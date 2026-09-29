"""Which column an Analysis's effect was read from (issues #213-#215).

GWAS-SSF permits an Analysis to report its effect as `beta`, `odds_ratio`, or a
signed `z_score`. The reader used to hardcode `beta`, so a harmonised file that
named every variant and reported a usable effect through another permitted
column yielded no beta and was dropped from the association stream -- an empty
result indistinguishable from "no association".

This module makes the choice a resolved fact about the file rather than an
assumption buried in a column lookup. Each kind's accepted spellings are
*enumerated*, not case-insensitive, so a header that is genuinely misspelled
still fails loudly (#214).

`beta = log(odds_ratio)`; the standard error GWAS-SSF reports is already on the
log scale, so it is carried through unchanged. A signed z is different: it is
not a change of units but an approximation that assumes a standardised phenotype
(`var(Y) = 1`), so the derived beta is in phenotype-SD units by construction and
`EffectSource.assumes_standardised` says so rather than leaving it invisible at
the call site (#215).

Resolving *which column* is only half of reading an effect. A full OGS-00011
resolve found 476 Analyses with no build-eligible row, 212 of which report an
effect and a precision the reader was not reading: `beta` empty on every row
while `odds_ratio` is populated, a 95% CI instead of a `standard_error`, or an
effect with only a p-value. `row_statistics` is that second half -- the one
per-row rule, applied by both the row-wise and the blocked projection -- and it
keeps the scale of the column that supplied *each row's* value, because a file
may spell the same effect two ways and use them in different rows (stores #176).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.stats import Z_975, inverse_normal_denominator


class EffectSourceKind(StrEnum):
    """The kind of effect an Analysis resolved to, independent of its spelling.

    The value is the kind's canonical spelling: a file whose column is `BETA`
    still resolves to `EffectSourceKind.BETA`, while
    `EffectSource.column_name` records the `BETA` spelling the file used (#214).
    """

    BETA = "beta"
    ODDS_RATIO = "odds_ratio"
    Z_SCORE = "z_score"


class CaseControlZScoreError(ValueError):
    """A signed z is a standardised effect, not a log-OR or log-hazard.

    The z-score formula yields a beta on the phenotype-SD scale. A case-control
    Analysis stores a log-OR or log-hazard, so deriving one through this path
    would silently relabel an effect it is not (#215).
    """


class UnsignedZScoreError(ValueError):
    """A z-score column with no negative value cannot be read as signed.

    A `|z|` or chi-square statistic has the same magnitude and no sign; reading
    it as a signed z would give every derived effect the wrong direction (#215).
    """


@dataclass(frozen=True)
class EffectSource:
    """One Analysis's resolved effect column and what must be done to it.

    `column_name` is the exact name the file's header carries -- reported to
    the caller rather than assumed. `is_derived` says the beta is computed
    from the column rather than read from it (`log(odds_ratio)`, or a z-score
    derivation). `assumes_standardised` says the derived beta is in
    phenotype-SD units by construction -- true for a signed z, whose formula
    assumes `var(Y) = 1` (#215) -- so a caller cannot obtain a z-derived effect
    without also obtaining that fact.

    `is_derived` and `assumes_standardised` describe `column_name` alone.
    `fallback_column_name`, when set, is the *second* effect column the file
    carries for the same value -- the `odds_ratio` beside a `beta` that is
    empty on every row (`GCST004030`) -- and a row that uses it has a derived
    beta (`log(odds_ratio)`) whichever column `column_name` names. `None` when
    the file names one effect column, which is most of them (stores #176).
    """

    column_name: str
    kind: EffectSourceKind
    is_derived: bool = False
    assumes_standardised: bool = False
    fallback_column_name: str | None = None


#: Candidate effect columns in precedence order. Each entry is the *enumerated*
#: spellings of one column -- not a case-insensitive match, so a genuinely
#: misspelled header still fails loudly -- followed by its kind, whether the
#: beta is derived from it, and whether that derivation assumes a standardised
#: phenotype. `beta` precedes `odds_ratio`, which precedes a signed `z`: the
#: earlier column is the source's own effect and needs no transform, and the
#: rule is explicit and tested rather than whichever lookup happened to run
#: last.
_EFFECT_COLUMNS: tuple[tuple[tuple[str, ...], EffectSourceKind, bool, bool], ...] = (
    (("beta", "BETA"), EffectSourceKind.BETA, False, False),
    (("odds_ratio",), EffectSourceKind.ODDS_RATIO, True, False),
    (("z_score", "Zscore", "ZScore", "z"), EffectSourceKind.Z_SCORE, True, True),
)

#: The per-row sample size a z-score derivation needs, enumerated the same way.
#: It is read from the row, never a study-level scalar (#215).
_SAMPLE_SIZE_SPELLINGS: tuple[str, ...] = ("n", "N")


def _matches(name: str | bytes, column: str) -> bool:
    """Whether a header cell names ``column``, ignoring surrounding whitespace.

    A padded spelling is the same column: the real `GCST006329` harmonised file
    carries both `beta ` (the column holding every value) and `beta` (all `NA`),
    and matching them exactly let the reader silently read the empty one --
    200,000 of 200,000 sampled rows dropped. Stripping here is what makes that
    collision a duplicate rather than two different columns.

    A bytes header is compared as bytes rather than decoded wholesale, so an
    unrelated non-UTF-8 column name cannot fail resolution.
    """
    if isinstance(name, bytes):
        return name.strip() == column.encode("utf-8")
    return name.strip() == column


def _text(name: str | bytes) -> str:
    """One header cell as `str`, decoding only the cell that matched."""
    return name.decode("utf-8") if isinstance(name, bytes) else name


def _resolve_spelling(
    header: Sequence[str] | Sequence[bytes], spelling: str, label: str
) -> str | bytes | None:
    """The one header cell matching ``spelling``, or ``None``; refuses a duplicate.

    A spelling named more than once cannot be interpreted honestly -- the real
    `GCST006329` carries `beta ` and `beta` -- and a last-wins lookup would
    silently read one of two columns.
    """
    matches = [name for name in header if _matches(name, spelling)]
    if len(matches) > 1:
        raise ValueError(f"Duplicate {label} {spelling!r} in header")
    return matches[0] if matches else None


def _resolve_candidate(
    header: Sequence[str] | Sequence[bytes], spellings: tuple[str, ...], label: str
) -> tuple[str, str | bytes] | None:
    """The one spelling this header carries, with its cell; refuses a conflict.

    Every spelling of the candidate is resolved, not just the winning one, so a
    duplicated lower-precedence column is refused even when a higher-precedence
    one is present -- precedence orders usable sources, it does not excuse a
    malformed header.

    Two *different* spellings of the same column is ambiguous: the reader cannot
    know which the file meant, so it refuses rather than preferring one.
    Duplicates are checked first, so a header with two `beta`s and a `BETA` is a
    duplicate rather than only ambiguous.
    """
    found = [(spelling, _resolve_spelling(header, spelling, label)) for spelling in spellings]
    present = [(spelling, cell) for spelling, cell in found if cell is not None]
    if len(present) > 1:
        names = " and ".join(repr(spelling) for spelling, _ in present)
        raise ValueError(f"Ambiguous {label}: header carries both {names}")
    return present[0] if present else None


def resolve_effect_source(header: Sequence[str] | Sequence[bytes]) -> EffectSource | None:
    """Resolve which effect column ``header`` names, or ``None`` if it names none.

    A candidate column named more than once raises `ValueError`: the header
    cannot be interpreted honestly (a real harmonised file, `GCST006329`,
    carries both `beta ` and `beta`), and a last-wins lookup would silently
    read one of two different columns. Two spellings of one column (`beta` and
    `BETA`, or two of a z-score's four) is likewise refused as ambiguous
    (#214, #215).

    `column_name` is the matched cell verbatim, padding and case included,
    because that exact spelling is what the caller must look the column up by;
    only the *match* ignores whitespace.

    A header carrying both a `beta` spelling and an `odds_ratio` spelling
    resolves to `beta`, and records the `odds_ratio` in
    `fallback_column_name`: `beta` is the source's own effect, and a file whose
    `beta` column is empty on every row still reports its effect through
    `odds_ratio` (`GCST004030`), which `row_statistics` reads per row.
    """
    matched = [
        (
            kind,
            is_derived,
            assumes_standardised,
            _resolve_candidate(header, spellings, "effect column"),
        )
        for spellings, kind, is_derived, assumes_standardised in _EFFECT_COLUMNS
    ]
    carried = {kind: match for kind, _is_derived, _standardised, match in matched}
    for kind, is_derived, assumes_standardised, match in matched:
        if match is not None:
            _, cell = match
            fallback = (
                carried[EffectSourceKind.ODDS_RATIO]
                if kind is EffectSourceKind.BETA
                else None
            )
            return EffectSource(
                column_name=_text(cell),
                kind=kind,
                is_derived=is_derived,
                assumes_standardised=assumes_standardised,
                fallback_column_name=None if fallback is None else _text(fallback[1]),
            )
    return None


def resolve_sample_size_column(header: Sequence[str] | Sequence[bytes]) -> str | None:
    """The header cell naming the per-row sample size, or ``None``.

    `n` and `N` are enumerated spellings, resolved under the same duplicate and
    ambiguity rules as an effect column: silently choosing one of two
    sample-size columns is exactly the wrong answer this seam exists to prevent.
    """
    match = _resolve_candidate(header, _SAMPLE_SIZE_SPELLINGS, label="sample-size column")
    return None if match is None else _text(match[1])


def derive_z_score_effect(
    z: float | None, effect_allele_frequency: float | None, sample_size: float | None
) -> tuple[float, float] | None:
    """``(beta, se)`` for a signed z on the phenotype-SD scale, or ``None``.

    For a signed z, effect-allele frequency ``f`` and per-row sample size ``N``::

        se   = 1 / sqrt(2 * f * (1 - f) * (N + z^2))
        beta = z * se

    The formula assumes a standardised phenotype, so the beta is in phenotype-SD
    units by construction; the caller carries that fact in
    `EffectSource.assumes_standardised` (#215). An EAF outside ``(0, 1)``, a
    non-positive ``N``, or any missing input yields ``None`` -- the row is
    dropped, never approximated with a substituted frequency or sample size.
    """
    if z is None or effect_allele_frequency is None or sample_size is None:
        return None
    if not 0.0 < effect_allele_frequency < 1.0 or sample_size <= 0.0:
        return None
    variance = 2.0 * effect_allele_frequency * (1.0 - effect_allele_frequency)
    se = 1.0 / math.sqrt(variance * (sample_size + z * z))
    return z * se, se


@dataclass(frozen=True)
class RowCells:
    """One row's candidate effect and precision columns, already read.

    Each value is a cell parsed to a float or `None` -- never a default standing
    in for a missing one. `effect` is the resolved effect column's own value, on
    its own scale: a beta, an odds ratio, or a signed z, according to
    `EffectSource.kind`. `odds_ratio` is the file's *second* effect column, the
    one `EffectSource.fallback_column_name` names. `frequency` and `sample_size`
    are read only by a signed z's derivation (#215).
    """

    effect: float | None = None
    odds_ratio: float | None = None
    standard_error: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    p_value: float | None = None
    frequency: float | None = None
    sample_size: float | None = None


@dataclass(frozen=True)
class RowStatistics:
    """One row's ``(beta, se)`` and which of its columns supplied them.

    `effect_from_odds_ratio_fallback` is true when the beta is
    `log(odds_ratio)` because the row's own `beta` cell was unusable.
    `se_from_ci` and `se_from_p_value` say the standard error was derived from
    the row's confidence interval or its p-value rather than read from
    `standard_error`. All three are false for a row whose effect or precision is
    simply unusable, and they are what the resolver's build-eligible counts
    report (stores #176).
    """

    beta: float | None
    se: float | None
    effect_from_odds_ratio_fallback: bool = False
    se_from_ci: bool = False
    se_from_p_value: bool = False


@dataclass(frozen=True)
class _RowEffect:
    """One row's effect, and the scale its own column put it on.

    `odds_ratio` is the raw value the beta was logged from, present exactly when
    the effect is on the log-odds scale -- which is what decides the scale a
    confidence interval for *this row* is read on. `is_fallback` says the
    value came from the file's second effect column rather than its first.
    """

    beta: float
    odds_ratio: float | None
    is_fallback: bool


def _positive(value: float | None) -> float | None:
    """`value` when it is positive and finite, else `None`."""
    if value is None or not math.isfinite(value) or value <= 0.0:
        return None
    return value


def _row_effect(source: EffectSource, cells: RowCells) -> _RowEffect | None:
    """The row's effect and its scale, or `None` when no column carries one.

    `beta` when it is finite, else `log(odds_ratio)` when the file carries that
    second column and the value is positive and finite -- the `beta` empty on
    every row while `odds_ratio` is populated is the same effect spelled the
    other way, not a different quantity. A file naming only one effect column
    behaves exactly as it did before stores #176: an unusable cell yields no
    effect, never the other column's value.
    """
    if source.kind is EffectSourceKind.ODDS_RATIO:
        raw = _positive(cells.effect)
        return None if raw is None else _RowEffect(math.log(raw), raw, False)
    if cells.effect is not None and math.isfinite(cells.effect):
        return _RowEffect(cells.effect, None, False)
    if source.fallback_column_name is None:
        return None
    raw = _positive(cells.odds_ratio)
    return None if raw is None else _RowEffect(math.log(raw), raw, True)


def _se_from_confidence_interval(
    effect: _RowEffect, lower: float | None, upper: float | None
) -> float | None:
    """The standard error a reported 95% interval implies, or `None`.

    The interval is read on the scale of the column that supplied *this row's*
    effect, and only when the row's own effect lies inside it: an interval on
    the other scale is rejected rather than converted, because nothing in the
    file says which scale it is and on the real files that carry one the wrong
    formula is 12 to 76 times off. A log-odds interval additionally needs both
    bounds positive; a beta interval needs only `lower < upper`.
    """
    if lower is None or upper is None:
        return None
    if not (math.isfinite(lower) and math.isfinite(upper)):
        return None
    if lower >= upper:
        return None
    if effect.odds_ratio is None:
        return _positive_interval(effect.beta, lower, upper, log_scale=False)
    return _positive_interval(effect.odds_ratio, lower, upper, log_scale=True)


def _positive_interval(
    value: float, lower: float, upper: float, *, log_scale: bool
) -> float | None:
    """The interval's standard error when `value` lies inside it, else `None`.

    The bounds are divided as they stand on a beta's own scale and logged on an
    odds ratio's, so the log form additionally needs a positive lower bound.
    Neither form converts the interval: an interval presented on the other scale
    fails the bracket test rather than being re-expressed.
    """
    if log_scale:
        if not lower > 0.0 or not lower <= value <= upper:
            return None
        return _positive((math.log(upper) - math.log(lower)) / (2.0 * Z_975))
    if not lower <= value <= upper:
        return None
    return _positive((upper - lower) / (2.0 * Z_975))


def _se_from_p_value(beta: float, p_value: float | None) -> float | None:
    """The standard error a two-sided p-value implies for a non-zero effect.

    ``se = |beta| / -Φ⁻¹(p / 2)``. `-Φ⁻¹(p / 2)` and not `Φ⁻¹(1 - p / 2)`, which
    rounds to infinity for a p small enough to matter here. `p = 1` has no
    effect size behind it, a p that underflows to `0` has none either, `p > 1`
    is a broken cell, and an effect of exactly zero has no direction: all four
    are unusable rather than approximated. `neg_log_10_p_value` is deliberately
    not read -- GWAS-SSF's own `p_value` is the column this rule is about.
    """
    if p_value is None or not math.isfinite(p_value) or not 0.0 < p_value < 1.0:
        return None
    if beta == 0.0:
        return None
    return _positive(abs(beta) / inverse_normal_denominator(p_value))


def row_statistics(source: EffectSource | None, cells: RowCells) -> RowStatistics:
    """One row's ``(beta, se)`` from the row's own columns (stores #176).

    A full OGS-00011 resolve found 476 Analyses with no build-eligible row, of
    which 212 report an effect and a precision the reader was not reading. The
    rules, in order:

    1. The effect is the resolved column's own value when it is usable, else
       `log(odds_ratio)` when the file carries that second column and the value
       is positive and finite. Unusable otherwise.
    2. The standard error is `standard_error` when it is positive and finite;
       else the row's 95% interval when its bounds are finite, in order and
       around the row's own effect on the row's own scale; else
       ``|beta| / -Φ⁻¹(p / 2)`` for a two-sided p in ``(0, 1)`` and a non-zero
       beta. The first usable one wins.
    3. A derived standard error must itself be positive and finite, or the row
       has none.

    A signed z is not read this way: `derive_z_score_effect` derives both
    statistics from the z, the frequency and the per-row sample size (#215),
    and the two fallbacks below are not defined for it. A row whose effect is
    unusable still reports whatever `standard_error` carries, as both row-wise
    projections have always reported it -- the association stream is what drops
    such a row.
    """
    if source is None:
        return RowStatistics(None, _positive(cells.standard_error))
    if source.kind is EffectSourceKind.Z_SCORE:
        derived = derive_z_score_effect(cells.effect, cells.frequency, cells.sample_size)
        return RowStatistics(None, None) if derived is None else RowStatistics(*derived)
    effect = _row_effect(source, cells)
    if effect is None:
        return RowStatistics(None, _positive(cells.standard_error))
    reported = _positive(cells.standard_error)
    if reported is not None:
        return RowStatistics(effect.beta, reported, effect.is_fallback)
    from_ci = _se_from_confidence_interval(effect, cells.ci_lower, cells.ci_upper)
    if from_ci is not None:
        return RowStatistics(effect.beta, from_ci, effect.is_fallback, se_from_ci=True)
    from_p = _se_from_p_value(effect.beta, cells.p_value)
    if from_p is not None:
        return RowStatistics(
            effect.beta, from_p, effect.is_fallback, se_from_p_value=True
        )
    return RowStatistics(effect.beta, None, effect.is_fallback)


def refuse_case_control_z_score(
    effect_source: EffectSource | None, stored_effect_scale: StoredEffectScale
) -> None:
    """Raise `CaseControlZScoreError` for a z-derived effect on a case-control scale.

    A signed z derives a phenotype-SD-standardised beta; a `log_or` or
    `log_hazard` Analysis stores a different quantity, so the derivation must
    not run for it (#215).
    """
    if (
        effect_source is not None
        and effect_source.kind is EffectSourceKind.Z_SCORE
        and stored_effect_scale in (StoredEffectScale.LOG_OR, StoredEffectScale.LOG_HAZARD)
    ):
        raise CaseControlZScoreError(
            "a signed z derives a phenotype-SD-standardised beta, not a "
            f"{stored_effect_scale.value} effect; refusing to derive one"
        )
