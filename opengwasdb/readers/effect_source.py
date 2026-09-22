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
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from opengwasdb.model.enums import StoredEffectScale


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
    """

    column_name: str
    kind: EffectSourceKind
    is_derived: bool = False
    assumes_standardised: bool = False


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
    for kind, is_derived, assumes_standardised, match in matched:
        if match is not None:
            _, cell = match
            return EffectSource(
                column_name=_text(cell),
                kind=kind,
                is_derived=is_derived,
                assumes_standardised=assumes_standardised,
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
