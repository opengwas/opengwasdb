"""Which column an Analysis's effect was read from (issue #213).

GWAS-SSF permits an Analysis to report its effect as either `beta` or
`odds_ratio`. The reader used to hardcode `beta`, so a harmonised file that
named every variant and reported a usable effect through the other permitted
spelling yielded no beta and was dropped from the association stream -- an
empty result indistinguishable from "no association".

This module makes the choice a resolved fact about the file rather than an
assumption buried in a column lookup, and is the seam the follow-ups build on:
a signed `z_score` column (#215) and the `BETA` spelling (#214) are more
sources, not more special cases in the reader. Each kind's accepted spellings
are *enumerated* -- `beta` and `BETA`, not a case-insensitive match -- so a
header that is genuinely misspelled still fails loudly (#214).

`beta = log(odds_ratio)`; the standard error GWAS-SSF reports is already on the
log scale, so it is carried through unchanged. An `odds_ratio` that is
non-positive or unparseable is unusable in exactly the way an unparseable
`beta` is: the row is dropped, never repaired with a substitute.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum


class EffectSourceKind(StrEnum):
    """The kind of effect an Analysis resolved to, independent of its spelling.

    The value is the kind's canonical spelling: a file whose column is `BETA`
    still resolves to `EffectSourceKind.BETA`, while
    `EffectSource.column_name` records the `BETA` spelling the file used (#214).
    """

    BETA = "beta"
    ODDS_RATIO = "odds_ratio"


@dataclass(frozen=True)
class EffectSource:
    """One Analysis's resolved effect column and what must be done to it.

    `column_name` is the exact name the file's header carries -- reported to
    the caller rather than assumed. `is_derived` says the beta is computed
    from the column rather than read from it (`log(odds_ratio)`), so a caller
    that needs the source's own value knows to look at the column itself.
    `assumes_standardised` is reserved for a source whose beta is a
    standardised effect (a signed `z_score`, issue #215); it is `False` for
    both spellings GWAS-SSF defines today, and a source that sets it must say
    so explicitly.
    """

    column_name: str
    kind: EffectSourceKind
    is_derived: bool = False
    assumes_standardised: bool = False


#: Candidate effect columns in precedence order. Each entry is the *enumerated*
#: spellings of one column -- not a case-insensitive match, so a genuinely
#: misspelled header still fails loudly -- followed by its kind and whether the
#: beta is derived from it. `beta` precedes `odds_ratio` because when a file
#: carries both, the beta is the source's own effect and needs no transform:
#: the rule is explicit and tested rather than whichever lookup happened to run
#: last.
_EFFECT_COLUMNS: tuple[tuple[tuple[str, ...], EffectSourceKind, bool], ...] = (
    (("beta", "BETA"), EffectSourceKind.BETA, False),
    (("odds_ratio",), EffectSourceKind.ODDS_RATIO, True),
)


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
    header: Sequence[str] | Sequence[bytes], spelling: str
) -> str | bytes | None:
    """The one header cell matching ``spelling``, or ``None``; refuses a duplicate.

    A spelling named more than once cannot be interpreted honestly -- the real
    `GCST006329` carries `beta ` and `beta` -- and a last-wins lookup would
    silently read one of two columns.
    """
    matches = [name for name in header if _matches(name, spelling)]
    if len(matches) > 1:
        raise ValueError(f"Duplicate effect column {spelling!r} in header")
    return matches[0] if matches else None


def _resolve_candidate(
    header: Sequence[str] | Sequence[bytes], spellings: tuple[str, ...]
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
    found = [(spelling, _resolve_spelling(header, spelling)) for spelling in spellings]
    present = [(spelling, cell) for spelling, cell in found if cell is not None]
    if len(present) > 1:
        names = " and ".join(repr(spelling) for spelling, _ in present)
        raise ValueError(f"Ambiguous effect column: header carries both {names}")
    return present[0] if present else None


def resolve_effect_source(header: Sequence[str] | Sequence[bytes]) -> EffectSource | None:
    """Resolve which effect column ``header`` names, or ``None`` if it names none.

    A candidate column named more than once raises `ValueError`: the header
    cannot be interpreted honestly (a real harmonised file, `GCST006329`,
    carries both `beta ` and `beta`), and a last-wins lookup would silently
    read one of two different columns. Two spellings of one column (`beta` and
    `BETA`) is likewise refused as ambiguous (#214).

    `column_name` is the matched cell verbatim, padding and case included,
    because that exact spelling is what the caller must look the column up by;
    only the *match* ignores whitespace.
    """
    matched = [
        (kind, is_derived, _resolve_candidate(header, spellings))
        for spellings, kind, is_derived in _EFFECT_COLUMNS
    ]
    for kind, is_derived, match in matched:
        if match is not None:
            _, cell = match
            return EffectSource(column_name=_text(cell), kind=kind, is_derived=is_derived)
    return None
