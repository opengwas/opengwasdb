"""Per-Analysis minor-allele-frequency filter input policy (stores #176).

Unlike the declared imputation score there is no declaration triple: the
frequency is the reader's own `effect_allele_frequency`, a column every GWAS-SSF
source the resolver and the Hybrid builder read already carries. This policy
only parses the explicit manifest value; it does not inspect source data or
apply the filter -- `opengwasdb.build.row_admission` is the one place that does,
for both callers.

The tests the column has to pass are `NaN` (or an absent column) meaning no MAF
filter, a finite value in [0, 0.5], and `0` disabling the filter. A value of
`NaN` and an absent column are one state (`unavailable`) because both mean the
same thing downstream; `0` (`disabled`) and a positive value (`filtered`) stay
distinct recorded facts, exactly as `InfoScorePolicy`'s states do.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

__all__ = ["MAF_COLUMNS", "MafPolicy", "MafState", "parse_maf_policy"]

#: The MAF filter's manifest column. One column, no declaration triple.
MAF_COLUMNS: tuple[str, ...] = ("maf_threshold",)


class MafState(StrEnum):
    """Why an Analysis does or does not apply its declared MAF threshold."""

    #: `NaN` or an absent column: no MAF filter at all.
    UNAVAILABLE = "unavailable"
    #: An explicit zero threshold: the filter is declared but drops nothing.
    DISABLED = "disabled"
    #: A positive threshold in (0, 0.5]: rows below it are dropped.
    FILTERED = "filtered"


def _state_for(threshold: float) -> MafState:
    """The state a validated finite threshold in [0, 0.5] means (stores #176)."""
    return MafState.DISABLED if threshold == 0 else MafState.FILTERED


@dataclass(frozen=True)
class MafPolicy:
    """A numeric MAF cut-off, or no filter.

    `state` separates three situations callers must not collapse into one
    absence: no threshold was declared (`UNAVAILABLE`), a zero threshold asked
    for no filtering (`DISABLED`), or a positive threshold filters rows
    (`FILTERED`). It is derived by `parse_maf_policy` and asserted here so a
    policy that contradicts its own threshold cannot be constructed and
    silently applied.
    """

    maf_threshold: float | None = None
    state: MafState = MafState.UNAVAILABLE

    def __post_init__(self) -> None:
        if self.maf_threshold is None:
            allowed = (MafState.UNAVAILABLE,)
        else:
            if not isfinite(self.maf_threshold) or not 0 <= self.maf_threshold <= 0.5:
                raise ValueError(
                    f"maf_threshold must be finite in [0, 0.5], got {self.maf_threshold!r}"
                )
            allowed = (_state_for(self.maf_threshold),)
        if self.state not in allowed:
            names = " or ".join(state.value for state in allowed)
            raise ValueError(
                f"maf_state {self.state.value!r} contradicts maf_threshold "
                f"{self.maf_threshold!r}; expected {names}"
            )


def parse_maf_policy(row: Mapping[str, str]) -> MafPolicy:
    """Parse one `analyses.tsv` row; fail on an unreadable threshold.

    A present, non-`NaN`, unparseable, non-finite, or out-of-range value raises:
    a MAF filter that silently did not apply is the failure this column exists to
    prevent. An absent column and a literal `NaN` are both `UNAVAILABLE`.
    """
    raw = row.get("maf_threshold")
    if raw is None or raw.strip() == "NaN":
        return MafPolicy()
    try:
        threshold = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"invalid maf_threshold {raw!r}; expected a number in [0, 0.5] or NaN"
        ) from exc
    if not isfinite(threshold) or not 0 <= threshold <= 0.5:
        raise ValueError(f"invalid maf_threshold {raw!r}; expected a number in [0, 0.5] or NaN")
    return MafPolicy(threshold, _state_for(threshold))
