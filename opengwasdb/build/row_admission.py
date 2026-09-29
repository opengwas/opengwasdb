"""The one row-admission rule the resolver scan and the Hybrid builder share (stores #176).

A declared INFO threshold and a declared MAF threshold both drop rows, and the
rule for which rows survive must not be spelled twice: the resolver's per-Analysis
record and the store a Hybrid build writes are compared to each other, so a
divergence between the two callers would be a silent mismatch the tests exist to
catch. `admit_rows` is that one rule: `keep = ~info_drop & ~maf_drop`.

MAF is `min(af, 1 - af)` for a finite `af` strictly inside `(0, 1)`. A row
whose `af` is missing, non-finite, outside `[0, 1]`, or exactly `0.0`/`1.0` is
retained and counted `maf_missing` -- the MAF rule has no frequency to judge it
by. A row is dropped by MAF only when the policy is `FILTERED`, the MAF is
available, and `MAF < maf_threshold`; equality passes.

When both filters would drop a row it is counted once, under INFO (INFO first):
`maf_below_threshold` never counts a row `info` already dropped, so the two
below-threshold tallies sum to the rows dropped by at most one filter each.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from opengwasdb.build.info_score_filter import (
    InfoScoreCounts,
    count_info_scores,
    retained_mask,
)
from opengwasdb.model.info_score_policy import InfoScorePolicy
from opengwasdb.model.maf_policy import MafPolicy, MafState

__all__ = ["Admission", "AdmissionCounts", "admit_rows"]


@dataclass(frozen=True)
class AdmissionCounts:
    """One caller's admitted-row dispositions, over the rows it observed.

    `info` is the declared-score tally; `admitted` is the rows the combined rule
    kept, which is what `canonical_rows_retained` (resolver) and
    `associations_retained` (builder) report -- not `info.retained`, which is the
    rows INFO alone keeps. `maf_below_threshold` counts rows dropped by MAF that
    INFO did not already drop; `maf_missing` counts rows with no usable
    frequency, which are retained.
    """

    info: InfoScoreCounts = field(default_factory=InfoScoreCounts)
    admitted: int = 0
    maf_below_threshold: int = 0
    maf_missing: int = 0

    def __add__(self, other: AdmissionCounts) -> AdmissionCounts:
        return AdmissionCounts(
            info=self.info + other.info,
            admitted=self.admitted + other.admitted,
            maf_below_threshold=self.maf_below_threshold + other.maf_below_threshold,
            maf_missing=self.maf_missing + other.maf_missing,
        )

    def as_dict(self) -> dict[str, Any]:
        """The shape a checkpoint record stores (nested `info` included)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AdmissionCounts:
        """Rebuild the record `as_dict` wrote."""
        return cls(
            info=InfoScoreCounts(**data["info"]),
            admitted=int(data["admitted"]),
            maf_below_threshold=int(data["maf_below_threshold"]),
            maf_missing=int(data["maf_missing"]),
        )


@dataclass(frozen=True)
class Admission:
    """One batch's admitted-row decision and the dispositions behind it."""

    keep: np.ndarray
    counts: AdmissionCounts


def admit_rows(
    scores: np.ndarray,
    statuses: np.ndarray,
    af: np.ndarray,
    info_policy: InfoScorePolicy,
    maf_policy: MafPolicy,
) -> Admission:
    """The rows one batch admits, by the shared INFO-then-MAF rule.

    `scores`/`statuses` are positionally parallel and are what the caller parsed
    from its declared score column; `af` is the positionally parallel source
    frequency (NaN where the source reports none). Both callers hand in the same
    three arrays' worth of data and get back the same mask, so a row one keeps
    the other keeps too.
    """
    info_keep = retained_mask(scores, statuses, info_policy)
    # Exactly 0 and exactly 1 are missing, not usable zeroes, as the reader's own
    # `parse_af` says (stores #176): a frequency of either describes a
    # monomorphic site and a file reporting one on every row is reporting a
    # placeholder. Such a row is retained and counted `maf_missing`, never
    # dropped for a MAF it does not have.
    available = np.isfinite(af) & (af > 0.0) & (af < 1.0)
    maf_below_threshold = 0
    # A state that declared no MAF filter records no MAF dispositions, exactly as
    # an undeclared INFO policy records none; `DISABLED` still reads the column,
    # so it counts the rows with no usable frequency while dropping none.
    maf_missing = 0 if maf_policy.state is MafState.UNAVAILABLE else int(
        np.count_nonzero(~available)
    )
    if maf_policy.state is MafState.FILTERED:
        threshold = maf_policy.maf_threshold
        assert threshold is not None, "a FILTERED MAF policy carries its threshold"
        maf = np.minimum(af, 1.0 - af)
        maf_drop = available & (maf < threshold)
        # INFO first: a row both filters would drop is counted there, not here.
        maf_below_threshold = int(np.count_nonzero(maf_drop & info_keep))
    else:
        maf_drop = np.zeros(len(af), dtype=bool)
    keep = info_keep & ~maf_drop
    return Admission(
        keep=keep,
        counts=AdmissionCounts(
            info=count_info_scores(scores, statuses, info_policy),
            admitted=int(np.count_nonzero(keep)),
            maf_below_threshold=maf_below_threshold,
            maf_missing=maf_missing,
        ),
    )
