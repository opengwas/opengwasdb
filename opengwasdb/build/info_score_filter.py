"""The single spelling of a declared INFO policy's retention rule (stores #175).

`opengwasdb.build.resolve` applies it to the canonical rows its metrics scan
yields and `opengwasdb.layouts.hybrid.build` applies it to the associations its
Source Reader yields, before evidence filtering and Dense/Overflow routing
respectively. Both call `retained_mask` here rather than restating it, so which
rows a declared policy keeps cannot drift between the resolver's record and the
store a build writes.

Only `FILTERED` drops a row, and a row whose *usable* score equals the threshold
is kept: equality passes. `DISABLED` (an explicit zero threshold with a declared
score), `UNAVAILABLE` (a literal `NaN`) and `LEGACY_ABSENT` (no threshold column
at all) are three different recorded facts that all mean "drop nothing" -- the
states stay distinguishable without filtering differently -- and a row whose
score is missing, malformed, non-finite or out of range is dropped under
`FILTERED` whatever its value.

What each caller *counts* differs by construction, and the names say so: the
resolver counts canonical-identity rows of a physical scan prefix
(`canonical_rows_*`), a builder counts the associations its reader yielded after
its own effect/SE admission. `InfoScoreCounts` is the shared disposition tally;
each caller records it under its own names.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from opengwasdb.model.info_score_policy import InfoScorePolicy, InfoScoreState
from opengwasdb.readers.interface import ImputationScoreStatus

__all__ = ["InfoScoreCounts", "count_info_scores", "retained_mask"]


def retained_mask(
    scores: np.ndarray, statuses: np.ndarray, policy: InfoScorePolicy
) -> np.ndarray:
    """Which positionally parallel observed rows a declared policy keeps.

    `scores` is what a caller parsed from the declared column
    (`MetricsChunk.imputation_score` or `ReaderAssociation.imputation_score`)
    and `statuses` each row's validity. A state that does not filter keeps every
    row, including rows carrying no score at all.
    """
    if policy.state is not InfoScoreState.FILTERED:
        return np.ones(len(scores), dtype=bool)
    threshold = policy.info_score_threshold
    assert threshold is not None, "a FILTERED policy carries its threshold"
    kept: np.ndarray = (statuses == ImputationScoreStatus.USABLE) & (scores >= threshold)
    return kept


@dataclass(frozen=True)
class InfoScoreCounts:
    """A caller's declared-score dispositions, over the rows it observed.

    `observed = retained + below_threshold + missing + malformed + nonfinite +
    out_of_range`, and `usable` counts the rows the declared column yielded a
    usable score for -- which is `retained + below_threshold` under `FILTERED`,
    and zero for a state that declared no score to read.
    """

    observed: int = 0
    retained: int = 0
    below_threshold: int = 0
    missing: int = 0
    malformed: int = 0
    nonfinite: int = 0
    out_of_range: int = 0
    usable: int = 0

    def __add__(self, other: InfoScoreCounts) -> InfoScoreCounts:
        return InfoScoreCounts(
            observed=self.observed + other.observed,
            retained=self.retained + other.retained,
            below_threshold=self.below_threshold + other.below_threshold,
            missing=self.missing + other.missing,
            malformed=self.malformed + other.malformed,
            nonfinite=self.nonfinite + other.nonfinite,
            out_of_range=self.out_of_range + other.out_of_range,
            usable=self.usable + other.usable,
        )


def count_info_scores(
    scores: np.ndarray, statuses: np.ndarray, policy: InfoScorePolicy
) -> InfoScoreCounts:
    """One block's score dispositions, over every observed row.

    `retained` comes from the same `retained_mask` the caller filters with, so
    the tally and the filter cannot disagree about which rows survived.
    """
    usable = statuses == ImputationScoreStatus.USABLE
    below_threshold = 0
    if policy.state is InfoScoreState.FILTERED:
        threshold = policy.info_score_threshold
        assert threshold is not None, "a FILTERED policy carries its threshold"
        below_threshold = int(np.count_nonzero(usable & (scores < threshold)))
    return InfoScoreCounts(
        observed=len(scores),
        retained=int(np.count_nonzero(retained_mask(scores, statuses, policy))),
        below_threshold=below_threshold,
        missing=int(np.count_nonzero(statuses == ImputationScoreStatus.MISSING)),
        malformed=int(np.count_nonzero(statuses == ImputationScoreStatus.MALFORMED)),
        nonfinite=int(np.count_nonzero(statuses == ImputationScoreStatus.NONFINITE)),
        out_of_range=int(np.count_nonzero(statuses == ImputationScoreStatus.OUT_OF_RANGE)),
        usable=int(np.count_nonzero(usable)),
    )
