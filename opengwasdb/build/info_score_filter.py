"""The single spelling of a declared INFO policy's retention rule (stores #175, #176).

`opengwasdb.build.resolve` applies it to the canonical rows its metrics scan
yields and `opengwasdb.layouts.hybrid.build` applies it to the associations its
Source Reader yields, before evidence filtering and Dense/Overflow routing
respectively. `opengwasdb.build.row_admission` combines it with the MAF filter
and both callers go through that, so which rows are admitted cannot drift between
the resolver's record and the store a build writes.

Only `FILTERED` drops a row, and only when the row carries a *usable* score (any
finite number, in range or not) strictly below the threshold: equality passes,
and a score above 1 passes any threshold <= 1 while a negative score falls below
any positive one. A row whose score is missing, malformed or non-finite carries
no number this rule has an opinion about, so it is retained. `DISABLED` (an
explicit zero threshold with a declared score), `UNAVAILABLE` (a literal `NaN`)
and `LEGACY_ABSENT` (no threshold column at all) are three different recorded
facts that all mean "drop nothing".

What each caller *counts* differs by construction, and the names say so: the
resolver counts canonical-identity rows of a physical scan prefix
(`canonical_rows_*`), a builder counts the associations its reader yielded after
its own effect/SE admission. `InfoScoreCounts` is the declared-score disposition
tally; each caller records it under its own names.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from opengwasdb.model.info_score_policy import InfoScorePolicy, InfoScoreState
from opengwasdb.readers.interface import ImputationScoreStatus

__all__ = [
    "InfoScoreCounts",
    "count_info_scores",
    "declared_score_state",
    "retained_mask",
]

#: The statuses that carry a finite number (stores #176). A score in either is a
#: *usable* score: it is compared against the declared threshold. The other
#: statuses carry no number and are never dropped by this rule.
_FINITE_STATUSES = (ImputationScoreStatus.USABLE, ImputationScoreStatus.OUT_OF_RANGE)


def retained_mask(
    scores: np.ndarray, statuses: np.ndarray, policy: InfoScorePolicy
) -> np.ndarray:
    """Which positionally parallel observed rows a declared policy keeps.

    `scores` is what a caller parsed from the declared column
    (`MetricsChunk.imputation_score` or `ReaderAssociation.imputation_score`)
    and `statuses` each row's validity. A state that does not filter keeps every
    row, including rows carrying no score at all. Only *usable* scores are
    compared against the threshold: a row whose score is missing, malformed or
    non-finite carries no number this rule has an opinion about and is kept; a
    row nothing was declared for carries none at all.
    """
    if policy.state is not InfoScoreState.FILTERED:
        return np.ones(len(scores), dtype=bool)
    threshold = policy.info_score_threshold
    assert threshold is not None, "a FILTERED policy carries its threshold"
    usable = np.isin(statuses, _FINITE_STATUSES)
    kept = np.ones(len(scores), dtype=bool)
    kept[usable] = scores[usable] >= threshold
    return kept


@dataclass(frozen=True)
class InfoScoreCounts:
    """A caller's declared-score dispositions, over the rows it observed.

    `observed = retained + below_threshold`: every row is either kept or dropped,
    and only a usable score below a positive threshold is dropped. `retained`
    is what the INFO rule alone keeps -- unscored rows included -- *before* any
    MAF filter, which `opengwasdb.build.row_admission.AdmissionCounts` applies on
    top. `usable` counts the rows the declared column yielded a usable score
    (a finite number) for, so it is the rows dropped plus the usable rows kept,
    not the whole of `retained`.
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
    `out_of_range` is informational -- usable scores outside [0, 1], which the
    filter treats like any other usable score.
    """
    usable = np.isin(statuses, _FINITE_STATUSES)
    below_threshold = 0
    if policy.state is InfoScoreState.FILTERED:
        threshold = policy.info_score_threshold
        assert threshold is not None, "a FILTERED policy carries its threshold"
        below_threshold = int(np.count_nonzero(scores[usable] < threshold))
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


def declared_score_state(policy: InfoScorePolicy, usable: int) -> InfoScoreState:
    """The recorded state of a declared-score filter (stores #176).

    A declared Analysis whose source yielded no usable score is retained whole
    and reported `no_usable_scores` rather than dropped or refused: an Analysis
    built from zero scored associations would otherwise look exactly like one
    whose source has no associations at all. A policy that declared no score is
    reported by its own state, whatever `usable` says (it is always zero, because
    no column was read).
    """
    if policy.imputation_score_declaration is not None and usable == 0:
        return InfoScoreState.NO_USABLE_SCORES
    return policy.state
