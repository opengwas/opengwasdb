"""Does the effect a build will store agree with the source's own p-value? (ADR 0069)

A source whose `beta` column holds odds ratios, or whose `odds_ratio` column
holds log-odds, is read without complaint: both are finite numbers. The z-score
the build would store from them (`beta / se`) is then wrong by a scale, and no
row looks wrong on its own. The source's own p-value is the independent witness:
for a source that is what it says it is, the two-sided p implied by `beta / se`
tracks the p-value the file reports, and for one that is not, it does not.

Two statistics are taken over a bounded sample of rows, where
``Δ = log10 p_implied - log10 p_source``: the median of ``|Δ|`` and the median of
signed ``Δ``. Logs, because the p-values here span 300 orders of magnitude and
an absolute difference of p is blind to all but the largest; computed in log
space, because ``2 * sf(|z|)`` underflows to 0 at the |z| this project sees
(137). Medians, because a minority of rows with a different test statistic
behind their p-value (a saddlepoint or Firth p beside a Wald beta) must not
decide a file's fate; a scale error moves every row.

The median ``|Δ|`` catches an effect on the wrong scale (an odds ratio in
``beta``), whose z is wrong by orders of magnitude. It cannot catch a stored z
that carries no signal (``beta`` 9.0 beside ``se`` 1000): then ``p_implied`` is
about 1 and ``Δ`` about ``-log10 p_source``, whose null median is log10 2 = 0.30,
under the ``|Δ|`` threshold. The median signed ``Δ`` catches that: every clean
source's is near 0, so a positive median above 0.1 is a deflated z (#370).

Rows whose standard error was *derived from* the p-value are excluded: their
z is the p-value's own quantile, so the comparison would agree by construction
and prove nothing (ADR 0055).

The module computes; `opengwasdb.build.resolve` decides, and the decision is a
refusal (ADR 0069, amending ADR 0044).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy.stats import norm

from opengwasdb.build.hash_sample import HashSample
from opengwasdb.readers.tabular import MetricsChunk

__all__ = [
    "DEFAULT_MAX_MEDIAN_ABS_DELTA",
    "DEFAULT_MAX_MEDIAN_DELTA",
    "DEFAULT_MIN_COMPARED_ROWS",
    "DEFAULT_EFFECT_CONSISTENCY_POLICY",
    "ConsistencySkip",
    "ConsistencySample",
    "EffectConsistency",
    "EffectConsistencyPolicy",
    "check_sample_bound",
    "log10_two_sided_p",
]

#: Refusal threshold on the median |Δ| (orders of magnitude of p). Chosen from
#: measurement; see ADR 0069 for the calibration.
DEFAULT_MAX_MEDIAN_ABS_DELTA = 1.0

#: Refusal threshold on the median signed Δ. A stored z deflated towards 0 makes
#: Δ ≈ -log10 p_source, whose null median is log10 2 = 0.30, which the |Δ| rule
#: above can never refuse. Chosen from measurement; see ADR 0069's addendum.
DEFAULT_MAX_MEDIAN_DELTA = 0.1

#: Fewest comparable rows a verdict is drawn from.
DEFAULT_MIN_COMPARED_ROWS = 50

_LN2 = math.log(2.0)
_LN10 = math.log(10.0)


@dataclass(frozen=True)
class EffectConsistencyPolicy:
    """When a source's effect is refused for contradicting its own p-value."""

    max_median_abs_delta: float = DEFAULT_MAX_MEDIAN_ABS_DELTA
    min_compared_rows: int = DEFAULT_MIN_COMPARED_ROWS
    max_median_delta: float = DEFAULT_MAX_MEDIAN_DELTA

    def __post_init__(self) -> None:
        if not (math.isfinite(self.max_median_abs_delta) and self.max_median_abs_delta > 0):
            raise ValueError(
                f"max_median_abs_delta must be positive, got {self.max_median_abs_delta!r}"
            )
        if not (math.isfinite(self.max_median_delta) and self.max_median_delta > 0):
            raise ValueError(f"max_median_delta must be positive, got {self.max_median_delta!r}")
        if self.min_compared_rows < 1:
            raise ValueError(f"min_compared_rows must be >= 1, got {self.min_compared_rows!r}")

    def as_fingerprint(self) -> dict[str, float | int]:
        return {
            "max_median_abs_delta": self.max_median_abs_delta,
            "max_median_delta": self.max_median_delta,
            "min_compared_rows": self.min_compared_rows,
        }


#: The documented default policy, shared so a default argument is not a call.
DEFAULT_EFFECT_CONSISTENCY_POLICY = EffectConsistencyPolicy()


class ConsistencySkip(StrEnum):
    """Why no verdict was drawn. A skip is recorded, never silent (ADR 0069)."""

    #: No admitted row carries a usable source p-value (no column, or none in
    #: (0, 1]) -- GWAS-VCF and BESD sources today.
    NO_SOURCE_P_VALUE = "no_source_p_value"
    #: Every admitted row with a source p-value had its SE derived from it.
    ALL_SE_FROM_P_VALUE = "all_se_from_p_value"
    #: Fewer comparable rows than the policy's minimum.
    TOO_FEW_ROWS = "too_few_rows"
    #: The Analysis declares its p-value a joint 2-df test (issue #371): the
    #: stored coefficient is valid, but that p tests more than it, so it cannot
    #: witness beta/se.
    JOINT_TEST_P_VALUE = "joint_test_p_value"


@dataclass(frozen=True)
class EffectConsistency:
    """One Analysis's measurement of stored z-score against source p-value.

    `n_rows_considered` counts every comparable row the scan saw; `n_rows_compared`
    is those in the bounded sample the medians are taken over. `skip_reason` is
    `None` exactly when a comparison was made.
    """

    n_rows_considered: int
    n_rows_compared: int
    sampled: bool
    median_abs_delta: float | None
    median_delta: float | None
    skip_reason: ConsistencySkip | None
    refused: bool = False
    #: Which rule refused: `"median_abs_delta"` or `"median_delta"`; `None` unless refused.
    refused_by: str | None = None


def check_sample_bound(evidence_sample: int, policy: EffectConsistencyPolicy) -> None:
    """Refuse a sample bound that can never reach the policy's minimum.

    A bound below `min_compared_rows` would skip every Analysis as `too_few_rows`
    without saying the configuration was the reason, so it raises where the two
    first meet.
    """
    if evidence_sample < policy.min_compared_rows:
        raise ValueError(
            f"evidence_sample ({evidence_sample}) is below the effect-consistency "
            f"min_compared_rows ({policy.min_compared_rows}): no Analysis could ever be "
            "checked. Raise evidence_sample or lower min_compared_rows."
        )


def log10_two_sided_p(z: np.ndarray) -> np.ndarray:
    """log10 of the two-sided normal p-value for `z`, computed in log space."""
    return (_LN2 + norm.logsf(np.abs(z))) / _LN10


class ConsistencySample:
    """Accumulate comparable rows' `(z, log10 p_source)`, bounded by a hash sample."""

    def __init__(self, k: int) -> None:
        self._sample: HashSample[tuple[float, float]] = HashSample(k)
        self.n_with_source_p = 0
        self.n_p_derived = 0

    def admit(self, chunk: MetricsChunk) -> None:
        """Offer one block of rows the build would admit."""
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            z = chunk.beta / chunk.se
        p = chunk.p_value
        usable = np.isfinite(z) & (chunk.se > 0) & np.isfinite(p) & (p > 0.0) & (p <= 1.0)
        self.n_with_source_p += int(np.count_nonzero(usable))
        derived = usable & chunk.se_from_p_value
        self.n_p_derived += int(np.count_nonzero(derived))
        rows = np.flatnonzero(usable & ~chunk.se_from_p_value)
        log_p = np.log10(p[rows])
        for index, value in zip(rows.tolist(), log_p.tolist(), strict=True):
            self._sample.offer(chunk.alid[index], (float(z[index]), value))

    def measure(self, policy: EffectConsistencyPolicy) -> EffectConsistency:
        considered = self._sample.offered
        retained = self._sample.retained()
        skip = self._skip(considered, len(retained), policy)
        if skip is not None:
            return EffectConsistency(considered, 0, False, None, None, skip)
        z = np.array([row[0] for row in retained])
        delta = log10_two_sided_p(z) - np.array([row[1] for row in retained])
        median_abs = float(np.median(np.abs(delta)))
        median_signed = float(np.median(delta))
        refused_by = None
        if median_abs > policy.max_median_abs_delta:
            refused_by = "median_abs_delta"
        elif median_signed > policy.max_median_delta:
            refused_by = "median_delta"
        return EffectConsistency(
            n_rows_considered=considered,
            n_rows_compared=len(retained),
            sampled=self._sample.sampled,
            median_abs_delta=median_abs,
            median_delta=median_signed,
            skip_reason=None,
            refused=refused_by is not None,
            refused_by=refused_by,
        )

    def skipped(self, reason: ConsistencySkip) -> EffectConsistency:
        """A recorded skip for a reason the caller knows, over the rows the scan saw."""
        return EffectConsistency(self._sample.offered, 0, False, None, None, reason)

    def _skip(
        self, considered: int, compared: int, policy: EffectConsistencyPolicy
    ) -> ConsistencySkip | None:
        if self.n_with_source_p == 0:
            return ConsistencySkip.NO_SOURCE_P_VALUE
        if considered == 0:
            return ConsistencySkip.ALL_SE_FROM_P_VALUE
        if compared < policy.min_compared_rows:
            return ConsistencySkip.TOO_FEW_ROWS
        return None
