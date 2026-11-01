"""Does the effect a build will store agree with the source's own p-value? (ADR 0069)

A source whose `beta` column holds odds ratios, or whose `odds_ratio` column
holds log-odds, is read without complaint: both are finite numbers. The z-score
the build would store from them (`beta / se`) is then wrong by a scale, and no
row looks wrong on its own. The source's own p-value is the independent witness:
for a source that is what it says it is, the two-sided p implied by `beta / se`
tracks the p-value the file reports, and for one that is not, it does not.

The statistic is the median over a bounded sample of rows of
``|log10 p_implied - log10 p_source|``. Logs, because the p-values here span
300 orders of magnitude and an absolute difference of p is blind to all but the
largest; computed in log space, because ``2 * sf(|z|)`` underflows to 0 at the
|z| this project sees (137). A median, because a minority of rows with a
different test statistic behind their p-value (a saddlepoint or Firth p beside a
Wald beta) must not decide a file's fate; a scale error moves every row.

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
    "DEFAULT_MIN_COMPARED_ROWS",
    "DEFAULT_EFFECT_CONSISTENCY_POLICY",
    "ConsistencySkip",
    "ConsistencySample",
    "EffectConsistency",
    "EffectConsistencyPolicy",
    "log10_two_sided_p",
]

#: Refusal threshold on the median |Δ| (orders of magnitude of p). Chosen from
#: measurement; see ADR 0069 for the calibration.
DEFAULT_MAX_MEDIAN_ABS_DELTA = 1.0

#: Fewest comparable rows a verdict is drawn from.
DEFAULT_MIN_COMPARED_ROWS = 50

_LN2 = math.log(2.0)
_LN10 = math.log(10.0)


@dataclass(frozen=True)
class EffectConsistencyPolicy:
    """When a source's effect is refused for contradicting its own p-value."""

    max_median_abs_delta: float = DEFAULT_MAX_MEDIAN_ABS_DELTA
    min_compared_rows: int = DEFAULT_MIN_COMPARED_ROWS

    def __post_init__(self) -> None:
        if not (math.isfinite(self.max_median_abs_delta) and self.max_median_abs_delta > 0):
            raise ValueError(
                f"max_median_abs_delta must be positive, got {self.max_median_abs_delta!r}"
            )
        if self.min_compared_rows < 1:
            raise ValueError(f"min_compared_rows must be >= 1, got {self.min_compared_rows!r}")

    def as_fingerprint(self) -> dict[str, float | int]:
        return {
            "max_median_abs_delta": self.max_median_abs_delta,
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
        skip = self._skip(considered, policy)
        if skip is not None:
            return EffectConsistency(considered, 0, False, None, None, skip)
        z = np.array([row[0] for row in retained])
        delta = log10_two_sided_p(z) - np.array([row[1] for row in retained])
        median_abs = float(np.median(np.abs(delta)))
        return EffectConsistency(
            n_rows_considered=considered,
            n_rows_compared=len(retained),
            sampled=self._sample.sampled,
            median_abs_delta=median_abs,
            median_delta=float(np.median(delta)),
            skip_reason=None,
            refused=median_abs > policy.max_median_abs_delta,
        )

    def _skip(self, considered: int, policy: EffectConsistencyPolicy) -> ConsistencySkip | None:
        if self.n_with_source_p == 0:
            return ConsistencySkip.NO_SOURCE_P_VALUE
        if considered == 0:
            return ConsistencySkip.ALL_SE_FROM_P_VALUE
        if considered < policy.min_compared_rows:
            return ConsistencySkip.TOO_FEW_ROWS
        return None
