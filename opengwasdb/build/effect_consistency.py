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

**Every build runs the check too (#510).** A Store built from a bundle whose
Resolution Records predate the check never had it: OGS-00032 and OGS-00033 were
resolved by opengwasdb 0.3.3, before the check shipped, and published ten odds
ratios stored as log-odds and one effect off by 10^5. So each builder hands
`verify_effect_consistency` one `ConsistencySource` per Analysis, beside its EAF
orientation check, and the shape is that check's: `check_effect_consistency`
measures, `enforce_effect_consistency` refuses the build naming every Analysis
that contradicts its p-value, and `apply_consistency_evidence` stamps each
Analysis's `analyses.tsv` columns. The build reads each source's first
`DEFAULT_BUILD_SCAN_ROWS` rows through the resolver's own projection and
admission, and samples them as the resolver does.
"""

from __future__ import annotations

import csv
import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm

from opengwasdb.admission import AdmissionCounts, AnalysisRules, SourceRows, counting_gate
from opengwasdb.build.hash_sample import HashSample
from opengwasdb.build.ordered_pool import fork_each
from opengwasdb.build.phenotype_sd import DEFAULT_EVIDENCE_SAMPLE
from opengwasdb.encoding.timing import log_phase
from opengwasdb.model.analyses import Analysis
from opengwasdb.model.enums import EffectConsistencyOutcome
from opengwasdb.model.manifest_columns import manifest_column
from opengwasdb.model.p_value_test import P_VALUE_TEST_COLUMN, PValueTest
from opengwasdb.readers.registry import SourceSpec, open_source
from opengwasdb.readers.tabular import MetricsChunk

log = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_BUILD_SCAN_ROWS",
    "DEFAULT_MAX_MEDIAN_ABS_DELTA",
    "DEFAULT_MAX_MEDIAN_DELTA",
    "DEFAULT_MIN_COMPARED_ROWS",
    "DEFAULT_EFFECT_CONSISTENCY_POLICY",
    "MR_MEGA_REFUSAL",
    "ConsistencyEvidence",
    "ConsistencySkip",
    "ConsistencySample",
    "ConsistencySource",
    "EffectConsistency",
    "EffectConsistencyError",
    "EffectConsistencyPolicy",
    "EffectConsistencyReport",
    "admit_block",
    "apply_consistency_evidence",
    "check_effect_consistency",
    "check_sample_bound",
    "declared_p_value_tests",
    "enforce_effect_consistency",
    "log10_two_sided_p",
    "refusal_text",
    "skipped_effect_consistency",
    "verify_effect_consistency",
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

#: Rows of each source a build reads for the check: the prefix ADR 0069's
#: calibration measured (1,000,000 rows per source), whose medians its whole-file
#: confirmation found within 0.0055 (clean) and 0.0128 (refused) of the whole
#: file's, against thresholds of 1.0 and 0.1. About 5 s per GWAS-SSF source.
DEFAULT_BUILD_SCAN_ROWS = 1_000_000

#: `refused_by` for an Analysis whose declared p-value test is MR-MEGA: refused
#: unread, as the resolver refuses it (issue #371).
MR_MEGA_REFUSAL = "p_value_test_mr_mega"

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

    def admit(self, chunk: MetricsChunk, keep: np.ndarray | None = None) -> None:
        """Offer one block of rows the build would admit: all of them, or those `keep` marks."""
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            z = chunk.beta / chunk.se
        p = chunk.p_value
        usable = np.isfinite(z) & (chunk.se > 0) & np.isfinite(p) & (p > 0.0) & (p <= 1.0)
        if keep is not None:
            usable &= keep
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


def refusal_text(measured: EffectConsistency, policy: EffectConsistencyPolicy) -> str:
    """Why `measured` was refused, in the words a Resolution Record and a build both use."""
    n = measured.n_rows_compared
    if measured.refused_by == "median_delta":
        return (
            "stored z is deflated against the source's own p-value: the median signed "
            f"(log10 p implied by beta/se - log10 p reported) is {measured.median_delta:.3g} "
            f"over {n} rows, above the threshold {policy.max_median_delta:g} "
            f"(median |delta| {measured.median_abs_delta:.3g}). beta/se carries less "
            "signal than the reported p-value: the standard error is probably not this "
            "effect's, or is on another scale."
        )
    return (
        "effect contradicts the source's own p-value: the median "
        f"|log10 p implied by beta/se - log10 p reported| is {measured.median_abs_delta:.3g} "
        f"(median signed {measured.median_delta:.3g}) over {n} "
        f"rows, above the threshold {policy.max_median_abs_delta:g}. The source's effect "
        "column is probably on another scale (an odds ratio in `beta`, or log-odds "
        "in `odds_ratio`) or its standard error is not this effect's."
    )


def admit_block(observed: MetricsChunk, rules: AnalysisRules) -> tuple[np.ndarray, AdmissionCounts]:
    """One block's keep mask and counts, by Association Admission, in the builder's orientation.

    Both come from one counting gate's judgement, so the counts the record
    reports and the rows the scan keeps cannot disagree (stores #176). The
    frequency is oriented exactly as `stream_associations` orients the
    builder's `eaf` (`1.0 - af_alt` on a flipped row): MAF is symmetric in exact
    arithmetic but not in floating point, so a row whose MAF lies on the
    threshold would otherwise be kept by one and dropped by the other.
    """
    gate = counting_gate(rules)
    keep = gate.source_rows(
        SourceRows(
            eaf=observed.a1_frequency(),
            imputation_score=observed.imputation_score,
            imputation_status=observed.imputation_score_status,
        )
    )
    return keep, gate.counts


# --- The build's check (#510): measure, enforce, stamp, as EAF orientation does ---


class EffectConsistencyError(ValueError):
    """A build refused because an Analysis's effect contradicts its source's p-value."""


def declared_p_value_tests(manifest_path: str | Path) -> dict[str, PValueTest | None]:
    """Each Analysis's declared `p_value_test` in a build manifest, by Analysis id.

    The column is optional in a build manifest, unlike the resolver's (ADR 0085):
    an Analysis with no declaration -- no column, or a blank cell -- is `None`,
    which the check reads as "checked" (`ConsistencySource`). A value that is not
    a test is refused, naming its Analysis.
    """
    with open(manifest_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        id_column = manifest_column(reader.fieldnames or [], "analysis_id")
        rows = list(reader)
    if id_column is None:
        raise ValueError(f"manifest {manifest_path} names no analysis_id (or trait_id) column")
    declared: dict[str, PValueTest | None] = {}
    for row in rows:
        analysis_id = row.get(id_column, "")
        raw = (row.get(P_VALUE_TEST_COLUMN) or "").strip()
        try:
            declared[analysis_id] = PValueTest(raw) if raw else None
        except ValueError as exc:
            allowed = ", ".join(test.value for test in PValueTest)
            raise ValueError(
                f"manifest {manifest_path}: analysis {analysis_id!r} has invalid "
                f"{P_VALUE_TEST_COLUMN} {raw!r}; expected one of {allowed}, or blank"
            ) from exc
    return declared


@dataclass(frozen=True)
class ConsistencySource:
    """One Analysis as a builder hands it to the check.

    `p_value_test` is the build manifest's declaration (ADR 0085), `None` where
    the manifest declares none. An undeclared Analysis is checked: if its p is a
    joint test after all, the build refuses it loudly and the refusal names the
    declaration that would skip it, so silence never stores a wrong z.
    """

    spec: SourceSpec
    rules: AnalysisRules
    p_value_test: PValueTest | None = None

    @property
    def analysis_id(self) -> str:
        return self.spec.analysis_id


@dataclass(frozen=True)
class ConsistencyEvidence:
    """One Analysis's outcome, and the measurement it was drawn from.

    `reason` is the skip's reason, or the rule that refused; blank on a pass.
    `measured` is `None` only for an Analysis refused unread (MR-MEGA).
    """

    analysis_id: str
    outcome: EffectConsistencyOutcome
    reason: str
    measured: EffectConsistency | None

    @classmethod
    def of(cls, analysis_id: str, measured: EffectConsistency) -> ConsistencyEvidence:
        if measured.skip_reason is not None:
            outcome, reason = EffectConsistencyOutcome.SKIPPED, measured.skip_reason.value
        elif measured.refused:
            outcome, reason = EffectConsistencyOutcome.REFUSED, str(measured.refused_by)
        else:
            outcome, reason = EffectConsistencyOutcome.PASSED, ""
        return cls(analysis_id, outcome, reason, measured)

    def columns(self) -> dict[str, str]:
        """The Analysis's `analyses.tsv` cells (`EFFECT_CONSISTENCY_COLUMNS`)."""
        m = self.measured
        measured = {"median_abs_delta": "", "median_delta": "", "n": ""}
        if m is not None and m.median_abs_delta is not None and m.median_delta is not None:
            measured = {
                "median_abs_delta": f"{m.median_abs_delta:.6g}",
                "median_delta": f"{m.median_delta:.6g}",
                "n": str(m.n_rows_compared),
            }
        return {
            "effect_consistency": self.outcome.value,
            "effect_consistency_reason": self.reason,
            **{f"effect_consistency_{name}": value for name, value in measured.items()},
        }

    def payload(self) -> dict[str, Any]:
        """A JSON form a resumed build reloads (`from_payload`)."""
        m = self.measured
        return {
            "analysis_id": self.analysis_id,
            "outcome": self.outcome.value,
            "reason": self.reason,
            "measured": None if m is None else {
                "n_rows_considered": m.n_rows_considered,
                "n_rows_compared": m.n_rows_compared,
                "sampled": m.sampled,
                "median_abs_delta": m.median_abs_delta,
                "median_delta": m.median_delta,
                "skip_reason": None if m.skip_reason is None else m.skip_reason.value,
                "refused": m.refused,
                "refused_by": m.refused_by,
            },
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ConsistencyEvidence:
        m = payload["measured"]
        measured = None if m is None else EffectConsistency(
            n_rows_considered=int(m["n_rows_considered"]),
            n_rows_compared=int(m["n_rows_compared"]),
            sampled=bool(m["sampled"]),
            median_abs_delta=m["median_abs_delta"],
            median_delta=m["median_delta"],
            skip_reason=None if m["skip_reason"] is None else ConsistencySkip(m["skip_reason"]),
            refused=bool(m["refused"]),
            refused_by=m["refused_by"],
        )
        return cls(
            str(payload["analysis_id"]),
            EffectConsistencyOutcome(payload["outcome"]),
            str(payload["reason"]),
            measured,
        )


@dataclass(frozen=True)
class EffectConsistencyReport:
    """Every Analysis's evidence, in the order the builder handed them over."""

    evidence: tuple[ConsistencyEvidence, ...]
    policy: EffectConsistencyPolicy = DEFAULT_EFFECT_CONSISTENCY_POLICY

    @property
    def refusals(self) -> list[ConsistencyEvidence]:
        return [e for e in self.evidence if e.outcome is EffectConsistencyOutcome.REFUSED]

    def payload(self) -> dict[str, Any]:
        """A JSON form a resumed build reloads (`from_payload`)."""
        return {"analyses": [e.payload() for e in self.evidence]}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> EffectConsistencyReport:
        return cls(tuple(ConsistencyEvidence.from_payload(item) for item in payload["analyses"]))


@dataclass(frozen=True)
class _ScanBounds:
    k: int
    scan_rows: int
    policy: EffectConsistencyPolicy


def measure_source(
    source: ConsistencySource,
    *,
    k: int = DEFAULT_EVIDENCE_SAMPLE,
    scan_rows: int = DEFAULT_BUILD_SCAN_ROWS,
    policy: EffectConsistencyPolicy = DEFAULT_EFFECT_CONSISTENCY_POLICY,
) -> ConsistencyEvidence:
    """One Analysis's evidence, from its source's first `scan_rows` rows.

    The rows are the resolver's: its metrics projection, admitted by the
    Analysis's own INFO and MAF rules, sampled bottom-`k` by ALID hash.
    """
    declared = source.p_value_test
    if declared is PValueTest.MR_MEGA:
        return ConsistencyEvidence(
            source.analysis_id, EffectConsistencyOutcome.REFUSED, MR_MEGA_REFUSAL, None
        )
    sample = ConsistencySample(k)
    if declared is PValueTest.JOINT_2DF:
        return ConsistencyEvidence.of(
            source.analysis_id, sample.skipped(ConsistencySkip.JOINT_TEST_P_VALUE)
        )
    reader = open_source(
        source.spec, imputation_score_declaration=source.rules.info.imputation_score_declaration
    )
    stream = reader.stream_metric_chunks()
    read = 0
    try:
        for chunk in stream:
            keep, _counts = admit_block(chunk, source.rules)
            remaining = scan_rows - read
            if remaining < len(chunk):
                keep = keep.copy()
                keep[remaining:] = False
            sample.admit(chunk, keep)
            read += len(chunk)
            if read >= scan_rows:
                break
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()
    return ConsistencyEvidence.of(source.analysis_id, sample.measure(policy))


def _measure_one(
    bounds: _ScanBounds, item: tuple[int, ConsistencySource]
) -> tuple[int, ConsistencyEvidence]:
    index, source = item
    return index, measure_source(
        source, k=bounds.k, scan_rows=bounds.scan_rows, policy=bounds.policy
    )


def check_effect_consistency(
    sources: Sequence[ConsistencySource],
    *,
    n_workers: int = 1,
    k: int = DEFAULT_EVIDENCE_SAMPLE,
    scan_rows: int = DEFAULT_BUILD_SCAN_ROWS,
    policy: EffectConsistencyPolicy = DEFAULT_EFFECT_CONSISTENCY_POLICY,
) -> EffectConsistencyReport:
    """Measure every Analysis, `n_workers` sources at a time, in the order given."""
    check_sample_bound(k, policy)
    if scan_rows < policy.min_compared_rows:
        raise ValueError(
            f"scan_rows ({scan_rows}) is below the effect-consistency min_compared_rows "
            f"({policy.min_compared_rows}): no Analysis could ever be checked"
        )
    found: dict[int, ConsistencyEvidence] = {}
    bounds = _ScanBounds(k, scan_rows, policy)
    for index, evidence in fork_each(_measure_one, bounds, enumerate(sources), n_workers):
        found[index] = evidence
    return EffectConsistencyReport(tuple(found[i] for i in range(len(sources))), policy)


def enforce_effect_consistency(report: EffectConsistencyReport) -> None:
    """Fail the build on every Analysis whose effect contradicts its p-value.

    Every refusal is named in one error, as `enforce_eaf_orientation` names every
    flipped Analysis, so the registry can exclude them all by reason at once. A
    skip is recorded and logged, never raised: it is a fact about the source.
    """
    for evidence in report.evidence:
        if evidence.outcome is EffectConsistencyOutcome.SKIPPED:
            log.info(
                "Effect consistency skipped for analysis %s: %s",
                evidence.analysis_id, evidence.reason,
            )
    refusals = report.refusals
    if not refusals:
        return
    detail = "; ".join(_refusal_line(e, report.policy) for e in refusals)
    raise EffectConsistencyError(
        f"{len(refusals)} Analysis/Analyses contradict their source's own p-value "
        f"(effect_consistency=refused, ADR 0069): {detail}. Exclude them by this reason, "
        "or, where an Analysis's p-value is a joint 2-df test, declare "
        "p_value_test=joint_2df for it in the build manifest"
    )


def _refusal_line(evidence: ConsistencyEvidence, policy: EffectConsistencyPolicy) -> str:
    if evidence.measured is None:
        return (
            f"{evidence.analysis_id}: refused by {evidence.reason} -- MR-MEGA's beta/se is a "
            "t-statistic, not a z (issue #371)"
        )
    return f"{evidence.analysis_id}: refused by {evidence.reason} -- " + refusal_text(
        evidence.measured, policy
    )


def verify_effect_consistency(
    sources: Sequence[ConsistencySource],
    *,
    n_workers: int = 1,
    k: int = DEFAULT_EVIDENCE_SAMPLE,
    scan_rows: int = DEFAULT_BUILD_SCAN_ROWS,
    policy: EffectConsistencyPolicy = DEFAULT_EFFECT_CONSISTENCY_POLICY,
) -> EffectConsistencyReport:
    """Run the check and stop the build if it refuses: the entry point every builder uses."""
    with log_phase(log, "Effect consistency verification"):
        report = check_effect_consistency(
            sources, n_workers=n_workers, k=k, scan_rows=scan_rows, policy=policy
        )
        enforce_effect_consistency(report)
    return report


def skipped_effect_consistency(
    analysis_ids: Iterable[str], reason: ConsistencySkip
) -> EffectConsistencyReport:
    """The report of a build whose Source Format carries no p-value to compare.

    A BESD build's metrics carry none (ADR 0069), so its Analyses are stamped
    `skipped` with the reason, never left blank as if no check had been asked.
    """
    return EffectConsistencyReport(
        tuple(
            ConsistencyEvidence(
                analysis_id, EffectConsistencyOutcome.SKIPPED, reason.value,
                EffectConsistency(0, 0, False, None, None, reason),
            )
            for analysis_id in analysis_ids
        )
    )


def apply_consistency_evidence(
    analyses: Sequence[Analysis], report: EffectConsistencyReport
) -> list[Analysis]:
    """Stamp each Analysis's effect-consistency columns from the report, by `analysis_id`.

    Every Analysis the build stores must have evidence: one the report does not
    name would be written blank, which reads as a Store built before the check,
    so it raises rather than stamp nothing.
    """
    by_id = {e.analysis_id: e for e in report.evidence}
    missing = [a.analysis_id for a in analyses if a.analysis_id not in by_id]
    if missing:
        raise ValueError(
            f"no effect-consistency evidence for {len(missing)} Analysis/Analyses "
            f"(first {missing[0]!r}): the build must check every Analysis it stores"
        )
    stamped = []
    for a in analyses:
        cells = by_id[a.analysis_id].columns()
        stamped.append(
            replace(
                a,
                effect_consistency=cells["effect_consistency"],
                effect_consistency_reason=cells["effect_consistency_reason"],
                effect_consistency_median_abs_delta=cells["effect_consistency_median_abs_delta"],
                effect_consistency_median_delta=cells["effect_consistency_median_delta"],
                effect_consistency_n=cells["effect_consistency_n"],
            )
        )
    return stamped
