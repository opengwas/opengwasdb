"""Resolve one Analysis's pre-build Analytical Metadata from one source scan (issue #207).

Phase B writes an Analysis's `assigned_ancestry` and its
`original_sd`/`original_sd_method` into `analyses.tsv` before a Release Bundle is
accepted (ADR 0029; `opengwasdb-stores`' Phase B/Phase A split). Both are
computed from the same source rows -- a frequency for the ancestry fit, a
standard error for the phenotype-SD median -- and the two stages used to open
the file separately, which on a genome-wide source means decompressing and
parsing it twice for one number each.

This module is the one seam that reads a source once and produces both. It
*computes*; it does not decide:

* **The method tier is the caller's.** `AnalysisRequest.original_sd_method` names
  the ADR-0029 tier to apply, exactly as the manifest column does
  (`opengwasdb.build.phenotype_sd_pipeline` reads that same column). Nothing here
  falls back from source AF to reference AF to a beta spread: a fallback recorded
  under the wrong method is a plausible number wearing the wrong name, which is
  the failure `original_sd_method` exists to prevent.
* **The acceptance policy is the caller's.** No tolerance, no dispersion
  threshold, no verdict, no `exclude_from_build`. The resolution reports the
  computation and the counts behind it; the caller applies its thresholds to
  them (ADR 0029's computation-versus-acceptance split).
* **The extraction panel is the caller's.** `extraction_panel` bounds which sites
  the ancestry fit sees. Passing the fixed 10,000-site QC panel keeps
  per-Analysis ancestry memory independent of both the source's row count and
  the reference's ~5.8M variants; whether that panel is scientifically
  interchangeable with the full reference is a question this module cannot
  answer and must not assume.

Memory is bounded by configuration, not by the source. The ancestry fit holds one
frequency per *panel* site; the phenotype-SD evidence is a deterministic
bottom-`k`-by-hash sample of the qualifying rows (`evidence_sample`, sharing
`opengwasdb.build.eaf_orientation.site_hash`'s selection rule), so a 40M-row file
and a 4M-row file cost the same. Above the bound the resolution says so --
`evidence_sampled`, `n_evidence_considered` -- rather than presenting a sampled
median as a whole-file one, and below it nothing is sampled at all.

Results are deterministic and independent of worker count: one Analysis is
resolved by one scan of one file, in file order, with the sample chosen by
variant hash rather than by position or arrival order.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from opengwasdb.ancestry.mixture import AncestryAssignment, Gates, assign_ancestry
from opengwasdb.ancestry.reference import AncestryReference
from opengwasdb.build.eaf_orientation import site_hash
from opengwasdb.build.phenotype_sd import (
    ESTIMATION_METHODS,
    PhenotypeSdEstimate,
    estimate_phenotype_sd,
    has_usable_sample_size,
    se_scale_samples,
)
from opengwasdb.model.enums import OriginalSdMethod, StoredEffectScale
from opengwasdb.readers.gwas_vcf import is_palindromic
from opengwasdb.readers.tabular import TabularMetricsRow

__all__ = [
    "AfReference",
    "AnalysisRequest",
    "AnalysisResolution",
    "DEFAULT_EVIDENCE_SAMPLE",
    "MetricsReader",
    "PhenotypeSdResolution",
    "ScanDiagnostics",
    "SdReason",
    "SdStatus",
    "resolve_analysis",
]

#: How many qualifying evidence rows one Analysis's phenotype-SD estimate is
#: drawn from. A robust median-implied-SD estimate needs nowhere near this many
#: sites (ADR 0029's estimator is a median over per-variant values, not a fit),
#: so the bound costs the estimate nothing measurable while capping what a
#: 40M-row file costs in memory. It matches
#: `opengwasdb.build.eaf_orientation.DEFAULT_SAMPLE_SITES`'s scale deliberately:
#: both are "an Analysis's own variants, bounded".
DEFAULT_EVIDENCE_SAMPLE = 20_000


class SdStatus(StrEnum):
    """Whether a phenotype SD was computed for this Analysis (issue #207)."""

    ESTIMATED = "estimated"
    SKIPPED = "skipped"
    UNAVAILABLE = "unavailable"


class SdReason(StrEnum):
    """Why no phenotype SD was computed.

    A fact about the inputs, never a verdict: `skipped` codes name a decision the
    caller already made or an input the caller did not declare, `unavailable`
    codes name evidence or a sample size that cannot support the estimate.
    """

    #: The Analysis is not quantitative, so there is no phenotype SD to estimate
    #: (`stored_effect_scale` is `log_or`/`log_hazard`), whatever tier was named.
    NON_QUANTITATIVE = "non_quantitative_effect_scale"
    #: The caller declared the scale already (`declared_standardised`) or took the
    #: value from the source (`source_provided`); there is nothing to estimate.
    SCALE_ALREADY_DECLARED = "scale_already_declared"
    #: No estimation tier was named, so no estimation was requested.
    NO_TIER_SELECTED = "no_tier_selected"
    #: The requested tier needs a declared reference for this Analysis's Assigned
    #: Ancestry, and the caller declared none (ADR 0019).
    NO_REFERENCE_RESOURCE_FOR_ANCESTRY = "no_reference_resource_for_ancestry"
    #: ADR-0029's formula is `sd_scale / sqrt(N)` as one quantity; without a
    #: usable `N` there is nothing to report.
    NO_USABLE_SAMPLE_SIZE = "no_usable_sample_size"
    #: The tier applies, but no retained row carried the values it needs.
    NO_QUALIFYING_EVIDENCE = "no_qualifying_evidence"


@dataclass(frozen=True)
class AfReference:
    """One declared reference allele-frequency source, keyed to Assigned Ancestry.

    `reference_id` is whatever stable name the caller declared it under. This
    module never invents one and never resolves a resource itself: which
    reference an ancestry uses is a release decision, and substituting one the
    caller did not declare is exactly the silent substitution ADR 0019 forbids.
    `frequencies` is A1-oriented `{canonical alid: frequency}`, the shape
    `opengwasdb.build.phenotype_sd_pipeline.load_af_reference` returns.
    """

    reference_id: str
    frequencies: Mapping[str, float]


@dataclass(frozen=True)
class AnalysisRequest:
    """One Analysis's identity and the caller's own method choices.

    `sample_size` is `None` when the manifest carries none -- never `0`, which
    would read as a number (ADR 0029). `stored_effect_scale` is Analytical
    Metadata the caller resolved from the manifest, not something a source file
    declares (issue #17).
    """

    analysis_id: str
    source_file: str | Path
    sample_size: float | None
    original_sd_method: OriginalSdMethod
    stored_effect_scale: StoredEffectScale = StoredEffectScale.SD


@dataclass(frozen=True)
class ScanDiagnostics:
    """What one source scan saw, before any threshold is applied to it."""

    source_file: str
    #: Rows the source yielded that named a canonical variant. Rows naming no
    #: variant at all are dropped by the reader and are not counted here.
    rows_read: int
    #: Distinct panel sites this source contributed a frequency at -- how much of
    #: a bounded `extraction_panel` the file actually covers.
    ancestry_sites: int


@dataclass(frozen=True)
class PhenotypeSdResolution:
    """The phenotype-SD half of one Analysis's resolution (ADR 0029).

    `status` separates the two ways there is no number, which must not be
    collapsed into one: `skipped` when the tier does not apply to this Analysis
    (a log-OR Analysis has no phenotype SD to standardise to; a caller that
    already declared the scale has nothing to estimate), `unavailable` when the
    tier applies but the sample size or the evidence cannot support it. Reading a
    case-control Analysis and a broken one as the same outcome is how a broken
    one gets through.

    `estimate` is present exactly when `status` is `ESTIMATED`, and `reason` is
    `None` exactly then. `n_estimate_inputs` counts the per-variant values the
    estimate is a median over, after the tier's own validity rule -- the number a
    caller's minimum-overlap threshold applies to.
    """

    status: SdStatus
    reason: SdReason | None
    estimate: PhenotypeSdEstimate | None
    reference_id: str
    n_evidence_considered: int
    n_estimate_inputs: int
    evidence_sampled: bool


@dataclass(frozen=True)
class AnalysisResolution:
    """Everything one source scan produced for one Analysis (issue #207).

    A non-empty `error` means the source could not be read at all: `ancestry` and
    `phenotype_sd` are then both `None`, and the caller records a controlled
    failure rather than inventing an outcome. Otherwise both are set -- an
    ancestry fit that failed its gates is still a fit, with `gate_reason` naming
    why (ADR 0028), and a skipped or unavailable phenotype SD is still a reported
    outcome.
    """

    analysis_id: str
    diagnostics: ScanDiagnostics
    ancestry: AncestryAssignment | None = None
    phenotype_sd: PhenotypeSdResolution | None = None
    error: str = ""


@runtime_checkable
class MetricsReader(Protocol):
    """A source reader that can stream one row's identity and statistics.

    Narrower than `SourceReader` on purpose. `stream_associations` cannot serve
    this resolver: it carries a `z` rather than the beta behind it, and it drops
    every row whose beta is unusable, which is a row ancestry assignment can
    still read a frequency from. No single-scan equivalent exists for a GWAS-VCF
    either -- the cheap read there is a targeted `bcftools -R` at reference sites,
    not a full-file row scan -- so a tabular reader satisfies this and
    `GwasVcfReader` does not, and `resolve_analysis` refuses one loudly rather
    than reading a 100 GB VCF row by row.
    """

    def stream_metrics(self) -> Iterator[TabularMetricsRow]: ...


@dataclass
class _Scan:
    """One source scan's mutable state, readable even if the scan raises mid-file.

    A truncated or corrupt source is a controlled per-Analysis failure, and the
    rows read before it failed are part of why; keeping them in an object the
    caller already holds means the failure path reports them instead of zeros.
    """

    panel_af: dict[str, float] = field(default_factory=dict)
    rows_read: int = 0


@dataclass
class _EvidenceSample:
    """A deterministic, bounded bottom-`k`-by-hash sample of evidence rows.

    One Analysis's qualifying rows, retained up to `k` and chosen by
    `site_hash` -- the rule `opengwasdb.build.eaf_orientation.select_sites`
    already uses to pick an Analysis's own variants deterministically, shared
    rather than re-spelled so a sample drawn here and a sample drawn there agree
    on which variants exist. Equal hashes are broken by source row order, which
    is also what keeps a repeated ALID's two rows both eligible.

    Below the bound nothing is sampled: every qualifying row is retained, in file
    order, so the arrays handed to the estimator are the ones
    `opengwasdb.build.phenotype_sd_pipeline` would have built from the same file
    and the estimate is identical rather than merely close.
    """

    k: int
    tier: OriginalSdMethod
    considered: int = 0
    _sequence: int = 0
    # `(-hash, sequence, se, af_alt, beta, alid)`. The sequence is unique, so the
    # heap never compares the payloads -- which is why a repeated ALID (same
    # hash) still gets two entries rather than one.
    _heap: list[tuple[int, int, float, float, float, str]] = field(default_factory=list)

    def admit(self, row: TabularMetricsRow) -> None:
        """Offer one row; it is retained only if it qualifies and outranks the heap."""
        if not self._qualifies(row):
            return
        self.considered += 1
        self._sequence += 1
        entry = (
            -site_hash(row.alid),
            self._sequence,
            _or_nan(row.se),
            _or_nan(row.af_alt),
            _or_nan(row.beta),
            row.alid,
        )
        if len(self._heap) < self.k:
            heapq.heappush(self._heap, entry)
        elif entry > self._heap[0]:
            heapq.heapreplace(self._heap, entry)

    def retained(self) -> list[tuple[float, float, float, str]]:
        """`(se, af_alt, beta, alid)` per retained row, in source row order."""
        return [
            (se, af, beta, alid)
            for _hash, _sequence, se, af, beta, alid in sorted(self._heap, key=_entry_sequence)
        ]

    def _qualifies(self, row: TabularMetricsRow) -> bool:
        """The row rule both stages share: `stream_associations`' own admission.

        A row needs a usable standard error and beta to be evidence at all --
        that is the filter the association stream applies, and matching it is what
        keeps a below-the-bound estimate identical to the one
        `estimate-phenotype-sd` produces from the same file. The source-MAF tier
        additionally needs a usable source frequency, which its own evidence
        gathering also requires. A tier that is not computed at all retains
        nothing.
        """
        if self.tier not in ESTIMATION_METHODS:
            return False
        if row.se is None or row.beta is None:
            return False
        if self.tier is OriginalSdMethod.ESTIMATED_FROM_SOURCE_MAF:
            return row.af_alt is not None
        return True


def _or_nan(value: float | None) -> float:
    """A missing value as NaN, so the heap stays one numeric shape (ADR 0029)."""
    return float("nan") if value is None else value


def _entry_sequence(entry: tuple[int, int, float, float, float, str]) -> int:
    return entry[1]


def _diagnostics(request: AnalysisRequest, scan: _Scan) -> ScanDiagnostics:
    return ScanDiagnostics(
        source_file=str(request.source_file),
        rows_read=scan.rows_read,
        ancestry_sites=len(scan.panel_af),
    )


def _scan(
    reader: MetricsReader, panel: Collection[str], scan: _Scan, evidence: _EvidenceSample
) -> None:
    """One pass over the source, feeding the ancestry fit and the SD evidence.

    This is the whole point of the module: the two stages read different things
    from the same row, so they are accumulated together rather than by two
    scans of a genome-wide file.
    """
    for row in reader.stream_metrics():
        scan.rows_read += 1
        _accumulate_ancestry(row, panel, scan.panel_af)
        evidence.admit(row)


def _accumulate_ancestry(
    row: TabularMetricsRow, panel: Collection[str], panel_af: dict[str, float]
) -> None:
    """Record this row's A1-oriented frequency when it is panel evidence.

    The filter is `opengwasdb.readers.tabular.extract_at_sites`'s, unchanged: a
    usable frequency, a usable standard error, no palindromic pair (neither the
    reader nor its callers have strand information to resolve A/T or C/G), and
    the site on the panel. Keeping it identical is what lets an Analysis resolved
    here match the same Analysis assigned by `assign_from_source`.

    Later rows win a repeated ALID, which is also what a dict built by
    `extract_at_sites` does.
    """
    if row.se is None or row.af_alt is None:
        return
    if is_palindromic(row.ref, row.alt) or row.alid not in panel:
        return
    panel_af[row.alid] = (1.0 - row.af_alt) if row.flipped else row.af_alt


def _skip_reason(request: AnalysisRequest) -> SdReason | None:
    """Why this Analysis has no phenotype SD to estimate, or `None` if it has one.

    Neither outcome here is an error: a non-quantitative Analysis has no phenotype
    SD to standardise to whatever tier was named, and a caller that already has
    the value has nothing left to estimate.
    """
    if request.stored_effect_scale in (StoredEffectScale.LOG_OR, StoredEffectScale.LOG_HAZARD):
        return SdReason.NON_QUANTITATIVE
    if request.original_sd_method is OriginalSdMethod.BINARY_TRAIT:
        return SdReason.NON_QUANTITATIVE
    if request.original_sd_method in (
        OriginalSdMethod.DECLARED_STANDARDISED,
        OriginalSdMethod.SOURCE_PROVIDED,
    ):
        return SdReason.SCALE_ALREADY_DECLARED
    if request.original_sd_method not in ESTIMATION_METHODS:
        return SdReason.NO_TIER_SELECTED
    return None


def _reference_for_ancestry(
    ancestry: AncestryAssignment, af_references: Mapping[str, AfReference]
) -> AfReference | None:
    """The declared reference for this Analysis's Assigned Ancestry, if any.

    An Unassigned Analysis has no ancestry to match a reference to, so it gets
    the same explicit outcome as an assigned one whose ancestry the caller
    declared no reference for -- never the reference of a nearby ancestry.
    """
    if ancestry.assigned_ancestry is None:
        return None
    return af_references.get(ancestry.assigned_ancestry)


def _tier_arrays(
    method: OriginalSdMethod,
    retained: list[tuple[float, float, float, str]],
    reference: Mapping[str, float] | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """`(se, af, beta)` for the requested tier, from the retained evidence.

    Mirrors `opengwasdb.build.phenotype_sd_pipeline`'s own evidence gathering,
    one tier at a time: the source-MAF tier takes the source's own frequency, the
    reference-MAF tier takes the declared reference's frequency at the row's
    canonical ALID (resolved *after* the ancestry fit chose which reference that
    is -- which is why the source is never re-read to find it), and the beta tier
    takes the oriented beta. A row a tier has no value for is left out rather
    than filled in.
    """
    if method is OriginalSdMethod.ESTIMATED_FROM_BETA_DISTRIBUTION:
        betas = [beta for _se, _af, beta, _alid in retained if math.isfinite(beta)]
        return None, None, np.asarray(betas, dtype=np.float64)
    se_values: list[float] = []
    af_values: list[float] = []
    for se, source_af, _beta, alid in retained:
        frequency: float | None
        if method is OriginalSdMethod.ESTIMATED_FROM_SOURCE_MAF:
            frequency = source_af
        else:
            frequency = None if reference is None else reference.get(alid)
        if frequency is None or not math.isfinite(frequency):
            continue
        se_values.append(se)
        af_values.append(frequency)
    return (
        np.asarray(se_values, dtype=np.float64),
        np.asarray(af_values, dtype=np.float64),
        None,
    )


def _estimate_inputs(
    method: OriginalSdMethod,
    se: np.ndarray | None,
    af: np.ndarray | None,
    beta: np.ndarray | None,
) -> int:
    """How many per-variant values the estimate is a median over.

    `se_scale_samples` is the estimator's own validity rule for the two AF tiers
    (a finite `se` and a frequency strictly inside (0, 1)), reused rather than
    restated so the count cannot disagree with the median it describes.
    """
    if method is OriginalSdMethod.ESTIMATED_FROM_BETA_DISTRIBUTION:
        return 0 if beta is None else int(np.isfinite(beta).sum())
    if se is None or af is None:
        return 0
    return int(se_scale_samples(se, af).size)


def _no_sd(
    status: SdStatus,
    reason: SdReason,
    evidence: _EvidenceSample,
    reference_id: str,
) -> PhenotypeSdResolution:
    return PhenotypeSdResolution(
        status=status,
        reason=reason,
        estimate=None,
        reference_id=reference_id,
        n_evidence_considered=evidence.considered,
        n_estimate_inputs=0,
        evidence_sampled=evidence.considered > evidence.k,
    )


def _resolve_phenotype_sd(
    request: AnalysisRequest,
    ancestry: AncestryAssignment,
    evidence: _EvidenceSample,
    af_references: Mapping[str, AfReference],
) -> PhenotypeSdResolution:
    """Apply the caller's requested tier to the evidence one scan collected."""
    skipped = _skip_reason(request)
    if skipped is not None:
        return _no_sd(SdStatus.SKIPPED, skipped, evidence, reference_id="")
    method = request.original_sd_method
    reference = (
        _reference_for_ancestry(ancestry, af_references)
        if method is OriginalSdMethod.ESTIMATED_FROM_REFERENCE_MAF
        else None
    )
    if method is OriginalSdMethod.ESTIMATED_FROM_REFERENCE_MAF and reference is None:
        return _no_sd(
            SdStatus.SKIPPED, SdReason.NO_REFERENCE_RESOURCE_FOR_ANCESTRY, evidence, ""
        )
    reference_id = "" if reference is None else reference.reference_id
    if not has_usable_sample_size(request.sample_size):
        return _no_sd(SdStatus.UNAVAILABLE, SdReason.NO_USABLE_SAMPLE_SIZE, evidence, reference_id)
    se, af, beta = _tier_arrays(
        method, evidence.retained(), None if reference is None else reference.frequencies
    )
    n_inputs = _estimate_inputs(method, se, af, beta)
    if n_inputs == 0:
        return _no_sd(SdStatus.UNAVAILABLE, SdReason.NO_QUALIFYING_EVIDENCE, evidence, reference_id)
    return PhenotypeSdResolution(
        status=SdStatus.ESTIMATED,
        reason=None,
        estimate=estimate_phenotype_sd(method, request.sample_size, se=se, af=af, beta=beta),
        reference_id=reference_id,
        n_evidence_considered=evidence.considered,
        n_estimate_inputs=n_inputs,
        evidence_sampled=evidence.considered > evidence.k,
    )


def resolve_analysis(
    request: AnalysisRequest,
    *,
    reader: MetricsReader,
    reference: AncestryReference,
    extraction_panel: Collection[str] | None = None,
    gates: Gates | None = None,
    af_references: Mapping[str, AfReference] | None = None,
    evidence_sample: int = DEFAULT_EVIDENCE_SAMPLE,
) -> AnalysisResolution:
    """Resolve one Analysis's ancestry and phenotype SD from one source scan.

    `reader` is the Analysis's own resolved source reader; this module never
    resolves one, so which Source Format a file is stays the caller's manifest
    fact rather than something inferred here.

    `extraction_panel` bounds the ancestry fit's sites (`None` uses every
    reference site, which is what assigning from the full reference does);
    `af_references` maps Assigned Ancestry to a declared reference-AF source and
    is read only by the reference-MAF tier, only after the ancestry fit has said
    which ancestry to look up.

    A source that cannot be read at all comes back as a resolution with `error`
    set rather than as an exception, because one unreadable file in a batch of
    thousands is a per-Analysis outcome (issue #207). A caller error -- a
    non-positive `evidence_sample`, a reader with no single-scan metrics path --
    raises, because it is a defect in the call, not in the data.
    """
    if evidence_sample <= 0:
        raise ValueError(
            f"evidence_sample must be a positive row count, got {evidence_sample!r}"
        )
    if not isinstance(reader, MetricsReader):
        raise TypeError(
            f"{type(reader).__name__} cannot stream source metrics; the one-pass resolver "
            f"reads tabular sources only (issue #207)"
        )
    scan = _Scan()
    evidence = _EvidenceSample(k=evidence_sample, tier=request.original_sd_method)
    panel: Collection[str] = reference.index if extraction_panel is None else extraction_panel
    try:
        _scan(reader, panel, scan, evidence)
    except (OSError, EOFError, ValueError) as exc:
        return AnalysisResolution(
            analysis_id=request.analysis_id,
            diagnostics=_diagnostics(request, scan),
            error=f"{type(exc).__name__}: {exc}",
        )
    ancestry = assign_ancestry(scan.panel_af, reference, gates)
    return AnalysisResolution(
        analysis_id=request.analysis_id,
        diagnostics=_diagnostics(request, scan),
        ancestry=ancestry,
        phenotype_sd=_resolve_phenotype_sd(request, ancestry, evidence, af_references or {}),
    )
