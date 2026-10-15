"""Resolve BESD Analyses: one `.esi` measurement per file, one SD per probe (issue #282).

Every Analysis is resolved, and a BESD probe is an Analysis (stores ADR 0036).
What a probe needs splits by where the evidence lives (ADR 0062):

* **Ancestry and allele orientation come from the `.esi`.** Its frequencies are
  per variant and shared by every probe, so they are measured once per file --
  in the manifest resolver's parent process, before workers fork -- and every
  probe's record carries that one measurement, under its `measurement_id`.
* **Phenotype SD comes from the probe.** Its own beta and SE from the `.besd`,
  its frequencies from the `.esi`, and its sample size from the manifest,
  because BESD carries no per-variant N.

Two refusals are specific to the format, and both are controlled failures of
the Analysis rather than a guessed value:

* a file set whose `.esi` reports no usable frequency at all -- no seventh
  column, or `NA` on every row -- has nothing to fit ancestry or check
  orientation against, so every probe in it fails;
* a probe whose tier asks for a phenotype SD and whose manifest row supplies
  no sample size fails, because nothing in a BESD file could ever supply one.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

from opengwasdb.ancestry.mixture import Gates
from opengwasdb.ancestry.reference import AncestryReference
from opengwasdb.build.phenotype_sd import has_usable_sample_size
from opengwasdb.build.resolve import (
    AfReference,
    AnalysisRequest,
    AnalysisResolution,
    ScanLimit,
    SourceAncestryMeasurement,
    measure_source_ancestry,
    needs_phenotype_sd,
    resolve_analysis,
    unresolved,
)
from opengwasdb.readers.assembly import assembly_plan
from opengwasdb.readers.besd import (
    BESD_CAPABILITY,
    BesdProbeReader,
    BesdSource,
    besd_member_files,
    load_besd_source,
)
from opengwasdb.readers.gwas_vcf import load_liftover

__all__ = [
    "BesdMeasurement",
    "besd_member_stats",
    "measure_besd_source",
    "resolve_besd_analysis",
]


@dataclass(frozen=True)
class BesdMeasurement:
    """One BESD file set, read and measured once; or why it could not be.

    Exactly one of (`source` and `ancestry`) or `error` is set, and the
    constructor refuses anything else. An `error` is every probe's controlled
    failure: a file set that cannot be read, aligned or measured cannot resolve
    any Analysis in it. `measurement_id` names the attempt either way.
    """

    measurement_id: str
    source: BesdSource | None = None
    ancestry: SourceAncestryMeasurement | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        measured = self.source is not None and self.ancestry is not None
        partial = (self.source is None) != (self.ancestry is None)
        if partial or measured == (self.error is not None):
            raise ValueError(
                "a BESD measurement is either a source and its ancestry, or an error"
            )


def besd_member_stats(prefix: str | Path) -> dict[str, dict[str, int | None]]:
    """Size and mtime of each member file: what a BESD record's resume fingerprint binds.

    `source_file` is a prefix, not a file, so the manifest resolver's own
    source stat sees nothing; without these a rewritten `.esi` would resume an
    old record. A missing member records `None`s and fails at measurement.
    """
    stats: dict[str, dict[str, int | None]] = {}
    for member in besd_member_files(prefix):
        try:
            st = member.stat()
            stats[member.suffix] = {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
        except OSError:
            stats[member.suffix] = {"bytes": None, "mtime_ns": None}
    return stats


def measure_besd_source(
    prefix: str | Path,
    *,
    source_build: str,
    reference_build: str,
    chain_file: str | Path | None,
    reference: AncestryReference,
    measurement_id: str,
    extraction_panel: Collection[str] | None = None,
    gates: Gates | None = None,
    scan_limit: ScanLimit | None = None,
) -> BesdMeasurement:
    """Read one file set's `.esi` and `.epi` and fit its ancestry, once."""
    try:
        source = _load_aligned(prefix, source_build, reference_build, chain_file)
    except (OSError, ValueError) as exc:
        return BesdMeasurement(measurement_id, error=f"{type(exc).__name__}: {exc}")
    if source.n_frequencies == 0:
        return BesdMeasurement(measurement_id, error=(
            f"BESD source {prefix}.esi reports no usable allele frequency on any of its "
            f"{source.n_rows} rows (no frequency column, or NA throughout): ancestry and "
            f"allele orientation cannot be measured, and no frequency is substituted"
        ))
    ancestry = measure_source_ancestry(
        source.alid, source.a1_frequencies(), source.palindromic,
        reference=reference, measurement_id=measurement_id,
        extraction_panel=extraction_panel, gates=gates, scan_limit=scan_limit,
    )
    return BesdMeasurement(measurement_id, source, ancestry)


def _load_aligned(
    prefix: str | Path, source_build: str, reference_build: str, chain_file: str | Path | None
) -> BesdSource:
    """The file set on the reference build: lifted by the declared chain, never by guess."""
    plan = assembly_plan(
        capability=BESD_CAPABILITY,
        source_build=source_build,
        reference_build=reference_build,
        chain_file=chain_file,
    )
    liftover = (
        load_liftover(plan.source_build, plan.reference_build, str(plan.chain_file))
        if plan.requires_liftover else None
    )
    return load_besd_source(prefix, liftover=liftover)


def resolve_besd_analysis(
    request: AnalysisRequest,
    probe_id: str,
    measurement: BesdMeasurement,
    *,
    reference: AncestryReference,
    gates: Gates | None = None,
    af_references: Mapping[str, AfReference] | None = None,
    evidence_sample: int,
    scan_limit: ScanLimit | None = None,
    variant_reference: Collection[str] | None = None,
) -> AnalysisResolution:
    """Resolve one probe against its file's shared measurement."""
    refusal = _refusal(request, measurement)
    if refusal is not None:
        # The file's measurement is recorded even on a refused probe (stores
        # ADR 0036): every Analysis's measurements, whether it passes or not.
        return unresolved(
            request.analysis_id, request.source_file, refusal,
            source_ancestry=measurement.ancestry,
            measurement_id=measurement.measurement_id,
        )
    assert measurement.source is not None
    return resolve_analysis(
        request,
        reader=BesdProbeReader(measurement.source, probe_id),
        reference=reference,
        gates=gates,
        af_references=af_references,
        evidence_sample=evidence_sample,
        scan_limit=scan_limit,
        variant_reference=variant_reference,
        source_ancestry=measurement.ancestry,
    )


def _refusal(request: AnalysisRequest, measurement: BesdMeasurement) -> str | None:
    """Why this probe cannot be resolved at all, or `None`.

    A missing sample size is refused only when the probe's tier estimates an
    SD; a declared-standardised or case-control probe needs no N (ADR 0062).
    """
    if needs_phenotype_sd(request) and not has_usable_sample_size(request.sample_size):
        return (
            f"Analysis {request.analysis_id}: original_sd_method "
            f"{request.original_sd_method.value!r} estimates a phenotype SD from the "
            f"sample size, and BESD carries none per variant; the manifest row must "
            f"supply a usable sample_size"
        )
    return measurement.error
