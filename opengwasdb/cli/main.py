"""OpenGWASDB command line interface."""

import csv
import json
import logging
import math
import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, cast

import numpy as np
import typer

from opengwasdb.ancestry.mixture import Gates
from opengwasdb.ancestry.pipeline import annotate_catalogue, read_source_manifest
from opengwasdb.ancestry.reference import load_reference
from opengwasdb.build.eaf_orientation import DEFAULT_SAMPLE_SITES
from opengwasdb.build.liftover import normalise_build
from opengwasdb.build.observed import build_dense_observed_from_sources
from opengwasdb.build.phenotype_sd_pipeline import (
    AfSource,
    estimate_manifest_phenotype_sd,
    load_af_reference,
    read_sd_manifest,
    write_sd_estimates,
)
from opengwasdb.build.resolve import DEFAULT_EVIDENCE_SAMPLE
from opengwasdb.build.resolve_manifest import resolve_analyses_manifest
from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
from opengwasdb.layouts.dense.complete import (
    complete_dense_store,
    resume_dense_completion,
)
from opengwasdb.layouts.dense.constants import DEFAULT_CHUNK_SHAPE
from opengwasdb.layouts.dense.overview import write_overview_html
from opengwasdb.layouts.dense.rho import (
    DEFAULT_RHO_MIN_NULLS,
    DEFAULT_RHO_WINDOW_BP,
    DEFAULT_RHO_Z_THRESH,
    build_dense_rho,
)
from opengwasdb.layouts.dense.top_hits import build_top_hit_indexes
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.layouts.hybrid.complete import complete_hybrid_store
from opengwasdb.layouts.ragged.build_besd import build_ragged_from_besd
from opengwasdb.layouts.ragged.build_ssf import build_ragged_from_ssf
from opengwasdb.layouts.ragged.complete import complete_ragged_store
from opengwasdb.layouts.ragged.top_hits import build_ragged_top_hit_indexes
from opengwasdb.model.analyses import read_analyses
from opengwasdb.query import query_store
from opengwasdb.query.facade import HybridStoreQuery, RaggedStoreQuery, StoreQuery
from opengwasdb.readers import known_capabilities
from opengwasdb.repair import repair_eaf_chunks
from opengwasdb.store import open_store
from opengwasdb.validation import validate_store
from opengwasdb.variants.reference import extract_variant_reference
from opengwasdb.variants.windows import (
    DEFAULT_MAP_SPILL_RECORDS,
    DEFAULT_REDUCTION_BATCH_SIZE,
    DEFAULT_WINDOW_SIZE_MB,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

app = typer.Typer(no_args_is_help=True)


class OutputFormat(StrEnum):
    """--format for the query-* commands (issue #104)."""

    tsv = "tsv"
    json = "json"


# EAF orientation options (issue #115, ADR 0037 §6), shared by every build
# command that can store a frequency. Declared once: a build path that took a
# reference under a different flag name would be a build path whose stores
# nobody could compare.
_EAF_REFERENCE_HELP = (
    "Reference frequencies to check each Analysis's stored EAF against: an LD "
    "panel directory (with --eaf-reference-ancestry) or a table with an 'eaf' "
    "column. A source reporting frequency against the other allele fails the build."
)
_EAF_ANCESTRY_HELP = "Population to read from an --eaf-reference panel directory, e.g. EUR"
_ALLOW_UNVERIFIED_HELP = (
    "Accept Analyses the supplied --eaf-reference could not verify (too little "
    "overlap or frequency spread) instead of failing. Recorded in the store's provenance."
)

_SOURCE_READER_CAPABILITY_HELP = (
    "Default Source Reader Capability for manifest rows that omit source_reader_capability "
    "(default: opengwasdb.gwas-vcf)"
)
_SOURCE_ASSEMBLY_HELP = (
    "Default source genome build for manifest rows that omit source_assembly "
    "(hg19/GRCh37 or hg38/GRCh38, default: hg19)"
)
_VARIANT_REFERENCE_HELP = (
    "Build against a precomputed variant axis instead of running Pass 1: a "
    "*.variant-ref.tsv.gz artifact, a plain ALID list, or a store variants.tsv.gz. "
    "Source variants absent from the reference are dropped; reference variants "
    "no study observes are stored as NaN."
)


def _validate_source_reader_capability(value: str | None) -> str | None:
    if value is None:
        return None
    if value not in known_capabilities():
        known = ", ".join(known_capabilities()) or "(none registered)"
        raise typer.BadParameter(
            f"unknown source reader capability {value!r}; known: {known}"
        )
    return value


def _validate_source_assembly(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return normalise_build(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


class ReportFormat(StrEnum):
    """Output format for info and validate commands (issue #175)."""

    text = "text"
    json = "json"


_REPORT_FORMAT_OPTION = typer.Option(
    ReportFormat.text,
    "--format",
    help="Output format: text (human readable) or json (machine readable)",
)

# Module-level option singletons rather than inline `typer.Option(...)` defaults:
# the call in a default value is what ruff's B008 flags, and these are shared by
# the one command that needs them.
_AF_SOURCE_OPTION = typer.Option(
    AfSource.source,
    "--af-source",
    help="Where the estimator's allele frequency comes from: source or reference",
)
_AF_REFERENCE_OPTION = typer.Option(
    None,
    "--af-reference",
    help=(
        "Reference frequency table (an 'eaf' column) or LD panel directory, "
        "required with --af-source reference"
    ),
)
_AF_REFERENCE_ANCESTRY_OPTION = typer.Option(
    None,
    help="Population to read from an --af-reference panel directory, e.g. EUR",
)
_N_WORKERS_OPTION = typer.Option(
    1, "--n-workers", "--workers", help="Fork process-pool size"
)


def _echo_summary(payload: dict[str, Any]) -> None:
    typer.echo(json.dumps(payload, sort_keys=True))


@app.command()
def info(
    store_path: Path,
    output_format: ReportFormat = _REPORT_FORMAT_OPTION,
) -> None:
    """Print basic manifest information for a local Store Release."""

    store = open_store(store_path)
    manifest = store.manifest
    if output_format is ReportFormat.json:
        payload = {
            "store_id": manifest.store_id,
            "release_id": manifest.release_id,
            "format_version": manifest.format_version,
            "primary_layout": manifest.primary_layout.value,
            "association_coverage": manifest.association_coverage.value,
            "completion_state": manifest.completion_state.value,
            "reference_assembly": manifest.reference_assembly,
            "encoding": manifest.encoding.to_manifest(),
        }
        typer.echo(json.dumps(payload, sort_keys=True))
        return

    typer.echo(f"store_id: {manifest.store_id}")
    typer.echo(f"release_id: {manifest.release_id}")
    typer.echo(f"format_version: {manifest.format_version}")
    typer.echo(f"primary_layout: {manifest.primary_layout.value}")
    typer.echo(f"association_coverage: {manifest.association_coverage.value}")
    typer.echo(f"completion_state: {manifest.completion_state.value}")
    typer.echo(f"reference_assembly: {manifest.reference_assembly}")
    # The encoding a release declares is what its bytes mean (ADR 0037), so it
    # belongs next to format_version rather than being something an operator
    # has to open manifest.json to see.
    encoding = manifest.encoding
    scale = "" if encoding.z.scale is None else f" (scale 1/{encoding.z.scale})"
    eaf = encoding.eaf.kind
    if encoding.eaf.residual_range is not None:
        eaf += f" (range +/-{encoding.eaf.residual_range:g})"
    if encoding.eaf.reference:
        eaf += " + reference"
    typer.echo(
        f"encoding: z={encoding.z.kind}{scale}, se={encoding.se.kind}, eaf={eaf}"
    )


@app.command("validate")
def validate_command(
    store_path: Path,
    output_format: ReportFormat = _REPORT_FORMAT_OPTION,
) -> None:
    """Validate a local Store Release."""

    result = validate_store(store_path)
    if output_format is ReportFormat.json:
        payload = {
            "ok": result.ok,
            "errors": result.errors,
            "warnings": result.warnings,
        }
        typer.echo(json.dumps(payload, sort_keys=True))
        if not result.ok:
            raise typer.Exit(1)
        return

    for warning in result.warnings:
        typer.echo(f"warning: {warning}", err=True)
    if result.ok:
        typer.echo("valid")
        return
    for error in result.errors:
        typer.echo(f"error: {error}", err=True)
    raise typer.Exit(1)


@app.command("repair-eaf-chunks")
def repair_eaf_chunks_command(store_path: Path) -> None:
    """Rechunk EAF baseline/reference arrays in an existing store in place."""
    repaired = repair_eaf_chunks(store_path)
    if not repaired:
        typer.echo("already repaired")
        return
    for item in repaired:
        typer.echo(f"{item.array}: {item.old_chunk} -> {item.new_chunk}")


@app.command("audit-eaf-orientation")
def audit_eaf_orientation_command(
    store_path: Path,
    eaf_reference: Path = typer.Option(..., help=_EAF_REFERENCE_HELP),
    eaf_reference_ancestry: str | None = typer.Option(None, help=_EAF_ANCESTRY_HELP),
    n_sites: int = typer.Option(
        DEFAULT_SAMPLE_SITES, help="How many variants to correlate over"
    ),
) -> None:
    """Re-run the EAF orientation check over a built store's stored frequencies.

    `validate` checks the evidence a build recorded, because a Store Release
    must be readable without the panel it was built against. This re-derives the
    answer from the store's own arrays, which is how a store built before the
    check existed gets one -- and how a recorded `passed` gets tested against a
    second reference rather than taken on trust (issue #115).

    Exits non-zero when an Analysis correlates negatively, or when the audit
    does not reproduce what the store recorded.
    """
    from opengwasdb.validation.eaf_audit import audit_eaf_orientation

    result = audit_eaf_orientation(
        store_path,
        eaf_reference,
        ancestry=eaf_reference_ancestry,
        n_sites=n_sites,
    )
    typer.echo(
        json.dumps(
            {
                "store_path": str(store_path),
                "recorded": result.recorded,
                "disagreements": list(result.disagreements),
                **result.report.provenance(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if result.ok:
        return
    for evidence in result.report.failures:
        typer.echo(
            f"error: analysis {evidence.analysis_id} reports EAF against the other "
            f"allele (r = {evidence.r:+.4f} over {evidence.n_overlap} variants)",
            err=True,
        )
    for analysis_id in result.disagreements:
        typer.echo(
            f"error: analysis {analysis_id} recorded "
            f"{result.recorded.get(analysis_id)!r} but this audit did not reproduce it",
            err=True,
        )
    raise typer.Exit(1)


@app.command("build-dense")
def build_dense_command(
    source_path: Path,
    output_path: Path,
    store_id: str = typer.Option(...),
    release_id: str = typer.Option(...),
    reference_assembly: str = typer.Option("GRCh37"),
    overwrite: bool = typer.Option(False),
) -> None:
    """Build a Dense Observed-Only store from a tiny TSV/CSV source."""

    result = build_dense_observed_from_sources(
        [source_path],
        output_path,
        store_id=store_id,
        release_id=release_id,
        reference_assembly=reference_assembly,
        overwrite=overwrite,
    )
    typer.echo(
        json.dumps(
            {
                "output_path": str(result.output_path),
                "n_variants": result.n_variants,
                "n_analyses": result.n_analyses,
            },
            sort_keys=True,
        )
    )


@app.command("extract-variant-reference")
def extract_variant_reference_command(
    manifest_path: Path,
    output_path: Annotated[
        Path, typer.Option("--output-path", help="Where to write the artifact")
    ],
    chain_file: Annotated[
        Path | None, typer.Option(help="hg19->hg38 chain file")
    ] = None,
    n_workers: Annotated[
        int, typer.Option(help="Fork pool size for the variant union")
    ] = 1,
    liftover_failure_threshold: Annotated[
        float, typer.Option(help="Maximum hg19 liftover failure rate")
    ] = 0.01,
    source_reader_capability: Annotated[
        str | None,
        typer.Option(
            callback=_validate_source_reader_capability, help=_SOURCE_READER_CAPABILITY_HELP
        ),
    ] = None,
    source_assembly: Annotated[
        str | None,
        typer.Option(callback=_validate_source_assembly, help=_SOURCE_ASSEMBLY_HELP),
    ] = None,
    window_size_mb: Annotated[
        float, typer.Option(help="Genomic window size in megabases")
    ] = DEFAULT_WINDOW_SIZE_MB,
    reduction_batch_size: Annotated[
        int, typer.Option(help="Shards merged per reduction task")
    ] = DEFAULT_REDUCTION_BATCH_SIZE,
    map_spill_records: Annotated[
        int,
        typer.Option(
            help="Variants a map worker buffers before spilling its window shards"
        ),
    ] = DEFAULT_MAP_SPILL_RECORDS,
) -> None:
    """Extract, lift and canonicalise a manifest's variant axis (#187) into the
    *.variant-ref.tsv.gz artifact `build-dense-vcf --variant-reference`
    consumes. --window-size-mb and --reduction-batch-size shape the parallel
    genomic tree-reduce (issue #188) and never change the artifact.
    --map-spill-records bounds a worker's peak memory (issue #194).
    """
    result = extract_variant_reference(
        manifest_path, output_path, chain_file=chain_file,
        liftover_failure_threshold=liftover_failure_threshold, n_workers=n_workers,
        source_reader_capability=source_reader_capability, source_assembly=source_assembly,
        window_size_mb=window_size_mb, reduction_batch_size=reduction_batch_size,
        map_spill_records=map_spill_records,
    )
    _echo_summary(
        {
            "output_path": str(result.output_path),
            "n_variants": result.n_variants,
            "n_source_keys": result.n_source_keys,
            "n_rsids": result.n_rsids,
        }
    )


@app.command("build-dense-vcf")
def build_dense_vcf_command(
    manifest_path: Path,
    output_path: Path,
    store_id: str = typer.Option(...),
    release_id: str = typer.Option(...),
    overwrite: bool = typer.Option(False),
    n_workers: int = typer.Option(1, help="Fork-based process pool size for Pass 1 and Pass 2"),
    chunk_variants: int = typer.Option(DEFAULT_CHUNK_SHAPE[0], help="Variant chunk size"),
    chunk_analyses: int = typer.Option(DEFAULT_CHUNK_SHAPE[1], help="Analysis chunk size"),
    eaf_reference: Path | None = typer.Option(None, help=_EAF_REFERENCE_HELP),
    eaf_reference_ancestry: str | None = typer.Option(None, help=_EAF_ANCESTRY_HELP),
    allow_unverified_eaf: bool = typer.Option(False, help=_ALLOW_UNVERIFIED_HELP),
    source_reader_capability: str | None = typer.Option(
        None, callback=_validate_source_reader_capability, help=_SOURCE_READER_CAPABILITY_HELP
    ),
    source_assembly: str | None = typer.Option(
        None, callback=_validate_source_assembly, help=_SOURCE_ASSEMBLY_HELP
    ),
    variant_reference: Annotated[Path | None, typer.Option(help=_VARIANT_REFERENCE_HELP)] = None,
) -> None:
    """Build a Dense Observed-Only store from a manifest of GWAS-VCF files.

    MANIFEST_PATH is a TSV with columns: trait_id, file_path, trait_name, n,
    stored_effect_scale (issue #17), original_sd_method, and original_sd (issue #18).
    VCF files are hg19 by default; liftover to hg38 is applied inline.
    --source-reader-capability and --source-assembly supply per-release defaults (#174).
    --variant-reference supplies a precomputed axis, bypassing Pass 1 (#185).
    """
    result = build_dense_from_vcf_manifest(
        manifest_path, output_path, store_id=store_id, release_id=release_id,
        overwrite=overwrite, n_workers=n_workers, chunk_shape=(chunk_variants, chunk_analyses),
        eaf_reference=eaf_reference, eaf_reference_ancestry=eaf_reference_ancestry,
        allow_unverified_eaf=allow_unverified_eaf,
        source_reader_capability=source_reader_capability, source_assembly=source_assembly,
        variant_reference=variant_reference,
    )
    _echo_summary({
        "output_path": str(result.output_path),
        "n_variants": result.n_variants,
        "n_analyses": result.n_analyses,
    })


@app.command("build-hybrid")
def build_hybrid_command(
    manifest_path: Path,
    output_path: Path,
    reference_panel: Path | None = typer.Option(
        None, help="Dense Component axis: reference-panel ALIDs (legacy; see --variant-reference)"
    ),
    variant_reference: Annotated[Path | None, typer.Option(help=_VARIANT_REFERENCE_HELP)] = None,
    store_id: str = typer.Option(...),
    release_id: str = typer.Option(...),
    overwrite: bool = typer.Option(False),
    n_workers: int = typer.Option(1, help="Fork-based process pool size for Pass 1 and Pass 2"),
    chunk_variants: int = typer.Option(DEFAULT_CHUNK_SHAPE[0], help="Zarr variant chunk size"),
    chunk_analyses: int = typer.Option(DEFAULT_CHUNK_SHAPE[1], help="Zarr analysis chunk size"),
    eaf_reference: Path | None = typer.Option(None, help=_EAF_REFERENCE_HELP),
    eaf_reference_ancestry: str | None = typer.Option(None, help=_EAF_ANCESTRY_HELP),
    allow_unverified_eaf: bool = typer.Option(False, help=_ALLOW_UNVERIFIED_HELP),
    capability: str | None = typer.Option(
        None, "--source-reader-capability", callback=_validate_source_reader_capability,
        help=_SOURCE_READER_CAPABILITY_HELP,
    ),
    assembly: str | None = typer.Option(
        None, "--source-assembly", callback=_validate_source_assembly,
        help=_SOURCE_ASSEMBLY_HELP,
    ),
) -> None:
    """Build a Hybrid store (Dense Component + Ragged Overflow) from a VCF manifest.

    MANIFEST_PATH is a TSV with columns: trait_id, file_path, trait_name, n,
    stored_effect_scale (issue #17), original_sd_method, and original_sd (issue #18).
    On-panel variants in --reference-panel fill the Dense Component; off-panel variants
    go to Ragged Overflow. --variant-reference supplies a precomputed axis and source
    map, bypassing Pass 1 (#186). --source-reader-capability and --source-assembly
    supply per-release defaults (#174).
    """
    res = build_hybrid_from_vcf_manifest(
        manifest_path, output_path, reference_panel=reference_panel,
        variant_reference=variant_reference,
        store_id=store_id, release_id=release_id, overwrite=overwrite, n_workers=n_workers,
        chunk_shape=(chunk_variants, chunk_analyses), eaf_reference=eaf_reference,
        eaf_reference_ancestry=eaf_reference_ancestry, allow_unverified_eaf=allow_unverified_eaf,
        source_reader_capability=capability, source_assembly=assembly,
    )
    _echo_summary(
        dict(
            output_path=str(res.output_path),
            n_variants=res.n_variants,
            n_analyses=res.n_analyses,
            n_panel=res.n_panel,
            n_off_panel=res.n_off_panel,
            n_overflow=res.n_overflow,
        )
    )


@app.command("build-hybrid-from-catalogue")
def build_hybrid_from_catalogue_command(
    catalogue_path: Path,
    output_path: Path,
    reference_panel: Path = typer.Option(..., help="Dense Component axis: reference-panel ALIDs"),
    store_id: str = typer.Option(...),
    release_id: str = typer.Option(...),
    stored_effect_scale: str = typer.Option(
        ..., help="Effect scale for every kept Analysis (issue #17): sd, log_or, or log_hazard"
    ),
    original_sd_method: str = typer.Option(
        ...,
        help=(
            "Phenotype-SD provenance for every kept Analysis (issue #18), e.g. "
            "declared_standardised, source_provided, estimated_from_source_maf, "
            "estimated_from_reference_maf, estimated_from_beta_distribution, binary_trait"
        ),
    ),
    ancestry: str = typer.Option("EUR", help="Assigned Ancestry to subset the Catalogue to"),
    original_sd: float | None = typer.Option(
        None,
        help=(
            "Phenotype SD to rescale by (issue #18); required when --original-sd-method is "
            "source_provided or one of the estimated_from_* tiers, omitted otherwise"
        ),
    ),
    overwrite: bool = typer.Option(False),
    n_workers: int = typer.Option(1, help="Fork-based process pool size for Pass 2"),
    chunk_variants: int = typer.Option(DEFAULT_CHUNK_SHAPE[0]),
    chunk_analyses: int = typer.Option(DEFAULT_CHUNK_SHAPE[1]),
    eaf_reference: Path | None = typer.Option(None, help=_EAF_REFERENCE_HELP),
    eaf_reference_ancestry: str | None = typer.Option(None, help=_EAF_ANCESTRY_HELP),
    allow_unverified_eaf: bool = typer.Option(False, help=_ALLOW_UNVERIFIED_HELP),
) -> None:
    """Subset the Catalogue to one ancestry and build a Hybrid store from it.

    Row-filters the Catalogue to ``assigned_ancestry == ANCESTRY`` (a manifest the
    unchanged build reads, plus STORED_EFFECT_SCALE and ORIGINAL_SD_METHOD/ORIGINAL_SD
    stamped onto every kept row -- the Catalogue itself never carries these columns,
    issues #17/#18), runs build-hybrid, and records per-Analysis Assigned Ancestry +
    Catalogue provenance in the store sidecar. Non-matching Analyses are absent from
    the store (still parked in the Catalogue).
    """
    from opengwasdb.ancestry.subset import build_hybrid_from_catalogue

    subset = build_hybrid_from_catalogue(
        catalogue_path,
        output_path,
        reference_panel=reference_panel,
        store_id=store_id,
        release_id=release_id,
        stored_effect_scale=stored_effect_scale,
        original_sd_method=original_sd_method,
        ancestry=ancestry,
        original_sd=original_sd,
        overwrite=overwrite,
        n_workers=n_workers,
        chunk_shape=(chunk_variants, chunk_analyses),
        eaf_reference=eaf_reference,
        eaf_reference_ancestry=eaf_reference_ancestry,
        allow_unverified_eaf=allow_unverified_eaf,
    )
    typer.echo(
        json.dumps(
            {
                "output_path": str(output_path),
                "ancestry": subset.ancestry,
                "subset_filter": subset.subset_filter,
                "n_total": subset.n_total,
                "n_kept": subset.n_kept,
                "catalogue_version": subset.catalogue_version,
            },
            sort_keys=True,
        )
    )


@app.command("assign-ancestry")
def assign_ancestry_command(
    manifest_path: Path,
    catalogue_path: Path,
    ancestry_reference: Path = typer.Option(
        ..., help="Ancestry Reference Panel: ref_freqs.hg38.tsv.gz"
    ),
    ancestry_groups: Path = typer.Option(
        ..., help="Fine→super-population map: ancestry_groups.tsv"
    ),
    maf_floor: float = typer.Option(0.01, help="Drop reference variants below this MAF"),
    tau: float = typer.Option(0.50, help="Gate: min dominant super-population proportion"),
    delta: float = typer.Option(0.20, help="Gate: min margin over the runner-up"),
    n_min: int = typer.Option(5_000, help="Gate: min overlapping reference sites"),
    residual_max: float = typer.Option(0.06, help="Gate: max RMS NNLS residual"),
    orientation_flip_r: float = typer.Option(
        -0.5,
        help=(
            "Gate: at or below this correlation against the reference consensus, an "
            "Analysis's EAF is reported as mis-oriented rather than as a residual failure"
        ),
    ),
    n_workers: int = typer.Option(1, "--n-workers", "--workers", help="Fork process-pool size"),
    catalogue_version: str = typer.Option("v1", help="Recorded in the Catalogue"),
    reference_version: str = typer.Option(
        "", help="Reference version stamp (default: reference filename)"
    ),
) -> None:
    """Annotate a raw source manifest into the versioned Analysis Catalogue.

    MANIFEST_PATH is a TSV with columns trait_id, file_path, trait_name, n and
    optional reported_population and source_reader_capability columns. AF is
    extracted at the reference sites (targeted, parallel), fit to the fine
    reference by NNLS, aggregated to super-populations, and gated into an
    Assigned Ancestry or Unassigned. Non-EUR/Unassigned Analyses are retained
    (parked) in the Catalogue.

    source_reader_capability selects the Source Format reader (issue #115); it
    defaults to opengwasdb.gwas-vcf, the only format this could read before, so
    a manifest written without the column keeps its meaning. Set it to
    opengwasdb.gwas-ssf or opengwasdb.finngen-r13 to assign a tabular family --
    without it those sources cannot be checked against reference frequencies at
    all, which is how GCST003566's inverted EAF column went unexamined.

    Each Analysis's EAF orientation is recorded in the Catalogue
    (eaf_orientation, eaf_orientation_r): an Analysis whose frequencies are
    anti-correlated with the reference is left Unassigned with
    gate_reason=eaf_orientation, which is the signal to exclude the source or
    report it upstream rather than to loosen a threshold.
    """
    reference = load_reference(ancestry_reference, ancestry_groups, maf_floor=maf_floor)
    gates = Gates(
        tau=tau,
        delta=delta,
        n_min=n_min,
        residual_max=residual_max,
        orientation_flip_r=orientation_flip_r,
    )
    source_rows = read_source_manifest(manifest_path)
    rows = annotate_catalogue(
        source_rows,
        reference,
        gates,
        catalogue_path,
        catalogue_version=catalogue_version,
        ancestry_reference_version=reference_version or ancestry_reference.name,
        n_workers=n_workers,
    )
    n_assigned = sum(1 for r in rows if r.assignment.assigned_ancestry is not None)
    typer.echo(
        json.dumps(
            {
                "catalogue_path": str(catalogue_path),
                "n_analyses": len(rows),
                "n_assigned": n_assigned,
                "n_parked": len(rows) - n_assigned,
                "superpops": reference.superpops,
            },
            sort_keys=True,
        )
    )


@app.command("estimate-phenotype-sd")
def estimate_phenotype_sd_command(
    manifest_path: Path,
    out_path: Path,
    af_source: AfSource = _AF_SOURCE_OPTION,
    af_reference: Path | None = _AF_REFERENCE_OPTION,
    af_reference_ancestry: str | None = _AF_REFERENCE_ANCESTRY_OPTION,
    n_workers: int = _N_WORKERS_OPTION,
) -> None:
    """Estimate a phenotype SD per Analysis from a canonical analyses.tsv.

    MANIFEST_PATH is the registry's canonical `analyses.tsv` (ADR 0034), read
    through the same column-alias resolver `assign-ancestry` uses. Each row's
    `source_file`/`source_reader_capability` resolves a `SourceReader`, which
    is read once for the se/af/beta evidence its caller-chosen
    `original_sd_method` needs -- no arrays on the command line.

    Output is a TSV keyed by analysis_id with the shared-core `analyses.tsv`
    spellings (analysis_id, original_sd, original_sd_method,
    original_sd_dispersion, notes), so a caller merges a column rather than
    translating a table. A missing or unusable `sample_size` reports
    `unavailable` rather than a fabricated estimate. This command computes the
    number; the tier, the tolerance, and whether a disagreement blocks a
    release stay registry decisions (ADR 0029).
    """
    if af_source is AfSource.reference and af_reference is None:
        raise typer.BadParameter("--af-source reference requires --af-reference")
    reference = (
        load_af_reference(af_reference, ancestry=af_reference_ancestry)
        if af_source is AfSource.reference and af_reference is not None
        else None
    )
    rows = read_sd_manifest(manifest_path)
    estimates = estimate_manifest_phenotype_sd(
        rows,
        af_source=af_source,
        af_reference=reference,
        n_workers=n_workers,
    )
    write_sd_estimates(out_path, estimates)
    _echo_summary(
        {
            "out_path": str(out_path),
            "n_analyses": len(estimates),
            "n_estimated": sum(
                1 for e in estimates if e.original_sd_method != "unavailable"
            ),
        }
    )


@app.command("resolve-analyses")
def resolve_analyses_command(
    manifest_path: Annotated[
        Path,
        typer.Argument(
            help="Path to canonical analyses.tsv manifest",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    records_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory to write versioned per-Analysis JSON records and index",
        ),
    ],
    ancestry_reference: Annotated[
        Path,
        typer.Option(
            "--ancestry-reference",
            help="Ancestry Reference Panel: ref_freqs.hg38.tsv.gz",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    ancestry_groups: Annotated[
        Path,
        typer.Option(
            "--ancestry-groups",
            help="Fine→super-population map: ancestry_groups.tsv",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    extraction_panel: Annotated[
        Path | None,
        typer.Option(
            "--extraction-panel",
            help="Variant list or QC panel file for bounded ancestry extraction",
        ),
    ] = None,
    af_reference: Annotated[
        list[str] | None,
        typer.Option(
            "--af-reference",
            help=(
                "Reference frequency table (or LD panel) for reference-MAF SD tier, "
                "optionally keyed by ancestry as ANCESTRY=PATH (e.g. EUR=/path/to/ukb.tsv.gz). "
                "Can be repeated."
            ),
        ),
    ] = None,
    af_reference_ancestry: Annotated[
        str | None,
        typer.Option(
            "--af-reference-ancestry",
            help="Default ancestry population for unkeyed --af-reference (default: EUR)",
        ),
    ] = None,
    default_source_reader_capability: Annotated[
        str | None,
        typer.Option(
            "--default-source-reader-capability",
            help="Default reader capability for manifest rows omitting source_reader_capability",
        ),
    ] = None,
    maf_floor: Annotated[
        float,
        typer.Option(help="Drop reference variants below this MAF"),
    ] = 0.01,
    tau: Annotated[
        float,
        typer.Option(help="Gate: min dominant super-population proportion"),
    ] = 0.50,
    delta: Annotated[
        float,
        typer.Option(help="Gate: min margin over the runner-up"),
    ] = 0.20,
    n_min: Annotated[
        int,
        typer.Option(help="Gate: min overlapping reference sites"),
    ] = 5_000,
    residual_max: Annotated[
        float,
        typer.Option(help="Gate: max RMS NNLS residual"),
    ] = 0.06,
    orientation_flip_r: Annotated[
        float,
        typer.Option(
            help="Gate: orientation correlation threshold for mis-oriented EAF reporting",
        ),
    ] = -0.5,
    evidence_sample: Annotated[
        int,
        typer.Option(
            help="Max qualifying evidence rows to retain per Analysis for SD estimation",
        ),
    ] = DEFAULT_EVIDENCE_SAMPLE,
    n_workers: int = _N_WORKERS_OPTION,
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Skip unchanged successful records"),
    ] = False,
    largest_first: Annotated[
        bool,
        typer.Option(
            "--largest-first/--manifest-order",
            help="Schedule larger sources first to avoid straggler tails",
        ),
    ] = True,
    reference_version: Annotated[
        str,
        typer.Option(help="Ancestry reference version stamp (default: reference filename)"),
    ] = "",
) -> None:
    """Resolve an entire analyses.tsv manifest into versioned per-Analysis records.

    Processes each Analysis in MANIFEST_PATH once using the bounded one-pass
    resolver (issue #207), producing an Assigned Ancestry and an estimated
    phenotype SD from a single source scan.

    Outputs are written to RECORDS_DIR as atomic {analysis_id}.json records
    plus a deterministic index.json. When --resume is active, successful records
    whose fingerprints (source content/mtime, tool version, references, extraction
    panel, gates, and method tiers) match the current run are preserved without
    re-executing. Missing, failed, or stale records are rerun.

    The Ancestry Reference Panel is loaded once in the parent process and
    fork-shared across workers. Ordinary source, parser, or statistical errors
    are isolated to the affected Analysis and recorded as controlled_failure,
    while systemic configuration errors fail the command immediately.
    """
    try:
        summary = resolve_analyses_manifest(
            manifest_path=manifest_path,
            records_dir=records_dir,
            ancestry_reference=ancestry_reference,
            ancestry_groups=ancestry_groups,
            extraction_panel=extraction_panel,
            af_references=af_reference,
            af_reference_ancestry=af_reference_ancestry,
            default_source_reader_capability=default_source_reader_capability,
            maf_floor=maf_floor,
            tau=tau,
            delta=delta,
            n_min=n_min,
            residual_max=residual_max,
            orientation_flip_r=orientation_flip_r,
            evidence_sample=evidence_sample,
            n_workers=n_workers,
            resume=resume,
            largest_first=largest_first,
            reference_version=reference_version,
        )
    except (ValueError, FileNotFoundError, TypeError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc

    _echo_summary(summary.as_dict())


@app.command("route-catalogue")
def route_catalogue_command(
    catalogue_path: Path,
    coverage_path: Path,
    out_path: Path,
    min_variants: int = typer.Option(
        500_000, help="Genome-wide coverage floor for store eligibility"
    ),
) -> None:
    """Add routing + coverage columns to an assignment Catalogue.

    Combines AF-recovered ancestry with a Reported-Population fallback and a
    genome-wide coverage gate (min variants, all autosomes, no single-chromosome
    concentration). Writes routing_ancestry, routing_source, and store_eligible so a
    build is the filter ``routing_ancestry == EUR and store_eligible``. Low-coverage
    Analyses keep their ancestry but are flagged store_eligible=false (not dropped).
    """
    from opengwasdb.ancestry.routing import finalize_catalogue

    tally = finalize_catalogue(
        catalogue_path, coverage_path, out_path, min_variants=min_variants
    )
    typer.echo(
        json.dumps(
            {
                "out_path": str(out_path),
                "store_eligible": tally["kept"],
                "rescued_via_reported": tally["reported_fallback"],
                "dropped_no_ancestry": tally["dropped_ancestry"],
                "dropped_low_coverage": tally["dropped_coverage"],
            },
            sort_keys=True,
        )
    )


@app.command("calibrate-ancestry")
def calibrate_ancestry_command(
    catalogue_path: Path,
    tau: float = typer.Option(None, help="If set with --out, relabel with this τ"),
    delta: float = typer.Option(None, help="If set with --out, relabel with this δ"),
    n_min: int = typer.Option(None, help="Overlap gate for relabel (default: keep)"),
    residual_max: float = typer.Option(None, help="Residual gate for relabel (default: keep)"),
    out: Path = typer.Option(None, help="Write a relabelled Catalogue with chosen gates"),
    report: Path = typer.Option(None, help="Write the disagreement report TSV"),
) -> None:
    """Cross-tabulate Assigned vs Reported Population and (optionally) relabel.

    Prints the Assigned×Reported cross-tab and operating-point counts and lists the
    Analyses where a routable Reported ancestry disagrees with the Assigned one.
    With --out and --tau/--delta it re-applies the chosen gates from the stored
    statistics (no AF re-extraction) and writes a relabelled, gate-stamped
    Catalogue. Reported Population calibrates/audits only — it never routes.
    """
    from opengwasdb.ancestry import calibrate

    with open(catalogue_path, newline="", encoding="utf-8") as fh:
        import csv as _csv

        reader = _csv.DictReader(fh, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    typer.echo(calibrate.format_crosstab(calibrate.crosstab(rows)))
    typer.echo("")
    op = calibrate.operating_point(rows)
    typer.echo(
        f"reported-EUR admitted as EUR: {op['reported_eur_admitted_eur']}/{op['reported_eur']}; "
        f"reported-Mixed → Unassigned: {op['reported_mixed_unassigned']}/{op['reported_mixed']}"
    )

    conflicts = calibrate.disagreements(rows)
    typer.echo(f"disagreements (Assigned ≠ Reported): {len(conflicts)}")
    if report is not None:
        calibrate.write_disagreements(report, conflicts)
        typer.echo(f"disagreement report → {report}")

    if out is not None:
        if tau is None or delta is None:
            raise typer.BadParameter("--tau and --delta are required with --out")
        base = Gates()
        gates = Gates(
            tau=tau,
            delta=delta,
            n_min=n_min if n_min is not None else base.n_min,
            residual_max=residual_max if residual_max is not None else base.residual_max,
        )
        relabelled = calibrate.relabel(rows, gates)
        calibrate.write_rows(out, relabelled, fieldnames)
        n_assigned = sum(1 for r in relabelled if r["assigned_ancestry"] != "Unassigned")
        typer.echo(
            f"relabelled with τ={tau} δ={delta}: {n_assigned}/{len(relabelled)} assigned → {out}"
        )


@app.command("complete-hybrid")
def complete_hybrid_command(
    source_path: Path,
    dest_path: Path,
    ld_panel: Path = typer.Option(..., help="Root of LD panel (ld_dir/{ancestry}/{chr}/...)"),
    ancestry: str = typer.Option("EUR"),
    min_cor: float = typer.Option(0.7),
    thresh: float = typer.Option(0.9),
    release_id: str = typer.Option(None),
    n_workers: int = typer.Option(1, help="LD-block process-pool size"),
    overwrite: bool = typer.Option(False),
) -> None:
    """Reference-complete a Hybrid store — impute only the Dense Component."""
    import time
    t0 = time.time()
    result = complete_hybrid_store(
        source_path,
        dest_path,
        ld_panel,
        ancestry=ancestry,
        min_cor=min_cor,
        thresh=thresh,
        release_id=release_id or None,
        n_workers=n_workers,
        overwrite=overwrite,
    )
    elapsed = time.time() - t0
    typer.echo(
        json.dumps(
            {
                "output_path": str(result.output_path),
                "n_variants": result.n_variants,
                "n_analyses": result.n_analyses,
                "n_panel": result.n_panel,
                "n_off_panel": result.n_off_panel,
                "n_overflow": result.n_overflow,
                "n_imputed": result.n_imputed,
                "elapsed_s": round(elapsed, 1),
            },
            sort_keys=True,
        )
    )


@app.command("build-ragged-besd")
def build_ragged_besd_command(
    besd_prefix: Path,
    output_path: Path,
    store_id: str = typer.Option(...),
    release_id: str = typer.Option(...),
    analyses: Path | None = typer.Option(
        None, "--analyses", help="Optional analyses.tsv manifest with metadata"
    ),
    tissue: str | None = typer.Option(None),
    source_build: str = typer.Option("hg38"),
    overwrite: bool = typer.Option(False),
) -> None:
    """Build a Ragged Observed-Only store from BESD files.

    BESD_PREFIX is the path without extension (.esi, .epi, .besd are appended).
    Use --source-build hg19 to liftover coordinates to hg38 inline.
    Optionally, --analyses overlays Analytical and Attribution Metadata from a manifest.
    """
    result = build_ragged_from_besd(
        besd_prefix,
        output_path,
        store_id=store_id,
        release_id=release_id,
        analyses_path=analyses,
        tissue=tissue or None,
        source_build=source_build,
        overwrite=overwrite,
    )
    _echo_summary(
        {
            "output_path": str(result.output_path),
            "n_variants": result.n_variants,
            "n_analyses": result.n_analyses,
            "n_associations": result.n_associations,
        }
    )


@app.command("build-ragged-ssf")
def build_ragged_ssf_command(
    manifest_path: Path,
    filtered_dir: Path,
    output_path: Path,
    store_id: str = typer.Option(...),
    release_id: str = typer.Option(...),
    stored_effect_scale: str = typer.Option("sd", help="sd, log_or, or log_hazard"),
    overwrite: bool = typer.Option(False),
    eaf_reference: Path | None = typer.Option(None, help=_EAF_REFERENCE_HELP),
    eaf_reference_ancestry: str | None = typer.Option(None, help=_EAF_ANCESTRY_HELP),
    allow_unverified_eaf: bool = typer.Option(False, help=_ALLOW_UNVERIFIED_HELP),
) -> None:
    """Build a Ragged Observed-Only store from filtered GWAS-SSF files.

    MANIFEST_PATH is a TSV with columns: analysis_index, analysis_id (or
    legacy trait_id), analysis_label (or legacy trait_name), trait_ontology_id,
    trait_ontology_label, trait_chr, trait_bp, sample_size (or legacy n),
    tissue, context, mhc, source_file (or legacy filtered_file).

    FILTERED_DIR holds one filtered GWAS-SSF ``.tsv.gz`` per analysis (as
    produced by the opengwasdb-stores download+filter step). Relative
    source_file paths are joined to FILTERED_DIR; absolute paths are used
    directly.
    """
    result = build_ragged_from_ssf(
        manifest_path,
        filtered_dir,
        output_path,
        store_id=store_id,
        release_id=release_id,
        stored_effect_scale=stored_effect_scale,
        overwrite=overwrite,
        eaf_reference=eaf_reference,
        eaf_reference_ancestry=eaf_reference_ancestry,
        allow_unverified_eaf=allow_unverified_eaf,
    )
    typer.echo(
        json.dumps(
            {
                "output_path": str(result.output_path),
                "n_variants": result.n_variants,
                "n_analyses": result.n_analyses,
                "n_associations": result.n_associations,
            },
            sort_keys=True,
        )
    )


@app.command("complete-ragged")
def complete_ragged_command(
    source_path: Path,
    dest_path: Path,
    ld_panel: Path = typer.Option(..., help="Root of LD panel (ld_dir/{ancestry}/{chr}/...)"),
    ancestry: str = typer.Option("EUR"),
    cis_window_bp: int = typer.Option(1_000_000),
    min_cor: float = typer.Option(0.7),
    release_id: str = typer.Option(None),
    overwrite: bool = typer.Option(False),
) -> None:
    """Produce a Reference-Completed ragged store from an observed-only store."""
    import time
    t0 = time.time()
    result = complete_ragged_store(
        source_path,
        dest_path,
        ld_panel,
        ancestry=ancestry,
        cis_window_bp=cis_window_bp,
        min_cor=min_cor,
        release_id=release_id or None,
        overwrite=overwrite,
    )
    elapsed = time.time() - t0
    typer.echo(
        json.dumps(
            {
                "output_path": str(result.output_path),
                "n_variants": result.n_variants,
                "n_analyses": result.n_analyses,
                "n_associations": result.n_associations,
                "n_imputed": result.n_imputed,
                "n_missing": result.n_missing,
                "elapsed_s": round(elapsed, 1),
            },
            sort_keys=True,
        )
    )


@app.command("complete-dense")
def complete_dense_command(
    source_path: Path,
    dest_path: Path,
    ld_panel: Path = typer.Option(..., help="Root of LD panel (ld_dir/{ancestry}/{chr}/...)"),
    ancestry: str = typer.Option("EUR"),
    min_cor: float = typer.Option(0.7),
    thresh: float = typer.Option(0.9),
    release_id: str = typer.Option(None),
    n_workers: int = typer.Option(1, help="LD-block process-pool size"),
    overwrite: bool = typer.Option(False),
) -> None:
    """Produce a Reference-Completed Dense store from a Full Coverage observed-only store."""
    import time
    t0 = time.time()
    result = complete_dense_store(
        source_path,
        dest_path,
        ld_panel,
        ancestry=ancestry,
        min_cor=min_cor,
        thresh=thresh,
        release_id=release_id or None,
        n_workers=n_workers,
        overwrite=overwrite,
    )
    elapsed = time.time() - t0
    typer.echo(
        json.dumps(
            {
                "output_path": str(result.output_path),
                "n_variants": result.n_variants,
                "n_analyses": result.n_analyses,
                "n_imputed": result.n_imputed,
                "n_missing_off_panel": result.n_missing_off_panel,
                "n_missing_imputation_failed": result.n_missing_imputation_failed,
                "elapsed_s": round(elapsed, 1),
            },
            sort_keys=True,
        )
    )


@app.command("complete-dense-resume")
def complete_dense_resume_command(
    checkpoint_dir: Path,
    n_workers: int = typer.Option(1, help="LD-block process-pool size"),
) -> None:
    """Resume an interrupted complete-dense run from its checkpoint directory.

    Takes only the checkpoint directory path; all other build parameters are
    loaded from the build_params.json written by the original run.
    """
    import time
    t0 = time.time()
    result = resume_dense_completion(checkpoint_dir, n_workers=n_workers)
    elapsed = time.time() - t0
    typer.echo(
        json.dumps(
            {
                "output_path": str(result.output_path),
                "n_variants": result.n_variants,
                "n_analyses": result.n_analyses,
                "n_imputed": result.n_imputed,
                "n_missing_off_panel": result.n_missing_off_panel,
                "n_missing_imputation_failed": result.n_missing_imputation_failed,
                "elapsed_s": round(elapsed, 1),
            },
            sort_keys=True,
        )
    )


@app.command("build-ragged-top-hits")
def build_ragged_top_hits_command(store_path: Path) -> None:
    """Build (or rebuild) the top-hit index for a Ragged store."""
    build_ragged_top_hit_indexes(store_path)
    typer.echo("done")


@app.command("build-dense-top-hits")
def build_dense_top_hits_command(store_path: Path) -> None:
    """Build (or rebuild) the top-hit index for a Dense store."""
    build_top_hit_indexes(store_path)
    typer.echo("done")


@app.command("build-dense-rho")
def build_dense_rho_command(
    store_path: Path,
    window_bp: int = typer.Option(
        DEFAULT_RHO_WINDOW_BP, help="Distance-thinning window (bp) over the store's own axis"
    ),
    z_thresh: float = typer.Option(
        DEFAULT_RHO_Z_THRESH, help="|z| cutoff for a variant to count as null"
    ),
    min_nulls: int = typer.Option(
        DEFAULT_RHO_MIN_NULLS, help="Minimum shared null-variant support; NaN below this"
    ),
    n_workers: int = typer.Option(1, help="Process pool size for the per-pair MLE"),
) -> None:
    """Build (or rebuild) the pairwise Rho Matrix for a Dense store."""
    build_dense_rho(
        store_path,
        window_bp=window_bp,
        z_thresh=z_thresh,
        min_nulls=min_nulls,
        n_workers=n_workers,
    )
    typer.echo("done")


@app.command("regenerate-overview")
def regenerate_overview_command(store_path: Path) -> None:
    """Rewrite overview.html from a store's already-persisted data --
    analyses.tsv, manifest.json, and a directory scan (issue #23 AC3, ADR
    0032). No other store artifact is rebuilt or touched."""
    table = read_analyses(store_path / "analyses.tsv")
    out_path = write_overview_html(store_path, table)
    typer.echo(f"wrote {out_path}")


_FORMAT_OPTION = typer.Option(
    OutputFormat.tsv,
    "--format",
    help="tsv (default): resolved, human-readable rows. json: the raw index-keyed result.",
)
_VARIANT_INFO_OPTION = typer.Option(
    False,
    "--variant-info",
    help=(
        "Include the per-variant rsid column in tsv output. Off by default: "
        "unlike chromosome/position/alleles (derived for free from the alid) "
        "and eaf (already materialised in the query result), rsid isn't "
        "derivable in-store and costs an extra variants.tsv.gz lookup that can "
        "dominate query time on a large result."
    ),
)


@app.command("query-phewas")
def query_phewas_command(
    store_path: Path,
    identifier: str,
    output_format: OutputFormat = _FORMAT_OPTION,
    include_variant_info: bool = _VARIANT_INFO_OPTION,
) -> None:
    """Extract one variant across all analyses (PheWAS)."""

    query = query_store(store_path)
    _emit(query, query.phewas(identifier), output_format, include_variant_info)


@app.command("query-range-phewas")
def query_range_phewas_command(
    store_path: Path,
    chromosome: str,
    start: int,
    end: int,
    output_format: OutputFormat = _FORMAT_OPTION,
    include_variant_info: bool = _VARIANT_INFO_OPTION,
) -> None:
    """Regional PheWAS: all variants in a genomic range across all analyses."""

    query = query_store(store_path)
    _emit(query, query.range_phewas(chromosome, start, end), output_format, include_variant_info)


@app.command("query-analysis")
def query_analysis_command(
    store_path: Path,
    analysis_id: str,
    output_format: OutputFormat = _FORMAT_OPTION,
    include_variant_info: bool = _VARIANT_INFO_OPTION,
) -> None:
    """Extract all finite associations for one analysis."""

    query = query_store(store_path)
    _emit(query, query.analysis(analysis_id), output_format, include_variant_info)


@app.command("query-lookup")
def query_lookup_command(
    store_path: Path,
    identifiers: str,
    analysis_ids: str,
    output_format: OutputFormat = _FORMAT_OPTION,
    include_variant_info: bool = _VARIANT_INFO_OPTION,
) -> None:
    """Query comma-separated variants against comma-separated analyses."""

    query = query_store(store_path)
    result = query.lookup(
        [item for item in identifiers.split(",") if item],
        [item for item in analysis_ids.split(",") if item],
    )
    _emit(query, result, output_format, include_variant_info)


@app.command("query-top-hits")
def query_top_hits_command(
    store_path: Path,
    threshold: float = typer.Option(5e-8),
    limit: int | None = typer.Option(None),
    output_format: OutputFormat = _FORMAT_OPTION,
    include_variant_info: bool = _VARIANT_INFO_OPTION,
) -> None:
    """Return ranked top-hit associations."""

    query = query_store(store_path)
    result = query.top_hits(threshold=threshold, limit=limit)
    _emit(query, result, output_format, include_variant_info)


QueryFacade = StoreQuery | RaggedStoreQuery | HybridStoreQuery


def _emit(
    query: QueryFacade,
    result: dict[str, np.ndarray],
    output_format: OutputFormat,
    include_variant_info: bool,
) -> None:
    if output_format is OutputFormat.json:
        _emit_json(result)
    else:
        _emit_tsv(query, result, include_variant_info)


def _emit_json(result: dict[str, np.ndarray]) -> None:
    rows = [
        {
            "variant_index": int(vi),
            "analysis_index": int(ai),
            "z": float(z),
            "se": float(se),
        }
        for vi, ai, z, se in zip(
            result["variant_index"],
            result["analysis_index"],
            result["z"],
            result["se"],
            strict=True,
        )
    ]
    typer.echo(json.dumps(rows, sort_keys=True))


_TSV_COLUMNS = (
    "analysis_id",
    "analysis_label",
    "chromosome",
    "position",
    "alid",
    "effect_allele",
    "other_allele",
    "z",
    "se",
    "p",
    "eaf",
    "association_status",
)
# The variant-info header is the default columns with rsid inserted after
# analysis_label, so the two can never drift apart.
_TSV_COLUMNS_WITH_VARIANT_INFO = (
    *_TSV_COLUMNS[:2],
    "rsid",
    *_TSV_COLUMNS[2:],
)


def _format_p(log10_p: float) -> str:
    """Two-sided p as a compact display string. Past the point float64 can no
    longer represent the value (|z| >~ 39; the FADS1/FADS2 window reaches
    |z| = 47.8) an explicit ``<1e-300`` is printed instead of a silent 0
    (issue #104)."""
    if math.isnan(log10_p):
        return "NA"
    if log10_p < -300:
        return "<1e-300"
    return f"{10.0 ** log10_p:.3g}"


def _format_eaf(eaf: object) -> str:
    """Display one resolved eaf, or ``.`` when the store has none.

    `resolve_rows` already turns an absent array and a NaN cell into the same
    ``.`` (ADR 0036); this handles that string and a missing key (a caller
    that resolved without eaf) identically, and formats a real value ``.6g``
    like z/se -- so the column never prints a bare ``nan``.
    """
    if eaf is None or isinstance(eaf, str):
        return eaf if eaf is not None else "."
    if not isinstance(eaf, (int, float)):
        return "."
    value = float(eaf)
    return "." if not math.isfinite(value) else f"{value:.6g}"


def _emit_tsv(
    query: QueryFacade, result: dict[str, np.ndarray], include_variant_info: bool
) -> None:
    # query.resolve() writes each output row lazily as this loop consumes it
    # rather than building a Python list/string of the whole formatted table
    # up front. rsid is the one identity field not derivable from the alid
    # (VariantAxis.identity_by_indices()), so it's the only part of this
    # join that still needs a variants.tsv.gz lookup -- resolve() only pays
    # for it when include_variant_info is set (issue #104 follow-up). eaf,
    # by contrast, came back in the query result itself, so it is always a
    # column now (issue #136) and never costs a new read.
    writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    writer.writerow(_TSV_COLUMNS_WITH_VARIANT_INFO if include_variant_info else _TSV_COLUMNS)
    for row in query.resolve(result, include_variant_info=include_variant_info):
        fields: list[object] = [
            row["analysis_id"],
            row["analysis_label"],
            row["chromosome"],
            row["position"],
            row["alid"],
            row["effect_allele"],
            row["other_allele"],
            f"{row['z']:.6g}",
            f"{row['se']:.6g}",
            _format_p(cast(float, row["log10_p"])),
        ]
        if include_variant_info:
            fields.insert(2, row["rsid"])
        # eaf always sits before association_status, which stays the last column.
        fields.append(_format_eaf(row.get("eaf")))
        fields.append(row["association_status"])
        writer.writerow(fields)
