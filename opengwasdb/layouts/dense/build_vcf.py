"""Two-pass Dense Observed-Only writer from GWAS-VCF manifests with inline liftover.

Variant Routing (`opengwasdb.variants.routing`) decides where every source
coordinate goes: Pass 1's lifted union or a Variant Reference, the names each
routed ALID carries, and the route table each Pass 2 worker routes one
Analysis's associations through. This module writes what routing places --
the variant axis and index, the per-column spills, and the Dense Component
bands -- and reads every source through a ``SourceReader`` resolved from its
row's ``source_reader_capability`` (issue #20).
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from opengwasdb.admission import AdmissionRecord, AnalysisRules, Reason, admission_record_for
from opengwasdb.admission.declared import declared_rules
from opengwasdb.admission.duplicates import record_duplicates
from opengwasdb.build.eaf_orientation import (
    DEFAULT_SAMPLE_SITES,
    EafOrientationReport,
    apply_orientation_evidence,
    sample_column_rows,
    site_hashes,
    verify_eaf_orientation,
)
from opengwasdb.build.liftover import LiftoverFailureError
from opengwasdb.build.ordered_pool import fork_each, ordered_map
from opengwasdb.encoding import (
    EafExceptionBuilder,
    EafMeasurements,
    EncodingMeasurements,
    StoreCodec,
    StoreEncoding,
    ZOverflowBuilder,
    eaf_baseline_from_grid,
    measure_eaf_sample,
    optimise_dense_se,
    positions_row_band,
    write_eaf_baseline,
)
from opengwasdb.encoding.association_codec import (
    CellValues,
    Storable,
    log_se_out_of_range,
    storable,
)
from opengwasdb.encoding.codec import require_parallel, se_float16
from opengwasdb.encoding.timing import PhaseTimer, format_duration, log_phase
from opengwasdb.encoding.timing import log_progress as log_phase_progress
from opengwasdb.index import initialise_schema, set_metadata
from opengwasdb.layouts.build_manifest import (
    ManifestRow,
    apply_se_divisor,
    column_map_provenance,
    manifest_row_to_analysis,
    read_manifest,
)
from opengwasdb.layouts.dense.build import (
    DenseBuildResult,
    write_analyses_tsv,
)
from opengwasdb.layouts.dense.constants import (
    DEFAULT_CHUNK_SHAPE,
    DEFAULT_DTYPE,
    dense_index_metadata,
    dense_layout_records,
    dense_provenance_block,
)
from opengwasdb.layouts.dense.top_hits import write_top_hit_indexes_for_store
from opengwasdb.layouts.release_manifest import observed_only_release_manifest
from opengwasdb.model.analyses import Analysis
from opengwasdb.model.enums import (
    AssociationCoverage,
    EafScope,
    PrimaryStorageLayout,
    coverage_for_layout,
)
from opengwasdb.readers.column_map import ColumnMap
from opengwasdb.readers.interface import StreamTally
from opengwasdb.readers.registry import open_source
from opengwasdb.store.arrays import ArrayRole, compressor, create_array
from opengwasdb.store.open import OpenGWASDBStore, StagedRelease
from opengwasdb.top_hits import TOP_HIT_THRESHOLDS, add_hit_counts, z_critical
from opengwasdb.variants import CanonicalVariant, write_variant_axis
from opengwasdb.variants.reference import VariantReference, read_variant_reference
from opengwasdb.variants.routing import (
    FromReference,
    LiftedUnion,
    RouteTable,
    VariantRouting,
    route_variants,
)

log = logging.getLogger(__name__)

SURVEY_PROGRESS_EVERY = 25
"""Analyses between the EAF survey's progress lines."""

# One compressor for every dense statistic array (z/se/eaf), so a new array
# cannot quietly ship with different settings from the ones beside it.
_DENSE_COMPRESSOR = compressor()

__all__ = ["build_dense_from_vcf_manifest", "LiftoverFailureError"]

# Top hits are harvested inline during Pass 2 rather than by a post-hoc scan of
# the full matrix (which had to reload ~200 GB of float32 and compute a p-value
# for every finite cell). Each column emits only the cells clearing the loosest
# threshold's |z| cutoff; the band-write phase accumulates them for the index.
_TOP_HIT_Z_CRIT = z_critical(max(TOP_HIT_THRESHOLDS))


def _spill_column(
    spill_dir: Path,
    col_idx: int,
    rows: np.ndarray,
    z: np.ndarray,
    se: np.ndarray,
    eaf: np.ndarray,
) -> None:
    """Atomically spill one resolved column to ``{spill_dir}/{col_idx}.npz``
    (temp-then-rename; both names end in .npz because np.savez appends that suffix
    unless already present).

    Top hits are NOT harvested here — they are harvested during the band-write
    phase from the *stored* float16 values, so the index matches exactly what a
    query reads back from the ``z`` array (issue 046).

    ``rows``, ``z``, ``se`` and ``eaf`` are one entry per cell of the column, and
    one of another length is refused before the spill file exists: the
    band-write phase reads the four back as one column, so a short one would put
    a cell's statistic under another's variant (issue #392)."""
    require_parallel(
        (("rows", rows), ("z", z), ("se", se), ("eaf", eaf)),
        where=f"dense-vcf spill, analysis_index {col_idx}",
    )
    final = spill_dir / f"{col_idx}.npz"
    tmp = spill_dir / f"{col_idx}.tmp.npz"
    np.savez(tmp, rows=rows, z=z, se=se, eaf=eaf)
    tmp.replace(final)



def log_progress(
    label: str, completed: int, total: int, start_time: float, extra: str, every: int
) -> None:
    """Log a phase's progress, elapsed time and ETA every ``every`` items and at the end.

    The Dense VCF and Hybrid builds' own line, ``N/M done (extra)``; the encoding
    passes log through `opengwasdb.encoding.timing.log_progress`, whose line has
    no ``done``.
    """
    if completed % every != 0 and completed != total:
        return
    elapsed = time.monotonic() - start_time
    eta = (elapsed / completed) * (total - completed) if completed else 0.0
    log.info(
        "%s: %d/%d done (%s) — elapsed %s, ETA %s",
        label,
        completed,
        total,
        extra,
        format_duration(elapsed),
        format_duration(eta),
    )



def build_dense_from_vcf_manifest(
    manifest_path: str | Path, output_path: str | Path, *,
    chain_file: str | Path | None = None, store_id: str, release_id: str,
    liftover_failure_threshold: float = 0.01,
    chunk_shape: tuple[int, int] = DEFAULT_CHUNK_SHAPE,
    dtype: str = DEFAULT_DTYPE, overwrite: bool = False, n_workers: int = 1,
    eaf_reference: str | Path | None = None,
    eaf_reference_ancestry: str | None = None, allow_unverified_eaf: bool = False,
    source_reader_capability: str | None = None, source_assembly: str | None = None,
    variant_reference: str | Path | None = None, column_map: ColumnMap | str | Path | None = None,
    association_coverage: AssociationCoverage | str = AssociationCoverage.GWAS,
    admission_record: str | Path | None = None,
) -> DenseBuildResult:
    """Build a Dense Observed-Only Store from a manifest of GWAS-VCF files.

    A thin orchestrator over the private phases below (issue #130): the build
    is staged atomically at ``output_path``, its two streaming passes never
    materialise the full association matrix (issue 043), and every phase runs
    before the store is committed. Keyword semantics live with the phase that
    consumes each -- the manifest columns and per-release defaults (#174) with
    `read_manifest`; ``chain_file`` and ``liftover_failure_threshold`` with the
    hg19→hg38 lift (issue #85); ``eaf_reference``/``eaf_reference_ancestry``/
    ``allow_unverified_eaf`` with EAF orientation verification (issue #115, ADR 0037 §6).
    ``variant_reference`` (issue #185) supplies a precomputed axis
    (``*.variant-ref.tsv.gz``, an ALID list, or a store ``variants.tsv.gz``);
    Pass 1 and liftover are bypassed save its rsid harvest (#255). ``column_map`` reads
    ``opengwasdb.column-map`` rows (#281). ``association_coverage`` is a build input
    (issue #298) and a Dense matrix can carry only ``gwas``, its default (ADR 0064).
    ``admission_record`` is Association Admission's record (#357), beside the Store.
    """
    coverage = coverage_for_layout(association_coverage, PrimaryStorageLayout.DENSE)
    manifest_rows, rules = _manifest_and_rules(
        manifest_path, source_reader_capability, source_assembly, column_map
    )

    out = Path(output_path)
    with OpenGWASDBStore.staging(out, overwrite=overwrite) as staged:
        prepared = _prepare_axis(
            staged, out, manifest_rows, chain_file, liftover_failure_threshold, chunk_shape,
            n_workers, variant_reference, rules,
        )
        # Phases 5-7 run inside one spill-dir lifetime: every spill is removed
        # even when a phase fails, keeping the staged release atomic.
        eaf_report, encoded = _spill_verify_and_encode(
            staged, out, manifest_rows, prepared, n_workers, chunk_shape, dtype,
            eaf_reference, eaf_reference_ancestry, allow_unverified_eaf,
            admission_record_for(out, admission_record),
        )
        # Phase 8: top-hit indexes, manifest and analyses.tsv metadata.
        _finalize_store(
            staged, prepared, encoded, eaf_report, store_id, release_id, chain_file,
            chunk_shape, dtype, allow_unverified_eaf, n_workers, coverage,
        )
        log.info(
            "Build complete: %d variants × %d analyses",
            len(prepared.axis.alids), len(prepared.axis.analyses),
        )

    return DenseBuildResult(out, len(prepared.axis.alids), len(prepared.axis.analyses))


def _manifest_and_rules(
    manifest_path: str | Path,
    source_reader_capability: str | None,
    source_assembly: str | None,
    column_map: ColumnMap | str | Path | None,
) -> tuple[list[ManifestRow], dict[str, AnalysisRules]]:
    """The manifest's rows, and each Analysis's declared INFO and MAF filters (D2)."""
    rows = read_manifest(
        manifest_path, default_source_reader_capability=source_reader_capability,
        default_source_assembly=source_assembly, column_map=column_map,
    )
    rules = declared_rules(
        manifest_path, {row.trait_id: row.source_reader_capability for row in rows}
    )
    return rows, rules


@dataclass(frozen=True)
class _AxisMetadata:
    """The store's axis: sorted hg38 ALIDs, the row maps, and Analysis records.

    ``analyses`` are the manifest's Analytical Metadata (issue #17, #86) --
    derived from each row, never from a VCF header.
    """

    alids: list[str]
    analysis_index: dict[str, int]
    analyses: list[Analysis]


@dataclass(frozen=True)
class _HitCandidates:
    """The cells the band-write's z pass harvested above the loosest threshold.

    Harvested from the *stored* values so the top-hit index matches exactly
    what a query reads back from the ``z`` array (issue 046)."""

    rows: np.ndarray
    cols: np.ndarray
    z: np.ndarray
    se: np.ndarray


@dataclass(frozen=True)
class _EncodedBands:
    """Phase-7 outcome the index and metadata phases consume."""

    encoding: StoreEncoding
    hits: _HitCandidates
    column_has_eaf: np.ndarray


@dataclass(frozen=True)
class _PreparedBuild:
    """What the preparation phase hands the spill phases: the store's axis and
    the fork-safe Pass 2 route table built against it.

    The routing's site map is consumed when the table is built, before any
    worker forks (issue 043).
    """

    axis: _AxisMetadata
    table: RouteTable
    #: Each Analysis's declared INFO and MAF filters, by Analysis id (D2).
    rules: Mapping[str, AnalysisRules] = field(default_factory=dict)
    #: The ``--variant-reference`` a single-pass build was given, or None for
    #: the inline two-pass build. Recorded in the store manifest's provenance
    #: so the axis's origin is auditable (issue #185).
    variant_reference: str | None = None
    #: The Column Map mapped rows were read through, as `column_map_provenance`
    #: records it, so the store names it as a Resolution Record does (#281).
    column_map: dict[str, str] | None = None


def _axis_metadata(alids: list[str], manifest_rows: Sequence[ManifestRow]) -> _AxisMetadata:
    """The store's axis: the routed ALIDs in genomic order, and the Analyses.

    The ALIDs' (chromosome, position, a1, a2) order gives the variant axis a
    stable, position-ordered row index; the manifest's own row order fixes the
    analysis column order.
    """
    return _AxisMetadata(
        alids=alids,
        analysis_index={row.trait_id: i for i, row in enumerate(manifest_rows)},
        analyses=[manifest_row_to_analysis(row) for row in manifest_rows],
    )


def _write_axis_and_index(
    staged: StagedRelease,
    routing: VariantRouting,
    axis: _AxisMetadata,
    rsid_by_alid: Mapping[str, str],
    chunk_shape: tuple[int, int],
) -> None:
    """Write the SQLite index and the tabix variant axis.

    The variant axis carries one canonical variant per stored row with its
    rsid (first named wins) and the source-build provenance each row resolved
    from (`VariantRouting.source_alids` -- collisions stay blank, never guessed).
    """
    write_index(staged, axis.alids, axis.analyses, chunk_shape)
    canonical_variants = [
        CanonicalVariant(
            chromosome=chrom,
            position=int(pos_str),
            effect_allele=a1,
            other_allele=a2,
        )
        for alid in axis.alids
        for chrom, pos_str, a1, a2 in [alid.split(":")]
    ]
    origin = routing.source_alids()
    source_alids = [origin.get(alid) for alid in axis.alids]
    del origin
    write_variant_axis(staged.path, canonical_variants, rsid_by_alid, source_alids)


def _routing_authority(
    manifest_rows: Sequence[ManifestRow],
    chain_file: str | Path | None,
    liftover_failure_threshold: float,
    variant_reference: str | Path | None,
) -> LiftedUnion | FromReference:
    """The manifest's own lifted union, or the precomputed reference (issue #185)."""
    if variant_reference is None:
        return LiftedUnion(chain_file, liftover_failure_threshold)
    return FromReference(
        _load_variant_reference(variant_reference, manifest_rows),
        chain_file,
        liftover_failure_threshold,
    )


def _load_variant_reference(
    variant_reference: str | Path, manifest_rows: Sequence[ManifestRow]
) -> VariantReference:
    """Read the reference and note when it cannot resolve the manifest's assembly.

    An identity reference (a plain ALID list or a store ``variants.tsv.gz``)
    carries no source-key mapping, so it resolves only rows already on hg38.
    The reference is the axis authority and no liftover runs in Pass 2, so a
    manifest declaring hg19 against one keeps only coordinates that happen to
    coincide; that is the documented off-reference behaviour, but it is worth
    saying so rather than letting the shortfall pass unremarked.
    """
    reference = read_variant_reference(variant_reference)
    log.info(
        "Single-pass build: variant axis loaded from %s (%d variants); "
        "Pass 1 variant union and liftover bypassed",
        variant_reference,
        len(reference.alids),
    )
    if not reference.explicit_source_keys:
        non_hg38 = sorted(
            {row.source_assembly for row in manifest_rows if row.source_assembly != "hg38"}
        )
        if non_hg38:
            log.warning(
                "Variant reference %s carries no source-key mapping and resolves only hg38 "
                "coordinates, but the manifest declares %s; variants whose source coordinate "
                "differs from the reference will be dropped",
                variant_reference,
                ", ".join(non_hg38),
            )
    return reference


def _prepare_axis(
    staged: StagedRelease,
    out: Path,
    manifest_rows: list[ManifestRow],
    chain_file: str | Path | None,
    liftover_failure_threshold: float,
    chunk_shape: tuple[int, int],
    n_workers: int,
    variant_reference: str | Path | None,
    rules: Mapping[str, AnalysisRules],
) -> _PreparedBuild:
    """Phases 1-4: route, write the axis and index, and build the Pass 2 table.

    hg19-declared rows are lifted to GRCh38 once (issue #85), or a precomputed
    ``variant_reference`` supplies the axis directly and Pass 1 only harvests
    the names it lacks (single-pass build, issues #185, #255). Either way the
    provenance-carrying variant axis is written and checked against the names
    it was written from, and the route table is built -- consuming the
    routing's site map before any worker forks (issue 043).
    """
    authority = _routing_authority(
        manifest_rows, chain_file, liftover_failure_threshold, variant_reference
    )
    routing = route_variants(
        [row.source for row in manifest_rows], authority, n_workers=n_workers
    )
    if routing.names.pending:
        # A Dense build drops the associations the reference does not route, so
        # nothing else names a variant; the harvest it held is freed here.
        routing.names = routing.names.finish()
        log.info(
            "Single-pass build: reference names no rsids; Pass 1 harvest resolved %d for "
            "the variant axis",
            len(routing.names.rsid_by_alid),
        )
    axis = _axis_metadata(routing.alids, manifest_rows)
    _write_axis_and_index(staged, routing, axis, routing.names.rsid_by_alid, chunk_shape)
    # The table just written must be the table the resolved map describes: a
    # partial harvest loss, an off-reference row dropped from the write, or a
    # table written from a stale map fails here, on the bytes (issue #255).
    routing.names.require_written(staged.path, axis.alids)
    return _PreparedBuild(
        axis=axis,
        # Its unmatched rows are kept by key so the gate can log them (D6).
        table=routing.table([axis.alids], keep_unmatched=True),
        variant_reference=str(variant_reference) if variant_reference is not None else None,
        column_map=column_map_provenance(manifest_rows),
        rules=rules,
    )


@dataclass(frozen=True)
class _Pass2Context:
    """What every Pass 2 worker reads: the route table, where to spill, and the
    Admission record its gate writes to."""

    table: RouteTable
    spill_dir: Path
    record: AdmissionRecord
    alids: Sequence[str]
    rules: Mapping[str, AnalysisRules]


def _route_and_spill(context: _Pass2Context, task: tuple[int, ManifestRow]) -> int:
    """Route one Analysis's column through the shared table and spill it.

    Returns the column index only: the compact result stays on disk, never in
    a pipe. An earlier design returned each column over IPC, and at
    genome-wide scale (~9.85M rows per file, thousands of files) that pipe
    traffic deadlocked the pool.
    """
    col_idx, row = task
    rules = context.rules[row.trait_id]
    reader = open_source(
        row.source, imputation_score_declaration=rules.info.imputation_score_declaration
    )
    tally = StreamTally()
    with context.record.gate(rules) as gate:
        routed = context.table.route_column(
            reader.stream_associations(tally),
            admit=gate.admit,
            imputation_scores=gate.reads_imputation_scores,
        )
        gate.streamed(tally)
        placed = routed.placed[0]
        record_duplicates(gate, placed, context.alids)
        gate.unplaced(Reason.OFF_AXIS, routed.unmatched.keys)
        se = apply_se_divisor(placed.se, row.se_divisor)
        where = f"build-dense-vcf, Analysis {row.trait_id}"
        kept = storable(CellValues(placed.z, se, placed.eaf), 0, 1, where=where)
        log_se_out_of_range(kept.se_out_of_range, where=where)
        _spill_stored(context.spill_dir, col_idx, placed.index, kept)
    return col_idx


def _spill_stored(spill_dir: Path, col_idx: int, rows: np.ndarray, kept: Storable) -> None:
    """Spill the column's rows `storable` keeps, with the values it returns.

    A spill holds rows, so a row it does not keep is left out, and its cell in
    the grid stays missing with no frequency.
    """
    stored, cells = kept.stored, kept.cells
    _spill_column(
        spill_dir,
        col_idx,
        rows[stored],
        np.asarray(cells.z)[stored],
        np.asarray(cells.se)[stored],
        np.asarray(cells.eaf)[stored],
    )


def _spill_columns(
    manifest_rows: Sequence[ManifestRow],
    spill_dir: Path,
    prepared: _PreparedBuild,
    admission_record: Path,
    n_workers: int,
    pass2_start: float,
) -> None:
    """Pass 2: route and spill one column per Analysis, here or in a fork pool,
    each through its Admission gate, then finish the record the gates wrote.

    With ``n_workers <= 1`` each column is routed in this process, one Analysis
    at a time, so peak memory is one routed column; with more, the workers
    inherit the route table at fork (`fork_each`) and write only their own
    ``.npz``, so nothing large crosses the pipe.
    """
    log.info(
        "Pass 2: resolving %d analyses × %d variants (n_workers=%d)",
        len(prepared.axis.analyses),
        len(prepared.axis.alids),
        n_workers,
    )
    id_by_col = {prepared.axis.analysis_index[row.trait_id]: row.trait_id for row in manifest_rows}
    tasks = [(prepared.axis.analysis_index[row.trait_id], row) for row in manifest_rows]
    record = AdmissionRecord.open(
        admission_record, [row.trait_id for row in manifest_rows], command="build-dense-vcf"
    )
    context = _Pass2Context(prepared.table, spill_dir, record, prepared.axis.alids, prepared.rules)
    columns = fork_each(_route_and_spill, context, tasks, n_workers)
    for i, col_idx in enumerate(columns):
        log_progress(
            "Pass 2",
            i + 1,
            len(prepared.axis.analyses),
            pass2_start,
            f"last: {id_by_col[col_idx]}",
            every=25,
        )
    record.finish()


def _survey_and_verify_eaf(
    spill_dir: Path,
    manifest_rows: Sequence[ManifestRow],
    axis: _AxisMetadata,
    *,
    eaf_reference: str | Path | None,
    eaf_reference_ancestry: str | None,
    allow_unverified_eaf: bool,
    n_workers: int,
) -> tuple[EafSpillSurvey, EafOrientationReport]:
    """Sample each Analysis's stored frequencies and verify their orientation.

    The check (issue #115, ADR 0037 §6) runs off the Pass 2 spills -- the exact
    values about to be stored -- and before any statistic array is written, so
    a build that would store a frequency column against the wrong allele fails
    here rather than after an hour of band-writing. The survey also measures
    the frequency spread the encoding plan reads, so both answer in one pass.
    ``allow_unverified_eaf`` records Analyses the supplied reference could not
    verify in the store's provenance instead of rejecting them.
    """
    id_by_col = {axis.analysis_index[row.trait_id]: row.trait_id for row in manifest_rows}
    with log_phase(log, "EAF site hashes"):
        hashes = site_hashes(axis.alids)
    eaf_survey = survey_eaf_spills(spill_dir, id_by_col, axis.alids, hashes, n_workers=n_workers)
    eaf_report = verify_eaf_orientation(
        eaf_survey.observations,
        eaf_reference=eaf_reference,
        eaf_reference_ancestry=eaf_reference_ancestry,
        allow_unverified=allow_unverified_eaf,
    )
    return eaf_survey, eaf_report


def _write_encoded_bands(
    staged: StagedRelease,
    spill_dir: Path,
    eaf_survey: EafSpillSurvey,
    axis: _AxisMetadata,
    chunk_shape: tuple[int, int],
    dtype: str,
    pass2_start: float,
    n_workers: int,
) -> _EncodedBands:
    """Create the statistic arrays and fill them from the spills.

    One encoding plan per build is decided here -- after Pass 2 and the EAF
    survey, because the ``eaf`` rules read the frequencies the sources
    actually carried -- and recorded in manifest.json (ADR 0037, issue #119).
    The z/se planes are then filled in chunk-column bands so peak memory is
    one band (issue 043), the eaf plane is written under the plan, and
    ``optimise_dense_se`` rewrites ``se`` under the plan's codec. The
    top-hit candidates are harvested in the same pass (issue 046).
    """
    n_variants, n_analyses = len(axis.alids), len(axis.analyses)
    encoding = StoreEncoding.decide(
        EncodingMeasurements(
            n_analyses=n_analyses,
            eaf=eaf_survey.measurements(
                n_cells=n_variants * n_analyses, n_variants=n_variants
            ),
        )
    )
    log.info("Encoding plan: %s", encoding.to_manifest())
    effective_chunks = create_dense_zarr(
        staged, n_variants, n_analyses, chunk_shape, dtype, encoding
    )
    rows, cols, z, se, column_has_eaf = write_dense_bands(
        staged,
        spill_dir,
        n_variants,
        n_analyses,
        effective_chunks,
        dtype,
        pass2_start,
        encoding,
        n_workers,
    )
    encoding = optimise_dense_se(staged.arrays(mode="a"), encoding, n_workers=n_workers)
    return _EncodedBands(
        encoding=encoding,
        hits=_HitCandidates(rows=rows, cols=cols, z=z, se=se),
        column_has_eaf=column_has_eaf,
    )


def _spill_verify_and_encode(
    staged: StagedRelease,
    out: Path,
    manifest_rows: Sequence[ManifestRow],
    prepared: _PreparedBuild,
    n_workers: int,
    chunk_shape: tuple[int, int],
    dtype: str,
    eaf_reference: str | Path | None,
    eaf_reference_ancestry: str | None,
    allow_unverified_eaf: bool,
    admission_record: Path,
) -> tuple[EafOrientationReport, _EncodedBands]:
    """Phases 5-7: spill each column, verify EAF, and write the encoded planes.

    Runs inside the spill dir's lifecycle: every spill is removed when this
    returns or raises, so a phase failure leaves nothing behind and the staged
    release stays atomic. ``n_workers`` > 1 resolves through the fork pool;
    EAF orientation is verified against ``eaf_reference`` before any array is
    written (issue #115, ADR 0037 §6). Pass 2 writes ``admission_record``.
    """
    spill_dir = Path(
        tempfile.mkdtemp(prefix=f".{out.name}.pass2spill.", dir=staged.path.parent)
    )
    try:
        pass2_start = time.monotonic()
        _spill_columns(
            manifest_rows,
            spill_dir,
            prepared,
            admission_record,
            n_workers,
            pass2_start,
        )
        with log_phase(log, "EAF survey and orientation verification"):
            eaf_survey, eaf_report = _survey_and_verify_eaf(
                spill_dir,
                manifest_rows,
                prepared.axis,
                eaf_reference=eaf_reference,
                eaf_reference_ancestry=eaf_reference_ancestry,
                allow_unverified_eaf=allow_unverified_eaf,
                n_workers=n_workers,
            )
        encoded = _write_encoded_bands(
            staged,
            spill_dir,
            eaf_survey,
            prepared.axis,
            chunk_shape,
            dtype,
            pass2_start,
            n_workers,
        )
    finally:
        shutil.rmtree(spill_dir, ignore_errors=True)
    return eaf_report, encoded


def _finalize_store(
    staged: StagedRelease,
    prepared: _PreparedBuild,
    encoded: _EncodedBands,
    eaf_report: EafOrientationReport,
    store_id: str,
    release_id: str,
    chain_file: str | Path | None,
    chunk_shape: tuple[int, int],
    dtype: str,
    allow_unverified_eaf: bool,
    n_workers: int,
    association_coverage: AssociationCoverage,
) -> None:
    """Phase 8: write the store's final metadata.

    The manifest names the encoding and the EAF-orientation evidence; the
    top-hit indexes are written from the band-write's stored-value harvest
    (issue 046); analyses.tsv is written last, with each Analysis's
    ``eaf_scope`` stamped from what was actually stored (ADR 0036) and the
    orientation evidence applied (issue #115).
    """
    axis = prepared.axis
    _write_manifest(
        staged,
        store_id,
        release_id,
        len(axis.alids),
        len(axis.analyses),
        chain_file,
        chunk_shape,
        dtype,
        encoding=encoded.encoding,
        eaf_orientation=eaf_report.provenance(allow_unverified=allow_unverified_eaf),
        variant_reference=prepared.variant_reference,
        column_map=prepared.column_map,
        association_coverage=association_coverage,
    )
    write_top_hit_indexes_for_store(
        staged.path, encoded.hits.rows, encoded.hits.cols, encoded.hits.z, encoded.hits.se,
        encoded.encoding, n_workers=n_workers,
    )
    analyses = apply_orientation_evidence(
        apply_eaf_scope(axis.analyses, encoded.column_has_eaf), eaf_report
    )
    write_analyses_tsv(staged.path, add_hit_counts(staged.path, analyses))


def write_index(
    staged: StagedRelease,
    hg38_alids: list[str],
    analyses: list[Analysis],
    chunk_shape: tuple[int, int],
) -> None:
    """Write a Dense release's ``index.sqlite``: its schema, counts and Dense layout."""
    with staged.index_connection() as connection:
        initialise_schema(connection)
        set_metadata(connection, "schema_version", 1)
        set_metadata(connection, "n_variants", len(hg38_alids))
        set_metadata(connection, "n_analyses", len(analyses))
        set_metadata(
            connection,
            "dense",
            dense_index_metadata((len(hg38_alids), len(analyses)), hint=chunk_shape),
        )
        connection.commit()


def apply_eaf_scope(analyses: list[Analysis], column_has_eaf: np.ndarray) -> list[Analysis]:
    """Stamp each Analysis's `eaf_scope` from what the build actually stored.

    Derived, never copied from the manifest (ADR 0036): only the build knows
    whether the source file turned out to carry a usable frequency, so a
    manifest claiming EAF for an Analysis whose file has none must not be able
    to make the store declare it. `column_has_eaf` is positional, matching
    `analysis_index`.
    """
    return [
        replace(
            analysis,
            eaf_scope=(
                EafScope.ASSOCIATION.value if bool(column_has_eaf[index]) else EafScope.ABSENT.value
            ),
        )
        for index, analysis in enumerate(analyses)
    ]


def _eaf_row_band(shard_rows: int, n_analyses: int) -> int:
    """Rows per band for the re-encode pass: whole shard rows, ~50M cells.

    A row band must be a whole number of Dense shards (issue #247): a band that
    ends inside a shard turns that shard's write into a read-modify-write.  The
    band is at least one shard, then as many whole shards as fit in the ~50M-cell
    memory target.
    """
    rows = max(int(shard_rows), 1)
    target = max(50_000_000 // max(n_analyses, 1), 1)
    return max(rows, (target // rows) * rows)


def _write_dense_eaf(
    staged: StagedRelease,
    spill_dir: Path,
    n_variants: int,
    n_analyses: int,
    effective_chunks: tuple[int, int],
    band_cols: int,
    codec: StoreCodec,
    pass2_start: float,
    n_workers: int,
    column_has_eaf: np.ndarray,
    encoding: StoreEncoding,
) -> None:
    """Write the `eaf` plane the store's plan declares.

    A `float32` plane is written straight from the spills, one chunk-column
    band at a time, exactly as ADR 0036 shipped it.

    An `int8_residual` plane cannot be, and this is the one place in the build
    where that costs a pass. The baseline is per *variant*, so encoding a cell
    needs every Analysis's frequency at that variant; the spills are per
    *Analysis*. So the frequencies are staged as `float32` in column bands, and
    then read back in row bands -- where a whole variant is resident -- to
    compute each baseline and encode against it. The staging array is deleted
    afterwards, and peak disk is 5 bytes per cell against the 4 the `float32`
    plane occupied on its own.

    What is written is decided by the plan, not by whether the array happens to
    be wanted here: a Hybrid release's two components share one plan, and a
    Dense Component with no EAF where the Ragged Overflow has some would
    otherwise declare a plane it does not have. An all-absent `int8` plane
    costs essentially nothing compressed.
    """
    if column_has_eaf.any() and encoding.eaf.is_absent:
        raise ValueError(
            "the encoding plan declares no eaf plane, but "
            f"{int(column_has_eaf.sum())} of {n_analyses} Analyses carried a frequency; "
            "the plan and the data disagree (ADR 0037 §2)"
        )
    if encoding.eaf.is_absent:
        return
    residual = codec.encoding.eaf.is_residual
    staging = _EAF_STAGING if residual else "eaf"
    _create_eaf_array(staged, n_variants, n_analyses, effective_chunks, name=staging)
    root = staged.arrays(mode="a")
    eaf_zarr = root[staging]
    loader = partial(_load_eaf_column, spill_dir, n_variants)
    with log_phase(log, "Band-write eaf"):
        for _c, _loaded, flush in _ordered_bands(
            loader, n_analyses, n_workers, n_variants, band_cols, "float32", "eaf"
        ):
            if flush is not None:
                _flush_band(eaf_zarr, flush, "Band-write eaf", n_analyses, pass2_start, band_cols)
    if not residual:
        return
    _encode_residual_dense_eaf(
        staged, n_variants, n_analyses, effective_chunks, codec, pass2_start
    )


def _encode_residual_dense_eaf(
    staged: StagedRelease,
    n_variants: int,
    n_analyses: int,
    effective_chunks: tuple[int, int],
    codec: StoreCodec,
    pass2_start: float,
) -> None:
    """Residual re-encode: transpose the staged float32 grid, then delete it.

    The staging array is only deleted once the baseline and exception tables it
    produced are both written -- a failed encode leaves the staging plane in
    place so a resume can tell the pass never finished. Full-float staging and
    exact exception values are preserved; nothing is encoded by column or
    clipped.
    """
    root = staged.arrays(mode="a")
    encoded = _create_eaf_array(
        staged,
        n_variants,
        n_analyses,
        effective_chunks,
        dtype=codec.eaf_dtype,
        fill_value=codec.eaf_fill_value,
    )
    baseline = np.full(n_variants, np.nan, dtype=np.float32)
    exceptions = EafExceptionBuilder()
    band_rows = _eaf_row_band(_shard_rows(encoded, effective_chunks), n_analyses)
    for r0 in range(0, n_variants, band_rows):
        r1 = min(r0 + band_rows, n_variants)
        block = np.asarray(root[_EAF_STAGING][r0:r1], dtype=np.float32)
        rows_baseline = eaf_baseline_from_grid(block)
        baseline[r0:r1] = rows_baseline
        encoded[r0:r1] = codec.encode_eaf(
            block,
            baseline=np.repeat(rows_baseline[:, None], n_analyses, axis=1),
            positions=positions_row_band(r0, n_analyses),
            exceptions=exceptions,
        )
        log_progress("Encode eaf", r1, n_variants, pass2_start, f"rows {r0}:{r1}", every=band_rows)
    write_eaf_baseline(root, baseline, compressor=_DENSE_COMPRESSOR)
    exceptions.table().write(root)
    log.info(
        "eaf: int8 residual at +/-%.1f, %d exception cell(s)",
        codec.encoding.eaf.residual_range,
        len(exceptions),
    )
    del root[_EAF_STAGING]


#: Name of the transient `float32` plane a residual-coded build stages into.
#: The spills are per Analysis and the baseline is per variant, so the grid has
#: to be transposed through something; staging it is the honest way to say so.
_EAF_STAGING = "eaf_source"


def _create_eaf_array(
    staged: StagedRelease,
    n_variants: int,
    n_analyses: int,
    effective_chunks: tuple[int, int],
    *,
    name: str = "eaf",
    dtype: str = "float32",
    fill_value: Any = float("nan"),
) -> Any:
    """Create the missing-filled `eaf` array (ADR 0036, ADR 0037 §2).

    Created after Pass 2, not alongside `z`/`se`, because how -- and whether --
    a build stores frequencies is only known once the sources have been read.
    A build whose plan says `absent` writes no array at all, so its store is
    byte-identical in shape to one built before EAF existed.

    Never `float16`: its spacing near 1.0 is 0.00049, and the canonical A1 is
    the lexicographically smaller allele rather than the minor one, so EAF near
    1 is ordinary here -- `float16` would silently round "MAF 1e-4" to "MAF 0"
    for half the ALID space.
    """
    root = staged.arrays(mode="a")
    return create_array(
        root,
        name,
        ArrayRole.DENSE_STATISTIC_PLANE,
        shape=(n_variants, n_analyses),
        dtype=dtype,
        fill_value=fill_value,
        compressor=_DENSE_COMPRESSOR,
        hint=effective_chunks,
        overwrite=True,
    )


def create_dense_zarr(
    staged: StagedRelease,
    n_variants: int,
    n_analyses: int,
    chunk_shape: tuple[int, int],
    dtype: str,
    encoding: StoreEncoding,
) -> tuple[int, int]:
    """Create the empty missing-filled z/se datasets and return the effective chunks.

    Chunk shape is clipped to the array dimensions (ADR-0021) so zarr's declared
    chunks match what is physically stored. The arrays are created without data;
    ``write_dense_bands`` fills them by chunk-column band afterwards. Each plane
    is filled with **its own** missing marker (spec §15): NaN for `se`, the
    reserved sentinel for a fixed-point `z`, so an untouched cell reads as
    missing under either encoding.
    """
    compressor = _DENSE_COMPRESSOR
    codec = StoreCodec(encoding)
    root = staged.arrays(mode="w")
    z_array = create_array(
        root,
        "z",
        ArrayRole.DENSE_STATISTIC_PLANE,
        shape=(n_variants, n_analyses),
        dtype=codec.z_dtype,
        fill_value=codec.z_fill_value,
        compressor=compressor,
        hint=chunk_shape,
    )
    # Chunk shape is clipped to the array dimensions (ADR 0021) by the role's
    # layout policy; read back the effective chunks it wrote.
    effective_chunks = (int(z_array.chunks[0]), int(z_array.chunks[1]))
    create_array(
        root,
        "se",
        # Scratch in float32, as dense.complete does: the band-writer fills
        # this before the SE encoding is decided, and an exact residual
        # exception must be the source's own value, not one already rounded
        # to the dtype the plane happened to start in (spec §6a).
        ArrayRole.DENSE_STATISTIC_PLANE,
        shape=(n_variants, n_analyses),
        dtype="float32",
        fill_value=float("nan"),
        compressor=compressor,
        hint=effective_chunks,
    )
    root.attrs["layout"] = "dense"
    root.attrs["completion_state"] = "observed_only"
    root.attrs.update(dense_layout_records((n_variants, n_analyses), hint=chunk_shape))
    return effective_chunks


# ── EAF orientation (issue #115, ADR 0037 §6) ────────────────────────────────
#
# Shared with the Hybrid builder, which spills the same per-column arrays. The
# check runs off the spills rather than re-reading the sources: the spilled
# `eaf` is exactly what the band-write is about to store, so what is verified
# is what a query will read back, and it costs no extra pass over the source
# files.


@dataclass(frozen=True)
class EafSpillSurvey:
    """One pass over the column spills, answering both EAF questions at once.

    The orientation check (§9.1) needs a deterministic per-Analysis sample of
    frequencies; the encoding tree (ADR 0037 §2) needs the residual spread over
    the same sample plus the build's exact cell counts. They are read together
    because the spills are large and the sample is the same one.
    """

    observations: dict[str, dict[str, float]]
    #: Cells the spills hold, whether or not they carry a frequency. For a
    #: Dense component this is not the plane's size -- the plane is the whole
    #: grid -- so the caller says which count the bytes should be reckoned on.
    n_spill_cells: int
    n_eaf_cells: int
    sample_rows: np.ndarray
    sample_values: np.ndarray

    def measurements(self, *, n_cells: int, n_variants: int) -> EafMeasurements:
        return measure_eaf_sample(
            self.sample_rows,
            self.sample_values,
            n_variants=n_variants,
            n_cells=n_cells,
            n_eaf_cells=self.n_eaf_cells,
        )


@dataclass(frozen=True)
class _SurveyContext:
    """Immutable per-call state a forked survey worker reads.

    Set as a module global immediately before `ordered_map` forks, so the
    workers inherit `hashes`/`row_map` without pickling them once per column.
    Cleared once the map is drained.
    """

    spill_dir: Path
    hashes: np.ndarray
    k: int
    suffix: str
    index_key: str
    row_map: np.ndarray | None


@dataclass(frozen=True)
class _SurveyColumn:
    """One Analysis's contribution to the survey: its sample and its counts.

    Carries the Analysis id so the parent can key `observations` and order the
    concatenated samples by Analysis, not by whichever column finished first.
    """

    analysis_id: str
    selected: np.ndarray
    values: np.ndarray
    n_spill_cells: int
    n_eaf_cells: int


_survey_context: _SurveyContext | None = None


def _survey_column(task: tuple[int, str]) -> _SurveyColumn | None:
    """Sample one Analysis's spill column. Runs in a forked worker.

    Returns `None` for a column with no spill file, which the parent reads as
    "this Analysis contributes nothing" rather than as a zero-valued sample.
    """
    col, analysis_id = task
    context = _survey_context
    if context is None:
        raise RuntimeError("survey context is not set; call survey_eaf_spills")
    path = context.spill_dir / f"{col}{context.suffix}.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        rows = data[context.index_key].astype(np.int64)
        if context.row_map is not None:
            rows = context.row_map[rows].astype(np.int64)
        eaf = np.asarray(data["eaf"], dtype=np.float64)
        n_spill_cells = int(eaf.size)
        n_eaf_cells = int(np.count_nonzero(np.isfinite(eaf)))
        selected, values = sample_column_rows(rows, eaf, context.hashes, k=context.k)
    return _SurveyColumn(
        analysis_id=analysis_id,
        selected=selected,
        values=values,
        n_spill_cells=n_spill_cells,
        n_eaf_cells=n_eaf_cells,
    )


@dataclass
class _SurveyTotals:
    """The running totals a survey accumulates across its columns."""

    n_spill_cells: int = 0
    n_eaf_cells: int = 0
    sample_rows: list[np.ndarray] = field(default_factory=list)
    sample_values: list[np.ndarray] = field(default_factory=list)


def _concatenate_or_empty(parts: list[np.ndarray], dtype: Any) -> np.ndarray:
    """The sample arrays concatenated in order, or an empty array of `dtype`."""
    return np.concatenate(parts) if parts else np.empty(0, dtype=dtype)


def _accumulate_survey_column(
    result: _SurveyColumn | None,
    alids: Sequence[str],
    observations: dict[str, dict[str, float]],
    totals: _SurveyTotals,
) -> None:
    """Fold one column's sample into the survey, in the order it is yielded."""
    if result is None:
        return
    totals.n_spill_cells += result.n_spill_cells
    totals.n_eaf_cells += result.n_eaf_cells
    totals.sample_rows.append(result.selected)
    totals.sample_values.append(result.values)
    observations[result.analysis_id].update(
        {
            alids[row]: float(value)
            for row, value in zip(result.selected.tolist(), result.values.tolist(), strict=True)
        }
    )


def _fold_survey(
    results: Iterable[_SurveyColumn | None],
    id_by_col: Mapping[int, str],
    alids: Sequence[str],
    *,
    label: str,
) -> EafSpillSurvey:
    """Fold the columns' samples into one survey, in the order they are yielded.

    `results` is lazy: the pool's work happens as this loop pulls from it, so
    the loop's own seconds are timed apart and reported once, summed over the
    Analyses, rather than as a phase of their own.
    """
    observations: dict[str, dict[str, float]] = {aid: {} for aid in id_by_col.values()}
    totals = _SurveyTotals()
    timer = PhaseTimer()
    total = len(id_by_col)
    started = time.monotonic()
    for completed, result in enumerate(results, start=1):
        with timer.phase("accumulate"):
            _accumulate_survey_column(result, alids, observations, totals)
        log_phase_progress(log, label, completed, total, started, every=SURVEY_PROGRESS_EVERY)
    log.info(
        "%s accumulation: done in %s across %d Analyses",
        label,
        format_duration(timer.seconds.get("accumulate", 0.0)),
        total,
    )
    return EafSpillSurvey(
        observations=observations,
        n_spill_cells=totals.n_spill_cells,
        n_eaf_cells=totals.n_eaf_cells,
        sample_rows=_concatenate_or_empty(totals.sample_rows, np.int64),
        sample_values=_concatenate_or_empty(totals.sample_values, np.float64),
    )


def survey_eaf_spills(
    spill_dir: Path,
    id_by_col: Mapping[int, str],
    alids: Sequence[str],
    hashes: np.ndarray,
    *,
    k: int = DEFAULT_SAMPLE_SITES,
    suffix: str = "",
    index_key: str = "rows",
    row_map: np.ndarray | None = None,
    n_workers: int = 1,
) -> EafSpillSurvey:
    """Sample each Analysis's frequencies and count them, from the spills.

    Every Analysis appears in `observations`, including ones whose source
    carried no frequency: an absent key would read as "not checked yet"
    downstream, where the honest answer is "checked, and there was nothing to
    check."

    `row_map` translates a spill's own row indices onto the axis `alids` and
    `hashes` describe — the Hybrid builder's Dense Component spills are indexed
    by dense row, while both of its components are sampled on the shared axis
    so that an Analysis living mostly off-panel is checked on the same footing
    as one sitting on it.

    The columns are independent, so `n_workers` > 1 surveys them through
    `ordered_map`: each worker loads, samples and counts one column, and the
    parent concatenates the samples and fills `observations` in Analysis order,
    never in completion order. `n_workers <= 1` is the serial path.
    """
    global _survey_context
    label = f"EAF survey ({suffix.lstrip('.')})" if suffix else "EAF survey"
    try:
        _survey_context = _SurveyContext(
            spill_dir=spill_dir,
            hashes=hashes,
            k=k,
            suffix=suffix,
            index_key=index_key,
            row_map=row_map,
        )
        with log_phase(log, label):
            results = ordered_map(_survey_column, list(id_by_col.items()), n_workers)
            return _fold_survey(results, id_by_col, alids, label=label)
    finally:
        _survey_context = None


@dataclass(frozen=True)
class _LoadedZeColumn:
    """One Analysis's decoded z column and everything the z pass derives from it.

    ``column`` is the Analysis's ``n_variants``-long encoded band column, with
    the plane's missing marker on every variant it does not observe -- the
    shape the band buffer wants, so the parent's share is a straight copy
    rather than a scatter. Shipping the dense column, not just its non-missing
    cells, is what keeps the process boundary cheap: for a Dense Component it
    is 2 bytes per variant against the spill's 10 bytes per association, and
    the scatter (and the fill) happen in the workers. The column index travels
    with it rather than being inferred from the order results arrive in, so a
    result can never be placed against the wrong Analysis (issue #220).
    """

    col: int
    column: np.ndarray
    has_eaf: bool
    overflow_index: np.ndarray
    overflow_value: np.ndarray
    hit_rows: np.ndarray
    hit_z: np.ndarray
    hit_se: np.ndarray


def _load_z_column(
    spill_dir: Path, n_variants: int, n_analyses: int, codec: StoreCodec, c: int
) -> _LoadedZeColumn:
    """Load one Analysis's spill and encode/harvest it, in a worker.

    A private overflow builder collects this column's out-of-range cells; the
    parent merges the returned entries in Analysis order, so the table is the
    one the serial loop built. The harvest thresholds the stored (quantised)
    z, exactly as the serial loop does -- not the source z.
    """
    with np.load(spill_dir / f"{c}.npz") as data:
        rows = data["rows"]
        overflow = ZOverflowBuilder()
        codes = codec.encode_z(
            data["z"],
            positions=rows.astype(np.int64) * n_analyses + c,
            overflow=overflow,
        )
        column = np.full(n_variants, codec.z_fill_value, dtype=codec.z_dtype)
        column[rows] = codes
        zc = codec.quantise_z(data["z"])  # what a query will read back
        hit = np.abs(zc) >= _TOP_HIT_Z_CRIT
        table = overflow.table()
        has_eaf = bool(np.isfinite(data["eaf"]).any())
        if np.any(hit):
            hit_rows = rows[hit]
            hit_z = zc[hit]
            # As a `float16` se plane holds it: the index's `se` is float32 and
            # must be the value the Store decodes at that cell, which for a
            # residual plan is the plane's own (`write_top_hit_indexes_for_store`
            # discards this and re-reads once the plan exists).
            hit_se = se_float16(data["se"][hit]).astype(np.float32)
        else:
            hit_rows = np.empty(0, dtype=np.int64)
            hit_z = np.empty(0, dtype=np.float32)
            hit_se = np.empty(0, dtype=np.float32)
        return _LoadedZeColumn(
            col=c,
            column=column,
            has_eaf=has_eaf,
            overflow_index=table.index,
            overflow_value=table.value,
            hit_rows=hit_rows,
            hit_z=hit_z,
            hit_se=hit_se,
        )


def _load_sparse_column(spill_dir: Path, n_variants: int, key: str, c: int) -> np.ndarray:
    """Scatter one spilled array onto a missing-filled float32 ``n_variants`` column."""
    with np.load(spill_dir / f"{c}.npz") as data:
        column = np.full(n_variants, np.nan, dtype=np.float32)
        column[data["rows"]] = data[key]
        return column


@dataclass(frozen=True)
class _LoadedColumn:
    """One Analysis's decoded band column, tagged with the Analysis it is from."""

    col: int
    column: np.ndarray


def _load_se_column(spill_dir: Path, n_variants: int, c: int) -> _LoadedColumn:
    """One Analysis's SE column, at the `float32` the source reported it.

    The column reaches the scratch plane unrounded because that plane is where
    a residual plan's exact exception table comes from (`_encode_band`), and
    spec §6a says such an exception is the source's own value. Narrowing an SE
    is the codec's: `_narrow_dense_se_to_float16` when the measured plan turns
    out to be `float16` (issue #355).
    """
    return _LoadedColumn(
        col=c, column=_load_sparse_column(spill_dir, n_variants, "se", c)
    )


def _load_eaf_column(spill_dir: Path, n_variants: int, c: int) -> _LoadedColumn:
    return _LoadedColumn(
        col=c, column=_load_sparse_column(spill_dir, n_variants, "eaf", c)
    )


def _ordered_bands(
    loader: Callable[[int], Any],
    n_analyses: int,
    n_workers: int,
    n_variants: int,
    band_cols: int,
    dtype: str,
    what: str,
) -> Iterator[tuple[int, Any, tuple[int, int, np.ndarray] | None]]:
    """Drive one band pass, yielding ``(column, loaded, band_to_write)``.

    ``loader(c)`` returns an object carrying ``.col`` and an ``n_variants``-long
    ``.column``; the band is filled a column at a time and yielded for writing
    at each band boundary, then replaced. So one band is the only large resident
    allocation, and loader results in flight are bounded by ``ordered_map``'s
    window. ``ordered_map`` yields in input order and a loader result whose
    ``.col`` does not match the Analysis being placed fails the build rather
    than being written into another Analysis's band slot (issue #220).
    """
    band: np.ndarray | None = None
    band_start = 0
    columns = ordered_map(loader, range(n_analyses), n_workers)
    for c, loaded in zip(range(n_analyses), columns, strict=True):
        if loaded.col != c:
            raise ValueError(
                f"band-write {what} columns arrived out of order: expected {c}, got {loaded.col}"
            )
        if band is None:
            band = np.empty((n_variants, band_cols), dtype=dtype)
            band_start = c
        local = c - band_start
        band[:, local] = loaded.column
        flush: tuple[int, int, np.ndarray] | None = None
        if local == band_cols - 1 or c == n_analyses - 1:
            flush = (band_start, c + 1, band)
            band = None
        yield c, loaded, flush


def _flush_band(
    arr: Any,
    flush: tuple[int, int, np.ndarray],
    label: str,
    n_analyses: int,
    pass2_start: float,
    band_cols: int,
) -> None:
    """Write one finished band to its plane and report progress."""
    start, stop, band = flush
    arr[:, start:stop] = band[:, : stop - start]
    log_progress(label, stop, n_analyses, pass2_start, f"cols {start}:{stop}", every=band_cols)


def _shard_columns(arr: Any, effective_chunks: tuple[int, int]) -> int:
    """The Analysis-axis width of a Dense plane's shard; the band width.

    A band write must cover whole shards, and a shard spans the whole Analysis
    axis of the band.  The shard is read from the array the builder just created
    rather than restated here, so this cannot drift from the seam's policy.  A
    v2 array (no shard) falls back to the inner chunk, which is what 0.1.0
    builds used before #247.
    """
    shards = getattr(arr, "shards", None)
    return int(shards[1]) if shards is not None else int(effective_chunks[1])


def _shard_rows(arr: Any, effective_chunks: tuple[int, int]) -> int:
    """The variant-axis height of a Dense plane's shard; the row-block unit.

    A row block must be a whole number of shards so that each of its writes
    covers whole shards.  Read from the array, as `_shard_columns` is.
    """
    shards = getattr(arr, "shards", None)
    return int(shards[0]) if shards is not None else int(effective_chunks[0])


def _write_dense_z_bands(
    root: Any,
    spill_dir: Path,
    n_variants: int,
    n_analyses: int,
    band_cols: int,
    codec: StoreCodec,
    dtype: str,
    pass2_start: float,
    n_workers: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray]:
    """z pass: encode, write, and harvest stored-value top hits.

    One ``(n_variants × band_cols)`` buffer is resident at a time; the columns
    filling the current band are loaded and encoded by ``n_workers`` forked
    workers and returned in Analysis order (``ordered_map``), so the overflow
    table, the top-hit candidate order and ``column_has_eaf`` are exactly the
    serial path's. At ``n_workers <= 1`` ``ordered_map`` runs the same loader
    in this process, which is the serial path. The top-hit harvest thresholds
    on the **quantised/decoded stored** z -- what a query reads back -- not the
    source z, and reads each hit's se straight from the spill rounded to the
    stored dtype. The overflow table is part of the z plane, not an addendum:
    it is written in the same pass that finished writing z. Returns the
    per-column hit parts and the per-Analysis ``column_has_eaf`` survey for
    the coordinator.
    """
    z_arr = root["z"]
    overflow = ZOverflowBuilder()
    hit_rows_parts: list[np.ndarray] = []
    hit_cols_parts: list[np.ndarray] = []
    hit_z_parts: list[np.ndarray] = []
    hit_se_parts: list[np.ndarray] = []
    column_has_eaf = np.zeros(n_analyses, dtype=bool)

    log.info(
        "Band-write z: %d analyses in bands of %d (n_workers=%d)",
        n_analyses,
        band_cols,
        n_workers,
    )
    loader = partial(_load_z_column, spill_dir, n_variants, n_analyses, codec)
    with log_phase(log, "Band-write z"):
        for c, loaded, flush in _ordered_bands(
            loader, n_analyses, n_workers, n_variants, band_cols, codec.z_dtype, "z"
        ):
            if loaded.overflow_index.size:
                overflow.add(loaded.overflow_index, loaded.overflow_value)
            column_has_eaf[c] = loaded.has_eaf
            if loaded.hit_rows.size:
                hit_rows_parts.append(loaded.hit_rows)
                hit_cols_parts.append(np.full(loaded.hit_rows.size, c, dtype=np.int64))
                hit_z_parts.append(loaded.hit_z)
                hit_se_parts.append(loaded.hit_se)
            if flush is not None:
                _flush_band(z_arr, flush, "Band-write z", n_analyses, pass2_start, band_cols)
    overflow.table().write(root)
    return hit_rows_parts, hit_cols_parts, hit_z_parts, hit_se_parts, column_has_eaf


def _write_dense_se_bands(
    root: Any,
    spill_dir: Path,
    n_variants: int,
    n_analyses: int,
    band_cols: int,
    pass2_start: float,
    n_workers: int,
) -> None:
    """se pass: the independent scratch write over one band at a time.

    ``z`` and ``se`` no longer share a dtype (ADR 0037), so this pass owns a
    fresh buffer of its own rather than reusing the z pass's. Its columns are
    loaded in parallel exactly as the z pass's are, and ``ordered_map`` keeps
    them in Analysis order so the written band is the serial one.

    The buffer takes the dtype of the plane ``create_dense_zarr`` created -- the
    `float32` scratch -- and not the build's ``dtype``: a `float16` band would
    round every SE before the codec decided the plan, and a residual plan's
    exact exception is read back out of this plane (issue #355, spec §6a).
    """
    se_arr = root["se"]
    loader = partial(_load_se_column, spill_dir, n_variants)
    with log_phase(log, "Band-write se"):
        for _c, _loaded, flush in _ordered_bands(
            loader, n_analyses, n_workers, n_variants, band_cols, str(se_arr.dtype), "se"
        ):
            if flush is not None:
                _flush_band(se_arr, flush, "Band-write se", n_analyses, pass2_start, band_cols)


def _concatenate_hits(
    parts: tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]],
    column_has_eaf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble the harvested top-hit candidates from their per-column parts."""
    hit_rows_parts, hit_cols_parts, hit_z_parts, hit_se_parts = parts
    if hit_rows_parts:
        return (
            np.concatenate(hit_rows_parts),
            np.concatenate(hit_cols_parts),
            np.concatenate(hit_z_parts),
            np.concatenate(hit_se_parts),
            column_has_eaf,
        )
    return (
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.float32),
        column_has_eaf,
    )


def write_dense_bands(
    staged: StagedRelease,
    spill_dir: Path,
    n_variants: int,
    n_analyses: int,
    effective_chunks: tuple[int, int],
    dtype: str,
    pass2_start: float,
    encoding: StoreEncoding,
    n_workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stream the retained per-column spills into the zarr in chunk-column bands.

    ``z`` and ``se`` are written in two separate passes so only one band is
    ever resident, and the top-hit harvest runs in the z-pass on the stored z.
    Each pass loads the band's columns across ``n_workers`` forked workers
    (issue #220); ``n_workers <= 1`` runs the loaders serially in-process.
    Spills are retained through the se pass and the EAF write, and are only
    unlinked once both consumed them. Returns the concatenated top-hit
    candidate arrays ``(rows, cols, z, se)`` plus the per-Analysis
    ``column_has_eaf`` survey for the index build and EAF decision.
    """
    root = staged.arrays(mode="a")
    band_cols = _shard_columns(root["z"], effective_chunks)
    codec = StoreCodec(encoding)
    rows_parts, cols_parts, z_parts, se_parts, column_has_eaf = _write_dense_z_bands(
        root, spill_dir, n_variants, n_analyses, band_cols, codec, dtype, pass2_start, n_workers
    )
    _write_dense_se_bands(
        root, spill_dir, n_variants, n_analyses, band_cols, pass2_start, n_workers
    )
    _write_dense_eaf(
        staged,
        spill_dir,
        n_variants,
        n_analyses,
        effective_chunks,
        band_cols,
        codec,
        pass2_start,
        n_workers,
        column_has_eaf,
        encoding,
    )
    for c in range(n_analyses):
        (spill_dir / f"{c}.npz").unlink(missing_ok=True)
    return _concatenate_hits((rows_parts, cols_parts, z_parts, se_parts), column_has_eaf)


def _write_manifest(
    staged: StagedRelease,
    store_id: str,
    release_id: str,
    n_variants: int,
    n_analyses: int,
    chain_file: str | Path | None,
    chunk_shape: tuple[int, int],
    dtype: str,
    encoding: StoreEncoding,
    association_coverage: AssociationCoverage,
    eaf_orientation: dict[str, Any] | None = None,
    variant_reference: str | None = None,
    column_map: dict[str, str] | None = None,
) -> None:
    manifest = observed_only_release_manifest(
        PrimaryStorageLayout.DENSE,
        provenance={
            "builder": (
                "opengwasdb.v0.1_dense_vcf_single_pass"
                if variant_reference is not None
                else "opengwasdb.v0.1_dense_vcf_two_pass"
            ),
            "chain_file": str(chain_file) if chain_file else "pyliftover_builtin_hg19_hg38",
            "n_variants": n_variants,
            "n_analyses": n_analyses,
            "dense": {
                **dense_provenance_block(
                    (n_variants, n_analyses), hint=chunk_shape, se_dtype=encoding.se.dtype
                ),
                "top_hit_thresholds": list(TOP_HIT_THRESHOLDS),
            },
            **({"eaf_orientation": eaf_orientation} if eaf_orientation is not None else {}),
            **(
                {"variant_reference": variant_reference}
                if variant_reference is not None
                else {}
            ),
            **({"column_map": column_map} if column_map is not None else {}),
        },
        store_id=store_id,
        release_id=release_id,
        encoding=encoding,
        association_coverage=association_coverage,
        reference_assembly="GRCh38",
    )
    staged.write_manifest(manifest)
