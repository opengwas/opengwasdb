"""Dense Reference Completion — enhancement pipeline.

Builds a Dense Reference-Completed Store Release from a Dense Observed-Only
`gwas`-coverage source, per ADR 0022 (dense axis = source ∪ reference panel)
and ADR 0023 (LD-block process-pool parallelism with checkpointed resume).

Pipeline shape:
  Phase 1 (sequential): enumerate the genome-wide LD block set, build the
    union variant axis, seed z/se from the source, compute the per-Analysis
    n_missing_off_panel scalar.
  Phase 2 (parallel, n_workers processes over LD blocks): each worker opens
    the source store and LD panel itself, imputes serially per Analysis
    within its block, and writes its own checkpoint file.
  Phase 3: shard the block results' fills by row band and write the
    completed planes a band at a time, across the same worker count (each band
    read once, completed and encoded whole, #321); then completion_quality
    rows, top-hit indexes, and manifest.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from opengwasdb.associations import Cells, DenseGridReader
from opengwasdb.completion.ancestry_filter import derive_impute_analysis_ids
from opengwasdb.completion.block import REGION_CAP_BP, read_as_planned, run_block
from opengwasdb.completion.checkpoint import (
    BlockCompletionResult,
    checkpoint_blocks_dir,
    checkpoint_dir_for,
    require_fresh_destination,
    sanitize_block_id,
    write_block_checkpoint,
)
from opengwasdb.completion.fill_shards import (
    CONFLICTED,
    FILL_RECORD_DTYPE,
    apply_fill_batch,
    by_row_band,
    clear_conflicted,
    fill_shard_path,
    iter_fill_records,
    log_dropped,
    shard_checkpoint_fills,
    sorted_union_map,
)
from opengwasdb.completion.ld_panel import (
    BlockPlacement,
    PanelPlacement,
    ceded_at_end,
    check_panel_has_chromosomes,
    list_all_blocks,
    list_chromosomes,
    panel_variant,
)
from opengwasdb.completion.manifest import (
    build_completion_provenance,
    completed_release_manifest,
)
from opengwasdb.completion.parallel import ordered_write, run_block_tasks
from opengwasdb.completion.reference_eaf import completed_eaf_scope, panel_reference_eaf
from opengwasdb.completion.schema import completion_quality_rollup, create_completion_quality_table
from opengwasdb.encoding import (
    EAF_BASELINE,
    EAF_REFERENCE,
    SE_MISSING,
    AssociationCodec,
    EafExceptionBuilder,
    EafExceptionTable,
    SeExceptionBuilder,
    SeExceptionTable,
    StoreCodec,
    StoreEncoding,
    ZOverflowBuilder,
    ZOverflowTable,
    write_dense_se_side_arrays,
    write_eaf_baseline,
    write_eaf_reference,
)
from opengwasdb.encoding.association_codec import CellValues, Placement, log_se_out_of_range
from opengwasdb.encoding.codec import SideTables, table_end_to_end
from opengwasdb.encoding.memory import pool_workers
from opengwasdb.encoding.timing import PhaseTimer, format_duration, log_phase, log_progress
from opengwasdb.index import initialise_schema, set_metadata
from opengwasdb.layouts.dense.build import write_analyses_tsv
from opengwasdb.layouts.dense.constants import (
    DEFAULT_CHUNK_SHAPE,
    DEFAULT_DTYPE,
    dense_layout_records,
)
from opengwasdb.layouts.dense.top_hits import build_top_hit_indexes
from opengwasdb.model.analyses import (
    Analysis,
    ancestry_impute_mask,
    read_analyses,
    read_analysis_records,
    reset_top_hit_counts,
)
from opengwasdb.model.enums import (
    AssociationCoverage,
    CompletionState,
    PrimaryStorageLayout,
)
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.store.arrays import ArrayRole, compressor, create_array
from opengwasdb.store.open import (
    OpenGWASDBStore,
    StagedRelease,
    check_writable_format_version,
    open_store,
)
from opengwasdb.top_hits import add_hit_counts
from opengwasdb.variants import (
    CanonicalVariant,
    VariantAxis,
    VariantRecord,
    chromosome_sort_key,
    parse_canonical_alid,
    write_variant_axis,
)

log = logging.getLogger(__name__)


_COMPRESSOR = compressor()
_LD_PANEL_ID = "eur-hg38-gpm"

#: What the Association Reader names the source in a refusal.
_SOURCE = "complete-dense, the source's Dense grid"


# ── Wall-time accounting ────────────────────────────────────────────────────

#: The five phases of a completion, in the order they run. A run's report
#: partitions its time over these; anything between them (the manifest, the
#: Analyses rows) is small and is not charged to any.
AXIS_PHASE = "Axis construction"
BLOCK_PHASE = "LD-block completion"
MERGE_PHASE = "Checkpoint merge"
BAND_PHASE = "Band write"
TOP_HIT_PHASE = "Top-Hit Index"

@contextmanager
def _completion_phase(timer: PhaseTimer, name: str) -> Iterator[None]:
    """Log `name`'s start and elapsed time and charge it to the run's report."""
    with log_phase(log, name), timer.phase(name):
        yield


def _log_phase_report(timer: PhaseTimer, started: float) -> None:
    log.info(
        "Reference Completion phase report (%s elapsed):\n%s",
        format_duration(time.monotonic() - started),
        timer.format_report(),
    )


@dataclass(frozen=True)
class CompletionResult:
    output_path: Path
    n_variants: int
    n_analyses: int
    n_imputed: int
    n_missing_off_panel: int
    n_missing_imputation_failed: int


# ── Phase 2: per-block worker ───────────────────────────────────────────────


@dataclass(frozen=True)
class _BlockTask:
    tsv_path: Path
    source_path: Path
    min_cor: float
    thresh: float
    checkpoint_path: Path
    #: How the plan's read of the block placed its rows, which the worker's must repeat.
    placement: BlockPlacement
    #: The variants at this block's end the next block owns (`shared_endpoints`).
    ceded: frozenset[str] = frozenset()


def _source_row_for(src_axis: VariantAxis, alid: str | None) -> int | None:
    """The source row an ALID maps to, or ``None`` for an absent/absent-looking one."""
    parsed = parse_canonical_alid(alid) if alid is not None else None
    rec = src_axis.by_alid(parsed) if parsed is not None else None
    return rec.variant_index if rec is not None else None


def _by_row(values: np.ndarray, n_analyses: int) -> np.ndarray:
    """A `variants` region's cells, read variant-major, as rows × Analyses."""
    return values.reshape(-1, n_analyses)


def _match_source_rows(
    source_path: str | Path, canonical_alids: list[str | None]
) -> tuple[list[int], list[int]]:
    """Map each canonical ALID to its source row, keeping absence as absence.

    ``None``, an unparsable ALID, and an ALID the source does not carry must all
    stay unmatched: the rows they own keep their NaN fill below, and a source
    position is never fabricated from row 0 or another sentinel.
    """
    src_axis = VariantAxis(source_path)
    try:
        src_rows = [_source_row_for(src_axis, alid) for alid in canonical_alids]
        matched_local = [i for i, r in enumerate(src_rows) if r is not None]
        matched_src = [row for row in src_rows if row is not None]
        return matched_local, matched_src
    finally:
        src_axis.close()


def _make_reader(task: _BlockTask):
    """dense's half of the ``run_block`` seam: read every Analysis's observed
    z/se at a block's positions as one matrix slice, opening the source store
    and LD panel itself so no payload beyond a lightweight block descriptor
    needs to be pickled into the worker process.

    The cells are read through the Association Reader unjudged (`inspect`): the
    band pass judges every source cell before any completed cell is stored.
    """

    def make_reader(block, canonical_alids: list[str | None]):
        src_store = open_store(task.source_path)
        source = DenseGridReader.open(
            src_store.arrays(mode="r"), src_store.manifest.encoding, source=_SOURCE
        )
        n_analyses = source.n_analyses

        matched_local, matched_src = _match_source_rows(task.source_path, canonical_alids)

        z_obs = np.full((len(canonical_alids), n_analyses), np.nan, dtype=np.float64)
        se_obs = np.full((len(canonical_alids), n_analyses), np.nan, dtype=np.float64)
        if matched_local:
            cells = source.inspect(Cells.variants(matched_src))
            z_obs[matched_local, :] = _by_row(cells.z, n_analyses)
            se_obs[matched_local, :] = _by_row(cells.se, n_analyses)

        def read(ai: int) -> tuple[np.ndarray, np.ndarray]:
            return z_obs[:, ai], se_obs[:, ai]

        return range(n_analyses), read

    return make_reader


def _run_block(task: _BlockTask) -> BlockCompletionResult | None:
    """Complete one LD block for every Analysis. Runs inside a worker process.

    Every Analysis is completed against every block, so a block cedes the same
    variants for each.
    """
    result = run_block(
        task.tsv_path, task.thresh, task.min_cor, REGION_CAP_BP,
        read_as_planned(task.placement, _make_reader(task)),
        ceded=lambda _analysis: task.ceded,
    )
    if result is None:
        return None
    write_block_checkpoint(task.checkpoint_path, result)
    # Fills stay on disk (the checkpoint); the parent reads them back from the
    # checkpoints in Phase 3. Returning them here would push potentially millions
    # of tuples per block through the pool result queue and accumulate them in the
    # parent (issue 044 follow-up), so return an empty-fills marker.
    return BlockCompletionResult(block_id=result.block_id, quality_rows=[], fills=[])


# ── Public entry points ─────────────────────────────────────────────────────


def complete_dense_store(
    source_path: str | Path,
    dest_path: str | Path,
    ld_dir: str | Path,
    *,
    ancestry: str = "EUR",
    min_cor: float = 0.7,
    thresh: float = 0.9,
    release_id: str | None = None,
    ld_panel_id: str = _LD_PANEL_ID,
    n_workers: int = 1,
    overwrite: bool = False,
    impute_analysis_ids: set[str] | None = None,
) -> CompletionResult:
    """Produce a Dense Reference-Completed Store Release from a `gwas`-coverage
    Dense Observed-Only source.

    source_path: existing Dense Observed-Only, `gwas`-coverage store.
    dest_path:   new store directory to create.
    ld_dir:      root of LD panel; blocks at ld_dir/{ancestry}/{chr}/{block}.*
    impute_analysis_ids: if given, only these analyses are imputed (the
        ancestry-match filter, ADR 0028); others are carried through
        observed-only. ``None`` auto-derives the filter from the source's
        ``assigned_ancestry`` column when present, imputing every analysis
        when it is not (no behaviour change for sources with no ancestry
        information).
    """
    dst = Path(dest_path)
    checkpoint_dir = checkpoint_dir_for(dst)
    require_fresh_destination(dst, checkpoint_dir, overwrite, "resume_dense_completion")
    source = open_store(source_path)
    check_writable_format_version(
        source.manifest.format_version, source=f"source release {Path(source_path)}"
    )
    check_panel_has_chromosomes(ld_dir, ancestry)

    if impute_analysis_ids is None:
        impute_analysis_ids = derive_impute_analysis_ids(
            read_analyses(Path(source_path) / "analyses.tsv").rows, ancestry
        )

    (checkpoint_dir / "blocks").mkdir(parents=True)
    build_params = {
        "source_path": str(Path(source_path).resolve()),
        "dest_path": str(dst.resolve()),
        "ld_dir": str(Path(ld_dir).resolve()),
        "ancestry": ancestry,
        "min_cor": min_cor,
        "thresh": thresh,
        "release_id": release_id,
        "ld_panel_id": ld_panel_id,
        "impute_analysis_ids": sorted(impute_analysis_ids)
        if impute_analysis_ids is not None
        else None,
    }
    (checkpoint_dir / "build_params.json").write_text(
        json.dumps(build_params, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    result = _run_completion(
        Path(source_path),
        dst,
        Path(ld_dir),
        ancestry=ancestry,
        min_cor=min_cor,
        thresh=thresh,
        release_id=release_id,
        ld_panel_id=ld_panel_id,
        n_workers=n_workers,
        checkpoint_dir=checkpoint_dir,
        impute_analysis_ids=impute_analysis_ids,
    )
    shutil.rmtree(checkpoint_dir)
    return result


def resume_dense_completion(
    checkpoint_dir: str | Path,
    *,
    n_workers: int = 1,
) -> CompletionResult:
    """Resume an interrupted complete_dense_store() run.

    Takes only the checkpoint directory path — all other build parameters are
    loaded from the build_params.json written on the first run, so a resumed
    run can never silently apply a different parameter set than the one its
    existing per-block checkpoints were computed under.
    """
    checkpoint_dir = Path(checkpoint_dir)
    params = json.loads((checkpoint_dir / "build_params.json").read_text(encoding="utf-8"))

    impute_ids = params.get("impute_analysis_ids")
    result = _run_completion(
        Path(params["source_path"]),
        Path(params["dest_path"]),
        Path(params["ld_dir"]),
        ancestry=params["ancestry"],
        min_cor=params["min_cor"],
        thresh=params["thresh"],
        release_id=params["release_id"],
        ld_panel_id=params["ld_panel_id"],
        n_workers=n_workers,
        checkpoint_dir=checkpoint_dir,
        impute_analysis_ids=set(impute_ids) if impute_ids is not None else None,
    )
    shutil.rmtree(checkpoint_dir)
    return result


# ── Shared pipeline core ────────────────────────────────────────────────────


@dataclass(frozen=True)
class _SourceAxis:
    """What Phase 1 reads from the Observed-Only source: its variant axis, its
    Analysis records, and the per-Analysis ancestry-match impute filter (ADR
    0028) derived from them. Read once and carried whole, so the filter can
    never come from a different read of the axis than the one it was built
    against."""

    src_variants: list[VariantRecord]
    src_analyses: list[Analysis]
    src_alid_to_idx: dict[str, int]
    n_analyses: int
    impute_mask: np.ndarray | None


@dataclass(frozen=True)
class _UnionAxis:
    """Phase 1's product: the merged variant axis (source ∪ panel, ADR 0022)
    and every map later phases read it through. One object, so a phase cannot
    pick up a mask or remap that was built against a different axis than the
    one it is writing."""

    merged_variants: list[CanonicalVariant]
    n_variants: int
    n_variants_new: int
    src_analyses: list[Analysis]
    n_analyses: int
    impute_mask: np.ndarray | None
    new_alid_to_idx: dict[str, int]
    on_panel: np.ndarray
    out_to_src: np.ndarray
    tsv_paths: list[Path]
    #: Per block's TSV, the variants at its end the next block owns.
    ceded: dict[Path, frozenset[str]]
    #: How the panel placed its rows: the completion record's `panel_indels`
    #: and `panel_snvs_unplaced` (`PanelPlacement.record`).
    panel_placement: dict[str, dict[str, object]]
    #: Per block's TSV, how the plan's read placed its rows (`BlockPlacement`).
    block_placement: dict[Path, BlockPlacement]


@dataclass(frozen=True)
class _CompletedArrays:
    """Phase 3's product: the encoding the completed arrays were written in,
    and the counts the metadata phases must state. ``n_missing_off_panel``
    stays per-Analysis so ``analyses.tsv`` can hold each row's own count."""

    encoding: StoreEncoding
    eaf_reference_present: bool
    n_missing_off_panel: np.ndarray
    n_missing_off_panel_total: int
    n_missing_imputation_failed: int
    total_imputed: int


def _read_source_axis(src: Path, impute_analysis_ids: set[str] | None) -> _SourceAxis:
    """Read the source's variant axis and Analysis records (ADR 0034) and
    derive the impute filter: ``None`` (no filter) imputes every Analysis,
    which is what a source with no ``assigned_ancestry`` column gets; a set
    keeps only the matching Analyses imputed and carries the rest through
    observed-only (ADR 0028)."""
    src_variant_axis = VariantAxis(src)
    src_variants = src_variant_axis.all()
    src_variant_axis.close()
    src_alid_to_idx = {v.alid: v.variant_index for v in src_variants}

    src_analyses = sorted(
        read_analysis_records(src / "analyses.tsv"), key=lambda a: int(a.analysis_index)
    )
    n_analyses = len(src_analyses)
    print(f"Source: {len(src_variants):,} variants, {n_analyses:,} analyses")

    impute_mask = ancestry_impute_mask(src_analyses, impute_analysis_ids)
    if impute_mask is not None:
        n_match = int(impute_mask.sum())
        print(f"Ancestry-match filter: imputing {n_match:,}/{n_analyses:,} analyses")
    return _SourceAxis(src_variants, src_analyses, src_alid_to_idx, n_analyses, impute_mask)


@dataclass(frozen=True)
class _Panel:
    """What Phase 1 reads of the LD panel: the block paths Phase 2 schedules,
    the canonical ALIDs the union axis must hold (ADR 0022), and, per block,
    the variants at its end the next block owns."""

    tsv_paths: list[Path]
    alids: set[str]
    ceded: dict[Path, frozenset[str]]
    #: How the panel placed its rows, for the completion record.
    placement: dict[str, dict[str, object]]
    #: Per block's TSV, how this read placed its rows (`BlockPlacement`).
    blocks: dict[Path, BlockPlacement]


def _enumerate_panel(ld_dir: Path, ancestry: str) -> _Panel:
    """Walk every LD block's TSV in chromosome order, collecting the block
    paths, the panel's canonical ALIDs and its shared endpoints."""
    print("Enumerating genome-wide LD blocks...")
    tsv_paths: list[Path] = []
    panel_alids: set[str] = set()
    ceded: dict[Path, frozenset[str]] = {}
    placement = PanelPlacement()
    for chrom in list_chromosomes(ld_dir, ancestry):
        blocks = list_all_blocks(ld_dir, ancestry, chrom)
        for block in blocks:
            tsv_paths.append(block.tsv_path)
            panel_alids.update(alid for alid in block.alids if alid is not None)
            placement.add(block)
        at_end = ceded_at_end(blocks)
        ceded.update(
            (block.tsv_path, at_end[block.block_id]) for block in blocks if block.block_id in at_end
        )
    print(
        f"LD panel: {len(tsv_paths):,} blocks, {len(panel_alids):,} panel variants, "
        f"{sum(map(len, ceded.values())):,} at an endpoint two blocks share"
    )
    placement.log(ld_dir, ancestry)
    return _Panel(tsv_paths, panel_alids, ceded, placement.record(), placement.blocks)


def _union_variant_table(
    src_variants: list[VariantRecord],
    src_alid_to_idx: dict[str, int],
    panel_alids: set[str],
) -> tuple[list[CanonicalVariant], list[CanonicalVariant], dict[str, int]]:
    """Append the panel variants the source does not already hold (deduplicated
    through canonical orientation) and sort the union into the store's
    canonical order. An ALID the panel cannot say what it is never becomes a
    variant the store claims to cover (`panel_variant`)."""
    present = set(src_alid_to_idx)
    new_canonical: list[CanonicalVariant] = []
    for alid in panel_alids:
        if alid in present:
            continue
        variant = panel_variant(alid)
        if variant is None:
            continue
        if variant.alid in present:
            continue
        present.add(variant.alid)
        new_canonical.append(variant)

    merged_variants: list[CanonicalVariant] = [
        CanonicalVariant(v.chromosome, v.position, v.effect_allele, v.other_allele)
        for v in src_variants
    ] + new_canonical
    merged_variants.sort(
        key=lambda v: (
            chromosome_sort_key(v.chromosome),
            v.position,
            v.effect_allele,
            v.other_allele,
        )
    )
    new_alid_to_idx: dict[str, int] = {v.alid: i for i, v in enumerate(merged_variants)}
    print(
        f"Union variant axis: {len(merged_variants):,} variants "
        f"({len(new_canonical):,} new panel variants)"
    )
    return merged_variants, new_canonical, new_alid_to_idx


def _write_union_axis_tables(
    staged: StagedRelease,
    *,
    source_variants: list[VariantRecord],
    merged_variants: list[CanonicalVariant],
    new_alid_to_idx: dict[str, int],
    panel_alids: set[str],
    n_variants: int,
    n_analyses: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Persist the union axis: ``variants.tsv.gz`` with the source's rsids
    carried across (issue #109), the ``on_panel`` mask, the inverse
    output-row → source-row remap (z/se are seeded band-by-band during the
    write, issue 044), and an ``index.sqlite`` holding the completion-quality
    table."""
    on_panel = np.zeros(n_variants, dtype=bool)
    for alid in panel_alids:
        idx = new_alid_to_idx.get(alid)
        if idx is not None:
            on_panel[idx] = True

    rsid_by_alid = {v.alid: v.rsid for v in source_variants if v.rsid}
    print("Writing variants.tsv.gz...")
    write_variant_axis(staged.path, merged_variants, rsid_by_alid)

    out_to_src = np.full(n_variants, -1, dtype=np.int64)
    for v in source_variants:
        out_to_src[new_alid_to_idx[v.alid]] = v.variant_index

    print("Writing index.sqlite...")
    with staged.index_connection() as dst_db:
        initialise_schema(dst_db)
        create_completion_quality_table(dst_db)
        set_metadata(dst_db, "schema_version", 2)
        set_metadata(dst_db, "n_variants", n_variants)
        set_metadata(dst_db, "n_analyses", n_analyses)
        dst_db.commit()
        # analyses.tsv is written after the band write, once
        # n_missing_off_panel is known (issue 044; issue #22).
    return on_panel, out_to_src


def _build_union_axis(
    src: Path,
    staged: StagedRelease,
    ld_dir: Path,
    ancestry: str,
    impute_analysis_ids: set[str] | None,
) -> _UnionAxis:
    """Phase 1 — axis union (ADR 0022): read the source, append the panel
    variants it lacks, and seed the staged store's variant tables from the
    merged axis."""
    source_axis = _read_source_axis(src, impute_analysis_ids)
    panel = _enumerate_panel(ld_dir, ancestry)
    panel_alids = panel.alids
    merged_variants, new_canonical, new_alid_to_idx = _union_variant_table(
        source_axis.src_variants, source_axis.src_alid_to_idx, panel_alids
    )
    on_panel, out_to_src = _write_union_axis_tables(
        staged,
        source_variants=source_axis.src_variants,
        merged_variants=merged_variants,
        new_alid_to_idx=new_alid_to_idx,
        panel_alids=panel_alids,
        n_variants=len(merged_variants),
        n_analyses=source_axis.n_analyses,
    )
    return _UnionAxis(
        merged_variants=merged_variants,
        n_variants=len(merged_variants),
        n_variants_new=len(new_canonical),
        src_analyses=source_axis.src_analyses,
        n_analyses=source_axis.n_analyses,
        impute_mask=source_axis.impute_mask,
        new_alid_to_idx=new_alid_to_idx,
        on_panel=on_panel,
        out_to_src=out_to_src,
        tsv_paths=panel.tsv_paths,
        ceded=panel.ceded,
        panel_placement=panel.placement,
        block_placement=panel.blocks,
    )


def _complete_pending_blocks(
    axis: _UnionAxis,
    src: Path,
    checkpoint_dir: Path,
    *,
    min_cor: float,
    thresh: float,
    n_workers: int,
) -> Path:
    """Phase 2 — schedule and run the LD-block completions over the process
    pool (ADR 0023). A block whose checkpoint already exists is skipped; each
    remaining block writes its own checkpoint, so nothing per block is held in
    the parent (issue 044)."""
    print(
        f"Running reference completion across {len(axis.tsv_paths):,} LD blocks "
        f"(n_workers={n_workers})..."
    )
    blocks_dir = checkpoint_blocks_dir(checkpoint_dir)

    pending: list[_BlockTask] = []
    n_existing = 0
    for tsv_path in axis.tsv_paths:
        block_id = f"{tsv_path.parent.name}/{tsv_path.stem}"
        ckpt_path = blocks_dir / f"{sanitize_block_id(block_id)}.npz"
        if ckpt_path.exists():
            n_existing += 1  # fills read from the checkpoint in Phase 3
        else:
            pending.append(
                _BlockTask(
                    tsv_path=tsv_path,
                    source_path=src,
                    min_cor=min_cor,
                    thresh=thresh,
                    checkpoint_path=ckpt_path,
                    placement=axis.block_placement[tsv_path],
                    ceded=axis.ceded.get(tsv_path, frozenset()),
                )
            )

    if pending:
        print(f"  {n_existing:,} blocks already checkpointed, {len(pending):,} remaining")
    run_block_tasks(pending, n_workers, _run_block)
    return blocks_dir


def _merge_checkpoint_fills(
    staged: StagedRelease,
    axis: _UnionAxis,
    blocks_dir: Path,
    ld_dir: Path,
    ancestry: str,
    source_encoding: StoreEncoding,
    n_workers: int = 1,
) -> tuple[StoreEncoding, bool, tuple[int, int], Path]:
    """Resolve every block checkpoint's fills to union rows and shard them by
    output row-band on disk, creating the empty completed planes alongside.
    The final zarr writer then reads only the shard for the band it is
    writing, so Phase 3 never needs a whole-genome fill array in RAM (issue
    044). ``impute_mask`` is applied here, where a worker's candidate fills
    become the release's (ADR 0028): dropping fills only, later, at the write
    left ``completion_quality`` -- and through it ``analyses.tsv`` -- counting
    cells the release does not hold."""
    print("Merging block results from checkpoints...")
    union = sorted_union_map(axis.new_alid_to_idx)

    # Reference EAF for imputed cells (ADR 0037 §4): an imputed cell's
    # frequency *is* the panel's -- one `float32` per variant, never a fallback
    # for observed cells. Asked for whatever the source declares, `absent`
    # included: a release whose Analyses reported no frequency still gains
    # imputed cells with the panel's frequency (issue #113).
    src_has_eaf = not source_encoding.eaf.is_absent
    eaf_reference = panel_reference_eaf(ld_dir, ancestry, axis.merged_variants)
    encoding = source_encoding.with_eaf_reference(eaf_reference is not None)
    effective_chunks = _create_completed_zarr(
        staged,
        axis.n_variants,
        axis.n_analyses,
        axis.on_panel,
        DEFAULT_CHUNK_SHAPE,
        DEFAULT_DTYPE,
        encoding,
        src_has_eaf=src_has_eaf,
        eaf_reference=eaf_reference,
    )
    band_rows = completion_band_rows(_dense_shard_rows(staged, effective_chunks))
    fill_shard_dir, quality_count = shard_checkpoint_fills(
        blocks_dir,
        staged,
        union,
        axis.n_variants,
        by_row_band(band_rows),
        impute_mask=axis.impute_mask,
        n_workers=n_workers,
    )
    print(f"Wrote {quality_count:,} completion quality rows")
    return encoding, eaf_reference is not None, effective_chunks, fill_shard_dir


def _stream_completed_arrays(
    staged: StagedRelease,
    src_root: Any,
    axis: _UnionAxis,
    blocks_dir: Path,
    ld_dir: Path,
    ancestry: str,
    source_encoding: StoreEncoding,
    timer: PhaseTimer,
    n_workers: int = 1,
) -> _CompletedArrays:
    """Phase 3 — stream the completed z/se/imputed matrix out of the sharded
    checkpoints into row-band writes (issue 044), across ``n_workers`` (#321),
    returning the counts the metadata phases must state. The merge and the
    band write are charged to ``timer`` separately."""
    with _completion_phase(timer, MERGE_PHASE):
        encoding, eaf_reference_present, effective_chunks, fill_shard_dir = (
            _merge_checkpoint_fills(
                staged, axis, blocks_dir, ld_dir, ancestry, source_encoding, n_workers
            )
        )
    print("Writing data.zarr (band-streamed)...")
    with _completion_phase(timer, BAND_PHASE):
        tallies = _write_completed_bands(
            staged,
            src_root,
            axis.out_to_src,
            axis.on_panel,
            fill_shard_dir,
            effective_chunks,
            axis.n_variants,
            axis.n_analyses,
            source_encoding,
            encoding,
            impute_mask=axis.impute_mask,
            name_row=lambda row: axis.merged_variants[row].alid,
            n_workers=n_workers,
        )
    shutil.rmtree(fill_shard_dir, ignore_errors=True)
    return _completed_arrays(axis, tallies, encoding, eaf_reference_present)


def _completed_arrays(
    axis: _UnionAxis, tallies: _BandTallies, encoding: StoreEncoding, eaf_reference_present: bool
) -> _CompletedArrays:
    """The band write's counts, said once and returned for the metadata phases."""
    log_se_out_of_range(
        tallies.se_out_of_range,
        [analysis.analysis_id for analysis in axis.src_analyses],
        where="complete-dense",
    )
    n_missing_off_panel = tallies.n_missing_off_panel
    n_missing_imputation_failed = tallies.n_missing_imputation_failed
    total_imputed = tallies.total_imputed
    n_missing_off_panel_total = int(n_missing_off_panel.sum())
    print(
        f"Completion done: {total_imputed:,} imputed, "
        f"{n_missing_imputation_failed:,} imputation-failed, "
        f"{n_missing_off_panel_total:,} off-panel missing"
    )
    return _CompletedArrays(
        encoding=encoding,
        eaf_reference_present=eaf_reference_present,
        n_missing_off_panel=n_missing_off_panel,
        n_missing_off_panel_total=n_missing_off_panel_total,
        n_missing_imputation_failed=n_missing_imputation_failed,
        total_imputed=total_imputed,
    )


def _checked_dense_source(manifest: StoreManifest, src: Path) -> str:
    """The source preconditions, checked with the other source preconditions
    rather than at manifest-write time: a completion that cannot honour its
    source's format should fail before it spends an hour imputing (ADR 0038
    §4), and Dense reference completion is only defined over a Dense,
    Observed-Only, GWAS Coverage source: completion fills the cells a source
    left missing, and only `gwas` makes every such cell mean \"the source did
    not report this\" rather than \"this was filtered out\" (issue #298)."""
    source_format_version = check_writable_format_version(
        manifest.format_version, source=f"source release {src}"
    )
    if manifest.primary_layout is not PrimaryStorageLayout.DENSE:
        raise ValueError(f"source store is not Dense (primary_layout={manifest.primary_layout})")
    if manifest.completion_state is not CompletionState.OBSERVED_ONLY:
        raise ValueError(
            f"source store is not Observed-Only (completion_state={manifest.completion_state})"
        )
    if manifest.association_coverage is not AssociationCoverage.GWAS:
        raise ValueError(
            "Dense reference completion only supports GWAS Coverage sources "
            f"(association_coverage={manifest.association_coverage.value})"
        )
    return source_format_version


def _write_completed_manifest(
    staged: StagedRelease,
    manifest: StoreManifest,
    source_format_version: str,
    axis: _UnionAxis,
    arrays: _CompletedArrays,
    *,
    release_id: str | None,
    ld_panel_id: str,
    ancestry: str,
    min_cor: float,
    thresh: float,
) -> None:
    """Phase 4a — the completed release's manifest and provenance. Written
    before analyses.tsv/overview.html below, because overview.html reads
    manifest.json fresh from output_path for its header (ADR 0032), so it must
    already reflect the completed release, not the source's. The completed
    release keeps its source's `format_version` and encoding -- completion
    writes into the source's arrays and therefore its encoding (ADR 0038 §4),
    the one addition being `eaf_reference` (ADR 0037 §4)."""
    completed_manifest = completed_release_manifest(
        manifest,
        encoding=arrays.encoding,
        release_id=release_id,
        source_format_version=source_format_version,
        completion_provenance=build_completion_provenance(
            ld_panel_id=ld_panel_id,
            ancestry=ancestry,
            min_cor=min_cor,
            thresh=thresh,
            n_variants_total=axis.n_variants,
            n_variants_new=axis.n_variants_new,
            n_imputed=arrays.total_imputed,
            n_missing_off_panel=arrays.n_missing_off_panel_total,
            n_missing_imputation_failed=arrays.n_missing_imputation_failed,
            **axis.panel_placement,
        ),
    )
    staged.write_manifest(_record_effective_dense_chunk_shape(completed_manifest, staged))


def _record_effective_dense_chunk_shape(
    manifest: StoreManifest, staged: StagedRelease
) -> StoreManifest:
    """State the chunk shape the completed Dense planes actually have.

    Completion writes the completed grid at `DEFAULT_CHUNK_SHAPE`, clipped to
    the array dimensions -- not at the source's build-wide hint.  A completed
    release therefore inherits a `provenance.dense.chunk_shape` describing the
    source's *intended* layout, which is not the completed arrays' layout
    whenever the two differ.  A manifest that describes one shape over arrays
    of another is the silent failure class #245's recorded-layout rule exists
    to catch, so completion records what it wrote (the root attrs already carry
    it; this makes the manifest agree).
    """
    dense = dict(manifest.provenance.get("dense", {}))
    if not dense:
        return manifest
    attrs = dict(staged.arrays(mode="r").attrs)
    if attrs.get("chunk_shape") is None:
        return manifest
    # Record the whole layout, not only the inner chunk: a completed release
    # that inherited the source's `shard_shape` or compressor but wrote arrays
    # with another is the same silent-failure class (issue #245, #247).  The
    # `index.sqlite` `dense` blob is the third recording 0.2.0 requires (#248),
    # and completion is the writer that produces it for a completed release.
    for key in ("chunk_shape", "shard_shape", "compressor", "zarr_format"):
        if key in attrs:
            dense[key] = attrs[key]
    with staged.index_connection() as connection:
        set_metadata(connection, "dense", dict(dense))
        connection.commit()
    return replace(manifest, provenance={**manifest.provenance, "dense": dense})


def _completed_analysis_rows(
    staged: StagedRelease,
    axis: _UnionAxis,
    arrays: _CompletedArrays,
    *,
    ancestry: str,
) -> list[Analysis]:
    """Phase 4b — the completed ``analyses.tsv`` rows: the source's rows with
    the completion rollup columns refreshed and the pre-completion Top-Hit
    Counts zeroed so ``add_hit_counts`` sets fresh post-completion counts
    rather than adding onto stale ones."""
    print("Writing analyses.tsv...")
    with staged.index_connection() as dst_db:
        quality_rollup = completion_quality_rollup(dst_db, axis.n_analyses)
    return reset_top_hit_counts(
        [
            replace(
                a,
                completed_against=(
                    ancestry if axis.impute_mask is None or axis.impute_mask[i] else ""
                ),
                # `eaf_scope` is derived from what the release actually holds, not
                # copied forward -- the declaration disagreeing with the arrays is
                # the defect that got through review on #106 (ADR 0037 §4).
                eaf_scope=completed_eaf_scope(a, quality_rollup[i], arrays.eaf_reference_present),
                completion_median_pearson_r=quality_rollup[i].median_pearson_r,
                completion_n_imputed_total=quality_rollup[i].n_imputed_total,
                completion_n_missing_total=str(int(arrays.n_missing_off_panel[i])),
            )
            for i, a in enumerate(axis.src_analyses)
        ]
    )


def _finalise_release(
    staged: StagedRelease,
    dst: Path,
    axis: _UnionAxis,
    arrays: _CompletedArrays,
    dst_analyses: list[Analysis],
) -> CompletionResult:
    """Phase 5 — top-hit indexes and the final summary. The indexes read the
    completed arrays, and ``analyses.tsv``'s Top-Hit Counts read the indexes
    (ADR 0032), so both are written once, here, after Phase 3 and 4."""
    print("Building top-hit indexes...")
    build_top_hit_indexes(staged.path, encoding=arrays.encoding)
    write_analyses_tsv(staged.path, add_hit_counts(staged.path, dst_analyses))
    result = CompletionResult(
        output_path=dst,
        n_variants=axis.n_variants,
        n_analyses=axis.n_analyses,
        n_imputed=arrays.total_imputed,
        n_missing_off_panel=arrays.n_missing_off_panel_total,
        n_missing_imputation_failed=arrays.n_missing_imputation_failed,
    )
    print(
        f"Reference completion complete: {result.n_variants:,} variants, "
        f"{result.n_analyses:,} analyses ({result.n_imputed:,} imputed, "
        f"{result.n_missing_off_panel:,} off-panel missing, "
        f"{result.n_missing_imputation_failed:,} imputation-failed)"
    )
    return result


def _run_completion(
    source_path: Path,
    dest_path: Path,
    ld_dir: Path,
    *,
    ancestry: str,
    min_cor: float,
    thresh: float,
    release_id: str | None,
    ld_panel_id: str,
    n_workers: int,
    checkpoint_dir: Path,
    impute_analysis_ids: set[str] | None = None,
) -> CompletionResult:
    """The Dense Reference Completion pipeline (ADR 0022, ADR 0023): a thin
    orchestrator over the five phases above, all inside one staged release so
    a failure leaves no half-written store behind (resume reads the per-block
    checkpoints Phase 2 left)."""
    src = Path(source_path)
    dst = Path(dest_path)
    timer = PhaseTimer()
    started = time.monotonic()
    with OpenGWASDBStore.staging(dst, overwrite=True) as staged:
        source = open_store(src)
        manifest = source.manifest
        source_format_version = _checked_dense_source(manifest, src)
        print(f"Source store: {manifest.store_id} / {manifest.release_id}")

        with _completion_phase(timer, AXIS_PHASE):
            axis = _build_union_axis(src, staged, ld_dir, ancestry, impute_analysis_ids)
        src_root = source.arrays(mode="r")
        with _completion_phase(timer, BLOCK_PHASE):
            blocks_dir = _complete_pending_blocks(
                axis, src, checkpoint_dir, min_cor=min_cor, thresh=thresh, n_workers=n_workers
            )
        arrays = _stream_completed_arrays(
            staged, src_root, axis, blocks_dir, ld_dir, ancestry, manifest.encoding, timer,
            n_workers,
        )
        _write_completed_manifest(
            staged,
            manifest,
            source_format_version,
            axis,
            arrays,
            release_id=release_id,
            ld_panel_id=ld_panel_id,
            ancestry=ancestry,
            min_cor=min_cor,
            thresh=thresh,
        )
        dst_analyses = _completed_analysis_rows(staged, axis, arrays, ancestry=ancestry)
        with _completion_phase(timer, TOP_HIT_PHASE):
            result = _finalise_release(staged, dst, axis, arrays, dst_analyses)
    _log_phase_report(timer, started)
    return result


# Row-band height for streaming the completed matrix — the seed/fill/write pass
# never holds the full (n_variants × n_analyses) matrices in RAM (issue 044).
_BAND_ROWS = 250_000


def completion_band_rows(shard_rows: int) -> int:
    """Rows per completion band: whole Dense shards, about `_BAND_ROWS` of them.

    The completion band writer sets ``z_arr[r0:r1]`` and the same for se,
    imputed and eaf, so a band that ends inside a shard turns every one of those
    writes into a read-modify-write.  The band is at least one shard, then as
    many whole shards as fit the ~`_BAND_ROWS` memory target (issue #247).
    """
    rows = max(int(shard_rows), 1)
    return max(rows, (_BAND_ROWS // rows) * rows)


# The observed-view check derives the same band from a completed Store's `z`
# (issue #289), so the rule is public; tests have always imported this name.
_completion_band_rows = completion_band_rows


def _dense_shard_rows(staged: StagedRelease, effective_chunks: tuple[int, int]) -> int:
    """The variant-axis height of the completed planes' shard, read from `z`."""
    shards = getattr(staged.arrays(mode="r")["z"], "shards", None)
    return int(shards[0]) if shards is not None else int(effective_chunks[0])


def _create_completed_planes(
    root: Any,
    codec: StoreCodec,
    n_variants: int,
    n_analyses: int,
    chunk_shape: tuple[int, int],
    src_has_eaf: bool,
) -> tuple[int, int]:
    """The completed grid's planes; returns the effective clipped chunks.

    z and se are missing-filled, imputed is 0-filled, and eaf appears only when
    the observed store carried one (ADR 0036) -- completion adds panel rows, it
    does not invent frequencies the source never reported.  Each plane gets its
    own missing marker (spec §15).

    SE is created in the destination's own dtype and coded as each band is
    written (#321), not staged as a `float32` plane and rewritten: the plane,
    and its fresh seam compressor, are the ones the rewrite made (issue #326).
    """
    grid = (n_variants, n_analyses)

    def plane(name: str, role: ArrayRole, plane_dtype: Any, fill: Any, hint: Any) -> Any:
        return create_array(
            root,
            name,
            role,
            shape=grid,
            dtype=plane_dtype,
            fill_value=fill,
            compressor=_COMPRESSOR,
            hint=hint,
        )

    z_array = plane(
        "z", ArrayRole.DENSE_STATISTIC_PLANE, codec.z_dtype, codec.z_fill_value, chunk_shape
    )
    # The role's layout policy clipped the chunk shape to the array dimensions
    # (ADR 0021); read the effective chunks back for the band writers.
    effective_chunks = (int(z_array.chunks[0]), int(z_array.chunks[1]))
    residual = codec.encoding.se.is_residual
    create_array(
        root,
        "se",
        ArrayRole.DENSE_STATISTIC_PLANE,
        shape=grid,
        dtype=codec.encoding.se.dtype,
        fill_value=SE_MISSING if residual else float("nan"),
        compressor=compressor(),
        hint=effective_chunks,
    )
    plane("imputed", ArrayRole.DENSE_IMPUTED_MASK, "uint8", 0, effective_chunks)
    if src_has_eaf:
        plane(
            "eaf",
            ArrayRole.DENSE_STATISTIC_PLANE,
            codec.eaf_dtype,
            codec.eaf_fill_value,
            effective_chunks,
        )
    return effective_chunks


def _create_completed_zarr(
    staged: StagedRelease,
    n_variants: int,
    n_analyses: int,
    on_panel: np.ndarray,
    chunk_shape: tuple[int, int],
    dtype: str,
    encoding: StoreEncoding,
    src_has_eaf: bool = False,
    eaf_reference: np.ndarray | None = None,
) -> tuple[int, int]:
    """Create empty z/se (missing-filled), imputed (0), and the 1-D on_panel
    datasets, plus `eaf` when the observed store carried one (ADR 0036). The
    matrices are filled by ``_write_completed_bands``; on_panel is small enough
    to write in one shot.

    The planes are created in the **source's** encoding, which completion
    preserves rather than re-stamping (ADR 0038 §4), and each is filled with
    its own missing marker (spec §15)."""
    root = staged.arrays(mode="w")
    effective_chunks = _create_completed_planes(
        root, StoreCodec(encoding), n_variants, n_analyses, chunk_shape, src_has_eaf
    )
    create_array(
        root,
        "on_panel",
        ArrayRole.DENSE_ON_PANEL,
        data=on_panel.astype(np.uint8),
        dtype="uint8",
        compressor=_COMPRESSOR,
        hint=effective_chunks,
    )
    if eaf_reference is not None:
        write_eaf_reference(root, eaf_reference, compressor=_COMPRESSOR)
    root.attrs["layout"] = "dense"
    root.attrs["completion_state"] = "reference_completed"
    root.attrs.update(dense_layout_records((n_variants, n_analyses), hint=chunk_shape))
    return effective_chunks


# ── the completed-band write ──────────────────────────────────────────────
#
# Each row band is written whole by one call, `_complete_band`: it reads the
# band's source rows once (z, SE, and the EAF the SE was decoded against),
# lands the band's fills, counts what is still missing, and encodes and writes
# every plane in the destination's encoding (`AssociationCodec`: z, then EAF,
# then SE against the EAF a reader decodes). A band is whole Dense shards, so
# the bands run on the completion write pool and each worker writes only the
# shards its band owns; the parent adds the bands' counts and side-table
# entries in band order (#321). A band holds its own cells and codes whole --
# its fills land anywhere in it -- and every read, count and encode inside it
# steps through `_STEP_CELLS` at a time.

#: Cells one step of a band's read, count and encode holds at a time, so their
#: decoded and `float64` intermediates are bounded whatever the band's width.
_STEP_CELLS = 1 << 24

#: A band worker's fixed footprint and its footprint per cell of its band, in
#: bytes: what the write pool divides the host's budget by (`_band_bytes`). A
#: band holds its z, SE and EAF (`float32`), its imputed mask and its codes,
#: 18 bytes a cell; the fixed part is a step's intermediates and the fork's floor.
_BAND_FIXED_BYTES = 4 * 1024**3
_BAND_BYTES_PER_CELL = 24


def _band_bytes(n_analyses: int, band_rows: int) -> int:
    """Bytes one band's worker holds."""
    return _BAND_FIXED_BYTES + _BAND_BYTES_PER_CELL * n_analyses * band_rows


#: What the parent holds of one band's outcome until it takes it: two counts a
#: band per Analysis, and the band's table entries, 12 bytes each, pickled and
#: unpickled. OGS-00009's completed release holds 6e-5 of its cells as SE
#: exceptions (about 30,000 a band) and far fewer z and EAF ones; a fortieth of
#: a byte a cell is room for 10^-3 of the cells (#321).
_BAND_RESULT_BYTES_PER_ANALYSIS = 16
_BAND_RESULT_CELLS_PER_BYTE = 40


def _band_result_bytes(n_analyses: int, band_rows: int) -> int:
    """Bytes the parent holds of one band's outcome."""
    return (
        _BAND_RESULT_BYTES_PER_ANALYSIS * n_analyses
        + n_analyses * band_rows // _BAND_RESULT_CELLS_PER_BYTE
    )


def _step_rows(n_analyses: int) -> int:
    """Rows of a band one step reads, counts or encodes."""
    return max(1, _STEP_CELLS // max(n_analyses, 1))


def _band_source_rows(
    out_to_src: np.ndarray, r0: int, r1: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rows of band ``[r0:r1)`` that carry a source cell, and the source rows
    they map to."""
    valid = np.where(out_to_src[r0:r1] >= 0)[0]
    return valid, out_to_src[r0:r1][valid]


@dataclass(frozen=True)
class _BandPlan:
    """What every band of the write reads: opened by the parent before the
    write pool forks, and inherited by its workers rather than pickled
    (`ordered_write`)."""

    root: Any
    source: DenseGridReader
    out_to_src: np.ndarray
    on_panel: np.ndarray
    fill_shard_dir: Path
    codec: AssociationCodec
    impute_mask: np.ndarray | None
    band_rows: int
    n_variants: int
    n_analyses: int
    #: Whether the source's frequencies are carried into the `eaf` plane.
    carries_eaf: bool

    def rows(self, band: int) -> tuple[int, int]:
        r0 = band * self.band_rows
        return r0, min(r0 + self.band_rows, self.n_variants)

    def steps(self, n_rows: int) -> Iterator[tuple[int, int]]:
        """``[0, n_rows)`` a step at a time."""
        step = _step_rows(self.n_analyses)
        for s0 in range(0, n_rows, step):
            yield s0, min(s0 + step, n_rows)


@dataclass(frozen=True)
class _BandOutcome:
    """What one band's write counted, and the side-table entries it owes."""

    band: int
    n_missing_off_panel: np.ndarray
    se_out_of_range: np.ndarray
    n_imputed: int
    n_missing_imputation_failed: int
    tables: SideTables
    #: The fill records at the cells two fills disagree on, none of them kept
    #: (D5), for the parent to log (D8), and how many cells they are at.
    dropped: np.ndarray
    n_conflicted_cells: int


def _complete_band(plan: _BandPlan, band: int) -> _BandOutcome:
    """Read, fill, count, encode and write one row band; runs in a pool worker.

    The source's cells are judged as they are read, before any fill, and every
    completed cell is judged whole (`storable`) before any plane holds it.
    """
    r0, r1 = plan.rows(band)
    cells = _seed_band(plan, r0, r1)
    n_missing_off_panel = np.zeros(plan.n_analyses, dtype=np.int64)
    _count_band_off_panel_missing(n_missing_off_panel, cells.z, plan.on_panel, r0, r1)
    assert cells.imputed is not None
    n_imputed, dropped, n_conflicted = _apply_fill_shard_records(
        cells.z, cells.se, cells.imputed, fill_shard_path(plan.fill_shard_dir, band), r0,
        impute_mask=plan.impute_mask,
    )
    n_failed = _count_band_imputation_failed(cells.z, plan.on_panel, r0, r1)
    codes, tables, se_out_of_range = _encode_band(plan, cells, r0)
    del cells
    _write_band(plan, codes, r0, r1)
    return _BandOutcome(
        band, n_missing_off_panel, se_out_of_range, n_imputed, n_failed, tables, dropped,
        n_conflicted,
    )


def _seed_band(plan: _BandPlan, r0: int, r1: int) -> CellValues:
    """The band's source cells, missing on every row no source cell maps to, and
    an imputed mask with nothing imputed yet."""
    shape = (r1 - r0, plan.n_analyses)
    z = np.full(shape, np.nan, dtype=np.float32)
    se = np.full(shape, np.nan, dtype=np.float32)
    eaf = np.full(shape, np.nan, dtype=np.float32) if plan.carries_eaf else None
    valid, rows = _band_source_rows(plan.out_to_src, r0, r1)
    for s0, s1 in plan.steps(len(valid)):
        _seed_paired_bands(z, se, eaf, plan.source, valid[s0:s1], rows[s0:s1])
    return CellValues(z, se, eaf, imputed=np.zeros(shape, dtype=np.uint8))


def _seed_paired_bands(
    zb: np.ndarray,
    sb: np.ndarray,
    eb: np.ndarray | None,
    source: DenseGridReader,
    valid: np.ndarray,
    rows: np.ndarray,
) -> None:
    """Seed band rows ``valid`` with source rows ``rows``: z, SE, and -- where
    the release carries them -- the frequencies, refusing a source cell the
    presence rule calls invalid (`InvalidAssociations`, ADR 0070).

    A fill lands where z is missing, so a half-missing source cell (spec §15)
    would come out with an imputed z beside an observed SE, or with its SE
    replaced by an imputed one. A missing marker other than the canonical NaN
    would be read as missing and, where nothing fills the cell, written into
    the completed planes as itself.

    One read: the frequencies are the ones the reader decoded the SE against,
    so the source's `eaf` plane is read once, not again for the `eaf` plane.
    """
    if not len(valid):
        return
    cells = source.inspect(Cells.variants(rows)).associations(_SOURCE)
    zb[valid, :] = _by_row(cells.z, source.n_analyses)
    sb[valid, :] = _by_row(cells.se, source.n_analyses)
    if eb is not None:
        eb[valid, :] = _by_row(cells.eaf, source.n_analyses)


def _count_band_off_panel_missing(
    n_missing_off_panel: np.ndarray,
    band: np.ndarray,
    on_panel: np.ndarray,
    r0: int,
    r1: int,
) -> None:
    """Add this band's still-missing off-panel cells to the per-Analysis count.

    Off-panel rows are never fill targets, so counting their NaN cells before
    the fills are applied and after would give the same answer; counting them
    here keeps the fills and the two missingness accounts in the same pass.
    """
    step = _step_rows(band.shape[1])
    for s0 in range(0, r1 - r0, step):
        off_local = s0 + np.flatnonzero(on_panel[r0 + s0 : min(r0 + s0 + step, r1)] == 0)
        if len(off_local):
            n_missing_off_panel += np.isnan(band[off_local, :]).sum(axis=0).astype(np.int64)


def _count_band_imputation_failed(
    band: np.ndarray, on_panel: np.ndarray, r0: int, r1: int
) -> int:
    """On-panel cells still missing once this band's fills were applied."""
    step = _step_rows(band.shape[1])
    failed = 0
    for s0 in range(0, r1 - r0, step):
        on_local = s0 + np.flatnonzero(on_panel[r0 + s0 : min(r0 + s0 + step, r1)] == 1)
        if len(on_local):
            failed += int(np.isnan(band[on_local, :]).sum())
    return failed


_NO_FILLS = np.empty(0, dtype=FILL_RECORD_DTYPE)


def _apply_fill_shard_records(
    zb: np.ndarray,
    sb: np.ndarray,
    imputed_band: np.ndarray,
    shard_path: Path,
    r0: int,
    *,
    impute_mask: np.ndarray | None = None,
) -> tuple[int, np.ndarray, int]:
    """Stream one band's fill-shard records into its NaN-seeded z and SE bands.

    A record lands only in a cell the source left missing -- it never
    overwrites an observed value -- and fills z and SE together and marks the
    cell imputed. Returns the count of cells filled, the records at cells two
    fills disagree on, and how many cells those are.

    A cell two records fill is resolved by Association Admission's duplicate
    rule (D5): fills that agree on z and SE are one, and fills that disagree
    are all dropped, leaving the cell missing; the caller logs each (D8). The
    block that starts at a shared endpoint alone fills it (`shared_endpoints`),
    so two fills reach a cell only where two blocks overlap elsewhere.

    It enforces the ancestry-match filter: a nonmatching Analysis reaching the
    write means the filter applied at checkpoint resolution and the one applied
    here disagree -- the disagreement that let ``completion_quality`` count
    cells the release did not hold -- so it is said, not silently re-filtered.
    """
    filled = 0
    for records in iter_fill_records(shard_path):
        lr = records["row"] - r0
        ai = records["ai"]
        if impute_mask is not None and len(ai) and not impute_mask[ai].all():
            raise ValueError(
                "fill shard contains analyses excluded by the ancestry-match filter "
                f"(first {int(ai[~impute_mask[ai]][0])}); the filter applied at "
                "checkpoint resolution and the one applied here disagree"
            )
        filled += apply_fill_batch(zb, sb, imputed_band, lr, ai, records["z"], records["se"])
    if imputed_band.max(initial=0) < CONFLICTED:
        return filled, _NO_FILLS, 0
    conflicted = imputed_band == CONFLICTED
    dropped = _conflicting_fill_records(shard_path, conflicted, r0)
    clear_conflicted(zb, sb, imputed_band, conflicted)
    return filled, dropped, int(conflicted.sum())


def _conflicting_fill_records(shard_path: Path, conflicted: np.ndarray, r0: int) -> np.ndarray:
    """Every fill record at a conflicted cell, in the shard's order."""
    parts = [
        records[conflicted[records["row"] - r0, records["ai"]]]
        for records in iter_fill_records(shard_path)
    ]
    return np.concatenate(parts) if parts else _NO_FILLS


def _log_conflicting_fills(
    dropped: np.ndarray, n_cells: int, name_row: Callable[[int], str]
) -> None:
    """Log every fill record at a conflicted cell, one line each (D8).

    A completion writes no Admission Record (ADR 0072), so its log is where
    the fills it drops are named, all of them.
    """
    n_rows = log_dropped("fill", (
        (analysis, name_row(row), z, se)
        for row, analysis, z, se in zip(
            dropped["row"].tolist(), dropped["ai"].tolist(),
            dropped["z"].tolist(), dropped["se"].tolist(), strict=True,
        )
    ))
    log.warning(
        "complete-dense: %d fill(s) at %d cell(s) disagree, so none is kept "
        "(duplicate_conflicting)",
        n_rows, n_cells,
    )


def _band_planes(plan: _BandPlan) -> tuple[str, ...]:
    return ("z", "se", "imputed", "eaf") if plan.carries_eaf else ("z", "se", "imputed")


def _encode_band(
    plan: _BandPlan, cells: CellValues, r0: int
) -> tuple[dict[str, np.ndarray], SideTables, np.ndarray]:
    """Judge and encode the band a step at a time: each plane's codes, the
    band's side-table entries, and its SEs above `float16`'s range per Analysis.

    Each step is placed at its own rows, so a cell's code and its side-table
    entry are the ones a whole-band encode gives it.
    """
    n_rows = len(cells.z)
    dtypes = {name: plan.root[name].dtype for name in _band_planes(plan)}
    codes = {name: np.empty((n_rows, plan.n_analyses), dtype=dt) for name, dt in dtypes.items()}
    z_overflow, eaf_exceptions, se_exceptions = (
        ZOverflowBuilder(), EafExceptionBuilder(), SeExceptionBuilder()
    )
    se_out_of_range = np.zeros(plan.n_analyses, dtype=np.int64)
    for s0, s1 in plan.steps(n_rows):
        step = CellValues(
            cells.z[s0:s1],
            cells.se[s0:s1],
            None if cells.eaf is None else cells.eaf[s0:s1],
            imputed=None if cells.imputed is None else cells.imputed[s0:s1],
        )
        encoded = plan.codec.encode(
            step,
            Placement.band(r0 + s0, r0 + s1, plan.n_analyses),
            where=f"complete-dense, rows {r0 + s0:,} to {r0 + s1:,}",
        )
        se_out_of_range += encoded.storable.se_out_of_range
        held = {"z": encoded.z, "se": encoded.se, "imputed": encoded.imputed, "eaf": encoded.eaf}
        for name in codes:
            codes[name][s0:s1] = held[name]
        for builder, table in (
            (z_overflow, encoded.tables.z_overflow),
            (eaf_exceptions, encoded.tables.eaf_exceptions),
            (se_exceptions, encoded.tables.se_exceptions),
        ):
            builder.add(table.index, table.value)
    tables = SideTables(z_overflow.table(), eaf_exceptions.table(), se_exceptions.table())
    return codes, tables, se_out_of_range


def _write_band(plan: _BandPlan, codes: dict[str, np.ndarray], r0: int, r1: int) -> None:
    """Each plane's rows of this band, written once: whole shards, which no
    other band's worker writes."""
    for name, values in codes.items():
        plan.root[name][r0:r1] = values


@dataclass
class _BandTallies:
    """What the band write counts, per Analysis or in total."""

    n_missing_off_panel: np.ndarray
    se_out_of_range: np.ndarray
    total_imputed: int = 0
    n_missing_imputation_failed: int = 0

    def add(self, outcome: _BandOutcome) -> None:
        self.n_missing_off_panel += outcome.n_missing_off_panel
        self.se_out_of_range += outcome.se_out_of_range
        self.total_imputed += outcome.n_imputed
        self.n_missing_imputation_failed += outcome.n_missing_imputation_failed


@dataclass
class _BandTables:
    """The bands' side-table entries, each band's a run kept in band order: laid
    end to end they are the tables (`table_end_to_end`), never sorted into place."""

    z_overflow: list[ZOverflowTable] = field(default_factory=list)
    eaf_exceptions: list[EafExceptionTable] = field(default_factory=list)
    se_exceptions: list[SeExceptionTable] = field(default_factory=list)

    def add(self, tables: SideTables) -> None:
        self.z_overflow.append(tables.z_overflow)
        self.eaf_exceptions.append(tables.eaf_exceptions)
        self.se_exceptions.append(tables.se_exceptions)


def _carried_eaf_baseline(
    src_root: Any, out_to_src: np.ndarray, n_variants: int
) -> np.ndarray | None:
    """The per-variant EAF baseline, carried from the source rows across the
    row remap; ``None`` when the source release carries none (ADR 0036).

    Carried with the values rather than recomputed, so a cell decoded from the
    source and re-encoded here lands on the same code -- completion moves
    values between two planes, it does not requantise them. Panel-only rows
    keep NaN: an imputed cell's frequency is the panel's, stored once per
    variant in ``eaf_reference`` and applied on read (ADR 0037 §4).
    """
    if EAF_BASELINE not in src_root:
        return None
    src_baseline = np.asarray(src_root[EAF_BASELINE][:], dtype=np.float32)
    out_baseline = np.full(n_variants, np.nan, dtype=np.float32)
    carried = out_to_src >= 0
    out_baseline[carried] = src_baseline[out_to_src[carried]]
    return out_baseline


def _source_se_coefficients(
    src_root: Any, encoding: StoreEncoding, n_analyses: int
) -> np.ndarray | None:
    """The source's SE model, which every completed SE is coded against under a
    residual plan, never refitted (issue #301); ``None`` for a `float16` plan."""
    if not encoding.se.is_residual:
        return None
    coefficients = np.asarray(src_root["se_coefficients"][:], dtype=np.float32)
    if coefficients.shape != (n_analyses, 2) or not np.all(np.isfinite(coefficients)):
        raise ValueError(f"se_coefficients must have finite shape ({n_analyses}, 2)")
    return coefficients


def _held_eaf_reference(root: Any) -> np.ndarray | None:
    """The panel frequencies as the release stores them, which an imputed
    cell's SE is coded against (ADR 0037 §4)."""
    if EAF_REFERENCE not in root:
        return None
    return np.asarray(root[EAF_REFERENCE][:], dtype=np.float32)


def _write_completed_bands(
    staged: StagedRelease,
    src_root: Any,
    out_to_src: np.ndarray,
    on_panel: np.ndarray,
    fill_shard_dir: Path,
    effective_chunks: tuple[int, int],
    n_variants: int,
    n_analyses: int,
    source_encoding: StoreEncoding,
    encoding: StoreEncoding,
    impute_mask: np.ndarray | None = None,
    name_row: Callable[[int], str] = str,
    n_workers: int = 1,
) -> _BandTallies:
    """Write every row band of the completed planes, across the write pool, and
    then the side tables and per-variant arrays the bands' codes need.

    Source z, SE and EAF are read decoded and written re-encoded through the
    source's plan -- completion moves values between two planes, it does not
    reinterpret them -- with the per-variant EAF baseline carried across the
    row remap and SE coded against the source's model. Fill records are read
    from per-band shard files in bounded chunks.

    ``impute_mask`` (bool per analysis; ``None`` = impute all) is the
    per-Analysis ancestry-match filter (ADR 0028): a masked-out analysis stays
    observed-only (NaN, ``imputed=0``) -- never imputed against a
    non-matching-ancestry panel. It is applied at checkpoint resolution, not
    here, so that ``completion_quality`` and the arrays are filtered by the
    same act; the band write only checks that the shards it reads honour it.
    """
    root = staged.arrays(mode="a")
    carries_eaf = "eaf" in root and "eaf" in src_root
    baseline = _carried_eaf_baseline(src_root, out_to_src, n_variants) if carries_eaf else None
    coefficients = _source_se_coefficients(src_root, encoding, n_analyses)
    plan = _BandPlan(
        root=root,
        source=DenseGridReader.open(src_root, source_encoding, source=_SOURCE),
        out_to_src=out_to_src,
        on_panel=on_panel,
        fill_shard_dir=fill_shard_dir,
        codec=AssociationCodec(
            encoding,
            eaf_baseline=baseline,
            eaf_reference=_held_eaf_reference(root),
            se_coefficients=coefficients,
        ),
        impute_mask=impute_mask,
        band_rows=completion_band_rows(_dense_shard_rows(staged, effective_chunks)),
        n_variants=n_variants,
        n_analyses=n_analyses,
        carries_eaf=carries_eaf,
    )
    tallies, tables = _run_bands(plan, n_workers, name_row)
    _write_side_arrays(root, tables, baseline, coefficients)
    return tallies


def _run_bands(
    plan: _BandPlan, n_workers: int, name_row: Callable[[int], str]
) -> tuple[_BandTallies, _BandTables]:
    """Every band across the write pool, its outcome taken in band order: the
    counts added, the side-table entries kept, and the fills a band dropped
    logged by their variant (D8)."""
    tallies = _BandTallies(
        n_missing_off_panel=np.zeros(plan.n_analyses, dtype=np.int64),
        se_out_of_range=np.zeros(plan.n_analyses, dtype=np.int64),
    )
    tables = _BandTables()
    n_bands = -(-plan.n_variants // plan.band_rows)
    workers = pool_workers(
        n_workers,
        _band_bytes(plan.n_analyses, plan.band_rows),
        what=BAND_PHASE,
        unit=f"{plan.band_rows:,}-row band of {plan.n_analyses:,} Analyses",
        units=n_bands,
        result_bytes=_band_result_bytes(plan.n_analyses, plan.band_rows),
    )
    started = time.monotonic()
    outcomes = ordered_write(_complete_band, plan, range(n_bands), workers)
    for band, outcome in enumerate(outcomes):
        if outcome.band != band:
            raise RuntimeError(f"band {outcome.band} arrived where band {band} belongs")
        tallies.add(outcome)
        tables.add(outcome.tables)
        if outcome.n_conflicted_cells:
            _log_conflicting_fills(outcome.dropped, outcome.n_conflicted_cells, name_row)
        log_progress(log, BAND_PHASE, band + 1, n_bands, started, every=max(1, n_bands // 20))
    return tallies, tables


def _write_side_arrays(
    root: Any,
    tables: _BandTables,
    baseline: np.ndarray | None,
    coefficients: np.ndarray | None,
) -> None:
    """The arrays the bands' codes cannot be decoded without: z's overflow
    table; the carried EAF baseline and its exception table, when the release
    carries one (a baseline-less plane has no residual codes to make exceptions
    for); and a residual SE plane's exception table and model."""
    table_end_to_end(ZOverflowTable, tables.z_overflow).write(root)
    if baseline is not None:
        write_eaf_baseline(root, baseline, compressor=_COMPRESSOR)
        table_end_to_end(EafExceptionTable, tables.eaf_exceptions).write(root)
    if coefficients is not None:
        write_dense_se_side_arrays(root, tables.se_exceptions, coefficients)
