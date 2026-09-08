"""Dense Reference Completion — enhancement pipeline.

Builds a Dense Reference-Completed Store Release from a Dense Observed-Only
Full Coverage source, per ADR 0022 (dense axis = source ∪ reference panel)
and ADR 0023 (LD-block process-pool parallelism with checkpointed resume).

Pipeline shape:
  Phase 1 (sequential): enumerate the genome-wide LD block set, build the
    union variant axis, seed z/se from the source, compute the per-Analysis
    n_missing_off_panel scalar.
  Phase 2 (parallel, n_workers processes over LD blocks): each worker opens
    the source store and LD panel itself, imputes serially per Analysis
    within its block, and writes its own checkpoint file.
  Phase 3 (sequential): merge all block results into the seeded z/se arrays,
    write the final zarr, completion_quality rows, top-hit indexes, and
    manifest.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numcodecs import Blosc

from opengwasdb.completion.ancestry_filter import derive_impute_analysis_ids
from opengwasdb.completion.block import REGION_CAP_BP, run_block
from opengwasdb.completion.checkpoint import (
    ALID_DTYPE,
    BlockCompletionResult,
    checkpoint_dir_for,
    require_fresh_destination,
    sanitize_block_id,
    write_block_checkpoint,
)
from opengwasdb.completion.ld_panel import (
    canonical_panel_alid as _canonical_panel_alid,
)
from opengwasdb.completion.ld_panel import (
    check_panel_has_chromosomes,
    list_all_blocks,
    list_chromosomes,
)
from opengwasdb.completion.manifest import (
    build_completion_provenance,
    completed_release_manifest,
)
from opengwasdb.completion.parallel import run_block_tasks
from opengwasdb.completion.reference_eaf import completed_eaf_scope, panel_reference_eaf
from opengwasdb.completion.schema import completion_quality_rollup, create_completion_quality_table
from opengwasdb.encoding import (
    EAF_BASELINE,
    DenseEafPlane,
    DenseSePlane,
    DenseZPlane,
    EafExceptionBuilder,
    StoreCodec,
    StoreEncoding,
    ZOverflowBuilder,
    positions_row_band,
    rewrite_dense_se,
    write_eaf_baseline,
    write_eaf_reference,
)
from opengwasdb.index import initialise_schema, set_metadata
from opengwasdb.layouts.dense.build import add_hit_counts, write_analyses_tsv
from opengwasdb.layouts.dense.constants import (
    DEFAULT_CHUNK_SHAPE,
    DEFAULT_COMPRESSOR,
    DEFAULT_DTYPE,
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
from opengwasdb.store.open import (
    OpenGWASDBStore,
    StagedRelease,
    check_writable_format_version,
    open_store,
)
from opengwasdb.variants import (
    CanonicalVariant,
    VariantAxis,
    VariantNormalisationError,
    VariantRecord,
    chromosome_sort_key,
    orient_to_canonical,
    parse_canonical_alid,
    write_variant_axis,
)

log = logging.getLogger(__name__)

_COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
_LD_PANEL_ID = "eur-hg38-gpm"


@dataclass(frozen=True)
class CompletionResult:
    output_path: Path
    n_variants: int
    n_analyses: int
    n_imputed: int
    n_missing_off_panel: int
    n_missing_imputation_failed: int


def _work_dir_for(dest_path: Path) -> Path:
    dest_path = Path(dest_path)
    return dest_path.parent / f".{dest_path.name}.tmp"


# ── Phase 2: per-block worker ───────────────────────────────────────────────


@dataclass(frozen=True)
class _BlockTask:
    tsv_path: Path
    source_path: Path
    min_cor: float
    thresh: float
    checkpoint_path: Path


def _make_reader(task: _BlockTask):
    """dense's half of the ``run_block`` seam: read every Analysis's observed
    z/se at a block's positions as one matrix slice, opening the source store
    and LD panel itself so no payload beyond a lightweight block descriptor
    needs to be pickled into the worker process.
    """

    def make_reader(block, canonical_alids: list[str | None]):
        src_axis = VariantAxis(task.source_path)
        try:
            src_store = open_store(task.source_path)
            src_root = src_store.arrays(mode="r")
            src_plane = DenseZPlane.open(src_root, src_store.manifest.encoding)
            src_se_plane = DenseSePlane.open(src_root, src_store.manifest.encoding)
            n_analyses = src_plane.n_analyses

            src_rows: list[int | None] = []
            for alid in canonical_alids:
                parsed = parse_canonical_alid(alid) if alid is not None else None
                rec = src_axis.by_alid(parsed) if parsed is not None else None
                src_rows.append(rec.variant_index if rec is not None else None)

            matched_local = [i for i, r in enumerate(src_rows) if r is not None]
            matched_src = [src_rows[i] for i in matched_local]

            z_obs = np.full((len(canonical_alids), n_analyses), np.nan, dtype=np.float64)
            se_obs = np.full((len(canonical_alids), n_analyses), np.nan, dtype=np.float64)
            if matched_local:
                z_obs[matched_local, :] = src_plane.rows(np.asarray(matched_src))
                se_obs[matched_local, :] = src_se_plane.rows(np.asarray(matched_src))
        finally:
            src_axis.close()

        def read(ai: int) -> tuple[np.ndarray, np.ndarray]:
            return z_obs[:, ai], se_obs[:, ai]

        return range(n_analyses), read

    return make_reader


def _run_block(task: _BlockTask) -> BlockCompletionResult | None:
    """Complete one LD block for every Analysis. Runs inside a worker process."""
    result = run_block(task.tsv_path, task.thresh, task.min_cor, REGION_CAP_BP, _make_reader(task))
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
    """Produce a Dense Reference-Completed Store Release from a Full Coverage
    Dense Observed-Only source.

    source_path: existing Dense Observed-Only, Full Coverage store.
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


def _enumerate_panel(ld_dir: Path, ancestry: str) -> tuple[list[Path], set[str]]:
    """Walk every LD block's TSV in chromosome order, collecting the block
    paths Phase 2 schedules and the canonical panel ALIDs the union axis must
    hold (ADR 0022)."""
    print("Enumerating genome-wide LD blocks...")
    tsv_paths: list[Path] = []
    panel_alids: set[str] = set()
    for chrom in list_chromosomes(ld_dir, ancestry):
        for block in list_all_blocks(ld_dir, ancestry, chrom):
            tsv_paths.append(block.tsv_path)
            for snp_id in block.snp_ids:
                ca = _canonical_panel_alid(snp_id)
                if ca is not None:
                    panel_alids.add(ca)
    print(f"LD panel: {len(tsv_paths):,} blocks, {len(panel_alids):,} panel variants")
    return tsv_paths, panel_alids


def _orient_panel_alid(alid: str) -> CanonicalVariant | None:
    """The store-canonical variant one panel ALID names, or ``None`` when the
    panel's identifier cannot be parsed or oriented. An ALID the panel cannot
    say what it is must never become a variant the store claims to cover."""
    parts = alid.split(":")
    if len(parts) != 4:
        return None
    chrom, pos_str, a1, a2 = parts
    try:
        cv_result = orient_to_canonical(chrom, int(pos_str), a1, a2)
    except (VariantNormalisationError, ValueError):
        return None
    return cv_result.variant


def _union_variant_table(
    src_variants: list[VariantRecord],
    src_alid_to_idx: dict[str, int],
    panel_alids: set[str],
) -> tuple[list[CanonicalVariant], list[CanonicalVariant], dict[str, int]]:
    """Append the panel variants the source does not already hold (deduplicated
    through canonical orientation) and sort the union into the store's
    canonical order."""
    present = set(src_alid_to_idx)
    new_canonical: list[CanonicalVariant] = []
    for alid in panel_alids:
        if alid in present:
            continue
        variant = _orient_panel_alid(alid)
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
    panel_paths, panel_alids = _enumerate_panel(ld_dir, ancestry)
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
        tsv_paths=panel_paths,
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
    blocks_dir = checkpoint_dir / "blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)

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
                )
            )

    if pending:
        print(f"  {n_existing:,} blocks already checkpointed, {len(pending):,} remaining")
    run_block_tasks(pending, n_workers, _run_block)
    return blocks_dir


def _sorted_union_map(new_alid_to_idx: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """The union ALID → row lookup, sorted by ALID, so checkpoint fills (which
    record ALIDs, not rows) resolve to union rows by binary search."""
    union_alids = np.fromiter(
        (alid.encode("ascii") for alid in new_alid_to_idx),
        dtype=ALID_DTYPE,
        count=len(new_alid_to_idx),
    )
    union_rows = np.fromiter(new_alid_to_idx.values(), dtype=np.int32, count=len(new_alid_to_idx))
    o = np.argsort(union_alids)
    return union_alids[o], union_rows[o]


def _merge_checkpoint_fills(
    staged: StagedRelease,
    axis: _UnionAxis,
    blocks_dir: Path,
    ld_dir: Path,
    ancestry: str,
    source_encoding: StoreEncoding,
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
    union_alids_s, union_rows_s = _sorted_union_map(axis.new_alid_to_idx)

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
    band_rows = _completion_band_rows(effective_chunks)
    fill_shard_dir, quality_count = _shard_checkpoint_fills_by_band(
        blocks_dir,
        staged,
        union_alids_s,
        union_rows_s,
        axis.n_variants,
        band_rows,
        impute_mask=axis.impute_mask,
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
) -> _CompletedArrays:
    """Phase 3 — stream the completed z/se/imputed matrix out of the sharded
    checkpoints into row-band writes (issue 044), returning the counts the
    metadata phases must state."""
    encoding, eaf_reference_present, effective_chunks, fill_shard_dir = _merge_checkpoint_fills(
        staged, axis, blocks_dir, ld_dir, ancestry, source_encoding
    )
    print("Writing data.zarr (band-streamed)...")
    n_missing_off_panel, n_missing_imputation_failed, total_imputed = _write_completed_bands(
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
    )
    shutil.rmtree(fill_shard_dir, ignore_errors=True)
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
    Observed-Only, Full Coverage source."""
    source_format_version = check_writable_format_version(
        manifest.format_version, source=f"source release {src}"
    )
    if manifest.primary_layout is not PrimaryStorageLayout.DENSE:
        raise ValueError(f"source store is not Dense (primary_layout={manifest.primary_layout})")
    if manifest.completion_state is not CompletionState.OBSERVED_ONLY:
        raise ValueError(
            f"source store is not Observed-Only (completion_state={manifest.completion_state})"
        )
    if manifest.association_coverage is not AssociationCoverage.FULL:
        raise ValueError(
            "Dense reference completion only supports Full Coverage sources "
            f"(association_coverage={manifest.association_coverage})"
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
        ),
    )
    staged.write_manifest(completed_manifest)


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
                completed_against=ancestry if axis.impute_mask is None or axis.impute_mask[i] else "",
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
    with OpenGWASDBStore.staging(dst, overwrite=True) as staged:
        source = open_store(src)
        manifest = source.manifest
        source_format_version = _checked_dense_source(manifest, src)
        print(f"Source store: {manifest.store_id} / {manifest.release_id}")

        axis = _build_union_axis(src, staged, ld_dir, ancestry, impute_analysis_ids)
        src_root = source.arrays(mode="r")
        blocks_dir = _complete_pending_blocks(
            axis,
            src,
            checkpoint_dir,
            min_cor=min_cor,
            thresh=thresh,
            n_workers=n_workers,
        )
        arrays = _stream_completed_arrays(
            staged, src_root, axis, blocks_dir, ld_dir, ancestry, manifest.encoding
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
        result = _finalise_release(staged, dst, axis, arrays, dst_analyses)
    return result


# Row-band height for streaming the completed matrix — the seed/fill/write pass
# never holds the full (n_variants × n_analyses) matrices in RAM (issue 044).
_BAND_ROWS = 250_000
_FILL_RECORD_DTYPE = np.dtype(
    [("row", np.int32), ("ai", np.int32), ("z", np.float32), ("se", np.float32)]
)
_FILL_RECORD_READ_COUNT = 5_000_000


def _completion_band_rows(effective_chunks: tuple[int, int]) -> int:
    return max(int(effective_chunks[0]), _BAND_ROWS)


def _fill_shard_path(fill_shard_dir: Path, band_index: int) -> Path:
    return fill_shard_dir / f"band-{band_index:06d}.bin"


def _iter_fill_records(path: Path) -> Iterator[np.ndarray]:
    if not path.exists():
        return
    with open(path, "rb") as fh:
        while True:
            records = np.fromfile(fh, dtype=_FILL_RECORD_DTYPE, count=_FILL_RECORD_READ_COUNT)
            if len(records) == 0:
                break
            yield records


def _insert_completion_quality_batch(
    db: Any,
    batch: list[tuple[int, str, float | None, int, int]],
) -> None:
    if not batch:
        return
    db.executemany(
        "INSERT INTO completion_quality "
        "(analysis_index, block_id, pearson_r, n_imputed, n_missing) "
        "VALUES (?, ?, ?, ?, ?)",
        batch,
    )
    db.commit()


def _shard_checkpoint_fills_by_band(
    blocks_dir: Path,
    staged: StagedRelease,
    union_alids_s: np.ndarray,
    union_rows_s: np.ndarray,
    n_variants: int,
    band_rows: int,
    impute_mask: np.ndarray | None = None,
) -> tuple[Path, int]:
    """Resolve checkpoint fills into raw row-band shard files.

    Checkpoints store fill rows by ALID because the union variant axis is built
    by the parent. This pass resolves those ALIDs once, writes compact
    ``(row, analysis, z, se)`` records to per-band files, and streams
    completion_quality directly into SQLite.

    ``impute_mask`` (bool per analysis; ``None`` = impute all) is applied here,
    because this is where a worker's *candidate* fills become the release's.
    The blocks are imputed for every Analysis and the ancestry-match filter
    (ADR 0028) is decided afterwards, so a nonmatching Analysis arrives with
    both fills and completion-quality rows. Dropping only the fills, later, at
    the write, left ``completion_quality`` -- and through it ``analyses.tsv``'s
    ``completion_n_imputed_total`` and ``eaf_scope`` -- counting cells the
    release does not contain. Filtered once, here, the table and the arrays
    cannot disagree.
    """
    fill_shard_dir = staged.path / "fill_shards"
    if fill_shard_dir.exists():
        shutil.rmtree(fill_shard_dir)
    fill_shard_dir.mkdir()

    quality_count = 0
    quality_batch: list[tuple[int, str, float | None, int, int]] = []
    quality_batch_size = 100_000

    with staged.index_connection() as dst_db:
        for ckpt in sorted(blocks_dir.glob("*.npz")):
            with np.load(ckpt, allow_pickle=False) as d:
                bid = str(d["block_id"][0])
                for ai, p, ni, nm in zip(
                    d["q_ai"], d["q_pearson"], d["q_nimp"], d["q_nmiss"], strict=True
                ):
                    if impute_mask is not None and not impute_mask[int(ai)]:
                        continue
                    quality_batch.append(
                        (
                            int(ai),
                            bid,
                            None if not np.isfinite(p) else float(p),
                            int(ni),
                            int(nm),
                        )
                    )
                    quality_count += 1
                    if len(quality_batch) >= quality_batch_size:
                        _insert_completion_quality_batch(dst_db, quality_batch)
                        quality_batch.clear()

                f_alid = d["f_alid"]
                if not len(f_alid):
                    continue
                if f_alid.dtype.kind == "U":
                    f_alid = f_alid.astype(ALID_DTYPE)

                idx = np.minimum(np.searchsorted(union_alids_s, f_alid), len(union_alids_s) - 1)
                matched = union_alids_s[idx] == f_alid
                if not matched.any():
                    continue

                rows = union_rows_s[idx[matched]].astype(np.int32, copy=False)
                in_bounds = (rows >= 0) & (rows < n_variants)
                if not in_bounds.any():
                    continue

                rows = rows[in_bounds]
                ai = d["f_ai"][matched].astype(np.int32, copy=False)[in_bounds]
                z = d["f_z"][matched].astype(np.float32, copy=False)[in_bounds]
                se = d["f_se"][matched].astype(np.float32, copy=False)[in_bounds]
                if impute_mask is not None:
                    keep = impute_mask[ai]
                    if not keep.any():
                        continue
                    rows, ai, z, se = rows[keep], ai[keep], z[keep], se[keep]
                band_ids = rows // band_rows

                for band_index in np.unique(band_ids):
                    selected = band_ids == band_index
                    records = np.empty(int(selected.sum()), dtype=_FILL_RECORD_DTYPE)
                    records["row"] = rows[selected]
                    records["ai"] = ai[selected]
                    records["z"] = z[selected]
                    records["se"] = se[selected]
                    with open(_fill_shard_path(fill_shard_dir, int(band_index)), "ab") as fh:
                        records.tofile(fh)

        _insert_completion_quality_batch(dst_db, quality_batch)

    return fill_shard_dir, quality_count


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
    effective_chunks = (min(chunk_shape[0], n_variants), min(chunk_shape[1], n_analyses))
    codec = StoreCodec(encoding)
    root = staged.arrays(mode="w")

    def plane(name: str, plane_dtype: Any, fill: Any) -> None:
        root.create_dataset(
            name,
            shape=(n_variants, n_analyses),
            chunks=effective_chunks,
            compressor=_COMPRESSOR,
            dtype=plane_dtype,
            fill_value=fill,
        )

    plane("z", codec.z_dtype, codec.z_fill_value)
    # Scratch in float32 so an exact residual exception is not rounded before
    # the destination's final SE encoding is written below.
    plane("se", "float32", float("nan"))
    plane("imputed", "uint8", 0)
    if src_has_eaf:
        # Never float16 -- see `build_vcf._create_eaf_array` for why it cannot
        # hold an EAF near 1 (ADR 0036). Created only when the observed store
        # had one: completion adds panel rows, it does not invent frequencies
        # the source never reported.
        plane("eaf", codec.eaf_dtype, codec.eaf_fill_value)
    root.create_dataset(
        "on_panel",
        data=on_panel.astype(np.uint8),
        chunks=(effective_chunks[0],),
        compressor=_COMPRESSOR,
        dtype="uint8",
    )
    if eaf_reference is not None:
        write_eaf_reference(root, eaf_reference, compressor=_COMPRESSOR)
    root.attrs["layout"] = "dense"
    root.attrs["completion_state"] = "reference_completed"
    root.attrs["compressor"] = DEFAULT_COMPRESSOR
    root.attrs["chunk_shape"] = list(effective_chunks)
    return effective_chunks


# ── completed-band writer phases ─────────────────────────────────────────
#
# `_write_completed_bands` splits the write into cohesive phases so each stays
# small enough to hold in the head at once: a z pass (seed + fills + missingness
# counters + overflow/imputed side tables), an se pass that fills exactly the
# same cells (source z/se missingness is consistent -- a validated store
# invariant), an eaf pass that carries observed frequencies across the row
# remap, and the residual-SE rewrite that finalises the scratch se plane.
# Every pass streams through its own float32 band buffer, so peak memory is ~one
# band rather than z + se + imputed held together (issue 044).


def _band_source_rows(
    out_to_src: np.ndarray, r0: int, r1: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rows of band ``[r0:r1)`` that carry a source cell, and the source rows
    they map to."""
    valid = np.where(out_to_src[r0:r1] >= 0)[0]
    return valid, out_to_src[r0:r1][valid]


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
    off_local = np.where(on_panel[r0:r1] == 0)[0]
    if len(off_local):
        n_missing_off_panel += np.isnan(band[off_local, :]).sum(axis=0).astype(np.int64)


def _count_band_imputation_failed(
    band: np.ndarray, on_panel: np.ndarray, r0: int, r1: int
) -> int:
    """On-panel cells still missing once this band's fills were applied."""
    on_local = np.where(on_panel[r0:r1] == 1)[0]
    if len(on_local):
        return int(np.isnan(band[on_local, :]).sum())
    return 0


def _apply_fill_shard_records(
    band: np.ndarray,
    imputed_band: np.ndarray | None,
    shard_path: Path,
    r0: int,
    value_field: str,
    *,
    impute_mask: np.ndarray | None = None,
    validate_mask: bool = False,
) -> int:
    """Stream one band's fill-shard records into a NaN-seeded band buffer.

    A record lands only in a cell the band still holds missing -- it never
    overwrites an observed value -- and marks the cell imputed when
    ``imputed_band`` is given (the z pass); the count of cells actually filled
    is returned. The se pass passes the same shard with ``imputed_band=None``,
    and deliberately no mask check: pass 1 read the same shards and would have
    raised, so a second filter is a second chance for the two passes to fill
    different cells (the missingness-consistency invariant).

    ``validate_mask=True`` (the z pass) enforces the ancestry-match filter: a
    nonmatching Analysis reaching the write means the filter applied at
    checkpoint resolution and the one applied here disagree -- the
    disagreement that let ``completion_quality`` count cells the release did
    not hold -- so it is said, not silently re-filtered.
    """
    filled = 0
    for records in _iter_fill_records(shard_path):
        lr = records["row"] - r0
        ai = records["ai"]
        if (
            validate_mask
            and impute_mask is not None
            and len(ai)
            and not impute_mask[ai].all()
        ):
            raise ValueError(
                "fill shard contains analyses excluded by the ancestry-match filter "
                f"(first {int(ai[~impute_mask[ai]][0])}); the filter applied at "
                "checkpoint resolution and the one applied here disagree"
            )
        fillable = ~np.isfinite(band[lr, ai])
        if fillable.any():
            lrm, aim = lr[fillable], ai[fillable]
            band[lrm, aim] = records[value_field][fillable]
            if imputed_band is not None:
                imputed_band[lrm, aim] = 1
            filled += int(fillable.sum())
    return filled


def _write_completed_z_bands(
    root: Any,
    src_plane: DenseZPlane,
    out_to_src: np.ndarray,
    on_panel: np.ndarray,
    fill_shard_dir: Path,
    codec: StoreCodec,
    overflow: ZOverflowBuilder,
    n_variants: int,
    n_analyses: int,
    band_rows: int,
    n_missing_off_panel: np.ndarray,
    impute_mask: np.ndarray | None,
) -> tuple[int, int]:
    """Seed z from the source, apply the fills, and write the z + imputed
    bands, one row-band at a time. Missingness is accounted here, in the pass
    that sees the fills land: per-Analysis off-panel missing accumulates into
    ``n_missing_off_panel`` (in place), and on-panel cells still NaN after the
    fills are the imputation failures. Out-of-range z cells go into the
    ``overflow`` builder's side table, which the caller writes once, after the
    whole pass, so the table is built by one codec plan.

    Returns ``(total_imputed, n_missing_imputation_failed)``.
    """
    z_arr, imp_arr = root["z"], root["imputed"]
    band = np.empty((band_rows, n_analyses), dtype=np.float32)
    total_imputed = 0
    n_missing_imputation_failed = 0
    for band_index, r0 in enumerate(range(0, n_variants, band_rows)):
        r1 = min(r0 + band_rows, n_variants)
        zb = band[: r1 - r0]
        zb[:] = np.nan
        imp_band = np.zeros((r1 - r0, n_analyses), dtype=np.uint8)

        valid, srows = _band_source_rows(out_to_src, r0, r1)
        if len(valid):
            zb[valid, :] = src_plane.rows(srows)

        _count_band_off_panel_missing(n_missing_off_panel, zb, on_panel, r0, r1)

        total_imputed += _apply_fill_shard_records(
            zb,
            imp_band,
            _fill_shard_path(fill_shard_dir, band_index),
            r0,
            "z",
            impute_mask=impute_mask,
            validate_mask=True,
        )

        n_missing_imputation_failed += _count_band_imputation_failed(zb, on_panel, r0, r1)

        z_arr[r0:r1] = codec.encode_z(
            zb, positions=positions_row_band(r0, n_analyses), overflow=overflow
        )
        imp_arr[r0:r1] = imp_band
    return total_imputed, n_missing_imputation_failed


def _write_completed_se_bands(
    root: Any,
    src_se_plane: DenseSePlane,
    out_to_src: np.ndarray,
    fill_shard_dir: Path,
    n_variants: int,
    n_analyses: int,
    band_rows: int,
) -> None:
    """Seed se from the source and apply the same fills pass 1 applied to z.

    No mask check and no counts here: pass 1 read the same shards (raising on
    a disagreement) and accounted the outcomes, and this pass must fill exactly
    the cells pass 1 filled for z and se to describe the same completed store.
    The scratch float32 se band is written as-is; the residual-SE rewrite
    finalises it afterwards.
    """
    se_arr = root["se"]
    band = np.empty((band_rows, n_analyses), dtype=np.float32)
    for band_index, r0 in enumerate(range(0, n_variants, band_rows)):
        r1 = min(r0 + band_rows, n_variants)
        sb = band[: r1 - r0]
        sb[:] = np.nan

        valid, srows = _band_source_rows(out_to_src, r0, r1)
        if len(valid):
            sb[valid, :] = src_se_plane.rows(np.asarray(srows, dtype=np.int64))

        _apply_fill_shard_records(
            sb, None, _fill_shard_path(fill_shard_dir, band_index), r0, "se"
        )

        se_arr[r0:r1] = sb


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


def _write_completed_eaf_bands(
    root: Any,
    src_root: Any,
    out_to_src: np.ndarray,
    source_encoding: StoreEncoding,
    encoding: StoreEncoding,
    n_variants: int,
    n_analyses: int,
    band_rows: int,
) -> None:
    """Carry observed frequencies across the row remap, one row-band at a time.

    Observed cells keep their source value and nothing else: an imputed cell's
    frequency is the panel's (stored once per variant and applied on read), and
    an observed cell whose source reported none stays absent. The exception
    side table is written only when the release carries a baseline -- a
    baseline-less eaf plane has no residual codes to make exceptions for.
    """
    src_eaf_plane = DenseEafPlane.open(src_root, source_encoding)
    eaf_codec = StoreCodec(encoding)
    exceptions = EafExceptionBuilder()
    out_baseline = _carried_eaf_baseline(src_root, out_to_src, n_variants)
    eaf_arr = root["eaf"]
    eaf_band = np.empty((band_rows, n_analyses), dtype=np.float32)
    for r0 in range(0, n_variants, band_rows):
        r1 = min(r0 + band_rows, n_variants)
        eb = eaf_band[: r1 - r0]
        eb[:] = np.nan
        valid = np.where(out_to_src[r0:r1] >= 0)[0]
        if len(valid):
            eb[valid, :] = src_eaf_plane.points(
                np.repeat(out_to_src[r0:r1][valid], n_analyses),
                np.tile(np.arange(n_analyses, dtype=np.int64), len(valid)),
            ).reshape(len(valid), n_analyses)
        band_baseline = (
            None
            if out_baseline is None
            else np.repeat(out_baseline[r0:r1, None], n_analyses, axis=1)
        )
        eaf_arr[r0:r1] = eaf_codec.encode_eaf(
            eb,
            baseline=band_baseline,
            positions=positions_row_band(r0, n_analyses),
            exceptions=exceptions,
        )
    if out_baseline is not None:
        write_eaf_baseline(root, out_baseline, compressor=_COMPRESSOR)
        exceptions.table().write(root)


def _rewrite_completed_residual_se(
    root: Any, src_root: Any, encoding: StoreEncoding
) -> None:
    """Finalise the scratch float32 se plane under the destination's plan.

    A residual plan needs the source release's per-Analysis coefficients (the
    prediction each cell's stored int8 residual is measured against); a
    non-residual plan is narrowed to float16. Either way the source
    coefficients are read only when the plan is residual.
    """
    source_coefficients = (
        np.asarray(src_root["se_coefficients"][:], dtype=np.float32)
        if encoding.se.is_residual
        else None
    )
    rewrite_dense_se(root, encoding, source_coefficients)


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
) -> tuple[np.ndarray, int, int]:
    """Seed z/se from the source, apply the imputed fills, and write
    z/se/imputed one row-band at a time, in cohesive phases: the z pass seeds
    from the source and applies the fills while accounting missingness and
    validating the fill shards against the ancestry-match filter; the se pass
    fills exactly the same cells (source z/se missingness is consistent -- a
    validated store invariant); the eaf pass carries observed frequencies
    across the row remap; and the residual-SE rewrite finalises the scratch
    float32 se plane. Each phase streams through its own float32 band buffer,
    so peak memory is ~one band rather than z + se + imputed held together,
    and each reads only its own source array once. Returns
    ``(n_missing_off_panel[n_analyses], n_missing_imputation_failed,
    total_imputed)``. Fill records are read from per-band shard files in
    bounded chunks.

    ``impute_mask`` (bool per analysis; ``None`` = impute all) is the
    per-Analysis ancestry-match filter (ADR 0028): a masked-out analysis stays
    observed-only (NaN, ``imputed=0``) -- never imputed against a
    non-matching-ancestry panel. It is applied at checkpoint resolution, not
    here, so that ``completion_quality`` and the arrays are filtered by the
    same act; this function only checks that the shards it reads honour it.
    """
    root = staged.arrays(mode="a")
    codec = StoreCodec(encoding)
    overflow = ZOverflowBuilder()
    band_rows = _completion_band_rows(effective_chunks)

    n_missing_off_panel = np.zeros(n_analyses, dtype=np.int64)

    src_se_plane = DenseSePlane.open(src_root, source_encoding)
    # Source z is read decoded and written re-encoded, through the same plan --
    # completion moves values between two planes, it does not reinterpret them.
    src_plane = DenseZPlane.open(src_root, source_encoding)

    total_imputed, n_missing_imputation_failed = _write_completed_z_bands(
        root,
        src_plane,
        out_to_src,
        on_panel,
        fill_shard_dir,
        codec,
        overflow,
        n_variants,
        n_analyses,
        band_rows,
        n_missing_off_panel,
        impute_mask,
    )
    overflow.table().write(root)

    _write_completed_se_bands(
        root, src_se_plane, out_to_src, fill_shard_dir, n_variants, n_analyses, band_rows
    )

    if "eaf" in root and "eaf" in src_root:
        _write_completed_eaf_bands(
            root,
            src_root,
            out_to_src,
            source_encoding,
            encoding,
            n_variants,
            n_analyses,
            band_rows,
        )

    _rewrite_completed_residual_se(root, src_root, encoding)

    return n_missing_off_panel, n_missing_imputation_failed, total_imputed
