"""Ragged Reference Completion — enhancement pipeline.

Reads an observed-only ragged store and writes a new Reference-Completed
Store Release by imputing z-scores and SE for all LD reference panel
variants within each Analysis's declared cis window.

Pipeline shape mirrors dense completion (ADR 0023): LD blocks are the
process-pool parallelism unit, checkpointed for resume. The difference from
dense is the unit of relevance -- a block matters only to the Analyses whose
cis window touches it, not to every Analysis genome-wide -- so Phase 1 also
records a block-to-Analyses mapping alongside the union variant axis, and
Phase 2's tasks are told which Analyses to complete rather than assuming all
of them:
  Phase 1 (sequential): scan every Analysis's cis window for touching LD
    blocks, collecting the union variant axis and a block_id -> Analysis
    indices mapping as it goes.
  Phase 2 (parallel, n_workers processes over touched LD blocks): each
    worker opens the source store and LD panel itself, completes every
    Analysis assigned to its block, and writes its own checkpoint file.
  Phase 3 (sequential): merge all block results, assemble each Analysis's
    completed association list (CSR), write completion_quality, top-hit
    indexes, and manifest.
"""
from __future__ import annotations

import json
import logging
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from numcodecs import Blosc

from opengwasdb.completion.ancestry_filter import derive_impute_analysis_ids
from opengwasdb.completion.block import REGION_CAP_BP, run_block
from opengwasdb.completion.checkpoint import (
    BlockCompletionResult,
    checkpoint_dir_for,
    read_block_checkpoint,
    sanitize_block_id,
    write_block_checkpoint,
)
from opengwasdb.completion.ld_panel import (
    LDBlock,
    blocks_over_variants,
    canonical_panel_alid,
    find_blocks,
)
from opengwasdb.completion.manifest import build_completion_provenance
from opengwasdb.completion.parallel import init_block_worker
from opengwasdb.completion.schema import completion_quality_rollup, create_completion_quality_table
from opengwasdb.encoding import StoreCodec, ZOverflowBuilder, positions_flat
from opengwasdb.layouts.dense.build import add_hit_counts
from opengwasdb.layouts.ragged.top_hits import build_ragged_top_hit_indexes
from opengwasdb.layouts.ragged.zarr_csr import RAGGED_ZARR_PATH, RaggedCSRReader
from opengwasdb.model.analyses import read_analysis_records, write_analysis_records
from opengwasdb.model.enums import CompletionState
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.store.open import (
    OpenGWASDBStore,
    check_writable_format_version,
    open_store,
)
from opengwasdb.variants.axis import (
    VariantAxis,
    write_variant_axis,
)
from opengwasdb.variants.normalise import (
    CanonicalVariant,
    VariantNormalisationError,
    chromosome_sort_key,
    orient_to_canonical,
)

log = logging.getLogger(__name__)

_COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
_ASSOC_CHUNK = 200_000
_OFFSET_CHUNK = 10_000

_LD_PANEL_ID = "eur-hg38-gpm"


@dataclass(frozen=True)
class CompletionResult:
    output_path: Path
    n_variants: int
    n_analyses: int
    n_associations: int
    n_imputed: int
    n_missing: int


# ── Phase 2: per-block worker ───────────────────────────────────────────────


@dataclass(frozen=True)
class _BlockTask:
    tsv_path: Path
    source_path: Path
    analysis_indices: list[int]
    min_cor: float
    thresh: float
    region_cap_bp: int | None
    checkpoint_path: Path


def _make_reader(task: _BlockTask):
    """ragged's half of the ``run_block`` seam: read one Analysis's observed
    z/se at a time from its CSR row, since (unlike dense's Full Coverage
    matrix) only the Analyses this task was assigned even have a cis window
    touching this block.
    """
    def make_reader(block, canonical_alids: list[str | None]):
        src_csr = RaggedCSRReader(task.source_path)
        src_variant_axis = VariantAxis(task.source_path)
        try:
            src_alids = [v.alid for v in src_variant_axis.all()]
        finally:
            src_variant_axis.close()

        def read(ai: int) -> tuple[np.ndarray, np.ndarray]:
            obs = src_csr.get_analysis(ai)
            obs_alid_to_z: dict[str, float] = {}
            obs_alid_to_se: dict[str, float] = {}
            # Observed EAF carried across the rebuild (ADR 0036). Reference
            # Completion adds panel rows to an Analysis; it does not change
            # what the source reported for the rows it already had.
            obs_alid_to_eaf: dict[str, float] = {}
            for vi_old, z_val, se_val, eaf_val in zip(
                obs.variant_index.tolist(),
                obs.z.tolist(),
                obs.se.tolist(),
                obs.eaf.tolist(),
                strict=True,
            ):
                alid = src_alids[vi_old]
                obs_alid_to_z[alid] = float(z_val)
                obs_alid_to_se[alid] = float(se_val)
                obs_alid_to_eaf[alid] = float(eaf_val)

            z_dense = np.array(
                [obs_alid_to_z.get(a, float("nan")) if a is not None else float("nan")
                 for a in canonical_alids],
                dtype=np.float64,
            )
            se_dense = np.array(
                [obs_alid_to_se.get(a, float("nan")) if a is not None else float("nan")
                 for a in canonical_alids],
                dtype=np.float64,
            )
            return z_dense, se_dense

        return task.analysis_indices, read

    return make_reader


def _run_block(task: _BlockTask) -> BlockCompletionResult | None:
    """Complete one LD block for the Analyses whose cis window touches it."""
    result = run_block(
        task.tsv_path, task.thresh, task.min_cor, task.region_cap_bp, _make_reader(task)
    )
    if result is None:
        return None
    write_block_checkpoint(task.checkpoint_path, result)
    # Fills stay on disk (the checkpoint); the parent reads them back in Phase 3
    # (mirrors dense — see issue 044).
    return BlockCompletionResult(block_id=result.block_id, quality_rows=[], fills=[])


# ── Public entry points ─────────────────────────────────────────────────────


def complete_ragged_store(
    source_path: str | Path,
    dest_path: str | Path,
    ld_dir: str | Path,
    *,
    ancestry: str = "EUR",
    cis_window_bp: int = 1_000_000,
    min_cor: float = 0.7,
    thresh: float = 0.9,
    release_id: str | None = None,
    ld_panel_id: str = _LD_PANEL_ID,
    n_workers: int = 1,
    overwrite: bool = False,
    impute_analysis_ids: set[str] | None = None,
    region_cap_bp: int | None = REGION_CAP_BP,
) -> CompletionResult:
    """Produce a Reference-Completed Store Release from an observed-only ragged store.

    source_path: existing observed-only ragged store.
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

    if dst.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {dst}. Use overwrite=True.")

    if checkpoint_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"A checkpoint directory already exists at {checkpoint_dir}. "
                "Use resume_ragged_completion() to continue it, or overwrite=True to discard it."
            )
        shutil.rmtree(checkpoint_dir)

    if impute_analysis_ids is None:
        impute_analysis_ids = derive_impute_analysis_ids(
            _read_analyses_rows(Path(source_path)), ancestry
        )

    (checkpoint_dir / "blocks").mkdir(parents=True)
    build_params = {
        "source_path": str(Path(source_path).resolve()),
        "dest_path": str(dst.resolve()),
        "ld_dir": str(Path(ld_dir).resolve()),
        "ancestry": ancestry,
        "cis_window_bp": cis_window_bp,
        "min_cor": min_cor,
        "thresh": thresh,
        "release_id": release_id,
        "ld_panel_id": ld_panel_id,
        "impute_analysis_ids": sorted(impute_analysis_ids)
        if impute_analysis_ids is not None
        else None,
        "region_cap_bp": region_cap_bp,
    }
    (checkpoint_dir / "build_params.json").write_text(
        json.dumps(build_params, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    result = _run_completion(
        Path(source_path), dst, Path(ld_dir),
        ancestry=ancestry, cis_window_bp=cis_window_bp, min_cor=min_cor, thresh=thresh,
        release_id=release_id, ld_panel_id=ld_panel_id,
        n_workers=n_workers, checkpoint_dir=checkpoint_dir,
        impute_analysis_ids=impute_analysis_ids, region_cap_bp=region_cap_bp,
    )
    shutil.rmtree(checkpoint_dir)
    return result


def resume_ragged_completion(
    checkpoint_dir: str | Path,
    *,
    n_workers: int = 1,
) -> CompletionResult:
    """Resume an interrupted complete_ragged_store() run.

    Takes only the checkpoint directory path -- all other build parameters
    are loaded from the build_params.json written on the first run, so a
    resumed run can never silently apply a different parameter set than the
    one its existing per-block checkpoints were computed under. Phase 1 (the
    union variant axis and block/Analysis mapping) is cheap and re-derived
    deterministically from source_path/ld_dir/cis_window_bp; only Phase 2's
    per-block results need to survive on disk to make resume skip work.
    """
    checkpoint_dir = Path(checkpoint_dir)
    params = json.loads((checkpoint_dir / "build_params.json").read_text())

    impute_ids = params.get("impute_analysis_ids")
    result = _run_completion(
        Path(params["source_path"]), Path(params["dest_path"]), Path(params["ld_dir"]),
        ancestry=params["ancestry"], cis_window_bp=params["cis_window_bp"],
        min_cor=params["min_cor"], thresh=params["thresh"],
        release_id=params["release_id"], ld_panel_id=params["ld_panel_id"],
        n_workers=n_workers, checkpoint_dir=checkpoint_dir,
        impute_analysis_ids=set(impute_ids) if impute_ids is not None else None,
        region_cap_bp=params.get("region_cap_bp"),
    )
    shutil.rmtree(checkpoint_dir)
    return result


def _read_analyses_rows(store_path: Path) -> list[dict]:
    return [
        {"analysis_id": a.analysis_id, "assigned_ancestry": a.assigned_ancestry}
        for a in read_analysis_records(store_path / "analyses.tsv")
    ]


# ── Shared pipeline core ────────────────────────────────────────────────────


def _run_completion(
    source_path: Path,
    dest_path: Path,
    ld_dir: Path,
    *,
    ancestry: str,
    cis_window_bp: int,
    min_cor: float,
    thresh: float,
    release_id: str | None,
    ld_panel_id: str,
    n_workers: int,
    checkpoint_dir: Path,
    impute_analysis_ids: set[str] | None,
    region_cap_bp: int | None,
) -> CompletionResult:
    src = Path(source_path)
    dst = Path(dest_path)

    source = open_store(src)
    manifest = source.manifest
    # See the dense path: refused before the work, not after it (ADR 0038 §4).
    source_format_version = check_writable_format_version(
        manifest.format_version, source=f"source release {src}"
    )
    print(f"Source store: {manifest.store_id} / {manifest.release_id}")
    print(f"  completion_state: {manifest.completion_state}")

    with OpenGWASDBStore.staging(dst, overwrite=True) as staged:
        # ── Phase 1: union variant axis + block/Analysis mapping ───────────
        src_variant_axis = VariantAxis(src)
        src_variants = src_variant_axis.all()
        src_variant_axis.close()
        src_alids: list[str] = [v.alid for v in src_variants]
        alid_to_src_idx: dict[str, int] = {v.alid: v.variant_index for v in src_variants}

        # analyses.tsv is the sole source of truth for Analytical Metadata
        # (ADR 0034, issue #69) -- including each Analysis's Trait genomic
        # position, so cis-window/LD-block scanning below reads it directly
        # rather than through a second, independently-shaped position file.
        src_analyses = sorted(
            read_analysis_records(src / "analyses.tsv"), key=lambda a: int(a.analysis_index)
        )
        n_analyses = len(src_analyses)

        print(f"Source: {len(src_alids):,} variants, {n_analyses:,} analyses")

        if impute_analysis_ids is None:
            impute_mask = None
        else:
            impute_mask = np.array(
                [a.analysis_id in impute_analysis_ids for a in src_analyses], dtype=bool
            )
            n_match = int(impute_mask.sum())
            print(f"Ancestry-match filter: imputing {n_match:,}/{n_analyses:,} analyses")

        # A Store Family with no single encoding gene per Analysis carries no
        # trait position at all (issue #102), so there is no cis window to scan
        # from. Those Analyses are enumerated from their own retained variants
        # instead -- every block holding an association they already have.
        # An Analysis with no Trait position has no cis window to scan from, so
        # its blocks are the ones it already holds enough observations in
        # (issue #102). Resolved for all such Analyses in one pass, keyed by
        # ALID, so the panel is read once rather than once per Analysis.
        src_csr = RaggedCSRReader(src)
        gene_target_less = [
            i for i, a in enumerate(src_analyses)
            if not (a.trait_chr and a.trait_bp) and (impute_mask is None or impute_mask[i])
        ]
        own_variant_blocks: dict[int, list[LDBlock]] = {}
        if gene_target_less:
            analyses_by_alid: dict[str, list[int]] = {}
            chromosomes: set[str] = set()
            for i in gene_target_less:
                for vi in src_csr.variant_indices(i).tolist():
                    analyses_by_alid.setdefault(src_alids[vi], []).append(i)
                    chromosomes.add(src_variants[vi].chromosome)
            print(
                f"  {len(gene_target_less):,} analyses have no Trait position; scanning "
                f"{len(chromosomes)} chromosome(s) of panel blocks for regions they hold"
            )
            own_variant_blocks = blocks_over_variants(
                ld_dir, ancestry, analyses_by_alid, sorted(chromosomes)
            )

        print("Scanning for LD blocks + new panel variants (cis windows where a "
              "Trait position exists, the Analysis's own variants otherwise)...")
        new_alids: set[str] = set()
        block_to_tsv: dict[str, Path] = {}
        block_to_analyses: dict[str, list[int]] = {}
        block_canonical_alids: dict[str, list[str | None]] = {}
        analysis_to_blocks: dict[int, list[str]] = {}

        for i, a in enumerate(src_analyses):
            # Ancestry-mismatched Analyses are left observed-only (ADR 0028):
            # not assigned to any block, not touched by Phase 2, no
            # completion_quality rows -- Phase 3's pass-through path below
            # (the same one "no trait position" already uses) carries their
            # original associations through completely unchanged.
            if impute_mask is not None and not impute_mask[i]:
                analysis_to_blocks[i] = []
                continue
            if a.trait_chr and a.trait_bp:
                start = max(1, int(a.trait_bp) - cis_window_bp)
                end = int(a.trait_bp) + cis_window_bp
                blocks = find_blocks(ld_dir, ancestry, a.trait_chr, start, end)
            else:
                blocks = own_variant_blocks.get(i, [])
            analysis_to_blocks[i] = [b.block_id for b in blocks]
            for block in blocks:
                block_to_analyses.setdefault(block.block_id, []).append(i)
                if block.block_id in block_to_tsv:
                    continue
                block_to_tsv[block.block_id] = block.tsv_path
                canonical_alids = [canonical_panel_alid(s) for s in block.snp_ids]
                block_canonical_alids[block.block_id] = canonical_alids
                for alid in canonical_alids:
                    if alid is not None and alid not in alid_to_src_idx:
                        new_alids.add(alid)

            if (i + 1) % 1000 == 0:
                print(f"  Scanned {i + 1:,} / {n_analyses:,} analyses, "
                      f"{len(new_alids):,} new LD panel variants, "
                      f"{len(block_to_tsv):,} blocks touched")

        print(f"New LD panel variants: {len(new_alids):,}")
        print(f"LD blocks touched: {len(block_to_tsv):,}")

        # ── Build merged variant table ──────────────────────────────────────
        new_canonical: list[CanonicalVariant] = []
        for alid in new_alids:
            try:
                parts = alid.split(":")
                if len(parts) != 4:
                    continue
                chrom, pos_str, a1, a2 = parts
                cv_result = orient_to_canonical(chrom, int(pos_str), a1, a2)
                if cv_result.variant.alid not in alid_to_src_idx:
                    new_canonical.append(cv_result.variant)
            except (VariantNormalisationError, ValueError):
                continue

        def _sort_key(v: CanonicalVariant) -> tuple[Any, int, str, str]:
            return (chromosome_sort_key(v.chromosome), v.position, v.effect_allele, v.other_allele)

        new_canonical.sort(key=_sort_key)
        all_src = [(*_sort_key(v), v) for v in src_variants]
        all_new = [(*_sort_key(v), v) for v in new_canonical]
        all_variants_sorted = sorted(all_src + all_new, key=lambda x: x[:4])

        merged_variants: list[CanonicalVariant] = [x[4] for x in all_variants_sorted]
        new_alid_to_idx: dict[str, int] = {v.alid: i for i, v in enumerate(merged_variants)}

        src_idx_remap: list[int] = [0] * len(src_alids)
        for v in src_variants:
            src_idx_remap[v.variant_index] = new_alid_to_idx[v.alid]

        print(f"Merged variant table: {len(merged_variants):,} variants")

        print("Writing variants.tsv.gz...")
        # Carry the observed store's rsids across (issue #109). Passing {} here
        # silently blanked the rsid column of every Reference-Completed Ragged
        # store: the eqtlgen-cis pilot had 49,967 rsids in 50,000 observed rows
        # and none at all in its completed sibling. Reference-Completed rows get
        # no rsid -- they are positions the reference panel supplies, which the
        # source never named.
        rsid_by_alid = {v.alid: v.rsid for v in src_variants if v.rsid}
        write_variant_axis(staged.path, merged_variants, rsid_by_alid)

        print("Writing index.sqlite...")
        dst_db = staged.index_connection()
        create_completion_quality_table(dst_db)
        dst_db.commit()

        # ── Phase 2: parallel LD-block completion ───────────────────────────
        print(f"Running reference completion across {len(block_to_tsv):,} LD blocks "
              f"(n_workers={n_workers})...")
        blocks_dir = checkpoint_dir / "blocks"
        blocks_dir.mkdir(parents=True, exist_ok=True)

        pending: list[_BlockTask] = []
        n_existing = 0
        for block_id, tsv_path in block_to_tsv.items():
            ckpt_path = blocks_dir / f"{sanitize_block_id(block_id)}.npz"
            if ckpt_path.exists():
                n_existing += 1
            else:
                pending.append(_BlockTask(
                    tsv_path=tsv_path, source_path=src,
                    analysis_indices=block_to_analyses[block_id],
                    min_cor=min_cor, thresh=thresh, region_cap_bp=region_cap_bp,
                    checkpoint_path=ckpt_path,
                ))

        if pending:
            print(f"  {n_existing:,} blocks already checkpointed, {len(pending):,} remaining")

        if n_workers <= 1:
            for i, task in enumerate(pending):
                _run_block(task)
                if (i + 1) % 200 == 0:
                    print(f"  {i + 1:,} / {len(pending):,} blocks")
        else:
            with ProcessPoolExecutor(
                max_workers=n_workers, initializer=init_block_worker
            ) as pool:
                futures = [pool.submit(_run_block, task) for task in pending]
                for i, fut in enumerate(as_completed(futures)):
                    fut.result()  # propagate worker errors; result is on disk
                    if (i + 1) % 200 == 0:
                        print(f"  {i + 1:,} / {len(pending):,} blocks")

        # ── Phase 3: merge checkpoints, assemble CSR, finalise ──────────────
        print("Merging block results from checkpoints...")
        fills_by_analysis: dict[int, dict[str, tuple[float, float]]] = {}
        quality_batch: list[tuple[int, str, float | None, int, int]] = []
        quality_batch_size = 100_000
        quality_count = 0

        def _flush_quality() -> None:
            nonlocal quality_count
            if quality_batch:
                dst_db.executemany(
                    "INSERT INTO completion_quality "
                    "(analysis_index, block_id, pearson_r, n_imputed, n_missing) "
                    "VALUES (?, ?, ?, ?, ?)",
                    quality_batch,
                )
                dst_db.commit()
                quality_count += len(quality_batch)
                quality_batch.clear()

        for ckpt in sorted(blocks_dir.glob("*.npz")):
            block_result = read_block_checkpoint(ckpt)
            for ai, pearson_r, n_imp, n_miss in block_result.quality_rows:
                quality_batch.append((ai, block_result.block_id, pearson_r, n_imp, n_miss))
                if len(quality_batch) >= quality_batch_size:
                    _flush_quality()
            for alid, ai, z, se in block_result.fills:
                fills_by_analysis.setdefault(ai, {})[alid] = (z, se)
        _flush_quality()
        print(f"Wrote {quality_count:,} completion quality rows")

        print("Assembling completed CSR per analysis...")
        all_vi: list[np.ndarray] = []
        all_z: list[np.ndarray] = []
        all_se: list[np.ndarray] = []
        all_eaf: list[np.ndarray] = []
        all_imp: list[np.ndarray] = []
        offsets: list[int] = [0]
        total_imputed = 0
        total_missing = 0

        for ai in range(n_analyses):
            obs = src_csr.get_analysis(ai)
            obs_vi_new = np.array(
                [src_idx_remap[vi] for vi in obs.variant_index.tolist()], dtype=np.int32
            )
            obs_z = obs.z.astype(np.float32)
            obs_se = obs.se.astype(np.float32)

            block_ids_here = analysis_to_blocks.get(ai, [])
            if not block_ids_here:
                all_vi.append(obs_vi_new)
                all_z.append(obs_z)
                all_se.append(obs_se)
                all_eaf.append(np.asarray(obs.eaf, dtype=np.float32))
                all_imp.append(np.zeros(len(obs_vi_new), dtype=np.uint8))
                offsets.append(offsets[-1] + len(obs_vi_new))
                continue

            obs_alid_to_z: dict[str, float] = {}
            obs_alid_to_se: dict[str, float] = {}
            # Observed EAF carried across the rebuild (ADR 0036). Reference
            # Completion adds panel rows to an Analysis; it does not change
            # what the source reported for the rows it already had.
            obs_alid_to_eaf: dict[str, float] = {}
            for vi_old, z_val, se_val, eaf_val in zip(
                obs.variant_index.tolist(),
                obs.z.tolist(),
                obs.se.tolist(),
                obs.eaf.tolist(),
                strict=True,
            ):
                alid = src_alids[vi_old]
                obs_alid_to_z[alid] = float(z_val)
                obs_alid_to_se[alid] = float(se_val)
                obs_alid_to_eaf[alid] = float(eaf_val)

            unique_ref_alids: list[str] = []
            seen_block_alids: set[str] = set()
            for block_id in block_ids_here:
                for alid in block_canonical_alids[block_id]:
                    if alid is not None and alid not in seen_block_alids:
                        seen_block_alids.add(alid)
                        unique_ref_alids.append(alid)

            fills_here = fills_by_analysis.get(ai, {})

            ref_vi: list[int] = []
            ref_z: list[float] = []
            ref_se: list[float] = []
            ref_eaf: list[float] = []
            ref_imp: list[int] = []
            seen_alids: set[str] = set()
            for alid in unique_ref_alids:
                vi = new_alid_to_idx.get(alid)
                if vi is None:
                    continue
                seen_alids.add(alid)
                if alid in obs_alid_to_z:
                    ref_vi.append(vi)
                    ref_z.append(obs_alid_to_z[alid])
                    ref_se.append(obs_alid_to_se[alid])
                    ref_eaf.append(obs_alid_to_eaf[alid])
                    ref_imp.append(0)
                elif alid in fills_here:
                    z_v, se_v = fills_here[alid]
                    ref_vi.append(vi)
                    ref_z.append(z_v)
                    ref_se.append(se_v)
                    # Imputed cells carry no EAF yet -- the panel's own EAF is
                    # available here but is not written until the completion
                    # checkpoint format carries it (ADR 0036, deferred half).
                    ref_eaf.append(float("nan"))
                    ref_imp.append(1)
                    total_imputed += 1
                else:
                    ref_vi.append(vi)
                    ref_z.append(float("nan"))
                    ref_se.append(float("nan"))
                    ref_eaf.append(float("nan"))
                    ref_imp.append(0)
                    total_missing += 1

            for vi_old, z_val, se_val, eaf_val in zip(
                obs.variant_index.tolist(),
                obs.z.tolist(),
                obs.se.tolist(),
                obs.eaf.tolist(),
                strict=True,
            ):
                alid = src_alids[vi_old]
                if alid not in seen_alids:
                    ref_vi.append(new_alid_to_idx[alid])
                    ref_z.append(float(z_val))
                    ref_se.append(float(se_val))
                    ref_eaf.append(float(eaf_val))
                    ref_imp.append(0)

            order = np.argsort(ref_vi)
            vi_arr = np.array(ref_vi, dtype=np.int32)[order]
            z_arr = np.array(ref_z, dtype=np.float32)[order]
            se_arr = np.array(ref_se, dtype=np.float16)[order]
            eaf_arr = np.array(ref_eaf, dtype=np.float32)[order]
            imp_arr = np.array(ref_imp, dtype=np.uint8)[order]

            all_vi.append(vi_arr)
            all_z.append(z_arr)
            all_se.append(se_arr)
            all_eaf.append(eaf_arr)
            all_imp.append(imp_arr)
            offsets.append(offsets[-1] + len(vi_arr))

            if (ai + 1) % 500 == 0:
                print(f"  {ai + 1:,} / {n_analyses:,} analyses | "
                      f"imputed {total_imputed:,} | missing {total_missing:,}")

        dst_db.commit()
        dst_db.close()

        print(f"Completion done: {total_imputed:,} imputed, {total_missing:,} missing")

        # ── Write zarr CSR with imputed mask ───────────────────────────────
        print("Writing zarr CSR...")
        n_total = offsets[-1]
        offsets_arr = np.array(offsets, dtype=np.int64)
        vi_all = np.concatenate(all_vi) if all_vi else np.empty(0, dtype=np.int32)
        z_all = np.concatenate(all_z) if all_z else np.empty(0, dtype=np.float32)
        se_all = np.concatenate(all_se) if all_se else np.empty(0, dtype=np.float16)
        eaf_all = np.concatenate(all_eaf) if all_eaf else np.empty(0, dtype=np.float32)
        imp_all = np.concatenate(all_imp) if all_imp else np.empty(0, dtype=np.uint8)

        ragged_path = staged.path / RAGGED_ZARR_PATH
        ragged_path.mkdir(parents=True, exist_ok=True)
        root = zarr.open_group(str(ragged_path), mode="w")
        root.create_dataset(
            "offsets", data=offsets_arr, chunks=(_OFFSET_CHUNK,),
            compressor=_COMPRESSOR, dtype=np.int64,
        )
        root.create_dataset(
            "variant_index", data=vi_all, chunks=(_ASSOC_CHUNK,),
            compressor=_COMPRESSOR, dtype=np.int32,
        )
        # Completion writes into the source's arrays, so it encodes with the
        # source's plan (ADR 0038 §4) -- the overflow table travels with the
        # plane it belongs to.
        codec = StoreCodec(manifest.encoding)
        z_overflow = ZOverflowBuilder()
        root.create_dataset(
            "z",
            data=codec.encode_z(z_all, positions=positions_flat(0), overflow=z_overflow),
            chunks=(_ASSOC_CHUNK,), compressor=_COMPRESSOR, dtype=codec.z_dtype,
        )
        z_overflow.table().write(root)
        root.create_dataset(
            "se", data=se_all, chunks=(_ASSOC_CHUNK,), compressor=_COMPRESSOR, dtype=np.float16
        )
        root.create_dataset(
            "imputed", data=imp_all, chunks=(_ASSOC_CHUNK,),
            compressor=_COMPRESSOR, dtype=np.uint8,
        )
        # Only when the observed store had EAF: a completed store must not
        # gain an all-NaN array its source never had (ADR 0036).
        if np.isfinite(eaf_all).any():
            root.create_dataset(
                "eaf", data=eaf_all, chunks=(_ASSOC_CHUNK,),
                compressor=_COMPRESSOR, dtype=np.float32,
            )
        root.attrs["layout"] = "ragged"
        root.attrs["completion_state"] = "reference_completed"
        root.attrs["n_analyses"] = n_analyses
        root.attrs["n_associations"] = n_total

        print("Building top-hit indexes...")
        build_ragged_top_hit_indexes(staged.path, encoding=manifest.encoding)

        print("Writing analyses.tsv...")
        with staged.index_connection() as quality_db:
            quality_rollup = completion_quality_rollup(quality_db, n_analyses)
        dst_analyses = [
            replace(
                a,
                completed_against=(
                    ancestry
                    if impute_analysis_ids is None or a.analysis_id in impute_analysis_ids
                    else ""
                ),
                completion_median_pearson_r=quality_rollup[i].median_pearson_r,
                completion_n_imputed_total=quality_rollup[i].n_imputed_total,
                completion_n_missing_total=quality_rollup[i].n_missing_total,
                # Completion changes z/se via imputation, so the source's
                # pre-completion Top-Hit Counts (carried forward from `a`) do
                # not apply here -- zero them so add_hit_counts below sets
                # fresh post-completion counts rather than adding onto stale
                # ones.
                n_hits_5e8="",
                n_hits_5e6="",
                n_hits_5e4="",
            )
            for i, a in enumerate(src_analyses)
        ]
        write_analysis_records(
            staged.path / "analyses.tsv", add_hit_counts(staged.path, dst_analyses)
        )

        new_release_id = release_id or f"{manifest.release_id}-completed"
        completed_manifest = StoreManifest(
            # Preserved with the format version below: a completed release is
            # written into its source's arrays, so it is in its source's encoding.
            encoding=manifest.encoding,
            store_id=manifest.store_id,
            release_id=new_release_id,
            # Preserved, not re-stamped -- see ADR 0038 §4 and the dense path.
            format_version=source_format_version,
            primary_layout=manifest.primary_layout,
            association_coverage=manifest.association_coverage,
            completion_state=CompletionState.REFERENCE_COMPLETED,
            reference_assembly=manifest.reference_assembly,
            created_at=datetime.now(UTC).isoformat(),
            provenance={
                **manifest.provenance,
                "source_release_id": manifest.release_id,
                "completion": build_completion_provenance(
                    ld_panel_id=ld_panel_id,
                    ancestry=ancestry,
                    min_cor=min_cor,
                    thresh=thresh,
                    n_variants_total=len(merged_variants),
                    n_variants_new=len(new_canonical),
                    cis_window_bp=cis_window_bp,
                    n_imputed=total_imputed,
                    n_missing=total_missing,
                ),
            },
        )
        staged.write_manifest(completed_manifest)

        result = CompletionResult(
            output_path=dst,
            n_variants=len(merged_variants),
            n_analyses=n_analyses,
            n_associations=n_total,
            n_imputed=total_imputed,
            n_missing=total_missing,
        )
        print(
            f"Reference completion complete: {result.n_variants:,} variants, "
            f"{result.n_analyses:,} analyses, {result.n_associations:,} associations "
            f"({result.n_imputed:,} imputed, {result.n_missing:,} missing)"
        )
    return result
