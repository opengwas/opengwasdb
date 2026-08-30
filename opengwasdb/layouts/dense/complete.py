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
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numcodecs import Blosc

from opengwasdb.build.eaf_orientation import panel_a1_eaf
from opengwasdb.completion.ancestry_filter import derive_impute_analysis_ids
from opengwasdb.completion.block import REGION_CAP_BP, run_block
from opengwasdb.completion.checkpoint import (
    ALID_DTYPE,
    BlockCompletionResult,
    checkpoint_dir_for,
    sanitize_block_id,
    write_block_checkpoint,
)
from opengwasdb.completion.ld_panel import (
    canonical_panel_alid as _canonical_panel_alid,
)
from opengwasdb.completion.ld_panel import (
    list_all_blocks,
    list_chromosomes,
)
from opengwasdb.completion.manifest import build_completion_provenance
from opengwasdb.completion.parallel import init_block_worker
from opengwasdb.completion.schema import completion_quality_rollup, create_completion_quality_table
from opengwasdb.encoding import (
    EAF_BASELINE,
    DenseEafPlane,
    DenseZPlane,
    EafExceptionBuilder,
    StoreCodec,
    StoreEncoding,
    ZOverflowBuilder,
    positions_row_band,
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
from opengwasdb.model.analyses import read_analyses, read_analysis_records
from opengwasdb.model.enums import (
    AssociationCoverage,
    CompletionState,
    EafScope,
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
                se_obs[matched_local, :] = src_root["se"].oindex[matched_src, :].astype(np.float64)
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

    if dst.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {dst}. Use overwrite=True.")

    if checkpoint_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"A checkpoint directory already exists at {checkpoint_dir}. "
                "Use resume_dense_completion() to continue it, or overwrite=True to discard it."
            )
        shutil.rmtree(checkpoint_dir)

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
        Path(source_path), dst, Path(ld_dir),
        ancestry=ancestry, min_cor=min_cor, thresh=thresh,
        release_id=release_id, ld_panel_id=ld_panel_id,
        n_workers=n_workers, checkpoint_dir=checkpoint_dir,
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
        Path(params["source_path"]), Path(params["dest_path"]), Path(params["ld_dir"]),
        ancestry=params["ancestry"], min_cor=params["min_cor"], thresh=params["thresh"],
        release_id=params["release_id"], ld_panel_id=params["ld_panel_id"],
        n_workers=n_workers, checkpoint_dir=checkpoint_dir,
        impute_analysis_ids=set(impute_ids) if impute_ids is not None else None,
    )
    shutil.rmtree(checkpoint_dir)
    return result


# ── Shared pipeline core ────────────────────────────────────────────────────


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
    src = Path(source_path)
    dst = Path(dest_path)
    with OpenGWASDBStore.staging(dst, overwrite=True) as staged:
        source = open_store(src)
        manifest = source.manifest
        # Checked here, with the other source preconditions, rather than at
        # manifest-write time: a completion that cannot honour its source's
        # format should fail before it spends an hour imputing (ADR 0038 §4).
        source_format_version = check_writable_format_version(
            manifest.format_version, source=f"source release {src}"
        )
        if manifest.primary_layout is not PrimaryStorageLayout.DENSE:
            raise ValueError(
                f"source store is not Dense (primary_layout={manifest.primary_layout})"
            )
        if manifest.completion_state is not CompletionState.OBSERVED_ONLY:
            raise ValueError(
                f"source store is not Observed-Only (completion_state={manifest.completion_state})"
            )
        if manifest.association_coverage is not AssociationCoverage.FULL:
            raise ValueError(
                "Dense reference completion only supports Full Coverage sources "
                f"(association_coverage={manifest.association_coverage})"
            )
        print(f"Source store: {manifest.store_id} / {manifest.release_id}")

        # ── Phase 1: union variant axis + seeded z/se ───────────────────────
        src_variant_axis = VariantAxis(src)
        src_variants = src_variant_axis.all()
        src_variant_axis.close()
        src_alid_to_idx = {v.alid: v.variant_index for v in src_variants}

        src_analyses = sorted(
            read_analysis_records(src / "analyses.tsv"), key=lambda a: int(a.analysis_index)
        )
        n_analyses = len(src_analyses)
        print(f"Source: {len(src_variants):,} variants, {n_analyses:,} analyses")

        # Per-Analysis ancestry-match filter (ADR 0028): only imputable analyses
        # get fills; the rest are carried through observed-only. None = impute all.
        if impute_analysis_ids is None:
            impute_mask = None
        else:
            impute_mask = np.array(
                [a.analysis_id in impute_analysis_ids for a in src_analyses], dtype=bool
            )
            n_match = int(impute_mask.sum())
            print(f"Ancestry-match filter: imputing {n_match:,}/{n_analyses:,} analyses")

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

        new_canonical: list[CanonicalVariant] = []
        seen_new: set[str] = set()
        for alid in panel_alids:
            if alid in src_alid_to_idx:
                continue
            parts = alid.split(":")
            if len(parts) != 4:
                continue
            chrom, pos_str, a1, a2 = parts
            try:
                cv_result = orient_to_canonical(chrom, int(pos_str), a1, a2)
            except (VariantNormalisationError, ValueError):
                continue
            if cv_result.variant.alid in src_alid_to_idx or cv_result.variant.alid in seen_new:
                continue
            seen_new.add(cv_result.variant.alid)
            new_canonical.append(cv_result.variant)

        merged_variants: list[CanonicalVariant] = [
            CanonicalVariant(v.chromosome, v.position, v.effect_allele, v.other_allele)
            for v in src_variants
        ] + new_canonical
        merged_variants.sort(
            key=lambda v: (
                chromosome_sort_key(v.chromosome), v.position, v.effect_allele, v.other_allele
            )
        )
        new_alid_to_idx: dict[str, int] = {v.alid: i for i, v in enumerate(merged_variants)}
        n_variants = len(merged_variants)
        print(
            f"Union variant axis: {n_variants:,} variants "
            f"({len(new_canonical):,} new panel variants)"
        )

        on_panel = np.zeros(n_variants, dtype=bool)
        for alid in panel_alids:
            idx = new_alid_to_idx.get(alid)
            if idx is not None:
                on_panel[idx] = True

        rsid_by_alid = {v.alid: v.rsid for v in src_variants if v.rsid}
        print("Writing variants.tsv.gz...")
        write_variant_axis(staged.path, merged_variants, rsid_by_alid)

        # Inverse map output_row -> source variant_index (-1 for panel-only rows).
        # z/se are seeded from the source band-by-band during the write (issue 044),
        # so the full source matrix is never loaded.
        src_root = source.arrays(mode="r")
        out_to_src = np.full(n_variants, -1, dtype=np.int64)
        for v in src_variants:
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

        # ── Phase 2: parallel LD-block completion ───────────────────────────
        print(f"Running reference completion across {len(tsv_paths):,} LD blocks "
              f"(n_workers={n_workers})...")
        blocks_dir = checkpoint_dir / "blocks"
        blocks_dir.mkdir(parents=True, exist_ok=True)

        pending: list[_BlockTask] = []
        n_existing = 0
        for tsv_path in tsv_paths:
            block_id = f"{tsv_path.parent.name}/{tsv_path.stem}"
            ckpt_path = blocks_dir / f"{sanitize_block_id(block_id)}.npz"
            if ckpt_path.exists():
                n_existing += 1  # fills read from the checkpoint in Phase 3
            else:
                pending.append(_BlockTask(
                    tsv_path=tsv_path, source_path=src,
                    min_cor=min_cor, thresh=thresh, checkpoint_path=ckpt_path,
                ))

        if pending:
            print(f"  {n_existing:,} blocks already checkpointed, {len(pending):,} remaining")
        # Each block writes its own checkpoint; the parent keeps nothing per block.
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

        # ── Phase 3: stream fills from checkpoints, band-write, finalise ────
        # Resolve checkpoint fill ALIDs to union rows and shard the fill records
        # by output row-band on disk. The final zarr writer then reads only the
        # shard for the band it is currently writing, so Phase 3 never needs a
        # whole-genome fill array in RAM.
        print("Merging block results from checkpoints...")
        union_alids = np.fromiter(
            (alid.encode("ascii") for alid in new_alid_to_idx),
            dtype=ALID_DTYPE,
            count=len(new_alid_to_idx),
        )
        union_rows = np.fromiter(
            new_alid_to_idx.values(), dtype=np.int32, count=len(new_alid_to_idx)
        )
        o = np.argsort(union_alids)
        union_alids_s = union_alids[o]
        union_rows_s = union_rows[o]

        # Reference EAF for imputed cells (ADR 0037 §4). An imputed cell's
        # frequency *is* the panel's -- identical for every Analysis imputed at
        # that variant -- so it is one `float32` per variant rather than
        # per-cell data. Observed cells never fall back to it: FinnGen's
        # frequencies differ from the EUR panel by up to 3000x.
        src_has_eaf = not manifest.encoding.eaf.is_absent
        eaf_reference = (
            _panel_reference_eaf(ld_dir, ancestry, merged_variants) if src_has_eaf else None
        )
        encoding = manifest.encoding.with_eaf_reference(eaf_reference is not None)
        effective_chunks = _create_completed_zarr(
            staged, n_variants, n_analyses, on_panel, DEFAULT_CHUNK_SHAPE, DEFAULT_DTYPE,
            encoding, src_has_eaf=src_has_eaf, eaf_reference=eaf_reference,
        )
        band_rows = _completion_band_rows(effective_chunks)
        fill_shard_dir, quality_count = _shard_checkpoint_fills_by_band(
            blocks_dir, staged, union_alids_s, union_rows_s, n_variants, band_rows
        )
        print(f"Wrote {quality_count:,} completion quality rows")
        del union_alids, union_rows, union_alids_s, union_rows_s, o

        print("Writing data.zarr (band-streamed)...")
        n_missing_off_panel, n_missing_imputation_failed, total_imputed = _write_completed_bands(
            staged, src_root, out_to_src, on_panel,
            fill_shard_dir,
            effective_chunks, n_variants, n_analyses,
            manifest.encoding,
            encoding,
            impute_mask=impute_mask,
        )
        shutil.rmtree(fill_shard_dir, ignore_errors=True)
        n_missing_off_panel_total = int(n_missing_off_panel.sum())
        print(
            f"Completion done: {total_imputed:,} imputed, "
            f"{n_missing_imputation_failed:,} imputation-failed, "
            f"{n_missing_off_panel_total:,} off-panel missing"
        )

        # Write the completed manifest.json before analyses.tsv/overview.html
        # below (rather than after, as this used to) -- overview.html reads
        # manifest.json fresh from output_path for its header (ADR 0032), so
        # it must already reflect the completed release, not the source's.
        new_release_id = release_id or f"{manifest.release_id}-completed"
        completed_manifest = StoreManifest(
            # Same reasoning as `format_version` below: the completed release
            # is in its source's encoding, because it is written into its
            # source's arrays. The one addition is `eaf_reference`, which
            # records a physical fact about *this* release: it carries panel
            # frequencies for the cells it just imputed (ADR 0037 §4).
            encoding=encoding,
            store_id=manifest.store_id,
            release_id=new_release_id,
            # Preserved, not re-stamped: completion writes into the source's
            # arrays and therefore its encoding, so the completed release is the
            # same format as its source (ADR 0038 §4), and was checked writable
            # at the top of this function.
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
                    n_variants_total=n_variants,
                    n_variants_new=len(new_canonical),
                    n_imputed=total_imputed,
                    n_missing_off_panel=n_missing_off_panel_total,
                    n_missing_imputation_failed=n_missing_imputation_failed,
                ),
            },
        )
        staged.write_manifest(completed_manifest)

        print("Writing analyses.tsv...")
        with staged.index_connection() as dst_db:
            quality_rollup = completion_quality_rollup(dst_db, n_analyses)
        dst_analyses = [
            replace(
                a,
                completed_against=ancestry if impute_mask is None or impute_mask[i] else "",
                # An Analysis that gained imputed cells in a release carrying
                # reference EAF now stores a frequency for them, whatever its
                # source reported (ADR 0037 §4). `eaf_scope` is derived from
                # what the release actually holds, not copied forward -- the
                # declaration disagreeing with the arrays is the defect that
                # got through review on #106.
                eaf_scope=_completed_eaf_scope(a, quality_rollup[i], eaf_reference is not None),
                completion_median_pearson_r=quality_rollup[i].median_pearson_r,
                completion_n_imputed_total=quality_rollup[i].n_imputed_total,
                completion_n_missing_total=str(int(n_missing_off_panel[i])),
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

        print("Building top-hit indexes...")
        build_top_hit_indexes(staged.path, encoding=manifest.encoding)
        write_analyses_tsv(staged.path, add_hit_counts(staged.path, dst_analyses))

        result = CompletionResult(
            output_path=dst,
            n_variants=n_variants,
            n_analyses=n_analyses,
            n_imputed=total_imputed,
            n_missing_off_panel=n_missing_off_panel_total,
            n_missing_imputation_failed=n_missing_imputation_failed,
        )
        print(
            f"Reference completion complete: {result.n_variants:,} variants, "
            f"{result.n_analyses:,} analyses ({result.n_imputed:,} imputed, "
            f"{result.n_missing_off_panel:,} off-panel missing, "
            f"{result.n_missing_imputation_failed:,} imputation-failed)"
        )
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
            records = np.fromfile(
                fh, dtype=_FILL_RECORD_DTYPE, count=_FILL_RECORD_READ_COUNT
            )
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
) -> tuple[Path, int]:
    """Resolve checkpoint fills into raw row-band shard files.

    Checkpoints store fill rows by ALID because the union variant axis is built
    by the parent. This pass resolves those ALIDs once, writes compact
    ``(row, analysis, z, se)`` records to per-band files, and streams
    completion_quality directly into SQLite.
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

                idx = np.minimum(
                    np.searchsorted(union_alids_s, f_alid), len(union_alids_s) - 1
                )
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


def _completed_eaf_scope(analysis: Any, rollup: Any, carries_reference: bool) -> str:
    """`eaf_scope` for a completed Analysis, from what the release now holds."""
    if carries_reference and int(rollup.n_imputed_total or 0) > 0:
        return str(EafScope.ASSOCIATION.value)
    return str(analysis.eaf_scope)


def _panel_reference_eaf(
    ld_dir: str | Path, ancestry: str, variants: Sequence[CanonicalVariant]
) -> np.ndarray | None:
    """One A1-oriented panel frequency per variant of the completed axis.

    `None` when the panel declares none, so a release only claims to carry
    reference EAF when it has some to carry. Variants the panel does not hold
    -- every off-panel row the source contributed -- are NaN, which is correct:
    they have no imputed cells to describe.
    """
    panel = panel_a1_eaf(ld_dir, ancestry)
    if not panel:
        return None
    reference = np.fromiter(
        (panel.get(v.alid, np.nan) for v in variants), dtype=np.float32, count=len(variants)
    )
    if not np.any(np.isfinite(reference)):
        return None
    print(
        f"Reference EAF: {int(np.count_nonzero(np.isfinite(reference))):,} of "
        f"{len(reference):,} variants carry a panel frequency"
    )
    return reference


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
    for name, plane_dtype, fill in (
        ("z", codec.z_dtype, codec.z_fill_value),
        ("se", dtype, float("nan")),
    ):
        root.create_dataset(
            name, shape=(n_variants, n_analyses), chunks=effective_chunks,
            compressor=_COMPRESSOR, dtype=plane_dtype, fill_value=fill,
        )
    root.create_dataset(
        "imputed", shape=(n_variants, n_analyses), chunks=effective_chunks,
        compressor=_COMPRESSOR, dtype="uint8", fill_value=0,
    )
    root.create_dataset(
        "on_panel", data=on_panel.astype(np.uint8),
        chunks=(effective_chunks[0],), compressor=_COMPRESSOR, dtype="uint8",
    )
    if src_has_eaf:
        # Never float16 -- see `build_vcf._create_eaf_array` for why it cannot
        # hold an EAF near 1 (ADR 0036). Created only when the observed store
        # had one: completion adds panel rows, it does not invent frequencies
        # the source never reported.
        root.create_dataset(
            "eaf", shape=(n_variants, n_analyses), chunks=effective_chunks,
            compressor=_COMPRESSOR, dtype=codec.eaf_dtype, fill_value=codec.eaf_fill_value,
        )
    if eaf_reference is not None:
        write_eaf_reference(root, eaf_reference, compressor=_COMPRESSOR)
    root.attrs["layout"] = "dense"
    root.attrs["completion_state"] = "reference_completed"
    root.attrs["compressor"] = DEFAULT_COMPRESSOR
    root.attrs["chunk_shape"] = list(effective_chunks)
    return effective_chunks


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
    """Seed z/se from the source, apply the imputed fills, and write z/se/imputed
    one row-band at a time. ``z`` and ``se`` are written in two passes over a
    **single reused** float32 band buffer (plus a uint8 imputed band in the z-pass),
    so peak memory is ~one band rather than z + se + imputed held together. Both
    passes fill the same cells because source z/se missingness is consistent (a
    validated store invariant) — each pass reads only its own source array once.
    Returns ``(n_missing_off_panel[n_analyses], n_missing_imputation_failed,
    total_imputed)``. Fill records are read from per-band shard files in bounded
    chunks.

    ``impute_mask`` (bool per analysis; ``None`` = impute all) is the per-Analysis
    ancestry-match filter (ADR 0028): fills for a masked-out analysis are dropped,
    so its cells stay observed-only (NaN, ``imputed=0``) — never imputed against a
    non-matching-ancestry panel.
    """
    root = staged.arrays(mode="a")
    z_arr, se_arr, imp_arr = root["z"], root["se"], root["imputed"]
    src_se = src_root["se"]
    # Source z is read decoded and written re-encoded, through the same plan --
    # completion moves values between two planes, it does not reinterpret them.
    src_plane = DenseZPlane.open(src_root, source_encoding)
    codec = StoreCodec(encoding)
    overflow = ZOverflowBuilder()
    band_rows = _completion_band_rows(effective_chunks)

    n_missing_off_panel = np.zeros(n_analyses, dtype=np.int64)
    n_missing_imputation_failed = 0
    total_imputed = 0
    band = np.empty((band_rows, n_analyses), dtype=np.float32)  # reused for z then se

    # Pass 1 — z + imputed mask + missingness counts.
    for band_index, r0 in enumerate(range(0, n_variants, band_rows)):
        r1 = min(r0 + band_rows, n_variants)
        br = r1 - r0
        zb = band[:br]
        zb[:] = np.nan
        imp_band = np.zeros((br, n_analyses), dtype=np.uint8)

        valid = np.where(out_to_src[r0:r1] >= 0)[0]
        if len(valid):
            srows = out_to_src[r0:r1][valid]
            zb[valid, :] = src_plane.rows(srows)

        off_local = np.where(on_panel[r0:r1] == 0)[0]
        if len(off_local):
            n_missing_off_panel += np.isnan(zb[off_local, :]).sum(axis=0).astype(np.int64)

        shard_path = _fill_shard_path(fill_shard_dir, band_index)
        for records in _iter_fill_records(shard_path):
            lr = records["row"] - r0
            ai = records["ai"]
            fillable = ~np.isfinite(zb[lr, ai])
            if impute_mask is not None:
                fillable &= impute_mask[ai]
            if fillable.any():
                lrm, aim = lr[fillable], ai[fillable]
                zb[lrm, aim] = records["z"][fillable]
                imp_band[lrm, aim] = 1
                total_imputed += int(fillable.sum())

        on_local = np.where(on_panel[r0:r1] == 1)[0]
        if len(on_local):
            n_missing_imputation_failed += int(np.isnan(zb[on_local, :]).sum())

        z_arr[r0:r1] = codec.encode_z(
            zb, positions=positions_row_band(r0, n_analyses), overflow=overflow
        )
        imp_arr[r0:r1] = imp_band

    overflow.table().write(root)

    # Pass 2 — se (same cells filled, by the missingness-consistency invariant).
    for band_index, r0 in enumerate(range(0, n_variants, band_rows)):
        r1 = min(r0 + band_rows, n_variants)
        br = r1 - r0
        sb = band[:br]
        sb[:] = np.nan

        valid = np.where(out_to_src[r0:r1] >= 0)[0]
        if len(valid):
            srows = out_to_src[r0:r1][valid]
            sb[valid, :] = src_se.oindex[srows, :].astype(np.float32)

        shard_path = _fill_shard_path(fill_shard_dir, band_index)
        for records in _iter_fill_records(shard_path):
            lr = records["row"] - r0
            ai = records["ai"]
            fillable = ~np.isfinite(sb[lr, ai])
            if impute_mask is not None:
                fillable &= impute_mask[ai]
            if fillable.any():
                sb[lr[fillable], ai[fillable]] = records["se"][fillable]

        se_arr[r0:r1] = sb

    # Pass 3 -- eaf. Observed frequencies are carried across the row remap and
    # nothing else: an imputed cell's frequency is the panel's, stored once per
    # variant in `eaf_reference` and applied on read (ADR 0037 §4), and an
    # observed cell whose source reported none stays absent. The per-variant
    # baseline travels with the values rather than being recomputed, so a cell
    # decoded from the source and re-encoded here lands on the same code --
    # completion moves values between two planes, it does not requantise them.
    if "eaf" in root and "eaf" in src_root:
        src_eaf_plane = DenseEafPlane.open(src_root, source_encoding)
        eaf_codec = StoreCodec(encoding)
        exceptions = EafExceptionBuilder()
        src_baseline = (
            np.asarray(src_root[EAF_BASELINE][:], dtype=np.float32)
            if EAF_BASELINE in src_root
            else None
        )
        out_baseline = (
            np.full(n_variants, np.nan, dtype=np.float32) if src_baseline is not None else None
        )
        if out_baseline is not None and src_baseline is not None:
            carried = out_to_src >= 0
            out_baseline[carried] = src_baseline[out_to_src[carried]]
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

    return n_missing_off_panel, n_missing_imputation_failed, total_imputed
