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
import sqlite3
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
    require_fresh_destination,
    sanitize_block_id,
    write_block_checkpoint,
)
from opengwasdb.completion.ld_panel import (
    LDBlock,
    blocks_over_variants,
    canonical_panel_alid,
    check_panel_has_chromosomes,
    find_blocks,
)
from opengwasdb.completion.manifest import build_completion_provenance
from opengwasdb.completion.parallel import run_block_tasks
from opengwasdb.completion.reference_eaf import completed_eaf_scope, panel_reference_eaf
from opengwasdb.completion.schema import completion_quality_rollup, create_completion_quality_table
from opengwasdb.encoding import (
    EAF_BASELINE,
    RaggedEafPlane,
    StoreCodec,
    StoreEncoding,
    ZOverflowBuilder,
    fit_se,
    positions_flat,
    write_eaf_csr,
    write_eaf_reference,
    write_se_csr,
)
from opengwasdb.layouts.dense.build import add_hit_counts
from opengwasdb.layouts.ragged.top_hits import build_ragged_top_hit_indexes
from opengwasdb.layouts.ragged.zarr_csr import RAGGED_ZARR_PATH, RaggedCSRReader
from opengwasdb.model.analyses import (
    Analysis,
    read_analysis_records,
    reset_top_hit_counts,
    write_analysis_records,
)
from opengwasdb.model.enums import CompletionState
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.store.open import (
    OpenGWASDBStore,
    StagedRelease,
    check_writable_format_version,
    open_store,
)
from opengwasdb.variants.axis import (
    VariantAxis,
    VariantRecord,
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


@dataclass(frozen=True)
class _ObservedAlidMaps:
    """One Analysis's observed z/se/eaf, each keyed by canonical ALID.

    A named record rather than a positional three-dict tuple because the fold
    feeds both the block reader and Phase 3's assembly (issue #130): a
    statistic placed in the wrong slot travels to both without raising -- an
    observed SE in the EAF slot is still a *valid* frequency -- and would
    re-encode a silent wrong answer into the completed store. Field access
    makes the statistic a caller reads explicit.
    """

    z_by_alid: dict[str, float]
    se_by_alid: dict[str, float]
    eaf_by_alid: dict[str, float]


def _observed_alid_maps(obs: Any, src_alids: list[str]) -> _ObservedAlidMaps:
    """Map one Analysis's observed CSR rows onto ``{alid: z/se/eaf}``. Observed
    EAF is carried across the rebuild (ADR 0036): Reference Completion adds
    panel rows to an Analysis; it does not change what the source reported for
    the rows it already had. Both the block reader and Phase 3's assembly fold
    the same observed rows this way."""
    obs_alid_to_z: dict[str, float] = {}
    obs_alid_to_se: dict[str, float] = {}
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
    return _ObservedAlidMaps(
        z_by_alid=obs_alid_to_z,
        se_by_alid=obs_alid_to_se,
        eaf_by_alid=obs_alid_to_eaf,
    )


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
            observed = _observed_alid_maps(obs, src_alids)

            z_dense = np.array(
                [
                    observed.z_by_alid.get(a, float("nan")) if a is not None else float("nan")
                    for a in canonical_alids
                ],
                dtype=np.float64,
            )
            se_dense = np.array(
                [
                    observed.se_by_alid.get(a, float("nan")) if a is not None else float("nan")
                    for a in canonical_alids
                ],
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
    require_fresh_destination(dst, checkpoint_dir, overwrite, "resume_ragged_completion")
    check_panel_has_chromosomes(ld_dir, ancestry)

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
        Path(source_path),
        dst,
        Path(ld_dir),
        ancestry=ancestry,
        cis_window_bp=cis_window_bp,
        min_cor=min_cor,
        thresh=thresh,
        release_id=release_id,
        ld_panel_id=ld_panel_id,
        n_workers=n_workers,
        checkpoint_dir=checkpoint_dir,
        impute_analysis_ids=impute_analysis_ids,
        region_cap_bp=region_cap_bp,
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
        Path(params["source_path"]),
        Path(params["dest_path"]),
        Path(params["ld_dir"]),
        ancestry=params["ancestry"],
        cis_window_bp=params["cis_window_bp"],
        min_cor=params["min_cor"],
        thresh=params["thresh"],
        release_id=params["release_id"],
        ld_panel_id=params["ld_panel_id"],
        n_workers=n_workers,
        checkpoint_dir=checkpoint_dir,
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

# The completion pipeline is one sequence of phases, each a module-level
# function that takes in what the phase before produced and hands on only what
# the phase after needs. `_run_completion` sequences the phases inside the
# staging context; a phase never reaches past its seam -- block discovery
# knows nothing of encodings, the CSR assembly nothing of checkpoint files it
# did not itself merge -- so the resume, counting, EAF and fail-loud rules
# stay next to the code that owns them rather than in one ~550-line core.


@dataclass
class _SourceState:
    """Everything read out of the observed-only source before any stage write:
    the variant axis, Analytical Metadata (ADR 0034) and the per-Analysis
    ancestry-match impute mask (ADR 0028)."""

    src: Path
    src_variants: list[VariantRecord]
    src_alids: list[str]
    src_analyses: list[Analysis]
    n_analyses: int
    impute_mask: np.ndarray | None


@dataclass
class _BlockPlan:
    """Phase 1's LD-block discovery (ADR 0023, issue #102): every block each
    Analysis is completed against, its tsv path and canonical panel ALIDs, and
    the panel ALIDs the source never held. Fills stay on disk in checkpoints;
    this is everything Phase 2 runs and Phase 3 resolves them against."""

    src_csr: RaggedCSRReader
    state: _SourceState
    analysis_to_blocks: dict[int, list[str]]
    block_to_tsv: dict[str, Path]
    block_to_analyses: dict[str, list[int]]
    block_canonical_alids: dict[str, list[str | None]]
    new_alids: set[str]


@dataclass
class _MergedAxis:
    """The union variant axis in canonical order, plus the maps that translate
    source variant indices and new panel ALIDs into it."""

    variants: list[CanonicalVariant]
    new_variants: list[CanonicalVariant]
    new_alid_to_idx: dict[str, int]
    src_idx_remap: list[int]


@dataclass
class _CsrCell:
    """One cell of an Analysis's completed CSR: index into the merged axis,
    its z/se/eaf values and the imputed flag. eaf is NaN on an imputed cell
    (deferred to `eaf_reference`, ADR 0037 §4) and on a missing one."""

    vi: int
    z: float
    se: float
    eaf: float
    imp: int


@dataclass
class _AnalysisCompleted:
    """One Analysis's rows in the completed CSR, in variant-index order."""

    vi: np.ndarray
    z: np.ndarray
    se: np.ndarray
    eaf: np.ndarray
    imp: np.ndarray
    n_imputed: int
    n_missing: int


@dataclass
class _CompletedCsr:
    """Phase 3's assembly output: per-Analysis arrays with the flat offsets,
    and the imputed/missing totals the later phases report and store."""

    offsets: list[int]
    vi: list[np.ndarray]
    z: list[np.ndarray]
    se: list[np.ndarray]
    eaf: list[np.ndarray]
    imp: list[np.ndarray]
    total_imputed: int
    total_missing: int


@dataclass
class _EncodePlan:
    """The completed release's encoding (ADR 0038 §4) and the per-variant EAF
    data it adds: the panel reference for imputed cells and the carried
    baseline for observed ones (ADR 0037 §2/§4)."""

    encoding: StoreEncoding
    codec: StoreCodec
    eaf_reference: np.ndarray | None
    out_baseline: np.ndarray | None


@dataclass
class _FlatCsr:
    """The assembled CSR concatenated into the flat arrays zarr stores."""

    offsets: np.ndarray
    vi: np.ndarray
    z: np.ndarray
    se: np.ndarray
    eaf: np.ndarray
    imp: np.ndarray


def _open_source(src: Path) -> tuple[StoreManifest, str]:
    """Open the observed-only source and refuse a format this build cannot
    write, before any of the work (ADR 0038 §4)."""
    source = open_store(src)
    manifest = source.manifest
    # See the dense path: refused before the work, not after it (ADR 0038 §4).
    source_format_version = check_writable_format_version(
        manifest.format_version, source=f"source release {src}"
    )
    print(f"Source store: {manifest.store_id} / {manifest.release_id}")
    print(f"  completion_state: {manifest.completion_state}")
    return manifest, source_format_version


def _read_source_state(
    src: Path, impute_analysis_ids: set[str] | None
) -> _SourceState:
    """Phase 1 read: the source's variant axis and Analytical Metadata (ADR
    0034), and the per-Analysis ancestry-match impute mask derived from it."""
    src_variant_axis = VariantAxis(src)
    src_variants = src_variant_axis.all()
    src_variant_axis.close()
    src_alids = [v.alid for v in src_variants]

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

    return _SourceState(
        src=src,
        src_variants=src_variants,
        src_alids=src_alids,
        src_analyses=src_analyses,
        n_analyses=n_analyses,
        impute_mask=impute_mask,
    )


def _has_trait_position(a: Analysis) -> bool:
    return bool(a.trait_chr) and bool(a.trait_bp)


def _imputable_analyses(state: _SourceState) -> list[tuple[int, Analysis]]:
    """Every Analysis, or -- under the ancestry-match filter (ADR 0028) --
    only those the filter admits. The others are left observed-only: Phase 3's
    pass-through path carries them through unchanged, so they never enter the
    block maps at all."""
    if state.impute_mask is None:
        return list(enumerate(state.src_analyses))
    return [
        (i, a) for i, a in enumerate(state.src_analyses) if bool(state.impute_mask[i])
    ]


def _gene_target_less_blocks(
    state: _SourceState, src_csr: RaggedCSRReader, ld_dir: Path, ancestry: str
) -> dict[int, list[LDBlock]]:
    """LD blocks for Analyses with no Trait position (issue #102): every block
    holding an association they already have, resolved in one ALID-keyed pass
    so the panel is read once rather than once per Analysis."""
    gene_target_less = [
        (i, a)
        for i, a in _imputable_analyses(state)
        if not _has_trait_position(a)
    ]
    if not gene_target_less:
        return {}
    analyses_by_alid: dict[str, list[int]] = {}
    chromosomes: set[str] = set()
    for i, _a in gene_target_less:
        for vi in src_csr.variant_indices(i).tolist():
            analyses_by_alid.setdefault(state.src_alids[vi], []).append(i)
            chromosomes.add(state.src_variants[vi].chromosome)
    print(
        f"  {len(gene_target_less):,} analyses have no Trait position; scanning "
        f"{len(chromosomes)} chromosome(s) of panel blocks for regions they hold"
    )
    return blocks_over_variants(ld_dir, ancestry, analyses_by_alid, sorted(chromosomes))


def _cis_window_blocks(
    a: Analysis, ld_dir: Path, ancestry: str, cis_window_bp: int
) -> list[LDBlock]:
    """The LD blocks touching one Analysis's cis window around its Trait
    position."""
    start = max(1, int(a.trait_bp) - cis_window_bp)
    end = int(a.trait_bp) + cis_window_bp
    return find_blocks(ld_dir, ancestry, a.trait_chr, start, end)


def _blocks_for_analysis(
    i: int,
    a: Analysis,
    own_variant_blocks: dict[int, list[LDBlock]],
    ld_dir: Path,
    ancestry: str,
    cis_window_bp: int,
) -> list[LDBlock]:
    """An Analysis's completable blocks: its cis window where it has a Trait
    position, the blocks it already holds enough observations in otherwise
    (issue #102)."""
    if _has_trait_position(a):
        return _cis_window_blocks(a, ld_dir, ancestry, cis_window_bp)
    return own_variant_blocks.get(i, [])


def _canonical_alids(block: LDBlock) -> list[str | None]:
    return [canonical_panel_alid(s) for s in block.snp_ids]


def _new_panel_alids(alids: list[str | None], source_alids: set[str]) -> set[str]:
    """The panel ALIDs among `alids` the source release never held."""
    return {a for a in alids if a is not None and a not in source_alids}


def _scan_block_maps(
    state: _SourceState,
    src_csr: RaggedCSRReader,
    own_variant_blocks: dict[int, list[LDBlock]],
    ld_dir: Path,
    ancestry: str,
    cis_window_bp: int,
) -> _BlockPlan:
    """Enumerate every imputable Analysis's touching LD blocks and build the
    `_BlockPlan` Phase 2 runs and Phase 3 resolves: the block -> Analysis
    map, each block's tsv path and canonical panel ALIDs, and the union of
    panel ALIDs the source never held."""
    source_alids = set(state.src_alids)
    new_alids: set[str] = set()
    block_to_tsv: dict[str, Path] = {}
    block_to_analyses: dict[str, list[int]] = {}
    block_canonical_alids: dict[str, list[str | None]] = {}
    analysis_to_blocks: dict[int, list[str]] = {}

    for i, a in _imputable_analyses(state):
        blocks = _blocks_for_analysis(i, a, own_variant_blocks, ld_dir, ancestry, cis_window_bp)
        analysis_to_blocks[i] = [b.block_id for b in blocks]
        for block in blocks:
            block_to_analyses.setdefault(block.block_id, []).append(i)
            if block.block_id in block_to_tsv:
                continue
            block_to_tsv[block.block_id] = block.tsv_path
            block_canonical_alids[block.block_id] = _canonical_alids(block)
            new_alids.update(_new_panel_alids(block_canonical_alids[block.block_id], source_alids))
        if (i + 1) % 1000 == 0:
            print(
                f"  Scanned {i + 1:,} / {state.n_analyses:,} analyses, "
                f"{len(new_alids):,} new LD panel variants, "
                f"{len(block_to_tsv):,} blocks touched"
            )

    return _BlockPlan(
        src_csr=src_csr,
        state=state,
        analysis_to_blocks=analysis_to_blocks,
        block_to_tsv=block_to_tsv,
        block_to_analyses=block_to_analyses,
        block_canonical_alids=block_canonical_alids,
        new_alids=new_alids,
    )


def _plan_blocks(
    state: _SourceState, ld_dir: Path, ancestry: str, cis_window_bp: int
) -> _BlockPlan:
    """Phase 1: scan every imputable Analysis's cis window (its own variants
    when it has no Trait position, issue #102) for touching LD blocks and
    return the plan Phase 2 runs and Phase 3 resolves."""
    src_csr = RaggedCSRReader(state.src)
    own_variant_blocks = _gene_target_less_blocks(state, src_csr, ld_dir, ancestry)
    print(
        "Scanning for LD blocks + new panel variants (cis windows where a "
        "Trait position exists, the Analysis's own variants otherwise)..."
    )
    plan = _scan_block_maps(state, src_csr, own_variant_blocks, ld_dir, ancestry, cis_window_bp)
    print(f"New LD panel variants: {len(plan.new_alids):,}")
    print(f"LD blocks touched: {len(plan.block_to_tsv):,}")
    return plan


def _sort_key(v: CanonicalVariant) -> tuple[Any, int, str, str]:
    return (chromosome_sort_key(v.chromosome), v.position, v.effect_allele, v.other_allele)


def _as_canonical(v: VariantRecord) -> CanonicalVariant:
    """A source axis row as a CanonicalVariant: the four fields the merged
    table is sorted and written by (a store axis is already canonical)."""
    return CanonicalVariant(
        chromosome=v.chromosome,
        position=v.position,
        effect_allele=v.effect_allele,
        other_allele=v.other_allele,
    )


def _merged_axis(state: _SourceState, new_alids: set[str]) -> _MergedAxis:
    """Merge the new panel ALIDs into the source axis: canonicalise, drop any
    that collide with source variants, sort under the store's key, and build
    the maps that translate source variant indices and panel ALIDs into the
    union."""
    source_alids = set(state.src_alids)
    new_variants: list[CanonicalVariant] = []
    for alid in new_alids:
        try:
            parts = alid.split(":")
            if len(parts) != 4:
                continue
            chrom, pos_str, a1, a2 = parts
            cv_result = orient_to_canonical(chrom, int(pos_str), a1, a2)
            if cv_result.variant.alid not in source_alids:
                new_variants.append(cv_result.variant)
        except (VariantNormalisationError, ValueError):
            continue
    new_variants.sort(key=_sort_key)
    variants = sorted(
        [_as_canonical(v) for v in state.src_variants] + new_variants, key=_sort_key
    )
    print(f"Merged variant table: {len(variants):,} variants")

    new_alid_to_idx = {v.alid: i for i, v in enumerate(variants)}
    src_idx_remap = [0] * len(state.src_alids)
    for v in state.src_variants:
        src_idx_remap[v.variant_index] = new_alid_to_idx[v.alid]
    return _MergedAxis(
        variants=variants,
        new_variants=new_variants,
        new_alid_to_idx=new_alid_to_idx,
        src_idx_remap=src_idx_remap,
    )


def _write_variant_axis(store_path: Path, state: _SourceState, axis: _MergedAxis) -> None:
    """Write variants.tsv.gz for the merged axis, carrying the observed
    release's rsids across (issue #109)."""
    print("Writing variants.tsv.gz...")
    # Carry the observed store's rsids across (issue #109). Passing {} here
    # silently blanked the rsid column of every Reference-Completed Ragged
    # store: the eqtlgen-cis pilot had 49,967 rsids in 50,000 observed rows
    # and none at all in its completed sibling. Reference-Completed rows get
    # no rsid -- they are positions the reference panel supplies, which the
    # source never named.
    rsid_by_alid = {v.alid: v.rsid for v in state.src_variants if v.rsid}
    write_variant_axis(store_path, axis.variants, rsid_by_alid)


def _run_pending_blocks(
    plan: _BlockPlan,
    *,
    checkpoint_dir: Path,
    n_workers: int,
    min_cor: float,
    thresh: float,
    region_cap_bp: int | None,
) -> None:
    """Phase 2: complete the LD blocks whose checkpoints do not yet exist,
    across the process pool. Each block writes its own checkpoint; the parent
    keeps nothing per block, so an interrupted run resumes from disk (ADR
    0023, issue 044)."""
    print(
        f"Running reference completion across {len(plan.block_to_tsv):,} LD blocks "
        f"(n_workers={n_workers})..."
    )
    blocks_dir = checkpoint_dir / "blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)

    pending: list[_BlockTask] = []
    n_existing = 0
    for block_id, tsv_path in plan.block_to_tsv.items():
        ckpt_path = blocks_dir / f"{sanitize_block_id(block_id)}.npz"
        if ckpt_path.exists():
            n_existing += 1
        else:
            pending.append(
                _BlockTask(
                    tsv_path=tsv_path,
                    source_path=plan.state.src,
                    analysis_indices=plan.block_to_analyses[block_id],
                    min_cor=min_cor,
                    thresh=thresh,
                    region_cap_bp=region_cap_bp,
                    checkpoint_path=ckpt_path,
                )
            )

    if pending:
        print(f"  {n_existing:,} blocks already checkpointed, {len(pending):,} remaining")

    run_block_tasks(pending, n_workers, _run_block)


def _merge_checkpoint_rows(
    blocks_dir: Path, dst_db: sqlite3.Connection
) -> tuple[dict[int, dict[str, tuple[float, float]]], int]:
    """Phase 3 merge: read every block checkpoint -- written this run or
    resumed from disk -- into `completion_quality` rows (batched) and the
    per-Analysis fill maps the CSR assembly layers onto the observed rows."""
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
    return fills_by_analysis, quality_count


def _checkpoint_and_assemble(
    staged: StagedRelease,
    plan: _BlockPlan,
    axis: _MergedAxis,
    *,
    checkpoint_dir: Path,
    n_workers: int,
    min_cor: float,
    thresh: float,
    region_cap_bp: int | None,
) -> _CompletedCsr:
    """Phases 2-3 together: run the pending LD blocks, merge their checkpoints
    into `completion_quality` and assemble the completed CSR. Owns the
    index.sqlite connection for the whole stretch -- ADR 0030 keeps the
    fine-grained completion_quality table SQLite-only, so it is created here
    and committed when the CSR assembly that reads it is done."""
    print("Writing index.sqlite...")
    dst_db = staged.index_connection()
    create_completion_quality_table(dst_db)
    dst_db.commit()

    _run_pending_blocks(
        plan,
        checkpoint_dir=checkpoint_dir,
        n_workers=n_workers,
        min_cor=min_cor,
        thresh=thresh,
        region_cap_bp=region_cap_bp,
    )
    fills_by_analysis, _ = _merge_checkpoint_rows(checkpoint_dir / "blocks", dst_db)
    csr = _assemble_completed_csr(plan, axis, fills_by_analysis)
    dst_db.commit()
    dst_db.close()
    print(f"Completion done: {csr.total_imputed:,} imputed, {csr.total_missing:,} missing")
    return csr


def _reference_alids(
    block_ids: list[str], block_canonical_alids: dict[str, list[str | None]]
) -> list[str]:
    """The block positions one Analysis is completed against, deduplicated in
    block order -- the rows the assembly classifies as observed, imputed or
    missing."""
    unique_ref_alids: list[str] = []
    seen_block_alids: set[str] = set()
    for block_id in block_ids:
        for alid in block_canonical_alids[block_id]:
            if alid is not None and alid not in seen_block_alids:
                seen_block_alids.add(alid)
                unique_ref_alids.append(alid)
    return unique_ref_alids


def _passthrough_analysis(obs: Any, src_idx_remap: list[int]) -> _AnalysisCompleted:
    """An Analysis no completable block touches: its source association list
    remapped onto the merged axis, every row marked observed."""
    vi = np.array([src_idx_remap[v] for v in obs.variant_index.tolist()], dtype=np.int32)
    return _AnalysisCompleted(
        vi=vi,
        z=obs.z.astype(np.float32),
        se=obs.se.astype(np.float32),
        eaf=np.asarray(obs.eaf, dtype=np.float32),
        imp=np.zeros(len(vi), dtype=np.uint8),
        n_imputed=0,
        n_missing=0,
    )


def _block_cells(
    block_ids: list[str],
    block_canonical_alids: dict[str, list[str | None]],
    new_alid_to_idx: dict[str, int],
    obs_alid_to_z: dict[str, float],
    obs_alid_to_se: dict[str, float],
    obs_alid_to_eaf: dict[str, float],
    fills: dict[str, tuple[float, float]],
) -> tuple[list[_CsrCell], set[str], int, int]:
    """One Analysis's block-position rows: observed cells carried across
    untouched (ADR 0036), fills as imputed cells whose frequency is deferred
    to `eaf_reference` (ADR 0037 §4), and block positions neither produced as
    missing (NaN) rows. Returns the rows, the ALIDs they consumed (so
    off-window observed rows are not appended twice), and the imputed/missing
    counts."""
    cells: list[_CsrCell] = []
    seen_alids: set[str] = set()
    n_imputed = 0
    n_missing = 0
    for alid in _reference_alids(block_ids, block_canonical_alids):
        vi = new_alid_to_idx.get(alid)
        if vi is None:
            continue
        seen_alids.add(alid)
        if alid in obs_alid_to_z:
            cells.append(
                _CsrCell(vi, obs_alid_to_z[alid], obs_alid_to_se[alid], obs_alid_to_eaf[alid], 0)
            )
        elif alid in fills:
            z_v, se_v = fills[alid]
            # Imputed cells carry no EAF yet -- the panel's own EAF is
            # available here but is not written until the completion
            # checkpoint format carries it (ADR 0036, deferred half).
            cells.append(_CsrCell(vi, z_v, se_v, float("nan"), 1))
            n_imputed += 1
        else:
            cells.append(_CsrCell(vi, float("nan"), float("nan"), float("nan"), 0))
            n_missing += 1
    return cells, seen_alids, n_imputed, n_missing


def _carried_cells(
    obs_alid_to_z: dict[str, float],
    obs_alid_to_se: dict[str, float],
    obs_alid_to_eaf: dict[str, float],
    seen_alids: set[str],
    new_alid_to_idx: dict[str, int],
) -> list[_CsrCell]:
    """Observed rows the block scan never produced (off-window variants the
    source already holds), carried through from the maps above."""
    cells: list[_CsrCell] = []
    for alid, z_val in obs_alid_to_z.items():
        if alid not in seen_alids:
            cells.append(
                _CsrCell(
                    new_alid_to_idx[alid], z_val, obs_alid_to_se[alid], obs_alid_to_eaf[alid], 0
                )
            )
    return cells


def _sort_cells(cells: list[_CsrCell], n_imputed: int, n_missing: int) -> _AnalysisCompleted:
    """Rows to arrays, ordered by variant index (a variant appears once per
    Analysis, so the sort is a permutation)."""
    rows = np.array(
        [(c.vi, c.z, c.se, c.eaf, c.imp) for c in cells],
        dtype=[("vi", "i4"), ("z", "f4"), ("se", "f4"), ("eaf", "f4"), ("imp", "u1")],
    )
    rows.sort(order="vi")
    return _AnalysisCompleted(
        vi=rows["vi"],
        z=rows["z"],
        se=rows["se"],
        eaf=rows["eaf"],
        imp=rows["imp"],
        n_imputed=n_imputed,
        n_missing=n_missing,
    )


def _completed_analysis(
    ai: int,
    *,
    src_csr: RaggedCSRReader,
    src_alids: list[str],
    src_idx_remap: list[int],
    new_alid_to_idx: dict[str, int],
    block_ids: list[str],
    block_canonical_alids: dict[str, list[str | None]],
    fills: dict[str, tuple[float, float]],
) -> _AnalysisCompleted:
    """One Analysis's completed rows: observed cells carried across (ADR
    0036), block fills as imputed, block positions neither produced as
    missing -- or, when no completable block touches the Analysis, its source
    association list untouched."""
    obs = src_csr.get_analysis(ai)
    if not block_ids:
        return _passthrough_analysis(obs, src_idx_remap)
    observed = _observed_alid_maps(obs, src_alids)
    cells, seen_alids, n_imputed, n_missing = _block_cells(
        block_ids,
        block_canonical_alids,
        new_alid_to_idx,
        observed.z_by_alid,
        observed.se_by_alid,
        observed.eaf_by_alid,
        fills,
    )
    cells += _carried_cells(
        observed.z_by_alid, observed.se_by_alid, observed.eaf_by_alid, seen_alids, new_alid_to_idx
    )
    return _sort_cells(cells, n_imputed, n_missing)


def _assemble_completed_csr(
    plan: _BlockPlan,
    axis: _MergedAxis,
    fills_by_analysis: dict[int, dict[str, tuple[float, float]]],
) -> _CompletedCsr:
    """Phase 3 assembly: one completed row block per Analysis -- remapped
    observed rows plus fills, sorted by variant index -- and the flat offsets
    and totals the later phases report."""
    print("Assembling completed CSR per analysis...")
    all_vi: list[np.ndarray] = []
    all_z: list[np.ndarray] = []
    all_se: list[np.ndarray] = []
    all_eaf: list[np.ndarray] = []
    all_imp: list[np.ndarray] = []
    offsets: list[int] = [0]
    total_imputed = 0
    total_missing = 0

    for ai in range(plan.state.n_analyses):
        completed = _completed_analysis(
            ai,
            src_csr=plan.src_csr,
            src_alids=plan.state.src_alids,
            src_idx_remap=axis.src_idx_remap,
            new_alid_to_idx=axis.new_alid_to_idx,
            block_ids=plan.analysis_to_blocks.get(ai, []),
            block_canonical_alids=plan.block_canonical_alids,
            fills=fills_by_analysis.get(ai, {}),
        )
        all_vi.append(completed.vi)
        all_z.append(completed.z)
        all_se.append(completed.se)
        all_eaf.append(completed.eaf)
        all_imp.append(completed.imp)
        offsets.append(offsets[-1] + len(completed.vi))
        total_imputed += completed.n_imputed
        total_missing += completed.n_missing

        if (ai + 1) % 500 == 0:
            print(
                f"  {ai + 1:,} / {plan.state.n_analyses:,} analyses | "
                f"imputed {total_imputed:,} | missing {total_missing:,}"
            )

    return _CompletedCsr(
        offsets=offsets,
        vi=all_vi,
        z=all_z,
        se=all_se,
        eaf=all_eaf,
        imp=all_imp,
        total_imputed=total_imputed,
        total_missing=total_missing,
    )


def _encode_plan(
    src: Path,
    ld_dir: Path,
    ancestry: str,
    manifest: StoreManifest,
    axis: _MergedAxis,
) -> _EncodePlan:
    """The completed release's encoding: its source's plan, since completion
    writes into the source's arrays (ADR 0038 §4) -- with `eaf_reference`
    added when the panel has frequencies (ADR 0037 §4, issue #113) and the
    source's EAF baseline carried across the variant remap (ADR 0037 §2)."""
    # Asked for whatever the source declares, `absent` included: a release
    # whose Analyses reported no frequency still gains imputed cells, and
    # those cells have the panel's frequency (issue #113). It gets an
    # `eaf_reference` array and no `eaf` plane -- NaN on every observed
    # cell, the panel's value on every imputed one.
    eaf_reference = panel_reference_eaf(ld_dir, ancestry, axis.variants)
    encoding = manifest.encoding.with_eaf_reference(eaf_reference is not None)
    codec = StoreCodec(encoding)
    src_baseline = _source_eaf_baseline(src)
    out_baseline = None
    if src_baseline is not None:
        out_baseline = np.full(len(axis.variants), np.nan, dtype=np.float32)
        out_baseline[np.asarray(axis.src_idx_remap, dtype=np.int64)] = src_baseline
    return _EncodePlan(
        encoding=encoding, codec=codec, eaf_reference=eaf_reference, out_baseline=out_baseline
    )


def _flatten_csr(csr: _CompletedCsr) -> _FlatCsr:
    """The flat arrays the zarr group stores, from the per-Analysis lists."""
    return _FlatCsr(
        offsets=np.array(csr.offsets, dtype=np.int64),
        vi=np.concatenate(csr.vi) if csr.vi else np.empty(0, dtype=np.int32),
        z=np.concatenate(csr.z) if csr.z else np.empty(0, dtype=np.float32),
        se=np.concatenate(csr.se) if csr.se else np.empty(0, dtype=np.float32),
        eaf=np.concatenate(csr.eaf) if csr.eaf else np.empty(0, dtype=np.float32),
        imp=np.concatenate(csr.imp) if csr.imp else np.empty(0, dtype=np.uint8),
    )


def _write_csr_id_arrays(root: Any, flat: _FlatCsr, codec: StoreCodec) -> None:
    """The CSR's index planes plus z: offsets, variant_index, the fixed-point
    z plane with its overflow table (ADR 0037 §1), and the imputed mask."""
    root.create_dataset(
        "offsets",
        data=flat.offsets,
        chunks=(_OFFSET_CHUNK,),
        compressor=_COMPRESSOR,
        dtype=np.int64,
    )
    root.create_dataset(
        "variant_index",
        data=flat.vi,
        chunks=(_ASSOC_CHUNK,),
        compressor=_COMPRESSOR,
        dtype=np.int32,
    )
    # Completion writes into the source's arrays, so it encodes with the
    # source's plan (ADR 0038 §4) -- the overflow table travels with the
    # plane it belongs to. The one addition is `eaf_reference`, which
    # records a physical fact about this release rather than reinterpreting
    # its source's bytes.
    z_overflow = ZOverflowBuilder()
    root.create_dataset(
        "z",
        data=codec.encode_z(flat.z, positions=positions_flat(0), overflow=z_overflow),
        chunks=(_ASSOC_CHUNK,),
        compressor=_COMPRESSOR,
        dtype=codec.z_dtype,
    )
    z_overflow.table().write(root)
    root.create_dataset(
        "imputed",
        data=flat.imp,
        chunks=(_ASSOC_CHUNK,),
        compressor=_COMPRESSOR,
        dtype=np.uint8,
    )


def _write_eaf_and_se_arrays(
    root: Any, encode_plan: _EncodePlan, flat: _FlatCsr, n_analyses: int
) -> None:
    """The EAF and SE planes plus the release attrs. Observed frequencies
    only: an imputed cell's frequency is the panel's, stored once per variant
    in `eaf_reference` and applied on read (ADR 0037 §4); an observed cell
    whose source reported none stays absent. The per-variant baseline travels
    with the values across the variant remap rather than being recomputed from
    them, so a value decoded from the source re-encodes to the same code."""
    if not encode_plan.encoding.eaf.is_absent:
        write_eaf_csr(
            root,
            encode_plan.codec,
            flat.vi,
            flat.eaf,
            baseline=encode_plan.out_baseline,
            compressor=_COMPRESSOR,
            chunks=(_ASSOC_CHUNK,),
        )
    if encode_plan.eaf_reference is not None:
        write_eaf_reference(root, encode_plan.eaf_reference, compressor=_COMPRESSOR)
    decoded_eaf = RaggedEafPlane.open(root, encode_plan.encoding, imputed=root["imputed"]).slice(
        0, len(flat.se)
    )
    analysis_index = np.searchsorted(flat.offsets[1:], np.arange(len(flat.se)), side="right")
    se_coefficients = (
        fit_se(
            flat.se,
            decoded_eaf,
            analysis_index,
            n_analyses=n_analyses,
            compressor=_COMPRESSOR,
            chunks=_ASSOC_CHUNK,
        )[0]
        if encode_plan.encoding.se.is_residual
        else None
    )
    write_se_csr(
        root,
        encode_plan.codec,
        flat.se,
        decoded_eaf,
        analysis_index,
        se_coefficients,
        compressor=_COMPRESSOR,
        chunks=(_ASSOC_CHUNK,),
    )
    root.attrs["layout"] = "ragged"
    root.attrs["completion_state"] = "reference_completed"
    root.attrs["n_analyses"] = n_analyses
    root.attrs["n_associations"] = len(flat.se)


def _write_completed_zarr(
    staged: StagedRelease, encode_plan: _EncodePlan, csr: _CompletedCsr, n_analyses: int
) -> None:
    """Phase 4: write the completed CSR as the store's ragged zarr group."""
    print("Writing zarr CSR...")
    ragged_path = staged.path / RAGGED_ZARR_PATH
    ragged_path.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(ragged_path), mode="w")
    flat = _flatten_csr(csr)
    _write_csr_id_arrays(root, flat, encode_plan.codec)
    _write_eaf_and_se_arrays(root, encode_plan, flat, n_analyses)


def _write_top_hits_and_analyses(
    staged: StagedRelease,
    state: _SourceState,
    encode_plan: _EncodePlan,
    *,
    ancestry: str,
) -> None:
    """Phase 5 metadata: top-hit indexes built with the completed encoding,
    and analyses.tsv -- rollup from index.sqlite (ADR 0030), `eaf_scope` from
    what the release holds, completion columns refreshed, and the source's
    pre-completion hit counts zeroed so add_hit_counts recomputes them."""
    print("Building top-hit indexes...")
    build_ragged_top_hit_indexes(staged.path, encoding=encode_plan.encoding)

    print("Writing analyses.tsv...")
    with staged.index_connection() as quality_db:
        quality_rollup = completion_quality_rollup(quality_db, state.n_analyses)
    dst_analyses = reset_top_hit_counts(
        [
            replace(
                a,
                completed_against=(
                    ancestry if state.impute_mask is None or bool(state.impute_mask[i]) else ""
                ),
                # An Analysis that gained imputed cells in a release carrying
                # reference EAF now stores a frequency for them, whatever its
                # source reported (ADR 0037 §4), so `eaf_scope` follows what
                # the release holds rather than being copied forward.
                eaf_scope=completed_eaf_scope(
                    a, quality_rollup[i], encode_plan.eaf_reference is not None
                ),
                completion_median_pearson_r=quality_rollup[i].median_pearson_r,
                completion_n_imputed_total=quality_rollup[i].n_imputed_total,
                completion_n_missing_total=quality_rollup[i].n_missing_total,
            )
            for i, a in enumerate(state.src_analyses)
        ]
    )
    write_analysis_records(
        staged.path / "analyses.tsv", add_hit_counts(staged.path, dst_analyses)
    )


def _completed_manifest(
    manifest: StoreManifest,
    *,
    source_format_version: str,
    release_id: str | None,
    ld_panel_id: str,
    ancestry: str,
    min_cor: float,
    thresh: float,
    cis_window_bp: int,
    axis: _MergedAxis,
    csr: _CompletedCsr,
    encode_plan: _EncodePlan,
) -> StoreManifest:
    """Phase 6 finalization: the completed release's manifest, with its
    provenance recording what completion added to its source."""
    new_release_id = release_id or f"{manifest.release_id}-completed"
    return StoreManifest(
        # Preserved with the format version below: a completed release is
        # written into its source's arrays, so it is in its source's
        # encoding -- plus `eaf_reference`, which says this release carries
        # panel frequencies for the cells it imputed (ADR 0037 §4).
        encoding=encode_plan.encoding,
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
                n_variants_total=len(axis.variants),
                n_variants_new=len(axis.new_variants),
                cis_window_bp=cis_window_bp,
                n_imputed=csr.total_imputed,
                n_missing=csr.total_missing,
            ),
        },
    )


def _completion_result(
    dst: Path, state: _SourceState, axis: _MergedAxis, csr: _CompletedCsr
) -> CompletionResult:
    result = CompletionResult(
        output_path=dst,
        n_variants=len(axis.variants),
        n_analyses=state.n_analyses,
        n_associations=csr.offsets[-1],
        n_imputed=csr.total_imputed,
        n_missing=csr.total_missing,
    )
    print(
        f"Reference completion complete: {result.n_variants:,} variants, "
        f"{result.n_analyses:,} analyses, {result.n_associations:,} associations "
        f"({result.n_imputed:,} imputed, {result.n_missing:,} missing)"
    )
    return result


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
    """The shared core: block discovery, checkpointed LD-block completion and
    CSR assembly, encoding, metadata and manifest, sequenced inside the
    staging context that makes a completed release atomic."""
    src = Path(source_path)
    dst = Path(dest_path)
    manifest, source_format_version = _open_source(src)

    with OpenGWASDBStore.staging(dst, overwrite=True) as staged:
        state = _read_source_state(src, impute_analysis_ids)
        plan = _plan_blocks(state, ld_dir, ancestry, cis_window_bp)
        axis = _merged_axis(state, plan.new_alids)
        _write_variant_axis(staged.path, state, axis)
        csr = _checkpoint_and_assemble(
            staged,
            plan,
            axis,
            checkpoint_dir=checkpoint_dir,
            n_workers=n_workers,
            min_cor=min_cor,
            thresh=thresh,
            region_cap_bp=region_cap_bp,
        )
        encode_plan = _encode_plan(src, ld_dir, ancestry, manifest, axis)
        _write_completed_zarr(staged, encode_plan, csr, state.n_analyses)
        _write_top_hits_and_analyses(staged, state, encode_plan, ancestry=ancestry)
        staged.write_manifest(
            _completed_manifest(
                manifest,
                source_format_version=source_format_version,
                release_id=release_id,
                ld_panel_id=ld_panel_id,
                ancestry=ancestry,
                min_cor=min_cor,
                thresh=thresh,
                cis_window_bp=cis_window_bp,
                axis=axis,
                csr=csr,
                encode_plan=encode_plan,
            )
        )
        result = _completion_result(dst, state, axis, csr)
    return result


def _source_eaf_baseline(source_path: Path) -> np.ndarray | None:
    """The observed release's per-variant `eaf_baseline`, or None if it has none.

    Read straight from the source's CSR group rather than recomputed from the
    values it holds: recomputing from *decoded* frequencies would shift each
    baseline by up to half a step and re-quantise every cell against the moved
    baseline, so a completed release would be less accurate than its source for
    no reason (ADR 0037 §2).
    """
    group = zarr.open_group(str(Path(source_path) / RAGGED_ZARR_PATH), mode="r")
    if EAF_BASELINE not in group:
        return None
    return np.asarray(group[EAF_BASELINE][:], dtype=np.float32)
