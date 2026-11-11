"""Ragged Reference Completion — enhancement pipeline.

Reads an observed-only ragged store and writes a new Reference-Completed
Store Release by imputing z-scores and SE for the LD reference panel
variants inside each Analysis's regions: its cis region, where it has a Trait
position, and its significant trans regions (`completion.regions`, ADR 0039,
issue #430).

Pipeline shape mirrors dense completion (ADR 0023): LD blocks are the
process-pool parallelism unit, checkpointed for resume. The difference from
dense is the unit of relevance -- a block matters only to the Analyses one of
whose regions it overlaps, not to every Analysis genome-wide -- so Phase 1 also
records a block-to-Analyses mapping alongside the union variant axis, and
Phase 2's tasks are told which Analyses to complete rather than assuming all
of them:
  Phase 1 (sequential): find the LD blocks each Analysis's regions overlap,
    collecting the union variant axis and a block_id -> Analysis indices
    mapping as it goes.
  Phase 2 (parallel, n_workers processes over touched LD blocks): each
    worker opens the source store and LD panel itself, completes every
    Analysis assigned to its block, fitting on all of the block but filling
    and counting only inside the Analysis's regions, and writes its own
    checkpoint file.
  Phase 3 (sequential): shard every block's fills by Analysis batch on disk
    (writing completion_quality as it goes, `completion.fill_shards`, shared
    with dense), assemble each Analysis's completed association list from its
    batch's shard, and write the CSR through `RaggedCSRWriter`, region by
    region; then top-hit indexes and manifest. Nothing in it holds every fill
    or every completed row of the Store at once (issue #424).
"""

from __future__ import annotations

import json
import logging
import shutil
from contextlib import closing
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from opengwasdb import top_hits
from opengwasdb.associations import Associations, Cells, CsrReader
from opengwasdb.completion import checkpoint, fill_shards
from opengwasdb.completion.ancestry_filter import derive_impute_analysis_ids
from opengwasdb.completion.block import (
    REGION_CAP_BP,
    panel_positions,
    read_as_planned,
    run_block,
)
from opengwasdb.completion.ld_panel import (
    BlockPlacement,
    LDBlock,
    PanelPlacement,
    SharedEndpoint,
    check_panel_has_chromosomes,
    panel_variant,
    shared_endpoints,
)
from opengwasdb.completion.manifest import (
    build_completion_provenance,
    completed_release_manifest,
)
from opengwasdb.completion.parallel import run_block_tasks
from opengwasdb.completion.reference_eaf import completed_eaf_scope, panel_reference_eaf
from opengwasdb.completion.regions import (
    Region,
    RegionTable,
    analysis_regions,
    inside,
    read_region_table,
    region_blocks,
)
from opengwasdb.completion.schema import completion_quality_rollup, create_completion_quality_table
from opengwasdb.encoding import (
    EAF_BASELINE,
    SE_EXCEPTION_BUDGET,
    StoreEncoding,
)
from opengwasdb.encoding.planes import SE_COEFFICIENTS
from opengwasdb.layouts.ragged.by_variant import build_variant_index, recorded_provenance
from opengwasdb.layouts.ragged.top_hits import build_ragged_top_hit_indexes
from opengwasdb.layouts.ragged.zarr_csr import (
    RAGGED_ZARR_PATH,
    RaggedCSRWriter,
    ReferenceCompletion,
    SeExceptionCounts,
)
from opengwasdb.model.analyses import (
    Analysis,
    ancestry_impute_mask,
    read_analysis_records,
    reset_top_hit_counts,
    write_analysis_records,
)
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.store import arrays as store_arrays
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
from opengwasdb.variants.normalise import CanonicalVariant, chromosome_sort_key
from opengwasdb.variants.routing import Duplicates, collapse_duplicates

log = logging.getLogger(__name__)

_LD_PANEL_ID = "eur-hg38-gpm"


@dataclass(frozen=True)
class CompletionResult:
    output_path: Path
    n_variants: int
    n_analyses: int
    n_associations: int
    n_imputed: int
    n_missing: int
    #: The `region_id` of each region no LD block of the panel overlaps, which is
    #: therefore completed nowhere (issue #430): a panel gap, reported, not refused.
    regions_without_block: tuple[str, ...] = ()


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
    #: The block's chromosome.
    chromosome: str
    #: Each assigned Analysis's regions: only its variants inside one are filled or counted.
    regions: dict[int, tuple[Region, ...]]
    #: How the plan's read of the block placed its rows, which the worker's must repeat.
    placement: BlockPlacement
    #: Per Analysis, the variants at this block's end the next block owns.
    ceded: dict[int, frozenset[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class _ObservedAlidMaps:
    """One Analysis's observed z/se/eaf, each keyed by canonical ALID.

    A named record rather than a positional three-dict tuple because a
    statistic placed in the wrong slot travels on without raising (issue
    #130) -- an observed SE in the EAF slot is still a *valid* frequency.
    Field access makes the statistic a caller reads explicit.
    """

    z_by_alid: dict[str, float]
    se_by_alid: dict[str, float]
    eaf_by_alid: dict[str, float]


def _admitted_observed(
    obs: Associations,
) -> tuple[Associations, Duplicates]:
    """One Analysis's observed rows, each variant's duplicates resolved by
    Association Admission's rule (`collapse_duplicates`, D5).

    A Store holds a (variant, Analysis) once, but a Store a builder wrote
    before D5 can hold one twice (a BESD probe naming a SNP twice). Rows that
    agree are one association, collapsed to the first; rows that disagree are
    all dropped, so the cell is missing and completion may impute it, rather
    than whichever row a map wrote last.
    """
    found = collapse_duplicates(obs.variant_index, obs.z, obs.se)
    if len(found.keep) == len(obs.variant_index):
        return obs, found
    return obs.take(found.keep), found


def _warn_observed_duplicates(
    ai: int, obs: Associations, found: Duplicates, src_alids: list[str]
) -> None:
    """Say what `_admitted_observed` dropped or collapsed of one Analysis's
    rows, ``obs``: a count, and a line for every dropped row."""
    if len(found.dropped):
        first = [src_alids[int(i)] for i in found.conflicting[:5]]
        log.warning(
            "Analysis %d: %d observed row(s) at %d variant(s) disagree, so none is "
            "carried (duplicate_conflicting); first: %s",
            ai, len(found.dropped), len(found.conflicting), first,
        )
        fill_shards.log_dropped("observed row", (
            (ai, src_alids[int(obs.variant_index[i])], float(obs.z[i]), float(obs.se[i]))
            for i in found.dropped.tolist()
        ))
    if found.collapsed:
        log.warning(
            "Analysis %d: %d observed row(s) repeat an agreeing row and are collapsed "
            "into it (duplicate_collapsed)",
            ai, found.collapsed,
        )


def _observed_alid_maps(obs: Associations, src_alids: list[str]) -> _ObservedAlidMaps:
    """Map one Analysis's observed CSR rows onto ``{alid: z/se/eaf}``. Observed
    EAF is carried across the rebuild (ADR 0036): Reference Completion adds
    panel rows to an Analysis; it does not change what the source reported for
    the rows it already had. The block reader folds the observed rows this
    way, and Phase 3's assembly reads the same rows as arrays
    (`_observed_rows`), each variant's duplicates resolved by D5 in both
    (`_admitted_observed`)."""
    obs, _ = _admitted_observed(obs)
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


def _open_source_sequence(src: Path) -> CsrReader:
    """The source release's association sequence, read through the Association
    Reader: every Analysis completion reads meets the presence rule (ADR 0070)."""
    return CsrReader.open(src, source=f"complete-ragged's source {src}")


def _make_reader(task: _BlockTask):
    """ragged's half of the ``run_block`` seam: read one Analysis's observed
    z/se at a time from its CSR row, since (unlike dense's `gwas`-coverage
    matrix) only the Analyses this task was assigned even have a cis window
    touching this block.
    """

    def make_reader(block, canonical_alids: list[str | None]):
        source = _open_source_sequence(task.source_path)
        src_variant_axis = VariantAxis(task.source_path)
        try:
            src_alids = [v.alid for v in src_variant_axis.all()]
        finally:
            src_variant_axis.close()

        def read(ai: int) -> tuple[np.ndarray, np.ndarray]:
            obs = source.read(Cells.analysis(ai))
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


def _outside(task: _BlockTask, analysis: int, positions: np.ndarray) -> np.ndarray:
    """The block's rows outside the Analysis's regions (issue #430), by position: the
    fit still sees every observation in the block, and the checkpoint's fills and
    `completion_quality` counts hold only the rows the Analysis is given. A panel row
    no ALID names is never one of those rows, inside or outside a region."""
    return ~inside(task.regions[analysis], task.chromosome, positions)


def _run_block(task: _BlockTask) -> checkpoint.BlockCompletionResult | None:
    """Complete one LD block for the Analyses one of whose regions it overlaps."""
    result = run_block(
        task.tsv_path, task.thresh, task.min_cor, task.region_cap_bp,
        read_as_planned(task.placement, _make_reader(task)),
        ceded=lambda analysis: task.ceded.get(analysis, frozenset()),
        outside=lambda analysis, positions: _outside(task, analysis, positions),
    )
    if result is None:
        return None
    checkpoint.write_block_checkpoint(task.checkpoint_path, result)
    # Fills stay on disk (the checkpoint); the parent reads them back in Phase 3
    # (mirrors dense — see issue 044).
    return checkpoint.BlockCompletionResult(block_id=result.block_id, quality_rows=[], fills=[])


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
    regions: str | Path | None = None,
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
    regions: the build's region table, its regions (`completion.regions`, issue #430).
    """
    table, impute_analysis_ids, resolved = _check_completion_inputs(
        Path(source_path), ld_dir, ancestry, regions, cis_window_bp, impute_analysis_ids
    )
    dst = Path(dest_path)
    checkpoint_dir = checkpoint.checkpoint_dir_for(dst)
    checkpoint.require_fresh_destination(dst, checkpoint_dir, overwrite, "resume_ragged_completion")

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
        "impute_analysis_ids": None if impute_analysis_ids is None else sorted(impute_analysis_ids),
        "region_cap_bp": region_cap_bp,
        **_region_params(table),
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
        regions=resolved,
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
    impute_ids = set(impute_ids) if impute_ids is not None else None
    regions = _resolve_regions(
        Path(params["source_path"]), _resumed_regions(checkpoint_dir, params),
        params["cis_window_bp"], impute_ids,
    )
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
        impute_analysis_ids=impute_ids,
        region_cap_bp=params.get("region_cap_bp"),
        regions=regions,
    )
    shutil.rmtree(checkpoint_dir)
    return result


def _region_params(table: RegionTable | None) -> dict[str, str | None]:
    """The region table's build parameters. Recorded, null included, so a resume
    knows the rule (issue #430) its checkpoints were written under, and refuses
    one that predates it (`_resumed_regions`)."""
    if table is None:
        return {"regions": None, "regions_sha256": None}
    return {"regions": str(table.path.resolve()), "regions_sha256": table.sha256}


def _resumed_regions(checkpoint_dir: Path, params: dict[str, Any]) -> RegionTable | None:
    """The region table a resumed completion's checkpoints were written against.

    Refused when the checkpoint predates the region rule (issue #430), whose blocks
    were filled whole, or when the table has changed since: either would land fills
    one rule chose beside rows another one chose.
    """
    if "regions" not in params:
        raise ValueError(
            f"{checkpoint_dir} was written by a complete-ragged that filled whole LD "
            "blocks, before completion was bounded to each Analysis's regions (issue "
            "#430); its block checkpoints cannot be resumed, so start the completion again"
        )
    if params["regions"] is None:
        return None
    table = read_region_table(
        params["regions"], StoreManifest.load(Path(params["source_path"])).reference_assembly
    )
    if table.sha256 != params["regions_sha256"]:
        raise ValueError(
            f"region table {params['regions']} has changed since {checkpoint_dir} was "
            "written; its block checkpoints were filled inside the regions it held then"
        )
    return table


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
    """Phase 1's LD-block discovery (ADR 0023, issue #430): each Analysis's
    regions and every block they overlap, each block's tsv path, chromosome,
    canonical panel ALIDs and their positions, and the panel ALIDs inside some
    Analysis's regions the source never held. Fills stay on disk in checkpoints;
    this is everything Phase 2 runs and Phase 3 resolves them against."""

    source: CsrReader
    state: _SourceState
    regions: dict[int, tuple[Region, ...]]
    analysis_to_blocks: dict[int, list[str]]
    block_to_tsv: dict[str, Path]
    block_to_analyses: dict[str, list[int]]
    block_chromosome: dict[str, str]
    block_canonical_alids: dict[str, list[str | None]]
    block_positions: dict[str, np.ndarray]
    new_alids: set[str]
    #: How the planned blocks placed their rows: the completion record's
    #: `panel_indels` and `panel_snvs_unplaced` (`PanelPlacement.record`).
    panel_placement: dict[str, dict[str, object]]
    #: Each block's end another planned block starts at (`shared_endpoints`).
    endpoints: dict[str, SharedEndpoint] = field(default_factory=dict)
    #: The `region_id` of each region no block overlaps (`RegionBlocks.uncovered`).
    uncovered: tuple[str, ...] = ()
    #: Per planned block's TSV, how the plan's read placed its rows (`BlockPlacement`).
    block_placement: dict[Path, BlockPlacement] = field(default_factory=dict)

    def ceded(self, block_id: str) -> dict[int, frozenset[str]]:
        """Per Analysis, the variants at ``block_id``'s end the next block owns.

        The block that starts at a shared endpoint owns it for an Analysis
        completed against both blocks. One completed against this block alone
        has no block after it there, so this block keeps its end, as a
        chromosome's last block does.
        """
        endpoint = self.endpoints.get(block_id)
        if endpoint is None:
            return {}
        return {
            analysis: endpoint.alids
            for analysis in self.block_to_analyses[block_id]
            if endpoint.following in self.analysis_to_blocks[analysis]
        }


@dataclass
class _MergedAxis:
    """The union variant axis in canonical order, plus the maps that translate
    source variant indices and new panel ALIDs into it."""

    variants: list[CanonicalVariant]
    new_variants: list[CanonicalVariant]
    new_alid_to_idx: dict[str, int]
    src_idx_remap: np.ndarray


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
    """Phase 3's assembly output: the writer holding every completed Analysis,
    in Analysis order, and the imputed/missing totals the later phases report
    and store."""

    writer: RaggedCSRWriter
    total_imputed: int
    total_missing: int


@dataclass
class _EncodePlan:
    """The completed release's encoding (ADR 0038 §4) and the per-variant EAF
    data it adds: the panel reference for imputed cells and the carried
    baseline for observed ones (ADR 0037 §2/§4). A residual SE plane carries
    its source's `se_coefficients` too, never a refit (issue #301)."""

    encoding: StoreEncoding
    eaf_reference: np.ndarray | None
    out_baseline: np.ndarray | None
    se_coefficients: np.ndarray | None


def _check_completion_inputs(
    src: Path,
    ld_dir: str | Path,
    ancestry: str,
    regions: str | Path | None,
    cis_window_bp: int,
    impute_analysis_ids: set[str] | None,
) -> tuple[RegionTable | None, set[str] | None, dict[int, tuple[Region, ...]]]:
    """The refusals that need no side effect, run before any is taken: the
    region table, read and checked whole, and the Analyses to impute, derived
    from the source's ancestry (ADR 0028) when not given, each with its regions.

    Ahead of `require_fresh_destination`, which under `overwrite` deletes a
    previous checkpoint: a run that is going to be refused for its source, its
    panel or its regions must not first discard what an earlier run left.
    """
    _refuse_sample_size_source(src)
    check_panel_has_chromosomes(ld_dir, ancestry)
    table = (
        None if regions is None
        else read_region_table(regions, StoreManifest.load(src).reference_assembly)
    )
    if impute_analysis_ids is None:
        impute_analysis_ids = derive_impute_analysis_ids(_read_analyses_rows(src), ancestry)
    resolved = _resolve_regions(src, table, cis_window_bp, impute_analysis_ids)
    return table, impute_analysis_ids, resolved


def _resolve_regions(
    src: Path, table: RegionTable | None, cis_window_bp: int, impute_analysis_ids: set[str] | None
) -> dict[int, tuple[Region, ...]]:
    """Each Analysis to impute's regions (`analysis_regions`), refusing what it refuses:
    a Store whose coverage is not `cis` with no table, an Analysis no region can be
    found for, and a table naming an Analysis the source lacks."""
    analyses, mask = _source_analyses(src, impute_analysis_ids)
    return analysis_regions(
        analyses, _imputable_indices(mask, len(analyses)), table, cis_window_bp,
        StoreManifest.load(src).association_coverage,
    )


def _refuse_sample_size_source(src: Path) -> None:
    """Refuse a source carrying a per-association sample-size plane (ADR 0067).

    Completion rewrites the whole CSR from decoded `z`/`se`/`eaf`, so an `n`
    plane would be dropped from the completed release while its Analyses still
    declared variant-level N -- and an imputed cell has no N of its own to
    carry. Carrying N through completion is not built, so it is refused by
    name, before any destination or checkpoint exists.
    """
    if StoreManifest.load(src).encoding.n is not None:
        raise ValueError(
            f"source release {src} carries a per-association sample-size plane "
            "(data.zarr/ragged/n); Ragged completion does not carry N through, and "
            "would drop it (ADR 0067)"
        )


def _open_source(src: Path) -> tuple[StoreManifest, str]:
    """Open the observed-only source and refuse a format this build cannot
    write, before any of the work (ADR 0038 §4)."""
    _refuse_sample_size_source(src)
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

    src_analyses, impute_mask = _source_analyses(src, impute_analysis_ids)
    n_analyses = len(src_analyses)
    print(f"Source: {len(src_alids):,} variants, {n_analyses:,} analyses")
    if impute_mask is not None:
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


def _source_analyses(
    src: Path, impute_analysis_ids: set[str] | None
) -> tuple[list[Analysis], np.ndarray | None]:
    """The source's Analyses in `analysis_index` order, and the ancestry-match
    impute mask derived from them (ADR 0028).

    analyses.tsv is the sole source of truth for Analytical Metadata (ADR 0034,
    issue #69) -- including each Analysis's Trait genomic position, so the
    region derivation reads it directly rather than through a second,
    independently-shaped position file."""
    analyses = sorted(
        read_analysis_records(src / "analyses.tsv"), key=lambda a: int(a.analysis_index)
    )
    return analyses, ancestry_impute_mask(analyses, impute_analysis_ids)


def _imputable_indices(impute_mask: np.ndarray | None, n_analyses: int) -> list[int]:
    """Every Analysis, or -- under the ancestry-match filter (ADR 0028) --
    only those the filter admits. The others are left observed-only: Phase 3's
    pass-through path carries them through unchanged, so they never enter the
    block maps at all."""
    return [i for i in range(n_analyses) if impute_mask is None or bool(impute_mask[i])]


def _imputable_analyses(state: _SourceState) -> list[tuple[int, Analysis]]:
    """`_imputable_indices` of the source state, with their Analyses."""
    return [
        (i, state.src_analyses[i])
        for i in _imputable_indices(state.impute_mask, state.n_analyses)
    ]


def _new_panel_alids(alids: list[str | None], source_alids: set[str]) -> set[str]:
    """The panel ALIDs among `alids` the source release never held."""
    return {a for a in alids if a is not None and a not in source_alids}


def _scan_block_maps(
    state: _SourceState,
    source: CsrReader,
    regions: dict[int, tuple[Region, ...]],
    blocks_of: dict[int, list[LDBlock]],
    ld_dir: Path,
    ancestry: str,
) -> _BlockPlan:
    """Map every imputable Analysis to the LD blocks its regions overlap and build
    the `_BlockPlan` Phase 2 runs and Phase 3 resolves: the block -> Analysis
    map, each block's tsv path, canonical panel ALIDs and positions, and the
    union of panel ALIDs inside some Analysis's regions the source never held.
    A panel variant outside every region is no row of any Analysis, so it is
    not added to the axis."""
    block_to_tsv: dict[str, Path] = {}
    block_to_analyses: dict[str, list[int]] = {}
    block_chromosome: dict[str, str] = {}
    block_canonical_alids: dict[str, list[str | None]] = {}
    block_positions: dict[str, np.ndarray] = {}
    analysis_to_blocks: dict[int, list[str]] = {}
    planned: list[LDBlock] = []
    placement = PanelPlacement()

    for i, _a in _imputable_analyses(state):
        blocks = blocks_of.get(i, [])
        analysis_to_blocks[i] = [b.block_id for b in blocks]
        for block in blocks:
            block_to_analyses.setdefault(block.block_id, []).append(i)
            if block.block_id in block_to_tsv:
                continue
            block_to_tsv[block.block_id] = block.tsv_path
            block_chromosome[block.block_id] = block.chrom
            block_canonical_alids[block.block_id] = block.alids
            block_positions[block.block_id] = panel_positions(block)
            planned.append(block)
            placement.add(block)

    return _BlockPlan(
        source=source,
        state=state,
        regions=regions,
        analysis_to_blocks=analysis_to_blocks,
        block_to_tsv=block_to_tsv,
        block_to_analyses=block_to_analyses,
        block_chromosome=block_chromosome,
        block_canonical_alids=block_canonical_alids,
        block_positions=block_positions,
        new_alids=_region_panel_alids(
            state, regions, block_to_analyses, block_chromosome,
            block_canonical_alids, block_positions,
        ),
        endpoints=shared_endpoints(planned),
        block_placement=placement.blocks,
        panel_placement=_logged_placement(placement, ld_dir, ancestry),
    )


def _logged_placement(
    placement: PanelPlacement, ld_dir: Path, ancestry: str
) -> dict[str, dict[str, object]]:
    placement.log(ld_dir, ancestry)
    return placement.record()


def _region_panel_alids(
    state: _SourceState,
    regions: dict[int, tuple[Region, ...]],
    block_to_analyses: dict[str, list[int]],
    block_chromosome: dict[str, str],
    block_canonical_alids: dict[str, list[str | None]],
    block_positions: dict[str, np.ndarray],
) -> set[str]:
    """The panel ALIDs the source never held that lie inside a region of an
    Analysis completed against their block: the variants completion adds."""
    source_alids = set(state.src_alids)
    new_alids: set[str] = set()
    for block_id, analyses in block_to_analyses.items():
        held = inside(
            [r for i in analyses for r in regions[i]],
            block_chromosome[block_id], block_positions[block_id],
        )
        alids = block_canonical_alids[block_id]
        new_alids.update(_new_panel_alids(
            [a for a, kept in zip(alids, held.tolist(), strict=True) if kept], source_alids
        ))
    return new_alids


def _plan_blocks(
    state: _SourceState, ld_dir: Path, ancestry: str, regions: dict[int, tuple[Region, ...]]
) -> _BlockPlan:
    """Phase 1: the LD blocks each imputable Analysis's regions overlap
    (`completion.regions`, issue #430), and the plan Phase 2 runs and Phase 3
    resolves. An Analysis with no region is carried Observed-Only."""
    source = _open_source_sequence(state.src)
    n_imputable = len(_imputable_analyses(state))
    kinds = [r.kind for held in regions.values() for r in held]
    print(
        f"Regions: {kinds.count('cis'):,} cis and {kinds.count('significant_trans'):,} "
        f"significant trans, over {len(regions):,} of {n_imputable:,} analyses to "
        f"impute; {n_imputable - len(regions):,} have none and are carried observed-only"
    )
    print("Scanning the LD blocks the regions overlap + new panel variants...")
    found = region_blocks(regions, ld_dir, ancestry)
    plan = _scan_block_maps(state, source, regions, found.blocks, ld_dir, ancestry)
    plan.uncovered = tuple(r.region_id for r in found.uncovered)
    print(_uncovered_line(plan.uncovered))
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
    """Merge the new panel ALIDs into the source axis: canonicalise
    (`panel_variant`), drop any that collide with source variants, sort under
    the store's key, and build the maps that translate source variant indices
    and panel ALIDs into the union."""
    source_alids = set(state.src_alids)
    new_variants: list[CanonicalVariant] = []
    for alid in new_alids:
        variant = panel_variant(alid)
        if variant is not None and variant.alid not in source_alids:
            new_variants.append(variant)
    new_variants.sort(key=_sort_key)
    variants = sorted(
        [_as_canonical(v) for v in state.src_variants] + new_variants, key=_sort_key
    )
    print(f"Merged variant table: {len(variants):,} variants")

    new_alid_to_idx = {v.alid: i for i, v in enumerate(variants)}
    src_idx_remap = np.zeros(len(state.src_alids), dtype=np.int64)
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
    blocks_dir = checkpoint.checkpoint_blocks_dir(checkpoint_dir)

    pending: list[_BlockTask] = []
    n_existing = 0
    for block_id, tsv_path in plan.block_to_tsv.items():
        ckpt_path = blocks_dir / f"{checkpoint.sanitize_block_id(block_id)}.npz"
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
                    chromosome=plan.block_chromosome[block_id],
                    regions={i: plan.regions[i] for i in plan.block_to_analyses[block_id]},
                    placement=plan.block_placement[tsv_path],
                    ceded=plan.ceded(block_id),
                )
            )

    if pending:
        print(f"  {n_existing:,} blocks already checkpointed, {len(pending):,} remaining")

    run_block_tasks(pending, n_workers, _run_block)


#: Fill records one Analysis batch's shard aims to hold: 2**28 records is 4 GiB,
#: read whole when the batch is assembled. A batch never splits an Analysis, so
#: one completed against more cells than this is a batch of its own. The shard
#: pass files each checkpoint's records one batch at a time, so its cost grows
#: with the number of batches: OGS-00006's 10.5B cells make about 39 of these.
_FILL_BATCH_CELLS = 1 << 28


@dataclass(frozen=True)
class _BlockRows:
    """One planned block's panel variants the merged axis holds, in block order:
    their merged-axis rows and their positions."""

    chromosome: str
    rows: np.ndarray
    positions: np.ndarray


def _block_rows(plan: _BlockPlan, axis: _MergedAxis) -> dict[str, _BlockRows]:
    """Each planned block's positions as merged-axis rows: the ALIDs the merged
    axis holds, in block order. An ALID it does not hold is no row of the
    completed Analysis, as before (`_merged_axis` drops what it cannot orient,
    and `_scan_block_maps` adds none outside every region)."""
    lookup = axis.new_alid_to_idx
    out: dict[str, _BlockRows] = {}
    for block_id, alids in plan.block_canonical_alids.items():
        held = [
            (k, lookup[alid]) for k, alid in enumerate(alids) if alid is not None and alid in lookup
        ]
        at = np.array([k for k, _row in held], dtype=np.int64)
        out[block_id] = _BlockRows(
            plan.block_chromosome[block_id],
            np.array([row for _k, row in held], dtype=np.int64),
            plan.block_positions[block_id][at],
        )
    return out


def _region_rows(plan: _BlockPlan, block_rows: dict[str, _BlockRows], ai: int) -> list[np.ndarray]:
    """One Analysis's merged-axis rows in each block it is completed against,
    inside its regions: the rows Phase 2 filled and counted for it, and the
    only ones completion gives it (issue #430)."""
    out = []
    for block_id in plan.analysis_to_blocks.get(ai, []):
        block = block_rows[block_id]
        out.append(block.rows[inside(plan.regions[ai], block.chromosome, block.positions)])
    return out


def _analysis_batches(
    plan: _BlockPlan, block_rows: dict[str, _BlockRows], budget: int = _FILL_BATCH_CELLS
) -> np.ndarray:
    """Each Analysis's fill-shard batch: consecutive Analyses, as many as fit
    `budget` cells. An Analysis has at most one fill per position of each block
    it is completed against, so that count bounds its shard's records."""
    batch_of = np.zeros(plan.state.n_analyses, dtype=np.int64)
    batch, held = 0, 0
    for ai in range(plan.state.n_analyses):
        cells = sum(len(rows) for rows in _region_rows(plan, block_rows, ai))
        if held and held + cells > budget:
            batch, held = batch + 1, 0
        batch_of[ai] = batch
        held += cells
    return batch_of


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
    eaf_reference: np.ndarray | None,
) -> _CompletedCsr:
    """Phases 2-3 together: run the pending LD blocks, shard their checkpoints'
    fills by Analysis batch on disk (writing `completion_quality` as it goes)
    and assemble each Analysis from its batch's shard, into a writer of a
    completion whose imputed cells read `eaf_reference`. ADR 0030 keeps the
    fine-grained completion_quality table SQLite-only, so it is created here.

    The shard pass is Dense's (`fill_shards.shard_checkpoint_fills`), keyed by
    Analysis batch rather than row band, so neither the fills nor the completed
    rows of the whole Store are ever held at once (issue #424)."""
    print("Writing index.sqlite...")
    with closing(staged.index_connection()) as dst_db:
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
    print("Merging block results from checkpoints...")
    union_alids_s, union_rows_s = fill_shards.sorted_union_map(axis.new_alid_to_idx)
    block_rows = _block_rows(plan, axis)
    batch_of = _analysis_batches(plan, block_rows)
    # No impute_mask here: Phase 1 is where the ancestry-match filter applies
    # (ADR 0028), since an Analysis it excludes is given no block, so its
    # checkpoints hold neither fills nor completion_quality rows to drop.
    fill_shard_dir, quality_count = fill_shards.shard_checkpoint_fills(
        checkpoint_dir / "blocks",
        staged,
        union_alids_s,
        union_rows_s,
        len(axis.variants),
        fill_shards.by_analysis_batch(batch_of),
    )
    print(f"Wrote {quality_count:,} completion quality rows")
    csr = _assemble_completed_csr(plan, axis, block_rows, fill_shard_dir, batch_of, eaf_reference)
    shutil.rmtree(fill_shard_dir)
    print(f"Completion done: {csr.total_imputed:,} imputed, {csr.total_missing:,} missing")
    return csr


def _batch_fills(fill_shard_dir: Path, batch: int) -> dict[int, np.ndarray]:
    """One batch shard's fill records, by Analysis, each in the order the
    checkpoints gave them (checkpoint order, then block order)."""
    shard = fill_shards.fill_shard_path(fill_shard_dir, batch)
    parts = list(fill_shards.iter_fill_records(shard))
    if not parts:
        return {}
    records = np.concatenate(parts)
    records = records[np.argsort(records["ai"], kind="stable")]
    analyses, starts = np.unique(records["ai"], return_index=True)
    ends = np.append(starts[1:], len(records))
    return {int(a): records[s:e] for a, s, e in zip(analyses, starts, ends, strict=True)}


def _assemble_completed_csr(
    plan: _BlockPlan,
    axis: _MergedAxis,
    block_rows: dict[str, _BlockRows],
    fill_shard_dir: Path,
    batch_of: np.ndarray,
    eaf_reference: np.ndarray | None,
) -> _CompletedCsr:
    """Phase 3 assembly: each Analysis's completed rows -- remapped observed
    rows, its fills and its missing block positions, sorted by variant index --
    added to the writer in Analysis order, one batch's fill shard read at a
    time. The writer judges each Analysis whole (`storable`); as a completion's
    (`ReferenceCompletion`), it logs the SEs above `float16`'s range once, as
    it writes, naming every Analysis."""
    print("Assembling completed CSR per analysis...")
    analysis_ids = [a.analysis_id for a in plan.state.src_analyses]
    writer = RaggedCSRWriter(
        len(axis.variants),
        ReferenceCompletion(tuple(analysis_ids), eaf_reference, where="complete-ragged"),
    )
    total_imputed = total_missing = 0
    no_fills = np.empty(0, dtype=fill_shards.FILL_RECORD_DTYPE)
    for batch in range(int(batch_of[-1]) + 1 if len(batch_of) else 0):
        fills = _batch_fills(fill_shard_dir, batch)
        for ai in np.flatnonzero(batch_of == batch).tolist():
            completed = _completed_analysis(
                ai,
                source=plan.source,
                src_alids=plan.state.src_alids,
                axis=axis,
                reference=_region_rows(plan, block_rows, ai),
                fills=fills.get(ai, no_fills),
            )
            writer.add_analysis(
                completed.vi, completed.z, completed.se, eaf=completed.eaf,
                imputed=completed.imp, where=f"complete-ragged, Analysis {analysis_ids[ai]}",
            )
            total_imputed += completed.n_imputed
            total_missing += completed.n_missing
            if (ai + 1) % 500 == 0:
                print(
                    f"  {ai + 1:,} / {plan.state.n_analyses:,} analyses | "
                    f"imputed {total_imputed:,} | missing {total_missing:,}"
                )
    return _CompletedCsr(writer=writer, total_imputed=total_imputed, total_missing=total_missing)


def _observed_rows(
    ai: int, source: CsrReader, src_alids: list[str], src_idx_remap: np.ndarray
) -> tuple[Associations, np.ndarray]:
    """One Analysis's observed rows, each variant's duplicates resolved by D5
    (`_admitted_observed`), and the merged-axis row of each.

    D1: the reader refuses a source cell the presence rule calls invalid,
    whatever completes the Analysis."""
    held = source.read(Cells.analysis(ai))
    obs, found = _admitted_observed(held)
    _warn_observed_duplicates(ai, held, found, src_alids)
    return obs, src_idx_remap[np.asarray(obs.variant_index, dtype=np.int64)]


def _completed_analysis(
    ai: int,
    *,
    source: CsrReader,
    src_alids: list[str],
    axis: _MergedAxis,
    reference: list[np.ndarray],
    fills: np.ndarray,
) -> _AnalysisCompleted:
    """One Analysis's completed rows over the merged axis, in variant order.

    Its rows are its block positions (`reference`, merged-axis rows of each
    block it is completed against, inside its regions) and its observed rows. An observed row is
    carried across untouched, a stored missing one too (ADR 0036); a block
    position the source holds no row at takes its fill, an imputed cell whose
    frequency is deferred to `eaf_reference` (ADR 0037 §4), or stays a
    missing (NaN) row. An Analysis
    no completable block touches is its source association list, remapped.
    """
    obs, obs_rows = _observed_rows(ai, source, src_alids, axis.src_idx_remap)
    in_reference = np.zeros(len(axis.variants), dtype=bool)
    for rows_of_block in reference:
        in_reference[rows_of_block] = True
    observed = np.zeros(len(axis.variants), dtype=bool)
    observed[obs_rows] = True
    held = in_reference | observed
    rows = np.flatnonzero(held)
    z, se, eaf = (np.full(len(rows), np.nan, dtype=np.float32) for _ in range(3))
    at = np.searchsorted(rows, obs_rows)
    z[at], se[at], eaf[at] = obs.z, obs.se, obs.eaf
    imp = np.zeros(len(rows), dtype=np.uint8)
    if len(fills):
        # A fill lands on a block position the source holds no row at: never on
        # an observed row, a stored missing one included, which is carried as
        # it stands. Its position among the Analysis's rows is a lookup over
        # the axis, as `searchsorted` over millions of fills is not.
        landed = (in_reference & ~observed)[fills["row"]]
        position = np.cumsum(held) - 1
        _apply_fills(ai, z, se, imp, fills[landed], position[fills["row"][landed]], axis)
    n_imputed = int(np.count_nonzero(imp))
    return _AnalysisCompleted(
        vi=rows.astype(np.int32),
        z=z,
        se=se,
        eaf=eaf,
        imp=imp,
        n_imputed=n_imputed,
        n_missing=int(np.count_nonzero(in_reference[rows] & ~observed[rows])) - n_imputed,
    )


def _apply_fills(
    ai: int,
    z: np.ndarray,
    se: np.ndarray,
    imp: np.ndarray,
    fills: np.ndarray,
    lr: np.ndarray,
    axis: _MergedAxis,
) -> None:
    """Land one Analysis's fills, at positions `lr` among its rows, on the
    rows the source holds no row at, by the shared duplicate rule (D5,
    `fill_shards.apply_fill_batch`): fills two blocks agree on are one, and
    fills they disagree on are all dropped, the cell left missing and every
    dropped fill logged (D8, `fill_shards.log_dropped`).

    All of the Analysis's fills are one batch, so the rule compares them all at
    once, as Ragged completion always did."""
    column = np.zeros(len(lr), dtype=np.int64)
    fill_shards.apply_fill_batch(
        z[:, None], se[:, None], imp[:, None], lr, column, fills["z"], fills["se"]
    )
    conflicted = imp == fill_shards.CONFLICTED
    if not conflicted.any():
        return
    dropped = fills[conflicted[lr]]
    names = [axis.variants[int(row)].alid for row in dropped["row"].tolist()]
    log.warning(
        "Analysis %d: %d fill(s) at %d variant(s) disagree across LD blocks, so none is kept "
        "(duplicate_conflicting); first: %s",
        ai, len(dropped), int(conflicted.sum()), names[:5],
    )
    fill_shards.log_dropped("fill", (
        (ai, name, z, s) for name, z, s in zip(
            names, dropped["z"].tolist(), dropped["se"].tolist(), strict=True
        )
    ))
    fill_shards.clear_conflicted(z, se, imp, conflicted)


def _encode_plan(
    src: Path,
    ld_dir: Path,
    ancestry: str,
    manifest: StoreManifest,
    axis: _MergedAxis,
    n_analyses: int,
) -> _EncodePlan:
    """The completed release's encoding: its source's plan, since completion
    writes into the source's arrays (ADR 0038 §4) -- with `eaf_reference`
    added when the panel has frequencies (ADR 0037 §4, issue #113), the
    source's EAF baseline carried across the variant remap (ADR 0037 §2) and,
    under a residual SE plan, the source's SE model (issue #301)."""
    # Asked for whatever the source declares, `absent` included: a release
    # whose Analyses reported no frequency still gains imputed cells, and
    # those cells have the panel's frequency (issue #113). It gets an
    # `eaf_reference` array and no `eaf` plane -- NaN on every observed
    # cell, the panel's value on every imputed one.
    eaf_reference = panel_reference_eaf(ld_dir, ancestry, axis.variants)
    encoding = manifest.encoding.with_eaf_reference(eaf_reference is not None)
    src_baseline = _source_eaf_baseline(src)
    out_baseline = None
    if src_baseline is not None:
        out_baseline = np.full(len(axis.variants), np.nan, dtype=np.float32)
        out_baseline[axis.src_idx_remap] = src_baseline
    return _EncodePlan(
        encoding=encoding,
        eaf_reference=eaf_reference,
        out_baseline=out_baseline,
        se_coefficients=(
            _source_se_coefficients(src, n_analyses) if encoding.se.is_residual else None
        ),
    )


def _se_exception_record(counts: SeExceptionCounts, analysis_ids: list[str]) -> dict[str, Any]:
    """Each Analysis's share of SEs held as exact exceptions, in `analyses.tsv`
    order, and the Analyses above the 2% budget (spec §6a, issue #301).

    The budget is a build's criterion for choosing a residual range. Completion
    keeps its source's range and model, so it measures the share it produced
    rather than enforcing it: the record and `_report_se_exceptions` are how a
    breach by imputed cells becomes visible. An Analysis with no SE has no
    share, recorded as null rather than 0. The writer counts as it codes each
    region (`SeExceptionCounts`).
    """
    n_se, n_exceptions = counts.n_se.tolist(), counts.n_exceptions.tolist()
    rates = [
        exact / n if n else None for exact, n in zip(n_exceptions, n_se, strict=True)
    ]
    analyses = [
        {
            "analysis_id": analysis_id,
            "n_se": int(n_se[i]),
            "n_exceptions": int(n_exceptions[i]),
            "n_imputed_se": int(counts.n_imputed_se[i]),
            "n_imputed_exceptions": int(counts.n_imputed_exceptions[i]),
            "exception_rate": rates[i],
        }
        for i, analysis_id in enumerate(analysis_ids)
    ]
    over = [
        analysis_id
        for analysis_id, rate in zip(analysis_ids, rates, strict=True)
        if rate is not None and rate > SE_EXCEPTION_BUDGET
    ]
    return {"budget": SE_EXCEPTION_BUDGET, "analyses": analyses, "over_budget": over}


def _report_se_exceptions(record: dict[str, Any]) -> None:
    """The exception tally on the console, and each Analysis over budget by name."""
    analyses = record["analyses"]
    total = sum(entry["n_exceptions"] for entry in analyses)
    imputed = sum(entry["n_imputed_exceptions"] for entry in analyses)
    cells = sum(entry["n_se"] for entry in analyses)
    print(
        f"SE exceptions: {total:,} of {cells:,} SEs ({imputed:,} on imputed cells); "
        f"{len(record['over_budget']):,} Analyses over the {record['budget']:.0%} budget"
    )
    over = [entry for entry in analyses if entry["analysis_id"] in set(record["over_budget"])]
    if not over:
        return
    log.warning(
        "complete-ragged: %s Analyses hold more than %s of their SEs as exact exceptions, "
        "the residual budget (spec §6a); the source's SE model and range are kept, never "
        "refitted or narrowed to float16 (issue #301): %s",
        f"{len(over):,}",
        f"{record['budget']:.0%}",
        "; ".join(
            f"{entry['analysis_id']}: {entry['exception_rate']:.2%} "
            f"({entry['n_exceptions']:,} of {entry['n_se']:,}, "
            f"{entry['n_imputed_exceptions']:,} imputed)"
            for entry in over
        ),
    )


def _write_completed_zarr(
    staged: StagedRelease,
    encode_plan: _EncodePlan,
    csr: _CompletedCsr,
    analyses: list[Analysis],
    n_variants: int,
) -> dict[str, Any] | None:
    """Phase 4: write the completed CSR as the store's ragged zarr group, then
    its variant-centric index (ADR 0060). Returns the SE exception record of a
    residual plane, reported here (issue #301).

    Written through the builders' writer, region by region (issue #424), with
    the source's plan: completion writes into the source's arrays, so it
    encodes with the source's plan (ADR 0038 §4), the one addition being
    `eaf_reference`. Observed frequencies only: an imputed cell's frequency is
    the panel's, stored once per variant and applied on read (ADR 0037 §4).
    The per-variant baseline travels with the values across the variant remap
    rather than being recomputed from them, so a value decoded from the source
    re-encodes to the same code.

    A residual SE plane is coded against the source's `se_coefficients`, never
    a refit (issue #301): a refit re-points every observed cell's code at a new
    model, which moved all 135,596 observed SEs of three Chen 2023 Analyses.
    Each observed cell is carried (`ReferenceCompletion`), so it decodes to exactly
    its source value, and an imputed cell outside the range is an exact
    exception. That may take an Analysis past the 2% budget; it is recorded and
    named, never met by a refit or a `float16` plane, either of which would
    move observed values.

    The index is rebuilt from the completed planes, never carried over from the
    observed source: the completion remaps the variant axis and adds imputed
    cells, so a copied index would answer with the source's rows.
    """
    print("Writing zarr CSR...")
    counts = csr.writer.flush(
        staged.path,
        encode_plan.encoding,
        eaf_baseline=encode_plan.out_baseline,
        se_coefficients=encode_plan.se_coefficients,
    )
    se_exceptions = (
        None
        if counts is None
        else _se_exception_record(counts, [analysis.analysis_id for analysis in analyses])
    )
    if se_exceptions is not None:
        _report_se_exceptions(se_exceptions)
    print("Building variant-centric index...")
    build_variant_index(staged.path, n_axis=n_variants)
    return se_exceptions


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
        staged.path / "analyses.tsv", top_hits.add_hit_counts(staged.path, dst_analyses)
    )


def _completed_manifest(
    manifest: StoreManifest,
    *,
    staged: StagedRelease,
    source_format_version: str,
    release_id: str | None,
    ld_panel_id: str,
    ancestry: str,
    min_cor: float,
    thresh: float,
    cis_window_bp: int,
    axis: _MergedAxis,
    panel_placement: dict[str, dict[str, object]],
    csr: _CompletedCsr,
    encode_plan: _EncodePlan,
    se_exceptions: dict[str, Any] | None,
) -> StoreManifest:
    """Phase 6 finalization: the completed release's manifest, with its
    provenance recording what completion added to its source -- and, for a
    residual SE plane, each Analysis's exception share (issue #301)."""
    return completed_release_manifest(
        manifest,
        encoding=encode_plan.encoding,
        release_id=release_id,
        source_format_version=source_format_version,
        completion_provenance=build_completion_provenance(
            ld_panel_id=ld_panel_id,
            ancestry=ancestry,
            min_cor=min_cor,
            thresh=thresh,
            n_variants_total=len(axis.variants),
            n_variants_new=len(axis.new_variants),
            cis_window_bp=cis_window_bp,
            n_imputed=csr.total_imputed,
            n_missing=csr.total_missing,
            **panel_placement,
            **({} if se_exceptions is None else {"se_exceptions": se_exceptions}),
        ),
        by_variant=recorded_provenance(staged.path),
    )


def _uncovered_line(uncovered: tuple[str, ...]) -> str:
    """The regions completed nowhere, by count and `region_id`."""
    return (
        f"Regions overlapping no LD block of the panel, completed nowhere: {len(uncovered):,}"
        + (f" ({', '.join(uncovered)})" if uncovered else "")
    )


def _completion_result(
    dst: Path, state: _SourceState, axis: _MergedAxis, csr: _CompletedCsr, plan: _BlockPlan
) -> CompletionResult:
    result = CompletionResult(
        output_path=dst,
        n_variants=len(axis.variants),
        n_analyses=state.n_analyses,
        n_associations=csr.writer.n_associations,
        n_imputed=csr.total_imputed,
        n_missing=csr.total_missing,
        regions_without_block=plan.uncovered,
    )
    print(
        f"Reference completion complete: {result.n_variants:,} variants, "
        f"{result.n_analyses:,} analyses, {result.n_associations:,} associations "
        f"({result.n_imputed:,} imputed, {result.n_missing:,} missing)"
    )
    print(_uncovered_line(result.regions_without_block))
    return result


def _run_completion(
    src: Path,
    dst: Path,
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
    regions: dict[int, tuple[Region, ...]],
) -> CompletionResult:
    """The shared core: block discovery, checkpointed LD-block completion and
    CSR assembly, encoding, metadata and manifest, sequenced inside the
    staging context that makes a completed release atomic."""
    manifest, source_format_version = _open_source(src)

    with OpenGWASDBStore.staging(dst, overwrite=True) as staged:
        state = _read_source_state(src, impute_analysis_ids)
        plan = _plan_blocks(state, ld_dir, ancestry, regions)
        axis = _merged_axis(state, plan.new_alids)
        _write_variant_axis(staged.path, state, axis)
        encode_plan = _encode_plan(src, ld_dir, ancestry, manifest, axis, state.n_analyses)
        csr = _checkpoint_and_assemble(
            staged, plan, axis,
            checkpoint_dir=checkpoint_dir,
            n_workers=n_workers,
            min_cor=min_cor,
            thresh=thresh,
            region_cap_bp=region_cap_bp,
            eaf_reference=encode_plan.eaf_reference,
        )
        se_exceptions = _write_completed_zarr(
            staged, encode_plan, csr, state.src_analyses, len(axis.variants)
        )
        _write_top_hits_and_analyses(staged, state, encode_plan, ancestry=ancestry)
        staged.write_manifest(
            _completed_manifest(
                manifest,
                staged=staged,
                source_format_version=source_format_version,
                release_id=release_id,
                ld_panel_id=ld_panel_id,
                ancestry=ancestry,
                min_cor=min_cor,
                thresh=thresh,
                cis_window_bp=cis_window_bp,
                axis=axis,
                panel_placement=plan.panel_placement,
                csr=csr,
                encode_plan=encode_plan,
                se_exceptions=se_exceptions,
            )
        )
        return _completion_result(dst, state, axis, csr, plan)


def _source_eaf_baseline(source_path: Path) -> np.ndarray | None:
    """The observed release's per-variant `eaf_baseline`, or None if it has none.

    Read straight from the source's CSR group rather than recomputed from the
    values it holds: recomputing from *decoded* frequencies would shift each
    baseline by up to half a step and re-quantise every cell against the moved
    baseline, so a completed release would be less accurate than its source for
    no reason (ADR 0037 §2).
    """
    group = store_arrays.open_group(Path(source_path) / RAGGED_ZARR_PATH)
    if EAF_BASELINE not in group:
        return None
    return np.asarray(group[EAF_BASELINE][:], dtype=np.float32)


def _source_se_coefficients(source_path: Path, n_analyses: int) -> np.ndarray:
    """The observed release's `se_coefficients`, which every completed cell is
    coded against (issue #301).

    Read, like the EAF baseline, rather than refitted: a model fitted to the
    completed rows decodes every observed code to a value its source never
    held. A residual source without a finite `(n_analyses, 2)` model cannot be
    completed in its own encoding (ADR 0038 §4), so it is refused here.
    """
    group = store_arrays.open_group(Path(source_path) / RAGGED_ZARR_PATH)
    if SE_COEFFICIENTS not in group:
        raise ValueError(
            f"{source_path}: its se plane is int8_residual but it holds no se_coefficients"
        )
    coefficients = np.asarray(group[SE_COEFFICIENTS][:], dtype=np.float32)
    if coefficients.shape != (n_analyses, 2) or not np.all(np.isfinite(coefficients)):
        raise ValueError(
            f"{source_path}: se_coefficients must be finite with shape ({n_analyses}, 2), "
            f"not {coefficients.shape}"
        )
    return coefficients
