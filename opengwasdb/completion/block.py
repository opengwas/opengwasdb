"""The per-LD-block Reference Completion runner.

``run_block`` owns the whole per-block task dense and ragged completion each
used to implement separately: load the block, load its eigendecomposition
(with the standard "cannot load -- write an empty checkpoint" failure path),
derive the block's canonical ALIDs/EAF/positions, run every assigned
Analysis's imputation through ``complete_block_for_analysis`` (including the
None-ALID bookkeeping), and assemble the block's ``completion_quality`` rows
and fills. The one thing that genuinely differs by layout -- how to read an
Analysis's observed z/se at this block's positions, and which Analyses are
even assigned to this block -- is a caller-supplied ``make_reader`` seam:
dense's reader is a matrix column slice from an already-opened source zarr
group; ragged's is a per-Analysis CSR lookup built into a dict. *Where the
imputed z/se get written back* stays with each caller's own Phase 3 (dense
band-writes into a zarr matrix; ragged assembles a CSR row), since that's a
storage-shape difference this module has no need to know about.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from opengwasdb.admission.cells import admissible
from opengwasdb.completion.checkpoint import BLOCK_FILL_DTYPE, BlockCompletionResult, QualityRow
from opengwasdb.completion.impute import impute_z_block, scalar_n_se
from opengwasdb.completion.ld_panel import (
    BlockPlacement,
    LDBlock,
    load_block,
    load_ld_eigenvectors,
)
from opengwasdb.completion.ld_panel import snp_position as _snp_position

log = logging.getLogger(__name__)

# The completion method recorded in a completed release's manifest
# (provenance["completion"]["method"]). Both layouts now run the identical
# elastic-net-on-eigenvectors kernel below, including the region z-cap QC, so
# they share one method string rather than recording two that imply a
# difference which no longer exists.
COMPLETION_METHOD = "elastic_net_eigenvectors_v2_regioncap"

# Region-based imputation z-cap (QC): an imputed |z| may not exceed the largest
# observed |z| within +/- this many bp of it (same LD block). Caps spurious
# large imputed z-scores from LD extrapolation. See pleiodb.
REGION_CAP_BP = 1_000_000

# One-dimensional prepared fills, in ascending local block index. Float64 keeps
# the kernel values exact until the checkpoint's established float32 cast.
PREPARED_FILL_DTYPE = np.dtype([("local_index", np.int64), ("z", np.float64), ("se", np.float64)])


@dataclass(frozen=True)
class BlockAnalysisResult:
    """Result of completing one LD block for one Analysis.

    ``pearson_r`` is ``None`` when imputation was never attempted (fewer than
    two observed positions in the block) and a float (possibly NaN) once it
    was -- the same three-way distinction the ``completion_quality`` table
    records. ``fills`` holds only positions actually filled, as
    a 1-D ``PREPARED_FILL_DTYPE`` array (int64 local index, float64 z/se) --
    the position's index into the block's
    own SNP list, not a store-wide variant index, since that mapping is the
    caller's responsibility (dense resolves it against the union axis by
    ALID; ragged resolves it the same way). ``n_missing`` is the count of
    block positions still unfilled after this call (the full block-missing
    count when nothing could be imputed).
    """

    pearson_r: float | None
    fills: np.ndarray
    n_missing: int


def _region_capped_array(
    z: np.ndarray, positions: np.ndarray, observed_positions: np.ndarray,
    observed_absz: np.ndarray, window_bp: int,
) -> np.ndarray:
    """Exact scalar region caps with range maxima over power-of-two spans.

    Two overlapping spans cover each inclusive window. Max only selects an
    existing observed value, so changing the reduction order changes no bits.
    Empty windows leave z untouched, including its sign when z is zero.
    """
    lo = np.searchsorted(observed_positions, positions - window_bp, side="left")
    hi = np.searchsorted(observed_positions, positions + window_bp, side="right")
    widths = hi - lo
    caps = np.full(len(z), np.inf)
    maxima = observed_absz
    span = 1
    while span <= len(observed_absz):
        use = (widths >= span) & (widths < 2 * span)
        caps[use] = np.maximum(maxima[lo[use]], maxima[hi[use] - span])
        if 2 * span > len(observed_absz):
            break
        maxima = np.maximum(maxima[:-span], maxima[span:])
        span *= 2
    return np.where(np.abs(z) <= caps, z, np.copysign(caps, z))


def complete_block_for_analysis(
    z_dense: np.ndarray,
    se_dense: np.ndarray,
    eaf: np.ndarray,
    positions: np.ndarray,
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    *,
    min_cor: float,
    region_cap_bp: int | None = REGION_CAP_BP,
) -> BlockAnalysisResult:
    """Impute one Analysis's missing positions within one LD block.

    Parameters
    ----------
    z_dense, se_dense : float64, length = n_ld_snps_in_block.
        NaN at positions not observed for this Analysis. Both must share the
        same missingness pattern (a validated store invariant).
    eaf : float64, length = n_ld_snps_in_block. Reference-panel EAF.
    positions : int64, length = n_ld_snps_in_block. Base-pair position of
        each block SNP, for the region z-cap.
    eigenvectors, eigenvalues : the block's LD eigendecomposition.
    min_cor : Pearson r quality gate (see ``impute_z_block``).
    region_cap_bp : window for the region z-cap QC, or ``None`` to disable it.
    """
    obs_mask = np.isfinite(z_dense)
    n_obs = int(obs_mask.sum())
    n_miss_block = int((~obs_mask).sum())
    if n_obs < 2:
        return BlockAnalysisResult(None, np.empty(0, dtype=PREPARED_FILL_DTYPE), n_miss_block)

    z_imp_arr, corr = impute_z_block(z_dense, eigenvectors, eigenvalues, min_cor=min_cor)
    if z_imp_arr is None:
        pearson_r = float(corr) if np.isfinite(corr) else None
        return BlockAnalysisResult(pearson_r, np.empty(0, dtype=PREPARED_FILL_DTYPE), n_miss_block)

    se_all = scalar_n_se(se_dense[obs_mask], eaf[obs_mask], eaf)

    candidates = np.flatnonzero(~obs_mask)
    candidates = candidates[admissible(z_imp_arr[candidates], se_all[candidates])]
    fills = np.empty(len(candidates), dtype=PREPARED_FILL_DTYPE)
    fills["local_index"] = candidates
    fills["z"] = z_imp_arr[candidates]
    fills["se"] = se_all[candidates]
    if region_cap_bp is not None:
        obs_idx = np.flatnonzero(obs_mask)
        o_order = np.argsort(positions[obs_idx])
        obs_pos_s = positions[obs_idx][o_order]
        obs_absz_s = np.abs(z_dense[obs_idx])[o_order]
        fills["z"] = _region_capped_array(
            fills["z"], positions[candidates], obs_pos_s, obs_absz_s, region_cap_bp
        )

    return BlockAnalysisResult(
        pearson_r=float(corr), fills=fills, n_missing=n_miss_block - len(fills)
    )


def panel_positions(block: LDBlock) -> np.ndarray:
    """Each of a block's panel variants' positions, in block order: the positions
    `run_block` fits and caps by, and Ragged completion bounds to an Analysis's
    regions (`completion.regions.inside`)."""
    return np.array([_snp_position(s) for s in block.snp_ids], dtype=np.int64)


# An Analysis's observed z/se at one block's positions, as dense NaN-filled
# arrays aligned with the block's own SNP order -- the seam ``run_block``
# hands each layout to supply its own way of reading that.
ObservedReader = Callable[[int], "tuple[np.ndarray, np.ndarray]"]

# (block, canonical_alids) -> (the Analysis indices assigned to this block,
# a reader for each one's observed z/se). Built once per block, not once per
# Analysis, since opening the source is the expensive part.
BlockReaderFactory = Callable[[LDBlock, list[str | None]], "tuple[Iterable[int], ObservedReader]"]


def read_as_planned(planned: BlockPlacement, make_reader: BlockReaderFactory) -> BlockReaderFactory:
    """``make_reader``, once the block a worker has read names its rows as the
    read the completion was planned from did (`BlockPlacement.require`)."""

    def checked(
        block: LDBlock, canonical_alids: list[str | None]
    ) -> tuple[Iterable[int], ObservedReader]:
        planned.require(block)
        return make_reader(block, canonical_alids)

    return checked


# The canonical ALIDs a block leaves to the block that follows it, for one
# Analysis (`completion.ld_panel.shared_endpoints`).
CededAlids = Callable[[int], "frozenset[str]"]

# (Analysis, the block's positions in block order) -> a mask of the rows outside
# every region the Analysis is completed in (Ragged, issue #430). By position, so
# it leaves out a row no ALID names as well as a variant.
OutsideRegions = Callable[[int, np.ndarray], np.ndarray]


def run_block(
    tsv_path: str | Path,
    thresh: float,
    min_cor: float,
    region_cap_bp: int | None,
    make_reader: BlockReaderFactory,
    ceded: CededAlids | None = None,
    outside: OutsideRegions | None = None,
) -> BlockCompletionResult | None:
    """Complete one LD block for every Analysis its caller assigns to it.

    Returns ``None`` when the block itself cannot be loaded (caller writes no
    checkpoint in that case -- there is nothing to record). When the block's
    eigendecomposition cannot be loaded, returns an empty result (still
    checkpoint-worthy, so a later run doesn't keep retrying an unreadable
    block).

    ``ceded`` names, per Analysis, the variants at this block's end the block starting
    there owns (ADR 0072); ``outside`` marks its rows outside its regions (#430). The
    block fits both but fills and counts neither, so a ceded cell is counted once.
    """
    block = load_block(tsv_path)
    if block is None:
        return None

    try:
        eigenvalues, eigenvectors = load_ld_eigenvectors(block, thresh)
    except Exception as exc:  # noqa: BLE001
        log.warning("Block %s: cannot load eigenvectors (%s) — skipping", block.block_id, exc)
        return BlockCompletionResult(block_id=block.block_id, quality_rows=[])

    return _complete_loaded_block(
        block, eigenvalues, eigenvectors, min_cor, region_cap_bp, make_reader, ceded, outside
    )


def _complete_loaded_block(
    block: LDBlock,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    min_cor: float,
    region_cap_bp: int | None,
    make_reader: BlockReaderFactory,
    ceded: CededAlids | None,
    outside: OutsideRegions | None,
) -> BlockCompletionResult:
    """Prepare and own each assigned Analysis's fills on the loaded panel axis."""
    # Canonical ALID per block variant (None where it names none); the LD
    # eigenvectors are indexed by block position, so every block SNP keeps
    # its slot even when unmatched. A panel codes its LD and EAF on each row's
    # EA, and the Store signs a statistic for its ALID's A1, so a row whose A1
    # is not its EA -- a panel deletion, spelt reference first -- is re-signed
    # (ADR 0079).
    canonical_alids = block.alids
    eigenvectors = block.oriented_eigenvectors(eigenvectors)
    eaf = block.eaf_for_a1()
    positions = panel_positions(block)

    analysis_indices, read_observed = make_reader(block, canonical_alids)

    quality_rows: list[QualityRow] = []
    fills: list[np.ndarray] = []
    named = np.array([alid is not None for alid in canonical_alids], dtype=bool)
    left_alone = _left_alone(canonical_alids, positions, ceded, outside)
    for ai in analysis_indices:
        z_dense, se_dense = read_observed(ai)
        block_result = complete_block_for_analysis(
            z_dense, se_dense, eaf, positions, eigenvectors, eigenvalues,
            min_cor=min_cor, region_cap_bp=region_cap_bp,
        )
        analysis_fills, quality = _owned_rows(
            ai, block_result, named, left_alone(ai), np.isnan(z_dense)
        )
        fills.append(analysis_fills)
        quality_rows.append(quality)

    return BlockCompletionResult(
        block_id=block.block_id, quality_rows=quality_rows, alids=canonical_alids,
        fills=np.concatenate(fills) if fills else np.empty(0, dtype=BLOCK_FILL_DTYPE),
    )


def _left_alone(
    canonical_alids: list[str | None],
    positions: np.ndarray,
    ceded: CededAlids | None,
    outside: OutsideRegions | None,
) -> Callable[[int], np.ndarray | None]:
    """Per Analysis, the block rows it neither fills nor counts: those no ALID names,
    which are never a Store row (the owner's ruling of 11 Nov 2026, ADR 0039), those
    it cedes, and those outside its regions; None where there are none."""
    named = np.array([alid is not None for alid in canonical_alids], dtype=bool)
    unnamed = None if named.all() else ~named
    ceded_mask = _ceded_masks(canonical_alids, ceded)

    def mask(analysis: int) -> np.ndarray | None:
        masks = [ceded_mask(analysis), unnamed]
        if outside is not None:
            masks.append(outside(analysis, positions))
        present = [m for m in masks if m is not None]
        return np.logical_or.reduce(present) if present else None

    return mask


def _ceded_masks(
    canonical_alids: list[str | None], ceded: CededAlids | None
) -> Callable[[int], np.ndarray | None]:
    """Per Analysis, which block positions it cedes; None where it cedes none.

    Built once per distinct set: a Dense block cedes the same for every Analysis.
    """
    masks: dict[frozenset[str], np.ndarray | None] = {frozenset(): None}

    def mask(analysis: int) -> np.ndarray | None:
        left = frozenset() if ceded is None else ceded(analysis)
        if left not in masks:
            masks[left] = np.array([alid in left for alid in canonical_alids])
        return masks[left]

    return mask


def _owned_rows(
    ai: int,
    result: BlockAnalysisResult,
    named: np.ndarray,
    ceded: np.ndarray | None,
    missing: np.ndarray,
) -> tuple[np.ndarray, QualityRow]:
    """One Analysis's array fills and quality, leaving unowned positions alone.

    Ceded, out-of-region and unnamed positions count as neither filled nor
    missing here (ADR 0072 and the owner's 11 Nov 2026 ruling, ADR 0039).
    """
    n_missing = result.n_missing
    prepared = result.fills
    if ceded is not None:
        keep = ~ceded[prepared["local_index"]]
        n_missing += int((~keep).sum()) - int((ceded & missing).sum())
        prepared = prepared[keep]
    local = prepared["local_index"]
    unnamed = ~named[local]
    if unnamed.any():
        raise AssertionError(
            f"block position {local[unnamed][0]} names no ALID but was not left alone"
        )
    fills = np.empty(len(prepared), dtype=BLOCK_FILL_DTYPE)
    fills["local_index"] = local
    fills["analysis_index"] = ai
    fills["z"] = prepared["z"]
    fills["se"] = prepared["se"]
    return fills, QualityRow(ai, result.pearson_r, len(fills), n_missing)
