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
from opengwasdb.completion.checkpoint import BlockCompletionResult, FillRow, QualityRow
from opengwasdb.completion.impute import impute_z_block, scalar_n_se
from opengwasdb.completion.ld_panel import (
    LDBlock,
    canonical_panel_alid,
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


@dataclass(frozen=True)
class BlockAnalysisResult:
    """Result of completing one LD block for one Analysis.

    ``pearson_r`` is ``None`` when imputation was never attempted (fewer than
    two observed positions in the block) and a float (possibly NaN) once it
    was -- the same three-way distinction the ``completion_quality`` table
    records. ``fills`` holds only positions actually filled, as
    ``(local_block_index, z, se)`` -- the position's index into the block's
    own SNP list, not a store-wide variant index, since that mapping is the
    caller's responsibility (dense resolves it against the union axis by
    ALID; ragged resolves it the same way). ``n_missing`` is the count of
    block positions still unfilled after this call (the full block-missing
    count when nothing could be imputed).
    """

    pearson_r: float | None
    fills: list[tuple[int, float, float]]
    n_missing: int


def _region_capped_z(
    zv: float,
    pos: int,
    obs_pos_sorted: np.ndarray,
    obs_absz_sorted: np.ndarray,
    window_bp: int = REGION_CAP_BP,
) -> float:
    """Clamp an imputed z-score so its magnitude does not exceed the largest
    observed |z| within +/- ``window_bp`` of ``pos`` (QC, per pleiodb). Returns
    ``zv`` unchanged when there is no observed variant in the window.

    ``obs_pos_sorted`` must be ascending, with ``obs_absz_sorted`` the matching
    |z| of the observed variants.
    """
    lo = int(np.searchsorted(obs_pos_sorted, pos - window_bp, side="left"))
    hi = int(np.searchsorted(obs_pos_sorted, pos + window_bp, side="right"))
    if hi <= lo:
        return zv
    z_region_max = float(obs_absz_sorted[lo:hi].max())
    if abs(zv) <= z_region_max:
        return zv
    return z_region_max if zv >= 0 else -z_region_max


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
        return BlockAnalysisResult(pearson_r=None, fills=[], n_missing=n_miss_block)

    z_imp_arr, corr = impute_z_block(z_dense, eigenvectors, eigenvalues, min_cor=min_cor)
    if z_imp_arr is None:
        pearson_r = float(corr) if np.isfinite(corr) else None
        return BlockAnalysisResult(pearson_r=pearson_r, fills=[], n_missing=n_miss_block)

    se_all = scalar_n_se(se_dense[obs_mask], eaf[obs_mask], eaf)

    obs_pos_s = obs_absz_s = None
    if region_cap_bp is not None:
        obs_idx = np.where(obs_mask)[0]
        o_order = np.argsort(positions[obs_idx])
        obs_pos_s = positions[obs_idx][o_order]
        obs_absz_s = np.abs(z_dense[obs_idx])[o_order]

    fills: list[tuple[int, float, float]] = []
    candidates = np.where(~obs_mask)[0]
    # Only a cell Association Admission would store as present is a fill; the
    # rest stay missing and are counted `n_missing`.
    for i in candidates[admissible(z_imp_arr[candidates], se_all[candidates])].tolist():
        zv, sev = float(z_imp_arr[i]), float(se_all[i])
        if region_cap_bp is not None:
            zv = _region_capped_z(zv, int(positions[i]), obs_pos_s, obs_absz_s, region_cap_bp)
        fills.append((int(i), zv, sev))

    return BlockAnalysisResult(
        pearson_r=float(corr), fills=fills, n_missing=n_miss_block - len(fills)
    )


# An Analysis's observed z/se at one block's positions, as dense NaN-filled
# arrays aligned with the block's own SNP order -- the seam ``run_block``
# hands each layout to supply its own way of reading that.
ObservedReader = Callable[[int], "tuple[np.ndarray, np.ndarray]"]

# (block, canonical_alids) -> (the Analysis indices assigned to this block,
# a reader for each one's observed z/se). Built once per block, not once per
# Analysis, since opening the source is the expensive part.
BlockReaderFactory = Callable[[LDBlock, list[str | None]], "tuple[Iterable[int], ObservedReader]"]


def run_block(
    tsv_path: str | Path,
    thresh: float,
    min_cor: float,
    region_cap_bp: int | None,
    make_reader: BlockReaderFactory,
) -> BlockCompletionResult | None:
    """Complete one LD block for every Analysis its caller assigns to it.

    Returns ``None`` when the block itself cannot be loaded (caller writes no
    checkpoint in that case -- there is nothing to record). When the block's
    eigendecomposition cannot be loaded, returns an empty result (still
    checkpoint-worthy, so a later run doesn't keep retrying an unreadable
    block).
    """
    block = load_block(tsv_path)
    if block is None:
        return None

    try:
        eigenvalues, eigenvectors = load_ld_eigenvectors(block, thresh)
    except Exception as exc:  # noqa: BLE001
        log.warning("Block %s: cannot load eigenvectors (%s) — skipping", block.block_id, exc)
        return BlockCompletionResult(block_id=block.block_id, quality_rows=[], fills=[])

    # Canonical ALID per block variant (None if it doesn't parse); the LD
    # eigenvectors are indexed by block position, so every block SNP keeps
    # its slot even when unmatched.
    canonical_alids = [canonical_panel_alid(s) for s in block.snp_ids]
    eaf = block.eaf
    positions = np.array([_snp_position(s) for s in block.snp_ids], dtype=np.int64)

    analysis_indices, read_observed = make_reader(block, canonical_alids)

    quality_rows: list[QualityRow] = []
    fills: list[FillRow] = []
    for ai in analysis_indices:
        z_dense, se_dense = read_observed(ai)
        block_result = complete_block_for_analysis(
            z_dense, se_dense, eaf, positions, eigenvectors, eigenvalues,
            min_cor=min_cor, region_cap_bp=region_cap_bp,
        )
        n_missing = block_result.n_missing
        n_filled = 0
        for local_idx, zv, sev in block_result.fills:
            alid = canonical_alids[local_idx]
            if alid is None:
                n_missing += 1
                continue
            fills.append(FillRow(alid, ai, zv, sev))
            n_filled += 1
        quality_rows.append(QualityRow(ai, block_result.pearson_r, n_filled, n_missing))

    return BlockCompletionResult(block_id=block.block_id, quality_rows=quality_rows, fills=fills)
