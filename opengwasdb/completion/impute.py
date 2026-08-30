"""LD-based z-score imputation kernel for Ragged Reference Completion.

Ported from pleiodb/src/pleiodb/impute.py (elastic-net on LD eigenvectors).
SE imputation uses the scalar-N model from the LD panel EAF.
"""
from __future__ import annotations

import logging

import numpy as np
import scipy.linalg
from sklearn.linear_model import ElasticNetCV

from opengwasdb.build.phenotype_sd import se_scale_constant

log = logging.getLogger(__name__)


def ld_pca(
    ld_matrix: np.ndarray,
    thresh: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecompose an LD matrix; return (eigenvalues, eigenvectors) in descending order.

    Truncated to the minimum number of components whose cumulative variance reaches *thresh*.
    """
    vals, vecs = scipy.linalg.eigh(ld_matrix)
    vals = vals[::-1]
    vecs = vecs[:, ::-1]
    vals = np.maximum(vals, 0.0)
    total = float(vals.sum()) or 1.0
    cumvar = np.cumsum(vals) / total
    n_comp = int(np.searchsorted(cumvar, thresh)) + 1
    n_comp = min(n_comp, len(vals))
    return vals[:n_comp], vecs[:, :n_comp]


def elastic_net_impute(
    z: np.ndarray,
    eigenvectors: np.ndarray,
    n_comp: int,
) -> np.ndarray | None:
    """Predict z-scores for all positions using elastic-net on LD eigenvectors.

    Fits on observed (non-NaN) rows; predicts for every row.
    Returns None when fitting is not possible (< 2 observed or all z identical).
    """
    mask_obs = np.isfinite(z)
    if mask_obs.sum() < 2:
        return None

    E = eigenvectors[:, :n_comp]
    z_obs = z[mask_obs]

    if len(np.unique(z_obs)) <= 1:
        return None

    n_obs = int(mask_obs.sum())
    cv = min(5, max(2, n_obs - 1))
    try:
        model = ElasticNetCV(l1_ratio=0.5, fit_intercept=False, cv=cv, max_iter=2000)
        model.fit(E[mask_obs], z_obs)
        return model.predict(E).astype(np.float64)
    except Exception as exc:  # noqa: BLE001
        log.debug("elastic_net_impute failed: %s", exc)
        return None


def _ratio_outlier_mask(truth: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Boolean keep-mask (True = not an outlier) based on two-pass ratio SD threshold."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(predicted != 0, truth / predicted, np.nan)

    keep = np.ones(len(ratio), dtype=bool)
    for _ in range(2):
        r = ratio[keep]
        if len(r) < 2:
            break
        m = np.nanmedian(r)
        s = np.nanstd(r)
        if s == 0:
            break
        keep[keep] = np.abs(r - m) <= 3 * s
    return keep


def min_observed_points(npoly: int = 3) -> int:
    """Fewest observed points a block needs before imputation can succeed.

    `poly_rescale` fits a degree-`npoly` polynomial and returns `pearson_r` of
    NaN below this, which `impute_z_block` rejects -- so a block with fewer
    observed points than this yields no fills however good its LD is. Exposed
    so callers deciding *which* blocks to attempt use the number the gate
    actually enforces rather than a second copy of it (issue #102).
    """
    return max(npoly + 1, 2)


def poly_rescale(
    truth: np.ndarray,
    predicted: np.ndarray,
    npoly: int = 3,
) -> tuple[np.ndarray, float]:
    """Rescale *predicted* to match the scale of *truth* via polynomial regression.

    Removes Cook's-distance ratio outliers (two passes), fits a degree-*npoly*
    polynomial through the origin.

    Returns (rescaled_array, pearson_r). pearson_r is NaN when too few clean points.
    """
    obs_mask = np.isfinite(truth)
    if obs_mask.sum() < min_observed_points(npoly):
        return predicted.copy(), float("nan")

    t_obs = truth[obs_mask]
    p_obs = predicted[obs_mask]

    keep = _ratio_outlier_mask(t_obs, p_obs)
    if keep.sum() < min_observed_points(npoly):
        return predicted.copy(), float("nan")

    t_clean = t_obs[keep]
    p_clean = p_obs[keep]

    while npoly > 1 and len(t_clean) < npoly + 1:
        npoly -= 1
    if len(t_clean) < 2:
        return predicted.copy(), float("nan")

    try:
        coeffs = np.polyfit(p_clean, t_clean, npoly)
    except np.linalg.LinAlgError:
        return predicted.copy(), float("nan")

    adjusted = np.polyval(coeffs, predicted)

    adj_obs = np.polyval(coeffs, p_obs[keep])
    if len(adj_obs) < 2 or np.std(adj_obs) == 0 or np.std(t_clean) == 0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(adj_obs, t_clean)[0, 1])

    return adjusted, corr


def impute_z_block(
    z_obs_dense: np.ndarray,
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    min_cor: float = 0.7,
) -> tuple[np.ndarray | None, float]:
    """Impute z-scores for one LD block using elastic-net on eigenvectors.

    Parameters
    ----------
    z_obs_dense : float64 array, length = n_ld_snps_in_block.
        NaN at positions not observed in the store.
    eigenvectors : (n_ld_snps, n_comp) float64 array.
    eigenvalues : (n_comp,) float64 array.
    min_cor : Pearson r quality gate.

    Returns
    -------
    (z_imputed, pearson_r) where z_imputed is None when imputation is rejected.
    pearson_r is NaN when imputation is skipped before the quality gate.
    """
    n_comp = len(eigenvalues)
    z_pred = elastic_net_impute(z_obs_dense, eigenvectors, n_comp)
    if z_pred is None:
        return None, float("nan")

    z_adj, corr = poly_rescale(z_obs_dense, z_pred)

    # Clamp to observed |z| range to prevent polynomial extrapolation spikes.
    z_obs_max = np.nanmax(np.abs(z_obs_dense))
    if np.isfinite(z_obs_max) and z_obs_max > 0:
        z_adj = np.clip(z_adj, -z_obs_max, z_obs_max)

    if not np.isfinite(corr) or corr < min_cor:
        log.debug("impute_z_block: pearson_r %.3f below min_cor %.3f – rejected", corr, min_cor)
        return None, corr

    return z_adj, corr


def scalar_n_se(
    se_obs: np.ndarray,
    eaf_obs: np.ndarray,
    eaf_ref: np.ndarray,
) -> np.ndarray:
    """Derive SE for reference-panel positions via the scalar-N model.

    se_scale = median(se_obs * sqrt(2 * eaf_obs * (1 - eaf_obs)))
    se_ref[i] = se_scale / sqrt(2 * eaf_ref[i] * (1 - eaf_ref[i]))

    Parameters
    ----------
    se_obs : SE values at observed positions (finite, positive).
    eaf_obs : EAF from the LD panel matched to observed positions.
    eaf_ref : EAF from the LD panel for the target (reference-panel) positions.

    Returns float64 array of length len(eaf_ref). NaN where EAF is 0, 1, or NaN.
    """
    se_scale = se_scale_constant(se_obs, eaf_obs)
    if np.isnan(se_scale):
        return np.full(len(eaf_ref), np.nan)

    het_ref = 2.0 * eaf_ref * (1.0 - eaf_ref)
    with np.errstate(divide="ignore", invalid="ignore"):
        se_out = np.where(
            (eaf_ref > 0) & (eaf_ref < 1) & np.isfinite(eaf_ref),
            se_scale / np.sqrt(het_ref),
            np.nan,
        )
    return se_out
