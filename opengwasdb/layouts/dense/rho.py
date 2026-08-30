"""Dense pairwise Rho Matrix: distance-thinned nulls, pleiodb-style CML estimator.

Rho is the correlation between two Analyses' statistics under the null
(sample overlap x phenotypic correlation) -- see issues/prd-dense-rho-matrix.md
and docs/adr/0025-dense-rho-matrix.md. The estimator mirrors pleiodb's
``estimate_rho_cml`` (truncated bivariate-normal conditional MLE, ADR 0025
"Estimator"): sufficient statistics on the shared both-null subset, minimised
against a precomputed normalising-constant grid.

Dense-only, opt-in, add-in-place (same pattern as ``top_hits.py``): a built
Dense store gains a ``data.zarr/rho`` group; nothing else changes.
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from numcodecs import Blosc
from scipy.optimize import minimize_scalar  # type: ignore[import-untyped]
from scipy.stats import multivariate_normal  # type: ignore[import-untyped]
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from opengwasdb.encoding import DenseZPlane, StoreEncoding
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.variants import VariantAxis

RHO_METHOD = "pleiodb-cml"
RHO_GRID_N = 2000
DEFAULT_RHO_WINDOW_BP = 15_000
DEFAULT_RHO_Z_THRESH = 1.0
DEFAULT_RHO_MIN_NULLS = 500
_RHO_BOUNDS = (-1.0 + 1e-6, 1.0 - 1e-6)
_RHO_CHUNK_ROWS = 1_000_000


# ── Distance-thinned variant selection (PRD "Independent variant selection") ─


def select_thinned_variants(
    chromosomes: np.ndarray, positions: np.ndarray, window_bp: int
) -> np.ndarray:
    """First variant per fixed-width genomic window, per chromosome.

    ``chromosomes``/``positions`` must be in the store's own row order, which
    is already sorted by ``(chromosome_sort_key, position)`` -- so distance
    thinning is a single left-to-right scan for the first row of each new
    window, with no separate genome-wide sort or LD panel needed.
    """
    n = len(chromosomes)
    if n == 0:
        return np.empty(0, dtype=np.int32)
    chromosomes = np.asarray(chromosomes)
    bucket = np.asarray(positions, dtype=np.int64) // int(window_bp)
    new_window = np.empty(n, dtype=bool)
    new_window[0] = True
    new_window[1:] = (chromosomes[1:] != chromosomes[:-1]) | (bucket[1:] != bucket[:-1])
    return np.nonzero(new_window)[0].astype(np.int32)


# ── pleiodb CML estimator (ADR 0025 "Estimator") ─────────────────────────────


def _box_probability(z_thresh: float, rho: float) -> float:
    """Pr(|X| < z_thresh, |Y| < z_thresh) for a standard bivariate normal
    (X, Y) with correlation ``rho`` -- the CML normalising constant."""
    rho = float(np.clip(rho, *_RHO_BOUNDS))
    z = float(z_thresh)
    mvn = multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]])
    return float(mvn.cdf([z, z]) - mvn.cdf([-z, z]) - mvn.cdf([z, -z]) + mvn.cdf([-z, -z]))


@lru_cache(maxsize=16)
def _cached_rho_grid(z_thresh: float, grid_n: int) -> tuple[np.ndarray, np.ndarray]:
    rho_grid = np.linspace(_RHO_BOUNDS[0], _RHO_BOUNDS[1], grid_n)
    log_p_grid = np.array([math.log(_box_probability(z_thresh, r)) for r in rho_grid])
    return rho_grid, log_p_grid


def compute_rho_grid(z_thresh: float, grid_n: int = RHO_GRID_N) -> tuple[np.ndarray, np.ndarray]:
    """Precompute (or fetch the cached) log normalising-constant grid keyed on
    ``z_thresh`` (``_GRID_N`` points spanning the open interval (-1, 1)). Pure
    function of its inputs, so cached process-wide -- a Dense build calls this
    once per (z_thresh, grid_n) regardless of how many pairs it estimates."""
    return _cached_rho_grid(float(z_thresh), int(grid_n))


def _neg_log_likelihood(
    rho: float,
    A: float,
    B: float,
    C: float,
    n: int,
    rho_grid: np.ndarray,
    log_p_grid: np.ndarray,
) -> float:
    """Truncated-bivariate-normal NLL for sufficient stats (A, B, C, n), up to
    an additive constant that does not depend on ``rho``."""
    log_p = float(np.interp(rho, rho_grid, log_p_grid))
    denom = 2.0 * (1.0 - rho * rho)
    return 0.5 * n * math.log(1.0 - rho * rho) + (A - 2.0 * rho * B + C) / denom + n * log_p


def _estimate_rho_from_stats(
    A: float,
    B: float,
    C: float,
    n: int,
    min_nulls: int,
    rho_grid: np.ndarray,
    log_p_grid: np.ndarray,
) -> float:
    if n < min_nulls:
        return float("nan")
    result = minimize_scalar(
        _neg_log_likelihood,
        bounds=_RHO_BOUNDS,
        method="bounded",
        args=(A, B, C, n, rho_grid, log_p_grid),
    )
    return float(result.x)


def estimate_rho_cml(
    z_j: np.ndarray,
    z_k: np.ndarray,
    z_thresh: float = DEFAULT_RHO_Z_THRESH,
    min_nulls: int = DEFAULT_RHO_MIN_NULLS,
) -> float:
    """Conditional-MLE Rho between two Z-score vectors (pleiodb ``rho.py``).

    Uses variants where both are finite and both are non-significant
    (``|z| < z_thresh``). NaN when fewer than ``min_nulls`` such variants exist.
    """
    z_j = np.asarray(z_j, dtype=np.float64)
    z_k = np.asarray(z_k, dtype=np.float64)
    mask = (
        np.isfinite(z_j) & np.isfinite(z_k) & (np.abs(z_j) < z_thresh) & (np.abs(z_k) < z_thresh)
    )
    n = int(mask.sum())
    if n < min_nulls:
        return float("nan")
    zj, zk = z_j[mask], z_k[mask]
    A = float(np.dot(zj, zj))
    B = float(np.dot(zj, zk))
    C = float(np.dot(zk, zk))
    rho_grid, log_p_grid = compute_rho_grid(z_thresh)
    return _estimate_rho_from_stats(A, B, C, n, min_nulls, rho_grid, log_p_grid)


# ── Batched sufficient statistics over all pairs ─────────────────────────────


def _null_matrices(
    root: Any, thinned_rows: np.ndarray, z_thresh: float, encoding: StoreEncoding
) -> tuple[np.ndarray, np.ndarray]:
    """Observed, nulls-zeroed Z and the null mask at the thinned rows.

    Imputed cells (Reference-Completed stores) are treated as missing --
    Rho is estimated from source-observed statistics only (PRD "Estimation
    inputs").
    """
    z = np.asarray(DenseZPlane.open(root, encoding).rows(thinned_rows), dtype=np.float64)
    finite = np.isfinite(z)
    if "imputed" in root:
        imputed = np.asarray(root["imputed"].oindex[thinned_rows, :], dtype=np.uint8)
        finite = finite & (imputed == 0)
    mask = finite & (np.abs(z) < z_thresh)
    z_nulls_zeroed = np.where(mask, z, 0.0)
    return z_nulls_zeroed, mask


def _pairwise_sufficient_statistics(
    z_nulls_zeroed: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(A, B, C, n) matrices for every Analysis pair via batched dot products.

    ``A[j, k] = sum_v mask[v,j] * mask[v,k] * z[v,j]**2`` -- analysis j's own
    sum-of-squares, restricted to the pair's *shared* null set; ``C = A.T`` is
    the same restricted to analysis k. ``B[j, k] = sum_v z[v,j] * z[v,k]``
    over the shared null set (``z_nulls_zeroed`` is already zero outside each
    column's own null set, so the product is automatically zero unless both
    are null). ``n[j, k]`` is the shared null-variant support count.
    """
    m = mask.astype(np.float64)
    z2_masked = (z_nulls_zeroed**2) * mask
    A = z2_masked.T @ m
    C = A.T
    B = z_nulls_zeroed.T @ z_nulls_zeroed
    n = np.rint(m.T @ m).astype(np.int64)
    return A, B, C, n


# ── Per-pair MLE, optionally parallelised over pairs (ADR 0023 pattern) ──────

_worker_thread_limiter: Any = None


def _init_rho_worker() -> None:
    """Cap each pool worker's BLAS threads to one -- see completion.py's
    ``_init_block_worker`` (ADR 0023): n_workers processes each spawning
    per-core BLAS threads would oversubscribe the CPU."""
    global _worker_thread_limiter
    _worker_thread_limiter = threadpool_limits(limits=1)


def _estimate_pairs_batch(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    n: np.ndarray,
    min_nulls: int,
    rho_grid: np.ndarray,
    log_p_grid: np.ndarray,
) -> np.ndarray:
    out = np.empty(len(A), dtype=np.float64)
    for i in range(len(A)):
        out[i] = _estimate_rho_from_stats(
            float(A[i]), float(B[i]), float(C[i]), int(n[i]), min_nulls, rho_grid, log_p_grid
        )
    return out


def _estimate_pairs_worker(
    task: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray],
) -> np.ndarray:
    A, B, C, n, min_nulls, rho_grid, log_p_grid = task
    return _estimate_pairs_batch(A, B, C, n, min_nulls, rho_grid, log_p_grid)


def _estimate_all_pairs(
    A_flat: np.ndarray,
    B_flat: np.ndarray,
    C_flat: np.ndarray,
    n_flat: np.ndarray,
    z_thresh: float,
    min_nulls: int,
    n_workers: int,
) -> np.ndarray:
    """Per-pair MLE over every flattened pair. ``n_workers`` is a pure runtime
    knob -- the serial and parallel paths share the same per-pair function and
    the same precomputed grid, so they produce identical results."""
    rho_grid, log_p_grid = compute_rho_grid(z_thresh)
    if n_workers <= 1 or len(A_flat) == 0:
        return _estimate_pairs_batch(
            A_flat, B_flat, C_flat, n_flat, min_nulls, rho_grid, log_p_grid
        )

    n_chunks = min(n_workers, len(A_flat))
    chunk_index = np.array_split(np.arange(len(A_flat)), n_chunks)
    tasks = [
        (A_flat[c], B_flat[c], C_flat[c], n_flat[c], min_nulls, rho_grid, log_p_grid)
        for c in chunk_index
    ]
    results: list[np.ndarray | None] = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_rho_worker) as pool:
        futures = {pool.submit(_estimate_pairs_worker, task): i for i, task in enumerate(tasks)}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    return np.concatenate(results)  # type: ignore[arg-type]


# ── Storage (PRD "Storage") ───────────────────────────────────────────────────


def write_rho_group(
    store_path: str | Path,
    rho_packed: np.ndarray,
    n_null_packed: np.ndarray,
    variant_index: np.ndarray,
    *,
    z_thresh: float,
    min_nulls: int,
    window_bp: int,
    n_analyses: int,
) -> None:
    """Write the packed strict-lower-triangle ``rho``/``n_null`` arrays plus
    provenance into ``data.zarr/rho``, replacing any existing group."""
    root = zarr.open_group(str(Path(store_path) / "data.zarr"), mode="a")
    if "rho" in root:
        del root["rho"]
    group = root.create_group("rho")
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

    chunk = max(1, min(len(rho_packed), _RHO_CHUNK_ROWS))
    group.create_dataset(
        "rho", data=rho_packed.astype("float16"), chunks=(chunk,), compressor=compressor
    )
    group.create_dataset(
        "n_null", data=n_null_packed.astype("int32"), chunks=(chunk,), compressor=compressor
    )
    vchunk = max(1, min(len(variant_index), _RHO_CHUNK_ROWS))
    group.create_dataset(
        "variant_index",
        data=variant_index.astype("int32"),
        chunks=(vchunk,),
        compressor=compressor,
    )
    group.attrs["z_thresh"] = float(z_thresh)
    group.attrs["min_nulls"] = int(min_nulls)
    group.attrs["grid_n"] = RHO_GRID_N
    group.attrs["window_bp"] = int(window_bp)
    group.attrs["n_variants_used"] = int(len(variant_index))
    group.attrs["observed_only"] = True
    group.attrs["method"] = RHO_METHOD
    group.attrs["n_analyses"] = int(n_analyses)


# ── Build entry point ────────────────────────────────────────────────────────


def build_dense_rho(
    store_path: str | Path,
    *,
    window_bp: int = DEFAULT_RHO_WINDOW_BP,
    z_thresh: float = DEFAULT_RHO_Z_THRESH,
    min_nulls: int = DEFAULT_RHO_MIN_NULLS,
    n_workers: int = 1,
) -> None:
    """(Re)build the pairwise Rho Matrix for a Dense store, in place.

    Distance-thins the store's own variant axis to approximately independent
    variants, estimates Rho for every Analysis pair with the pleiodb CML
    estimator on their shared observed non-significant Z-scores, and writes
    ``data.zarr/rho`` -- the same add-in-place pattern as the top-hit index.
    Dense-only: Ragged stores have no shared variant axis to thin.
    """
    store_path = Path(store_path)
    encoding = StoreManifest.load(store_path).encoding
    root = zarr.open_group(str(store_path / "data.zarr"), mode="r")
    n_analyses = int(root["z"].shape[1])

    variant_axis = VariantAxis(store_path)
    try:
        records = variant_axis.all()
    finally:
        variant_axis.close()
    chromosomes = np.array([r.chromosome for r in records])
    positions = np.array([r.position for r in records], dtype=np.int64)
    thinned_rows = select_thinned_variants(chromosomes, positions, window_bp)

    z_nulls_zeroed, mask = _null_matrices(root, thinned_rows, z_thresh, encoding)
    A, B, C, n = _pairwise_sufficient_statistics(z_nulls_zeroed, mask)

    rows, cols = np.tril_indices(n_analyses, k=-1)
    A_flat, B_flat, C_flat, n_flat = A[rows, cols], B[rows, cols], C[rows, cols], n[rows, cols]

    rho_flat = _estimate_all_pairs(A_flat, B_flat, C_flat, n_flat, z_thresh, min_nulls, n_workers)

    write_rho_group(
        store_path,
        rho_flat,
        n_flat,
        thinned_rows,
        z_thresh=z_thresh,
        min_nulls=min_nulls,
        window_bp=window_bp,
        n_analyses=n_analyses,
    )


# ── Query reader (facade support, PRD "Query API") ──────────────────────────


class DenseRhoReader:
    """Unpack a stored packed strict-lower-triangle Rho Matrix into symmetric
    form. The packed arrays are tiny at any supported scale (PRD "Storage" /
    "Known Limitation"), so they -- and the reconstructed full matrix, once
    asked for -- are held in memory rather than re-read per query."""

    def __init__(self, group: zarr.Group, n_analyses: int):
        self.group = group
        self.n_analyses = n_analyses
        self._packed_rho = np.asarray(group["rho"][:], dtype=np.float32)
        self._packed_n_null = np.asarray(group["n_null"][:], dtype=np.int64)
        self._full_rho: np.ndarray | None = None
        self._full_n_null: np.ndarray | None = None

    def _ensure_full(self) -> None:
        if self._full_rho is not None:
            return
        n = self.n_analyses
        rows, cols = np.tril_indices(n, k=-1)
        full_rho = np.eye(n, dtype=np.float32)
        full_n_null = np.zeros((n, n), dtype=np.int64)
        full_rho[rows, cols] = self._packed_rho
        full_rho[cols, rows] = self._packed_rho
        full_n_null[rows, cols] = self._packed_n_null
        full_n_null[cols, rows] = self._packed_n_null
        self._full_rho = full_rho
        self._full_n_null = full_n_null

    def pair(self, i: int, j: int) -> tuple[float, int]:
        """Rho and support between analysis indices ``i`` and ``j``."""
        self._ensure_full()
        assert self._full_rho is not None and self._full_n_null is not None
        return float(self._full_rho[i, j]), int(self._full_n_null[i, j])

    def row(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Rho and support of analysis index ``i`` against every analysis
        (including itself: rho=1.0, n_null=0 at the diagonal)."""
        self._ensure_full()
        assert self._full_rho is not None and self._full_n_null is not None
        return self._full_rho[i, :].copy(), self._full_n_null[i, :].copy()

    def matrix(self, indices: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Full symmetric matrix, or the dense submatrix for ``indices``."""
        self._ensure_full()
        assert self._full_rho is not None and self._full_n_null is not None
        if indices is None:
            return self._full_rho.copy(), self._full_n_null.copy()
        idx = np.asarray(indices, dtype=np.int64)
        if len(idx) == 0:
            return (
                np.empty((0, 0), dtype=np.float32),
                np.empty((0, 0), dtype=np.int64),
            )
        return self._full_rho[np.ix_(idx, idx)], self._full_n_null[np.ix_(idx, idx)]
