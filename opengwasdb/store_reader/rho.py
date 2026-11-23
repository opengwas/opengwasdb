"""Reading a Dense release's Rho Matrix (ADR 0025): the stored packed lower triangle, unpacked.

Estimating and writing the matrix is `opengwasdb.layouts.dense.rho`'s.
"""

from __future__ import annotations

import numpy as np
import zarr


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
