"""Residual covariance to A1-signed correlation (ADR 0101; opengwas/ogld#89 ruling)."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from opengwasdb.ld.build.importing.parts import read_part

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def diagonal_mask(diagonal: FloatArray) -> tuple[npt.NDArray[np.bool_], dict[str, int]]:
    """Undefined residual correlations drop variants, never substitute diagonal = 1."""
    finite = np.isfinite(diagonal)
    return finite & (diagonal > 0), {
        "nonfinite_diagonal": int((~finite).sum()),
        "nonpositive_diagonal": int((finite & (diagonal <= 0)).sum()),
    }


def correlation(
    raw: npt.NDArray[Any],
    row_diagonal: FloatArray,
    column_diagonal: FloatArray,
    row_sign: IntArray,
    column_sign: IntArray,
) -> FloatArray:
    """Normalise before quantising, preserving opengwas/ogld#89's float64 operation order."""
    result = np.array(raw, dtype=np.float64)
    result /= np.sqrt(row_diagonal[:, None] * column_diagonal)
    result *= row_sign[:, None]
    result *= column_sign
    return result


class MatrixSource:
    """Read indexed parts with a bounded per-chromosome decoded-block working set."""

    def __init__(self, folder: Path, metadata: dict[str, Any], *, cache_blocks: int = 48):
        self.folder = folder
        self.paths: dict[str, Path] = {}
        self.size = int(metadata["blockSize"])
        if not 0 < self.size <= 4096 or metadata["nRows"] != metadata["nCols"]:
            raise ValueError("expected square BlockMatrix with blockSize at most 4096")
        self.rows = int(metadata["nRows"])
        n = (self.rows + self.size - 1) // self.size
        self.parts = {
            (z % n, z // n): name
            for z, name in zip(metadata["maybeFiltered"], metadata["partFiles"], strict=True)
        }
        if len(self.parts) != len(metadata["partFiles"]):
            raise ValueError("duplicate BlockMatrix block indices")
        self._cache: OrderedDict[str, FloatArray] = OrderedDict()
        self._limit = cache_blocks

    def block(self, row: int, column: int) -> FloatArray:
        """A missing allocation is refused, not synthesised as zeros."""
        if row > column:
            return self.block(column, row).T
        name = self.parts[row, column]
        if name not in self._cache:
            part = read_part(self.paths.get(name, self.folder / name))
            expected = (
                min(self.size, self.rows - row * self.size),
                min(self.size, self.rows - column * self.size),
            )
            if part.shape != expected:
                raise ValueError(f"{name}: shape {part.shape}, expected {expected}")
            self._cache[name] = part
            if len(self._cache) > self._limit:
                self._cache.popitem(last=False)
        self._cache.move_to_end(name)
        return self._cache[name]

    def diagonals(self, indices: IntArray) -> FloatArray:
        result = np.full(len(indices), np.nan)
        for block in np.unique(indices // self.size):
            at = np.flatnonzero(indices // self.size == block)
            result[at] = self.block(int(block), int(block)).diagonal()[indices[at] % self.size]
        return result

    def covariance(self, rows: IntArray, columns: IntArray) -> FloatArray:
        """Only allocated source values are numbers; missing coverage stays NaN."""
        result = np.full((len(rows), len(columns)), np.nan)
        rb, cb = rows // self.size, columns // self.size
        for row_block in np.unique(rb):
            r = np.flatnonzero(rb == row_block)
            local_r = rows[r] % self.size
            for col_block in np.unique(cb):
                if (min(row_block, col_block), max(row_block, col_block)) not in self.parts:
                    continue
                c = np.flatnonzero(cb == col_block)
                local_c = columns[c] % self.size
                block = self.block(int(row_block), int(col_block))
                if row_block == col_block:
                    held = block[
                        np.minimum(local_r[:, None], local_c), np.maximum(local_r[:, None], local_c)
                    ]
                else:
                    held = block[np.ix_(local_r, local_c)]
                result[np.ix_(r, c)] = held
        return result
