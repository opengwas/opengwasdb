"""Decoded views over a store's `z` plane.

A `DenseZPlane` wraps the zarr array and its overflow table and offers the
read shapes the query, Rho, top-hit and validation paths actually use. Callers
get `float32` z-scores and never see a sentinel, a scale, or the difference
between a `float16` legacy store and a fixed-point one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from opengwasdb.encoding.codec import (
    StoreCodec,
    ZOverflowBuilder,
    ZOverflowTable,
    positions_pairs,
    positions_row_band,
    positions_rows_cols,
)
from opengwasdb.encoding.plan import StoreEncoding


class DenseZPlane:
    """The `n_variants x n_analyses` z grid, decoded on read."""

    def __init__(self, array: Any, codec: StoreCodec, group: Any = None) -> None:
        self._array = array
        self._codec = codec
        self._group = group

    @classmethod
    def open(cls, group: Any, encoding: StoreEncoding, *, name: str = "z") -> DenseZPlane:
        """Open the plane `name` in `group` under a store's declared plan."""
        return cls(
            group[name],
            StoreCodec(encoding, z_overflow=ZOverflowTable.read(group)),
            group,
        )

    @property
    def array(self) -> Any:
        """The raw zarr array -- for shape, chunks and dtype only."""
        return self._array

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self._array.shape)

    @property
    def n_analyses(self) -> int:
        return int(self._array.shape[1])

    def band(self, r0: int, r1: int) -> np.ndarray:
        """Rows `[r0:r1)`, all analyses."""
        return self._codec.decode_z(
            self._array[r0:r1], positions=positions_row_band(r0, self.n_analyses)
        )

    def column(self, col: int) -> np.ndarray:
        """One analysis, every variant."""
        raw = self._array[:, col]
        n_analyses = self.n_analyses

        def resolve(mask: np.ndarray) -> np.ndarray:
            return np.flatnonzero(mask).astype(np.int64) * n_analyses + int(col)

        return self._codec.decode_z(raw, positions=resolve)

    def row(self, row: int) -> np.ndarray:
        """One variant, every analysis."""
        return self._codec.decode_z(
            self._array[row, :], positions=positions_row_band(row, self.n_analyses)
        )

    def rows(self, row_indices: np.ndarray) -> np.ndarray:
        """A set of variants, every analysis. Contiguous runs read as a slice."""
        row_indices = np.asarray(row_indices)
        if len(row_indices) == 0:
            return np.empty((0, self.n_analyses), dtype=np.float32)
        start, stop = int(row_indices[0]), int(row_indices[-1]) + 1
        contiguous = stop - start == len(row_indices) and np.array_equal(
            row_indices, np.arange(start, stop, dtype=row_indices.dtype)
        )
        if contiguous:
            return self.band(start, stop)
        return self._codec.decode_z(
            self._array.oindex[row_indices, :],
            positions=positions_rows_cols(
                row_indices, np.arange(self.n_analyses), self.n_analyses
            ),
        )

    def block(
        self, row_indices: Sequence[int] | np.ndarray, col_indices: Sequence[int] | np.ndarray
    ) -> np.ndarray:
        """The cross product `rows x cols` (zarr orthogonal indexing)."""
        return self._codec.decode_z(
            self._array.oindex[list(row_indices), list(col_indices)],
            positions=positions_rows_cols(row_indices, col_indices, self.n_analyses),
        )

    def points(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Elementwise cells `(rows[i], cols[i])` (zarr coordinate indexing)."""
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        if len(rows) == 0:
            return np.empty(0, dtype=np.float32)
        return self._codec.decode_z(
            self._array.vindex[rows, cols],
            positions=positions_pairs(rows, cols, self.n_analyses),
        )

    # ---- writing ---------------------------------------------------------

    def patch(self, rows: np.ndarray, cols: np.ndarray, values: np.ndarray) -> None:
        """Overwrite the cells `(rows[i], cols[i])` with `values[i]`.

        The one partial in-place write against a built plane -- Hybrid
        completion folding a crossed-over association into the Dense Component
        (issue #99). The overflow table is rewritten with it, not after it: a
        patched cell can enter or leave the representable range, and a table
        left describing the value a cell *used* to have is exactly the kind of
        stale-by-one-call-site defect the codec exists to make impossible.
        """
        if self._group is None:
            raise ValueError("this plane was opened without its group and cannot be written")
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        if len(rows) == 0:
            return
        positions = positions_pairs(rows, cols, self.n_analyses)
        builder = ZOverflowBuilder()
        self._array.vindex[rows, cols] = self._codec.encode_z(
            values, positions=positions, overflow=builder
        )
        if not self._codec.encoding.z.is_fixed_point:
            # A float plane has no overflow table, and must not acquire an
            # empty one just because it was written through here.
            return
        previous = self._codec.z_overflow or ZOverflowTable.empty()
        untouched = ~np.isin(previous.index, positions)
        merged = ZOverflowBuilder()
        merged.add(previous.index[untouched], previous.value[untouched])
        new_table = builder.table()
        merged.add(new_table.index, new_table.value)
        table = merged.table()
        table.write(self._group)
        self._codec = StoreCodec(self._codec.encoding, z_overflow=table)
