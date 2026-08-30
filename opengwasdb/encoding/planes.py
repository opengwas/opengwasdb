"""Decoded views over a store's statistic planes.

A `DenseZPlane` wraps the zarr array and its overflow table and offers the
read shapes the query, Rho, top-hit and validation paths actually use. Callers
get `float32` z-scores and never see a sentinel, a scale, or the difference
between a `float16` legacy store and a fixed-point one.

`DenseEafPlane` and `RaggedEafPlane` do the same for `eaf`, which needs more
company than `z`: a per-variant baseline, an exception table, and -- on a
Reference-Completed release -- the imputed mask and the per-variant reference
frequency. Gathering those onto the cells being read is the whole job, and it
lives here rather than at each of the query facade's result sites, because a
site that gathered the baseline against the wrong variant axis would return
frequencies that are wrong and plausible (issue #99, issue #106).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from opengwasdb.encoding.codec import (
    EAF_BASELINE,
    EAF_REFERENCE,
    EafExceptionBuilder,
    EafExceptionTable,
    StoreCodec,
    ZOverflowBuilder,
    ZOverflowTable,
    positions_at,
    positions_flat,
    positions_pairs,
    positions_row_band,
    positions_rows_cols,
)
from opengwasdb.encoding.plan import EafBaselineError, StoreEncoding


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


class _EafPlaneBase:
    """Shared plumbing for the decoded `eaf` views.

    Holds the four arrays a decode may need and refuses, loudly, a release
    whose plan promises one that is not there. "The plan says what should be
    there and validation checks that it is" is only true if the read path
    stops rather than substituting an absence (issue #119).
    """

    def __init__(
        self,
        array: Any,
        codec: StoreCodec,
        *,
        baseline: Any = None,
        reference: Any = None,
        imputed: Any = None,
        group: Any = None,
    ) -> None:
        self._array = array
        self._codec = codec
        self._baseline = baseline
        self._reference = reference
        self._imputed = imputed
        self._group = group
        encoding = codec.encoding.eaf
        if array is None and not (encoding.is_absent or encoding.is_optional_plane):
            raise EafBaselineError(
                f"this release declares an eaf encoding of {encoding.kind!r} but carries "
                "no eaf array; the store and its manifest disagree"
            )
        if encoding.is_residual and baseline is None:
            raise EafBaselineError(
                "this release declares an int8_residual eaf plane but carries no "
                f"{EAF_BASELINE} array, without which the plane cannot be decoded"
            )
        if encoding.reference and reference is None:
            raise EafBaselineError(
                "this release declares reference EAF for its imputed cells but carries "
                f"no {EAF_REFERENCE} array"
            )
        if encoding.reference and imputed is None:
            raise EafBaselineError(
                "this release declares reference EAF for its imputed cells but carries "
                "no imputed mask to say which cells those are (spec §9, §15)"
            )

    @property
    def has_values(self) -> bool:
        """Whether this component holds frequencies at all (ADR 0036)."""
        return self._array is not None

    @property
    def carries_reference(self) -> bool:
        return bool(self._codec.encoding.eaf.reference)

    @staticmethod
    def _missing(shape: int | tuple[int, ...]) -> np.ndarray:
        return np.full(shape, np.nan, dtype=np.float32)

    @staticmethod
    def _gather(array: Any, rows: np.ndarray) -> np.ndarray | None:
        """One per-variant value per cell, or None when there is no array.

        The gather that has to be against the *right* variant axis: a Hybrid
        release's Dense Component is panel-sized where its Ragged Overflow
        covers the shared union, and a baseline gathered against the other
        one decodes to plausible, wrong frequencies (issue #99, #106).
        """
        if array is None:
            return None
        return np.asarray(np.asarray(array[:], dtype=np.float32)[rows], dtype=np.float32)


class DenseEafPlane(_EafPlaneBase):
    """The `n_variants x n_analyses` frequency grid, decoded on read."""

    @classmethod
    def open(cls, group: Any, encoding: StoreEncoding, *, name: str = "eaf") -> DenseEafPlane:
        return cls(
            group[name] if name in group else None,
            StoreCodec(encoding, eaf_exceptions=EafExceptionTable.read(group)),
            baseline=group[EAF_BASELINE] if EAF_BASELINE in group else None,
            reference=group[EAF_REFERENCE] if EAF_REFERENCE in group else None,
            imputed=group["imputed"] if "imputed" in group else None,
            group=group,
        )

    @property
    def n_analyses(self) -> int:
        return int(self._array.shape[1])

    def points(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Elementwise cells `(rows[i], cols[i])`."""
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        if self._array is None or len(rows) == 0:
            return self._missing(len(rows))
        return self._codec.decode_eaf(
            self._array.vindex[rows, cols],
            baseline=self._gather(self._baseline, rows),
            positions=positions_pairs(rows, cols, self.n_analyses),
            imputed=(
                self._imputed.vindex[rows, cols].astype(bool)
                if self.carries_reference
                else None
            ),
            reference=self._gather(self._reference, rows),
        )

    def band(self, r0: int, r1: int) -> np.ndarray:
        """Rows `[r0:r1)`, all analyses."""
        n_analyses = self.n_analyses
        if self._array is None:
            return self._missing((r1 - r0, n_analyses))
        per_row = self._gather(self._baseline, np.arange(r0, r1, dtype=np.int64))
        return self._codec.decode_eaf(
            self._array[r0:r1],
            baseline=None if per_row is None else per_row[:, None].repeat(n_analyses, axis=1),
            positions=positions_row_band(r0, n_analyses),
            imputed=self._imputed[r0:r1].astype(bool) if self.carries_reference else None,
            reference=self._reference_band(r0, r1, n_analyses),
        )

    def patch(self, rows: np.ndarray, cols: np.ndarray, values: np.ndarray) -> None:
        """Overwrite the cells `(rows[i], cols[i])` with the frequencies
        `values[i]`, against the baselines already recorded for those variants.

        The one partial in-place write against a built plane -- Hybrid
        completion folding a crossed-over association into the Dense Component
        (issue #99). The exception table is rewritten with it, not after it: a
        patched cell can enter or leave the representable range, and a table
        left describing the value a cell *used* to have is exactly the kind of
        stale-by-one-call-site defect the codec exists to make impossible.

        The baseline is *not* recomputed. It describes the Dense Component's
        own observed cells; a crossed-over cell is one more observation at that
        variant, and moving the baseline to accommodate it would re-quantise
        every cell already coded against it.
        """
        if self._group is None:
            raise EafBaselineError(
                "this plane was opened without its group and cannot be written"
            )
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        if self._array is None or len(rows) == 0:
            return
        positions = positions_pairs(rows, cols, self.n_analyses)
        builder = EafExceptionBuilder()
        self._array.vindex[rows, cols] = self._codec.encode_eaf(
            values,
            baseline=self._gather(self._baseline, rows),
            positions=positions,
            exceptions=builder,
        )
        if not self._codec.encoding.eaf.is_residual:
            return
        previous = self._codec.eaf_exceptions or EafExceptionTable.empty()
        untouched = ~np.isin(previous.index, positions)
        merged = EafExceptionBuilder()
        merged.add(previous.index[untouched], previous.value[untouched])
        added = builder.table()
        merged.add(added.index, added.value)
        table = merged.table()
        table.write(self._group)
        self._codec = StoreCodec(self._codec.encoding, eaf_exceptions=table)

    def _reference_band(self, r0: int, r1: int, n_analyses: int) -> np.ndarray | None:
        per_row = self._gather(self._reference, np.arange(r0, r1, dtype=np.int64))
        if per_row is None:
            return None
        return per_row[:, None].repeat(n_analyses, axis=1)

class RaggedEafPlane(_EafPlaneBase):
    """The flat CSR frequency sequence, decoded on read.

    A CSR cell's flat position is its ordinal in the concatenated arrays, and
    its variant is `variant_index[position]` -- so unlike the Dense grid, the
    baseline gather needs a read of the CSR's own variant index.
    """

    def __init__(
        self,
        array: Any,
        codec: StoreCodec,
        variant_index: Any,
        *,
        baseline: Any = None,
        reference: Any = None,
        imputed: Any = None,
        group: Any = None,
    ) -> None:
        super().__init__(
            array, codec, baseline=baseline, reference=reference, imputed=imputed, group=group
        )
        self._variant_index = variant_index

    @classmethod
    def open(
        cls, group: Any, encoding: StoreEncoding, *, imputed: Any = None
    ) -> RaggedEafPlane:
        return cls(
            group["eaf"] if "eaf" in group else None,
            StoreCodec(encoding, eaf_exceptions=EafExceptionTable.read(group)),
            group["variant_index"],
            baseline=group[EAF_BASELINE] if EAF_BASELINE in group else None,
            reference=group[EAF_REFERENCE] if EAF_REFERENCE in group else None,
            imputed=imputed,
            group=group,
        )

    def slice(self, start: int, end: int) -> np.ndarray:
        """`eaf[start:end]` in flat CSR order."""
        start, end = int(start), int(end)
        if self._array is None or end <= start:
            return self._missing(max(end - start, 0))
        rows = np.asarray(self._variant_index[start:end], dtype=np.int64)
        return self._codec.decode_eaf(
            self._array[start:end],
            baseline=self._gather(self._baseline, rows),
            positions=positions_flat(start),
            imputed=(
                np.asarray(self._imputed[start:end], dtype=bool)
                if self.carries_reference
                else None
            ),
            reference=self._gather(self._reference, rows),
        )

    def at(self, positions: np.ndarray) -> np.ndarray:
        """Frequencies at arbitrary flat CSR positions."""
        positions = np.asarray(positions, dtype=np.int64)
        if self._array is None or len(positions) == 0:
            return self._missing(len(positions))
        rows = np.asarray(self._variant_index[:], dtype=np.int64)[positions]
        return self._codec.decode_eaf(
            np.asarray(self._array[:])[positions],
            baseline=self._gather(self._baseline, rows),
            positions=positions_at(positions),
            imputed=self._imputed_at(positions),
            reference=self._gather(self._reference, rows),
        )

    def _imputed_at(self, positions: np.ndarray) -> np.ndarray | None:
        if not self.carries_reference:
            return None
        return np.asarray(np.asarray(self._imputed[:], dtype=bool)[positions], dtype=bool)



# ── Writing an `eaf` plane ──────────────────────────────────────────────────
#
# The plane, its per-variant baseline and its exception table are one artifact
# in three arrays: a component that has any of them must have all of them.
# These helpers write them together for the two shapes that occur, so no
# builder can produce two of the three.


def write_per_variant_array(
    group: Any,
    name: str,
    values: np.ndarray,
    *,
    compressor: Any = None,
    chunk: int | None = None,
) -> None:
    """Write (or replace) one `float32` per variant of a component's axis."""
    data = np.asarray(values, dtype=np.float32)
    if name in group:
        del group[name]
    group.create_dataset(
        name,
        data=data,
        chunks=(min(chunk or max(len(data), 1), max(len(data), 1)),),
        compressor=compressor,
        dtype="float32",
    )


def write_eaf_baseline(
    group: Any, baseline: np.ndarray, *, compressor: Any = None, chunk: int | None = None
) -> None:
    """Write the per-variant `eaf_baseline` the residual coding decodes against."""
    write_per_variant_array(
        group, EAF_BASELINE, baseline, compressor=compressor, chunk=chunk
    )


def write_eaf_reference(
    group: Any, reference: np.ndarray, *, compressor: Any = None, chunk: int | None = None
) -> None:
    """Write the per-variant reference-panel frequency (ADR 0037 §4).

    One `float32` per variant, and ~0 bytes per cell: an imputed cell's EAF is
    the panel's, identical for every Analysis imputed at that variant, so it is
    a per-variant constant rather than per-cell data.
    """
    write_per_variant_array(
        group, EAF_REFERENCE, reference, compressor=compressor, chunk=chunk
    )


def write_eaf_csr(
    group: Any,
    codec: StoreCodec,
    variant_index: np.ndarray,
    values: np.ndarray,
    *,
    baseline: np.ndarray | None,
    compressor: Any = None,
    chunks: tuple[int, ...] | None = None,
) -> None:
    """Encode and write a CSR component's `eaf` plane and everything it needs.

    A CSR cell's flat position is its ordinal in the concatenated arrays, which
    is what its exception entry is keyed on.
    """
    encoding = codec.encoding.eaf
    if encoding.is_absent:
        return
    if encoding.is_optional_plane and not np.any(np.isfinite(values)):
        # ADR 0036's contract, which only a `format_version` 1.0 release is in:
        # the plane's *presence* is what says the release has frequencies, so a
        # component with none must not acquire an all-NaN one.
        return
    exceptions = EafExceptionBuilder()
    per_cell = (
        None
        if baseline is None
        else np.asarray(baseline, dtype=np.float32)[
            np.asarray(variant_index, dtype=np.int64)
        ]
    )
    codes = codec.encode_eaf(
        values, baseline=per_cell, positions=positions_flat(0), exceptions=exceptions
    )
    if "eaf" in group:
        del group["eaf"]
    group.create_dataset(
        "eaf", data=codes, chunks=chunks, compressor=compressor, dtype=codec.eaf_dtype
    )
    if encoding.is_residual:
        assert baseline is not None
        write_eaf_baseline(group, baseline, compressor=compressor)
        exceptions.table().write(group)
