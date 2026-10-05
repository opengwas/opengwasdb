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
from functools import cached_property
from typing import Any, NamedTuple

import numpy as np

from opengwasdb.encoding.codec import (
    EAF_BASELINE,
    EAF_REFERENCE,
    EafExceptionBuilder,
    EafExceptionTable,
    SeExceptionBuilder,
    SeExceptionTable,
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
from opengwasdb.store.arrays import (
    ArrayRole,
    component_variant_chunk,
    compressor_of,
    create_array,
)

SE_COEFFICIENTS = "se_coefficients"


class EafRead(NamedTuple):
    """One region's decoded frequencies, and the imputed mask read with them.

    `imputed` is None when the component holds no imputed mask; when it is not
    None it is the mask the panel substitution was applied with, so a caller
    that needs it for Association Status does not read the array a second
    time. Both arrays are the region the caller asked for -- a whole column, a
    row band, a rows x analyses block -- so a shared read can be indexed with
    the same finite/observed mask for SE decoding and for the result column
    (#253).
    """

    values: np.ndarray
    imputed: np.ndarray | None


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
            positions=positions_rows_cols(row_indices, np.arange(self.n_analyses), self.n_analyses),
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


class DenseSePlane:
    """Dense Standard Errors decoded against the corresponding physical EAF."""

    def __init__(
        self,
        array: Any,
        codec: StoreCodec,
        eaf: DenseEafPlane | None,
        coefficients: Any,
        group: Any,
    ):
        self._array, self._codec, self._eaf, self._coefficients = array, codec, eaf, coefficients
        self._group = group

    @classmethod
    def open(cls, group: Any, encoding: StoreEncoding) -> DenseSePlane:
        residual = encoding.se.is_residual
        required = ("se_exception_index", "se_exception_value", SE_COEFFICIENTS)
        if residual and any(name not in group for name in required):
            missing = [name for name in required if name not in group]
            raise ValueError(f"int8_residual se plane is missing required arrays: {missing}")
        return cls(
            group["se"],
            StoreCodec(encoding, se_exceptions=SeExceptionTable.read(group)),
            DenseEafPlane.open(group, encoding) if residual else None,
            group[SE_COEFFICIENTS] if residual else None,
            group,
        )

    @property
    def n_analyses(self) -> int:
        return int(self._array.shape[1])

    def _decode(
        self, raw: np.ndarray, eaf: np.ndarray, analyses: np.ndarray, positions: Any
    ) -> np.ndarray:
        if not self._codec.encoding.se.is_residual:
            return self._codec.decode_se(
                raw,
                eaf=eaf,
                analysis_index=analyses,
                coefficients=np.empty((0, 2)),
                positions=positions,
            )
        return self._codec.decode_se(
            raw,
            eaf=eaf,
            analysis_index=analyses,
            coefficients=np.asarray(self._coefficients[:], dtype=np.float32),
            positions=positions,
        )

    def band(self, r0: int, r1: int, *, eaf: np.ndarray | None = None) -> np.ndarray:
        """Rows `[r0:r1)`, all analyses.

        `eaf` is a pre-read decoded block of exactly these cells (#253): the
        query facade reads a region's frequencies once and hands the same array
        to SE decoding and to the result's `eaf` column. When it is omitted the
        plane reads its own, as every other caller does.
        """
        raw = np.asarray(self._array[r0:r1])
        ai = np.broadcast_to(np.arange(self.n_analyses), raw.shape)
        if eaf is None:
            eaf = (
                self._eaf.band(r0, r1)
                if self._eaf is not None
                else np.empty(raw.shape, dtype=np.float32)
            )
        return self._decode(
            raw, np.asarray(eaf, dtype=np.float32), ai, positions_row_band(r0, self.n_analyses)
        )

    def column(self, col: int, *, eaf: np.ndarray | None = None) -> np.ndarray:
        """One analysis, every variant. `eaf` is the pre-read block of these cells."""
        raw = np.asarray(self._array[:, col])
        rows = np.arange(len(raw), dtype=np.int64)
        if eaf is None:
            eaf = (
                self._eaf.points(rows, np.full(len(raw), col))
                if self._eaf is not None
                else np.empty(raw.shape, dtype=np.float32)
            )
        return self._decode(
            raw,
            np.asarray(eaf, dtype=np.float32),
            np.full(len(raw), col),
            positions_pairs(rows, np.full(len(raw), col), self.n_analyses),
        )

    def row(self, row: int, *, eaf: np.ndarray | None = None) -> np.ndarray:
        """One variant, every analysis. `eaf` is the pre-read block of these cells."""
        block = self.band(row, row + 1, eaf=None if eaf is None else np.asarray(eaf)[None, :])
        return np.asarray(block[0], dtype=np.float32)

    def rows(self, rows: np.ndarray, *, eaf: np.ndarray | None = None) -> np.ndarray:
        """A set of variants, every analysis. `eaf` is the pre-read block of these cells."""
        rows = np.asarray(rows, dtype=np.int64)
        if len(rows) == 0:
            return np.empty((0, self.n_analyses), dtype=np.float32)
        return self.block(rows, np.arange(self.n_analyses), eaf=eaf)

    def block(
        self,
        rows: Sequence[int] | np.ndarray,
        cols: Sequence[int] | np.ndarray,
        *,
        eaf: np.ndarray | None = None,
    ) -> np.ndarray:
        """The cross product `rows x cols` (zarr orthogonal indexing)."""
        r, c = np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
        raw = np.asarray(self._array.oindex[r, c])
        ai = np.broadcast_to(c, raw.shape)
        if eaf is None:
            er, ec = np.meshgrid(r, c, indexing="ij")
            eaf = (
                self._eaf.points(er.ravel(), ec.ravel()).reshape(raw.shape)
                if self._eaf is not None
                else np.empty(raw.shape, dtype=np.float32)
            )
        return self._decode(
            raw, np.asarray(eaf, dtype=np.float32), ai, positions_rows_cols(r, c, self.n_analyses)
        )

    def points(
        self, rows: np.ndarray, cols: np.ndarray, *, eaf: np.ndarray | None = None
    ) -> np.ndarray:
        """Elementwise cells `(rows[i], cols[i])` (zarr coordinate indexing)."""
        rows, cols = np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
        raw = np.asarray(self._array.vindex[rows, cols])
        if eaf is None:
            eaf = (
                self._eaf.points(rows, cols)
                if self._eaf is not None
                else np.empty(raw.shape, dtype=np.float32)
            )
        return self._decode(
            raw,
            np.asarray(eaf, dtype=np.float32),
            cols,
            positions_pairs(rows, cols, self.n_analyses),
        )

    def patch(self, rows: np.ndarray, cols: np.ndarray, values: np.ndarray) -> None:
        """Patch physical SE cells and keep their exact-exception table aligned."""
        rows, cols = np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
        if not self._codec.encoding.se.is_residual:
            self._array.vindex[rows, cols] = np.asarray(values, dtype=np.float16)
            return
        assert self._eaf is not None and self._coefficients is not None
        positions = positions_pairs(rows, cols, self.n_analyses)
        builder = SeExceptionBuilder()
        self._array.vindex[rows, cols] = self._codec.encode_se(
            values,
            eaf=self._eaf.points(rows, cols),
            analysis_index=cols,
            coefficients=np.asarray(self._coefficients[:], dtype=np.float32),
            positions=positions,
            exceptions=builder,
        )
        previous = self._codec.se_exceptions or SeExceptionTable.empty()
        merged = SeExceptionBuilder()
        keep = ~np.isin(previous.index, positions)
        merged.add(previous.index[keep], previous.value[keep])
        added = builder.table()
        merged.add(added.index, added.value)
        table = merged.table()
        table.write(self._group, compressor=compressor_of(self._array))
        self._codec = StoreCodec(self._codec.encoding, se_exceptions=table)


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
        if array is None and not encoding.is_absent:
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

    @property
    def can_report_frequencies(self) -> bool:
        """Whether reading this plane can return anything but NaN.

        Not the same as `has_values`. A release with no `eaf` array still
        reports the panel's frequency on its imputed cells (issue #113), so a
        caller that short-circuits on the array's absence drops exactly the
        values that release holds -- silently, and only for the imputed cells.
        """
        return self.has_values or self.carries_reference

    @staticmethod
    def _missing(shape: int | tuple[int, ...]) -> np.ndarray:
        return np.full(shape, np.nan, dtype=np.float32)

    def _read_imputed_mask(self, want: bool) -> bool:
        """Whether a read must fetch the imputed mask.

        The panel substitution needs it whenever the plan declares reference
        EAF; Association Status needs it on any release that has the array, so
        `want` lets the query facade ask for it without a second read.
        """
        return self._imputed is not None and (self.carries_reference or want)

    def _empty_read(self, shape: int | tuple[int, ...], want_imputed: bool) -> EafRead:
        mask = np.empty(shape, dtype=np.uint8) if self._read_imputed_mask(want_imputed) else None
        return EafRead(self._missing(shape), mask)

    def _no_plane(
        self,
        shape: int | tuple[int, ...],
        *,
        imputed: np.ndarray | None,
        reference: np.ndarray | None,
    ) -> np.ndarray:
        """What a component with no `eaf` plane reads: NaN, unless a panel speaks.

        A completed release whose Analyses reported no frequency at all still
        holds one for every cell it imputed -- the panel's (issue #113). There
        is nothing to decode, but the substitution is the same substitution, so
        it goes through the codec rather than being written out a second time
        here.
        """
        blank = self._missing(shape)
        if not self.carries_reference:
            return blank
        return self._codec.decode_eaf(blank, imputed=imputed, reference=reference)

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
        rows = np.asarray(rows, dtype=np.int64)
        if len(rows) == 0:
            return np.empty(0, dtype=np.float32)
        start, stop = int(rows[0]), int(rows[-1]) + 1
        if stop - start == len(rows) and np.array_equal(
            rows, np.arange(start, stop, dtype=rows.dtype)
        ):
            return np.asarray(array[start:stop], dtype=np.float32)
        return np.asarray(array.oindex[rows], dtype=np.float32)


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
        """The grid's width -- read from a sibling plane when `eaf` is absent.

        A component can legitimately have no `eaf` array while the rest of the
        release is a full grid: a completed release carrying only the panel's
        frequencies (issue #113) declares `absent` and holds no plane. The
        width is a property of the release, not of this plane.
        """
        for candidate in (self._array, self._imputed):
            if candidate is not None:
                return int(candidate.shape[1])
        return self._sibling_width

    @cached_property
    def _sibling_width(self) -> int:
        """The `z` plane's width, opened once: under zarr 3 every
        ``name in group`` and ``group[name]`` reads metadata from the store, and
        a regional query asks for the width on every call (#244)."""
        if self._group is not None and "z" in self._group:
            return int(self._group["z"].shape[1])
        raise EafBaselineError(
            "this component has no eaf plane and no sibling plane to take its width from"
        )

    @property
    def n_variants(self) -> int:
        """The grid's height -- from whichever plane carries the variant axis."""
        for candidate in (self._array, self._imputed, self._reference, self._baseline):
            if candidate is not None:
                return int(candidate.shape[0])
        return self._sibling_height

    @cached_property
    def _sibling_height(self) -> int:
        """The `z` plane's height, for a component holding no axis-bearing array."""
        if self._group is not None and "z" in self._group:
            return int(self._group["z"].shape[0])
        raise EafBaselineError(
            "this component has no eaf plane and no sibling plane to take its height from"
        )

    def read_points(
        self, rows: np.ndarray, cols: np.ndarray, *, want_imputed: bool = False
    ) -> EafRead:
        """Elementwise cells `(rows[i], cols[i])`, with the imputed mask."""
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        if len(rows) == 0:
            return self._empty_read(0, want_imputed)
        imputed = (
            self._imputed.vindex[rows, cols].astype(np.uint8)
            if self._read_imputed_mask(want_imputed)
            else None
        )
        reference = self._gather(self._reference, rows)
        if self._array is None:
            values = self._no_plane(len(rows), imputed=imputed, reference=reference)
        else:
            values = self._codec.decode_eaf(
                self._array.vindex[rows, cols],
                baseline=self._gather(self._baseline, rows),
                positions=positions_pairs(rows, cols, self.n_analyses),
                imputed=imputed,
                reference=reference,
            )
        return EafRead(values, imputed)

    def points(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Elementwise cells `(rows[i], cols[i])`."""
        return self.read_points(rows, cols).values

    def read_band(self, r0: int, r1: int, *, want_imputed: bool = False) -> EafRead:
        """Rows `[r0:r1)`, all analyses, with the imputed mask."""
        n_analyses = self.n_analyses
        imputed = (
            self._imputed[r0:r1].astype(np.uint8)
            if self._read_imputed_mask(want_imputed)
            else None
        )
        reference = self._reference_band(r0, r1, n_analyses)
        if self._array is None:
            values = self._no_plane((r1 - r0, n_analyses), imputed=imputed, reference=reference)
        else:
            per_row = self._gather(self._baseline, np.arange(r0, r1, dtype=np.int64))
            values = self._codec.decode_eaf(
                self._array[r0:r1],
                baseline=None if per_row is None else per_row[:, None].repeat(n_analyses, axis=1),
                positions=positions_row_band(r0, n_analyses),
                imputed=imputed,
                reference=reference,
            )
        return EafRead(values, imputed)

    def band(self, r0: int, r1: int) -> np.ndarray:
        """Rows `[r0:r1)`, all analyses."""
        return self.read_band(r0, r1).values

    def read_row(self, row: int, *, want_imputed: bool = False) -> EafRead:
        """One variant, every analysis, with the imputed mask."""
        block = self.read_band(row, row + 1, want_imputed=want_imputed)
        return EafRead(block.values[0], None if block.imputed is None else block.imputed[0])

    def read_rows(self, row_indices: np.ndarray, *, want_imputed: bool = False) -> EafRead:
        """A set of variants, every analysis; contiguous runs read as a band."""
        rows = np.asarray(row_indices, dtype=np.int64)
        if len(rows) == 0:
            return self._empty_read((0, self.n_analyses), want_imputed)
        start, stop = int(rows[0]), int(rows[-1]) + 1
        contiguous = stop - start == len(rows) and np.array_equal(
            rows, np.arange(start, stop, dtype=rows.dtype)
        )
        if contiguous:
            return self.read_band(start, stop, want_imputed=want_imputed)
        return self.read_block(rows, np.arange(self.n_analyses), want_imputed=want_imputed)

    def read_block(
        self,
        rows: Sequence[int] | np.ndarray,
        cols: Sequence[int] | np.ndarray,
        *,
        want_imputed: bool = False,
    ) -> EafRead:
        """The cross product `rows x cols` (zarr orthogonal indexing)."""
        r = np.asarray(rows, dtype=np.int64)
        c = np.asarray(cols, dtype=np.int64)
        if len(r) == 0 or len(c) == 0:
            return self._empty_read((len(r), len(c)), want_imputed)
        imputed = (
            self._imputed.oindex[r, c].astype(np.uint8)
            if self._read_imputed_mask(want_imputed)
            else None
        )
        per_row = self._gather(self._baseline, r)
        baseline = None if per_row is None else per_row[:, None].repeat(len(c), axis=1)
        reference = self._reference_block(r, c)
        if self._array is None:
            values = self._no_plane((len(r), len(c)), imputed=imputed, reference=reference)
        else:
            values = self._codec.decode_eaf(
                self._array.oindex[r, c],
                baseline=baseline,
                positions=positions_rows_cols(r, c, self.n_analyses),
                imputed=imputed,
                reference=reference,
            )
        return EafRead(values, imputed)

    def read_column(self, col: int, *, want_imputed: bool = False) -> EafRead:
        """One analysis, every variant, with the imputed mask."""
        n_variants = self.n_variants
        rows = np.arange(n_variants, dtype=np.int64)
        cols = np.full(n_variants, int(col), dtype=np.int64)
        imputed = (
            self._imputed[:, col].astype(np.uint8)
            if self._read_imputed_mask(want_imputed)
            else None
        )
        reference = self._gather(self._reference, rows)
        if self._array is None:
            values = self._no_plane(n_variants, imputed=imputed, reference=reference)
        else:
            values = self._codec.decode_eaf(
                self._array[:, col],
                baseline=self._gather(self._baseline, rows),
                positions=positions_pairs(rows, cols, self.n_analyses),
                imputed=imputed,
                reference=reference,
            )
        return EafRead(values, imputed)

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
            raise EafBaselineError("this plane was opened without its group and cannot be written")
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

    def _reference_block(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray | None:
        per_row = self._gather(self._reference, rows)
        if per_row is None:
            return None
        return per_row[:, None].repeat(len(cols), axis=1)


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
    def open(cls, group: Any, encoding: StoreEncoding, *, imputed: Any = None) -> RaggedEafPlane:
        return cls(
            group["eaf"] if "eaf" in group else None,
            StoreCodec(encoding, eaf_exceptions=EafExceptionTable.read(group)),
            group["variant_index"],
            baseline=group[EAF_BASELINE] if EAF_BASELINE in group else None,
            reference=group[EAF_REFERENCE] if EAF_REFERENCE in group else None,
            imputed=imputed,
            group=group,
        )

    def read_slice(self, start: int, end: int, *, want_imputed: bool = False) -> EafRead:
        """`eaf[start:end]` in flat CSR order, with the imputed mask."""
        start, end = int(start), int(end)
        if end <= start:
            return self._empty_read(end - start, want_imputed)
        rows = np.asarray(self._variant_index[start:end], dtype=np.int64)
        imputed = (
            np.asarray(self._imputed[start:end], dtype=np.uint8)
            if self._read_imputed_mask(want_imputed)
            else None
        )
        reference = self._gather(self._reference, rows)
        if self._array is None:
            values = self._no_plane(end - start, imputed=imputed, reference=reference)
        else:
            values = self._codec.decode_eaf(
                self._array[start:end],
                baseline=self._gather(self._baseline, rows),
                positions=positions_flat(start),
                imputed=imputed,
                reference=reference,
            )
        return EafRead(values, imputed)

    def slice(self, start: int, end: int) -> np.ndarray:
        """`eaf[start:end]` in flat CSR order."""
        return self.read_slice(start, end).values

    def read_at(self, positions: np.ndarray, *, want_imputed: bool = False) -> EafRead:
        """Frequencies at arbitrary flat CSR positions, with the imputed mask."""
        positions = np.asarray(positions, dtype=np.int64)
        if len(positions) == 0:
            return self._empty_read(0, want_imputed)
        rows = np.asarray(self._variant_index.oindex[positions], dtype=np.int64)
        imputed = (
            np.asarray(self._imputed.oindex[positions], dtype=np.uint8)
            if self._read_imputed_mask(want_imputed)
            else None
        )
        reference = self._gather(self._reference, rows)
        if self._array is None:
            values = self._no_plane(len(positions), imputed=imputed, reference=reference)
        else:
            values = self._codec.decode_eaf(
                np.asarray(self._array.oindex[positions]),
                baseline=self._gather(self._baseline, rows),
                positions=positions_at(positions),
                imputed=imputed,
                reference=reference,
            )
        return EafRead(values, imputed)

    def at(self, positions: np.ndarray) -> np.ndarray:
        """Frequencies at arbitrary flat CSR positions."""
        return self.read_at(positions).values


class RaggedSePlane:
    """CSR Standard Errors decoded using CSR ordinals as exception keys."""

    def __init__(self, group: Any, encoding: StoreEncoding, eaf: RaggedEafPlane):
        self._array = group["se"]
        self._encoding = encoding
        self._eaf = eaf
        self._variant_index = group["variant_index"]
        self._offsets = group["offsets"]
        residual = encoding.se.is_residual
        required = ("se_exception_index", "se_exception_value", SE_COEFFICIENTS)
        if residual and any(name not in group for name in required):
            missing = [name for name in required if name not in group]
            raise ValueError(f"int8_residual se plane is missing required arrays: {missing}")
        self._coefficients = group[SE_COEFFICIENTS] if residual else None
        self._codec = StoreCodec(encoding, se_exceptions=SeExceptionTable.read(group))

    @classmethod
    def open(
        cls, group: Any, encoding: StoreEncoding, eaf: RaggedEafPlane | None = None
    ) -> RaggedSePlane:
        return cls(
            group,
            encoding,
            eaf
            or RaggedEafPlane.open(
                group, encoding, imputed=group["imputed"] if "imputed" in group else None
            ),
        )

    def _analysis_indices(self, positions: np.ndarray) -> np.ndarray:
        offsets = np.asarray(self._offsets[:], dtype=np.int64)
        return np.searchsorted(offsets[1:], positions, side="right").astype(np.int64)

    def slice(
        self,
        start: int,
        end: int,
        *,
        analysis_index: int | None = None,
        eaf: np.ndarray | None = None,
    ) -> np.ndarray:
        """Decoded `se[start:end]`; `eaf` is a pre-read block of these cells (#253)."""
        raw = np.asarray(self._array[start:end])
        ai = (
            np.full(len(raw), analysis_index, dtype=np.int64)
            if analysis_index is not None
            else self._analysis_indices(np.arange(start, end))
        )
        coef = (
            np.asarray(self._coefficients[:], dtype=np.float32)
            if self._coefficients is not None
            else np.empty((0, 2))
        )
        if eaf is None:
            eaf = self._eaf.slice(start, end)
        return self._codec.decode_se(
            raw,
            eaf=np.asarray(eaf, dtype=np.float32),
            analysis_index=ai,
            coefficients=coef,
            positions=positions_flat(start),
        )

    def at(
        self, positions: np.ndarray, *, analysis_index: np.ndarray, eaf: np.ndarray | None = None
    ) -> np.ndarray:
        """Decoded SE at CSR ordinals; `eaf` is a pre-read block of these cells (#253)."""
        positions = np.asarray(positions, dtype=np.int64)
        coef = (
            np.asarray(self._coefficients[:], dtype=np.float32)
            if self._coefficients is not None
            else np.empty((0, 2))
        )
        if eaf is None:
            eaf = self._eaf.at(positions)
        return self._codec.decode_se(
            np.asarray(self._array.oindex[positions]),
            eaf=np.asarray(eaf, dtype=np.float32),
            analysis_index=analysis_index,
            coefficients=coef,
            positions=positions_at(positions),
        )


# ── Writing an `eaf` plane ──────────────────────────────────────────────────
#
# The plane, its per-variant baseline and its exception table are one artifact
# in three arrays: a component that has any of them must have all of them.
# These helpers write them together for the two shapes that occur, so no
# builder can produce two of the three.


def _write_per_variant_array(
    group: Any,
    name: str,
    values: np.ndarray,
    *,
    compressor: Any = None,
    chunk: int | None = None,
) -> None:
    """Write (or replace) one `float32` per variant of a component's axis.

    The component plane's variant-axis chunk is read here and passed in as
    `component_chunk`, so the layout policy itself never inspects `group`.
    """
    create_array(
        group,
        name,
        ArrayRole.PER_VARIANT,
        data=np.asarray(values, dtype=np.float32),
        dtype="float32",
        compressor=compressor,
        hint=chunk,
        component_chunk=component_variant_chunk(group),
        overwrite=True,
    )


def write_eaf_baseline(
    group: Any, baseline: np.ndarray, *, compressor: Any = None, chunk: int | None = None
) -> None:
    """Write the per-variant `eaf_baseline` the residual coding decodes against."""
    _write_per_variant_array(group, EAF_BASELINE, baseline, compressor=compressor, chunk=chunk)


def write_eaf_reference(
    group: Any, reference: np.ndarray, *, compressor: Any = None, chunk: int | None = None
) -> None:
    """Write the per-variant reference-panel frequency (ADR 0037 §4).

    One `float32` per variant, and ~0 bytes per cell: an imputed cell's EAF is
    the panel's, identical for every Analysis imputed at that variant, so it is
    a per-variant constant rather than per-cell data.
    """
    _write_per_variant_array(group, EAF_REFERENCE, reference, compressor=compressor, chunk=chunk)


def write_se_coefficients(group: Any, coefficients: np.ndarray, *, compressor: Any = None) -> None:
    """Write (or replace) the two `float32` decode parameters per Analysis."""
    create_array(
        group,
        SE_COEFFICIENTS,
        ArrayRole.SE_COEFFICIENTS,
        data=np.asarray(coefficients, dtype=np.float32),
        dtype="float32",
        compressor=compressor,
        overwrite=True,
    )


def _write_se_arrays(
    group: Any,
    codec: StoreCodec,
    codes: np.ndarray,
    coefficients: np.ndarray | None,
    exceptions: SeExceptionBuilder | None,
    *,
    role: ArrayRole,
    compressor: Any,
    chunks: tuple[int, ...] | None,
) -> None:
    """Replace the plane and, for a residual plan, both of its side arrays.

    The plane, its coefficients and its exception table are one artifact in
    three arrays, exactly as the `eaf` writers above treat theirs: writing them
    from one place is what stops a builder producing two of the three.
    """
    create_array(
        group,
        "se",
        role,
        data=codes,
        dtype=codec.encoding.se.dtype,
        compressor=compressor,
        hint=chunks,
        overwrite=True,
    )
    if not codec.encoding.se.is_residual:
        return
    assert coefficients is not None and exceptions is not None
    write_se_coefficients(group, coefficients, compressor=compressor)
    exceptions.table().write(group, compressor=compressor)


def _write_se_plane(
    group: Any,
    codec: StoreCodec,
    values: np.ndarray,
    eaf: np.ndarray,
    analysis_index: np.ndarray,
    coefficients: np.ndarray | None,
    positions: Any,
    compressor: Any,
    chunks: tuple[int, ...] | None,
    role: ArrayRole,
) -> None:
    """Encode a plane and write it with both of its side arrays, or as float16.

    Dense and CSR differ only in how a cell's Analysis and flat position are
    derived; everything downstream of that is one path, so a plane cannot be
    written by one route and its exception table by another.
    """
    data = np.asarray(values, dtype=np.float32)
    if not codec.encoding.se.is_residual:
        _write_se_arrays(
            group,
            codec,
            data.astype(np.float16),
            None,
            None,
            role=role,
            compressor=compressor,
            chunks=chunks,
        )
        return
    if coefficients is None:
        raise ValueError("residual SE needs se_coefficients")
    exceptions = SeExceptionBuilder()
    codes = codec.encode_se(
        data,
        eaf=eaf,
        analysis_index=analysis_index,
        coefficients=coefficients,
        positions=positions,
        exceptions=exceptions,
    )
    _write_se_arrays(
        group,
        codec,
        codes,
        coefficients,
        exceptions,
        role=role,
        compressor=compressor,
        chunks=chunks,
    )


def write_se_dense(
    group: Any,
    codec: StoreCodec,
    values: np.ndarray,
    eaf: np.ndarray,
    coefficients: np.ndarray | None,
    *,
    compressor: Any = None,
    chunks: tuple[int, ...] | None = None,
) -> None:
    """Write a Dense SE plane and all residual decode parameters atomically."""
    data = np.asarray(values, dtype=np.float32)
    n_analyses = data.shape[1]
    _write_se_plane(
        group,
        codec,
        data,
        eaf,
        np.broadcast_to(np.arange(n_analyses), data.shape),
        coefficients,
        positions_row_band(0, n_analyses),
        compressor,
        chunks,
        ArrayRole.DENSE_STATISTIC_PLANE,
    )


def write_se_csr(
    group: Any,
    codec: StoreCodec,
    values: np.ndarray,
    eaf: np.ndarray,
    analysis_index: np.ndarray,
    coefficients: np.ndarray | None,
    *,
    compressor: Any = None,
    chunks: tuple[int, ...] | None = None,
) -> None:
    """Write a Ragged SE plane with ordinal-keyed exact exceptions."""
    _write_se_plane(
        group,
        codec,
        values,
        eaf,
        analysis_index,
        coefficients,
        positions_flat(0),
        compressor,
        chunks,
        ArrayRole.ASSOCIATION_SEQUENCE,
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
    exceptions = EafExceptionBuilder()
    per_cell = (
        None
        if baseline is None
        else np.asarray(baseline, dtype=np.float32)[np.asarray(variant_index, dtype=np.int64)]
    )
    codes = codec.encode_eaf(
        values, baseline=per_cell, positions=positions_flat(0), exceptions=exceptions
    )
    create_array(
        group,
        "eaf",
        ArrayRole.ASSOCIATION_SEQUENCE,
        data=codes,
        dtype=codec.eaf_dtype,
        compressor=compressor,
        hint=chunks,
        overwrite=True,
    )
    if encoding.is_residual:
        assert baseline is not None
        write_eaf_baseline(group, baseline, compressor=compressor)
        exceptions.table().write(group)
