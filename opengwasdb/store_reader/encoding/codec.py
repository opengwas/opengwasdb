"""`StoreDecoder` -- the only place stored statistic bytes become physical values.

Every builder, completion pass, query adapter and validation rule that touches
`z` goes through this module. The point is not the arithmetic, which is three
lines; it is that there is **one site**. Encoding multiplies the number of
places a plane's bytes are interpreted, and every silent defect this release
stage has closed was a single call site out of step with its siblings.

`decode_z` is independently meaningful. `decode_se` deliberately requires
decoded EAF, Analysis indices, coefficients, and flat positions: there is no
raw-only convenience that could silently interpret residual codes as physical
Standard Errors.

Cells an integer plane cannot hold live in a sparse side table keyed by the
cell's **flat position** in its plane -- row × n_analyses + column for a Dense
grid, the association's ordinal for a CSR. `z` calls its table an *overflow*
table (a value beyond the fixed-point range) and `eaf` an *exception* table (a
residual beyond the range, a frequency of 0 or 1, or a variant with no usable
baseline), but they are the same structure and the same lookup. Decoding such a
cell needs to know where it is, which is why the read helpers take a position
resolver rather than just the bytes.

`eaf` decoding needs two things beyond the bytes: the variant's baseline, and
-- on a Reference-Completed release that carries one -- the imputed mask and
the per-variant reference frequency. `decode_eaf` requires them rather than
defaulting them, because the failure mode of getting this half-right is a
panel frequency silently standing in for a cohort's own, which real data says
can be wrong by 3000x (ADR 0037 §4).

Encoding is `opengwasdb.encoding.codec`'s (#464): its `StoreCodec` is this
decoder with the encoders added, and it writes the side tables these read.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, NoReturn, Self

import numpy as np

from opengwasdb.store_reader.arrays import array_length
from opengwasdb.store_reader.encoding.plan import (
    EAF_ABSENT,
    EAF_EXCEPTION,
    N_MISSING,
    SE_EXCEPTION,
    SE_MISSING,
    Z_MISSING,
    Z_OVERFLOW,
    EafBaselineError,
    StoreEncoding,
)

#: The key of a residual SE plane's per-Analysis model, beside the plane.
SE_COEFFICIENTS = "se_coefficients"
SE_EXCEPTION_INDEX = "se_exception_index"
SE_EXCEPTION_VALUE = "se_exception_value"

#: Where a plane's overflow table lives, in the same zarr group as the plane.
Z_OVERFLOW_INDEX = "z_overflow_index"
Z_OVERFLOW_VALUE = "z_overflow_value"

#: Where an `eaf` plane's exception table and per-variant baseline live.
EAF_EXCEPTION_INDEX = "eaf_exception_index"
EAF_EXCEPTION_VALUE = "eaf_exception_value"
EAF_BASELINE = "eaf_baseline"

#: The per-variant reference-panel frequency for a component's imputed cells.
EAF_REFERENCE = "eaf_reference"

#: Flat positions of some subset of a plane's cells: either an array parallel
#: to the values, or a callable taking the boolean mask of the cells that need
#: resolving and returning their positions. The callable form exists so a read
#: of a 10M-cell column does not have to materialise 10M positions to resolve
#: the two that overflowed.
PositionSource = np.ndarray | Callable[[np.ndarray], np.ndarray] | None


def positions_for(mask: np.ndarray, positions: PositionSource, *, what: str) -> np.ndarray:
    if positions is None:
        raise ValueError(
            f"{what}: this plane has out-of-range (overflow) z cells, whose exact values "
            "live in the plane's overflow table keyed by flat position, but no positions "
            "were supplied to resolve them"
        )
    if callable(positions):
        return np.asarray(positions(mask), dtype=np.int64)
    resolved = np.asarray(positions)
    if resolved.shape != mask.shape:
        raise ValueError(
            f"{what}: positions shape {resolved.shape} does not match values shape {mask.shape}"
        )
    return np.asarray(resolved[mask], dtype=np.int64)


def se_float16(values: np.ndarray) -> np.ndarray:
    """SEs as a `float16` SE plane holds them: the one rounding rule for an SE.

    Rounded to nearest, so an SE above 65,504 and below 65,520 becomes 65,504,
    and one of 65,520 or more becomes `+inf`; either stays an association (spec
    §15, issue #337). Everything that needs an SE as a `float16` plane holds it
    comes through here or through `StoreCodec.float16_se`, so a plane cannot be
    measured under one rounding and written under another. A build holds its SEs
    at the source's `float32` until a plan is chosen, because a residual plan's
    exact exception is that value (issue #355, spec §6a).
    """
    with np.errstate(over="ignore"):
        narrowed: np.ndarray = np.asarray(values).astype(np.float16)
    return narrowed


def se_log_prediction(
    eaf: np.ndarray, analysis_index: np.ndarray, coefficients: np.ndarray
) -> np.ndarray:
    """`log(se)` as an Analysis' model predicts it from each cell's frequency.

    Unchecked: a model or index that cannot predict a cell yields a non-finite
    prediction, which `se_residual_codes` makes an exception. `decode_se` checks
    its inputs and then predicts through here, so both share the expression.
    """
    f = np.asarray(eaf, dtype=np.float64)
    ai = np.asarray(analysis_index, dtype=np.int64)
    coef = np.asarray(coefficients)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log(2.0 * f * (1.0 - f))
        return np.asarray(
            coef[ai, 0].astype(np.float64) + coef[ai, 1].astype(np.float64) * x,
            dtype=np.float64,
        )


def se_from_log(log_se: np.ndarray) -> np.ndarray:
    """The SE a decoded `log(se)` is read as: the one expression `decode_se` and
    `se_residual_codes` share, so the encoder judges the value a reader gets."""
    with np.errstate(over="ignore", invalid="ignore"):
        decoded: np.ndarray = np.exp(np.asarray(log_se, dtype=np.float64)).astype(np.float32)
    return decoded


@dataclass(frozen=True)
class SparseExactTable:
    """Exact `float32` values for the cells an integer plane cannot hold.

    Sorted by flat position so a lookup is a binary search over a contiguous
    `int64` array. Tiny by construction -- 6,346 cells of 24.8e9 for `z` on
    `ukb-b`, a 74 KB side table against a multi-gigabyte store (ADR 0037).

    Subclasses supply only the array names and the word they use for the cells
    they hold, so `z` and `eaf` cannot drift apart in how they sort, dedupe,
    look up or refuse an entry that is not there.
    """

    index: np.ndarray  # int64, sorted, unique
    value: np.ndarray  # float32

    #: Overridden per plane.
    index_name: ClassVar[str] = ""
    value_name: ClassVar[str] = ""
    what: ClassVar[str] = ""

    def __post_init__(self) -> None:
        if self.index.ndim != 1 or self.value.ndim != 1 or len(self.index) != len(self.value):
            raise ValueError(f"{self.what} table index and value must be parallel 1-D arrays")
        if self.index.dtype != np.int64 or self.value.dtype != np.float32:
            raise ValueError(f"{self.what} table must use int64 index and float32 value")
        if len(self.index) > 1 and np.any(self.index[1:] <= self.index[:-1]):
            raise ValueError(f"{self.what} table index must be sorted and unique")

    @classmethod
    def empty(cls) -> Self:
        return cls(index=np.empty(0, dtype=np.int64), value=np.empty(0, dtype=np.float32))

    def __len__(self) -> int:
        return int(len(self.index))

    def lookup(self, positions: np.ndarray) -> np.ndarray:
        """Exact values at `positions`. A position with no entry is an error:
        a store whose plane says "look in the table" and whose table does not
        say what the value was has lost the cell, and it must not read as
        anything.
        """
        positions = np.asarray(positions, dtype=np.int64)
        slot = np.searchsorted(self.index, positions)
        found = slot < len(self.index)
        hit = np.zeros(len(positions), dtype=bool)
        hit[found] = self.index[slot[found]] == positions[found]
        if not np.all(hit):
            missing = positions[~hit][:5].tolist()
            raise ValueError(
                f"{self.what} table has no entry for cell(s) at flat "
                f"position(s) {missing}; the store's plane and its table disagree"
            )
        return np.asarray(self.value[slot], dtype=np.float32)

    @classmethod
    def read(cls, group: Any) -> Self:
        """Read the table beside its plane; empty when the store has none."""
        if cls.index_name not in group or cls.value_name not in group:
            return cls.empty()
        return cls(
            index=np.asarray(group[cls.index_name][:], dtype=np.int64),
            value=np.asarray(group[cls.value_name][:], dtype=np.float32),
        )

    @classmethod
    def open(cls, group: Any) -> SparseExactTable:
        """A view of the table that reads only the window a lookup needs.

        The decode paths use this rather than `read`, which materialises the
        whole table: OGS-00011's EAF exception table is 180,396,687 entries
        (about 2.1 GiB), read at open and again per process.  This holds the
        arrays, not their bytes (ADR 0060, #252 review round 1).
        """
        if cls.index_name not in group or cls.value_name not in group:
            return cls.empty()
        return WindowedExactTable.of(group, cls)


@dataclass(frozen=True)
class WindowedExactTable(SparseExactTable):
    """A lazy view of an exact table stored in a group.

    `lookup` reads only the sorted index's window covering the requested
    positions, by bisecting the on-disk index one inner chunk at a time, so a
    process never holds the whole table and a query's table read is O(cells in
    the answer).  The bisect costs O(log(E / chunk)) inner-chunk reads -- about
    10 chunk reads of 200,000 int64 for OGS-00011's 180,396,687-entry table.
    A clustered set is then read as one span; a scattered set is scanned in
    multi-chunk windows, each read whole when it holds enough positions and
    chunk by chunk when it holds few (review round 4, finding 2).
    """

    index_zarr: Any = None
    value_zarr: Any = None
    what_name: str = ""
    #: Table entries one scattered-scan window holds: read in one zarr call and
    #: matched with one vectorised `searchsorted`.  About 96 MB per window for
    #: an int64 index plus a float32 value (review round 3, finding 1).
    scan_window_entries: ClassVar[int] = 8_000_000
    #: Positions a scan window must hold to be worth reading whole.  A chunk's
    #: index plus value costs about 12 ms and a window about 95-100 ms, so the
    #: two strategies break even near ten positions; below this, each distinct
    #: chunk is read on its own and a sparse set does not pay the window
    #: (review round 4, finding 2).
    dense_window_positions: ClassVar[int] = 12
    #: The first and last element of each chunk a lookup has read or bisected
    #: through.  Cached per table for the process's life: a second lookup over
    #: the same table reuses them instead of re-reading the chunks
    #: (review round 2, finding 3).
    chunk_first: dict[int, int] = field(default_factory=dict)
    chunk_last: dict[int, int] = field(default_factory=dict)

    @classmethod
    def of(cls, group: Any, table_type: type[SparseExactTable]) -> WindowedExactTable:
        return cls(
            index=np.empty(0, dtype=np.int64),
            value=np.empty(0, dtype=np.float32),
            index_zarr=group[table_type.index_name],
            value_zarr=group[table_type.value_name],
            what_name=table_type.what or table_type.index_name,
        )

    def _chunk_edge(self, chunk_index: int, total: int, chunk: int, *, first: bool) -> int:
        """One chunk's first or last element, cached across lookups."""
        cache = self.chunk_first if first else self.chunk_last
        cached = cache.get(chunk_index)
        if cached is not None:
            return cached
        start = chunk_index * chunk
        stop = min(start + chunk, total)
        data = np.asarray(self.index_zarr[start:stop], dtype=np.int64)
        self.chunk_first[chunk_index] = int(data[0])
        self.chunk_last[chunk_index] = int(data[-1])
        return self.chunk_first[chunk_index] if first else self.chunk_last[chunk_index]

    def _chunk_after(self, target: int, *, strict: bool) -> tuple[int, int, int]:
        """Bisect the index chunks, returning `(chunk index, inner chunk, total)`.

        `strict=False` finds the first chunk whose **last** element is `>=`
        target (the chunk that could hold a lower bound); `strict=True` finds
        the first chunk whose **first** element is `>` target (one past the
        last chunk that can hold it).  Each step reads one chunk's edge, cached
        afterwards, so a repeated lookup pays nothing for the bisect
        (review round 2, finding 3).
        """
        total = array_length(self.index_zarr)
        if total == 0:
            return 0, 1, 0
        chunk = max(1, int(self.index_zarr.chunks[0]))
        n_chunks = max(1, -(-total // chunk))
        low, high = 0, n_chunks
        while low < high:
            mid = (low + high) // 2
            edge = self._chunk_edge(mid, total, chunk, first=strict)
            past = edge <= target if strict else edge < target
            if past:
                low = mid + 1
            else:
                high = mid
        return low, chunk, total

    def _chunk(self, chunk_index: int, total: int, chunk: int) -> tuple[np.ndarray, np.ndarray]:
        """One inner chunk's index and value, the unit a lookup reads."""
        start = chunk_index * chunk
        stop = min(start + chunk, total)
        index = np.asarray(self.index_zarr[start:stop], dtype=np.int64)
        value = np.asarray(self.value_zarr[start:stop], dtype=np.float32)
        if len(index):
            self.chunk_first[chunk_index] = int(index[0])
            self.chunk_last[chunk_index] = int(index[-1])
        return index, value

    def _scan_chunks(
        self,
        sorted_positions: np.ndarray,
        order: np.ndarray,
        out: np.ndarray,
        found: np.ndarray,
        total: int,
        chunk: int,
    ) -> None:
        """Match sorted positions against the table, by density.

        A window is a whole number of chunks.  If it holds at least
        `dense_window_positions` of the lookup's positions -- a Dense column,
        or any clustered set -- it is read in **one** zarr call and matched
        with one vectorised `searchsorted`, so a Dense column's ~1 M exception
        cells over a 165 M-entry table cost O(span / window) reads rather than
        one read per inner chunk (review round 3, finding 1).  A window
        holding fewer positions is not read: each distinct chunk under it is
        read on its own, so a **sparse** scattered set costs one chunk per
        position and not a ~96 MB window per cell (review round 4, finding 2).
        Peak memory is one window plus the positions.
        """
        window = max(int(self.scan_window_entries), chunk)
        # A window must be a whole number of chunks: a chunk straddling a
        # partial window's end could hold a target beyond it, and the loop
        # would not advance (review round 4, finding 3).
        window -= window % chunk
        i = 0
        n = len(sorted_positions)
        while i < n:
            chunk_index = self._chunk_after(int(sorted_positions[i]), strict=False)[0]
            if chunk_index * chunk >= total:
                break
            start = (chunk_index * chunk // window) * window
            stop = min(start + window, total)
            j = self._window_end(sorted_positions, i, stop, total, chunk)
            if j == i:
                raise RuntimeError(
                    f"the scattered scan made no progress at position "
                    f"{int(sorted_positions[i])} (window of {window} entries, "
                    f"chunk of {chunk}); the window does not cover its own "
                    f"first position"
                )
            if j - i >= self.dense_window_positions:
                self._read_window(
                    start, stop, sorted_positions[i:j], order[i:j], out, found
                )
            else:
                self._match_chunks(
                    sorted_positions[i:j], order[i:j], out, found, total, chunk
                )
            i = j

    def _window_end(
        self,
        sorted_positions: np.ndarray,
        i: int,
        stop: int,
        total: int,
        chunk: int,
    ) -> int:
        """One past the last position the window ending at `stop` covers."""
        high = self._chunk_edge((stop - 1) // chunk, total, chunk, first=False)
        return i + int(np.searchsorted(sorted_positions[i:], high, side="right"))

    def _read_window(
        self,
        start: int,
        stop: int,
        segment: np.ndarray,
        segment_order: np.ndarray,
        out: np.ndarray,
        found: np.ndarray,
    ) -> None:
        """Read a dense window's index and value, then match the segment."""
        block_index = np.asarray(self.index_zarr[start:stop], dtype=np.int64)
        block_value = np.asarray(self.value_zarr[start:stop], dtype=np.float32)
        self._match_block(block_index, block_value, segment, segment_order, out, found)

    @staticmethod
    def _match_block(
        block_index: np.ndarray,
        block_value: np.ndarray,
        segment: np.ndarray,
        segment_order: np.ndarray,
        out: np.ndarray,
        found: np.ndarray,
    ) -> None:
        """One `searchsorted` over a read window's entries."""
        slot = np.searchsorted(block_index, segment)
        in_bounds = slot < len(block_index)
        hit = np.zeros(len(segment), dtype=bool)
        hit[in_bounds] = block_index[slot[in_bounds]] == segment[in_bounds]
        matched = segment_order[hit]
        out[matched] = block_value[slot[hit]]
        found[matched] = True

    def _match_chunks(
        self,
        segment: np.ndarray,
        segment_order: np.ndarray,
        out: np.ndarray,
        found: np.ndarray,
        total: int,
        chunk: int,
    ) -> None:
        """Read each distinct chunk a sparse segment falls in, once."""
        blocks: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for k, target in enumerate(segment):
            target = int(target)
            chunk_index = self._chunk_after(target, strict=False)[0]
            if chunk_index not in blocks:
                blocks[chunk_index] = self._chunk(chunk_index, total, chunk)
            block_index, block_value = blocks[chunk_index]
            slot = int(np.searchsorted(block_index, target, side="left"))
            if slot < len(block_index) and int(block_index[slot]) == target:
                out[segment_order[k]] = block_value[slot]
                found[segment_order[k]] = True

    def _first_ge(self, target: int) -> int:
        """Table-entry index of the first entry `>= target` (an element index)."""
        total = array_length(self.index_zarr)
        chunk_index, chunk, _ = self._chunk_after(target, strict=False)
        if chunk_index * chunk >= total:
            return total
        block, _ = self._chunk(chunk_index, total, chunk)
        return chunk_index * chunk + int(np.searchsorted(block, target, side="left"))

    def _last_le(self, target: int) -> int:
        """One past the table-entry index of the last entry `<= target`."""
        total = array_length(self.index_zarr)
        chunk_index, chunk, _ = self._chunk_after(target, strict=True)
        if chunk_index == 0:
            return 0
        block, _ = self._chunk(chunk_index - 1, total, chunk)
        start = (chunk_index - 1) * chunk
        return start + int(np.searchsorted(block, target, side="right"))

    def _lookup_span(self, first: int, last: int, positions: np.ndarray) -> np.ndarray:
        """One vectorised `searchsorted` over the entries `[first, last)`."""
        index = np.asarray(self.index_zarr[first:last], dtype=np.int64)
        value = np.asarray(self.value_zarr[first:last], dtype=np.float32)
        slot = np.searchsorted(index, positions)
        found = slot < len(index)
        hit = np.zeros(len(positions), dtype=bool)
        hit[found] = index[slot[found]] == positions[found]
        if not np.all(hit):
            self._raise_missing(positions[~hit])
        return np.asarray(value[slot], dtype=np.float32)

    def _raise_missing(self, missing_positions: np.ndarray) -> NoReturn:
        missing = np.asarray(missing_positions, dtype=np.int64)[:5].tolist()
        raise ValueError(
            f"{self.what_name} table has no entry for cell(s) at flat "
            f"position(s) {missing}; the store's plane and its table disagree"
        )

    def lookup(self, positions: np.ndarray) -> np.ndarray:
        """Exact values at `positions`, reading only the entries they cover.

        A **clustered** set -- a decode block, or a whole Analysis's contiguous
        range -- is read as one slice of the table and `searchsorted` once, so
        its cost is O(table entries in the range) and it matches the eager
        lookup the code used before the table was windowed (review round 2,
        finding 3's regression).  A **scattered** set, where the entries between
        the first and last position would be far more than the positions
        themselves, is scanned window by window: a window holding enough of the
        positions is read whole, a sparser window reads only the distinct
        chunks its positions fall in (review round 4, finding 2).
        """
        positions = np.asarray(positions, dtype=np.int64)
        if len(positions) == 0:
            return np.empty(0, dtype=np.float32)
        if array_length(self.index_zarr) == 0:
            self._raise_missing(positions)
        first = self._first_ge(int(positions.min()))
        last = self._last_le(int(positions.max()))
        if last - first <= 4 * len(positions) + 1024:
            return self._lookup_span(first, last, positions)
        order = np.argsort(positions, kind="stable")
        out = np.empty(len(positions), dtype=np.float32)
        found = np.zeros(len(positions), dtype=bool)
        self._scan_chunks(
            positions[order],
            order,
            out,
            found,
            array_length(self.index_zarr),
            max(1, int(self.index_zarr.chunks[0])),
        )
        if not np.all(found):
            self._raise_missing(positions[~found])
        return out


@dataclass(frozen=True)
class ZOverflowTable(SparseExactTable):
    """Exact `float32` z for the cells the fixed-point plane cannot hold.

    Not a clip and not a build failure: the `ukb-b` survey found 568 genuine
    associations above |z| = 64 (ADR 0037 §1).
    """

    index_name: ClassVar[str] = Z_OVERFLOW_INDEX
    value_name: ClassVar[str] = Z_OVERFLOW_VALUE
    what: ClassVar[str] = "z overflow"


@dataclass(frozen=True)
class EafExceptionTable(SparseExactTable):
    """Exact `float32` EAF for the cells the residual coding cannot express.

    Three kinds of cell land here, and all three are stored exactly rather
    than approximated: a residual outside the declared range, a frequency of
    exactly 0 or 1 (which has no logit), and a cell at a variant with no
    usable baseline (ADR 0037 §2).
    """

    index_name: ClassVar[str] = EAF_EXCEPTION_INDEX
    value_name: ClassVar[str] = EAF_EXCEPTION_VALUE
    what: ClassVar[str] = "eaf exception"


@dataclass(frozen=True)
class SeExceptionTable(SparseExactTable):
    """Exact physical SE for cells not representable by the residual plane."""

    index_name: ClassVar[str] = SE_EXCEPTION_INDEX
    value_name: ClassVar[str] = SE_EXCEPTION_VALUE
    what: ClassVar[str] = "se exception"


def logit(f: np.ndarray) -> np.ndarray:
    """`log(f / (1 - f))`, NaN outside the open interval (0, 1)."""
    f = np.asarray(f, dtype=np.float64)
    out = np.full(f.shape, np.nan, dtype=np.float64)
    usable = np.isfinite(f) & (f > 0.0) & (f < 1.0)
    out[usable] = np.log(f[usable]) - np.log1p(-f[usable])
    return out


def expit(x: np.ndarray) -> np.ndarray:
    """The inverse of `logit`, evaluated without overflowing at either tail."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty(x.shape, dtype=np.float64)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    tail = np.exp(x[~positive])
    out[~positive] = tail / (1.0 + tail)
    return out


class StoreDecoder:
    """Decode a store's statistic planes under a declared plan.

    One instance per plane-bearing component: a Hybrid release's Dense
    Component and its Ragged Overflow each carry their own overflow table
    while sharing one `StoreEncoding`.
    """

    def __init__(
        self,
        encoding: StoreEncoding,
        *,
        z_overflow: SparseExactTable | None = None,
        eaf_exceptions: SparseExactTable | None = None,
        se_exceptions: SparseExactTable | None = None,
    ) -> None:
        self.encoding = encoding
        self.z_overflow = z_overflow
        self.eaf_exceptions = eaf_exceptions
        self.se_exceptions = se_exceptions

    # ---- se --------------------------------------------------------------

    def float16_se(self, values: np.ndarray) -> np.ndarray:
        """These SEs as a `float16` plan's plane stores them, at `float16`.

        The non-residual plan's rounding, asked of the codec so no builder or
        checker narrows an SE itself (issue #355). A Dense build harvests a
        top-hit candidate's SE this way before its plan is decided, because
        under a `float16` plan that is the value a query decodes at the cell;
        `write_top_hit_indexes_for_store` re-reads the plane under a residual
        one, whose cells hold codes. Validation re-applies it to a source value
        to compare the cell a `float16` plane stores, and to a table entry to
        ask whether that entry is one such rounding (spec §20.0).
        """
        return se_float16(values)

    @staticmethod
    def _se_prediction(
        eaf: np.ndarray, analysis_index: np.ndarray, coefficients: np.ndarray
    ) -> np.ndarray:
        StoreDecoder._require_se_model(eaf, analysis_index, coefficients)
        return se_log_prediction(eaf, analysis_index, coefficients)

    @staticmethod
    def _require_se_model(
        eaf: np.ndarray, analysis_index: np.ndarray, coefficients: np.ndarray
    ) -> None:
        """Refuse a model or an index that cannot predict these cells."""
        f_shape = np.shape(eaf)
        ai = np.asarray(analysis_index, dtype=np.int64)
        coef = np.asarray(coefficients)
        if coef.ndim != 2 or coef.shape[1] != 2:
            raise ValueError(f"se_coefficients shape {coef.shape} must be (n_analyses, 2)")
        if not np.all(np.isfinite(coef)):
            raise ValueError("se_coefficients must contain only finite values")
        if ai.shape != f_shape:
            raise ValueError(f"analysis_index shape {ai.shape} does not match EAF shape {f_shape}")
        if np.any(ai < 0) or np.any(ai >= len(coef)):
            raise ValueError("analysis_index is outside se_coefficients")

    @staticmethod
    def _checked_se_inputs(codes: np.ndarray, eaf: np.ndarray) -> np.ndarray:
        """The frequency plane a residual decode needs, or why it cannot be used."""
        if codes.dtype != np.int8:
            raise ValueError(
                f"se plane has dtype {codes.dtype}, but manifest declares int8_residual"
            )
        f = np.asarray(eaf, dtype=np.float32)
        if f.shape != codes.shape:
            raise ValueError(f"EAF shape {f.shape} does not match se shape {codes.shape}")
        return f

    def decode_se(
        self,
        raw: np.ndarray,
        *,
        eaf: np.ndarray,
        analysis_index: np.ndarray,
        coefficients: np.ndarray,
        positions: PositionSource = None,
    ) -> np.ndarray:
        codes = np.asarray(raw)
        if not self.encoding.se.is_residual:
            if codes.dtype.kind in "iu":
                raise ValueError(f"se plane has dtype {codes.dtype}, but manifest declares float16")
            return codes.astype(np.float32)
        f = self._checked_se_inputs(codes, eaf)
        exceptional = codes == SE_EXCEPTION
        # Every cell carrying a standard error owes a finite frequency, exact
        # exceptions included. `encode_se` refuses to create one without a
        # frequency, so a plane that has one did not come from this encoder and
        # guessing what it meant is worse than refusing it (issue #159).
        present = codes != SE_MISSING
        if np.any(present & ~np.isfinite(f)):
            raise ValueError(
                "residual SE cannot be decoded without a finite EAF for every cell carrying "
                "a standard error"
            )
        predicted = present & ~exceptional
        prediction = self._se_prediction(f, analysis_index, coefficients)
        out = se_from_log(prediction + codes.astype(np.float64) * self.encoding.se.step)
        out[codes == SE_MISSING] = np.nan
        if np.any(exceptional):
            if self.se_exceptions is None:
                raise ValueError("se plane has exception cells but no exception table")
            out[exceptional] = self.se_exceptions.lookup(
                positions_for(exceptional, positions, what="decode_se")
            )
        if np.any(predicted & ~np.isfinite(out)):
            raise ValueError("se residual prediction is non-finite")
        return out

    # ---- z ---------------------------------------------------------------

    @property
    def z_dtype(self) -> str:
        return self.encoding.z.dtype

    @property
    def z_fill_value(self) -> Any:
        """The value a freshly created z plane is filled with: "missing"."""
        return Z_MISSING if self.encoding.z.is_fixed_point else float("nan")

    @property
    def se_fill_value(self) -> Any:
        """The value a freshly created `se` plane is filled with: "missing".

        A `residual` plane's missing marker is its reserved sentinel, not a
        NaN it cannot hold (ADR 0038 §6). An Indexed Variant Subset creates its
        planes before writing every band, so the fill is what an unwritten cell
        would read as -- and it must be the missing marker, not the dtype
        default.
        """
        return SE_MISSING if self.encoding.se.is_residual else float("nan")

    def decode_z(self, raw: np.ndarray, *, positions: PositionSource = None) -> np.ndarray:
        """Stored bytes -> `float32` z-scores, with NaN for missing cells."""
        codes = np.asarray(raw)
        if not self.encoding.z.is_fixed_point:
            # The guard matters more in this direction than the other: reading
            # a fixed-point plane as `float16` does not fail, it returns
            # z-scores a thousand times too large.
            if codes.dtype.kind in "iu":
                raise ValueError(
                    f"z plane has dtype {codes.dtype}, but this store declares "
                    f"{self.encoding.z.kind}; the store and its manifest disagree"
                )
            return np.asarray(codes, dtype=np.float32)

        if codes.dtype != np.int16:
            raise ValueError(
                f"z plane has dtype {codes.dtype}, but this store declares "
                f"{self.encoding.z.kind}; the store and its manifest disagree"
            )
        scale = self.encoding.z.scale
        assert scale is not None
        out = codes.astype(np.float32) / np.float32(scale)
        out[codes == Z_MISSING] = np.float32(np.nan)
        out_of_range = codes == Z_OVERFLOW
        if np.any(out_of_range):
            if self.z_overflow is None:
                raise ValueError(
                    "z plane has out-of-range (overflow) cells but this codec was built "
                    "without the plane's overflow table"
                )
            out[out_of_range] = self.z_overflow.lookup(
                positions_for(out_of_range, positions, what="decode_z")
            )
        return out

    def missing_mask(self, raw: np.ndarray) -> np.ndarray:
        """Which cells of a stored z plane are missing (spec §15).

        A plane's missing marker is defined by its declared encoding, not by
        its dtype: NaN for a float plane, the reserved sentinel for an integer
        one. Callers checking missingness -- the paired-NaN invariant, the
        imputed-mask rules -- ask this rather than assuming NaN, which is what
        lets an integer plane express a contract it cannot hold a NaN for.
        """
        codes = np.asarray(raw)
        if not self.encoding.z.is_fixed_point:
            return np.asarray(np.isnan(codes), dtype=bool)
        return np.asarray(codes == Z_MISSING, dtype=bool)

    # ---- eaf --------------------------------------------------------------

    @property
    def eaf_dtype(self) -> str:
        return self.encoding.eaf.dtype

    @property
    def eaf_fill_value(self) -> Any:
        """The value a freshly created `eaf` plane is filled with: "no EAF"."""
        return EAF_ABSENT if self.encoding.eaf.is_residual else float("nan")

    def decode_eaf(
        self,
        raw: np.ndarray,
        *,
        baseline: np.ndarray | None = None,
        positions: PositionSource = None,
        imputed: np.ndarray | None = None,
        reference: np.ndarray | None = None,
    ) -> np.ndarray:
        """Stored bytes -> `float32` frequencies, with NaN where there is none.

        On a release whose plan declares `eaf_reference`, `imputed` and
        `reference` are **required**, not optional: an imputed cell's frequency
        is the panel's, and a decode that quietly skipped the substitution
        would return NaN for every imputed cell in a store that has the value.
        An observed cell never falls back to the panel -- FinnGen's frequencies
        differ from the EUR panel by up to 3000x (ADR 0037 §4).
        """
        codes = np.asarray(raw)
        if not self.encoding.eaf.is_residual:
            if codes.dtype.kind in "iu":
                raise ValueError(
                    f"eaf plane has dtype {codes.dtype}, but this store declares "
                    f"{self.encoding.eaf.kind}; the store and its manifest disagree"
                )
            out = np.asarray(codes, dtype=np.float32)
            return self._apply_eaf_reference(out, imputed=imputed, reference=reference)

        if codes.dtype != np.int8:
            raise ValueError(
                f"eaf plane has dtype {codes.dtype}, but this store declares "
                f"{self.encoding.eaf.kind}; the store and its manifest disagree"
            )
        if baseline is None:
            raise EafBaselineError(
                "decoding an int8_residual eaf plane needs the per-cell eaf_baseline; "
                "an int8 residual plane is meaningless without it (ADR 0037)"
            )
        b = np.asarray(baseline, dtype=np.float32).astype(np.float64)
        if b.shape != codes.shape:
            raise EafBaselineError(
                f"baseline shape {b.shape} does not match eaf plane shape {codes.shape}"
            )
        with np.errstate(invalid="ignore"):
            out = expit(logit(b) + codes.astype(np.float64) * self.encoding.eaf.step)
        out = out.astype(np.float32)
        out[codes == EAF_ABSENT] = np.float32(np.nan)
        exceptional = codes == EAF_EXCEPTION
        if np.any(exceptional):
            if self.eaf_exceptions is None:
                raise ValueError(
                    "eaf plane has exception cells but this codec was built without the "
                    "plane's exception table"
                )
            out[exceptional] = self.eaf_exceptions.lookup(
                positions_for(exceptional, positions, what="decode_eaf")
            )
        return self._apply_eaf_reference(out, imputed=imputed, reference=reference)

    def _apply_eaf_reference(
        self,
        decoded: np.ndarray,
        *,
        imputed: np.ndarray | None,
        reference: np.ndarray | None,
    ) -> np.ndarray:
        """Substitute panel frequencies on this component's imputed cells."""
        if not self.encoding.eaf.reference:
            return decoded
        if imputed is None or reference is None:
            raise EafBaselineError(
                "this release declares reference EAF for its imputed cells, so decoding "
                "needs the imputed mask and the per-cell eaf_reference; without them "
                "every imputed cell would read as NaN in a store that holds the value "
                "(ADR 0037 §4)"
            )
        mask = np.asarray(imputed).astype(bool)
        panel = np.asarray(reference, dtype=np.float32)
        if mask.shape != decoded.shape or panel.shape != decoded.shape:
            raise EafBaselineError(
                f"imputed mask {mask.shape} and reference {panel.shape} must both match "
                f"the decoded eaf shape {decoded.shape}"
            )
        decoded = decoded.copy()
        decoded[mask] = panel[mask]
        return decoded

    def eaf_absent_mask(self, raw: np.ndarray) -> np.ndarray:
        """Which cells of a stored `eaf` plane carry no frequency.

        Read through the plane's declared encoding, like `missing_mask` for
        `z`: NaN for a float plane, the reserved code for a residual one.
        """
        codes = np.asarray(raw)
        if not self.encoding.eaf.is_residual:
            return np.asarray(np.isnan(codes), dtype=bool)
        return np.asarray(codes == EAF_ABSENT, dtype=bool)

    # ---- n ---------------------------------------------------------------

    def decode_n(self, raw: np.ndarray) -> np.ndarray:
        """Decoded per-association sample sizes (ADR 0067).

        `float64`, which holds every `uint32` count exactly, with NaN for the
        reserved missing code -- the result contract's one missing marker. A
        release that declares no `n` plane has nothing to decode, and asking
        is a caller defect rather than a request for an all-missing column.
        """
        if self.encoding.n is None:
            raise ValueError("this release declares no n plane, so there is no n to decode")
        codes = np.asarray(raw)
        if codes.dtype != np.dtype(self.encoding.n.dtype):
            raise ValueError(
                f"n codes have dtype {codes.dtype} but the plan declares {self.encoding.n.dtype}"
            )
        out = codes.astype(np.float64)
        out[codes == N_MISSING] = np.nan
        return out


def positions_flat(start: int) -> Callable[[np.ndarray], np.ndarray]:
    """A contiguous 1-D run of a CSR plane starting at flat position `start`."""

    def resolve(mask: np.ndarray) -> np.ndarray:
        return np.flatnonzero(mask).astype(np.int64) + start

    return resolve


def positions_at(indices: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """An arbitrary gather from a 1-D plane at flat positions `indices`."""
    flat = np.asarray(indices, dtype=np.int64)

    def resolve(mask: np.ndarray) -> np.ndarray:
        return np.asarray(flat[mask], dtype=np.int64)

    return resolve


def positions_row_band(row_start: int, n_analyses: int) -> Callable[[np.ndarray], np.ndarray]:
    """A full-width row band of a Dense grid, rows `row_start...`."""
    offset = int(row_start) * int(n_analyses)

    def resolve(mask: np.ndarray) -> np.ndarray:
        return np.flatnonzero(mask).astype(np.int64) + offset

    return resolve


def positions_rows_cols(
    rows: Sequence[int] | np.ndarray, cols: Sequence[int] | np.ndarray, n_analyses: int
) -> Callable[[np.ndarray], np.ndarray]:
    """A Dense block at the cross product of `rows` x `cols`, in that order."""
    row_idx = np.asarray(rows, dtype=np.int64)
    col_idx = np.asarray(cols, dtype=np.int64)

    def resolve(mask: np.ndarray) -> np.ndarray:
        i, j = np.divmod(np.flatnonzero(mask).astype(np.int64), len(col_idx))
        positions: np.ndarray = row_idx[i] * n_analyses + col_idx[j]
        return positions

    return resolve


def positions_pairs(rows: np.ndarray, cols: np.ndarray, n_analyses: int) -> np.ndarray:
    """Elementwise Dense cells `(rows[i], cols[i])`."""
    return np.asarray(rows, dtype=np.int64) * int(n_analyses) + np.asarray(cols, dtype=np.int64)
