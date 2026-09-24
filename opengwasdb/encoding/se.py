"""Bounded-memory Dense SE fitting and rewriting after EAF is available."""

from __future__ import annotations

import logging
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from opengwasdb.build.ordered_pool import ordered_map
from opengwasdb.encoding.codec import (
    EXACT_TABLE_CHUNK,
    SE_EXCEPTION_INDEX,
    SE_EXCEPTION_VALUE,
    SeExceptionBuilder,
    SeExceptionTable,
    StoreCodec,
    positions_flat,
    positions_row_band,
    se_residual_codes,
)
from opengwasdb.encoding.measure import packed_chunk_bytes, solve_log_se
from opengwasdb.encoding.plan import (
    SE_MISSING,
    SE_RANGE_CANDIDATES,
    EncodingMeasurements,
    SeMeasurements,
    StoreEncoding,
)
from opengwasdb.encoding.planes import DenseEafPlane, write_se_coefficients
from opengwasdb.encoding.timing import PhaseTimer, log_phase, log_progress

log = logging.getLogger(__name__)


def _optional_phase(timer: PhaseTimer | None, name: str) -> AbstractContextManager[None]:
    """A timer phase when one is being kept, a no-op otherwise.

    The per-chunk passes run inside a worker when a pool is in use, where a
    parent's ``PhaseTimer`` cannot be updated from the child. A caller therefore
    keeps the fine-grained phases only on the ``n_workers <= 1`` path and
    charges one coarse phase around the whole pool instead.
    """
    return timer.phase(name) if timer is not None else nullcontext()


@dataclass(frozen=True, eq=False, kw_only=True)
class OverflowCells:
    """A named, validated Hybrid Overflow Component cell bundle.

    SE values, EAF values and Analysis indices travel together everywhere the
    Hybrid residual-SE optimiser fits or measures a shared plan. Naming them
    (and validating them here) turns the old ``(se, eaf, analysis_index)``
    positional tuple into something a swapped array, a wrong rank, or an
    out-of-range Analysis index cannot pass silently (issue #162).
    """

    se_values: np.ndarray
    eaf_values: np.ndarray
    analysis_indices: np.ndarray
    n_analyses: int

    def __post_init__(self) -> None:
        se = _normalise_overflow_values(
            self.se_values, "OverflowCells se_values and eaf_values must be one-dimensional"
        )
        eaf = _normalise_overflow_values(
            self.eaf_values, "OverflowCells se_values and eaf_values must be one-dimensional"
        )
        if len(se) != len(eaf):
            raise ValueError("OverflowCells se_values and eaf_values must have equal lengths")
        indices = _normalise_analysis_indices(self.analysis_indices, len(se))
        n_analyses = _normalise_analysis_count(self.n_analyses)
        if np.any(indices < 0) or np.any(indices >= n_analyses):
            raise ValueError(
                f"OverflowCells analysis_indices must lie within [0, {n_analyses})"
            )

        object.__setattr__(self, "se_values", se)
        object.__setattr__(self, "eaf_values", eaf)
        object.__setattr__(self, "analysis_indices", indices)
        object.__setattr__(self, "n_analyses", n_analyses)


def _normalise_overflow_values(values: Any, message: str) -> np.ndarray:
    """One OverflowCells parallel vector as a one-dimensional array, or `message`."""
    out = np.asarray(values)
    if out.ndim != 1:
        raise ValueError(message)
    return out


def _normalise_analysis_indices(values: Any, n: int) -> np.ndarray:
    """The OverflowCells Analysis-index vector as canonical ``int64``.

    Validates rank, the required length and integral values, keeping the
    vector otherwise untouched so callers see exactly what they stored.
    """
    ai = np.asarray(values)
    if ai.ndim != 1:
        raise ValueError("OverflowCells analysis_indices must be one-dimensional")
    if len(ai) != n:
        raise ValueError(
            "OverflowCells analysis_indices length must match se_values/eaf_values"
        )
    if ai.dtype.kind not in ("i", "u"):
        if (
            ai.dtype.kind != "f"
            or not np.all(np.isfinite(ai))
            or not np.all(np.mod(ai, 1.0) == 0.0)
        ):
            raise ValueError("OverflowCells analysis_indices must be integers")
    return ai.astype(np.int64, copy=False)


def _normalise_analysis_count(value: Any) -> int:
    """The non-negative Analysis count OverflowCells requires, as an ``int``."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("OverflowCells n_analyses must be a non-negative integer")
    n_analyses: int = int(value)
    if n_analyses < 0:
        raise ValueError("OverflowCells n_analyses must be a non-negative integer")
    return n_analyses


class _ComponentCost(NamedTuple):
    """What one component's cells cost, per candidate residual range.

    `n_finite` and `exception_counts` are per Analysis, not pooled: issue #118
    asks the plane to revert when *any* Analysis fits badly, and a pooled share
    lets one GCST007320-shaped Analysis hide behind its well-fitting
    neighbours -- which is precisely the case the issue was raised about.
    """

    float_bytes: int
    n_finite: np.ndarray
    exception_counts: dict[float, np.ndarray]
    candidate_bytes: dict[float, int]

    @classmethod
    def empty(cls, n_analyses: int = 0) -> _ComponentCost:
        zeros = np.zeros(n_analyses, dtype=np.int64)
        return cls(
            0,
            zeros,
            {candidate: zeros.copy() for candidate in SE_RANGE_CANDIDATES},
            dict.fromkeys(SE_RANGE_CANDIDATES, 0),
        )

    @property
    def total_finite(self) -> int:
        return int(self.n_finite.sum())


def _packed(compressor: Any, value: np.ndarray) -> int:
    data = np.ascontiguousarray(value)
    return len(compressor.encode(data)) if compressor is not None else data.nbytes


def _packed_coefficients(compressor: Any, coefficients: np.ndarray) -> int:
    """Coefficient-array bytes as `write_se_coefficients` will store them.

    The writer declares a (min(n_analyses, 1024), 2) chunk with the numeric
    default fill, so a row count that is not a multiple of 1024 leaves an edge
    chunk zarr pads to a full 1024 rows before compressing it. Each grid row
    is charged through the shared `packed_chunk_bytes`, which pads that edge
    chunk to its declared shape (issue #158).
    """
    chunk_rows = max(1, min(len(coefficients), 1024))
    return sum(
        packed_chunk_bytes(compressor, coefficients[r0 : r0 + chunk_rows], (chunk_rows, 2), 0)
        for r0 in range(0, len(coefficients), chunk_rows)
    )


class _SideTableCost:
    """Compressed bytes for one sorted side table, bounded to one chunk.

    The rewrite's arrays are pre-sized with a chunk of
    ``max(1, min(count, EXACT_TABLE_CHUNK))`` and written slot by slot, so a
    table that fits one chunk is stored whole and a longer one has a final
    edge chunk zarr pads to a full ``EXACT_TABLE_CHUNK`` with its fill. The
    in-extent rows are held in two fixed buffers and flushed as they fill; the
    final partial flush is charged through the shared `packed_chunk_bytes` so
    that edge chunk is not measured short (issue #158).
    """

    def __init__(self, compressor: Any, n_analyses: int) -> None:
        self._compressor = compressor
        self._index = np.empty(EXACT_TABLE_CHUNK, dtype=np.int64)
        self._value = np.empty(EXACT_TABLE_CHUNK, dtype=np.float32)
        self._used = 0
        self._flushed_rows = 0
        self.count = np.zeros(n_analyses, dtype=np.int64)
        self.compressed_bytes = 0

    def add(self, index: np.ndarray, value: np.ndarray, analyses: np.ndarray) -> None:
        index = np.asarray(index, dtype=np.int64)
        value = np.asarray(value, dtype=np.float32)
        self.count += np.bincount(np.asarray(analyses, dtype=np.int64), minlength=len(self.count))
        offset = 0
        while offset < len(index):
            take = min(EXACT_TABLE_CHUNK - self._used, len(index) - offset)
            end = offset + take
            self._index[self._used : self._used + take] = index[offset:end]
            self._value[self._used : self._used + take] = value[offset:end]
            self._used += take
            offset = end
            if self._used == EXACT_TABLE_CHUNK:
                self._flush()

    def _flush(self) -> None:
        self.compressed_bytes += _packed(self._compressor, self._index[: self._used])
        self.compressed_bytes += _packed(self._compressor, self._value[: self._used])
        self._flushed_rows += self._used
        self._used = 0

    def finish(self) -> tuple[np.ndarray, int]:
        if self._used:
            # A flush has happened only when the table outgrew one chunk, and
            # then the final chunk is stored at a full EXACT_TABLE_CHUNK with
            # its fill (0 for both the int64 index and the float32 value). A
            # table that fits one chunk is stored at its own length and must
            # not be padded up to a chunk it will never occupy.
            chunk = EXACT_TABLE_CHUNK if self._flushed_rows else self._used
            self.compressed_bytes += packed_chunk_bytes(
                self._compressor, self._index[: self._used], (chunk,), 0
            )
            self.compressed_bytes += packed_chunk_bytes(
                self._compressor, self._value[: self._used], (chunk,), 0
            )
            self._used = 0
        return self.count, self.compressed_bytes


def _add_fit_sums(
    se: np.ndarray,
    eaf: np.ndarray,
    analysis_index: np.ndarray,
    count: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    sxx: np.ndarray,
    sxy: np.ndarray,
) -> bool:
    values = np.asarray(se, dtype=np.float64).ravel()
    frequencies = np.asarray(eaf, dtype=np.float64).ravel()
    analyses = np.asarray(analysis_index, dtype=np.int64).ravel()
    finite = np.isfinite(values)
    eligible = bool(np.all(np.isfinite(frequencies[finite])))
    use = finite & (values > 0) & (frequencies > 0) & (frequencies < 1)
    if not np.any(use):
        return eligible
    x = np.log(2 * frequencies[use] * (1 - frequencies[use]))
    y = np.log(values[use])
    selected = analyses[use]
    length = len(count)
    count += np.bincount(selected, minlength=length)
    sx += np.bincount(selected, weights=x, minlength=length)
    sy += np.bincount(selected, weights=y, minlength=length)
    sxx += np.bincount(selected, weights=x * x, minlength=length)
    sxy += np.bincount(selected, weights=x * y, minlength=length)
    return eligible


def _candidate_codes(
    se: np.ndarray,
    eaf: np.ndarray,
    analysis_index: np.ndarray,
    coefficients: np.ndarray,
    candidate: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(se, dtype=np.float64)
    frequencies = np.asarray(eaf, dtype=np.float64)
    analyses = np.asarray(analysis_index, dtype=np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log(2 * frequencies * (1 - frequencies))
        prediction = coefficients[analyses, 0] + coefficients[analyses, 1] * x
        residual = np.log(values) - prediction
    return se_residual_codes(values, residual, candidate / 127)


def _charged(
    sides: dict[float, _SideTableCost], code_bytes: dict[float, int]
) -> tuple[dict[float, np.ndarray], dict[float, int]]:
    """Per-Analysis exception counts and total compressed bytes, per candidate."""
    counts: dict[float, np.ndarray] = {}
    total: dict[float, int] = {}
    for candidate, side in sides.items():
        counts[candidate], side_compressed = side.finish()
        total[candidate] = code_bytes[candidate] + side_compressed
    return counts, total


def _band_chunk_bytes(
    compressor: Any, band: np.ndarray, row_chunk: int, col_chunk: int, fill_value: Any
) -> int:
    """Compressed bytes of one row band stored as the plane's physical chunks.

    A band is one physical row-chunk cell of the plane; its final band is
    short when the plane's row extent does not divide the chunk. Each of the
    band's column cells is charged through the shared `packed_chunk_bytes`,
    which pads a cell running past the plane's extent out to the full declared
    chunk with the array's fill value before compressing it, exactly as zarr
    stores it (issue #158).
    """
    return sum(
        packed_chunk_bytes(
            compressor, band[:, c0 : c0 + col_chunk], (row_chunk, col_chunk), fill_value
        )
        for c0 in range(0, band.shape[1], col_chunk)
    )


@dataclass
class _MeasureContext:
    """The read-only state one SE-measurement worker band needs.

    Set as a module global before the pool forks and cleared after, so the
    zarr planes and coefficient array are inherited rather than pickled per
    chunk (`dense.build_vcf`'s Pass 2 does the same).
    """

    source: Any
    eaf_plane: DenseEafPlane
    coefficients: np.ndarray
    analysis_index: np.ndarray
    row_chunk: int
    col_chunk: int
    compressor: Any
    float16_fill: Any
    n_analyses: int
    timer: PhaseTimer | None


_MEASURE: _MeasureContext | None = None


class _BandMeasurement(NamedTuple):
    """One row chunk's contribution to the measured candidate costs.

    The exception rows leave the worker unaccumulated: the side table's
    compressed size depends on the order rows are appended in, so only the
    parent, which consumes chunks in row order, may charge it.
    """

    float_bytes: int
    code_bytes: dict[float, int]
    finite: np.ndarray
    exceptions: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]]


def _measure_one_band(r0: int) -> _BandMeasurement:
    """Measure one row chunk: its ``float16`` cost, candidate codes, exceptions."""
    ctx = _MEASURE
    if ctx is None:
        raise RuntimeError("SE measurement worker ran without a measurement context")
    r1 = min(r0 + ctx.row_chunk, int(ctx.source.shape[0]))
    with _optional_phase(ctx.timer, "measure.read"):
        values = np.asarray(ctx.source[r0:r1], dtype=np.float32)
        frequencies = ctx.eaf_plane.band(r0, r1)
    finite = np.isfinite(values).sum(axis=0).astype(np.int64)
    ai = ctx.analysis_index[: r1 - r0]
    with _optional_phase(ctx.timer, "measure.float16"):
        float_bytes = _band_chunk_bytes(
            ctx.compressor,
            values.astype(np.float16),
            ctx.row_chunk,
            ctx.col_chunk,
            ctx.float16_fill,
        )
    code_bytes: dict[float, int] = {}
    exceptions: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for candidate in SE_RANGE_CANDIDATES:
        with _optional_phase(ctx.timer, "measure.code"):
            raw, exceptional = _candidate_codes(
                values, frequencies, ai, ctx.coefficients, candidate
            )
        with _optional_phase(ctx.timer, "measure.compress"):
            code_bytes[candidate] = _band_chunk_bytes(
                ctx.compressor, raw, ctx.row_chunk, ctx.col_chunk, SE_MISSING
            )
        exceptions[candidate] = (
            positions_row_band(r0, ctx.n_analyses)(exceptional),
            values[exceptional],
            ai[exceptional],
        )
    return _BandMeasurement(float_bytes, code_bytes, finite, exceptions)


def _accumulate_measurement(
    band: _BandMeasurement,
    sides: dict[float, _SideTableCost],
    code_bytes: dict[float, int],
    chunk_timer: PhaseTimer | None,
) -> tuple[int, np.ndarray]:
    """Fold one band's measurement into the running total, in chunk order."""
    for candidate in SE_RANGE_CANDIDATES:
        code_bytes[candidate] += band.code_bytes[candidate]
        rows, values, analyses = band.exceptions[candidate]
        with _optional_phase(chunk_timer, "measure.exceptions"):
            sides[candidate].add(rows, values, analyses)
    return band.float_bytes, band.finite


def _measure_dense(
    source: Any,
    eaf_plane: DenseEafPlane,
    coefficients: np.ndarray,
    timer: PhaseTimer,
    n_workers: int,
) -> _ComponentCost:
    """Measure every candidate range over the plane, one row chunk at a time.

    Row chunks are independent, so ``n_workers > 1`` measures them in a fork
    pool. The reduction runs in row-chunk order -- the order the serial pass
    accumulates -- which is what keeps the exception side table's compressed
    size, and therefore the encoding decision, identical either way.
    """
    global _MEASURE
    n_rows, n_analyses = map(int, source.shape)
    row_chunk, col_chunk = map(int, source.chunks)
    compressor = source.compressor
    # The planes this decision writes declare their own fills: the int8 codes
    # plane `_rewrite_dense` produces is created with `SE_MISSING` as its fill,
    # and a float32 scratch plane is narrowed to `float16` with NaN. A source
    # already stored as `float16` (a migration input) is left untouched and
    # keeps the fill its own writer declared. Each measured plane is charged
    # at its padded size with the fill its own writer declares (issue #158).
    float16_fill: Any = (
        source.fill_value if source.dtype == np.dtype("float16") else float("nan")
    )
    sides = {candidate: _SideTableCost(compressor, n_analyses) for candidate in SE_RANGE_CANDIDATES}
    code_bytes = dict.fromkeys(SE_RANGE_CANDIDATES, 0)
    float_bytes = 0
    finite_per_analysis = np.zeros(n_analyses, dtype=np.int64)
    columns = np.arange(n_analyses, dtype=np.int64)
    starts = range(0, n_rows, row_chunk)
    n_chunks = len(starts)
    chunk_timer = timer if n_workers <= 1 else None
    _MEASURE = _MeasureContext(
        source, eaf_plane, coefficients,
        np.broadcast_to(columns, (row_chunk, n_analyses)),
        row_chunk, col_chunk, compressor, float16_fill, n_analyses, chunk_timer,
    )
    started = time.monotonic()
    try:
        with log_phase(log, "SE measurement"):
            umbrella = timer.phase("measure.parallel") if n_workers > 1 else nullcontext()
            with umbrella:
                bands = ordered_map(_measure_one_band, starts, n_workers)
                for index, band in enumerate(bands, start=1):
                    added_bytes, finite = _accumulate_measurement(
                        band, sides, code_bytes, chunk_timer
                    )
                    float_bytes += added_bytes
                    finite_per_analysis += finite
                    log_progress(
                        log, "SE measurement", index, n_chunks, started,
                        every=max(1, n_chunks // 20),
                    )
    finally:
        _MEASURE = None
    return _ComponentCost(float_bytes, finite_per_analysis, *_charged(sides, code_bytes))


@dataclass
class _OverflowContext:
    """Read-only state one overflow-measurement chunk needs (see `_MeasureContext`)."""

    values: np.ndarray
    frequencies: np.ndarray
    analyses: np.ndarray
    coefficients: np.ndarray
    compressor: Any
    chunk: int
    n_analyses: int


_OVERFLOW: _OverflowContext | None = None


def _measure_overflow_band(start: int) -> _BandMeasurement:
    """Measure one flat overflow chunk: codes, compression and exceptions."""
    ctx = _OVERFLOW
    if ctx is None:
        raise RuntimeError("SE overflow measurement ran without a context")
    end = min(start + ctx.chunk, len(ctx.values))
    se = np.asarray(ctx.values[start:end], dtype=np.float32)
    eaf = np.asarray(ctx.frequencies[start:end], dtype=np.float32)
    ai = np.asarray(ctx.analyses[start:end], dtype=np.int64)
    finite = np.bincount(ai[np.isfinite(se)], minlength=ctx.n_analyses).astype(np.int64)
    # The Ragged Overflow store writes both planes whole (`data=`, numeric
    # default fill), so every edge chunk -- the final one of a length that does
    # not divide the chunk -- is padded with 0 before it is compressed. Charged
    # through the shared `packed_chunk_bytes` (issue #158).
    float_bytes = packed_chunk_bytes(ctx.compressor, se.astype(np.float16), (ctx.chunk,), 0)
    code_bytes: dict[float, int] = {}
    exceptions: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for candidate in SE_RANGE_CANDIDATES:
        raw, exceptional = _candidate_codes(se, eaf, ai, ctx.coefficients, candidate)
        code_bytes[candidate] = packed_chunk_bytes(ctx.compressor, raw, (ctx.chunk,), 0)
        exceptions[candidate] = (
            positions_flat(start)(exceptional),
            se[exceptional],
            ai[exceptional],
        )
    return _BandMeasurement(float_bytes, code_bytes, finite, exceptions)


def _measure_overflow(
    overflow: OverflowCells,
    coefficients: np.ndarray,
    compressor: Any,
    chunk: int,
    n_analyses: int,
    timer: PhaseTimer | None = None,
    n_workers: int = 1,
) -> _ComponentCost:
    """A Hybrid Overflow Component's per-candidate costs, in its own chunks.

    The Overflow Component's flat planes are written whole (`data=`, numeric
    default fill) in `chunk`-sized chunks, so a cell count that does not
    divide the chunk leaves an edge chunk zarr pads with 0 before compressing
    it; both the `float16` alternative and every candidate's codes are charged
    at that padded size (issue #158). The flat chunks are independent and run
    across ``n_workers``; the parent folds their side tables in chunk order, as
    the serial pass does, so the decision is unchanged (issue #221).
    """
    global _OVERFLOW
    values = np.asarray(overflow.se_values).ravel()
    frequencies = np.asarray(overflow.eaf_values).ravel()
    analyses = np.asarray(overflow.analysis_indices).ravel()
    sides = {candidate: _SideTableCost(compressor, n_analyses) for candidate in SE_RANGE_CANDIDATES}
    code_bytes = dict.fromkeys(SE_RANGE_CANDIDATES, 0)
    float_bytes = 0
    finite_per_analysis = np.zeros(n_analyses, dtype=np.int64)
    starts = range(0, len(values), chunk)
    n_chunks = len(starts)
    _OVERFLOW = _OverflowContext(
        values, frequencies, analyses, coefficients, compressor, chunk, n_analyses
    )
    started = time.monotonic()
    try:
        with log_phase(log, "SE overflow measurement"):
            with _optional_phase(timer, "measure.overflow"):
                bands = ordered_map(_measure_overflow_band, starts, n_workers)
                for index, band in enumerate(bands, start=1):
                    added_bytes, finite = _accumulate_measurement(
                        band, sides, code_bytes, None
                    )
                    float_bytes += added_bytes
                    finite_per_analysis += finite
                    log_progress(
                        log, "SE overflow measurement", index, n_chunks, started,
                        every=max(1, n_chunks // 20),
                    )
    finally:
        _OVERFLOW = None
    return _ComponentCost(float_bytes, finite_per_analysis, *_charged(sides, code_bytes))


def _empty_exception_arrays(group: Any, count: int, compressor: Any) -> tuple[Any, Any]:
    """Allocate the side table the streaming rewrite fills in position order.

    Sized from the rewrite's codes-only count pass rather than grown: the
    rewrite visits row chunks in order, so the exceptions arrive already sorted
    and can be written straight into their final slots.
    """
    for name in (SE_EXCEPTION_INDEX, SE_EXCEPTION_VALUE):
        if name in group:
            del group[name]
    chunks = (max(1, min(count, EXACT_TABLE_CHUNK)),)
    return (
        group.create_dataset(
            SE_EXCEPTION_INDEX,
            shape=(count,),
            chunks=chunks,
            compressor=compressor,
            dtype="int64",
        ),
        group.create_dataset(
            SE_EXCEPTION_VALUE,
            shape=(count,),
            chunks=chunks,
            compressor=compressor,
            dtype="float32",
        ),
    )


@dataclass
class _CountContext:
    """Read-only state one codes-only count band needs (see `_MeasureContext`)."""

    source: Any
    eaf_plane: DenseEafPlane
    coefficients: np.ndarray
    residual_range: float
    analysis_index: np.ndarray
    row_chunk: int
    n_rows: int


_COUNT: _CountContext | None = None


def _count_one_band(r0: int) -> int:
    """The exception count one row chunk contributes under a decided range."""
    ctx = _COUNT
    if ctx is None:
        raise RuntimeError("SE rewrite count worker ran without a count context")
    r1 = min(r0 + ctx.row_chunk, ctx.n_rows)
    _, exceptional = _candidate_codes(
        np.asarray(ctx.source[r0:r1], dtype=np.float32),
        ctx.eaf_plane.band(r0, r1),
        ctx.analysis_index[: r1 - r0],
        ctx.coefficients,
        ctx.residual_range,
    )
    return int(exceptional.sum())


def _count_dense_exceptions(
    source: Any,
    eaf_plane: DenseEafPlane,
    coefficients: np.ndarray,
    residual_range: float,
    timer: PhaseTimer,
    n_workers: int,
) -> int:
    """Exact exception count for one range, from codes alone (issue #145).

    The rewrite sizes its side table from this pass rather than from the
    measurement: computing int8 codes costs none of the compression the
    measurement pays for, so the count survives #146 sampling the measurement
    while staying exact. It runs `_candidate_codes`, which shares
    `se_residual_codes` with the codec's `encode_se`, so the count it returns
    is the count the rewrite's own encode will produce -- and the rewrite's
    cursor check still fails loudly if the two ever disagree.
    """
    n_rows = int(source.shape[0])
    row_chunk = int(source.chunks[0])
    n_analyses = int(source.shape[1])
    analysis_index = np.broadcast_to(np.arange(n_analyses, dtype=np.int64), (row_chunk, n_analyses))
    starts = range(0, n_rows, row_chunk)
    n_chunks = len(starts)
    global _COUNT
    _COUNT = _CountContext(source, eaf_plane, coefficients, residual_range,
                           analysis_index, row_chunk, n_rows)
    count = 0
    started = time.monotonic()
    try:
        with log_phase(log, "SE rewrite count"):
            with timer.phase("rewrite.count"):
                counts = ordered_map(_count_one_band, starts, n_workers)
                for index, band_count in enumerate(counts, start=1):
                    count += band_count
                    log_progress(
                        log, "SE rewrite count", index, n_chunks, started,
                        every=max(1, n_chunks // 20),
                    )
    finally:
        _COUNT = None
    return count


class _RewriteContext(NamedTuple):
    """Read-only state one rewrite band needs (see `_MeasureContext`)."""

    source: Any
    eaf_plane: DenseEafPlane
    codec: StoreCodec
    analysis_index: np.ndarray
    coefficients: np.ndarray
    row_chunk: int
    n_rows: int
    n_analyses: int
    timer: PhaseTimer | None


_REWRITE: _RewriteContext | None = None


def _encode_band(r0: int) -> tuple[int, np.ndarray, SeExceptionTable]:
    """Read, code and return one row band; the parent writes it in band order."""
    ctx = _REWRITE
    if ctx is None:
        raise RuntimeError("SE rewrite worker ran without a rewrite context")
    r1 = min(r0 + ctx.row_chunk, ctx.n_rows)
    with _optional_phase(ctx.timer, "rewrite.read"):
        values = np.asarray(ctx.source[r0:r1], dtype=np.float32)
        frequencies = ctx.eaf_plane.band(r0, r1)
    exceptions = SeExceptionBuilder()
    with _optional_phase(ctx.timer, "rewrite.encode"):
        raw = ctx.codec.encode_se(
            values,
            eaf=frequencies,
            analysis_index=ctx.analysis_index[: r1 - r0],
            coefficients=ctx.coefficients,
            positions=positions_row_band(r0, ctx.n_analyses),
            exceptions=exceptions,
        )
    return r0, raw, exceptions.table()


@dataclass
class _RewriteSink:
    """The arrays and bookkeeping the rewrite's in-order parent writes into."""

    pending: Any
    exception_index: Any
    exception_value: Any
    row_chunk: int
    n_rows: int
    n_chunks: int
    chunk_timer: PhaseTimer | None


def _run_rewrite_bands(
    sink: _RewriteSink, starts: range, n_workers: int, timer: PhaseTimer
) -> int:
    """Encode every band across the pool and write them back in row order.

    The parent consumes ``ordered_map`` in row order, so the pending plane and
    the exception side table are filled exactly as the serial pass fills them.
    """
    cursor = 0
    started = time.monotonic()
    umbrella = timer.phase("rewrite.parallel") if n_workers > 1 else nullcontext()
    with umbrella:
        encoded = ordered_map(_encode_band, starts, n_workers)
        for index, (r0, raw, table) in enumerate(encoded, start=1):
            r1 = min(r0 + sink.row_chunk, sink.n_rows)
            with _optional_phase(sink.chunk_timer, "rewrite.write"):
                sink.pending[r0:r1] = raw
            with _optional_phase(sink.chunk_timer, "rewrite.exceptions"):
                end = cursor + len(table)
                sink.exception_index[cursor:end] = table.index
                sink.exception_value[cursor:end] = table.value
                cursor = end
            log_progress(
                log, "SE rewrite", index, sink.n_chunks, started,
                every=max(1, sink.n_chunks // 20),
            )
    return cursor


def _finish_rewrite(
    group: Any, coefficients: np.ndarray, cursor: int, exception_count: int, compressor: Any
) -> None:
    """Swap the coded plane in and record the coefficients, checking the table size."""
    if cursor != exception_count:
        raise RuntimeError(
            f"SE codes-only pass counted {exception_count} exceptions but rewrite produced {cursor}"
        )
    del group["se"]
    group.move("se_pending", "se")
    write_se_coefficients(group, coefficients, compressor=compressor)


def _rewrite_dense(
    group: Any,
    encoding: StoreEncoding,
    coefficients: np.ndarray,
    exception_count: int,
    timer: PhaseTimer,
    n_workers: int,
) -> None:
    """Encode the float32 scratch plane under an already-decided residual plan.

    `exception_count` must come from `_count_dense_exceptions` -- the
    rewrite's own codes-only pass -- not from a measurement, which #146 will
    stop making exhaustive (issue #145). The cursor check below is the
    plane-versus-table guarantee: a rewrite that produces a different number
    of exceptions than it allocated fails loudly rather than writing a short
    or padded table.

    The coding is chunk-independent and runs across ``n_workers``; the parent
    writes each band back in row order, so the exception table is filled in the
    same order as the serial pass and the stored plane is unchanged.
    """
    global _REWRITE
    source = group["se"]
    n_rows, n_analyses = map(int, source.shape)
    row_chunk = int(source.chunks[0])
    compressor = source.compressor
    pending = group.create_dataset(
        "se_pending",
        shape=source.shape,
        chunks=source.chunks,
        compressor=compressor,
        dtype="int8",
        fill_value=SE_MISSING,
    )
    exception_index, exception_value = _empty_exception_arrays(group, exception_count, compressor)
    analysis_index = np.broadcast_to(
        np.arange(n_analyses, dtype=np.int64), (row_chunk, n_analyses)
    )
    starts = range(0, n_rows, row_chunk)
    chunk_timer = timer if n_workers <= 1 else None
    _REWRITE = _RewriteContext(
        source, DenseEafPlane.open(group, encoding), StoreCodec(encoding),
        analysis_index, coefficients, row_chunk, n_rows, n_analyses, chunk_timer,
    )
    sink = _RewriteSink(
        pending, exception_index, exception_value, row_chunk, n_rows, len(starts), chunk_timer
    )
    try:
        with log_phase(log, "SE rewrite"):
            cursor = _run_rewrite_bands(sink, starts, n_workers, timer)
    finally:
        _REWRITE = None
    _finish_rewrite(group, coefficients, cursor, exception_count, compressor)


@dataclass
class _FitContext:
    """Read-only state one fit band needs (see `_MeasureContext`)."""

    source: Any
    eaf_plane: DenseEafPlane
    analysis_index: np.ndarray
    row_chunk: int
    n_rows: int
    n_analyses: int


_FIT: _FitContext | None = None


class _FitSums(NamedTuple):
    """One row chunk's contribution to the per-Analysis log-SE fit."""

    eligible: bool
    counts: np.ndarray
    sx: np.ndarray
    sy: np.ndarray
    sxx: np.ndarray
    sxy: np.ndarray


def _fit_one_band(r0: int) -> _FitSums:
    """One row chunk's five per-Analysis sums, for the parent to add back."""
    ctx = _FIT
    if ctx is None:
        raise RuntimeError("SE fit worker ran without a fit context")
    r1 = min(r0 + ctx.row_chunk, ctx.n_rows)
    count = np.zeros(ctx.n_analyses, dtype=np.int64)
    sx, sy, sxx, sxy = (np.zeros(ctx.n_analyses) for _ in range(4))
    eligible = _add_fit_sums(
        ctx.source[r0:r1],
        ctx.eaf_plane.band(r0, r1),
        ctx.analysis_index[: r1 - r0],
        count,
        sx,
        sy,
        sxx,
        sxy,
    )
    return _FitSums(eligible, count, sx, sy, sxx, sxy)


@dataclass
class _FitAccumulators:
    """The five per-Analysis running sums of the log-SE fit."""

    count: np.ndarray
    sx: np.ndarray
    sy: np.ndarray
    sxx: np.ndarray
    sxy: np.ndarray

    @classmethod
    def zeros(cls, n_analyses: int) -> _FitAccumulators:
        return cls(
            np.zeros(n_analyses, dtype=np.int64),
            *(np.zeros(n_analyses) for _ in range(4)),
        )

    def add_band(self, sums: _FitSums) -> None:
        self.count += sums.counts
        self.sx += sums.sx
        self.sy += sums.sy
        self.sxx += sums.sxx
        self.sxy += sums.sxy


def _accumulate_fit_bands(
    starts: range, n_workers: int, acc: _FitAccumulators
) -> bool:
    """Add every row chunk's partial sums in row order; return the AND of eligibility."""
    eligible = True
    started = time.monotonic()
    sums_iter = ordered_map(_fit_one_band, starts, n_workers)
    for index, sums in enumerate(sums_iter, start=1):
        eligible = eligible and sums.eligible
        acc.add_band(sums)
        log_progress(
            log, "SE fit", index, len(starts), started, every=max(1, len(starts) // 20)
        )
    return eligible


def _fit_shared_coefficients(
    source: Any,
    eaf_plane: DenseEafPlane,
    overflow: OverflowCells | None,
    timer: PhaseTimer,
    n_workers: int,
) -> tuple[np.ndarray, bool]:
    """Least squares of `log(se)` on `log(2f(1-f))`, per Analysis, in one pass.

    Accumulated as sums rather than held as a design matrix: the Dense plane is
    read one physical row chunk at a time, and `overflow`'s flat CSR cells join the
    same sums so both Hybrid components are fitted by one model.

    The whole pass is one `fit` phase: it is a full read of the plane, and issue
    #144 wants to know what each full read costs before deciding which to merge.

    Row chunks are independent, so ``n_workers > 1`` fits them in a fork pool.
    Their partial sums are added back in row-chunk order -- exactly the order
    the serial pass adds them -- so the floating-point result, and the
    coefficients derived from it, are bit-for-bit the same either way.
    """
    global _FIT
    n_rows, n_analyses = map(int, source.shape)
    row_chunk = int(source.chunks[0])
    acc = _FitAccumulators.zeros(n_analyses)
    analysis_index = np.broadcast_to(np.arange(n_analyses, dtype=np.int64), (row_chunk, n_analyses))
    starts = range(0, n_rows, row_chunk)
    _FIT = _FitContext(source, eaf_plane, analysis_index, row_chunk, n_rows, n_analyses)
    try:
        with log_phase(log, "SE fit"):
            with timer.phase("fit"):
                eligible = _accumulate_fit_bands(starts, n_workers, acc)
                if overflow is not None:
                    eligible &= _add_fit_sums(
                        overflow.se_values,
                        overflow.eaf_values,
                        overflow.analysis_indices,
                        acc.count,
                        acc.sx,
                        acc.sy,
                        acc.sxx,
                        acc.sxy,
                    )
    finally:
        _FIT = None
    coefficients, solved = solve_log_se(
        acc.count.astype(np.float64), acc.sx, acc.sy, acc.sxx, acc.sxy
    )
    return coefficients, eligible and solved


def _aligned(counts: np.ndarray, n_analyses: int) -> np.ndarray:
    """A per-Analysis count vector padded to the Dense component's width."""
    if len(counts) == n_analyses:
        return counts
    out = np.zeros(n_analyses, dtype=np.int64)
    out[: len(counts)] = counts[:n_analyses]
    return out


def _charged_jointly(
    dense_bytes: int,
    overflow_bytes: int,
    *,
    dense_float: int,
    overflow_float: int | None,
    joint_float: int,
) -> int:
    """A candidate's cost, or `float16`'s when it does not beat it everywhere.

    Reporting a non-saving candidate as costing exactly what `float16` costs is
    what lets the central decision tree reject it without a Hybrid-only branch.
    `overflow_float` is None when that component has no finite cell to charge;
    `joint_float` is always the pair's real `float16` cost, so the number
    returned here and the one reported as the `float16` baseline agree.
    """
    joint = dense_bytes + overflow_bytes
    saves = (
        dense_bytes < dense_float
        and (overflow_float is None or overflow_bytes < overflow_float)
        and joint < joint_float
    )
    return joint if saves else joint_float


def _shared_measurements(
    dense: _ComponentCost,
    overflow: _ComponentCost,
    *,
    eligible: bool,
    dense_coefficient_bytes: int,
    overflow_coefficient_bytes: int,
) -> SeMeasurements:
    """Charge every candidate against `float16` jointly *and* per component.

    A shared kind must save bytes in each non-empty component as well as over
    the pair. A candidate that fails either test is reported as costing exactly
    what `float16` costs, so the central decision tree rejects it without
    needing a Hybrid-only branch.
    """
    joint_float = dense.float_bytes + overflow.float_bytes
    overflow_finite = overflow.total_finite
    # A Hybrid Analysis has cells in both components, so its share is taken
    # over the pair rather than per component.
    finite = dense.n_finite + _aligned(overflow.n_finite, len(dense.n_finite))
    fractions: dict[float, float] = {}
    charged: dict[float, int] = {}
    carrying = finite > 0
    for candidate in SE_RANGE_CANDIDATES:
        exceptions = dense.exception_counts[candidate] + _aligned(
            overflow.exception_counts[candidate], len(dense.n_finite)
        )
        fractions[candidate] = (
            float(np.max(exceptions[carrying] / finite[carrying])) if np.any(carrying) else 0.0
        )
        charged[candidate] = _charged_jointly(
            dense.candidate_bytes[candidate] + dense_coefficient_bytes,
            overflow.candidate_bytes[candidate]
            + (overflow_coefficient_bytes if overflow_finite else 0),
            dense_float=dense.float_bytes,
            overflow_float=overflow.float_bytes if overflow_finite else None,
            joint_float=joint_float,
        )
    return SeMeasurements(
        eligible=eligible,
        exception_fraction=fractions,
        worst_relative_error={
            candidate: float(np.expm1(candidate / 254)) for candidate in SE_RANGE_CANDIDATES
        },
        compressed_bytes=charged,
        float16_compressed_bytes=joint_float,
    )


def _measure_overflow_component(
    overflow: OverflowCells | None,
    coefficients: np.ndarray,
    compressor: Any,
    chunk: int,
    n_analyses: int,
    timer: PhaseTimer,
    n_workers: int,
) -> tuple[_ComponentCost, int]:
    """A Hybrid Overflow Component's cost, or an empty one when there is none."""
    if overflow is None:
        return _ComponentCost.empty(n_analyses), 0
    return (
        _measure_overflow(
            overflow, coefficients, compressor, chunk, n_analyses, timer, n_workers
        ),
        _packed_coefficients(compressor, coefficients),
    )


def _fall_back_to_float16(
    group: Any, encoding: StoreEncoding
) -> tuple[StoreEncoding, np.ndarray | None]:
    """Narrow the scratch plane and report no coefficients.

    Both exits from the decision reach here: a store with no EAF cannot fit a
    model at all, and one whose fit does not earn its bytes declines it. Either
    way the plane must end up in the `float16` its manifest declares, and a
    caller must not be handed coefficients no array was coded against.
    """
    _narrow_dense_se_to_float16(group)
    return encoding, None


def _select_se_encoding(
    encoding: StoreEncoding,
    n_analyses: int,
    dense: _ComponentCost,
    overflow_cost: _ComponentCost,
    *,
    eligible: bool,
    dense_coefficient_bytes: int,
    overflow_coefficient_bytes: int,
) -> StoreEncoding:
    """Charge both components' candidates and take the one central decision."""
    measured = _shared_measurements(
        dense,
        overflow_cost,
        eligible=eligible,
        dense_coefficient_bytes=dense_coefficient_bytes,
        overflow_coefficient_bytes=overflow_coefficient_bytes,
    )
    se_choice = StoreEncoding.decide(EncodingMeasurements(n_analyses, se=measured)).se
    return StoreEncoding(z=encoding.z, se=se_choice, eaf=encoding.eaf)


def _rewrite_selected(
    group: Any,
    source: Any,
    eaf_plane: DenseEafPlane,
    coefficients: np.ndarray,
    selected: StoreEncoding,
    timer: PhaseTimer,
    n_workers: int,
) -> None:
    """Count the decided range's exceptions, then rewrite the plane under it.

    The table is sized by the rewrite's own codes-only count rather than by the
    measurement, so the two can never disagree about it (issue #145).
    """
    residual_range = selected.se.residual_range
    if residual_range is None:
        raise ValueError("residual SE needs a residual_range")
    exception_count = _count_dense_exceptions(
        source, eaf_plane, coefficients, residual_range, timer, n_workers
    )
    _rewrite_dense(group, selected, coefficients, exception_count, timer, n_workers)


def _check_overflow_width(overflow: OverflowCells | None, n_analyses: int) -> None:
    """Fail loudly when a Hybrid overflow was fitted over a different width."""
    if overflow is not None and overflow.n_analyses != n_analyses:
        raise ValueError(
            f"overflow declares {overflow.n_analyses} analyses but the Dense "
            f"component has {n_analyses}"
        )


def optimise_dense_se_joint(
    group: Any,
    encoding: StoreEncoding,
    *,
    overflow: OverflowCells | None = None,
    overflow_compressor: Any = None,
    overflow_chunk: int = 200_000,
    timer: PhaseTimer | None = None,
    n_workers: int = 1,
) -> tuple[StoreEncoding, np.ndarray | None]:
    """Select and rewrite Dense SE, optionally fitting a shared CSR component.

    The Dense plane is read one physical row chunk at a time; `overflow`'s flat
    CSR cells join the same fit and every selection gate, while each component's
    chunks and side table are charged separately.

    A caller that passes a ``PhaseTimer`` gets wall-clock accounting for the
    fit, measurement and rewrite passes (issue #144). ``n_workers > 1`` runs the
    independent row chunks of the fit, measurement, count and rewrite in a fork
    pool; the chosen encoding and every written array are the serial path's
    (issue #221).
    """
    timer = timer or PhaseTimer()
    if encoding.eaf.is_absent:
        return _fall_back_to_float16(group, encoding)
    source = group["se"]
    n_analyses = int(source.shape[1])
    _check_overflow_width(overflow, n_analyses)
    eaf_plane = DenseEafPlane.open(group, encoding)
    coefficients, eligible = _fit_shared_coefficients(
        source, eaf_plane, overflow, timer, n_workers
    )

    compressor = source.compressor
    dense = _measure_dense(source, eaf_plane, coefficients, timer, n_workers)
    overflow_cost, overflow_coefficient_bytes = _measure_overflow_component(
        overflow, coefficients, overflow_compressor or compressor, overflow_chunk, n_analyses,
        timer, n_workers,
    )

    selected = _select_se_encoding(
        encoding,
        n_analyses,
        dense,
        overflow_cost,
        eligible=eligible,
        dense_coefficient_bytes=_packed_coefficients(compressor, coefficients),
        overflow_coefficient_bytes=overflow_coefficient_bytes,
    )
    se_choice = selected.se
    log.info("SE fit: %s", _format_solution(coefficients, eligible, selected))
    log.info("SE phase timings:\n%s", timer.format_report())
    if not se_choice.is_residual:
        return _fall_back_to_float16(group, selected)
    _rewrite_selected(group, source, eaf_plane, coefficients, selected, timer, n_workers)
    log.info("SE phase timings:\n%s", timer.format_report())
    return selected, coefficients


def _format_solution(
    coefficients: np.ndarray, eligible: bool, selected: StoreEncoding
) -> str:
    """One line naming what the fit found and what it selected."""
    if not eligible:
        return "no eligible model (non-finite EAF); falling back to float16"
    worst = float(np.max(np.abs(coefficients))) if len(coefficients) else 0.0
    return (
        f"{len(coefficients)} Analyses, max |coefficient| {worst:.4g}, "
        f"selected se={selected.se.kind} range={selected.se.residual_range}"
    )


def _narrow_dense_se_to_float16(group: Any) -> None:
    """Bring a `float32` scratch plane down to the declared `float16`.

    Builders keep the scratch in `float32` so an exact exception is the
    source's own value rather than one already rounded (spec §6a). When the
    measured decision is `float16` after all, the plane still has to end up in
    the encoding the manifest declares, so it is narrowed here rather than
    left wider than its own declaration.
    """
    source = group["se"]
    if source.dtype == np.dtype("float16"):
        return
    n_rows = int(source.shape[0])
    row_chunk = int(source.chunks[0])
    pending = group.create_dataset(
        "se_pending",
        shape=source.shape,
        chunks=source.chunks,
        compressor=source.compressor,
        dtype="float16",
        fill_value=np.nan,
    )
    for r0 in range(0, n_rows, row_chunk):
        r1 = min(r0 + row_chunk, n_rows)
        pending[r0:r1] = np.asarray(source[r0:r1], dtype=np.float16)
    del group["se"]
    group.move("se_pending", "se")


def optimise_dense_se(group: Any, encoding: StoreEncoding, n_workers: int = 1) -> StoreEncoding:
    """Measure, select, and rewrite a Dense SE plane by physical chunks."""
    return optimise_dense_se_joint(group, encoding, n_workers=n_workers)[0]


def rewrite_dense_se(
    group: Any,
    encoding: StoreEncoding,
    coefficients: np.ndarray | None = None,
    timer: PhaseTimer | None = None,
    n_workers: int = 1,
) -> None:
    """Encode a float32 Dense scratch plane under an already-decided plan."""
    timer = timer or PhaseTimer()
    source = group["se"]
    n_analyses = int(source.shape[1])
    if not encoding.se.is_residual:
        _narrow_dense_se_to_float16(group)
        return
    if coefficients is None:
        raise ValueError("residual SE needs se_coefficients")
    stored_coefficients = np.asarray(coefficients, dtype=np.float32)
    if stored_coefficients.shape != (n_analyses, 2) or not np.all(np.isfinite(stored_coefficients)):
        raise ValueError(f"se_coefficients must have finite shape ({n_analyses}, 2)")
    eaf_plane = DenseEafPlane.open(group, encoding)
    assert encoding.se.residual_range is not None
    exception_count = _count_dense_exceptions(
        source, eaf_plane, stored_coefficients, encoding.se.residual_range, timer, n_workers
    )
    _rewrite_dense(group, encoding, stored_coefficients, exception_count, timer, n_workers)
