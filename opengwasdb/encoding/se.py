"""Bounded-memory Dense SE fitting and rewriting after EAF is available."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from opengwasdb.encoding.codec import (
    EXACT_TABLE_CHUNK,
    SE_EXCEPTION_INDEX,
    SE_EXCEPTION_VALUE,
    SeExceptionBuilder,
    StoreCodec,
    positions_flat,
    positions_row_band,
    se_residual_codes,
)
from opengwasdb.encoding.measure import solve_log_se
from opengwasdb.encoding.plan import (
    SE_MISSING,
    SE_RANGE_CANDIDATES,
    EncodingMeasurements,
    SeMeasurements,
    StoreEncoding,
)
from opengwasdb.encoding.planes import DenseEafPlane, write_se_coefficients
from opengwasdb.encoding.timing import PhaseTimer

OverflowCells = tuple[np.ndarray, np.ndarray, np.ndarray]


class ComponentCost(NamedTuple):
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
    def empty(cls, n_analyses: int = 0) -> ComponentCost:
        zeros = np.zeros(n_analyses, dtype=np.int64)
        return cls(
            0,
            zeros,
            {candidate: zeros.copy() for candidate in SE_RANGE_CANDIDATES},
            dict.fromkeys(SE_RANGE_CANDIDATES, 0),
        )

    def total_exceptions(self, candidate: float) -> int:
        return int(self.exception_counts[candidate].sum())

    @property
    def total_finite(self) -> int:
        return int(self.n_finite.sum())


def _packed(compressor: Any, value: np.ndarray) -> int:
    data = np.ascontiguousarray(value)
    return len(compressor.encode(data)) if compressor is not None else data.nbytes


def _packed_coefficients(compressor: Any, coefficients: np.ndarray) -> int:
    return sum(
        _packed(compressor, coefficients[start : start + 1024])
        for start in range(0, len(coefficients), 1024)
    )


class _SideTableCost:
    """Compressed bytes for one sorted side table, bounded to one chunk."""

    def __init__(self, compressor: Any, n_analyses: int) -> None:
        self._compressor = compressor
        self._index = np.empty(EXACT_TABLE_CHUNK, dtype=np.int64)
        self._value = np.empty(EXACT_TABLE_CHUNK, dtype=np.float32)
        self._used = 0
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
        self._used = 0

    def finish(self) -> tuple[np.ndarray, int]:
        if self._used:
            self._flush()
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


def _charge_in_column_chunks(compressor: Any, band: np.ndarray, col_chunk: int) -> int:
    """Compressed bytes for one row band, charged as the physical chunks it becomes."""
    return sum(
        _packed(compressor, band[:, c0 : c0 + col_chunk])
        for c0 in range(0, band.shape[1], col_chunk)
    )


def _measure_dense(
    source: Any,
    eaf_plane: DenseEafPlane,
    coefficients: np.ndarray,
    timer: PhaseTimer,
) -> ComponentCost:
    n_rows, n_analyses = map(int, source.shape)
    row_chunk, col_chunk = map(int, source.chunks)
    compressor = source.compressor
    sides = {candidate: _SideTableCost(compressor, n_analyses) for candidate in SE_RANGE_CANDIDATES}
    code_bytes = dict.fromkeys(SE_RANGE_CANDIDATES, 0)
    float_bytes = 0
    finite_per_analysis = np.zeros(n_analyses, dtype=np.int64)
    columns = np.arange(n_analyses, dtype=np.int64)
    analysis_index = np.broadcast_to(columns, (row_chunk, n_analyses))
    for r0 in range(0, n_rows, row_chunk):
        r1 = min(r0 + row_chunk, n_rows)
        with timer.phase("measure.read"):
            values = np.asarray(source[r0:r1], dtype=np.float32)
            frequencies = eaf_plane.band(r0, r1)
            finite_per_analysis += np.isfinite(values).sum(axis=0).astype(np.int64)
        ai = analysis_index[: r1 - r0]
        with timer.phase("measure.float16"):
            float_bytes += _charge_in_column_chunks(
                compressor, values.astype(np.float16), col_chunk
            )
        for candidate in SE_RANGE_CANDIDATES:
            with timer.phase("measure.code"):
                raw, exceptional = _candidate_codes(
                    values, frequencies, ai, coefficients, candidate
                )
            with timer.phase("measure.compress"):
                code_bytes[candidate] += _charge_in_column_chunks(compressor, raw, col_chunk)
            with timer.phase("measure.exceptions"):
                sides[candidate].add(
                    positions_row_band(r0, n_analyses)(exceptional),
                    values[exceptional],
                    ai[exceptional],
                )
    return ComponentCost(float_bytes, finite_per_analysis, *_charged(sides, code_bytes))


def _measure_overflow(
    overflow: OverflowCells,
    coefficients: np.ndarray,
    compressor: Any,
    chunk: int,
    n_analyses: int,
    timer: PhaseTimer | None = None,
) -> ComponentCost:
    values, frequencies, analyses = (np.asarray(part).ravel() for part in overflow)
    sides = {candidate: _SideTableCost(compressor, n_analyses) for candidate in SE_RANGE_CANDIDATES}
    code_bytes = dict.fromkeys(SE_RANGE_CANDIDATES, 0)
    float_bytes = 0
    finite_per_analysis = np.zeros(n_analyses, dtype=np.int64)

    def run() -> None:
        nonlocal float_bytes, finite_per_analysis
        for start in range(0, len(values), chunk):
            end = min(start + chunk, len(values))
            se = np.asarray(values[start:end], dtype=np.float32)
            eaf = np.asarray(frequencies[start:end], dtype=np.float32)
            ai = np.asarray(analyses[start:end], dtype=np.int64)
            finite_per_analysis += np.bincount(ai[np.isfinite(se)], minlength=n_analyses).astype(
                np.int64
            )
            float_bytes += _packed(compressor, se.astype(np.float16))
            for candidate in SE_RANGE_CANDIDATES:
                raw, exceptional = _candidate_codes(se, eaf, ai, coefficients, candidate)
                code_bytes[candidate] += _packed(compressor, raw)
                sides[candidate].add(
                    positions_flat(start)(exceptional),
                    se[exceptional],
                    ai[exceptional],
                )

    if timer is None:
        run()
    else:
        with timer.phase("measure.overflow"):
            run()
    return ComponentCost(float_bytes, finite_per_analysis, *_charged(sides, code_bytes))


def _empty_exception_arrays(group: Any, count: int, compressor: Any) -> tuple[Any, Any]:
    """Allocate the side table the streaming rewrite fills in position order.

    Sized from the measurement pass rather than grown: the rewrite visits row
    chunks in order, so the exceptions arrive already sorted and can be written
    straight into their final slots.
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


def _rewrite_dense(
    group: Any,
    encoding: StoreEncoding,
    coefficients: np.ndarray,
    exception_count: int,
    timer: PhaseTimer,
) -> None:
    source = group["se"]
    n_rows, n_analyses = map(int, source.shape)
    row_chunk = int(source.chunks[0])
    compressor = source.compressor
    eaf_plane = DenseEafPlane.open(group, encoding)
    pending = group.create_dataset(
        "se_pending",
        shape=source.shape,
        chunks=source.chunks,
        compressor=compressor,
        dtype="int8",
        fill_value=SE_MISSING,
    )
    exception_index, exception_value = _empty_exception_arrays(group, exception_count, compressor)
    codec = StoreCodec(encoding)
    cursor = 0
    columns = np.arange(n_analyses, dtype=np.int64)
    analysis_index = np.broadcast_to(columns, (row_chunk, n_analyses))
    for r0 in range(0, n_rows, row_chunk):
        r1 = min(r0 + row_chunk, n_rows)
        with timer.phase("rewrite.read"):
            values = np.asarray(source[r0:r1], dtype=np.float32)
            frequencies = eaf_plane.band(r0, r1)
        exceptions = SeExceptionBuilder()
        with timer.phase("rewrite.encode"):
            raw = codec.encode_se(
                values,
                eaf=frequencies,
                analysis_index=analysis_index[: r1 - r0],
                coefficients=coefficients,
                positions=positions_row_band(r0, n_analyses),
                exceptions=exceptions,
            )
        with timer.phase("rewrite.write"):
            pending[r0:r1] = raw
        with timer.phase("rewrite.exceptions"):
            table = exceptions.table()
            end = cursor + len(table)
            exception_index[cursor:end] = table.index
            exception_value[cursor:end] = table.value
            cursor = end
    if cursor != exception_count:
        raise RuntimeError(
            f"SE measurement counted {exception_count} exceptions but rewrite produced {cursor}"
        )
    del group["se"]
    group.move("se_pending", "se")
    write_se_coefficients(group, coefficients, compressor=compressor)


def _fit_shared_coefficients(
    source: Any,
    eaf_plane: DenseEafPlane,
    overflow: OverflowCells | None,
    timer: PhaseTimer,
) -> tuple[np.ndarray, bool]:
    """Least squares of `log(se)` on `log(2f(1-f))`, per Analysis, in one pass.

    Accumulated as sums rather than held as a design matrix: the Dense plane is
    read one physical row chunk at a time, and `overflow`'s flat CSR cells join the
    same sums so both Hybrid components are fitted by one model.

    The whole pass is one `fit` phase: it is a full read of the plane, and issue
    #144 wants to know what each full read costs before deciding which to merge.
    """
    n_rows, n_analyses = map(int, source.shape)
    row_chunk = int(source.chunks[0])
    count = np.zeros(n_analyses, dtype=np.int64)
    sx, sy, sxx, sxy = (np.zeros(n_analyses) for _ in range(4))
    eligible = True
    analysis_index = np.broadcast_to(np.arange(n_analyses, dtype=np.int64), (row_chunk, n_analyses))
    with timer.phase("fit"):
        for r0 in range(0, n_rows, row_chunk):
            r1 = min(r0 + row_chunk, n_rows)
            eligible &= _add_fit_sums(
                source[r0:r1],
                eaf_plane.band(r0, r1),
                analysis_index[: r1 - r0],
                count,
                sx,
                sy,
                sxx,
                sxy,
            )
        if overflow is not None:
            eligible &= _add_fit_sums(*overflow, count, sx, sy, sxx, sxy)
    coefficients, solved = solve_log_se(count.astype(np.float64), sx, sy, sxx, sxy)
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
    dense: ComponentCost,
    overflow: ComponentCost,
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


def optimise_dense_se_joint(
    group: Any,
    encoding: StoreEncoding,
    *,
    overflow: OverflowCells | None = None,
    overflow_compressor: Any = None,
    overflow_chunk: int = 200_000,
    timer: PhaseTimer | None = None,
) -> tuple[StoreEncoding, np.ndarray | None]:
    """Select and rewrite Dense SE, optionally fitting a shared CSR component.

    The Dense plane is read one physical row chunk at a time. When ``overflow``
    is supplied (Hybrid), its flat CSR cells contribute to the same fit and
    all selection gates, while each component's actual chunks and side table
    are charged separately.

    A caller that passes a ``PhaseTimer`` gets wall-clock accounting for the
    fit, measurement and rewrite passes (issue #144); one that does not still
    pays for them, the timer is just not retained.
    """
    timer = timer or PhaseTimer()
    if encoding.eaf.is_absent:
        narrow_dense_se_to_float16(group)
        return encoding, None
    source = group["se"]
    n_analyses = int(source.shape[1])
    eaf_plane = DenseEafPlane.open(group, encoding)
    coefficients, eligible = _fit_shared_coefficients(source, eaf_plane, overflow, timer)

    compressor = source.compressor
    dense = _measure_dense(source, eaf_plane, coefficients, timer)
    overflow_cost = ComponentCost.empty(n_analyses)
    overflow_coefficient_bytes = 0
    if overflow is not None:
        overflow_compressor = overflow_compressor or compressor
        overflow_cost = _measure_overflow(
            overflow,
            coefficients,
            overflow_compressor,
            overflow_chunk,
            n_analyses,
            timer,
        )
        overflow_coefficient_bytes = _packed_coefficients(overflow_compressor, coefficients)

    measured = _shared_measurements(
        dense,
        overflow_cost,
        eligible=eligible,
        dense_coefficient_bytes=_packed_coefficients(compressor, coefficients),
        overflow_coefficient_bytes=overflow_coefficient_bytes,
    )
    se_choice = StoreEncoding.decide(EncodingMeasurements(n_analyses, se=measured)).se
    selected = StoreEncoding(z=encoding.z, se=se_choice, eaf=encoding.eaf)
    if not se_choice.is_residual:
        narrow_dense_se_to_float16(group)
        return selected, None
    assert se_choice.residual_range is not None
    _rewrite_dense(
        group,
        selected,
        coefficients,
        dense.total_exceptions(se_choice.residual_range),
        timer,
    )
    return selected, coefficients


def narrow_dense_se_to_float16(group: Any) -> None:
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


def optimise_dense_se(group: Any, encoding: StoreEncoding) -> StoreEncoding:
    """Measure, select, and rewrite a Dense SE plane by physical chunks."""
    return optimise_dense_se_joint(group, encoding)[0]


def rewrite_dense_se(
    group: Any,
    encoding: StoreEncoding,
    coefficients: np.ndarray | None = None,
    timer: PhaseTimer | None = None,
) -> None:
    """Encode a float32 Dense scratch plane under an already-decided plan."""
    timer = timer or PhaseTimer()
    source = group["se"]
    n_rows, n_analyses = map(int, source.shape)
    row_chunk = int(source.chunks[0])
    if not encoding.se.is_residual:
        narrow_dense_se_to_float16(group)
        return
    if coefficients is None:
        raise ValueError("residual SE needs se_coefficients")
    stored_coefficients = np.asarray(coefficients, dtype=np.float32)
    if stored_coefficients.shape != (n_analyses, 2) or not np.all(np.isfinite(stored_coefficients)):
        raise ValueError(f"se_coefficients must have finite shape ({n_analyses}, 2)")
    eaf_plane = DenseEafPlane.open(group, encoding)
    analysis_index = np.broadcast_to(np.arange(n_analyses, dtype=np.int64), (row_chunk, n_analyses))
    exception_count = 0
    assert encoding.se.residual_range is not None
    with timer.phase("rewrite.count"):
        for r0 in range(0, n_rows, row_chunk):
            r1 = min(r0 + row_chunk, n_rows)
            _, exceptional = _candidate_codes(
                source[r0:r1],
                eaf_plane.band(r0, r1),
                analysis_index[: r1 - r0],
                stored_coefficients,
                encoding.se.residual_range,
            )
            exception_count += int(exceptional.sum())
    _rewrite_dense(group, encoding, stored_coefficients, exception_count, timer)
