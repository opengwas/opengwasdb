"""Bounded-memory Dense SE fitting and rewriting after EAF is available."""

from __future__ import annotations

from typing import Any

import numpy as np

from opengwasdb.encoding.codec import (
    EXACT_TABLE_CHUNK,
    SE_EXCEPTION_INDEX,
    SE_EXCEPTION_VALUE,
    SeExceptionBuilder,
    StoreCodec,
    positions_flat,
    positions_row_band,
)
from opengwasdb.encoding.plan import (
    SE_CODE_MAX,
    SE_CODE_MIN,
    SE_RANGE_CANDIDATES,
    EncodingMeasurements,
    SeMeasurements,
    StoreEncoding,
)
from opengwasdb.encoding.planes import DenseEafPlane, write_se_coefficients

SeFitExtra = tuple[np.ndarray, np.ndarray, np.ndarray]


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

    def __init__(self, compressor: Any) -> None:
        self._compressor = compressor
        self._index = np.empty(EXACT_TABLE_CHUNK, dtype=np.int64)
        self._value = np.empty(EXACT_TABLE_CHUNK, dtype=np.float32)
        self._used = 0
        self.count = 0
        self.compressed_bytes = 0

    def add(self, index: np.ndarray, value: np.ndarray) -> None:
        index = np.asarray(index, dtype=np.int64)
        value = np.asarray(value, dtype=np.float32)
        self.count += len(index)
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

    def finish(self) -> tuple[int, int]:
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
    finite = np.isfinite(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log(2 * frequencies * (1 - frequencies))
        prediction = coefficients[analyses, 0] + coefficients[analyses, 1] * x
        code = np.rint((np.log(values) - prediction) / (candidate / 127))
    ordinary = finite & (values > 0) & np.isfinite(code)
    ordinary &= (code >= SE_CODE_MIN) & (code <= SE_CODE_MAX)
    exceptional = finite & ~ordinary
    raw = np.full(values.shape, -128, dtype=np.int8)
    raw[exceptional] = -127
    raw[ordinary] = code[ordinary].astype(np.int8)
    return raw, exceptional


def _charged(
    sides: dict[float, _SideTableCost], code_bytes: dict[float, int]
) -> tuple[dict[float, int], dict[float, int]]:
    """Exception counts and total compressed bytes, per candidate range."""
    counts: dict[float, int] = {}
    total: dict[float, int] = {}
    for candidate, side in sides.items():
        counts[candidate], side_compressed = side.finish()
        total[candidate] = code_bytes[candidate] + side_compressed
    return counts, total


def _measure_dense(
    source: Any,
    eaf_plane: DenseEafPlane,
    coefficients: np.ndarray,
) -> tuple[int, int, dict[float, int], dict[float, int]]:
    n_rows, n_analyses = map(int, source.shape)
    row_chunk, col_chunk = map(int, source.chunks)
    compressor = source.compressor
    sides = {candidate: _SideTableCost(compressor) for candidate in SE_RANGE_CANDIDATES}
    code_bytes = dict.fromkeys(SE_RANGE_CANDIDATES, 0)
    float_bytes = 0
    total_finite = 0
    columns = np.arange(n_analyses, dtype=np.int64)
    analysis_index = np.broadcast_to(columns, (row_chunk, n_analyses))
    for r0 in range(0, n_rows, row_chunk):
        r1 = min(r0 + row_chunk, n_rows)
        values = np.asarray(source[r0:r1], dtype=np.float32)
        frequencies = eaf_plane.band(r0, r1)
        ai = analysis_index[: r1 - r0]
        total_finite += int(np.isfinite(values).sum())
        for c0 in range(0, n_analyses, col_chunk):
            c1 = min(c0 + col_chunk, n_analyses)
            float_bytes += _packed(compressor, values[:, c0:c1].astype(np.float16))
        for candidate in SE_RANGE_CANDIDATES:
            raw, exceptional = _candidate_codes(values, frequencies, ai, coefficients, candidate)
            for c0 in range(0, n_analyses, col_chunk):
                c1 = min(c0 + col_chunk, n_analyses)
                code_bytes[candidate] += _packed(compressor, raw[:, c0:c1])
            sides[candidate].add(
                positions_row_band(r0, n_analyses)(exceptional),
                values[exceptional],
            )
    return float_bytes, total_finite, *_charged(sides, code_bytes)


def _measure_extra(
    extra: SeFitExtra,
    coefficients: np.ndarray,
    compressor: Any,
    chunk: int,
) -> tuple[int, int, dict[float, int], dict[float, int]]:
    values, frequencies, analyses = (np.asarray(part).ravel() for part in extra)
    sides = {candidate: _SideTableCost(compressor) for candidate in SE_RANGE_CANDIDATES}
    code_bytes = dict.fromkeys(SE_RANGE_CANDIDATES, 0)
    float_bytes = 0
    total_finite = 0
    for start in range(0, len(values), chunk):
        end = min(start + chunk, len(values))
        se = np.asarray(values[start:end], dtype=np.float32)
        eaf = np.asarray(frequencies[start:end], dtype=np.float32)
        ai = np.asarray(analyses[start:end], dtype=np.int64)
        total_finite += int(np.isfinite(se).sum())
        float_bytes += _packed(compressor, se.astype(np.float16))
        for candidate in SE_RANGE_CANDIDATES:
            raw, exceptional = _candidate_codes(se, eaf, ai, coefficients, candidate)
            code_bytes[candidate] += _packed(compressor, raw)
            sides[candidate].add(
                positions_flat(start)(exceptional),
                se[exceptional],
            )
    return float_bytes, total_finite, *_charged(sides, code_bytes)


def _rewrite_dense(
    group: Any,
    encoding: StoreEncoding,
    coefficients: np.ndarray,
    exception_count: int,
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
        fill_value=-128,
    )
    for name in (SE_EXCEPTION_INDEX, SE_EXCEPTION_VALUE):
        if name in group:
            del group[name]
    exception_index = group.create_dataset(
        SE_EXCEPTION_INDEX,
        shape=(exception_count,),
        chunks=(max(1, min(exception_count, EXACT_TABLE_CHUNK)),),
        compressor=compressor,
        dtype="int64",
    )
    exception_value = group.create_dataset(
        SE_EXCEPTION_VALUE,
        shape=(exception_count,),
        chunks=(max(1, min(exception_count, EXACT_TABLE_CHUNK)),),
        compressor=compressor,
        dtype="float32",
    )
    codec = StoreCodec(encoding)
    cursor = 0
    columns = np.arange(n_analyses, dtype=np.int64)
    analysis_index = np.broadcast_to(columns, (row_chunk, n_analyses))
    for r0 in range(0, n_rows, row_chunk):
        r1 = min(r0 + row_chunk, n_rows)
        values = np.asarray(source[r0:r1], dtype=np.float32)
        exceptions = SeExceptionBuilder()
        pending[r0:r1] = codec.encode_se(
            values,
            eaf=eaf_plane.band(r0, r1),
            analysis_index=analysis_index[: r1 - r0],
            coefficients=coefficients,
            positions=positions_row_band(r0, n_analyses),
            exceptions=exceptions,
        )
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


def optimise_dense_se_joint(
    group: Any,
    encoding: StoreEncoding,
    *,
    extra: SeFitExtra | None = None,
    extra_compressor: Any = None,
    extra_chunk: int = 200_000,
) -> tuple[StoreEncoding, np.ndarray | None]:
    """Select and rewrite Dense SE, optionally fitting a shared CSR component.

    The Dense plane is read one physical row chunk at a time. When ``extra``
    is supplied (Hybrid), its flat CSR cells contribute to the same fit and
    all selection gates, while each component's actual chunks and side table
    are charged separately.
    """
    if encoding.eaf.is_absent:
        return encoding, None
    source = group["se"]
    n_rows, n_analyses = map(int, source.shape)
    row_chunk = int(source.chunks[0])
    eaf_plane = DenseEafPlane.open(group, encoding)
    count = np.zeros(n_analyses, dtype=np.int64)
    sx = np.zeros(n_analyses)
    sy = np.zeros(n_analyses)
    sxx = np.zeros(n_analyses)
    sxy = np.zeros(n_analyses)
    eligible = True
    columns = np.arange(n_analyses, dtype=np.int64)
    analysis_index = np.broadcast_to(columns, (row_chunk, n_analyses))
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
    if extra is not None:
        eligible &= _add_fit_sums(*extra, count, sx, sy, sxx, sxy)
    denominator = count * sxx - sx * sx
    eligible &= bool(np.all(count >= 2) and np.all(np.abs(denominator) > 0))
    coefficients = np.full((n_analyses, 2), np.nan, dtype=np.float32)
    if eligible:
        coefficients[:, 1] = ((count * sxy - sx * sy) / denominator).astype(np.float32)
        coefficients[:, 0] = ((sy - coefficients[:, 1] * sx) / count).astype(np.float32)
        eligible &= bool(np.all(np.isfinite(coefficients)))

    dense_float_bytes, dense_finite, dense_counts, dense_bytes = _measure_dense(
        source, eaf_plane, coefficients
    )
    compressor = source.compressor
    coefficient_bytes = _packed_coefficients(compressor, coefficients)
    for candidate in SE_RANGE_CANDIDATES:
        dense_bytes[candidate] += coefficient_bytes
    float_bytes = dense_float_bytes
    candidate_bytes = dense_bytes.copy()
    extra_counts = dict.fromkeys(SE_RANGE_CANDIDATES, 0)
    extra_finite = 0
    extra_float = 0
    extra_bytes = dict.fromkeys(SE_RANGE_CANDIDATES, 0)
    if extra is not None:
        extra_compressor = extra_compressor or compressor
        extra_float, extra_finite, extra_counts, extra_bytes = _measure_extra(
            extra, coefficients, extra_compressor, extra_chunk
        )
        float_bytes += extra_float
        extra_coefficient_bytes = _packed_coefficients(extra_compressor, coefficients)
        for candidate in SE_RANGE_CANDIDATES:
            candidate_bytes[candidate] += extra_bytes[candidate] + extra_coefficient_bytes

    exception_fraction: dict[float, float] = {}
    for candidate in SE_RANGE_CANDIDATES:
        dense_fraction = dense_counts[candidate] / max(dense_finite, 1)
        extra_fraction = extra_counts[candidate] / max(extra_finite, 1)
        exception_fraction[candidate] = max(dense_fraction, extra_fraction)
        # A shared kind must save bytes in each non-empty component as well as
        # jointly.  Mark a failing candidate as non-saving so the central
        # decision tree rejects it without adding a Hybrid-only tree.
        extra_candidate_bytes = extra_bytes[candidate] + (
            _packed_coefficients(extra_compressor, coefficients) if extra is not None else 0
        )
        if (
            dense_bytes[candidate] >= dense_float_bytes
            or (extra_finite and extra_candidate_bytes >= extra_float)
            or candidate_bytes[candidate] >= float_bytes
        ):
            candidate_bytes[candidate] = float_bytes

    measured = SeMeasurements(
        eligible=eligible,
        exception_fraction=exception_fraction,
        worst_relative_error={
            candidate: float(np.expm1(candidate / 254)) for candidate in SE_RANGE_CANDIDATES
        },
        compressed_bytes=candidate_bytes,
        float16_compressed_bytes=float_bytes,
    )
    se_choice = StoreEncoding.decide(EncodingMeasurements(n_analyses, se=measured)).se
    selected = StoreEncoding(z=encoding.z, se=se_choice, eaf=encoding.eaf)
    if not se_choice.is_residual:
        return selected, None
    assert se_choice.residual_range is not None
    _rewrite_dense(group, selected, coefficients, dense_counts[se_choice.residual_range])
    return selected, coefficients


def optimise_dense_se(group: Any, encoding: StoreEncoding) -> StoreEncoding:
    """Measure, select, and rewrite a Dense SE plane by physical chunks."""
    return optimise_dense_se_joint(group, encoding)[0]


def rewrite_dense_se(
    group: Any,
    encoding: StoreEncoding,
    coefficients: np.ndarray | None = None,
) -> None:
    """Encode a float32 Dense scratch plane under an already-decided plan."""
    source = group["se"]
    n_rows, n_analyses = map(int, source.shape)
    row_chunk = int(source.chunks[0])
    compressor = source.compressor
    if not encoding.se.is_residual:
        pending = group.create_dataset(
            "se_pending",
            shape=source.shape,
            chunks=source.chunks,
            compressor=compressor,
            dtype="float16",
            fill_value=np.nan,
        )
        for r0 in range(0, n_rows, row_chunk):
            r1 = min(r0 + row_chunk, n_rows)
            pending[r0:r1] = np.asarray(source[r0:r1], dtype=np.float16)
        del group["se"]
        group.move("se_pending", "se")
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
    _rewrite_dense(group, encoding, stored_coefficients, exception_count)
