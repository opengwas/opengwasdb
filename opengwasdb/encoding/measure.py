"""What a build measures about its own frequencies before choosing an encoding.

`StoreEncoding.decide()` is deliberately not allowed to look at a store's
layout, its analysis count or which arrays happen to exist -- only at an
`EafMeasurements` summary produced here (ADR 0037 §2, issue #119). Nothing
stops a Dense manifest spanning several cohorts, and a store that assumed
otherwise would clip against a range chosen for data it does not hold.

The two entry points differ only in how cells are grouped into variants: a
Dense grid groups by row, a CSR by `variant_index`. Both compute the same
baselines the writer will compute, so the exception fraction the tree reads is
the exception fraction the build will actually produce.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from opengwasdb.encoding.codec import (
    EXACT_TABLE_CHUNK,
    eaf_baseline_from_grid,
    eaf_baseline_from_pairs,
    logit,
    se_residual_codes,
)
from opengwasdb.encoding.plan import (
    EAF_CODE_HALF,
    EAF_CODE_MAX,
    EAF_CODE_MIN,
    EAF_RANGE_CANDIDATES,
    SE_RANGE_CANDIDATES,
    EafMeasurements,
    SeMeasurements,
)

#: The largest number of an axis's chunks a measurement will visit. The bound
#: is deliberately a constant, not a share: it is what keeps the survey cost
#: from growing with the store (issue #146). A plane with fewer chunks than
#: this is measured exhaustively; a larger one is measured on an even spread
#: of whole chunks and the byte totals are scaled up by what was seen.
MAX_MEASURED_CHUNKS = 64


@dataclass(frozen=True)
class ChunkSample:
    """A deterministic even-spread of whole chunks over one axis.

    Selection is a pure function of ``(axis_length, step)``: the same plane is
    always sampled in the same places, so two builds of the same store make the
    same decision (issue #146 AC2). `starts` are ascending axis offsets of the
    chunks to visit, in physical order; whole chunks are sampled, never
    individual cells, because a chunk is the unit zarr compresses and the
    measurement is estimating compressed size.
    """

    starts: tuple[int, ...]
    total_chunks: int

    @property
    def sampled_chunks(self) -> int:
        return len(self.starts)

    @property
    def is_full(self) -> bool:
        """Whether every chunk was sampled -- the plane was measured in full."""
        return self.sampled_chunks == self.total_chunks

    def to_manifest(self) -> dict[str, int]:
        return {
            "sampled_chunks": self.sampled_chunks,
            "total_chunks": self.total_chunks,
        }


def sample_chunks(
    axis_length: int, step: int, *, max_chunks: int = MAX_MEASURED_CHUNKS
) -> ChunkSample:
    """Pick at most `max_chunks` whole chunks, spread evenly over the axis.

    Chunk `i` of `total` is visited when ``i * total // max_chunks`` lands on
    it, so the sample always includes the axis's first chunk and spreads the
    rest at an even stride. With `total <= max_chunks` every chunk is sampled
    and the decision is exactly the exhaustive one; a larger plane is sampled
    and the caller scales byte totals by the cells it saw.
    """
    total = max(1, int(np.ceil(axis_length / max(step, 1))))
    keep = min(total, max(max_chunks, 1))
    starts = tuple(i * step for i in (int(i * total // keep) for i in range(keep)))
    return ChunkSample(starts=starts, total_chunks=total)


@dataclass
class SeMeasurementRecord:
    """What the SE measurement saw, for the manifest's provenance (issue #146).

    A decision this format stores must say how it was reached: a reader that
    assumes an exhaustive survey when the range was chosen from a sample is
    assuming work that never happened. The record lives in `manifest.json`
    provenance -- it is about the build, not about decoding -- so it does not
    extend the `encoding` block or change what a reader must implement.
    """

    dense: ChunkSample | None = None
    csr: ChunkSample | None = None

    @classmethod
    def of(cls, dense: ChunkSample | None, csr: ChunkSample | None = None) -> SeMeasurementRecord:
        return cls(dense=dense, csr=csr)

    def to_manifest(self) -> dict[str, dict[str, int] | str]:
        def describe(sample: ChunkSample | None) -> dict[str, int] | str:
            if sample is None:
                return "none"
            return sample.to_manifest()

        return {"dense": describe(self.dense), "csr": describe(self.csr)}


def _packed_bytes(compressor: object | None, data: np.ndarray) -> int:
    """Compressed size of one array, or its raw size when nothing compresses it."""
    encode = getattr(compressor, "encode", None)
    return len(encode(np.ascontiguousarray(data))) if encode is not None else data.nbytes


def _chunk_starts(length: int, step: int) -> range:
    return range(0, length, max(step, 1))


def _packed_1d(compressor: object | None, data: np.ndarray, chunk: int) -> int:
    return sum(
        _packed_bytes(compressor, data[start : start + chunk])
        for start in _chunk_starts(len(data), chunk)
    )


def _packed_2d(compressor: object | None, data: np.ndarray, chunk: tuple[int, int]) -> int:
    return sum(
        _packed_bytes(compressor, data[r0 : r0 + chunk[0], c0 : c0 + chunk[1]])
        for r0 in _chunk_starts(data.shape[0], chunk[0])
        for c0 in _chunk_starts(data.shape[1], chunk[1])
    )


def _packed_chunks(
    compressor: object | None, data: np.ndarray, chunk_shape: tuple[int, ...] | int | None
) -> int:
    """Compressed size the way zarr will actually store it: chunk by chunk.

    Compressing an array whole flatters it against the same array cut into
    chunks, so the size the decision compares must be measured in the shape it
    will be written in. An array whose chunking is not stated, or does not
    match its own rank, is charged whole.
    """
    shaped = np.asarray(data)
    shape = (chunk_shape,) if isinstance(chunk_shape, int) else chunk_shape
    if shape is None or shaped.ndim != len(shape):
        return _packed_bytes(compressor, shaped)
    if shaped.ndim == 1:
        return _packed_1d(compressor, shaped, shape[0])
    if shaped.ndim == 2:
        return _packed_2d(compressor, shaped, (shape[0], shape[1]))
    return _packed_bytes(compressor, shaped)


def solve_log_se(
    count: np.ndarray, sx: np.ndarray, sy: np.ndarray, sxx: np.ndarray, sxy: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Per-Analysis OLS of `log(se)` on `log(2f(1-f))`, from accumulated sums.

    The single site that turns sums into coefficients. The Dense path
    accumulates them one row chunk at a time and the Ragged path in one pass,
    but a store must not be able to get one answer from one and a different
    answer from the other.

    Returns the coefficients and whether every Analysis could be fitted: fewer
    than two usable cells, or a degenerate spread of frequencies, is not a bad
    fit but no fit, and sends the whole plane back to `float16`.
    """
    n_analyses = len(count)
    coefficients = np.full((n_analyses, 2), np.nan, dtype=np.float32)
    denominator = count * sxx - sx * sx
    if not (bool(np.all(count >= 2)) and bool(np.all(np.abs(denominator) > 0))):
        return coefficients, False
    slope = (count * sxy - sx * sy) / denominator
    coefficients[:, 1] = slope.astype(np.float32)
    coefficients[:, 0] = ((sy - slope * sx) / count).astype(np.float32)
    return coefficients, bool(np.all(np.isfinite(coefficients)))


def _solve_from_all_cells(
    s: np.ndarray, f: np.ndarray, ai: np.ndarray, n_analyses: int
) -> tuple[np.ndarray, bool]:
    """One vectorised bincount pass for per-Analysis coefficients.

    The coefficients are stored data, so the fit is deliberately exhaustive
    even though the measurement below samples (issue #146/#147): sampling the
    fit would change what a store contains, not just how its range was chosen.
    """
    use = (s > 0) & (f > 0) & (f < 1)
    selected = ai[use]
    x = np.log(2 * f[use] * (1 - f[use]))
    y = np.log(s[use])
    return solve_log_se(
        np.bincount(selected, minlength=n_analyses).astype(np.float64),
        np.bincount(selected, weights=x, minlength=n_analyses),
        np.bincount(selected, weights=y, minlength=n_analyses),
        np.bincount(selected, weights=x * x, minlength=n_analyses),
        np.bincount(selected, weights=x * y, minlength=n_analyses),
    )


def _charge_stored_unit(
    compressor: object | None, data: np.ndarray, *, columns: int, col_chunk: int
) -> int:
    """Compressed bytes of one measured unit in the shape it is stored in.

    A CSR unit is one flat chunk. A grid unit is a row band cut into its
    column chunks, mirroring the 2-D partition `_packed_chunks` applies to a
    whole grid -- so a full sample charges exactly what the whole plane does.
    """
    if columns == 1:
        return _packed_bytes(compressor, np.ascontiguousarray(data))
    shaped = np.ascontiguousarray(data).reshape(-1, columns)
    return sum(
        _packed_bytes(compressor, shaped[:, c0 : c0 + col_chunk])
        for c0 in range(0, columns, col_chunk)
    )


def fit_se(
    se: np.ndarray,
    eaf: np.ndarray,
    analysis_index: np.ndarray,
    *,
    n_analyses: int,
    compressor: object | None = None,
    chunks: tuple[int, ...] | int | None = None,
    measure_max_chunks: int = MAX_MEASURED_CHUNKS,
    record: SeMeasurementRecord | None = None,
) -> tuple[np.ndarray, SeMeasurements]:
    """Fit per-Analysis log-SE models and measure the candidate costs.

    EAF must already have made its encode/decode round trip. The compressed
    comparison includes codes, coefficients, and exact side-table rows.

    The measurement runs on a bounded, deterministic sample of the axis's
    whole chunks and scales byte totals up by the cells it saw (issue #147);
    the coefficients and the eligibility sweep stay exhaustive. A plane with
    fewer chunks than the cap is measured in full, byte-identical to the
    pre-sampling survey.
    """
    original = np.asarray(se)
    if original.ndim not in (1, 2):
        raise ValueError(f"se must be 1-D (CSR) or 2-D (grid), got {original.ndim}-D")
    se_shape = original.shape
    s = original.astype(np.float64).ravel()
    f = np.asarray(eaf, dtype=np.float64).ravel()
    ai = np.asarray(analysis_index, dtype=np.int64).ravel()
    if not (s.shape == f.shape == ai.shape):
        raise ValueError("se, eaf, and analysis_index must have the same shape")
    finite = np.isfinite(s)
    eligible = bool(np.all(np.isfinite(f[finite])))
    coef = np.full((n_analyses, 2), np.nan, dtype=np.float32)
    if eligible:
        coef, solved = _solve_from_all_cells(s, f, ai, n_analyses)
        eligible = solved

    # The measurement unit is one whole chunk of the axis the plane is stored
    # on: for a CSR that is a flat run of associations, for a grid a run of
    # whole rows (all analyses, cut into column chunks for the byte charge).
    if original.ndim == 2:
        columns = int(se_shape[1])
        row_chunk = (
            int(chunks[0]) if isinstance(chunks, tuple) and len(chunks) == 2 else int(se_shape[0])
        )
        col_chunk = int(chunks[1]) if isinstance(chunks, tuple) and len(chunks) == 2 else columns
        unit_len = row_chunk * columns
        sample = sample_chunks(int(se_shape[0]), row_chunk, max_chunks=measure_max_chunks)
    else:
        columns = 1
        col_chunk = 1
        chunk_len = (
            int(chunks)
            if isinstance(chunks, int)
            else int(chunks[0])
            if isinstance(chunks, tuple)
            else int(len(s))
        )
        unit_len = chunk_len
        sample = sample_chunks(int(len(s)), chunk_len, max_chunks=measure_max_chunks)
    if record is not None:
        record.csr = sample
    if sample.is_full:
        return coef, _measure_full(
            s, f, ai, coef, compressor, chunks, se_shape, n_analyses, eligible
        )

    float16_bytes = 0
    code_bytes = dict.fromkeys(SE_RANGE_CANDIDATES, 0)
    side_bytes = dict.fromkeys(SE_RANGE_CANDIDATES, 0)
    finite_per_analysis = np.zeros(n_analyses, dtype=np.int64)
    exception_counts = {
        candidate: np.zeros(n_analyses, dtype=np.int64) for candidate in SE_RANGE_CANDIDATES
    }
    sampled_cells = 0
    coefficient_bytes = _packed_chunks(compressor, coef, (min(max(len(coef), 1), 1024), 2))
    # Each sampled chunk keeps its Analysis indexes: a chunk is a full-width row
    # band of a grid or a contiguous CSR run, and the per-Analysis exception
    # share is taken over the sample so the scale cancels (issue #146 AC6).
    for axis_start in sample.starts:
        start = axis_start * columns
        end = min(start + unit_len, len(s))
        unit_s = s[start:end]
        unit_f = f[start:end]
        unit_ai = ai[start:end]
        sampled_cells += unit_s.size
        present = np.isfinite(unit_s)
        finite_per_analysis += np.bincount(unit_ai[present], minlength=n_analyses)
        float16_bytes += _charge_stored_unit(
            compressor, unit_s.astype(np.float16), columns=columns, col_chunk=col_chunk
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            prediction = coef[unit_ai, 0] + coef[unit_ai, 1] * np.log(2 * unit_f * (1 - unit_f))
            residual = np.log(unit_s) - prediction
        for candidate in SE_RANGE_CANDIDATES:
            step = candidate / 127.0
            stored, exceptions = se_residual_codes(unit_s, residual, step)
            code_bytes[candidate] += _charge_stored_unit(
                compressor, stored, columns=columns, col_chunk=col_chunk
            )
            exception_counts[candidate] += np.bincount(unit_ai[exceptions], minlength=n_analyses)
            if np.any(exceptions):
                flat = np.arange(start, end, dtype=np.int64)[exceptions]
                side_bytes[candidate] += _packed_bytes(compressor, flat) + _packed_bytes(
                    compressor, unit_s[exceptions].astype(np.float32)
                )
    scale = len(s) / max(sampled_cells, 1)
    carrying = finite_per_analysis > 0
    fractions: dict[float, float] = {}
    errors: dict[float, float] = {}
    sizes: dict[float, int] = {}
    for candidate in SE_RANGE_CANDIDATES:
        per_analysis = exception_counts[candidate]
        fractions[candidate] = (
            float(np.max(per_analysis[carrying] / finite_per_analysis[carrying]))
            if np.any(carrying)
            else 0.0
        )
        errors[candidate] = float(np.expm1(candidate / 254))
        sizes[candidate] = (
            int(round((code_bytes[candidate] + side_bytes[candidate]) * scale)) + coefficient_bytes
        )
    return coef, SeMeasurements(
        eligible=eligible and bool(np.all(np.isfinite(coef))),
        exception_fraction=fractions,
        worst_relative_error=errors,
        compressed_bytes=sizes,
        float16_compressed_bytes=int(round(float16_bytes * scale)),
    )


def _measure_full(
    s: np.ndarray,
    f: np.ndarray,
    ai: np.ndarray,
    coef: np.ndarray,
    compressor: object | None,
    chunks: tuple[int, ...] | int | None,
    se_shape: tuple[int, ...],
    n_analyses: int,
    eligible: bool,
) -> SeMeasurements:
    """The exhaustive survey, kept byte-identical for planes under the cap.

    A plane with fewer chunks than `MAX_MEASURED_CHUNKS` is measured in full;
    keeping the pre-sampling arithmetic here (rather than running the sampled
    loop over every chunk) is what lets the small pilots stay byte-for-byte
    unchanged, which the parity tests and #146 AC3 depend on.
    """
    residual = np.full(s.shape, np.nan, dtype=np.float64)
    use = np.isfinite(s) & (s > 0) & (f > 0) & (f < 1)
    selected = ai[use]
    x = np.log(2 * f[use] * (1 - f[use]))
    y = np.log(s[use])
    residual[use] = y - (
        coef[selected, 0].astype(np.float64) + coef[selected, 1].astype(np.float64) * x
    )
    finite_per_analysis = np.bincount(ai[np.isfinite(s)], minlength=n_analyses)
    carrying = finite_per_analysis > 0
    coefficient_bytes = _packed_chunks(compressor, coef, (min(max(len(coef), 1), 1024), 2))
    fractions: dict[float, float] = {}
    errors: dict[float, float] = {}
    sizes: dict[float, int] = {}
    for candidate in SE_RANGE_CANDIDATES:
        step = candidate / 127.0
        stored, exceptions = se_residual_codes(s, residual, step)
        per_analysis = np.bincount(ai[exceptions], minlength=n_analyses)
        fractions[candidate] = (
            float(np.max(per_analysis[carrying] / finite_per_analysis[carrying]))
            if np.any(carrying)
            else 0.0
        )
        errors[candidate] = float(np.expm1(step / 2.0))
        sizes[candidate] = (
            _packed_chunks(compressor, stored.reshape(se_shape), chunks)
            + coefficient_bytes
            + _packed_chunks(
                compressor, np.flatnonzero(exceptions).astype(np.int64), EXACT_TABLE_CHUNK
            )
            + _packed_chunks(compressor, s[exceptions].astype(np.float32), EXACT_TABLE_CHUNK)
        )
    return SeMeasurements(
        eligible=eligible and bool(np.all(np.isfinite(coef))),
        exception_fraction=fractions,
        worst_relative_error=errors,
        compressed_bytes=sizes,
        float16_compressed_bytes=_packed_chunks(
            compressor, s.astype(np.float16).reshape(se_shape), chunks
        ),
    )


def fit_se_grid(
    se: np.ndarray,
    eaf: np.ndarray,
    *,
    compressor: object | None = None,
    chunks: tuple[int, int] | None = None,
    measure_max_chunks: int = MAX_MEASURED_CHUNKS,
    record: SeMeasurementRecord | None = None,
) -> tuple[np.ndarray, SeMeasurements]:
    block = np.asarray(se)
    if block.ndim != 2:
        raise ValueError("dense se must be a 2-D grid")
    ai = np.broadcast_to(np.arange(block.shape[1]), block.shape)
    return fit_se(
        block,
        eaf,
        ai,
        n_analyses=block.shape[1],
        compressor=compressor,
        chunks=chunks,
        measure_max_chunks=measure_max_chunks,
        record=record,
    )


def _exception_fractions(residual: np.ndarray, n_eaf_cells: int) -> dict[float, float]:
    """Share of EAF-bearing cells that each candidate range cannot code.

    `residual` holds one entry per EAF-bearing cell, NaN where the cell has no
    residual at all -- a frequency of exactly 0 or 1, or a variant with no
    usable baseline. Those cells are exceptions at every range, and counting
    them here is what keeps the tree's byte estimate honest about the side
    table it is about to ask for.
    """
    if n_eaf_cells == 0:
        return dict.fromkeys(EAF_RANGE_CANDIDATES, 0.0)
    unusable = ~np.isfinite(residual)
    fractions: dict[float, float] = {}
    for candidate in EAF_RANGE_CANDIDATES:
        step = candidate / EAF_CODE_HALF
        with np.errstate(invalid="ignore"):
            codes = np.rint(residual / step)
            codable = ~unusable & (codes >= EAF_CODE_MIN) & (codes <= EAF_CODE_MAX)
        fractions[candidate] = float(np.count_nonzero(~codable) / n_eaf_cells)
    return fractions


def measure_eaf(
    variant_index: np.ndarray, values: np.ndarray, *, n_variants: int
) -> EafMeasurements:
    """Measure a CSR component's frequencies (flat cells, one variant each)."""
    return measure_eaf_sample(
        variant_index,
        values,
        n_variants=n_variants,
        n_cells=int(np.asarray(values).size),
        n_eaf_cells=int(np.count_nonzero(np.isfinite(np.asarray(values, dtype=np.float64)))),
    )


def measure_eaf_sample(
    variant_index: np.ndarray,
    values: np.ndarray,
    *,
    n_variants: int,
    n_cells: int,
    n_eaf_cells: int,
) -> EafMeasurements:
    """Residual spread from a sample of cells, against the build's real totals.

    A Dense build cannot read its own grid variant-by-variant before it has
    written it -- the spills it holds are per Analysis -- so the spread is
    measured on the deterministic per-variant sample the EAF orientation check
    already draws (§9.1), while the cell and variant counts that decide the
    *bytes* are the build's exact ones. The range is a policy choice that a
    sample settles; the baselines the writer then computes are exact, and a
    cell the sample did not anticipate becomes an exception, which is stored
    exactly rather than clipped.
    """
    values = np.asarray(values, dtype=np.float64)
    if n_eaf_cells == 0 or values.size == 0:
        return EafMeasurements(
            n_cells=int(n_cells), n_eaf_cells=int(n_eaf_cells), n_variants=int(n_variants)
        )
    variant_index = np.asarray(variant_index, dtype=np.int64)
    finite = np.isfinite(values)
    axis_length = int(variant_index.max()) + 1 if variant_index.size else 0
    baseline = eaf_baseline_from_pairs(variant_index, values, axis_length)
    residual = _residual(values[finite], baseline[variant_index[finite]])
    return EafMeasurements(
        n_cells=int(n_cells),
        n_eaf_cells=int(n_eaf_cells),
        n_variants=int(n_variants),
        exception_fraction=_exception_fractions(residual, int(np.count_nonzero(finite))),
    )


def measure_eaf_grid(values: np.ndarray) -> EafMeasurements:
    """Measure a Dense block's frequencies (one row per variant).

    For a build that streams its grid, call this per band and combine the
    results with `combine_eaf_measurements`: the baseline is per row, so a band is a complete
    unit of measurement and no cell is counted twice.
    """
    block = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(block)
    n_eaf_cells = int(np.count_nonzero(finite))
    if n_eaf_cells == 0:
        return EafMeasurements(n_cells=int(block.size), n_eaf_cells=0, n_variants=block.shape[0])
    baseline = eaf_baseline_from_grid(block)
    rows = np.nonzero(finite)[0]
    residual = _residual(block[finite], baseline[rows])
    return EafMeasurements(
        n_cells=int(block.size),
        n_eaf_cells=n_eaf_cells,
        n_variants=int(block.shape[0]),
        exception_fraction=_exception_fractions(residual, n_eaf_cells),
    )


def combine_eaf_measurements(parts: list[EafMeasurements]) -> EafMeasurements:
    """Sum measurements over the components (or bands) one plan will cover.

    A Hybrid release's Dense Component and its Ragged Overflow get one
    `decide()` between them, because they partition one Analysis's
    associations and a result contract that differed by component would be
    inconsistent (issue #119). Each still writes its own baseline array, which
    is why `n_variants` adds rather than maxes.
    """
    usable = [part for part in parts if part is not None]
    if not usable:
        return EafMeasurements()
    n_eaf_cells = sum(part.n_eaf_cells for part in usable)
    fractions: dict[float, float] = {}
    if n_eaf_cells:
        for candidate in EAF_RANGE_CANDIDATES:
            weighted = sum(
                part.exception_fraction.get(candidate, 0.0) * part.n_eaf_cells for part in usable
            )
            fractions[candidate] = float(weighted / n_eaf_cells)
    return EafMeasurements(
        n_cells=sum(part.n_cells for part in usable),
        n_eaf_cells=n_eaf_cells,
        n_variants=sum(part.n_variants for part in usable),
        exception_fraction=fractions,
    )


def _residual(values: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        residual = logit(values) - logit(np.asarray(baseline, dtype=np.float32).astype(np.float64))
    return np.asarray(residual, dtype=np.float64)
