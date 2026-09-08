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

from typing import Any

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


def _packed_bytes(compressor: object | None, data: np.ndarray) -> int:
    """Compressed size of one array, or its raw size when nothing compresses it."""
    encode = getattr(compressor, "encode", None)
    return len(encode(np.ascontiguousarray(data))) if encode is not None else data.nbytes


def _chunk_starts(length: int, step: int) -> range:
    return range(0, length, max(step, 1))


def _full_chunk(piece: np.ndarray, chunk_shape: tuple[int, ...], fill_value: Any) -> np.ndarray:
    """`piece` as the chunk zarr will actually store: exactly `chunk_shape`.

    zarr never stores a partial chunk: a chunk whose span runs past the array's
    extent is padded out to the declared chunk shape with the array's fill
    value *before* it is compressed. Measuring the slice at the size it happens
    to be undercharges the final chunk of every plane whose extent is not a
    multiple of its chunk -- and does it most on exactly the small or
    awkwardly-shaped planes where the compressed-bytes margin is narrowest
    (issue #158). An interior chunk is returned unchanged.
    """
    if tuple(piece.shape) == tuple(chunk_shape):
        return piece
    pads = tuple((0, want - have) for have, want in zip(piece.shape, chunk_shape, strict=True))
    return np.pad(piece, pads, mode="constant", constant_values=fill_value)


def packed_chunk_bytes(
    compressor: object | None, piece: np.ndarray, chunk_shape: tuple[int, ...], fill_value: Any
) -> int:
    """Compressed bytes of the one zarr chunk `piece` is stored as.

    zarr never stores a partial chunk: a chunk whose span runs past an array's
    extent is padded out to the declared chunk shape with the array's fill
    value *before* it is compressed, so the bytes `piece` will occupy on disk
    are the bytes of `piece` padded to `chunk_shape`. This is the single
    chunk-accounting implementation: the whole-grid fits below call it through
    `_packed_1d`/`_packed_2d`, and the streamed Dense/Hybrid/migration
    optimiser in `encoding/se.py` charges its row bands through the same
    function, so no measurement path can leave an edge chunk short of zarr's
    padded size (issue #158). An interior chunk -- one whose shape already
    matches `chunk_shape` -- is compressed exactly as it is.
    """
    return _packed_bytes(compressor, _full_chunk(np.asarray(piece), tuple(chunk_shape), fill_value))


def _packed_1d(
    compressor: object | None, data: np.ndarray, chunk: int, fill_value: Any
) -> int:
    return sum(
        packed_chunk_bytes(compressor, data[start : start + chunk], (chunk,), fill_value)
        for start in _chunk_starts(len(data), chunk)
    )


def _packed_2d(
    compressor: object | None, data: np.ndarray, chunk: tuple[int, int], fill_value: Any
) -> int:
    return sum(
        packed_chunk_bytes(
            compressor, data[r0 : r0 + chunk[0], c0 : c0 + chunk[1]], chunk, fill_value
        )
        for r0 in _chunk_starts(data.shape[0], chunk[0])
        for c0 in _chunk_starts(data.shape[1], chunk[1])
    )


def _packed_chunks(
    compressor: object | None,
    data: np.ndarray,
    chunk_shape: tuple[int, ...] | int | None,
    fill_value: Any,
) -> int:
    """Compressed size the way zarr will actually store it: chunk by chunk.

    Compressing an array whole flatters it against the same array cut into
    chunks, so the size the decision compares must be measured in the shape it
    will be written in. zarr pads an edge chunk out to the declared chunk shape
    with the array's fill value before compressing it, so a measured slice is
    padded the same way first, chunk by chunk through `packed_chunk_bytes`
    (issue #158). An array whose chunking is not stated, or does not match its
    own rank, is charged whole.
    """
    shaped = np.asarray(data)
    shape = (chunk_shape,) if isinstance(chunk_shape, int) else chunk_shape
    if shape is None or shaped.ndim != len(shape):
        return _packed_bytes(compressor, shaped)
    if shaped.ndim == 1:
        return _packed_1d(compressor, shaped, shape[0], fill_value)
    if shaped.ndim == 2:
        return _packed_2d(compressor, shaped, (shape[0], shape[1]), fill_value)
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


def _fit_log_se(
    s: np.ndarray, f: np.ndarray, ai: np.ndarray, n_analyses: int, finite: np.ndarray
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Fit every Analysis in one pass, and keep each fitted cell's residual.

    One pass of `np.bincount` rather than a mask per Analysis: the Ragged
    builders hold the whole store in memory already, and a loop over Analyses
    would make the fit quadratic in a dimension that grows.
    """
    residual = np.full(s.shape, np.nan, dtype=np.float64)
    use = finite & (s > 0) & (f > 0) & (f < 1)
    selected = ai[use]
    x = np.log(2 * f[use] * (1 - f[use]))
    y = np.log(s[use])
    coefficients, eligible = solve_log_se(
        np.bincount(selected, minlength=n_analyses).astype(np.float64),
        np.bincount(selected, weights=x, minlength=n_analyses),
        np.bincount(selected, weights=y, minlength=n_analyses),
        np.bincount(selected, weights=x * x, minlength=n_analyses),
        np.bincount(selected, weights=x * y, minlength=n_analyses),
    )
    if not eligible:
        return coefficients, residual, False
    residual[use] = y - (
        coefficients[selected, 0].astype(np.float64)
        + coefficients[selected, 1].astype(np.float64) * x
    )
    return coefficients, residual, True


#: zarr pads an edge chunk's out-of-extent cells with the array's declared
#: fill value before compressing it (issue #158). The arrays `fit_se` measures
#: are created by the layouts' writers from whole arrays (`data=` with no
#: explicit fill), so the value zarr declares -- and pads with -- is its
#: numeric default, 0. The fill is passed per measured array rather than
#: assumed, so a writer that declares a semantic fill (its plane's missing
#: marker) is charged for the padding it will actually write.
_ZARR_DEFAULT_FILL: Any = 0


def _candidate_bytes(
    compressor: object | None,
    stored: np.ndarray,
    chunks: tuple[int, ...] | int | None,
    coefficients: np.ndarray,
    exceptions: np.ndarray,
    values: np.ndarray,
) -> int:
    """Everything one candidate range actually costs on disk.

    Codes, coefficients and both side arrays: comparing only the codes against
    `float16` would accept a range whose exception table more than gives the
    saving back. Each array is charged in the chunks -- and padded with the
    fill value -- its eventual writer declares, so an edge chunk that zarr will
    store full-size is not measured short (issue #158).
    """
    n_exceptions = int(np.count_nonzero(exceptions))
    # The writers size a side table's chunks to its own row count when that is
    # smaller than EXACT_TABLE_CHUNK, so a table that fits one chunk is charged
    # at its own length and nothing beyond it is invented.
    exception_chunk = (max(1, min(n_exceptions, EXACT_TABLE_CHUNK)),)
    return (
        _packed_chunks(compressor, stored, chunks, _ZARR_DEFAULT_FILL)
        + _packed_chunks(
            compressor,
            coefficients,
            (min(max(len(coefficients), 1), 1024), 2),
            _ZARR_DEFAULT_FILL,
        )
        + _packed_chunks(
            compressor,
            np.flatnonzero(exceptions).astype(np.int64),
            exception_chunk,
            _ZARR_DEFAULT_FILL,
        )
        + _packed_chunks(
            compressor, values[exceptions].astype(np.float32), exception_chunk, _ZARR_DEFAULT_FILL
        )
    )


def fit_se(
    se: np.ndarray,
    eaf: np.ndarray,
    analysis_index: np.ndarray,
    *,
    n_analyses: int,
    compressor: object | None = None,
    chunks: tuple[int, ...] | int | None = None,
) -> tuple[np.ndarray, SeMeasurements]:
    """Fit per-Analysis log-SE models and measure the complete candidate costs.

    EAF must already have made its encode/decode round trip. The compressed
    comparison includes codes, coefficients, and exact side-table rows.
    """
    se_shape = np.asarray(se).shape
    s = np.asarray(se, dtype=np.float64).ravel()
    f = np.asarray(eaf, dtype=np.float64).ravel()
    ai = np.asarray(analysis_index, dtype=np.int64).ravel()
    if not (s.shape == f.shape == ai.shape):
        raise ValueError("se, eaf, and analysis_index must have the same shape")
    finite = np.isfinite(s)
    eligible = bool(np.all(np.isfinite(f[finite])))
    coef = np.full((n_analyses, 2), np.nan, dtype=np.float32)
    residual = np.full(s.shape, np.nan, dtype=np.float64)
    if eligible:
        coef, residual, eligible = _fit_log_se(s, f, ai, n_analyses, finite)

    # Per Analysis, not pooled: issue #118 reverts the plane when *any*
    # Analysis fits badly, and a pooled share lets one bad Analysis hide
    # behind its well-fitting neighbours.
    finite_per_analysis = np.bincount(ai[finite], minlength=n_analyses)
    carrying = finite_per_analysis > 0
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
        sizes[candidate] = _candidate_bytes(
            compressor, stored.reshape(se_shape), chunks, coef, exceptions, s
        )
    return coef, SeMeasurements(
        eligible=eligible and bool(np.all(np.isfinite(coef))),
        exception_fraction=fractions,
        worst_relative_error=errors,
        compressed_bytes=sizes,
        float16_compressed_bytes=_packed_chunks(
            compressor, s.astype(np.float16).reshape(se_shape), chunks, _ZARR_DEFAULT_FILL
        ),
    )


def fit_se_grid(
    se: np.ndarray,
    eaf: np.ndarray,
    *,
    compressor: object | None = None,
    chunks: tuple[int, int] | None = None,
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
