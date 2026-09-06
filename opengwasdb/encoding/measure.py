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

import numpy as np

from opengwasdb.encoding.codec import (
    EXACT_TABLE_CHUNK,
    eaf_baseline_from_grid,
    eaf_baseline_from_pairs,
    logit,
)
from opengwasdb.encoding.plan import (
    EAF_CODE_HALF,
    EAF_CODE_MAX,
    EAF_CODE_MIN,
    EAF_RANGE_CANDIDATES,
    SE_CODE_MAX,
    SE_CODE_MIN,
    SE_RANGE_CANDIDATES,
    EafMeasurements,
    SeMeasurements,
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
        for col in range(n_analyses):
            use = finite & (s > 0) & (ai == col) & (f > 0) & (f < 1)
            if np.count_nonzero(use) < 2:
                eligible = False
                break
            x = np.log(2 * f[use] * (1 - f[use]))
            design = np.column_stack((np.ones(len(x)), x))
            fitted = np.linalg.lstsq(design, np.log(s[use]), rcond=None)[0]
            coef[col] = fitted.astype(np.float32)
            residual[use] = np.log(s[use]) - design @ fitted
    fractions: dict[float, float] = {}
    errors: dict[float, float] = {}
    sizes: dict[float, int] = {}

    def packed_bytes(data: np.ndarray) -> int:
        encode = getattr(compressor, "encode", None)
        return len(encode(np.ascontiguousarray(data))) if encode is not None else data.nbytes

    def packed_chunks(data: np.ndarray, chunk_shape: tuple[int, ...] | int | None) -> int:
        shaped = np.asarray(data)
        if chunk_shape is None:
            return packed_bytes(shaped)
        if isinstance(chunk_shape, int):
            chunk_shape = (chunk_shape,)
        if shaped.ndim != len(chunk_shape):
            return packed_bytes(shaped)
        total = 0
        if shaped.ndim == 1:
            for start in range(0, len(shaped), chunk_shape[0]):
                total += packed_bytes(shaped[start : start + chunk_shape[0]])
            return total
        if shaped.ndim == 2:
            for r0 in range(0, shaped.shape[0], chunk_shape[0]):
                for c0 in range(0, shaped.shape[1], chunk_shape[1]):
                    total += packed_bytes(
                        shaped[
                            r0 : r0 + chunk_shape[0],
                            c0 : c0 + chunk_shape[1],
                        ]
                    )
            return total
        return packed_bytes(shaped)

    denominator = max(int(np.count_nonzero(finite)), 1)
    for candidate in SE_RANGE_CANDIDATES:
        step = candidate / 127.0
        codes = np.rint(residual / step)
        ordinary = (
            finite
            & (s > 0)
            & np.isfinite(residual)
            & (codes >= SE_CODE_MIN)
            & (codes <= SE_CODE_MAX)
        )
        exceptions = finite & ~ordinary
        stored = np.full(s.shape, -128, dtype=np.int8)
        stored[exceptions] = -127
        stored[ordinary] = codes[ordinary].astype(np.int8)
        fractions[candidate] = float(np.count_nonzero(exceptions) / denominator)
        errors[candidate] = float(np.expm1(step / 2.0))
        positions = np.flatnonzero(exceptions).astype(np.int64)
        exact = s[exceptions].astype(np.float32)
        sizes[candidate] = (
            packed_chunks(stored.reshape(se_shape), chunks)
            + packed_chunks(coef, (min(max(n_analyses, 1), 1024), 2))
            + packed_chunks(positions, EXACT_TABLE_CHUNK)
            + packed_chunks(exact, EXACT_TABLE_CHUNK)
        )
    return coef, SeMeasurements(
        eligible=eligible and bool(np.all(np.isfinite(coef))),
        exception_fraction=fractions,
        worst_relative_error=errors,
        compressed_bytes=sizes,
        float16_compressed_bytes=packed_chunks(s.astype(np.float16).reshape(se_shape), chunks),
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
