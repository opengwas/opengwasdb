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
    eaf_baseline_from_grid,
    eaf_baseline_from_pairs,
    logit,
)
from opengwasdb.encoding.plan import (
    EAF_CODE_HALF,
    EAF_CODE_MAX,
    EAF_CODE_MIN,
    EAF_RANGE_CANDIDATES,
    EafMeasurements,
)


def exception_fractions(residual: np.ndarray, n_eaf_cells: int) -> dict[float, float]:
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
        exception_fraction=exception_fractions(residual, int(np.count_nonzero(finite))),
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
        exception_fraction=exception_fractions(residual, n_eaf_cells),
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
                part.exception_fraction.get(candidate, 0.0) * part.n_eaf_cells
                for part in usable
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
        residual = logit(values) - logit(
            np.asarray(baseline, dtype=np.float32).astype(np.float64)
        )
    return np.asarray(residual, dtype=np.float64)
