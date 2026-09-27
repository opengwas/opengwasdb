"""Streaming the Ragged Overflow Component's cells (issue #228).

The joint SE fit and the CSR flush both held a whole Overflow plane in memory,
which is ~1.3 TB and ~1.1 TB respectively on OGS-00011's 15,078,327,210 cells.
Streaming them has to change the footprint and nothing else: the oracle in
every equivalence test here is the materialising path itself, so the two cannot
drift apart silently.
"""

from __future__ import annotations

import numpy as np
import pytest

from opengwasdb.encoding import EncodingMeasurements, StoreEncoding
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRWriter

# Small enough that the cells amortise the per-variant baseline: a residual
# plane also stores one float32 per variant, so a plane with far more
# variants than cells is correctly encoded as float32 instead.
_N_VARIANTS = 2_000


def _writer(sizes, *, seed=0, without_eaf=()):
    """A writer holding one Analysis per entry of `sizes`, sorted by variant.

    Each variant has one true frequency that every Analysis reports with a
    little noise, because that is what makes a per-variant baseline worth
    storing: independent frequencies per Analysis have nothing for a residual
    to be small against, and the encoding tree correctly picks `float32`.

    `without_eaf` names Analyses whose source reported no frequency, which the
    writer stores as all-NaN (ADR 0036) -- the mix a real release carries.
    """
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.05, 0.95, _N_VARIANTS)
    writer = RaggedCSRWriter(_N_VARIANTS)
    for i, count in enumerate(sizes):
        vi = np.sort(rng.choice(_N_VARIANTS, size=count, replace=False)).astype(np.int32)
        z = rng.standard_normal(count).astype(np.float32)
        se = np.abs(rng.standard_normal(count) * 0.1 + 0.2).astype(np.float32)
        noisy = np.clip(truth[vi] + rng.normal(0, 0.002, count), 1e-4, 1 - 1e-4)
        eaf = None if i in without_eaf else noisy.astype(np.float32)
        writer.add_analysis(vi, z, se, eaf)
    return writer


def _residual_encoding(writer, n_analyses):
    encoding = StoreEncoding.decide(
        EncodingMeasurements(n_analyses=n_analyses, eaf=writer.eaf_measurements())
    )
    assert encoding.eaf.is_residual, "fixture must select a residual EAF plane to be meaningful"
    return encoding


# ── Seam: the streamed cell source ───────────────────────────────────────────


@pytest.mark.parametrize("cell_budget", [1, 10, 500, 10_000])
def test_streamed_batches_reconstruct_the_materialised_cells(cell_budget):
    """Concatenating the batches must give back exactly what `se_fit_inputs`
    builds in one piece -- same values, same order, same dtypes."""
    sizes = [1500, 1, 0, 1800, 900, 1900]
    writer = _writer(sizes, without_eaf=(2, 4))
    encoding = _residual_encoding(writer, len(sizes))

    whole = writer.se_fit_inputs(encoding)
    batches = list(writer.se_fit_batches(encoding, cell_budget=cell_budget))

    for field in ("se_values", "eaf_values", "analysis_indices"):
        streamed = (
            np.concatenate([getattr(b, field) for b in batches])
            if batches
            else np.empty(0, dtype=getattr(whole, field).dtype)
        )
        expected = getattr(whole, field)
        assert streamed.dtype == expected.dtype, field
        np.testing.assert_array_equal(streamed, expected, err_msg=field)
