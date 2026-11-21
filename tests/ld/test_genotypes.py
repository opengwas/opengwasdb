"""The shared genotype arithmetic against the frozen builder and independent Test Store (#90)."""

from __future__ import annotations

import build_fixtures as made
import genotypes_base as base
import numpy as np
import pytest
import testdata

from ogld.format import REncoding


@pytest.mark.parametrize("k", [0, 10])
@pytest.mark.parametrize("scale", [511, 32767])
def test_codes_equal_the_base_builder_for_every_near_band_pair(k: int, scale: int) -> None:
    from ogld.genotypes import covariates, prepare, r

    v = testdata.variants()
    rows = np.flatnonzero(v.passes_filters)
    calls = made.hard_calls()[rows].astype(np.float64)
    is_alt = made.a1_signs()[rows] > 0
    names = list(testdata.pcs().names[:k])
    values = testdata.pcs().values[:, :k]
    old = base.unit_vectors(calls, base.covariates(names, values, calls.shape[1]), is_alt)
    expected = old @ old.T
    block = prepare(calls, covariates(names, values, calls.shape[1]), is_alt)
    actual = r(block, block)
    near = (v.chrom[rows, None] == v.chrom[rows][None, :]) & (
        np.abs(v.pos[rows, None] - v.pos[rows][None, :]) <= 3_000_000
    )
    near &= np.triu(np.ones(near.shape, dtype=bool), 1)
    assert near.sum() > 100_000, "compare all Near Band pairs, not an empty slice"
    enc = REncoding(scale)
    np.testing.assert_array_equal(enc.quantise(actual[near]), enc.quantise(expected[near]))
