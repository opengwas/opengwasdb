"""Threshold-query parity and the hit-only clump seam (#91)."""

import dataclasses
from pathlib import Path

import build_fixtures
import numpy as np
import pytest
import testdata
from contract.support import TEST_STORE
from test_genotype_store import genotype_store as genotype_store

import ogld
from ogld.genotypes.reader import GenotypeReader


def _equal(actual: object, expected: object) -> None:
    assert type(actual) is type(expected)
    for field in dataclasses.fields(actual):
        a, b = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b, err_msg=field.name)
        else:
            assert a == b, field.name


def test_proxies_equal_the_test_store(genotype_store: Path) -> None:
    names = build_fixtures.alids(np.flatnonzero(testdata.variants().passes_filters)[::11])
    assert len(names) > 10
    _equal(
        ogld.open(genotype_store).proxies([*names, "absent"], r2=0.01, kb=3000),
        ogld.open(TEST_STORE).proxies([*names, "absent"], r2=0.01, kb=3000),
    )


def test_clump_computes_only_hit_rows(
    genotype_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = np.flatnonzero(testdata.variants().passes_filters)[::7]
    names = build_fixtures.alids(rows)
    assert len(names) > 20
    store = ogld.open(genotype_store)
    hits = set(store._reader.rows_of(names).tolist())
    computed = []
    original = GenotypeReader._data

    def counted(self, rows):
        computed.extend(rows.tolist())
        return original(self, rows)

    monkeypatch.setattr(GenotypeReader, "_data", counted)
    scores = np.arange(len(names), dtype=float)
    actual = store.clump([*names, "absent"], [*scores, 0], r2=0.01, kb=3000)
    expected = ogld.open(TEST_STORE).clump([*names, "absent"], [*scores, 0], r2=0.01, kb=3000)
    _equal(actual, expected)
    assert computed
    assert set(computed) <= hits
    assert len(computed) == len(hits), "each component decodes its hits once, not the panel window"
