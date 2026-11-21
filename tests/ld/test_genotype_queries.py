"""Threshold-query parity and the hit-only clump seam (#91)."""

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import build_fixtures
import numpy as np
import pytest
import testdata
from contract.support import TEST_STORE
from test_genotype_store import genotype_store as genotype_store

import ogld
from ogld.genotypes import reader as genotype_reader
from ogld.genotypes.pgen import Genotypes
from ogld.genotypes.reader import GenotypeReader


def _equal(actual: object, expected: object) -> None:
    assert type(actual) is type(expected)
    for field in dataclasses.fields(actual):
        a, b = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b, err_msg=field.name)
        elif dataclasses.is_dataclass(a):
            _equal(a, b)
        else:
            assert a == b, field.name


@pytest.mark.parametrize("fill", [None, 0.0])
@pytest.mark.parametrize("centre", [20000000, 25000000])
def test_regions_equal_the_test_store(
    genotype_store: Path, fill: float | None, centre: int
) -> None:
    actual = ogld.open(genotype_store).region("chr22", centre, 1500000, fill=fill)
    expected = ogld.open(TEST_STORE).region("22", centre, 1500000, fill=fill)
    assert len(actual.r) > 10
    _equal(actual, expected)


def test_parallel_pgen_reads_preserve_codes_and_duplicates(
    genotype_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(genotype_reader, "_READ_ROWS", 128)
    names = build_fixtures.alids(np.flatnonzero(testdata.variants().passes_filters))
    assert len(names) > 256
    names = [*names[::-1], names[0]]
    actual = ogld.open(genotype_store).matrix(names)
    expected = ogld.open(TEST_STORE).matrix(names)
    _equal(actual, expected)


def test_failed_parallel_pgen_read_refuses_the_entire_query(
    genotype_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(genotype_reader, "_READ_ROWS", 128)
    names = build_fixtures.alids(np.flatnonzero(testdata.variants().passes_filters))
    assert len(names) > 256

    def broken(self, rows, out):
        raise ogld.InputError("panel.pgen: injected native read failure")

    monkeypatch.setattr(Genotypes, "rows_into", broken)
    with pytest.raises(ogld.InputError, match=r"panel\.pgen"):
        ogld.open(genotype_store).matrix(names)


def test_concurrent_requests_keep_one_blas_thread_until_both_finish(genotype_store: Path) -> None:
    done = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("genotype_threads.py")), str(genotype_store)],
        env={**os.environ, "OPENBLAS_NUM_THREADS": "2"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr


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
