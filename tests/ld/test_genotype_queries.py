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

import ogld
from ogld.genotypes import reader as genotype_reader
from ogld.genotypes.pgen import Genotypes, PgenFileset
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


def test_each_request_enters_the_blas_guard_once(
    genotype_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import contextmanager

    entries: list[int] = []
    original = genotype_reader._one_blas_thread

    @contextmanager
    def counted():
        entries.append(len(entries))
        with original():
            yield

    monkeypatch.setattr(genotype_reader, "_one_blas_thread", counted)
    names = build_fixtures.alids(np.flatnonzero(testdata.variants().passes_filters)[::9])
    assert len(names) > 20, "the fixture must span enough leads for a per-row guard to show"
    store = ogld.open(genotype_store)
    scores = np.arange(len(names), dtype=float)
    checks = [
        ("matrix", lambda: store.matrix(names)),
        ("region", lambda: store.region("22", 25000000, 1500000)),
        ("proxies", lambda: store.proxies(names, r2=0.01, kb=3000)),
        ("clump", lambda: store.clump(names, scores, r2=0.01, kb=3000)),
    ]
    for label, call in checks:
        before = len(entries)
        call()
        entered = len(entries) - before
        assert entered == 1, f"{label} entered the BLAS guard {entered} times"


def test_a_long_lived_reader_reuses_and_closes_its_pgen_handles(
    genotype_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[Path] = []
    closed: list[Path] = []
    original_open = Genotypes.__init__
    original_close = getattr(Genotypes, "close", None)

    def counted_open(self: Genotypes, fileset: PgenFileset) -> None:
        original_open(self, fileset)
        opened.append(self._fileset.pgen)

    def counted_close(self: Genotypes) -> None:
        closed.append(self._fileset.pgen)
        if original_close is not None:
            original_close(self)

    monkeypatch.setattr(Genotypes, "__init__", counted_open)
    monkeypatch.setattr(Genotypes, "close", counted_close, raising=False)
    names = build_fixtures.alids(np.flatnonzero(testdata.variants().passes_filters)[::9])
    store = ogld.open(genotype_store)
    store.matrix(names)
    first = len(opened)
    assert first >= 1, "the fixture must decode at least one fileset"
    store.matrix(names)
    store.proxies([names[0]], r2=0.01, kb=3000)
    assert len(opened) == first, f"a second request rebuilt {len(opened) - first} pgen handles"
    store._reader.close()
    assert sorted(closed) == sorted(opened), "close() closes every handle the reader kept"


def test_the_handle_bound_evicts_and_closes(
    genotype_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reader = ogld.open(genotype_store)._reader
    filesets = [reader._filesets[0]] * 3  # three pool entries, each a real fileset to open
    pool = genotype_reader._PgenHandles(filesets, keep=2)
    opened: list[int] = []
    closed: list[int] = []
    original_open = Genotypes.__init__

    def counted_open(self: Genotypes, fileset: PgenFileset) -> None:
        opened.append(len(opened))
        original_open(self, fileset)

    monkeypatch.setattr(Genotypes, "__init__", counted_open)
    monkeypatch.setattr(Genotypes, "close", lambda self: closed.append(len(closed)))
    for index in range(3):
        with pool.lease(index):
            pass
    # The third lease needs a handle and the bound is two, so one idle handle is evicted and
    # closed, and two stay for the next decode.
    assert len(opened) == 3, "each fileset's first lease opens one handle"
    assert len(closed) == 1, f"the bound evicted and closed {len(closed)} handles, not one"
    assert len(pool._idle) == 2
    pool.close()
    assert len(closed) == 3, "close() closes every handle left idle"


def test_a_lead_outside_its_window_is_refused_by_name(genotype_store: Path) -> None:
    store = ogld.open(genotype_store)
    reader = store._reader
    rows = np.flatnonzero(testdata.variants().passes_filters)[:5]
    names = build_fixtures.alids(rows)
    hits = reader.rows_of(names)
    chrom = str(reader.loci_of(hits)[0][0])
    start, stop = reader.chromosome_range(chrom)
    local = hits - start
    absent = next(row for row in range(stop - start) if row not in set(local.tolist()))
    alid = str(reader.alids_of(np.array([start + absent], dtype=np.int64))[0])
    with reader.clump_rows(chrom, local, 3000.0) as partners:
        with pytest.raises(ogld.InputError, match=alid):
            partners(absent)
