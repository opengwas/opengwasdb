"""Chromosome scheduling changes reads, not result order or greedy edges (#40)."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pytest
from store_copies import TEST_STORE

import ogld


@pytest.mark.parametrize("r2", [0, 0.001, 0.01, 0.1, 1])
def test_interleaved_chromosomes_keep_request_order_and_answers(r2: float) -> None:
    store = ogld.open(TEST_STORE)
    reader = store._reader
    rows = np.array([500, 4, 700, 0, 500, 7, 701, 4])
    names = reader.alids_of(rows).tolist()
    scores = np.array([5, 4, 3, 2, 1, 5, 4, 4], dtype=float)
    variants = reader.variants(rows)
    assert set(variants.chrom) == {"21", "22"}
    got = store.clump([*names, "bad"], [*scores, 1], r2=r2, kb=3000)
    assert got.absent == ("bad",)
    np.testing.assert_array_equal(got.alid, variants.alid)
    np.testing.assert_array_equal(got.chrom, variants.chrom)
    np.testing.assert_array_equal(got.position, variants.position)
    # Independent existing one-chromosome runs are the scheduling oracle.
    for chrom in ("21", "22"):
        at = np.flatnonzero(variants.chrom == chrom)
        one = store.clump([names[i] for i in at], scores[at], r2=r2, kb=3000)
        np.testing.assert_array_equal(got.lead[at], one.lead)
        np.testing.assert_array_equal(got.r[at], one.r)


def test_clump_reads_only_returned_columns_and_never_returns_to_a_chromosome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ogld.open(TEST_STORE)
    reader = store._reader
    rows = np.arange(store.provenance["n_variants"])
    names = reader.alids_of(rows).tolist()
    columns: Counter[str] = Counter()
    visited: list[str] = []
    read = reader._read
    partners = reader.partners

    def recording(array: Any, coords: Any) -> Any:
        columns[array.path] += len(coords)
        return read(array, coords)

    # Record partner chromosomes without polluting the column-read assertion.
    chrom = reader.variants(rows).chrom

    def paired(rows: Any, **kwargs: Any) -> Any:
        visited.extend(chrom[rows].tolist())
        return partners(rows, **kwargs)

    monkeypatch.setattr(reader, "_read", recording)
    monkeypatch.setattr(reader, "partners", paired)
    got = store.clump(names, np.random.default_rng(40).random(len(rows)), r2=0.01, kb=3000)
    assert len(got.leads()) > 1
    assert set(visited) == {"21", "22"}
    assert not {"variants/a1", "variants/a2", "variants/a1_frequency"} & columns.keys()
    assert len(np.flatnonzero(np.array(visited[1:]) != visited[:-1])) == 1
