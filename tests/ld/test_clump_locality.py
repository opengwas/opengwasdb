"""Scheduling changes reads, not result order or greedy edges (#40; partner-rows §9.2)."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pytest
from store_copies import TEST_STORE

import ogld


@pytest.mark.parametrize("r2", [0.01, 0.1, 1])
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


def test_clump_reads_only_returned_columns_and_one_partner_row_per_lead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clump reads the Position and ALID columns clump returns, no allele column, and one
    Partner Rows row per lead: never a row of a hit a stronger lead took (partner-rows §8.2)."""
    store = ogld.open(TEST_STORE)
    reader = store._reader
    rows = np.arange(store.provenance["n_variants"])
    names = reader.alids_of(rows).tolist()
    columns: Counter[str] = Counter()
    reads: list[tuple[str, int]] = []
    read = reader._read
    partner_row = reader.partner_row

    def recording(array: Any, coords: Any) -> Any:
        columns[array.path] += len(coords)
        return read(array, coords)

    def recorded(chrom: str, local_row: int) -> Any:
        reads.append((chrom, local_row))
        return partner_row(chrom, local_row)

    monkeypatch.setattr(reader, "_read", recording)
    monkeypatch.setattr(reader, "partner_row", recorded)
    score = np.random.default_rng(40).random(len(rows))
    got = store.clump(names, score, r2=0.01, kb=3000)
    assert len(got.leads()) > 1
    assert not {"variants/a1", "variants/a2", "variants/a1_frequency"} & columns.keys()
    leads = set(got.leads().tolist())
    assert len(reads) == len(leads), "one Partner Rows row per lead, and no other"
    assert len({(chrom, local) for chrom, local in reads}) == len(reads), "no row read twice"
    for chrom, local in reads:
        row = reader.chromosome_range(chrom)[0] + local
        assert reader.alids_of(np.array([row]))[0] in leads, "a row read is a lead's"
