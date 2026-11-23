"""The variant table stands alone: one class answers every variant read (ogld#94).

`VariantTable` owns the Variant Set's rows and the ALID lookup, so a reader composes it and the
Genotype Store reader (opengwas/ogld#91) will compose the same one. This holds it to `StoreReader`'s
own answers on the Test Store, over a present ALID, an absent one, a duplicate and one wider than
the index's slot — with the store's committed companion-less index, and with a companion derived
over a copy of it (ADR 0096, ADR 0098). A table needs none of the LD Store's LD: it is opened here
on its own, from a location, and never through a reader.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import zarr
from store_copies import TEST_STORE, copy_store

from opengwasdb.ld.store import StoreReader
from opengwasdb.ld.store.assembly import derive_alid_mmap
from opengwasdb.ld.store.layout import ALID_INDEX_WIDTH
from opengwasdb.ld.store.location import ChunkCache
from opengwasdb.ld.store.reader import open_variant_table
from opengwasdb.ld.store.variants import VariantTable


class Counting(ChunkCache):
    """A chunk cache that records which arrays it was asked for."""

    def __init__(self) -> None:
        super().__init__()
        self.taken: list[str] = []

    def take(self, array: zarr.Array[Any], coords: npt.ArrayLike) -> npt.NDArray[Any]:
        self.taken.append(array.path)
        return super().take(array, coords)

    def runs(
        self, array: zarr.Array[Any], first: npt.ArrayLike, length: npt.ArrayLike
    ) -> list[npt.NDArray[Any]]:
        self.taken.append(array.path)
        return super().runs(array, first, length)


def table_alids(store: Path) -> list[str]:
    """Every ALID the store's variant table holds, in Variant Set row order (ld-store §2)."""
    return np.asarray(zarr.open_group(store, mode="r")["variants/alid"][:]).tolist()


@pytest.fixture(scope="module", params=["canonical index", "local companion"])
def store(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The Test Store as committed, and a copy of it with a derived companion (ADR 0098)."""
    if request.param == "canonical index":
        assert not (TEST_STORE / "variants" / "alid_mmap").exists(), "no companion is committed"
        return TEST_STORE
    copy = copy_store(tmp_path_factory.mktemp("companion") / "store")
    derive_alid_mmap(copy)
    assert (copy / "variants" / "alid_mmap").is_dir(), "the companion must be there to be tested"
    return copy


def test_a_variant_table_gives_the_same_answers_as_the_reader_it_leaves_out(store: Path) -> None:
    names = table_alids(store)
    present = next(a for a in names if len(a.encode("utf-8")) < ALID_INDEX_WIDTH)
    wide = next(a for a in names if len(a.encode("utf-8")) > ALID_INDEX_WIDTH)
    absent = "22:16050075:A:T"
    # Assert the fixture is meaningful before asserting on it: the four asks are one of each case.
    assert absent not in names, "the absent ALID must be absent"
    assert len(wide.encode("utf-8")) > ALID_INDEX_WIDTH, "the over-width ALID must not fit the slot"
    asked = [present, absent, present, wide]
    rows = np.array([0, 1, names.index(present), names.index(wide), 500], dtype=np.int64)

    table: VariantTable = open_variant_table(store)
    reader = StoreReader.open(store)
    try:
        # The companion, where one is published, is the path the fitting names take (ADR 0098).
        np.testing.assert_array_equal(table.rows_of(asked), reader.rows_of(asked))
        assert table.rows_of(asked).tolist() == [
            names.index(present),
            -1,
            names.index(present),
            names.index(wide),
        ]
        assert (table._alid_mmap is not None) == (store / "variants/alid_mmap").is_dir()

        table_chrom, table_position = table.loci_of(rows)
        reader_chrom, reader_position = reader.loci_of(rows)
        np.testing.assert_array_equal(table_chrom, reader_chrom)
        np.testing.assert_array_equal(table_position, reader_position)

        for chrom in table.names:
            assert table.chromosome_range(chrom) == reader.chromosome_range(chrom)

        # A literal region of the Test Store and the rows its positions alone give (ld-store §2):
        # three rows, so an empty answer fails here, and the delegate is not compared with itself.
        assert table.rows_in("22", 16_020_000, 16_030_000).tolist() == [468, 469, 470]
        np.testing.assert_array_equal(table.alids_of(rows), reader.alids_of(rows))

        ours, theirs = table.variants(rows), reader.variants(rows)
        np.testing.assert_array_equal(ours.chrom, theirs.chrom)
        np.testing.assert_array_equal(ours.position, theirs.position)
        np.testing.assert_array_equal(ours.alid, theirs.alid)
        np.testing.assert_array_equal(ours.a1, theirs.a1)
        np.testing.assert_array_equal(ours.a2, theirs.a2)
        np.testing.assert_array_equal(ours.a1_frequency, theirs.a1_frequency)
        assert len(ours) == len(rows), "the table read the rows it was asked for"
    finally:
        table.close()
        reader.close()


def test_a_replaced_chunk_cache_reaches_every_variant_read() -> None:
    """The chunk cache has one owner, so a replaced cache is the one both halves read through.

    `tests/ld/test_bench.py` and `scripts/ld/bench/latency.py` replace `reader._cache` to measure
    what a read holds, and a variant read that kept the cache it was constructed with would be
    measured against the wrong budget — and one reader would keep its positions and its pairs in two
    caches under two budgets (ld-store §10). So the test replaces it exactly as the bench does.
    """
    reader = StoreReader.open(TEST_STORE)
    present = next(a for a in table_alids(TEST_STORE) if len(a.encode("utf-8")) < ALID_INDEX_WIDTH)
    cache = Counting()
    reader._cache = cache
    assert cache.taken == [], "a cache just swapped in has served nothing"
    try:
        reader.alids_of(np.array([0, 1]))
        reader.rows_of([present])
        reader.loci_of(np.array([0, 1]))
        assert reader._variant_table._opened is reader._opened, "one opened store, not two copies"
        assert "variants/alid" in cache.taken, f"alids_of read through the old cache: {cache.taken}"
        assert "variants/position" in cache.taken, (
            f"loci_of read through the old cache: {cache.taken}"
        )
        assert any(path.startswith("variants/alid_index/") for path in cache.taken), (
            f"rows_of searched the index through the old cache: {cache.taken}"
        )
        # The store's own facts are one reference too, not a copy each half holds: a rename names
        # the store for both, and the provenance the reads judge rows by is the same record.
        assert reader._variant_table.provenance is reader.provenance
        reader._location = "/renamed/store"
        assert reader._variant_table.location == "/renamed/store"
    finally:
        reader.close()


def test_a_table_answers_with_the_ld_arrays_gone(tmp_path: Path) -> None:
    """The variant half needs no LD Store's LD: the root metadata and `variants/` are enough
    (opengwas/ogld#94).

    A Genotype Store reader composes this table over a store that holds no Near Band, Far Pairs or
    Partner Rows, so a variant read that reached one would fail there. Here the three groups are
    absent, and the six reads answer what the committed store's table answers.
    """
    stripped = tmp_path / "variants-only"
    shutil.copytree(TEST_STORE / "variants", stripped / "variants")
    shutil.copy2(TEST_STORE / "zarr.json", stripped / "zarr.json")
    assert sorted(path.name for path in stripped.iterdir()) == ["variants", "zarr.json"], (
        "near/, far/ and partners/ must be gone for this to mean anything"
    )
    whole = open_variant_table(TEST_STORE)
    table = open_variant_table(stripped)
    try:
        names = table_alids(stripped)
        present = next(a for a in names if len(a.encode("utf-8")) < ALID_INDEX_WIDTH)
        wide = next(a for a in names if len(a.encode("utf-8")) > ALID_INDEX_WIDTH)
        asked = [present, "22:16050075:A:T", present, wide]
        rows = np.array([0, 1, names.index(present), 500], dtype=np.int64)
        np.testing.assert_array_equal(table.rows_of(asked), whole.rows_of(asked))
        assert table.rows_of(asked).tolist()[2] == names.index(present), "the present ALID is found"
        np.testing.assert_array_equal(table.loci_of(rows)[0], whole.loci_of(rows)[0])
        np.testing.assert_array_equal(table.loci_of(rows)[1], whole.loci_of(rows)[1])
        assert table.chromosome_range("22") == whole.chromosome_range("22")
        assert table.rows_in("22", 16_020_000, 16_030_000).tolist() == [468, 469, 470]
        np.testing.assert_array_equal(table.alids_of(rows), whole.alids_of(rows))
        ours, theirs = table.variants(rows), whole.variants(rows)
        for column in ("chrom", "position", "alid", "a1", "a2", "a1_frequency"):
            np.testing.assert_array_equal(
                ours.columns()[column], theirs.columns()[column], err_msg=column
            )
    finally:
        table.close()
        whole.close()
