"""The variant table stands alone: one class answers every variant read (ogld#94).

`VariantTable` owns the Variant Set's rows and the ALID lookup, so a reader composes it and the
Genotype Store reader (#91) will compose the same one. This holds it to `StoreReader`'s own
answers on the Test Store, over a present ALID, an absent one, a duplicate and one wider than the
index's slot — with the store's committed companion-less index, and with a companion derived over
a copy of it (ADR 0010, ADR 0012). A table needs none of the LD Store's LD: it is opened here on
its own, from a location, and never through a reader.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr
from store_copies import TEST_STORE, copy_store

from ogld.store import StoreReader
from ogld.store.assembly import derive_alid_mmap
from ogld.store.layout import ALID_INDEX_WIDTH
from ogld.store.reader import open_variant_table
from ogld.store.variants import VariantTable


def table_alids(store: Path) -> list[str]:
    """Every ALID the store's variant table holds, in Variant Set row order (spec §2)."""
    return np.asarray(zarr.open_group(store, mode="r")["variants/alid"][:]).tolist()


@pytest.fixture(scope="module", params=["canonical index", "local companion"])
def store(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The Test Store as committed, and a copy of it with a derived companion (ADR 0012)."""
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
        # The companion, where one is published, is the path the fitting names take (ADR 0012).
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

        span = (16_000_000, 17_000_000)
        np.testing.assert_array_equal(table.rows_in("22", *span), reader.rows_in("22", *span))
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
