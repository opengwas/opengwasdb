"""The Store Reader's read-only release refuses a Variant Axis it cannot read (#464).

A Store Release's 0.2.x Variant Table is read through a `LegacyVariantTable`
that opengwasdb supplies (`opengwasdb.variants.legacy_table`). A release opened
by the Store Reader alone has none, and asking it for a Variant Axis must say
so rather than return an axis that cannot answer.
"""

from __future__ import annotations

from pathlib import Path

from opengwasdb.store.open import open_store
from opengwasdb.store_reader.release import StoreRelease, open_release
from opengwasdb.variants.legacy_table import TabixVariantTable


def test_a_0_3_0_release_opened_without_a_table_reader_reads_its_sidecars(
    dense_fixture_store: Path,
) -> None:
    """No 0.3.0 read uses the table (#474); a 0.2 release is still refused
    (`tests/test_open_release_without_legacy.py`)."""
    from opengwasdb.store_reader.variant_axis import NoVariantTable

    release = open_release(dense_fixture_store)
    assert type(release) is StoreRelease and release.legacy is None
    axis = release.variant_axis(aliases=False)
    assert isinstance(axis._legacy, NoVariantTable)
    assert axis.n_variants > 0 and axis.by_index(0) is not None


def test_open_store_supplies_the_table_reader(dense_fixture_store: Path) -> None:
    release = open_store(dense_fixture_store)
    axis = release.variant_axis(aliases=True)
    try:
        assert isinstance(axis._legacy, TabixVariantTable)
        assert axis.n_variants > 0 and axis.by_index(0) is not None
    finally:
        axis.close()
