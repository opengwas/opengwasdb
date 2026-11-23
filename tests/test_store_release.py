"""The Store Reader's read-only release refuses a Variant Axis it cannot read (#464).

A Store Release's 0.2.x Variant Table is read through a `LegacyVariantTable`
that opengwasdb supplies (`opengwasdb.variants.legacy_table`). A release opened
by the Store Reader alone has none, and asking it for a Variant Axis must say
so rather than return an axis that cannot answer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from opengwasdb.store.open import open_store
from opengwasdb.store_reader.release import StoreRelease, open_release
from opengwasdb.variants.legacy_table import TabixVariantTable


def test_a_release_opened_without_a_table_reader_refuses_a_variant_axis(
    dense_fixture_store: Path,
) -> None:
    release = open_release(dense_fixture_store)
    assert type(release) is StoreRelease and release.legacy is None
    with pytest.raises(ValueError, match="without a reader for its Store Variant Table"):
        release.variant_axis(aliases=False)


def test_open_store_supplies_the_table_reader(dense_fixture_store: Path) -> None:
    release = open_store(dense_fixture_store)
    axis = release.variant_axis(aliases=True)
    try:
        assert isinstance(axis._legacy, TabixVariantTable)
        assert axis.n_variants > 0 and axis.by_index(0) is not None
    finally:
        axis.close()
