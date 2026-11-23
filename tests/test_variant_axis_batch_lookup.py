"""Vectorised identifier -> row-index resolution for dense lookup (issue #3).

`indices_by_identifiers` must resolve many identifiers to Store-local Variant
Indices without materialising a full `VariantRecord` per identifier (which
opens `variants.tsv.gz` on every call, see `by_index`), while still falling
back cleanly for alias/rsid identifiers and for stores built before the
mmap'd ALID search index existed.
"""

from __future__ import annotations

import sqlite3

import pytest
from format_0_2_3 import with_fixed_width_alid_index

from opengwasdb.index.sqlite import connect, initialise_schema
from opengwasdb.store_reader.alid_index import MissingAlidIndex
from opengwasdb.store_reader.format import CURRENT_FORMAT_VERSION
from opengwasdb.store_reader.variant_axis import variant_alid_bytes_path, variant_alid_rows_path
from opengwasdb.variants.axis import VariantAxis, write_variant_axis
from opengwasdb.variants.normalise import CanonicalVariant


def _variant(chrom: str, pos: int, a1: str, a2: str) -> CanonicalVariant:
    return CanonicalVariant(chromosome=chrom, position=pos, effect_allele=a1, other_allele=a2)


def _build_axis(tmp_path, aliases: list[tuple[str, int]] | None = None) -> sqlite3.Connection:
    variants = [
        _variant("1", 1_000, "A", "G"),
        _variant("1", 2_000, "C", "T"),
        _variant("1", 3_000, "A", "T"),
    ]
    write_variant_axis(tmp_path, variants, {})
    connection = connect(tmp_path / "aliases.sqlite")
    initialise_schema(connection)
    if aliases:
        connection.executemany(
            "INSERT INTO variant_aliases(alias, variant_index) VALUES (?, ?)", aliases
        )
        connection.commit()
    return connection


def test_resolves_multiple_canonical_alids(tmp_path):
    connection = _build_axis(tmp_path)
    axis = VariantAxis(tmp_path, connection, format_version=CURRENT_FORMAT_VERSION)
    try:
        indices = axis.indices_by_identifiers(["1:1000:A:G", "1:3000:A:T"])
        assert sorted(indices.tolist()) == [0, 2]
        assert indices.dtype == "int32"
    finally:
        axis.close()
        connection.close()


def test_drops_missing_alids(tmp_path):
    connection = _build_axis(tmp_path)
    axis = VariantAxis(tmp_path, connection, format_version=CURRENT_FORMAT_VERSION)
    try:
        indices = axis.indices_by_identifiers(["1:1000:A:G", "1:9999:A:G"])
        assert indices.tolist() == [0]
    finally:
        axis.close()
        connection.close()


def test_resolves_rsid_aliases_alongside_canonical_alids(tmp_path):
    connection = _build_axis(tmp_path, aliases=[("rs1", 0), ("rs3", 2)])
    axis = VariantAxis(tmp_path, connection, format_version=CURRENT_FORMAT_VERSION)
    try:
        indices = axis.indices_by_identifiers(["rs1", "1:2000:C:T", "rs3", "unknown_rsid"])
        assert sorted(indices.tolist()) == [0, 1, 2]
    finally:
        axis.close()
        connection.close()


def test_matches_by_identifier_row_for_row(tmp_path):
    connection = _build_axis(tmp_path, aliases=[("rs1", 0)])
    axis = VariantAxis(tmp_path, connection, format_version=CURRENT_FORMAT_VERSION)
    try:
        identifiers = ["1:1000:A:G", "rs1", "1:3000:A:T", "does:not:exist:x"]
        expected = sorted(
            record.variant_index
            for id_ in identifiers
            if (record := axis.by_identifier(id_)) is not None
        )
        actual = sorted(axis.indices_by_identifiers(identifiers).tolist())
        assert actual == expected
    finally:
        axis.close()
        connection.close()


def test_a_0_2_release_without_its_alid_index_is_refused_not_scanned(tmp_path):
    """A 0.2 release carries the fixed-width pair; without it the axis is refused, where
    it once answered every ALID by scanning the table (#467 review)."""
    connection = _build_axis(tmp_path)
    with_fixed_width_alid_index(tmp_path)
    variant_alid_bytes_path(tmp_path).unlink()
    variant_alid_rows_path(tmp_path).unlink()
    try:
        with pytest.raises(MissingAlidIndex, match="variant_alid_bytes.npy"):
            VariantAxis(tmp_path, connection, format_version="0.2.3")
    finally:
        connection.close()


def test_empty_identifiers_returns_empty_array(tmp_path):
    connection = _build_axis(tmp_path)
    axis = VariantAxis(tmp_path, connection, format_version=CURRENT_FORMAT_VERSION)
    try:
        indices = axis.indices_by_identifiers([])
        assert indices.tolist() == []
        assert indices.dtype == "int32"
    finally:
        axis.close()
        connection.close()
