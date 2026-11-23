"""A Store's 0.2.x Store Variant Table, read through pysam, and its `variant_aliases`.

Format 0.2.x keeps the Store Variant Table as a BGZF file with a tabix index,
which only pysam reads, and a Store built before the rsid index names rsids in
`index.sqlite`'s `variant_aliases` table. The Store Reader may import neither
pysam nor sqlite3 (#464): its Variant Axis asks a `LegacyVariantTable` (a
Protocol in `opengwasdb.store_reader.variant_axis`) for these reads, and this
module is the one implementation, and the only place opengwasdb reads that
table: `TabixVariantTable`, its opener `open_legacy_variant_table`, and the
whole-table scan `iter_variant_records`. #465 makes the table lazy; #476
deletes this module with the 0.2.x table.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path

import pysam

from opengwasdb.store_reader.variant_axis import (
    VariantRecord,
    parse_variant_line,
    variant_table_path,
)


def iter_variant_records(table_path: str | Path) -> Iterator[VariantRecord]:
    """Stream a Store Variant Table row by row, in `variant_index` order.

    What `VariantAxis.all` materialises. A caller that only needs one field per
    row -- the EAF orientation audit needs `variant_index` and `alid` -- would
    otherwise pay tens of millions of `VariantRecord` objects to get at them,
    on stores where that is gigabytes.
    """
    for line in _table_lines(table_path):
        yield parse_variant_line(line)


def _table_lines(table_path: str | Path) -> Iterator[str]:
    """A Store Variant Table's rows as text, header lines skipped."""
    with pysam.BGZFile(str(table_path), "r", None) as handle:
        for raw_line in handle:
            line = raw_line.decode("utf-8")
            if line.startswith("#"):
                continue
            yield line


class TabixVariantTable:
    """A Store's 0.2.x Variant Table, read through pysam, and its `variant_aliases`.

    Implements the Store Reader's `LegacyVariantTable` Protocol: the tabix range
    fetches, the BGZF seek per row, the whole-table scan and the SQLite rsid
    fallback the Variant Axis used to make itself (#464). `aliases` is an open
    `index.sqlite` connection, or None for a Store whose rsids are not looked up
    there (a Ragged release); a connection this object opened (`owns_aliases`)
    it closes with the table.
    """

    def __init__(
        self,
        store_path: str | Path,
        aliases: sqlite3.Connection | None = None,
        *,
        owns_aliases: bool = False,
    ) -> None:
        self.table_path = variant_table_path(store_path)
        self._tabix = pysam.TabixFile(str(self.table_path))
        self._aliases = aliases
        self._owns_aliases = owns_aliases

    def fetch(self, chromosome: str, start: int, end: int) -> Iterable[str]:
        try:
            return list(self._tabix.fetch(chromosome, start, end))
        except ValueError:
            return []

    def line_at(self, offset: int) -> str:
        with pysam.BGZFile(str(self.table_path), "r", None) as handle:
            handle.seek(offset)
            line: str = handle.readline().decode("utf-8")
        return line

    def rows(self) -> Iterator[str]:
        return _table_lines(self.table_path)

    def alias_index(self, alias: str) -> int | None:
        if self._aliases is None:
            return None
        row = self._aliases.execute(
            """
            SELECT variant_index
            FROM variant_aliases
            WHERE alias = ?
            ORDER BY variant_index
            LIMIT 1
            """,
            (alias,),
        ).fetchone()
        return None if row is None else int(row["variant_index"])

    def close(self) -> None:
        self._tabix.close()
        if self._owns_aliases and self._aliases is not None:
            self._aliases.close()


def open_legacy_variant_table(store_path: Path, *, aliases: bool) -> TabixVariantTable:
    """The Store Reader's `LegacyTableOpener`: a Store's Variant Table, with its
    `index.sqlite` opened for `variant_aliases` when `aliases` asks."""
    connection = None
    if aliases:
        connection = sqlite3.connect(str(Path(store_path) / "index.sqlite"))
        connection.row_factory = sqlite3.Row
    return TabixVariantTable(store_path, connection, owns_aliases=True)
