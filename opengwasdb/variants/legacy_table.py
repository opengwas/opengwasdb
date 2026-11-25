"""A Store's 0.2.x Store Variant Table, read through pysam, and its `variant_aliases`.

Format 0.2.x keeps the Store Variant Table as a BGZF file with a tabix index,
which only pysam reads, and a Store built before the rsid index names rsids in
`index.sqlite`'s `variant_aliases` table. A 0.3.0 release has no such table (ADR
0102, #471): its Variant Axis never asks `alias_index`, and its facade never names
the index to open. The Store Reader may import neither
pysam nor sqlite3 (#464): its Variant Axis asks a `LegacyVariantTable` (a
Protocol in `opengwasdb.store_reader.variant_axis`) for these reads, and this
module is the one implementation, and the only place opengwasdb reads that
table: `TabixVariantTable`, its opener `open_legacy_variant_table`, and the
whole-table scan `iter_variant_records`. Nothing is opened until a read needs
it (#465): `pysam` and `sqlite3` are imported, and the tabix handle and the
`index.sqlite` connection opened, on first use, so a query a search index
answers never loads either. #476 deletes this module with the 0.2.x table.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3

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
    """A Store Variant Table's rows as text, header lines skipped.

    A missing table raises `FileNotFoundError`: pysam's BGZF reader crashes the
    process on one, and a 0.3.0 Hybrid's Dense Component carries none (#472).
    """
    import pysam

    if not Path(table_path).is_file():
        raise FileNotFoundError(f"no Store Variant Table at {table_path}")
    with pysam.BGZFile(str(table_path), "r", None) as handle:
        for raw_line in handle:
            line = raw_line.decode("utf-8")
            if line.startswith("#"):
                continue
            yield line


def open_index_read_only(path: Path) -> sqlite3.Connection:
    """A release's `index.sqlite`, opened read-only, rows as `sqlite3.Row`.

    The one read-only opener of `index.sqlite`: the alias lookup here and the
    validator's `opengwasdb.index.release_index.ReleaseIndex` both open it through
    this (#473). Read-only means a missing index is an error, never a new empty
    file (#334). `sqlite3` is imported on use, so opening a release does not load
    it (#465). Writers open the index read-write through
    `opengwasdb.store.open.connect_index`.
    """
    import sqlite3

    uri = f"{Path(path).resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


class TabixVariantTable:
    """A Store's 0.2.x Variant Table, read through pysam, and its `variant_aliases`.

    Implements the Store Reader's `LegacyVariantTable` Protocol: the tabix range
    fetches, the BGZF seek per row, the whole-table scan and the SQLite rsid
    fallback the Variant Axis used to make itself (#464). Opened lazily (#465):
    constructing one touches no file; the tabix handle opens on the first `fetch`
    and the `index.sqlite` connection on the first `alias_index`.

    `aliases` is an open `index.sqlite` connection the caller keeps; or
    `aliases_path` names the file to open on first use (and close with the
    table); or neither, for a Store whose rsids are not looked up there (a
    Ragged release).

    Threads: one lock guards the first open of each handle and every use of the
    connection this table opens, which is created with `check_same_thread=False`
    and read-only, so any thread may call `alias_index` and none runs two
    statements at once. A pysam `TabixFile` is not shared between threads here
    either: `fetch` holds the same lock. After `close()` every read raises
    `ValueError`, as a closed handle did before the handles were lazy.
    """

    def __init__(
        self,
        store_path: str | Path,
        aliases: sqlite3.Connection | None = None,
        *,
        aliases_path: Path | None = None,
    ) -> None:
        if aliases is not None and aliases_path is not None:
            raise ValueError("pass an open `aliases` connection or an `aliases_path`, not both")
        self.table_path = variant_table_path(store_path)
        self._tabix_handle: pysam.TabixFile | None = None
        self._aliases = aliases
        self._aliases_path = aliases_path
        self._lock = threading.Lock()
        self._closed = False

    def _require_open(self) -> None:
        if self._closed:
            raise ValueError(f"the Variant Table of {self.table_path.parent} is closed")

    def fetch(self, chromosome: str, start: int, end: int) -> Iterable[str]:
        with self._lock:
            self._require_open()
            if self._tabix_handle is None:
                import pysam

                self._tabix_handle = pysam.TabixFile(str(self.table_path))
            try:
                return list(self._tabix_handle.fetch(chromosome, start, end))
            except ValueError:
                return []

    def line_at(self, offset: int) -> str:
        self._require_open()
        import pysam

        with pysam.BGZFile(str(self.table_path), "r", None) as handle:
            handle.seek(offset)
            line: str = handle.readline().decode("utf-8")
        return line

    def rows(self) -> Iterator[str]:
        self._require_open()
        return _table_lines(self.table_path)

    def alias_index(self, alias: str) -> int | None:
        with self._lock:
            self._require_open()
            if self._aliases is None and self._aliases_path is not None:
                self._aliases = open_index_read_only(self._aliases_path)
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
        with self._lock:
            self._closed = True
            if self._tabix_handle is not None:
                self._tabix_handle.close()
                self._tabix_handle = None
            if self._aliases_path is not None and self._aliases is not None:
                self._aliases.close()
                self._aliases = None


def open_legacy_variant_table(store_path: Path, *, aliases: bool) -> TabixVariantTable:
    """The Store Reader's `LegacyTableOpener`: a Store's Variant Table, with its
    `index.sqlite` consulted for `variant_aliases` when `aliases` asks. Neither
    file is opened until a read needs it."""
    return TabixVariantTable(
        store_path, aliases_path=Path(store_path) / "index.sqlite" if aliases else None
    )
