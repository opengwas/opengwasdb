"""A Store Release's `index.sqlite`, read: the named reads the engine makes of it (#473).

Format 0.2.x keeps a few facts in `index.sqlite`: the `metadata` table (the
`dense` layout recording among them), Reference Completion's
`completion_quality` rows, and, for a Store built before the rsid index, the
`variant_aliases` table. The Store Reader may not import sqlite3 (#464), so,
like the 0.2.x Store Variant Table (`opengwasdb.variants.legacy_table`), these
reads live beside it as its SQLite implementation, and count as the Store
Reader (#473 ruling 1): the validator and the observed-view check ask for them
here, by what they return, and no other read module opens `index.sqlite`.
Writing it -- the schema, metadata and quality rows a build or a completion
writes -- is `opengwasdb.index.sqlite`'s and the writers'. #476 removes SQLite
from every read path.

The index is opened read-only, by the one read-only opener the alias lookup also
uses (`opengwasdb.variants.legacy_table.open_index_read_only`), with rows as
`sqlite3.Row` as the readers before it had them: what a read returns, and what a
missing table or column raises (`UnreadableIndex`), are as they were, and a read
never creates a missing index (#334).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import TracebackType
from typing import Any

from opengwasdb.index.sqlite import get_metadata
from opengwasdb.store_reader.release import release_paths
from opengwasdb.variants.legacy_table import open_index_read_only

#: What a read of a table or column the index lacks raises: sqlite3's own error,
#: named here so a caller can catch it without importing sqlite3.
UnreadableIndex = sqlite3.OperationalError

#: The `completion_quality` columns a count of its rows reads, in this order.
QUALITY_COUNTS = ("analysis_index", "block_id", "n_imputed", "n_missing")
#: The `completion_quality` columns that say which blocks it records.
QUALITY_BLOCKS = ("analysis_index", "block_id")


class ReleaseIndex:
    """The `index.sqlite` of the release at `release`, open for reading.

    A context manager: leaving it closes the connection.
    """

    def __init__(self, release: str | Path) -> None:
        self._connection = open_index_read_only(release_paths(Path(release))["index"])

    def __enter__(self) -> ReleaseIndex:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def names(self, *, tables_only: bool) -> set[str]:
        """The named objects the index holds: its tables, or with `tables_only` false
        its indexes and views as well."""
        query = "SELECT name FROM sqlite_master"
        if tables_only:
            query += " WHERE type='table'"
        return {row[0] for row in self._connection.execute(query)}

    def metadata(self, key: str, default: Any = None) -> Any:
        """One `metadata` value, decoded, or `default` when the index records none."""
        return get_metadata(self._connection, key, default=default)

    def completion_quality_columns(self) -> set[str]:
        """The columns `completion_quality` has; empty when there is no such table."""
        return {
            row[1]
            for row in self._connection.execute("PRAGMA table_info(completion_quality)").fetchall()
        }

    def completion_quality_rows_outside(self, n_analyses: int) -> int:
        """How many `completion_quality` rows name an Analysis outside ``[0, n_analyses)``."""
        count: int = self._connection.execute(
            "SELECT COUNT(*) FROM completion_quality "
            "WHERE analysis_index < 0 OR analysis_index >= ?",
            (n_analyses,),
        ).fetchone()[0]
        return count

    def completion_quality(
        self, columns: tuple[str, ...] = QUALITY_COUNTS
    ) -> list[tuple[Any, ...]]:
        """Every `completion_quality` row's `columns`, the values as stored.

        `columns` is `QUALITY_COUNTS` -- what a count of the rows needs -- or
        `QUALITY_BLOCKS`, which reads only the two key columns, so a table missing
        a count column still says which blocks it records. `UnreadableIndex` when
        the table, or one of the columns, is not there.
        """
        if columns not in (QUALITY_COUNTS, QUALITY_BLOCKS):
            raise ValueError(f"completion_quality reads {QUALITY_COUNTS} or {QUALITY_BLOCKS}")
        query = f"SELECT {', '.join(columns)} FROM completion_quality"
        return [tuple(row) for row in self._connection.execute(query)]

    def variant_aliases(self) -> list[tuple[Any, Any]]:
        """Every `variant_aliases` row as ``(alias, variant_index)``, the values as
        stored. `UnreadableIndex` when there is no such table."""
        return [
            (row["alias"], row["variant_index"])
            for row in self._connection.execute(
                "SELECT alias, variant_index FROM variant_aliases"
            ).fetchall()
        ]
