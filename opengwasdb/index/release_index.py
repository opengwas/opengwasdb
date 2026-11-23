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

The connection is opened as the readers before it opened it
(`sqlite3.connect`, rows as `sqlite3.Row`), so what a read returns, and what a
missing table or column raises (`UnreadableIndex`), are as they were.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import TracebackType
from typing import Any

from opengwasdb.index.sqlite import get_metadata

#: What a read of a table or column the index lacks raises: sqlite3's own error,
#: named here so a caller can catch it without importing sqlite3.
UnreadableIndex = sqlite3.OperationalError

#: The name every release keeps its index under.
INDEX_FILE = "index.sqlite"


class ReleaseIndex:
    """The `index.sqlite` of the release at `release`, open for reading.

    A context manager: leaving it closes the connection.
    """

    def __init__(self, release: str | Path) -> None:
        self._connection = sqlite3.connect(str(Path(release) / INDEX_FILE))
        self._connection.row_factory = sqlite3.Row

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

    def table_names(self) -> set[str]:
        """The tables the index holds."""
        return {
            row[0]
            for row in self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }

    def schema_names(self) -> set[str]:
        """Every named object the index holds: its tables, indexes and views."""
        return {row[0] for row in self._connection.execute("SELECT name FROM sqlite_master")}

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

    def completion_quality(self) -> list[tuple[Any, Any, Any, Any]]:
        """Every `completion_quality` row as ``(analysis_index, block_id, n_imputed,
        n_missing)``, the values as stored. `UnreadableIndex` when there is no such
        table."""
        return [
            (row[0], row[1], row[2], row[3])
            for row in self._connection.execute(
                "SELECT analysis_index, block_id, n_imputed, n_missing FROM completion_quality"
            )
        ]

    def completion_quality_blocks(self) -> list[tuple[Any, Any]]:
        """Every `completion_quality` row's ``(analysis_index, block_id)``, as stored.
        `UnreadableIndex` when there is no such table."""
        return [
            (row[0], row[1])
            for row in self._connection.execute(
                "SELECT analysis_index, block_id FROM completion_quality"
            )
        ]

    def variant_aliases(self) -> list[tuple[Any, Any]]:
        """Every `variant_aliases` row as ``(alias, variant_index)``, the values as
        stored. `UnreadableIndex` when there is no such table."""
        return [
            (row["alias"], row["variant_index"])
            for row in self._connection.execute(
                "SELECT alias, variant_index FROM variant_aliases"
            ).fetchall()
        ]
