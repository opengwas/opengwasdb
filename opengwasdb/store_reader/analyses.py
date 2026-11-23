"""In-memory `analyses.tsv` index (ADR 0030, issue #22).

`analyses.tsv` is the sole source of truth for Analytical Metadata -- there is
no SQLite `analyses` table to query anymore. Both of the facade's access
patterns (a single indexed point lookup by `analysis_id`, and a full-table
scan) translate directly to reading the file into a dict once at store-open.
"""

from __future__ import annotations

from collections.abc import ItemsView
from pathlib import Path

from opengwasdb.model.analyses import read_analyses


class AnalysesIndex:
    """`analyses.tsv` read once, keyed by `analysis_id` and by `analysis_index`."""

    def __init__(self, store_path: str | Path):
        table = read_analyses(Path(store_path) / "analyses.tsv")
        self._by_id: dict[str, dict[str, str]] = {}
        self._by_index: dict[int, dict[str, str]] = {}
        for row in table.rows:
            self._by_id[row["analysis_id"]] = row
            self._by_index[int(row["analysis_index"])] = row

    def by_id(self, analysis_id: str) -> dict[str, str] | None:
        return self._by_id.get(analysis_id)

    def all(self) -> dict[int, dict[str, str]]:
        return dict(self._by_index)

    def items(self) -> ItemsView[int, dict[str, str]]:
        """A live view over (analysis_index, row) pairs for internal iteration
        -- unlike `all()`, this doesn't copy, so it's for callers that only
        read (a caller wanting an owned dict to mutate should use `all()`)."""
        return self._by_index.items()
