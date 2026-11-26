"""`opengwasdb recount-top-hits`: set a release's Top-Hit Counts from its stored tiers, in
place (issue #509, spec §21.4, ADR 0107).

`n_hits_5e8`, `n_hits_5e6` and `n_hits_5e4` count the rows each Analysis has in the
matching tier (ADR 0032), which `analyses.top_hit_counts` holds them to. Until #509,
`complete-hybrid` folded panel crossovers into the Dense Component and rebuilt its tiers
without recounting them, so a completed Hybrid published counts its tiers contradict
(OGS-00011: 1,028 of 3,245 Analyses at 5e-8). This repairs such a release without a
rebuild:

- Each `analyses.tsv` the release carries is recounted from the tiers it counts
  (`top_hits.counted_components`, the rule's owner too): a Hybrid's Dense Component's own
  table first, where it carries one (before 0.3.0, #472), then the root's. Each row is
  matched to its tier slot by `analysis_index`, as the rule matches it, and the table keeps
  its row order. The counts are set, not added onto.
- Every table and tier is read, and every count computed, before anything is written. A
  release the tiers cannot count (`top_hits.UncountableTiers`: a component with no Top-Hit
  Index, a missing or malformed tier, an `analysis_index` that is not 0 to n-1 each once)
  or a table without the three columns is refused whole.
- Only a table whose counts change is written, as `clump` writes its counts
  (`model.analyses.write_analysis_columns`: every other column and value as read,
  `n_loci_5e8` and `n_unclumped_5e8` included, the file replaced atomically).
- Then each table's `overview.html`, where the release carries one, is regenerated if it
  differs from what the table renders (`overview.refresh_overview_html`), written
  atomically. A run killed between a table and its overview is therefore completed by the
  next run, which finds the counts right and the page stale. A release whose counts and
  pages are right is left untouched.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from opengwasdb.model.analyses import TOP_HIT_COUNT_COLUMNS, write_analysis_columns
from opengwasdb.store.open import OpenGWASDBStore, open_store
from opengwasdb.store_reader.analyses import AnalysesTable, read_analyses
from opengwasdb.store_reader.enums import PrimaryStorageLayout
from opengwasdb.store_reader.format import UnsupportedFormatVersion
from opengwasdb.store_reader.release import release_arrays
from opengwasdb.top_hits import UncountableTiers, counted_components, counted_hits, table_slots


class RecountRefusal(Exception):
    """A release `recount_top_hits` cannot recount; nothing was written."""


@dataclass(frozen=True)
class RecountedTable:
    """One `analyses.tsv` and what recounting it changes: the new columns, in the table's
    row order, and per column how many Analyses' counts change."""

    path: Path
    n_analyses: int
    columns: dict[str, list[str]]
    changed: dict[str, int]
    n_analyses_changed: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "table": str(self.path),
            "analyses": self.n_analyses,
            "analyses_changed": self.n_analyses_changed,
            "changed": self.changed,
        }


@dataclass(frozen=True)
class RecountRun:
    """What one run did, for the CLI: each table recounted, in the order written, and the
    overviews it regenerated."""

    store: str
    seconds: float
    tables: list[RecountedTable] = field(default_factory=list)
    overviews: list[Path] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "store": self.store,
            "seconds": round(self.seconds, 3),
            "tables": [table.as_dict() for table in self.tables],
            "written": [str(t.path) for t in self.tables if t.n_analyses_changed],
            "overviews_written": [str(path) for path in self.overviews],
        }


def recount_top_hits(store_path: str | Path) -> RecountRun:
    """Recount the Top-Hit Counts of the release at `store_path` in place (module
    docstring). Every refusal comes before anything is written."""
    from opengwasdb.overview import refresh_overview_html

    started = time.perf_counter()
    try:
        release = open_store(store_path)
    except UnsupportedFormatVersion as exc:
        raise RecountRefusal(str(exc)) from exc
    planned = [_recounted(each) for each in _table_holders(release)]
    overviews = []
    for table in planned:
        if table.n_analyses_changed:
            write_analysis_columns(table.path, table.columns)
        page = table.path.parent / "overview.html"
        if page.is_file() and refresh_overview_html(page.parent, read_analyses(table.path)):
            overviews.append(page)
    return RecountRun(str(release.path), time.perf_counter() - started, planned, overviews)


def _table_holders(release: OpenGWASDBStore) -> list[OpenGWASDBStore]:
    """The releases whose own `analyses.tsv` is recounted, in the order written: a
    Hybrid's Dense Component first, where it carries its own table, then the release."""
    if release.manifest.primary_layout is not PrimaryStorageLayout.HYBRID:
        return [release]
    dense = release.dense_component()
    return [dense, release] if dense.analyses_path != release.analyses_path else [release]


def _recounted(holder: OpenGWASDBStore) -> RecountedTable:
    """`holder`'s table recounted from the tiers it counts, each row by its
    `analysis_index`; refused where it cannot be (module docstring)."""
    table = _recountable_table(holder.analyses_path)
    columns = _recounted_columns(holder, table)
    differs = _differing(table, columns)
    return RecountedTable(
        path=holder.analyses_path,
        n_analyses=len(table.rows),
        columns=columns,
        changed={c: sum(c in d for d in differs) for c in TOP_HIT_COUNT_COLUMNS},
        n_analyses_changed=sum(1 for d in differs if d),
    )


def _recounted_columns(holder: OpenGWASDBStore, table: AnalysesTable) -> dict[str, list[str]]:
    """The three columns of `holder`'s table as its tiers count them, in the table's row
    order, each row at its `analysis_index`'s slot; refused where they cannot be counted."""
    roots = {path: release_arrays(path) for path in counted_components(holder)}
    try:
        slots = table_slots(table.rows)
        counts = counted_hits(roots, len(table.rows))
    except UncountableTiers as exc:
        raise RecountRefusal(f"{holder.analyses_path} cannot be recounted: {exc}") from exc
    return {c: [str(counts[c][slot]) for slot in slots] for c in TOP_HIT_COUNT_COLUMNS}


def _differing(table: AnalysesTable, columns: dict[str, list[str]]) -> list[set[str]]:
    """Per row, the Top-Hit Count columns whose recounted value differs from the table's."""
    return [
        {c for c in TOP_HIT_COUNT_COLUMNS if row.get(c, "") != columns[c][i]}
        for i, row in enumerate(table.rows)
    ]


def _recountable_table(table_path: Path) -> AnalysesTable:
    """The table at `table_path`, refused unless it holds the three columns."""
    if not table_path.is_file():
        raise RecountRefusal(f"{table_path} does not exist")
    table = read_analyses(table_path)
    missing = [c for c in TOP_HIT_COUNT_COLUMNS if c not in table.fieldnames]
    if missing:
        raise RecountRefusal(f"{table_path} has no column {', '.join(missing)} to recount")
    return table


__all__ = ["RecountRefusal", "RecountRun", "RecountedTable", "recount_top_hits"]
