"""`opengwasdb recount-top-hits`: set a release's Top-Hit Counts from its stored tiers, in
place (issue #509, spec §21.4).

`n_hits_5e8`, `n_hits_5e6` and `n_hits_5e4` count the rows each Analysis has in the
matching tier (ADR 0032), which `analyses.top_hit_counts` holds them to. Until #509,
`complete-hybrid` folded panel crossovers into the Dense Component and rebuilt its tiers
without recounting them, so a completed Hybrid published counts its tiers contradict
(OGS-00011: 1,028 of 3,245 Analyses at 5e-8). This repairs such a release without a
rebuild:

- Each `analyses.tsv` the release carries is recounted from the tiers it counts: a Dense
  or Ragged release's own; a Hybrid's Dense Component's own, where it carries a table
  (before 0.3.0, #472), then the root's from both components. The counts are reset, not
  added onto, as `complete-ragged` does.
- Every table and tier is read, and every count computed, before anything is written: a
  release with no Top-Hit Index in a component it counts, a tier whose offsets are
  malformed or count other than its table's Analyses, a row whose `analysis_index` is not
  its position, or a table without the three columns is refused whole.
- Only a table whose counts change is written, as `clump` writes its counts
  (`model.analyses.write_analysis_columns`: every other column and value as read, the
  file replaced atomically), and its `overview.html` regenerated as
  `regenerate-overview` does. A release whose counts are right is left untouched.
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
from opengwasdb.top_hits import has_top_hit_index, misshapen_tier, summed_hit_counts


class RecountRefusal(Exception):
    """A release `recount_top_hits` cannot recount; nothing was written."""


@dataclass(frozen=True)
class RecountedTable:
    """One `analyses.tsv` and what recounting it changes: the new columns, and per
    column how many Analyses' counts change."""

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
    """What one run did, for the CLI: each table recounted, in the order written."""

    store: str
    seconds: float
    tables: list[RecountedTable] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "store": self.store,
            "seconds": round(self.seconds, 3),
            "tables": [table.as_dict() for table in self.tables],
            "written": [str(t.path) for t in self.tables if t.n_analyses_changed],
        }


def recount_top_hits(store_path: str | Path) -> RecountRun:
    """Recount the Top-Hit Counts of the release at `store_path` in place (module
    docstring). Every refusal comes before anything is written."""
    started = time.perf_counter()
    try:
        release = open_store(store_path)
    except UnsupportedFormatVersion as exc:
        raise RecountRefusal(str(exc)) from exc
    planned = [_recounted(table, counted) for table, counted in _tables(release)]
    for table in planned:
        if table.n_analyses_changed:
            _write(table)
    return RecountRun(str(release.path), time.perf_counter() - started, planned)


def _tables(release: OpenGWASDBStore) -> list[tuple[Path, list[Path]]]:
    """Each `analyses.tsv` the release carries, in the order written, with the releases
    whose tiers it counts: a Hybrid's Dense Component's own table first, where it carries
    one, then the root's, which counts both components."""
    if release.manifest.primary_layout is not PrimaryStorageLayout.HYBRID:
        return [(release.analyses_path, [release.path])]
    dense = release.dense_component()
    tables = []
    if dense.analyses_path != release.analyses_path:
        tables.append((dense.analyses_path, [dense.path]))
    tables.append((release.analyses_path, [release.path, dense.path]))
    return tables


def _recounted(table_path: Path, counted: list[Path]) -> RecountedTable:
    """`table_path`'s Top-Hit Counts recounted from the tiers of `counted`, refused where
    they cannot be (module docstring)."""
    table = _recountable_table(table_path)
    columns = {
        column: [str(n) for n in counts]
        for column, counts in _tier_counts(table_path, counted, len(table.rows)).items()
    }
    differs = _differing(table, columns)
    return RecountedTable(
        path=table_path,
        n_analyses=len(table.rows),
        columns=columns,
        changed={c: sum(c in d for d in differs) for c in TOP_HIT_COUNT_COLUMNS},
        n_analyses_changed=sum(1 for d in differs if d),
    )


def _differing(table: AnalysesTable, columns: dict[str, list[str]]) -> list[set[str]]:
    """Per row, the Top-Hit Count columns whose recounted value differs from the table's."""
    return [
        {c for c in TOP_HIT_COUNT_COLUMNS if row.get(c, "") != columns[c][i]}
        for i, row in enumerate(table.rows)
    ]


def _recountable_table(table_path: Path) -> AnalysesTable:
    """The table at `table_path`, refused unless it holds the three columns and each row
    is at its own `analysis_index`, the order a tier counts in."""
    if not table_path.is_file():
        raise RecountRefusal(f"{table_path} does not exist")
    table = read_analyses(table_path)
    missing = [c for c in TOP_HIT_COUNT_COLUMNS if c not in table.fieldnames]
    if missing:
        raise RecountRefusal(f"{table_path} has no column {', '.join(missing)} to recount")
    for position, row in enumerate(table.rows):
        if row.get("analysis_index") != str(position):
            raise RecountRefusal(
                f"{table_path}: row {position} has analysis_index "
                f"{row.get('analysis_index')!r}; a tier counts Analyses in analysis_index "
                "order, so each row must be at its own"
            )
    return table


def _tier_counts(table_path: Path, counted: list[Path], n_analyses: int) -> dict[str, list[int]]:
    """The tiers' counts for `table_path`, summed over `counted`; refused where a
    component has no Top-Hit Index or a tier cannot be counted."""
    for path in counted:
        if not has_top_hit_index(path):
            raise RecountRefusal(f"{path} has no Top-Hit Index to count {table_path} from")
        problem = misshapen_tier(path, n_analyses)
        if problem:
            raise RecountRefusal(f"{problem}: {table_path} cannot be recounted from it")
    try:
        return summed_hit_counts(counted, n_analyses)
    except (KeyError, ValueError) as exc:
        raise RecountRefusal(f"{table_path} cannot be recounted: {exc}") from exc


def _write(table: RecountedTable) -> None:
    """The recounted columns, then the overview beside the table, as `clump` writes its
    counts and `regenerate-overview` its overview."""
    from opengwasdb.overview import write_overview_html

    write_analysis_columns(table.path, table.columns)
    if (table.path.parent / "overview.html").exists():
        write_overview_html(table.path.parent, read_analyses(table.path))


__all__ = ["RecountRefusal", "RecountRun", "RecountedTable", "recount_top_hits"]
