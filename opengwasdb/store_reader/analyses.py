"""`analyses.tsv` as a reader reads it, and the in-memory index the facade keeps of it.

`analyses.tsv` is the sole source of truth for Analytical Metadata (ADR 0030,
issue #22) -- there is no SQLite `analyses` table to query anymore. Both of the
facade's access patterns (a single indexed point lookup by `analysis_id`, and a
full-table scan) translate directly to reading the file into a dict once at
store-open (`AnalysesIndex`). Reading the table is `read_analyses`, with the
standard library's csv (#464); its contract, writing and validation are
`opengwasdb.model.analyses`'.
"""

from __future__ import annotations

import csv
from collections.abc import ItemsView, Mapping
from dataclasses import dataclass
from pathlib import Path


class UnknownAnalysisError(ValueError):
    """A query named an `analysis_id` the Store Release does not hold.

    An empty answer would read as "no association"; a name nothing holds is a
    mistake the caller must see (#483).
    """


@dataclass(frozen=True)
class AnalysesTable:
    """A parsed `analyses.tsv`: column order plus raw string rows.

    Rows are kept as raw strings, like `opengwasdb.ancestry.catalogue`'s
    Catalogue reader -- `analyses.tsv` is inherently a text format, and typed
    access belongs to callers that know which columns they need, not to this
    module.
    """

    fieldnames: tuple[str, ...]
    rows: tuple[dict[str, str], ...]


def read_analyses(path: str | Path) -> AnalysesTable:
    """Read an `analyses.tsv` file, preserving column order and every field.

    Columns beyond the known contract (registry-only build inputs, or
    anything else) are read without error and carried through in each row's
    dict unchanged -- the superset property a manifest and a built store's
    analyses.tsv are related by.
    """
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = tuple(reader.fieldnames or ())
        rows = tuple(dict(row) for row in reader)
    return AnalysesTable(fieldnames=fieldnames, rows=rows)


def analysis_label(row: Mapping[str, str]) -> str:
    """The best-available human label for one `analyses.tsv` row.

    `analysis_label`, or the `analysis_id` when the release supplies no label --
    a store built without a label for an Analysis (and every release predating
    ADR 0034, whose table has no `analysis_label` column at all) still has to be
    browsable and sortable under something a person can read.

    Deliberately the *only* implementation of that fallback: `overview.html`'s
    Rho tab and anything else that wants a display label call this rather than
    spelling the chain out again (issue #57). It is a presentation helper, so it
    returns a label rather than writing one back into the row: a blank
    `analysis_label` in a store's `analyses.tsv` is an honest "no label was
    supplied", and `analyses_table()` hands that column through unchanged.
    """
    return row.get("analysis_label") or row.get("analysis_id", "")


class AnalysesIndex:
    """`analyses.tsv` read once, keyed by `analysis_id` and by `analysis_index`.

    Built from a release's directory, or from `table`, the `analyses.tsv` a release
    reads when it is not its own (a 0.3.0 Hybrid Dense Component's is its Hybrid's,
    `StoreRelease.analyses_path`, #472).
    """

    def __init__(self, store_path: str | Path, *, table: str | Path | None = None):
        path = Path(table) if table is not None else Path(store_path) / "analyses.tsv"
        rows = read_analyses(path).rows
        self._by_id: dict[str, dict[str, str]] = {}
        self._by_index: dict[int, dict[str, str]] = {}
        for row in rows:
            self._by_id[row["analysis_id"]] = row
            self._by_index[int(row["analysis_index"])] = row

    def by_id(self, analysis_id: str) -> dict[str, str] | None:
        return self._by_id.get(analysis_id)

    def require_index(self, analysis_id: str) -> int:
        """The Store-local Analysis index of `analysis_id`; raises `UnknownAnalysisError`."""
        row = self._by_id.get(analysis_id)
        if row is None:
            raise UnknownAnalysisError(
                f"unknown analysis_id {analysis_id!r}: not in this Store Release"
            )
        return int(row["analysis_index"])

    def all(self) -> dict[int, dict[str, str]]:
        return dict(self._by_index)

    def items(self) -> ItemsView[int, dict[str, str]]:
        """A live view over (analysis_index, row) pairs for internal iteration
        -- unlike `all()`, this doesn't copy, so it's for callers that only
        read (a caller wanting an owned dict to mutate should use `all()`)."""
        return self._by_index.items()
