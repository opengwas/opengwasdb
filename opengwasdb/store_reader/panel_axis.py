"""A Hybrid Dense Component's Variant Axis: the shared axis seen through `dense_to_shared`.

From format 0.3.0 a Hybrid's nested Dense Component carries no variant identity of
its own (#472, ADR 0102 §4). Its rows are still panel-local, 0..n_panel-1, the rows
of its grid, and `dense_to_shared.npy` names the shared row of each. `PanelVariantAxis`
answers every question a `VariantAxis` answers, in panel rows, by asking the shared
axis and translating:

- **panel row to shared row** is an index into the map;
- **shared row to panel row** is `hybrid.panel_rows`, the one search of the map, which
  the Hybrid query facade asks too. A shared row off the panel is no panel row, and is
  left out of every answer, as the Dense Component's own axis never held it.

So an identifier is resolved once, against the shared sidecars: there is no second
ALID, position or rsID index to disagree with the first (#481). A record's
`variant_index` is its panel row; every other column is the shared row's.

It is a `VariantAxis` to its callers, and overrides every method one answers. Its
`alid_index` is the shared ALID index seen the same way (`PanelAlidIndex`), for the
readers that search ALIDs without the axis. The component has no other index of its
own, so asking it for one (`rsid_index`, `position_sidecar`), for an index's
`problems`, or for a tabix read of a table it does not carry is refused by name
rather than answered from the shared axis's rows.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import replace
from typing import Any, ClassVar

import numpy as np

from opengwasdb.store_reader.alid import ParsedAlid
from opengwasdb.store_reader.alid_index import ABSENT, AlidIndex
from opengwasdb.store_reader.hybrid import panel_rows
from opengwasdb.store_reader.variant_axis import (
    VARIANTS_TABLE_COLUMNS,
    VariantAxis,
    VariantRecord,
)


class NoComponentIndex(TypeError):
    """A 0.3.0 Dense Component was asked for an index it does not carry (#472)."""


class PanelAlidIndex:
    """The shared ALID index of a 0.3.0 Hybrid, in its Dense Component's panel rows: an
    `AlidIndex` whose rows are the panel's. An ALID of a shared row off the panel is
    `ABSENT` here, as the component holds no such row."""

    FILENAMES: ClassVar[tuple[str, ...]] = ()

    def __init__(self, shared: AlidIndex, dense_to_shared: np.ndarray, n_shared: int) -> None:
        self._shared = shared
        self._dense_to_shared = dense_to_shared
        self._n_shared = n_shared

    def search(self, alids: Sequence[str]) -> np.ndarray:
        rows = np.asarray(self._shared.search(alids), dtype=np.int64)
        held = rows >= 0
        on_panel, panel = panel_rows(self._dense_to_shared, rows[held])
        rows[np.flatnonzero(held)] = np.where(on_panel, panel, ABSENT)
        return rows

    def alids(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        inside = (rows >= 0) & (rows < self._dense_to_shared.size)
        out = np.full(rows.shape, None, dtype=object)
        if inside.any():
            shared = self._dense_to_shared[rows[inside]].astype(np.int64)
            out[inside] = self._shared.alids(shared)
        return out

    def unindexed_rows(self, n_rows: int) -> np.ndarray:
        shared = self._shared.unindexed_rows(self._n_shared)
        on_panel, panel = panel_rows(self._dense_to_shared, shared)
        return panel[on_panel]

    def problems(self, table_rows: int, expected_indexed: int) -> Iterator[str]:
        raise NoComponentIndex(
            "a Hybrid's Dense Component carries no ALID index of its own from format 0.3.0 "
            "(#472): the shared one is judged at the release's root"
        )


class PanelVariantAxis(VariantAxis):
    """The shared `VariantAxis` of a 0.3.0 Hybrid, in its Dense Component's panel rows."""

    def __init__(self, shared: VariantAxis, dense_to_shared: np.ndarray) -> None:
        # Deliberately not `VariantAxis.__init__`: this axis opens no index of its own.
        self._shared = shared
        self._dense_to_shared = np.asarray(dense_to_shared)
        self.store_path = shared.store_path

    # ── the translation ──────────────────────────────────────────────────────
    @property
    def shared(self) -> VariantAxis:
        """The shared axis this one reads."""
        return self._shared

    @property
    def dense_to_shared(self) -> np.ndarray:
        """Each panel row's shared row (`dense_to_shared.npy`)."""
        return self._dense_to_shared

    def _to_shared(self, panel: np.ndarray) -> np.ndarray:
        """The shared row of each panel row; -1 for one outside the panel."""
        rows = np.asarray(panel, dtype=np.int64).reshape(-1)
        inside = (rows >= 0) & (rows < self.n_variants)
        out = np.full(rows.shape, -1, dtype=np.int64)
        out[inside] = self._dense_to_shared[rows[inside]]
        return out

    def _to_panel(self, shared: np.ndarray) -> np.ndarray:
        """The panel row of each shared row, in order, the shared rows off the panel left out."""
        shared = np.asarray(shared, dtype=np.int64).reshape(-1)
        on_panel, rows = panel_rows(self._dense_to_shared, shared)
        panel: np.ndarray = rows[on_panel].astype("int32")
        return panel

    def _record(self, record: VariantRecord | None) -> VariantRecord | None:
        """A shared record renumbered to its panel row; None for one off the panel."""
        if record is None:
            return None
        rows = self._to_panel(np.array([record.variant_index]))
        return replace(record, variant_index=int(rows[0])) if len(rows) else None

    # ── what a VariantAxis answers ───────────────────────────────────────────
    @property
    def n_variants(self) -> int:
        """The panel's rows: one per entry of `dense_to_shared.npy`."""
        return int(self._dense_to_shared.size)

    @property
    def reads_table(self) -> bool:
        """Never: a 0.3.0 axis reads its sidecars (#469)."""
        return False

    def _refused(self, what: str) -> NoComponentIndex:
        return NoComponentIndex(
            f"a Hybrid's Dense Component carries no {what} of its own from format 0.3.0 "
            f"(#472): its rows are named by the shared axis at {self._shared.store_path}"
        )

    @property
    def alid_index(self) -> PanelAlidIndex:
        """The shared ALID index in panel rows (`PanelAlidIndex`)."""
        return PanelAlidIndex(
            self._shared.alid_index, self._dense_to_shared, self._shared.n_variants
        )

    @property
    def rsid_index(self) -> Any:
        raise self._refused("rsID index")

    @property
    def position_sidecar(self) -> Any:
        raise self._refused("position sidecar")

    def tabix_range(self, chromosome: str, start: int, end: int) -> list[VariantRecord]:
        raise self._refused("Variant Table or tabix index")

    def close(self) -> None:
        self._shared.close()

    def index_of(self, identifier: str) -> int | None:
        rows = self.indices_by_identifiers([identifier])
        return int(rows.min()) if len(rows) else None

    def indices_by_identifiers(self, identifiers: Sequence[str]) -> np.ndarray:
        """The panel rows the identifiers name, in the shared axis's order
        (`VariantAxis.indices_by_identifiers`), those off the panel left out."""
        return self._to_panel(self._shared.indices_by_identifiers(identifiers))

    def indices_by_alias(self, alias: str) -> np.ndarray:
        return self._to_panel(self._shared.indices_by_alias(alias))

    def range_indices(self, chromosome: str, start: int, end: int) -> np.ndarray:
        return self._to_panel(self._shared.range_indices(chromosome, start, end))

    def identity_by_indices(self, variant_index: np.ndarray) -> dict[str, np.ndarray]:
        shared = self._to_shared(variant_index)
        if (shared < 0).any():
            outside = int(np.asarray(variant_index).reshape(-1)[shared < 0][0])
            raise ValueError(f"no variant row at index {outside}")
        return self._shared.identity_by_indices(shared)

    def rsids_by_indices(self, variant_index: np.ndarray) -> np.ndarray:
        """Each panel row's rsID; None for one that names none or lies outside the panel."""
        shared = self._to_shared(variant_index)
        out = np.full(shared.size, None, dtype=object)
        inside = shared >= 0
        if inside.any():
            out[inside] = self._shared.rsids_by_indices(shared[inside])
        return out

    def variants_table(self) -> dict[int, dict[str, Any]]:
        rows = np.arange(self.n_variants, dtype=np.int64)
        shared = self._dense_to_shared.astype(np.int64)
        columns = {
            name: values.tolist()
            for name, values in self._shared.identity_by_indices(shared).items()
        }
        columns["rsid"] = self._shared.rsids_by_indices(shared).tolist()
        return {
            row: {name: columns[name][row] for name in VARIANTS_TABLE_COLUMNS}
            for row in rows.tolist()
        }

    # ── the record reads: the shared table's rows (builders, completion, validator) ──
    def by_index(self, variant_index: int) -> VariantRecord | None:
        [shared] = self._to_shared(np.array([variant_index])).tolist()
        return None if shared < 0 else self._record(self._shared.by_index(int(shared)))

    def by_indices(self, indices: Iterable[int]) -> dict[int, VariantRecord]:
        panel = np.array(sorted({int(i) for i in indices}), dtype=np.int64)
        shared = self._to_shared(panel)
        found = self._shared.by_indices(shared[shared >= 0].tolist())
        out: dict[int, VariantRecord] = {}
        for row, shared_row in zip(panel.tolist(), shared.tolist(), strict=True):
            record = found.get(int(shared_row)) if shared_row >= 0 else None
            if record is not None:
                out[row] = replace(record, variant_index=row)
        return out

    def by_identifier(self, identifier: str) -> VariantRecord | None:
        index = self.index_of(identifier)
        return None if index is None else self.by_index(index)

    def by_alid(self, parsed: ParsedAlid) -> VariantRecord | None:
        return self._record(self._shared.by_alid(parsed))

    def records(self) -> Iterator[VariantRecord]:
        """Every panel row, streamed in panel order, from one pass of the shared table."""
        wanted = iter(enumerate(self._dense_to_shared.tolist()))
        nxt = next(wanted, None)
        for record in self._shared.records():
            if nxt is None:
                return
            row, shared_row = nxt
            if record.variant_index == shared_row:
                yield replace(record, variant_index=row)
                nxt = next(wanted, None)
        if nxt is not None:
            raise ValueError(
                f"dense_to_shared maps panel row {nxt[0]} to shared row {nxt[1]}, which the "
                "shared Variant Table does not hold in order"
            )

    def all(self) -> list[VariantRecord]:
        return list(self.records())

    def range(self, chromosome: str, start: int, end: int) -> list[VariantRecord]:
        return [
            record
            for record in map(self._record, self._shared.range(chromosome, start, end))
            if record is not None
        ]

    def indices_by_source_alid(self, wanted: Iterable[str]) -> dict[str, int] | None:
        found = self._shared.indices_by_source_alid(wanted)
        if found is None:
            return None
        names = list(found)
        rows = np.array([found[name] for name in names], dtype=np.int64)
        on_panel, panel = panel_rows(self._dense_to_shared, rows)
        return {
            name: int(row)
            for name, hit, row in zip(names, on_panel.tolist(), panel.tolist(), strict=True)
            if hit
        }
