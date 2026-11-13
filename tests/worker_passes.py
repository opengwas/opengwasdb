"""Validation at one worker and at several, on clean and corrupted releases (#315).

Shared by the tests of `validate --n-workers`, which are split across files so
pytest's workers run them side by side: the releases' corruptions, the cases each
is run as, and a recorder of which band each forked worker read and when.

Every case's release is narrowed until its grid spans more bands than two a worker,
and says so before anything is asserted about it: a grid read in one or two bands
would pass at any worker count whatever the reduction did. Corruptions sit in bands
the parent does not read, so a worker meets them.
"""

from __future__ import annotations

import functools
import multiprocessing
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from shutil import copytree

import numpy as np
import pytest
import zarr
from eaf_tables import EXCEPTION_EVERY, SMALL_VARIANTS, rewrite_primary_table
from pool_budget import FIXTURE_UNIT_BYTES

from opengwasdb.encoding import banding
from opengwasdb.encoding.codec import (
    EAF_EXCEPTION,
    EAF_EXCEPTION_INDEX,
    EAF_EXCEPTION_VALUE,
    Z_OVERFLOW,
)
from opengwasdb.encoding.plan import Z_MISSING
from opengwasdb.layouts.dense import indexed_subsets
from opengwasdb.store import open_store
from opengwasdb.validation import run, scans, validate_store
from opengwasdb.validation.rule import Kind, ValidationOptions
from opengwasdb.validation.validate import release_targets

#: The pool every equality is taken at: the parent's band, then two or more a worker.
WORKERS = 3
#: The fewest bands a fixture's grid may span.
MIN_BANDS = 2 * WORKERS + 1
#: The subset every fixture publishes.
SUBSET = "hm3"


def _grid_path(store: Path) -> Path:
    """Where the release's Dense planes are: a Hybrid's Dense Component's."""
    nested = store / "dense" / "data.zarr"
    return nested if nested.exists() else store / "data.zarr"


def _grid(store: Path) -> zarr.Group:
    return zarr.open_group(str(_grid_path(store)), mode="r+")


def _index(store: Path) -> zarr.Group:
    return zarr.open_group(str(_grid_path(store) / "indexed_subsets" / SUBSET), mode="r+")


def _narrow(monkeypatch: pytest.MonkeyPatch, store: Path, rows: int) -> None:
    """Bands of `rows` rows of the release's grid."""
    width = int(_grid(store)["z"].shape[1])
    monkeypatch.setattr(banding, "DECODED_BAND_BYTES", banding.BYTES_PER_CELL * width * rows)


def _bands(store: Path) -> list[int]:
    """How many bands each Dense grid of the release is read in."""
    targets = release_targets(open_store(store), ValidationOptions())
    try:
        return [scans._association_blocks(t) for t in targets if t.kind is Kind.GRID]
    finally:
        for target in targets:
            target.close()


def _report(store: Path, n_workers: int) -> list[tuple[str, str, str, str, tuple[str, ...]]]:
    findings = validate_store(store, n_workers=n_workers).findings
    return [(f.rule, f.severity.value, f.target, f.message, f.reasons) for f in findings]


def _failed(report: list[tuple[str, str, str, str, tuple[str, ...]]]) -> set[str]:
    return {rule for rule, severity, *_rest in report if severity == "error"}


# ── Corruptions, each in a band the parent does not read ─────────────────────


def _step(array: zarr.Array, cell: tuple[int, ...]) -> None:
    """One code step away from what the array holds there, in its own dtype."""
    values = np.asarray(array[:])
    if np.issubdtype(values.dtype, np.floating):
        values[cell] = values[cell] + np.float32(0.01)
    else:
        value = int(values[cell])
        values[cell] = value + 1 if value < 100 else value - 1
    array[:] = values


def _slot(store: Path, row: int) -> int:
    """The subset slot holding primary row `row`."""
    rows = np.asarray(_index(store)["variant_index"][:], dtype=np.int64)
    slot = int(np.searchsorted(rows, row))
    assert rows[slot] == row, f"row {row} must be in the subset"
    return slot


def subset_cell(plane: str, row: int, analysis: int) -> Callable[[Path], None]:
    def corrupt(store: Path) -> None:
        _step(_index(store)[plane], (analysis, _slot(store, row)))

    return corrupt


def primary_cell(plane: str, row: int, analysis: int) -> Callable[[Path], None]:
    def corrupt(store: Path) -> None:
        _step(_grid(store)[plane], (row, analysis))

    return corrupt


def unpaired(row: int, analysis: int) -> Callable[[Path], None]:
    """Z missing where SE is not."""

    def corrupt(store: Path) -> None:
        z = _grid(store)["z"]
        values = np.asarray(z[:])
        values[row, analysis] = Z_MISSING if values.dtype == np.int16 else np.nan
        z[:] = values

    return corrupt


def _table(store: Path) -> tuple[np.ndarray, np.ndarray]:
    root = _grid(store)
    return (
        np.asarray(root[EAF_EXCEPTION_INDEX][:], dtype=np.int64),
        np.asarray(root[EAF_EXCEPTION_VALUE][:], dtype=np.float32),
    )


def table_entry_dropped(position: int) -> Callable[[Path], None]:
    """The primary table loses the entry for one exception cell."""

    def corrupt(store: Path) -> None:
        positions, values = _table(store)
        keep = positions != position
        assert not keep.all(), f"the table must describe position {position}"
        rewrite_primary_table(store, positions[keep], values[keep])

    return corrupt


def table_entry_added(position: int) -> Callable[[Path], None]:
    """The primary table describes a cell its plane does not mark."""

    def corrupt(store: Path) -> None:
        positions, values = _table(store)
        assert position not in positions
        at = int(np.searchsorted(positions, position))
        rewrite_primary_table(
            store, np.insert(positions, at, position), np.insert(values, at, np.float32(0.5))
        )

    return corrupt


def index_table_value(entry: int) -> Callable[[Path], None]:
    def corrupt(store: Path) -> None:
        _step(_index(store)[EAF_EXCEPTION_VALUE], (entry,))

    return corrupt


def tier_z(tier: str, row: int) -> Callable[[Path], None]:
    """One Top-Hit tier cell's Z, in a row past the parent's band, made wrong."""

    def corrupt(store: Path) -> None:
        group = _grid(store)["top_hits"][tier]
        rows = np.asarray(group["variant_index"][:], dtype=np.int64)
        held = np.flatnonzero(rows >= row)
        assert len(held), f"tier {tier} must hold a cell at row {row} or later"
        _step(group["z"], (int(held[0]),))

    return corrupt


#: Takes a table's positions and values and returns the two it is rewritten with.
TableChange = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def table_rewritten(change: TableChange) -> Callable[[Path], None]:
    """The primary table rewritten by `change`, which takes and returns its two arrays."""

    def corrupt(store: Path) -> None:
        rewrite_primary_table(store, *change(*_table(store)))

    return corrupt


def _swapped(at: int) -> TableChange:
    """Two neighbouring positions out of order (#314's disordered table)."""

    def change(positions: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        positions = positions.copy()
        positions[at], positions[at + 1] = positions[at + 1], positions[at]
        return positions, values

    return change


def _last_repeated(positions: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The last entry twice (#387's table that is not unique)."""
    return np.concatenate((positions, positions[-1:])), np.concatenate((values, values[-1:]))


def _past_the_grid(positions: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """An entry for a cell past the grid's last (#387's past-last-span entry)."""
    beyond = SMALL_VARIANTS * 2 + 5
    return np.concatenate((positions, [beyond])), np.concatenate((values, np.float32([0.5])))


def swapped_cells(
    plane: str, one: tuple[int, int], other: tuple[int, int]
) -> Callable[[Path], None]:
    """Two cells of the index exchanged: every value still there, two in each other's
    cell (#314)."""

    def corrupt(store: Path) -> None:
        codes = np.asarray(_index(store)[plane][:])
        assert codes[one] != codes[other], "the swap must move two different values"
        codes[one], codes[other] = codes[other], codes[one]
        _index(store)[plane][:] = codes

    return corrupt


def index_undecodable(row: int, analysis: int) -> Callable[[Path], None]:
    """An index Z cell marked overflow with no overflow entry: the index cannot be
    decoded there (#314)."""

    def corrupt(store: Path) -> None:
        z = _index(store)["z"]
        codes = np.asarray(z[:])
        cell = (analysis, _slot(store, row))
        assert codes[cell] not in (Z_OVERFLOW, Z_MISSING), "the cell must be ordinary"
        codes[cell] = Z_OVERFLOW
        z[:] = codes

    return corrupt


def eaf_cells_undecodable(*cells: tuple[int, int]) -> Callable[[Path], None]:
    """Each cell's frequency coded as an exception the table holds no entry for, so its
    band's EAF -- and the SE decoded against it -- cannot be decoded, in words naming
    that band's cell."""

    def corrupt(store: Path) -> None:
        eaf = _grid(store)["eaf"]
        codes = np.asarray(eaf[:])
        width = codes.shape[1]
        positions, values = _table(store)
        for row, analysis in cells:
            codes[row, analysis] = EAF_EXCEPTION
        eaf[:] = codes
        keep = ~np.isin(positions, [row * width + analysis for row, analysis in cells])
        rewrite_primary_table(store, positions[keep], values[keep])

    return corrupt


def _clean(_store: Path) -> None:
    return None


@dataclass(frozen=True)
class Case:
    """A seed, what is done to a copy of it, its band rows, and the rules it must fail."""

    seed: str
    corrupt: Callable[[Path], None]
    rows: int
    fails: frozenset[str] = frozenset()
    table_page: int | None = None
    #: Index slots per comparison band and cells per unit, narrowed as #314's tests do.
    units: int | None = None
    #: A band whose worker is held until another band has been read, so the two
    #: finish out of band order.
    slow: tuple[int, int] | None = None


#: Rows a band of each seed holds, and so how many bands: 16, 10, 10, 12 and 10.
RICH, SMALL, COMPLETED, VARIED, HYBRID = 20, 400, 20, 10, 4
_SUBSET_FAILS = frozenset({"subsets.indexed"})
#: A band whose EAF, so its SE, cannot be decoded: the plane and its table disagree,
#: and the subset's table is no longer the release's remapped.
_UNDECODABLE = frozenset(
    {"associations.decodable", "side_tables.eaf_exceptions", "subsets.indexed"}
)
#: A table entry with no marked cell, or a marked cell turned ordinary.
_UNMATCHED = frozenset({"side_tables.eaf_exceptions", "subsets.indexed"})
#: A table the plan rule finds out of order, which every value rule stands behind.
_DISORDERED = frozenset({"encoding.eaf", "subsets.indexed"})
#: An exception cell of the small seed in its eighth band: every twentieth variant's,
#: in its second Analysis.
_SMALL_EXCEPTION = 3_000 * 2 + 1
assert 3_000 % EXCEPTION_EVERY == 0

CASES: dict[str, Case] = {
    "rich-clean": Case("rich_seed", _clean, RICH),
    "rich-subset-z": Case("rich_seed", subset_cell("z", 200, 1), RICH, _SUBSET_FAILS),
    "rich-subset-se": Case("rich_seed", subset_cell("se", 200, 2), RICH, _SUBSET_FAILS),
    "rich-subset-eaf": Case("rich_seed", subset_cell("eaf", 200, 3), RICH, _SUBSET_FAILS),
    "rich-eaf-entry-dropped": Case(
        "rich_seed", table_entry_dropped(20 * 4 + 1), RICH, _UNDECODABLE,
    ),
    "rich-unpaired": Case(
        "rich_seed", unpaired(250, 2), RICH, frozenset({"associations.markers_paired"})
    ),
    "small-clean": Case("small_seed", _clean, SMALL, table_page=64),
    "small-entry-dropped": Case(
        "small_seed", table_entry_dropped(_SMALL_EXCEPTION), SMALL, _UNDECODABLE,
        table_page=64,
    ),
    "small-entry-stray": Case(
        "small_seed", table_entry_added(2_501 * 2), SMALL, _UNMATCHED, table_page=64,
    ),
    "small-index-table-value": Case(
        "small_seed", index_table_value(-1), SMALL, _SUBSET_FAILS, table_page=64
    ),
    "small-primary-eaf": Case(
        "small_seed", primary_cell("eaf", 3_500, 0), SMALL, _UNMATCHED, table_page=64
    ),
    "completed-clean": Case("completed_seed", _clean, COMPLETED),
    "completed-subset-imputed": Case(
        "completed_seed", subset_cell("imputed", 150, 1), COMPLETED, _SUBSET_FAILS
    ),
    "varied-clean": Case("varied_seed", _clean, VARIED),
    "varied-top-hit-z": Case(
        "varied_seed", tier_z("p_5e_04", 100), VARIED, frozenset({"top_hits.z"})
    ),
    "small-table-disordered": Case(
        "small_seed", table_rewritten(_swapped(63)), SMALL, _DISORDERED, table_page=64,
    ),
    "small-table-repeated": Case(
        "small_seed", table_rewritten(_last_repeated), SMALL, _DISORDERED, table_page=64,
    ),
    "small-past-last-span": Case(
        "small_seed", table_rewritten(_past_the_grid), SMALL,
        frozenset({"side_tables.eaf_exceptions"}), table_page=64,
    ),
    "varied-swapped-cells": Case(
        "varied_seed", swapped_cells("z", (0, 41), (0, 45)), VARIED, _SUBSET_FAILS, units=20
    ),
    "completed-index-undecodable": Case(
        "completed_seed", index_undecodable(150, 1), COMPLETED, _SUBSET_FAILS, units=20
    ),
    # Two bands whose SE cannot be decoded, each for its own cell; the earlier one
    # finishes last. Added in the order the workers finished, the report would name
    # the later band's cell.
    "rich-two-undecodable-bands": Case(
        "rich_seed", eaf_cells_undecodable((20, 1), (110, 2)), RICH, _UNDECODABLE,
        slow=(1, 5),
    ),
    "hybrid-subset-eaf": Case("hybrid_seed", subset_cell("eaf", 30, 1), HYBRID, _SUBSET_FAILS),
}


class _Finished:
    """Which band each worker read, in the order they finished, across the fork pool.

    It records one pass: the driver's read of the first band starts it again.
    """

    def __init__(self, n_bands: int) -> None:
        context = multiprocessing.get_context("fork")
        self.count = context.Value("i", 0)
        self.order = context.Array("i", n_bands)
        self.pids = context.Array("i", n_bands)
        self.events = [context.Event() for _ in range(n_bands)]

    def start(self) -> None:
        """Forget the last pass: its order, who read what, and which bands were read."""
        with self.count.get_lock():
            self.count.value = 0
        for band, event in enumerate(self.events):
            self.pids[band] = 0
            event.clear()

    def finished(self, band: int) -> None:
        with self.count.get_lock():
            self.order[self.count.value] = band
            self.count.value += 1
        self.pids[band] = os.getpid()
        self.events[band].set()

    def wait_for(self, band: int) -> None:
        assert self.events[band].wait(timeout=60), f"band {band} was never read"


def _reads_bands(
    monkeypatch: pytest.MonkeyPatch,
    rows: int,
    finished: _Finished,
    *,
    slow: int | None = None,
    after: int = 0,
    unreadable: frozenset[int] = frozenset(),
) -> None:
    """Every band records who finished it, and when: once its folds and riders have
    measured it, so its outcome is ready to go back. Band `slow`, in a worker, is
    read only once band `after` has finished, and each band in `unreadable` fails
    to read with its own words, finishing there."""
    read = scans.GridSplit.read
    read = getattr(read, "__wrapped__", read)
    measure = getattr(run._measured, "__wrapped__", run._measured)
    driver = os.getpid()

    @functools.wraps(read)
    def recorded(self: scans.GridSplit, unit: scans.GridBand) -> scans.Block:
        band = unit.r0 // rows
        if band == 0:  # the driver's first read: a pass begins
            finished.start()
        if band == slow and os.getpid() != driver:
            finished.wait_for(after)
        if band in unreadable:
            finished.finished(band)
            raise OSError(f"band {band}'s chunks cannot be read")
        return read(self, unit)

    @functools.wraps(measure)
    def measured(*args: object) -> object:
        outcome = measure(*args)
        block = args[-1]
        if isinstance(block, scans.Block) and block.analysis is None:  # a Dense band
            finished.finished(block.offset // block.n_analyses // rows)
        return outcome

    monkeypatch.setattr(scans.GridSplit, "read", recorded)
    monkeypatch.setattr(run, "_measured", measured)


def worker_floor_pinned() -> object:
    """An autouse fixture holding a worker's floor at a fixture's size, so the host's
    memory never stands between a test and the workers it counts on (`pool_budget`)."""

    @pytest.fixture(autouse=True)
    def _fixture_sized_workers(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(scans, "WORKER_FLOOR_BYTES", FIXTURE_UNIT_BYTES)

    return _fixture_sized_workers


def cases_of(*seeds: str) -> object:
    """A `case` fixture over the cases built on `seeds`: a copy of the case's seed,
    corrupted, and narrowed into its bands."""

    @pytest.fixture(params=sorted(name for name, case in CASES.items() if case.seed in seeds))
    def case(
        request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[Path, Case]:
        chosen = CASES[request.param]
        store = copytree(request.getfixturevalue(chosen.seed), tmp_path / "store.opengwasdb")
        chosen.corrupt(store)
        _narrow(monkeypatch, store, chosen.rows)
        if chosen.table_page is not None:
            monkeypatch.setattr(scans, "SPAN_PAGE", chosen.table_page)
        if chosen.units is not None:
            units = chosen.units
            monkeypatch.setattr(indexed_subsets, "_physical_write_unit", lambda _array: units)
            monkeypatch.setattr(indexed_subsets, "EXHAUSTIVE_VALIDATION_BAND_CELLS", units)
        if chosen.slow is not None:
            slow, after = chosen.slow
            finished = _Finished(max(_bands(store)))
            _reads_bands(monkeypatch, chosen.rows, finished, slow=slow, after=after)
        return store, chosen

    return case


def assert_the_same_report(store: Path, chosen: Case) -> None:
    """The case's release spans its bands, fails what it must, and reports the same at
    one worker and at `WORKERS`."""
    bands = _bands(store)
    assert bands and min(bands) >= MIN_BANDS, f"each grid must span {MIN_BANDS} bands: {bands}"

    serial = _report(store, 1)
    spread = _report(store, WORKERS)

    assert _failed(spread) == chosen.fails, spread
    assert spread == serial
