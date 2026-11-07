"""The subset check reads the Dense EAF exception table once per validation pass (#387).

The primary release's EAF exception table is one physical file on a full-size
Store Release (a Zarr v3 shard: 237,027,522 entries on OGS-00016, 1,136,991,041
on OGS-00011), so every band the subset comparison decoded used to read all of
it again: 1,128 times on OGS-00016's HapMap3 subset, 2,114 times on OGS-00011's,
which is 3.2 TB and 28.8 TB of the same bytes.

What the value comparison needs is not the whole table but the cells the subset
holds, keyed by the primary's flat positions. The side-table rule reads the
whole table to prove the index's copy is the primary's, remapped; the comparison
reuses that read. The first test counts the reads the way `test_validation_reads`
counts a Dense grid's; the others hold the comparison to what it reported before.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr
from pytest import MonkeyPatch, approx, fixture
from store_reads import element_reads, times_read
from test_indexed_subsets import (
    N_ANALYSES,
    _build,
    _index_group,
    _root,
    _write_standard_variant_list,
    build_rich_store,
)

from opengwasdb.encoding.codec import EAF_EXCEPTION_INDEX, EAF_EXCEPTION_VALUE
from opengwasdb.layouts.dense import indexed_subsets
from opengwasdb.validation import validate_store

NAME = "hm3"


@fixture
def indexed_store(tmp_path: Path) -> Path:
    """A published subset over a release of its own, so a corruption cannot leak.

    The builder is `test_indexed_subsets`', imported as a function rather than as
    a fixture, so this module's tests run on their own and the release is the one
    that suite publishes over.
    """
    store = build_rich_store(tmp_path / "store.opengwasdb")
    _build(store, NAME, _write_standard_variant_list(tmp_path / "subset.alid.txt"))
    return store


def _blocks(store: Path) -> int:
    """How many band-tiles the exhaustive comparison decodes, as it computes them."""
    group = _index_group(store, NAME)
    subset = np.asarray(group["variant_index"][:], dtype=np.int64)
    write_unit = indexed_subsets._physical_write_unit(group["z"])
    bands = indexed_subsets._output_write_bands(len(subset), write_unit)
    tiles = indexed_subsets._analysis_tiles(
        N_ANALYSES, write_unit, indexed_subsets.EXHAUSTIVE_VALIDATION_BAND_CELLS, 1
    )
    return len(bands) * len(tiles)


def _several_blocks(store: Path, monkeypatch: MonkeyPatch) -> int:
    """Force several blocks: the fixture is six variants wide and one band.

    Its physical write unit is the whole subset, so a cell budget below one band
    is what splits it into a tile per Analysis. A single block would make a
    per-block read of the table indistinguishable from one read per pass.
    """
    monkeypatch.setattr(indexed_subsets, "EXHAUSTIVE_VALIDATION_BAND_CELLS", 1)
    blocks = _blocks(store)
    assert blocks > 1, "the pass must decode several blocks"
    return blocks


def test_the_eaf_exception_table_is_read_once_per_pass(
    indexed_store: Path, monkeypatch: MonkeyPatch
) -> None:
    root = _root(indexed_store)
    group = _index_group(indexed_store, NAME)
    # The fixture must reach the path before anything is asserted about it: a
    # subset with no exception cell decodes without consulting any table.
    assert len(root[EAF_EXCEPTION_INDEX][:]) > 0, "the primary must carry exceptions"
    assert len(group[EAF_EXCEPTION_INDEX][:]) > 0, "the index must carry the subset's"
    _several_blocks(indexed_store, monkeypatch)

    with element_reads(monkeypatch) as counts:
        result = validate_store(indexed_store, rules={"subsets.indexed"})

    assert result.ok, result.errors
    read = times_read(counts, root[EAF_EXCEPTION_INDEX])
    assert read == 1.0, f"one pass read the primary EAF exception table {read} times"
    assert times_read(counts, root[EAF_EXCEPTION_VALUE]) == 1.0


def test_a_corrupted_eaf_cell_in_a_later_block_is_still_found(
    indexed_store: Path, monkeypatch: MonkeyPatch
) -> None:
    """A table held from the first block must not answer for a later one.

    The last Analysis is the last block the comparison decodes, and its last
    variant's EAF code is stepped by one cell, so the decoded frequency moves
    while every structural invariant stays intact.
    """
    group = _index_group(indexed_store, NAME)
    codes = np.asarray(group["eaf"][:])
    step = 1 if codes[N_ANALYSES - 1, -1] < 127 else -1
    codes[N_ANALYSES - 1, -1] = codes[N_ANALYSES - 1, -1] + step
    group["eaf"][:] = codes
    _several_blocks(indexed_store, monkeypatch)

    result = validate_store(indexed_store, rules={"subsets.indexed"})

    assert result.failed() == {"subsets.indexed"}, result.errors
    assert result.errors == [
        f"indexed subset {NAME!r} eaf values differ from the primary plane; the index "
        "is stale or corrupt"
    ], result.errors


#: The report a store of this fixture produces, rule and severity and message:
#: the ones a Store Release is judged on and the ones it is not.
_UNVERIFIED = [
    (
        "eaf_orientation.unverified",
        "warning",
        ".",
        f"analysis 'a{col}' stores EAF whose orientation is unverified: general Dense "
        "source supplied no orientation reference",
    )
    for col in range(N_ANALYSES)
]


def _report(store: Path) -> list[tuple[str, str, str, str]]:
    return [(f.rule, f.severity.value, f.target, f.message) for f in validate_store(store).findings]


def test_the_whole_report_is_the_one_the_inherited_check_produced(indexed_store: Path) -> None:
    """`validate`'s report is a contract, and this change moves none of it.

    The test pins every record of a clean release and of one whose index holds a
    corrupted EAF cell: the finding's rule, severity, target and message. The
    report is the same as the code before this change produced; what changed is
    how many times the table behind it is read, which no report can show.
    """
    assert _report(indexed_store) == _UNVERIFIED

    group = _index_group(indexed_store, NAME)
    codes = np.asarray(group["eaf"][:])
    step = 1 if codes[0, 0] < 127 else -1
    codes[0, 0] = codes[0, 0] + step
    group["eaf"][:] = codes

    assert _report(indexed_store) == [
        ("subsets.indexed", "error", "data.zarr",
         f"indexed subset {NAME!r} eaf values differ from the primary plane; the index "
         "is stale or corrupt"),
        *_UNVERIFIED,
        ("fidelity.source", "skip", "data.zarr", "subsets.indexed"),
    ]


def _table(path: Path, positions: np.ndarray, values: np.ndarray) -> zarr.Group:
    """One exception table as a Zarr group, the way a plane carries it."""
    group = zarr.open_group(str(path), mode="w")
    group.create_array(EAF_EXCEPTION_INDEX, data=positions)
    group.create_array(EAF_EXCEPTION_VALUE, data=values)
    return group


def test_a_table_read_in_several_windows_keeps_every_cell(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Every window of the table contributes its cells, and only its own.

    A six-entry primary table over a four-variant subset, read two entries at a
    time. Two of the table's cells sit at a variant the subset does not hold, one
    in the second window and one in the third, so a pass that kept every window's
    cells regardless of the subset -- or only the first window's -- describes the
    index wrongly and fails the comparison here.
    """
    n_analyses, subset = 3, np.array([1, 4, 7, 9], dtype=np.int64)
    positions = np.array([3, 11, 14, 16, 21, 28], dtype=np.int64)
    values = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06], dtype=np.float32)
    assert [int(position) // n_analyses for position in positions] == [1, 3, 4, 5, 7, 9]
    kept = [3, 14, 21, 28]
    kept_values = [0.01, 0.03, 0.05, 0.06]
    remapped = indexed_subsets._remap_side_table(
        np.asarray(kept, dtype=np.int64),
        np.asarray(kept_values, dtype=np.float32),
        subset,
        n_analyses,
    )
    root = _table(tmp_path / "root.zarr", positions, values)
    group = _table(tmp_path / "index.zarr", *remapped)
    monkeypatch.setattr(indexed_subsets, "EXCEPTION_TABLE_WINDOW", 2)

    errors: list[str] = []
    cells = indexed_subsets._check_eaf_exceptions(
        "hm3", group, root, subset, n_analyses, errors
    )

    assert errors == []
    assert cells.index.tolist() == kept
    assert cells.value.tolist() == approx(kept_values)
