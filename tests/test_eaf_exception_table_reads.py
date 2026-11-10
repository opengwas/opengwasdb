"""The Dense EAF exception table is read once, in windows, per validation pass (#387).

The table is one physical file on a full-size Store Release: a Zarr v3 shard of
237,027,522 entries on OGS-00016 and 1,136,991,041 on OGS-00011, 2.7 GiB and
12.7 GiB of `int64` and `float32`.  Two rules read it, and both used to read it
whole: the Indexed Variant Subset check, once per band it compared, and
`side_tables.eaf_exceptions`, whose judge loaded it beside every marked position
the scan had kept.

The fixture is 4,000 variants wide and deliberately small in everything but the
number of exception cells: a frequency far from its variant's baseline at every
second variant, so the table spans several windows and the scan several bands.
The tests assert that shape before asserting anything about the reads.
"""

from __future__ import annotations

from pathlib import Path
from shutil import copytree

import numpy as np
import zarr
from dense_sources import build_dense_from_records
from pytest import MonkeyPatch, TempPathFactory, approx, fixture, mark, param
from store_reads import element_reads, times_read
from test_indexed_subsets import (
    _build,
    _index_group,
    _root,
    _write_standard_variant_list,
    build_rich_store,
)

from opengwasdb.encoding import banding
from opengwasdb.encoding.codec import EAF_EXCEPTION_INDEX, EAF_EXCEPTION_VALUE
from opengwasdb.layouts.dense import indexed_subsets
from opengwasdb.layouts.dense.build import NormalisedAssociation
from opengwasdb.layouts.dense.indexed_subsets import validate_indexed_subsets
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.store import arrays as store_arrays
from opengwasdb.store.arrays import create_array
from opengwasdb.validation import validate_store
from opengwasdb.validation.rules import side_tables
from opengwasdb.variants import CanonicalVariant

NAME = "hm3"
#: Wider than one Indexed Variant Subset inner chunk (65,536), so the index's
#: planes span several physical write units and the comparison several bands.
WIDE_VARIANTS = 65_600
WIDE_ANALYSES = 2
#: Variants one comparison band holds.  The store's own write unit is one inner
#: chunk of 65,536, which a subset of this width spans two of; narrowing the unit
#: the comparison reads is what leaves several bands to tell apart.
BAND_VARIANTS = 16_000
#: Every twentieth variant carries, in one Analysis, a frequency no residual
#: coding can express: 3,280 exact cells, enough to span several windows without
#: making the residual plan lose to a plain float32 plane.
WIDE_EXCEPTION_EVERY = 20


def _wide_records() -> list[NormalisedAssociation]:
    freqs = np.linspace(0.05, 0.95, WIDE_VARIANTS, dtype=np.float32)
    records: list[NormalisedAssociation] = []
    for col in range(WIDE_ANALYSES):
        for row in range(WIDE_VARIANTS):
            eaf = float(freqs[row])
            if col == WIDE_ANALYSES - 1 and row % WIDE_EXCEPTION_EVERY == 0:
                eaf = 0.999  # far from every baseline: an exception, stored exactly
            records.append(
                NormalisedAssociation(
                    analysis_id=f"a{col}",
                    variant=CanonicalVariant("1", row + 1, "A", "G"),
                    z=1.0,
                    se=0.2,
                    eaf=eaf,
                )
            )
    return records


@fixture(scope="module")
def wide_store(tmp_path_factory: TempPathFactory) -> Path:
    store = tmp_path_factory.mktemp("eaf-wide") / "store.opengwasdb"
    build_dense_from_records(
        _wide_records(),
        store,
        store_id="eaf-wide",
        release_id="wide-v1",
        reference_assembly="GRCh38",
        chunk_shape=(1_000, WIDE_ANALYSES),
    )
    return store


def _table_length(store: Path) -> int:
    return int(zarr.open_group(str(store / "data.zarr"), mode="r")[EAF_EXCEPTION_INDEX].shape[0])


@fixture
def wide_subset(tmp_path: Path, wide_store: Path) -> Path:
    """A published subset over the whole wide store."""

    store = copytree(wide_store, tmp_path / "store.opengwasdb")
    listing = tmp_path / "hm3.alid.txt"
    listing.write_text(
        "\n".join(f"1:{row + 1}:A:G" for row in range(WIDE_VARIANTS)) + "\n", encoding="utf-8"
    )
    _build(store, NAME, listing)
    return store


@fixture
def indexed_store(tmp_path: Path) -> Path:
    """A published subset over `test_indexed_subsets`' rich release."""
    store = build_rich_store(tmp_path / "store.opengwasdb")
    _build(store, NAME, _write_standard_variant_list(tmp_path / "subset.alid.txt"))
    return store


def _blocks_and_windows(
    store: Path, monkeypatch: MonkeyPatch, window: int
) -> tuple[int, int]:
    """The comparison's block count and the table's window count, and a check of both."""
    monkeypatch.setattr(indexed_subsets, "EXCEPTION_TABLE_WINDOW", window)
    monkeypatch.setattr(indexed_subsets, "_physical_write_unit", lambda _array: BAND_VARIANTS)
    group = _index_group(store, NAME)
    subset = np.asarray(group["variant_index"][:], dtype=np.int64)
    write_unit = indexed_subsets._physical_write_unit(group["z"])
    bands = indexed_subsets._output_write_bands(len(subset), write_unit)
    tiles = indexed_subsets._analysis_tiles(
        WIDE_ANALYSES, write_unit, indexed_subsets.EXHAUSTIVE_VALIDATION_BAND_CELLS, 1
    )
    windows = -(-_table_length(store) // window)
    return len(bands) * len(tiles), windows


def test_the_fixture_spans_several_bands_and_several_windows(
    wide_subset: Path, monkeypatch: MonkeyPatch
) -> None:
    root = _root(wide_subset)
    group = _index_group(wide_subset, NAME)
    assert root[EAF_EXCEPTION_INDEX].shape[0] > 0, "the primary must carry exceptions"
    assert group[EAF_EXCEPTION_INDEX].shape[0] > 0, "the index must carry the subset's"
    blocks, windows = _blocks_and_windows(wide_subset, monkeypatch, window=64)
    assert blocks > 2, f"the comparison must decode several bands, not {blocks} blocks"
    assert windows > 2, f"the table must span several windows, not {windows}"


def test_the_subset_eaf_table_is_read_once_per_pass(
    wide_subset: Path, monkeypatch: MonkeyPatch
) -> None:
    root = _root(wide_subset)
    _blocks_and_windows(wide_subset, monkeypatch, window=64)

    with element_reads(monkeypatch) as counts:
        result = validate_store(wide_subset, rules={"subsets.indexed"})

    assert result.ok, result.errors
    read = times_read(counts, root[EAF_EXCEPTION_INDEX])
    assert read == 1.0, f"one pass read the primary EAF exception table {read} times"
    assert times_read(counts, root[EAF_EXCEPTION_VALUE]) == 1.0


def test_a_corrupted_eaf_cell_in_a_later_band_is_still_found(
    wide_subset: Path, monkeypatch: MonkeyPatch
) -> None:
    """A table held from the first band must not answer for a later one."""
    group = _index_group(wide_subset, NAME)
    blocks, _windows = _blocks_and_windows(wide_subset, monkeypatch, window=64)
    write_unit = BAND_VARIANTS
    last_band = indexed_subsets._output_write_bands(
        int(group["z"].shape[1]), write_unit
    )[-1][0]
    assert last_band > 0, "the fixture must have more than one band"
    codes = np.asarray(group["eaf"][:])
    codes[WIDE_ANALYSES - 1, last_band + 1] = codes[WIDE_ANALYSES - 1, last_band + 1] + 1
    group["eaf"][:] = codes

    result = validate_store(wide_subset, rules={"subsets.indexed"})

    assert blocks > 1
    assert result.errors == [
        f"indexed subset {NAME!r} eaf values differ from the primary plane; the index "
        "is stale or corrupt"
    ], result.errors


def test_an_exception_value_corrupted_in_a_later_window_is_still_found(
    wide_subset: Path, monkeypatch: MonkeyPatch
) -> None:
    """Nor may a table read from the first window answer for a later one."""
    group = _index_group(wide_subset, NAME)
    _blocks_and_windows(wide_subset, monkeypatch, window=64)
    entries = int(group[EAF_EXCEPTION_VALUE].shape[0])
    assert entries > 64, "the index's table must span several windows"
    values = np.asarray(group[EAF_EXCEPTION_VALUE][:])
    values[entries - 1] = values[entries - 1] + np.float32(0.01)
    group[EAF_EXCEPTION_VALUE][:] = values

    result = validate_store(wide_subset, rules={"subsets.indexed"})

    assert result.errors == [
        f"indexed subset {NAME!r} eaf exception table is not the primary release's "
        "remapped table; the index is stale or corrupt"
    ], result.errors


def test_the_release_eaf_table_is_never_read_whole(
    wide_store: Path, monkeypatch: MonkeyPatch
) -> None:
    """`side_tables.eaf_exceptions` reconciles block by block; neither half loads it.

    Reading it whole is 12.7 GiB on OGS-00011, and the judge then sorted the
    1.14 billion positions the scan had marked beside it: the peak of a full
    `validate`, against this project's 1 GiB budget for one pass.
    """
    assert _table_length(wide_store) >= 300, "the fixture must carry a real table"
    monkeypatch.setattr(banding, "DECODED_BAND_BYTES", 48_000)  # 250 rows, 16 bands
    monkeypatch.setattr(side_tables, "SPAN_PAGE", 64)
    whole: list[str] = []

    class NoWholeTable:
        """The table's names, and a `read` that says the whole table was loaded."""

        index_name = side_tables.EafExceptionTable.index_name
        value_name = side_tables.EafExceptionTable.value_name

        @staticmethod
        def read(_group: object) -> object:
            whole.append("read")
            raise AssertionError("the whole eaf exception table was read")

    monkeypatch.setattr(side_tables, "EafExceptionTable", NoWholeTable)

    result = validate_store(wide_store, rules={"side_tables.eaf_exceptions"})

    assert result.ok, result.errors
    assert whole == []


def _drop_last_entry(root: zarr.Group) -> None:
    """Rewrite one exception table without its last cell, through the array seam."""
    for name, dtype in ((EAF_EXCEPTION_INDEX, "int64"), (EAF_EXCEPTION_VALUE, "float32")):
        kept = np.asarray(root[name][:])[:-1]
        del root[name]
        create_array(
            root,
            name,
            store_arrays.ArrayRole.EXCEPTION_TABLE,
            data=np.asarray(kept, dtype=dtype),
            compressor=None,
        )


def test_an_entry_missing_from_a_later_band_is_still_found(
    wide_store: Path, monkeypatch: MonkeyPatch
) -> None:
    """A dropped table cell is named in the band that marks it, with its position."""
    monkeypatch.setattr(banding, "DECODED_BAND_BYTES", 48_000)  # 250 rows, 16 bands
    monkeypatch.setattr(side_tables, "SPAN_PAGE", 64)
    root = zarr.open_group(str(wide_store / "data.zarr"), mode="r+")
    dropped = int(np.asarray(root[EAF_EXCEPTION_INDEX][:])[-1])
    assert dropped >= WIDE_VARIANTS, "the dropped cell must sit past the first band"
    _drop_last_entry(root)

    result = validate_store(wide_store, rules={"side_tables.eaf_exceptions"})

    assert result.errors == [
        f"data.zarr/eaf: 1 eaf exception cell(s) have no entry in the eaf exception "
        f"table (first at flat position {dropped})"
    ], result.errors


def _table(path: Path, positions: np.ndarray, values: np.ndarray) -> zarr.Group:
    """One exception table as a Zarr group, the way a plane carries it."""
    group = zarr.open_group(str(path), mode="w")
    group.create_array(EAF_EXCEPTION_INDEX, data=positions)
    group.create_array(EAF_EXCEPTION_VALUE, data=values)
    return group


@mark.parametrize(
    ("positions", "kept", "hitless_window"),
    [
        param(
            [6, 7, 8, 9, 12, 21, 28, 30, 33],
            [12, 21, 28],
            True,
            id="a-window-with-no-hit",
        ),
        param([3, 6, 7, 8, 12, 21, 28], [3, 12, 21, 28], False, id="a-short-last-window"),
    ],
)
def test_a_table_read_in_windows_of_four_keeps_every_cell(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    positions: list[int],
    kept: list[int],
    hitless_window: bool,
) -> None:
    """Every window contributes its cells, and only those the subset holds.

    Windows of four over a table whose positions are one of them entirely outside
    the subset and whose last window is short: the reader takes each page's cells
    below the block's end, so both are the loop's own branches.
    """
    n_analyses, subset = 3, np.array([1, 4, 7, 9], dtype=np.int64)
    variants = [position // n_analyses for position in positions]
    assert len(positions) % 4 != 0, "the last window must be short"
    if hitless_window:
        assert any(
            all(variant not in subset.tolist() for variant in variants[i : i + 4])
            for i in range(0, len(positions), 4)
        ), "a window must hold no cell of the subset's"
    values = np.asarray([0.01 * (i + 1) for i in range(len(positions))], dtype=np.float32)
    kept_values = [float(values[positions.index(position)]) for position in kept]
    remapped = indexed_subsets._remap_side_table(
        np.asarray(kept, dtype=np.int64),
        np.asarray(kept_values, dtype=np.float32),
        subset,
        n_analyses,
    )
    root = _table(tmp_path / "root.zarr", np.asarray(positions, dtype=np.int64), values)
    group = _table(tmp_path / "index.zarr", *remapped)
    monkeypatch.setattr(indexed_subsets, "EXCEPTION_TABLE_WINDOW", 4)

    errors: list[str] = []
    cells = indexed_subsets._check_eaf_exceptions(
        "hm3", group, root, subset, n_analyses, errors
    )

    assert errors == []
    assert cells.index.tolist() == kept
    assert cells.value.tolist() == approx(kept_values)


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
    for col in range(4)
]


def _report(store: Path) -> list[tuple[str, str, str, str]]:
    return [
        (f.rule, f.severity.value, f.target, f.message)
        for f in validate_store(store).findings
    ]


def test_the_whole_report_is_the_one_the_inherited_check_produced(
    indexed_store: Path,
) -> None:
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
        (
            "subsets.indexed",
            "error",
            "data.zarr",
            f"indexed subset {NAME!r} eaf values differ from the primary plane; the "
            "index is stale or corrupt",
        ),
        *_UNVERIFIED,
        ("fidelity.source", "skip", "data.zarr", "subsets.indexed"),
    ]


def _rewrite_primary_table(
    store: Path, positions: np.ndarray, values: np.ndarray
) -> None:
    """Replace the release's own exception table, through the array seam."""
    root = zarr.open_group(str(store / "data.zarr"), mode="r+")
    for name, data, dtype in (
        (EAF_EXCEPTION_INDEX, positions, "int64"),
        (EAF_EXCEPTION_VALUE, values, "float32"),
    ):
        del root[name]
        create_array(
            root,
            name,
            store_arrays.ArrayRole.EXCEPTION_TABLE,
            data=np.asarray(data, dtype=dtype),
            compressor=None,
        )


def _subset_errors(store: Path) -> list[str]:
    """The subset validation alone, judged as `subsets.indexed` judges it."""
    root = _root(store)
    n_variants, n_analyses = (int(size) for size in root["z"].shape)
    errors: list[str] = []
    validate_indexed_subsets(
        store, root, StoreManifest.load(store), n_variants, n_analyses, errors
    )
    return errors


_STAGING_FINDING = (
    "temporary indexed-subset entry '.tmp' is not a published index; a staging or "
    "replacement directory left by an interrupted build must be removed"
)


@mark.parametrize(
    "unsorted",
    [
        param(False, id="a-repeated-position"),
        param(True, id="an-unsorted-position"),
    ],
)
def test_a_primary_table_that_is_not_sorted_and_unique_is_a_finding(
    wide_subset: Path, unsorted: bool
) -> None:
    """A corrupt release table is a finding, and the entries after it are still judged.

    Building the index's own table from it raises, because a table that is not
    sorted and unique is not one an index can be the remapping of. That must be
    the subset's finding -- every namespace entry after it still has to be read
    -- and not an exception that ends the rule with nothing reported at all.
    """
    root = _root(wide_subset)
    positions = np.asarray(root[EAF_EXCEPTION_INDEX][:])
    values = np.asarray(root[EAF_EXCEPTION_VALUE][:])
    if unsorted:
        first, second = int(positions[0]), int(positions[1])
        positions = np.concatenate(([second, first], positions[2:]))
    else:
        positions = np.concatenate((positions, positions[-1:]))
        values = np.concatenate((values, values[-1:]))
    (wide_subset / "data.zarr" / "indexed_subsets" / ".tmp").mkdir()
    _rewrite_primary_table(wide_subset, positions, values)

    errors = _subset_errors(wide_subset)

    assert errors == [
        _STAGING_FINDING,
        f"indexed subset {NAME!r} eaf exception table is not the primary release's "
        "remapped table; the index is stale or corrupt",
    ]


def test_a_table_whose_values_are_not_parallel_with_its_positions_is_a_finding(
    wide_subset: Path,
) -> None:
    """A value array that is longer than the positions is one read away from nothing."""
    root = _root(wide_subset)
    positions = np.asarray(root[EAF_EXCEPTION_INDEX][:])
    values = np.asarray(root[EAF_EXCEPTION_VALUE][:])
    _rewrite_primary_table(
        wide_subset, positions, np.concatenate((values, np.float32([0.5])))
    )

    errors = _subset_errors(wide_subset)

    assert errors == [
        f"indexed subset {NAME!r} eaf exception table is not the primary release's "
        "remapped table; the index is stale or corrupt"
    ]
