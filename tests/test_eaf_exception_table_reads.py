"""No rule of `validate` reads the Dense EAF exception table whole (#387).

The table is one physical file on a full-size Store Release: a Zarr v3 shard of
237,027,522 entries on OGS-00016 and 1,136,991,041 on OGS-00011, 2.7 GiB and
12.7 GiB of `int64` and `float32`.  Three rules read it — the plan rule
`encoding.eaf`, the Indexed Variant Subset check and `side_tables.eaf_exceptions`
— and each used to read it whole, or once per band.

Two fixtures, each asserted before anything is asserted about the reads:

- the **wide** one is 65,600 variants wide with 3,280 exception cells, so the
  index's planes span several physical write units and the table several pages;
- the **small** one is 4,000 variants wide with 400, which is enough for the
  release-level rules to cross pages and be read in many bands.

Every corruption is planted on a copy or on a fixture of its own: nothing here
opens a shared store for writing.
"""

from __future__ import annotations

from pathlib import Path
from shutil import copytree

import numpy as np
import zarr
from dense_sources import build_dense_from_records
from pytest import MonkeyPatch, TempPathFactory, approx, fixture, mark, param
from store_reads import largest_read
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
from opengwasdb.layouts.dense.indexed_subsets import (
    INDEXED_SUBSETS_GROUP,
    validate_indexed_subsets,
)
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.store import arrays as store_arrays
from opengwasdb.store.arrays import create_array
from opengwasdb.store.open import open_store
from opengwasdb.validation import validate_store
from opengwasdb.validation.rule import Kind, Target
from opengwasdb.validation.rules import encoding as encoding_rules
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
#: coding can express: 3,280 exact cells on the wide fixture and 400 on the small
#: one, enough to span several pages without making the residual plan lose to a
#: plain float32 plane.
EXCEPTION_EVERY = 20
SMALL_VARIANTS = 4_000
ANALYSES = 2


def _records(variants: int) -> list[NormalisedAssociation]:
    freqs = np.linspace(0.05, 0.95, variants, dtype=np.float32)
    records: list[NormalisedAssociation] = []
    for col in range(ANALYSES):
        for row in range(variants):
            eaf = float(freqs[row])
            if col == ANALYSES - 1 and row % EXCEPTION_EVERY == 0:
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


def _build_store(path: Path, variants: int, store_id: str) -> Path:
    build_dense_from_records(
        _records(variants),
        path,
        store_id=store_id,
        release_id=f"{store_id}-v1",
        reference_assembly="GRCh38",
        chunk_shape=(1_000, ANALYSES),
    )
    return path


@fixture(scope="module")
def wide_seed(tmp_path_factory: TempPathFactory) -> Path:
    """The wide release with its HapMap3 subset, published once for the module."""
    store = _build_store(
        tmp_path_factory.mktemp("eaf-wide") / "store.opengwasdb", WIDE_VARIANTS, "eaf-wide"
    )
    listing = store.parent / "hm3.alid.txt"
    listing.write_text(
        "\n".join(f"1:{row + 1}:A:G" for row in range(WIDE_VARIANTS)) + "\n", encoding="utf-8"
    )
    _build(store, NAME, listing)
    return store


@fixture
def wide_subset(tmp_path: Path, wide_seed: Path) -> Path:
    """A copy of the wide release, so a test cannot change the module's."""
    return copytree(wide_seed, tmp_path / "store.opengwasdb")


@fixture(scope="module")
def small_seed(tmp_path_factory: TempPathFactory) -> Path:
    """A release whose table crosses pages but whose grid is 16 bands wide."""
    root = tmp_path_factory.mktemp("eaf-small")
    store = _build_store(root / "store.opengwasdb", SMALL_VARIANTS, "eaf-small")
    listing = root / "hm3.alid.txt"
    listing.write_text(
        "\n".join(f"1:{row + 1}:A:G" for row in range(SMALL_VARIANTS)) + "\n", encoding="utf-8"
    )
    _build(store, NAME, listing)
    return store


@fixture
def small_copy(tmp_path: Path, small_seed: Path) -> Path:
    return copytree(small_seed, tmp_path / "store.opengwasdb")


@fixture
def indexed_store(tmp_path: Path) -> Path:
    """A published subset over `test_indexed_subsets`' rich release."""
    store = build_rich_store(tmp_path / "store.opengwasdb")
    _build(store, NAME, _write_standard_variant_list(tmp_path / "subset.alid.txt"))
    return store


def _table(store: Path) -> tuple[int, int]:
    """The release's exception table length and first position."""
    root = _root(store)
    entries = int(root[EAF_EXCEPTION_INDEX].shape[0])
    first = int(root[EAF_EXCEPTION_INDEX][0]) if entries else -1
    return entries, first


def _rewrite_primary_table(store: Path, positions: np.ndarray, values: np.ndarray) -> None:
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


# ── The subset check reads the table once, in windows ───────────────────────


def _blocks_and_windows(store: Path, monkeypatch: MonkeyPatch, window: int) -> tuple[int, int]:
    """The comparison's block count and the table's window count, and a check of both."""
    monkeypatch.setattr(indexed_subsets, "EXCEPTION_TABLE_WINDOW", window)
    monkeypatch.setattr(indexed_subsets, "_physical_write_unit", lambda _array: BAND_VARIANTS)
    group = _index_group(store, NAME)
    subset = np.asarray(group["variant_index"][:], dtype=np.int64)
    write_unit = BAND_VARIANTS
    bands = indexed_subsets._output_write_bands(len(subset), write_unit)
    tiles = indexed_subsets._analysis_tiles(
        ANALYSES, write_unit, indexed_subsets.EXHAUSTIVE_VALIDATION_BAND_CELLS, 1
    )
    windows = -(-_table(store)[0] // window)
    return len(bands) * len(tiles), windows


def test_the_fixture_spans_several_bands_and_several_windows(
    wide_subset: Path, monkeypatch: MonkeyPatch
) -> None:
    root = _root(wide_subset)
    group = _index_group(wide_subset, NAME)
    assert root[EAF_EXCEPTION_INDEX].shape[0] > 0, "the primary must carry exceptions"
    assert group[EAF_EXCEPTION_INDEX].shape[0] > 0, "the index must carry the subset's"
    blocks, windows = _blocks_and_windows(wide_subset, monkeypatch, window=64)
    assert blocks >= 5, f"the comparison must decode several bands, not {blocks} blocks"
    assert windows >= 50, f"the table must span several windows, not {windows}"


def test_no_consumer_reads_the_whole_table(
    small_copy: Path, monkeypatch: MonkeyPatch
) -> None:
    """A full `validate` pages through the release's table, and never loads it.

    Every rule that reads it is asked here at once: the plan rule, the subset check
    and the side-table rule that the association scan folds over.  A whole read of
    it is 2.7 GiB on OGS-00016 and 12.7 GiB on OGS-00011.
    """
    entries, _first = _table(small_copy)
    assert entries >= 300, "the fixture must carry a real table"
    monkeypatch.setattr(indexed_subsets, "EXCEPTION_TABLE_WINDOW", 64)
    monkeypatch.setattr(banding, "DECODED_BAND_BYTES", 48_000)
    monkeypatch.setattr(side_tables, "SPAN_PAGE", 64)
    monkeypatch.setattr(encoding_rules, "EAF_TABLE_PAGE", 64)
    whole = _no_whole_table(monkeypatch)

    # Every rule that reads the table, and what they run after: the plan rule, the
    # subset check and the scan the side-table rule folds over.
    result = validate_store(
        small_copy,
        rules={"encoding.eaf", "subsets.indexed", "side_tables.eaf_exceptions"},
    )

    assert result.ok, result.errors
    assert whole == []


def _no_whole_table(monkeypatch: MonkeyPatch) -> list[str]:
    """Make `EafExceptionTable.read` refuse the release's table, and only that one.

    A subset's own copy of the table is `O(exception count)` and small; the
    release's is the shard this ticket is about.
    """
    original = side_tables.EafExceptionTable.read
    whole: list[str] = []

    def read(group: object) -> object:
        if INDEXED_SUBSETS_GROUP in str(getattr(group, "store_path", "")):
            return original(group)  # type: ignore[arg-type]
        whole.append(str(getattr(group, "store_path", group)))
        raise AssertionError("the release's whole eaf exception table was read")

    monkeypatch.setattr(side_tables.EafExceptionTable, "read", read)
    return whole


def test_a_corrupted_eaf_cell_in_a_later_band_is_still_found(
    wide_subset: Path, monkeypatch: MonkeyPatch
) -> None:
    """A table held from the first band must not answer for a later one."""
    group = _index_group(wide_subset, NAME)
    blocks, _windows = _blocks_and_windows(wide_subset, monkeypatch, window=64)
    write_unit = BAND_VARIANTS
    last_band = indexed_subsets._output_write_bands(int(group["z"].shape[1]), write_unit)[-1][0]
    assert last_band > 0, "the fixture must have more than one band"
    codes = np.asarray(group["eaf"][:])
    codes[ANALYSES - 1, last_band + 1] = codes[ANALYSES - 1, last_band + 1] + 1
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


def test_an_entry_missing_from_a_later_band_is_still_found(
    small_copy: Path, monkeypatch: MonkeyPatch
) -> None:
    """A dropped table cell is named in the band that marks it, with its position."""
    monkeypatch.setattr(banding, "DECODED_BAND_BYTES", 48_000)  # 250 rows: 16 bands
    monkeypatch.setattr(side_tables, "SPAN_PAGE", 64)
    root = zarr.open_group(str(small_copy / "data.zarr"), mode="r+")
    dropped = int(np.asarray(root[EAF_EXCEPTION_INDEX][:])[-1])
    assert dropped >= 250 * ANALYSES, "the dropped cell must sit past the first band"
    _drop_last_entry(root)

    result = validate_store(small_copy, rules={"side_tables.eaf_exceptions"})

    assert result.errors == [
        f"data.zarr/eaf: 1 eaf exception cell(s) have no entry in the eaf exception "
        f"table (first at flat position {dropped})"
    ], result.errors


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


# ── The release's side table is judged block by block ───────────────────────


def _side_table_errors(store: Path, monkeypatch: MonkeyPatch) -> list[str]:
    """The side-table rule alone, over sixteen bands and pages of 64 entries."""
    monkeypatch.setattr(banding, "DECODED_BAND_BYTES", 48_000)
    monkeypatch.setattr(side_tables, "SPAN_PAGE", 64)
    return validate_store(store, rules={"side_tables.eaf_exceptions"}).errors


def test_a_stray_entry_in_a_later_span_is_found(
    small_copy: Path, monkeypatch: MonkeyPatch
) -> None:
    """A table entry for a cell the plane does not mark, in the last band's span.

    The appended position is the grid's last cell; its variant is not one of the
    ones that carry an exception, so nothing marks it.
    """
    root = zarr.open_group(str(small_copy / "data.zarr"), mode="r+")
    positions = np.asarray(root[EAF_EXCEPTION_INDEX][:])
    values = np.asarray(root[EAF_EXCEPTION_VALUE][:])
    stray = SMALL_VARIANTS * ANALYSES - 1
    assert stray > int(positions[-1]), "the stray must be past every real entry"
    _rewrite_primary_table(
        small_copy,
        np.concatenate((positions, [stray])),
        np.concatenate((values, np.float32([0.5]))),
    )

    assert _side_table_errors(small_copy, monkeypatch) == [
        f"data.zarr/eaf: eaf exception table describes 1 cell(s) that are not marked "
        f"eaf exception (first at flat position {stray})"
    ]


def test_an_entry_past_the_last_span_is_found(
    small_copy: Path, monkeypatch: MonkeyPatch
) -> None:
    """A table entry for a cell past the grid's last is no cell at all."""
    root = zarr.open_group(str(small_copy / "data.zarr"), mode="r+")
    positions = np.asarray(root[EAF_EXCEPTION_INDEX][:])
    values = np.asarray(root[EAF_EXCEPTION_VALUE][:])
    beyond = SMALL_VARIANTS * ANALYSES + 5
    _rewrite_primary_table(
        small_copy,
        np.concatenate((positions, [beyond])),
        np.concatenate((values, np.float32([0.5]))),
    )

    assert _side_table_errors(small_copy, monkeypatch) == [
        f"data.zarr/eaf: eaf exception table describes 1 cell(s) that are not marked "
        f"eaf exception (first at flat position {beyond})"
    ]


def test_a_component_with_no_cells_reports_every_entry_stray(
    small_seed: Path, monkeypatch: MonkeyPatch
) -> None:
    """Nothing was read, so every entry the table holds is past the last span.

    The scan's own count of what it took is zero there, and the judgement has to
    come from the table's length instead.
    """
    entries, first = _table(small_seed)
    assert entries > 1
    target = _target(small_seed)
    monkeypatch.setattr(side_tables, "_spans", _no_spans)

    findings = list(
        side_tables._judge_eaf_exceptions(target, side_tables.Reconciled(returned=0))
    )

    assert findings == [
        f"data.zarr/eaf: eaf exception table describes {entries} cell(s) that are not "
        f"marked eaf exception (first at flat position {first})"
    ]


def _no_spans(_target: Target) -> None:
    """No block was measured, so the rule never opened its reader."""
    return None


def _target(store: Path) -> Target:
    """The grid target `validate` builds for a Dense release."""
    opened = open_store(store)
    return Target(
        Kind.GRID,
        "data.zarr",
        opened,
        options=None,
        group=zarr.open_group(str(store / "data.zarr"), mode="r"),
        encoding=StoreManifest.load(store).encoding,
    )


# ── The plan rule pages through the table too ───────────────────────────────


def _plan_errors(store: Path, monkeypatch: MonkeyPatch) -> list[str]:
    """The plan rule alone, over pages of 64 entries of a 400-entry table."""
    monkeypatch.setattr(encoding_rules, "EAF_TABLE_PAGE", 64)
    return validate_store(store, rules={"encoding.eaf"}).errors


@mark.parametrize(
    ("change", "message"),
    [
        param("disorder", "is not sorted by position, or repeats one", id="a-disorder"),
        param("duplicate", "is not sorted by position, or repeats one", id="a-duplicate"),
        param("frequency", "holds a value that is not a frequency in [0, 1]", id="out-of-range"),
    ],
)
def test_the_plan_rule_judges_across_a_page_boundary(
    small_copy: Path, monkeypatch: MonkeyPatch, change: str, message: str
) -> None:
    """A defect across the seam between two pages is judged as reading whole was.

    Every one of these sits at the first page boundary, so a rule that judged one
    page at a time would miss it; each is also read with no single read larger
    than a page.
    """
    root = zarr.open_group(str(small_copy / "data.zarr"), mode="r+")
    positions = np.asarray(root[EAF_EXCEPTION_INDEX][:], dtype=np.int64)
    values = np.asarray(root[EAF_EXCEPTION_VALUE][:], dtype=np.float32)
    assert len(positions) > 130, "the table must span several pages of 64"
    if change == "disorder":
        positions[64], positions[63] = positions[63], positions[64]
    elif change == "duplicate":
        positions[64] = positions[63]
    else:
        values[64] = np.float32(1.5)
    _rewrite_primary_table(small_copy, positions, values)
    monkeypatch.setattr(encoding_rules, "EAF_TABLE_PAGE", 64)

    with largest_read(monkeypatch) as largest:
        errors = validate_store(small_copy, rules={"encoding.eaf"}).errors

    assert errors == [f"data.zarr eaf exception table {message}"], errors
    assert largest[str(root[EAF_EXCEPTION_INDEX].store_path)] <= 64


# ── The window walk keeps every cell ────────────────────────────────────────


def _table_group(path: Path, positions: np.ndarray, values: np.ndarray) -> zarr.Group:
    """One exception table as a Zarr group, the way a plane carries it."""
    group = zarr.open_group(str(path), mode="w")
    group.create_array(EAF_EXCEPTION_INDEX, data=positions)
    group.create_array(EAF_EXCEPTION_VALUE, data=values)
    return group


@mark.parametrize(
    ("positions", "kept", "hitless_window"),
    [
        param([6, 7, 8, 9, 12, 21, 28, 30, 33], [12, 21, 28], True, id="a-window-with-no-hit"),
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
    the subset and whose last window is short: the windowed read takes each
    window's cells whose variant the subset holds, so both are its own branches.
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
    root = _table_group(tmp_path / "root.zarr", np.asarray(positions, dtype=np.int64), values)
    group = _table_group(tmp_path / "index.zarr", *remapped)
    monkeypatch.setattr(indexed_subsets, "EXCEPTION_TABLE_WINDOW", 4)

    errors: list[str] = []
    cells = indexed_subsets._check_eaf_exceptions(
        "hm3", group, root, subset, n_analyses, errors
    )

    assert errors == []
    assert cells.index.tolist() == kept
    assert cells.value.tolist() == approx(kept_values)


# ── Corrupt tables are findings, and the report does not move ───────────────


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
_NOT_THE_REMAPPING = (
    f"indexed subset {NAME!r} eaf exception table is not the primary release's "
    "remapped table; the index is stale or corrupt"
)


@mark.parametrize(
    "unsorted",
    [param(False, id="a-repeated-position"), param(True, id="an-unsorted-position")],
)
def test_a_primary_table_that_is_not_sorted_and_unique_is_a_finding(
    wide_subset: Path, unsorted: bool
) -> None:
    """A corrupt release table is a finding, and the entries after it are still judged.

    Building the index's own table from it raises, because a table that is not
    sorted and unique is not one an index can be the remapping of. That must be
    the subset's finding — every namespace entry after it still has to be read —
    and not an exception that ends the rule with nothing reported at all.
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
    (wide_subset / "data.zarr" / INDEXED_SUBSETS_GROUP / ".tmp").mkdir()
    _rewrite_primary_table(wide_subset, positions, values)

    assert _subset_errors(wide_subset) == [_STAGING_FINDING, _NOT_THE_REMAPPING]


def test_a_table_whose_values_are_not_parallel_with_its_positions_is_a_finding(
    wide_subset: Path,
) -> None:
    """A value array that is longer than the positions is one read away from nothing."""
    root = _root(wide_subset)
    positions = np.asarray(root[EAF_EXCEPTION_INDEX][:])
    values = np.asarray(root[EAF_EXCEPTION_VALUE][:])
    _rewrite_primary_table(wide_subset, positions, np.concatenate((values, np.float32([0.5]))))

    assert _subset_errors(wide_subset) == [_NOT_THE_REMAPPING]


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
        (f.rule, f.severity.value, f.target, f.message) for f in validate_store(store).findings
    ]


def test_the_whole_report_is_the_one_the_inherited_check_produced(
    indexed_store: Path,
) -> None:
    """`validate`'s report is a contract, and this change moves none of it.

    The test pins every record of a clean release and of one whose index holds a
    corrupted EAF cell: the finding's rule, severity, target and message.
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
