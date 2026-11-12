"""`validate` reads each statistic plane once, Indexed Variant Subsets and all (#314).

The subset check compared every cell of an Indexed Variant Subset against the
primary planes by reading them again, one comparison unit at a time, after the
association scan had read them once already: on OGS-00009 that second read was
11,030 s of a 16,974 s validation. It now rides the scan's pass. And the release's
EAF exception table, which four rules each streamed, is read once per pass and
handed to all of them.

Two fixtures, each asserted meaningful before anything is asserted about it:

- a Reference-Completed residual release of 200 variants, read in five bands,
  whose subset of every third variant spans all five bands and eight comparison
  units;
- #387's small observed release of 4,000 variants, whose 400-entry exception table
  is read in pages of 64, with a subset of every variant.

Every corruption is planted on a copy.
"""

from __future__ import annotations

from pathlib import Path
from shutil import copytree as copy_release

import numpy as np
import pytest
import zarr
from eaf_tables import rewrite_primary_table
from residual_stores import dense_completed
from store_reads import element_reads, times_read
from test_indexed_subsets import _build, _index_group, _root

from opengwasdb.encoding import banding
from opengwasdb.encoding.codec import EAF_EXCEPTION_INDEX, EAF_EXCEPTION_VALUE, Z_OVERFLOW
from opengwasdb.layouts.dense import indexed_subsets
from opengwasdb.store.arrays import inner_chunk_of
from opengwasdb.validation import scans, validate_store
from opengwasdb.validation.rule import Severity
from opengwasdb.variants import VariantAxis

NAME = "hm3"
#: Forty-row bands of the two-Analysis grid: five over its 200 variants.
BAND_ROWS = 40
#: Index slots per comparison band, and cells per comparison unit: one Analysis a tile.
WRITE_UNIT = 20
UNIT_CELLS = 20
#: The planes a Dense grid's value rules read, and the per-variant arrays beside them.
PLANES = ("z", "se", "eaf", "imputed", "eaf_baseline", "eaf_reference")


def _message(words: str) -> str:
    return f"indexed subset {NAME!r} {words}"


Z_DIFFER = _message("z values differ from the primary plane; the index is stale or corrupt")
EAF_DIFFER = _message("eaf values differ from the primary plane; the index is stale or corrupt")
SE_DIFFER = _message(
    "se values differ from the primary plane or cannot be decoded; the index is stale or "
    "corrupt"
)
MASK_DIFFER = _message(
    "imputed mask disagrees with the primary release; the index is stale or corrupt"
)


@pytest.fixture(scope="module")
def completed_seed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The completed release with a subset of every third variant, built once."""
    root = tmp_path_factory.mktemp("one-pass")
    store, _observed, _panel = dense_completed(root)
    axis = VariantAxis(store)
    try:
        alids = [axis.by_index(row).alid for row in range(0, axis.n_variants, 3)]
    finally:
        axis.close()
    listing = root / "subset.alid.txt"
    listing.write_text("\n".join(alids) + "\n", encoding="utf-8")
    _build(store, NAME, listing)
    return store


@pytest.fixture
def completed(tmp_path: Path, completed_seed: Path) -> Path:
    """A copy of the completed release, so a test cannot change the module's."""
    return copy_release(completed_seed, tmp_path / "store.opengwasdb")


@pytest.fixture
def narrow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bands and comparison units small enough for the fixture to hold several."""
    monkeypatch.setattr(banding, "DECODED_BAND_BYTES", banding.BYTES_PER_CELL * 2 * BAND_ROWS)
    monkeypatch.setattr(indexed_subsets, "_physical_write_unit", lambda _array: WRITE_UNIT)
    monkeypatch.setattr(indexed_subsets, "EXHAUSTIVE_VALIDATION_BAND_CELLS", UNIT_CELLS)


def _subset(store: Path) -> np.ndarray:
    return np.asarray(_index_group(store, NAME)["variant_index"][:], dtype=np.int64)


def test_the_subset_spans_several_bands_and_units(completed: Path, narrow: None) -> None:
    root = _root(completed)
    group = _index_group(completed, NAME)
    subset = _subset(completed)
    n_variants, n_analyses = (int(size) for size in root["z"].shape)
    bands = list(banding.band_limits(n_variants, n_analyses, inner_chunk_of(root["z"])[0]))
    assert len(bands) == 5, bands
    assert all(np.any((subset >= r0) & (subset < r1)) for r0, r1 in bands)
    write_unit = indexed_subsets._physical_write_unit(group["z"])
    write_bands = indexed_subsets._output_write_bands(len(subset), write_unit)
    tiles = indexed_subsets._analysis_tiles(
        n_analyses, write_unit, indexed_subsets.EXHAUSTIVE_VALIDATION_BAND_CELLS, 1
    )
    assert (len(write_bands), len(tiles)) == (4, 2)
    assert int(np.asarray(root["imputed"][:]).sum()) > 0, "the release must impute cells"
    assert int(np.asarray(group["imputed"][:]).sum()) > 0, "and the subset must hold one"


def test_a_subset_is_compared_without_reading_the_planes_again(
    completed: Path, narrow: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each plane is read once, by the scan, and the index once, by the comparison."""
    root = _root(completed)
    group = _index_group(completed, NAME)

    with element_reads(monkeypatch) as counts:
        result = validate_store(completed)

    assert result.ok, result.errors
    assert "subsets.indexed" not in result.skipped
    assert {name: times_read(counts, root[name]) for name in PLANES} == dict.fromkeys(
        PLANES, 1.0
    )
    assert times_read(counts, group["z"]) == 1.0, "the comparison must read the index"


def _ordinary_cell(store: Path, after_slot: int, analysis: int) -> int:
    """The first slot past `after_slot` whose cell is observed, with ordinary codes."""
    group = _index_group(store, NAME)
    imputed = np.asarray(group["imputed"][analysis])
    z, se, eaf = (np.asarray(group[name][analysis]) for name in ("z", "se", "eaf"))
    for slot in range(after_slot, len(imputed)):
        if imputed[slot] == 0 and abs(int(se[slot])) < 100 and abs(int(eaf[slot])) < 100:
            if abs(int(z[slot])) < 30_000:
                return slot
    raise AssertionError("the fixture has no ordinary cell to corrupt")


def _bump(store: Path, plane: str, analysis: int, slot: int, by: int = 1) -> None:
    group = _index_group(store, NAME)
    codes = np.asarray(group[plane][:])
    codes[analysis, slot] = codes[analysis, slot] + by if plane != "imputed" else 1
    group[plane][:] = codes


@pytest.mark.parametrize(
    ("plane", "message"),
    [("z", Z_DIFFER), ("se", SE_DIFFER), ("eaf", EAF_DIFFER), ("imputed", MASK_DIFFER)],
)
def test_a_cell_corrupted_in_a_later_band_is_found(
    completed: Path, narrow: None, plane: str, message: str
) -> None:
    """A cell whose primary row the scan reads in its third band, in the third unit."""
    slot = _ordinary_cell(completed, 2 * WRITE_UNIT + 1, 0)
    assert _subset(completed)[slot] >= 2 * BAND_ROWS, "the cell must sit past two bands"
    _bump(completed, plane, 0, slot)

    result = validate_store(completed)

    assert result.failed() == {"subsets.indexed"}, result.errors
    assert result.errors == [message]


def test_an_earlier_tile_s_finding_comes_first(completed: Path, narrow: None) -> None:
    """The scan reads both Analyses of a band at once; the comparison still reports the
    unit it has always reached first, the first Analysis's, whatever it finds there."""
    late_se = _ordinary_cell(completed, 2 * WRITE_UNIT + 15, 0)
    early_z = _ordinary_cell(completed, 2 * WRITE_UNIT + 1, 1)
    assert late_se // WRITE_UNIT == early_z // WRITE_UNIT == 2, "one band of the index"
    assert early_z < late_se
    _bump(completed, "se", 0, late_se)
    _bump(completed, "z", 1, early_z)

    assert validate_store(completed).errors == [SE_DIFFER]


def test_an_earlier_band_s_finding_comes_first(completed: Path, narrow: None) -> None:
    """A later index band's finding waits for an earlier one's, in any Analysis."""
    first = _ordinary_cell(completed, WRITE_UNIT + 1, 1)
    later = _ordinary_cell(completed, 3 * WRITE_UNIT + 1, 0)
    _bump(completed, "eaf", 1, first)
    _bump(completed, "z", 0, later)

    assert validate_store(completed).errors == [EAF_DIFFER]


def test_a_cell_the_index_cannot_decode_is_named_as_before(
    completed: Path, narrow: None
) -> None:
    """A finding worded from the decoder's own error keeps its words and position."""
    slot = _ordinary_cell(completed, 2 * WRITE_UNIT + 1, 1)
    n_subset = len(_subset(completed))
    group = _index_group(completed, NAME)
    codes = np.asarray(group["z"][:])
    codes[1, slot] = Z_OVERFLOW
    group["z"][:] = codes
    position = n_subset + slot

    assert validate_store(completed).errors == [
        _message(
            f"z cannot be decoded: z overflow table has no entry for cell(s) at flat "
            f"position(s) [{position}]; the store's plane and its table disagree"
        )
    ]


# ── The exception table, once per pass ───────────────────────────────────────


def _pages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sixteen bands of the small release, and its table in pages of 64."""
    monkeypatch.setattr(banding, "DECODED_BAND_BYTES", 48_000)
    monkeypatch.setattr(scans, "SPAN_PAGE", 64, raising=False)


def test_the_exception_table_is_read_once_per_pass(
    small_copy: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scan's decode, the side-table rule, the plan rule and the subset check all
    read the table; the pass reads each entry once, for all four."""
    root = _root(small_copy)
    entries = int(root[EAF_EXCEPTION_INDEX].shape[0])
    assert entries > 4 * 64, "the table must span several pages"
    assert NAME in zarr.open_group(str(small_copy / "data.zarr"), mode="r")["indexed_subsets"]
    _pages(monkeypatch)

    with element_reads(monkeypatch) as counts:
        result = validate_store(small_copy)

    assert result.ok, result.errors
    assert not result.skipped, result.skipped
    assert times_read(counts, root[EAF_EXCEPTION_INDEX]) == 1.0
    assert times_read(counts, root[EAF_EXCEPTION_VALUE]) == 1.0


def test_an_exception_value_corrupted_in_a_later_page_is_found(
    small_copy: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The subset keeps its cells of every page the pass reads, the last one too."""
    group = _index_group(small_copy, NAME)
    entries = int(group[EAF_EXCEPTION_VALUE].shape[0])
    assert entries > 4 * 64, "the index's table must span several pages"
    values = np.asarray(group[EAF_EXCEPTION_VALUE][:])
    values[entries - 1] = values[entries - 1] + np.float32(0.01)
    group[EAF_EXCEPTION_VALUE][:] = values
    _pages(monkeypatch)

    assert validate_store(small_copy).errors == [
        _message(
            "eaf exception table is not the primary release's remapped table; the index "
            "is stale or corrupt"
        )
    ]


@pytest.mark.parametrize(
    "stray_n",
    [pytest.param(False, id="the-plan-rule-alone"), pytest.param(True, id="and-a-second")],
)
def test_a_disordered_table_stops_the_value_rules_as_it_did(
    small_copy: Path,
    monkeypatch: pytest.MonkeyPatch,
    stray_n: bool,
) -> None:
    """The plan rule is decided by the pass now, and the rules standing on it are
    skipped behind it with the reasons they gave when it was decided first: behind
    it alone, or behind it and a second plan rule in the order they run after them,
    and with nothing the read met in the disordered table reported by any of them.

    Alone, the value rules measure in the pass before the plan rule is decided, and
    are skipped once it is; with a second, they are stopped before the read, and
    their skips wait for it."""
    root = _root(small_copy)
    positions = np.asarray(root[EAF_EXCEPTION_INDEX][:], dtype=np.int64)
    values = np.asarray(root[EAF_EXCEPTION_VALUE][:], dtype=np.float32)
    positions[64], positions[63] = positions[63], positions[64]
    rewrite_primary_table(small_copy, positions, values)
    if stray_n:
        writable = zarr.open_group(str(small_copy / "data.zarr"), mode="r+")
        writable.create_array("n", shape=root["z"].shape, dtype="uint16")
    _pages(monkeypatch)

    result = validate_store(small_copy)

    plan = ("encoding.eaf", "encoding.n") if stray_n else ("encoding.eaf",)
    assert set(plan) <= result.failed()
    decodable = result.of("associations.decodable")
    assert [(f.severity, f.reasons) for f in decodable] == [(Severity.SKIP, plan)]
    reported = {f.rule for f in result.findings if f.severity is not Severity.SKIP}
    assert not reported & {
        "associations.decodable",
        "side_tables.eaf_exceptions",
        "associations.eaf_range",
        "top_hits.eaf",
    }
