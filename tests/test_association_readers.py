"""One contract for every association reader (ADR 0070).

The same table of associations is written into a Dense grid and a CSR, and
each reader is asked the same regions. What differs between them is only what
each layout stores: a grid holds no association at a missing cell, a CSR holds
a row for every attempted pair. Presence, Status, the frequency column and the
§15 refusal are asserted identical, against the table -- never against another
reader. The Indexed Variant Subset adapter is pinned to its own Dense release.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
import residual_stores
import zarr
from presence_fixtures import build_stores
from store_reads import chunk_reads, duplicate_chunk_keys
from test_indexed_subsets import _build, alid, build_rich_store
from test_ragged_residual_completion import RaggedResidualScenario

from opengwasdb.associations import (
    Associations,
    Cells,
    CsrReader,
    DenseGridReader,
    InvalidAssociations,
    State,
    SubsetReader,
)
from opengwasdb.encoding import StoreCodec
from opengwasdb.encoding.codec import EAF_ABSENT, Z_MISSING
from opengwasdb.encoding.plan import EafEncoding, SeEncoding, StoreEncoding, ZEncoding
from opengwasdb.layouts.dense.indexed_subsets import INDEXED_SUBSETS_GROUP
from opengwasdb.layouts.ragged.by_variant import VariantIndexError, build_variant_index
from opengwasdb.layouts.ragged.zarr_csr import RAGGED_ZARR_PATH, RaggedCSRWriter
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.query import query_store
from opengwasdb.store.open import open_store
from opengwasdb.variants import VariantAxis

N_VARIANTS = 8
N_ANALYSES = 3
NAN = np.nan
ENCODING = StoreEncoding(
    z=ZEncoding(kind="int16_fixed", scale=1024),
    se=SeEncoding(kind="float16"),
    eaf=EafEncoding(kind="float32"),
)


class Cell(NamedTuple):
    variant: int
    analysis: int
    z: float
    se: float
    eaf: float
    flag: int


#: Every attempted cell; a grid's other cells are missing. Values are exact in
#: float16 SE and the int16 Z code (scale 1024), so decoded equals written.
TABLE = (
    Cell(0, 0, 1.5, 0.125, 0.25, 0),
    Cell(1, 0, 2.0, np.inf, 0.375, 0),  # +inf SE: present (spec §15, #338)
    Cell(2, 0, NAN, NAN, NAN, 0),  # a stored missing row
    Cell(3, 0, -1.0, 0.25, 0.5, 1),
    Cell(1, 1, 0.5, 0.375, 0.625, 0),
    Cell(4, 1, 3.0, 0.0625, 0.75, 1),
    Cell(5, 2, 1.0, 0.125, NAN, 0),
)

#: One §15-invalid cell each, added to the table at an otherwise empty slot.
INVALID = {
    "partial": Cell(6, 2, 1.0, NAN, 0.5, 0),
    "flag": Cell(6, 2, 1.0, 0.5, 0.5, 2),
    "imputed_missing": Cell(6, 2, NAN, NAN, NAN, 1),
}


def _write_dense(
    path: Path, table: tuple[Cell, ...], encoding: StoreEncoding = ENCODING
) -> zarr.Group:
    codec = StoreCodec(encoding)
    z = np.full((N_VARIANTS, N_ANALYSES), NAN, np.float32)
    se = np.full_like(z, NAN)
    eaf = np.full_like(z, NAN)
    flags = np.zeros(z.shape, np.uint8)
    for c in table:
        z[c.variant, c.analysis] = c.z
        se[c.variant, c.analysis] = c.se
        eaf[c.variant, c.analysis] = c.eaf
        flags[c.variant, c.analysis] = c.flag
    group = zarr.open_group(path, mode="w")
    group.create_array("z", data=codec.encode_z(z))
    group.create_array("se", data=se.astype(np.float16))
    group.create_array("eaf", data=eaf)
    group.create_array("imputed", data=flags)
    return group


def _write_csr(path: Path, table: tuple[Cell, ...], encoding: StoreEncoding = ENCODING) -> Path:
    """The table as an association sequence. The writer refuses a cell with one
    of z and SE missing, so such a cell is written with a stand-in SE and its
    SE planted as missing afterwards."""
    writer = RaggedCSRWriter(N_VARIANTS)
    flags: list[int] = []
    unpaired: list[int] = []
    for analysis in range(N_ANALYSES):
        rows = sorted((c for c in table if c.analysis == analysis), key=lambda c: c.variant)
        se = np.array([c.se for c in rows], np.float32)
        z = np.array([c.z for c in rows], np.float32)
        half = np.isnan(se) & ~np.isnan(z)
        unpaired.extend((np.flatnonzero(half) + len(flags)).tolist())
        writer.add_analysis(
            np.array([c.variant for c in rows], np.int32),
            z,
            np.where(half, np.float32(1.0), se),
            eaf=np.array([c.eaf for c in rows], np.float32),
        )
        flags.extend(c.flag for c in rows)
    writer.flush(path, encoding)
    ragged = zarr.open_group(path / RAGGED_ZARR_PATH, mode="a")
    for position in unpaired:
        ragged["se"][position] = np.float16(np.nan)
    ragged.create_array("imputed", data=np.array(flags, np.uint8))
    build_variant_index(path, n_axis=N_VARIANTS)
    return path


def _readers(
    tmp_path: Path, table: tuple[Cell, ...], encoding: StoreEncoding = ENCODING
) -> dict[str, DenseGridReader | CsrReader]:
    # A bare grid or sequence records no completion state; this one carries a mask.
    dense = DenseGridReader.open(
        _write_dense(tmp_path / "grid.zarr", table, encoding), encoding, requires_mask=True
    )
    csr = CsrReader.open(
        _write_csr(tmp_path / "csr", table, encoding), encoding, requires_mask=True
    )
    csr.open_variant_index(N_VARIANTS)
    return {"dense": dense, "csr": csr}


@pytest.fixture(scope="module")
def readers(tmp_path_factory: pytest.TempPathFactory) -> dict[str, DenseGridReader | CsrReader]:
    return _readers(tmp_path_factory.mktemp("readers"), TABLE)


def test_fixture_holds_every_valid_state_and_an_infinite_se(readers) -> None:
    found = readers["csr"].inspect(Cells.analyses(range(N_ANALYSES)))
    assert set(found.state.tolist()) == {State.OBSERVED, State.MISSING, State.IMPUTED}
    assert np.isinf(found.se).sum() == 1
    # Decoded equals written, so the assertions below compare against the table.
    by_cell = {(c.variant, c.analysis): c for c in TABLE}
    for v, a, z in zip(found.variant_index, found.analysis_index, found.z, strict=True):
        assert np.array_equal(z, by_cell[(int(v), int(a))].z, equal_nan=True)


def _expected(cells: list[Cell]) -> dict[str, list]:
    status = ["missing" if np.isnan(c.z) else ("imputed" if c.flag else "observed") for c in cells]
    return {
        "variant_index": [c.variant for c in cells],
        "analysis_index": [c.analysis for c in cells],
        "z": [c.z for c in cells],
        "se": [c.se for c in cells],
        "eaf": [c.eaf for c in cells],
        "association_status": status,
    }


def _assert_rows(result, cells: list[Cell]) -> None:
    expected = _expected(cells)
    for key in ("variant_index", "analysis_index", "association_status"):
        assert result[key].tolist() == expected[key], key
    for key in ("z", "se", "eaf"):
        assert np.array_equal(result[key], np.array(expected[key], np.float32), equal_nan=True), key


def _stored(layout: str, cells: list[Cell]) -> list[Cell]:
    """What a layout returns of these cells: a grid stores no missing one."""
    return cells if layout == "csr" else [c for c in cells if not np.isnan(c.z)]


@pytest.mark.parametrize("layout", ["dense", "csr"])
def test_analysis_region_returns_the_analysis_variants_ascending(readers, layout) -> None:
    cells = sorted((c for c in TABLE if c.analysis == 0), key=lambda c: c.variant)
    _assert_rows(readers[layout].read(Cells.analysis(0)), _stored(layout, cells))


@pytest.mark.parametrize("layout", ["dense", "csr"])
def test_observed_only_drops_imputed_rows_and_keeps_missing_ones(readers, layout) -> None:
    cells = sorted((c for c in TABLE if c.analysis == 0 and c.flag == 0), key=lambda c: c.variant)
    _assert_rows(
        readers[layout].read(Cells.analysis(0), observed_only=True), _stored(layout, cells)
    )


@pytest.mark.parametrize("layout", ["dense", "csr"])
def test_variant_region_returns_every_analysis_at_the_variants(readers, layout) -> None:
    cells = sorted((c for c in TABLE if c.variant in (1, 4)), key=lambda c: (c.variant, c.analysis))
    _assert_rows(readers[layout].read(Cells.variants([1, 4])), cells)


@pytest.mark.parametrize("layout", ["dense", "csr"])
def test_cross_region_is_analysis_major_in_request_order(readers, layout) -> None:
    wanted = (0, 1, 3, 4)
    cells = [
        c
        for a in (1, 0)
        for c in sorted(TABLE, key=lambda c: c.variant)
        if c.analysis == a and c.variant in wanted
    ]
    _assert_rows(readers[layout].read(Cells.cross(wanted, [1, 0])), _stored(layout, cells))


@pytest.mark.parametrize("layout", ["dense", "csr"])
def test_pairs_region_answers_each_named_cell_in_request_order(readers, layout) -> None:
    cells = [TABLE[3], TABLE[0], TABLE[1]]
    found = readers[layout].read(
        Cells.pairs(np.array([c.variant for c in cells]), np.array([c.analysis for c in cells]))
    )
    _assert_rows(found, cells)


def test_a_csr_refuses_a_pair_it_does_not_hold(readers) -> None:
    with pytest.raises(LookupError, match="not rows of this component"):
        readers["csr"].read(Cells.pairs(np.array([7]), np.array([0])))


def test_a_csr_reads_flat_positions_in_the_order_given(readers) -> None:
    rows = sorted(TABLE, key=lambda c: (c.analysis, c.variant))
    found = readers["csr"].read(Cells.positions(np.array([3, 0, 3])))
    _assert_rows(found, [rows[3], rows[0], rows[3]])


# ── region reads: a span of Analyses, and an Analysis a window at a time ─────


@pytest.mark.parametrize(
    ("positions", "run"),
    [
        ([5, 6, 7], (5, 8)),
        ([3], (3, 4)),
        ([], None),
        ([0, 2, 1, 3], None),  # the right ends and length, out of order
        ([0, 1, 1, 3], None),  # the right ends and length, with a repeat
        ([0, 1, 3], None),
        ([2, 1, 0], None),
    ],
)
def test_positions_read_as_a_slice_only_when_they_are_one_ascending_run(positions, run) -> None:
    from opengwasdb.associations.csr import _run

    assert _run(np.array(positions, dtype=np.int64)) == run


def _in_order(analyses: range) -> list[Cell]:
    return sorted(
        (c for c in TABLE if c.analysis in analyses), key=lambda c: (c.analysis, c.variant)
    )


@pytest.mark.parametrize("span", [(0, 3), (1, 3), (2, 3), (0, 1), (1, 1), (3, 3)])
@pytest.mark.parametrize("layout", ["dense", "csr"])
def test_a_span_of_analyses_reads_as_each_analysis_in_turn(readers, layout, span) -> None:
    first, stop = span
    found = readers[layout].read(Cells.analyses(first, stop))
    _assert_rows(found, _stored(layout, _in_order(range(first, stop))))


def test_a_dense_span_reads_each_plane_once(dense_residual, monkeypatch) -> None:
    release, _observed, _panel, _imputed = dense_residual
    grid = _grid(release.store)
    assert grid.n_analyses >= 2, "the span must hold more than one Analysis to mean anything"
    with chunk_reads(monkeypatch, _EAF_ARRAYS) as reads:
        found = grid.read(Cells.analyses(0, grid.n_analyses))
    _assert_read_once(reads, "span")
    each = [grid.read(Cells.analysis(a)) for a in range(grid.n_analyses)]
    assert found.count == sum(part.count for part in each) > 0
    for key in found:
        joined = np.concatenate([part[key] for part in each])
        assert np.array_equal(found[key], joined, equal_nan=joined.dtype.kind == "f"), key


def _size(window: Cells) -> int:
    """The cells a window names: its flat positions, or its variants × Analyses."""
    if window.kind == "positions":
        assert window.pos is not None
        return len(window.pos)
    assert window.kind == "cross" and window.vi is not None and window.ai is not None
    return len(window.vi) * len(window.ai)


@pytest.mark.parametrize("max_cells", [1, 2, 3, 4, 100])
@pytest.mark.parametrize("analysis", [0, 1, 2])
@pytest.mark.parametrize("layout", ["dense", "csr"])
def test_windows_read_an_analysis_once_in_order_within_max_cells(
    readers, layout, analysis, max_cells
) -> None:
    reader = readers[layout]
    windows = list(reader.windows(Cells.analysis(analysis), max_cells=max_cells))
    assert windows and all(0 < _size(window) <= max_cells for window in windows)
    _assert_rows(
        Associations.concat([reader.read(window) for window in windows]),
        _stored(layout, _in_order(range(analysis, analysis + 1))),
    )


@pytest.mark.parametrize("max_cells", [1, 3, 5, 100])
@pytest.mark.parametrize("layout", ["dense", "csr"])
def test_windows_with_no_region_read_the_whole_component(readers, layout, max_cells) -> None:
    reader = readers[layout]
    windows = list(reader.windows(max_cells=max_cells))
    assert all(0 < _size(window) <= max_cells for window in windows)
    _assert_rows(
        Associations.concat([reader.read(window) for window in windows]),
        _stored(layout, _in_order(range(N_ANALYSES))),
    )


@pytest.mark.parametrize("layout", ["dense", "csr"])
def test_windows_keep_the_order_a_list_of_analyses_names(readers, layout) -> None:
    reader = readers[layout]
    windows = list(reader.windows(Cells.analyses([2, 0]), max_cells=2))
    expected = _in_order(range(2, 3)) + _in_order(range(0, 1))
    _assert_rows(
        Associations.concat([reader.read(window) for window in windows]),
        _stored(layout, expected),
    )


@pytest.mark.parametrize("layout", ["dense", "csr"])
def test_a_window_s_z_is_the_z_of_its_cells(readers, layout) -> None:
    reader = readers[layout]
    windows = list(reader.windows(max_cells=3))
    assert len(windows) > 1
    for window in windows:
        assert np.array_equal(reader.z(window), reader.inspect(window).z, equal_nan=True)


@pytest.mark.parametrize("layout", ["dense", "csr"])
def test_a_window_s_z_refuses_a_non_canonical_marker(tmp_path, layout) -> None:
    """A scan thresholds Z, and a NaN clears no cutoff, so a window's Z refuses the
    marker every read refuses rather than let the comparison drop the cell (spec §15)."""
    float_z = StoreEncoding(z=ZEncoding(kind="float16"), se=ENCODING.se, eaf=ENCODING.eaf)
    reader = _readers(tmp_path, (*TABLE, Cell(6, 2, NAN, NAN, NAN, 0)), float_z)[layout]
    if layout == "dense":
        _plant_payload(zarr.open_group(tmp_path / "grid.zarr", mode="r+")["z"], (6, 2))
    else:
        ragged = zarr.open_group(tmp_path / "csr" / RAGGED_ZARR_PATH, mode="r+")
        start = int(np.asarray(ragged["offsets"][2]))
        _plant_payload(ragged["z"], (start + 1,))  # Analysis 2 holds variants 5 and 6
    windows = list(reader.windows(Cells.analysis(2), max_cells=1))
    held = [reader.inspect(window).variant_index.tolist() for window in windows]
    assert [6] in held, "a window must hold the planted cell"
    for window, variants in zip(windows, held, strict=True):
        if variants == [6]:
            with pytest.raises(InvalidAssociations, match=r"variant 6, analysis 2: non_canonical"):
                reader.z(window)
        else:
            assert np.array_equal(reader.z(window), reader.inspect(window).z, equal_nan=True)


@pytest.mark.parametrize("layout", ["dense", "csr"])
def test_windows_refuse_an_empty_window(readers, layout) -> None:
    with pytest.raises(ValueError, match="at least one"):
        list(readers[layout].windows(max_cells=0))


#: Regions naming an Analysis outside the fixture's three, and the first such index.
OUTSIDE = {
    "analysis -1": (Cells.analysis(-1), -1),
    "analysis n": (Cells.analysis(N_ANALYSES), N_ANALYSES),
    "a span past the end": (Cells.analyses(2, N_ANALYSES + 1), N_ANALYSES),
    "a list holding -1": (Cells.analyses([0, -1]), -1),
}

#: The other regions that name Analyses.
OUTSIDE_PAIRED = {
    "cross with -1": (Cells.cross([5], [-1]), -1),
    "cross with n": (Cells.cross([5], [N_ANALYSES]), N_ANALYSES),
    "pairs with -1": (Cells.pairs(np.array([5]), np.array([-1])), -1),
}


def _outside(index: int) -> str:
    return rf"Analysis index {index} is outside \[0, {N_ANALYSES}\)"


@pytest.mark.parametrize("region", sorted(OUTSIDE))
@pytest.mark.parametrize("layout", ["dense", "csr"])
def test_windows_refuse_an_analysis_outside_the_axis(readers, layout, region) -> None:
    """Asked for, Analysis -1 would read the last Analysis's cells as its own, and an
    empty scan reads as "no hits": the refusal comes before any window does."""
    cells, index = OUTSIDE[region]
    with pytest.raises(IndexError, match=_outside(index)):
        readers[layout].windows(cells, max_cells=3)


@pytest.mark.parametrize("region", sorted(OUTSIDE | OUTSIDE_PAIRED))
@pytest.mark.parametrize("call", ["read", "inspect"])
@pytest.mark.parametrize("layout", ["dense", "csr"])
def test_a_region_naming_an_analysis_outside_the_axis_is_refused(
    readers, layout, call, region
) -> None:
    cells, index = (OUTSIDE | OUTSIDE_PAIRED)[region]
    with pytest.raises(IndexError, match=_outside(index)):
        getattr(readers[layout], call)(cells)


def test_a_dense_window_s_z_refuses_an_analysis_outside_the_axis(readers) -> None:
    with pytest.raises(IndexError, match=_outside(-1)):
        readers["dense"].z(Cells.cross([5], [-1]))


#: Regions naming a variant outside the fixture's eight, and the first such index.
OUTSIDE_VARIANTS = {
    "variants -1": (Cells.variants([-1]), -1),
    "variants n": (Cells.variants([N_VARIANTS]), N_VARIANTS),
    "variants past the end": (Cells.variants(np.arange(6, N_VARIANTS + 2)), N_VARIANTS),
    "cross with variant -1": (Cells.cross([-1, 1], [0]), -1),
    "cross with variant n": (Cells.cross([1, N_VARIANTS], [0]), N_VARIANTS),
    "pairs with variant -1": (Cells.pairs(np.array([-1]), np.array([0])), -1),
    "pairs with variant n": (Cells.pairs(np.array([N_VARIANTS]), np.array([0])), N_VARIANTS),
}

#: The CSR's rows, one for each attempted cell.
N_ROWS = len(TABLE)

#: Positions regions outside the CSR's rows, and the first such position.
OUTSIDE_POSITIONS = {
    "position -1": (Cells.positions(np.array([-1])), -1),
    "position n": (Cells.positions(np.array([N_ROWS])), N_ROWS),
    "a run past the end": (Cells.positions(np.arange(N_ROWS - 2, N_ROWS + 1)), N_ROWS),
    "scattered, one past the end": (Cells.positions(np.array([3, N_ROWS + 4, 0])), N_ROWS + 4),
}


@pytest.mark.parametrize("region", sorted(OUTSIDE_VARIANTS))
@pytest.mark.parametrize("call", ["read", "inspect"])
@pytest.mark.parametrize("layout", ["dense", "csr"])
def test_a_region_naming_a_variant_outside_the_axis_is_refused(
    readers, layout, call, region
) -> None:
    """Variant -1 would read the last variant's cells as its own on a grid, and on a
    CSR read as a variant with no associations."""
    cells, index = OUTSIDE_VARIANTS[region]
    with pytest.raises(IndexError, match=rf"variant index {index} is outside \[0, {N_VARIANTS}\)"):
        getattr(readers[layout], call)(cells)


@pytest.mark.parametrize("region", sorted(OUTSIDE_POSITIONS))
@pytest.mark.parametrize("call", ["read", "inspect", "z"])
def test_a_csr_refuses_a_position_outside_its_rows(readers, call, region) -> None:
    cells, index = OUTSIDE_POSITIONS[region]
    with pytest.raises(IndexError, match=rf"position {index} is outside \[0, {N_ROWS}\)"):
        getattr(readers["csr"], call)(cells)


def test_a_dense_window_s_z_refuses_a_variant_outside_the_axis(readers) -> None:
    with pytest.raises(IndexError, match=rf"variant index -1 is outside \[0, {N_VARIANTS}\)"):
        readers["dense"].z(Cells.cross([-1], [0]))


@pytest.mark.parametrize(
    "cells",
    [Cells.cross([1], [0]), Cells.pairs(np.array([1]), np.array([0]))],
    ids=["cross", "pairs"],
)
def test_a_csr_reads_variants_only_on_the_axis_its_index_gives(tmp_path, cells) -> None:
    """A CSR holds no record of its variant axis, so a region naming variants is
    checked against the one `open_variant_index` is given, and refused without it."""
    csr = CsrReader.open(_write_csr(tmp_path / "csr", TABLE), ENCODING, requires_mask=True)
    with pytest.raises(VariantIndexError, match="open the variant index"):
        csr.read(cells)
    csr.open_variant_index(N_VARIANTS)
    assert csr.read(cells).count == 1


@pytest.mark.parametrize("layout", ["dense", "csr"])
def test_windows_refuse_a_region_that_is_not_analyses(readers, layout) -> None:
    with pytest.raises(TypeError, match="Analyses"):
        list(readers[layout].windows(Cells.variants([1])))


def _assert_windows_join_to_reads(
    reader: DenseGridReader | CsrReader, region: Cells | None, whole, max_cells: int
) -> None:
    """`region`'s windows, read and joined, are `whole`, its own read."""
    windows = list(reader.windows(region, max_cells=max_cells))
    assert len(windows) > 1 and all(_size(window) <= max_cells for window in windows)
    joined = Associations.concat([reader.read(window) for window in windows])
    for key in whole:
        assert np.array_equal(joined[key], whole[key], equal_nan=key in ("z", "se", "eaf")), key


def test_windows_read_a_hybrid_overflow_as_its_whole_analyses(hybrid_residual) -> None:
    overflow = _sequence(hybrid_residual, requires_mask=False)
    sizes = np.diff(overflow.structure.offsets)
    busiest = int(np.argmax(sizes))
    assert sizes[busiest] > 2, "an Analysis must span several windows to mean anything"
    region = Cells.analysis(busiest)
    _assert_windows_join_to_reads(overflow, region, overflow.read(region), max_cells=2)
    every = overflow.read(Cells.analyses(range(overflow.n_analyses)))
    _assert_windows_join_to_reads(overflow, None, every, max_cells=2)


def test_windows_read_a_built_dense_release_as_its_whole_analyses(dense_residual) -> None:
    """Windows of 37 variants, the last one short, meet imputed cells: each one's
    frequency is the panel's, and its SE is decoded from that frequency."""
    release, _observed, _panel, imputed = dense_residual
    grid = _grid(release.store)
    assert grid.n_variants % 37 and imputed[:, 1].any()
    region = Cells.analysis(1)
    _assert_windows_join_to_reads(grid, region, grid.read(region), max_cells=37)
    every = grid.read(Cells.analyses(0, grid.n_analyses))
    _assert_windows_join_to_reads(grid, None, every, max_cells=37)


@pytest.mark.parametrize("bad", sorted(INVALID))
@pytest.mark.parametrize("layout", ["dense", "csr"])
def test_a_region_holding_an_invalid_cell_is_refused_and_inspect_counts_it(
    tmp_path, layout, bad
) -> None:
    reader = _readers(tmp_path, (*TABLE, INVALID[bad]))[layout]
    with pytest.raises(InvalidAssociations, match=r"variant 6, analysis 2"):
        reader.read(Cells.analysis(2))
    # A region without the cell still reads.
    assert reader.read(Cells.analysis(0)).count
    found = reader.inspect(Cells.analysis(2))
    assert found.tally().invalid == 1


def test_a_variant_region_without_the_index_is_refused(tmp_path) -> None:
    csr = CsrReader.open(_write_csr(tmp_path / "csr", TABLE), ENCODING, requires_mask=True)
    with pytest.raises(VariantIndexError):
        csr.read(Cells.variants([1]))


def test_n_rides_with_its_rows(tmp_path) -> None:
    writer = RaggedCSRWriter(N_VARIANTS)
    writer.add_analysis(
        np.array([2, 5], np.int32),
        np.array([1.0, 2.0], np.float32),
        np.array([0.5, 0.25], np.float32),
        n=np.array([1000, 2000], np.uint32),
    )
    encoding = StoreEncoding(
        z=ENCODING.z, se=ENCODING.se, eaf=EafEncoding(kind="absent"), n=_n_encoding()
    )
    writer.flush(tmp_path, encoding)
    csr = CsrReader.open(tmp_path, encoding, requires_mask=False)
    assert csr.read(Cells.analysis(0))["n"].tolist() == [1000.0, 2000.0]
    assert csr.read(Cells.positions(np.array([1])))["n"].tolist() == [2000.0]
    grid = DenseGridReader.open(_write_dense(tmp_path / "g", TABLE), ENCODING, requires_mask=True)
    assert "n" not in grid.read(Cells.analysis(0))


def _n_encoding():
    from opengwasdb.encoding.plan import NEncoding

    return NEncoding()


# ── the Indexed Variant Subset adapter ───────────────────────────────────────


@pytest.fixture(scope="module")
def subset_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("subset_reader")
    store = build_rich_store(tmp / "rich.opengwasdb")
    alids = tmp / "subset.alid.txt"
    alids.write_text("\n".join(alid(v) for v in (3, 42, 200, 7)) + "\n", encoding="utf-8")
    _build(store, "few", alids)
    return store


def test_a_subset_inspects_exactly_as_its_dense_release(subset_store: Path) -> None:
    store = open_store(subset_store)
    subset = SubsetReader.open(subset_store, "few")
    grid = DenseGridReader.open(store.arrays(mode="r"), store.manifest.encoding)
    variants = np.array(sorted((3, 42, 200, 7)))
    for analysis in range(grid.n_analyses):
        mine = subset.inspect(Cells.analysis(analysis))
        theirs = grid.inspect(Cells.cross(variants, [analysis]))
        assert mine.variant_index.tolist() == theirs.variant_index.tolist()
        assert mine.state.tolist() == theirs.state.tolist()
        for plane in ("z", "se", "eaf"):
            assert np.array_equal(
                getattr(mine, plane), getattr(theirs, plane), equal_nan=True
            ), plane
        read = subset.read(Cells.analysis(analysis))
        assert read["variant_index"].tolist() == [
            int(v) for v, s in zip(theirs.variant_index, theirs.state, strict=True) if s != 0
        ]


def test_a_subset_reads_one_analysis_only(subset_store: Path) -> None:
    with pytest.raises(TypeError, match="one Analysis"):
        SubsetReader.open(subset_store, "few").read(Cells.variants([3]))


@pytest.mark.parametrize("method", ["read", "inspect"])
@pytest.mark.parametrize("bad", [-1, "n_analyses"])
def test_a_subset_refuses_an_analysis_outside_its_axis_with_index_error(
    subset_store: Path, method: str, bad: int | str
) -> None:
    """The same `IndexError` every other reader raises (issue #393)."""
    subset = SubsetReader.open(subset_store, "few")
    index = subset._subset.n_analyses if bad == "n_analyses" else bad
    with pytest.raises(IndexError, match="Analysis index"):
        getattr(subset, method)(Cells.analysis(index))


def test_a_completed_component_without_its_mask_is_refused(tmp_path) -> None:
    """Its mask says which rows are imputed; without one every row would pass for observed."""
    group = _write_dense(tmp_path / "grid.zarr", TABLE)
    del group["imputed"]
    csr = _write_csr(tmp_path / "csr", TABLE)
    shutil.rmtree(csr / RAGGED_ZARR_PATH / "imputed")
    with pytest.raises(ValueError, match="carries no imputed mask"):
        DenseGridReader.open(group, ENCODING, requires_mask=True)
    with pytest.raises(ValueError, match="carries no imputed mask"):
        CsrReader.open(csr, ENCODING, requires_mask=True)
    # Observed by construction (an Observed-Only release, a Hybrid's Overflow): read as such.
    found = DenseGridReader.open(group, ENCODING, requires_mask=False).read(Cells.analysis(0))
    assert set(found["association_status"].tolist()) == {"observed"}


# ── a non-canonical missing marker (spec §15) ────────────────────────────────

#: A float16 NaN other than `np.nan`'s 0x7e00: a payload a reader must refuse.
PAYLOAD_NAN = 0x7E01


def _plant_payload(array: zarr.Array, index: tuple[int, ...]) -> None:
    """Write `PAYLOAD_NAN`'s bits into one cell of a float16 plane, and check they stuck."""
    array[index] = np.array([PAYLOAD_NAN], np.uint16).view(np.float16)[0]
    assert int(np.asarray(array[index]).view(np.uint16)) == PAYLOAD_NAN


@pytest.mark.parametrize("layout", ["dense", "csr"])
def test_a_non_canonical_marker_is_refused_and_inspect_counts_it(tmp_path, layout) -> None:
    # Both markers set, so the cell is missing by every other rule: only its payload is wrong.
    reader = _readers(tmp_path, (*TABLE, Cell(6, 2, NAN, NAN, NAN, 0)))[layout]
    if layout == "dense":
        _plant_payload(zarr.open_group(tmp_path / "grid.zarr", mode="r+")["se"], (6, 2))
    else:
        ragged = zarr.open_group(tmp_path / "csr" / RAGGED_ZARR_PATH, mode="r+")
        start = int(np.asarray(ragged["offsets"][2]))
        _plant_payload(ragged["se"], (start + 1,))  # Analysis 2 holds variants 5 and 6
    with pytest.raises(InvalidAssociations, match="non_canonical"):
        reader.read(Cells.analysis(2))
    assert reader.inspect(Cells.analysis(2)).tally().count(State.NON_CANONICAL) == 1
    assert reader.read(Cells.analysis(0)).count  # a region without the cell still reads


def test_a_subset_refuses_a_non_canonical_marker(tmp_path) -> None:
    """A subset built on a float16-SE Dense Component can hold a payload NaN too."""
    hybrid = build_stores(tmp_path)["hybrid"]
    dense = open_store(hybrid / "dense")
    alids = [str(row.alid) for row in VariantAxis(dense.path).all()]
    _build(hybrid, "all", _alid_list(tmp_path / "all.txt", alids))
    subset = zarr.open_group(
        hybrid / "dense" / "data.zarr" / INDEXED_SUBSETS_GROUP / "all", mode="r+"
    )
    subset["z"][0, 0] = Z_MISSING
    _plant_payload(subset["se"], (0, 0))
    with pytest.raises(InvalidAssociations, match="non_canonical"):
        SubsetReader.open(hybrid, "all").read(Cells.analysis(0))


def _alid_list(path: Path, alids: list[str]) -> Path:
    path.write_text("\n".join(alids) + "\n", encoding="utf-8")
    return path


# ── residual SE: every adapter decodes SE through the frequency it reads ─────
#
# Residual SE (ADR 0037 §3) is predicted from EAF, so each region reads the
# imputed mask, then EAF with the panel substituted on imputed cells, then SE
# from that one EAF read (#253). These pin, for every adapter and region kind,
# that each of those arrays is read once, and that the shared read lines up:
# observed cells carry the cohort's frequency and imputed ones the panel's,
# `observed_only` cuts every column together, and a missing grid cell drops out
# rather than shifting its neighbours' frequencies.

_EAF_ARRAYS = ("eaf", "eaf_baseline", "imputed")


def _observed_z(row: int) -> float:
    """The residual Dense source's z: 8 on every 50th variant, 1 elsewhere."""
    return 8.0 if row % 50 == 0 else 1.0


class _Release(NamedTuple):
    store: Path
    analysis: dict[str, int]  # Analysis ID -> index
    alid: dict[int, str]  # variant index -> ALID


def _release(store: Path) -> _Release:
    with query_store(store) as query:
        return _Release(
            store,
            {row["analysis_id"]: i for i, row in query.analyses_table().items()},
            {i: row["alid"] for i, row in query.variants_table().items()},
        )


def _grid(store: Path, component: str = "") -> DenseGridReader:
    release = open_store(store / component if component else store)
    return DenseGridReader.open(release.arrays(mode="r"), release.manifest.encoding)


def _sequence(store: Path, *, requires_mask: bool | None = None) -> CsrReader:
    reader = CsrReader.open(store, requires_mask=requires_mask)
    reader.open_variant_index(len(_release(store).alid))
    return reader


@pytest.fixture(scope="module")
def dense_residual(tmp_path_factory: pytest.TempPathFactory):
    store, observed, panel = residual_stores.dense_completed(
        tmp_path_factory.mktemp("residual_dense")
    )
    imputed = np.asarray(zarr.open_group(store / "data.zarr", mode="r")["imputed"][:], bool)
    return _release(store), observed, panel, imputed


@pytest.fixture(scope="module")
def dense_residual_subset(dense_residual, tmp_path_factory: pytest.TempPathFactory) -> str:
    """An Indexed Variant Subset of the residual release, over imputed and observed rows."""
    release, _observed, _panel, imputed = dense_residual
    rows = sorted({0, 1, 50, *np.flatnonzero(imputed[:, 1])[:3].tolist()})
    path = tmp_path_factory.mktemp("residual_subset") / "list.txt"
    _build(release.store, "residual", _alid_list(path, [release.alid[r] for r in rows]))
    return "residual"


@pytest.fixture(scope="module")
def ragged_residual(tmp_path_factory: pytest.TempPathFactory):
    scenario = RaggedResidualScenario(tmp_path_factory)
    return scenario, _release(scenario.completed)


@pytest.fixture(scope="module")
def hybrid_residual(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return residual_stores.hybrid_completed(tmp_path_factory.mktemp("residual_hybrid"))


def test_the_residual_releases_are_residual_and_impute(
    dense_residual, ragged_residual, hybrid_residual
) -> None:
    release, observed, panel, imputed = dense_residual
    manifest = StoreManifest.load(release.store)
    assert manifest.encoding.se.is_residual and imputed.any()
    root = zarr.open_group(release.store / "data.zarr", mode="r")
    assert np.all(np.asarray(root["se"][:])[imputed] != 0), "imputed cells carry se codes"
    # An imputed cell's raw EAF is the absent code: only the panel substitution
    # gives it a frequency, and the panel differs from the cohort's.
    assert np.all(np.asarray(root["eaf"][:])[imputed] == EAF_ABSENT)
    observed_row = int(np.argmax(~imputed[:, 1]))
    assert not np.isclose(panel[observed_row], observed[observed_row])
    scenario, _ = ragged_residual
    encoding = StoreManifest.load(scenario.completed).encoding
    assert encoding.se.is_residual and encoding.eaf.reference and scenario.result.n_imputed
    for part in (hybrid_residual, hybrid_residual / "dense"):
        assert StoreManifest.load(part).encoding.se.is_residual, part


def _dense_regions(imputed: np.ndarray) -> dict[str, Cells]:
    row = int(np.flatnonzero(imputed[:, 1])[0])
    return {
        "analysis": Cells.analysis(1),
        "variant": Cells.variants([row]),
        "band": Cells.variants(np.arange(imputed.shape[0])),
        "cross": Cells.cross([0, row], [0, 1]),
        "pairs": Cells.pairs(np.array([0, row]), np.array([0, 1])),
    }


def _assert_read_once(reads: dict[str, list[str]], label: str, *, expect=_EAF_ARRAYS) -> None:
    assert duplicate_chunk_keys(reads) == {}, f"{label} re-read EAF chunks"
    for name in expect:
        assert reads[name], f"{label} must read {name} for this to mean anything"


@pytest.mark.parametrize("region", ["analysis", "variant", "band", "cross", "pairs"])
def test_a_dense_region_reads_each_frequency_array_once(dense_residual, region, monkeypatch):
    release, _observed, _panel, imputed = dense_residual
    grid = _grid(release.store)
    with chunk_reads(monkeypatch, _EAF_ARRAYS) as reads:
        grid.read(_dense_regions(imputed)[region])
    _assert_read_once(reads, region)


def _assert_dense_answers(found, observed, panel, label: str) -> None:
    """Every Dense cell decodes to the source's or the panel's value, by its status."""
    rows = found["variant_index"].astype(np.int64)
    status = found["association_status"]
    is_observed, is_imputed = status == "observed", status == "imputed"
    assert is_observed.any() and np.all(is_observed | is_imputed), label
    np.testing.assert_allclose(found["eaf"][is_observed], observed[rows[is_observed]], rtol=0.02)
    np.testing.assert_allclose(found["eaf"][is_imputed], panel[rows[is_imputed]], rtol=0.02)
    assert np.all(np.isfinite(found["se"])) and np.all(found["se"] > 0), label
    np.testing.assert_allclose(
        found["z"][is_observed], [_observed_z(int(r)) for r in rows[is_observed]], rtol=1e-3
    )


def _assert_observed_only_lockstep(full, filtered, label: str) -> None:
    """`observed_only` drops the imputed rows from every column together."""
    keep = full["association_status"] != "imputed"
    assert keep.any() and not keep.all(), f"{label} must hold observed and imputed rows"
    assert filtered.count == int(keep.sum()), label
    for key in full:
        np.testing.assert_array_equal(full[key][keep], filtered[key], err_msg=f"{label}:{key}")


def test_a_dense_region_decodes_the_source_and_the_panel_unmixed(dense_residual) -> None:
    release, observed, panel, imputed = dense_residual
    grid = _grid(release.store)
    seen_imputed = False
    for name, region in _dense_regions(imputed).items():
        found = grid.read(region)
        _assert_dense_answers(found, observed, panel, name)
        seen_imputed |= bool((found["association_status"] == "imputed").any())
        if name in ("analysis", "band", "cross"):
            _assert_observed_only_lockstep(found, grid.read(region, observed_only=True), name)
    assert seen_imputed, "the regions above must reach imputed cells"


def test_a_missing_grid_cell_drops_out_rather_than_shifting_frequencies(tmp_path) -> None:
    """The shared read is the whole column; a mask that slid would hand every
    later cell its neighbour's frequency -- wrong, plausible and silent (#253)."""
    store = residual_stores.dense_observed_missing(tmp_path)
    frequencies = np.linspace(0.05, 0.95, 600, dtype=np.float32)
    found = _grid(store).read(Cells.analysis(_release(store).analysis["b"]))
    rows = found["variant_index"].tolist()
    assert residual_stores.MISSING_VARIANT - 1 not in rows and len(rows) == 599
    np.testing.assert_allclose(found["eaf"], frequencies[found["variant_index"]], rtol=0.02)


def test_a_subset_reads_once_and_decodes_as_its_release(
    dense_residual, dense_residual_subset, monkeypatch
) -> None:
    release, observed, panel, _imputed = dense_residual
    subset = SubsetReader.open(release.store, dense_residual_subset)
    with chunk_reads(monkeypatch, _EAF_ARRAYS) as reads:
        found = subset.read(Cells.analysis(1))
    assert duplicate_chunk_keys(reads) == {}, "the subset re-read EAF chunks"
    _assert_dense_answers(found, observed, panel, "subset")
    _assert_observed_only_lockstep(
        found, subset.read(Cells.analysis(1), observed_only=True), "subset"
    )


def _csr_regions(release: _Release, reader: CsrReader) -> dict[str, Cells]:
    first = int(reader.read(Cells.analysis(release.analysis["a"]))["variant_index"][0])
    start, end = reader.structure.span(release.analysis["b"])
    return {
        "analysis": Cells.analysis(release.analysis["b"]),
        "analyses": Cells.analyses([release.analysis["b"]]),
        "cross": Cells.cross([first], [release.analysis["b"]]),
        "positions": Cells.positions(np.arange(start, end)),
    }


@pytest.mark.parametrize("region", ["analysis", "analyses", "cross", "positions"])
def test_a_csr_region_reads_each_frequency_array_once(ragged_residual, region, monkeypatch):
    _scenario, release = ragged_residual
    reader = _sequence(release.store)
    regions = _csr_regions(release, reader)
    with chunk_reads(monkeypatch, _EAF_ARRAYS) as reads:
        reader.read(regions[region])
    # A sequence without a baseline array decodes its float32 EAF directly; the
    # mask is read because the plan carries reference EAF.
    _assert_read_once(reads, region, expect=("eaf", "imputed"))


def test_a_csr_region_decodes_the_source_and_the_panel_unmixed(ragged_residual) -> None:
    scenario, release = ragged_residual
    expected, panel = scenario.expected_observed(), scenario.panel_eaf
    analysis_id = {i: a for a, i in release.analysis.items()}
    reader = _sequence(release.store)
    seen_imputed = 0
    for name, region in _csr_regions(release, reader).items():
        found = reader.read(region)
        assert found.count, name
        for vi, ai, se, eaf, status in zip(
            found["variant_index"], found["analysis_index"], found["se"], found["eaf"],
            found["association_status"], strict=True,
        ):
            alid = release.alid[int(vi)]
            if status == "missing":
                assert np.isnan(se) and np.isnan(eaf), name
            elif status == "imputed":
                seen_imputed += 1
                np.testing.assert_allclose(eaf, panel[alid], rtol=0.02)
                assert np.isfinite(se) and se > 0
            else:
                want_se, want_eaf = expected[analysis_id[int(ai)]][alid]
                np.testing.assert_allclose(eaf, want_eaf, rtol=0.02)
                np.testing.assert_allclose(se, want_se, rtol=0.02)
    assert seen_imputed, "the regions above must reach imputed cells"
    region = Cells.analysis(release.analysis["b"])
    _assert_observed_only_lockstep(
        reader.read(region), reader.read(region, observed_only=True), "csr analysis"
    )


def test_each_hybrid_component_reads_once_and_cuts_in_lockstep(hybrid_residual, monkeypatch):
    analysis = _release(hybrid_residual).analysis["trait_b"]
    # The Overflow is observed by construction (ADR 0026), in a completed release too.
    grid = _grid(hybrid_residual, "dense")
    overflow = _sequence(hybrid_residual, requires_mask=False)
    # The Overflow's busiest Analysis: completion moved one crossover into the grid.
    busiest = int(np.argmax(np.diff(overflow.structure.offsets)))
    for label, reader, column in (
        ("dense component", grid, analysis),
        ("overflow", overflow, busiest),
    ):
        with chunk_reads(monkeypatch, _EAF_ARRAYS) as reads:
            found = reader.read(Cells.analysis(column))
        assert found.count, label
        assert duplicate_chunk_keys(reads) == {}, f"{label} re-read EAF chunks"
        assert reads["eaf"], f"{label} must read eaf for this to mean anything"
    region = Cells.analysis(analysis)
    _assert_observed_only_lockstep(
        grid.read(region), grid.read(region, observed_only=True), "dense component"
    )


# ── a completed release read directly, without its mask ─────────────────────


def _without_mask(store: Path, components: tuple[str, ...]) -> None:
    """Delete the imputed mask, and the panel frequencies it would key, so the release
    reads like a completion whose panel carried no EAF: nothing else refuses it."""
    for component in components:
        for name in ("imputed", "eaf_reference", "by_variant/imputed"):
            shutil.rmtree(store / component / name, ignore_errors=True)
    path = store / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["encoding"]["eaf"]["reference"] = False
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_a_direct_open_of_a_completed_release_without_its_mask_is_refused(
    dense_residual, ragged_residual, tmp_path
) -> None:
    """The reader infers the requirement from the release itself; no caller has to ask."""
    dense = Path(shutil.copytree(dense_residual[0].store, tmp_path / "dense"))
    _without_mask(dense, ("data.zarr",))
    release = open_store(dense)
    assert release.manifest.completion_state.value == "reference_completed"
    with pytest.raises(ValueError, match="carries no imputed mask"):
        DenseGridReader.open(release.arrays(mode="r"), release.manifest.encoding)
    ragged = Path(shutil.copytree(ragged_residual[1].store, tmp_path / "ragged"))
    _without_mask(ragged, (RAGGED_ZARR_PATH,))
    with pytest.raises(ValueError, match="carries no imputed mask"):
        CsrReader.open(ragged)


def test_a_reader_that_cannot_tell_whether_a_mask_is_owed_refuses(tmp_path) -> None:
    """A grid or sequence that records no completion state cannot say whether it owes
    a mask; the caller must."""
    group = _write_dense(tmp_path / "grid.zarr", TABLE)
    with pytest.raises(ValueError, match="cannot tell"):
        DenseGridReader.open(group, ENCODING)
    csr = _write_csr(tmp_path / "csr", TABLE)  # its writer records the state; drop it
    del zarr.open_group(csr / RAGGED_ZARR_PATH, mode="r+").attrs["completion_state"]
    with pytest.raises(ValueError, match="cannot tell"):
        CsrReader.open(csr, ENCODING)
