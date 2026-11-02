"""One contract for every association reader (ADR 0068).

The same table of associations is written into a Dense grid and a CSR, and
each reader is asked the same regions. What differs between them is only what
each layout stores: a grid holds no association at a missing cell, a CSR holds
a row for every attempted pair. Presence, Status, the frequency column and the
§15 refusal are asserted identical, against the table -- never against another
reader. The Indexed Variant Subset adapter is pinned to its own Dense release.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
import zarr
from test_indexed_subsets import _build, alid, build_rich_store

from opengwasdb.associations import (
    Cells,
    CsrReader,
    DenseGridReader,
    InvalidAssociations,
    State,
    SubsetReader,
)
from opengwasdb.encoding import StoreCodec
from opengwasdb.encoding.plan import EafEncoding, SeEncoding, StoreEncoding, ZEncoding
from opengwasdb.layouts.ragged.by_variant import VariantIndexError, build_variant_index
from opengwasdb.layouts.ragged.zarr_csr import RAGGED_ZARR_PATH, RaggedCSRWriter
from opengwasdb.store.open import open_store

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


def _write_dense(path: Path, table: tuple[Cell, ...]) -> zarr.Group:
    codec = StoreCodec(ENCODING)
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


def _write_csr(path: Path, table: tuple[Cell, ...]) -> Path:
    writer = RaggedCSRWriter(N_VARIANTS)
    flags: list[int] = []
    for analysis in range(N_ANALYSES):
        rows = sorted((c for c in table if c.analysis == analysis), key=lambda c: c.variant)
        writer.add_analysis(
            np.array([c.variant for c in rows], np.int32),
            np.array([c.z for c in rows], np.float32),
            np.array([c.se for c in rows], np.float32),
            eaf=np.array([c.eaf for c in rows], np.float32),
        )
        flags.extend(c.flag for c in rows)
    writer.flush(path, ENCODING)
    ragged = zarr.open_group(path / RAGGED_ZARR_PATH, mode="a")
    ragged.create_array("imputed", data=np.array(flags, np.uint8))
    build_variant_index(path, n_axis=N_VARIANTS)
    return path


def _readers(tmp_path: Path, table: tuple[Cell, ...]) -> dict[str, DenseGridReader | CsrReader]:
    dense = DenseGridReader.open(_write_dense(tmp_path / "grid.zarr", table), ENCODING)
    csr = CsrReader.open(_write_csr(tmp_path / "csr", table), ENCODING)
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
    csr = CsrReader.open(_write_csr(tmp_path / "csr", TABLE), ENCODING)
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
    csr = CsrReader.open(tmp_path, encoding)
    assert csr.read(Cells.analysis(0))["n"].tolist() == [1000.0, 2000.0]
    assert csr.read(Cells.positions(np.array([1])))["n"].tolist() == [2000.0]
    assert "n" not in DenseGridReader.open(_write_dense(tmp_path / "g", TABLE), ENCODING).read(
        Cells.analysis(0)
    )


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
