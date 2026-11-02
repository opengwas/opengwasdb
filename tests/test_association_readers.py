"""One contract for every association reader (ADR 0070).

The same table of associations is written into a Dense grid and a CSR, and
each reader is asked the same regions. What differs between them is only what
each layout stores: a grid holds no association at a missing cell, a CSR holds
a row for every attempted pair. Presence, Status, the frequency column and the
§15 refusal are asserted identical, against the table -- never against another
reader. The Indexed Variant Subset adapter is pinned to its own Dense release.
"""

from __future__ import annotations

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
    found = DenseGridReader.open(group, ENCODING).read(Cells.analysis(0))
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


def _sequence(store: Path) -> CsrReader:
    reader = CsrReader.open(store)
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
    grid, overflow = _grid(hybrid_residual, "dense"), _sequence(hybrid_residual)
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
