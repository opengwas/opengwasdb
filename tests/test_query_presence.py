"""One presence rule for every layout's queries (spec §15, ADR 0068).

An association is present exactly when neither Z nor SE carries its plane's
missing marker. An SE above a `float16` plane's range is stored `+inf` beside a
finite Z (issue #338), and every layout returns that association, observed. A
cell spec §15 forbids -- one marker without the other -- is refused by every
layout rather than dropped (Dense) or labelled missing (Ragged), as each did.

Each fixture plants the value into every copy a query reads (a Ragged
component's by-variant index holds its own), and the plant is asserted to have
landed before anything is asserted about a query.
"""

from __future__ import annotations

from pathlib import Path
from shutil import copytree

import numpy as np
import pytest
import zarr
from conftest import SOURCE_HEADER, SOURCE_ROWS
from test_query_metadata_reads import _BUILDERS

from opengwasdb.associations import Associations, InvalidAssociations
from opengwasdb.build.observed import build_dense_observed_from_sources
from opengwasdb.layouts.hybrid.layout import dense_component_path, dense_to_shared_path
from opengwasdb.query import query_store

#: (builder, Analysis, shared variant index, analysis index) of one present cell.
CELLS = {
    "dense": ("dense", "a1", 1, 0),
    "ragged": ("ragged", "trait_a", 1, 0),
    "hybrid_on_panel": ("hybrid", "trait_a", 2, 0),
    "hybrid_overflow": ("hybrid", "trait_a", 1, 0),
}


@pytest.fixture(scope="module")
def stores(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("presence")
    source = base / "associations.tsv"
    source.write_text(SOURCE_HEADER + "\n" + "\n".join(SOURCE_ROWS) + "\n", encoding="utf-8")
    dense = base / "store.opengwasdb"
    build_dense_observed_from_sources(
        [source], dense, store_id="f", release_id="v", reference_assembly="GRCh37"
    )
    built = {}
    for name, build in _BUILDERS.items():
        (base / name).mkdir()
        built[name] = build(base / name, dense)
    return built


def _csr_position(group: zarr.Group, variant: int, analysis: int) -> int:
    offsets = np.asarray(group["offsets"][:], dtype=np.int64)
    start, end = int(offsets[analysis]), int(offsets[analysis + 1])
    found = np.flatnonzero(np.asarray(group["variant_index"][start:end]) == variant)
    return start + int(found[0])


def _plant_csr(store: Path, variant: int, analysis: int, se: float) -> None:
    """The SE of one CSR row, in the sequence and in its by-variant copy."""
    ragged = zarr.open_group(store / "data.zarr" / "ragged", mode="r+")
    ragged["se"][_csr_position(ragged, variant, analysis)] = se
    index = ragged["by_variant"]
    start, end = (int(x) for x in index["offsets"][variant : variant + 2])
    rows = np.flatnonzero(np.asarray(index["analysis_index"][start:end]) == analysis)
    index["se"][start + int(rows[0])] = se


def _plant(stores: dict[str, Path], case: str, se: float, tmp_path: Path) -> Path:
    builder, _, variant, analysis = CELLS[case]
    store = Path(copytree(stores[builder], tmp_path / "store"))
    if case == "dense":
        zarr.open_group(store / "data.zarr", mode="r+")["se"][variant, analysis] = se
    elif case == "hybrid_on_panel":
        panel_row = int(np.flatnonzero(np.load(dense_to_shared_path(store)) == variant)[0])
        grid = zarr.open_group(dense_component_path(store) / "data.zarr", mode="r+")
        grid["se"][panel_row, analysis] = se
    else:
        _plant_csr(store, variant, analysis, se)
    return store


def _row(result, variant: int, analysis: int) -> dict[str, object]:
    at = np.flatnonzero(
        (result["variant_index"] == variant) & (result["analysis_index"] == analysis)
    )
    assert len(at) == 1, f"expected one row for ({variant}, {analysis}), got {len(at)}"
    return {key: result[key][int(at[0])] for key in result}


def _queries(store: Path, case: str):
    _, analysis_id, variant, _ = CELLS[case]
    query = query_store(store)
    alid = query.variants_table()[variant]["alid"]
    return query, {
        "analysis": lambda: query.analysis(analysis_id),
        "phewas": lambda: query.phewas(alid),
        "lookup": lambda: query.lookup([alid], [analysis_id]),
    }


@pytest.mark.parametrize("case", sorted(CELLS))
def test_an_infinite_se_is_returned_observed_on_every_layout(stores, case, tmp_path) -> None:
    store = _plant(stores, case, np.inf, tmp_path)
    _, _, variant, analysis = CELLS[case]
    query, shapes = _queries(store, case)
    with query:
        for name, run in shapes.items():
            row = _row(run(), variant, analysis)
            assert np.isposinf(row["se"]), name
            assert np.isfinite(row["z"]), name
            assert row["association_status"] == "observed", name


@pytest.mark.parametrize("case", sorted(CELLS))
def test_a_lone_missing_se_is_refused_on_every_layout(stores, case, tmp_path) -> None:
    store = _plant(stores, case, np.nan, tmp_path)
    query, shapes = _queries(store, case)
    with query:
        for run in shapes.values():
            with pytest.raises(InvalidAssociations, match="partial"):
                run()


def test_dense_lookup_is_analysis_major_as_ragged_is(stores) -> None:
    for name in ("dense", "ragged"):
        with query_store(stores[name]) as query:
            alids = [row["alid"] for row in query.variants_table().values()]
            ids = [row["analysis_id"] for row in query.analyses_table().values()]
            result = query.lookup(alids, ids)
            assert result["analysis_index"].tolist() == sorted(
                result["analysis_index"].tolist()
            ), name
            assert len(set(result["analysis_index"].tolist())) == 2, name


@pytest.mark.parametrize("name", sorted(_BUILDERS))
def test_every_query_returns_an_association_record(stores, name) -> None:
    with query_store(stores[name]) as query:
        analysis_id = next(iter(query.analyses_table().values()))["analysis_id"]
        alid = next(iter(query.variants_table().values()))["alid"]
        results = {
            "analysis": query.analysis(analysis_id),
            "unknown": query.analysis("no-such-analysis"),
            "phewas": query.phewas(alid),
            "range_phewas": query.range_phewas("1", 1, 10**9),
            "lookup": query.lookup([alid], [analysis_id]),
            "top_hits": query.top_hits(threshold=5e-4),
        }
        for shape, result in results.items():
            assert isinstance(result, Associations), shape
            assert set(result) >= {"variant_index", "z", "se", "eaf", "association_status"}
