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

import numpy as np
import pytest
from presence_fixtures import CELLS, build_stores, plant
from test_query_metadata_reads import _BUILDERS

from opengwasdb.associations import Associations, InvalidAssociations
from opengwasdb.query import query_store


@pytest.fixture(scope="module")
def stores(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return build_stores(tmp_path_factory.mktemp("presence"))


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
    store = plant(stores, case, np.inf, tmp_path)
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
    store = plant(stores, case, np.nan, tmp_path)
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
