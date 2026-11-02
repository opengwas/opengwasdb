"""One presence rule for every layout's queries (spec §15, ADR 0070).

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
from shutil import rmtree

import numpy as np
import pytest
import residual_stores
from presence_fixtures import CELLS, build_stores, plant
from test_hybrid_build import HG19_POS_1, HG19_POS_2, HG19_POS_3, _make_manifest, _make_vcf, _panel
from test_query_metadata_reads import _BUILDERS
from test_ragged_residual_completion import RaggedResidualScenario

from opengwasdb.associations import Associations, InvalidAssociations
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.layouts.ragged.top_hits import build_ragged_top_hit_indexes
from opengwasdb.query import query_store
from opengwasdb.top_hits import TopHitTier


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


# ── a top-hit tier without its imputed mask ──────────────────────────────────


@pytest.fixture(scope="module")
def ragged_completed(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A Reference-Completed Ragged release whose one tier holds every cell."""
    scenario = RaggedResidualScenario(tmp_path_factory)
    build_ragged_top_hit_indexes(scenario.completed, thresholds=(1.0,))
    return scenario.completed


def _hide_tier_array(monkeypatch: pytest.MonkeyPatch, hidden: str) -> None:
    """Make every tier look like an index built before it carried `hidden`."""
    original = TopHitTier.has

    def has(self: TopHitTier, name: str) -> bool:
        return False if name == hidden else original(self, name)

    monkeypatch.setattr(TopHitTier, "has", has)


def test_a_ragged_tier_without_its_mask_takes_the_flags_from_the_sequence(
    ragged_completed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with query_store(ragged_completed) as query:
        indexed = query.top_hits(threshold=1.0)
    statuses = indexed["association_status"].tolist()
    assert "imputed" in statuses and "observed" in statuses  # the mask decides something
    _hide_tier_array(monkeypatch, "imputed")
    with query_store(ragged_completed) as query:
        older = query.top_hits(threshold=1.0)
        observed_only = query.top_hits(threshold=1.0, observed_only=True)
    assert older["association_status"].tolist() == statuses
    assert "imputed" not in observed_only["association_status"].tolist()
    assert len(observed_only["z"]) == statuses.count("observed") + statuses.count("missing")


def test_a_completed_release_without_its_mask_is_refused(tmp_path: Path) -> None:
    """Without the mask, every present cell of a completed grid would pass for observed."""
    store, _, _ = residual_stores.dense_completed(tmp_path)
    with query_store(store) as query:
        statuses = set(query.analysis("b")["association_status"].tolist())
    assert "imputed" in statuses  # the mask decides something here
    rmtree(store / "data.zarr" / "imputed")
    with pytest.raises(ValueError, match="carries no imputed mask"):
        query_store(store)


# ── Hybrid lookup order ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def hybrid_two_sided(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A Hybrid release whose two Analyses each hold on-panel and off-panel rows."""
    tmp = tmp_path_factory.mktemp("hybrid_two_sided")
    rows = {
        "trait_a": ["2.0:0.5:0.2", "1.5:0.3:0.3", "0.6:0.2:0.4"],
        "trait_b": ["6.0:0.5:0.25", "1.1:0.3:0.35", "1.2:0.3:0.45"],
    }
    entries = []
    for trait, values in rows.items():
        lines = [
            f"1\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t.\tES:SE:AF\t{value}\n"
            for (pos, ref, alt), value in zip(
                ((HG19_POS_1, "A", "G"), (HG19_POS_2, "C", "T"), (HG19_POS_3, "G", "A")),
                values,
                strict=True,
            )
        ]
        entries.append((trait, _make_vcf(tmp, trait, lines), trait.replace("_", " ")))
    store = tmp / "hybrid.opengwasdb"
    build_hybrid_from_vcf_manifest(
        _make_manifest(tmp, entries), store, reference_panel=_panel(tmp),
        store_id="hybrid-order", release_id="v1", n_workers=1,
    )
    return store


def test_hybrid_lookup_is_analysis_major_across_both_components(hybrid_two_sided) -> None:
    with query_store(hybrid_two_sided) as query:
        alids = [row["alid"] for row in query.variants_table().values()]
        on_panel = query._on_panel_mask(np.arange(len(alids)))
        assert on_panel.any() and not on_panel.all()  # both components hold the variants
        for order in (["trait_a", "trait_b"], ["trait_b", "trait_a"]):
            result = query.lookup(alids, order)
            ids = [query.analyses_table()[int(i)]["analysis_id"] for i in result["analysis_index"]]
            runs = [k for i, k in enumerate(ids) if i == 0 or ids[i - 1] != k]
            assert runs == order, ids  # one contiguous run per Analysis, in request order
            for analysis_id in order:  # and each run holds both components' rows
                rows = result["variant_index"][[i == analysis_id for i in ids]]
                assert on_panel[rows].any() and (~on_panel[rows]).any(), analysis_id
