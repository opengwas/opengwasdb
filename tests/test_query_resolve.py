from __future__ import annotations

import math

import numpy as np
import pytest
from dense_sources import build_dense_fixture

from opengwasdb.query import log10_p_two_sided, query_store


def test_log10_p_two_sided_matches_known_values():
    # z=0 -> p=1 -> log10(p)=0; z=1.959964 -> p~0.05 -> log10(p)~-1.301
    log10_p = log10_p_two_sided(np.array([0.0, 1.959964]))
    assert math.isclose(log10_p[0], 0.0, abs_tol=1e-9)
    assert math.isclose(log10_p[1], math.log10(0.05), abs_tol=1e-3)


def test_log10_p_two_sided_stays_finite_past_float64_underflow():
    # 2*norm.sf(47.8) underflows to exactly 0.0 in float64 (FADS1/FADS2, issue #104);
    # the log-space computation must stay finite and strictly negative.
    log10_p = log10_p_two_sided(np.array([47.8]))
    assert np.isfinite(log10_p[0])
    assert log10_p[0] < -300

    from scipy import stats

    assert 2.0 * stats.norm.sf(47.8) == 0.0  # sanity: confirms the naive path underflows


def test_log10_p_two_sided_is_sign_independent():
    assert log10_p_two_sided(np.array([-6.0]))[0] == log10_p_two_sided(np.array([6.0]))[0]


def test_store_query_resolve_joins_variant_and_analysis_identity(dense_store_path):
    with query_store(dense_store_path) as query:
        result = query.phewas("rs1")
        rows = sorted(query.resolve(result), key=lambda r: r["analysis_id"])

    assert [r["analysis_id"] for r in rows] == ["a1", "a2"]
    assert [r["analysis_label"] for r in rows] == ["Height primary", "Disease primary"]
    # rsid is the one identity field not derivable from the alid alone, and is
    # the expensive part of this join at scale -- omitted from the row
    # entirely unless a caller opts in (issue #104 follow-up).
    assert all("rsid" not in r for r in rows)
    assert all(r["chromosome"] == "1" for r in rows)
    assert all(r["position"] == 100 for r in rows)
    assert [round(r["z"], 3) for r in rows] == [2.0, 6.0]
    assert [round(r["se"], 3) for r in rows] == [0.1, 0.2]
    assert all(r["association_status"] == "observed" for r in rows)
    # p derived from z, not silently omitted.
    for r in rows:
        expected_p = 2.0 * (1.0 - _norm_cdf(abs(r["z"])))
        assert math.isclose(10.0 ** r["log10_p"], expected_p, rel_tol=1e-3)


def test_store_query_resolve_include_variant_info(dense_store_path):
    with query_store(dense_store_path) as query:
        result = query.phewas("rs1")
        rows = sorted(
            query.resolve(result, include_variant_info=True), key=lambda r: r["analysis_id"]
        )
    assert all(r["rsid"] == "rs1" for r in rows)


def test_store_query_resolve_is_empty_for_empty_result(dense_store_path):
    with query_store(dense_store_path) as query:
        result = query.phewas("rs-does-not-exist")
        assert list(query.resolve(result)) == []


def test_resolve_association_status_round_trips(dense_store_path):
    # Every status value StoreQuery/RaggedStoreQuery can emit (ADR-0013) must
    # survive the join unchanged -- an "imputed" or "missing" row is only
    # distinguishable from "observed" on the command line via this column
    # (issue #104).
    with query_store(dense_store_path) as query:
        result = query.phewas("rs1")
        synthetic = dict(result)
        synthetic["association_status"] = np.array(
            ["imputed", "missing"][: len(result["z"])], dtype=object
        )
        rows = list(query.resolve(synthetic))
    assert [r["association_status"] for r in rows] == ["imputed", "missing"]


@pytest.fixture
def rsid_gap_store_path(tmp_path):
    header = (
        "analysis_id\tphenotype_id\tphenotype_label\tanalysis_label\tchromosome"
        "\tposition\teffect_allele\tother_allele\tz\tse\trsid\tstored_effect_scale"
    )
    # Empty rsid field, matching FinnGen/GWAS-Catalog-hybrid stores (issue #104).
    row = "a1\tp1\tHeight\tHeight primary\t1\t100\tA\tG\t2.0\t0.1\t\tsd"
    source = tmp_path / "no_rsid.tsv"
    source.write_text(header + "\n" + row + "\n", encoding="utf-8")
    store_path = tmp_path / "no-rsid.opengwasdb"
    build_dense_fixture(
        [source], store_path, store_id="no-rsid", release_id="v1", reference_assembly="GRCh37"
    )
    return store_path


def test_resolve_rsid_placeholder_when_store_has_none(rsid_gap_store_path):
    with query_store(rsid_gap_store_path) as query:
        result = query.analysis("a1")
        rows = list(query.resolve(result, include_variant_info=True))
    assert len(rows) == 1
    assert rows[0]["rsid"] == "."


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
