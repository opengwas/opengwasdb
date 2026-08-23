from __future__ import annotations

import math
import shutil

import numpy as np
import pytest
from typer.testing import CliRunner

from opengwasdb.build.observed import build_dense_observed_from_sources
from opengwasdb.cli.main import app
from opengwasdb.encoding import DenseZPlane
from opengwasdb.layouts.dense.rho import (
    build_dense_rho,
    estimate_rho_cml,
    select_thinned_variants,
)
from opengwasdb.query import query_store
from opengwasdb.store.open import open_store
from opengwasdb.validation import validate_store

RHO_SOURCE_HEADER = "\t".join(
    [
        "analysis_id",
        "phenotype_id",
        "phenotype_label",
        "analysis_label",
        "chromosome",
        "position",
        "effect_allele",
        "other_allele",
        "z",
        "se",
        "rsid",
        "stored_effect_scale",
    ]
)

N_PER_CHROM = 150
N_CHROM = 2
N_ANALYSES = 4
TRUE_RHO_01 = 0.4


def _make_rho_source(tmp_path, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = N_PER_CHROM * N_CHROM
    base = rng.standard_normal(n)
    z = {
        0: base,
        1: TRUE_RHO_01 * base + math.sqrt(1 - TRUE_RHO_01**2) * rng.standard_normal(n),
    }
    for a in range(2, N_ANALYSES):
        z[a] = rng.standard_normal(n)

    rows = []
    v = 0
    for c in range(N_CHROM):
        chrom = str(c + 1)
        for p in range(N_PER_CHROM):
            pos = (p + 1) * 100  # 100bp spacing -- >> any window_bp used below
            for a in range(N_ANALYSES):
                rows.append(
                    f"a{a}\tp{a}\tTrait{a}\tTrait{a} primary\t{chrom}\t{pos}\tA\tG\t"
                    f"{z[a][v]:.6f}\t1.0\trs{v}\tsd"
                )
            v += 1
    path = tmp_path / "rho_source.tsv"
    path.write_text(RHO_SOURCE_HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def rho_store_path(tmp_path):
    source = _make_rho_source(tmp_path)
    store_path = tmp_path / "rho-store.opengwasdb"
    build_dense_observed_from_sources(
        [source],
        store_path,
        store_id="rho-fixture",
        release_id="observed-v1",
        reference_assembly="GRCh37",
    )
    return store_path


# ── Distance-thinned variant selection ───────────────────────────────────────


def test_select_thinned_variants_keeps_first_per_window_per_chromosome():
    chromosomes = np.array(["1", "1", "1", "2", "2"])
    positions = np.array([100, 200, 20100, 100, 25000])

    idx = select_thinned_variants(chromosomes, positions, window_bp=10_000)

    assert idx.tolist() == [0, 2, 3, 4]


def test_select_thinned_variants_empty_input():
    assert select_thinned_variants(np.array([]), np.array([]), window_bp=1000).tolist() == []


# ── Estimator (ADR 0025 "Estimator") ─────────────────────────────────────────
# No vendored pleiodb reference is available in this repo, so correctness is
# validated against a known generative model rather than a golden fixture.


def test_estimate_rho_cml_recovers_known_correlation_and_nan_below_min_nulls():
    rng = np.random.default_rng(1)
    true_rho = 0.5
    n = 20_000
    x = rng.standard_normal(n)
    y = true_rho * x + math.sqrt(1 - true_rho**2) * rng.standard_normal(n)

    estimate = estimate_rho_cml(x, y, z_thresh=1.0, min_nulls=100)
    assert estimate == pytest.approx(true_rho, abs=0.1)

    # Independent (rho=0) pair also recovers near zero.
    z_indep = rng.standard_normal(n)
    estimate_zero = estimate_rho_cml(x, z_indep, z_thresh=1.0, min_nulls=100)
    assert estimate_zero == pytest.approx(0.0, abs=0.1)

    # Below min_nulls -> NaN, regardless of how good the data is.
    assert math.isnan(estimate_rho_cml(x[:10], y[:10], z_thresh=1.0, min_nulls=500))


def test_estimate_rho_cml_ignores_missing_and_significant_cells():
    rng = np.random.default_rng(2)
    n = 2000
    x = rng.standard_normal(n)
    y = 0.3 * x + math.sqrt(1 - 0.3**2) * rng.standard_normal(n)

    baseline = estimate_rho_cml(x, y, z_thresh=1.0, min_nulls=100)

    x_polluted = x.copy()
    y_polluted = y.copy()
    x_polluted[:50] = np.nan  # missing
    y_polluted[50:100] = 10.0  # significant -> excluded by |z| < z_thresh

    polluted = estimate_rho_cml(x_polluted, y_polluted, z_thresh=1.0, min_nulls=100)
    assert polluted == pytest.approx(baseline, abs=0.02)


# ── Build round-trip (issue 047) ─────────────────────────────────────────────


def test_build_dense_rho_round_trips_against_direct_estimation(rho_store_path):
    build_dense_rho(rho_store_path, window_bp=50, z_thresh=1.0, min_nulls=20, n_workers=1)

    result = validate_store(rho_store_path)
    assert result.ok, result.errors

    root = open_store(rho_store_path).arrays(mode="r")
    rho_group = root["rho"]
    assert rho_group.attrs["z_thresh"] == 1.0
    assert rho_group.attrs["min_nulls"] == 20
    assert rho_group.attrs["window_bp"] == 50
    assert rho_group.attrs["method"] == "pleiodb-cml"
    assert rho_group.attrs["observed_only"] is True
    assert rho_group.attrs["n_analyses"] == N_ANALYSES
    thinned = np.asarray(rho_group["variant_index"][:])
    assert rho_group.attrs["n_variants_used"] == len(thinned)
    # 100bp spacing >> 50bp window -> thinning keeps every variant.
    assert len(thinned) == N_PER_CHROM * N_CHROM

    query = query_store(rho_store_path)
    pair = query.rho("a0", "a1")
    assert len(pair["rho"]) == 1
    stored_rho = float(pair["rho"][0])
    stored_n = int(pair["n_null"][0])

    z_all = DenseZPlane.open(root, open_store(rho_store_path).manifest.encoding).band(
        0, root["z"].shape[0]
    )
    z0 = z_all[thinned, 0].astype(np.float64)
    z1 = z_all[thinned, 1].astype(np.float64)
    direct = estimate_rho_cml(z0, z1, z_thresh=1.0, min_nulls=20)
    mask = np.isfinite(z0) & np.isfinite(z1) & (np.abs(z0) < 1.0) & (np.abs(z1) < 1.0)

    assert stored_n == int(mask.sum())
    assert not math.isnan(direct)
    assert not math.isnan(stored_rho)
    # Rho itself is stored float16 (ADR 0025).
    assert stored_rho == pytest.approx(direct, abs=1e-2)


def test_rho_is_nan_exactly_below_min_nulls(rho_store_path):
    build_dense_rho(rho_store_path, window_bp=50, z_thresh=1.0, min_nulls=20, n_workers=1)

    root = open_store(rho_store_path).arrays(mode="r")
    rho_vals = np.asarray(root["rho"]["rho"][:], dtype=np.float32)
    n_vals = np.asarray(root["rho"]["n_null"][:], dtype=np.int64)

    assert np.array_equal(np.isfinite(rho_vals), n_vals >= 20)


def test_imputed_cells_are_excluded_from_null_support(rho_store_path):
    root = open_store(rho_store_path).arrays(mode="a")
    n_variants, n_analyses = root["z"].shape
    imputed = np.zeros((n_variants, n_analyses), dtype="uint8")
    imputed[::2, 0] = 1
    imputed[::2, 1] = 1
    root.create_dataset("imputed", data=imputed, chunks=root["z"].chunks, dtype="uint8")

    build_dense_rho(rho_store_path, window_bp=50, z_thresh=1.0, min_nulls=1, n_workers=1)

    rho_group = open_store(rho_store_path).arrays(mode="r")["rho"]
    thinned = np.asarray(rho_group["variant_index"][:])
    plane = DenseZPlane.open(root, open_store(rho_store_path).manifest.encoding)
    z0 = np.asarray(plane.column(0), dtype=np.float64)[thinned]
    z1 = np.asarray(plane.column(1), dtype=np.float64)[thinned]
    imp0 = imputed[:, 0][thinned]
    imp1 = imputed[:, 1][thinned]
    expected_mask = (
        np.isfinite(z0)
        & np.isfinite(z1)
        & (np.abs(z0) < 1.0)
        & (np.abs(z1) < 1.0)
        & (imp0 == 0)
        & (imp1 == 0)
    )

    query = query_store(rho_store_path)
    pair = query.rho("a0", "a1")
    assert int(pair["n_null"][0]) == int(expected_mask.sum())


# ── Query surface (issue 048) ────────────────────────────────────────────────


def test_query_surface_agrees_across_accessors(rho_store_path):
    build_dense_rho(rho_store_path, window_bp=50, z_thresh=1.0, min_nulls=5, n_workers=1)
    query = query_store(rho_store_path)
    ids = [f"a{i}" for i in range(N_ANALYSES)]

    long = query.rho(ids)
    assert len(long["rho"]) == N_ANALYSES * (N_ANALYSES - 1) // 2

    matrix = query.rho_matrix(ids)
    assert matrix["rho"].shape == (N_ANALYSES, N_ANALYSES)
    assert np.allclose(np.diag(matrix["rho"]), 1.0)
    assert np.array_equal(matrix["rho"], matrix["rho"].T)
    assert np.array_equal(matrix["n_null"], matrix["n_null"].T)

    for aid_a, aid_b, r, n in zip(
        long["analysis_id_a"], long["analysis_id_b"], long["rho"], long["n_null"], strict=True
    ):
        i, j = ids.index(aid_a), ids.index(aid_b)
        assert float(matrix["rho"][i, j]) == pytest.approx(float(r))
        assert int(matrix["n_null"][i, j]) == int(n)

        row = query.rho_row(aid_a)
        pos = list(row["analysis_id"]).index(aid_b)
        assert float(row["rho"][pos]) == pytest.approx(float(r))
        assert int(row["n_null"][pos]) == int(n)


def test_rho_matrix_submatrix_matches_full_matrix(rho_store_path):
    build_dense_rho(rho_store_path, window_bp=50, z_thresh=1.0, min_nulls=5, n_workers=1)
    query = query_store(rho_store_path)

    full = query.rho_matrix()
    sub = query.rho_matrix(["a2", "a0"])

    assert sub["rho"].shape == (2, 2)
    assert sub["rho"][0, 1] == pytest.approx(float(full["rho"][2, 0]))
    assert sub["rho"][1, 0] == pytest.approx(float(full["rho"][0, 2]))


def test_rho_query_handles_unknown_ids_and_missing_group(dense_store_path, rho_store_path):
    # No rho group built -> every accessor is empty, not an error.
    no_rho_query = query_store(dense_store_path)
    assert len(no_rho_query.rho("a1", "a2")["rho"]) == 0
    assert len(no_rho_query.rho_row("a1")["rho"]) == 0
    assert no_rho_query.rho_matrix()["rho"].shape == (0, 0)

    build_dense_rho(rho_store_path, window_bp=50, z_thresh=1.0, min_nulls=5, n_workers=1)
    query = query_store(rho_store_path)
    assert len(query.rho("a0", "unknown-analysis")["rho"]) == 0
    assert len(query.rho_row("unknown-analysis")["rho"]) == 0


# ── Parallel/serial parity (issue 050) ───────────────────────────────────────


def test_parallel_build_matches_serial_build(rho_store_path, tmp_path):
    parallel_path = tmp_path / "rho-store-parallel.opengwasdb"
    shutil.copytree(rho_store_path, parallel_path)

    build_dense_rho(rho_store_path, window_bp=50, z_thresh=1.0, min_nulls=5, n_workers=1)
    build_dense_rho(parallel_path, window_bp=50, z_thresh=1.0, min_nulls=5, n_workers=3)

    serial_group = open_store(rho_store_path).arrays(mode="r")["rho"]
    parallel_group = open_store(parallel_path).arrays(mode="r")["rho"]

    serial_rho = np.asarray(serial_group["rho"][:])
    parallel_rho = np.asarray(parallel_group["rho"][:])
    assert np.array_equal(np.isnan(serial_rho), np.isnan(parallel_rho))
    finite = ~np.isnan(serial_rho)
    assert np.array_equal(serial_rho[finite], parallel_rho[finite])
    assert np.array_equal(
        np.asarray(serial_group["n_null"][:]), np.asarray(parallel_group["n_null"][:])
    )


# ── CLI (issue 050) ───────────────────────────────────────────────────────────


def test_cli_build_dense_rho(rho_store_path):
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "build-dense-rho",
            str(rho_store_path),
            "--window-bp", "50",
            "--z-thresh", "1.0",
            "--min-nulls", "5",
            "--n-workers", "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "done"

    validation = validate_store(rho_store_path)
    assert validation.ok, validation.errors

    query = query_store(rho_store_path)
    matrix = query.rho_matrix()
    assert matrix["rho"].shape == (N_ANALYSES, N_ANALYSES)


def test_cli_build_dense_rho_defaults(rho_store_path):
    # CLI defaults must match the pleiodb/PRD defaults.
    runner = CliRunner()
    result = runner.invoke(app, ["build-dense-rho", str(rho_store_path)])
    assert result.exit_code == 0, result.output

    root = open_store(rho_store_path).arrays(mode="r")
    rho_group = root["rho"]
    assert rho_group.attrs["z_thresh"] == 1.0
    assert rho_group.attrs["min_nulls"] == 500
