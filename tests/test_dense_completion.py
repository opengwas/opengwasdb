"""Integration tests for Dense reference completion (ADR 0022, ADR 0023)."""

from __future__ import annotations

import gzip
import io
from pathlib import Path

import numpy as np
import pytest

from opengwasdb.build.observed import build_dense_observed_from_sources
from opengwasdb.completion.checkpoint import checkpoint_dir_for
from opengwasdb.completion.ld_panel import (
    canonical_panel_alid,
    load_block,
    load_ld_eigenvectors,
    snp_position,
)
from opengwasdb.layouts.dense import complete as complete_module
from opengwasdb.layouts.dense.complete import (
    complete_dense_store,
    resume_dense_completion,
)
from opengwasdb.model.analyses import read_analyses, write_analyses
from opengwasdb.query import query_store
from opengwasdb.store.open import open_store
from opengwasdb.validation.validate import validate_store


def _assert_stores_identical(dst_a: Path, dst_b: Path) -> None:
    """Assert two completed stores hold byte-identical z/se/imputed arrays.

    The parity guarantee for completion is stronger than matching cell counts:
    the imputed z/se *values* (and the imputed/on_panel masks) must match too.
    z/se are stored float16, so exact equality (NaN-aware) is the right check.
    """
    root_a = open_store(dst_a).arrays(mode="r")
    root_b = open_store(dst_b).arrays(mode="r")
    for name in ("z", "se"):
        arr_a = root_a[name][:]
        arr_b = root_b[name][:]
        assert arr_a.shape == arr_b.shape, f"{name} shape differs"
        assert np.array_equal(arr_a, arr_b, equal_nan=True), f"{name} values differ"
    for name in ("imputed", "on_panel"):
        assert np.array_equal(root_a[name][:], root_b[name][:]), f"{name} mask differs"

SOURCE_HEADER = "\t".join(
    [
        "analysis_id", "phenotype_id", "phenotype_label", "analysis_label",
        "chromosome", "position", "effect_allele", "other_allele",
        "z", "se", "rsid", "stored_effect_scale",
    ]
)

# a1 observes 4 of the 5 chr1 block positions (900000/950000/1000000/1100000), leaving
# 1050000 as a genuine imputation target — enough points for poly_rescale's quality fit.
# a2 observes only chr1:1,000,000 → its block vector has 1 obs point → never imputed.
# chr1:1,200,000 A/G — observed by a1 only; OFF-PANEL (not in any LD block) → a2's
#                       n_missing_off_panel picks this up.
SOURCE_ROWS = [
    "a1\tp1\tHeight\tHeight primary\t1\t900000\tA\tG\t1.0\t0.15\trs0\tsd",
    "a1\tp1\tHeight\tHeight primary\t1\t950000\tA\tC\t1.8\t0.15\trs0b\tsd",
    "a1\tp1\tHeight\tHeight primary\t1\t1000000\tA\tG\t2.0\t0.15\trs1\tsd",
    "a2\tp2\tDisease\tDisease primary\t1\t1000000\tA\tG\t6.0\t0.20\trs1\tlog_or",
    "a1\tp1\tHeight\tHeight primary\t1\t1100000\tA\tC\t3.0\t0.20\trs2\tsd",
    "a1\tp1\tHeight\tHeight primary\t1\t1200000\tA\tG\t1.5\t0.30\trs3\tsd",
]


def _write_ld_block(
    block_dir: Path,
    block_name: str,
    snps: list[tuple[str, float, int]],
    seed: int = 0,
    *,
    write_matrix: bool = True,
    write_npz: bool = False,
    npz_k: int | None = None,
    snp_id_style: str = "alid",
) -> None:
    """Write one flat-layout LD block.

    A block may carry an LD matrix, an eigendecomposition, or both — panels built
    under ADR 0031 ship only the decomposition, so tests need to construct each
    combination. ``npz_k`` truncates the stored eigenvectors to force the
    under-resolved case. ``snp_id_style`` selects the panel's identifier
    convention: canonical ALIDs, or the legacy ``chr:pos_ref_alt`` form used by the
    production UKB EUR panel.
    """
    block_dir.mkdir(parents=True, exist_ok=True)
    n = len(snps)

    tsv_lines = ["CHR\tSNP\tOA\tEA\tEAF\tBP"]
    for alid, eaf, bp in snps:
        chrom, pos, a1, a2 = alid.split(":")
        snp_id = alid if snp_id_style == "alid" else f"{chrom}:{pos}_{a1}_{a2}"
        tsv_lines.append(f"{chrom}\t{snp_id}\t{a2}\t{a1}\t{eaf}\t{bp}")
    (block_dir / f"{block_name}.tsv").write_text("\n".join(tsv_lines) + "\n")

    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    ld = A @ A.T + np.eye(n) * n * 0.1

    if write_matrix:
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
            for row in ld:
                gz.write(("\t".join(f"{v:.6f}" for v in row) + "\n").encode())
        (block_dir / f"{block_name}.unphased.vcor1.gz").write_bytes(buf.getvalue())

    if write_npz:
        vals, vecs = np.linalg.eigh(ld)
        vals = vals[::-1]
        vecs = vecs[:, ::-1]
        k = vecs.shape[1] if npz_k is None else min(npz_k, vecs.shape[1])
        np.savez_compressed(
            block_dir / f"{block_name}.ldeig", values=vals, vectors=vecs[:, :k]
        )


def _make_ld_panel(tmp_path: Path) -> Path:
    """Two blocks: chr1 (mixes observed + one new panel SNP) and chr2 (pure panel-only)."""
    root = tmp_path / "ld_panel"
    _write_ld_block(
        root / "EUR" / "1", "900000-1300000",
        [
            ("1:900000:A:G", 0.35, 900_000),     # observed by a1
            ("1:950000:A:C", 0.32, 950_000),     # observed by a1
            ("1:1000000:A:G", 0.30, 1_000_000),  # observed by a1, a2
            ("1:1050000:C:T", 0.40, 1_050_000),  # new — imputation target
            ("1:1100000:A:C", 0.45, 1_100_000),  # observed by a1
        ],
        seed=0,
    )
    _write_ld_block(
        root / "EUR" / "2", "1-500000",
        [
            ("2:100000:A:G", 0.20, 100_000),  # new — not observed by anyone
            ("2:200000:C:G", 0.25, 200_000),  # new — not observed by anyone
        ],
        seed=1,
    )
    return root


@pytest.fixture
def source_path(tmp_path: Path) -> Path:
    path = tmp_path / "associations.tsv"
    path.write_text(SOURCE_HEADER + "\n" + "\n".join(SOURCE_ROWS) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def observed_store(tmp_path: Path, source_path: Path) -> Path:
    out = tmp_path / "obs.opengwasdb"
    build_dense_observed_from_sources(
        [source_path], out,
        store_id="test", release_id="obs-v1", reference_assembly="GRCh38",
    )
    return out


@pytest.fixture
def ld_panel(tmp_path: Path) -> Path:
    return _make_ld_panel(tmp_path)


@pytest.fixture
def completed_store(tmp_path: Path, observed_store: Path, ld_panel: Path) -> Path:
    dst = tmp_path / "comp.opengwasdb"
    complete_dense_store(
        observed_store, dst, ld_panel,
        ancestry="EUR", min_cor=0.0, release_id="comp-v1",
    )
    return dst


class TestCompletionFiles:
    def test_creates_store_directory(self, completed_store):
        assert completed_store.exists()
        assert (completed_store / "manifest.json").exists()
        assert (completed_store / "variants.tsv.gz").exists()
        assert (completed_store / "index.sqlite").exists()

    def test_no_leftover_checkpoint_or_work_dir(self, tmp_path, completed_store):
        assert not (tmp_path / f".{completed_store.name}.checkpoint").exists()
        assert not (tmp_path / f".{completed_store.name}.tmp").exists()

    def test_manifest_completion_state(self, completed_store):
        import json
        m = json.loads((completed_store / "manifest.json").read_text())
        assert m["completion_state"] == "reference_completed"
        assert "completion" in m["provenance"]

    def test_variant_table_larger_than_source(self, completed_store, observed_store):
        from opengwasdb.variants.axis import VariantAxis
        obs_ax = VariantAxis(observed_store)
        comp_ax = VariantAxis(completed_store)
        # +2 (chr1 new panel SNP, off-panel chr1:1200000 already in source) +2 (chr2 new panel SNPs)
        assert comp_ax.n_variants == obs_ax.n_variants + 3
        obs_ax.close()
        comp_ax.close()

    def test_imputed_and_on_panel_arrays_present(self, completed_store):
        root = open_store(completed_store).arrays(mode="r")
        assert "imputed" in root
        assert "on_panel" in root
        assert root["imputed"].shape == root["z"].shape
        assert root["on_panel"].shape[0] == root["z"].shape[0]

    def test_off_panel_variant_never_imputed(self, completed_store):
        root = open_store(completed_store).arrays(mode="r")
        on_panel = root["on_panel"][:]
        imputed = root["imputed"][:]
        assert not np.any(imputed[on_panel == 0, :] == 1)

    def test_completion_quality_table_exists(self, completed_store):
        import sqlite3
        conn = sqlite3.connect(str(completed_store / "index.sqlite"))
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        conn.close()
        assert "completion_quality" in tables

    def test_analyses_tsv_carries_forward_analysis_label_with_no_phenotype_columns(
        self, completed_store
    ):
        # ADR 0034/issue #68: completion must carry the unified schema
        # forward (no phenotype_id/phenotype_label), refreshing only the
        # completion-rollup and Top-Hit Count columns.
        from opengwasdb.model.analyses import read_analyses

        table = read_analyses(completed_store / "analyses.tsv")
        assert "phenotype_id" not in table.fieldnames
        assert "phenotype_label" not in table.fieldnames
        rows = {r["analysis_id"]: r for r in table.rows}
        assert rows["a1"]["analysis_label"] == "Height primary"
        assert rows["a2"]["analysis_label"] == "Disease primary"

    def test_analyses_tsv_has_completion_n_missing_total(self, completed_store):
        from opengwasdb.model.analyses import read_analyses

        rows = {
            r["analysis_id"]: int(r["completion_n_missing_total"])
            for r in read_analyses(completed_store / "analyses.tsv").rows
        }
        # a1 observed the off-panel chr1:1200000 variant; a2 never did.
        assert rows["a1"] == 0
        assert rows["a2"] == 1

    def test_analyses_tsv_hit_counts_match_the_completed_stores_own_index(self, completed_store):
        # Completion changes z/se via imputation, so persisted Top-Hit Counts
        # (ADR 0032) must reflect *this* store's post-completion top-hit
        # index, not be carried forward (or double-counted) from the
        # pre-completion source's counts.
        from opengwasdb.layouts.dense.top_hits import read_top_hit_counts
        from opengwasdb.model.analyses import read_analyses

        rows = sorted(
            read_analyses(completed_store / "analyses.tsv").rows,
            key=lambda r: int(r["analysis_index"]),
        )
        expected = read_top_hit_counts(completed_store, len(rows))
        for column, values in expected.items():
            assert [int(r[column]) for r in rows] == values

    def test_overwrite_raises_without_flag(
        self, tmp_path, observed_store, ld_panel, completed_store
    ):
        with pytest.raises(FileExistsError):
            complete_dense_store(observed_store, completed_store, ld_panel, min_cor=0.0)

    def test_stray_file_fails_validation(self, completed_store):
        # Issue #80: the closed-envelope check applies to Reference-Completed
        # releases too -- Reference Completion adds no new top-level entry,
        # so the same allowed set governs both Observed-Only and
        # Reference-Completed Dense releases.
        (completed_store / "traits.tsv.gz").write_bytes(b"stray")

        result = validate_store(completed_store)

        assert not result.ok
        assert any(
            "unexpected store entry" in error and "traits.tsv.gz" in error
            for error in result.errors
        )


class TestRegionCap:
    """QC z-cap: an imputed |z| is clamped to the max observed |z| within
    +/- REGION_CAP_BP (per pleiodb)."""

    def _obs(self):
        # observed variants at 1.0/2.0/5.0 Mb with |z| = 3, 4, 2 (already sorted).
        pos = np.array([1_000_000, 2_000_000, 5_000_000], dtype=np.int64)
        absz = np.array([3.0, 4.0, 2.0], dtype=np.float64)
        return pos, absz

    def test_caps_overshoot_to_window_max(self):
        from opengwasdb.completion.block import _region_capped_z

        pos, absz = self._obs()
        # at 1.5 Mb, window +/-1 Mb -> [1.0, 2.0] Mb (|z| 3, 4) -> cap to 4
        assert _region_capped_z(10.0, 1_500_000, pos, absz) == pytest.approx(4.0)
        assert _region_capped_z(-10.0, 1_500_000, pos, absz) == pytest.approx(-4.0)

    def test_within_window_unchanged(self):
        from opengwasdb.completion.block import _region_capped_z

        pos, absz = self._obs()
        assert _region_capped_z(2.5, 1_500_000, pos, absz) == pytest.approx(2.5)

    def test_no_observed_in_window_uncapped(self):
        from opengwasdb.completion.block import _region_capped_z

        pos, absz = self._obs()
        # 10 Mb is >1 Mb from every observed variant -> no window -> unchanged
        assert _region_capped_z(9.9, 10_000_000, pos, absz) == pytest.approx(9.9)

    def test_window_excludes_distant_large_z(self):
        from opengwasdb.completion.block import _region_capped_z

        pos, absz = self._obs()
        # at 5 Mb, window includes only the 5 Mb observed (|z|=2) -> cap to 2,
        # NOT the larger |z|=4 at 2 Mb which is out of the +/-1 Mb window.
        assert _region_capped_z(8.0, 5_000_000, pos, absz) == pytest.approx(2.0)


class TestValidation:
    def test_valid_completed_store_passes(self, completed_store):
        result = validate_store(completed_store)
        assert result.ok, result.errors

    def test_observed_store_still_passes(self, observed_store):
        result = validate_store(observed_store)
        assert result.ok, result.errors

    def test_corrupt_off_panel_imputed_fails(self, completed_store):
        root = open_store(completed_store).arrays(mode="r+")
        on_panel = root["on_panel"][:]
        off_panel_rows = np.where(on_panel == 0)[0]
        assert len(off_panel_rows) > 0
        imputed = root["imputed"][:]
        imputed[off_panel_rows[0], 0] = 1
        root["imputed"][:] = imputed
        result = validate_store(completed_store)
        assert not result.ok

    def test_corrupt_imputed_nan_z_fails(self, completed_store):
        # An imputed=1 cell whose z is NaN must be caught by the streamed
        # finite-check band pass (issue 045), not just a full-matrix load. Mark
        # an on-panel cell imputed with a NaN z (on-panel so it isolates the
        # finite check from the off-panel-never-imputed check).
        root = open_store(completed_store).arrays(mode="r+")
        on_panel = root["on_panel"][:]
        on_panel_rows = np.where(on_panel == 1)[0]
        assert len(on_panel_rows) > 0
        r, c = int(on_panel_rows[0]), 0
        imputed = root["imputed"][:]
        imputed[r, c] = 1
        root["imputed"][:] = imputed
        # "Missing" is whatever this store's declared encoding marks it with
        # (spec §15) -- NaN for a float plane, the reserved sentinel for a
        # fixed-point one -- so write it through the codec rather than assuming.
        from opengwasdb.encoding import StoreCodec

        codec = StoreCodec(open_store(completed_store).manifest.encoding)
        z = root["z"][:]
        z[r, c] = codec.encode_z(np.array([np.nan]))[0]
        root["z"][:] = z
        result = validate_store(completed_store)
        assert not result.ok
        assert any("missing z" in e for e in result.errors), result.errors

    def test_corrupt_top_hit_imputed_index_fails(self, completed_store):
        from opengwasdb.layouts.dense.top_hits import threshold_key

        root = open_store(completed_store).arrays(mode="r+")
        group = root[f"top_hits/{threshold_key(5e-4)}"]
        assert "imputed" in group
        values = group["imputed"][:]
        assert len(values) > 0
        values[0] = 1 - values[0]
        group["imputed"][:] = values

        result = validate_store(completed_store)
        assert not result.ok
        assert any("imputed value inconsistent" in e for e in result.errors), result.errors


class TestSourceFidelity:
    """validate_store(source=...) cross-checks store values against the source.

    The observed store is same-assembly (GRCh38 in, GRCh38 store), so the join
    goes through the canonical-ALID path with no liftover — no bcftools needed.
    """

    def test_fidelity_passes_against_source(self, observed_store, source_path):
        result = validate_store(observed_store, source=source_path)
        assert result.ok, result.errors

    def test_fidelity_detects_sign_flip(self, observed_store, source_path):
        # Flip the sign of one observed, below-threshold cell: internal validation
        # still passes (|z| and the top-hit index are untouched), so only the
        # source-fidelity check can catch it.
        res = query_store(observed_store).lookup(["1:1000000:A:G"], ["a1"])
        assert len(res["z"]) == 1
        row, col = int(res["variant_index"][0]), int(res["analysis_index"][0])

        root = open_store(observed_store).arrays(mode="r+")
        z = root["z"][:]
        z[row, col] = -z[row, col]
        root["z"][:] = z

        assert validate_store(observed_store).ok  # internal checks still pass
        result = validate_store(observed_store, source=source_path)
        assert not result.ok
        assert any("source-fidelity" in e for e in result.errors), result.errors


class TestQuery:
    def test_analysis_a1_includes_imputed_by_default(self, completed_store):
        q = query_store(completed_store)
        result = q.analysis("a1")
        assert "association_status" in result
        statuses = set(result["association_status"].tolist())
        assert "observed" in statuses
        q.close()

    def test_analysis_observed_only_excludes_imputed(self, completed_store):
        q = query_store(completed_store)
        all_result = q.analysis("a1")
        obs_result = q.analysis("a1", observed_only=True)
        assert len(obs_result["z"]) <= len(all_result["z"])
        assert "imputed" not in set(obs_result["association_status"].tolist())
        q.close()

    def test_association_status_field_always_present(self, completed_store):
        q = query_store(completed_store)
        result = q.analysis("a1")
        assert len(result["association_status"]) == len(result["z"])
        q.close()

    def test_observed_store_has_status_observed_only(self, observed_store):
        q = query_store(observed_store)
        result = q.analysis("a1")
        assert set(result["association_status"].tolist()).issubset({"observed"})
        q.close()

    def test_top_hits_observed_only(self, completed_store):
        q = query_store(completed_store)
        result = q.top_hits(threshold=1.0, observed_only=True)
        assert "imputed" not in set(result["association_status"].tolist())
        q.close()

    def test_analysis_top_hits_match_global_subset_and_use_indexed_status(self, completed_store):
        q = query_store(completed_store)
        global_result = q.top_hits(threshold=5e-4)
        expected = global_result["analysis_index"] == 0

        def fail_imputed_pairs(rows, cols):  # noqa: ARG001
            raise AssertionError("selected top hits must use indexed imputed flags")

        q._imputed_pairs = fail_imputed_pairs
        selected = q.top_hits(analysis_id="a1", threshold=5e-4)
        observed = q.top_hits(analysis_id="a1", threshold=5e-4, observed_only=True)

        for name in ("variant_index", "analysis_index", "z", "se", "association_status"):
            np.testing.assert_array_equal(selected[name], global_result[name][expected])
        assert "imputed" not in set(observed["association_status"].tolist())
        q.close()

    def test_top_hit_index_stores_imputed_flags(self, completed_store):
        from opengwasdb.layouts.dense.top_hits import threshold_key

        root = open_store(completed_store).arrays(mode="r")
        group = root[f"top_hits/{threshold_key(5e-4)}"]
        assert "imputed" in group

        rows = group["variant_index"][:].astype(np.int64)
        cols = group["analysis_index"][:].astype(np.int64)
        indexed = group["imputed"][:].astype(np.uint8)
        gathered = root["imputed"].vindex[rows, cols].astype(np.uint8)
        np.testing.assert_array_equal(indexed, gathered)

    def test_context_manager(self, completed_store):
        with query_store(completed_store) as q:
            assert q.analyses_table()

    def test_top_hits_uses_indexed_imputed_flags(self, completed_store):
        q = query_store(completed_store)

        def fail_imputed_pairs(rows, cols):  # noqa: ARG001
            raise AssertionError("top_hits should read indexed imputed flags")

        q._imputed_pairs = fail_imputed_pairs
        result = q.top_hits(threshold=5e-4)
        assert len(result["association_status"]) == len(result["z"])
        q.close()

    def test_top_hits_uses_indexed_reference_frequency(self, completed_store):
        q = query_store(completed_store)
        expected = q.top_hits(threshold=5e-4)["eaf"]

        def fail_eaf_pairs(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("top_hits should read indexed frequencies")

        q._eaf_pairs = fail_eaf_pairs
        result = q.top_hits(threshold=5e-4)
        np.testing.assert_array_equal(result["eaf"], expected)
        q.close()

    def test_top_hits_does_not_open_frequency_plane(self, completed_store, monkeypatch):
        from opengwasdb.encoding import DenseEafPlane

        def fail_open(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("top_hits must not open the frequency plane")

        monkeypatch.setattr(DenseEafPlane, "open", fail_open)
        with query_store(completed_store) as q:
            result = q.top_hits(threshold=5e-4)
        assert len(result["eaf"]) > 0

    def test_range_phewas_reads_imputed_status_as_block(self, completed_store):
        root = open_store(completed_store).arrays(mode="r+")
        imputed = root["imputed"][:]
        imputed[0, 0] = 1
        root["imputed"][:] = imputed

        q = query_store(completed_store)

        def fail_imputed_pairs(rows, cols):  # noqa: ARG001
            raise AssertionError("range_phewas should read imputed flags as a block")

        q._imputed_pairs = fail_imputed_pairs
        result = q.range_phewas("1", 900_000, 1_100_000)
        assert len(result["association_status"]) == len(result["z"])
        assert "imputed" in set(result["association_status"].tolist())
        q.close()

    def test_top_hits_falls_back_without_indexed_imputed_flags(self, completed_store):
        from opengwasdb.layouts.dense.top_hits import threshold_key

        q = query_store(completed_store)
        indexed = q.top_hits(threshold=5e-4)
        q.close()

        root = open_store(completed_store).arrays(mode="r+")
        del root[f"top_hits/{threshold_key(5e-4)}"]["imputed"]

        q = query_store(completed_store)
        fallback = q.top_hits(threshold=5e-4)
        q.close()

        for name in ("variant_index", "analysis_index", "z", "se", "association_status"):
            np.testing.assert_array_equal(indexed[name], fallback[name])

    def test_top_hits_falls_back_without_indexed_frequency(self, completed_store):
        from opengwasdb.layouts.dense.top_hits import threshold_key

        q = query_store(completed_store)
        indexed = q.top_hits(threshold=5e-4)
        q.close()

        root = open_store(completed_store).arrays(mode="r+")
        del root[f"top_hits/{threshold_key(5e-4)}"]["eaf"]

        q = query_store(completed_store)
        fallback = q.top_hits(threshold=5e-4)
        q.close()
        np.testing.assert_array_equal(indexed["eaf"], fallback["eaf"])


class TestResume:
    def test_resume_completes_after_simulated_crash(
        self, tmp_path, observed_store, ld_panel, monkeypatch
    ):
        dst = tmp_path / "resumed.opengwasdb"
        calls = {"n": 0}
        real_run_block = complete_module._run_block

        def flaky_run_block(task):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("simulated crash")
            return real_run_block(task)

        monkeypatch.setattr(complete_module, "_run_block", flaky_run_block)
        with pytest.raises(RuntimeError, match="simulated crash"):
            complete_dense_store(observed_store, dst, ld_panel, ancestry="EUR", min_cor=0.0)

        checkpoint_dir = checkpoint_dir_for(dst)
        assert checkpoint_dir.exists()
        assert not dst.exists()
        assert not complete_module._work_dir_for(dst).exists()
        checkpointed = list((checkpoint_dir / "blocks").glob("*.npz"))
        assert len(checkpointed) == 1  # exactly one block finished before the crash

        monkeypatch.undo()
        result = resume_dense_completion(checkpoint_dir)
        assert dst.exists()
        assert not checkpoint_dir.exists()
        assert result.n_variants > 0

        assert validate_store(dst).ok

    def test_resume_matches_fresh_run(self, tmp_path, observed_store, ld_panel):
        fresh_dst = tmp_path / "fresh.opengwasdb"
        fresh = complete_dense_store(
            observed_store, fresh_dst, ld_panel, ancestry="EUR", min_cor=0.0
        )

        resumable_dst = tmp_path / "resumable.opengwasdb"
        checkpoint_dir = checkpoint_dir_for(resumable_dst)
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "blocks").mkdir()
        import json
        (checkpoint_dir / "build_params.json").write_text(
            json.dumps({
                "source_path": str(Path(observed_store).resolve()),
                "dest_path": str(resumable_dst.resolve()),
                "ld_dir": str(Path(ld_panel).resolve()),
                "ancestry": "EUR", "min_cor": 0.0, "thresh": 0.9,
                "release_id": None, "ld_panel_id": "eur-hg38-gpm",
            }),
            encoding="utf-8",
        )
        resumed = resume_dense_completion(checkpoint_dir)

        assert resumed.n_variants == fresh.n_variants
        assert resumed.n_imputed == fresh.n_imputed
        assert resumed.n_missing_off_panel == fresh.n_missing_off_panel
        assert resumed.n_missing_imputation_failed == fresh.n_missing_imputation_failed
        _assert_stores_identical(resumable_dst, fresh_dst)


class TestParallel:
    def test_two_workers_matches_serial(self, tmp_path, observed_store, ld_panel):
        serial_dst = tmp_path / "serial.opengwasdb"
        serial = complete_dense_store(
            observed_store, serial_dst, ld_panel, ancestry="EUR", min_cor=0.0, n_workers=1
        )
        parallel_dst = tmp_path / "parallel.opengwasdb"
        parallel = complete_dense_store(
            observed_store, parallel_dst, ld_panel, ancestry="EUR", min_cor=0.0, n_workers=2
        )
        assert parallel.n_variants == serial.n_variants
        assert parallel.n_imputed == serial.n_imputed
        assert parallel.n_missing_off_panel == serial.n_missing_off_panel
        assert parallel.n_missing_imputation_failed == serial.n_missing_imputation_failed
        _assert_stores_identical(parallel_dst, serial_dst)
        assert validate_store(parallel_dst).ok


class TestPanelArtifacts:
    """Which block artifacts the panel loader requires (ADR 0031, spec §13.1).

    Completion consumes eigenvectors only, so a block is loadable when it can
    produce them by either route. Requiring the LD matrix would make
    eigendecomposition-only panels silently empty rather than failing loudly.
    """

    @staticmethod
    def _block(tmp_path: Path, name: str, **kwargs) -> Path:
        _write_ld_block(
            tmp_path / "panel" / "EUR" / "1", name,
            [
                ("1:900000:A:G", 0.35, 900_000),
                ("1:950000:A:C", 0.32, 950_000),
                ("1:1000000:A:G", 0.30, 1_000_000),
                ("1:1050000:C:T", 0.40, 1_050_000),
            ],
            **kwargs,
        )
        return tmp_path / "panel" / "EUR" / "1" / f"{name}.tsv"

    def test_loads_with_eigendecomposition_only(self, tmp_path):
        tsv = self._block(tmp_path, "blk", write_matrix=False, write_npz=True)
        block = load_block(tsv)
        assert block is not None
        assert block.ld_path is None
        vals, vecs = load_ld_eigenvectors(block, thresh=0.9)
        assert vecs.shape[0] == 4
        assert vecs.shape[1] >= 1

    def test_loads_with_matrix_only(self, tmp_path):
        tsv = self._block(tmp_path, "blk", write_matrix=True, write_npz=False)
        block = load_block(tsv)
        assert block is not None
        assert block.ldeig_npz_path is None
        vals, vecs = load_ld_eigenvectors(block, thresh=0.9)
        assert vecs.shape[0] == 4

    def test_loads_with_both(self, tmp_path):
        tsv = self._block(tmp_path, "blk", write_matrix=True, write_npz=True)
        block = load_block(tsv)
        assert block is not None
        assert block.ld_path is not None and block.ldeig_npz_path is not None

    def test_skips_when_neither_present(self, tmp_path):
        tsv = self._block(tmp_path, "blk", write_matrix=False, write_npz=False)
        assert load_block(tsv) is None

    def test_under_resolved_block_warns_and_returns_stored(self, tmp_path, caplog):
        """A stored component count too small for the requested variance must be
        surfaced — silently returning fewer components degrades imputation with no
        signal (spec §13.1)."""
        tsv = self._block(tmp_path, "blk", write_matrix=False, write_npz=True, npz_k=1)
        block = load_block(tsv)
        with caplog.at_level("WARNING"):
            vals, vecs = load_ld_eigenvectors(block, thresh=0.99)
        assert vecs.shape[1] == 1
        assert "under-resolved" in caplog.text
        assert block.block_id in caplog.text

    def test_sufficient_components_do_not_warn(self, tmp_path, caplog):
        tsv = self._block(tmp_path, "blk", write_matrix=False, write_npz=True)
        block = load_block(tsv)
        with caplog.at_level("WARNING"):
            load_ld_eigenvectors(block, thresh=0.9)
        assert "under-resolved" not in caplog.text


class TestPanelIdentifiers:
    """Panel SNP ids normalise to canonical store ALIDs regardless of convention."""

    def test_legacy_underscore_form(self):
        assert canonical_panel_alid("22:17238320_A_G") == "22:17238320:A:G"

    def test_canonical_form_passes_through(self):
        assert canonical_panel_alid("22:17238320:A:G") == "22:17238320:A:G"

    def test_chr_prefix_stripped(self):
        assert canonical_panel_alid("chr22:17238320_A_G") == "22:17238320:A:G"

    def test_orientation_canonicalised(self):
        """A panel entry recorded in the opposite allele order must resolve to the
        same variant, or imputed effect directions would silently invert."""
        assert canonical_panel_alid("22:17238320_G_A") == canonical_panel_alid(
            "22:17238320_A_G"
        )

    def test_unparseable_returns_none(self):
        assert canonical_panel_alid("rs12345") is None
        assert canonical_panel_alid("") is None

    def test_position_from_either_form(self):
        assert snp_position("22:17238320_A_G") == 17238320
        assert snp_position("22:17238320:A:G") == 17238320
        assert snp_position("rs12345") == -1


class TestPanelFormatIndependence:
    """Completion must not depend on how a panel stores LD or names its variants.

    The production EUR panel uses legacy ``chr:pos_ref_alt`` ids and ships LD
    matrices; panels built under ADR 0031 use canonical ALIDs and ship only
    eigendecompositions. Both must yield the same store — otherwise the storage
    contract silently changes results.
    """

    @staticmethod
    def _panel(root: Path, **kwargs) -> Path:
        _write_ld_block(
            root / "EUR" / "1", "900000-1300000",
            [
                ("1:900000:A:G", 0.35, 900_000),
                ("1:950000:A:C", 0.32, 950_000),
                ("1:1000000:A:G", 0.30, 1_000_000),
                ("1:1050000:C:T", 0.40, 1_050_000),
                ("1:1100000:A:C", 0.45, 1_100_000),
            ],
            seed=0, **kwargs,
        )
        _write_ld_block(
            root / "EUR" / "2", "1-500000",
            [
                ("2:100000:A:G", 0.20, 100_000),
                ("2:200000:C:G", 0.25, 200_000),
            ],
            seed=1, **kwargs,
        )
        return root

    def test_eigendecomposition_only_matches_matrix_panel(
        self, tmp_path, observed_store, completed_store
    ):
        """npz-only, legacy-id panel produces a store identical to the canonical
        matrix-backed fixture — same variants, same imputed/missing counts, same
        arrays."""
        root = self._panel(
            tmp_path / "legacy_panel",
            write_matrix=False, write_npz=True, snp_id_style="legacy",
        )
        dst = tmp_path / "legacy.opengwasdb"
        result = complete_dense_store(
            observed_store, dst, root,
            ancestry="EUR", min_cor=0.0, release_id="comp-v1",
        )
        assert validate_store(dst).ok
        assert result.n_variants == 8
        _assert_stores_identical(dst, completed_store)


# AR(1)-correlated block (rho=0.9): unlike the iid-random matrix _write_ld_block
# uses elsewhere, its leading eigenvectors are smooth basis functions, so a smooth
# z profile projects onto a handful of components and elastic-net can actually
# recover a held-out point. SOURCE_ROWS above never gives ElasticNetCV enough
# signal to fit anything but zero, so TestPanelFormatIndependence's byte-identical
# check is silently comparing two all-missing outputs — real proof that
# eigendecomposition-only panels still produce genuine imputed cells needs its
# own fixture.
_SIGNAL_N = 12
_SIGNAL_MISSING_IDX = 6  # left out of the observed source rows; the imputation target
_SIGNAL_POSITIONS = [100_000 * (i + 1) for i in range(_SIGNAL_N)]
_SIGNAL_Z_TRUE = np.sin(np.linspace(0, 3, _SIGNAL_N)) * 5 + np.arange(_SIGNAL_N) * 0.1


@pytest.fixture
def signal_source_path(tmp_path: Path) -> Path:
    """One analysis, one AR(1) block, every position observed except the target."""
    rows = [SOURCE_HEADER]
    for i, (pos, z) in enumerate(zip(_SIGNAL_POSITIONS, _SIGNAL_Z_TRUE, strict=True)):
        if i == _SIGNAL_MISSING_IDX:
            continue
        rows.append(
            f"a1\tp1\tHeight\tHeight primary\t1\t{pos}\tA\tG\t{z:.6f}\t0.2\trs{i}\tsd"
        )
    path = tmp_path / "signal_associations.tsv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def signal_observed_store(tmp_path: Path, signal_source_path: Path) -> Path:
    out = tmp_path / "signal_obs.opengwasdb"
    build_dense_observed_from_sources(
        [signal_source_path], out,
        store_id="test", release_id="obs-v1", reference_assembly="GRCh38",
    )
    return out


@pytest.fixture
def signal_panel_npz_only(tmp_path: Path) -> Path:
    """A single eigendecomposition-only block (no .unphased.vcor1.gz at all)
    covering every position in `signal_source_path`, with genuine AR(1) LD
    structure so imputation of the held-out position actually succeeds."""
    root = tmp_path / "signal_panel"
    snps = [
        (f"1:{pos}:A:G", 0.3, pos) for pos in _SIGNAL_POSITIONS
    ]
    block_dir = root / "EUR" / "1"
    block_dir.mkdir(parents=True, exist_ok=True)
    tsv_lines = ["CHR\tSNP\tOA\tEA\tEAF\tBP"]
    for alid, eaf, bp in snps:
        chrom, _pos, a1, a2 = alid.split(":")
        tsv_lines.append(f"{chrom}\t{alid}\t{a2}\t{a1}\t{eaf}\t{bp}")
    (block_dir / "100000-1200000.tsv").write_text("\n".join(tsv_lines) + "\n")

    idx = np.arange(_SIGNAL_N)
    ld = 0.9 ** np.abs(idx[:, None] - idx[None, :])
    vals, vecs = np.linalg.eigh(ld)
    vals, vecs = vals[::-1], vecs[:, ::-1]
    np.savez_compressed(block_dir / "100000-1200000.ldeig.npz", values=vals, vectors=vecs)
    return root


class TestEigendecompositionOnlyPanelImputesRealCells:
    """opengwasdb#10: a panel shipping only .ldeig.npz must complete end to end
    and actually impute the held-out position, not merely match a fixture whose
    own imputation happens to fail identically on both sides."""

    def test_completes_with_imputed_cells(
        self, tmp_path, signal_observed_store, signal_panel_npz_only
    ):
        for chrom_dir in (signal_panel_npz_only / "EUR").iterdir():
            for f in chrom_dir.iterdir():
                assert not f.name.endswith(".unphased.vcor1.gz"), (
                    "fixture must not write LD matrices — that's the point of this test"
                )

        dst = tmp_path / "signal_comp.opengwasdb"
        complete_dense_store(
            signal_observed_store, dst, signal_panel_npz_only,
            ancestry="EUR", min_cor=0.0, release_id="comp-signal-v1",
        )

        root = open_store(dst).arrays(mode="r")
        assert root["imputed"][:].sum() > 0


# ── The declaration and the arrays are two accounts of the same cells ────────
#
# `analyses.tsv` says how many cells an Analysis gained; `data.zarr/imputed`
# holds them. Nothing made the two agree, and the ancestry-match filter made
# them disagree in a way no query would show: the excluded Analysis reads
# observed-only and its metadata claimed otherwise (ADR 0028, ADR 0037 §4).
#
# These use the signal fixtures because the small `completed_store` imputes
# nothing at all -- against it every rule below is vacuously satisfied.


@pytest.fixture
def imputing_completed_store(
    tmp_path: Path, signal_observed_store: Path, signal_panel_npz_only: Path
) -> Path:
    """A completed store that actually gained cells, and says so.

    Not a *valid* store: its source carries EAF with no orientation evidence,
    so `validate_store` objects for an unrelated reason. The tests below
    therefore assert on the presence of their own error rather than on
    `result.ok`, and check it is absent first.
    """
    dst = tmp_path / "signal_declared.opengwasdb"
    complete_dense_store(
        signal_observed_store, dst, signal_panel_npz_only,
        ancestry="EUR", min_cor=0.0, release_id="comp-signal-v1",
    )
    assert open_store(dst).arrays(mode="r")["imputed"][:].sum() > 0, (
        "nothing was imputed, so nothing below is under test"
    )
    assert int(read_analyses(dst / "analyses.tsv").rows[0]["completion_n_imputed_total"]) > 0
    return dst


def _rewrite_analyses(store: Path, **columns: str) -> None:
    """Overwrite `columns` on the first Analysis row of a store."""
    table = read_analyses(store / "analyses.tsv")
    table.rows[0].update(columns)
    write_analyses(store / "analyses.tsv", table)


def _matching(store: Path, *fragments: str) -> list[str]:
    return [
        e for e in validate_store(store).errors if all(f in e for f in fragments)
    ]


def test_validator_rejects_imputed_cells_declared_but_not_held(imputing_completed_store: Path):
    """A count of imputed cells the arrays do not contain."""
    fragments = ("declares completion_n_imputed_total", "no imputed=1 cell")
    assert not _matching(imputing_completed_store, *fragments)

    arrays = open_store(imputing_completed_store).arrays(mode="a")
    band = arrays["imputed"][:, :]
    band[:, 0] = 0
    arrays["imputed"][:, :] = band

    assert _matching(imputing_completed_store, *fragments)


def test_validator_rejects_imputed_cells_held_but_not_declared(imputing_completed_store: Path):
    """The same disagreement the other way round: cells nothing accounts for."""
    fragment = "imputed cells its metadata does not account for"
    assert not _matching(imputing_completed_store, fragment)

    _rewrite_analyses(imputing_completed_store, completion_n_imputed_total="0")

    assert _matching(imputing_completed_store, fragment)


def test_validator_rejects_a_completion_count_without_a_completion(
    imputing_completed_store: Path,
):
    """`completed_against` blank and a nonzero count: not completed, yet counted.

    Read from `analyses.tsv` alone, so it holds for Ragged and Hybrid too,
    where there is no dense `imputed` matrix to compare against.
    """
    assert not _matching(imputing_completed_store, "completed_against is blank")

    _rewrite_analyses(imputing_completed_store, completed_against="")

    assert _matching(imputing_completed_store, "completed_against is blank")
