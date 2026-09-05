"""Integration tests for ragged reference completion (issues 039-042)."""

from __future__ import annotations

import gzip
import io
import struct
from pathlib import Path

import numpy as np
import pytest

from opengwasdb.layouts.dense.top_hits import threshold_key
from opengwasdb.layouts.ragged.build_besd import build_ragged_from_besd
from opengwasdb.layouts.ragged.complete import complete_ragged_store
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRReader
from opengwasdb.query import query_store
from opengwasdb.store.open import open_store
from opengwasdb.validation.validate import validate_store

# ── Synthetic BESD fixture (reused from test_ragged_build_besd.py) ─────────

def _write_esi(path: Path, snps: list[dict]) -> None:
    with open(path, "w") as fh:
        for s in snps:
            fh.write(f"{s['chr']}\t{s['snp_id']}\t0\t{s['bp']}\t{s['a1']}\t{s['a2']}\tNA\n")


def _write_epi(path: Path, probes: list[dict]) -> None:
    with open(path, "w") as fh:
        for p in probes:
            fh.write(f"{p['chr']}\t{p['probe_id']}\t0\t{p['bp']}\t{p.get('gene', 'NA')}\t+\n")


def _write_besd_sparse_3f(path: Path, n_probes: int, probe_assocs: list[list[tuple]]) -> None:
    rowid, val, cols = [], [], []
    offset = 0
    for assocs in probe_assocs:
        n = len(assocs)
        cols.append(offset)
        cols.append(offset + n)
        for snp_idx, beta, _ in assocs:
            rowid.append(snp_idx); val.append(beta)
        for _, _, se in assocs:
            rowid.append(0); val.append(se)
        offset += 2 * n
    cols.append(offset)
    col_num = (n_probes << 1) + 1
    with open(path, "wb") as fh:
        fh.write(struct.pack("<I", 0x40400000))
        fh.write(struct.pack("<Q", len(val)))
        fh.write(struct.pack(f"<{col_num}Q", *cols))
        fh.write(struct.pack(f"<{len(val)}I", *rowid))
        fh.write(struct.pack(f"<{len(val)}f", *val))


def _make_besd_fixture(tmp_path: Path) -> Path:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    snps = [
        {"chr": "1", "snp_id": "rs1001", "bp": 1_000_000, "a1": "A", "a2": "G"},
        {"chr": "1", "snp_id": "rs1002", "bp": 1_100_000, "a1": "C", "a2": "T"},
        {"chr": "1", "snp_id": "rs1003", "bp": 1_200_000, "a1": "A", "a2": "C"},
    ]
    probes = [
        {"chr": "1", "probe_id": "ENSG00000000001", "bp": 1_050_000, "gene": "GENE1"},
        {"chr": "1", "probe_id": "ENSG00000000002", "bp": 1_150_000, "gene": "GENE2"},
    ]
    probe_assocs = [
        [(0, 0.1, 0.02), (1, -0.2, 0.03)],
        [(1, 0.5, 0.05), (2, -0.15, 0.025)],
    ]
    _write_esi(fixture / "test.esi", snps)
    _write_epi(fixture / "test.epi", probes)
    _write_besd_sparse_3f(fixture / "test.besd", len(probes), probe_assocs)
    return fixture / "test"


# ── Synthetic LD panel fixture ──────────────────────────────────────────────

def _make_ld_panel(tmp_path: Path, chrom: str, start: int, end: int) -> Path:
    """Create a tiny synthetic LD panel block directory (flat layout)."""
    block_name = f"{start}-{end}"
    panel_dir = tmp_path / "ld_panel" / "EUR" / chrom
    panel_dir.mkdir(parents=True)

    # Two reference-panel variants, one of which is NOT in the observed store
    ref_snps = [
        ("1:1000000_A_G", 0.3, 1_000_000),   # same as rs1001
        ("1:1050500_C_T", 0.4, 1_050_500),   # new — not in observed store
        ("1:1100000_C_T", 0.45, 1_100_000),  # same as rs1002
    ]
    n = len(ref_snps)

    tsv_lines = ["CHR\tSNP\tOA\tEA\tEAF\tBP"]
    for alid, eaf, bp in ref_snps:
        chrom_, rest = alid.split(":")
        bp_str = rest.split("_")[0]
        a1, a2 = rest.split("_")[1], rest.split("_")[2]
        tsv_lines.append(f"{chrom_}\t{alid}\t{a2}\t{a1}\t{eaf}\t{bp_str}")
    (panel_dir / f"{block_name}.tsv").write_text("\n".join(tsv_lines) + "\n")

    # Random positive-definite LD matrix
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n))
    ld = A @ A.T + np.eye(n) * n * 0.1

    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        for row in ld:
            gz.write(("\t".join(f"{v:.6f}" for v in row) + "\n").encode())
    (panel_dir / f"{block_name}.unphased.vcor1.gz").write_bytes(buf.getvalue())

    return tmp_path / "ld_panel"


# ── Tests ───────────────────────────────────────────────────────────────────

@pytest.fixture
def observed_store(tmp_path):
    prefix = _make_besd_fixture(tmp_path)
    out = tmp_path / "obs.opengwasdb"
    build_ragged_from_besd(prefix, out, store_id="test", release_id="obs-v1", tissue="Blood")
    return out


@pytest.fixture
def ld_panel(tmp_path):
    return _make_ld_panel(tmp_path, "1", 900_000, 1_300_000)


@pytest.fixture
def completed_store(tmp_path, observed_store, ld_panel):
    dst = tmp_path / "comp.opengwasdb"
    complete_ragged_store(
        observed_store, dst, ld_panel,
        ancestry="EUR", cis_window_bp=500_000, min_cor=0.0, release_id="comp-v1",
    )
    return dst


class TestTopHitCountsRefreshedByCompletion:
    """Issue #73: completion recomputes Top-Hit Counts from the post-
    completion top-hit index rather than carrying forward pre-completion
    counts, which would go stale once imputation changes the association
    list a threshold is evaluated against."""

    def test_completed_top_hit_counts_match_completed_top_hit_index(self, completed_store):
        from opengwasdb.layouts.dense.top_hits import read_top_hit_counts
        from opengwasdb.model.analyses import read_analyses

        table = read_analyses(completed_store / "analyses.tsv")
        rows = sorted(table.rows, key=lambda r: int(r["analysis_index"]))
        expected = read_top_hit_counts(completed_store, len(rows))
        for column in ("n_hits_5e8", "n_hits_5e6", "n_hits_5e4"):
            assert [int(r[column]) for r in rows] == expected[column]

    def test_completed_top_hit_counts_visible_via_query_facade(self, completed_store):
        from opengwasdb.query import query_store

        q = query_store(completed_store)
        analyses = q.analyses_table()
        q.close()
        assert any(int(row["n_hits_5e4"]) > 0 for row in analyses.values())

    def test_completion_overwrites_stale_pre_completion_counts(
        self, tmp_path, observed_store, ld_panel
    ):
        """A source analyses.tsv carrying an obviously wrong (stale)
        pre-completion count must not survive into the completed store --
        add_hit_counts recomputes it from the completed store's own
        top-hit index rather than adding onto whatever the source said."""
        from opengwasdb.model.analyses import read_analyses, write_analyses

        analyses_path = observed_store / "analyses.tsv"
        table = read_analyses(analyses_path)
        rows = [{**row, "n_hits_5e4": "999"} for row in table.rows]
        write_analyses(analyses_path, type(table)(fieldnames=table.fieldnames, rows=tuple(rows)))

        dst = tmp_path / "comp_stale.opengwasdb"
        complete_ragged_store(
            observed_store, dst, ld_panel, ancestry="EUR", cis_window_bp=500_000, min_cor=0.0,
        )

        dst_rows = read_analyses(dst / "analyses.tsv").rows
        assert all(int(r["n_hits_5e4"]) != 999 for r in dst_rows)


class TestCompletionRollupColumns:
    """Issue #70: analyses.tsv's completion-rollup columns
    (completion_median_pearson_r, completion_n_imputed_total,
    completion_n_missing_total) are populated per Analysis, cross-checked
    against an independent read of the completion_quality SQLite table
    (which stays SQLite-only per ADR 0030 -- only the rollup travels into
    analyses.tsv)."""

    def test_rollup_columns_match_completion_quality_table(self, completed_store):
        import sqlite3

        from opengwasdb.model.analyses import read_analyses

        table = read_analyses(completed_store / "analyses.tsv")
        rows = {int(r["analysis_index"]): r for r in table.rows}
        for column in (
            "completion_median_pearson_r", "completion_n_imputed_total",
            "completion_n_missing_total",
        ):
            assert column in table.fieldnames

        conn = sqlite3.connect(str(completed_store / "index.sqlite"))
        conn.row_factory = sqlite3.Row
        try:
            for ai, row in rows.items():
                quality_rows = conn.execute(
                    "SELECT pearson_r, n_imputed, n_missing FROM completion_quality "
                    "WHERE analysis_index = ?",
                    (ai,),
                ).fetchall()
                expected_imputed = sum(int(r["n_imputed"]) for r in quality_rows)
                expected_missing = sum(int(r["n_missing"]) for r in quality_rows)
                assert int(row["completion_n_imputed_total"]) == expected_imputed
                assert int(row["completion_n_missing_total"]) == expected_missing

                r_values = [
                    float(r["pearson_r"]) for r in quality_rows if r["pearson_r"] is not None
                ]
                if r_values:
                    assert row["completion_median_pearson_r"] != ""
                    assert float(row["completion_median_pearson_r"]) == pytest.approx(
                        float(np.median(r_values))
                    )
                else:
                    assert row["completion_median_pearson_r"] == ""
        finally:
            conn.close()

    def test_completed_against_recorded_when_ancestry_matches(self, completed_store):
        from opengwasdb.model.analyses import read_analyses

        rows = read_analyses(completed_store / "analyses.tsv").rows
        # This fixture's build carries no assigned_ancestry column, so
        # derive_impute_analysis_ids() returns None (impute everything) and
        # every Analysis is completed against the requested panel ancestry.
        assert {r["completed_against"] for r in rows} == {"EUR"}


class TestCompletionFiles:
    def test_creates_store_directory(self, completed_store):
        assert completed_store.exists()
        assert (completed_store / "manifest.json").exists()
        assert (completed_store / "variants.tsv.gz").exists()
        assert (completed_store / "analyses.tsv").exists()
        assert (completed_store / "index.sqlite").exists()

    def test_does_not_carry_forward_traits_tsv(self, completed_store):
        # traits.tsv.gz (issue 034) is retired (issue #69): Reference
        # Completion no longer copies it forward, since analyses.tsv is the
        # sole source of truth for a completed Analysis's Trait position too.
        assert not (completed_store / "traits.tsv.gz").exists()

    def test_manifest_completion_state(self, completed_store):
        import json
        m = json.loads((completed_store / "manifest.json").read_text())
        assert m["completion_state"] == "reference_completed"
        assert "completion" in m["provenance"]

    def test_imputed_array_present(self, completed_store):
        root = open_store(completed_store).arrays(mode="r")["ragged"]
        assert "imputed" in root

    def test_imputed_array_aligned_with_z(self, completed_store):
        root = open_store(completed_store).arrays(mode="r")["ragged"]
        assert len(root["imputed"]) == len(root["z"])

    def test_imputed_values_are_0_or_1(self, completed_store):
        root = open_store(completed_store).arrays(mode="r")["ragged"]
        imp = root["imputed"][:]
        assert np.all((imp == 0) | (imp == 1))

    def test_imputed_1_rows_have_finite_z(self, completed_store):
        root = open_store(completed_store).arrays(mode="r")["ragged"]
        z = root["z"][:].astype("float32")
        imp = root["imputed"][:]
        assert np.all(np.isfinite(z[imp == 1]))

    def test_variant_table_larger_than_source(self, completed_store, observed_store):
        from opengwasdb.variants.axis import VariantAxis
        obs_ax = VariantAxis(observed_store)
        comp_ax = VariantAxis(completed_store)
        assert comp_ax.n_variants >= obs_ax.n_variants
        obs_ax.close()
        comp_ax.close()

    def test_completion_keeps_the_observed_store_rsids(self, completed_store, observed_store):
        """Issue #109: completion rewrote variants.tsv.gz and passed no rsids,
        so every Reference-Completed Ragged store silently lost them -- the
        eqtlgen-cis pilot went from 49,967 named rows to none. Rows the panel
        adds are genuinely unnamed and stay blank."""
        from opengwasdb.variants.axis import VariantAxis

        obs_ax = VariantAxis(observed_store)
        comp_ax = VariantAxis(completed_store)
        try:
            observed_rsids = {r.alid: r.rsid for r in obs_ax.all() if r.rsid}
            assert observed_rsids, "fixture must name some variants for this to mean anything"
            completed_rsids = {r.alid: r.rsid for r in comp_ax.all() if r.rsid}
            assert completed_rsids == observed_rsids
            alid, rsid = next(iter(observed_rsids.items()))
            record = comp_ax.by_identifier(rsid)
            assert record is not None and record.alid == alid
        finally:
            obs_ax.close()
            comp_ax.close()

    def test_completion_does_not_lose_observed_eaf(self, tmp_path, ld_panel):
        """Completion rewrites the CSR, so observed EAF has to be carried across
        it explicitly (ADR 0036) -- the same way rsids do (issue #109). The BESD
        fixture has no frequencies at all, so this builds an SSF store that
        does, rather than asserting a vacuous equality on empty arrays."""
        import gzip

        from opengwasdb.layouts.ragged.build_ssf import build_ragged_from_ssf
        from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRReader

        filtered_dir = tmp_path / "filtered"
        filtered_dir.mkdir()
        with gzip.open(filtered_dir / "a.tsv.gz", "wt", encoding="utf-8") as fh:
            fh.write(
                "chromosome\tbase_pair_location\teffect_allele\tother_allele"
                "\tbeta\tstandard_error\teffect_allele_frequency\n"
            )
            fh.write("1\t1000000\tA\tG\t0.1\t0.02\t0.3\n")
            fh.write("1\t1100000\tC\tT\t-0.2\t0.03\t0.45\n")
        manifest = tmp_path / "ssf_manifest.tsv"
        manifest.write_text(
            "analysis_index\tanalysis_id\tfiltered_file\tn\tassigned_ancestry\n"
            "0\tanalysis_a\ta.tsv.gz\t1000\tEUR\n",
            encoding="utf-8",
        )
        observed = tmp_path / "ssf_obs.opengwasdb"
        build_ragged_from_ssf(manifest, filtered_dir, observed, store_id="t", release_id="obs")

        obs_reader = RaggedCSRReader(observed)
        observed_eaf = {
            int(vi): float(e)
            for vi, e in zip(
                obs_reader.get_analysis(0).variant_index,
                obs_reader.get_analysis(0).eaf,
                strict=True,
            )
            if np.isfinite(e)
        }
        assert observed_eaf, "fixture must carry EAF for this to mean anything"

        completed = tmp_path / "ssf_comp.opengwasdb"
        complete_ragged_store(
            observed, completed, ld_panel,
            ancestry="EUR", cis_window_bp=500_000, min_cor=0.0, release_id="comp",
        )

        from opengwasdb.variants.axis import VariantAxis

        obs_axis, comp_axis = VariantAxis(observed), VariantAxis(completed)
        try:
            obs_alid = {r.variant_index: r.alid for r in obs_axis.all()}
            comp_alid = {r.variant_index: r.alid for r in comp_axis.all()}
        finally:
            obs_axis.close()
            comp_axis.close()

        comp = RaggedCSRReader(completed).get_analysis(0)
        completed_eaf = {
            comp_alid[int(vi)]: float(e)
            for vi, e in zip(comp.variant_index, comp.eaf, strict=True)
            if np.isfinite(e)
        }
        assert completed_eaf == {obs_alid[vi]: e for vi, e in observed_eaf.items()}

    def test_completion_quality_table_exists(self, completed_store):
        import sqlite3
        conn = sqlite3.connect(str(completed_store / "index.sqlite"))
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        conn.close()
        assert "completion_quality" in tables

    def test_overwrite_raises_without_flag(self, tmp_path, observed_store, ld_panel, completed_store):
        with pytest.raises(FileExistsError):
            complete_ragged_store(observed_store, completed_store, ld_panel, min_cor=0.0)


class TestPanelAlidCanonicalisation:
    """Before the fix, ragged completion resolved panel ids with a bare
    ``chr``-prefix strip instead of ``canonical_panel_alid()``, so an
    underscore-form id (``chr:pos_ref_alt``, the production UKB EUR panel's
    convention) never matched the colon-delimited canonical variant table --
    a panel-only variant was silently dropped from the completed store
    rather than imputed. The ``ld_panel``/``completed_store`` fixtures above
    already use underscore-form ids, but their block is too small (one
    unobserved target, two observed points) for ``poly_rescale``'s degree-3
    fit ever to succeed regardless of the alid bug, so that gap alone
    wouldn't be caught by a test asserting actual imputation. This test uses
    a block large enough (7 observed points) for imputation to genuinely
    succeed, so the assertion demonstrates the fix rather than an
    unreachable target (issue #52).
    """

    @staticmethod
    def _write_underscore_panel(panel_dir: Path, snps: list[tuple[int, str, str, float]]) -> None:
        panel_dir.mkdir(parents=True)
        tsv_lines = ["CHR\tSNP\tOA\tEA\tEAF\tBP"]
        for bp, a1, a2, eaf in snps:
            snp_id = f"1:{bp}_{a1}_{a2}"  # production-panel underscore form
            tsv_lines.append(f"1\t{snp_id}\t{a2}\t{a1}\t{eaf}\t{bp}")
        (panel_dir / "block.tsv").write_text("\n".join(tsv_lines) + "\n")

        n = len(snps)
        rng = np.random.default_rng(0)
        A = rng.standard_normal((n, n))
        ld = A @ A.T + np.eye(n) * n * 0.1
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
            for row in ld:
                gz.write(("\t".join(f"{v:.6f}" for v in row) + "\n").encode())
        (panel_dir / "block.unphased.vcor1.gz").write_bytes(buf.getvalue())

    def test_new_panel_variant_in_underscore_form_is_imputed(self, tmp_path):
        from opengwasdb.variants import parse_canonical_alid
        from opengwasdb.variants.axis import VariantAxis

        # 7 observed positions (alleles already alphabetically A1<A2, so no
        # orientation flip) plus one new, unobserved target at 1050000.
        observed_bp = [800_000, 850_000, 900_000, 950_000, 1_000_000, 1_100_000, 1_150_000]
        z_values = [1.0, -1.2, 2.0, -0.5, 1.5, -2.0, 0.8]
        se = 0.15

        snps = [
            {"chr": "1", "snp_id": f"rs{i}", "bp": bp, "a1": "A", "a2": "G"}
            for i, bp in enumerate(observed_bp)
        ]
        probes = [{"chr": "1", "probe_id": "ENSG00000000099", "bp": 975_000, "gene": "GENE99"}]
        probe_assocs = [[(i, z * se, se) for i, z in enumerate(z_values)]]

        fixture = tmp_path / "fixture"
        fixture.mkdir()
        _write_esi(fixture / "test.esi", snps)
        _write_epi(fixture / "test.epi", probes)
        _write_besd_sparse_3f(fixture / "test.besd", len(probes), probe_assocs)

        observed_store = tmp_path / "obs.opengwasdb"
        build_ragged_from_besd(
            fixture / "test", observed_store, store_id="test", release_id="obs-v1", tissue="Blood"
        )

        panel_dir = tmp_path / "ld_panel_us" / "EUR" / "1"
        self._write_underscore_panel(
            panel_dir,
            [(bp, "A", "G", 0.35) for bp in observed_bp] + [(1_050_000, "C", "T", 0.40)],
        )

        completed_store = tmp_path / "comp.opengwasdb"
        complete_ragged_store(
            observed_store, completed_store, tmp_path / "ld_panel_us",
            ancestry="EUR", cis_window_bp=500_000, min_cor=0.0, release_id="comp-v1",
        )

        comp_ax = VariantAxis(completed_store)
        rec = comp_ax.by_alid(parse_canonical_alid("1:1050000:C:T"))
        comp_ax.close()
        assert rec is not None, "new panel variant missing from the completed variant table"

        q = query_store(completed_store)
        result = q.analysis("ENSG00000000099::Blood")
        q.close()

        idx = np.where(result["variant_index"] == rec.variant_index)[0]
        assert len(idx) == 1, "new panel variant missing from the completed association list"
        assert result["association_status"][idx[0]] == "imputed"
        assert np.isfinite(result["z"][idx[0]])


_SSF_HEADER = [
    "chromosome", "base_pair_location", "effect_allele", "other_allele",
    "beta", "standard_error", "rsid", "variant_id",
]


def _write_ssf_filtered(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\t".join(_SSF_HEADER) + "\n")
        for row in rows:
            fh.write("\t".join(str(row.get(col, "")) for col in _SSF_HEADER) + "\n")


def _write_ssf_manifest(path: Path, rows: list[dict]) -> None:
    header = [
        "analysis_index", "analysis_id", "trait_id",
        "analysis_label", "trait_ontology_id", "trait_ontology_label",
        "trait_chr", "trait_bp", "n", "tissue", "context", "mhc", "filtered_file",
        "assigned_ancestry",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for row in rows:
            fh.write("\t".join(str(row.get(col, "")) for col in header) + "\n")


class TestAncestryMatchedRaggedCompletion:
    """Before the fix, complete_ragged_store had no impute_analysis_ids
    parameter at all, and even once added, the derivation was inert against
    real ragged sources: neither ragged builder's SQLite analyses schema
    carried assigned_ancestry, so derive_impute_analysis_ids always saw no
    ancestry information and imputed everything. build_ragged_from_ssf now
    reads an optional assigned_ancestry manifest column (mirroring dense's
    build_vcf.py), and complete_ragged_store derives the filter from it --
    this test proves an EUR panel actually imputes an EUR-assigned Analysis
    while leaving an AFR-assigned one observed-only, and that
    completed_against is recorded per Analysis (issue #52).
    """

    def test_only_matching_ancestry_analysis_is_imputed(self, tmp_path):
        import sqlite3

        from opengwasdb.layouts.ragged.build_ssf import build_ragged_from_ssf
        from opengwasdb.model.analyses import read_analyses
        from opengwasdb.variants import parse_canonical_alid
        from opengwasdb.variants.axis import VariantAxis

        observed_bp = [800_000, 850_000, 900_000, 950_000, 1_000_000, 1_100_000, 1_150_000]
        z_values = [1.0, -1.2, 2.0, -0.5, 1.5, -2.0, 0.8]
        se = 0.15

        filtered_dir = tmp_path / "filtered"
        filtered_dir.mkdir()
        for analysis_id in ("eur_trait", "afr_trait"):
            _write_ssf_filtered(filtered_dir / f"{analysis_id}.tsv.gz", [
                {
                    "chromosome": "1", "base_pair_location": bp,
                    "effect_allele": "A", "other_allele": "G",
                    "beta": z * se, "standard_error": se, "rsid": f"rs{bp}",
                }
                for bp, z in zip(observed_bp, z_values, strict=True)
            ])

        manifest = tmp_path / "manifest.tsv"
        _write_ssf_manifest(manifest, [
            {"analysis_index": 0, "analysis_id": "eur_trait", "trait_id": "T1",
             "trait_chr": "1", "trait_bp": 975_000, "n": 5000, "mhc": "FALSE",
             "filtered_file": "eur_trait.tsv.gz", "assigned_ancestry": "EUR"},
            {"analysis_index": 1, "analysis_id": "afr_trait", "trait_id": "T2",
             "trait_chr": "1", "trait_bp": 975_000, "n": 5000, "mhc": "FALSE",
             "filtered_file": "afr_trait.tsv.gz", "assigned_ancestry": "AFR"},
        ])

        observed_store = tmp_path / "obs.opengwasdb"
        build_ragged_from_ssf(
            manifest, filtered_dir, observed_store, store_id="test", release_id="obs-v1"
        )

        obs_table = read_analyses(observed_store / "analyses.tsv")
        rows = {r["analysis_id"]: r["assigned_ancestry"] for r in obs_table.rows}
        assert rows == {"eur_trait": "EUR", "afr_trait": "AFR"}

        panel_dir = tmp_path / "ld_panel_us" / "EUR" / "1"
        TestPanelAlidCanonicalisation._write_underscore_panel(
            panel_dir,
            [(bp, "A", "G", 0.35) for bp in observed_bp] + [(1_050_000, "C", "T", 0.40)],
        )

        completed_store = tmp_path / "comp.opengwasdb"
        complete_ragged_store(
            observed_store, completed_store, tmp_path / "ld_panel_us",
            ancestry="EUR", cis_window_bp=500_000, min_cor=0.0, release_id="comp-v1",
        )

        comp_table = read_analyses(completed_store / "analyses.tsv")
        completed_against = {r["analysis_id"]: r["completed_against"] for r in comp_table.rows}
        assert completed_against == {"eur_trait": "EUR", "afr_trait": ""}

        comp_ax = VariantAxis(completed_store)
        rec = comp_ax.by_alid(parse_canonical_alid("1:1050000:C:T"))
        comp_ax.close()
        assert rec is not None

        obs_ax = VariantAxis(observed_store)
        afr_observed_alids = set()
        obs_q = query_store(observed_store)
        afr_observed = obs_q.analysis("afr_trait")
        obs_q.close()
        for vi in afr_observed["variant_index"].tolist():
            afr_observed_alids.add(obs_ax.by_index(vi).alid)
        obs_ax.close()

        q = query_store(completed_store)
        eur_result = q.analysis("eur_trait")
        afr_result = q.analysis("afr_trait")
        q.close()

        eur_idx = np.where(eur_result["variant_index"] == rec.variant_index)[0]
        assert len(eur_idx) == 1
        assert eur_result["association_status"][eur_idx[0]] == "imputed"
        assert np.isfinite(eur_result["z"][eur_idx[0]])

        # Ancestry-mismatched (ADR 0028: "left observed-only"): the new panel
        # variant must not appear at all, and the association list must be
        # exactly the untouched source (compared by ALID, since variant_index
        # spaces differ between the observed and completed variant tables) --
        # not expanded with a "missing" row for a reference-panel position
        # this Analysis was never completed against, even though the same
        # block/data shape succeeded for EUR.
        assert len(np.where(afr_result["variant_index"] == rec.variant_index)[0]) == 0
        assert len(afr_result["z"]) == len(afr_observed["z"]) == len(observed_bp)
        assert set(afr_result["association_status"].tolist()) == {"observed"}
        comp_ax2 = VariantAxis(completed_store)
        afr_result_alids = {comp_ax2.by_index(vi).alid for vi in afr_result["variant_index"].tolist()}
        comp_ax2.close()
        assert afr_result_alids == afr_observed_alids

        afr_index = next(
            int(r["analysis_index"]) for r in comp_table.rows if r["analysis_id"] == "afr_trait"
        )
        with sqlite3.connect(str(completed_store / "index.sqlite")) as conn:
            afr_quality_rows = conn.execute(
                "SELECT COUNT(*) FROM completion_quality WHERE analysis_index = ?", (afr_index,)
            ).fetchone()[0]
        assert afr_quality_rows == 0


class TestParallelAndResume:
    def test_n_workers_matches_serial(self, tmp_path, observed_store, ld_panel):
        serial_dst = tmp_path / "serial.opengwasdb"
        serial = complete_ragged_store(
            observed_store, serial_dst, ld_panel,
            ancestry="EUR", cis_window_bp=500_000, min_cor=0.0, release_id="serial",
        )
        parallel_dst = tmp_path / "parallel.opengwasdb"
        parallel = complete_ragged_store(
            observed_store, parallel_dst, ld_panel,
            ancestry="EUR", cis_window_bp=500_000, min_cor=0.0, release_id="parallel",
            n_workers=2,
        )
        assert parallel.n_variants == serial.n_variants
        assert parallel.n_imputed == serial.n_imputed
        assert parallel.n_missing == serial.n_missing
        assert parallel.n_associations == serial.n_associations

    def test_resume_matches_fresh_run(self, tmp_path, observed_store, ld_panel):
        import json

        from opengwasdb.completion.checkpoint import checkpoint_dir_for
        from opengwasdb.layouts.ragged.complete import resume_ragged_completion

        fresh_dst = tmp_path / "fresh.opengwasdb"
        fresh = complete_ragged_store(
            observed_store, fresh_dst, ld_panel,
            ancestry="EUR", cis_window_bp=500_000, min_cor=0.0,
        )

        resumable_dst = tmp_path / "resumable.opengwasdb"
        checkpoint_dir = checkpoint_dir_for(resumable_dst)
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "blocks").mkdir()
        (checkpoint_dir / "build_params.json").write_text(
            json.dumps({
                "source_path": str(Path(observed_store).resolve()),
                "dest_path": str(resumable_dst.resolve()),
                "ld_dir": str(Path(ld_panel).resolve()),
                "ancestry": "EUR", "cis_window_bp": 500_000,
                "min_cor": 0.0, "thresh": 0.9,
                "release_id": None, "ld_panel_id": "eur-hg38-gpm",
                "impute_analysis_ids": None, "region_cap_bp": 1_000_000,
            }),
            encoding="utf-8",
        )
        resumed = resume_ragged_completion(checkpoint_dir)

        assert resumed.n_variants == fresh.n_variants
        assert resumed.n_imputed == fresh.n_imputed
        assert resumed.n_missing == fresh.n_missing
        assert resumed.n_associations == fresh.n_associations
        assert not checkpoint_dir.exists()
        assert validate_store(resumable_dst).ok


class TestValidation:
    def test_valid_completed_store_passes(self, completed_store):
        result = validate_store(completed_store)
        assert result.ok, result.errors

    def test_observed_store_still_passes(self, observed_store):
        result = validate_store(observed_store)
        assert result.ok, result.errors

    def test_corrupt_imputed_array_fails(self, completed_store):
        root = open_store(completed_store).arrays(mode="r+")["ragged"]
        imp = root["imputed"][:]
        # Set imputed=1 where z is NaN
        z = root["z"][:].astype("float32")
        nan_positions = np.where(~np.isfinite(z))[0]
        if len(nan_positions) > 0:
            imp[nan_positions[0]] = 1
            root["imputed"][:] = imp
            result = validate_store(completed_store)
            assert not result.ok

    def test_stray_file_in_completed_store_fails(self, completed_store):
        # Issue #80: the closed-envelope check applies to Reference-Completed
        # releases too, not just Observed-Only ones -- Reference Completion
        # adds no new top-level entry, so the same allowed set governs both.
        (completed_store / "traits.tsv.gz").write_bytes(b"stray")

        result = validate_store(completed_store)

        assert not result.ok
        assert any(
            "unexpected store entry" in error and "traits.tsv.gz" in error
            for error in result.errors
        )


class TestQuery:
    def test_top_hits_uses_indexed_reference_frequency(self, completed_store):
        q = query_store(completed_store)
        expected = q.top_hits(threshold=5e-4)["eaf"]

        def fail_eaf_pairs(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("top_hits should read indexed frequencies")

        q._csr.eaf_pairs = fail_eaf_pairs
        result = q.top_hits(threshold=5e-4)
        np.testing.assert_array_equal(result["eaf"], expected)
        q.close()

    def test_selected_top_hits_match_global_and_filter_imputed(self, completed_store):
        q = query_store(completed_store)
        global_result = q.top_hits(threshold=5e-4)
        selected = q.top_hits(analysis_id="ENSG00000000001::Blood", threshold=5e-4)
        observed = q.top_hits(
            analysis_id="ENSG00000000001::Blood", threshold=5e-4, observed_only=True
        )
        expected = global_result["analysis_index"] == 0
        for name in ("variant_index", "analysis_index", "z", "se", "association_status"):
            np.testing.assert_array_equal(selected[name], global_result[name][expected])
        assert "imputed" not in set(observed["association_status"].tolist())
        q.close()

    def test_top_hits_fallback_matches_indexed_path(self, completed_store):
        """The full-CSR-scan fallback and the precomputed-index fast path must
        return the identical result (same order, same rows) for the same
        call -- issue 051. Force the fallback by deleting the precomputed
        top-hit group for a threshold the store already built, and compare
        both the default call and an observed_only+limit call, since the two
        filters interact (limit must apply after observed_only) and that
        interaction is exactly what previously diverged between the two
        paths."""
        threshold = 5e-4
        path = f"top_hits/{threshold_key(threshold)}"
        q = query_store(completed_store)
        indexed_default = q.top_hits(threshold=threshold)
        indexed_filtered = q.top_hits(threshold=threshold, observed_only=True, limit=1)
        q.close()
        assert len(indexed_default["z"]) > 0

        root = open_store(completed_store).arrays(mode="r+")
        del root[path]

        q2 = query_store(completed_store)
        assert path not in q2.store.arrays(mode="r")
        fallback_default = q2.top_hits(threshold=threshold)
        fallback_filtered = q2.top_hits(threshold=threshold, observed_only=True, limit=1)
        for name in ("variant_index", "analysis_index", "z", "se", "association_status"):
            np.testing.assert_array_equal(fallback_default[name], indexed_default[name])
            np.testing.assert_array_equal(fallback_filtered[name], indexed_filtered[name])
        q2.close()

    def test_top_hits_fallback_excludes_earliest_hit_when_forced_imputed(self, completed_store):
        """Force the genomically-first significant hit in one analysis to
        read as imputed, so observed_only must drop it and limit=1 must
        still return the *next* (observed) hit rather than an empty result.
        This is the exact scenario the original bug got wrong: applying
        limit before observed_only would slice down to the now-imputed
        first hit, then filter it away, leaving nothing -- the default-args
        parity test above can't distinguish that because this fixture's
        completion run happens to impute nothing (0 imputed, 3 missing), so
        observed_only is otherwise a no-op on it."""
        threshold = 5e-4
        root = open_store(completed_store).arrays(mode="r+")
        del root[f"top_hits/{threshold_key(threshold)}"]

        ragged = root["ragged"]
        offsets = ragged["offsets"][:]
        start, end = int(offsets[0]), int(offsets[1])  # analysis_index 0 == ENSG00000000001::Blood
        # Decoded, not raw: a missing cell in a fixed-point plane is a sentinel,
        # which `isfinite` would happily accept (ADR 0037).
        z_segment = RaggedCSRReader(completed_store).z_slice(start, end)
        finite = np.where(np.isfinite(z_segment))[0]
        assert len(finite) >= 2, "fixture must have >=2 finite associations in analysis 0"
        forced_offset = start + int(finite[0])
        second_variant_index = int(ragged["variant_index"][start + int(finite[1])])

        imputed = ragged["imputed"][:]
        imputed[forced_offset] = 1
        ragged["imputed"][:] = imputed

        q = query_store(completed_store)
        result = q.top_hits(
            analysis_id="ENSG00000000001::Blood", threshold=threshold, observed_only=True, limit=1
        )
        assert len(result["z"]) == 1
        assert result["association_status"][0] == "observed"
        assert int(result["variant_index"][0]) == second_variant_index
        q.close()

    def test_top_hits_fallback_filters_by_analysis_id(self, completed_store):
        threshold = 5e-4
        root = open_store(completed_store).arrays(mode="r+")
        del root[f"top_hits/{threshold_key(threshold)}"]
        q = query_store(completed_store)
        selected = q.top_hits(analysis_id="ENSG00000000001::Blood", threshold=threshold)
        assert len(selected["z"]) > 0
        assert set(selected["analysis_index"].tolist()) == {0}
        q.close()

    def test_variants_table_and_analyses_table(self, completed_store):
        q = query_store(completed_store)
        variants = q.variants_table()
        analyses = q.analyses_table()
        assert len(variants) > 0
        assert len(analyses) > 0
        assert all(
            {"alid", "chromosome", "position", "effect_allele", "other_allele", "rsid"}
            <= v.keys()
            for v in variants.values()
        )
        assert all("analysis_id" in a for a in analyses.values())
        q.close()

    def test_context_manager(self, completed_store):
        with query_store(completed_store) as q:
            assert q.analyses_table()

    def test_analysis_returns_imputed_by_default(self, completed_store):
        q = query_store(completed_store)
        result = q.analysis("ENSG00000000001::Blood")
        assert "association_status" in result
        statuses = set(result["association_status"].tolist())
        # Should have at least "observed"; may also have "imputed" or "missing"
        assert "observed" in statuses
        q.close()

    def test_analysis_observed_only_excludes_imputed(self, completed_store):
        q = query_store(completed_store)
        all_result = q.analysis("ENSG00000000001::Blood")
        obs_result = q.analysis("ENSG00000000001::Blood", observed_only=True)
        # observed-only must be a subset
        assert len(obs_result["z"]) <= len(all_result["z"])
        statuses = set(obs_result["association_status"].tolist())
        assert "imputed" not in statuses
        q.close()

    def test_association_status_field_always_present(self, completed_store):
        q = query_store(completed_store)
        result = q.analysis("ENSG00000000001::Blood")
        assert "association_status" in result
        assert len(result["association_status"]) == len(result["z"])
        q.close()

    def test_observed_store_has_status_observed(self, observed_store):
        q = query_store(observed_store)
        result = q.analysis("ENSG00000000001::Blood")
        assert "association_status" in result
        statuses = set(result["association_status"].tolist())
        assert statuses.issubset({"observed"})
        q.close()

    def test_range_by_analysis_observed_only(self, completed_store):
        q = query_store(completed_store)
        result = q.range_by_analysis("1", 900_000, 1_300_000, observed_only=True)
        statuses = set(result["association_status"].tolist())
        assert "imputed" not in statuses
        q.close()


# Panel SNPs for the gene-target-less fixture: six in one chr1 block, of which
# the store observes five. Enough observed points for `poly_rescale` to fit
# (`min_observed_points()` == 4), and one panel-only variant to prove the block
# was enumerated.
_RICH_PANEL_SNPS = [
    ("1:900000:A:G", 0.35, 900_000),
    ("1:950000:A:C", 0.32, 950_000),
    ("1:1000000:A:G", 0.30, 1_000_000),
    ("1:1050500:C:T", 0.40, 1_050_500),   # panel-only: never observed by the store
    ("1:1100000:C:T", 0.45, 1_100_000),
    ("1:1150000:A:G", 0.28, 1_150_000),
]
_RICH_OBSERVED = [snp for snp in _RICH_PANEL_SNPS if snp[0] != "1:1050500:C:T"]


def _make_rich_ld_panel(tmp_path: Path) -> Path:
    """One chr1 block over `_RICH_PANEL_SNPS`, in the panel's native id form."""
    panel_dir = tmp_path / "rich_panel" / "EUR" / "1"
    panel_dir.mkdir(parents=True)
    lines = ["CHR\tSNP\tOA\tEA\tEAF\tBP"]
    for alid, eaf, _bp in _RICH_PANEL_SNPS:
        chrom, pos, a1, a2 = alid.split(":")
        lines.append(f"{chrom}\t{chrom}:{pos}_{a1}_{a2}\t{a2}\t{a1}\t{eaf}\t{pos}")
    (panel_dir / "900000-1300000.tsv").write_text("\n".join(lines) + "\n")

    n = len(_RICH_PANEL_SNPS)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n))
    ld = A @ A.T + np.eye(n) * n * 0.1
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        for row in ld:
            gz.write(("\t".join(f"{v:.6f}" for v in row) + "\n").encode())
    (panel_dir / "900000-1300000.unphased.vcor1.gz").write_bytes(buf.getvalue())
    return tmp_path / "rich_panel"


def _make_rich_besd_fixture(tmp_path: Path) -> Path:
    """One probe observing five of the block's six panel SNPs."""
    fixture = tmp_path / "rich_fixture"
    fixture.mkdir()
    snps = []
    for alid, _eaf, bp in _RICH_OBSERVED:
        chrom, pos, a1, a2 = alid.split(":")
        snps.append({"chr": chrom, "snp_id": f"rs{pos}", "bp": bp, "a1": a1, "a2": a2})
    probes = [{"chr": "1", "probe_id": "ENSG00000000001", "bp": 1_050_000, "gene": "GENE1"}]
    # Distinct effect sizes: `elastic_net_impute` refuses a block whose observed
    # z-scores are all identical.
    assocs = [[(i, 0.1 * (i + 1), 0.02) for i in range(len(snps))]]
    _write_esi(fixture / "test.esi", snps)
    _write_epi(fixture / "test.epi", probes)
    _write_besd_sparse_3f(fixture / "test.besd", len(probes), assocs)
    return fixture / "test"


class TestGeneTargetLessAnalyses:
    """Issue #102: a Store Family with no single encoding gene per Analysis
    (small-molecule metabolomics, say) has no `trait_chr`/`trait_bp` at all --
    by design, and documented as such in the registry's schema. Block
    enumeration was scoped entirely to a cis window around that position, so
    for those families Reference Completion was a **complete no-op**: 0 blocks
    enumerated, 0 new panel variants, 0 imputed, and a release stamped
    `reference_completed` whose associations are byte-identical to its
    observed-only source. All four of opengwasdb-stores'
    `metabolome-plasma-2023` full releases (4,443 real Analyses) completed
    this way and did nothing.

    These tests assert on **block enumeration** -- did the store gain the
    panel-only variants of a block it should have completed -- rather than on
    imputed counts, because whether `ElasticNetCV` converges on a fixture this
    small is not deterministic (`test_dense_completion.py` notes the same
    fragility). Enumeration is what #102 is about: the blocks were never
    reached at all.
    """

    #: In the fixture's panel block and never observed by the store, so it can
    #: only reach the completed store if that block was enumerated.
    PANEL_ONLY_ALID = "1:1050500:C:T"

    @pytest.fixture
    def rich_observed_store(self, tmp_path):
        prefix = _make_rich_besd_fixture(tmp_path)
        out = tmp_path / "rich_obs.opengwasdb"
        build_ragged_from_besd(prefix, out, store_id="test", release_id="obs-v1", tissue="Blood")
        return out

    @pytest.fixture
    def rich_panel(self, tmp_path):
        return _make_rich_ld_panel(tmp_path)

    @staticmethod
    def _drop_trait_positions(store: Path) -> None:
        """Blank `trait_chr`/`trait_bp` and nothing else, so the only
        difference from a store that does get completed is the one thing this
        issue is about."""
        from opengwasdb.model.analyses import read_analyses, write_analyses

        table = read_analyses(store / "analyses.tsv")
        rows = [{**row, "trait_chr": "", "trait_bp": ""} for row in table.rows]
        write_analyses(
            store / "analyses.tsv", type(table)(fieldnames=table.fieldnames, rows=tuple(rows))
        )

    @staticmethod
    def _alids(store: Path) -> set[str]:
        from opengwasdb.variants.axis import VariantAxis

        axis = VariantAxis(store)
        try:
            return {v.alid for v in axis.all()}
        finally:
            axis.close()

    def test_the_fixture_is_meaningful_before_anything_is_asserted_about_it(
        self, rich_observed_store, rich_panel
    ):
        """The Analysis must observe enough of the block to be imputable at
        all (`min_observed_points()`), and the panel-only variant must be
        absent from the source -- or the assertions below prove nothing."""
        from opengwasdb.completion.impute import min_observed_points

        observed = self._alids(rich_observed_store)
        assert self.PANEL_ONLY_ALID not in observed
        panel_alids = {alid for alid, _eaf, _bp in _RICH_PANEL_SNPS}
        assert len(observed & panel_alids) >= min_observed_points()
        assert "1:1050500_C_T" in (
            rich_panel / "EUR" / "1" / "900000-1300000.tsv"
        ).read_text()

    def test_an_analysis_with_a_trait_position_still_uses_its_cis_window(
        self, tmp_path, rich_observed_store, rich_panel
    ):
        """The unchanged path, asserted so the new one cannot be mistaken for
        it."""
        dst = tmp_path / "with_positions.opengwasdb"
        complete_ragged_store(
            rich_observed_store, dst, rich_panel,
            ancestry="EUR", cis_window_bp=500_000, min_cor=0.0,
        )
        assert self.PANEL_ONLY_ALID in self._alids(dst)

    def test_an_analysis_with_no_trait_position_is_completed_from_its_own_variants(
        self, tmp_path, rich_observed_store, rich_panel
    ):
        self._drop_trait_positions(rich_observed_store)
        dst = tmp_path / "no_positions.opengwasdb"

        complete_ragged_store(
            rich_observed_store, dst, rich_panel,
            ancestry="EUR", cis_window_bp=500_000, min_cor=0.0,
        )

        assert self.PANEL_ONLY_ALID in self._alids(dst), (
            "no LD block was enumerated for a gene-target-less Analysis: reference "
            "completion did nothing but relabel the store"
        )

    def test_a_block_the_analysis_barely_touches_is_not_enumerated(
        self, tmp_path, rich_observed_store, rich_panel
    ):
        """Below `min_observed_points()` the block cannot be fitted, so
        enumerating it would add its panel variants as missing rows and impute
        none of them -- and spec §17 forbids expanding a suggestive singleton
        into a whole region. Here the Analysis keeps only two of the block's
        SNPs."""
        from opengwasdb.model.analyses import read_analyses, write_analyses
        from opengwasdb.variants.axis import VariantAxis

        self._drop_trait_positions(rich_observed_store)
        # Rebuild the source with only two of the block's panel SNPs observed.
        thin = tmp_path / "thin_fixture"
        thin.mkdir()
        snps = []
        for alid, _eaf, bp in _RICH_OBSERVED[:2]:
            chrom, pos, a1, a2 = alid.split(":")
            snps.append({"chr": chrom, "snp_id": f"rs{pos}", "bp": bp, "a1": a1, "a2": a2})
        _write_esi(thin / "test.esi", snps)
        _write_epi(thin / "test.epi", [
            {"chr": "1", "probe_id": "ENSG00000000001", "bp": 1_050_000, "gene": "GENE1"}
        ])
        _write_besd_sparse_3f(
            thin / "test.besd", 1, [[(0, 0.1, 0.02), (1, 0.3, 0.02)]]
        )
        thin_store = tmp_path / "thin_obs.opengwasdb"
        build_ragged_from_besd(
            thin / "test", thin_store, store_id="test", release_id="obs-v1", tissue="Blood"
        )
        table = read_analyses(thin_store / "analyses.tsv")
        write_analyses(
            thin_store / "analyses.tsv",
            type(table)(
                fieldnames=table.fieldnames,
                rows=tuple({**r, "trait_chr": "", "trait_bp": ""} for r in table.rows),
            ),
        )

        dst = tmp_path / "thin_completed.opengwasdb"
        result = complete_ragged_store(
            thin_store, dst, rich_panel, ancestry="EUR", cis_window_bp=500_000, min_cor=0.0,
        )

        assert result.n_imputed == 0
        axis = VariantAxis(dst)
        try:
            assert self.PANEL_ONLY_ALID not in {v.alid for v in axis.all()}
        finally:
            axis.close()

    def test_an_analysis_whose_variants_touch_no_block_is_left_alone(
        self, tmp_path, rich_observed_store
    ):
        """The pass-through path still exists -- it is now reached by holding
        no completable region, rather than by having no gene target."""
        self._drop_trait_positions(rich_observed_store)
        empty_panel = tmp_path / "empty_panel"
        (empty_panel / "EUR" / "22").mkdir(parents=True)
        dst = tmp_path / "no_blocks.opengwasdb"

        result = complete_ragged_store(
            rich_observed_store, dst, empty_panel,
            ancestry="EUR", cis_window_bp=500_000, min_cor=0.0,
        )

        assert result.n_imputed == 0
        assert self._alids(dst) == self._alids(rich_observed_store)
