"""Integration test for ragged BESD builder (issue 036)."""

import struct
from pathlib import Path

import numpy as np
import pytest

from opengwasdb.layouts.ragged.build_besd import build_ragged_from_besd
from opengwasdb.layouts.ragged.top_hits import build_ragged_top_hit_indexes
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRReader
from opengwasdb.store.open import open_store


# ── Synthetic BESD fixture ────────────────────────────────────────────────────

def _write_esi(path: Path, snps: list[dict]) -> None:
    with open(path, "w") as fh:
        for s in snps:
            freq = s.get("freq", "NA")
            fh.write(f"{s['chr']}\t{s['snp_id']}\t0\t{s['bp']}\t{s['a1']}\t{s['a2']}\t{freq}\n")


def _write_epi(path: Path, probes: list[dict]) -> None:
    with open(path, "w") as fh:
        for p in probes:
            gene = p.get("gene", "NA")
            fh.write(f"{p['chr']}\t{p['probe_id']}\t0\t{p['bp']}\t{gene}\t+\n")


def _write_besd_sparse_3f(
    path: Path,
    n_probes: int,
    probe_assocs: list[list[tuple[int, float, float]]],
) -> None:
    """Write a minimal SPARSE_FILE_TYPE_3F BESD file.

    BESD layout: for each probe p, val contains [betas | SEs] and rowid contains
    [snp_indices | zeros] at the same offsets. cols[2p]=beta_start, cols[2p+1]=se_start.
    """
    rowid: list[int] = []
    val: list[float] = []
    cols: list[int] = []
    offset = 0

    for assocs in probe_assocs:
        n = len(assocs)
        cols.append(offset)          # beta_start for this probe
        cols.append(offset + n)      # se_start for this probe
        # Beta positions: meaningful snp_idx and beta values
        for snp_idx, beta, _ in assocs:
            rowid.append(snp_idx)
            val.append(beta)
        # SE positions: snp_idx is don't-care; SE values follow immediately
        for _, _, se in assocs:
            rowid.append(0)
            val.append(se)
        offset += 2 * n

    cols.append(offset)   # final sentinel
    val_num = len(val)
    col_num = (n_probes << 1) + 1

    with open(path, "wb") as fh:
        fh.write(struct.pack("<I", 0x40400000))         # magic 3F
        fh.write(struct.pack("<Q", val_num))             # val_num
        fh.write(struct.pack(f"<{col_num}Q", *cols))    # column offsets
        fh.write(struct.pack(f"<{val_num}I", *rowid))   # row indices
        fh.write(struct.pack(f"<{val_num}f", *val))     # beta/SE values


def _make_besd_fixture(tmp_path: Path) -> Path:
    """Create a 3-probe synthetic BESD dataset in tmp_path/fixture/."""
    fixture = tmp_path / "fixture"
    fixture.mkdir()

    snps = [
        {"chr": "1", "snp_id": "rs1001", "bp": 1_000_000, "a1": "A", "a2": "G"},
        {"chr": "1", "snp_id": "rs1002", "bp": 1_100_000, "a1": "C", "a2": "T"},
        {"chr": "1", "snp_id": "rs1003", "bp": 1_200_000, "a1": "A", "a2": "C"},
        {"chr": "2", "snp_id": "rs2001", "bp": 2_000_000, "a1": "G", "a2": "T"},
        {"chr": "2", "snp_id": "rs2002", "bp": 2_100_000, "a1": "A", "a2": "T"},
    ]
    probes = [
        {"chr": "1", "probe_id": "ENSG00000000001", "bp": 1_050_000, "gene": "GENE1"},
        {"chr": "1", "probe_id": "ENSG00000000002", "bp": 1_150_000, "gene": "GENE2"},
        {"chr": "2", "probe_id": "ENSG00000000003", "bp": 2_050_000, "gene": "GENE3"},
    ]
    # probe 0: SNPs 0,1 (rs1001, rs1002)
    # probe 1: SNP 2 (rs1003)
    # probe 2: SNPs 3,4 (rs2001, rs2002)
    probe_assocs = [
        [(0, 0.1, 0.02), (1, -0.2, 0.03)],
        [(2, 0.5, 0.05)],
        [(3, 0.3, 0.04), (4, -0.15, 0.025)],
    ]

    _write_esi(fixture / "test.esi", snps)
    _write_epi(fixture / "test.epi", probes)
    _write_besd_sparse_3f(fixture / "test.besd", len(probes), probe_assocs)
    return fixture / "test"


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_build_creates_store_files(tmp_path):
    prefix = _make_besd_fixture(tmp_path)
    out = tmp_path / "out.opengwasdb"

    result = build_ragged_from_besd(
        prefix, out,
        store_id="test", release_id="v1",
        tissue="Whole_Blood",
    )

    assert out.exists()
    assert (out / "manifest.json").exists()
    assert (out / "variants.tsv.gz").exists()
    assert (out / "analyses.tsv").exists()
    # traits.tsv.gz (issue 034) is retired (issue #69): a Trait's genomic
    # position lives in analyses.tsv's own trait_chr/trait_bp columns now,
    # not a second, independently-shaped tabix-indexed position file.
    assert not (out / "traits.tsv.gz").exists()
    assert (out / "index.sqlite").exists()
    assert (out / "data.zarr" / "ragged").exists()

    assert result.n_variants == 5
    assert result.n_analyses == 3


def test_manifest_primary_layout(tmp_path):
    import json
    prefix = _make_besd_fixture(tmp_path)
    out = tmp_path / "out.opengwasdb"
    build_ragged_from_besd(prefix, out, store_id="test", release_id="v1")

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["primary_layout"] == "ragged"
    assert manifest["completion_state"] == "observed_only"
    assert manifest["reference_assembly"] == "GRCh38"


def test_analyses_tsv_carries_probe_identities(tmp_path):
    from opengwasdb.model.analyses import read_analyses

    prefix = _make_besd_fixture(tmp_path)
    out = tmp_path / "out.opengwasdb"
    build_ragged_from_besd(prefix, out, store_id="test", release_id="v1", tissue="Whole_Blood")

    table = read_analyses(out / "analyses.tsv")
    assert len(table.rows) == 3
    gene_ontology_ids = {r["trait_ontology_id"] for r in table.rows}
    assert gene_ontology_ids == {
        "ENSEMBL:ENSG00000000001", "ENSEMBL:ENSG00000000002", "ENSEMBL:ENSG00000000003",
    }


def test_zarr_csr_associations(tmp_path):
    prefix = _make_besd_fixture(tmp_path)
    out = tmp_path / "out.opengwasdb"
    build_ragged_from_besd(prefix, out, store_id="test", release_id="v1")

    csr = RaggedCSRReader(out)
    assert csr.n_analyses == 3
    assert csr.n_associations == 5  # 2 + 1 + 2

    # Probe 0 has 2 associations
    a0 = csr.get_analysis(0)
    assert len(a0.variant_index) == 2

    # Probe 1 has 1 association
    a1 = csr.get_analysis(1)
    assert len(a1.variant_index) == 1

    # Probe 2 has 2 associations
    a2 = csr.get_analysis(2)
    assert len(a2.variant_index) == 2


def test_z_scores_computed_correctly(tmp_path):
    prefix = _make_besd_fixture(tmp_path)
    out = tmp_path / "out.opengwasdb"
    build_ragged_from_besd(prefix, out, store_id="test", release_id="v1")

    csr = RaggedCSRReader(out)
    # Probe 1: single assoc beta=0.5, se=0.05 → z=10.0
    a1 = csr.get_analysis(1)
    assert len(a1.z) == 1
    assert abs(float(a1.z[0]) - 10.0) < 0.001  # fixed-point 1/1024 (ADR 0037)


def test_top_hit_index_built_inline(tmp_path):
    """build_ragged_from_besd auto-builds the top-hit index."""
    prefix = _make_besd_fixture(tmp_path)
    out = tmp_path / "out.opengwasdb"
    build_ragged_from_besd(prefix, out, store_id="test", release_id="v1")

    root = open_store(out).arrays(mode="r")
    assert "top_hits" in root
    # At least one threshold group must exist
    assert len(list(root["top_hits"].keys())) > 0


def test_top_hit_index_schema(tmp_path):
    """Each threshold group has the expected arrays and attribute."""
    prefix = _make_besd_fixture(tmp_path)
    out = tmp_path / "out.opengwasdb"
    build_ragged_from_besd(prefix, out, store_id="test", release_id="v1")
    build_ragged_top_hit_indexes(out)  # idempotent rebuild

    root = open_store(out).arrays(mode="r")
    for key in root["top_hits"]:
        group = root["top_hits"][key]
        for name in ("variant_index", "analysis_index", "abs_z", "z", "se", "p_value"):
            assert name in group, f"missing {name} in {key}"
        n = len(group["variant_index"])
        for name in ("analysis_index", "abs_z", "z", "se", "p_value"):
            assert len(group[name]) == n
        assert "threshold" in group.attrs


def test_top_hit_z_values_match_csr(tmp_path):
    """Top-hit z values round-trip through the stored CSR correctly."""
    prefix = _make_besd_fixture(tmp_path)
    out = tmp_path / "out.opengwasdb"
    build_ragged_from_besd(prefix, out, store_id="test", release_id="v1")

    csr = RaggedCSRReader(out)
    root = open_store(out).arrays(mode="r")

    # Use the loosest threshold to get all 5 hits
    loosest_key = sorted(root["top_hits"].keys())[-1]
    group = root["top_hits"][loosest_key]
    vis = group["variant_index"][:].astype(int)
    ais = group["analysis_index"][:].astype(int)
    zs = group["z"][:].astype("float32")

    offsets = csr._offsets[:]
    vi_all = csr._variant_index[:]
    z_all = csr.z_all()

    for vi, ai, z_hit in zip(vis, ais, zs):
        start, end = int(offsets[ai]), int(offsets[ai + 1])
        pos = start + int(np.searchsorted(vi_all[start:end], vi))
        assert pos < end and int(vi_all[pos]) == vi
        assert np.isclose(float(z_all[pos]), float(z_hit), rtol=1e-2, atol=1e-2)


def test_top_hits_query_uses_index(tmp_path):
    """RaggedStoreQuery.top_hits reads from the precomputed index."""
    from opengwasdb.query import query_store
    prefix = _make_besd_fixture(tmp_path)
    out = tmp_path / "out.opengwasdb"
    build_ragged_from_besd(prefix, out, store_id="test", release_id="v1")

    q = query_store(out)
    # Threshold 5e-4 should return all 5 associations (all have |z| > 3.48)
    result = q.top_hits(threshold=5e-4)
    assert len(result["z"]) == 5
    assert "variant_index" in result and "analysis_index" in result


def test_top_hits_query_selects_one_analysis_from_index(tmp_path):
    from opengwasdb.query import query_store

    prefix = _make_besd_fixture(tmp_path)
    out = tmp_path / "out.opengwasdb"
    build_ragged_from_besd(prefix, out, store_id="test", release_id="v1")

    q = query_store(out)
    global_result = q.top_hits(threshold=5e-4)
    selected = q.top_hits(analysis_id="ENSG00000000001", threshold=5e-4, limit=1)

    assert selected["analysis_index"].tolist() == [0]
    assert selected["variant_index"].tolist() == global_result["variant_index"][:1].tolist()
    assert q.top_hits(analysis_id="unknown", threshold=5e-4)["z"].size == 0


def test_validation_rejects_ragged_top_hit_offsets(tmp_path):
    from opengwasdb.validation import validate_store

    prefix = _make_besd_fixture(tmp_path)
    out = tmp_path / "out.opengwasdb"
    build_ragged_from_besd(prefix, out, store_id="test", release_id="v1")
    root = open_store(out).arrays(mode="r+")
    offsets = root["top_hits/p_5e_04/analysis_offsets"][:]
    offsets[-1] -= 1
    root["top_hits/p_5e_04/analysis_offsets"][:] = offsets

    result = validate_store(out)
    assert not result.ok
    assert any("invalid analysis offsets" in error for error in result.errors)


def test_validation_checks_threshold_for_legacy_index_without_eaf(tmp_path):
    from opengwasdb.validation import validate_store

    prefix = _make_besd_fixture(tmp_path)
    out = tmp_path / "store.opengwasdb"
    build_ragged_from_besd(prefix, out, store_id="s", release_id="r")
    root = open_store(out).arrays(mode="a")
    group = root["top_hits/p_5e_04"]
    assert "eaf" not in group
    group.attrs["threshold"] = 1e-20

    result = validate_store(out)
    assert not result.ok
    assert any("above threshold" in error for error in result.errors)


def test_analyses_tsv_has_no_sql_table_and_carries_molecular_columns(tmp_path):
    """Issue #69: analyses.tsv is the sole source of truth for Analytical
    Metadata; the store's index.sqlite carries no `analyses` table, and the
    molecular/context columns the BESD path already collects (gene, tissue,
    genomic position) are carried into the shared schema."""
    import sqlite3

    from opengwasdb.model.analyses import read_analyses

    prefix = _make_besd_fixture(tmp_path)
    out = tmp_path / "out.opengwasdb"
    build_ragged_from_besd(prefix, out, store_id="test", release_id="v1", tissue="Blood")

    conn = sqlite3.connect(str(out / "index.sqlite"))
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert "analyses" not in tables

    table = read_analyses(out / "analyses.tsv")
    rows = {r["analysis_id"]: r for r in table.rows}
    assert rows["ENSG00000000001::Blood"]["analysis_label"] == "GENE1"
    assert rows["ENSG00000000001::Blood"]["trait_ontology_id"] == "ENSEMBL:ENSG00000000001"
    assert rows["ENSG00000000001::Blood"]["trait_ontology_label"] == "Ensembl"
    assert rows["ENSG00000000001::Blood"]["tissue"] == "Blood"
    assert rows["ENSG00000000001::Blood"]["trait_chr"] == "1"
    assert rows["ENSG00000000001::Blood"]["trait_bp"] == "1050000"


def test_top_hit_counts_in_analyses_tsv_match_top_hit_index(tmp_path):
    """Issue #73: Ragged builds persist per-Analysis Top-Hit Counts in
    analyses.tsv, matching what the store's own top-hit index actually
    contains -- not zero, not stale."""
    from opengwasdb.layouts.dense.top_hits import read_top_hit_counts
    from opengwasdb.model.analyses import read_analyses

    prefix = _make_besd_fixture(tmp_path)
    out = tmp_path / "out.opengwasdb"
    build_ragged_from_besd(prefix, out, store_id="test", release_id="v1")

    table = read_analyses(out / "analyses.tsv")
    rows = sorted(table.rows, key=lambda r: int(r["analysis_index"]))
    expected = read_top_hit_counts(out, len(rows))
    for column in ("n_hits_5e8", "n_hits_5e6", "n_hits_5e4"):
        assert [int(r[column]) for r in rows] == expected[column]
    # All 5 associations clear the loosest threshold (test_top_hits_query_uses_index).
    assert sum(int(r["n_hits_5e4"]) for r in rows) == 5


def test_analyses_table_groups_by_gene(tmp_path):
    """Issue #71/#81: a caller who used to group Ragged Analyses via the
    retired dual analysis_id-or-trait_id lookup can instead filter
    analyses_table() (the same shape Dense/Hybrid return) on
    trait_ontology_id (ADR 0035: gene identity travels via
    analysis_label/trait_ontology_id, not dedicated gene_id/gene_name
    columns)."""
    from opengwasdb.query import query_store

    prefix = _make_besd_fixture(tmp_path)
    out = tmp_path / "out.opengwasdb"
    build_ragged_from_besd(prefix, out, store_id="test", release_id="v1")

    q = query_store(out)
    analyses = q.analyses_table()
    q.close()

    by_gene_ontology_id = {row["trait_ontology_id"]: idx for idx, row in analyses.items()}
    assert by_gene_ontology_id["ENSEMBL:ENSG00000000002"] is not None
    matched = [
        idx for idx, row in analyses.items()
        if row["trait_ontology_id"] == "ENSEMBL:ENSG00000000002"
    ]
    assert len(matched) == 1
    assert analyses[matched[0]]["analysis_id"] == "ENSG00000000002"


def test_range_by_analysis_matches_trait_position(tmp_path):
    """Issue #69/#71: genomic-range-by-Trait-position lookup keeps working
    after retiring both the SQL analyses table and the tabix-indexed
    traits.tsv.gz sidecar (issue #69: not a second, independently-shaped
    metadata file) -- analyses.tsv's own trait_chr/trait_bp columns are now
    the sole source of truth `range_by_analysis()` scans."""
    from opengwasdb.query import query_store

    prefix = _make_besd_fixture(tmp_path)
    out = tmp_path / "out.opengwasdb"
    build_ragged_from_besd(prefix, out, store_id="test", release_id="v1")

    from opengwasdb.model.analyses import read_analyses

    # Independent oracle: compute the expected analysis_index set by hand
    # from the same analyses.tsv rows the query facade reads, rather than
    # hardcoding the expected indices, so a filtering-logic bug in
    # range_by_analysis() (wrong comparison operator, wrong column) would
    # be caught even though both sides ultimately read the same file.
    table = read_analyses(out / "analyses.tsv")
    expected = {
        int(r["analysis_index"])
        for r in table.rows
        if r["trait_chr"] == "1" and 1_000_000 <= int(r["trait_bp"]) <= 1_200_000
    }
    assert expected == {0, 1}  # chr1 probes at 1_050_000/1_150_000; chr2 probe excluded

    q = query_store(out)
    result = q.range_by_analysis("1", 1_000_000, 1_200_000)
    q.close()

    assert set(result["analysis_index"].tolist()) == expected


def test_overwrite_flag(tmp_path):
    prefix = _make_besd_fixture(tmp_path)
    out = tmp_path / "out.opengwasdb"

    build_ragged_from_besd(prefix, out, store_id="test", release_id="v1")
    with pytest.raises(FileExistsError):
        build_ragged_from_besd(prefix, out, store_id="test", release_id="v1")

    # With overwrite=True should succeed
    build_ragged_from_besd(prefix, out, store_id="test", release_id="v1", overwrite=True)
    assert out.exists()
