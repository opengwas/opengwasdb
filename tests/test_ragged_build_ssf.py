"""Integration test for the ragged GWAS-SSF builder (ticket 074).

Mirrors tests/test_ragged_build_besd.py's structure -- build_ragged_from_ssf
is the BESD builder's sibling for filtered GWAS-SSF sources.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

from opengwasdb.layouts.ragged.build_ssf import build_ragged_from_ssf
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRReader
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.query import query_store
from opengwasdb.store.open import open_store
from opengwasdb.validation import validate_store
from opengwasdb.variants.axis import VariantAxis

_SSF_HEADER = [
    "chromosome",
    "base_pair_location",
    "effect_allele",
    "other_allele",
    "beta",
    "standard_error",
    "effect_allele_frequency",
    "rsid",
    "variant_id",
]


def _write_filtered(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\t".join(_SSF_HEADER) + "\n")
        for row in rows:
            fh.write("\t".join(str(row.get(col, "")) for col in _SSF_HEADER) + "\n")


#: Every column the Ragged SSF builder's manifest may carry, in written order.
_MANIFEST_HEADER = [
    "analysis_index",
    "analysis_id",
    "trait_id",
    "analysis_label",
    "trait_ontology_id",
    "trait_ontology_label",
    "trait_chr",
    "trait_bp",
    "n",
    "tissue",
    "context",
    "mhc",
    "filtered_file",
]


def _write_one_association(filtered_dir: Path, name: str = "trait_a.tsv.gz") -> None:
    """One unambiguous association, for tests about the manifest rather than the rows."""
    _write_filtered(
        filtered_dir / name,
        [
            {
                "chromosome": "1",
                "base_pair_location": 100_000,
                "effect_allele": "A",
                "other_allele": "G",
                "beta": 1.0,
                "standard_error": 0.5,
                "rsid": "rs1",
            },
        ],
    )


def _write_manifest(path: Path, rows: list[dict]) -> None:
    header = _MANIFEST_HEADER
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for row in rows:
            fh.write("\t".join(str(row.get(col, "")) for col in header) + "\n")


def _make_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Two analyses sharing one variant (1:100000:A:G).

    trait_a (analysis_index 0):
      row1: chr1:100000 A/G, beta=1.0 se=0.5, rsid=rs1        -> A canonical, no flip, z=2.0
      row2: chr1:200000 G/A, beta=1.0 se=0.25, rsid=rs2       -> A canonical (A<G), flip, z=-4.0
      row3: chr1:300000 C/T, beta=1.0 se=NA (unparseable)     -> dropped
      row4: chr1:400000 C/T, beta=1.0 se=0 (non-positive)     -> dropped
      row5: chr1:600000 C/T, beta=inf se=0.5 (non-finite beta) -> dropped

    trait_b (analysis_index 1):
      row1: chr2:500000 C/T, beta=2.0 se=0.5, rsid="" variant_id=rs5fb
            -> no flip, z=4.0, rsid via fallback
      row2: chr1:100000 A/G, beta=0.5 se=0.25, no rsid
            -> same variant as trait_a row1, no flip, z=2.0
    """
    filtered_dir = tmp_path / "filtered"
    filtered_dir.mkdir()

    _write_filtered(
        filtered_dir / "trait_a.tsv.gz",
        [
            {
                "chromosome": "1",
                "base_pair_location": 100_000,
                "effect_allele": "A",
                "other_allele": "G",
                "beta": 1.0,
                "standard_error": 0.5,
                "rsid": "rs1",
            },
            {
                "chromosome": "1",
                "base_pair_location": 200_000,
                "effect_allele": "G",
                "other_allele": "A",
                "beta": 1.0,
                "standard_error": 0.25,
                "rsid": "rs2",
            },
            {
                "chromosome": "1",
                "base_pair_location": 300_000,
                "effect_allele": "C",
                "other_allele": "T",
                "beta": 1.0,
                "standard_error": "NA",
            },
            {
                "chromosome": "1",
                "base_pair_location": 400_000,
                "effect_allele": "C",
                "other_allele": "T",
                "beta": 1.0,
                "standard_error": 0.0,
            },
            {
                "chromosome": "1",
                "base_pair_location": 600_000,
                "effect_allele": "C",
                "other_allele": "T",
                "beta": "inf",
                "standard_error": 0.5,
            },
        ],
    )
    _write_filtered(
        filtered_dir / "trait_b.tsv.gz",
        [
            {
                "chromosome": "2",
                "base_pair_location": 500_000,
                "effect_allele": "C",
                "other_allele": "T",
                "beta": 2.0,
                "standard_error": 0.5,
                "variant_id": "rs5fb",
            },
            {
                "chromosome": "1",
                "base_pair_location": 100_000,
                "effect_allele": "A",
                "other_allele": "G",
                "beta": 0.5,
                "standard_error": 0.25,
            },
        ],
    )

    manifest = tmp_path / "manifest.tsv"
    _write_manifest(
        manifest,
        [
            {
                "analysis_index": 0,
                "analysis_id": "trait_a",
                "trait_id": "T1",
                "analysis_label": "GENE1",
                "trait_ontology_id": "ENSEMBL:ENSG00001",
                "trait_ontology_label": "Ensembl",
                "trait_chr": "1",
                "trait_bp": 150_000,
                "n": 5000,
                "tissue": "Liver",
                "context": "",
                "mhc": "FALSE",
                "filtered_file": "trait_a.tsv.gz",
            },
            {
                "analysis_index": 1,
                "analysis_id": "trait_b",
                "trait_id": "T2",
                "analysis_label": "GENE2",
                "trait_ontology_id": "ENSEMBL:ENSG00002",
                "trait_ontology_label": "Ensembl",
                "trait_chr": "2",
                "trait_bp": 500_000,
                "n": 6000,
                "tissue": "Blood",
                "context": "",
                "mhc": "FALSE",
                "filtered_file": "trait_b.tsv.gz",
            },
        ],
    )
    return manifest, filtered_dir


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_build_creates_store_files(tmp_path):
    manifest, filtered_dir = _make_fixture(tmp_path)
    out = tmp_path / "out.opengwasdb"

    result = build_ragged_from_ssf(manifest, filtered_dir, out, store_id="test", release_id="v1")

    assert out.exists()
    assert (out / "manifest.json").exists()
    assert (out / "variants.tsv.gz").exists()
    assert (out / "analyses.tsv").exists()
    assert (out / "index.sqlite").exists()
    assert (out / "data.zarr" / "ragged").exists()

    assert result.n_variants == 3  # 1:100000, 1:200000, 2:500000
    assert result.n_analyses == 2
    assert result.n_associations == 4  # 2 valid rows per analysis


def test_ssf_build_queries_and_validates_residual_se(tmp_path):
    filtered_dir = tmp_path / "filtered"
    filtered_dir.mkdir()
    frequencies = np.linspace(0.05, 0.95, 600, dtype=np.float32)
    expected = np.exp(
        -3.0
        - 0.5 * np.log(2 * frequencies * (1 - frequencies))
        + 0.1 * np.sin(np.arange(len(frequencies)) * 0.07)
    ).astype(np.float32)
    _write_filtered(
        filtered_dir / "trait.tsv.gz",
        [
            {
                "chromosome": "1",
                "base_pair_location": row + 1,
                "effect_allele": "A",
                "other_allele": "G",
                "beta": float(expected[row]),
                "standard_error": float(expected[row]),
                "effect_allele_frequency": float(frequencies[row]),
            }
            for row in range(len(frequencies))
        ],
    )
    manifest = tmp_path / "manifest.tsv"
    _write_manifest(
        manifest,
        [
            {
                "analysis_index": 0,
                "analysis_id": "trait",
                "trait_id": "T1",
                "analysis_label": "Trait",
                "n": 1000,
                "mhc": "FALSE",
                "filtered_file": "trait.tsv.gz",
            }
        ],
    )
    out = tmp_path / "out.opengwasdb"

    build_ragged_from_ssf(
        manifest,
        filtered_dir,
        out,
        store_id="test",
        release_id="v1",
        allow_unverified_eaf=True,
    )

    assert StoreManifest.load(out).encoding.se.is_residual
    with query_store(out) as query:
        result = query.analysis("trait")
    np.testing.assert_allclose(result["se"], expected, rtol=0.01)
    assert validate_store(out).ok


def test_manifest_fields(tmp_path):
    import json

    manifest, filtered_dir = _make_fixture(tmp_path)
    out = tmp_path / "out.opengwasdb"
    build_ragged_from_ssf(manifest, filtered_dir, out, store_id="test", release_id="v1")

    data = json.loads((out / "manifest.json").read_text())
    assert data["primary_layout"] == "ragged"
    assert data["completion_state"] == "observed_only"
    assert data["association_coverage"] == "cis_and_signals"
    assert data["reference_assembly"] == "GRCh38"
    assert data["provenance"]["stored_effect_scale"] == "sd"


def test_stored_effect_scale_populated_per_analysis(tmp_path):
    """Issue #69: stored_effect_scale is a real per-Analysis analyses.tsv
    column, not only a store-wide value threaded through provenance."""
    from opengwasdb.model.analyses import read_analyses

    manifest, filtered_dir = _make_fixture(tmp_path)
    out = tmp_path / "out.opengwasdb"
    build_ragged_from_ssf(
        manifest,
        filtered_dir,
        out,
        store_id="test",
        release_id="v1",
        stored_effect_scale="log_or",
    )

    table = read_analyses(out / "analyses.tsv")
    assert {r["analysis_id"]: r["stored_effect_scale"] for r in table.rows} == {
        "trait_a": "log_or",
        "trait_b": "log_or",
    }


def test_analyses_tsv_carries_trait_positions(tmp_path):
    # traits.tsv.gz (issue 034) is retired (issue #69): trait_chr/trait_bp
    # live only in analyses.tsv now, and range_by_analysis() reads them
    # directly rather than through a second, independently-shaped file.
    from opengwasdb.model.analyses import read_analyses
    from opengwasdb.query import query_store

    manifest, filtered_dir = _make_fixture(tmp_path)
    out = tmp_path / "out.opengwasdb"
    build_ragged_from_ssf(manifest, filtered_dir, out, store_id="test", release_id="v1")
    assert not (out / "traits.tsv.gz").exists()

    table = read_analyses(out / "analyses.tsv")
    assert {r["analysis_id"] for r in table.rows} == {"trait_a", "trait_b"}

    q = query_store(out)
    result = q.range_by_analysis("1", 100_000, 200_000)
    q.close()
    assert set(result["analysis_index"].tolist()) == {0}  # trait_a only (chr1:150_000)


def test_zarr_csr_associations_and_z_values(tmp_path):
    manifest, filtered_dir = _make_fixture(tmp_path)
    out = tmp_path / "out.opengwasdb"
    build_ragged_from_ssf(manifest, filtered_dir, out, store_id="test", release_id="v1")

    csr = RaggedCSRReader(out)
    assert csr.n_analyses == 2
    assert csr.n_associations == 4

    a0 = csr.get_analysis(0)
    a1 = csr.get_analysis(1)
    assert len(a0.variant_index) == 2  # the three malformed rows were dropped
    assert len(a1.variant_index) == 2

    # trait_a row1: A canonical (A<G), no flip, beta/se = 1.0/0.5 = 2.0
    variants = VariantAxis(out).all()
    alid_by_index = {v.variant_index: v.alid for v in variants}
    a0_z = dict(zip((alid_by_index[int(i)] for i in a0.variant_index), a0.z, strict=True))
    a0_se = dict(zip((alid_by_index[int(i)] for i in a0.variant_index), a0.se, strict=True))
    assert a0_z["1:100000:A:G"] == pytest.approx(2.0, rel=5e-3)
    assert a0_se["1:100000:A:G"] == pytest.approx(0.5, rel=5e-3)
    # trait_a row2: G/A -> A canonical -> flip -> z = -(1.0/0.25) = -4.0; se unaffected by flip
    assert a0_z["1:200000:A:G"] == pytest.approx(-4.0, rel=5e-3)
    assert a0_se["1:200000:A:G"] == pytest.approx(0.25, rel=5e-3)

    a1_z = dict(zip((alid_by_index[int(i)] for i in a1.variant_index), a1.z, strict=True))
    a1_se = dict(zip((alid_by_index[int(i)] for i in a1.variant_index), a1.se, strict=True))
    assert a1_z["2:500000:C:T"] == pytest.approx(4.0, rel=5e-3)
    assert a1_se["2:500000:C:T"] == pytest.approx(0.5, rel=5e-3)
    assert a1_z["1:100000:A:G"] == pytest.approx(2.0, rel=5e-3)  # shared variant
    assert a1_se["1:100000:A:G"] == pytest.approx(0.25, rel=5e-3)


def test_rsid_prefers_rsid_column_falls_back_to_variant_id(tmp_path):
    manifest, filtered_dir = _make_fixture(tmp_path)
    out = tmp_path / "out.opengwasdb"
    build_ragged_from_ssf(manifest, filtered_dir, out, store_id="test", release_id="v1")

    variants = VariantAxis(out).all()
    rsid_by_alid = {v.alid: v.rsid for v in variants}
    assert rsid_by_alid["1:100000:A:G"] == "rs1"  # from trait_a's rsid column
    assert rsid_by_alid["1:200000:A:G"] == "rs2"
    assert rsid_by_alid["2:500000:C:T"] == "rs5fb"  # rsid column blank, variant_id fallback


def test_top_hit_index_built_inline(tmp_path):
    manifest, filtered_dir = _make_fixture(tmp_path)
    out = tmp_path / "out.opengwasdb"
    build_ragged_from_ssf(manifest, filtered_dir, out, store_id="test", release_id="v1")

    root = open_store(out).arrays(mode="r")
    assert "top_hits" in root
    assert len(list(root["top_hits"].keys())) > 0


def test_validate_store_passes(tmp_path):
    from opengwasdb.validation import validate_store

    manifest, filtered_dir = _make_fixture(tmp_path)
    out = tmp_path / "out.opengwasdb"
    build_ragged_from_ssf(manifest, filtered_dir, out, store_id="test", release_id="v1")

    result = validate_store(out)
    assert result.ok, result.errors


def test_overwrite_flag(tmp_path):
    manifest, filtered_dir = _make_fixture(tmp_path)
    out = tmp_path / "out.opengwasdb"

    build_ragged_from_ssf(manifest, filtered_dir, out, store_id="test", release_id="v1")
    with pytest.raises(FileExistsError):
        build_ragged_from_ssf(manifest, filtered_dir, out, store_id="test", release_id="v1")

    build_ragged_from_ssf(
        manifest, filtered_dir, out, store_id="test", release_id="v1", overwrite=True
    )
    assert out.exists()


def test_invalid_stored_effect_scale_rejected(tmp_path):
    manifest, filtered_dir = _make_fixture(tmp_path)
    out = tmp_path / "out.opengwasdb"

    with pytest.raises(ValueError, match="stored_effect_scale"):
        build_ragged_from_ssf(
            manifest,
            filtered_dir,
            out,
            store_id="test",
            release_id="v1",
            stored_effect_scale="sd_units",
        )
    assert not out.exists()  # fails before any I/O


def test_non_dense_analysis_index_fails_loudly(tmp_path):
    filtered_dir = tmp_path / "filtered"
    filtered_dir.mkdir()
    _write_one_association(filtered_dir)
    manifest = tmp_path / "manifest.tsv"
    # analysis_index jumps 0 -> 2, not dense.
    _write_manifest(
        manifest,
        [
            {
                "analysis_index": 0,
                "analysis_id": "trait_a",
                "trait_id": "T1",
                "filtered_file": "trait_a.tsv.gz",
            },
            {
                "analysis_index": 2,
                "analysis_id": "trait_b",
                "trait_id": "T2",
                "filtered_file": "trait_a.tsv.gz",
            },
        ],
    )
    out = tmp_path / "out.opengwasdb"

    with pytest.raises(ValueError, match="analysis_index must be 0..n-1"):
        build_ragged_from_ssf(manifest, filtered_dir, out, store_id="test", release_id="v1")


def test_manifest_without_trait_id_builds(tmp_path):
    """A gene-target-less Store Family (e.g. small-molecule metabolomics) has
    no single encoding gene, so its manifest omits trait_id entirely -- per
    docs/release-metadata-schema.md's documented convention that its absence
    (not a blank value) is how a reviewer tells the two family shapes apart.
    trait_id is unused by this builder's own output, so a manifest without it
    must build identically to one with it."""
    filtered_dir = tmp_path / "filtered"
    filtered_dir.mkdir()
    _write_one_association(filtered_dir)
    manifest = tmp_path / "manifest.tsv"
    # The same manifest as every other test's, with `trait_id` absent rather
    # than blank — which is the distinction this test exists to make.
    header = [column for column in _MANIFEST_HEADER if column != "trait_id"]
    with open(manifest, "w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        row = {
            "analysis_index": 0,
            "analysis_id": "metabolite_a",
            "analysis_label": "Metabolite A",
            "filtered_file": "trait_a.tsv.gz",
        }
        fh.write("\t".join(str(row.get(col, "")) for col in header) + "\n")
    out = tmp_path / "out.opengwasdb"

    result = build_ragged_from_ssf(manifest, filtered_dir, out, store_id="test", release_id="v1")

    assert result.n_analyses == 1
    assert result.n_variants == 1
    from opengwasdb.validation import validate_store

    assert validate_store(out).ok


def test_duplicate_canonical_variant_within_analysis_is_resolved(tmp_path):
    """issue #101: a GWAS-SSF file can carry more than one row that
    canonicalizes to the same variant within one analysis. Rows with
    identical (z, se) are a harmless duplicate submission and collapse to
    one association; rows that disagree are dropped for that cell entirely
    rather than silently keeping an arbitrary one."""
    filtered_dir = tmp_path / "filtered"
    filtered_dir.mkdir()
    _write_filtered(
        filtered_dir / "trait_a.tsv.gz",
        [
            # 1:100000 A/G appears twice with identical beta/se -> collapses to one.
            {
                "chromosome": "1",
                "base_pair_location": 100_000,
                "effect_allele": "A",
                "other_allele": "G",
                "beta": 1.0,
                "standard_error": 0.5,
                "rsid": "rs1",
            },
            {
                "chromosome": "1",
                "base_pair_location": 100_000,
                "effect_allele": "A",
                "other_allele": "G",
                "beta": 1.0,
                "standard_error": 0.5,
                "rsid": "rs1",
            },
            # 1:200000 A/G appears twice with conflicting beta -> dropped entirely.
            {
                "chromosome": "1",
                "base_pair_location": 200_000,
                "effect_allele": "A",
                "other_allele": "G",
                "beta": 0.5,
                "standard_error": 0.2,
                "rsid": "rs2",
            },
            {
                "chromosome": "1",
                "base_pair_location": 200_000,
                "effect_allele": "A",
                "other_allele": "G",
                "beta": -0.5,
                "standard_error": 0.2,
                "rsid": "rs2",
            },
            # 1:300000 A/G, single row, unaffected control.
            {
                "chromosome": "1",
                "base_pair_location": 300_000,
                "effect_allele": "A",
                "other_allele": "G",
                "beta": 2.0,
                "standard_error": 1.0,
                "rsid": "rs3",
            },
        ],
    )
    manifest = tmp_path / "manifest.tsv"
    _write_manifest(
        manifest,
        [
            {
                "analysis_index": 0,
                "analysis_id": "trait_a",
                "trait_id": "T1",
                "filtered_file": "trait_a.tsv.gz",
            },
        ],
    )
    out = tmp_path / "out.opengwasdb"

    result = build_ragged_from_ssf(manifest, filtered_dir, out, store_id="test", release_id="v1")

    # 1:100000 (collapsed) + 1:300000 (control) survive; 1:200000 (conflict) dropped.
    assert result.n_associations == 2

    csr = RaggedCSRReader(out)
    a0 = csr.get_analysis(0)
    variants = VariantAxis(out).all()
    alid_by_index = {v.variant_index: v.alid for v in variants}
    surviving_alids = {alid_by_index[int(i)] for i in a0.variant_index}
    assert surviving_alids == {"1:100000:A:G", "1:300000:A:G"}

    from opengwasdb.validation import validate_store

    assert validate_store(out).ok


def test_cli_build_ragged_ssf(tmp_path):
    from typer.testing import CliRunner

    from opengwasdb.cli.main import app

    manifest, filtered_dir = _make_fixture(tmp_path)
    out = tmp_path / "out.opengwasdb"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "build-ragged-ssf",
            str(manifest),
            str(filtered_dir),
            str(out),
            "--store-id",
            "cli-test",
            "--release-id",
            "v1",
        ],
    )
    assert result.exit_code == 0, result.output
    import json

    summary = json.loads(result.output.strip().splitlines()[-1])
    assert summary["n_variants"] == 3
    assert summary["n_analyses"] == 2

    validate = runner.invoke(app, ["validate", str(out)])
    assert validate.exit_code == 0, validate.output
