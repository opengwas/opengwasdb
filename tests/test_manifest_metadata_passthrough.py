"""Manifest Analytical + Attribution Metadata passthrough (issues #86, #83).

Every layout's builder must carry the shared-core `analyses.tsv` columns a
manifest supplies into the built store verbatim, and must leave them blank
rather than fabricate them when the manifest omits them. Dense/Hybrid got
this in #86; Ragged in #83.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.layouts.ragged.build_ssf import build_ragged_from_ssf
from opengwasdb.model.analyses import read_analyses


def _source_vcf(tmp_path: Path) -> Path:
    path = tmp_path / "binary.vcf"
    path.write_text(
        "##fileformat=VCFv4.2\n"
        '##FORMAT=<ID=ES,Number=A,Type=Float,Description="Effect size">\n'
        '##FORMAT=<ID=SE,Number=A,Type=Float,Description="Standard error">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSTUDY1\n"
        "1\t100000\t.\tA\tG\t.\tPASS\t.\tES:SE\t0.6:0.3\n",
        encoding="utf-8",
    )
    return path


def _manifest(tmp_path: Path, source: Path) -> Path:
    path = tmp_path / "manifest.tsv"
    path.write_text(
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale"
        "\toriginal_sd_method\toriginal_sd\tassigned_ancestry"
        "\tancestry_assignment_method\tsample_size_kind\tsample_size_scope"
        "\tn_cases\tn_controls\toriginal_effect_scale\n"
        f"binary\t{source}\tBinary trait\t10000\tlog_or\tbinary_trait\t\tEUR"
        "\tsource_trusted_no_af\tcase_control\tanalysis_level\t1000\t9000\tlog_or\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("layout", ["dense", "hybrid"])
def test_manifest_analytical_metadata_passes_through_unchanged(tmp_path: Path, layout: str):
    source = _source_vcf(tmp_path)
    manifest = _manifest(tmp_path, source)
    store = tmp_path / f"{layout}.opengwasdb"

    if layout == "dense":
        build_dense_from_vcf_manifest(
            manifest, store, store_id="metadata-test", release_id="r1"
        )
        analyses_paths = [store / "analyses.tsv"]
    else:
        panel = tmp_path / "panel.txt"
        panel.write_text("1:100000:A:G\n", encoding="utf-8")
        build_hybrid_from_vcf_manifest(
            manifest,
            store,
            reference_panel=panel,
            store_id="metadata-test",
            release_id="r1",
        )
        analyses_paths = [store / "analyses.tsv", store / "dense" / "analyses.tsv"]

    expected = {
        "sample_size_kind": "case_control",
        "sample_size_scope": "analysis_level",
        "sample_size": "10000",
        "n_cases": "1000",
        "n_controls": "9000",
        "original_effect_scale": "log_or",
        "ancestry_assignment_method": "source_trusted_no_af",
    }
    for analyses_path in analyses_paths:
        row = read_analyses(analyses_path).rows[0]
        assert {column: row[column] for column in expected} == expected


# ── Ragged (issue #83) ────────────────────────────────────────────────────────
#
# The Ragged SSF builder read none of the columns above, so a Ragged store
# built from a manifest that carried them silently lost every one of them --
# blank columns in the built `analyses.tsv`, no warning, no error. Same
# passthrough contract as Dense/Hybrid above, exercised through Ragged's own
# (differently shaped) manifest.

_RAGGED_SSF_HEADER = (
    "chromosome\tbase_pair_location\teffect_allele\tother_allele\tbeta\tstandard_error\trsid\n"
)


def _ragged_source(tmp_path: Path) -> Path:
    filtered_dir = tmp_path / "filtered"
    filtered_dir.mkdir()
    with gzip.open(filtered_dir / "trait_a.tsv.gz", "wt", encoding="utf-8") as fh:
        fh.write(_RAGGED_SSF_HEADER)
        fh.write("1\t100000\tA\tG\t0.6\t0.3\trs1\n")
    return filtered_dir


def _ragged_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "ragged_manifest.tsv"
    path.write_text(
        "analysis_index\tanalysis_id\tanalysis_label\tfiltered_file\tn"
        "\tassigned_ancestry\tancestry_assignment_method\tsample_size_kind"
        "\tsample_size_scope\tn_cases\tn_controls\toriginal_effect_scale"
        "\toriginal_sd_method\tlicense\tpublication_doi\tpublication_pmid"
        "\tconsortium\tfirst_author\tancestry_prop_EUR\n"
        "0\tbinary\tBinary trait\ttrait_a.tsv.gz\t10000"
        "\tEUR\tsource_trusted_no_af\tcase_control"
        "\tanalysis_level\t1000\t9000\tlog_or"
        "\tbinary_trait\tCC-BY-4.0\t10.1000/xyz\t12345678"
        "\tGIANT\tSmith\t0.98\n",
        encoding="utf-8",
    )
    return path


def test_ragged_manifest_analytical_metadata_passes_through_unchanged(tmp_path: Path):
    filtered_dir = _ragged_source(tmp_path)
    manifest = _ragged_manifest(tmp_path)
    store = tmp_path / "ragged.opengwasdb"

    build_ragged_from_ssf(
        manifest, filtered_dir, store, store_id="metadata-test", release_id="r1",
        association_coverage="cis_and_signals",
    )

    row = read_analyses(store / "analyses.tsv").rows[0]
    expected = {
        "sample_size_kind": "case_control",
        "sample_size_scope": "analysis_level",
        "sample_size": "10000",
        "n_cases": "1000",
        "n_controls": "9000",
        "original_effect_scale": "log_or",
        "original_sd_method": "binary_trait",
        "ancestry_assignment_method": "source_trusted_no_af",
        "license": "CC-BY-4.0",
        "publication_doi": "10.1000/xyz",
        "publication_pmid": "12345678",
        "consortium": "GIANT",
        "first_author": "Smith",
        "ancestry_prop_EUR": "0.98",
    }
    assert {column: row[column] for column in expected} == expected


def test_ragged_manifest_without_optional_columns_leaves_them_blank(tmp_path: Path):
    """Absent is blank, never fabricated -- the manifest producer owns these."""
    filtered_dir = _ragged_source(tmp_path)
    manifest = tmp_path / "bare_manifest.tsv"
    manifest.write_text(
        "analysis_index\tanalysis_id\tfiltered_file\tn\n0\tbinary\ttrait_a.tsv.gz\t10000\n",
        encoding="utf-8",
    )
    store = tmp_path / "bare.opengwasdb"

    build_ragged_from_ssf(
        manifest, filtered_dir, store, store_id="metadata-test", release_id="r1",
        association_coverage="cis_and_signals",
    )

    row = read_analyses(store / "analyses.tsv").rows[0]
    blank = (
        "ancestry_assignment_method", "sample_size_kind", "sample_size_scope",
        "n_cases", "n_controls", "original_effect_scale", "original_sd_method",
        "license", "publication_doi", "publication_pmid", "consortium", "first_author",
    )
    assert all(row[column] == "" for column in blank)
    assert row["sample_size"] == "10000"
