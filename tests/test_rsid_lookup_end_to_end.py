"""rsid lookups resolve against a built store, in every layout (issue #109).

The bug this covers is silent: before it, `ogdb query-phewas <store> rs123`
printed a header row and nothing else, exactly as it would for a variant with
no associations, even though the store's own `variants.tsv.gz` named the
variant. Dense and Hybrid additionally never captured rsids at all -- their
source readers dropped them -- so the fix spans reader, builder and axis; a
test at any one of those layers alone would have passed while the store stayed
unqueryable.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.layouts.ragged.build_ssf import build_ragged_from_ssf
from opengwasdb.query import query_store
from opengwasdb.variants import VariantAxis

_ALID = "1:100000:A:G"
_RSID = "rs16902359"


def _vcf(tmp_path: Path) -> Path:
    """A GWAS-VCF naming its variant in the ID column, as real ones do."""
    path = tmp_path / "study.vcf"
    path.write_text(
        "##fileformat=VCFv4.2\n"
        '##FILTER=<ID=PASS,Description="All filters passed">\n'
        '##FORMAT=<ID=ES,Number=A,Type=Float,Description="Effect size">\n'
        '##FORMAT=<ID=SE,Number=A,Type=Float,Description="Standard error">\n'
        '##FORMAT=<ID=EZ,Number=A,Type=Float,Description="Z-score">\n'
        "##SAMPLE=<ID=STUDY1,StudyType=Continuous>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSTUDY1\n"
        f"1\t100000\t{_RSID}\tG\tA\t.\tPASS\t.\tES:SE\t0.6:0.3\n",
        encoding="utf-8",
    )
    return path


def _vcf_manifest(tmp_path: Path, source: Path) -> Path:
    path = tmp_path / "manifest.tsv"
    path.write_text(
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale\toriginal_sd_method\n"
        f"study1\t{source}\tStudy 1\t10000\tsd\tdeclared_standardised\n",
        encoding="utf-8",
    )
    return path


def _ssf_source(tmp_path: Path) -> Path:
    filtered_dir = tmp_path / "filtered"
    filtered_dir.mkdir()
    with gzip.open(filtered_dir / "study1.tsv.gz", "wt", encoding="utf-8") as fh:
        fh.write(
            "chromosome\tbase_pair_location\teffect_allele\tother_allele"
            "\tbeta\tstandard_error\trsid\n"
        )
        fh.write(f"1\t100000\tA\tG\t0.6\t0.3\t{_RSID}\n")
    return filtered_dir


def _build(tmp_path: Path, layout: str) -> Path:
    store = tmp_path / f"{layout}.opengwasdb"
    if layout == "ragged":
        manifest = tmp_path / "ragged_manifest.tsv"
        manifest.write_text(
            "analysis_index\tanalysis_id\tfiltered_file\tn\n0\tstudy1\tstudy1.tsv.gz\t10000\n",
            encoding="utf-8",
        )
        build_ragged_from_ssf(
            manifest, _ssf_source(tmp_path), store, store_id="rsid-test", release_id="r1",
            association_coverage="cis_and_signals",
        )
        return store

    manifest = _vcf_manifest(tmp_path, _vcf(tmp_path))
    if layout == "dense":
        build_dense_from_vcf_manifest(manifest, store, store_id="rsid-test", release_id="r1")
        return store

    panel = tmp_path / "panel.txt"
    panel.write_text(f"{_ALID}\n", encoding="utf-8")
    build_hybrid_from_vcf_manifest(
        manifest, store, reference_panel=panel, store_id="rsid-test", release_id="r1"
    )
    return store


@pytest.mark.parametrize("layout", ["dense", "ragged", "hybrid"])
def test_built_store_records_the_source_rsid(tmp_path: Path, layout: str):
    store = _build(tmp_path, layout)
    axis = VariantAxis(store)
    try:
        record = axis.by_index(0)
        assert record is not None
        assert record.alid == _ALID
        assert record.rsid == _RSID
    finally:
        axis.close()


@pytest.mark.parametrize("layout", ["dense", "ragged", "hybrid"])
def test_rsid_resolves_to_the_same_variant_as_its_alid(tmp_path: Path, layout: str):
    store = _build(tmp_path, layout)
    axis = VariantAxis(store)
    try:
        by_rsid = axis.by_identifier(_RSID)
        by_alid = axis.by_identifier(_ALID)
        assert by_rsid is not None, "rsid lookup returned nothing -- the issue #109 bug"
        assert by_rsid == by_alid
    finally:
        axis.close()


@pytest.mark.parametrize("layout", ["dense", "ragged", "hybrid"])
def test_phewas_by_rsid_matches_phewas_by_alid(tmp_path: Path, layout: str):
    """The user-facing shape of the bug: same variant, two names, one answer."""
    store = _build(tmp_path, layout)
    with query_store(store) as query:
        by_rsid = query.phewas(_RSID)
        by_alid = query.phewas(_ALID)

    assert len(by_alid["z"]) == 1
    assert by_rsid["variant_index"].tolist() == by_alid["variant_index"].tolist()
    assert by_rsid["z"].tolist() == by_alid["z"].tolist()
