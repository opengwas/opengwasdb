from __future__ import annotations

import gzip
from pathlib import Path
from shutil import copytree
from typing import TYPE_CHECKING

import pytest
import shared_releases

from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.layouts.ragged.build_ssf import build_ragged_from_ssf

if TYPE_CHECKING:
    from test_ragged_residual_completion import RaggedResidualScenario

SOURCE_HEADER = "\t".join(
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


SOURCE_ROWS = [
    # Generated tiny fixture. Shape intentionally has overlap and missing cells.
    "a1\tp1\tHeight\tHeight primary\t1\t100\tA\tG\t2.0\t0.1\trs1\tsd",
    "a2\tp2\tDisease\tDisease primary\t1\t100\tA\tG\t6.0\t0.2\trs1\tlog_or",
    # Source effect allele is not canonical A1, so stored z is negated.
    "a1\tp1\tHeight\tHeight primary\t1\t200\tT\tC\t3.0\t0.2\trs2\tsd",
    # Source effect allele is not canonical A1, so stored z becomes positive.
    "a2\tp2\tDisease\tDisease primary\t1\t300\tG\tA\t-6.0\t0.5\trs3\tlog_or",
]


@pytest.fixture
def source_path(tmp_path: Path) -> Path:
    path = tmp_path / "associations.tsv"
    path.write_text(SOURCE_HEADER + "\n" + "\n".join(SOURCE_ROWS) + "\n", encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def dense_fixture_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """`shared_releases.dense_fixture_store`, built once per session."""
    return shared_releases.dense_fixture_store(tmp_path_factory.mktemp("dense-fixture"))


@pytest.fixture
def dense_store_path(tmp_path: Path, dense_fixture_store: Path) -> Path:
    """A private copy of the shared fixture Store, which is never written to."""
    return Path(copytree(dense_fixture_store, tmp_path / "store.opengwasdb"))


@pytest.fixture
def ragged_store_path(tmp_path):
    """A minimal Ragged Observed-Only store (SSF ingestion path)."""
    header = [
        "chromosome", "base_pair_location", "effect_allele", "other_allele",
        "beta", "standard_error", "rsid",
    ]
    rows = [
        {"chromosome": "1", "base_pair_location": 100, "effect_allele": "A",
         "other_allele": "G", "beta": 0.3, "standard_error": 0.1, "rsid": "rs1"},
        {"chromosome": "1", "base_pair_location": 200, "effect_allele": "A",
         "other_allele": "G", "beta": -0.2, "standard_error": 0.1, "rsid": "rs2"},
    ]
    filtered_dir = tmp_path / "filtered"
    filtered_dir.mkdir()
    with gzip.open(filtered_dir / "t1.tsv.gz", "wt", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for row in rows:
            fh.write("\t".join(str(row[c]) for c in header) + "\n")

    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "analysis_index\tanalysis_id\ttrait_id\t"
        "analysis_label\ttrait_ontology_id\ttrait_ontology_label\ttrait_chr\ttrait_bp\t"
        "n\ttissue\tcontext\tmhc\tfiltered_file\tsource_genome_build\n"
        "0\tt1\tT1\tGENE1\tENSEMBL:ENSG1\tEnsembl\t1\t150\t1000\tBlood\t\tFALSE\tt1.tsv.gz\tGRCh38\n",
        encoding="utf-8",
    )
    store_path = tmp_path / "ragged.opengwasdb"
    build_ragged_from_ssf(
        manifest, filtered_dir, store_path, store_id="fixture", release_id="v1",
        association_coverage="cis_and_signals",
    )
    return store_path


@pytest.fixture
def hybrid_store_path(tmp_path):
    """A minimal Hybrid store: two on-panel analyses, so its Dense Component
    (dense_component_path) is where index.sqlite/analyses.tsv actually live."""
    header = (
        "##fileformat=VCFv4.2\n"
        "##FORMAT=<ID=ES,Number=A,Type=Float,Description=\"Effect size\">\n"
        "##FORMAT=<ID=SE,Number=A,Type=Float,Description=\"Standard error\">\n"
        "##SAMPLE=<ID=S,StudyType=Continuous>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS\n"
    )
    vcf_a = tmp_path / "trait_a.vcf"
    vcf_a.write_text(header + "1\t100000\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n", encoding="utf-8")
    vcf_b = tmp_path / "trait_b.vcf"
    vcf_b.write_text(header + "1\t100000\t.\tA\tG\t.\tPASS\t.\tES:SE\t1.0:0.4\n", encoding="utf-8")
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale"
        "\toriginal_sd_method\toriginal_sd\n"
        f"trait_a\t{vcf_a}\tTrait A\t1000\tsd\tdeclared_standardised\t\n"
        f"trait_b\t{vcf_b}\tTrait B\t1000\tsd\tdeclared_standardised\t\n",
        encoding="utf-8",
    )
    panel = tmp_path / "panel.txt"
    panel.write_text("1:100000:A:G\n", encoding="utf-8")
    store_path = tmp_path / "hybrid.opengwasdb"
    build_hybrid_from_vcf_manifest(
        manifest, store_path, reference_panel=panel, store_id="fixture", release_id="v1"
    )
    return store_path


@pytest.fixture(scope="session")
def residual_ragged_scenario(
    tmp_path_factory: pytest.TempPathFactory,
) -> RaggedResidualScenario:
    """`shared_releases.residual_ragged`, built once per session."""
    return shared_releases.residual_ragged(tmp_path_factory)


@pytest.fixture(scope="session")
def panel_only_eaf_ragged_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """`shared_releases.panel_only_eaf_ragged`, built once per session."""
    return shared_releases.panel_only_eaf_ragged(
        tmp_path_factory.mktemp("panel-only-eaf-ragged")
    )
