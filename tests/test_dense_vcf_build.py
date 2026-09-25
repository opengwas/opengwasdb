"""End-to-end tests for the two-pass VCF dense build pipeline.

All fixtures use synthetic GWAS-VCF files written to tmp_path.
Real pyliftover is used with known hg19 positions that map successfully to hg38.

Known positions:
  hg19 1:100000  → hg38 1:100000   (REF=A, ALT=G → ALID 1:100000:A:G, flip=True  → z=-z)
  hg19 1:1000000 → hg38 1:1064620  (REF=C, ALT=T → ALID 1:1064620:C:T, flip=True  → z=-z)
  hg19 1:1500000 → hg38 1:1564620  (REF=G, ALT=A → ALID 1:1564620:A:G, flip=False → z unchanged)
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
from cli_output import normalize_cli_output
from residual_fixtures import write_gwas_vcf_with_eaf
from store_assertions import assert_same_band_arrays, assert_same_top_hits

from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
from opengwasdb.layouts.dense.top_hits import threshold_key
from opengwasdb.model.analyses import read_analyses
from opengwasdb.query import query_store
from opengwasdb.readers import GWAS_SSF_CAPABILITY
from opengwasdb.store.open import open_store
from opengwasdb.validation import validate_store
from opengwasdb.variants.reference import read_variant_reference, write_variant_reference

# hg19 positions used in fixtures and their expected hg38 positions
HG19_POS_1 = 100_000   # → hg38 100000  REF=A ALT=G  (ALT>REF → flip, stored z = -z)
HG19_POS_2 = 1_000_000  # → hg38 1064620 REF=C ALT=T  (ALT>REF → flip, stored z = -z)
HG19_POS_3 = 1_500_000  # → hg38 1564620 REF=G ALT=A  (ALT<REF → no flip)

HG38_ALID_1 = "1:100000:A:G"
HG38_ALID_2 = "1:1064620:C:T"
HG38_ALID_3 = "1:1564620:A:G"


def _vcf_header(study_type: str = "Continuous") -> str:
    return (
        "##fileformat=VCFv4.2\n"
        "##FILTER=<ID=PASS,Description=\"All filters passed\">\n"
        "##FORMAT=<ID=ES,Number=A,Type=Float,Description=\"Effect size\">\n"
        "##FORMAT=<ID=SE,Number=A,Type=Float,Description=\"Standard error\">\n"
        "##FORMAT=<ID=EZ,Number=A,Type=Float,Description=\"Z-score\">\n"
        f"##SAMPLE=<ID=STUDY1,StudyType={study_type}>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSTUDY1\n"
    )


def _make_vcf(tmp_path: Path, name: str, rows: list[str], study_type: str = "Continuous") -> Path:
    path = tmp_path / f"{name}.vcf"
    path.write_text(_vcf_header(study_type) + "".join(rows), encoding="utf-8")
    return path


def _make_manifest(
    tmp_path: Path,
    entries: list[tuple[str, Path, str]],
    scales: dict[str, str] | None = None,
    sd_methods: dict[str, str] | None = None,
    sds: dict[str, str] | None = None,
) -> Path:
    """Write the build manifest: trait_id/file_path/trait_name/n (also the
    Analysis Catalogue's BUILD_COLUMNS) plus the required stored_effect_scale
    (issue #17) and original_sd_method/original_sd (issue #18). `scales`/
    `sd_methods`/`sds` override the value per trait_id (defaults `"sd"` and
    `"declared_standardised"` -- i.e. no rescaling, `original_sd` blank),
    letting a test declare values that disagree with its VCF's own
    ``##SAMPLE`` header -- the manifest always wins.
    """
    scales = scales or {}
    sd_methods = sd_methods or {}
    sds = sds or {}
    manifest = tmp_path / "manifest.tsv"
    lines = [
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale"
        "\toriginal_sd_method\toriginal_sd"
    ]
    for trait_id, file_path, trait_name in entries:
        scale = scales.get(trait_id, "sd")
        sd_method = sd_methods.get(trait_id, "declared_standardised")
        sd = sds.get(trait_id, "")
        lines.append(
            f"{trait_id}\t{file_path}\t{trait_name}\t1000\t{scale}\t{sd_method}\t{sd}"
        )
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


@pytest.fixture
def two_trait_store(tmp_path):
    """Store built from two VCF fixtures with three variants each."""
    vcf1 = _make_vcf(
        tmp_path,
        "trait_a",
        [
            f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n",  # z=4.0, flip→-4.0
            f"1\t{HG19_POS_2}\t.\tC\tT\t.\tPASS\t.\tES:SE\t1.5:0.3\n",  # z=5.0, flip→-5.0
            f"1\t{HG19_POS_3}\t.\tG\tA\t.\tPASS\t.\tES:SE\t0.6:0.2\n",  # z=3.0, no flip
        ],
    )
    vcf2 = _make_vcf(
        tmp_path,
        "trait_b",
        [
            f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t6.0:0.5\n",  # z=12.0, flip→-12.0
            f"1\t{HG19_POS_3}\t.\tG\tA\t.\tPASS\t.\tES:SE\t1.2:0.3\n",  # z=4.0, no flip
        ],
        # The ieu-a-7 scenario: header says Continuous, but this is really a
        # case-control trait -- the manifest (below) declares log_or, and
        # that value must win (issue #17), not this header.
        study_type="Continuous",
    )
    manifest = _make_manifest(
        tmp_path,
        [("trait_a", vcf1, "Trait A"), ("trait_b", vcf2, "Trait B")],
        scales={"trait_b": "log_or"},
    )
    store_path = tmp_path / "store.opengwasdb"
    build_dense_from_vcf_manifest(
        manifest,
        store_path,
        store_id="test-store",
        release_id="v1",
    )
    return store_path


def test_build_creates_standard_store_envelope(two_trait_store):
    assert (two_trait_store / "manifest.json").exists()
    assert (two_trait_store / "index.sqlite").exists()
    assert (two_trait_store / "data.zarr").exists()
    assert (two_trait_store / "variants.tsv.gz").exists()
    assert (two_trait_store / "variant_offsets.npy").exists()


def test_validate_store_passes(two_trait_store):
    result = validate_store(two_trait_store)
    assert result.ok, result.errors


def test_manifest_json_has_grch38_assembly(two_trait_store):
    manifest = json.loads((two_trait_store / "manifest.json").read_text())
    assert manifest["reference_assembly"] == "GRCh38"
    assert manifest["completion_state"] == "observed_only"


def test_store_has_correct_dimensions(two_trait_store):
    root = open_store(two_trait_store).arrays(mode="r")
    assert root["z"].shape == (3, 2)
    assert root["se"].shape == (3, 2)


def test_allele_flip_z_negated_when_alt_not_a1(two_trait_store):
    """Variants where ALT > REF (A1=REF) should have z negated."""
    query = query_store(two_trait_store)
    # Use lookup to get trait_a's z for ALID_1 directly
    result = query.lookup([HG38_ALID_1], ["trait_a"])
    assert len(result["z"]) == 1
    # ALT=G > REF=A → z was negated; ES=2.0/SE=0.5=4.0 → stored z=-4.0
    assert result["z"][0] == pytest.approx(-4.0, rel=5e-3)


def test_z_not_negated_when_alt_is_a1(two_trait_store):
    """Variants where ALT < REF (A1=ALT) should preserve z sign."""
    query = query_store(two_trait_store)
    result = query.lookup([HG38_ALID_3], ["trait_a"])
    assert len(result["z"]) == 1
    # ALT=A < REF=G → A is A1, no flip; ES=0.6/SE=0.2=3.0 → stored z=3.0
    assert result["z"][0] == pytest.approx(3.0, rel=5e-3)


def test_missing_cells_are_absent(two_trait_store):
    """trait_b does not have a value for HG38_ALID_2; only one analysis returned."""
    query = query_store(two_trait_store)
    result = query.phewas(HG38_ALID_2)
    # Only trait_a has data for variant at HG38_ALID_2
    assert len(result["z"]) == 1
    analyses = query.analyses_table()
    trait_b_idx = next(k for k, v in analyses.items() if v["analysis_id"] == "trait_b")
    assert trait_b_idx not in result["analysis_index"].tolist()


def test_range_query_returns_expected_variants(two_trait_store):
    query = query_store(two_trait_store)
    result = query.range_phewas("1", 50_000, 200_000)
    variants = query.variants_table()
    alids = {variants[int(vi)]["alid"] for vi in result["variant_index"]}
    assert HG38_ALID_1 in alids


def test_analysis_query_returns_all_variants_for_trait(two_trait_store):
    query = query_store(two_trait_store)
    result = query.analysis("trait_a")
    assert len(result["z"]) == 3
    assert all(np.isfinite(result["z"]))


def test_stored_effect_scale_comes_from_manifest_not_header(two_trait_store):
    """The ieu-a-7 fix (issue #17): trait_b's VCF header says
    ``StudyType=Continuous``, but its manifest row declares
    ``stored_effect_scale=log_or`` -- the built store must record the
    manifest's value, not the header's."""
    query = query_store(two_trait_store)
    analyses = query.analyses_table()
    by_id = {v["analysis_id"]: v for v in analyses.values()}
    assert by_id["trait_a"]["stored_effect_scale"] == "sd"
    assert by_id["trait_b"]["stored_effect_scale"] == "log_or"


def test_analyses_tsv_has_no_phenotype_columns(two_trait_store):
    """ADR 0034/issue #68: phenotype_id/phenotype_label are retired with no
    replacement raw-identifier column."""
    table = read_analyses(two_trait_store / "analyses.tsv")
    assert "phenotype_id" not in table.fieldnames
    assert "phenotype_label" not in table.fieldnames
    assert "trait_id" not in table.fieldnames


def test_ontology_and_attribution_columns_blank_when_manifest_omits_them(two_trait_store):
    """A bare manifest supplies no ontology/attribution columns -- those
    fields must be blank, never fabricated (ADR 0034/issue #68)."""
    table = read_analyses(two_trait_store / "analyses.tsv")
    for row in table.rows:
        for column in (
            "trait_ontology_id",
            "trait_ontology_label",
            "license",
            "publication_doi",
            "publication_pmid",
            "consortium",
            "first_author",
        ):
            assert row[column] == ""


def test_ontology_and_attribution_columns_populated_from_manifest(tmp_path):
    """When a manifest supplies trait-ontology/Attribution columns, they flow
    straight into analyses.tsv (ADR 0034/issue #68)."""
    vcf = _make_vcf(
        tmp_path,
        "trait_a",
        [f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n"],
    )
    manifest_path = tmp_path / "manifest.tsv"
    manifest_path.write_text(
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale\toriginal_sd_method"
        "\toriginal_sd\ttrait_ontology_id\ttrait_ontology_label\tlicense"
        "\tpublication_doi\tpublication_pmid\tconsortium\tfirst_author\n"
        f"trait_a\t{vcf}\tTrait A\t1000\tsd\tdeclared_standardised\t\t"
        "EFO:0001073\tbody height\tCC0\t10.1000/xyz\t12345678\tGIANT\tJ. Smith\n",
        encoding="utf-8",
    )
    store_path = tmp_path / "store.opengwasdb"
    build_dense_from_vcf_manifest(manifest_path, store_path, store_id="test-store", release_id="v1")

    table = read_analyses(store_path / "analyses.tsv")
    row = table.rows[0]
    assert row["trait_ontology_id"] == "EFO:0001073"
    assert row["trait_ontology_label"] == "body height"
    assert row["license"] == "CC0"
    assert row["publication_doi"] == "10.1000/xyz"
    assert row["publication_pmid"] == "12345678"
    assert row["consortium"] == "GIANT"
    assert row["first_author"] == "J. Smith"
    assert row["analysis_label"] == "Trait A"


def test_missing_required_manifest_field_fails_the_build_loudly(tmp_path):
    """A manifest missing stored_effect_scale must fail the build with a
    clear error before any I/O, not fall back to VCF-header inference or a
    silent default (issue #17)."""
    rows = [f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n"]
    vcf = _make_vcf(tmp_path, "trait_a", rows)
    manifest_path = tmp_path / "manifest.tsv"
    # No stored_effect_scale column at all.
    manifest_path.write_text(
        "trait_id\tfile_path\ttrait_name\tn\n"
        f"trait_a\t{vcf}\tTrait A\t1000\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="stored_effect_scale"):
        build_dense_from_vcf_manifest(
            manifest_path, tmp_path / "store.opengwasdb", store_id="s", release_id="r"
        )


def test_missing_original_sd_method_fails_the_build_loudly(tmp_path):
    """A manifest missing original_sd_method must fail the build the same way
    a missing stored_effect_scale does (issue #18)."""
    rows = [f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n"]
    vcf = _make_vcf(tmp_path, "trait_a", rows)
    manifest_path = tmp_path / "manifest.tsv"
    manifest_path.write_text(
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale\n"
        f"trait_a\t{vcf}\tTrait A\t1000\tsd\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="original_sd_method"):
        build_dense_from_vcf_manifest(
            manifest_path, tmp_path / "store.opengwasdb", store_id="s", release_id="r"
        )


def test_continuous_trait_rescaled_by_manifest_original_sd(tmp_path):
    """issue #18 AC1: a continuous-trait Analysis with a manifest-supplied
    original_sd != 1 has its se divided by that SD in the built store; z is
    unchanged (z = beta/se is invariant to dividing both by the same
    constant)."""
    vcf = _make_vcf(
        tmp_path,
        "trait_a",
        [f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n"],  # z=4.0, flip→-4.0, se=0.5
    )
    manifest = _make_manifest(
        tmp_path,
        [("trait_a", vcf, "Trait A")],
        sd_methods={"trait_a": "source_provided"},
        sds={"trait_a": "2.0"},
    )
    store_path = tmp_path / "store.opengwasdb"
    build_dense_from_vcf_manifest(manifest, store_path, store_id="s", release_id="r")

    result = query_store(store_path).analysis("trait_a")
    assert result["z"][0] == pytest.approx(-4.0, rel=5e-3)
    assert result["se"][0] == pytest.approx(0.25, rel=5e-3)  # 0.5 / 2.0


def test_binary_trait_never_rescaled(tmp_path):
    """issue #18 AC2: original_sd_method=binary_trait is never rescaled by an
    inapplicable SD scalar, regardless of stored_effect_scale."""
    vcf = _make_vcf(
        tmp_path,
        "trait_b",
        [f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n"],
        study_type="CaseControl",
    )
    manifest = _make_manifest(
        tmp_path,
        [("trait_b", vcf, "Trait B")],
        scales={"trait_b": "log_or"},
        sd_methods={"trait_b": "binary_trait"},
    )
    store_path = tmp_path / "store.opengwasdb"
    build_dense_from_vcf_manifest(manifest, store_path, store_id="s", release_id="r")

    result = query_store(store_path).analysis("trait_b")
    assert result["se"][0] == pytest.approx(0.5, rel=5e-3)


def test_original_sd_method_unavailable_fails_the_build_loudly(tmp_path):
    """issue #18 AC3: original_sd_method=unavailable is handled the same way
    #17 handles any other missing required field -- flagged/failed, not
    silently assumed to be 1."""
    rows = [f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n"]
    vcf = _make_vcf(tmp_path, "trait_a", rows)
    manifest = _make_manifest(
        tmp_path, [("trait_a", vcf, "Trait A")], sd_methods={"trait_a": "unavailable"}
    )

    with pytest.raises(ValueError, match="unavailable"):
        build_dense_from_vcf_manifest(
            manifest, tmp_path / "store.opengwasdb", store_id="s", release_id="r"
        )


def test_sd_rescale_method_without_original_sd_fails_the_build_loudly(tmp_path):
    """A method that carries an SD magnitude (e.g. source_provided) but no
    usable original_sd value must fail loudly rather than silently skip
    rescaling (issue #18)."""
    rows = [f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n"]
    vcf = _make_vcf(tmp_path, "trait_a", rows)
    manifest = _make_manifest(
        tmp_path, [("trait_a", vcf, "Trait A")], sd_methods={"trait_a": "source_provided"}
    )  # original_sd left blank

    with pytest.raises(ValueError, match="original_sd"):
        build_dense_from_vcf_manifest(
            manifest, tmp_path / "store.opengwasdb", store_id="s", release_id="r"
        )


def test_stray_original_sd_without_a_rescale_method_fails_the_build_loudly(tmp_path):
    """A tier that carries no SD magnitude (declared_standardised, binary_trait)
    must reject a stray original_sd value rather than silently ignoring it --
    a manifest declaring both is self-contradictory (issue #18)."""
    rows = [f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n"]
    vcf = _make_vcf(tmp_path, "trait_a", rows)
    manifest = _make_manifest(
        tmp_path,
        [("trait_a", vcf, "Trait A")],
        sd_methods={"trait_a": "declared_standardised"},
        sds={"trait_a": "1.5"},
    )

    with pytest.raises(ValueError, match="original_sd"):
        build_dense_from_vcf_manifest(
            manifest, tmp_path / "store.opengwasdb", store_id="s", release_id="r"
        )


def test_liftover_failure_above_threshold_raises(tmp_path):
    """A manifest where all VCF positions fail liftover raises LiftoverFailureError."""
    from opengwasdb.build.liftover import LiftoverFailureError

    vcf = _make_vcf(
        tmp_path,
        "bad_trait",
        [
            "1\t200000\t.\tA\tG\t.\tPASS\t.\tES:SE\t1.0:0.5\n",
            "1\t300000\t.\tC\tT\t.\tPASS\t.\tES:SE\t0.5:0.2\n",
        ],
    )
    manifest = _make_manifest(tmp_path, [("bad_trait", vcf, "Bad Trait")])

    with pytest.raises(LiftoverFailureError):
        build_dense_from_vcf_manifest(
            manifest,
            tmp_path / "store.opengwasdb",
            store_id="s",
            release_id="r",
            liftover_failure_threshold=0.01,
        )


# --- source_assembly (issue #85): per-row declared source genome build ---


def test_read_manifest_defaults_source_assembly_to_hg19(tmp_path):
    from opengwasdb.layouts.dense.build_vcf import _read_manifest

    vcf = _make_vcf(
        tmp_path, "trait_a", [f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n"]
    )
    manifest = _make_manifest(tmp_path, [("trait_a", vcf, "Trait A")])

    rows = _read_manifest(manifest)

    assert rows[0].source_assembly == "hg19"


def test_read_manifest_normalises_source_assembly_aliases(tmp_path):
    from opengwasdb.layouts.dense.build_vcf import _read_manifest

    vcf = _make_vcf(
        tmp_path, "trait_a", [f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n"]
    )
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale"
        "\toriginal_sd_method\toriginal_sd\tsource_assembly\n"
        f"trait_a\t{vcf}\tTrait A\t1000\tsd\tdeclared_standardised\t\tGRCh38\n",
        encoding="utf-8",
    )

    rows = _read_manifest(manifest)

    assert rows[0].source_assembly == "hg38"


def test_read_manifest_rejects_invalid_source_assembly(tmp_path):
    from opengwasdb.layouts.dense.build_vcf import _read_manifest

    vcf = _make_vcf(
        tmp_path, "trait_a", [f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n"]
    )
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale"
        "\toriginal_sd_method\toriginal_sd\tsource_assembly\n"
        f"trait_a\t{vcf}\tTrait A\t1000\tsd\tdeclared_standardised\t\thg17\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source_assembly"):
        _read_manifest(manifest)


def test_canonical_analyses_tsv_columns_build_cleanly(tmp_path):
    """Issue #170: the registry's canonical ``analyses.tsv`` column names are
    accepted directly -- no ``analysis_id``->``trait_id`` rename needed."""
    from opengwasdb.layouts.dense.build_vcf import _read_manifest

    vcf = _make_vcf(
        tmp_path, "trait_a", [f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n"]
    )
    manifest = tmp_path / "canonical_manifest.tsv"
    manifest.write_text(
        "analysis_id\tsource_file\tanalysis_label\tsample_size"
        "\tstored_effect_scale\toriginal_sd_method\n"
        f"trait_a\t{vcf}\tTrait A\t1234\tsd\tdeclared_standardised\n",
        encoding="utf-8",
    )

    # The reader maps every canonical name onto the builder's own row fields.
    rows = _read_manifest(manifest)
    assert len(rows) == 1
    assert rows[0].trait_id == "trait_a"
    assert rows[0].file_path == str(vcf)
    assert rows[0].trait_name == "Trait A"
    assert rows[0].n == 1234

    # ...and the whole build runs to completion from that manifest.
    store_path = tmp_path / "canonical-store.opengwasdb"
    result = build_dense_from_vcf_manifest(
        manifest, store_path, store_id="canonical", release_id="v1"
    )
    assert result.n_analyses == 1
    analyses = read_analyses(store_path / "analyses.tsv")
    assert analyses.rows[0]["analysis_id"] == "trait_a"
    assert analyses.rows[0]["analysis_label"] == "Trait A"
    assert analyses.rows[0]["sample_size"] == "1234"


def test_blank_analysis_label_is_preserved_in_dense_build(tmp_path):
    """ADR 0034: a manifest that includes analysis_label column with a blank value
    preserves the blank in analyses.tsv rather than falling back to analysis_id."""
    vcf = _make_vcf(
        tmp_path, "trait_blank", [f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n"]
    )
    manifest = tmp_path / "blank_label_manifest.tsv"
    manifest.write_text(
        "analysis_id\tsource_file\tanalysis_label\tsample_size"
        "\tstored_effect_scale\toriginal_sd_method\n"
        f"trait_blank\t{vcf}\t\t1234\tsd\tdeclared_standardised\n",
        encoding="utf-8",
    )

    store_path = tmp_path / "blank-label-store.opengwasdb"
    result = build_dense_from_vcf_manifest(
        manifest, store_path, store_id="blank_label", release_id="v1"
    )
    assert result.n_analyses == 1
    analyses = read_analyses(store_path / "analyses.tsv")
    assert analyses.rows[0]["analysis_id"] == "trait_blank"
    assert analyses.rows[0]["analysis_label"] == ""


def _manifest_with_source_assembly(
    tmp_path: Path, entries: list[tuple[str, Path, str, str]]
) -> Path:
    """Like `_make_manifest`, plus a `source_assembly` column per entry."""
    manifest = tmp_path / "manifest.tsv"
    lines = [
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale"
        "\toriginal_sd_method\toriginal_sd\tsource_assembly"
    ]
    for trait_id, file_path, trait_name, source_assembly in entries:
        lines.append(
            f"{trait_id}\t{file_path}\t{trait_name}\t1000\tsd\tdeclared_standardised\t"
            f"\t{source_assembly}"
        )
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def test_hg38_source_assembly_is_not_lifted(tmp_path):
    """issue #85: a row declaring source_assembly=hg38 passes through with no
    liftover -- HG19_POS_2 (1,000,000) is a position the real hg19->hg38
    chain shifts to 1,064,620 (see this file's module docstring); if it were
    lifted a second time despite the hg38 declaration, the stored position
    would be 1,064,620, not the source file's own 1,000,000.
    """
    vcf = _make_vcf(
        tmp_path, "trait_ssf",
        [f"1\t{HG19_POS_2}\t.\tC\tT\t.\tPASS\t.\tES:SE\t1.5:0.3\n"],  # z=5.0, flip->-5.0
    )
    manifest = _manifest_with_source_assembly(tmp_path, [("trait_ssf", vcf, "Trait SSF", "hg38")])
    store_path = tmp_path / "store.opengwasdb"

    build_dense_from_vcf_manifest(manifest, store_path, store_id="s", release_id="r")

    query = query_store(store_path)
    result = query.analysis("trait_ssf")
    vt = query.variants_table()
    query.close()

    assert len(result["z"]) == 1
    variant = vt[int(result["variant_index"][0])]
    assert variant["position"] == HG19_POS_2
    assert result["z"][0] == pytest.approx(-5.0, rel=5e-3)


def test_mixed_hg19_and_hg38_manifest_builds_correctly(tmp_path):
    """issue #85: one manifest mixing a default (hg19) row and an
    explicitly-hg38 row lifts only the hg19 row -- the scenario the bug
    report specifically named (a GWAS-VCF row alongside a harmonised
    GWAS-SSF row in one build). trait_vcf's position 1,500,000 (HG19_POS_3)
    genuinely shifts under the real chain (-> 1,564,620), proving liftover
    ran for it; trait_ssf's position 1,000,000 (HG19_POS_2) would shift too
    if lifted, so it staying put proves the hg38 declaration skipped it.
    """
    vcf_hg19 = _make_vcf(
        tmp_path, "trait_vcf",
        [f"1\t{HG19_POS_3}\t.\tG\tA\t.\tPASS\t.\tES:SE\t0.6:0.2\n"],  # lifted -> 1564620
    )
    vcf_hg38 = _make_vcf(
        tmp_path, "trait_ssf",
        [f"1\t{HG19_POS_2}\t.\tC\tT\t.\tPASS\t.\tES:SE\t1.5:0.3\n"],  # not lifted; stays 1000000
    )
    manifest = _manifest_with_source_assembly(
        tmp_path,
        [("trait_vcf", vcf_hg19, "Trait VCF", ""), ("trait_ssf", vcf_hg38, "Trait SSF", "hg38")],
    )
    store_path = tmp_path / "store.opengwasdb"

    build_dense_from_vcf_manifest(manifest, store_path, store_id="s", release_id="r")

    query = query_store(store_path)
    vt = query.variants_table()
    vcf_result = query.analysis("trait_vcf")
    ssf_result = query.analysis("trait_ssf")
    query.close()

    assert vt[int(vcf_result["variant_index"][0])]["position"] == 1_564_620
    assert vt[int(ssf_result["variant_index"][0])]["position"] == HG19_POS_2


def test_cross_assembly_tuple_collision_is_dropped_not_misattributed(tmp_path, caplog):
    """issue #85 code review follow-up: an hg38-declared row and an
    hg19-declared row sharing an identical raw (chrom, pos, ref, alt) string
    are two different physical loci on two different builds -- the hg38
    string is a literal coordinate, the hg19 string is a *pre-lift*
    coordinate bound for a different hg38 position. Binding both to one
    stored row would silently misattribute one row's association to the
    other's variant, so the shared tuple must be dropped from both rather
    than guessed.
    """
    import logging

    colliding_row = f"1\t{HG19_POS_2}\t.\tC\tT\t.\tPASS\t.\tES:SE\t1.5:0.3\n"
    # A third, non-colliding row keeps the store non-empty -- an all-variants-
    # dropped build hits an unrelated pre-existing limitation elsewhere in the
    # zarr band-write path (a zero-width dense matrix), out of scope here.
    clean_row = f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n"
    vcf_hg19 = _make_vcf(tmp_path, "trait_vcf", [colliding_row, clean_row])
    vcf_hg38 = _make_vcf(tmp_path, "trait_ssf", [colliding_row])
    manifest = _manifest_with_source_assembly(
        tmp_path,
        [("trait_vcf", vcf_hg19, "Trait VCF", ""), ("trait_ssf", vcf_hg38, "Trait SSF", "hg38")],
    )
    store_path = tmp_path / "store.opengwasdb"

    with caplog.at_level(logging.WARNING):
        build_dense_from_vcf_manifest(manifest, store_path, store_id="s", release_id="r")

    assert "raw variant tuple" in caplog.text

    query = query_store(store_path)
    vcf_result = query.analysis("trait_vcf")
    ssf_result = query.analysis("trait_ssf")
    query.close()

    # The colliding variant is absent from both traits; the clean one survives.
    assert len(vcf_result["z"]) == 1
    assert vcf_result["z"][0] == pytest.approx(-4.0, rel=5e-3)
    assert len(ssf_result["z"]) == 0


def test_match_batch_handles_an_empty_lookup_without_crashing():
    """issue #85 code review follow-up: a manifest whose entire variant set is
    dropped (e.g. every variant is an ambiguous cross-assembly collision)
    leaves `keys_sorted` empty; `_match_batch` must report no matches rather
    than crash indexing an empty array (`np.searchsorted` on an empty array
    always returns 0, so `keys_sorted[len(keys_sorted) - 1]` -> `keys_sorted[-1]`
    on a zero-length array raised IndexError before this guard)."""
    from opengwasdb.layouts.dense.build_vcf import _match_batch

    rows, z, se, eaf = _match_batch(
        ["1"], [100], ["A"], ["G"], [1.0], [0.5], [float("nan")],
        keys_sorted=np.empty(0, dtype="S1"), rows_sorted=np.empty(0, dtype=np.int32),
    )

    assert len(rows) == 0
    assert len(z) == 0
    assert len(se) == 0
    assert len(eaf) == 0


def test_liftover_failure_threshold_scoped_to_hg19_group_not_diluted_by_hg38_rows(tmp_path):
    """issue #85: liftover_failure_threshold is computed over the hg19 group's
    own denominator (`_lift_manifest_variants` calls `build_liftover_lookup`
    with only that group's tuples), not the whole manifest -- otherwise a
    large hg38-sourced (e.g. GWAS-SSF) manifest could mask a genuinely broken
    hg19 source. 2/2 hg19 variants fail liftover (100%, over threshold) here,
    but 2/302 against the *whole* manifest (300 hg38 passthrough rows added)
    would be under the 1% threshold -- so this only raises if the two groups
    are scored separately, not summed.
    """
    from opengwasdb.build.liftover import LiftoverFailureError

    bad_vcf = _make_vcf(
        tmp_path, "bad_trait",
        [
            "1\t200000\t.\tA\tG\t.\tPASS\t.\tES:SE\t1.0:0.5\n",
            "1\t300000\t.\tC\tT\t.\tPASS\t.\tES:SE\t0.5:0.2\n",
        ],
    )
    good_hg38_vcf = _make_vcf(
        tmp_path, "good_trait",
        [
            f"1\t{5_000_000 + i}\t.\tC\tT\t.\tPASS\t.\tES:SE\t1.0:0.5\n"
            for i in range(300)
        ],
    )
    manifest = _manifest_with_source_assembly(
        tmp_path,
        [
            ("bad_trait", bad_vcf, "Bad Trait", ""),
            ("good_trait", good_hg38_vcf, "Good Trait", "hg38"),
        ],
    )

    with pytest.raises(LiftoverFailureError):
        build_dense_from_vcf_manifest(
            manifest, tmp_path / "store.opengwasdb",
            store_id="s", release_id="r", liftover_failure_threshold=0.01,
        )


class TestParallel:
    def test_two_workers_matches_serial(self, tmp_path):
        vcf1 = _make_vcf(
            tmp_path,
            "trait_a",
            [
                f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n",
                f"1\t{HG19_POS_2}\t.\tC\tT\t.\tPASS\t.\tES:SE\t1.5:0.3\n",
                f"1\t{HG19_POS_3}\t.\tG\tA\t.\tPASS\t.\tES:SE\t0.6:0.2\n",
            ],
        )
        vcf2 = _make_vcf(
            tmp_path,
            "trait_b",
            [
                f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t6.0:0.5\n",
                f"1\t{HG19_POS_3}\t.\tG\tA\t.\tPASS\t.\tES:SE\t1.2:0.3\n",
            ],
            study_type="CaseControl",
        )
        manifest = _make_manifest(
            tmp_path,
            [("trait_a", vcf1, "Trait A"), ("trait_b", vcf2, "Trait B")],
        )

        serial_path = tmp_path / "serial.opengwasdb"
        build_dense_from_vcf_manifest(
            manifest, serial_path, store_id="s", release_id="r", n_workers=1
        )
        parallel_path = tmp_path / "parallel.opengwasdb"
        build_dense_from_vcf_manifest(
            manifest, parallel_path, store_id="s", release_id="r", n_workers=2
        )

        assert validate_store(parallel_path).ok

        serial_root = open_store(serial_path).arrays(mode="r")
        parallel_root = open_store(parallel_path).arrays(mode="r")
        serial_z = serial_root["z"][:]
        parallel_z = parallel_root["z"][:]
        serial_se = serial_root["se"][:]
        parallel_se = parallel_root["se"][:]

        assert serial_z.shape == parallel_z.shape
        np.testing.assert_array_equal(np.isnan(serial_z), np.isnan(parallel_z))
        np.testing.assert_allclose(
            serial_z[~np.isnan(serial_z)], parallel_z[~np.isnan(parallel_z)]
        )
        np.testing.assert_allclose(
            serial_se[~np.isnan(serial_se)], parallel_se[~np.isnan(parallel_se)]
        )

        # Inline-harvested top hits must match between serial and parallel.
        for key in ("p_5e_04", "p_5e_06"):
            s = serial_root[f"top_hits/{key}"]
            p = parallel_root[f"top_hits/{key}"]
            np.testing.assert_array_equal(s["variant_index"][:], p["variant_index"][:])
            np.testing.assert_array_equal(s["analysis_index"][:], p["analysis_index"][:])
            np.testing.assert_array_equal(s["z"][:], p["z"][:])


def _assert_dense_stores_match(serial_path: Path, parallel_path: Path) -> None:
    from opengwasdb.variants.axis import iter_variant_records

    assert validate_store(parallel_path).ok
    serial_records = {
        r.alid: r for r in iter_variant_records(serial_path / "variants.tsv.gz")
    }
    parallel_records = {
        r.alid: r for r in iter_variant_records(parallel_path / "variants.tsv.gz")
    }
    assert serial_records.keys() == parallel_records.keys()
    for alid in serial_records:
        assert dict(serial_records[alid]) == dict(parallel_records[alid]), alid

    serial_root = open_store(serial_path).arrays(mode="r")
    parallel_root = open_store(parallel_path).arrays(mode="r")
    for name in ("z", "se"):
        s = serial_root[name][:]
        p = parallel_root[name][:]
        assert s.shape == p.shape
        np.testing.assert_array_equal(np.isnan(s), np.isnan(p))
        np.testing.assert_allclose(s[~np.isnan(s)], p[~np.isnan(p)])


class TestPass1Parallel:
    """Stage 1 (variant-union) parallelism must be bit-for-bit deterministic.

    The union and rsid maps are compared directly (before liftover) and again
    through the built store's variant axis, rsid column, and statistic arrays
    (after liftover and ALID generation).
    """

    def test_pass1_parallel_union_and_rsids_match_serial(self, tmp_path):
        from opengwasdb.layouts.dense.build_vcf import (
            _collect_manifest_variant_sites,
            _read_manifest,
        )

        vcf1 = _make_vcf(
            tmp_path,
            "trait_a",
            [
                f"1\t{HG19_POS_1}\trsZ\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n",
                f"1\t{HG19_POS_2}\t.\tC\tT\t.\tPASS\t.\tES:SE\t1.5:0.3\n",  # no rsid
                f"1\t{HG19_POS_3}\trs300\tG\tA\t.\tPASS\t.\tES:SE\t0.6:0.2\n",
            ],
        )
        vcf2 = _make_vcf(
            tmp_path,
            "trait_b",
            [
                f"1\t{HG19_POS_1}\trsA\tA\tG\t.\tPASS\t.\tES:SE\t6.0:0.5\n",  # must lose
                f"1\t{HG19_POS_2}\trs200\tC\tT\t.\tPASS\t.\tES:SE\t1.2:0.3\n",  # fills blank
                "1\t2000000\trs400\tG\tA\t.\tPASS\t.\tES:SE\t1.0:0.2\n",
            ],
        )
        manifest = _make_manifest(
            tmp_path, [("trait_a", vcf1, "Trait A"), ("trait_b", vcf2, "Trait B")]
        )
        rows = _read_manifest(manifest)

        serial = _collect_manifest_variant_sites(rows, n_workers=1)
        parallel = _collect_manifest_variant_sites(rows, n_workers=2)

        assert serial == parallel
        tuples_by_assembly, rsid_by_site = parallel
        assert tuples_by_assembly == {
            "hg19": {
                ("1", HG19_POS_1, "A", "G"),
                ("1", HG19_POS_2, "C", "T"),
                ("1", HG19_POS_3, "G", "A"),
                ("1", 2_000_000, "G", "A"),
            }
        }
        # First named rsid wins across shards -- and by manifest order, not
        # rsid-string order ("rsZ" sorts after "rsA" but came first).
        assert rsid_by_site[("1", HG19_POS_1, "A", "G")] == "rsZ"
        assert rsid_by_site[("1", HG19_POS_2, "C", "T")] == "rs200"
        assert rsid_by_site[("1", HG19_POS_3, "G", "A")] == "rs300"

    def test_parallel_stage1_store_matches_serial_variant_axis_and_arrays(self, tmp_path):
        """Parallel Pass 1 produces the same union, ALIDs, rsids, and dense
        arrays as serial Pass 1 -- including a mixed hg19/hg38 manifest so
        the liftover and passthrough groups are both exercised."""
        from opengwasdb.variants.axis import iter_variant_records

        vcf_a = _make_vcf(
            tmp_path,
            "trait_a",
            [
                f"1\t{HG19_POS_1}\trs1\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n",
                f"1\t{HG19_POS_2}\trs2\tC\tT\t.\tPASS\t.\tES:SE\t1.5:0.3\n",
                f"1\t{HG19_POS_3}\t.\tG\tA\t.\tPASS\t.\tES:SE\t0.6:0.2\n",
            ],
        )
        vcf_b = _make_vcf(
            tmp_path,
            "trait_b",
            [
                f"1\t{HG19_POS_1}\trsX\tA\tG\t.\tPASS\t.\tES:SE\t6.0:0.5\n",  # must lose to rs1
                f"1\t{HG19_POS_3}\trs3\tG\tA\t.\tPASS\t.\tES:SE\t1.2:0.3\n",
            ],
        )
        vcf_ssf = _make_vcf(
            tmp_path,
            "trait_ssf",
            ["1\t5000000\trs5\tC\tT\t.\tPASS\t.\tES:SE\t0.8:0.4\n"],
        )
        manifest = _manifest_with_source_assembly(
            tmp_path,
            [
                ("trait_a", vcf_a, "Trait A", ""),
                ("trait_b", vcf_b, "Trait B", ""),
                ("trait_ssf", vcf_ssf, "Trait SSF", "hg38"),
            ],
        )

        serial_path = tmp_path / "serial.opengwasdb"
        parallel_path = tmp_path / "parallel.opengwasdb"
        build_dense_from_vcf_manifest(
            manifest, serial_path, store_id="s", release_id="r", n_workers=1
        )
        build_dense_from_vcf_manifest(
            manifest, parallel_path, store_id="s", release_id="r", n_workers=3
        )

        _assert_dense_stores_match(serial_path, parallel_path)

        parallel_records = {
            r.alid: r for r in iter_variant_records(parallel_path / "variants.tsv.gz")
        }
        # The rsid column carries the first-named rsid from the manifest order,
        # including a blank on the first occurrence filled by a later file.
        assert parallel_records[HG38_ALID_1].rsid == "rs1"
        assert parallel_records[HG38_ALID_2].rsid == "rs2"
        assert parallel_records[HG38_ALID_3].rsid == "rs3"
        assert parallel_records["1:5000000:C:T"].rsid == "rs5"

        serial_root = open_store(serial_path).arrays(mode="r")
        parallel_root = open_store(parallel_path).arrays(mode="r")
        for key in ("p_5e_04", "p_5e_06", "p_5e_08"):
            s = serial_root[f"top_hits/{key}"]
            p = parallel_root[f"top_hits/{key}"]
            np.testing.assert_array_equal(s["variant_index"][:], p["variant_index"][:])
            np.testing.assert_array_equal(s["analysis_index"][:], p["analysis_index"][:])
            np.testing.assert_array_equal(s["z"][:], p["z"][:])

    def test_pass1_parallel_supports_generic_readers_mixed_manifest(self, tmp_path):
        """The parallel stage 1 variant-union routine is generic across file types:
        a mixed manifest containing GWAS-VCF and GWAS-SSF files produces the exact
        same variant union, rsid mapping, and dense store arrays in parallel as serial."""
        vcf_path = _make_vcf(
            tmp_path,
            "trait_vcf",
            [
                f"1\t{HG19_POS_1}\trs_vcf1\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n",
                f"1\t{HG19_POS_2}\t.\tC\tT\t.\tPASS\t.\tES:SE\t1.5:0.3\n",
            ],
        )
        ssf_path = tmp_path / "trait_ssf.tsv.gz"
        _write_ssf(
            ssf_path,
            [
                {
                    "chromosome": "1",
                    "base_pair_location": HG19_POS_2,
                    "effect_allele": "T",
                    "other_allele": "C",
                    "beta": 1.5,
                    "standard_error": 0.3,
                },
                {
                    "chromosome": "1",
                    "base_pair_location": HG19_POS_3,
                    "effect_allele": "A",
                    "other_allele": "G",
                    "beta": 0.6,
                    "standard_error": 0.2,
                },
            ],
        )

        manifest = tmp_path / "mixed_manifest.tsv"
        header = (
            "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale\t"
            "original_sd_method\toriginal_sd\tsource_reader_capability\tsource_assembly\n"
        )
        row_vcf = (
            f"trait_vcf\t{vcf_path}\tTrait VCF\t1000\tsd\tdeclared_standardised\t\t"
            "opengwasdb.gwas-vcf\thg19\n"
        )
        row_ssf = (
            f"trait_ssf\t{ssf_path}\tTrait SSF\t1000\tsd\tdeclared_standardised\t\t"
            f"{GWAS_SSF_CAPABILITY}\thg19\n"
        )
        manifest.write_text(header + row_vcf + row_ssf, encoding="utf-8")

        serial_path = tmp_path / "serial_mixed.opengwasdb"
        parallel_path = tmp_path / "parallel_mixed.opengwasdb"
        build_dense_from_vcf_manifest(
            manifest, serial_path, store_id="s", release_id="r", n_workers=1
        )
        build_dense_from_vcf_manifest(
            manifest, parallel_path, store_id="s", release_id="r", n_workers=2
        )

        _assert_dense_stores_match(serial_path, parallel_path)


class TestTopHitHarvest:
    def test_store_without_frequency_builds_compatible_index(self, two_trait_store):
        root = open_store(two_trait_store).arrays(mode="r")
        for key in root["top_hits"]:
            assert "eaf" not in root[f"top_hits/{key}"]
        with query_store(two_trait_store) as query:
            result = query.top_hits(threshold=5e-4)
        assert len(result["eaf"]) > 0
        assert np.all(np.isnan(result["eaf"]))

    def test_harvest_matches_full_scan(self, tmp_path):
        """Top hits harvested during Pass 2 must equal a full-matrix rescan."""
        from opengwasdb.layouts.dense.top_hits import (
            build_top_hit_indexes,
            threshold_key,
        )

        # trait z-scores: 4.0 and 5.0 clear the loosest tier; 3.0 does not.
        vcf1 = _make_vcf(
            tmp_path,
            "trait_a",
            [
                f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n",   # z=4.0
                f"1\t{HG19_POS_2}\t.\tC\tT\t.\tPASS\t.\tES:SE\t1.5:0.3\n",   # z=5.0
                f"1\t{HG19_POS_3}\t.\tG\tA\t.\tPASS\t.\tES:SE\t0.6:0.2\n",   # z=3.0
            ],
        )
        vcf2 = _make_vcf(
            tmp_path,
            "trait_b",
            [f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t6.0:0.5\n"],     # z=12.0
        )
        manifest = _make_manifest(
            tmp_path, [("trait_a", vcf1, "Trait A"), ("trait_b", vcf2, "Trait B")]
        )
        store = tmp_path / "store.opengwasdb"
        build_dense_from_vcf_manifest(manifest, store, store_id="s", release_id="r", n_workers=2)

        root = open_store(store).arrays(mode="r")
        harvested = {
            t: root[f"top_hits/{threshold_key(t)}"]["z"][:]
            for t in (5e-4, 5e-6, 5e-8)
        }

        # Rebuild the same index by full-matrix scan and compare.
        build_top_hit_indexes(store)
        for t in (5e-4, 5e-6, 5e-8):
            rescanned = root[f"top_hits/{threshold_key(t)}"]["z"][:]
            np.testing.assert_array_equal(harvested[t], rescanned)

        # Sanity: the loosest tier caught the three |z|>=3.4808 cells
        # (z = -12, -5, -4); the z=3.0 cell is below the 3.4808 cutoff.
        assert sorted(harvested[5e-4].tolist()) == [-12.0, -5.0, -4.0]

    def test_index_z_equals_stored_matrix(self, tmp_path):
        """Issue 046: the top-hit index z must equal the stored matrix value
        exactly -- as decoded through the store's own encoding (ADR 0037) --
        so the index agrees with what a query reads from `z`, and the store
        validates cleanly."""
        from opengwasdb.layouts.dense.top_hits import threshold_key

        vcf = _make_vcf(
            tmp_path,
            "trait_a",
            [
                f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n",  # z=4.0
                f"1\t{HG19_POS_2}\t.\tC\tT\t.\tPASS\t.\tES:SE\t1.5:0.3\n",  # z=5.0
                f"1\t{HG19_POS_3}\t.\tG\tA\t.\tPASS\t.\tES:SE\t0.6:0.2\n",  # z=3.0
            ],
        )
        manifest = _make_manifest(tmp_path, [("trait_a", vcf, "Trait A")])
        store = tmp_path / "store.opengwasdb"
        build_dense_from_vcf_manifest(manifest, store, store_id="s", release_id="r", n_workers=2)

        assert validate_store(store).ok

        from opengwasdb.encoding import DenseZPlane

        opened = open_store(store)
        root = opened.arrays(mode="r")
        z_plane = DenseZPlane.open(root, opened.manifest.encoding)
        for t in (5e-4, 5e-6, 5e-8):
            g = root[f"top_hits/{threshold_key(t)}"]
            assert "imputed" not in g
            rows = g["variant_index"][:]
            cols = g["analysis_index"][:]
            index_z = g["z"][:]
            gathered = z_plane.points(rows, cols)
            np.testing.assert_array_equal(index_z, gathered)


class TestBandStreaming:
    def _three_trait_manifest(self, tmp_path):
        rows = [
            f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n",
            f"1\t{HG19_POS_2}\t.\tC\tT\t.\tPASS\t.\tES:SE\t1.5:0.3\n",
            f"1\t{HG19_POS_3}\t.\tG\tA\t.\tPASS\t.\tES:SE\t0.6:0.2\n",
        ]
        entries = []
        for k in range(3):
            vcf = _make_vcf(tmp_path, f"trait_{k}", rows)
            entries.append((f"trait_{k}", vcf, f"Trait {k}"))
        return _make_manifest(tmp_path, entries)

    def test_short_final_band_matches_single_band(self, tmp_path):
        """A 2-wide analysis chunk over 3 analyses (bands [0:2],[2:3]) must equal
        a single-band build byte-for-byte — exercises the short final band."""
        manifest = self._three_trait_manifest(tmp_path)

        single = tmp_path / "single.opengwasdb"
        build_dense_from_vcf_manifest(
            manifest, single, store_id="s", release_id="r", n_workers=2,
            chunk_shape=(1000, 1000),
        )
        banded = tmp_path / "banded.opengwasdb"
        build_dense_from_vcf_manifest(
            manifest, banded, store_id="s", release_id="r", n_workers=2,
            chunk_shape=(1000, 2),
        )

        assert validate_store(banded).ok
        rs = open_store(single).arrays(mode="r")
        rb = open_store(banded).arrays(mode="r")
        for name in ("z", "se"):
            a, b = rs[name][:], rb[name][:]
            np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
            np.testing.assert_array_equal(a[~np.isnan(a)], b[~np.isnan(b)])
        # banded store really used a 2-wide analysis chunk
        assert rb["z"].chunks[1] == 2


def _eaf_vcf(tmp_path: Path, name: str, spec: list[tuple[int, float, float, float, float]]) -> Path:
    """A GWAS-VCF with frequencies, from ``(pos, z, es, se, af)`` rows."""
    return write_gwas_vcf_with_eaf(
        tmp_path / f"{name}.vcf",
        [
            f"1\t{pos}\t.\tA\tG\t.\tPASS\t.\tES:SE:EZ:AF\t{es}:{se}:{z}:{af}\n"
            for pos, z, es, se, af in spec
        ],
    )


class TestParallelBandWrite:
    """Issue #220: the z/se/eaf band passes load a band's columns across
    ``--n-workers`` while every order-dependent output stays the serial path's.

    The fixture deliberately spans two bands (a 2-wide analysis chunk over four
    Analyses of different sizes, so columns finish out of order), carries a z
    overflow cell in each band, has top hits in both bands, and holds
    frequencies. A reduction that combined results in completion order, or
    against the wrong Analysis, cannot pass this by luck.
    """

    def _manifest(self, tmp_path: Path) -> Path:
        # 137 and -120 are outside the int16 fixed-point plane (|z| <= ~32) and
        # land in the overflow table; 4.0 and 5.0 clear the loosest top-hit
        # tier's |z| >= 3.4808. Column sizes differ (1, 2, 3, 2).
        entries = [
            (
                "trait_a",
                _eaf_vcf(tmp_path, "trait_a", [(HG19_POS_1, 137.0, 13.7, 0.1, 0.2)]),
                "Trait A",
            ),
            (
                "trait_b",
                _eaf_vcf(
                    tmp_path,
                    "trait_b",
                    [(HG19_POS_2, 4.0, 2.0, 0.5, 0.3), (HG19_POS_3, 1.0, 1.0, 1.0, 0.4)],
                ),
                "Trait B",
            ),
            (
                "trait_c",
                _eaf_vcf(
                    tmp_path,
                    "trait_c",
                    [
                        (HG19_POS_1, -120.0, -12.0, 0.1, 0.25),
                        (HG19_POS_2, 5.0, 2.5, 0.5, 0.35),
                        (HG19_POS_3, 0.5, 0.5, 1.0, 0.45),
                    ],
                ),
                "Trait C",
            ),
            (
                "trait_d",
                _eaf_vcf(
                    tmp_path,
                    "trait_d",
                    [(HG19_POS_1, 0.2, 0.2, 1.0, 0.5), (HG19_POS_3, 3.6, 1.8, 0.5, 0.55)],
                ),
                "Trait D",
            ),
        ]
        return _make_manifest(tmp_path, entries)

    def _build(self, manifest: Path, out: Path, n_workers: int) -> Path:
        build_dense_from_vcf_manifest(
            manifest,
            out,
            store_id="s",
            release_id="r",
            n_workers=n_workers,
            chunk_shape=(1000, 2),
            allow_unverified_eaf=True,
        )
        return out

    def test_parallel_band_write_matches_serial(self, tmp_path):
        manifest = self._manifest(tmp_path)
        serial = self._build(manifest, tmp_path / "serial.opengwasdb", 1)
        parallel = self._build(manifest, tmp_path / "parallel.opengwasdb", 3)

        assert validate_store(parallel).ok
        rs = open_store(serial).arrays(mode="r")
        rp = open_store(parallel).arrays(mode="r")

        # The fixture is only meaningful if it exercises the paths under test.
        assert rs["z_overflow_index"][:].size >= 2, "fixture carries no z overflow cells"
        assert rs["eaf"][:].shape == rp["eaf"][:].shape
        assert np.isfinite(rs["eaf"][:]).any(), "fixture carries no frequencies"
        bands = {int(c) // 2 for c in rs[f"top_hits/{threshold_key(5e-4)}"]["analysis_index"][:]}
        assert bands == {0, 1}, f"fixture's top hits do not span both bands: {bands}"

        assert_same_band_arrays(rs, rp)
        assert_same_top_hits(
            rs, rp, ("variant_index", "analysis_index", "z", "se", "eaf")
        )

        serial_scopes = {
            row["analysis_id"]: row["eaf_scope"]
            for row in read_analyses(serial / "analyses.tsv").rows
        }
        parallel_scopes = {
            row["analysis_id"]: row["eaf_scope"]
            for row in read_analyses(parallel / "analyses.tsv").rows
        }
        assert serial_scopes == parallel_scopes
        assert set(serial_scopes.values()) == {"association"}, (
            "fixture must have every Analysis carrying a frequency for "
            "column_has_eaf to mean anything"
        )

    def test_a_pool_that_yields_out_of_order_fails_loudly(self, tmp_path, monkeypatch):
        """If a worker pool ever returned a column's result against another
        Analysis, the build must stop rather than write it into the wrong
        band slot (issue #220). The real ``ordered_map`` preserves input
        order; this replaces it with a deliberately out-of-order reduction."""
        import opengwasdb.layouts.dense.build_vcf as build_vcf

        def reversed_map(fn, items, n_workers, max_in_flight=None):
            results = [fn(item) for item in items]
            yield from reversed(results)

        monkeypatch.setattr(build_vcf, "ordered_map", reversed_map)
        with pytest.raises(ValueError, match="arrived out of order"):
            self._build(self._manifest(tmp_path), tmp_path / "store.opengwasdb", 2)


class TestForkSafeLookup:
    def test_last_wins_dedup_on_collision(self, tmp_path):
        """Two source variants mapping to the same row (as a liftover collision
        would) resolve to one row, keeping the last stream occurrence."""
        from opengwasdb.layouts.dense.build_vcf import (
            _build_variant_key_index,
            _resolve_column,
        )

        # Both hg19 keys map to the same hg38 ALID → same row 0.
        hg19_lookup = {("1", 100, "A", "G"): "1:100:A:G", ("1", 200, "A", "G"): "1:100:A:G"}
        keys, rows = _build_variant_key_index(hg19_lookup, {"1:100:A:G": 0})

        vcf = _make_vcf(
            tmp_path,
            "t",
            [
                "1\t100\t.\tA\tG\t.\tPASS\t.\tES:SE\t1.0:0.5\n",  # z=2.0, flip → -2.0
                "1\t200\t.\tA\tG\t.\tPASS\t.\tES:SE\t3.0:0.5\n",  # z=6.0, flip → -6.0 (later)
            ],
        )
        r, z, _se, _eaf = _resolve_column(str(vcf), keys, rows)
        assert r.tolist() == [0]
        assert z[0] == pytest.approx(-6.0, rel=5e-3)  # last occurrence wins

    def test_absent_variant_not_mismapped(self, tmp_path):
        """A variant not in the panel must be dropped, not snapped to a neighbour."""
        from opengwasdb.layouts.dense.build_vcf import (
            _build_variant_key_index,
            _resolve_column,
        )

        hg19_lookup = {("1", 100, "A", "G"): "1:100:A:G", ("1", 300, "A", "G"): "1:300:A:G"}
        keys, rows = _build_variant_key_index(
            hg19_lookup, {"1:100:A:G": 0, "1:300:A:G": 1}
        )
        vcf = _make_vcf(
            tmp_path,
            "t",
            [
                "1\t100\t.\tA\tG\t.\tPASS\t.\tES:SE\t1.0:0.5\n",  # in panel → row 0
                "1\t200\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n",  # ABSENT → dropped
            ],
        )
        r, _z, _se, _eaf = _resolve_column(str(vcf), keys, rows)
        assert r.tolist() == [0]  # only the in-panel variant, no mis-map to row 1

    def test_batched_matches_whole_file(self, tmp_path, monkeypatch):
        """A tiny batch size (many batches + a cross-batch collision) resolves to
        the same column as processing the whole file at once."""
        import opengwasdb.layouts.dense.build_vcf as bv

        # pos 100 and 200 collide onto row 0; pos 300 -> row 1.
        hg19_lookup = {
            ("1", 100, "A", "G"): "1:100:A:G",
            ("1", 200, "A", "G"): "1:100:A:G",
            ("1", 300, "C", "T"): "1:300:C:T",
        }
        keys, rows = bv._build_variant_key_index(
            hg19_lookup, {"1:100:A:G": 0, "1:300:C:T": 1}
        )
        vcf = _make_vcf(
            tmp_path,
            "t",
            [
                "1\t100\t.\tA\tG\t.\tPASS\t.\tES:SE\t1.0:0.5\n",  # row 0
                "1\t300\t.\tC\tT\t.\tPASS\t.\tES:SE\t2.0:0.5\n",  # row 1
                "1\t200\t.\tA\tG\t.\tPASS\t.\tES:SE\t3.0:0.5\n",  # row 0 again, last wins
            ],
        )
        whole = bv._resolve_column(str(vcf), keys, rows)
        monkeypatch.setattr(bv, "_RESOLVE_BATCH", 1)  # one association per batch
        batched = bv._resolve_column(str(vcf), keys, rows)

        for a, b in zip(whole, batched, strict=True):
            np.testing.assert_array_equal(a, b)
        r, z, _se, _eaf = batched
        assert sorted(r.tolist()) == [0, 1]
        # row 0 kept the later (pos 200) occurrence: z=6.0 flipped to -6.0
        assert z[r.tolist().index(0)] == pytest.approx(-6.0, rel=5e-3)


def test_sorted_alids_matches_reference_with_long_keys():
    """`_sorted_alids` preserves `_alid_sort_key` order even when a rare
    long indel would have padded a fixed-width numpy string array to 200+ bytes."""
    from opengwasdb.layouts.dense.build_vcf import _alid_sort_key, _sorted_alids

    long_a = "A" * 200
    long_c = "C" * 150
    alids = {
        "1:10:A:G",
        "1:2:C:T",
        "2:1:A:G",
        "10:9:G:C",
        "X:100:A:C",
        "Y:1:G:T",
        "M:5:A:C",
        "MT:5:A:C",
        f"1:10:A:{long_a}",
        f"1:10:A:{long_c}",
        f"1:10:C:{long_a}",
    }

    assert _sorted_alids(alids) == sorted(alids, key=_alid_sort_key)


def test_axis_metadata_long_keys_match_reference_sort():
    """`_axis_metadata` sorts the variant axis exactly like the scalar
    `sorted(..., key=_alid_sort_key)` reference, including >100-byte alleles."""
    from opengwasdb.layouts.dense.build_vcf import (
        _alid_sort_key,
        _axis_metadata,
        _ManifestRow,
    )

    long_a = "A" * 200
    source_lookup = {
        ("1", 10, "A", "G"): "1:10:A:G",
        ("2", 5, "C", "T"): "2:5:C:T",
        ("1", 10, "A", long_a): f"1:10:A:{long_a}",
        ("X", 5, "A", "C"): "X:5:A:C",
        ("10", 2, "G", "C"): "10:2:C:G",
    }
    row = _ManifestRow(
        trait_id="t",
        file_path="unused",
        trait_name="Trait",
        n=1000,
        stored_effect_scale="sd",
        se_divisor=1.0,
        source_reader_capability="opengwasdb.gwas-vcf",
        source_assembly="hg38",
        original_sd="",
        assigned_ancestry="",
    )

    axis, variant_index = _axis_metadata(source_lookup, [row])
    expected = sorted(set(source_lookup.values()), key=_alid_sort_key)

    assert axis.alids == expected
    assert variant_index == {alid: i for i, alid in enumerate(expected)}


def test_build_variant_key_index_long_keys_sorted_and_searchable():
    """The Pass 2 key index stays sorted in ASCII byte order and binary-searchable
    when a 300-byte allele is present, without padding every key to that width."""
    from opengwasdb.layouts.dense.build_vcf import _build_variant_key_index

    long_alt = "A" * 300
    long_alt_c = "C" * 250
    source_lookup = {
        ("1", 10, "A", "G"): "1:10:A:G",
        ("2", 5, "C", "T"): "2:5:C:T",
        ("1", 10, "A", long_alt): f"1:10:A:{long_alt}",
        ("1", 10, "A", long_alt_c): f"1:10:A:{long_alt_c}",
        ("1", 20, "G", "C"): "1:20:C:G",
    }
    variant_index = {
        "1:10:A:G": 0,
        "2:5:C:T": 1,
        f"1:10:A:{long_alt}": 2,
        f"1:10:A:{long_alt_c}": 3,
        "1:20:C:G": 4,
    }

    keys, rows = _build_variant_key_index(source_lookup, variant_index)

    # Variable-length Python bytes, not an S array padded to the 300-byte max.
    assert keys.dtype == object
    assert list(keys) == sorted(keys)

    expected = sorted(
        (
            (f"{chrom}:{pos}:{ref}:{alt}".encode(), variant_index[alid])
            for (chrom, pos, ref, alt), alid in source_lookup.items()
        ),
        key=lambda kv: kv[0],
    )
    assert list(keys) == [key for key, _ in expected]
    assert rows.tolist() == [row for _, row in expected]

    # `np.searchsorted` must find the long and short keys and not snap a missing
    # key to a neighbour.
    query = np.array(
        [b"1:10:A:G", b"1:10:A:" + long_alt.encode(), b"9:9:A:G"],
        dtype="S",
    )
    idx = np.searchsorted(keys, query)
    idx_clip = np.minimum(idx, len(keys) - 1)
    assert (keys[idx_clip] == query).tolist() == [True, True, False]


def test_ez_preferred_over_es_se(tmp_path):
    """When EZ is present and finite, it is used instead of ES/SE."""
    vcf = _make_vcf(
        tmp_path,
        "ez_trait",
        [f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tEZ:ES:SE\t7.5:2.0:0.5\n"],
    )
    manifest = _make_manifest(tmp_path, [("ez_trait", vcf, "EZ Trait")])
    store_path = tmp_path / "store.opengwasdb"
    build_dense_from_vcf_manifest(manifest, store_path, store_id="s", release_id="r")

    query = query_store(store_path)
    result = query.analysis("ez_trait")
    assert len(result["z"]) == 1
    # EZ=7.5, ALT>REF → flip → stored z = -7.5
    assert result["z"][0] == pytest.approx(-7.5, rel=5e-3)


def test_build_honours_source_reader_capability_column(tmp_path, monkeypatch):
    """A non-GWAS-VCF, non-bcftools reader can drive a build end-to-end when a
    manifest row declares a different source_reader_capability (issue #20) --
    the builder never assumes GWAS-VCF, it resolves whatever the manifest names."""
    from opengwasdb.readers import registry as readers_registry
    from opengwasdb.readers.fake import FakeReader
    from opengwasdb.readers.interface import ReaderAssociation

    fake_capability = "opengwasdb.test-fake"

    def _fake_factory(path, stored_effect_scale):
        return FakeReader(
            associations=[
                ReaderAssociation(
                    chromosome="1",
                    position=HG19_POS_1,
                    ref="A",
                    alt="G",
                    z=-2.0,
                    se=0.5,
                    stored_effect_scale=stored_effect_scale,
                )
            ]
        )

    monkeypatch.setitem(readers_registry._READERS, fake_capability, _fake_factory)

    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale"
        "\toriginal_sd_method\toriginal_sd\tsource_reader_capability\n"
        f"fake_trait\tunused-placeholder-path\tFake Trait\t1000\tsd"
        f"\tdeclared_standardised\t\t{fake_capability}\n",
        encoding="utf-8",
    )
    store_path = tmp_path / "store.opengwasdb"
    build_dense_from_vcf_manifest(manifest, store_path, store_id="s", release_id="r")

    query = query_store(store_path)
    result = query.analysis("fake_trait")
    assert len(result["z"]) == 1
    assert result["z"][0] == pytest.approx(-2.0, rel=5e-3)


_SSF_HEADER = [
    "chromosome",
    "base_pair_location",
    "effect_allele",
    "other_allele",
    "beta",
    "standard_error",
]


def _write_ssf(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\t".join(_SSF_HEADER) + "\n")
        for row in rows:
            fh.write("\t".join(str(row.get(col, "")) for col in _SSF_HEADER) + "\n")


def test_gwas_ssf_capability_builds_a_dense_store(tmp_path):
    """issue #84: build_dense_from_vcf_manifest resolves the registered
    GWAS-SSF capability the same way build_hybrid_from_vcf_manifest does
    (test_hybrid_build.py's equivalent test) -- same dense builder, no
    source-format branching, just a manifest row naming GWAS_SSF_CAPABILITY.
    Same z pattern (-4.0, -5.0, 3.0) as `two_trait_store`'s trait_a, sourced
    from a filtered/harmonised GWAS-SSF file instead of a VCF.
    """
    ssf_path = tmp_path / "trait_ssf.tsv.gz"
    _write_ssf(
        ssf_path,
        [
            # effect_allele=G, other_allele=A -> A1=A, effect != A1 -> flip -> z=-4.0
            {
                "chromosome": "1", "base_pair_location": HG19_POS_1,
                "effect_allele": "G", "other_allele": "A",
                "beta": 2.0, "standard_error": 0.5,
            },
            # effect_allele=T, other_allele=C -> A1=C, effect != A1 -> flip -> z=-5.0
            {
                "chromosome": "1", "base_pair_location": HG19_POS_2,
                "effect_allele": "T", "other_allele": "C",
                "beta": 1.5, "standard_error": 0.3,
            },
            # effect_allele=A, other_allele=G -> A1=A, effect == A1 -> no flip -> z=3.0
            {
                "chromosome": "1", "base_pair_location": HG19_POS_3,
                "effect_allele": "A", "other_allele": "G",
                "beta": 0.6, "standard_error": 0.2,
            },
        ],
    )
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale"
        "\toriginal_sd_method\toriginal_sd\tsource_reader_capability\n"
        f"trait_ssf\t{ssf_path}\tTrait SSF\t1000\tsd\tdeclared_standardised\t\t"
        f"{GWAS_SSF_CAPABILITY}\n",
        encoding="utf-8",
    )
    store_path = tmp_path / "store_ssf.opengwasdb"
    build_dense_from_vcf_manifest(
        manifest, store_path, store_id="dense-ssf-test", release_id="v1"
    )

    result = validate_store(store_path)
    assert result.ok, result.errors

    query = query_store(store_path)
    r = query.analysis("trait_ssf")
    query.close()

    assert set(np.round(r["z"], 1).tolist()) == {-4.0, -5.0, 3.0}


def test_finngen_r13_hg38_capability_builds_and_queries_dense_store(tmp_path):
    """A FinnGen endpoint stays on GRCh38 and round-trips through Dense."""
    from opengwasdb.readers import FINNGEN_R13_CAPABILITY

    source = Path(__file__).parent / "fixtures" / "finngen_r13.tsv"
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale"
        "\toriginal_sd_method\toriginal_sd\tsource_reader_capability\tsource_assembly\n"
        f"finngen_r13_test\t{source}\tFinnGen R13 test\t500000\tlog_or"
        f"\tbinary_trait\t\t{FINNGEN_R13_CAPABILITY}\tGRCh38\n",
        encoding="utf-8",
    )
    store_path = tmp_path / "finngen_r13.opengwasdb"

    build_dense_from_vcf_manifest(
        manifest, store_path, store_id="finngen-r13-test", release_id="r13"
    )

    validation = validate_store(store_path)
    assert validation.ok, validation.errors
    query = query_store(store_path)
    result = query.analysis("finngen_r13_test")
    variants = query.variants_table()
    query.close()
    assert {variants[int(index)]["position"] for index in result["variant_index"]} == {
        13668,
        17017,
        19234,
        98536,
    }
    np.testing.assert_allclose(
        np.sort(result["z"]), np.array([-0.7, 0.2, 0.4, 1.4]), atol=0.05
    )


def test_axis_source_alids_record_the_prelift_hg19_coordinate(tmp_path):
    """The variant axis carries each row's source-build ALID as provenance: a
    row stored at hg38 1:1064620 because its source file sat at hg19
    1:1000000 must say so, not claim the hg38 coordinate it was stored under.

    HG19_POS_1 is an identity lift (hg38 1:100000), so its source ALID equals
    the stored ALID; HG19_POS_2 genuinely moves. The fixture must contain a
    row that really moved or the assertion proves nothing -- both ALIDs are
    asserted so the set itself proves the lift happened.
    """
    from opengwasdb.variants.axis import iter_variant_records

    vcf = _make_vcf(
        tmp_path,
        "provenance_trait",
        [
            f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n",
            f"1\t{HG19_POS_2}\t.\tC\tT\t.\tPASS\t.\tES:SE\t1.5:0.3\n",
        ],
    )
    manifest = _make_manifest(tmp_path, [("provenance_trait", vcf, "Provenance Trait")])
    store_path = tmp_path / "store.opengwasdb"
    build_dense_from_vcf_manifest(manifest, store_path, store_id="s", release_id="r")

    records = {r.alid: r for r in iter_variant_records(store_path / "variants.tsv.gz")}
    assert set(records) == {HG38_ALID_1, HG38_ALID_2}
    # identity lift: the source and stored ALIDs coincide.
    assert records[HG38_ALID_1].source_alid == HG38_ALID_1
    # genuine translation: the source ALID is the hg19 coordinate, not hg38's.
    assert records[HG38_ALID_2].source_alid == f"1:{HG19_POS_2}:C:T"


def test_source_alids_blank_an_ambiguous_liftover_collision():
    """A stored row that several source variants resolved to is ambiguous, so
    its source_alid provenance is blank rather than guessed -- unless the
    source variants are one physical locus (same position, ref/alt reported
    either way round), whose identical canonical origin is safe to record.
    """
    from opengwasdb.layouts.dense.build_vcf import _source_alids_by_alid

    # Two distinct source positions lift onto one stored row -> ambiguous.
    # Single-source rows are recorded verbatim; a second file reporting the
    # same position with ref/alt swapped records the same origin, not None.
    source_lookup = {
        ("1", 100, "A", "G"): "1:100:A:G",  # row 0
        ("1", 200, "A", "G"): "1:100:A:G",  # distinct origin -> ambiguous
        ("1", 300, "C", "T"): "1:300:C:T",  # row 1, single origin
        ("1", 400, "A", "G"): "1:400:A:G",  # row 2
        ("1", 400, "G", "A"): "1:400:A:G",  # same locus, flipped report
    }
    assert _source_alids_by_alid(source_lookup, ["1:100:A:G", "1:300:C:T", "1:400:A:G"]) == [
        None,
        "1:300:C:T",
        "1:400:A:G",
    ]


def test_standalone_gwas_vcf_build_codes_residual_se_and_falls_back_without_eaf(tmp_path):
    from opengwasdb.model.manifest import StoreManifest

    """Issue #139/#142: a standalone Dense VCF build can select residual SE.

    Residual Dense-from-VCF evidence has been a review probe: the Hybrid
    builder shares ``build_vcf.py``'s helpers, but no committed test asserts
    ``is_residual`` on a release ``build_dense_from_vcf_manifest`` built on
    its own. Two EAF-carrying VCFs whose SE tracks the per-MAF model select
    ``int8_residual`` end to end -- queries return physical float32 values and
    validation passes -- while the same rows without an AF column must fall
    back to ``float16`` (and an absent EAF plane) yet still build, query and
    validate.
    """
    n_variants = 600
    frequencies = np.linspace(0.05, 0.95, n_variants, dtype=np.float64)

    def model_se(col: int, freq: float, phase: int) -> float:
        return float(
            np.exp(
                (-3.0 + col * 0.2)
                - 0.5 * np.log(2 * freq * (1 - freq))
                + 0.12 * np.sin(phase * (0.07 + col * 0.01))
            )
        )

    def rows_with_af(col: int) -> list[str]:
        lines: list[str] = []
        for i in range(n_variants):
            freq = frequencies[i]
            se = model_se(col, freq, i)
            z = 8.0 if i % 50 == 0 else 1.0
            effect = f"{z * se:.6f}:{se:.6f}"
            lines.append(
                f"1\t{(i + 1) * 1000}\t.\tA\tG\t.\tPASS\t.\tES:SE:AF"
                f"\t{effect}:{freq:.6f}\n"
            )
        return lines

    def rows_without_af(col: int) -> list[str]:
        lines: list[str] = []
        for i in range(n_variants):
            freq = frequencies[i]
            se = model_se(col, freq, i)
            z = 8.0 if i % 50 == 0 else 1.0
            lines.append(
                f"1\t{(i + 1) * 1000}\t.\tA\tG\t.\tPASS\t.\tES:SE"
                f"\t{z * se:.6f}:{se:.6f}\n"
            )
        return lines

    def manifest(vcfs: dict[str, Path], name: str) -> Path:
        path = tmp_path / f"{name}.manifest.tsv"
        path.write_text(
            "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale"
            "\toriginal_sd_method\tsource_assembly\n"
            + "\n".join(
                f"{tid}\t{vcf}\tTrait {tid}\t1000\tsd\tdeclared_standardised\thg38"
                for tid, vcf in vcfs.items()
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def expected_for(col: int) -> np.ndarray:
        return np.array(
            [model_se(col, frequencies[i], i) for i in range(n_variants)], dtype=np.float32
        )

    vcfs_with_af = {
        tid: write_gwas_vcf_with_eaf(tmp_path / f"{tid}.vcf", rows_with_af(col))
        for col, tid in enumerate(("t0", "t1"))
    }
    residual = tmp_path / "residual.opengwasdb"
    build_dense_from_vcf_manifest(
        manifest(vcfs_with_af, "residual"),
        residual,
        store_id="s",
        release_id="r",
        chunk_shape=(100, 2),
    )
    encoding = StoreManifest.load(residual).encoding
    assert encoding.se.is_residual, "the AF-bearing twin must residual-code its se plane"
    root = open_store(residual).arrays(mode="r")
    assert root["se"].dtype == np.dtype("int8")
    with query_store(residual) as query:
        for tid, col in (("t0", 0), ("t1", 1)):
            result = query.analysis(tid)
            assert result["se"].dtype == np.dtype("float32")
            np.testing.assert_allclose(result["se"], expected_for(col), rtol=0.01)
            assert np.isfinite(result["se"]).all()
        hits = query.top_hits(threshold=5e-8)
        assert len(hits["z"]) == 24  # every 50th of 600 variants x 2 Analyses carries |z|=8
        assert hits["se"].dtype == np.dtype("float32")
        assert np.isfinite(hits["se"]).all()
    assert validate_store(residual).ok, validate_store(residual).errors

    no_af = {tid: write_gwas_vcf_with_eaf(tmp_path / f"{tid}.noaf.vcf", rows_without_af(col))
             for col, tid in enumerate(("t0", "t1"))}
    fallback = tmp_path / "fallback.opengwasdb"
    build_dense_from_vcf_manifest(
        manifest(no_af, "fallback"),
        fallback,
        store_id="s",
        release_id="r",
        chunk_shape=(100, 2),
    )
    fallback_encoding = StoreManifest.load(fallback).encoding
    assert not fallback_encoding.se.is_residual
    assert fallback_encoding.se.kind == "float16"
    with query_store(fallback) as query:
        for tid, col in (("t0", 0), ("t1", 1)):
            result = query.analysis(tid)
            assert result["se"].dtype == np.dtype("float32")
            assert np.isfinite(result["se"]).all()
            np.testing.assert_allclose(result["se"], expected_for(col), rtol=0.01)
    assert validate_store(fallback).ok, validate_store(fallback).errors


def test_cli_default_source_assembly_applies_to_dense_build(tmp_path):
    """Issue #174: source_assembly CLI default applies to rows that omit the column."""
    vcf = _make_vcf(
        tmp_path, "trait_cli_ass",
        [f"1\t{HG19_POS_2}\t.\tC\tT\t.\tPASS\t.\tES:SE\t1.5:0.3\n"],
    )
    manifest = tmp_path / "manifest_no_assembly.tsv"
    manifest.write_text(
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale\toriginal_sd_method\n"
        f"trait_cli_ass\t{vcf}\tTrait CLI Ass\t1000\tsd\tdeclared_standardised\n",
        encoding="utf-8",
    )
    store_path = tmp_path / "store.opengwasdb"
    build_dense_from_vcf_manifest(
        manifest, store_path, store_id="s", release_id="r", source_assembly="hg38"
    )

    query = query_store(store_path)
    result = query.analysis("trait_cli_ass")
    vt = query.variants_table()
    query.close()
    assert len(result["z"]) == 1
    # hg38 default means no liftover -> stays HG19_POS_2 (1,000,000)
    assert vt[int(result["variant_index"][0])]["position"] == HG19_POS_2


def test_cli_default_source_assembly_per_row_override_dense(tmp_path):
    """Issue #174: per-row source_assembly overrides CLI default option."""
    vcf_hg38 = _make_vcf(
        tmp_path, "trait_row_hg38",
        [f"1\t{HG19_POS_2}\t.\tC\tT\t.\tPASS\t.\tES:SE\t1.5:0.3\n"],
    )
    vcf_hg19 = _make_vcf(
        tmp_path, "trait_row_hg19",
        [f"1\t{HG19_POS_3}\t.\tG\tA\t.\tPASS\t.\tES:SE\t0.6:0.2\n"],
    )
    manifest = _manifest_with_source_assembly(
        tmp_path,
        [
            ("trait_row_hg38", vcf_hg38, "Row hg38", "hg38"),
            ("trait_row_hg19", vcf_hg19, "Row hg19", "hg19"),
        ],
    )
    store_path = tmp_path / "store.opengwasdb"
    # CLI default is hg19, but trait_row_hg38 declares hg38 in manifest row
    build_dense_from_vcf_manifest(
        manifest, store_path, store_id="s", release_id="r", source_assembly="hg19"
    )
    query = query_store(store_path)
    r38 = query.analysis("trait_row_hg38")
    r19 = query.analysis("trait_row_hg19")
    vt = query.variants_table()
    query.close()
    assert vt[int(r38["variant_index"][0])]["position"] == HG19_POS_2  # not lifted
    assert vt[int(r19["variant_index"][0])]["position"] == 1564620     # lifted from 1500000


def test_cli_default_source_reader_capability_applies_to_dense_build(tmp_path):
    """Issue #174: source_reader_capability CLI default applies to rows that omit the column."""
    from opengwasdb.readers import FINNGEN_R13_CAPABILITY

    source = Path(__file__).parent / "fixtures" / "finngen_r13.tsv"
    manifest = tmp_path / "manifest_no_cap.tsv"
    manifest.write_text(
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale\toriginal_sd_method\n"
        f"finngen_cli_test\t{source}\tFinnGen CLI\t500000\tlog_or\tbinary_trait\n",
        encoding="utf-8",
    )
    store_path = tmp_path / "finngen_cli.opengwasdb"
    build_dense_from_vcf_manifest(
        manifest,
        store_path,
        store_id="finngen-cli",
        release_id="r13",
        source_reader_capability=FINNGEN_R13_CAPABILITY,
        source_assembly="GRCh38",
    )
    assert validate_store(store_path).ok


def test_cli_invalid_options_fail_at_parse_time_dense(tmp_path):
    """Issue #174: invalid CLI options fail at parse time."""
    from typer.testing import CliRunner

    from opengwasdb.cli.main import app

    runner = CliRunner()
    args_base = [
        "build-dense-vcf",
        str(tmp_path / "manifest.tsv"),
        str(tmp_path / "out.opengwasdb"),
        "--store-id", "s",
        "--release-id", "r",
    ]

    res_cap = runner.invoke(app, [*args_base, "--source-reader-capability", "unknown_capability"])
    assert res_cap.exit_code != 0
    clean_cap_output = normalize_cli_output(res_cap.output)
    assert "unknown source reader capability 'unknown_capability'" in clean_cap_output
    assert "known: opengwasdb.finngen-r13" in clean_cap_output

    res_ass = runner.invoke(app, [*args_base, "--source-assembly", "unknown_build"])
    assert res_ass.exit_code != 0
    clean_ass_output = normalize_cli_output(res_ass.output)
    assert "unknown genome build 'unknown_build'" in clean_ass_output.lower()
    assert "use hg19/hg38 or aliases grch37/grch38" in clean_ass_output.lower()


def test_direct_api_empty_source_assembly_fails_loudly(tmp_path):
    """Passing source_assembly='' fails loudly rather than silently defaulting to hg19."""
    vcf = _make_vcf(
        tmp_path, "trait_empty_ass",
        [f"1\t{HG19_POS_2}\t.\tC\tT\t.\tPASS\t.\tES:SE\t1.5:0.3\n"],
    )
    manifest = tmp_path / "manifest_empty_ass.tsv"
    manifest.write_text(
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale\toriginal_sd_method\n"
        f"trait_empty_ass\t{vcf}\tTrait Empty Ass\t1000\tsd\tdeclared_standardised\n",
        encoding="utf-8",
    )
    store_path = tmp_path / "store.opengwasdb"
    with pytest.raises(ValueError, match=r"[Uu]nknown genome build ''"):
        build_dense_from_vcf_manifest(
            manifest, store_path, store_id="s", release_id="r", source_assembly=""
        )

    with pytest.raises(ValueError, match=r"unknown source reader capability ''"):
        build_dense_from_vcf_manifest(
            manifest, store_path, store_id="s", release_id="r", source_reader_capability=""
        )



# ── single-pass builds from a precomputed variant reference (issue #185) ─────


def _write_reference_from_manifest(
    tmp_path: Path, manifest: Path, name: str = "panel.variant-ref.tsv.gz"
) -> Path:
    """Write the artifact the inline two-pass Pass 1 would recompute.

    Uses the builder's own Pass 1 (union + liftover + first-named rsid) so the
    reference genuinely matches the manifest's variant union.
    """
    from opengwasdb.layouts.dense.build_vcf import (
        _lift_manifest_variants,
        _read_manifest,
        _sorted_alids,
    )

    rows = _read_manifest(manifest)
    source_lookup, rsid_by_alid = _lift_manifest_variants(
        rows, chain_file=None, liftover_failure_threshold=0.01
    )
    path = tmp_path / name
    write_variant_reference(
        path, _sorted_alids(source_lookup.values()), source_lookup, rsid_by_alid
    )
    return path


def _assert_dense_stores_data_identical(reference: Path, candidate: Path) -> None:
    """The stored data (z/se, axis, top hits, analyses) matches exactly.

    ``manifest.json`` is excluded: it carries a ``created_at`` timestamp and the
    axis-origin provenance, which are expected to differ between runs.
    """
    import gzip

    from opengwasdb.layouts.dense.top_hits import threshold_key
    from opengwasdb.variants.axis import iter_variant_records

    with gzip.open(reference / "variants.tsv.gz", "rt", encoding="utf-8") as a:
        reference_axis = a.read()
    with gzip.open(candidate / "variants.tsv.gz", "rt", encoding="utf-8") as b:
        candidate_axis = b.read()
    assert reference_axis == candidate_axis
    np.testing.assert_array_equal(
        np.load(reference / "variant_offsets.npy"), np.load(candidate / "variant_offsets.npy")
    )
    assert {r.alid for r in iter_variant_records(reference / "variants.tsv.gz")} == {
        r.alid for r in iter_variant_records(candidate / "variants.tsv.gz")
    }

    reference_root = open_store(reference).arrays(mode="r")
    candidate_root = open_store(candidate).arrays(mode="r")
    for name in ("z", "se"):
        np.testing.assert_array_equal(reference_root[name][:], candidate_root[name][:])
    for threshold in (5e-4, 5e-6, 5e-8):
        key = f"top_hits/{threshold_key(threshold)}"
        for field in ("variant_index", "analysis_index", "z", "se"):
            np.testing.assert_array_equal(
                reference_root[key][field][:], candidate_root[key][field][:]
            )
    assert (reference / "analyses.tsv").read_text() == (candidate / "analyses.tsv").read_text()


class TestVariantReferenceSinglePass:
    def _mixed_manifest(self, tmp_path: Path) -> Path:
        vcf_hg19 = _make_vcf(
            tmp_path,
            "trait_hg19",
            [
                f"1\t{HG19_POS_1}\trs1\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n",
                f"1\t{HG19_POS_2}\t.\tC\tT\t.\tPASS\t.\tES:SE\t1.5:0.3\n",
                f"1\t{HG19_POS_3}\trs3\tG\tA\t.\tPASS\t.\tES:SE\t0.6:0.2\n",
            ],
        )
        vcf_second = _make_vcf(
            tmp_path,
            "trait_second",
            [
                # Same site as trait_hg19's rs1, named differently: first named
                # in manifest order must still win.
                f"1\t{HG19_POS_1}\trsX\tA\tG\t.\tPASS\t.\tES:SE\t6.0:0.5\n",
                f"1\t{HG19_POS_3}\trs3b\tG\tA\t.\tPASS\t.\tES:SE\t1.2:0.3\n",
            ],
        )
        vcf_hg38 = _make_vcf(
            tmp_path, "trait_hg38",
            ["1\t5000000\trs5\tC\tT\t.\tPASS\t.\tES:SE\t0.8:0.4\n"],
        )
        return _manifest_with_source_assembly(
            tmp_path,
            [
                ("trait_hg19", vcf_hg19, "Trait hg19", ""),
                ("trait_second", vcf_second, "Trait second", ""),
                ("trait_hg38", vcf_hg38, "Trait hg38", "hg38"),
            ],
        )

    def test_single_pass_matches_two_pass_bit_for_bit(self, tmp_path):
        manifest = self._mixed_manifest(tmp_path)
        reference = _write_reference_from_manifest(tmp_path, manifest)

        two_pass = tmp_path / "two-pass.opengwasdb"
        single_pass = tmp_path / "single-pass.opengwasdb"
        build_dense_from_vcf_manifest(
            manifest, two_pass, store_id="s", release_id="r", n_workers=2
        )
        build_dense_from_vcf_manifest(
            manifest, single_pass, store_id="s", release_id="r", n_workers=2,
            variant_reference=reference,
        )

        assert validate_store(single_pass).ok
        _assert_dense_stores_data_identical(two_pass, single_pass)

        provenance = json.loads((single_pass / "manifest.json").read_text())["provenance"]
        assert provenance["variant_reference"] == str(reference)
        assert provenance["builder"] == "opengwasdb.v0.1_dense_vcf_single_pass"

    def test_single_pass_never_runs_the_union_pass(self, tmp_path, monkeypatch):
        """The reference supplies the axis, so the Pass 1 variant-union read is
        never invoked -- proven by making it raise."""
        import opengwasdb.layouts.dense.build_vcf as build_vcf

        manifest = self._mixed_manifest(tmp_path)
        reference = _write_reference_from_manifest(tmp_path, manifest)

        def _boom(*_args, **_kwargs):
            raise AssertionError("Pass 1 ran despite a supplied variant reference")

        monkeypatch.setattr(build_vcf, "_collect_manifest_variant_sites", _boom)
        store = tmp_path / "union-bypassed.opengwasdb"
        build_dense_from_vcf_manifest(
            manifest, store, store_id="s", release_id="r", variant_reference=reference
        )
        assert validate_store(store).ok

    def test_single_pass_logs_pass1_bypass(self, tmp_path, caplog):
        import logging

        manifest = self._mixed_manifest(tmp_path)
        reference = _write_reference_from_manifest(tmp_path, manifest)
        store = tmp_path / "logged.opengwasdb"

        with caplog.at_level(logging.INFO):
            build_dense_from_vcf_manifest(
                manifest, store, store_id="s", release_id="r", variant_reference=reference
            )

        assert "Single-pass build: variant axis loaded from" in caplog.text
        assert "Pass 1: collecting source variants" not in caplog.text

    def test_omitted_reference_still_runs_two_pass(self, tmp_path, caplog):
        import logging

        manifest = self._mixed_manifest(tmp_path)
        store = tmp_path / "two-pass.opengwasdb"

        with caplog.at_level(logging.INFO):
            build_dense_from_vcf_manifest(manifest, store, store_id="s", release_id="r")

        assert "Pass 1" in caplog.text
        provenance = json.loads((store / "manifest.json").read_text())["provenance"]
        assert "variant_reference" not in provenance
        assert provenance["builder"] == "opengwasdb.v0.1_dense_vcf_two_pass"

    def test_plain_alid_list_drops_off_reference_and_keeps_unobserved_nan(self, tmp_path):
        vcf = _make_vcf(
            tmp_path,
            "trait_hg38",
            [
                "1\t100000\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n",    # in reference
                "1\t200000\t.\tC\tT\t.\tPASS\t.\tES:SE\t1.5:0.3\n",    # off-reference
                "1\t5000000\t.\tG\tA\t.\tPASS\t.\tES:SE\t0.6:0.2\n",   # in reference
            ],
        )
        manifest = _manifest_with_source_assembly(
            tmp_path, [("trait_hg38", vcf, "Trait hg38", "hg38")]
        )
        reference = tmp_path / "panel.alids"
        reference.write_text("1:100000:A:G\n1:5000000:A:G\n1:9000000:C:T\n", encoding="utf-8")
        store = tmp_path / "plain-list.opengwasdb"

        build_dense_from_vcf_manifest(
            manifest, store, store_id="s", release_id="r", variant_reference=reference
        )

        assert validate_store(store).ok
        from opengwasdb.encoding import DenseZPlane
        from opengwasdb.variants.axis import iter_variant_records

        records = {r.alid for r in iter_variant_records(store / "variants.tsv.gz")}
        assert records == {"1:100000:A:G", "1:5000000:A:G", "1:9000000:C:T"}
        assert "1:200000:C:T" not in records

        opened = open_store(store)
        z_column = DenseZPlane.open(opened.arrays(mode="r"), opened.manifest.encoding).column(0)
        assert z_column.shape == (3,)
        assert z_column[0] == pytest.approx(-4.0, rel=5e-3)  # 100000, A1=A, flipped
        assert z_column[1] == pytest.approx(3.0, rel=5e-3)   # 5000000, A1=A, no flip
        assert np.isnan(z_column[2])                          # 9000000, unobserved

    def test_store_variants_table_is_accepted_as_reference(self, tmp_path):
        vcf = _make_vcf(
            tmp_path,
            "trait_hg38",
            [
                "1\t100000\trs1\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n",
                "1\t5000000\trs5\tG\tA\t.\tPASS\t.\tES:SE\t0.6:0.2\n",
            ],
        )
        manifest = _manifest_with_source_assembly(
            tmp_path, [("trait_hg38", vcf, "Trait hg38", "hg38")]
        )
        source_store = tmp_path / "source.opengwasdb"
        build_dense_from_vcf_manifest(manifest, source_store, store_id="s", release_id="r")

        rebuilt = tmp_path / "rebuilt.opengwasdb"
        build_dense_from_vcf_manifest(
            manifest, rebuilt, store_id="s", release_id="r",
            variant_reference=source_store / "variants.tsv.gz",
        )

        assert validate_store(rebuilt).ok
        _assert_dense_stores_data_identical(source_store, rebuilt)


class TestVariantReferenceErrors:
    def _one_row_manifest(self, tmp_path: Path) -> Path:
        vcf = _make_vcf(
            tmp_path, "trait_a",
            [f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n"],
        )
        return _make_manifest(tmp_path, [("trait_a", vcf, "Trait A")])

    def test_identity_reference_with_hg19_manifest_warns_and_drops(self, tmp_path, caplog):
        """An identity panel resolves only hg38 coordinates; an hg19 manifest is
        accepted (off-reference variants drop as documented), but the assembly
        mismatch is logged so it cannot pass unremarked."""
        import logging

        from opengwasdb.variants.axis import iter_variant_records

        vcf = _make_vcf(
            tmp_path, "trait_hg19",
            [
                f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n",
                f"1\t{HG19_POS_2}\t.\tC\tT\t.\tPASS\t.\tES:SE\t1.5:0.3\n",
            ],
        )
        manifest = _make_manifest(tmp_path, [("trait_hg19", vcf, "Trait hg19")])
        reference = tmp_path / "panel.alids"
        reference.write_text("1:100000:A:G\n", encoding="utf-8")
        store = tmp_path / "identity.opengwasdb"

        with caplog.at_level(logging.WARNING):
            build_dense_from_vcf_manifest(
                manifest, store, store_id="s", release_id="r", variant_reference=reference
            )

        assert "carries no source-key mapping" in caplog.text
        assert validate_store(store).ok
        assert {r.alid for r in iter_variant_records(store / "variants.tsv.gz")} == {
            "1:100000:A:G"
        }

    def test_artifact_round_trips_source_keys_and_rsids(self, tmp_path):
        source_lookup = {
            ("1", HG19_POS_1, "A", "G"): HG38_ALID_1,
            ("1", HG19_POS_2, "C", "T"): HG38_ALID_2,
            ("1", HG19_POS_3, "G", "A"): HG38_ALID_3,
        }
        rsid_by_alid = {HG38_ALID_1: "rs1", HG38_ALID_3: "rs3"}
        path = tmp_path / "roundtrip.variant-ref.tsv.gz"
        write_variant_reference(path, list(source_lookup.values()), source_lookup, rsid_by_alid)

        reference = read_variant_reference(path)

        assert reference.explicit_source_keys is True
        assert set(reference.alids) == set(source_lookup.values())
        assert reference.rsid_by_alid == rsid_by_alid
        for key, alid in source_lookup.items():
            assert reference.source_lookup[key] == alid

    def test_missing_file_fails_loudly(self, tmp_path):
        manifest = self._one_row_manifest(tmp_path)
        with pytest.raises(ValueError, match="does not exist"):
            build_dense_from_vcf_manifest(
                manifest, tmp_path / "store.opengwasdb", store_id="s", release_id="r",
                variant_reference=tmp_path / "absent.variant-ref.tsv.gz",
            )

    def test_empty_reference_fails_loudly(self, tmp_path):
        import gzip

        manifest = self._one_row_manifest(tmp_path)
        empty = tmp_path / "empty.variant-ref.tsv.gz"
        with gzip.open(empty, "wt", encoding="utf-8"):
            pass
        with pytest.raises(ValueError, match="is empty"):
            build_dense_from_vcf_manifest(
                manifest, tmp_path / "store.opengwasdb", store_id="s", release_id="r",
                variant_reference=empty,
            )

    def test_reference_with_no_alid_rows_fails_loudly(self, tmp_path):
        import gzip

        manifest = self._one_row_manifest(tmp_path)
        header_only = tmp_path / "header-only.variant-ref.tsv.gz"
        with gzip.open(header_only, "wt", encoding="utf-8") as handle:
            handle.write("#alid\tchromosome\tposition\ta1\ta2\trsid\tsource_keys\n")
        with pytest.raises(ValueError, match="contained no ALIDs"):
            build_dense_from_vcf_manifest(
                manifest, tmp_path / "store.opengwasdb", store_id="s", release_id="r",
                variant_reference=header_only,
            )

    def test_invalid_alid_fails_loudly(self, tmp_path):
        manifest = self._one_row_manifest(tmp_path)
        bad = tmp_path / "bad.alids"
        bad.write_text("1:100000:A:G\nnot-an-alid\n", encoding="utf-8")
        with pytest.raises(ValueError, match="invalid ALID"):
            build_dense_from_vcf_manifest(
                manifest, tmp_path / "store.opengwasdb", store_id="s", release_id="r",
                variant_reference=bad,
            )

    def test_mismatched_artifact_columns_fail_loudly(self, tmp_path):
        import gzip

        manifest = self._one_row_manifest(tmp_path)
        bad = tmp_path / "mismatch.variant-ref.tsv.gz"
        with gzip.open(bad, "wt", encoding="utf-8") as handle:
            handle.write("#alid\tchromosome\tposition\ta1\ta2\trsid\tsource_keys\n")
            handle.write("1:100000:A:G\t1\t999\tA\tG\t\t1:100000:A:G\n")
        with pytest.raises(ValueError, match="row columns describe"):
            build_dense_from_vcf_manifest(
                manifest, tmp_path / "store.opengwasdb", store_id="s", release_id="r",
                variant_reference=bad,
            )

    def test_tabular_reference_without_known_columns_fails_loudly(self, tmp_path):
        import gzip

        bad = tmp_path / "unknown.tsv.gz"
        with gzip.open(bad, "wt", encoding="utf-8") as handle:
            handle.write("foo\tbar\n1\t2\n")
        with pytest.raises(ValueError, match="neither an 'alid' nor a 'source_keys' column"):
            read_variant_reference(bad)

    def test_variant_artifact_missing_required_column_fails_loudly(self, tmp_path):
        import gzip

        bad = tmp_path / "missing-column.variant-ref.tsv.gz"
        with gzip.open(bad, "wt", encoding="utf-8") as handle:
            handle.write("#alid\tchromosome\tposition\ta1\ta2\n1:100000:A:G\t1\t100000\tA\tG\n")
        with pytest.raises(ValueError, match="missing the column"):
            read_variant_reference(bad)

    def test_conflicting_source_key_fails_loudly(self, tmp_path):
        import gzip

        bad = tmp_path / "conflict.variant-ref.tsv.gz"
        with gzip.open(bad, "wt", encoding="utf-8") as handle:
            handle.write("#alid\tchromosome\tposition\ta1\ta2\trsid\tsource_keys\n")
            handle.write("1:100000:A:G\t1\t100000\tA\tG\t\t1:200000:A:G\n")
            handle.write("1:100000:C:T\t1\t100000\tC\tT\t\t1:200000:A:G\n")
        with pytest.raises(ValueError, match="maps to both"):
            read_variant_reference(bad)

    def test_duplicate_alid_in_artifact_fails_loudly(self, tmp_path):
        import gzip

        bad = tmp_path / "duplicate.variant-ref.tsv.gz"
        with gzip.open(bad, "wt", encoding="utf-8") as handle:
            handle.write("#alid\tchromosome\tposition\ta1\ta2\trsid\tsource_keys\n")
            handle.write("1:100000:A:G\t1\t100000\tA\tG\t\t1:100000:A:G\n")
            handle.write("1:100000:A:G\t1\t100000\tA\tG\t\t1:100000:A:G\n")
        with pytest.raises(ValueError, match="duplicate ALID"):
            read_variant_reference(bad)

    def test_invalid_source_key_fails_loudly(self, tmp_path):
        import gzip

        bad = tmp_path / "bad-key.variant-ref.tsv.gz"
        with gzip.open(bad, "wt", encoding="utf-8") as handle:
            handle.write("#alid\tchromosome\tposition\ta1\ta2\trsid\tsource_keys\n")
            handle.write("1:100000:A:G\t1\t100000\tA\tG\t\t1:not-a-position:A:G\n")
        with pytest.raises(ValueError, match="invalid source key"):
            read_variant_reference(bad)


def test_cli_variant_reference_builds_single_pass(tmp_path):
    """The CLI exposes --variant-reference and produces a valid store (#185)."""
    from typer.testing import CliRunner

    from opengwasdb.cli.main import app

    vcf = _make_vcf(
        tmp_path, "trait_hg38",
        [
            "1\t100000\trs1\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n",
            "1\t5000000\trs5\tG\tA\t.\tPASS\t.\tES:SE\t0.6:0.2\n",
        ],
    )
    manifest = _manifest_with_source_assembly(
        tmp_path, [("trait_hg38", vcf, "Trait hg38", "hg38")]
    )
    reference = tmp_path / "panel.alids"
    reference.write_text("1:100000:A:G\n1:5000000:A:G\n", encoding="utf-8")
    store = tmp_path / "cli.opengwasdb"

    result = CliRunner().invoke(
        app,
        [
            "build-dense-vcf", str(manifest), str(store),
            "--store-id", "s", "--release-id", "r",
            "--variant-reference", str(reference),
        ],
    )

    assert result.exit_code == 0, result.output
    assert validate_store(store).ok
    z = open_store(store).arrays(mode="r")["z"][:]
    assert z.shape == (2, 1)
