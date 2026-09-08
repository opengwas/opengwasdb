from __future__ import annotations

import gzip

import numpy as np
import pytest

from opengwasdb.layouts.dense.rho import build_dense_rho
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.layouts.ragged.build_ssf import build_ragged_from_ssf
from opengwasdb.model.analyses import read_analyses, write_analyses
from opengwasdb.store.open import open_store
from opengwasdb.validation import validate_store


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
        "n\ttissue\tcontext\tmhc\tfiltered_file\n"
        "0\tt1\tT1\tGENE1\tENSEMBL:ENSG1\tEnsembl\t1\t150\t1000\tBlood\t\tFALSE\tt1.tsv.gz\n",
        encoding="utf-8",
    )
    store_path = tmp_path / "ragged.opengwasdb"
    build_ragged_from_ssf(manifest, filtered_dir, store_path, store_id="fixture", release_id="v1")
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


def test_validator_rejects_missing_required_array(dense_store_path):
    root = open_store(dense_store_path).arrays(mode="a")
    del root["se"]

    result = validate_store(dense_store_path)

    assert not result.ok
    assert "missing data.zarr/se" in result.errors


def test_validator_rejects_negative_se(dense_store_path):
    root = open_store(dense_store_path).arrays(mode="a")
    root["se"][0, 0] = -0.1

    result = validate_store(dense_store_path)

    assert not result.ok
    assert "se contains negative finite values" in result.errors


def test_validator_rejects_inconsistent_missingness(dense_store_path):
    root = open_store(dense_store_path).arrays(mode="a")
    root["se"][0, 0] = float("nan")

    result = validate_store(dense_store_path)

    assert not result.ok
    assert "z and se missingness is inconsistent" in result.errors


def test_validator_rejects_invalid_stored_effect_scale(dense_store_path):
    analyses_path = dense_store_path / "analyses.tsv"
    table = read_analyses(analyses_path)
    for row in table.rows:
        if row["analysis_id"] == "a1":
            row["stored_effect_scale"] = "kg"
    write_analyses(analyses_path, table)

    result = validate_store(dense_store_path)

    assert not result.ok
    assert any("invalid stored_effect_scale" in error for error in result.errors)


def test_validator_rejects_inconsistent_top_hit_index(dense_store_path):
    root = open_store(dense_store_path).arrays(mode="a")
    root["top_hits"]["p_5e_08"]["z"][0] = 0.0

    result = validate_store(dense_store_path)

    assert not result.ok
    assert any("top-hit index p_5e_08" in error for error in result.errors)


def test_validator_rejects_invalid_top_hit_offsets(dense_store_path):
    root = open_store(dense_store_path).arrays(mode="r+")
    offsets = root["top_hits/p_5e_08/analysis_offsets"][:]
    offsets[-1] -= 1
    root["top_hits/p_5e_08/analysis_offsets"][:] = offsets

    result = validate_store(dense_store_path)

    assert not result.ok
    assert any("invalid analysis offsets" in error for error in result.errors)


def test_validator_rejects_missing_variant_axis_file(dense_store_path):
    (dense_store_path / "variants.tsv.gz").unlink()

    result = validate_store(dense_store_path)

    assert not result.ok
    assert "missing variants.tsv.gz" in result.errors


def test_validator_rejects_bad_variant_offset(dense_store_path):
    offsets_path = dense_store_path / "variant_offsets.npy"
    offsets = np.load(offsets_path)
    offsets[1] = offsets[0]
    np.save(offsets_path, offsets)

    result = validate_store(dense_store_path)

    assert not result.ok
    assert any("variant offset for row 1" in error for error in result.errors)


def test_validator_rejects_leftover_sqlite_analyses_table(dense_store_path):
    # ADR 0030/issue #22: analyses.tsv is the sole source of truth for
    # Analytical Metadata; a store carrying a leftover SQLite `analyses`
    # table (store-format spec §20) is invalid even though nothing else
    # about it is malformed.
    connection = open_store(dense_store_path).index_connection()
    try:
        connection.execute("CREATE TABLE analyses (analysis_index INTEGER PRIMARY KEY)")
        connection.commit()
    finally:
        connection.close()

    result = validate_store(dense_store_path)

    assert not result.ok
    assert any("still contains an analyses table" in error for error in result.errors)


def test_validator_accepts_store_without_sqlite_analyses_table(dense_store_path):
    result = validate_store(dense_store_path)

    assert result.ok, result.errors


def test_validator_rejects_leftover_sqlite_analyses_table_ragged(ragged_store_path):
    # Issue #69/#72: Ragged retired its SQLite `analyses` table onto
    # analyses.tsv too -- the same rule Dense already enforces must catch a
    # Ragged release that still carries one.
    connection = open_store(ragged_store_path).index_connection()
    try:
        connection.execute("CREATE TABLE analyses (analysis_index INTEGER PRIMARY KEY)")
        connection.commit()
    finally:
        connection.close()

    result = validate_store(ragged_store_path)

    assert not result.ok
    assert any("still contains an analyses table" in error for error in result.errors)


def test_validator_rejects_missing_ragged_csr_array(ragged_store_path):
    # The Ragged validator's CSR-structure seam must not turn a missing
    # parallel array into a decode crash in a later seam.
    result = validate_store(ragged_store_path)
    assert result.ok, result.errors
    ragged = open_store(ragged_store_path).arrays(mode="a")["ragged"]
    assert "se" in ragged and len(ragged["se"]) == int(ragged["offsets"][-1])  # fixture sanity
    del ragged["se"]

    result = validate_store(ragged_store_path)

    assert not result.ok
    assert "missing data.zarr/ragged/se" in result.errors


def test_validator_rejects_ragged_csr_array_length_mismatch(ragged_store_path):
    # A parallel array shorter than `offsets` implies is a structural error
    # that must be reported, not read as the store silently losing rows.
    ragged = open_store(ragged_store_path).arrays(mode="r")["ragged"]
    n_assoc = int(ragged["offsets"][-1])
    assert n_assoc > 1  # fixture sanity: the truncation must actually shorten z
    assert len(ragged["z"]) == n_assoc
    ragged = open_store(ragged_store_path).arrays(mode="a")["ragged"]
    z = ragged["z"][:]
    del ragged["z"]
    ragged.create_dataset("z", data=z[:-1])

    result = validate_store(ragged_store_path)

    assert not result.ok
    assert any(
        "data.zarr/ragged/z has" in error and "entries but offsets imply" in error
        for error in result.errors
    )


def test_validator_rejects_ragged_csr_array_outside_declared_plan(ragged_store_path):
    # The manifest's encoding is the contract (issue #119): an array whose
    # dtype disagrees with it must be reported as such, and later seams must
    # not try to decode values under a plan the structure already contradicted.
    ragged = open_store(ragged_store_path).arrays(mode="r")["ragged"]
    assert str(ragged["z"].dtype) == "int16"  # fixture sanity
    ragged = open_store(ragged_store_path).arrays(mode="a")["ragged"]
    z = ragged["z"][:]
    del ragged["z"]
    ragged.create_dataset("z", data=z.astype(np.float32))

    result = validate_store(ragged_store_path)

    assert not result.ok
    assert any(
        "data.zarr/ragged/z has dtype float32 but the manifest declares int16_fixed"
        in error
        for error in result.errors
    )


def test_validator_rejects_negative_ragged_se(ragged_store_path):
    # Same rule as the Dense band pass: a decoded negative finite se is a
    # wrong answer a query would serve unchanged, so it is a hard error.
    ragged = open_store(ragged_store_path).arrays(mode="r")["ragged"]
    assert np.all(np.asarray(ragged["se"][:]) > 0)  # fixture sanity
    ragged = open_store(ragged_store_path).arrays(mode="a")["ragged"]
    ragged["se"][0] = np.float16(-0.1)

    result = validate_store(ragged_store_path)

    assert not result.ok
    assert "se contains negative finite values" in result.errors


def test_validator_rejects_ragged_analyses_row_count_mismatch(ragged_store_path):
    # analyses.tsv is the sole source of truth for Analytical Metadata, so a
    # row count that disagrees with the CSR offsets is a store whose metadata
    # no longer describes its arrays (ADR 0030, spec §11/§20).
    result = validate_store(ragged_store_path)
    assert result.ok, result.errors
    analyses_path = ragged_store_path / "analyses.tsv"
    table = read_analyses(analyses_path)
    assert len(table.rows) == 1  # fixture sanity: dropping it must change the count
    write_analyses(analyses_path, type(table)(fieldnames=table.fieldnames, rows=()))

    result = validate_store(ragged_store_path)

    assert not result.ok
    assert any(
        "analyses.tsv has 0 rows but" in error and "offsets imply 1" in error
        for error in result.errors
    )


def test_validator_accepts_migrated_ragged_store(ragged_store_path):
    result = validate_store(ragged_store_path)

    assert result.ok, result.errors


def test_validator_accepts_migrated_hybrid_store(hybrid_store_path):
    # Issue #72: "a correctly-migrated Dense, Hybrid, and Ragged release...
    # succeeds" -- Hybrid's Dense Component reuses _validate_dense_store,
    # so this covers the same two new rules via that shared path.
    result = validate_store(hybrid_store_path)

    assert result.ok, result.errors


def test_validator_rejects_leftover_sqlite_analyses_table_hybrid(hybrid_store_path):
    from opengwasdb.layouts.hybrid.layout import dense_component_path

    connection = open_store(dense_component_path(hybrid_store_path)).index_connection()
    try:
        connection.execute("CREATE TABLE analyses (analysis_index INTEGER PRIMARY KEY)")
        connection.commit()
    finally:
        connection.close()

    result = validate_store(hybrid_store_path)

    assert not result.ok
    assert any("still contains an analyses table" in error for error in result.errors)


def test_validator_rejects_duplicate_analysis_id_hybrid(hybrid_store_path):
    from opengwasdb.layouts.hybrid.layout import dense_component_path

    analyses_path = dense_component_path(hybrid_store_path) / "analyses.tsv"
    table = read_analyses(analyses_path)
    rows = [{**row, "analysis_id": table.rows[0]["analysis_id"]} for row in table.rows]
    write_analyses(analyses_path, type(table)(fieldnames=table.fieldnames, rows=tuple(rows)))

    result = validate_store(hybrid_store_path)

    assert not result.ok
    assert any("more than one row for analysis_id" in error for error in result.errors)


def test_validator_rejects_duplicate_analysis_id(dense_store_path):
    # Issue #72: analysis_id uniqueness within analyses.tsv is enforced for
    # every layout, not just documented (store-format spec §7a).
    analyses_path = dense_store_path / "analyses.tsv"
    table = read_analyses(analyses_path)
    rows = list(table.rows)
    rows[1] = {**rows[1], "analysis_id": rows[0]["analysis_id"]}
    write_analyses(analyses_path, type(table)(fieldnames=table.fieldnames, rows=tuple(rows)))

    result = validate_store(dense_store_path)

    assert not result.ok
    assert any("more than one row for analysis_id" in error for error in result.errors)


def test_validator_rejects_stray_file_in_dense_store(dense_store_path):
    # Issue #80: the closed envelope catches a file no build path writes --
    # this is the class of bug that let a stray Ragged traits.tsv.gz side-file
    # go unnoticed until a human spotted it (issues #69-#73 / PR #78).
    (dense_store_path / "traits.tsv.gz").write_bytes(b"stray")

    result = validate_store(dense_store_path)

    assert not result.ok
    assert any(
        "unexpected store entry" in error and "traits.tsv.gz" in error
        for error in result.errors
    )


def test_validator_rejects_stray_directory_in_ragged_store(ragged_store_path):
    (ragged_store_path / "extra_dir").mkdir()

    result = validate_store(ragged_store_path)

    assert not result.ok
    assert any(
        "unexpected store entry" in error and "extra_dir" in error for error in result.errors
    )


def test_validator_rejects_stray_file_at_hybrid_top_level(hybrid_store_path):
    (hybrid_store_path / "traits.tsv.gz").write_bytes(b"stray")

    result = validate_store(hybrid_store_path)

    assert not result.ok
    assert any(
        "unexpected store entry" in error and "traits.tsv.gz" in error
        for error in result.errors
    )


def test_validator_rejects_stray_file_in_hybrid_dense_component(hybrid_store_path):
    from opengwasdb.layouts.hybrid.layout import dense_component_path

    (dense_component_path(hybrid_store_path) / "stray.txt").write_text("x")

    result = validate_store(hybrid_store_path)

    assert not result.ok
    assert any(
        "unexpected store entry" in error and "stray.txt" in error for error in result.errors
    )


def test_validator_accepts_clean_dense_ragged_hybrid_stores(
    dense_store_path, ragged_store_path, hybrid_store_path
):
    # Issue #80 acceptance criterion: a correctly-built release passes with no
    # false positives -- overview.html, the Hybrid dense/ subdirectory, and
    # its dense_to_shared.npy must all be recognised as legitimate, not just
    # rejected as unexpected.
    for store_path in (dense_store_path, ragged_store_path, hybrid_store_path):
        result = validate_store(store_path)
        assert result.ok, (store_path, result.errors)


def test_validator_rejects_retired_gene_id_column(dense_store_path):
    # Issue #81/ADR 0035: a built store's analyses.tsv carrying the retired
    # gene_id/gene_name columns (superseded by analysis_label/
    # trait_ontology_id/trait_ontology_label) is invalid against the current
    # schema, the same way ADR 0034's phenotype_id/phenotype_label/trait_id
    # retirement already is.
    analyses_path = dense_store_path / "analyses.tsv"
    table = read_analyses(analyses_path)
    fieldnames = (*table.fieldnames, "gene_id")
    rows = tuple({**row, "gene_id": "ENSG00000000001"} for row in table.rows)
    write_analyses(analyses_path, type(table)(fieldnames=fieldnames, rows=rows))

    result = validate_store(dense_store_path)

    assert not result.ok
    assert any("retired column" in error and "gene_id" in error for error in result.errors)


def test_validator_accepts_duplicate_trait_ontology_id(dense_store_path):
    # Issue #72: duplicate trait_ontology_id is expected and valid (several
    # Analyses of the same curated Trait) as long as analysis_id stays unique.
    analyses_path = dense_store_path / "analyses.tsv"
    table = read_analyses(analyses_path)
    rows = [{**row, "trait_ontology_id": "EFO:0000001"} for row in table.rows]
    write_analyses(analyses_path, type(table)(fieldnames=table.fieldnames, rows=tuple(rows)))

    result = validate_store(dense_store_path)

    assert result.ok, result.errors


def test_validator_rejects_invalid_original_sd_method(dense_store_path):
    analyses_path = dense_store_path / "analyses.tsv"
    table = read_analyses(analyses_path)
    for row in table.rows:
        if row["analysis_id"] == "a1":
            row["original_sd_method"] = "bogus_method"
    write_analyses(analyses_path, table)

    result = validate_store(dense_store_path)

    assert not result.ok
    assert any("invalid original_sd_method" in error for error in result.errors)


def test_validator_accepts_valid_original_sd_method(dense_store_path):
    analyses_path = dense_store_path / "analyses.tsv"
    table = read_analyses(analyses_path)
    for row in table.rows:
        row["original_sd_method"] = "declared_standardised"
    write_analyses(analyses_path, table)

    result = validate_store(dense_store_path)

    assert result.ok, result.errors


def test_validator_rejects_invalid_ancestry_assignment_method(dense_store_path):
    analyses_path = dense_store_path / "analyses.tsv"
    table = read_analyses(analyses_path)
    for row in table.rows:
        if row["analysis_id"] == "a2":
            row["ancestry_assignment_method"] = "bogus_method"
    write_analyses(analyses_path, table)

    result = validate_store(dense_store_path)

    assert not result.ok
    assert any("invalid ancestry_assignment_method" in error for error in result.errors)


def test_validator_accepts_valid_ancestry_assignment_method(dense_store_path):
    analyses_path = dense_store_path / "analyses.tsv"
    table = read_analyses(analyses_path)
    for row in table.rows:
        row["ancestry_assignment_method"] = "af_assigned"
    write_analyses(analyses_path, table)

    result = validate_store(dense_store_path)

    assert result.ok, result.errors


def test_validator_rejects_analyses_tsv_missing_referenced_analysis_index(dense_store_path):
    # store-format spec §20: analyses.tsv must cover every analysis_index
    # referenced by the store's positional indexing -- a gap (here, a2's
    # index jumps from 1 to 2, leaving 1 uncovered) is the one relationship
    # SQLite cannot enforce as a foreign key, since analyses.tsv is a
    # separate file.
    analyses_path = dense_store_path / "analyses.tsv"
    table = read_analyses(analyses_path)
    for row in table.rows:
        if row["analysis_id"] == "a2":
            row["analysis_index"] = "2"
    write_analyses(analyses_path, table)

    result = validate_store(dense_store_path)

    assert not result.ok
    assert any(
        "analysis_index values are not exactly the range" in error for error in result.errors
    )


def test_validator_accepts_analyses_tsv_covering_every_analysis_index(dense_store_path):
    result = validate_store(dense_store_path)

    assert result.ok, result.errors


# ── Rho Matrix (issue 049) ────────────────────────────────────────────────────
# window_bp=1 keeps every variant (100/200/300bp spacing); z_thresh=10.0 makes
# every finite z "null" -- the fixture then has exactly one shared-null variant
# (rs1, the only row both a1 and a2 have data for) for the single a1/a2 pair.


def _build_fixture_rho(dense_store_path, min_nulls: int = 1) -> None:
    build_dense_rho(dense_store_path, window_bp=1, z_thresh=10.0, min_nulls=min_nulls, n_workers=1)


def test_validator_accepts_store_without_rho_group(dense_store_path):
    result = validate_store(dense_store_path)

    assert result.ok, result.errors


def test_validator_accepts_well_formed_rho_matrix(dense_store_path):
    _build_fixture_rho(dense_store_path)

    result = validate_store(dense_store_path)

    assert result.ok, result.errors


def test_validator_rejects_rho_wrong_packed_length(dense_store_path):
    _build_fixture_rho(dense_store_path)
    root = open_store(dense_store_path).arrays(mode="a")
    del root["rho"]["rho"]
    root["rho"].create_dataset("rho", data=np.array([0.1, 0.2], dtype="float16"))

    result = validate_store(dense_store_path)

    assert not result.ok
    assert any("rho array has" in error for error in result.errors)


def test_validator_rejects_rho_finite_with_insufficient_support(dense_store_path):
    _build_fixture_rho(dense_store_path)
    root = open_store(dense_store_path).arrays(mode="a")
    root["rho"]["n_null"][:] = 0  # below min_nulls=1, but rho stays finite

    result = validate_store(dense_store_path)

    assert not result.ok
    assert any("rho finiteness is inconsistent" in error for error in result.errors)


def test_validator_rejects_rho_out_of_range(dense_store_path):
    _build_fixture_rho(dense_store_path)
    root = open_store(dense_store_path).arrays(mode="a")
    root["rho"]["rho"][:] = np.array([1.5], dtype="float16")

    result = validate_store(dense_store_path)

    assert not result.ok
    assert any("rho contains values outside" in error for error in result.errors)


def test_validator_rejects_rho_missing_provenance_attr(dense_store_path):
    _build_fixture_rho(dense_store_path)
    root = open_store(dense_store_path).arrays(mode="a")
    del root["rho"].attrs["window_bp"]

    result = validate_store(dense_store_path)

    assert not result.ok
    assert any("missing provenance attrs" in error for error in result.errors)


def test_validator_rejects_rho_variant_index_length_mismatch(dense_store_path):
    _build_fixture_rho(dense_store_path)
    root = open_store(dense_store_path).arrays(mode="a")
    root["rho"].attrs["n_variants_used"] = 999

    result = validate_store(dense_store_path)

    assert not result.ok
    assert any("rho variant_index has" in error for error in result.errors)
