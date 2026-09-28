"""Provider-declared INFO policy at the public analyses.tsv and resolver seams (#175)."""

from __future__ import annotations

import pytest

from opengwasdb.ancestry.mixture import Gates
from opengwasdb.build.resolve_manifest import _build_analysis_fingerprints, read_resolve_manifest
from opengwasdb.model.analyses import read_analyses, validate_analyses
from opengwasdb.model.info_score_policy import parse_info_score_policy
from opengwasdb.readers.gwas_ssf import GwasSsfReader
from opengwasdb.readers.interface import ImputationScoreKind

BASE = {
    "analysis_id": "GCST1",
    "source_file": "example.tsv.gz",
    "source_reader_capability": "opengwasdb.gwas-ssf",
    "stored_effect_scale": "sd",
    "sample_size_kind": "total",
    "sample_size_scope": "analysis_level",
    "sample_size": "1000",
    "original_effect_scale": "sd",
    "original_sd_method": "source_provided",
    "ancestry_assignment_method": "source_trusted_no_af",
}
DECLARATION = {
    "imputation_score_column": "quality_metric",
    "imputation_score_kind": "imputation_info",
    "imputation_score_provenance": (
        "Provider documentation: Table 2 reports per-variant imputation INFO"
    ),
}


def _manifest(tmp_path, **updates):
    row = {**BASE, **updates}
    path = tmp_path / "analyses.tsv"
    path.write_text("\t".join(row) + "\n" + "\t".join(row.values()) + "\n")
    return path


def test_public_manifest_roundtrip_validates_declared_policy(tmp_path):
    path = _manifest(tmp_path, info_score_threshold="0.8", **DECLARATION)
    table = read_analyses(path)
    assert len(table.rows) == 1
    assert validate_analyses(table) == []
    policy = read_resolve_manifest(path)[0].info_score_policy
    assert policy.info_score_threshold == 0.8
    assert policy.imputation_score_declaration is not None
    assert policy.imputation_score_declaration.column_name == "quality_metric"
    assert policy.imputation_score_declaration.kind is ImputationScoreKind.IMPUTATION_INFO
    assert (
        policy.imputation_score_declaration.provenance
        == DECLARATION["imputation_score_provenance"]
    )


@pytest.mark.parametrize("updates", [
    {},
    {"info_score_threshold": "NaN"},
    {"info_score_threshold": "0", **DECLARATION},
    {"info_score_threshold": "1", **{**DECLARATION, "imputation_score_kind": "imputation_r2"}},
])
def test_legacy_and_unavailable_and_boundaries_are_parsed(tmp_path, updates):
    path = _manifest(tmp_path, **updates)
    assert validate_analyses(read_analyses(path)) == []
    policy = read_resolve_manifest(path)[0].info_score_policy
    unavailable = updates.get("info_score_threshold") in (None, "NaN")
    assert (policy.info_score_threshold is None) == unavailable
    assert (policy.imputation_score_declaration is None) == (
        "imputation_score_column" not in updates
    )


@pytest.mark.parametrize("updates,fragment", [
    ({"info_score_threshold": "0.7"}, "imputation_score_column"),
    ({"info_score_threshold": "0.7", "imputation_score_column": "INFO"}, "imputation_score_kind"),
    ({"info_score_threshold": "NaN", **DECLARATION}, "NaN"),
    ({**DECLARATION}, "info_score_threshold"),
    ({"info_score_threshold": ""}, "info_score_threshold"),
    ({"info_score_threshold": "NA"}, "info_score_threshold"),
    ({"info_score_threshold": "inf"}, "info_score_threshold"),
    ({"info_score_threshold": "-0.1", **DECLARATION}, "info_score_threshold"),
    ({"info_score_threshold": "1.01", **DECLARATION}, "info_score_threshold"),
    (
        {"info_score_threshold": "0.5", **DECLARATION, "imputation_score_kind": "frequency"},
        "imputation_score_kind",
    ),
    (
        {"info_score_threshold": "0.5", **DECLARATION, "imputation_score_provenance": ""},
        "imputation_score_provenance",
    ),
    (
        {"info_score_threshold": "0.5", **DECLARATION,
         "source_reader_capability": "opengwasdb.gwas-vcf"},
        "source_reader_capability",
    ),
])
def test_public_and_resolver_reject_unsafe_manifest(tmp_path, updates, fragment):
    path = _manifest(tmp_path, **updates)
    errors = validate_analyses(read_analyses(path))
    assert errors and fragment in errors[0]
    with pytest.raises(ValueError, match=fragment):
        read_resolve_manifest(path)


def test_fingerprint_records_policy_and_invalidates_resume(tmp_path):
    old = read_resolve_manifest(_manifest(tmp_path))[0]
    new = read_resolve_manifest(
        _manifest(tmp_path, info_score_threshold="0.8", **DECLARATION)
    )[0]
    kwargs = dict(
        opengwasdb_version="test", opengwasdb_git_hash="test",
        ancestry_reference_id="ref", ancestry_reference_sha256="ref-sha",
        ancestry_groups_sha256="groups-sha", extraction_panel_sha256=None,
        extraction_panel_variants=None, variant_reference_sha256=None,
        af_references_fp=[], gates=Gates(), maf_floor=0.01,
        evidence_sample=100, scan_limit=None,
    )
    previous = _build_analysis_fingerprints(old, **kwargs)
    current = _build_analysis_fingerprints(new, **kwargs)
    assert previous["fingerprint_digest"] != current["fingerprint_digest"]
    config = current["resolution_config"]
    assert config["info_score_threshold"] == 0.8
    assert config["imputation_score_column"] == "quality_metric"
    assert config["imputation_score_kind"] == "imputation_info"
    assert config["imputation_score_provenance"] == DECLARATION["imputation_score_provenance"]
    for change in (
        {"info_score_threshold": "0.7"},
        {"imputation_score_column": "another_column"},
        {"imputation_score_kind": "imputation_r2"},
        {"imputation_score_provenance": "Different provider evidence"},
    ):
        changed = read_resolve_manifest(_manifest(
            tmp_path, **{"info_score_threshold": "0.8", **DECLARATION, **change}
        ))[0]
        assert _build_analysis_fingerprints(changed, **kwargs)["fingerprint_digest"] != (
            current["fingerprint_digest"]
        )


def test_policy_never_uses_header_or_provenance_from_column_name():
    assert parse_info_score_policy({**BASE, "INFO": "0.9"}).imputation_score_declaration is None
    with pytest.raises(ValueError, match="imputation_score_provenance"):
        parse_info_score_policy({
            **BASE, "info_score_threshold": "0.5", **DECLARATION,
            "imputation_score_provenance": "quality_metric",
        })


def test_declared_ambiguous_source_header_is_rejected(tmp_path):
    source = tmp_path / "duplicate.tsv"
    source.write_text("quality_metric\tquality_metric\n0.5\t0.8\n")
    policy = parse_info_score_policy({**BASE, "info_score_threshold": "0.5", **DECLARATION})
    assert policy.imputation_score_declaration is not None
    reader = GwasSsfReader(source, imputation_score_declaration=policy.imputation_score_declaration)
    with pytest.raises(ValueError, match="must occur exactly once"):
        _ = reader.imputation_score_column


def test_duplicate_manifest_policy_column_is_ambiguous(tmp_path):
    path = _manifest(tmp_path, info_score_threshold="0.8", **DECLARATION)
    text = path.read_text()
    path.write_text(
        text.replace("info_score_threshold", "info_score_threshold\tinfo_score_threshold")
    )
    errors = validate_analyses(read_analyses(path))
    assert any("ambiguous INFO policy column" in error for error in errors)
    with pytest.raises(ValueError, match="ambiguous INFO policy column"):
        read_resolve_manifest(path)
