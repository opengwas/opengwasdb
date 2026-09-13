"""Unit tests for manifest column alias resolution (issue #170, #172, #177; ADR 0034)."""

from __future__ import annotations

import pytest

import opengwasdb.model.manifest_columns as mc


def test_manifest_column_prefers_canonical_over_legacy():
    # When canonical and legacy are both present, canonical always wins.
    fieldnames = ["analysis_id", "trait_id", "sample_size", "n", "source_file", "filtered_file"]
    assert mc.manifest_column(fieldnames, "analysis_id") == "analysis_id"
    assert mc.manifest_column(fieldnames, "sample_size") == "sample_size"
    assert mc.manifest_column(fieldnames, "source_file") == "source_file"
    assert mc.manifest_column(fieldnames, "analysis_label") is None


def test_manifest_column_resolves_legacy_aliases():
    fieldnames = ["trait_id", "n", "filtered_file", "trait_name"]
    assert mc.manifest_column(fieldnames, "analysis_id") == "trait_id"
    assert mc.manifest_column(fieldnames, "sample_size") == "n"
    assert mc.manifest_column(fieldnames, "source_file") == "filtered_file"
    assert mc.manifest_column(fieldnames, "analysis_label") == "trait_name"


def test_manifest_column_resolves_multiple_legacy_aliases_for_source_file():
    assert mc.manifest_column(["file_path"], "source_file") == "file_path"
    assert mc.manifest_column(["filtered_file"], "source_file") == "filtered_file"
    # Precedence order: source_file > file_path > filtered_file
    all_three = ["source_file", "file_path", "filtered_file"]
    assert mc.manifest_column(all_three, "source_file") == "source_file"
    assert mc.manifest_column(["filtered_file", "file_path"], "source_file") == "file_path"


def test_manifest_column_returns_none_when_absent():
    assert mc.manifest_column(["other_col", "another_col"], "analysis_id") is None
    assert mc.manifest_column([], "sample_size") is None


def test_required_manifest_column_returns_resolved():
    assert mc.required_manifest_column(["analysis_id"], "analysis_id") == "analysis_id"
    assert mc.required_manifest_column(["trait_id"], "analysis_id") == "trait_id"
    assert mc.required_manifest_column(["filtered_file"], "source_file") == "filtered_file"


def test_required_manifest_column_raises_with_informative_message():
    with pytest.raises(
        ValueError,
        match=r"manifest /tmp/test\.tsv is missing required column: 'analysis_id' "
        r"\(legacy name 'trait_id'\)",
    ):
        mc.required_manifest_column(["other"], "analysis_id", "/tmp/test.tsv")

    with pytest.raises(
        ValueError,
        match=r"manifest /tmp/test\.tsv is missing required column: 'source_file' "
        r"\(legacy aliases: 'file_path', 'filtered_file'\)",
    ):
        mc.required_manifest_column(["other"], "source_file", "/tmp/test.tsv")

    with pytest.raises(
        ValueError,
        match=r"missing required column: 'unaliased_col'",
    ):
        mc.required_manifest_column(["other"], "unaliased_col")


def test_require_columns_checks_presence():
    mc.require_columns(["a", "b", "c"], "/tmp/test.tsv", "a", "b")
    with pytest.raises(
        ValueError, match=r"manifest /tmp/test\.tsv is missing required column: 'missing_col'"
    ):
        mc.require_columns(["a", "b"], "/tmp/test.tsv", "a", "missing_col")


def test_manifest_trait_name():
    row = {"analysis_label": "Trait Label", "trait_name": "Legacy Name"}
    assert mc.manifest_trait_name(row, "analysis_label", "trait_1") == "Trait Label"
    assert mc.manifest_trait_name(row, "trait_name", "trait_1") == "Legacy Name"
    assert mc.manifest_trait_name(row, None, "trait_1") == "trait_1"
    # When label column is present but blank:
    # Dense default (fallback_on_blank=False) preserves blank per ADR 0034.
    assert mc.manifest_trait_name({"analysis_label": ""}, "analysis_label", "trait_1") == ""
    assert mc.manifest_trait_name(
        {"analysis_label": ""}, "analysis_label", "trait_1", fallback_on_blank=False
    ) == ""
    # Ancestry mode (fallback_on_blank=True) falls back to trait_id.
    assert mc.manifest_trait_name(
        {"analysis_label": ""}, "analysis_label", "trait_1", fallback_on_blank=True
    ) == "trait_1"


def test_manifest_n():
    row = {"sample_size": "1000", "n": "500", "empty": ""}
    assert mc.manifest_n(row, "sample_size") == 1000
    assert mc.manifest_n(row, "n") == 500
    assert mc.manifest_n(row, None) == 0
    assert mc.manifest_n(row, "empty") == 0
    assert mc.manifest_n({}, "sample_size") == 0


def test_resolve_manifest_columns():
    cols = mc.resolve_manifest_columns(["analysis_id", "source_file", "sample_size"])
    assert isinstance(cols, mc.ManifestColumns)
    assert cols.analysis_id == "analysis_id"
    assert cols.source_file == "source_file"
    assert cols.sample_size == "sample_size"
    assert cols.analysis_label is None

    legacy_cols = mc.resolve_manifest_columns(["trait_id", "filtered_file", "trait_name", "n"])
    assert legacy_cols.analysis_id == "trait_id"
    assert legacy_cols.source_file == "filtered_file"
    assert legacy_cols.analysis_label == "trait_name"
    assert legacy_cols.sample_size == "n"

