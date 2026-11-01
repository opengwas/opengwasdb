from __future__ import annotations

from opengwasdb.model.analyses import (
    REQUIRED_COLUMNS,
    AnalysesTable,
    ColumnClass,
    analysis_label,
    classify_column,
    read_analyses,
    to_json_schema,
    validate_analyses,
    write_analyses,
)

_MINIMAL_ROW = {
    "analysis_id": "ieu-a-7",
    "stored_effect_scale": "log_or",
    "sample_size_kind": "case_control",
    "sample_size_scope": "analysis_level",
    "sample_size": "184305",
    "original_effect_scale": "log_or",
    "original_sd_method": "binary_trait",
    "ancestry_assignment_method": "source_trusted_no_af",
    "n_cases": "60801",
    "n_controls": "123504",
}


def _table(fieldnames: list[str], rows: list[dict[str, str]]) -> AnalysesTable:
    return AnalysesTable(fieldnames=tuple(fieldnames), rows=tuple(rows))


def test_round_trip_preserves_every_field(tmp_path):
    fieldnames = [*_MINIMAL_ROW.keys(), "source_file", "checksum"]
    row = {**_MINIMAL_ROW, "source_file": "/fixture/ieu-a-7.vcf.gz", "checksum": "abc123"}
    table = _table(fieldnames, [row])
    path = tmp_path / "analyses.tsv"

    write_analyses(path, table)
    reread = read_analyses(path)

    assert reread.fieldnames == table.fieldnames
    assert reread.rows == table.rows


def test_validate_accepts_a_conforming_row():
    table = _table(list(_MINIMAL_ROW.keys()), [dict(_MINIMAL_ROW)])

    assert validate_analyses(table) == []


def test_validate_rejects_out_of_vocabulary_value():
    row = {**_MINIMAL_ROW, "stored_effect_scale": "z_score"}
    table = _table(list(row.keys()), [row])

    errors = validate_analyses(table)

    assert len(errors) == 1
    assert "ieu-a-7" in errors[0]
    assert "stored_effect_scale" in errors[0]
    assert "z_score" in errors[0]


def test_validate_rejects_missing_required_column():
    fieldnames = [c for c in _MINIMAL_ROW if c != "sample_size_scope"]
    row = {k: v for k, v in _MINIMAL_ROW.items() if k != "sample_size_scope"}
    table = _table(fieldnames, [row])

    errors = validate_analyses(table)

    assert len(errors) == 1
    assert "sample_size_scope" in errors[0]


def test_validate_rejects_blank_value_in_a_present_required_column():
    row = {**_MINIMAL_ROW, "sample_size_scope": ""}
    table = _table(list(row.keys()), [row])

    errors = validate_analyses(table)

    assert len(errors) == 1
    assert "ieu-a-7" in errors[0]
    assert "sample_size_scope" in errors[0]


def test_validate_requires_case_control_counts_for_log_or():
    row = {k: v for k, v in _MINIMAL_ROW.items() if k not in ("n_cases", "n_controls")}
    table = _table(list(row.keys()), [row])

    errors = validate_analyses(table)

    assert len(errors) == 2
    assert any("no n_cases" in e for e in errors)
    assert any("no n_controls" in e for e in errors)


def test_validate_does_not_require_case_control_counts_for_sd():
    row = {**_MINIMAL_ROW, "stored_effect_scale": "sd", "original_sd_method": "source_provided"}
    row = {k: v for k, v in row.items() if k not in ("n_cases", "n_controls")}
    table = _table(list(row.keys()), [row])

    assert validate_analyses(table) == []


def test_extra_columns_pass_through_unchanged(tmp_path):
    fieldnames = [*_MINIMAL_ROW.keys(), "inclusion_reason", "an_unrecognised_future_column"]
    row = {
        **_MINIMAL_ROW,
        "inclusion_reason": "pilot batch",
        "an_unrecognised_future_column": "surprise",
    }
    table = _table(fieldnames, [row])
    path = tmp_path / "analyses.tsv"
    write_analyses(path, table)

    reread = read_analyses(path)

    assert validate_analyses(reread) == []
    assert reread.rows[0]["inclusion_reason"] == "pilot batch"
    assert reread.rows[0]["an_unrecognised_future_column"] == "surprise"


def test_analysis_label_prefers_the_row_label_over_its_id():
    row = {"analysis_id": "ieu-a-7", "analysis_label": "Body mass index"}

    assert analysis_label(row) == "Body mass index"


def test_analysis_label_falls_back_to_the_id_when_the_label_is_blank():
    # A blank label is "no label was supplied", not a label of ``""`` -- the
    # one display label `analyses_table()`/overview.html have always shown.
    assert analysis_label({"analysis_id": "a2", "analysis_label": ""}) == "a2"


def test_analysis_label_falls_back_to_the_id_when_the_column_is_absent():
    # A release whose analyses.tsv predates ADR 0034 carries no label column.
    assert analysis_label({"analysis_id": "a3"}) == "a3"


def test_classify_column_shared_core():
    assert classify_column("stored_effect_scale") is ColumnClass.SHARED_CORE
    assert classify_column("ancestry_prop_EUR") is ColumnClass.SHARED_CORE
    assert classify_column("ancestry_prop_AFR") is ColumnClass.SHARED_CORE


def test_classify_column_store_only():
    assert classify_column("completion_median_pearson_r") is ColumnClass.STORE_ONLY
    assert classify_column("completed_against") is ColumnClass.STORE_ONLY


def test_classify_column_registry_only():
    assert classify_column("checksum") is ColumnClass.REGISTRY_ONLY
    # A genuinely unknown column still classifies as registry-only, matching
    # the superset property: this module does not need to know every
    # registry build-input column name in advance.
    assert classify_column("some_future_registry_column") is ColumnClass.REGISTRY_ONLY


def test_to_json_schema_encodes_the_superset_property():
    schema = to_json_schema()

    assert schema["additionalProperties"] is True
    assert set(REQUIRED_COLUMNS) <= schema["properties"].keys()
    assert schema["required"] == list(REQUIRED_COLUMNS)
    assert "log_or" in schema["properties"]["stored_effect_scale"]["enum"]
    assert "sd" in schema["properties"]["stored_effect_scale"]["enum"]
