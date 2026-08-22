"""Shared Analysis Metadata model + analyses.tsv writer/reader (issue #67).

Exercises `opengwasdb.model.analyses.Analysis`/`analyses_table_from_records`/
`write_analysis_records`/`read_analysis_records` by building a table
directly -- no layout builder depends on this model yet (that migration is
issue #68 onward), so these tests never go through a builder entrypoint.
"""

from __future__ import annotations

import dataclasses

import pytest

from opengwasdb.model.analyses import (
    ANALYSIS_COLUMNS,
    RETIRED_ANALYSIS_COLUMNS,
    Analysis,
    analyses_table_from_records,
    read_analyses,
    read_analysis_records,
    write_analysis_records,
)

_RETIRED_COLUMNS = set(RETIRED_ANALYSIS_COLUMNS)

_FULL_ANALYSIS = Analysis(
    analysis_id="ieu-a-7",
    analysis_label="Body mass index",
    trait_ontology_id="EFO:0004340",
    trait_ontology_label="body mass index",
    tissue="whole_blood",
    context="cis",
    trait_chr="7",
    trait_bp="127881331",
    stored_effect_scale="sd",
    original_effect_scale="sd",
    original_sd="1.0",
    original_sd_method="source_provided",
    original_sd_dispersion="0.02",
    assigned_ancestry="EUR",
    ancestry_assignment_method="source_trusted_no_af",
    ancestry_prop={"EUR": "0.98", "AFR": "0.02"},
    sample_size_kind="cohort",
    sample_size_scope="analysis_level",
    sample_size="184305",
    n_cases="60801",
    n_controls="123504",
    eaf_scope="association",
    license="CC0",
    publication_doi="10.1000/example",
    publication_pmid="12345678",
    consortium="GIANT",
    first_author="Yengo L",
    completed_against="1kg_eur",
    completion_median_pearson_r="0.91",
    completion_n_imputed_total="1200",
    completion_n_missing_total="3",
    eaf_orientation="passed",
    eaf_orientation_r="0.999200",
    eaf_orientation_n="18422",
    n_hits_5e8="42",
    n_hits_5e6="120",
    n_hits_5e4="900",
)

_MINIMAL_ANALYSIS = Analysis(analysis_id="ieu-a-8")


def test_analysis_model_has_no_retired_identifier_fields():
    fields = {f for f in _FULL_ANALYSIS.__dataclass_fields__}
    assert fields.isdisjoint(_RETIRED_COLUMNS)


def test_analysis_columns_match_the_adr_0035_order():
    # `eaf_scope` joined the list in ADR 0036, between the sample-size
    # interpretation columns and the effect-scale ones it sits alongside.
    # `eaf_orientation*` joined it for issue #115, alongside the other
    # build-time evidence columns a store carries and a manifest cannot.
    assert ANALYSIS_COLUMNS == (
        "analysis_index",
        "analysis_id",
        "analysis_label",
        "trait_ontology_id",
        "trait_ontology_label",
        "tissue",
        "context",
        "trait_chr",
        "trait_bp",
        "stored_effect_scale",
        "assigned_ancestry",
        "ancestry_assignment_method",
        "sample_size_kind",
        "sample_size_scope",
        "sample_size",
        "n_cases",
        "n_controls",
        "eaf_scope",
        "original_effect_scale",
        "original_sd",
        "original_sd_method",
        "original_sd_dispersion",
        "license",
        "publication_doi",
        "publication_pmid",
        "consortium",
        "first_author",
        "completed_against",
        "completion_median_pearson_r",
        "completion_n_imputed_total",
        "completion_n_missing_total",
        "eaf_orientation",
        "eaf_orientation_r",
        "eaf_orientation_n",
        "n_hits_5e8",
        "n_hits_5e6",
        "n_hits_5e4",
    )
    assert _RETIRED_COLUMNS.isdisjoint(ANALYSIS_COLUMNS)


def test_minimal_analysis_is_representable_with_every_optional_field_blank():
    table = analyses_table_from_records([_MINIMAL_ANALYSIS])
    row = table.rows[0]

    assert row["analysis_id"] == "ieu-a-8"
    for column in ANALYSIS_COLUMNS:
        if column in ("analysis_index", "analysis_id"):
            continue
        assert row[column] == "", f"{column} was fabricated: {row[column]!r}"


def test_the_round_trip_fixture_populates_every_column():
    """Guard on the round-trip tests below, which are only as complete as this
    fixture is.

    `_analysis_row`/`_analysis_from_row` map each column by hand, so a field
    added to `Analysis` and to `ANALYSIS_COLUMNS` but forgotten in the mappers
    writes a permanently blank column -- and a round-trip test whose fixture
    leaves that field blank passes anyway, proving nothing. This asserts the
    fixture is meaningful before anything is asserted about it.
    """
    blank = [
        name
        for name, value in dataclasses.asdict(_FULL_ANALYSIS).items()
        if name != "analysis_index" and not value
    ]
    assert blank == [], f"_FULL_ANALYSIS leaves {blank} blank, so the round trip skips them"


def test_write_then_read_round_trips_every_populated_column(tmp_path):
    path = tmp_path / "analyses.tsv"

    write_analysis_records(path, [_FULL_ANALYSIS])
    reread = read_analysis_records(path)

    # analysis_index isn't a field the caller sets on _FULL_ANALYSIS -- the
    # writer assigns it from list position (index 0 here), and read
    # populates it back from the row, so the only expected diff is that.
    assert reread == [dataclasses.replace(_FULL_ANALYSIS, analysis_index="0")]


def test_write_then_read_round_trips_every_blank_column(tmp_path):
    path = tmp_path / "analyses.tsv"

    write_analysis_records(path, [_MINIMAL_ANALYSIS])
    reread = read_analysis_records(path)

    assert reread == [dataclasses.replace(_MINIMAL_ANALYSIS, analysis_index="0")]


def test_ancestry_prop_reads_back_blank_for_a_population_the_row_has_none_for(tmp_path):
    # Sharing a table with _FULL_ANALYSIS gives every row an ancestry_prop_*
    # column per population seen anywhere in the table; a row with no
    # proportion for a given population reads back "" for it -- the TSV
    # itself cannot distinguish "no proportion for this population" from "no
    # ancestry_prop entry at all", so this is the honest round trip, not
    # equality with the original in-memory Analysis.
    path = tmp_path / "analyses.tsv"

    write_analysis_records(path, [_FULL_ANALYSIS, _MINIMAL_ANALYSIS])
    reread = read_analysis_records(path)

    assert reread[1].ancestry_prop == {"AFR": "", "EUR": ""}
    assert reread[0].ancestry_prop == _FULL_ANALYSIS.ancestry_prop


def test_round_trip_assigns_analysis_index_by_position(tmp_path):
    path = tmp_path / "analyses.tsv"

    write_analysis_records(path, [_FULL_ANALYSIS, _MINIMAL_ANALYSIS])
    table = read_analyses(path)

    assert table.rows[0]["analysis_index"] == "0"
    assert table.rows[0]["analysis_id"] == _FULL_ANALYSIS.analysis_id
    assert table.rows[1]["analysis_index"] == "1"
    assert table.rows[1]["analysis_id"] == _MINIMAL_ANALYSIS.analysis_id


def test_ancestry_prop_splices_sorted_population_columns():
    table = analyses_table_from_records([_FULL_ANALYSIS])

    assert "ancestry_prop_AFR" in table.fieldnames
    assert "ancestry_prop_EUR" in table.fieldnames
    prop_columns = [c for c in table.fieldnames if c.startswith("ancestry_prop_")]
    assert prop_columns == sorted(prop_columns)
    assert table.rows[0]["ancestry_prop_EUR"] == "0.98"
    assert table.rows[0]["ancestry_prop_AFR"] == "0.02"


def test_ancestry_prop_column_blank_for_analyses_without_that_population():
    table = analyses_table_from_records([_FULL_ANALYSIS, _MINIMAL_ANALYSIS])

    minimal_row = table.rows[1]
    assert minimal_row["ancestry_prop_EUR"] == ""
    assert minimal_row["ancestry_prop_AFR"] == ""


def test_duplicate_analysis_id_is_rejected_by_the_writer():
    duplicate = Analysis(analysis_id=_MINIMAL_ANALYSIS.analysis_id, analysis_label="different row")

    with pytest.raises(ValueError, match=_MINIMAL_ANALYSIS.analysis_id):
        analyses_table_from_records([_MINIMAL_ANALYSIS, duplicate])


def test_duplicate_analysis_id_is_rejected_when_writing_to_disk(tmp_path):
    duplicate = Analysis(analysis_id=_MINIMAL_ANALYSIS.analysis_id, analysis_label="different row")

    with pytest.raises(ValueError, match=_MINIMAL_ANALYSIS.analysis_id):
        write_analysis_records(tmp_path / "analyses.tsv", [_MINIMAL_ANALYSIS, duplicate])


def test_blank_analysis_id_is_rejected():
    with pytest.raises(ValueError, match="analysis_id"):
        Analysis(analysis_id="")


def test_read_analysis_records_populates_analysis_index_field(tmp_path):
    path = tmp_path / "analyses.tsv"

    write_analysis_records(path, [_FULL_ANALYSIS, _MINIMAL_ANALYSIS])
    reread = read_analysis_records(path)

    assert reread[0].analysis_index == "0"
    assert reread[1].analysis_index == "1"


def test_writer_ignores_a_caller_supplied_analysis_index():
    # analysis_index must equal list position (it's also the column index a
    # Dense/Hybrid store's z/se arrays are keyed by), so the writer is
    # authoritative regardless of what a caller passes in.
    stale = Analysis(analysis_id="ieu-a-9", analysis_index="99")

    table = analyses_table_from_records([stale])

    assert table.rows[0]["analysis_index"] == "0"
