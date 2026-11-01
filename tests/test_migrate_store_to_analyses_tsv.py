"""Tests for scripts/migrate_store_to_analyses_tsv.py (issue #24).

There is no old-layout fixture builder left in the codebase (issue #22 removed
the SQLite `analyses` table and the `ancestry.tsv` sidecar from every build
path) -- so these tests build a store the current way, then hand-revert it to
the pre-#22 layout the migration script targets.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

from opengwasdb.model.analyses import read_analyses
from opengwasdb.validation import validate_store

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "migrate_store_to_analyses_tsv.py"
)
_spec = importlib.util.spec_from_file_location("migrate_store_to_analyses_tsv", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
migrate_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(migrate_module)

_OLD_SIDECAR_FIELDS = ["trait_id", "assigned_ancestry", "completed_against"]


def _stamp_as_written_before_338(store_path: Path) -> None:
    """A release with an old `analyses.tsv` layout predates every later column, so it is
    stamped as the last release written before `n_missing_se_out_of_range` became required
    (#338): the migration cannot know that count and does not write it."""
    manifest_path = store_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format_version"] = "0.2.0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _revert_to_old_layout(store_path: Path, ancestry: dict[str, str] | None = None) -> None:
    """Convert a freshly-built (post-#22) store back into the pre-#22 layout:
    a SQLite `analyses` table instead of `analyses.tsv`, and (optionally) an
    `ancestry.tsv` + `ancestry_provenance.json` sidecar pair."""
    table = read_analyses(store_path / "analyses.tsv")
    (store_path / "analyses.tsv").unlink()
    (store_path / "overview.html").unlink()
    _stamp_as_written_before_338(store_path)

    connection = sqlite3.connect(store_path / "index.sqlite")
    try:
        connection.execute(
            """
            CREATE TABLE analyses (
                analysis_index INTEGER PRIMARY KEY,
                analysis_id TEXT NOT NULL UNIQUE,
                phenotype_id TEXT,
                phenotype_label TEXT,
                analysis_label TEXT,
                stored_effect_scale TEXT NOT NULL
            )
            """
        )
        # phenotype_id/phenotype_label aren't columns of a current build's
        # analyses.tsv (retired, ADR 0034) -- synthesise plausible pre-#22
        # values here rather than sourcing them from `table.rows`, since the
        # point of this fixture is a historical layout the current builder no
        # longer produces.
        connection.executemany(
            "INSERT INTO analyses(analysis_index, analysis_id, phenotype_id, "
            "phenotype_label, analysis_label, stored_effect_scale) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    int(row["analysis_index"]), row["analysis_id"],
                    f"p_{row['analysis_id']}", row["analysis_label"] or None,
                    row["analysis_label"] or None, row["stored_effect_scale"],
                )
                for row in table.rows
            ],
        )
        connection.commit()
    finally:
        connection.close()

    if ancestry:
        sidecar = store_path / "ancestry.tsv"
        with open(sidecar, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_OLD_SIDECAR_FIELDS, delimiter="\t")
            writer.writeheader()
            for analysis_id, assigned in ancestry.items():
                writer.writerow(
                    {
                        "trait_id": analysis_id,
                        "assigned_ancestry": assigned,
                        "completed_against": "",
                    }
                )
        (store_path / "ancestry_provenance.json").write_text(
            json.dumps(
                {
                    "catalogue_version": "cat-v1",
                    "subset_filter": "assigned_ancestry == EUR",
                    "ancestry_reference_version": "prive2022-hg38",
                    "n_analyses": len(ancestry),
                },
                indent=2, sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def _sqlite_tables(store_path: Path) -> set[str]:
    connection = sqlite3.connect(store_path / "index.sqlite")
    try:
        return {
            r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        connection.close()


def test_migrate_preserves_analyses_and_ancestry(dense_store_path):
    _revert_to_old_layout(dense_store_path, ancestry={"a1": "EUR", "a2": "AFR"})

    before = migrate_module.read_old_layout_analysis_ids(dense_store_path)
    after = migrate_module.migrate_store(dense_store_path)
    assert before == after == ["a1", "a2"]

    table = read_analyses(dense_store_path / "analyses.tsv")
    rows = {r["analysis_id"]: r for r in table.rows}
    assert rows["a1"]["assigned_ancestry"] == "EUR"
    assert rows["a2"]["assigned_ancestry"] == "AFR"
    assert rows["a1"]["ancestry_assignment_method"] == "af_assigned"
    # phenotype_id/phenotype_label are retired (ADR 0034) -- migrate_store no
    # longer writes them, and the old layout's analysis_label carries through.
    assert "phenotype_id" not in table.fieldnames
    assert "phenotype_label" not in table.fieldnames
    assert rows["a1"]["analysis_label"] == "Height primary"

    # Top-Hit Counts (ADR 0032) are recovered for real from the pre-existing
    # top-hit index (fixture z-values: a1=2.0,-3.0 -- no hits at any
    # threshold; a2=6.0,6.0 -- hits at all three), not left blank.
    for column in ("n_hits_5e8", "n_hits_5e6", "n_hits_5e4"):
        assert rows["a1"][column] == "0"
        assert rows["a2"][column] == "2"

    assert (dense_store_path / "overview.html").exists()
    assert not (dense_store_path / "ancestry.tsv").exists()
    assert not (dense_store_path / "ancestry_provenance.json").exists()
    assert "analyses" not in _sqlite_tables(dense_store_path)

    manifest = json.loads((dense_store_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["provenance"]["ancestry"]["catalogue_version"] == "cat-v1"
    assert manifest["provenance"]["ancestry"]["n_analyses"] == 2


def test_migrate_translates_legacy_sd_units_to_current_vocabulary(dense_store_path):
    # Pre-issue-#16 stores (and every real UKB-B production store) were built
    # when StoredEffectScale's SD member was still spelled "sd_units"; issue
    # #16 renamed it to "sd" without a migration path. A store carrying the
    # old spelling must not fail validation after migration -- the value is a
    # known historical rename, not something migrate_store has to guess.
    _revert_to_old_layout(dense_store_path)
    connection = sqlite3.connect(dense_store_path / "index.sqlite")
    try:
        connection.execute(
            "UPDATE analyses SET stored_effect_scale = 'sd_units' WHERE analysis_id = 'a1'"
        )
        connection.commit()
    finally:
        connection.close()

    migrate_module.migrate_store(dense_store_path)

    rows = {r["analysis_id"]: r for r in read_analyses(dense_store_path / "analyses.tsv").rows}
    assert rows["a1"]["stored_effect_scale"] == "sd"
    assert rows["a2"]["stored_effect_scale"] == "log_or"  # untouched -- already current vocabulary

    result = validate_store(dense_store_path)
    assert result.ok, result.errors


def test_migrate_writes_unavailable_sd_provenance_not_a_guess(dense_store_path):
    _revert_to_old_layout(dense_store_path)
    migrate_module.migrate_store(dense_store_path)

    table = read_analyses(dense_store_path / "analyses.tsv")
    assert table.rows
    for row in table.rows:
        assert row["original_sd_method"] == "unavailable"
        assert row["assigned_ancestry"] == ""
        assert row["ancestry_assignment_method"] == ""


def test_migrated_store_passes_validation(dense_store_path):
    _revert_to_old_layout(dense_store_path, ancestry={"a1": "EUR"})
    migrate_module.migrate_store(dense_store_path)

    result = validate_store(dense_store_path)
    assert result.ok, result.errors


def test_migrate_raises_when_already_migrated(dense_store_path):
    _revert_to_old_layout(dense_store_path)
    migrate_module.migrate_store(dense_store_path)

    with pytest.raises(ValueError, match="no analyses table"):
        migrate_module.migrate_store(dense_store_path)


def _revert_to_pre_adr0034_tsv(store_path: Path) -> None:
    """Rewrite a freshly-built (post-ADR-0034) store's `analyses.tsv` back
    into the shape `write_analyses_tsv` produced before the unified
    `Analysis` model landed (issue #68): `phenotype_id`/`phenotype_label`
    instead of `trait_ontology_id`/`trait_ontology_label`, and
    `analysis_label` defaulted to `analysis_id` rather than genuinely
    resolved -- matching every real store built in that window (e.g. the
    2026-08-05 `ukb-b-c128-completed` production store)."""
    _stamp_as_written_before_338(store_path)
    table = read_analyses(store_path / "analyses.tsv")
    old_fields = [
        "analysis_index", "analysis_id", "phenotype_id", "phenotype_label", "analysis_label",
        "stored_effect_scale", "assigned_ancestry", "ancestry_assignment_method",
        "sample_size_kind", "sample_size_scope", "sample_size", "n_cases", "n_controls",
        "original_effect_scale", "original_sd", "original_sd_method", "original_sd_dispersion",
        "completed_against", "completion_median_pearson_r", "completion_n_imputed_total",
        "completion_n_missing_total", "n_hits_5e8", "n_hits_5e6", "n_hits_5e4",
    ]
    old_rows = [
        {
            "analysis_index": row["analysis_index"],
            "analysis_id": row["analysis_id"],
            "phenotype_id": row["analysis_id"],
            "phenotype_label": row["analysis_label"],
            "analysis_label": row["analysis_id"],
            **{f: row.get(f, "") for f in old_fields if f not in
               {"analysis_index", "analysis_id", "phenotype_id", "phenotype_label", "analysis_label"}},
        }
        for row in table.rows
    ]
    with open(store_path / "analyses.tsv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=old_fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(old_rows)


def test_migrate_analyses_tsv_in_place_recovers_label_and_drops_retired_columns(dense_store_path):
    _revert_to_pre_adr0034_tsv(dense_store_path)
    assert migrate_module.has_pre_adr0034_analyses_tsv(dense_store_path)

    before = [r["analysis_id"] for r in read_analyses(dense_store_path / "analyses.tsv").rows]
    after = migrate_module.migrate_analyses_tsv_in_place(dense_store_path)
    assert before == after == ["a1", "a2"]

    table = read_analyses(dense_store_path / "analyses.tsv")
    assert "phenotype_id" not in table.fieldnames
    assert "phenotype_label" not in table.fieldnames
    assert "trait_ontology_id" in table.fieldnames

    rows = {r["analysis_id"]: r for r in table.rows}
    # phenotype_label carried the real description; the pre-migration
    # analysis_label was just analysis_id -- recovering from phenotype_label
    # here is the whole point of this migration path.
    assert rows["a1"]["analysis_label"] == "Height primary"

    result = validate_store(dense_store_path)
    assert result.ok, result.errors


def test_migrate_analyses_tsv_in_place_preserves_completion_and_hit_counts(dense_store_path):
    _revert_to_pre_adr0034_tsv(dense_store_path)
    before_rows = {
        r["analysis_id"]: r for r in read_analyses(dense_store_path / "analyses.tsv").rows
    }

    migrate_module.migrate_analyses_tsv_in_place(dense_store_path)

    after_rows = {r["analysis_id"]: r for r in read_analyses(dense_store_path / "analyses.tsv").rows}
    for analysis_id, before in before_rows.items():
        after = after_rows[analysis_id]
        for column in ("completed_against", "completion_median_pearson_r",
                       "completion_n_imputed_total", "completion_n_missing_total",
                       "n_hits_5e8", "n_hits_5e6", "n_hits_5e4"):
            assert after[column] == before[column]


def test_migrate_analyses_tsv_in_place_raises_when_already_migrated(dense_store_path):
    with pytest.raises(ValueError, match="already migrated"):
        migrate_module.migrate_analyses_tsv_in_place(dense_store_path)
