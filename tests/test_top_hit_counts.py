"""`n_hits_*` against the tiers: the rule `analyses.top_hit_counts` and the repair
`opengwasdb recount-top-hits` (issue #509).

The fixtures are the six `digest_releases` (Dense, Ragged and Hybrid, each Observed-Only
and Reference-Completed; the completed Hybrid folds a panel crossover). A stale count is
planted by rewriting one Analysis's `n_hits_5e8` in place, as a release published before
#509's fix holds one.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from format_0_2_0 import as_0_2_0
from format_0_2_3 import as_0_2_3
from typer.testing import CliRunner

from opengwasdb.cli.main import app
from opengwasdb.model.analyses import TOP_HIT_COUNT_COLUMNS, write_analysis_columns
from opengwasdb.store.top_hit_counts import RecountRefusal, recount_top_hits
from opengwasdb.store_reader.analyses import read_analyses
from opengwasdb.validation import validate_store
from opengwasdb.validation.rule import Severity

RULE = "analyses.top_hit_counts"
LAYOUTS = ["dense_observed", "dense_completed", "ragged_observed", "ragged_completed",
           "hybrid_observed", "hybrid_completed"]


def _copy(digest_releases: dict[str, Path], name: str, tmp_path: Path) -> Path:
    return Path(shutil.copytree(digest_releases[name], tmp_path / name))


def _counts(table: Path, column: str = "n_hits_5e8") -> list[str]:
    return [row[column] for row in read_analyses(table).rows]


def _plant_stale(table: Path, column: str = "n_hits_5e8") -> str:
    """Add one to the first Analysis's `column`; return its `analysis_id`."""
    rows = read_analyses(table).rows
    values = [row[column] for row in rows]
    values[0] = str(int(values[0]) + 1)
    write_analysis_columns(table, {column: values})
    return rows[0]["analysis_id"]


def _rule_findings(release: Path) -> list:
    return [f for f in validate_store(release, rules={RULE}).findings if f.rule == RULE]


@pytest.mark.parametrize("name", LAYOUTS)
def test_every_layout_counts_its_tiers(digest_releases, name):
    release = digest_releases[name]
    counted = [int(n) for c in TOP_HIT_COUNT_COLUMNS for n in _counts(release / "analyses.tsv", c)]
    assert any(counted), "the fixture holds top hits to count"

    assert _rule_findings(release) == []


@pytest.mark.parametrize("name", ["dense_observed", "ragged_completed", "hybrid_completed"])
def test_a_stale_count_fails_the_rule(digest_releases, tmp_path, name):
    release = _copy(digest_releases, name, tmp_path)
    analysis = _plant_stale(release / "analyses.tsv")

    (finding,) = _rule_findings(release)

    assert finding.severity is Severity.ERROR
    assert "n_hits_5e8 disagrees with its tier on 1 of" in finding.message
    assert repr(analysis) in finding.message and "recount-top-hits" in finding.message


def test_a_stale_count_warns_on_a_release_before_0_2_2(digest_releases, tmp_path):
    """A release no current builder writes is warned, not failed (`severity_owed`)."""
    release = as_0_2_0(as_0_2_3(_copy(digest_releases, "dense_observed", tmp_path)))
    assert _rule_findings(release) == [], "the 0.2.0 form counts its tiers"
    _plant_stale(release / "analyses.tsv")

    (finding,) = _rule_findings(release)

    assert finding.severity is Severity.WARNING


def test_a_0_2_3_hybrids_dense_component_table_is_held_to_its_own_tiers(
    digest_releases, tmp_path
):
    release = as_0_2_3(_copy(digest_releases, "hybrid_completed", tmp_path))
    assert (release / "dense" / "analyses.tsv").is_file(), "a 0.2.3 Dense Component's own"
    assert _rule_findings(release) == []
    _plant_stale(release / "dense" / "analyses.tsv")

    (finding,) = _rule_findings(release)

    assert finding.target == "dense"


@pytest.mark.parametrize("name", ["dense_observed", "ragged_completed", "hybrid_completed"])
def test_the_command_repairs_a_stale_count_and_nothing_else(digest_releases, tmp_path, name):
    release = _copy(digest_releases, name, tmp_path)
    table = release / "analyses.tsv"
    before = read_analyses(table)
    _plant_stale(table)
    assert _rule_findings(release), "the planted count is stale"

    run = recount_top_hits(release)

    (recounted,) = run.tables
    assert recounted.changed == {"n_hits_5e8": 1, "n_hits_5e6": 0, "n_hits_5e4": 0}
    assert recounted.n_analyses_changed == 1
    assert read_analyses(table) == before, "every other column and value as read"
    assert _rule_findings(release) == []
    again = recount_top_hits(release)
    assert again.as_dict()["written"] == [], "a right release is left untouched"


def test_the_command_recounts_a_0_2_3_hybrids_dense_table_then_its_root(
    digest_releases, tmp_path
):
    release = as_0_2_3(_copy(digest_releases, "hybrid_completed", tmp_path))
    _plant_stale(release / "dense" / "analyses.tsv")
    _plant_stale(release / "analyses.tsv", "n_hits_5e4")

    run = recount_top_hits(release)

    assert [t.path for t in run.tables] == [
        release / "dense" / "analyses.tsv", release / "analyses.tsv",
    ]
    assert [t.changed for t in run.tables] == [
        {"n_hits_5e8": 1, "n_hits_5e6": 0, "n_hits_5e4": 0},
        {"n_hits_5e8": 0, "n_hits_5e6": 0, "n_hits_5e4": 1},
    ]
    assert _rule_findings(release) == []


def test_the_command_regenerates_the_overview(digest_releases, tmp_path):
    """The overview renders every column of `analyses.tsv`: after the recount it is the
    page `regenerate-overview` writes from the repaired table, not the stale one."""
    from opengwasdb.overview import write_overview_html

    release = _copy(digest_releases, "dense_observed", tmp_path)
    overview = release / "overview.html"
    assert overview.is_file(), "the fixture carries an overview"
    _plant_stale(release / "analyses.tsv")
    write_overview_html(release, read_analyses(release / "analyses.tsv"))
    stale = overview.read_text(encoding="utf-8")

    recount_top_hits(release)
    recounted = overview.read_text(encoding="utf-8")

    assert recounted != stale
    write_overview_html(release, read_analyses(release / "analyses.tsv"))
    assert overview.read_text(encoding="utf-8") == recounted


def test_a_release_without_a_top_hit_index_is_refused_and_untouched(digest_releases, tmp_path):
    release = _copy(digest_releases, "dense_observed", tmp_path)
    _plant_stale(release / "analyses.tsv")
    held = (release / "analyses.tsv").read_bytes()
    shutil.rmtree(release / "data.zarr" / "top_hits")

    with pytest.raises(RecountRefusal, match="has no Top-Hit Index"):
        recount_top_hits(release)
    assert (release / "analyses.tsv").read_bytes() == held


def test_a_row_out_of_analysis_index_order_is_refused(digest_releases, tmp_path):
    release = _copy(digest_releases, "dense_observed", tmp_path)
    table = release / "analyses.tsv"
    indexes = [row["analysis_index"] for row in read_analyses(table).rows]
    assert len(indexes) >= 2, "the fixture has two Analyses to swap"
    write_analysis_columns(table, {"analysis_index": [indexes[1], indexes[0], *indexes[2:]]})
    held = table.read_bytes()

    with pytest.raises(RecountRefusal, match="analysis_index"):
        recount_top_hits(release)
    assert table.read_bytes() == held


def test_the_cli_prints_what_changed_and_refuses_with_exit_1(digest_releases, tmp_path):
    release = _copy(digest_releases, "hybrid_completed", tmp_path)
    _plant_stale(release / "analyses.tsv")
    runner = CliRunner()

    done = runner.invoke(app, ["recount-top-hits", str(release)])

    assert done.exit_code == 0, done.output
    payload = json.loads(done.output)
    assert payload["tables"][0]["changed"]["n_hits_5e8"] == 1
    assert payload["written"] == [str(release / "analyses.tsv")]
    shutil.rmtree(release / "data.zarr" / "top_hits")
    refused = runner.invoke(app, ["recount-top-hits", str(release)])
    assert refused.exit_code == 1 and "has no Top-Hit Index" in refused.output


def test_a_tier_with_malformed_offsets_skips_the_rule_and_is_refused(digest_releases, tmp_path):
    """No count can be read through offsets that do not partition the tier:
    `top_hits.structure` fails it, the count rule says it could not judge, and the
    command writes nothing."""
    from opengwasdb.store.open import open_store

    release = _copy(digest_releases, "dense_observed", tmp_path)
    root = open_store(release).arrays(mode="r+")
    offsets = root["top_hits/p_5e_08/analysis_offsets"][:]
    offsets[-1] -= 1
    root["top_hits/p_5e_08/analysis_offsets"][:] = offsets
    held = (release / "analyses.tsv").read_bytes()

    (finding,) = _rule_findings(release)
    assert finding.severity is Severity.SKIP and "invalid analysis offsets" in finding.message
    with pytest.raises(RecountRefusal, match="invalid analysis offsets"):
        recount_top_hits(release)
    assert (release / "analyses.tsv").read_bytes() == held
