"""The Reference Completion profiler's artifact says what it measured (#309).

The profiler splits a real LD block's time into stages and times the band write
over a slice of bands. What would mislead is a split whose shares do not add up
to the whole, or an artifact that cannot say which revision, argv and host load
produced it. Both are pinned here, on the Dense completion fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from dense_sources import build_dense_fixture
from test_dense_completion import SOURCE_HEADER, SOURCE_ROWS, _make_ld_panel

from benchmarks.profile_reference_completion import StageClock, choose_block, main


def test_nested_stages_are_charged_exclusively() -> None:
    now = [0.0]
    clock = StageClock(lambda: now[0])
    with clock.stage("outer"):
        now[0] += 2
        with clock.stage("inner"):
            now[0] += 5
        now[0] += 2
    assert clock.timer.seconds == {"outer": 4.0, "inner": 5.0}
    assert clock.timer.total() == 9.0
    assert clock.calls == {"outer": 1, "inner": 1}


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "associations.tsv"
    source.write_text(SOURCE_HEADER + "\n" + "\n".join(SOURCE_ROWS) + "\n", encoding="utf-8")
    observed = tmp_path / "obs.opengwasdb"
    build_dense_fixture(
        [source], observed, store_id="test", release_id="obs-v1", reference_assembly="GRCh38"
    )
    return observed, _make_ld_panel(tmp_path)


def test_choose_block_picks_by_variant_count(inputs: tuple[Path, Path]) -> None:
    _, panel = inputs
    paths = sorted(panel.glob("EUR/*/*.tsv"))
    assert len(paths) == 2
    assert choose_block(paths, "largest") == panel / "EUR" / "1" / "900000-1300000.tsv"
    assert choose_block(paths, "/elsewhere/1-2.tsv") == Path("/elsewhere/1-2.tsv")


def test_block_artifact_records_revision_argv_load_and_a_whole_split(
    inputs: tuple[Path, Path], tmp_path: Path
) -> None:
    observed, panel = inputs
    out = tmp_path / "block.json"
    argv = ["block", str(observed), str(panel), "--work", str(tmp_path / "w"),
            "--output", str(out), "--block", "largest"]
    assert main(argv) == 0

    artifact = json.loads(out.read_text())
    assert artifact["argv"][1:] == argv
    assert artifact["commit"]
    assert artifact["opengwasdb_fingerprint"]
    for key in ("host_load_start", "host_load_end"):
        assert artifact[key]["load_1m"] >= 0
    assert artifact["block_variants"] == 5
    assert artifact["n_analyses"] == 2
    stages = {entry["stage"]: entry for entry in artifact["stages"]}
    assert {"ld_panel_read", "source_read", "elastic_net_fit", "checkpoint_write"} <= set(stages)
    assert stages["elastic_net_fit"]["calls"] >= 1
    assert sum(entry["share"] for entry in stages.values()) == pytest.approx(1.0)
    assert artifact["total_seconds"] == pytest.approx(
        sum(entry["seconds"] for entry in stages.values())
    )


def test_bands_artifact_names_the_slice_and_splits_the_write(
    inputs: tuple[Path, Path], tmp_path: Path
) -> None:
    observed, panel = inputs
    out = tmp_path / "bands.json"
    argv = ["bands", str(observed), str(panel), "--work", str(tmp_path / "w"),
            "--output", str(out), "--bands", "1", "--n-workers", "1"]
    assert main(argv) == 0

    artifact = json.loads(out.read_text())
    assert artifact["argv"][1:] == argv
    assert artifact["host_load_start"]["load_1m"] >= 0
    assert artifact["bands_run"] == 1
    assert artifact["bands_in_full_matrix"] == 1
    assert artifact["blocks_run"] == 1, "only chromosome 1's block starts in the first band"
    assert artifact["blocks_in_panel"] == 2
    stages = {entry["stage"] for entry in artifact["stages"]}
    assert {
        "axis_construction_full", "ld_block_phase_slice", "merge_shard_fills_and_quality",
        "band_seed_from_source", "band_apply_fills", "band_judge_and_write",
        "band_residual_se_rewrite",
    } <= stages
    assert sum(entry["share"] for entry in artifact["stages"]) == pytest.approx(1.0)
    assert not list((tmp_path / "w").glob("*.opengwasdb")), "the slice must not be published"
