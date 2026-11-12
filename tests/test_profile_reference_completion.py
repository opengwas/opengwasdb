"""The Reference Completion profiler's artifact says what it measured (#309).

What would mislead is a split that does not account for the call it came from, a
cleanup that deletes what it did not make or leaves what it did, or an artifact
that names code it did not run. Each is checked against something measured
separately from the stages themselves.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from dense_sources import build_dense_from_records
from test_dense_completion import _make_ld_panel

from benchmarks import profile_reference_completion as profiler
from benchmarks.profile_reference_completion import (
    StageClock,
    check_accounts_for_wall,
    choose_block,
    main,
    recorded_full_run,
    require_new_directory,
    scale_to_full_run,
)
from opengwasdb.layouts.dense.build import NormalisedAssociation
from opengwasdb.variants import CanonicalVariant


class FakeClock:
    """A clock that moves only when the test says so."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_nested_stages_are_charged_exclusively() -> None:
    clock = FakeClock()
    stages = StageClock(clock)
    with stages.stage("outer"):
        clock.now += 2
        with stages.stage("inner"):
            clock.now += 5
        clock.now += 2
    assert stages.timer.seconds == {"outer": 4.0, "inner": 5.0}
    assert stages.calls == {"outer": 1, "inner": 1}


def test_add_moves_seconds_a_callee_timed_out_of_the_running_stage() -> None:
    clock = FakeClock()
    stages = StageClock(clock)
    with stages.stage("rewrite"):
        clock.now += 10
        stages.add("rewrite.read", 4.0)
    assert stages.timer.seconds == {"rewrite": 6.0, "rewrite.read": 4.0}


def test_accounted_time_must_match_the_calls_wall_time() -> None:
    check_accounts_for_wall(98.0, 100.0)
    with pytest.raises(RuntimeError, match="missing from the table"):
        check_accounts_for_wall(60.0, 100.0)
    with pytest.raises(RuntimeError, match="counted twice"):
        check_accounts_for_wall(140.0, 100.0)


@pytest.fixture
def ld_panel(tmp_path: Path) -> Path:
    return _make_ld_panel(tmp_path)


def test_choose_block_takes_the_middle_of_three_different_sizes(tmp_path: Path) -> None:
    paths = []
    for name, n in (("a", 9), ("b", 3), ("c", 6)):
        path = tmp_path / f"{name}.tsv"
        path.write_text("CHR\tSNP\n" + "1\tx\n" * n)
        paths.append(path)
    assert choose_block(paths, "median") == tmp_path / "c.tsv"
    assert choose_block(paths, "largest") == tmp_path / "a.tsv"
    assert choose_block(paths, "/elsewhere/1-2.tsv") == Path("/elsewhere/1-2.tsv")


def test_work_directory_must_be_new(tmp_path: Path) -> None:
    existing = tmp_path / "keep"
    existing.mkdir()
    (existing / "precious").write_text("x")
    with pytest.raises(SystemExit, match="refusing"):
        require_new_directory(existing)
    assert (existing / "precious").read_text() == "x"


def _records() -> list[NormalisedAssociation]:
    """a1 observes three panel positions with a frequency, a2 observes one."""

    def record(analysis: str, position: int, z: float) -> NormalisedAssociation:
        return NormalisedAssociation(
            analysis, CanonicalVariant("1", position, "A", "G"), z=z, se=0.2, eaf=0.3
        )

    return [
        record("a1", 900_000, 3.0), record("a1", 950_000, 2.5), record("a1", 1_000_000, 4.0),
        record("a2", 1_000_000, 2.0),
    ]


@pytest.fixture
def inputs(tmp_path: Path, ld_panel: Path) -> tuple[Path, Path]:
    observed = tmp_path / "obs.opengwasdb"
    build_dense_from_records(
        _records(), observed,
        store_id="test", release_id="obs-v1", reference_assembly="GRCh38",
    )
    return observed, ld_panel


def _run(tmp_path: Path, command: str, source: Path, panel: Path, *extra: str) -> dict:
    out = tmp_path / f"{command}.json"
    argv = [command, str(source), str(panel), "--work", str(tmp_path / f"{command}-work"),
            "--output", str(out), "--allow-dirty", "--load-period", "0.01", *extra]
    assert main(argv) == 0
    artifact = json.loads(out.read_text())
    assert artifact["argv"][1:] == argv
    return artifact


def test_block_artifact_accounts_for_its_wall_time_and_cleans_up(
    inputs: tuple[Path, Path], tmp_path: Path
) -> None:
    observed, panel = inputs
    artifact = _run(tmp_path, "block", observed, panel, "--block", "largest")

    assert artifact["block_variants"] == 5
    assert artifact["n_analyses"] == 2
    assert artifact["distinct_observed_sets"] == 2
    assert artifact["accounted_seconds"] == pytest.approx(
        artifact["wall_seconds"], rel=0.05, abs=0.05
    )
    stages = {entry["stage"]: entry for entry in artifact["stages"]}
    assert {"ld_panel_read", "source_read", "elastic_net_fit", "owned_rows",
            "checkpoint_write"} <= set(stages)
    assert stages["elastic_net_fit"]["calls"] >= 1
    assert artifact["commit"]
    assert "worktree_clean" in artifact
    assert len(artifact["host_load_samples"]) >= 2
    assert not (tmp_path / "block-work").exists(), "the profile's own scratch is removed"


def test_a_run_whose_outer_stage_is_missing_fails_the_wall_check(
    inputs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Time spent outside every stage is not hidden: the same run passes with the
    outer stage in place and fails with it left out."""
    observed, panel = inputs
    from opengwasdb.layouts.dense import complete as dense

    clock = FakeClock()
    run_block = dense._run_block

    def slow_block(task):
        clock.now += 100.0
        return run_block(task)

    monkeypatch.setattr(dense, "_run_block", slow_block)
    work = tmp_path / "ok-work"
    body = profiler.profile_block(observed, panel, "EUR", work, "largest", clock=clock)
    assert body["wall_seconds"] == 100.0
    assert body["accounted_seconds"] == 100.0

    original = StageClock.stage

    def without_outer(self, name):
        if name == "block_other":
            from contextlib import nullcontext
            return nullcontext()
        return original(self, name)

    monkeypatch.setattr(StageClock, "stage", without_outer)
    with pytest.raises(RuntimeError, match="missing from the table"):
        profiler.profile_block(
            observed, panel, "EUR", tmp_path / "bad-work", "largest", clock=clock
        )


def test_a_dirty_tree_is_refused(
    inputs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed, panel = inputs
    monkeypatch.setattr("benchmarks._artifact.tree_is_clean", lambda: False)
    with pytest.raises(SystemExit, match="uncommitted"):
        main(["block", str(observed), str(panel), "--work", str(tmp_path / "w"),
              "--output", str(tmp_path / "o.json")])
    assert not (tmp_path / "o.json").exists()


def test_bands_artifact_splits_each_pass_and_scales_to_the_full_run(
    inputs: tuple[Path, Path], tmp_path: Path
) -> None:
    observed, panel = inputs
    artifact = _run(tmp_path, "bands", observed, panel, "--bands", "1", "--n-workers", "1")

    assert artifact["bands_run"] == 1
    assert artifact["blocks_run"] == 1, "only chromosome 1's block starts in the first band"
    assert artifact["blocks_in_panel"] == 2
    assert artifact["accounted_seconds"] == pytest.approx(
        artifact["wall_seconds"], rel=0.05, abs=0.05
    )
    stages = {entry["stage"] for entry in artifact["stages"]}
    assert {
        "axis_construction_full", "ld_block_phase_slice", "merge_shard_fills_and_quality",
        "merge_union_map", "merge_reference_eaf", "merge_create_planes",
        "band_read_source", "band_read_fills", "band_compute_apply_fills",
        "band_compute_judge", "band_compute_encode_z", "band_compute_encode_eaf",
        "band_compute_encode_se", "band_write", "band_side_arrays",
    } <= stages
    assert "band_write_pool" not in stages
    full = artifact["scaled_to_full_run"]
    assert full["stages"]["band_write"]["factor"] == 1.0
    assert not (tmp_path / "bands-work").exists()


def test_a_band_write_on_the_pool_is_timed_as_one_stage(
    inputs: tuple[Path, Path], tmp_path: Path
) -> None:
    """The pool's workers run the bands where this process's clock cannot see,
    so their time is one stage, and the wall check still holds."""
    observed, panel = inputs
    artifact = _run(
        tmp_path, "bands", observed, panel, "--bands", "1", "--n-workers", "1",
        "--write-workers", "2",
    )

    assert artifact["write_workers"] == 2
    stages = {entry["stage"] for entry in artifact["stages"]}
    assert "band_write_pool" in stages
    assert not {"band_read_source", "band_write", "band_write_other"} & stages
    assert artifact["accounted_seconds"] == pytest.approx(
        artifact["wall_seconds"], rel=0.05, abs=0.05
    )


def test_scaling_follows_bands_and_rows() -> None:
    summary = {
        "bands_in_full_matrix": 56, "bands_run": 2, "union_variants": 11_200_000,
        "slice_rows": 400_000,
        "stages": [
            {"stage": "band_write", "seconds": 10.0},
            {"stage": "band_read_source", "seconds": 5.0},
            {"stage": "merge_shard_fills_and_quality", "seconds": 100.0},
            {"stage": "axis_construction_full", "seconds": 250.0},
        ],
    }
    scaled = scale_to_full_run(summary)
    assert scaled["stages"]["band_write"]["seconds"] == 280.0
    assert scaled["stages"]["merge_shard_fills_and_quality"]["seconds"] == 2800.0
    assert "axis_construction_full" not in scaled["stages"]
    assert scaled["band_write_seconds"] == 420.0


def test_recorded_full_run_reads_the_figures_from_the_step_record(tmp_path: Path) -> None:
    stderr = "\n".join([
        "2026-10-23 00:55:42 INFO opengwasdb.completion.ancestry_filter: start",
        "2026-10-23 06:10:20 WARNING opengwasdb.completion.ld_panel: Block 22/1-2 stores",
        "2026-10-23 12:15:16 INFO opengwasdb.encoding.se: SE rewrite count: start",
        "2026-10-23 13:49:27 INFO opengwasdb.encoding.se: SE rewrite: done in 51m59s",
    ])
    record = tmp_path / "complete.json"
    record.write_text(json.dumps({"stderr": stderr, "elapsed_seconds": 50682.3}))
    figures = recorded_full_run(record)
    assert figures["blocks_at_least_seconds"] == 5 * 3600 + 14 * 60 + 38
    assert figures["post_block_at_most_seconds"] == 7 * 3600 + 39 * 60 + 7
    assert figures["se_rewrite_seconds"] == 1 * 3600 + 34 * 60 + 11
