"""The build-cost harness's pure parts: its time parser, load gate and plan (#249).

The harness itself runs hours-long builds; these pin the pieces whose failure
would silently record a wrong number -- a mis-parsed wall time or RSS, a load
gate that starts a build on a busy node, or a plan that runs both sides with the
wrong argv or cwd.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.measure_build_cost import (
    GUARD_ENV,
    PILOTS,
    _hms,
    build_plan,
    merge_artifacts,
    parse_time_v,
    run_step,
    wait_for_quiet_load,
)

SAMPLE = """\
\tCommand being timed: "opengwasdb build-dense-vcf"
\tUser time (seconds): 900.10
\tSystem time (seconds): 12.34
\tElapsed (wall clock) time (h:mm:ss or m:ss): 8:58.12
\tMaximum resident set size (kbytes): 10180324
\tExit status: 0
"""


def test_parse_time_v_reads_wall_seconds_and_peak_rss() -> None:
    seconds, maxrss = parse_time_v(SAMPLE)
    assert seconds == pytest.approx(538.12)
    assert maxrss == 10180324


def test_parse_time_v_handles_an_hour_long_build() -> None:
    seconds, maxrss = parse_time_v(
        "Elapsed (wall clock) time (h:mm:ss or m:ss): 2:29:22.00\n"
        "Maximum resident set size (kbytes): 38257368\n"
    )
    assert seconds == pytest.approx(2 * 3600 + 29 * 60 + 22)
    assert maxrss == 38257368


def test_parse_time_v_fails_loudly_without_the_numbers() -> None:
    with pytest.raises(ValueError, match="did not report"):
        parse_time_v("opengwasdb: everything worked\n")


def test_hms_formats_the_wall_time() -> None:
    assert _hms(4.51) == "0:00:05"
    assert _hms(538.12) == "0:08:58"
    assert _hms(8962) == "2:29:22"


def test_wait_for_quiet_load_returns_the_reading_below_the_threshold() -> None:
    assert wait_for_quiet_load(3.0, 15.0, 60.0, load_fn=lambda: 2.5) == 2.5


def test_wait_for_quiet_load_refuses_to_start_a_busy_build() -> None:
    """A load at or above the threshold does not become a measurement."""
    with pytest.raises(SystemExit, match="not starting a build on a busy node"):
        # `max_wait=0` means the deadline is the first reading; no sleep is used.
        wait_for_quiet_load(3.0, 15.0, 0.0, load_fn=lambda: 37.9)


def _args(tmp_path: Path, order: str = "base,head", sides: str = "base,head", suffix: str = ""):
    class _Args:
        pass

    args = _Args()
    args.base_cmd_list = ["pixi", "run", "-e", "dev", "opengwasdb"]
    args.head_cmd_list = ["pixi", "run", "-e", "dev", "opengwasdb"]
    args.base_cwd = tmp_path / "base"
    args.head_cwd = tmp_path / "head"
    args.work = tmp_path / "work"
    args.pilot = ["ragged-besd", "hybrid"]
    args.order = order
    args.sides = sides
    args.sides_list = [side for side in sides.split(",") if side]
    args.suffix = suffix
    return args


def test_build_plan_alternates_the_order_and_separates_the_stores(tmp_path: Path) -> None:
    plan = build_plan(_args(tmp_path))
    assert [step["side"] for step in plan] == ["base", "head", "head", "base"]
    assert [step["pilot"] for step in plan] == ["ragged-besd", "ragged-besd", "hybrid", "hybrid"]
    stores = {str(step["store"]) for step in plan}
    assert len(stores) == 4  # one output directory per pilot per side
    for step in plan:
        expected_cwd = tmp_path / step["side"]
        assert step["cwd"] == expected_cwd
        assert str(step["store"]) in step["command"]


def test_build_plan_can_run_only_the_head_side_with_a_suffix(tmp_path: Path) -> None:
    """The guard-on proof builds reuse the head argv but a separate store path."""
    plan = build_plan(_args(tmp_path, sides="head", suffix="-guard"))
    assert [step["side"] for step in plan] == ["head", "head"]
    for step in plan:
        assert str(step["store"]).endswith("-guard.opengwasdb")
        assert "-guard" in str(step["log"])


def test_build_plan_refuses_an_unknown_side(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="--sides"):
        build_plan(_args(tmp_path, sides="middle"))


def test_build_plan_uses_the_registered_hybrid_argv(tmp_path: Path) -> None:
    """OGS-00004's registered argv: the two-pass `--reference-panel` form."""
    plan = build_plan(_args(tmp_path))
    hybrid = next(step for step in plan if step["pilot"] == "hybrid")
    argv = hybrid["command"]
    assert "build-hybrid" in argv
    assert "--reference-panel" in argv
    assert "/data/opengwasdb/reference/alid-panel/EUR-variants.tsv.gz" in argv
    assert "--variant-reference" not in argv  # the single-pass flag is #255


def test_every_pilot_names_a_builder_command() -> None:
    for pilot, argv_for in PILOTS.items():
        argv = argv_for(Path("/tmp/out.opengwasdb"))
        assert argv[0].startswith("build-"), (pilot, argv)
        assert str(Path("/tmp/out.opengwasdb")) in argv


def test_run_step_records_the_guard_environment(tmp_path: Path, monkeypatch) -> None:
    """An artifact says which guard configuration produced each number (#249 r1)."""
    kwargs = dict(
        side="head",
        pilot="dense",
        command=["true"],
        cwd=tmp_path,
        out=tmp_path / "out",
        log_path=tmp_path / "log",
        max_load=1000.0,
        poll_seconds=0.01,
        max_wait=5.0,
    )
    monkeypatch.setenv(GUARD_ENV, "1")
    assert run_step(**kwargs)["guard_env"] == "1"
    monkeypatch.delenv(GUARD_ENV, raising=False)
    step = run_step(**kwargs)
    assert step["guard_env"] is None
    assert step["exit_code"] == 0


def _write_payload(tmp_path: Path, name: str, payload: dict) -> Path:
    """Write one artifact payload and return its path."""
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _artifact(tmp_path: Path, name: str, commit_sha: str, steps: list[dict]) -> Path:
    payload = {
        "artifact": "test",
        "commit": commit_sha,
        "measured_at": "2026-10-08T00:00:00+00:00",
        "base_cwd": "/base",
        "base_revision": "f168ef1",
        "head_cwd": "/head",
        "head_revision": "f48ab2a",
        "load_threshold_1m": 3.0,
        "n_workers_note": "note",
        "production_config": "guard_env is null on every step",
        "timing_note": "time -v on the CLI process",
        "steps": [{"guard_env": None, **step} for step in steps],
    }
    return _write_payload(tmp_path, name, payload)


def test_merge_artifacts_keeps_run_order_and_its_own_provenance(tmp_path: Path) -> None:
    """The committed file is the harness's merge of the documented runs (#249 r1)."""
    small = _artifact(tmp_path, "small.json", "f48ab2a", [{"pilot": "dense", "side": "base"}])
    hybrid = _artifact(tmp_path, "hybrid.json", "f48ab2a", [{"pilot": "hybrid", "side": "head"}])
    merged = merge_artifacts([small, hybrid])
    assert [step["pilot"] for step in merged["steps"]] == ["dense", "hybrid"]
    assert [run["n_steps"] for run in merged["runs"]] == [1, 1]
    assert merged["commit"] == "f48ab2a"
    assert merged["production_config"] == "guard_env is null on every step"
    assert merged["timing_note"] == "time -v on the CLI process"


def test_merge_artifacts_refuses_two_code_versions(tmp_path: Path) -> None:
    first = _artifact(tmp_path, "a.json", "aaaaaaa", [{"pilot": "dense", "side": "base"}])
    second = _artifact(tmp_path, "b.json", "bbbbbbb", [{"pilot": "hybrid", "side": "head"}])
    with pytest.raises(SystemExit, match="two code versions"):
        merge_artifacts([first, second])


def test_merge_artifacts_refuses_a_duplicate_step(tmp_path: Path) -> None:
    """A pilot/side measured twice would make the before/after table ambiguous."""
    first = _artifact(tmp_path, "a.json", "f48ab2a", [{"pilot": "dense", "side": "base"}])
    second = _artifact(tmp_path, "b.json", "f48ab2a", [{"pilot": "dense", "side": "base"}])
    with pytest.raises(SystemExit, match="duplicate pilot/side"):
        merge_artifacts([first, second])


def test_merge_artifacts_refuses_a_guard_on_step(tmp_path: Path) -> None:
    """The merged file is the timed production-config one; the guard must be unset."""
    steps = [{"pilot": "dense", "side": "head", "guard_env": "1"}]
    guarded = _artifact(tmp_path, "guarded.json", "f48ab2a", steps)
    with pytest.raises(SystemExit, match="guard must be unset"):
        merge_artifacts([guarded])


def test_merge_artifacts_refuses_a_run_without_the_notes(tmp_path: Path) -> None:
    """A run from before the notes existed cannot be merged (#249 r3)."""
    path = _write_payload(
        tmp_path,
        "old.json",
        {
            "artifact": "test",
            "commit": "f48ab2a",
            "base_revision": "f168ef1",
            "head_revision": "f48ab2a",
            "steps": [{"pilot": "dense", "side": "base", "guard_env": None}],
        },
    )
    with pytest.raises(SystemExit, match="cannot be merged"):
        merge_artifacts([path])


def test_merge_artifacts_refuses_a_step_without_guard_env(tmp_path: Path) -> None:
    """A missing `guard_env` is missing data, not 'unset' (#249 r3)."""
    path = _write_payload(
        tmp_path,
        "no-guard.json",
        {
            "artifact": "test",
            "commit": "f48ab2a",
            "base_revision": "f168ef1",
            "head_revision": "f48ab2a",
            "production_config": "note",
            "timing_note": "note",
            "steps": [{"pilot": "dense", "side": "base"}],
        },
    )
    with pytest.raises(SystemExit, match="record no"):
        merge_artifacts([path])
