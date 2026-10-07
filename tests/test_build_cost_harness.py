"""The build-cost harness's pure parts: its time parser, load gate and plan (#249).

The harness itself runs hours-long builds; these pin the pieces whose failure
would silently record a wrong number -- a mis-parsed wall time or RSS, a load
gate that starts a build on a busy node, or a plan that runs both sides with the
wrong argv or cwd.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.measure_build_cost import (
    PILOTS,
    _hms,
    build_plan,
    parse_time_v,
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
