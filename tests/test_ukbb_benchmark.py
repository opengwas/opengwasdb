from datetime import timedelta

from benchmarks.benchmark_ukbb_dense import _build_seconds


def test_build_duration_accepts_millisecond_timestamps(tmp_path):
    log = tmp_path / "build.log"
    log.write_text(
        "2026-09-04 10:00:00,125 INFO Pass 1: starting\n"
        "2026-09-04 11:02:03,875 INFO Build complete\n"
    )

    assert _build_seconds(log) == timedelta(hours=1, minutes=2, seconds=3.75).total_seconds()
