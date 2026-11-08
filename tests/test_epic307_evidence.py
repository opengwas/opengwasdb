"""The epic #307 evidence harness: step records, windows, reader A/B, comparison.

The harness's only real job is to report what the records say, so these pin the
places where it could quietly report something else:

- the parser reads exactly what the package's own emitters write (a round trip
  through `log_phase` and both `log_progress`), and a spelling it has stopped
  understanding is counted rather than skipped;
- a missing field is reported as missing, never defaulted to a plausible zero;
- a window flags the phases and steps it overlaps, on the log lines' clock,
  across midnight and every night the records span;
- a phase's clean rate comes from the intervals outside every window, and the
  pace it reached on its last line before one is recorded beside it;
- every figure in the epic's tables is measured against the record that holds
  it, with a reason when it does not reproduce;
- the reader A/B interleaves its sides, refuses a row count that moved, and
  scores the control tree separately from the parse floor.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import re
import subprocess
import time
from datetime import date, datetime
from datetime import time as clock_time
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from benchmarks import _epic307_figures, _reader_ab
from benchmarks._epic307_figures import (
    FIGURES,
    Derive,
    Figure,
    compare,
    measure_figure,
    reproduces,
)
from benchmarks._step_records import (
    Phase,
    Window,
    clean_rate,
    daily_windows,
    measure_store,
    parse_phases,
    parse_progress,
    parse_record,
    parse_window,
    parser_suspects,
)
from benchmarks.measure_epic307_evidence import main
from opengwasdb.cli.main import LOG_DATE_FORMAT, LOG_FORMAT


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []
        self.formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


def _captured(logger: logging.Logger) -> tuple[_Capture, logging.Handler]:
    capture = _Capture()
    logger.addHandler(capture)
    logger.setLevel(logging.INFO)
    return capture, capture


def _inside_window_detail(detail: dict) -> bool:
    return detail.get("inside_window") is True


def _at(text: str) -> datetime:
    return datetime.fromisoformat(text)


def _line(at: str, message: str) -> str:
    return f"{at} INFO opengwasdb.some.phase: {message}"


def _record(
    tmp_path: Path,
    stderr: str | None,
    *,
    step: str = "build",
    wall: float | None = 1234.5,
    start: str | None = "2026-10-22T00:00:00.000000Z",
    end: str | None = "2026-10-22T00:20:34.500000Z",
    extra: dict | None = None,
) -> Path:
    records = tmp_path / "records"
    records.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "store_id": "OGS-00000",
        "exit_code": 0,
        "success": True,
        "start_time": start,
        "end_time": end,
        "elapsed_seconds": wall,
        "argv": ["opengwasdb", "build-dense-vcf", "analyses.tsv", "store.opengwasdb"],
        "opengwasdb_rev": "d405de0a7e1be29b1fc1d2e0dfb1090f607ba08c",
        "opengwasdb_version": "0.3.0",
    }
    if stderr is not None:
        payload["stderr"] = stderr
    payload.update(extra or {})
    path = records / f"{step}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --- 1. the parser reads what the emitters write ----------------------------


def test_the_parser_reads_what_the_real_emitters_write(monkeypatch):
    """A round trip through the package's own phase and progress emitters.

    The harness copies their spellings instead of calling them, so this is what
    notices when an emitter's format moves and the parser has stopped matching
    it. The clock is driven, because a block that takes no time renders as `0s`
    and would never exercise the hour-and-minute spelling.
    """
    from opengwasdb.encoding import timing
    from opengwasdb.encoding.timing import log_phase, log_progress
    from opengwasdb.layouts.dense.build_vcf import log_progress as dense_progress

    class _Clock:
        def __init__(self) -> None:
            self.now = 0.0

        def monotonic(self) -> float:
            return self.now

    clock = _Clock()
    monkeypatch.setattr(timing, "time", clock)

    logger = logging.getLogger("epic307.roundtrip")
    dense_logger = logging.getLogger("opengwasdb.layouts.dense.build_vcf")
    levels = (logger.level, dense_logger.level)
    capture, handler = _captured(logger)
    dense_capture, dense_handler = _captured(dense_logger)
    try:
        with log_phase(logger, "Band-write se"):
            clock.now = 5288.0
        clock.now = 5288.0
        log_progress(
            logger, "Band-write se", 2514, 2514, clock.now - 90, every=2514
        )
        dense_progress("Pass 2", 2514, 2514, time.monotonic() - 3215, "last: ukb-b-9981", every=25)
    finally:
        logger.removeHandler(handler)
        dense_logger.removeHandler(dense_handler)
        # `_captured` raised both loggers to INFO; leaving them there changes
        # what every later test's log capture sees.
        logger.setLevel(levels[0])
        dense_logger.setLevel(levels[1])

    stderr = "\n".join(capture.lines + dense_capture.lines) + "\n"

    (phase,) = parse_phases(stderr)
    assert phase.label == "Band-write se"
    assert phase.start is not None and phase.end is not None
    # The emitter's own spelling, hour and minute, parses back to its seconds.
    assert phase.logged_duration == "1h28m"
    assert phase.logged_duration_seconds == 5280
    progress = parse_progress(stderr)
    assert [(sample.completed, sample.total) for sample in progress["Band-write se"]] == [
        (2514, 2514)
    ]
    assert progress["Band-write se"][0].elapsed_seconds == 90
    assert [(sample.completed, sample.elapsed_seconds) for sample in progress["Pass 2"]] == [
        (2514, 3215)
    ]
    assert parser_suspects(stderr)["lines"] == 0


def test_a_spelling_the_parser_no_longer_matches_is_counted(tmp_path):
    """`phases: []` must never pass for a step that logged none."""
    stderr = "\n".join(
        [
            _line("2026-10-22 00:00:00", "Band-write z: done in 1h28"),
            _line("2026-10-22 00:00:01", "Pass 2: 25/2514 done (last: x) - elapsed 1m, ETA 2m"),
            "2026-10-22T00:00:02Z INFO opengwasdb.x: Band-write z: start",
            _line("2026-10-22 00:00:03", "Band-write eaf: start"),
        ]
    )

    suspects = parser_suspects(stderr)

    assert suspects["lines"] == 3
    assert len(suspects["examples"]) == 3
    # The fourth line is a real phase start and parses.
    assert [phase.label for phase in parse_phases(stderr)] == ["Band-write eaf"]


def test_ordinary_log_lines_are_not_counted_as_unparsed():
    stderr = "\n".join(
        [
            _line("2026-10-22 00:02:04", "Pass 1 extraction: 2514 files -> 154 window group(s)"),
            _line("2026-10-22 00:07:24", "Pass 1 merge: 154 window(s) in 29m00s"),
            _line(
                "2026-10-22 18:52:31",
                "SE fit: 2514 Analyses, max |coefficient| 11.5, selected se=int8_residual",
            ),
        ]
    )

    assert parser_suspects(stderr) == {"lines": 0, "examples": []}


# --- 2. missing data is reported, never defaulted ---------------------------


def test_a_records_revision_is_read_from_whichever_key_it_holds(tmp_path):
    revision_path = _record(tmp_path, "", step="se-reencode", wall=None)
    payload = json.loads(revision_path.read_text())
    payload.pop("opengwasdb_rev")
    payload["opengwasdb_revision"] = "5fc07666202d43500d5e48bbc48f1e4c9e34ef0f"
    revision_path.write_text(json.dumps(payload), encoding="utf-8")

    record = parse_record(revision_path)

    assert record.revision_key == "opengwasdb_revision"
    assert record.revision == "5fc07666202d43500d5e48bbc48f1e4c9e34ef0f"


def test_a_missing_stderr_is_missing_not_an_empty_one(tmp_path):
    path = _record(tmp_path, None)

    record = parse_record(path)

    assert record.stderr_present is False
    assert record.stderr_lines is None
    assert record.unparsed == {"lines": 0, "examples": []}
    assert "stderr" in record.missing_fields
    assert record.to_dict([])["stderr_lines"] is None


def test_missing_step_fields_are_named_rather_than_defaulted(tmp_path):
    path = _record(tmp_path, "", extra={"store_id": "OGS-00000"})
    payload = json.loads(path.read_text())
    for key in ("exit_code", "success", "elapsed_seconds", "opengwasdb_rev"):
        payload.pop(key)
    path.write_text(json.dumps(payload), encoding="utf-8")

    record = parse_record(path)

    assert set(record.missing_fields) >= {"exit_code", "success", "elapsed_seconds"}
    assert "opengwasdb_revision" in record.missing_fields
    assert record.wall_seconds is None
    assert record.exit_code is None
    assert record.success is None
    assert record.revision is None


def test_a_phase_without_its_done_line_is_unfinished_not_zero():
    phases = parse_phases(_line("2026-10-22 00:01:00", "Validate: start") + "\n")

    assert phases == [
        Phase(
            label="Validate",
            start=_at("2026-10-22T00:01:00"),
            end=None,
            logged_duration=None,
            logged_duration_seconds=None,
        )
    ]
    assert phases[0].elapsed_seconds is None


def test_a_record_yields_the_steps_wall_time_and_each_phases_span(tmp_path):
    path = _record(
        tmp_path,
        "\n".join(
            [
                _line("2026-10-22 00:01:00", "Band-write z: start"),
                _line("2026-10-22 00:02:30", "Band-write z: done in 1m30s"),
                _line("2026-10-20 21:55:18", "Top-hit scan: start"),
                _line("2026-10-20 23:30:19", "Top-hit scan: done in 1h35m"),
            ]
        )
        + "\n",
    )

    step = parse_record(path).to_dict([])

    assert step["wall_seconds"] == 1234.5
    assert [(p["label"], p["start"], p["end"]) for p in step["phases"]] == [
        ("Band-write z", "2026-10-22T00:01:00", "2026-10-22T00:02:30"),
        ("Top-hit scan", "2026-10-20T21:55:18", "2026-10-20T23:30:19"),
    ]
    # 5,701 s between the two lines, 5,700 s in the minute-rounded `done in`.
    assert [p["elapsed_seconds"] for p in step["phases"]] == [90.0, 5701.0]
    assert [p["logged_duration_seconds"] for p in step["phases"]] == [90, 5700]


# --- 3. windows -------------------------------------------------------------


def test_a_window_flags_the_phases_it_overlaps_and_no_others():
    window = parse_window("2026-10-22T01:06/2026-10-22T16:10")

    assert [
        window.overlaps(_at("2026-10-22T09:00:00"), _at("2026-10-22T10:00:00")),
        window.overlaps(_at("2026-10-22T17:00:00"), _at("2026-10-22T18:00:00")),
        window.overlaps(_at("2026-10-22T00:30:00"), _at("2026-10-22T01:30:00")),
        window.overlaps(_at("2026-10-22T16:00:00"), _at("2026-10-22T16:30:00")),
        window.overlaps(_at("2026-10-21T23:00:00"), _at("2026-10-22T01:06:00")),
    ] == [True, False, True, True, False]


def test_a_window_that_lands_before_its_start_crosses_midnight():
    window = parse_window("2026-10-22T22:00/2026-10-23T06:00")

    assert window.start == _at("2026-10-22T22:00:00")
    assert window.end == _at("2026-10-23T06:00:00")
    assert window.overlaps(_at("2026-10-22T23:30:00"), _at("2026-10-23T00:30:00")) is True
    assert window.overlaps(_at("2026-10-22T20:00:00"), _at("2026-10-22T22:00:00")) is False
    assert parse_window("2026-10-22T22:00/2026-10-22T06:00").end == _at("2026-10-23T06:00:00")


def test_a_phase_that_straddles_midnight_is_flagged_by_a_nightly_window(tmp_path):
    path = _record(
        tmp_path,
        "\n".join(
            [
                _line("2026-10-22 23:30:00", "Band-write eaf: start"),
                _line("2026-10-23 00:30:00", "Band-write eaf: done in 1h00m"),
            ]
        )
        + "\n",
    )

    step = parse_record(path).to_dict([parse_window("2026-10-22T22:00/2026-10-23T06:00")])

    (phase,) = step["phases"]
    assert phase["elapsed_seconds"] == 3600.0
    assert phase["overlaps_window"] is True
    assert phase["window_index"] == 0
    assert phase["seconds_inside_windows"] == 3600.0


def test_a_daily_window_covers_every_night_the_records_span():
    windows = daily_windows(
        "01:06-16:10", "Europe/London", first=date(2026, 10, 22), until=date(2026, 10, 25)
    )

    assert [window.date for window in windows] == ["2026-10-22", "2026-10-23", "2026-10-24"]
    # Each is the named zone's 01:06 to 16:10, expressed on the log clock.
    for window in windows:
        night = date.fromisoformat(window.date or "")
        start = datetime.combine(night, clock_time(1, 6), tzinfo=ZoneInfo("Europe/London"))
        end = datetime.combine(night, clock_time(16, 10), tzinfo=ZoneInfo("Europe/London"))
        assert window.start == start.astimezone().replace(tzinfo=None)
        assert window.end == end.astimezone().replace(tzinfo=None)
        assert (window.end - window.start).total_seconds() == 15 * 3600 + 4 * 60


def test_a_daily_window_keeps_its_wall_clock_across_the_dst_change():
    windows = daily_windows(
        "01:06-16:10", "Europe/London", first=date(2026, 10, 24), until=date(2026, 10, 27)
    )

    # 25 Oct 2026 is the night the clocks go back. The window is a wall-clock
    # reading, so every night starts at 01:06 and ends at 16:10; what changes is
    # the UTC offset those readings stand for, which is why the zone is named.
    offsets = {
        window.date: window.start.replace(tzinfo=ZoneInfo("Europe/London")).utcoffset()
        for window in windows
    }
    assert offsets["2026-10-24"] != offsets["2026-10-26"]
    assert {(window.start.hour, window.start.minute) for window in windows} == {(1, 6)}
    assert {(window.end.hour, window.end.minute) for window in windows} == {(16, 10)}


def test_a_step_with_no_phases_is_flagged_when_the_step_itself_overlaps(tmp_path):
    path = _record(
        tmp_path,
        "",
        step="validate",
        start="2026-10-22T18:00:00.000000Z",
        end="2026-10-22T21:00:00.000000Z",
    )

    step = parse_record(path).to_dict([parse_window("2026-10-22T01:06/2026-10-22T19:30:00")])

    assert step["phases"] == []
    assert step["overlaps_window"] is True
    assert step["span_source"] == "record_times"


def test_a_phase_with_no_end_has_an_unknown_window_flag(tmp_path):
    """Unknown is not "outside": a phase missing its done line has no span."""
    path = _record(
        tmp_path,
        _line("2026-10-22 09:00:00", "Ragged Overflow CSR flush: start")
        + "\n"
        + _line("2026-10-22 09:05:00", "Band-write z: start")
        + "\n"
        + _line("2026-10-22 09:10:00", "Band-write z: done in 5m00s"),
    )

    step = parse_record(path).to_dict([parse_window("2026-10-22T01:06/2026-10-22T16:10")])

    unfinished, closed = step["phases"]
    assert unfinished["overlaps_window"] is None
    assert unfinished["window_index"] is None
    assert unfinished["seconds_inside_windows"] is None
    assert closed["overlaps_window"] is True
    assert step["window_unknown"] == 1


def test_a_stderr_from_another_run_is_flagged(tmp_path):
    path = _record(
        tmp_path,
        _line("2026-10-19 14:59:11", "Band-write z: start") + "\n",
        start="2026-10-22T16:48:00.000000Z",
        end="2026-10-23T09:00:00.000000Z",
    )

    step = parse_record(path).to_dict([])

    assert step["stderr_outside_record"] is True
    assert step["record_span"]["first"] == "2026-10-22T17:48:00"
    assert step["stderr_span"]["first"] == "2026-10-19T14:59:11"


def test_a_stderr_from_the_records_own_run_is_not_flagged(tmp_path):
    path = _record(
        tmp_path,
        _line("2026-10-22 01:00:00", "Band-write z: start") + "\n",
        start="2026-10-22T00:00:00.000000Z",
        end="2026-10-22T02:00:00.000000Z",
    )

    assert parse_record(path).to_dict([])["stderr_outside_record"] is False


# --- 4. the clean rate ------------------------------------------------------


PROGRESS_LINES = "\n".join(
    [
        _line("2026-10-22 00:01:40", "Pass 2: 100/550 done (last: a) — elapsed 1m40s, ETA 7m30s"),
        _line("2026-10-22 00:03:20", "Pass 2: 200/550 done (last: b) — elapsed 3m20s, ETA 5m50s"),
        _line("2026-10-22 01:00:00", "Pass 2: 220/550 done (last: c) — elapsed 1h00m, ETA 1h30m"),
        _line("2026-10-22 02:00:00", "Pass 2: 250/550 done (last: d) — elapsed 2h00m, ETA 3h00m"),
        _line("2026-10-22 02:01:40", "Pass 2: 350/550 done (last: e) — elapsed 2h01m, ETA 1m40s"),
        _line("2026-10-22 02:03:20", "Pass 2: 450/550 done (last: f) — elapsed 2h03m, ETA 50s"),
        _line("2026-10-22 02:05:00", "Pass 2: 550/550 done (last: g) — elapsed 2h05m, ETA 0s"),
    ]
)


def test_a_progress_phase_is_parsed_from_both_line_flavours():
    samples = parse_progress(
        "\n".join(
            [
                _line(
                    "2026-10-22 00:00:01",
                    "Band-write se: 1024/2514 done (cols 0:1024) — elapsed 1m00s, ETA 1m30s",
                ),
                _line("2026-10-22 00:02:01", "Band-write se: 2048/2514 — elapsed 3m00s, ETA 45s"),
            ]
        )
    )

    assert [(s.completed, s.total, s.elapsed_seconds) for s in samples["Band-write se"]] == [
        (1024, 2514, 60),
        (2048, 2514, 180),
    ]


def test_clean_rate_uses_only_the_intervals_outside_the_window():
    samples = parse_progress(PROGRESS_LINES)["Pass 2"]
    window = Window(
        start=_at("2026-10-22T00:10:00"), end=_at("2026-10-22T02:00:00"), text="synthetic"
    )

    rate = clean_rate(samples, [window])

    # Four clean intervals, each 100 items in 100 s. The intervals that touch the
    # window (200->220->250) are excluded, so none of the starved time is
    # charged to the phase. Pooled: 400 items in 400 s.
    assert rate["intervals"] == 4
    assert rate["items"] == 400
    assert rate["seconds"] == 400.0
    assert rate["seconds_per_item"] == 1.0
    assert rate["projected_total_seconds"] == 550.0
    assert rate["median_interval_seconds_per_item"] == 1.0
    assert rate["median_projected_total_seconds"] == 550.0
    assert rate["samples_inside_window"] == 1
    assert rate["overlaps_window"] is True
    # The whole-phase rate is kept beside it, which is what the window inflates.
    assert rate["whole_phase"]["seconds_per_item"] == pytest.approx(7400 / 450)
    # The pace the phase had reached on its last clean line before the window.
    assert rate["prefix"] == {
        "at": "2026-10-22T00:03:20",
        "completed": 200,
        "elapsed_seconds": 200,
        "items": 200,
        "elapsed_source": "log_elapsed",
        "seconds_per_item": 1.0,
        "projected_total_seconds": 550.0,
        "reason": None,
    }
    assert rate["first_inside_window"] == {
        "at": "2026-10-22T01:00:00",
        "completed": 220,
        "elapsed_seconds": 3600,
        "seconds_per_item": pytest.approx(3600 / 220),
    }


def test_the_median_clean_pace_survives_an_hour_that_was_slow_for_another_reason():
    """A six-day phase meets hours no window covers and that are still slow."""
    stderr = "\n".join(
        [
            _line(
                "2026-10-14 23:00:00",
                "Pass 2: 100/400 done (last: a) — elapsed 1m40s, ETA 5m00s",
            ),
            _line(
                "2026-10-14 23:01:40",
                "Pass 2: 200/400 done (last: b) — elapsed 3m20s, ETA 3m20s",
            ),
            _line(
                "2026-10-14 23:03:20",
                "Pass 2: 300/400 done (last: c) — elapsed 5m00s, ETA 1m40s",
            ),
            _line(
                "2026-10-15 00:36:40",
                "Pass 2: 400/400 done (last: d) — elapsed 1h38m20s, ETA 0s",
            ),
        ]
    )
    samples = parse_progress(stderr)["Pass 2"]

    rate = clean_rate(samples, [parse_window("2026-10-22T01:06/2026-10-22T16:10")])

    # Intervals of 1.0, 1.0 and 18.0 s per item: the pooled rate is dragged by
    # the last one, the median is what the phase usually keeps.
    assert rate["seconds_per_item"] == pytest.approx(5800 / 300)
    assert rate["median_interval_seconds_per_item"] == 1.0
    assert rate["median_projected_total_seconds"] == 400.0


def test_an_unreadable_elapsed_does_not_read_as_no_line_before_the_window():
    """A `5ms` counter does not parse, and must not turn into `None`."""
    stderr = "\n".join(
        [
            _line(
                "2026-10-22 00:00:00",
                "Pass 2: 100/400 done (last: a) — elapsed 1m00s, ETA 3m00s",
            ),
            _line(
                "2026-10-22 00:01:00",
                "Pass 2: 200/400 done (last: b) — elapsed 5ms, ETA 2m00s",
            ),
            _line(
                "2026-10-22 03:00:00",
                "Pass 2: 400/400 done (last: c) — elapsed 3h00m, ETA 0s",
            ),
        ]
    )
    samples = parse_progress(stderr)["Pass 2"]
    assert samples[1].elapsed_seconds is None

    rate = clean_rate(samples, [parse_window("2026-10-22T02:00/2026-10-22T03:00")])

    assert rate["prefix"] is not None
    assert rate["prefix"]["completed"] == 200
    assert rate["prefix"]["elapsed_source"] == "sample_timestamps"
    # The items finished between the two lines, not the cumulative count: the
    # first line already reported 100 of them.
    assert rate["prefix"]["items"] == 200 - 100
    assert rate["prefix"]["elapsed_seconds"] == 60
    assert rate["prefix"]["seconds_per_item"] == pytest.approx((60 - 0) / (200 - 100))


def test_slow_intervals_outside_every_window_are_listed_with_their_time():
    """The window is an assumption; an hour it does not cover that is still slow
    must be named rather than averaged in."""
    stderr = "\n".join(
        [
            _line(
                "2026-10-15 10:00:00",
                "Pass 2: 100/400 done (last: a) — elapsed 1m40s, ETA 5m00s",
            ),
            _line(
                "2026-10-15 10:01:40",
                "Pass 2: 200/400 done (last: b) — elapsed 3m20s, ETA 3m20s",
            ),
            _line(
                "2026-10-15 10:03:20",
                "Pass 2: 300/400 done (last: c) — elapsed 5m00s, ETA 1m40s",
            ),
            _line(
                "2026-10-15 12:00:00",
                "Pass 2: 400/400 done (last: d) — elapsed 2h00m, ETA 0s",
            ),
        ]
    )
    samples = parse_progress(stderr)["Pass 2"]

    rate = clean_rate(samples, [parse_window("2026-10-14T01:06/2026-10-14T16:10")])

    assert rate["median_interval_seconds_per_item"] == 1.0
    assert rate["seconds_per_item"] == pytest.approx(7200 / 300)
    (slow,) = rate["slow_intervals"]
    assert slow["from"] == "2026-10-15T10:03:20"
    assert slow["to"] == "2026-10-15T12:00:00"
    assert slow["items"] == 100
    assert slow["seconds"] == 7000.0
    assert slow["times_median"] == 70.0


def test_clean_rate_refuses_a_phase_with_no_clean_interval():
    samples = parse_progress(PROGRESS_LINES)["Pass 2"]

    rate = clean_rate(samples, [parse_window("2026-10-22T00:00/2026-10-22T06:00")])

    assert rate["seconds_per_item"] is None
    assert rate["intervals"] == 0
    assert rate["projected_total_seconds"] is None
    assert rate["prefix"] is None
    assert rate["first_inside_window"]["completed"] == 100


# --- 5. a store's whole step set --------------------------------------------


def test_measure_store_refuses_a_directory_whose_records_disagree(tmp_path):
    _record(tmp_path, "", step="build")
    _record(tmp_path, "", step="validate", extra={"store_id": "OGS-00001"})

    with pytest.raises(SystemExit, match="disagree"):
        measure_store(tmp_path)


def test_measure_store_records_every_step_record(tmp_path):
    _record(tmp_path, "", step="build")
    _record(tmp_path, "", step="top-hits", wall=42.0)

    store = measure_store(tmp_path)

    assert store["label"] == tmp_path.name
    assert [record.step for record in store["records"]] == ["build", "top-hits"]
    assert [record.wall_seconds for record in store["records"]] == [1234.5, 42.0]


# --- 6. the comparison with the epic ----------------------------------------


def _evidence(tmp_path: Path, windows: list[Window] | None = None) -> dict:
    store = measure_store(tmp_path)
    return {
        "stores": {store["label"]: {record.step: record for record in store["records"]}},
        "windows": windows or [],
        "reader_ab": [],
    }


def test_the_comparison_reports_a_figure_from_the_record_that_holds_it(tmp_path):
    _record(
        tmp_path,
        "\n".join(
            [
                _line("2026-10-21 23:38:23", "Pass 1: collecting source variants from 2514 files"),
                _line("2026-10-22 00:12:45", "Pass 2: resolving 2514 analyses x 9847698 variants"),
            ]
        )
        + "\n",
    )
    evidence = _evidence(tmp_path)
    evidence["stores"] = {"OGS-00009": evidence["stores"][tmp_path.name]}

    entry = measure_figure(
        Figure(
            key="build_pass1",
            figure="Pass 1",
            epic="34 min",
            epic_seconds=2040.0,
            derive=Derive(
                "interval",
                store="OGS-00009",
                step="build",
                after=r"^Pass 1: collecting",
                until=r"^Pass 2: resolving",
            ),
            method="the two marker lines",
        ),
        evidence,
    )

    assert entry["measured_seconds"] == 2062.0
    assert entry["measured"] == "34m22s"
    assert entry["verdict"] == "reproduces"
    assert entry["reason"] is None


def test_a_progress_figure_is_projected_from_the_clean_pace(tmp_path):
    _record(
        tmp_path,
        "\n".join(
            [
                _line(
                    "2026-10-22 20:00:00",
                    "Pass 2: 100/400 done (last: a) — elapsed 1m40s, ETA 5m00s",
                ),
                _line(
                    "2026-10-22 20:01:40",
                    "Pass 2: 200/400 done (last: b) — elapsed 3m20s, ETA 3m20s",
                ),
                _line(
                    "2026-10-22 20:03:20",
                    "Pass 2: 300/400 done (last: c) — elapsed 5m00s, ETA 1m40s",
                ),
                _line(
                    "2026-10-22 20:05:00",
                    "Pass 2: 400/400 done (last: d) — elapsed 6m40s, ETA 0s",
                ),
            ]
        )
        + "\n",
    )
    evidence = _evidence(tmp_path, [parse_window("2026-10-22T01:06/2026-10-22T16:10")])
    evidence["stores"] = {"OGS-00009": evidence["stores"][tmp_path.name]}

    entry = measure_figure(
        Figure(
            key="pass2",
            figure="Pass 2",
            epic="6m40s",
            epic_seconds=400.0,
            derive=Derive("projection", store="OGS-00009", step="build", label="Pass 2"),
            method="the clean pace over 400",
        ),
        evidence,
    )

    # 300 items in 300 s, projected over the phase's 400: 400 s.
    assert entry["measured_seconds"] == 400.0
    assert entry["detail"]["seconds_per_item"] == 1.0
    assert entry["detail"]["samples_inside_window"] == 0
    assert entry["verdict"] == "reproduces"


def test_a_figure_whose_store_is_not_in_the_run_says_so():
    entry = measure_figure(
        Figure(
            key="x",
            figure="somewhere else",
            epic="1 h",
            epic_seconds=3600.0,
            derive=Derive("step", store="OGS-99999", step="build"),
            method="the step record",
        ),
        {"stores": {}, "windows": [], "reader_ab": []},
    )

    assert entry["measured"] is None
    assert entry["verdict"] == "unavailable"
    assert entry["detail"]["reason"] == "OGS-99999.build is not in this run"
    assert entry["reason"]


def test_every_epic_figure_is_measured_or_says_why_not():
    comparison = compare({"stores": {}, "windows": [], "reader_ab": []})

    assert comparison["rule"]
    assert comparison["epic"]["issue"] == 307
    assert comparison["epic"]["read_at"]
    assert comparison["repeated_figures"]
    assert len(comparison["figures"]) == len(FIGURES) > 30
    for entry in comparison["figures"]:
        assert entry["figure"], entry["key"]
        assert entry["epic"], entry["key"]
        assert entry["method"], entry["key"]
        assert entry["verdict"] == "unavailable", entry["key"]
        assert entry["reason"], entry["key"]


def test_every_figure_names_a_derivation_the_harness_can_dispatch():
    """A kind filed under the record readers but handled elsewhere loses its
    number silently, which is how the paired reader rows once read as absent."""
    known = set(_epic307_figures.RECORD_KINDS) | set(_epic307_figures.OTHER_KINDS)
    assert not set(_epic307_figures.RECORD_KINDS) & set(_epic307_figures.OTHER_KINDS)
    used: set[str] = set()
    for figure in FIGURES:
        used.add(figure.derive.kind)
        used.update(part.kind for _, part in figure.derive.parts)
    assert used <= known, sorted(used - known)


def test_no_reason_text_carries_a_measured_number():
    """A reason is prose; the numbers live in the row's measured fields.

    A hand-typed number in a sentence goes stale the moment the data moves,
    which is what the predicate beside the sentence exists to prevent.
    """
    for figure in FIGURES:
        if figure.reason is None:
            continue
        text, holds = figure.reason
        assert callable(holds), figure.key
        assert not re.search(r"\d+h\d|\d+m\d|\d+\.\d+\s*s|\d+\s*(?:min|seconds?)\b", text), (
            figure.key,
            text,
        )


def test_the_reproduces_rule_is_five_percent_or_a_second_and_honours_a_bound():
    assert reproduces(3600.0, 3600.0, "point") is True
    assert reproduces(3700.0, 3600.0, "point") is True
    assert reproduces(3800.0, 3600.0, "point") is False
    assert reproduces(1.5, 1.0, "point") is True
    assert reproduces(2.5, 1.0, "point") is False
    assert reproduces(25.5, None, "range", (22.0, 29.0)) is True
    assert reproduces(30.0, None, "range", (22.0, 29.0)) is False
    assert reproduces(15356.6, 11400.0, "lower_bound") is True
    assert reproduces(10000.0, 11400.0, "lower_bound") is False
    assert reproduces(None, 3600.0, "point") is False


def test_a_composed_figure_is_a_lower_bound_and_says_what_it_left_out(tmp_path):
    _record(
        tmp_path,
        "\n".join(
            [
                _line("2026-10-22 00:00:00", "Inside: start"),
                _line("2026-10-22 00:10:00", "Inside: done in 10m00s"),
                _line("2026-10-22 20:00:00", "Outside: start"),
                _line("2026-10-22 20:05:00", "Outside: done in 5m00s"),
            ]
        )
        + "\n",
    )
    evidence = _evidence(tmp_path, [parse_window("2026-10-21T23:00/2026-10-22T00:30")])
    evidence["stores"] = {"OGS-00009": evidence["stores"][tmp_path.name]}

    entry = measure_figure(
        Figure(
            key="composed",
            figure="a sum",
            epic="10 min",
            epic_seconds=600.0,
            derive=Derive(
                "composed",
                parts=(
                    ("Inside", Derive("phase", store="OGS-00009", step="build", label="Inside")),
                    ("Outside", Derive("phase", store="OGS-00009", step="build", label="Outside")),
                ),
            ),
            method="the clean components",
            epic_like_for_like=_epic307_figures.LikeForLike(
                "the epic's value for the component it can measure", (("Outside", 300.0),)
            ),
            reason=("a component is left out", lambda detail: bool(detail.get("excluded"))),
        ),
        evidence,
    )

    # The clean component only, and the row is a lower bound, not a smaller
    # figure that pretends to be the whole.
    assert entry["measured_seconds"] == 300.0
    assert entry["verdict"] == "lower_bound"
    assert [part["name"] for part in entry["excluded"]] == ["Inside"]
    assert entry["excluded"][0]["seconds"] == 600.0
    assert entry["excluded"][0]["reason"] == "ran inside a window"
    assert entry["like_for_like"]["verdict"] == "reproduces"
    assert entry["reason"] == "a component is left out"


def test_a_reason_whose_predicate_no_longer_holds_is_not_printed(tmp_path):
    _record(
        tmp_path,
        _line("2026-10-22 20:00:00", "Outside: start")
        + "\n"
        + _line("2026-10-22 20:05:00", "Outside: done in 5m00s"),
    )
    evidence = _evidence(tmp_path, [parse_window("2026-10-22T01:06/2026-10-22T16:10")])
    evidence["stores"] = {"OGS-00009": evidence["stores"][tmp_path.name]}

    entry = measure_figure(
        Figure(
            key="outside",
            figure="outside the window",
            epic="10 min",
            epic_seconds=600.0,
            derive=Derive("phase", store="OGS-00009", step="build", label="Outside"),
            method="the phase's span",
            reason=("it ran inside the window", _inside_window_detail),
        ),
        evidence,
    )

    assert entry["verdict"] == "differs"
    assert entry["inside_window"] is False
    assert entry["reason"] == "reason no longer holds"


def test_an_unavailable_figure_carries_the_derivations_own_reason():
    entry = measure_figure(
        Figure(
            key="gone",
            figure="not on disk",
            epic="1 h",
            epic_seconds=3600.0,
            derive=Derive(
                "unavailable", reason="the record is not on disk", store="S", step="build",
                check="no_phases",
            ),
            method="not available",
            reason=("the record is not on disk", lambda detail: detail["record_present"] is False),
        ),
        {"stores": {}, "windows": [], "reader_ab": []},
    )

    assert entry["verdict"] == "unavailable"
    assert entry["measured"] is None
    assert entry["reason"] == "the record is not on disk"


def _synthetic_finngen(tmp_path: Path) -> Path:
    """A one-row FinnGen-shaped source, enough to exercise the A/B's plumbing."""
    source = tmp_path / "finngen_R13_SYNTHETIC.gz"
    with gzip.open(source, "wt", encoding="utf-8") as fh:
        fh.write("#chrom\tpos\tref\talt\n")
    return source


def _control(tmp_path: Path, fingerprint: str | None = None) -> dict:
    return {
        "module_path": tmp_path,
        "revision": "d405de0",
        "opengwasdb_fingerprint": fingerprint or "fingerprint-reader_base",
    }


class _FakeChild:
    """A `run_child` stand-in: one payload per side, keeping every call's argv."""

    def __init__(self, seconds: dict[str, float], rows: dict[str, int] | None = None) -> None:
        self.seconds = seconds
        self.rows = rows or {}
        self.calls: list[list[str]] = []

    def __call__(self, script: Path, argv: list[str], marker: str) -> dict:
        self.calls.append(argv)
        side = argv[argv.index("--side") + 1]
        return {
            "side": side,
            "seconds": self.seconds[side],
            "rows": self.rows.get(side, 1000),
            "peak_rss_kb": 1234,
            "opengwasdb_path": "/tmp/opengwasdb",
            "opengwasdb_fingerprint": f"fingerprint-{side}",
        }


def test_the_reader_ab_interleaves_its_sides_and_scores_them(tmp_path, monkeypatch):
    source = tmp_path / "finngen_R13_SYNTHETIC.gz"
    with gzip.open(source, "wt", encoding="utf-8") as fh:
        fh.write("#chrom\tpos\tref\talt\tbeta\tsebeta\taf_alt\n")
        fh.write("1\t1\tA\tG\t0.1\t0.2\t0.3\n")
    fake = _FakeChild({"reader_head": 4.0, "reader_base": 8.0, "parse_floor": 1.0})
    monkeypatch.setattr(_reader_ab, "run_child", fake)
    control = {
        "module_path": tmp_path,
        "revision": "d405de0",
        "opengwasdb_fingerprint": "fingerprint-reader_base",
    }

    record = _reader_ab.measure_source(
        "opengwasdb.finngen-r13", source, repetitions=6, control=control, warm=False
    )

    sides = [argv[argv.index("--side") + 1] for argv in fake.calls]
    assert sides == [
        "reader_head",
        "reader_base",
        "parse_floor",
        "parse_floor",
        "reader_base",
        "reader_head",
    ] * 3
    assert record["medians"] == {"reader_head": 4.0, "reader_base": 8.0, "parse_floor": 1.0}
    against_floor = record["comparisons"]["reader_head_vs_parse_floor"]
    assert against_floor["median_ratio"] == 4.0
    assert against_floor["min_ratio"] == against_floor["max_ratio"] == 4.0
    assert against_floor["verdict"] == "slower"
    against_control = record["comparisons"]["reader_head_vs_reader_base"]
    assert against_control["median_ratio"] == 0.5
    assert against_control["verdict"] == "faster"
    assert [round_row["round"] for round_row in against_control["rounds"]] == [0, 1, 2, 3, 4, 5]
    assert record["control"]["revision"] == "d405de0"
    assert record["fingerprints"]["reader_base"] == ["fingerprint-reader_base"] * 6
    assert record["samples"][0]["opengwasdb_fingerprint"] == "fingerprint-reader_head"


def test_the_reader_ab_refuses_a_reader_that_reads_more_rows_than_the_file(tmp_path, monkeypatch):
    source = tmp_path / "ukb-b-99999.vcf.gz"
    source.write_bytes(gzip.compress(b""))
    fake = _FakeChild(
        {"reader_head": 4.0, "parse_floor": 1.0}, rows={"reader_head": 1001, "parse_floor": 1000}
    )
    monkeypatch.setattr(_reader_ab, "run_child", fake)

    with pytest.raises(SystemExit, match="associations from a"):
        _reader_ab.measure_source(
            "opengwasdb.gwas-vcf", source, repetitions=2, warm=False
        )


def test_the_reader_ab_refuses_a_control_that_changes_the_row_count(tmp_path, monkeypatch):
    source = _synthetic_finngen(tmp_path)
    fake = _FakeChild(
        {"reader_head": 4.0, "reader_base": 2.0, "parse_floor": 1.0},
        rows={"reader_head": 1000, "reader_base": 999, "parse_floor": 1000},
    )
    monkeypatch.setattr(_reader_ab, "run_child", fake)

    with pytest.raises(SystemExit, match="is not a speed-up"):
        _reader_ab.measure_source(
            "opengwasdb.finngen-r13",
            source,
            repetitions=2,
            control=_control(tmp_path),
            warm=False,
        )


def test_the_reader_ab_refuses_a_control_that_ran_other_code(tmp_path, monkeypatch):
    """A harness module importing opengwasdb at top level would silently make
    the control this checkout; the fingerprint is what notices."""
    source = _synthetic_finngen(tmp_path)
    fake = _FakeChild({"reader_head": 4.0, "reader_base": 2.0, "parse_floor": 1.0})
    monkeypatch.setattr(_reader_ab, "run_child", fake)

    with pytest.raises(SystemExit, match="was not the code that ran"):
        _reader_ab.measure_source(
            "opengwasdb.finngen-r13",
            source,
            repetitions=2,
            control=_control(tmp_path, "the-archived-revision"),
            warm=False,
        )


def test_the_reader_ab_refuses_a_control_that_is_this_checkout(tmp_path, monkeypatch):
    source = _synthetic_finngen(tmp_path)

    class _SameCode(_FakeChild):
        def __call__(self, script: Path, argv: list[str], marker: str) -> dict:
            payload = super().__call__(script, argv, marker)
            payload["opengwasdb_fingerprint"] = "this-checkout"
            return payload

    monkeypatch.setattr(
        _reader_ab,
        "run_child",
        _SameCode({"reader_head": 4.0, "reader_base": 4.0, "parse_floor": 1.0}),
    )

    with pytest.raises(SystemExit, match="did not take effect"):
        _reader_ab.measure_source(
            "opengwasdb.finngen-r13",
            source,
            repetitions=2,
            control=_control(tmp_path, "this-checkout"),
            warm=False,
        )


def test_a_control_whose_paired_ratios_straddle_one_is_not_resolved(tmp_path, monkeypatch):
    source = _synthetic_finngen(tmp_path)

    class _Mixed(_FakeChild):
        """Head alternates between slower and faster than the control."""

        head_calls = 0

        def __call__(self, script: Path, argv: list[str], marker: str) -> dict:
            payload = super().__call__(script, argv, marker)
            if payload["side"] == "reader_head":
                type(self).head_calls += 1
                payload["seconds"] = 6.0 if type(self).head_calls == 1 else 2.0
            return payload

    fake = _Mixed({"reader_head": 4.0, "reader_base": 4.0, "parse_floor": 1.0})
    monkeypatch.setattr(_reader_ab, "run_child", fake)

    record = _reader_ab.measure_source(
        "opengwasdb.finngen-r13",
        source,
        repetitions=2,
        control=_control(tmp_path),
        warm=False,
    )

    comparison = record["comparisons"]["reader_head_vs_reader_base"]
    assert comparison["verdict"] == "not resolved at this load"
    assert comparison["min_ratio"] < 1 < comparison["max_ratio"]


def test_the_reader_ab_refuses_a_row_count_that_moves_between_repetitions(tmp_path, monkeypatch):
    source = tmp_path / "finngen_R13_SYNTHETIC.gz"
    with gzip.open(source, "wt", encoding="utf-8") as fh:
        fh.write("#chrom\tpos\tref\talt\n")
    calls = {"parse_floor": 0}

    def run(script: Path, argv: list[str], marker: str) -> dict:
        side = argv[argv.index("--side") + 1]
        calls["parse_floor"] += side == "parse_floor"
        rows = 1000 if calls["parse_floor"] == 1 else 999
        return {
            "side": side,
            "seconds": 1.0,
            "rows": 500 if side == "reader_head" else rows,
            "peak_rss_kb": 1,
            "opengwasdb_path": "/tmp/opengwasdb",
            "opengwasdb_fingerprint": "f",
        }

    monkeypatch.setattr(_reader_ab, "run_child", run)

    with pytest.raises(SystemExit, match="different rows"):
        _reader_ab.measure_source(
            "opengwasdb.finngen-r13", source, repetitions=2, warm=False
        )


def test_the_reader_ab_needs_two_repetitions(tmp_path):
    with pytest.raises(ValueError, match="at least two"):
        _reader_ab.measure_source("opengwasdb.finngen-r13", tmp_path / "x", repetitions=1)


def test_a_child_prints_one_result_line_and_names_the_code_it_ran(tmp_path, capsys):
    source = tmp_path / "finngen_R13_SYNTHETIC.gz"
    with gzip.open(source, "wt", encoding="utf-8") as fh:
        fh.write("#chrom\tpos\tref\talt\tbeta\tsebeta\taf_alt\n")
        for index in range(50):
            fh.write(f"1\t{index + 1}\tA\tG\t0.1\t0.2\t0.3\n")

    assert (
        main(
            [
                "--child",
                "--capability",
                "opengwasdb.finngen-r13",
                "--path",
                str(source),
                "--side",
                "parse_floor",
            ]
        )
        == 0
    )

    (line,) = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("READER_AB ")
    ]
    payload = json.loads(line.split(" ", 1)[1])
    assert payload["rows"] == 50
    assert payload["opengwasdb_path"].endswith("/opengwasdb")
    assert len(payload["opengwasdb_fingerprint"]) == 64


def test_an_unknown_source_capability_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="unknown Source Reader Capability"):
        main(
            [
                "--store",
                f"x={tmp_path}",
                "--source",
                f"opengwasdb.nope={tmp_path / 'x'}",
                "--output",
                str(tmp_path / "out.json"),
            ]
        )


def test_parse_window_refuses_text_that_names_no_span():
    with pytest.raises(ValueError, match="/"):
        parse_window("2026-10-22T01:06")


def test_the_artifacts_times_are_log_clock_not_record_clock(tmp_path):
    """A record's UTC times and its local log lines are one clock in the artifact."""
    path = _record(
        tmp_path,
        _line("2026-10-22 00:00:00", "Band-write z: start") + "\n",
        start="2026-10-21T23:00:00.000000Z",
        end="2026-10-22T01:00:00.000000Z",
    )

    step = parse_record(path).to_dict([])

    # The record's UTC start is the log clock's midnight in BST, which is where
    # its first log line sits.
    assert step["start_time"] == "2026-10-21T23:00:00.000000Z"
    assert step["log_start"] == "2026-10-22T00:00:00"
    assert step["log_start"] == step["first_log_line"]
    assert step["span_source"] == "record_times"


def test_the_host_block_records_load_and_uptime():
    from benchmarks.measure_epic307_evidence import _host

    reading = _host()

    assert reading["load_1m"] >= 0.0
    assert reading["uptime_seconds"] > 0.0
    assert reading["hostname"]


def _commit(repo: Path, *tracked: str) -> None:
    """`git init` and one commit holding `tracked`, in `repo`."""
    for command in (
        ["git", "init", "-q"],
        ["git", "add", *tracked],
        ["git", "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "one"],
    ):
        subprocess.run(command, cwd=repo, check=True)


def test_the_harness_refuses_to_write_an_artifact_its_own_commit_holds(tmp_path):
    """The artifact names the commit it was made at, so it must not be re-run
    at the commit that already holds a different one."""
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "evidence.json").write_text("{}", encoding="utf-8")
    _commit(repo, "docs/evidence.json")

    from benchmarks.measure_epic307_evidence import _committed_artifact

    here = os.getcwd()
    try:
        os.chdir(repo)
        assert _committed_artifact(Path("docs/evidence.json")) is True
        assert _committed_artifact(Path("docs/other.json")) is False
    finally:
        os.chdir(here)


def test_a_window_may_not_precede_the_records(tmp_path):
    _record(tmp_path, "", step="build", start="2026-10-22T00:00:00.000000Z")
    output = tmp_path / "out.json"

    with pytest.raises(SystemExit, match="is not after the first record"):
        main(
            [
                "--store",
                f"OGS-00000={tmp_path}",
                "--daily-window",
                "01:06-16:10",
                "--tz",
                "Europe/London",
                "--until",
                "2026-10-01",
                "--output",
                str(output),
            ]
        )


def test_the_one_off_window_is_recorded_with_its_interval(tmp_path):
    _record(tmp_path, _line("2026-10-22 09:00:00", "Validate: start") + "\n")
    output = tmp_path / "out.json"

    assert (
        main(
            [
                "--store",
                f"OGS-00000={tmp_path}",
                "--window",
                "2026-10-22T01:06/2026-10-22T16:10",
                "--daily-window",
                "01:06-16:10",
                "--tz",
                "Europe/London",
                "--until",
                "2026-10-24",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    artifact = json.loads(output.read_text())
    assert artifact["windows"]["one_off"][0]["text"] == "2026-10-22T01:06/2026-10-22T16:10"
    assert artifact["windows"]["daily"]["basis"].startswith("assumed to repeat nightly")
    assert artifact["windows"]["daily"]["nights"] == 2
    assert len(artifact["windows"]["intervals"]) == 3
    assert artifact["comparison"]["figures"]
    assert artifact["host"]["start"]["uptime_seconds"] > 0
    assert artifact["argv"][0] == "--store"


# --- fix round 3 -------------------------------------------------------------
#
# Each test below changes the data a claim rests on and watches the claim go.
# A predicate that cannot fail is the defect these exist to catch.


def _mkstore(tmp_path: Path, label: str, steps: dict[str, str | None], **kwargs) -> dict:
    """A Store directory named `label` holding one record per step."""
    root = tmp_path / label
    for step, stderr in steps.items():
        _record(root, stderr, step=step, start=None, end=None, **kwargs)
    return measure_store(root, label=label)


def _ev(*stores: dict, windows: list[Window] | None = None, **extra) -> dict:
    return {
        "stores": {
            store["label"]: {record.step: record for record in store["records"]}
            for store in stores
        },
        "windows": windows or [],
        "reader_ab": [],
        **extra,
    }


def _figure(key: str) -> Figure:
    return next(figure for figure in FIGURES if figure.key == key)


def _window(text: str, date: str | None = None) -> Window:
    window = parse_window(text)
    return Window(window.start, window.end, window.text, date)


NIGHT = "2026-10-22T01:06/2026-10-22T16:10"


# H3: a window's seconds are a union, never a sum


def test_a_one_off_window_and_the_daily_window_for_the_same_night_count_once(tmp_path):
    store = _mkstore(
        tmp_path,
        "OGS-00009",
        {
            "build": _line("2026-10-22 02:00:00", "Band-write z: start")
            + "\n"
            + _line("2026-10-22 04:00:00", "Band-write z: done in 2h00m")
            + "\n"
        },
    )
    windows = [_window(NIGHT), _window(NIGHT, "2026-10-22")]

    entry = measure_figure(
        Figure("s", "a step", "2 h", 7200.0, Derive("step", store="OGS-00009", step="build"), "m"),
        _ev(store, windows=windows),
    )

    assert entry["detail"]["seconds_inside_windows"] == 7200.0
    assert entry["detail"]["span_seconds"] == 7200.0


def test_a_step_forty_percent_inside_the_window_is_not_mostly_inside(tmp_path):
    # 22:06 on 21 Oct to 03:06 on 22 Oct is five hours; the window covers the last two.
    store = _mkstore(
        tmp_path,
        "OGS-00009",
        {
            "complete": _line("2026-10-21 22:06:00", "Reference completion: start")
            + "\n"
            + _line("2026-10-22 03:06:00", "Reference completion: done in 5h00m")
            + "\n"
        },
    )
    windows = [_window(NIGHT), _window(NIGHT, "2026-10-22")]

    entry = measure_figure(_figure("completion"), _ev(store, windows=windows))

    assert entry["detail"]["seconds_inside_windows"] == 7200.0
    assert entry["detail"]["span_seconds"] == 18000.0
    assert entry["verdict"] == "differs"
    assert entry["reason"] == "reason no longer holds"


def test_a_step_sixty_percent_inside_the_window_is_mostly_inside(tmp_path):
    store = _mkstore(
        tmp_path,
        "OGS-00009",
        {
            "complete": _line("2026-10-21 23:06:00", "Reference completion: start")
            + "\n"
            + _line("2026-10-22 04:06:00", "Reference completion: done in 5h00m")
            + "\n"
        },
    )

    entry = measure_figure(_figure("completion"), _ev(store, windows=[_window(NIGHT)]))

    # 01:06 to 04:06 is three of the five hours.
    assert entry["detail"]["seconds_inside_windows"] == 10800.0
    assert entry["detail"]["span_seconds"] == 18000.0
    assert entry["reason"] and "no longer holds" not in entry["reason"]


def test_phase_seconds_inside_windows_count_nested_phases_once_and_unknown_as_unknown(tmp_path):
    nested = _mkstore(
        tmp_path,
        "nested",
        {
            "build": "\n".join(
                [
                    _line("2026-10-22 02:00:00", "Outer: start"),
                    _line("2026-10-22 02:30:00", "Inner: start"),
                    _line("2026-10-22 03:00:00", "Inner: done in 30m00s"),
                    _line("2026-10-22 04:00:00", "Outer: done in 2h00m"),
                ]
            )
            + "\n"
        },
    )
    unfinished = _mkstore(
        tmp_path,
        "unfinished",
        {
            "build": "\n".join(
                [
                    _line("2026-10-22 02:00:00", "Outer: start"),
                    _line("2026-10-22 04:00:00", "Other: start"),
                    _line("2026-10-22 05:00:00", "Other: done in 1h00m"),
                    _line("2026-10-22 06:00:00", "Orphan: start"),
                ]
            )
            + "\n"
        },
    )
    step = Figure("s", "s", "x", 1.0, Derive("step", store="x", step="build"), "m")

    def measured(store: dict, label: str) -> dict:
        evidence = _ev(store, windows=[_window(NIGHT)])
        evidence["stores"] = {"x": evidence["stores"][label]}
        return measure_figure(step, evidence)["detail"]

    assert measured(nested, "nested")["phases_seconds_inside_windows"] == 7200.0
    detail = measured(unfinished, "unfinished")
    assert detail["phases_seconds_inside_windows"] is None
    assert detail["phases_unknown"] == 2


# H2: a like-for-like sum names the parts it holds


def _three_part_figure(**overrides) -> Figure:
    parts = tuple(
        (name, Derive("phase", store="S", step="build", label=name)) for name in ("A", "B", "C")
    )
    spec = {
        "key": "sum",
        "figure": "a sum",
        "epic": "5 min",
        "epic_seconds": 300.0,
        "derive": Derive("composed", parts=parts),
        "method": "the clean parts",
        "epic_like_for_like": _epic307_figures.LikeForLike(
            "the epic's values for A, B and C", (("A", 100.0), ("B", 100.0), ("C", 100.0))
        ),
    }
    spec.update(overrides)
    return Figure(**spec)


def _abc(tmp_path: Path, windows: list[Window] | None) -> dict:
    lines = []
    for index, name in enumerate("ABC"):
        hour = 18 + index
        lines += [
            _line(f"2026-10-22 {hour}:00:00", f"{name}: start"),
            _line(f"2026-10-22 {hour}:01:40", f"{name}: done in 1m40s"),
        ]
    store = _mkstore(tmp_path, "S", {"build": "\n".join(lines) + "\n"})
    return _ev(store, windows=windows)


def test_a_like_for_like_sum_with_every_part_compares_against_the_epics_parts(tmp_path):
    entry = measure_figure(_three_part_figure(), _abc(tmp_path, [_window(NIGHT)]))

    like = entry["like_for_like"]
    assert like["verdict"] == "reproduces"
    assert like["epic_seconds"] == 300.0
    assert like["missing"] == [] and like["unexpected"] == []


def test_a_like_for_like_sum_missing_a_part_is_incomplete_not_reproduces(tmp_path):
    # Part A runs inside the window, so the sum holds B and C and is 200 s.
    windows = [_window("2026-10-22T17:30/2026-10-22T18:30")]

    entry = measure_figure(_three_part_figure(), _abc(tmp_path, windows))

    like = entry["like_for_like"]
    assert entry["measured_seconds"] == 200.0
    assert like["verdict"] == "incomplete"
    assert like["missing"] == ["A"]
    assert like["epic_seconds"] is None


def test_a_like_for_like_sum_with_an_extra_part_is_not_compared_either(tmp_path):
    figure = _three_part_figure(
        epic_like_for_like=_epic307_figures.LikeForLike(
            "A and B only", (("A", 100.0), ("B", 100.0))
        )
    )

    like = measure_figure(figure, _abc(tmp_path, [_window(NIGHT)]))["like_for_like"]

    assert like["verdict"] == "incomplete"
    assert like["unexpected"] == ["C"]


def test_the_build_totals_like_for_like_names_the_six_parts_the_epic_summed():
    like = _figure("build_clean").epic_like_for_like
    assert like is not None
    parts = {name for name, _ in _figure("build_clean").derive.parts}
    names = {name for name, _ in like.components}
    assert names <= parts and len(names) == 6
    assert sum(seconds for _, seconds in like.components) == 16380.0
    assert "Pass 1, liftover, key lookup" in names


# H4: a predicate that always passes is not a predicate


def _finngen_ab(revision: str = "d405de0a7e1be29b1fc1d2e0dfb1090f607ba08c", rows: int = 21326102):
    return {
        "label": "finngen_R13_AB1_ACTINOMYCOSIS",
        "n_rows": rows,
        "medians": {"reader_base": 173.6, "reader_head": 176.5, "parse_floor": 18.65},
        "samples": [
            {"side": "reader_base", "seconds": 173.6},
            {"side": "parse_floor", "seconds": 18.65},
        ],
        "control": {"revision": revision, "fingerprint": "f"},
        "comparisons": {},
    }


def test_the_control_rows_reason_needs_the_epics_revision_and_the_epics_row_count():
    figure = _figure("reader_finngen")

    held = measure_figure(figure, _ev(reader_ab=[_finngen_ab()]))
    other_revision = measure_figure(figure, _ev(reader_ab=[_finngen_ab(revision="abc1234ffff")]))
    other_rows = measure_figure(figure, _ev(reader_ab=[_finngen_ab(rows=21326101)]))

    assert held["verdict"] == "differs"
    assert held["reason"] and "no longer holds" not in held["reason"]
    assert other_revision["reason"] == "reason no longer holds"
    assert other_rows["reason"] == "reason no longer holds"


def _build_with_pass1(tmp_path: Path, windows: list[Window]) -> dict:
    store = _mkstore(
        tmp_path,
        "OGS-00009",
        {
            "build": _line("2026-10-22 02:00:00", "Pass 1: collecting source variants from 3 files")
            + "\n"
            + _line("2026-10-22 03:00:00", "Pass 2: resolving 3 analyses x 9 variants")
            + "\n"
        },
    )
    return _ev(store, windows=windows)


def test_the_build_totals_reason_needs_a_part_that_ran_inside_a_window(tmp_path):
    figure = _figure("build_clean")

    inside = measure_figure(figure, _build_with_pass1(tmp_path / "a", [_window(NIGHT)]))
    outside = measure_figure(figure, _build_with_pass1(tmp_path / "b", []))

    assert inside["verdict"] == outside["verdict"] == "lower_bound"
    assert inside["reason"] and "no longer holds" not in inside["reason"]
    # Every part but Pass 1 is missing from the record, which is not the same as
    # having run inside the window.
    assert outside["reason"] == "reason no longer holds"


def _unavailable(key: str, evidence: dict) -> dict:
    entry = measure_figure(_figure(key), evidence)
    assert entry["verdict"] == "unavailable", entry
    return entry


def test_an_unavailable_reason_about_a_record_needs_that_record_to_show_it(tmp_path):
    bare = _mkstore(tmp_path / "a", "OGS-00009", {"validate": _line("2026-10-20 01:00:00", "x")})
    phased = _mkstore(
        tmp_path / "b",
        "OGS-00009",
        {
            "validate": _line("2026-10-20 01:00:00", "Subset comparison: start")
            + "\n"
            + _line("2026-10-20 02:00:00", "Subset comparison: done in 1h00m")
        },
    )

    held = _unavailable("ogs00009_subset_check", _ev(bare))
    moved = _unavailable("ogs00009_subset_check", _ev(phased))
    absent = _unavailable("ogs00009_subset_check", _ev())

    assert held["reason"] and "no longer holds" not in held["reason"]
    assert moved["reason"] == "reason no longer holds"
    assert absent["reason"] == "reason no longer holds"


@pytest.mark.parametrize(
    ("key", "phase"),
    [("ogs00012_completion_block", "Completion block"), ("ogs00012_band_write", "Band write")],
)
def test_a_completion_row_without_its_phase_needs_the_record_to_lack_it(tmp_path, key, phase):
    bare = _mkstore(
        tmp_path / "a",
        "OGS-00012",
        {"complete": _line("2026-10-22 01:00:00", "SE rewrite: start") + "\n"},
    )
    phased = _mkstore(
        tmp_path / "b",
        "OGS-00012",
        {
            "complete": _line("2026-10-22 01:00:00", f"{phase}: start")
            + "\n"
            + _line("2026-10-22 02:00:00", f"{phase}: done in 1h00m")
            + "\n"
        },
    )

    held = _unavailable(key, _ev(bare))
    moved = _unavailable(key, _ev(phased))

    assert held["reason"] and "no longer holds" not in held["reason"]
    assert moved["reason"] == "reason no longer holds"


def test_the_pyspy_row_needs_no_record_to_mention_pyspy(tmp_path):
    quiet = _mkstore(tmp_path / "a", "OGS-00009", {"build": _line("2026-10-22 01:00:00", "x")})
    loud = _mkstore(
        tmp_path / "b",
        "OGS-00009",
        {"build": _line("2026-10-22 01:00:00", "py-spy dump of worker 3")},
    )

    held = _unavailable("pyspy_shares", _ev(quiet))
    moved = _unavailable("pyspy_shares", _ev(loud))
    empty = _unavailable("pyspy_shares", _ev())

    assert held["reason"] and "no longer holds" not in held["reason"]
    assert moved["reason"] == "reason no longer holds"
    assert empty["reason"] == "reason no longer holds"


def test_the_after_the_change_row_needs_every_record_to_predate_the_epic(tmp_path):
    before = _mkstore(
        tmp_path / "a", "OGS-00009", {"build": _line("2026-11-06 23:59:00", "x") + "\n"}
    )
    after = _mkstore(
        tmp_path / "b", "OGS-00009", {"build": _line("2026-11-08 00:01:00", "x") + "\n"}
    )

    held = _unavailable("after_steps_2_to_4", _ev(before))
    moved = _unavailable("after_steps_2_to_4", _ev(after))

    assert held["reason"] and "no longer holds" not in held["reason"]
    assert moved["reason"] == "reason no longer holds"


def test_a_roll_up_is_not_re_derived_only_while_a_part_is_not_a_clean_reproduction():
    figure = _figure("completion_total")

    open_part = measure_figure(
        figure, _ev(verdicts={"completion": "differs", "completion_validate": "reproduces"})
    )
    all_clean = measure_figure(
        figure, _ev(verdicts={"completion": "reproduces", "completion_validate": "reproduces"})
    )

    assert open_part["reason"] and "no longer holds" not in open_part["reason"]
    assert all_clean["reason"] == "reason no longer holds"


def test_an_unavailable_derivation_must_name_what_it_checks():
    with pytest.raises(ValueError, match="check"):
        measure_figure(
            Figure("g", "gone", "1 h", 3600.0, Derive("unavailable", reason="stored text"), "m"),
            _ev(),
        )


def test_every_unavailable_figure_has_a_check_and_a_predicate():
    for figure in FIGURES:
        if figure.derive.kind == "unavailable":
            assert figure.derive.check, figure.key
            assert figure.reason is not None, figure.key


# Spec 5 / AC2: the reader-to-floor ratio, beside the epic's


def _ratio_ab(control: float, floor: float, label: str = "finngen_R13_AB1_ACTINOMYCOSIS") -> dict:
    return {
        "label": label,
        "n_rows": 1,
        "medians": {"reader_base": control, "parse_floor": floor, "reader_head": control},
        "samples": [],
        "control": {"revision": "d405de0", "fingerprint": "f"},
        "comparisons": {},
    }


def test_the_control_to_floor_ratio_is_reported_beside_the_epics():
    figure = _figure("reader_finngen_control_over_floor")

    entry = measure_figure(figure, _ev(reader_ab=[_ratio_ab(173.6222, 18.6544)]))

    assert entry["epic"].startswith("7.78")
    assert entry["measured_seconds"] == pytest.approx(9.307, abs=0.001)
    assert entry["measured"] == "9.31x"
    assert entry["verdict"] == "differs"
    assert entry["detail"]["epic_control_seconds"] == 205.5
    assert entry["detail"]["epic_floor_seconds"] == 26.4
    assert entry["reason"] and "no longer holds" not in entry["reason"]


def test_the_ratio_reason_is_withdrawn_when_the_floor_did_not_move_more_than_the_control():
    figure = _figure("reader_finngen_control_over_floor")

    # The floor is the epic's own, and the control sped up: the ratio is lower.
    entry = measure_figure(figure, _ev(reader_ab=[_ratio_ab(173.6, 26.4)]))

    assert entry["verdict"] == "differs"
    assert entry["reason"] == "reason no longer holds"


def test_a_ratio_is_not_judged_on_a_second_of_slack():
    # 2.44 against 1.96 is a quarter apart; a one-second tolerance would call them equal.
    assert reproduces(2.44, 1.96, "point", slack=0.0) is False
    entry = measure_figure(
        _figure("reader_ukb_control_over_floor"),
        _ev(reader_ab=[_ratio_ab(13.5754, 5.5694, "ukb-b-10001")]),
    )
    assert entry["verdict"] == "differs"
    assert entry["measured"] == "2.44x"


def test_a_per_analysis_figure_is_not_judged_on_a_second_of_slack(tmp_path):
    store = _mkstore(
        tmp_path,
        "OGS-00009",
        {
            "build": "\n".join(
                [
                    _line(
                        "2026-10-22 02:00:00",
                        "Pass 2: 100/400 done (last: a) — elapsed 3m20s, ETA 5m00s",
                    ),
                    _line(
                        "2026-10-22 02:00:10",
                        "Pass 2: 200/400 done (last: b) — elapsed 6m40s, ETA 3m20s",
                    ),
                ]
            )
            + "\n"
        },
    )

    entry = measure_figure(_figure("pass2_before_window"), _ev(store, windows=[_window(NIGHT)]))

    # 2.0 s per Analysis against the epic's 1.53 is a 30% miss.
    assert entry["measured_seconds"] == 2.0
    assert entry["verdict"] == "differs"


# H6: OGS-00016 Pass 2's clean interval, against the epic's range


def _pass2_lines(rates: list[int]) -> str:
    """Pass 2 progress lines, one per interval, `rates` seconds per Analysis each."""
    lines = []
    when = datetime(2026, 10, 22, 17, 0, 0)
    done = 0
    lines.append(
        _line(
            when.isoformat(sep=" "),
            f"Pass 2: {done}/{10 * len(rates)} done (last: a) — elapsed 0s, ETA 1h",
        )
    )
    from datetime import timedelta

    for rate in rates:
        when += timedelta(seconds=rate * 10)
        done += 10
        lines.append(
            _line(
                when.isoformat(sep=" "),
                f"Pass 2: {done}/{10 * len(rates)} done (last: a) — elapsed 1h, ETA 1h",
            )
        )
    return "\n".join(lines) + "\n"


def _pass2_row(tmp_path: Path, rates: list[int]) -> dict:
    store = _mkstore(tmp_path, "OGS-00016", {"build": _pass2_lines(rates)})
    return measure_figure(_figure("ogs00016_pass2_clean_interval"), _ev(store))


def test_the_ogs00016_pass2_row_is_the_median_clean_interval_against_the_epics_range(tmp_path):
    entry = _pass2_row(tmp_path, [25, 26, 25, 24, 25, 26, 25, 1000])

    assert entry["measured_seconds"] == 25.0
    assert entry["verdict"] == "reproduces"
    assert entry["detail"]["pooled_seconds_per_item"] > 100
    (slow,) = entry["detail"]["slow_intervals"]
    assert slow["seconds_per_item"] == 1000
    assert slow["cause"] == "unknown"
    assert entry["notes"] and "no longer holds" not in entry["notes"][0]


def test_the_ogs00016_pass2_row_differs_outside_the_range_and_drops_its_note_without_slow_intervals(
    tmp_path,
):
    entry = _pass2_row(tmp_path, [40, 41, 40, 39, 40, 41])

    assert entry["measured_seconds"] == 40.0
    assert entry["verdict"] == "differs"
    assert entry["detail"]["slow_intervals"] == []
    assert entry["notes"] == ["note no longer holds"]


# Spec 4: the EAF survey ran different code in September and in October


def _survey_store(
    tmp_path: Path, label: str, *, done: str, plan: str, total: int, revision: str
) -> dict:
    return _mkstore(
        tmp_path,
        label,
        {
            "build": _line(done, f"Pass 2: {total}/{total} done (last: a) — elapsed 1h, ETA 0s")
            + "\n"
            + _line(plan, "Encoding plan: {'version': 3}")
            + "\n"
        },
        extra={"opengwasdb_rev": revision},
    )


def _ancestry(contains: dict[tuple[str, str], bool | None]):
    return lambda ancestor, descendant: contains.get((ancestor, descendant))


def test_the_october_survey_note_needs_the_parallel_survey_and_a_window_all_around_it(tmp_path):
    revision = "d405de0a7e1be29b1fc1d2e0dfb1090f607ba08c"
    inside = _survey_store(
        tmp_path / "a", "OGS-00009", done="2026-10-22 02:00:00", plan="2026-10-22 05:32:30",
        total=2514, revision=revision,
    )
    partly = _survey_store(
        tmp_path / "b", "OGS-00009", done="2026-10-22 14:00:00", plan="2026-10-22 17:32:30",
        total=2514, revision=revision,
    )
    parallel = _ancestry({("ea08bec", revision): True})
    serial = _ancestry({("ea08bec", revision): False})
    windows = [_window(NIGHT)]
    figure = _figure("eaf_survey_oct")

    held = measure_figure(figure, _ev(inside, windows=windows, is_ancestor=parallel))
    only_partly = measure_figure(figure, _ev(partly, windows=windows, is_ancestor=parallel))
    old_code = measure_figure(figure, _ev(inside, windows=windows, is_ancestor=serial))
    unknown = measure_figure(figure, _ev(inside, windows=windows, is_ancestor=_ancestry({})))

    assert held["notes"] and "no longer holds" not in held["notes"][0]
    assert "wholly" in held["notes"][0] and "parallel" in held["notes"][0]
    assert only_partly["notes"] == ["note no longer holds"]
    assert old_code["notes"] == ["note no longer holds"]
    assert unknown["notes"] == ["note no longer holds"]


def test_the_september_survey_note_needs_the_serial_survey_and_no_window(tmp_path):
    revision = "f8125322f9d9b71db8b05d1c18d507612bf654e2"
    sept = _survey_store(
        tmp_path / "a", "OGS-00009-sept", done="2026-09-19 19:54:06", plan="2026-09-19 21:49:39",
        total=2024, revision=revision,
    )
    figure = _figure("eaf_survey_sept")
    serial = _ancestry({("ea08bec", revision): False})
    parallel = _ancestry({("ea08bec", revision): True})

    held = measure_figure(figure, _ev(sept, is_ancestor=serial))
    moved = measure_figure(figure, _ev(sept, is_ancestor=parallel))
    in_window = measure_figure(
        figure,
        _ev(sept, windows=[_window("2026-09-19T20:00/2026-09-19T22:00")], is_ancestor=serial),
    )

    assert held["measured_seconds"] == 6933.0
    assert held["verdict"] == "reproduces"
    assert "serial" in held["notes"][0] and "no longer holds" not in held["notes"][0]
    assert moved["notes"] == ["note no longer holds"]
    assert in_window["notes"] == ["note no longer holds"]


def test_the_survey_row_that_scales_september_names_what_ran_in_october(tmp_path):
    revision = "d405de0a7e1be29b1fc1d2e0dfb1090f607ba08c"
    october = _survey_store(
        tmp_path, "OGS-00009", done="2026-10-22 02:00:00", plan="2026-10-22 05:32:30",
        total=2514, revision=revision,
    )

    entry = measure_figure(
        _figure("build_eaf_survey"),
        _ev(
            october,
            windows=[_window(NIGHT)],
            is_ancestor=_ancestry({("ea08bec", revision): True}),
        ),
    )

    assert entry["verdict"] == "differs"
    assert "parallel" in entry["reason"] and "wholly" in entry["reason"]
    assert "serial" in entry["reason"]


def test_ancestry_is_asked_of_git(tmp_path):
    from benchmarks._artifact import is_ancestor

    def git(*argv: str) -> str:
        return subprocess.run(
            ["git", "-c", "user.email=a@b", "-c", "user.name=a", *argv],
            cwd=tmp_path, check=True, capture_output=True, text=True,
        ).stdout.strip()

    git("init", "-q")
    (tmp_path / "f").write_text("1")
    git("add", "f")
    git("commit", "-qm", "one")
    first = git("rev-parse", "HEAD")
    (tmp_path / "f").write_text("2")
    git("commit", "-qam", "two")
    second = git("rev-parse", "HEAD")

    assert is_ancestor(first, second, repo=tmp_path) is True
    assert is_ancestor(second, first, repo=tmp_path) is False
    assert is_ancestor("0" * 40, second, repo=tmp_path) is None


# H5: the A/B's verdict is a paired test


def test_the_sign_test_and_wilcoxon_p_match_hand_worked_values():
    sign = _reader_ab.sign_test_p
    wilcoxon = _reader_ab.wilcoxon_p

    # Nine of ten above one: 2 * (1 + 10) / 1024.
    assert sign([1.1] * 9 + [0.9]) == pytest.approx(22 / 1024)
    # Six of six: 2 / 64. Four of four is 2 / 16, which no one would call a result.
    assert sign([1.1] * 6) == pytest.approx(2 / 64)
    assert sign([1.1] * 4) == pytest.approx(2 / 16)
    # The one below is also the smallest in magnitude, rank 1 of 10: 4 / 1024.
    ratios = [1.01 + 0.01 * index for index in range(9)]
    assert wilcoxon([0.99, *[1.0 + 0.1 * (index + 1) for index in range(9)]]) == pytest.approx(
        4 / 1024
    )
    assert wilcoxon([1.1] * 6) == pytest.approx(2 / 64)
    assert len(ratios) == 9
    # A tie at one drops out rather than voting either way.
    assert sign([1.0, 1.0, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]) == pytest.approx(2 / 64)


def _paired(ratios: list[float]) -> dict:
    samples = []
    for index, ratio in enumerate(ratios):
        samples += [
            {"round": index, "side": "reader_head", "seconds": 10.0 * ratio, "load_1m_before": 1.0},
            {"round": index, "side": "reader_base", "seconds": 10.0, "load_1m_before": 1.0},
        ]
    return _reader_ab._paired(samples, "reader_head", "reader_base")


def test_a_direction_needs_six_even_rounds_and_both_p_values_under_five_percent():
    six_slower = _paired([1.05, 1.02, 1.08, 1.03, 1.04, 1.06])
    assert six_slower["verdict"] == "slower"
    assert six_slower["sign_test_p"] == pytest.approx(2 / 64)
    assert six_slower["n_rounds"] == 6

    assert _paired([0.95, 0.9, 0.92, 0.97, 0.94, 0.96])["verdict"] == "faster"
    # The same shape on three rounds proves nothing: all-on-one-side is p = 0.25.
    assert _paired([1.05, 1.02, 1.08])["verdict"] == "not resolved at this load"
    assert _paired([1.05, 1.02, 1.08, 1.03])["verdict"] == "not resolved at this load"
    # An odd number of rounds cannot be balanced for who ran first.
    assert _paired([1.05, 1.02, 1.08, 1.03, 1.04, 1.06, 1.07])["verdict"] == (
        "not resolved at this load"
    )
    # Four of six above one: p = 0.69.
    mixed = _paired([1.05, 0.98, 1.08, 0.97, 1.04, 1.06])
    assert mixed["verdict"] == "not resolved at this load"
    assert mixed["above_one"] == 4


def test_the_ukb_round_two_data_resolves_as_slower():
    ratios = [1.08, 1.12, 1.05, 1.09, 1.07, 1.10, 1.06, 1.08, 1.04, 0.97]

    comparison = _paired(ratios)

    assert comparison["above_one"] == 9
    assert comparison["sign_test_p"] == pytest.approx(0.0215, abs=0.0001)
    assert comparison["wilcoxon_p"] < 0.05
    assert comparison["verdict"] == "slower"
    assert comparison["median_ratio"] == 1.075


def test_the_head_leads_the_control_in_half_the_rounds(tmp_path, monkeypatch):
    source = _synthetic_finngen(tmp_path)
    fake = _FakeChild({"reader_head": 4.0, "reader_base": 4.1, "parse_floor": 1.0})
    monkeypatch.setattr(_reader_ab, "run_child", fake)

    record = _reader_ab.measure_source(
        "opengwasdb.finngen-r13", source, repetitions=6, control=_control(tmp_path), warm=False
    )

    comparison = record["comparisons"]["reader_head_vs_reader_base"]
    assert comparison["head_first_rounds"] == 3
    assert [entry["head_first"] for entry in comparison["rounds"]] == [True, False] * 3


def test_a_paired_row_prints_the_p_values_the_verdict_rests_on():
    ab = {
        "label": "ukb-b-10001",
        "n_rows": 1,
        "medians": {"reader_head": 14.4, "reader_base": 13.5},
        "samples": [],
        "control": {"revision": "d405de0", "fingerprint": "f"},
        "comparisons": {
            "reader_head_vs_reader_base": _paired([1.05, 1.02, 1.08, 1.03, 1.04, 1.06])
        },
    }

    entry = measure_figure(_figure("reader_ukb_head_vs_control"), _ev(reader_ab=[ab]))

    assert entry["verdict"] == "slower"
    assert entry["detail"]["sign_test_p"] == pytest.approx(2 / 64)
    assert entry["detail"]["wilcoxon_p"] == pytest.approx(2 / 64)
    assert entry["detail"]["n_rounds"] == 6


# N5: the control's fingerprint is the archive's, not a tree on disk's


def _repo_with_package(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "opengwasdb").mkdir(parents=True)
    (repo / "opengwasdb" / "__init__.py").write_text("A = 1\n")
    (repo / "opengwasdb" / "a.py").write_text("B = 2\n")
    _commit(repo, "opengwasdb")
    return repo


def test_a_control_revision_is_exported_and_fingerprinted_from_git_archive(tmp_path):
    import hashlib

    from benchmarks._artifact import export_revision

    repo = _repo_with_package(tmp_path)
    expected = hashlib.sha256(
        b"__init__.py" + b"A = 1\n" + b"a.py" + b"B = 2\n"
    ).hexdigest()
    # An uncommitted edit must not reach the fingerprint: the archive is the commit's.
    (repo / "opengwasdb" / "a.py").write_text("B = 3\n")

    control = export_revision("HEAD", tmp_path / "control", repo=repo)

    assert control["opengwasdb_fingerprint"] == expected
    assert control["revision"] == subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert (tmp_path / "control" / "opengwasdb" / "a.py").read_text() == "B = 2\n"
    assert control["module_path"] == tmp_path / "control"


def test_an_unknown_control_revision_is_refused(tmp_path):
    from benchmarks._artifact import export_revision

    repo = _repo_with_package(tmp_path)

    with pytest.raises(SystemExit, match="not a commit"):
        export_revision("0" * 40, tmp_path / "control", repo=repo)


def test_the_harness_takes_a_control_revision_and_no_control_tree():
    from benchmarks.measure_epic307_evidence import _parser

    options = {action.dest for action in _parser()._actions}
    assert "control_revision" in options
    assert "control_module_path" not in options


# J1: the output-path guard resolves the path, needs a work tree, runs first


def test_the_committed_artifact_guard_resolves_an_absolute_output(tmp_path):
    from benchmarks.measure_epic307_evidence import _committed_artifact

    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "evidence.json").write_text("{}", encoding="utf-8")
    _commit(repo, "docs/evidence.json")

    here = os.getcwd()
    try:
        os.chdir(repo)
        assert _committed_artifact(repo / "docs" / "evidence.json") is True
        assert _committed_artifact(repo / "docs" / ".." / "docs" / "evidence.json") is True
        assert _committed_artifact(repo / "docs" / "other.json") is False
        assert _committed_artifact(tmp_path / "elsewhere.json") is False
    finally:
        os.chdir(here)


def test_the_guard_refuses_to_run_outside_a_git_work_tree(tmp_path):
    from benchmarks.measure_epic307_evidence import _committed_artifact

    here = os.getcwd()
    try:
        os.chdir(tmp_path)
        with pytest.raises(SystemExit, match="git work tree"):
            _committed_artifact(Path("out.json"))
    finally:
        os.chdir(here)


def test_the_committed_artifact_is_refused_before_any_reader_samples(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "evidence.json").write_text("{}", encoding="utf-8")
    _commit(repo, "docs/evidence.json")
    _record(tmp_path / "store", _line("2026-10-22 09:00:00", "Validate: start") + "\n")
    source = _synthetic_finngen(tmp_path)

    def must_not_run(*args, **kwargs):
        raise AssertionError("the A/B ran before the output path was checked")

    monkeypatch.setattr(_reader_ab, "measure_source", must_not_run)
    here = os.getcwd()
    try:
        os.chdir(repo)
        with pytest.raises(SystemExit, match="already committed"):
            main(
                [
                    "--store", f"OGS-00000={tmp_path / 'store'}",
                    "--source", f"opengwasdb.finngen-r13={source}",
                    "--output", str(repo / "docs" / "evidence.json"),
                ]
            )
    finally:
        os.chdir(here)


# the daily window is declared from the night the epic observed it


def test_a_daily_window_starts_when_it_is_declared_to_start(tmp_path):
    _record(tmp_path, _line("2026-09-20 09:00:00", "Validate: start") + "\n", start=None, end=None)
    output = tmp_path / "out.json"

    assert (
        main(
            [
                "--store", f"OGS-00000={tmp_path}",
                "--daily-window", "01:06-16:10",
                "--tz", "Europe/London",
                "--daily-from", "2026-10-22",
                "--until", "2026-10-24",
                "--output", str(output),
            ]
        )
        == 0
    )

    artifact = json.loads(output.read_text())
    assert artifact["windows"]["daily"]["from"] == "2026-10-22"
    assert artifact["windows"]["daily"]["nights"] == 2
    assert artifact["stores"][0]["steps"][0]["overlaps_window"] is False


# H1: the README points at the artifact for its numbers


def test_the_readme_does_not_restate_the_artifacts_load():
    readme = (Path(__file__).resolve().parent.parent / "benchmarks" / "README.md").read_text()
    start = readme.index("### `measure_epic307_evidence.py`")
    following = readme.find("\n### ", start + 5)
    section = readme[start:] if following < 0 else readme[start:following]

    assert not re.search(r"\bload\b[^.|\n]*\d", section), "the README quotes a load"
    assert "host" in section


# fix round 4


def _inside_pass2(tmp_path: Path, lines: list[tuple[str, int]], windows: list[Window]) -> dict:
    """OGS-00016 Pass 2 progress lines, `(time, completed)`, against `windows`."""
    stderr = "\n".join(
        _line(at, f"Pass 2: {done}/100 done (last: a) — elapsed 1h, ETA 1h") for at, done in lines
    )
    store = _mkstore(tmp_path, "OGS-00016", {"build": stderr + "\n"})
    return measure_figure(_figure("ogs00016_pass2_inside_interval"), _ev(store, windows=windows))


def test_the_inside_row_is_the_median_and_range_of_intervals_that_overlap_a_window(tmp_path):
    lines = [
        ("2026-10-22 00:00:00", 0),
        ("2026-10-22 02:00:00", 10),  # straddles the window's start: 720 s each
        ("2026-10-22 03:00:00", 15),  # wholly inside: 720 s each
        ("2026-10-22 05:00:00", 20),  # wholly inside: 1,440 s each
        ("2026-10-22 18:00:00", 30),  # straddles the window's end: 4,680 s each
    ]

    entry = _inside_pass2(tmp_path, lines, [_window(NIGHT)])

    detail = entry["detail"]
    assert detail["intervals"] == 4
    assert detail["straddling_intervals"] == 2
    assert detail["min_seconds_per_item"] == 720.0
    assert detail["max_seconds_per_item"] == 4680.0
    assert entry["measured_seconds"] == 1080.0
    assert entry["verdict"] == "reproduces"


def test_the_inside_rows_reason_needs_an_interval_that_straddles_a_windows_edge(tmp_path):
    slow = [("2026-10-22 02:00:00", 0), ("2026-10-22 03:00:00", 2), ("2026-10-22 04:00:00", 4)]
    straddling = [
        ("2026-10-22 00:00:00", 0),
        ("2026-10-22 02:00:00", 2),
        ("2026-10-22 03:00:00", 4),
    ]
    windows = [_window(NIGHT)]

    wholly = _inside_pass2(tmp_path / "a", slow, windows)
    edge = _inside_pass2(tmp_path / "b", straddling, windows)

    # 1,800 s per Analysis is above the epic's range in both; only the second straddles.
    assert wholly["verdict"] == edge["verdict"] == "differs"
    assert wholly["detail"]["straddling_intervals"] == 0
    assert wholly["reason"] == "reason no longer holds"
    assert edge["reason"] and "no longer holds" not in edge["reason"]


def test_the_inside_row_is_unavailable_when_no_interval_overlaps_a_window(tmp_path):
    lines = [("2026-10-22 18:00:00", 0), ("2026-10-22 19:00:00", 10)]

    entry = _inside_pass2(tmp_path, lines, [_window(NIGHT)])

    assert entry["verdict"] == "unavailable"
    assert entry["measured_seconds"] is None


def test_the_control_record_names_a_revision_and_fingerprint_and_no_path(tmp_path, monkeypatch):
    source = _synthetic_finngen(tmp_path)
    monkeypatch.setattr(
        _reader_ab,
        "run_child",
        _FakeChild({"reader_head": 4.0, "reader_base": 4.0, "parse_floor": 1.0}),
    )

    record = _reader_ab.measure_source(
        "opengwasdb.finngen-r13", source, repetitions=2, control=_control(tmp_path), warm=False
    )

    assert record["control"]["revision"] == "d405de0"
    assert record["control"]["fingerprint"]
    assert "module_path" not in record["control"]
    assert str(tmp_path) not in json.dumps(record["control"])
