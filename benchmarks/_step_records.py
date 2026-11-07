"""What an orchestrated Store Release's step records say about its wall time.

A record under `stores/<STORE>/records/<step>.json` holds the step's argv, its
`elapsed_seconds`, and the step's stderr. The stderr is where the package's own
phase and progress lines live:

* `log_phase` (`opengwasdb/encoding/timing.py`) writes `LABEL: start` and, on
  the way out, `LABEL: done in 1h28m` -- minute-rounded, because
  `format_duration` drops the seconds;
* `timing.log_progress` writes `LABEL: n/N — elapsed …, ETA …`, and the Dense
  VCF and Hybrid builders' own copy (`layouts/dense/build_vcf.py`) writes
  `LABEL: n/N done (extra) — elapsed …, ETA …`.

This module turns those three spellings into phases with a start, an end and a
span, progress samples, and the clean rate a phase achieved outside the nightly
I/O window. It copies the spellings rather than calling the emitters, so a
change to an emitter would leave the parser matching nothing: `parser_suspects`
counts, per record, the lines that look like a phase or progress line yet did
not parse, and the artifact carries that count so an empty `phases` list is
never mistaken for a real zero (see `tests/test_epic307_evidence.py`'s
round-trip test, which emits through the real functions).

Times are compared on the log lines' own clock -- the host's local time as
written on each line -- and a record's UTC `start_time`/`end_time` are
converted onto it. Windows are half-open `[start, end)`.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

#: One logging record as every phase module writes it.
_LOG_LINE = re.compile(
    r"^(?P<at>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (?P<level>[A-Z]+) "
    r"(?P<logger>[\w.]+): (?P<message>.*)$"
)
#: `log_phase`'s two lines. A label carries spaces and parentheses, never a colon.
_PHASE_START = re.compile(r"^(?P<label>[^:]+): start$")
_PHASE_DONE = re.compile(r"^(?P<label>[^:]+): done in (?P<duration>\d+(?:h\d+m|m\d+s|s))$")
#: Both progress flavours, whose only difference is the optional `done (extra)`.
_PROGRESS = re.compile(
    r"^(?P<label>[^:]+): (?P<completed>\d+)/(?P<total>\d+)(?: done)?"
    r"(?: \((?P<extra>.*)\))? — elapsed (?P<elapsed>[0-9hms]+), ETA .+$"
)
#: What a phase or progress line looks like to a reader, regardless of whether
#: this module can parse it: a line that matches these but not the parsers
#: above is a log format the harness has stopped understanding. They are loose
#: on purpose -- a progress line that lost its em dash or its `n/N` spelling
#: still says `elapsed` or `ETA`, and a phase line still begins `start`/`done`
#: -- and narrow enough that an ordinary line carrying a ratio (`4160/9851866
#: variants failed`) is not mistaken for one.
_SUSPECT_PHASE = re.compile(r"^[^:]+: (?:start|done)\b")
_SUSPECT_PROGRESS = re.compile(r"^[^:]+: .*(?:\belapsed\b|\bETA\b)")
#: A line that begins like a log record: the timestamp spelling itself moved.
_SUSPECT_LOG_LINE = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}")
_DURATION = re.compile(r"^(?:(?P<hours>\d+)h)?(?:(?P<minutes>\d+)m)?(?:(?P<seconds>\d+)s)?$")

#: The keys a record is expected to carry. A missing one is named in
#: `missing_fields` rather than defaulted to a plausible value.
RECORD_KEYS = (
    "step",
    "store_id",
    "start_time",
    "end_time",
    "elapsed_seconds",
    "exit_code",
    "success",
    "argv",
    "stderr",
)


@dataclass(frozen=True)
class Window:
    """A half-open interval of log-clock time, `[start, end)`.

    A phase ending exactly at `start`, or starting exactly at `end`, is outside
    it. A daily window carries the date it was expanded for, so a reader can
    see which night a flag came from.
    """

    start: datetime
    end: datetime
    text: str
    date: str | None = None

    def overlaps(self, start: datetime | None, end: datetime | None) -> bool:
        if start is None or end is None:
            return False
        return start < self.end and end > self.start

    def contains(self, at: datetime) -> bool:
        return self.start <= at < self.end

    def overlap_seconds(self, start: datetime | None, end: datetime | None) -> float:
        if start is None or end is None:
            return 0.0
        latest = max(start, self.start)
        earliest = min(end, self.end)
        return max(0.0, (earliest - latest).total_seconds())

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "text": self.text,
            "date": self.date,
        }


def _log_clock(value: str) -> datetime:
    """A log-clock datetime from `YYYY-MM-DD[ T]HH:MM[:SS]`.

    A value carrying an offset is converted to the host's local time and then
    stripped of its zone, because the log lines are written in local time and
    everything has to be compared on one clock.
    """
    parsed = datetime.fromisoformat(value.strip().replace(" ", "T"))
    if parsed.tzinfo is None:
        return parsed
    return parsed.astimezone().replace(tzinfo=None)


def _record_clock(value: str) -> datetime:
    """A record's ISO timestamp on the log clock.

    A record's `start_time`/`end_time` are UTC and carry an offset; a naive one
    is taken as UTC, which is what every orchestrator here writes, and the
    assumption is recorded in the step's `normalised` list.
    """
    parsed = datetime.fromisoformat(value.strip().replace(" ", "T"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC).astimezone().replace(tzinfo=None)
    return parsed.astimezone().replace(tzinfo=None)


def parse_window(text: str) -> Window:
    """`START/END` in log-clock time; an END before START ends the next day."""
    start_text, separator, end_text = text.partition("/")
    if not separator or not start_text.strip() or not end_text.strip():
        raise ValueError(f"a window is START/END, got {text!r} (no `/` in it)")
    start = _log_clock(start_text)
    end = _log_clock(end_text)
    if end <= start:
        end += timedelta(days=1)
    return Window(start, end, text)


def parse_daily_window(text: str) -> tuple[time, time]:
    """`HH:MM-HH:MM` as local start and end times of a daily window."""
    start_text, separator, end_text = text.partition("-")
    if not separator:
        raise ValueError(f"a daily window is HH:MM-HH:MM, got {text!r}")
    return (
        time.fromisoformat(start_text.strip()),
        time.fromisoformat(end_text.strip()),
    )


def daily_windows(
    text: str, tz: str, *, first: date, until: date
) -> list[Window]:
    """One window per night from `first` up to (not including) `until`.

    The observation the epic made on 22 Oct is a clock reading in a named zone;
    a night whose date falls on the other side of a DST change would otherwise
    be flagged an hour out. `until` is exclusive: the backup stopped covering
    `/data/opengwasdb` on 2 Nov 2026, so `--until 2026-11-02` gives the last
    window on 1 Nov.
    """
    start_time, end_time = parse_daily_window(text)
    zone = ZoneInfo(tz)
    windows: list[Window] = []
    day = first
    while day < until:
        start = datetime.combine(day, start_time, tzinfo=zone)
        end = datetime.combine(day, end_time, tzinfo=zone)
        if end <= start:
            end += timedelta(days=1)
        windows.append(
            Window(
                start.astimezone().replace(tzinfo=None),
                end.astimezone().replace(tzinfo=None),
                text,
                day.isoformat(),
            )
        )
        day += timedelta(days=1)
    return windows


def parse_duration(text: str) -> int | None:
    """`1h28m` / `3m05s` / `7s` as whole seconds, `None` for anything else."""
    match = _DURATION.match(text)
    if match is None or not any(match.groupdict().values()):
        return None
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = int(match.group("seconds") or 0)
    return hours * 3600 + minutes * 60 + seconds


def format_duration(seconds: float | None) -> str | None:
    """`1h28m45s` / `3m05s` / `7s`, for an artifact that has to be read."""
    if seconds is None:
        return None
    whole = int(round(seconds))
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def _first_overlap(
    start: datetime | None, end: datetime | None, windows: Sequence[Window]
) -> int | None:
    for index, window in enumerate(windows):
        if window.overlaps(start, end):
            return index
    return None


def seconds_inside(
    start: datetime | None, end: datetime | None, windows: Sequence[Window]
) -> float:
    """Seconds of `[start, end)` covered by the windows, counted once.

    Windows may overlap: a one-off window and the daily window for the same
    night are the same interval, and adding both would report twice the time
    inside.
    """
    if start is None or end is None:
        return 0.0
    covered = sorted(
        (max(start, window.start), min(end, window.end))
        for window in windows
        if window.overlaps(start, end)
    )
    total = 0.0
    reach: datetime | None = None
    for low, high in covered:
        begin = low if reach is None else max(low, reach)
        if high > begin:
            total += (high - begin).total_seconds()
        reach = high if reach is None else max(high, reach)
    return total


def seconds_inside_union(
    spans: Sequence[tuple[datetime, datetime]], windows: Sequence[Window]
) -> float:
    """Seconds covered by any of the `spans` and any window, counted once.

    Phases nest (`Top-hit EAF gather` holds `Top-hit gather`), so adding each
    phase's seconds inside the windows would count the inner one twice.
    """
    merged: list[tuple[datetime, datetime]] = []
    for low, high in sorted(spans):
        if merged and low <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], high))
        else:
            merged.append((low, high))
    return sum(seconds_inside(low, high, windows) for low, high in merged)


@dataclass(frozen=True)
class Phase:
    """One `log_phase` block, from its start line to its done line."""

    label: str
    start: datetime | None
    end: datetime | None
    logged_duration: str | None
    logged_duration_seconds: int | None

    @property
    def elapsed_seconds(self) -> float | None:
        if self.start is None or self.end is None:
            return None
        return (self.end - self.start).total_seconds()

    def to_dict(self, windows: Sequence[Window]) -> dict[str, Any]:
        """The phase and its window flags; `None` where a time is not known.

        A phase missing one of its two lines has no known span, so whether it
        overlapped a window is unknown too -- not false, which would read as
        "outside".
        """
        known = self.start is not None and self.end is not None
        index = _first_overlap(self.start, self.end, windows) if known else None
        return {
            "label": self.label,
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "elapsed_seconds": self.elapsed_seconds,
            "logged_duration": self.logged_duration,
            "logged_duration_seconds": self.logged_duration_seconds,
            "complete": known,
            "overlaps_window": (index is not None) if known else None,
            "window_index": index,
            "seconds_inside_windows": (
                round(seconds_inside(self.start, self.end, windows), 3) if known else None
            ),
        }


@dataclass(frozen=True)
class ProgressSample:
    """One progress line: its completed count and the phase's elapsed counter."""

    at: datetime
    completed: int
    total: int
    elapsed_seconds: int | None


def parse_progress(stderr: str) -> dict[str, list[ProgressSample]]:
    """Progress lines by label, in log order."""
    samples: dict[str, list[ProgressSample]] = {}
    for at, message in _messages(stderr):
        matched = _PROGRESS.match(message)
        if matched is None:
            continue
        samples.setdefault(matched.group("label"), []).append(
            ProgressSample(
                at=at,
                completed=int(matched.group("completed")),
                total=int(matched.group("total")),
                elapsed_seconds=parse_duration(matched.group("elapsed")),
            )
        )
    return samples


def _messages(stderr: str) -> list[tuple[datetime, str]]:
    """`(log-clock timestamp, message)` for every whole log line."""
    out: list[tuple[datetime, str]] = []
    for line in stderr.splitlines():
        matched = _LOG_LINE.match(line)
        if matched is not None:
            out.append((_log_clock(matched.group("at")), matched.group("message")))
    return out


def parser_suspects(stderr: str) -> dict[str, Any]:
    """Lines that look like phase or progress lines but did not parse.

    A record whose stderr the harness has stopped understanding must not read
    as a step that logged no phases. The three classes are counted together
    because all three mean the same thing: an emitter's spelling moved.
    """
    unparsed: list[str] = []
    for line in stderr.splitlines():
        matched = _LOG_LINE.match(line)
        if matched is None:
            if _SUSPECT_LOG_LINE.match(line):
                unparsed.append(line[:200])
            continue
        message = matched.group("message")
        if (
            _PHASE_START.match(message)
            or _PHASE_DONE.match(message)
            or _PROGRESS.match(message)
        ):
            continue
        if _SUSPECT_PHASE.match(message) or _SUSPECT_PROGRESS.match(message):
            unparsed.append(line[:200])
    return {"lines": len(unparsed), "examples": unparsed[:5]}


def parse_phases(stderr: str) -> list[Phase]:
    """Every `label: start` … `label: done in X` pair, in start order.

    Phases nest (`Top-hit EAF gather` contains `Top-hit gather`), so a done
    line closes the nearest open phase of its own label rather than the most
    recent one. A start without a done line stays open -- it was still running
    when the record was written, or the record's stderr was truncated -- and is
    reported with `end: None` rather than given an invented one.
    """
    open_phases: list[dict[str, Any]] = []
    for at, message in _messages(stderr):
        started = _PHASE_START.match(message)
        if started is not None:
            open_phases.append({"label": started.group("label"), "start": at})
            continue
        done = _PHASE_DONE.match(message)
        if done is None:
            continue
        label = done.group("label")
        for index in range(len(open_phases) - 1, -1, -1):
            if open_phases[index]["label"] == label:
                open_phases[index].update(
                    end=at,
                    logged_duration=done.group("duration"),
                    logged_duration_seconds=parse_duration(done.group("duration")),
                )
                break
        else:
            open_phases.append(
                {
                    "label": label,
                    "start": None,
                    "end": at,
                    "logged_duration": done.group("duration"),
                    "logged_duration_seconds": parse_duration(done.group("duration")),
                }
            )
    return [
        Phase(
            label=phase["label"],
            start=phase["start"],
            end=phase.get("end"),
            logged_duration=phase.get("logged_duration"),
            logged_duration_seconds=phase.get("logged_duration_seconds"),
        )
        for phase in open_phases
    ]


def clean_rate(
    samples: Sequence[ProgressSample], windows: Sequence[Window]
) -> dict[str, Any]:
    """The phase's pace outside the windows, and the pace it reached before one.

    An interval counts only when no part of it falls inside a window: one
    endpoint inside is enough to charge the window's I/O contention to the
    phase, and an interval spanning a window with an endpoint on each side is
    the worst case of all. `projected_total_seconds` extends the clean pace
    over the phase's last line's `N`. `whole_phase` is the same projection over
    every interval, which is what a window-inflated rate looks like.
    """
    inside = sum(1 for sample in samples if any(w.contains(sample.at) for w in windows))
    window_index = (
        _first_overlap(samples[0].at, samples[-1].at, windows) if samples else None
    )
    clean_intervals = [
        (first, second)
        for first, second in zip(samples, samples[1:], strict=False)
        if not any(window.overlaps(first.at, second.at) for window in windows)
    ]
    all_intervals = list(zip(samples, samples[1:], strict=False))
    total = samples[-1].total if samples else 0
    median_rate = _median_interval(clean_intervals)
    return {
        "samples": len(samples),
        "samples_inside_window": inside,
        "overlaps_window": window_index is not None,
        "window_index": window_index,
        "first_inside_window": _first_inside(samples, windows),
        "intervals": len(clean_intervals),
        "items": _items(clean_intervals),
        "seconds": _seconds(clean_intervals),
        "seconds_per_item": _per_item(clean_intervals),
        "median_interval_seconds_per_item": median_rate,
        "projected_total_seconds": _projected(clean_intervals, total),
        "median_projected_total_seconds": _median_projected(clean_intervals, total),
        "slow_intervals": _slow_intervals(clean_intervals, median_rate),
        "inside": _inside_intervals(all_intervals, windows),
        "whole_phase": {
            "intervals": len(all_intervals),
            "items": _items(all_intervals),
            "seconds": _seconds(all_intervals),
            "seconds_per_item": _per_item(all_intervals),
            "median_interval_seconds_per_item": _median_interval(all_intervals),
            "projected_total_seconds": _projected(all_intervals, total),
        },
        "prefix": _prefix(samples, windows, total),
        "total": total,
        "completed": samples[-1].completed if samples else 0,
        "first_sample": samples[0].at.isoformat() if samples else None,
        "last_sample": samples[-1].at.isoformat() if samples else None,
    }


def _inside_intervals(
    intervals: Sequence[tuple[ProgressSample, ProgressSample]], windows: Sequence[Window]
) -> dict[str, Any]:
    """The intervals that overlap a window: their own rates, median and range.

    An interval straddling a window's edge blends in-window and clean pace, so
    those are counted apart from the ones a window covers entirely.
    """
    overlapping = [
        (first, second)
        for first, second in intervals
        if any(window.overlaps(first.at, second.at) for window in windows)
    ]
    rates = [
        (second.at - first.at).total_seconds() / (second.completed - first.completed)
        for first, second in overlapping
        if second.completed > first.completed and second.at > first.at
    ]
    straddling = sum(
        1
        for first, second in overlapping
        if not any(w.start <= first.at and second.at <= w.end for w in windows)
    )
    return {
        "intervals": len(rates),
        "straddling_intervals": straddling,
        "median_interval_seconds_per_item": _median_interval(overlapping),
        "min_seconds_per_item": min(rates) if rates else None,
        "max_seconds_per_item": max(rates) if rates else None,
    }


def _items(intervals: Sequence[tuple[ProgressSample, ProgressSample]]) -> int:
    return sum(second.completed - first.completed for first, second in intervals)


def _seconds(intervals: Sequence[tuple[ProgressSample, ProgressSample]]) -> float:
    return sum((second.at - first.at).total_seconds() for first, second in intervals)


def _per_item(intervals: Sequence[tuple[ProgressSample, ProgressSample]]) -> float | None:
    items, seconds = _items(intervals), _seconds(intervals)
    return seconds / items if items > 0 and seconds > 0 else None


def _median_interval(
    intervals: Sequence[tuple[ProgressSample, ProgressSample]]
) -> float | None:
    """The median of the intervals' own rates.

    A phase that spans days meets hours that no declared window covers and that
    are still slow (another build on the host, a lock, a spill), so the pooled
    rate can be dragged far from the pace the phase usually keeps. The median
    is that usual pace, and the pooled value stays beside it.
    """
    rates = sorted(
        (second.at - first.at).total_seconds() / (second.completed - first.completed)
        for first, second in intervals
        if second.completed > first.completed
        and (second.at - first.at).total_seconds() > 0
    )
    if not rates:
        return None
    middle = len(rates) // 2
    return rates[middle] if len(rates) % 2 else (rates[middle - 1] + rates[middle]) / 2


def _slow_intervals(
    intervals: Sequence[tuple[ProgressSample, ProgressSample]], median: float | None
) -> list[dict[str, Any]]:
    """The clean intervals more than three times the median pace, with their time.

    A window is an assumption about *when* the host is slow. An interval that
    sits outside every window and still runs an order of magnitude slower than
    the phase's usual pace is the data arguing with that assumption, so the row
    that reports the median also lists these, and says the cause is unknown.
    """
    if median is None or median <= 0:
        return []
    slow: list[dict[str, Any]] = []
    for first, second in intervals:
        items = second.completed - first.completed
        seconds = (second.at - first.at).total_seconds()
        if items <= 0 or seconds <= 0:
            continue
        rate = seconds / items
        if rate > 3 * median:
            slow.append(
                {
                    "from": first.at.isoformat(),
                    "to": second.at.isoformat(),
                    "items": items,
                    "seconds": seconds,
                    "seconds_per_item": rate,
                    "times_median": rate / median,
                    "cause": "unknown",
                }
            )
    return slow


def _median_projected(
    intervals: Sequence[tuple[ProgressSample, ProgressSample]], total: int
) -> float | None:
    median = _median_interval(intervals)
    return None if median is None else median * total


def _projected(
    intervals: Sequence[tuple[ProgressSample, ProgressSample]], total: int
) -> float | None:
    per_item = _per_item(intervals)
    return None if per_item is None else per_item * total


def _first_inside(
    samples: Sequence[ProgressSample], windows: Sequence[Window]
) -> dict[str, Any] | None:
    """The first progress line a window covers, with the phase's own counter."""
    for sample in samples:
        if any(window.contains(sample.at) for window in windows):
            return {
                "at": sample.at.isoformat(),
                "completed": sample.completed,
                "elapsed_seconds": sample.elapsed_seconds,
                "seconds_per_item": (
                    sample.elapsed_seconds / sample.completed
                    if sample.elapsed_seconds and sample.completed
                    else None
                ),
            }
    return None


def _prefix(
    samples: Sequence[ProgressSample], windows: Sequence[Window], total: int
) -> dict[str, Any] | None:
    """The pace the phase had reached on its last line before the first window.

    This is the figure a hand reconstruction quotes when it says "2,100
    Analyses in 53m25s, before the window". It is the phase's own cumulative
    counter, so it is only clean when the phase began outside the windows, and
    the count it divides by is that same counter's.

    When the line's `elapsed` does not parse, the pace is taken from the time
    between the phase's first line and this one over the items finished in that
    span -- not over this line's cumulative count, which includes items the
    first line already reported. `items` and `elapsed_source` name which basis
    was used, so an unreadable counter cannot pass for "there was no line
    before the window" and cannot quietly divide by the wrong count.
    """
    if not windows:
        return None
    first_window = min(window.start for window in windows)
    before = [sample for sample in samples if sample.at < first_window]
    if not before:
        return None
    sample = before[-1]
    elapsed = sample.elapsed_seconds
    source = "log_elapsed"
    items = sample.completed
    if elapsed is None:
        elapsed = int((sample.at - samples[0].at).total_seconds())
        items = sample.completed - samples[0].completed
        source = "sample_timestamps"
    per_item = elapsed / items if items > 0 and elapsed > 0 else None
    return {
        "at": sample.at.isoformat(),
        "completed": sample.completed,
        "elapsed_seconds": elapsed,
        "items": items,
        "elapsed_source": source,
        "seconds_per_item": per_item,
        "projected_total_seconds": None if per_item is None else per_item * total,
        "reason": (
            None
            if per_item is not None
            else "no items or no time between the first line and this one"
        ),
    }


@dataclass(frozen=True)
class Record:
    """One step record, parsed but not yet flagged against any window."""

    path: Path
    step: str | None
    store_id: str | None
    start_time: str | None
    end_time: str | None
    wall_seconds: float | None
    exit_code: int | None
    success: bool | None
    skipped: bool | None
    argv: list[str] | None
    revision: str | None
    revision_key: str | None
    missing_fields: tuple[str, ...]
    normalised: tuple[str, ...]
    stderr_present: bool
    stderr_lines: int | None
    unparsed: dict[str, Any] = field(default_factory=dict)
    first_log_line: datetime | None = None
    last_log_line: datetime | None = None
    messages: tuple[tuple[datetime, str], ...] = ()
    phases: tuple[Phase, ...] = ()
    progress: dict[str, tuple[ProgressSample, ...]] = field(default_factory=dict)
    log_start: datetime | None = None
    log_end: datetime | None = None
    span_source: str = "missing"
    stderr_outside_record: bool | None = None
    record_span: tuple[datetime | None, datetime | None] = (None, None)

    def phase(self, label: str) -> Phase | None:
        for phase in self.phases:
            if phase.label == label:
                return phase
        return None

    def phases_named(self, label: str) -> list[Phase]:
        return [phase for phase in self.phases if phase.label == label]

    def interval(self, after: str, until: str) -> dict[str, Any] | None:
        """Seconds from the last line matching `after` to the next `until` one."""
        start = None
        for at, message in self.messages:
            if re.search(after, message):
                start = at
        if start is None:
            return None
        for at, message in self.messages:
            if at > start and re.search(until, message):
                return {
                    "from": start.isoformat(),
                    "to": at.isoformat(),
                    "after": after,
                    "until": until,
                    "seconds": (at - start).total_seconds(),
                }
        return None

    def to_dict(self, windows: Sequence[Window]) -> dict[str, Any]:
        known = self.log_start is not None and self.log_end is not None
        index = _first_overlap(self.log_start, self.log_end, windows) if known else None
        phases = [phase.to_dict(windows) for phase in self.phases]
        return {
            "step": self.step,
            "store_id": self.store_id,
            "record": str(self.path),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "log_start": self.log_start.isoformat() if self.log_start else None,
            "log_end": self.log_end.isoformat() if self.log_end else None,
            "span_source": self.span_source,
            "wall_seconds": self.wall_seconds,
            "exit_code": self.exit_code,
            "success": self.success,
            "skipped": self.skipped,
            "argv": self.argv,
            "revision": self.revision,
            "revision_key": self.revision_key,
            "missing_fields": list(self.missing_fields),
            "normalised": list(self.normalised),
            "stderr_present": self.stderr_present,
            "stderr_lines": self.stderr_lines,
            "unparsed": self.unparsed,
            "first_log_line": self.first_log_line.isoformat() if self.first_log_line else None,
            "last_log_line": self.last_log_line.isoformat() if self.last_log_line else None,
            "overlaps_window": (index is not None) if known else None,
            "window_index": index,
            "seconds_inside_windows": (
                round(seconds_inside(self.log_start, self.log_end, windows), 3)
                if known
                else None
            ),
            "window_unknown": sum(1 for phase in phases if phase["overlaps_window"] is None),
            "stderr_outside_record": self.stderr_outside_record,
            "record_span": {
                "first": self.record_span[0].isoformat() if self.record_span[0] else None,
                "last": self.record_span[1].isoformat() if self.record_span[1] else None,
            },
            "stderr_span": {
                "first": self.first_log_line.isoformat() if self.first_log_line else None,
                "last": self.last_log_line.isoformat() if self.last_log_line else None,
            },
            "phases": phases,
            "progress": {
                label: clean_rate(samples, windows)
                for label, samples in sorted(self.progress.items())
            },
        }


def _has_offset(value: str | None) -> bool:
    """Whether a record timestamp spells its zone."""
    if not value:
        return False
    return datetime.fromisoformat(value.strip().replace(" ", "T")).tzinfo is not None


def _span(
    payload: dict[str, Any], first_log: datetime | None, last_log: datetime | None
) -> tuple[datetime | None, datetime | None, str]:
    """The step's span on the log clock, preferring the record's own timestamps."""
    if payload.get("start_time") or payload.get("end_time"):
        start = _record_clock(payload["start_time"]) if payload.get("start_time") else None
        end = _record_clock(payload["end_time"]) if payload.get("end_time") else None
        if start is not None or end is not None:
            return start, end, "record_times"
    if first_log is not None or last_log is not None:
        return first_log, last_log, "log_lines"
    return None, None, "missing"


def _stderr_outside_record(
    start: str | None, end: str | None, first_log: datetime | None, last_log: datetime | None
) -> bool | None:
    """Whether the stderr's own timestamps fall outside the record's dates.

    A record is written when its step runs, and its stderr lines carry the walk
    of that run. Hours or days apart means the stderr is not from this step --
    OGS-00011's build says 26-27 Oct while its lines run 23-24 Oct -- and every
    window flag taken from those lines describes a different night than the
    record claims. `None` when either span is unknown.
    """
    if first_log is None or last_log is None or not start or not end:
        return None
    slack = timedelta(minutes=1)
    record_start = _record_clock(start)
    record_end = _record_clock(end)
    return first_log < record_start - slack or last_log > record_end + slack


def parse_record(path: str | Path) -> Record:
    """One `records/<step>.json`, with everything it says about its own time."""
    record_path = Path(path)
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    missing = tuple(key for key in RECORD_KEYS if key not in payload)
    normalised: list[str] = []
    stderr = payload.get("stderr")
    stderr_present = isinstance(stderr, str)
    stderr_text = stderr if stderr_present else ""
    messages = tuple(_messages(stderr_text))
    first_log = messages[0][0] if messages else None
    last_log = messages[-1][0] if messages else None
    revision_key = next(
        (key for key in ("opengwasdb_revision", "opengwasdb_rev") if payload.get(key)),
        None,
    )
    log_start, log_end, span_source = _span(payload, first_log, last_log)
    record_span = (
        _record_clock(payload["start_time"]) if payload.get("start_time") else None,
        _record_clock(payload["end_time"]) if payload.get("end_time") else None,
    )
    stderr_outside = _stderr_outside_record(
        payload.get("start_time"), payload.get("end_time"), first_log, last_log
    )
    for key in ("start_time", "end_time"):
        if payload.get(key) and not _has_offset(payload[key]):
            normalised.append(f"{key}_naive_taken_as_utc")
    if not stderr_present:
        normalised.append("stderr_missing")
    if revision_key is None:
        missing = (*missing, "opengwasdb_revision")
    return Record(
        path=record_path,
        step=payload.get("step"),
        store_id=payload.get("store_id"),
        start_time=payload.get("start_time"),
        end_time=payload.get("end_time"),
        wall_seconds=payload.get("elapsed_seconds"),
        exit_code=payload.get("exit_code"),
        success=payload.get("success"),
        skipped=payload.get("skipped"),
        argv=payload.get("argv"),
        revision=payload.get(revision_key) if revision_key else None,
        revision_key=revision_key,
        missing_fields=missing,
        normalised=tuple(normalised),
        stderr_present=stderr_present,
        stderr_lines=len(stderr_text.splitlines()) if stderr_present else None,
        unparsed=parser_suspects(stderr_text),
        first_log_line=first_log,
        last_log_line=last_log,
        messages=messages,
        phases=tuple(parse_phases(stderr_text)),
        progress={
            label: tuple(samples) for label, samples in parse_progress(stderr_text).items()
        },
        log_start=log_start,
        log_end=log_end,
        span_source=span_source,
        stderr_outside_record=stderr_outside,
        record_span=record_span,
    )


def measure_store(path: str | Path, label: str | None = None) -> dict[str, Any]:
    """Every step record under one Store Release, in file order.

    `label` names the Store in the artifact; a replaced or retired Store's
    directory is named for its timestamp, not for the Store it held.
    """
    store = Path(path).resolve()
    records = sorted((store / "records").glob("*.json"))
    if not records:
        raise SystemExit(f"{store}: no records/*.json; not an orchestrated Store Release")
    parsed = [parse_record(record) for record in records]
    ids = {entry.store_id for entry in parsed if entry.store_id}
    if len(ids) > 1:
        raise SystemExit(f"{store}: records disagree about their store id: {sorted(ids)}")
    counts = Counter(entry.step for entry in parsed)
    duplicate_steps = sorted(
        name for name, count in counts.items() if count > 1 and name is not None
    )
    if duplicate_steps:
        raise SystemExit(f"{store}: two records claim the step(s) {duplicate_steps}")
    return {
        "label": label or store.name,
        "path": str(store),
        "store_id": ids.pop() if ids else None,
        "records": parsed,
    }


def store_span(records: Iterable[Record]) -> tuple[datetime | None, datetime | None]:
    """The earliest and latest log-clock instant any of these records names."""
    starts = [record.log_start for record in records if record.log_start]
    ends = [record.log_end for record in records if record.log_end]
    return (min(starts) if starts else None, max(ends) if ends else None)


def first_date(records: Iterable[Record]) -> date | None:
    start, _ = store_span(records)
    return start.date() if start else None
