"""Epic #307's figures, each measured against the record that holds it.

The epic's tables were assembled by hand. This module holds that table as
data -- the epic's own wording and value beside a derivation from the parsed
step records (or the reader A/B) -- so that re-running the harness puts a
measured value next to every figure and says whether the record reproduces it.

Three rules keep an unknown from reading as known:

* a figure whose record is absent, unlogged or off disk is `unavailable`, with
  the derivation's own reason, never a zero;
* a figure that is a sum reports `lower_bound` when any of its components is
  excluded, lists the excluded ones with their raw spans, and is compared
  like-for-like only against the epic's own values for the components it could
  measure;
* a "why not" reason is a sentence *and* a predicate the run evaluates against
  the row's own detail -- including the rows that say a figure is unavailable,
  whose predicate reads the record the claim is about. A reason whose predicate
  no longer holds is printed as `reason no longer holds`, so a re-run cannot
  keep an explanation the data has left behind. A `note` is the same pair for
  something worth saying whether or not the figure reproduced. No reason text
  carries a measured number.

`reproduces` is decided by one rule, recorded in the artifact: a point value
within 5% (or one second, for a value measured in seconds), a lower bound that
the measurement meets, a range the measurement falls inside.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from benchmarks._artifact import is_ancestor
from benchmarks._step_records import (
    Record,
    clean_rate,
    format_duration,
    seconds_inside,
    seconds_inside_union,
)

#: The revision the epic measured the reader at.
EPIC_REVISION = "d405de0"

#: The commit that parallelised the EAF survey (#219).
PARALLEL_SURVEY_COMMIT = "ea08bec"

REPO = Path(__file__).resolve().parents[1]

#: How far a measured value may sit from the epic's and still count as the same
#: figure. One second keeps a step measured in seconds from being judged on a
#: percentage of a rounding error.
TOLERANCE = 0.05
TOLERANCE_SECONDS = 1.0

EPIC_ISSUE = 307
EPIC_READ_AT = "2026-11-07"

REPRODUCES_RULE = (
    "a point value reproduces when it is within 5% of the epic's (or one second, for a "
    "value measured in seconds); a lower bound reproduces when the measurement meets it; "
    "a range reproduces when the measurement falls inside it. A sum with any component "
    "excluded is a lower bound and is not called yes or no."
)


#: The derivations that read a step record, and so need the store and step to be
#: in the run. Everything else -- the reader A/B, a composition, a figure with
#: no record -- is dispatched without one, and a kind in neither list is an
#: error rather than a silently missing number.
RECORD_KINDS = (
    "step",
    "phase",
    "phase_clean",
    "interval",
    "projection",
    "median_projection",
    "median_interval",
    "inside_interval",
    "first_inside",
    "gap",
)
OTHER_KINDS = ("unavailable", "reader", "reader_paired", "reader_ratio", "composed")

#: What an `unavailable` derivation can check about the run, so that the
#: sentence saying why a figure is missing is tested against the records.
UNAVAILABLE_CHECKS = ("no_phases", "no_phase_matching", "no_mention", "predates_epic", "roll_up")


@dataclass(frozen=True)
class Measured:
    """What a derivation found: a number, how far it can be trusted, its basis."""

    seconds: float | None
    clean: bool
    detail: dict[str, Any] = field(default_factory=dict)
    excluded: tuple[dict[str, Any], ...] = ()
    inside_window: bool | None = None


#: A reason is prose and a predicate over the row's own `detail`. The prose is
#: printed only while the predicate holds.
Reason = tuple[str, Callable[[Mapping[str, Any]], bool]]


@dataclass(frozen=True)
class Derive:
    """Where one figure's measured value comes from."""

    kind: str
    store: str | None = None
    step: str | None = None
    label: str | None = None
    after: str | None = None
    until: str | None = None
    source: str | None = None
    side: str | None = None
    other_store: str | None = None
    other_step: str | None = None
    parts: tuple[tuple[str, Derive], ...] = ()
    reason: str | None = None
    epic_seconds: float | None = None
    #: A commit whose presence in the record's revision is reported beside it.
    contains: str | None = None
    epic_rows: int | None = None
    epic_control_seconds: float | None = None
    epic_floor_seconds: float | None = None
    #: An `unavailable` derivation's check, and what it looks for.
    check: str | None = None
    pattern: str | None = None
    components: tuple[str, ...] = ()


@dataclass(frozen=True)
class LikeForLike:
    """The epic's own values for the parts of a sum the record can measure.

    The comparison is made only when the sum holds exactly these parts: a sum
    missing one is a smaller number, not a different measurement of the same
    thing.
    """

    label: str
    components: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class Figure:
    """One row of the epic's tables, and how to measure it."""

    key: str
    figure: str
    epic: str
    epic_seconds: float | None
    derive: Derive
    method: str
    unit: str = "seconds"
    comparison: str = "point"
    epic_range: tuple[float, float] | None = None
    epic_like_for_like: LikeForLike | None = None
    reason: Reason | None = None
    notes: tuple[Reason, ...] = ()


# --- reasons' predicates ----------------------------------------------------


def _inside_window(detail: Mapping[str, Any]) -> bool:
    return detail.get("inside_window") is True


def _no_clean_rate(detail: Mapping[str, Any]) -> bool:
    return detail.get("inside_window") is True and detail.get("clean_rate") is None


def _onset_in_first_inside(detail: Mapping[str, Any]) -> bool:
    first = detail.get("first_inside_seconds_per_item")
    clean = detail.get("seconds_per_item")
    return first is not None and clean is not None and first > clean


def _scan_overlaps_window(detail: Mapping[str, Any]) -> bool:
    return "Top-hit scan" in (detail.get("phases_inside_window") or [])


def _started_inside_window(detail: Mapping[str, Any]) -> bool:
    return detail.get("start_inside_window") is True


def _mostly_inside_window(detail: Mapping[str, Any]) -> bool:
    inside = detail.get("seconds_inside_windows")
    span = detail.get("span_seconds")
    return inside is not None and span is not None and span > 0 and inside > 0.5 * span


def _gap_is_short(detail: Mapping[str, Any]) -> bool:
    seconds = detail.get("seconds")
    return seconds is not None and 0 <= seconds <= 60


def _faster_than_the_epic(detail: Mapping[str, Any]) -> bool:
    samples = [value for value in detail.get("samples") or [] if value is not None]
    epic = detail.get("epic_seconds")
    return bool(samples) and epic is not None and all(value < epic for value in samples)


def _same_code_and_rows(detail: Mapping[str, Any]) -> bool:
    """The control is the epic's revision reading the epic's file."""
    revision = detail.get("control_revision")
    rows = detail.get("epic_rows")
    return (
        isinstance(revision, str)
        and revision.startswith(EPIC_REVISION)
        and rows is not None
        and detail.get("n_rows") == rows
    )


def _floor_fell_further_than_the_control(detail: Mapping[str, Any]) -> bool:
    """The ratio is larger than the epic's because the floor shrank by more."""
    floor = detail.get("floor_vs_epic")
    control = detail.get("control_vs_epic")
    ratio = detail.get("ratio")
    epic_ratio = detail.get("epic_ratio")
    return (
        None not in (floor, control, ratio, epic_ratio)
        and floor < control
        and ratio > epic_ratio
    )


def _not_resolved(detail: Mapping[str, Any]) -> bool:
    return detail.get("verdict") == "not resolved at this load"


def _excluded_part_ran_inside_a_window(detail: Mapping[str, Any]) -> bool:
    """A part is left out because a window covers it, not merely because it is missing."""
    return any(part.get("inside_window") is True for part in detail.get("excluded") or [])


def _inside_intervals_straddle_an_edge(detail: Mapping[str, Any]) -> bool:
    return bool(detail.get("straddling_intervals"))


def _has_slow_intervals(detail: Mapping[str, Any]) -> bool:
    return bool(detail.get("slow_intervals"))


def _parallel_survey_wholly_inside(detail: Mapping[str, Any]) -> bool:
    ancestry = detail.get("ancestry") or {}
    return ancestry.get("contained") is True and detail.get("fully_inside") is True


def _serial_survey_outside_windows(detail: Mapping[str, Any]) -> bool:
    ancestry = detail.get("ancestry") or {}
    return ancestry.get("contained") is False and detail.get("inside_window") is False


def _record_has_no_phases(detail: Mapping[str, Any]) -> bool:
    return detail.get("record_present") is True and detail.get("phases_in_record") == 0


def _record_lacks_the_phase(detail: Mapping[str, Any]) -> bool:
    return detail.get("record_present") is True and not detail.get("matching_labels")


def _no_record_mentions_it(detail: Mapping[str, Any]) -> bool:
    return bool(detail.get("records_searched")) and detail.get("records_mentioning") == 0


def _every_record_predates_the_epic(detail: Mapping[str, Any]) -> bool:
    latest = detail.get("latest_record_end")
    return latest is not None and datetime.fromisoformat(latest).date() < date.fromisoformat(
        str(detail.get("epic_read_at"))
    )


def _a_part_is_not_a_clean_reproduction(detail: Mapping[str, Any]) -> bool:
    verdicts = list((detail.get("components") or {}).values())
    return bool(verdicts) and None not in verdicts and any(v != "reproduces" for v in verdicts)


# --- measurement ------------------------------------------------------------


def _record(derive: Derive, evidence: Mapping[str, Any]) -> Record | None:
    steps: Mapping[str, Record] = evidence["stores"].get(derive.store or "", {})
    return steps.get(derive.step or "")


def _overlaps(start: datetime | None, end: datetime | None, evidence: Mapping[str, Any]) -> bool:
    return any(window.overlaps(start, end) for window in evidence["windows"])


def _measure(derive: Derive, evidence: Mapping[str, Any]) -> Measured:
    if derive.kind == "unavailable":
        return _measure_unavailable(derive, evidence)
    if derive.kind in RECORD_KINDS:
        record = _record(derive, evidence)
        if record is None:
            return Measured(
                None, False, {"reason": f"{derive.store}.{derive.step} is not in this run"}
            )
        return _measure_in_record(derive, record, evidence)
    if derive.kind == "reader":
        return _measure_reader(derive, evidence)
    if derive.kind == "reader_paired":
        return _measure_reader_paired(derive, evidence)
    if derive.kind == "reader_ratio":
        return _measure_reader_ratio(derive, evidence)
    if derive.kind == "composed":
        return _measure_composed(derive, evidence)
    raise ValueError(f"unknown derivation {derive.kind!r}")


def _measure_in_record(
    derive: Derive, record: Record, evidence: Mapping[str, Any]
) -> Measured:
    if derive.kind == "step":
        return _measure_step(derive, record, evidence)
    if derive.kind == "phase":
        return _measure_phase(derive, record, evidence)
    if derive.kind == "phase_clean":
        return _measure_phase_clean(derive, record, evidence)
    if derive.kind == "interval":
        return _measure_interval(derive, record, evidence)
    if derive.kind in ("projection", "median_projection"):
        return _measure_projection(derive, record, evidence)
    if derive.kind == "inside_interval":
        return _measure_inside_interval(derive, record, evidence)
    if derive.kind == "median_interval":
        return _measure_median_interval(derive, record, evidence)
    if derive.kind == "first_inside":
        return _measure_first_inside(derive, record, evidence)
    if derive.kind == "gap":
        return _measure_gap(derive, evidence)
    raise ValueError(f"unknown in-record derivation {derive.kind!r}")


def _measure_step(
    derive: Derive, record: Record, evidence: Mapping[str, Any]
) -> Measured:
    known = record.log_start is not None and record.log_end is not None
    inside = _overlaps(record.log_start, record.log_end, evidence) if known else None
    windows = evidence["windows"]
    flags = [phase.to_dict(windows) for phase in record.phases]
    unknown = sum(1 for phase in record.phases if phase.start is None or phase.end is None)
    spans = [
        (phase.start, phase.end)
        for phase in record.phases
        if phase.start is not None and phase.end is not None
    ]
    return Measured(
        record.wall_seconds,
        inside is False,
        {
            "record": str(record.path),
            "span_source": record.span_source,
            "wall_seconds": record.wall_seconds,
            "span_seconds": (
                (record.log_end - record.log_start).total_seconds()
                if record.log_start is not None and record.log_end is not None
                else None
            ),
            "seconds_inside_windows": (
                seconds_inside(record.log_start, record.log_end, windows) if known else None
            ),
            "start_inside_window": (
                None
                if inside is None or record.log_start is None
                else any(window.contains(record.log_start) for window in windows)
            ),
            # Unfinished phases have no span, so the total is unknown, not short.
            "phases_seconds_inside_windows": (
                None if unknown else round(seconds_inside_union(spans, windows), 3)
            ),
            "phases_unknown": unknown,
            "phases_inside_window": [
                flag["label"] for flag in flags if flag["overlaps_window"]
            ],
        },
        inside_window=inside,
    )


def _measure_phase(
    derive: Derive, record: Record, evidence: Mapping[str, Any]
) -> Measured:
    phase = record.phase(derive.label or "")
    if phase is None:
        return Measured(None, False, {"reason": f"no `{derive.label}` phase in this record"})
    known = phase.start is not None and phase.end is not None
    inside = _overlaps(phase.start, phase.end, evidence) if known else None
    return Measured(
        phase.elapsed_seconds,
        inside is False,
        {
            "logged_duration_seconds": phase.logged_duration_seconds,
            "start": phase.start.isoformat() if phase.start else None,
            "end": phase.end.isoformat() if phase.end else None,
        },
        inside_window=inside,
    )


def _measure_phase_clean(
    derive: Derive, record: Record, evidence: Mapping[str, Any]
) -> Measured:
    """A phase's span if no window covers it, else its clean-rate projection.

    A phase that ended seconds after the snapshot began is not a polluted half
    hour, but its span still is not clean: the projection from its lines outside
    every window is, and that is what a sum of clean work wants. A phase with
    neither is reported with its span, `clean: False` and no clean rate, so a
    composition can leave it out and say why.
    """
    span_derive = Derive("phase", derive.store, derive.step, derive.label)
    span = _measure_in_record(span_derive, record, evidence)
    if span.clean:
        return span
    samples = record.progress.get(derive.label or "")
    rate = clean_rate(samples, evidence["windows"]) if samples else None
    if rate is None or rate["projected_total_seconds"] is None:
        return Measured(
            span.seconds,
            False,
            {**span.detail, "clean_rate": None},
            inside_window=span.inside_window,
        )
    return Measured(
        rate["projected_total_seconds"],
        True,
        {
            "projected": True,
            "span_seconds": span.seconds,
            "seconds_per_item": rate["seconds_per_item"],
            "items": rate["items"],
            "total": rate["total"],
            "clean_rate": rate["seconds_per_item"],
        },
        inside_window=span.inside_window,
    )


def _measure_interval(
    derive: Derive, record: Record, evidence: Mapping[str, Any]
) -> Measured:
    interval = record.interval(derive.after or "", derive.until or "")
    if interval is None:
        return Measured(
            None,
            False,
            {"reason": f"no `{derive.after}` … `{derive.until}` pair in this record"},
        )
    start = datetime.fromisoformat(interval["from"])
    end = datetime.fromisoformat(interval["to"])
    inside = _overlaps(start, end, evidence)
    covered = seconds_inside(start, end, evidence["windows"])
    return Measured(
        interval["seconds"],
        not inside,
        {
            **{key: interval[key] for key in ("from", "to", "after", "until")},
            "span_seconds": interval["seconds"],
            "seconds_inside_windows": covered,
            "fully_inside": covered >= interval["seconds"] > 0,
            **_ancestry(derive, record, evidence),
        },
        inside_window=inside,
    )


def _ancestry(derive: Derive, record: Record, evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Whether the record's revision holds the commit the row's claim turns on.

    `None` is "git cannot say" (an unknown revision, a shallow clone), never a
    no: a claim about what code ran is made only on a yes or a no.
    """
    if not derive.contains:
        return {}
    ask = evidence.get("is_ancestor")
    revision = record.revision
    contained = ask(derive.contains, revision) if ask is not None and revision else None
    return {
        "revision": revision,
        "ancestry": {"commit": derive.contains, "contained": contained},
    }


def _progress_rate(
    derive: Derive, record: Record, evidence: Mapping[str, Any]
) -> dict[str, Any] | Measured:
    """The phase's progress rate, or the `Measured` that says it has none."""
    samples = record.progress.get(derive.label or "")
    if not samples:
        return Measured(None, False, {"reason": f"no `{derive.label}` progress lines"})
    return clean_rate(samples, evidence["windows"])


def _measure_projection(
    derive: Derive, record: Record, evidence: Mapping[str, Any]
) -> Measured:
    rate = _progress_rate(derive, record, evidence)
    if isinstance(rate, Measured):
        return rate
    median = derive.kind == "median_projection"
    first_inside = rate["first_inside_window"] or {}
    return Measured(
        rate["median_projected_total_seconds"] if median else rate["projected_total_seconds"],
        (rate["median_interval_seconds_per_item"] if median else rate["seconds_per_item"])
        is not None,
        {
            "seconds_per_item": rate["seconds_per_item"],
            "median_interval_seconds_per_item": rate["median_interval_seconds_per_item"],
            "first_inside_seconds_per_item": first_inside.get("seconds_per_item"),
            "items": rate["items"],
            "total": rate["total"],
            "samples": rate["samples"],
            "samples_inside_window": rate["samples_inside_window"],
            "slow_intervals": rate["slow_intervals"],
        },
        inside_window=rate["overlaps_window"],
    )


def _measure_inside_interval(
    derive: Derive, record: Record, evidence: Mapping[str, Any]
) -> Measured:
    """The median pace of the progress intervals that overlap a window, and their range."""
    rate = _progress_rate(derive, record, evidence)
    if isinstance(rate, Measured):
        return rate
    inside = rate["inside"]
    median = inside["median_interval_seconds_per_item"]
    return Measured(
        median,
        median is not None,
        {
            "intervals": inside["intervals"],
            "straddling_intervals": inside["straddling_intervals"],
            "min_seconds_per_item": inside["min_seconds_per_item"],
            "max_seconds_per_item": inside["max_seconds_per_item"],
        },
        inside_window=True if inside["intervals"] else None,
    )


def _measure_median_interval(
    derive: Derive, record: Record, evidence: Mapping[str, Any]
) -> Measured:
    rate = _progress_rate(derive, record, evidence)
    if isinstance(rate, Measured):
        return rate
    return Measured(
        rate["median_interval_seconds_per_item"],
        rate["median_interval_seconds_per_item"] is not None,
        {
            "pooled_seconds_per_item": rate["seconds_per_item"],
            "intervals": rate["intervals"],
            "samples": rate["samples"],
            "samples_inside_window": rate["samples_inside_window"],
            "slow_intervals": rate["slow_intervals"],
        },
        inside_window=rate["overlaps_window"],
    )


def _measure_first_inside(
    derive: Derive, record: Record, evidence: Mapping[str, Any]
) -> Measured:
    samples = record.progress.get(derive.label or "")
    rate = clean_rate(samples, evidence["windows"]) if samples else None
    inside = rate["first_inside_window"] if rate else None
    if not inside:
        return Measured(None, False, {"reason": "no progress line inside a window"})
    return Measured(
        inside["seconds_per_item"],
        False,
        {
            "at": inside["at"],
            "completed": inside["completed"],
            "elapsed_seconds": inside["elapsed_seconds"],
        },
        inside_window=True,
    )


def _measure_gap(derive: Derive, evidence: Mapping[str, Any]) -> Measured:
    """Seconds from a phase ending in one record to the next step's first line.

    `Top-hit index write` finishes inside the build, and the standalone rebuild
    is a separate step record; the gap between them is what the epic's "ten
    seconds" describes.
    """
    first = _record(Derive("step", derive.store, derive.step), evidence)
    second = _record(Derive("step", derive.other_store, derive.other_step), evidence)
    if first is None or second is None:
        return Measured(
            None,
            False,
            {"reason": f"{derive.store}.{derive.step} or {derive.other_store} is not in this run"},
        )
    phase = first.phase(derive.label or "")
    if phase is None or phase.end is None or second.log_start is None:
        return Measured(
            None,
            False,
            {"reason": "the phase's end or the next step's start is not in the records"},
        )
    outcome = (second.log_start - phase.end).total_seconds()
    return Measured(
        outcome,
        True,
        {
            "seconds": outcome,
            "from": phase.end.isoformat(),
            "to": second.log_start.isoformat(),
            "phase": phase.label,
            "next": f"{derive.other_store}.{derive.other_step}",
        },
        inside_window=_overlaps(phase.end, second.log_start, evidence),
    )


def _reader_source(derive: Derive, evidence: Mapping[str, Any]) -> Mapping[str, Any] | None:
    sources: Sequence[Mapping[str, Any]] = evidence["reader_ab"]
    for source in sources:
        if source["label"] == derive.source:
            return source
    return None


def _no_reader(derive: Derive) -> Measured:
    return Measured(None, False, {"reason": f"no reader A/B for {derive.source}"})


def _measure_reader(derive: Derive, evidence: Mapping[str, Any]) -> Measured:
    """One side's median, and the samples behind it."""
    source = _reader_source(derive, evidence)
    if source is None:
        return _no_reader(derive)
    side = derive.side or ""
    seconds = source["medians"].get(side)
    control = source.get("control") or {}
    return Measured(
        seconds,
        seconds is not None,
        {
            "samples": [
                sample["seconds"] for sample in source["samples"] if sample["side"] == side
            ],
            "epic_seconds": derive.epic_seconds,
            "n_rows": source["n_rows"],
            "epic_rows": derive.epic_rows,
            "control_revision": control.get("revision"),
        },
        inside_window=None,
    )


def _measure_reader_paired(derive: Derive, evidence: Mapping[str, Any]) -> Measured:
    """The per-round ratios of head against the control, and their verdict."""
    source = _reader_source(derive, evidence)
    if source is None:
        return _no_reader(derive)
    comparison = (source.get("comparisons") or {}).get("reader_head_vs_reader_base")
    if not comparison:
        return Measured(
            None, False, {"reason": "the run has no control side to pair against"}
        )
    verdict = comparison.get("verdict")
    return Measured(
        comparison.get("median_ratio"),
        verdict in ("slower", "faster"),
        {
            "verdict": verdict,
            "rounds": comparison.get("rounds"),
            "median_ratio": comparison.get("median_ratio"),
            "min_ratio": comparison.get("min_ratio"),
            "max_ratio": comparison.get("max_ratio"),
            "n_rounds": comparison.get("n_rounds"),
            "above_one": comparison.get("above_one"),
            "sign_test_p": comparison.get("sign_test_p"),
            "wilcoxon_p": comparison.get("wilcoxon_p"),
            "head_first_rounds": comparison.get("head_first_rounds"),
            "verdict_basis": comparison.get("verdict_basis"),
            "control_revision": (source.get("control") or {}).get("revision"),
        },
        inside_window=None,
    )


def _measure_reader_ratio(derive: Derive, evidence: Mapping[str, Any]) -> Measured:
    """The control reader's median over the parse floor's, beside the epic's.

    The epic's `7.78x` and `1.96x` are ratios of its own two timings. Whatever
    moved the absolute times, the ratio is the quantity that says how far the
    reader sits above the floor, so it is reported on its own.
    """
    source = _reader_source(derive, evidence)
    if source is None:
        return _no_reader(derive)
    control = source["medians"].get("reader_base")
    floor = source["medians"].get("parse_floor")
    epic_control, epic_floor = derive.epic_control_seconds, derive.epic_floor_seconds
    if not control or not floor or not epic_control or not epic_floor:
        return Measured(
            None, False, {"reason": f"no control and floor medians for {derive.source}"}
        )
    return Measured(
        control / floor,
        True,
        {
            "ratio": control / floor,
            "control_seconds": control,
            "floor_seconds": floor,
            "epic_control_seconds": epic_control,
            "epic_floor_seconds": epic_floor,
            "epic_ratio": epic_control / epic_floor,
            "control_vs_epic": control / epic_control,
            "floor_vs_epic": floor / epic_floor,
        },
        inside_window=None,
    )


def _measure_unavailable(derive: Derive, evidence: Mapping[str, Any]) -> Measured:
    """The facts an unavailable figure's reason rests on, read from the run.

    A figure is unavailable because a record is absent, a phase unlogged, or
    the epic's number a roll-up. Each is a claim about the run, so each is
    looked up here rather than asserted.
    """
    check = derive.check
    if check not in UNAVAILABLE_CHECKS:
        raise ValueError(
            f"an unavailable figure needs a check from {UNAVAILABLE_CHECKS}, got {check!r}"
            f" (reason {derive.reason!r})"
        )
    records = [
        record for steps in evidence["stores"].values() for record in steps.values()
    ]
    facts: dict[str, Any] = {"check": check}
    if check in ("no_phases", "no_phase_matching"):
        record = _record(derive, evidence)
        facts["record_present"] = record is not None
        if record is not None:
            labels = [phase.label for phase in record.phases] + sorted(record.progress)
            facts["phases_in_record"] = len(record.phases)
            facts["phase_labels"] = labels
            facts["matching_labels"] = [
                label
                for label in labels
                if derive.pattern and re.search(derive.pattern, label, re.IGNORECASE)
            ]
    elif check == "no_mention":
        pattern = re.compile(derive.pattern or "", re.IGNORECASE)
        facts["records_searched"] = len(records)
        facts["records_mentioning"] = sum(
            1
            for record in records
            if any(pattern.search(message) for _, message in record.messages)
            or pattern.search(" ".join(record.argv or []))
        )
    elif check == "predates_epic":
        ends = [end for record in records for end in (record.log_end, record.last_log_line) if end]
        facts["latest_record_end"] = max(ends).isoformat() if ends else None
        facts["epic_read_at"] = EPIC_READ_AT
    else:
        verdicts = evidence.get("verdicts") or {}
        facts["components"] = {key: verdicts.get(key) for key in derive.components}
    return Measured(None, False, facts, inside_window=None)


def _measure_composed(derive: Derive, evidence: Mapping[str, Any]) -> Measured:
    parts: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    total = 0.0
    for name, part in derive.parts:
        measured = _measure(part, evidence)
        included = measured.seconds is not None and measured.clean
        if included:
            total += measured.seconds or 0.0
        else:
            excluded.append(
                {
                    "name": name,
                    "measured": format_duration(measured.seconds),
                    "seconds": measured.seconds,
                    "reason": measured.detail.get("reason")
                    or ("ran inside a window" if measured.inside_window else None),
                    "inside_window": measured.inside_window,
                }
            )
        parts.append(
            {
                "name": name,
                "seconds": measured.seconds,
                "measured": format_duration(measured.seconds),
                "included": included,
                "inside_window": measured.inside_window,
                "detail": measured.detail,
            }
        )
    found = any(part["seconds"] is not None for part in parts)
    return Measured(
        total if found else None,
        not excluded and found,
        {"parts": parts, "excluded": excluded},
        excluded=tuple(excluded),
        inside_window=None,
    )


# --- rendering and the verdict ----------------------------------------------


def _render(seconds: float | None, unit: str) -> str | None:
    if seconds is None:
        return None
    if unit == "seconds":
        return format_duration(seconds)
    if unit == "s/analysis":
        return f"{seconds:.4f} s/Analysis"
    if unit == "s":
        return f"{seconds:.1f} s"
    if unit == "ratio":
        return f"{seconds:.2f}x"
    raise ValueError(f"unknown unit {unit!r}")


def reproduces(
    measured: float | None,
    epic: float | None,
    comparison: str,
    epic_range: tuple[float, float] | None = None,
    *,
    slack: float = TOLERANCE_SECONDS,
) -> bool:
    """Whether the measurement is the epic's figure, by the artifact's rule.

    `slack` is the absolute tolerance beside the 5%: one second for a value in
    seconds, and none for a ratio or a rate, where a second is not small.
    """
    if comparison == "range":
        if measured is None or epic_range is None:
            return False
        return epic_range[0] <= measured <= epic_range[1]
    if measured is None or epic is None:
        return False
    if comparison == "lower_bound":
        return measured >= epic
    return abs(measured - epic) <= max(TOLERANCE * epic, slack)


def _reason(figure: Figure, measured: Measured, verdict: str) -> str | None:
    """The row's explanation, or `reason no longer holds` when the data moved.

    A resolved direction needs no excuse, and a figure that reproduced has
    nothing to explain.
    """
    if verdict in ("reproduces", "slower", "faster"):
        return None
    detail_reason = measured.detail.get("reason")
    if verdict == "unavailable" and detail_reason and figure.derive.kind != "unavailable":
        return str(detail_reason)
    if figure.reason is not None:
        text, holds = figure.reason
        data = {**measured.detail, "inside_window": measured.inside_window}
        return text if holds(data) else "reason no longer holds"
    if detail_reason:
        return str(detail_reason)
    return "the measurement differs from the epic's and no reason is recorded"


def _notes(figure: Figure, measured: Measured) -> list[str]:
    """What the row says whether or not it reproduced, each checked like a reason."""
    data = {**measured.detail, "inside_window": measured.inside_window}
    return [text if holds(data) else "note no longer holds" for text, holds in figure.notes]


def _like_for_like(figure: Figure, measured: Measured) -> dict[str, Any] | None:
    """The sum against the epic's values for the same parts, or `incomplete`.

    The comparison is made only when the parts the sum holds are exactly the
    parts the epic's like-for-like value is a sum of. With one missing, or one
    extra, the two totals are of different things.
    """
    spec = figure.epic_like_for_like
    if spec is None or measured.seconds is None:
        return None
    parts = measured.detail.get("parts") or []
    included = [part["name"] for part in parts if part["included"]]
    expected = [name for name, _ in spec.components]
    missing = [name for name in expected if name not in included]
    unexpected = [name for name in included if name not in expected]
    complete = not missing and not unexpected
    epic_seconds = sum(seconds for _, seconds in spec.components) if complete else None
    if epic_seconds is None:
        verdict = "incomplete"
    else:
        verdict = "reproduces" if reproduces(measured.seconds, epic_seconds, "point") else "differs"
    return {
        "epic": spec.label,
        "epic_seconds": epic_seconds,
        "measured_seconds": round(measured.seconds, 4),
        "verdict": verdict,
        "included": included,
        "expected": expected,
        "missing": missing,
        "unexpected": unexpected,
    }


def measure_figure(figure: Figure, evidence: Mapping[str, Any]) -> dict[str, Any]:
    """One comparison row: the epic's value, the measured one, and why."""
    measured = _measure(figure.derive, evidence)
    slack = TOLERANCE_SECONDS if figure.unit in ("seconds", "s") else 0.0
    if measured.seconds is None:
        verdict = "unavailable"
    elif figure.comparison == "paired_ratio":
        verdict = str(measured.detail.get("verdict") or "not resolved at this load")
    elif measured.excluded:
        verdict = "lower_bound"
    elif reproduces(
        measured.seconds, figure.epic_seconds, figure.comparison, figure.epic_range, slack=slack
    ):
        verdict = "reproduces"
    else:
        verdict = "differs"
    return {
        "key": figure.key,
        "figure": figure.figure,
        "epic": figure.epic,
        "epic_seconds": figure.epic_seconds,
        "epic_range": list(figure.epic_range) if figure.epic_range else None,
        "unit": figure.unit,
        "measured": _render(measured.seconds, figure.unit),
        "measured_seconds": None if measured.seconds is None else round(measured.seconds, 4),
        "verdict": verdict,
        "inside_window": measured.inside_window,
        "method": figure.method,
        "detail": measured.detail,
        "like_for_like": _like_for_like(figure, measured),
        "excluded": list(measured.excluded),
        "reason": _reason(figure, measured, verdict),
        "notes": _notes(figure, measured),
    }


# --- the epic's own markers -------------------------------------------------


def _pass1() -> Derive:
    return Derive(
        "interval",
        store="OGS-00009",
        step="build",
        after=r"^Pass 1: collecting source variants",
        until=r"^Pass 2: resolving",
    )


#: The September build and OGS-00010's completion are not the published Stores'
#: current records: the first was replaced when OGS-00009 was rebuilt, the second
#: was retired. Their records are read from where those Stores' files now sit.
SEPT_BUILD = "OGS-00009-sept"
RETIRED_COMPLETION = "OGS-00010-retired"


def _eaf_survey(
    store: str = "OGS-00009",
    after: str = r"^Pass 2: 2514/2514 done",
    contains: str | None = None,
) -> Derive:
    return Derive(
        "interval",
        store=store,
        step="build",
        after=after,
        until=r"^Encoding plan:",
        contains=contains,
    )


def _planes() -> Derive:
    return Derive(
        "interval",
        store="OGS-00009",
        step="build",
        after=r"^Encoding plan:",
        until=r"^Band-write z: start$",
    )


def _residual(store: str = "OGS-00009", after: str = r"^Band-write eaf: done in") -> Derive:
    return Derive(
        "interval",
        store=store,
        step="build",
        after=after,
        until=r"^eaf: int8 residual",
    )


def _se_and_index() -> Derive:
    return Derive(
        "interval",
        store="OGS-00009",
        step="build",
        after=r"^SE fit: start$",
        until=r"^Top-hit index write: done in",
    )


def _phase(store: str, step: str, label: str, kind: str = "phase") -> Derive:
    return Derive(kind, store=store, step=step, label=label)


def _projection(store: str, step: str, label: str, kind: str = "projection") -> Derive:
    return Derive(kind, store=store, step=step, label=label)


def _unavailable_figure(
    key: str,
    figure: str,
    epic: str,
    epic_seconds: float | None,
    reason: Reason,
    **derive: Any,
) -> Figure:
    """A figure the run cannot measure, with the check its reason rests on."""
    return Figure(
        key=key,
        figure=figure,
        epic=epic,
        epic_seconds=epic_seconds,
        derive=Derive("unavailable", reason=reason[0], **derive),
        method="not available",
        reason=reason,
    )


def _control_row(
    key: str, label: str, source: str, epic: str, epic_seconds: float, epic_rows: int
) -> Figure:
    """The epic's own revision, read over the epic's own file, measured here."""
    return Figure(
        key=key,
        figure=label,
        epic=epic,
        epic_seconds=epic_seconds,
        derive=Derive(
            "reader", source=source, side="reader_base", epic_rows=epic_rows
        ),
        method="the control side's median, fresh interpreters, interleaved with head and floor",
        unit="s",
        reason=(
            "the control is the epic's revision reading the epic's file, so the code and the "
            "input are the same and what differs is the run's conditions (host load, cache "
            "state, library versions); this run records the load of every sample and does not "
            "separate the rest",
            _same_code_and_rows,
        ),
    )


def _floor_row(
    key: str, label: str, source: str, epic: str, epic_seconds: float, engine: str
) -> Figure:
    """The parse floor, which the epic measured once."""
    return Figure(
        key=key,
        figure=label,
        epic=epic,
        epic_seconds=epic_seconds,
        derive=Derive("reader", source=source, side="parse_floor", epic_seconds=epic_seconds),
        method="the parse-floor side's median, same samples",
        unit="s",
        reason=(
            f"every sample of the floor is faster than the epic's single reading, by the same "
            f"{engine} method; this run does not attribute the difference",
            _faster_than_the_epic,
        ),
    )


def _ratio_row(
    key: str, label: str, source: str, epic: str, epic_ratio: float, control: float, floor: float
) -> Figure:
    """Control over floor: the reader's height above its floor, which absolute times hide."""
    return Figure(
        key=key,
        figure=label,
        epic=epic,
        epic_seconds=epic_ratio,
        derive=Derive(
            "reader_ratio",
            source=source,
            epic_control_seconds=control,
            epic_floor_seconds=floor,
        ),
        method="the control side's median over the parse-floor side's, same samples",
        unit="ratio",
        reason=(
            "the parse floor ran faster than the epic's reading by a larger fraction than the "
            "control reader did, so the control sits further above it; this run does not say why",
            _floor_fell_further_than_the_control,
        ),
    )


def _paired_row(key: str, label: str, source: str) -> Figure:
    """Head against the control, round by round: the regression question."""
    return Figure(
        key=key,
        figure=label,
        epic="1.00x (the epic measured d405de0)",
        epic_seconds=1.0,
        derive=Derive("reader_paired", source=source, side="reader_head"),
        method="the median of the per-round paired ratios head / control; a direction only "
        "from an even number of at least six rounds with the sign test and the Wilcoxon "
        "signed-rank test both under 0.05",
        unit="ratio",
        comparison="paired_ratio",
        reason=(
            "the rounds do not separate head from the control by both paired tests, or there "
            "are too few of them; see `verdict_basis`",
            _not_resolved,
        ),
    )


FIGURES: tuple[Figure, ...] = (
    # --- what each step is bound by -----------------------------------------
    Figure(
        key="pass2_before_window",
        figure="OGS-00009 Pass 2, per Analysis, before the window",
        epic="1.53 s each (2,100 Analyses in 53m25s, 32 workers)",
        epic_seconds=1.53,
        derive=Derive("first_inside", store="OGS-00009", step="build", label="Pass 2"),
        method="the first progress line a window covers: its own elapsed / its completed count",
        unit="s/analysis",
    ),
    Figure(
        key="ogs00016_scan",
        figure="OGS-00016 standalone Top-Hit Index scan phase",
        epic="scan 5,701 s (99.5% of the step)",
        epic_seconds=5701.0,
        derive=_phase("OGS-00016", "top-hits", "Top-hit scan"),
        method="the phase's span from its two log lines",
        unit="s",
    ),
    Figure(
        key="ogs00016_pass2_clean_interval",
        figure="OGS-00016 Pass 2, per Analysis, outside the window",
        epic="22-29 s each outside the window (200-1,700 s inside it)",
        epic_seconds=None,
        epic_range=(22.0, 29.0),
        comparison="range",
        derive=Derive("median_interval", store="OGS-00016", step="build", label="Pass 2"),
        method="the median of the progress intervals outside every window, each interval's own "
        "seconds per Analysis; the pooled pace and the slow intervals are beside it",
        unit="s/analysis",
        notes=(
            (
                "the clean intervals slower than three times the median are listed in "
                "`slow_intervals`, each with its time; none lies inside a declared window, and "
                "their cause is unknown",
                _has_slow_intervals,
            ),
        ),
    ),
    Figure(
        key="ogs00016_pass2_inside_interval",
        figure="OGS-00016 Pass 2, per Analysis, inside the window",
        epic="200-1,700 s each inside the window (22-29 s outside it)",
        epic_seconds=None,
        epic_range=(200.0, 1700.0),
        comparison="range",
        derive=Derive("inside_interval", store="OGS-00016", step="build", label="Pass 2"),
        method="the median of the progress intervals that overlap a window, each interval's "
        "own seconds per Analysis; their minimum and maximum are beside it",
        unit="s/analysis",
        reason=(
            "an interval that straddles a window's edge blends in-window and clean pace, so "
            "the ends of the range are not purely in-window readings",
            _inside_intervals_straddle_an_edge,
        ),
    ),
    Figure(
        key="ogs00016_inline_gap",
        figure="OGS-00016: build's inline index write to the standalone rebuild",
        epic="ten seconds (the inline write finishes before the rebuild starts)",
        epic_seconds=10.0,
        derive=Derive(
            "gap",
            store="OGS-00016",
            step="build",
            label="Top-hit index write",
            other_store="OGS-00016",
            other_step="top-hits",
        ),
        method="the build's `Top-hit index write` end to the rebuild's first log line",
        unit="s",
        reason=(
            "the epic's figure is a rounded reading of this interval, which the two records "
            "bound between a phase end and the next step's first line",
            _gap_is_short,
        ),
    ),
    Figure(
        key="ogs00016_top_hits",
        figure="OGS-00016 standalone Top-Hit Index rebuild",
        epic="1.59 h",
        epic_seconds=5724.0,
        derive=Derive("step", store="OGS-00016", step="top-hits"),
        method="the step record's elapsed_seconds",
    ),
    Figure(
        key="ogs00009_top_hits",
        figure="OGS-00009 standalone Top-Hit Index rebuild",
        epic="1.35 h",
        epic_seconds=4860.0,
        derive=Derive("step", store="OGS-00009", step="top-hits"),
        method="the step record's elapsed_seconds",
    ),
    Figure(
        key="ogs00012_top_hits",
        figure="OGS-00012 standalone Top-Hit Index rebuild",
        epic="20.87 h inside the window",
        epic_seconds=75132.0,
        derive=Derive("step", store="OGS-00012", step="top-hits"),
        method="the step record's elapsed_seconds",
        reason=(
            "the record holds the step's total, and the scan phase it contains overlaps the "
            "nightly window, so the total is not a clean rate",
            _scan_overlaps_window,
        ),
    ),
    Figure(
        key="ogs00016_validate",
        figure="OGS-00016 validation",
        epic="7.03 h",
        epic_seconds=25308.0,
        derive=Derive("step", store="OGS-00016", step="validate"),
        method="the step record's elapsed_seconds",
    ),
    Figure(
        key="ogs00013_validate",
        figure="OGS-00013 validation",
        epic="5.83 h",
        epic_seconds=20988.0,
        derive=Derive("step", store="OGS-00013", step="validate"),
        method="the step record's elapsed_seconds",
    ),
    Figure(
        key="ogs00009_validate",
        figure="OGS-00009 validation",
        epic="at least 3h10m, still running at 23:50 BST",
        epic_seconds=11400.0,
        derive=Derive("step", store="OGS-00009", step="validate"),
        method="the step record's elapsed_seconds; the epic saw the same run unfinished",
        comparison="lower_bound",
    ),
    Figure(
        key="ogs00009_subset_check",
        figure="OGS-00009 Indexed Variant Subset comparison inside validation",
        epic="over an hour (the HM3 subset itself built in 13 min)",
        epic_seconds=3600.0,
        derive=Derive("unavailable", store="OGS-00009", step="validate", check="no_phases"),
        method="not available",
        reason=(
            "validation logs no per-check phase before #308; the record holds the whole step's "
            "elapsed time and nothing inside it",
            _record_has_no_phases,
        ),
    ),
    Figure(
        key="eaf_survey_oct",
        figure="OGS-00009 EAF survey and orientation check (Oct)",
        epic="3h32m for 2,514 Analyses, partly in the window",
        epic_seconds=12720.0,
        derive=_eaf_survey(contains=PARALLEL_SURVEY_COMMIT),
        method="last Pass 2 progress line to the `Encoding plan` line: the survey logs no phase",
        notes=(
            (
                "this survey ran the parallel code (the build's revision contains the commit "
                "that parallelised it) and lies wholly inside the nightly window, not partly",
                _parallel_survey_wholly_inside,
            ),
        ),
    ),
    Figure(
        key="eaf_survey_sept",
        figure="OGS-00009 EAF survey and orientation check (Sept, clean)",
        epic="1h55m33s for 2,024 Analyses",
        epic_seconds=6933.0,
        derive=_eaf_survey(
            store=SEPT_BUILD, after=r"^Pass 2: 2024/2024 done", contains=PARALLEL_SURVEY_COMMIT
        ),
        method="last Pass 2 progress line to the `Encoding plan` line, in the September "
        "build's own record",
        notes=(
            (
                "this survey ran the serial code (the build's revision predates the commit that "
                "parallelised it) and lies outside every declared window",
                _serial_survey_outside_windows,
            ),
        ),
    ),
    Figure(
        key="sept_build",
        figure="OGS-00009 build (Sept)",
        epic="5.92 h",
        epic_seconds=21312.0,
        derive=Derive("step", store=SEPT_BUILD, step="build"),
        method="the replaced Store's step record, elapsed_seconds",
    ),
    Figure(
        key="residual_encode_oct",
        figure="OGS-00009 residual EAF encode (Oct, format 0.2.0)",
        epic="1h28m45s",
        epic_seconds=5325.0,
        derive=_residual(),
        method="`Band-write eaf: done` to the `eaf: int8 residual` line: the encode logs progress, "
        "not a phase",
    ),
    Figure(
        key="residual_encode_sept",
        figure="OGS-00009 residual EAF encode (Sept, format 0.1.0)",
        epic="30m01s",
        epic_seconds=1801.0,
        derive=_residual(store=SEPT_BUILD, after=r"^Band-write eaf: 2024/2024 done"),
        method="the last `Band-write eaf` line to the `eaf: int8 residual` line, in the "
        "September build's own record (it logged progress, not a phase)",
    ),
    Figure(
        key="ogs00010_completion",
        figure="OGS-00010 Reference Completion (Sept)",
        epic="8.80 h (2,024 Analyses, 1,357 LD blocks, 32 workers)",
        epic_seconds=31680.0,
        derive=Derive("step", store=RETIRED_COMPLETION, step="complete"),
        method="the retired Store's step record, elapsed_seconds",
    ),
    _unavailable_figure(
        "ogs00012_completion_block",
        "OGS-00012 Reference Completion block phase",
        "4h28m for 1,357 blocks x 1,669 Analyses (0.227 worker-s)",
        16080.0,
        (
            "Reference Completion logs no block phase before #309; the record holds only its SE "
            "and Top-Hit Index phases",
            _record_lacks_the_phase,
        ),
        store="OGS-00012",
        step="complete",
        check="no_phase_matching",
        pattern="block",
    ),
    _unavailable_figure(
        "ogs00012_band_write",
        "OGS-00012 Reference Completion band write",
        "over 1.5 h in one process at about 26% CPU",
        5400.0,
        (
            "Reference Completion logs no band-write phase before #309; the live observation is "
            "in no record",
            _record_lacks_the_phase,
        ),
        store="OGS-00012",
        step="complete",
        check="no_phase_matching",
        pattern="band",
    ),
    _unavailable_figure(
        "pyspy_shares",
        "The epic's py-spy shares (Pass 2 worker, validation, completion)",
        "reader ~39% / key match ~38%; Blosc decode 58%; store reads 9%",
        None,
        (
            "the shares are py-spy samples of live jobs; no record in this run mentions them, "
            "and a re-run would sample whatever runs then",
            _no_record_mentions_it,
        ),
        check="no_mention",
        pattern=r"py[-_ ]?spy",
    ),
    # --- the reader A/B, one real file each ----------------------------------
    _control_row(
        "reader_finngen",
        "FinnGen R13 AB1_ACTINOMYCOSIS, the epic's revision (control)",
        "finngen_R13_AB1_ACTINOMYCOSIS",
        "205.5 s (21,326,102 rows)",
        205.5,
        21326102,
    ),
    _floor_row(
        "reader_finngen_floor",
        "FinnGen R13 AB1_ACTINOMYCOSIS, parse floor",
        "finngen_R13_AB1_ACTINOMYCOSIS",
        "26.4 s (pandas C engine, seven columns plus z)",
        26.4,
        "pandas",
    ),
    _control_row(
        "reader_ukb",
        "ukb-b-10001 GWAS-VCF, the epic's revision (control)",
        "ukb-b-10001",
        "16.1 s (5,352,901 associations)",
        16.1,
        5352901,
    ),
    _floor_row(
        "reader_ukb_floor",
        "ukb-b-10001 GWAS-VCF, parse floor",
        "ukb-b-10001",
        "8.2 s (bcftools query alone)",
        8.2,
        "bcftools",
    ),
    _ratio_row(
        "reader_finngen_control_over_floor",
        "FinnGen R13 AB1_ACTINOMYCOSIS, the epic's revision over the parse floor",
        "finngen_R13_AB1_ACTINOMYCOSIS",
        "7.78x (205.5 s over 26.4 s)",
        7.78,
        205.5,
        26.4,
    ),
    _ratio_row(
        "reader_ukb_control_over_floor",
        "ukb-b-10001 GWAS-VCF, the epic's revision over the parse floor",
        "ukb-b-10001",
        "1.96x (16.1 s over 8.2 s)",
        1.96,
        16.1,
        8.2,
    ),
    _paired_row(
        "reader_finngen_head_vs_control",
        "FinnGen reader, this checkout against the epic's revision",
        "finngen_R13_AB1_ACTINOMYCOSIS",
    ),
    _paired_row(
        "reader_ukb_head_vs_control",
        "ukb-b-10001 reader, this checkout against the epic's revision",
        "ukb-b-10001",
    ),
    # --- OGS-00009 end to end, outside the window ----------------------------
    Figure(
        key="build_pass1",
        figure="OGS-00009 Pass 1, liftover, key lookup",
        epic="34 min",
        epic_seconds=2040.0,
        derive=_pass1(),
        method="first log line to the `Pass 2: resolving` line",
    ),
    Figure(
        key="build_pass2",
        figure="OGS-00009 Pass 2",
        epic="64 min",
        epic_seconds=3840.0,
        derive=_projection("OGS-00009", "build", "Pass 2"),
        method="the clean pace over intervals outside every window, projected over 2,514",
        reason=(
            "the epic projects the pace of its first in-window line, which already carries the "
            "window's onset; the pace outside the window is faster",
            _onset_in_first_inside,
        ),
    ),
    Figure(
        key="build_eaf_survey",
        figure="OGS-00009 EAF survey and orientation check",
        epic="2h23m (Sept scaled to 2,514)",
        epic_seconds=8580.0,
        derive=_eaf_survey(contains=PARALLEL_SURVEY_COMMIT),
        method="as the Oct EAF survey row",
        reason=(
            "the epic scaled September's survey, which ran the serial code; October's ran the "
            "parallel code, wholly inside the window, so the two are different code and the "
            "October one cannot be measured cleanly",
            _parallel_survey_wholly_inside,
        ),
    ),
    Figure(
        key="build_planes",
        figure="OGS-00009 statistic-plane creation",
        epic="15 min",
        epic_seconds=900.0,
        derive=_planes(),
        method="the `Encoding plan` line to `Band-write z: start`",
        reason=(
            "the epic counts this as measured, but the interval runs wholly inside the window, "
            "so it is a window reading",
            _inside_window,
        ),
    ),
    Figure(
        key="build_band_z",
        figure="OGS-00009 band-write z",
        epic="35 min",
        epic_seconds=2100.0,
        derive=_phase("OGS-00009", "build", "Band-write z", kind="phase_clean"),
        method="the phase's span, or its clean-rate projection when a window covers it",
        reason=(
            "the epic inferred this from se and eaf; here the phase runs inside the window and "
            "logs no interval outside it",
            _no_clean_rate,
        ),
    ),
    Figure(
        key="build_band_se",
        figure="OGS-00009 band-write se",
        epic="30 min",
        epic_seconds=1800.0,
        derive=_phase("OGS-00009", "build", "Band-write se"),
        method="the phase's span",
    ),
    Figure(
        key="build_band_eaf",
        figure="OGS-00009 band-write eaf",
        epic="38 min",
        epic_seconds=2280.0,
        derive=_phase("OGS-00009", "build", "Band-write eaf"),
        method="the phase's span",
    ),
    Figure(
        key="build_residual",
        figure="OGS-00009 residual EAF encode",
        epic="1h29m",
        epic_seconds=5340.0,
        derive=_residual(),
        method="as the residual encode row",
    ),
    Figure(
        key="build_se_and_index",
        figure="OGS-00009 SE fit and rewrite; inline Top-Hit Index",
        epic="18 min",
        epic_seconds=1080.0,
        derive=_se_and_index(),
        method="`SE fit: start` to `Top-hit index write: done`",
    ),
    Figure(
        key="build_clean",
        figure="OGS-00009 build, clean phases",
        epic="7h46m (nine components: Pass 1, Pass 2, EAF survey, planes, z, se, eaf, "
        "residual, SE and index)",
        epic_seconds=27960.0,
        derive=Derive(
            "composed",
            parts=(
                ("Pass 1, liftover, key lookup", _pass1()),
                ("Pass 2 (clean projection)", _projection("OGS-00009", "build", "Pass 2")),
                ("EAF survey and orientation check", _eaf_survey()),
                ("Statistic-plane creation", _planes()),
                ("Band-write z", _phase("OGS-00009", "build", "Band-write z", "phase_clean")),
                ("Band-write se", _phase("OGS-00009", "build", "Band-write se", "phase_clean")),
                ("Band-write eaf", _phase("OGS-00009", "build", "Band-write eaf", "phase_clean")),
                ("Residual EAF encode", _residual()),
                ("SE fit, rewrite and inline Top-Hit Index", _se_and_index()),
            ),
        ),
        method="the sum of the components that are clean and directly measured, with every "
        "component listed and the excluded ones given their raw spans",
        epic_like_for_like=LikeForLike(
            "the epic's own values for the six components the record can measure cleanly",
            (
                ("Pass 1, liftover, key lookup", 2040.0),
                ("Pass 2 (clean projection)", 3840.0),
                ("Band-write se", 1800.0),
                ("Band-write eaf", 2280.0),
                ("Residual EAF encode", 5340.0),
                ("SE fit, rewrite and inline Top-Hit Index", 1080.0),
            ),
        ),
        reason=(
            "components of the epic's total ran inside the window here, so the sum leaves them "
            "out and is a lower bound",
            _excluded_part_ran_inside_a_window,
        ),
    ),
    Figure(
        key="build_standalone_rebuild",
        figure="OGS-00009 standalone Top-Hit Index rebuild",
        epic="1h21m",
        epic_seconds=4860.0,
        derive=Derive("step", store="OGS-00009", step="top-hits"),
        method="the step record's elapsed_seconds",
    ),
    Figure(
        key="build_hm3",
        figure="OGS-00009 Indexed Variant Subset (HM3)",
        epic="13 min",
        epic_seconds=780.0,
        derive=Derive("step", store="OGS-00009", step="indexed-subset-hm3"),
        method="the step record's elapsed_seconds",
    ),
    Figure(
        key="completion",
        figure="OGS-00009 Reference Completion",
        epic="about 10h54m (Sept OGS-00010 scaled)",
        epic_seconds=39240.0,
        derive=Derive("step", store="OGS-00009", step="complete"),
        method="the step record's elapsed_seconds",
        reason=(
            "a direct measurement of the October run, most of it inside that night's window, "
            "against a September projection from another store and panel; completion logs no "
            "block phase before #309, so the window's share cannot be attributed",
            _mostly_inside_window,
        ),
    ),
    Figure(
        key="completion_validate",
        figure="OGS-00009 validation of the completed release",
        epic="about 3h30m (2.3-3.9 h range)",
        epic_seconds=12600.0,
        derive=Derive("step", store="OGS-00009", step="completion-validate"),
        method="the step record's elapsed_seconds",
        reason=(
            "a direct measurement, above the range projected from other stores, and it starts "
            "inside that night's window",
            _started_inside_window,
        ),
    ),
    # --- the epic's roll-ups and its after-the-change column -----------------
    _unavailable_figure(
        "ogs00009_total",
        "OGS-00009 total (build through validation)",
        "at least 12h30m",
        45000.0,
        (
            "a sum of the rows above, at least one of which is not a clean reproduction, so "
            "the sum is not re-derived as a point value",
            _a_part_is_not_a_clean_reproduction,
        ),
        check="roll_up",
        components=("build_clean", "build_standalone_rebuild", "build_hm3", "ogs00009_validate"),
    ),
    _unavailable_figure(
        "completion_total",
        "Completion total",
        "about 14h24m",
        51840.0,
        (
            "a sum of the rows above, at least one of which is not a clean reproduction, so "
            "the sum is not re-derived as a point value",
            _a_part_is_not_a_clean_reproduction,
        ),
        check="roll_up",
        components=("completion", "completion_validate"),
    ),
    _unavailable_figure(
        "both",
        "Both stores",
        "about 27 h",
        97200.0,
        (
            "a sum of the rows above, at least one of which is not a clean reproduction, so "
            "the sum is not re-derived as a point value",
            _a_part_is_not_a_clean_reproduction,
        ),
        check="roll_up",
        components=("ogs00009_total", "completion_total"),
    ),
    _unavailable_figure(
        "after_steps_2_to_4",
        "Every 'after steps 2-4' value in the epic's tables",
        "19.4 h, and the per-phase projections under it",
        None,
        (
            "projections of changes the epic's steps 2-4 make; every record in this run was "
            "written before the epic was read, so none was made with them",
            _every_record_predates_the_epic,
        ),
        check="predates_epic",
    ),
)


REPEATED_FIGURES = (
    "OGS-00009's standalone Top-Hit Index rebuild appears in two of the epic's tables "
    "(`ogs00009_top_hits` at 1.35 h and `build_standalone_rebuild` at 1h21m) and is measured "
    "from the same record under both keys; its validation appears once, in both tables' terms. "
    "Nothing is deduplicated, so a reader can see that the two epic values agree."
)


def compare(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Every epic figure measured against this run's records.

    A roll-up row reads the verdicts of the rows above it, so the rows are
    measured in order and each verdict is handed to the ones after.
    """
    verdicts: dict[str, str] = {}
    figures: list[dict[str, Any]] = []
    for figure in FIGURES:
        row = measure_figure(figure, {**evidence, "verdicts": verdicts})
        verdicts[figure.key] = row["verdict"]
        figures.append(row)
    return {
        "epic": {
            "issue": EPIC_ISSUE,
            "read_at": EPIC_READ_AT,
            "note": "the epic's values are quoted from its Evidence tables on that date",
        },
        "rule": REPRODUCES_RULE,
        "figures": figures,
        "repeated_figures": REPEATED_FIGURES,
    }


def _git_ancestry(commit: str, revision: str) -> bool | None:
    return is_ancestor(commit, revision, repo=REPO)


def evidence(
    stores: Sequence[Mapping[str, Any]],
    windows: Sequence[Any],
    reader_ab: Sequence[Mapping[str, Any]],
    *,
    is_ancestor_in: Callable[[str, str], bool | None] | None = None,
) -> dict[str, Any]:
    """The lookup the derivations read: records by store and step, and the A/B.

    `is_ancestor_in(commit, revision)` answers the ancestry questions; it asks
    this repository's git unless a caller supplies another.
    """
    return {
        "is_ancestor": is_ancestor_in or _git_ancestry,
        "stores": {
            store["label"]: {record.step: record for record in store["records"]}
            for store in stores
        },
        "windows": list(windows),
        "reader_ab": list(reader_ab),
    }
