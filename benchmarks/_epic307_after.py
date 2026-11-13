"""Epic #307's after-measurement: OGS-00009 rebuilt end to end on the finished engine.

#322 rebuilt OGS-00009 with every engine ticket of the epic merged and ran it as
the Store Release Build plans it after opengwasdb-stores#357 and #358: the Dense
build with its inline Top-Hit Index and no standalone rebuild, validation across
workers, the hm3 subset, Reference Completion and the completed release's checks.
This module sets each phase of that run beside two things:

* #311's baseline -- the same derivation `_epic307_figures` makes from the
  October records of the published OGS-00009, so the two columns are measured
  the same way; and
* every projection the epic made for that phase ("after steps 2-4", and the two
  step-5 scenarios), each marked `met` or `missed`.

A projection is met when the measurement is no more than 5% (or one second)
above it. A miss carries a reason in `_epic307_figures`' form: a sentence and a
predicate over the row's own detail, printed as `reason no longer holds` when
the data has moved away from it. A row whose after value cannot be measured is
`unmeasured`, never a zero. The one exception is a step the plan no longer
holds: the standalone Top-Hit Index rebuild is measured as zero seconds because
the after store has no record of it, and its row says so.

Milestone #388's line -- a Store Release the size of OGS-00009, Reference
Completion included, in an evening (about 9 h) -- is judged on the after run's
span from its first step's start to its last step's end.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from benchmarks._epic307_figures import (
    FIGURES,
    TOLERANCE,
    TOLERANCE_SECONDS,
    Derive,
    Measured,
    _measure,
)
from benchmarks._step_records import Record, format_duration

#: The Store whose October records are #311's baseline.
BASELINE_STORE = "OGS-00009"

#: Stands for the after Store's label in the derivations below.
AFTER = "<after>"

#: Milestone #388's "an evening": the hours between the backup window's end and
#: its start, as the milestone states them.
EVENING_SECONDS = 9 * 3600.0

SCENARIOS = {
    "steps_2_4": "after steps 2-4 (#312-#319, opengwasdb-stores#357/#358)",
    "residual": "with the residual EAF encode parallelised as well (step 5, #320)",
    "survey": "if the EAF survey's cause also yields (step 5, #310)",
}

#: The epic's issue and the date its projections were read.
EPIC_ISSUE = 307
EPIC_READ_AT = "2026-11-13"

MET_RULE = (
    "a projection is met when the after value is at most 5% (or one second) above it; "
    "a projection of zero is met only by a step the plan no longer runs"
)

Reason = tuple[str, Callable[[Mapping[str, Any]], bool]]


@dataclass(frozen=True)
class Projection:
    scenario: str
    seconds: float
    epic: str


@dataclass(frozen=True)
class Row:
    """One phase, measured on both Stores, with the epic's projections for it."""

    key: str
    phase: str
    epic_now: str
    projections: tuple[Projection, ...] = ()
    #: `_epic307_figures.FIGURES` key whose derivation is the baseline.
    baseline_figure: str | None = None
    #: The baseline's derivation, for a row no figure of the epic's measures.
    baseline: Derive | None = None
    #: The after Store's derivation, with `AFTER` as its store.
    after: Derive | None = None
    #: A roll-up: the keys of the rows it sums, on both sides.
    parts: tuple[str, ...] = ()
    #: Printed for a miss while its predicate holds of the row's detail.
    miss_reasons: tuple[Reason, ...] = ()
    #: Printed whatever the verdict, while its predicate holds.
    notes: tuple[Reason, ...] = ()


def _p(scenario: str, seconds: float, epic: str) -> Projection:
    return Projection(scenario, seconds, epic)


def _reads_gwas_vcf(detail: Mapping[str, Any]) -> bool:
    argv = detail["attribution"].get("build_argv") or []
    return "opengwasdb.gwas-vcf" in argv


#: Why Pass 2 misses ~30 min. The epic's figure halved October's, which is what
#: it said a faster parse could give a VCF source at most.
PASS2_PER_ROW: Reason = (
    "OGS-00009's sources are GWAS-VCF, and that reader offers no association batches: "
    "Pass 2 still reads each association one at a time, so it gained the integer key join "
    "(#316) but none of the columnar parse the epic's figure assumed (#318 and #319 gave "
    "batches to FinnGen and GWAS-SSF sources only)",
    _reads_gwas_vcf,
)


def _after_interval(after: str, until: str) -> Derive:
    return Derive("interval", store=AFTER, step="build", after=after, until=until)


def _after_phase(step: str, label: str) -> Derive:
    return Derive("phase", store=AFTER, step=step, label=label)


def _after_step(step: str) -> Derive:
    return Derive("step", store=AFTER, step=step)


ROWS: tuple[Row, ...] = (
    Row(
        "pass1",
        "Pass 1, liftover, key lookup",
        "34 min",
        (_p("steps_2_4", 2040.0, "34 min"),),
        baseline_figure="build_pass1",
        after=_after_interval(r"^Pass 1: collecting source variants", r"^Pass 2: resolving"),
    ),
    Row(
        "pass2",
        "Pass 2",
        "64 min",
        (_p("steps_2_4", 1800.0, "~30 min"),),
        baseline_figure="build_pass2",
        after=_after_interval(r"^Pass 2: resolving", r"^Pass 2: 2514/2514 done"),
        miss_reasons=(PASS2_PER_ROW,),
    ),
    Row(
        "eaf_survey",
        "EAF survey and orientation check",
        "2h23m",
        (_p("steps_2_4", 8580.0, "2h23m"), _p("survey", 1800.0, "~30 min if its cause yields")),
        baseline_figure="build_eaf_survey",
        after=_after_interval(r"^Pass 2: 2514/2514 done", r"^Encoding plan:"),
    ),
    Row(
        "planes",
        "Statistic-plane creation",
        "15 min",
        (_p("steps_2_4", 900.0, "15 min"),),
        baseline_figure="build_planes",
        after=_after_interval(r"^Encoding plan:", r"^Band-write z: start$"),
    ),
    Row(
        "band_z",
        "Band-write z",
        "35 min",
        (_p("steps_2_4", 2100.0, "35 min"),),
        baseline_figure="build_band_z",
        after=_after_phase("build", "Band-write z"),
    ),
    Row(
        "band_se",
        "Band-write se",
        "30 min",
        (_p("steps_2_4", 1800.0, "30 min"),),
        baseline_figure="build_band_se",
        after=_after_phase("build", "Band-write se"),
    ),
    Row(
        "band_eaf",
        "Band-write eaf",
        "38 min",
        (_p("steps_2_4", 2280.0, "38 min"),),
        baseline_figure="build_band_eaf",
        after=_after_phase("build", "Band-write eaf"),
    ),
    Row(
        "residual",
        "Residual EAF encode",
        "1h29m",
        (_p("steps_2_4", 5340.0, "1h29m"), _p("residual", 900.0, "~15 min if parallelised")),
        baseline_figure="build_residual",
        after=_after_interval(r"^Band-write eaf: done in", r"^eaf: int8 residual"),
    ),
    Row(
        "se_and_index",
        "SE fit and rewrite; inline Top-Hit Index",
        "18 min",
        (_p("steps_2_4", 1080.0, "18 min"),),
        baseline_figure="build_se_and_index",
        after=_after_interval(r"^SE fit: start$", r"^Top-hit index write: done in"),
    ),
    Row(
        "build",
        "Build",
        "7h46m",
        (_p("steps_2_4", 25920.0, "7h12m"),),
        baseline=Derive("step", store=BASELINE_STORE, step="build"),
        after=_after_step("build"),
    ),
    Row(
        "top_hits",
        "Standalone Top-Hit Index rebuild",
        "1h21m",
        (_p("steps_2_4", 0.0, "0 (not planned after a Dense build)"),),
        baseline_figure="build_standalone_rebuild",
        after=Derive("not_planned", store=AFTER, step="top-hits"),
    ),
    Row(
        "hm3",
        "Indexed Variant Subset (HM3)",
        "13 min",
        (_p("steps_2_4", 780.0, "13 min"),),
        baseline_figure="build_hm3",
        after=_after_step("indexed-subset-hm3"),
    ),
    Row(
        "validate",
        "Validation",
        "at least 3h10m",
        (_p("steps_2_4", 1500.0, "~25 min"),),
        baseline_figure="ogs00009_validate",
        after=_after_step("validate"),
    ),
    Row(
        "ogs00009_total",
        "OGS-00009 total (build, Top-Hit Index, subset, validation)",
        "at least 12h30m",
        (
            _p("steps_2_4", 28200.0, "~7h50m"),
            _p("residual", 23760.0, "~6h36m"),
            _p("survey", 16980.0, "~4h43m"),
        ),
        parts=("build", "top_hits", "hm3", "validate"),
    ),
    Row(
        "completion",
        "Reference Completion",
        "~10h54m",
        (_p("steps_2_4", 39240.0, "~10h54m"),),
        baseline_figure="completion",
        after=_after_step("complete"),
    ),
    Row(
        "completion_validate",
        "Validation of the completed release",
        "~3h30m",
        (_p("steps_2_4", 2400.0, "~40 min"),),
        baseline_figure="completion_validate",
        after=_after_step("completion-validate"),
    ),
    Row(
        "completion_total",
        "Completion total (completion and its validation)",
        "~14h24m",
        (_p("steps_2_4", 41640.0, "~11h34m"),),
        parts=("completion", "completion_validate"),
    ),
    Row(
        "both",
        "Both (the epic's sum)",
        "~27 h",
        (
            _p("steps_2_4", 69840.0, "~19.4 h"),
            _p("residual", 65520.0, "~18.2 h"),
            _p("survey", 58680.0, "~16.3 h"),
        ),
        parts=("ogs00009_total", "completion_total"),
    ),
)

#: Steps the epic's sums leave out but a Store Release Build runs; the after
#: Store Release's own total is every planned step.
UNPROJECTED_STEPS = (
    "overview",
    "completion-indexed-subset-hm3",
    "completion-overview",
    "completion-observed-view-check",
)

#: Phases inside the after run's long steps that the epic did not project, kept
#: for attribution: completion's (#309) and the EAF survey's (#310).
ATTRIBUTION_PHASES = (
    ("build", "EAF survey and orientation verification"),
    ("build", "Encode eaf"),
    ("complete", "Axis construction"),
    ("complete", "LD-block completion"),
    ("complete", "Checkpoint merge"),
    ("complete", "Band write"),
    ("complete", "Top-Hit Index"),
)


def _store_records(evidence: Mapping[str, Any], label: str) -> Mapping[str, Record]:
    stores: Mapping[str, Mapping[str, Record]] = evidence["stores"]
    if label not in stores:
        raise SystemExit(f"--after {label!r} names no --store of this run")
    return stores[label]


def _measure_after(derive: Derive, label: str, evidence: Mapping[str, Any]) -> Measured:
    """The after Store's value: `_epic307_figures`' derivation, or a step not planned."""
    concrete = replace(derive, store=label)
    if concrete.kind == "not_planned":
        record = _store_records(evidence, label).get(concrete.step or "")
        if record is None:
            return Measured(0.0, True, {"planned": False, "step": concrete.step})
        return Measured(record.wall_seconds, True, {"planned": True, "step": concrete.step})
    return _measure(concrete, evidence)


def _baseline(row: Row, evidence: Mapping[str, Any]) -> Measured:
    if row.baseline is not None:
        return _measure(row.baseline, evidence)
    figure = next(figure for figure in FIGURES if figure.key == row.baseline_figure)
    return _measure(figure.derive, evidence)


def _verdict(after: float | None, projection: Projection) -> str:
    if after is None:
        return "unmeasured"
    if projection.seconds == 0.0:
        return "met" if after == 0.0 else "missed"
    allowed = projection.seconds + max(TOLERANCE * projection.seconds, TOLERANCE_SECONDS)
    return "met" if after <= allowed else "missed"


def _side(measured: Measured) -> dict[str, Any]:
    return {
        "seconds": None if measured.seconds is None else round(measured.seconds, 3),
        "rendered": format_duration(measured.seconds),
        "clean": measured.clean,
        "inside_window": measured.inside_window,
        "detail": measured.detail,
    }


def _rolled_up(parts: Sequence[Mapping[str, Any]], side: str) -> dict[str, Any]:
    """A sum of rows already measured; unknown if any part is."""
    values = [part[side]["seconds"] for part in parts]
    known = None not in values
    inside = [part["key"] for part in parts if part[side]["inside_window"]]
    seconds = round(sum(values), 3) if known else None
    return {
        "seconds": seconds,
        "rendered": format_duration(seconds),
        "clean": known and not inside,
        "inside_window": bool(inside) if known else None,
        "detail": {"parts": [part["key"] for part in parts], "parts_inside_window": inside},
    }


def _reasons(reasons: Sequence[Reason], data: Mapping[str, Any]) -> list[str]:
    return [text if holds(data) else "reason no longer holds" for text, holds in reasons]


def _row_detail(
    row: Row, after: Mapping[str, Any], baseline: Mapping[str, Any], attribution: Mapping[str, Any]
) -> dict[str, Any]:
    """What a reason's predicate may read: both sides and the after run's phases."""
    return {
        "after_seconds": after["seconds"],
        "baseline_seconds": baseline["seconds"],
        "baseline_inside_window": baseline["inside_window"],
        "after_detail": after["detail"],
        "attribution": attribution,
    }


def _finish_row(
    row: Row, after: dict[str, Any], baseline: dict[str, Any], attribution: Mapping[str, Any]
) -> dict[str, Any]:
    data = _row_detail(row, after, baseline, attribution)
    projections = []
    for projection in row.projections:
        verdict = _verdict(after["seconds"], projection)
        projections.append(
            {
                "scenario": projection.scenario,
                "scenario_text": SCENARIOS[projection.scenario],
                "epic": projection.epic,
                "epic_seconds": projection.seconds,
                "verdict": verdict,
                "reasons": _reasons(row.miss_reasons, data) if verdict == "missed" else [],
            }
        )
    ratio = (
        round(baseline["seconds"] / after["seconds"], 3)
        if baseline["seconds"] and after["seconds"]
        else None
    )
    return {
        "key": row.key,
        "phase": row.phase,
        "epic_now": row.epic_now,
        "baseline": baseline,
        "after": after,
        "baseline_over_after": ratio,
        "projections": projections,
        "notes": _reasons(row.notes, data),
    }


def attribution(evidence: Mapping[str, Any], label: str) -> dict[str, Any]:
    """The after run's phases the epic did not project, and its unprojected steps."""
    phases = {}
    for step, phase in ATTRIBUTION_PHASES:
        measured = _measure(Derive("phase", store=label, step=step, label=phase), evidence)
        seconds = None if measured.seconds is None else round(measured.seconds, 3)
        phases[f"{step}: {phase}"] = seconds
    records = _store_records(evidence, label)
    steps = {
        step: None if records.get(step) is None else records[step].wall_seconds
        for step in UNPROJECTED_STEPS
    }
    build = records.get("build")
    return {
        "phases": phases,
        "unprojected_steps": steps,
        "build_argv": list(build.argv) if build is not None and build.argv else None,
    }


def _block_phase_fills_the_overrun(detail: Mapping[str, Any]) -> bool:
    block = detail["attribution"]["phases"].get("complete: LD-block completion")
    over, span = detail["over_by_seconds"], detail["span_seconds"]
    return block is not None and over is not None and block > over and block > 0.4 * span


#: Why the evening is missed, while the data says so.
EVENING_MISS_REASONS: tuple[Reason, ...] = (
    (
        "Reference Completion's LD-block phase, which no ticket in this epic changed, is the "
        "largest part of the run, and the run without it would fit in an evening: the block "
        "fits (one elastic-net fit with cross-validation per LD block and Analysis) are what "
        "remains between this Store Release and the milestone",
        _block_phase_fills_the_overrun,
    ),
)


def store_release(
    evidence: Mapping[str, Any], label: str, attributed: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The after Store Release's wall time, every planned step, and #388's line."""
    records = _store_records(evidence, label)
    walls = {step: record.wall_seconds for step, record in records.items()}
    starts = [record.log_start for record in records.values() if record.log_start]
    ends = [record.log_end for record in records.values() if record.log_end]
    span = (max(ends) - min(starts)).total_seconds() if starts and ends else None
    summed = None if None in walls.values() else round(sum(v for v in walls.values() if v), 3)
    over = None if span is None else round(span - EVENING_SECONDS, 3)
    verdict = "unmeasured" if span is None else ("met" if span <= EVENING_SECONDS else "missed")
    data = {
        "attribution": attributed or attribution(evidence, label),
        "over_by_seconds": over,
        "span_seconds": span,
    }
    reasons = _reasons(EVENING_MISS_REASONS, data) if verdict == "missed" else []
    return {
        "steps": walls,
        "steps_summed_seconds": summed,
        "span_seconds": None if span is None else round(span, 3),
        "span": format_duration(span),
        "first_start": min(starts).isoformat() if starts else None,
        "last_end": max(ends).isoformat() if ends else None,
        "milestone_388": {
            "line": (
                "a Store Release the size of OGS-00009, Reference Completion included, "
                "finishes in an evening (about 9 h), outside the backup window"
            ),
            "target_seconds": EVENING_SECONDS,
            "verdict": verdict,
            "reasons": reasons,
            "over_by_seconds": over,
            "over_by": format_duration(abs(over)) if over is not None else None,
            "measured_on": "the span from the first step's start to the last step's end",
        },
    }


def compare_after(evidence: Mapping[str, Any], label: str) -> dict[str, Any]:
    """Every row of the epic's OGS-00009 table, before and after, against its projections."""
    _store_records(evidence, label)
    attributed = attribution(evidence, label)
    done: dict[str, dict[str, Any]] = {}
    for row in ROWS:
        if row.parts:
            parts = [done[key] for key in row.parts]
            after, baseline = _rolled_up(parts, "after"), _rolled_up(parts, "baseline")
        else:
            if row.after is None:
                raise ValueError(f"row {row.key} has neither a derivation nor parts")
            after = _side(_measure_after(row.after, label, evidence))
            baseline = _side(_baseline(row, evidence))
        done[row.key] = _finish_row(row, after, baseline, attributed)
    return {
        "after_store": label,
        "baseline_store": BASELINE_STORE,
        "epic": {"issue": EPIC_ISSUE, "read_at": EPIC_READ_AT, "scenarios": SCENARIOS},
        "rule": MET_RULE,
        "rows": list(done.values()),
        "attribution": attributed,
        "store_release": store_release(evidence, label, attributed),
    }
