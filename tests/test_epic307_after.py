"""Epic #307's after-measurement (#322): each phase before and after, against its projections.

These pin what could make the after table say something its records do not:

- a projection is met within the stated tolerance and missed beyond it, and an
  unmeasured phase is `unmeasured`, never met;
- the baseline column is measured from OGS-00009's records and the after column
  from the after Store's, by the same derivations;
- a step the plan no longer runs is zero only when the after Store has no record
  of it;
- a sum with an unknown part is unknown;
- milestone #388 is judged on the after run's span.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks._epic307_after import (
    EVENING_SECONDS,
    ROWS,
    Projection,
    _verdict,
    compare_after,
)
from benchmarks._epic307_figures import evidence
from benchmarks._step_records import measure_store


def _line(at: str, message: str) -> str:
    return f"{at} INFO opengwasdb.some.phase: {message}"


def _store(root: Path, label: str, steps: dict[str, tuple[str, float]]) -> dict:
    """A Store whose records are `step -> (stderr, wall seconds)`."""
    records = root / label / "records"
    records.mkdir(parents=True)
    for step, (stderr, wall) in steps.items():
        payload = {
            "step": step,
            "store_id": "OGS-00009",
            "exit_code": 0,
            "success": True,
            "start_time": None,
            "end_time": None,
            "elapsed_seconds": wall,
            "argv": ["opengwasdb", step],
            "stderr": stderr,
            "opengwasdb_rev": "8acd22b02a7f61656cf9079e38939fb0ff255581",
        }
        (records / f"{step}.json").write_text(json.dumps(payload), encoding="utf-8")
    return measure_store(root / label, tz="Europe/London", label=label)


def _build_log(day: str, pass1_minutes: int) -> str:
    """A build log whose Pass 1 runs `pass1_minutes` from midnight on `day`."""
    end = f"{day} 00:{pass1_minutes:02d}:00"
    return "\n".join(
        [
            _line(f"{day} 00:00:00", "Pass 1: collecting source variants from 2514 files"),
            _line(end, "Pass 2: resolving 2514 analyses × 9847698 variants (n_workers=32)"),
        ]
    )


def _evidence(*stores: dict) -> dict:
    return evidence(stores, [], [], is_ancestor_in=lambda commit, revision: None)


def _row(after: dict, key: str) -> dict:
    return next(row for row in after["rows"] if row["key"] == key)


def test_a_projection_is_met_within_five_percent_and_missed_beyond():
    projection = Projection("steps_2_4", 1000.0, "1000 s")
    assert _verdict(1050.0, projection) == "met"
    assert _verdict(1050.5, projection) == "missed"
    assert _verdict(None, projection) == "unmeasured"
    small = Projection("steps_2_4", 10.0, "10 s")
    assert _verdict(11.0, small) == "met", "one second of slack for a short phase"
    assert _verdict(11.5, small) == "missed"
    zero = Projection("steps_2_4", 0.0, "0")
    assert _verdict(0.0, zero) == "met"
    assert _verdict(0.5, zero) == "missed", "a step still planned never meets a zero"


def test_each_column_is_measured_from_its_own_stores_records(tmp_path):
    baseline = _store(tmp_path, "OGS-00009", {"build": (_build_log("2026-10-21", 34), 70000.0)})
    after = _store(tmp_path, "after", {"build": (_build_log("2026-11-13", 20), 14000.0)})
    assert baseline["records"][0].wall_seconds != after["records"][0].wall_seconds

    result = compare_after(_evidence(baseline, after), "after")

    pass1 = _row(result, "pass1")
    assert pass1["baseline"]["seconds"] == 34 * 60
    assert pass1["after"]["seconds"] == 20 * 60
    assert pass1["projections"][0]["verdict"] == "met"
    build = _row(result, "build")
    assert build["baseline"]["seconds"] == 70000.0
    assert build["after"]["seconds"] == 14000.0
    assert build["baseline_over_after"] == 5.0
    assert build["projections"][0]["verdict"] == "met"


def test_a_step_the_plan_no_longer_runs_is_zero_only_without_its_record(tmp_path):
    baseline = _store(tmp_path, "OGS-00009", {"top-hits": ("", 4852.0)})
    without = _store(tmp_path, "without", {"build": ("", 100.0)})
    still = _store(tmp_path, "still", {"build": ("", 100.0), "top-hits": ("", 300.0)})

    dropped = _row(compare_after(_evidence(baseline, without), "without"), "top_hits")
    kept = _row(compare_after(_evidence(baseline, still), "still"), "top_hits")

    assert dropped["after"]["seconds"] == 0.0
    assert dropped["after"]["detail"] == {"planned": False, "step": "top-hits"}
    assert dropped["projections"][0]["verdict"] == "met"
    assert kept["after"]["seconds"] == 300.0
    assert kept["projections"][0]["verdict"] == "missed"


def test_a_sum_with_an_unmeasured_part_is_unmeasured(tmp_path):
    baseline = _store(tmp_path, "OGS-00009", {"build": ("", 70000.0)})
    after = _store(tmp_path, "after", {"build": ("", 14000.0), "indexed-subset-hm3": ("", 600.0)})

    result = compare_after(_evidence(baseline, after), "after")

    assert _row(result, "validate")["after"]["seconds"] is None
    total = _row(result, "ogs00009_total")
    assert total["after"]["seconds"] is None
    assert {p["verdict"] for p in total["projections"]} == {"unmeasured"}
    assert {p["verdict"] for p in _row(result, "both")["projections"]} == {"unmeasured"}


def test_milestone_388_is_judged_on_the_runs_span(tmp_path):
    baseline = _store(tmp_path, "OGS-00009", {"build": ("", 70000.0)})
    first = _line("2026-11-13 02:00:00", "Pass 1: collecting source variants from 2514 files")
    last = _line("2026-11-13 12:00:00", "Validation: done in 10m00s")
    after = _store(
        tmp_path,
        "after",
        {"build": (first, 30000.0), "completion-validate": (last, 600.0)},
    )

    release = compare_after(_evidence(baseline, after), "after")["store_release"]

    assert release["span_seconds"] == 10 * 3600
    assert release["milestone_388"]["verdict"] == "missed"
    assert release["milestone_388"]["over_by_seconds"] == 10 * 3600 - EVENING_SECONDS
    assert release["steps_summed_seconds"] == 30600.0


def test_an_after_label_that_names_no_store_is_refused(tmp_path):
    baseline = _store(tmp_path, "OGS-00009", {"build": ("", 70000.0)})
    with pytest.raises(SystemExit, match="names no --store"):
        compare_after(_evidence(baseline), "after")


def test_every_row_is_measured_or_summed_and_every_projection_names_a_scenario():
    keys = [row.key for row in ROWS]
    assert len(keys) == len(set(keys))
    for row in ROWS:
        assert (row.after is None) == bool(row.parts), row.key
        for part in row.parts:
            assert keys.index(part) < keys.index(row.key), (row.key, part)
        assert row.projections, row.key


def _pass2_log(day: str, minutes: int) -> str:
    return "\n".join(
        [
            _line(f"{day} 01:00:00", "Pass 2: resolving 2514 analyses × 9847698 variants"),
            _line(
                f"{day} 01:{minutes:02d}:00",
                "Pass 2: 2514/2514 done (last: x) — elapsed 1s, ETA 0s",
            ),
        ]
    )


def _with_argv(store: dict, step: str, argv: list[str]) -> dict:
    record = next(r for r in store["records"] if r.step == step)
    store["records"] = [replace(r, argv=argv) if r is record else r for r in store["records"]]
    return store


def test_a_missed_pass2_says_why_only_while_its_reason_holds(tmp_path):
    """Pass 2's reason is that GWAS-VCF reads per row: printed for a GWAS-VCF build,
    and `reason no longer holds` for a build whose argv names another reader."""
    baseline = _store(tmp_path, "OGS-00009", {"build": ("", 70000.0)})
    vcf = _with_argv(
        _store(tmp_path, "vcf", {"build": (_pass2_log("2026-11-13", 40), 9000.0)}),
        "build",
        ["opengwasdb", "build-dense-vcf", "--source-reader-capability", "opengwasdb.gwas-vcf"],
    )
    finngen = _with_argv(
        _store(tmp_path, "finngen", {"build": (_pass2_log("2026-11-13", 40), 9000.0)}),
        "build",
        ["opengwasdb", "build-dense-vcf", "--source-reader-capability", "opengwasdb.finngen-r13"],
    )

    held = _row(compare_after(_evidence(baseline, vcf), "vcf"), "pass2")["projections"][0]
    moved = _row(compare_after(_evidence(baseline, finngen), "finngen"), "pass2")["projections"][0]

    assert held["verdict"] == moved["verdict"] == "missed"
    assert held["reasons"] and "GWAS-VCF" in held["reasons"][0]
    assert moved["reasons"] == ["reason no longer holds"]


def _completion_log(block_hours: int) -> str:
    return "\n".join(
        [
            _line("2026-11-13 02:00:00", "LD-block completion: start"),
            _line(f"2026-11-13 {2 + block_hours:02d}:00:00", "LD-block completion: done in 1s"),
        ]
    )


@pytest.mark.parametrize(("block_hours", "holds"), [(7, True), (1, False)])
def test_the_evening_miss_names_the_block_phase_only_when_it_explains_the_overrun(
    tmp_path, block_hours, holds
):
    baseline = _store(tmp_path, "OGS-00009", {"build": ("", 70000.0)})
    first = _line("2026-11-13 00:00:00", "Pass 1: collecting source variants from 2514 files")
    last = _line("2026-11-13 12:00:00", "Validation: done in 10m00s")
    after = _store(
        tmp_path,
        "after",
        {
            "build": (first, 7200.0),
            "complete": (_completion_log(block_hours), 30000.0),
            "completion-validate": (last, 600.0),
        },
    )

    line = compare_after(_evidence(baseline, after), "after")["store_release"]["milestone_388"]

    assert line["verdict"] == "missed" and line["over_by_seconds"] == 3 * 3600
    if holds:
        assert line["reasons"] and "LD-block" in line["reasons"][0]
    else:
        assert line["reasons"] == ["reason no longer holds"]


def test_no_reason_text_carries_a_measured_number():
    from benchmarks._epic307_after import EVENING_MISS_REASONS

    texts = [text for row in ROWS for text, _ in (*row.miss_reasons, *row.notes)]
    texts += [text for text, _ in EVENING_MISS_REASONS]
    assert texts
    for text in texts:
        assert not re.search(r"\d+h\d|\d+m\d|\d+\.\d+\s*s|\d+\s*(?:min|seconds?|h)\b", text), text
