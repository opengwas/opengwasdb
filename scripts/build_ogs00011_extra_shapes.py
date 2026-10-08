#!/usr/bin/env python3
"""Aggregate #250's two #252 Hybrid extra shapes from the raw one-shape runs.

`benchmarks/ogs00011_extra_shapes.py` runs #252's probe
(`benchmarks.ogs00011_ab.measure_one_shape`) and appends one JSON record per
repetition, carrying the environment it ran in. This collapses the committed
`opengwasdb_ogs00011_extra_shapes.jsonl` into a per-column
median/p95/peak-RSS/count/digest artifact, with each column's environment and
each run's start, wait and per-repetition loads. It refuses to publish if the
three columns do not return the same rows and sha256, or if one column label
covers runs from different environments.

Run from the repository root:

    python3 scripts/build_ogs00011_extra_shapes.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT = Path("docs/benchmark-output")
RAW = "opengwasdb_ogs00011_extra_shapes.jsonl"
AGG = "opengwasdb_ogs00011_extra_shapes_0_2_0.json"
COLUMN_ORDER = ["a-2.18-0.1.0", "b-code-0.1.0", "c-code-0.2.0"]
SHAPE_ORDER = ["phewas_off_axis", "bulk_overflow_heavy"]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def _group(raw: Path) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for line in raw.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            grouped.setdefault((record["column"], record["shape"]), []).append(record)
    return grouped


def _row(records: list[dict]) -> dict:
    if any(record["timed_out"] for record in records):
        raise SystemExit("a one-shape run hit the time limit; refusing to publish")
    elapsed = [float(record["elapsed_ms"]) for record in records]
    return {
        "repetitions": len(records),
        "median_ms": round(sorted(elapsed)[len(elapsed) // 2], 3),
        "p95_ms": round(_percentile(elapsed, 0.95), 3),
        "peak_mib": round(max(float(record["peak_mb"]) for record in records), 1),
        "result_count": records[0]["result_count"],
        "sha256": records[0]["sha256"],
        # In repetition order, so the report can say which repetition supplied the
        # median and which repetition started at a load of 3 or more (#250 r2, 3b).
        "elapsed_ms": [round(float(record["elapsed_ms"]), 3) for record in records],
        "start_load_1m": [round(float(record["load_start"][0]), 2) for record in records],
        "run": _run(records),
    }


def _shape_rows(grouped: dict[tuple[str, str], list[dict]], shape: str) -> dict[str, dict]:
    rows = {}
    for column in COLUMN_ORDER:
        records = grouped.get((column, shape))
        if not records:
            raise SystemExit(f"{RAW}: no run for {column} / {shape}")
        rows[column] = _row(records)
    counts = {row["result_count"] for row in rows.values()}
    digests = {row["sha256"] for row in rows.values()}
    if len(counts) != 1 or len(digests) != 1:
        raise SystemExit(f"{RAW}: the columns for {shape} do not agree; refusing to publish")
    return rows


#: What names a column's code and interpreter; every run under one label must agree.
ENVIRONMENT_FIELDS = (
    "python", "zarr", "hostname", "opengwasdb_path", "opengwasdb_fingerprint",
    "commit", "probe_path", "probe_sha256", "runner_path", "runner_sha256",
)
#: What one runner invocation records once, for all of its repetitions.
RUN_FIELDS = ("measured_at", "waited_for_load_s", "waited_timed_out")


def _environment(record: dict) -> dict:
    """One record's environment, or `recorded: false` if it predates the fields."""
    if not all(field in record for field in ENVIRONMENT_FIELDS):
        return {"recorded": False}
    return {"recorded": True} | {field: record[field] for field in ENVIRONMENT_FIELDS}


def _environments(grouped: dict[tuple[str, str], list[dict]]) -> dict[str, dict]:
    """The one environment each column ran under, across every shape it ran.

    A column whose records predate the fields is reported as `recorded: false`
    rather than silently omitted (#250 review r2, major 3). A column whose runs
    disagree on any field is refused: the label would otherwise name code it
    did not all run.
    """
    out: dict[str, dict] = {}
    for (column, _shape), records in grouped.items():
        for record in records:
            environment = _environment(record)
            known = out.setdefault(column, environment)
            differing = sorted(key for key in known.keys() | environment.keys()
                               if known.get(key) != environment.get(key))
            if differing:
                raise SystemExit(f"{RAW}: column {column} ran under different {differing}")
    return out


def _run(records: list[dict]) -> dict:
    """When one runner invocation started, how long it waited for quiet, and if it gave up."""
    if not all(field in record for record in records for field in RUN_FIELDS):
        return {"recorded": False}
    runs = {tuple(record[field] for field in RUN_FIELDS) for record in records}
    if len(runs) != 1:
        raise SystemExit(f"{RAW}: one column and shape holds {len(runs)} runs; keep one")
    return {"recorded": True} | {field: records[0][field] for field in RUN_FIELDS}


def _aggregate(grouped: dict[tuple[str, str], list[dict]]) -> dict:
    artifact: dict = {
        "task": "#250",
        "runner": "benchmarks/ogs00011_extra_shapes.py",
        "probe": "benchmarks.ogs00011_ab.measure_one_shape",
        # The runner calls the probe once per repetition in one interpreter per
        # column and shape, so repetition 1 is the only cold-process one.
        "repetitions_share_one_interpreter": True,
        "raw": RAW,
        "environments": _environments(grouped),
        "shapes": {shape: _shape_rows(grouped, shape) for shape in SHAPE_ORDER},
    }
    artifact["identity"] = {
        shape: {
            "identical": True,
            "counts": {column: row["result_count"] for column, row in rows.items()},
            "digests": {column: row["sha256"] for column, row in rows.items()},
        }
        for shape, rows in artifact["shapes"].items()
    }
    return artifact


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=OUT / RAW)
    ap.add_argument("--output", type=Path, default=OUT / AGG)
    args = ap.parse_args()

    artifact = _aggregate(_group(args.raw))
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    for shape, rows in artifact["shapes"].items():
        print(shape, {column: (row["median_ms"], row["peak_mib"]) for column, row in rows.items()})


if __name__ == "__main__":
    main()
