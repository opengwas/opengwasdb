#!/usr/bin/env python3
"""Compare physical-SE query latency before and after format-3 residual coding."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from opengwasdb.query import query_store


def _bytes(path: Path) -> int:
    result = subprocess.run(["du", "-sb", str(path)], check=True, capture_output=True, text=True)
    return int(result.stdout.split()[0])


def _timing(call: Callable[[], dict[str, np.ndarray]], repetitions: int) -> dict[str, object]:
    call()
    samples = []
    count = 0
    for _ in range(repetitions):
        started = time.perf_counter()
        result = call()
        samples.append((time.perf_counter() - started) * 1000)
        count = len(result["se"])
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "samples_ms": samples,
        "result_count": count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with query_store(args.before) as query:
        analysis_id = str(query.analyses_table()[0]["analysis_id"])
        hits = query.top_hits(analysis_id=analysis_id, threshold=5e-8, limit=1)
        variant_index = int(hits["variant_index"][0])
        variant = query._variant_axis.by_index(variant_index)
        assert variant is not None
        alid = variant.alid
        region = (variant.chromosome, max(0, variant.position - 50_000), variant.position + 50_000)

    stores = []
    for label, path in (("float16", args.before), ("int8_residual", args.after)):
        with query_store(path) as query:
            calls = {
                "analysis": lambda q=query: q.analysis(analysis_id),
                "phewas": lambda q=query: q.phewas(alid),
                "lookup": lambda q=query: q.lookup([alid], [analysis_id]),
                "range": lambda q=query: q.range_phewas(*region),
                "top_hits": lambda q=query: q.top_hits(
                    analysis_id=analysis_id, threshold=5e-8
                ),
            }
            stores.append(
                {
                    "label": label,
                    "path": str(path),
                    "se_bytes": sum(
                        _bytes(path / "data.zarr" / name)
                        for name in (
                            "se",
                            "se_coefficients",
                            "se_exception_index",
                            "se_exception_value",
                        )
                        if (path / "data.zarr" / name).exists()
                    ),
                    "queries": {
                        name: _timing(call, args.repetitions) for name, call in calls.items()
                    },
                }
            )
    payload = {
        "repetitions": args.repetitions,
        "selection": {"analysis_id": analysis_id, "alid": alid, "region": region},
        "stores": stores,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
