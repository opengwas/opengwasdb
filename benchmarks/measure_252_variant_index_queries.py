#!/usr/bin/env python3
"""Cold vs warm off-axis PheWAS, and the scattered exception-table probe (#252).

Three measurements on one indexed store, for the round-2 review:

* **open** -- `query_store` time and the reader's own cost, with the process's
  `maxrss`;
* **cold** -- a fresh process's first off-axis PheWAS, from process start
  (open included): the number the ADR must quote, since the query optimises
  nothing before it;
* **warm** -- the same query after `ByVariantReader.warm()`, p50 and p90 over
  `--reps`;
* **scattered** -- `WindowedExactTable.lookup` on 20 exception cells chosen
  across the table, the shape the scan route and `lookup` hits take.  Reports
  the wall time and the process-RSS delta, so a window that is not bounded by
  the positions (round-2 finding 3) is visible.

    pixi run -e dev python benchmarks/measure_252_variant_index_queries.py \
        --store /data/opengwasdb/work/epic252/OGS-00011-0.2.0 \
        --output docs/benchmark-output/opengwasdb_252_variant_index_queries.json
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np

from benchmarks._artifact import provenance, write_artifact
from benchmarks._query_shapes import probe_variant_alid
from benchmarks._rss import RssSampler, rss_mb
from opengwasdb.query import query_store

#: Taken *after* the imports, so the name says what it measures: the wall from
#: this point (imports excluded, about 0.9 s of them) through the cold answer.
_PROCESS_START = time.perf_counter()


def _maxrss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def measure(store: Path, reps: int, scattered_cells: int) -> dict:
    started = time.perf_counter()
    query = query_store(store)
    open_s = time.perf_counter() - started
    out: dict = {"store": str(store), "open_s": round(open_s, 3), "reps": reps}
    try:
        alid = probe_variant_alid(query, off_panel=True)
        if alid is None:
            raise SystemExit(f"{store}: no off-axis variant to query")
        reader = getattr(query, "_by_variant", None)
        if reader is None:
            raise SystemExit(f"{store}: no variant index")

        # Cold: the first call loads the codec's windowed tables.
        cold_start = time.perf_counter()
        cold = query.phewas(alid)
        out["cold_query_s"] = round(time.perf_counter() - cold_start, 4)
        out["cold_rows"] = int(len(cold["z"]))
        # Cold **including open**: from after the imports through the first
        # answer.  `cold_sampled_rss_mb` is this process's own RSS (statm);
        # `maxrss` is reported separately because under `pixi run` it starts at
        # ~2 GiB inherited from the launcher (review round 3, finding 2).
        out["cold_after_imports_wall_s"] = round(time.perf_counter() - _PROCESS_START, 3)
        out["cold_sampled_rss_mb"] = round(rss_mb(), 1)

        reader.warm()
        warm: list[float] = []
        for _ in range(reps):
            first = time.perf_counter()
            query.phewas(alid)
            warm.append(time.perf_counter() - first)
        out["warm_p50_s"] = round(_percentile(warm, 50), 4)
        out["warm_p90_s"] = round(_percentile(warm, 90), 4)

        table = reader._codec.eaf_exceptions
        index = np.asarray(table.index_zarr[:], dtype=np.int64)
        if len(index) == 0:
            out["scattered"] = {"skipped": "the store has no EAF exception table"}
        else:
            ordinals = np.linspace(0, len(index) - 1, scattered_cells).astype(np.int64)
            positions = index[ordinals].copy()
            baseline = rss_mb()
            with RssSampler() as sampler:
                scattered_start = time.perf_counter()
                table.lookup(positions)
                scattered_s = time.perf_counter() - scattered_start
            peak = max(sampler.peak_mb, rss_mb())
            out["scattered"] = {
                "cells": len(positions),
                "table_entries": int(len(index)),
                "seconds": round(scattered_s, 4),
                "delta_mb": round(peak - baseline, 1),
                "process_peak_mb": round(peak, 1),
            }
    finally:
        query.close()
    out["maxrss_gib"] = round(_maxrss_gib(), 3)
    return out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=40, help="warm repetitions")
    parser.add_argument("--scattered-cells", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = {
        "harness": "benchmarks/measure_252_variant_index_queries.py",
        **provenance(),
        **measure(args.store, args.reps, args.scattered_cells),
    }
    write_artifact(args.output, artifact)
    return 0


if __name__ == "__main__":
    sys.exit(main())
