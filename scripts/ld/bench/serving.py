#!/usr/bin/env python3
"""What serving LD costs a process that holds the HGDP+1kGP stores (ogld#9).

Two measurements, each written as JSON lines to `--out`, and the tables of every record:

- `memory`: one process opens the five stores, as a worker that serves them would, and records
  its resident memory (`VmRSS` and its peak `VmHWM`, from /proc/self/status) after the opens,
  after loading every chromosome's positions and `indptr` arrays (Near Band and Far Pairs: what
  a long-lived reader keeps, ld-store §10), and after filling each store's chunk
  cache with 3 Mb regions until it holds 95% of its budget. Each record carries every store's
  cache count beside the process's memory. A store that runs out of regions first stops the run.
- `lag`: the main thread runs an asyncio event loop, as a web service does, and ticks every
  5 ms while an ogld query runs on one worker thread, as ogdb-api runs a store query
  (`asyncio.to_thread`). It records how late each tick woke: how long a service's own requests
  would wait while ogld reads. The queries are ogld#8's own (`latency.clump_query`,
  `matrix_query`, `region_query`), so each record carries what its query answered, and runs under
  different zarr settings can be held to the same answer.
- `tables`: renders records as `docs/benchmarks/serving-tables.md`: ogld#8's step-1 tables of the
  `bench-latency` runs (`report.step1_tables`), then the memory and lag runs, one row a record.

    pixi run -e dev bench-serving memory --out memory.jsonl
    pixi run -e dev bench-serving lag --out lag.jsonl
    pixi run -e dev bench-serving lag --engine-zarr --out lag.jsonl
    pixi run -e dev bench-serving tables docs/benchmarks/results/9/*.jsonl.gz \
        > docs/benchmarks/serving-tables.md

`bench-serving` keeps BLAS to one thread, as `bench-latency` does. zarr's pool is
`--zarr-threads` (6), and `lag` adds one worker thread. The stores are only read.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import statistics
import sys
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Final, TextIO

# The read benchmark beside this script: where the stores and Analyses are, its queries, and its
# tables (scripts/bench/latency.py and report.py).
import latency
import report

#: Each fill read's radius, so a 3 Mb region; the regions' spacing along a chromosome; and the
#: share of a cache's budget that counts as full.
FILL_RADIUS: Final = 1_500_000
FILL_STEP: Final = 20_000_000
FULL: Final = 0.95
#: The clump `lag` times: one of ogld#8's four Analyses, at r² 0.01 within 3 Mb.
LAG_ANALYSIS: Final = "OGS-00011__GCST90319613"
TICK: Final = 0.005
GIB: Final = 1 << 30


def parse_status(text: str) -> dict[str, int]:
    """`VmRSS` and `VmHWM` of a /proc/<pid>/status text, in bytes (the kernel gives kB)."""
    fields = {}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key in ("VmRSS", "VmHWM"):
            fields[key] = int(value.split()[0]) * 1024
    if len(fields) != 2:
        raise ValueError(f"the status text names {sorted(fields)}, not VmRSS and VmHWM")
    return fields


def status() -> dict[str, int]:
    """This process's resident memory and its peak, in bytes, as the kernel counts them."""
    return parse_status(Path("/proc/self/status").read_text())


def write(out: TextIO, record: dict[str, Any]) -> None:
    out.write(json.dumps(record) + "\n")
    out.flush()
    print(json.dumps(record), file=sys.stderr)


def held(store: Any) -> int:
    """The bytes an opened store's chunk cache holds now. ogld exposes the budget
    (`StoreReader.cache_bytes`) but not the count, so this reads it from outside, as ogld#8's
    benchmark reads the reader's internals (`latency.instrument`)."""
    return int(store._reader._cache._bytes)


def memory_record(step: str, stores: dict[str, Any], **extra: Any) -> dict[str, Any]:
    gc.collect()
    return {
        "measurement": "memory",
        "step": step,
        **status(),
        "cache_bytes": {panel: held(store) for panel, store in stores.items()},
        "threads": threading.active_count(),
        "load": list(os.getloadavg()),
        **extra,
    }


def load_chromosomes(store: Any) -> int:
    """Load every chromosome's kept arrays, the Far Pairs' `indptr` too, through the reader's
    own loaders (ld-store §10), so the count is what a reader that has met every chromosome keeps.
    No public read loads a chromosome without reading pairs too. Returns the chromosomes."""
    reader = store._reader
    for name in reader._ranges:
        reader._far_pairs(reader._chromosome(name))
    return len(reader._ranges)


def centres(store: Any, radius: int = FILL_RADIUS, step: int = FILL_STEP) -> list[tuple[str, int]]:
    """Regions of `radius`, `step` apart along each chromosome in turn, each inside it."""
    reader = store._reader
    found = []
    for name in reader._ranges:
        position = reader._chromosome(name).position
        first, last = int(position[0]) + radius, int(position[-1]) - radius
        found += [(name, c) for c in range(first, last, step)]
    return found


def fill(store: Any, places: Iterable[tuple[str, int]], radius: int = FILL_RADIUS) -> int:
    """Read regions at `places` until the store's cache holds `FULL` of its budget; returns how
    many. Running out of places first is refused: a record of a cache short of full would read
    as the memory of a full one."""
    budget = store._reader.cache_bytes
    remaining = iter(places)
    read = 0
    while held(store) < FULL * budget:
        place = next(remaining, None)
        if place is None:
            raise SystemExit(
                f"ran out of regions after {read}, with the cache at {held(store):,} of "
                f"{budget:,} bytes, short of {FULL:.0%}"
            )
        store.region(place[0], place[1], radius)
        read += 1
    return read


def memory(args: argparse.Namespace, out: TextIO) -> None:
    import opengwasdb.ld

    stores: dict[str, Any] = {}
    write(out, memory_record("imported", stores))
    for panel in args.panels:
        started = time.perf_counter()
        stores[panel] = opengwasdb.ld.open(latency.recipe.store_path(panel))
        write(out, memory_record("opened", stores, panel=panel, seconds=elapsed(started)))
    for panel, store in stores.items():
        started = time.perf_counter()
        n = load_chromosomes(store)
        extra = {"panel": panel, "chromosomes": n, "seconds": elapsed(started)}
        write(out, memory_record("chromosomes", stores, **extra))
    for panel, store in stores.items():
        started = time.perf_counter()
        n = fill(store, centres(store))
        write(
            out, memory_record("filled", stores, panel=panel, regions=n, seconds=elapsed(started))
        )


def elapsed(started: float) -> float:
    return time.perf_counter() - started


# ------------------------------------------------------------------ the event loop's lag


async def lags_while(going: Callable[[], bool]) -> list[float]:
    """How late each `TICK` sleep on the running event loop woke, for as long as `going()`."""
    lags = []
    while going():
        before = time.perf_counter()
        await asyncio.sleep(TICK)
        lags.append(time.perf_counter() - before - TICK)
    return lags


def for_seconds(seconds: float) -> Callable[[], bool]:
    stop = time.perf_counter() + seconds
    return lambda: time.perf_counter() < stop


def summary(lags: list[float]) -> dict[str, float]:
    """The lags' median, 99th percentile (the nearest rank at or below it), largest, and the
    ticks more than 100 ms late, in ms. No tick is refused: it would read as no lag."""
    if not lags:
        raise ValueError("no tick was measured")
    ordered = sorted(lags)
    return {
        "ticks": len(ordered),
        "lag_p50_ms": 1000 * statistics.median(ordered),
        "lag_p99_ms": 1000 * ordered[int(0.99 * (len(ordered) - 1))],
        "lag_max_ms": 1000 * ordered[-1],
        "late_over_100ms": sum(lag > 0.1 for lag in ordered),
    }


async def lag_of(query: Callable[[], Any]) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    started = time.perf_counter()
    done = loop.run_in_executor(None, query)
    lags = await lags_while(lambda: not done.done())
    seconds = elapsed(started)
    return {"query_seconds": seconds, **summary(lags), "result": done.result()}


def lag_queries() -> dict[str, latency.Query]:
    """Each query `lag` times, ogld#8's own, its inputs chosen first on a reader of their own
    (ogld#8's `Sample`), so choosing them warms nothing a timing reads."""
    alids, z = latency.hits(latency.analyses([LAG_ANALYSIS])[0], None)
    sample = latency.Sample(latency.recipe.store_path("EUR"))
    scattered = sample.alids(sample.rows(500, None, latency.SEED))
    return {
        "clump": latency.clump_query(alids, z, 0.01),
        "matrix_500_genome": latency.matrix_query(scattered),
        "region_3mb_chr22": latency.region_query("22", 32_000_000, FILL_RADIUS),
    }


def lag(args: argparse.Namespace, out: TextIO) -> None:
    """Each query, cold: the store's pages evicted (ogld#8's `evict`) and the store opened
    inside the timing, on the worker thread, as a job opens it."""
    import opengwasdb.ld

    idle = asyncio.run(lags_while(for_seconds(5.0)))
    write(out, {"measurement": "lag", "query": "idle", **summary(idle), **context(args)})
    queries = lag_queries()
    path = latency.recipe.store_path("EUR")
    for repeat in range(args.repeats):
        for name in args.queries:
            latency.evict(path)
            record = asyncio.run(lag_of(lambda name=name: queries[name](opengwasdb.ld.open(path))))
            write(
                out,
                {"measurement": "lag", "query": name, "repeat": repeat, **record, **context(args)},
            )


def context(args: argparse.Namespace) -> dict[str, Any]:
    import zarr

    return {
        "engine_zarr": args.engine_zarr,
        "codec_pipeline": zarr.config.get("codec_pipeline"),
        "load": list(os.getloadavg()),
    }


def engine_zarr() -> None:
    """The process-wide zarr settings the engine's array seam makes on import (opengwasdb ADR
    0056, `opengwasdb/store/arrays.py`), which an ogdb-api process that imports both packages
    would read ogld's stores under. Blosc's threads are left alone: ogld's arrays are zstd."""
    import zarr

    zarr.config.set(
        {
            "array.write_empty_chunks": True,
            "codec_pipeline.path": "zarr.core.codec_pipeline.FusedCodecPipeline",
            "codec_pipeline.max_workers": 1,
        }
    )


# ------------------------------------------------------------------ the tables


#: One run of a records file: the file, its environment record, and the records after it.
Run = tuple[Path, dict[str, Any], list[dict[str, Any]]]


def runs(paths: Iterable[Path]) -> list[Run]:
    """Each run of each file: its environment record and the records it wrote after it."""
    found: list[Run] = []
    for path in paths:
        for line in report.text_of(path).splitlines():
            record = json.loads(line)
            if record.get("kind") == "environment":
                found.append((path, record, []))
            else:
                found[-1][2].append(record)
    return found


def hhmm(environment: dict[str, Any]) -> str:
    return str(environment["started"])[11:16]


def memory_rows(found: list[Run]) -> list[list[str]]:
    rows = []
    for _, environment, records in found:
        for r in records:
            done = r.get("chromosomes", r.get("regions", "—"))
            rows.append(
                [
                    hhmm(environment),
                    r["step"],
                    r.get("panel", "—"),
                    f"{r['VmRSS'] / GIB:.2f}",
                    f"{r['VmHWM'] / GIB:.2f}",
                    f"{sum(r['cache_bytes'].values()) / GIB:.2f}",
                    f"{done}",
                    f"{r['seconds']:.1f}" if "seconds" in r else "—",
                    f"{r['load'][0]:.1f}",
                ]
            )
    return rows


def lag_rows(found: list[Run]) -> list[list[str]]:
    rows = []
    for _, environment, records in found:
        for r in records:
            answer = r.get("result") or {}
            rows.append(
                [
                    hhmm(environment),
                    "engine's" if r["engine_zarr"] else "default",
                    r["query"],
                    f"{r['query_seconds']:.2f}" if "query_seconds" in r else "—",
                    f"{r['lag_p50_ms']:.2f}",
                    f"{r['lag_p99_ms']:.1f}",
                    f"{r['lag_max_ms']:.1f}",
                    str(r["late_over_100ms"]),
                    f"{r['ticks']:,}",
                    ", ".join(f"{k} {v:,}" for k, v in answer.items()) or "—",
                    f"{r['load'][0]:.1f}",
                ]
            )
    return rows


MEMORY_HEADER: Final = [
    "run", "step", "store", "resident, GiB", "peak, GiB", "caches counted, GiB",
    "chromosomes or regions", "seconds", "load",
]  # fmt: skip
LAG_HEADER: Final = [
    "run", "zarr settings", "query", "query, s", "lag p50, ms", "lag p99, ms", "lag max, ms",
    "ticks over 100 ms", "ticks", "answer", "load",
]  # fmt: skip


def measured(found: list[Run], kind: str) -> list[Run]:
    """The runs of one `bench-serving` measurement."""
    return [run for run in found if run[1].get("measurement") == kind]


def latency_tables(found: list[Run]) -> list[str]:
    """The `bench-latency` runs among `found`, through ogld#8's step-1 tables."""
    paths = sorted({path for path, environment, _ in found if "measurement" not in environment})
    timings, environments = report.read(paths)
    lines = ["## The clump and read runs (`bench-latency step1`)", ""]
    lines += [f"Runs: {len(environments)}; records: {len(timings)}.", ""]
    return lines + report.step1_tables(timings)


def tables(paths: list[Path]) -> str:
    """The records of `paths` as `serving-tables.md`: `bench-latency`'s runs through ogld#8's
    step-1 tables, and `bench-serving`'s, one row a record, in the order they were written."""
    found = runs(paths)
    lines = ["# What serving LD costs: tables (ogld#9)", ""]
    lines += [
        "Generated by `scripts/bench/serving.py tables` from `docs/benchmarks/results/9/`.",
        "",
    ]
    lines += [f"Runs: {len(found)}; records: {sum(len(run[2]) for run in found)}.", ""]
    lines += latency_tables(found)
    lines += ["## One process holding the five stores (`bench-serving memory`)", ""]
    lines += [report.table(MEMORY_HEADER, memory_rows(measured(found, "memory"))), ""]
    lines += ["## Sharing a process (`bench-serving lag`)", ""]
    lines += [report.table(LAG_HEADER, lag_rows(measured(found, "lag"))), ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="measurement", required=True)
    for name in ("memory", "lag"):
        each = sub.add_parser(name)
        each.add_argument("--out", type=Path, required=True)
        each.add_argument("--zarr-threads", type=int, default=6)
    sub.choices["memory"].add_argument("--panels", nargs="+", default=list(latency.recipe.PANELS))
    timed = sub.choices["lag"]
    timed.add_argument("--queries", nargs="+", default=list(LAG_QUERIES))
    timed.add_argument("--repeats", type=int, default=1, help="rounds of the queries")
    timed.add_argument("--engine-zarr", action="store_true", help="the engine's zarr settings")
    sub.add_parser("tables").add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()
    if args.measurement == "tables":
        print(tables(args.results))
        return
    measure(args)


LAG_QUERIES: Final = ("clump", "matrix_500_genome", "region_3mb_chr22")


def measure(args: argparse.Namespace) -> None:
    import zarr

    zarr.config.set({"threading.max_workers": args.zarr_threads})
    if getattr(args, "engine_zarr", False):
        engine_zarr()
    with args.out.open("a") as out:
        write(out, {**latency.environment(args), "measurement": args.measurement})
        (memory if args.measurement == "memory" else lag)(args, out)


if __name__ == "__main__":
    main()
