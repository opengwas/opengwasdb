#!/usr/bin/env python3
"""What serving LD costs a process that holds the HGDP+1kGP stores (ogld#9).

Two measurements, each written as JSON lines to `--out`:

- `memory`: one process opens the five stores, as a worker that serves them would, and records
  its resident memory (`VmRSS` and its peak `VmHWM`, from /proc/self/status) after the opens,
  after loading every chromosome's positions and `indptr` arrays (Near Band, Proxy Index and
  Far Pairs: what a long-lived reader keeps, spec §10), and after filling each store's chunk
  cache with 3 Mb regions until it holds 95% of its budget. Each record carries every store's
  cache count beside the process's memory.
- `lag`: the main thread runs an asyncio event loop, as a web service does, and ticks every
  5 ms while an ogld query runs on one worker thread, as ogdb-api runs a store query
  (`asyncio.to_thread`). It records how late each tick woke: how long a service's own requests
  would wait while ogld reads.

    pixi run -e dev python scripts/bench/serving.py memory --out memory.jsonl
    pixi run -e dev python scripts/bench/serving.py lag --out lag.jsonl

Threads: zarr's pool is `--zarr-threads` (6), BLAS 1 when run as above with
`OPENBLAS_NUM_THREADS=1`, and `lag` adds one worker thread. The stores are only read.
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
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final, TextIO

# The read benchmark's helpers, beside this script: where the stores and Analyses are come from
# their owners through it (scripts/bench/latency.py).
import latency

#: The region width that fills a cache, and the share of the budget that counts as full.
FILL_RADIUS: Final = 1_500_000
FULL: Final = 0.95
#: The clump `lag` times: one of ogld#8's four Analyses, at r² 0.01 within 3 Mb.
LAG_ANALYSIS: Final = "OGS-00011__GCST90319613"
TICK: Final = 0.005


def status() -> dict[str, int]:
    """This process's resident memory and its peak, in bytes, as the kernel counts them."""
    fields = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("VmRSS", "VmHWM"):
            fields[key] = int(value.split()[0]) * 1024
    return fields


def write(out: TextIO, record: dict[str, Any]) -> None:
    out.write(json.dumps(record) + "\n")
    out.flush()
    print(json.dumps(record), file=sys.stderr)


def cached(stores: dict[str, Any]) -> dict[str, int]:
    """Each store's chunk-cache count, as the cache bounds it (spec §10)."""
    return {panel: int(store._reader._cache._bytes) for panel, store in stores.items()}


def memory_record(step: str, stores: dict[str, Any], **extra: Any) -> dict[str, Any]:
    gc.collect()
    return {
        "measurement": "memory",
        "step": step,
        **status(),
        "cache_bytes": cached(stores),
        "threads": threading.active_count(),
        "load": list(os.getloadavg()),
        **extra,
    }


def load_chromosomes(store: Any) -> int:
    """Load every chromosome's kept arrays, the Far Pairs' `indptr` too, through the reader's
    own loaders (spec §10), so the count is what a reader that has met every chromosome keeps.
    Returns the chromosomes loaded."""
    reader = store._reader
    for name in reader._ranges:
        reader._far_pairs(reader._chromosome(name))
    return len(reader._ranges)


def centres(store: Any) -> list[tuple[str, int]]:
    """3 Mb regions, 20 Mb apart along each chromosome in turn, each inside its chromosome."""
    reader = store._reader
    found = []
    for name in reader._ranges:
        position = reader._chromosome(name).position
        first, last = int(position[0]) + FILL_RADIUS, int(position[-1]) - FILL_RADIUS
        found += [(name, c) for c in range(first, last, 20_000_000)]
    return found


def fill(store: Any) -> int:
    """Read 3 Mb regions until the store's cache holds `FULL` of its budget; returns how many."""
    cache = store._reader._cache
    read = 0
    for chrom, centre in centres(store):
        if cache._bytes >= FULL * cache.budget:
            break
        store.region(chrom, centre, FILL_RADIUS)
        read += 1
    return read


def memory(args: argparse.Namespace, out: TextIO) -> None:
    import ogld

    stores: dict[str, Any] = {}
    write(out, memory_record("imported", stores))
    for panel in args.panels:
        started = time.perf_counter()
        stores[panel] = ogld.open(latency.recipe.store_path(panel))
        write(out, memory_record("opened", stores, panel=panel, seconds=elapsed(started)))
    for panel, store in stores.items():
        started = time.perf_counter()
        n = load_chromosomes(store)
        write(
            out,
            memory_record(
                "chromosomes", stores, panel=panel, chromosomes=n, seconds=elapsed(started)
            ),
        )
    for panel, store in stores.items():
        started = time.perf_counter()
        n = fill(store)
        write(
            out, memory_record("filled", stores, panel=panel, regions=n, seconds=elapsed(started))
        )


def elapsed(started: float) -> float:
    return time.perf_counter() - started


# ------------------------------------------------------------------ the event loop's lag


async def ticking(done: asyncio.Future[Any] | None, seconds: float) -> list[float]:
    """How late each `TICK` sleep woke, until `done` resolves, or for `seconds` when None."""
    lags = []
    stop = time.perf_counter() + seconds
    while (done is None and time.perf_counter() < stop) or (done is not None and not done.done()):
        before = time.perf_counter()
        await asyncio.sleep(TICK)
        lags.append(time.perf_counter() - before - TICK)
    return lags


def summary(lags: list[float]) -> dict[str, float]:
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
    lags = await ticking(done, 0)
    seconds = elapsed(started)
    return {"query_seconds": seconds, **summary(lags), "result": done.result()}


def lag_queries() -> dict[str, Callable[[Any], dict[str, int]]]:
    """Each query `lag` times, its inputs chosen first, on a reader of their own (ogld#8's
    `Sample`), so choosing them warms nothing a timing reads."""
    alids, z = latency.hits(latency.analyses([LAG_ANALYSIS])[0], None)
    sample = latency.Sample(latency.recipe.store_path("EUR"))
    scattered = sample.alids(sample.rows(500, None, latency.SEED))
    return {
        "clump": lambda store: {
            "leads": len(store.clump(alids, z, r2=0.01, kb=3000).leads()),
            "hits": len(alids),
        },
        "matrix_500_genome": lambda store: {"variants": len(store.matrix(scattered).variants.alid)},
        "region_3mb_chr22": lambda store: {
            "variants": len(store.region("22", 32_000_000, FILL_RADIUS).variants.alid)
        },
    }


def lag(args: argparse.Namespace, out: TextIO) -> None:
    """Each query, cold: the store's pages evicted (ogld#8's `evict`) and the store opened
    inside the timing, on the worker thread, as a job opens it."""
    import ogld

    idle = asyncio.run(ticking(None, 5.0))
    write(out, {"measurement": "lag", "query": "idle", **summary(idle), **context(args)})
    queries = lag_queries()
    path = latency.recipe.store_path("EUR")
    for repeat in range(args.repeats):
        for name in args.queries:
            latency.evict(path)
            record = asyncio.run(lag_of(lambda name=name: queries[name](ogld.open(path))))
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("measurement", choices=("memory", "lag"))
    parser.add_argument("--panels", nargs="+", default=list(latency.recipe.PANELS))
    parser.add_argument(
        "--queries", nargs="+", default=["clump", "matrix_500_genome", "region_3mb_chr22"]
    )
    parser.add_argument("--zarr-threads", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=1, help="`lag`: rounds of the queries")
    parser.add_argument(
        "--engine-zarr", action="store_true", help="read under the engine's zarr settings"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import zarr

    zarr.config.set({"threading.max_workers": args.zarr_threads})
    if args.engine_zarr:
        engine_zarr()
    with args.out.open("a") as out:
        write(out, {**latency.environment(args), "measurement": args.measurement})
        (memory if args.measurement == "memory" else lag)(args, out)


if __name__ == "__main__":
    main()
