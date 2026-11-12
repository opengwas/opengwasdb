#!/usr/bin/env python3
"""Time LD Store reads from local disk and over HTTP from a static host (ogld#8).

Two plans, each a list of cases timed on each target:

- `step1`: the published HGDP+1kGP EUR store (k = 10, layout C). Clumping each of
  opengwas/opengwasdb#403's 37 Analyses' 5e-8 tiers at r² 0.01 and 0.001 within 3 Mb; region
  matrices 3 Mb and 10 Mb wide, the second reading the Proxy Index rows for its pairs beyond
  near_bp (spec §11); a matrix of 500 scattered variants, on chr22 and genome-wide; and proxies
  at r² 0.8 within 10 Mb for 1, 10 and 100 variants.
- `step2`: EUR chr22 built in each Near Band layout of spec §6 (`build_layouts.py`). Clumping
  the 37 Analyses' chr22 hits below t_far (the column read), a 3 Mb region, 500 scattered
  variants, and single pairs.

A target is `local`, or `http<ms>`: the store's directory served by `http_static.py`'s static
host, read through its proxy with <ms> added to every request. That is loopback with injected
latency, not a real WAN. Each record says what a timing read: the client's zarr reads and
bytes per array, its chunk-cache reads, and over HTTP the requests and bytes the proxy saw.

States: `cold` evicts the store's pages from the OS cache (posix_fadvise) and opens the store
inside the timing; `open` keeps the opened reader and its chromosomes' positions and `indptr`,
but empties its chunk cache and evicts the pages, as a long-lived service meets a new query;
`warm` repeats the query on the reader as the last run left it.

    pixi run -e dev bench-latency step1 --target local http0 --out results.jsonl

Threads: zarr's pool is `--zarr-threads` (6), BLAS 1; the static host and the proxy run one
event loop and at most two worker threads each. So a timing uses at most 8 threads busy at once
over HTTP, and 7 locally. Every record carries the node's load average before and after.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, TextIO

import http_static
import numpy as np

HERE: Final = Path(__file__).resolve().parent

STORES: Final = Path("/data/opengwasdb/work/ogld/stores")
EUR: Final = "hgdp1kgp-v2-EUR-k10.store"
ANALYSES: Final = Path("/data/opengwasdb/work/ogld/realdata/403-analyses")
LAYOUT_DIR: Final = Path("/data/opengwasdb/work/ogld/scratch/8/layouts")
#: The region centres on chr22: inside its band, each with 5 Mb either side on the chromosome.
CENTRES: Final = (25_000_000, 32_000_000, 40_000_000)
SEED: Final = 8

Query = Callable[[Any], dict[str, Any]]


# ------------------------------------------------------------------ what a timing read


@dataclass
class Tally:
    """What the client read during one timing: zarr store reads, chunk-cache reads, and the
    seconds spent loading chromosomes and scattering the Proxy Index into a matrix."""

    gets: Counter[str] = field(default_factory=Counter)
    get_bytes: Counter[str] = field(default_factory=Counter)
    cache_reads: Counter[str] = field(default_factory=Counter)
    chunks: Counter[str] = field(default_factory=Counter)
    decoded_bytes: Counter[str] = field(default_factory=Counter)
    seconds: Counter[str] = field(default_factory=Counter)

    def as_json(self) -> dict[str, Any]:
        return {name: dict(value) for name, value in vars(self).items()}


TALLY = Tally()


def _counting_get(original: Callable[..., Any]) -> Callable[..., Any]:
    async def get(self: Any, key: str, prototype: Any = None, byte_range: Any = None) -> Any:
        out = await original(self, key, prototype, byte_range)
        name = http_static.array_of(key)
        TALLY.gets[name] += 1
        TALLY.get_bytes[name] += 0 if out is None else len(out)
        return out

    return get


def _counting_run(original: Callable[..., Any]) -> Callable[..., Any]:
    def read_run(self: Any, array: Any, first: int, stop: int, size: int) -> Any:
        name = http_static.array_of(array.path + "/c/0")
        TALLY.cache_reads[name] += 1
        TALLY.chunks[name] += stop - first
        TALLY.decoded_bytes[name] += (stop - first) * size * array.dtype.itemsize
        return original(self, array, first, stop, size)

    return read_run


def _timing(original: Callable[..., Any], label: str) -> Callable[..., Any]:
    def timed(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            TALLY.seconds[label] += time.perf_counter() - start

    return timed


def _loading(original: Callable[..., Any]) -> Callable[..., Any]:
    def chromosome(self: Any, name: str) -> Any:
        if name in self._loaded:
            return self._loaded[name]
        return _timing(original, "chromosome_loads")(self, name)

    return chromosome


def _concurrent_chunks(self: Any, array: Any, wanted: np.ndarray, size: int) -> list[np.ndarray]:
    """`ChunkCache._chunks`, with every run of missing chunks fetched at once (`--reader
    concurrent`): a prototype of one lever, benchmark code only.

    ogld's `_chunks` reads each run of neighbouring missing chunks in its own Zarr call, one
    after another, and over HTTP each call waits at least two round trips: the shard index,
    then the chunks. This starts every run's read together on zarr's event loop, at most
    zarr's async concurrency at a time, and keeps each chunk as `_read_run` does. It decodes
    the same chunks, so it changes no answer; only when the requests are made.
    """
    found = self._kept_chunks(array.path, wanted.tolist())
    missing = np.array([k for k in wanted.tolist() if k not in found], dtype=np.int64)
    split = np.split(missing, np.flatnonzero(np.diff(missing) != 1) + 1)
    runs = [(int(run[0]), int(run[-1]) + 1) for run in split if len(run)]
    for (first, stop), data in zip(runs, _fetched(array, runs, size), strict=True):
        found.update(_kept_run(self, array, first, stop, size, np.asarray(data)))
    return [found[k] for k in wanted.tolist()]


def _fetched(array: Any, runs: list[tuple[int, int]], size: int) -> list[Any]:
    """Each run's elements, read concurrently through zarr's async API."""
    import asyncio

    import zarr
    from zarr.core.sync import sync

    async def gathered() -> list[Any]:
        gate = asyncio.Semaphore(int(zarr.config.get("async.concurrency")))

        async def one(first: int, stop: int) -> Any:
            async with gate:
                return await array._async_array.getitem(slice(first * size, stop * size))

        return list(await asyncio.gather(*(one(a, b) for a, b in runs)))

    return sync(gathered()) if runs else []


def _kept_run(
    cache: Any, array: Any, first: int, stop: int, size: int, data: np.ndarray
) -> dict[int, np.ndarray]:
    """Chunks first … stop − 1 of `data`, copied out, counted and kept, as `_read_run` does."""
    from ogld.store import location

    name = http_static.array_of(array.path + "/c/0")
    TALLY.cache_reads[name] += 1
    TALLY.chunks[name] += stop - first
    TALLY.decoded_bytes[name] += (stop - first) * size * array.dtype.itemsize
    copies = [data[(k - first) * size : (k - first + 1) * size].copy() for k in range(first, stop)]
    read = {first + i: (chunk, location._counted(chunk)) for i, chunk in enumerate(copies)}
    with cache._lock:
        return {k: cache._keep((array.path, k), *entry) for k, entry in read.items()}


def instrument(zarr_threads: int, reader: str, cache_mib: int | None) -> None:
    """Bound zarr's threads and count what each timing reads. The counting wraps the store's
    reads from outside ogld: the benchmark measures the reader, it does not change it, except
    with `reader` "concurrent", the prototype `_concurrent_chunks`."""
    import zarr
    from zarr.storage import FsspecStore, LocalStore

    from ogld.store import location
    from ogld.store.location import ChunkCache
    from ogld.store.reader import StoreReader

    zarr.config.set({"threading.max_workers": zarr_threads})
    if cache_mib is not None:
        # A reader makes its cache with the module's budget when it opens (`ChunkCache()`).
        location.CACHE_BYTES = cache_mib << 20
    if reader == "concurrent":
        ChunkCache._chunks = _concurrent_chunks
    _wrap(LocalStore, "get", _counting_get)
    _wrap(FsspecStore, "get", _counting_get)
    _wrap(ChunkCache, "_read_run", _counting_run)
    _wrap(StoreReader, "_chromosome", _loading)
    for owner, name, label in (
        (StoreReader, "_scatter_far", "far_scatter"),
        # The string reads and the cache's per-string count, the 7% of #14's profile.
        (StoreReader, "rows_of", "rows_of"),
        (StoreReader, "variants", "variants"),
        (location, "_counted", "counted"),
    ):
        _wrap(owner, name, lambda original, label=label: _timing(original, label))


def _wrap(owner: Any, name: str, wrapper: Callable[[Any], Any]) -> None:
    """Replace `owner.name` with `wrapper` of it: how the benchmark counts, from outside."""
    setattr(owner, name, wrapper(getattr(owner, name)))


# ------------------------------------------------------------------ where a store is read from


class Host:
    """`http_static.py`'s static host for `root` and its latency proxy, as child processes."""

    def __init__(self, root: Path) -> None:
        self._children: list[subprocess.Popen[str]] = []
        static = self._start("static", str(root))
        self.url = f"http://127.0.0.1:{self._start('proxy', f'http://127.0.0.1:{static}')}"

    def _start(self, role: str, target: str) -> str:
        child = subprocess.Popen(
            [sys.executable, str(HERE / "http_static.py"), role, target],
            stdout=subprocess.PIPE,
            text=True,
        )
        self._children.append(child)
        assert child.stdout is not None
        ready = child.stdout.readline().split()
        if ready[:1] != ["ready"]:
            raise SystemExit(f"http_static.py {role} did not start: {ready}")
        return ready[1]

    def control(self, action: str) -> dict[str, Any]:
        request = urllib.request.Request(f"{self.url}/__ogld/{action}", method="POST")
        with urllib.request.urlopen(request) as response:
            answer: dict[str, Any] = json.load(response)
        return answer

    def close(self) -> None:
        for child in self._children:
            child.terminate()
            child.wait()


@dataclass(frozen=True)
class Target:
    """Local disk, or HTTP with `latency_ms` added to every request."""

    name: str
    latency_ms: float | None

    @classmethod
    def parse(cls, text: str) -> Target:
        if text == "local":
            return cls(text, None)
        if text.startswith("http") and text[4:].isdigit():
            return cls(text, float(text[4:]))
        raise argparse.ArgumentTypeError(f"a target is local or http<ms>, not {text!r}")


def evict(store: Path) -> None:
    """Drop the store's file pages from the OS cache, so the next read comes from disk."""
    for path in store.rglob("*"):
        if path.is_file():
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)


# ------------------------------------------------------------------ one timing


@dataclass
class Bench:
    """Times queries on the stores under `root`, from one target, and writes each record."""

    root: Path
    target: Target
    host: Host | None
    out: TextIO
    step: str
    reader: str

    def location(self, store: str) -> str:
        if self.host is None:
            return str(self.root / store)
        return f"{self.host.url}/{store}/"

    def timed(self, fn: Callable[[], dict[str, Any]], **meta: Any) -> dict[str, Any]:
        """Run `fn` once, timed, and write its record: what it read, and the load around it."""
        global TALLY
        TALLY = Tally()
        if self.host is not None:
            self.host.control("reset")
        before, start = os.getloadavg(), time.perf_counter()
        result = fn()
        seconds = time.perf_counter() - start
        record = {
            "step": self.step,
            "reader": self.reader,
            "cache_bytes": _cache_bytes(),
            "target": self.target.name,
            "latency_ms": self.target.latency_ms,
            **meta,
            "seconds": seconds,
            "load_before": before,
            "load_after": os.getloadavg(),
            "client": TALLY.as_json(),
            "wire": None if self.host is None else self.host.control("stats"),
            "result": result,
        }
        self.out.write(json.dumps(record) + "\n")
        self.out.flush()
        return record

    def cold(self, store: str, query: Query, **meta: Any) -> Any:
        """Evict, open the store and run `query`, all timed; the opened store is returned."""
        evict(self.root / store)
        held: dict[str, Any] = {}

        def run() -> dict[str, Any]:
            start = time.perf_counter()
            held["store"] = _open(self.location(store))
            opened = time.perf_counter() - start
            return {"open_seconds": opened, **query(held["store"])}

        self.timed(run, store=store, state="cold", **meta)
        return held["store"]

    def again(self, store: str, opened: Any, query: Query, state: str, **meta: Any) -> None:
        """Run `query` on the opened store: `open` with its chunk cache emptied and the pages
        evicted, `warm` as the last run left it."""
        if state == "open":
            from ogld.store.location import ChunkCache

            opened._reader._cache = ChunkCache()
            evict(self.root / store)
        self.timed(lambda: query(opened), store=store, state=state, **meta)


def _cache_bytes() -> int:
    from ogld.store import location

    return int(location.CACHE_BYTES)


def _open(location: str) -> Any:
    import ogld

    return ogld.open(location)


# ------------------------------------------------------------------ the queries


def hits(analysis: Path, chrom: str | None) -> tuple[list[str], np.ndarray]:
    """An Analysis's 5e-8 tier: ALIDs and |z|, on one chromosome when `chrom` is given."""
    table = np.loadtxt(analysis / "hits.tsv", dtype=str, skiprows=1, ndmin=2)
    if chrom is not None:
        table = table[np.char.startswith(table[:, 0], f"{chrom}:")]
    return list(table[:, 0]), np.abs(table[:, 1].astype(np.float64))


def clump_query(alids: list[str], z: np.ndarray, r2: float) -> Query:
    def query(store: Any) -> dict[str, Any]:
        clumps = store.clump(alids, z, r2=r2, kb=3000)
        return {"hits": len(alids), "leads": len(clumps.leads())}

    return query


def region_query(chrom: str, centre: int, radius: int) -> Query:
    def query(store: Any) -> dict[str, Any]:
        m = store.region(chrom, centre, radius)
        return {"variants": len(m.variants.alid), "stored": int((~np.isnan(m.r)).sum())}

    return query


def matrix_query(alids: list[str]) -> Query:
    def query(store: Any) -> dict[str, Any]:
        m = store.matrix(alids)
        return {"variants": len(alids), "stored": int((~np.isnan(m.r)).sum())}

    return query


def pairs_query(first: np.ndarray, second: np.ndarray) -> Query:
    def query(store: Any) -> dict[str, Any]:
        r = store._reader.r_of_pairs(first, second)
        return {"pairs": len(first), "stored": int((~np.isnan(r)).sum())}

    return query


def proxies_query(alids: list[str]) -> Query:
    def query(store: Any) -> dict[str, Any]:
        found = store.proxies(alids, r2=0.8, kb=10_000)
        return {"variants": len(alids), "proxies": len(found.proxy)}

    return query


# ------------------------------------------------------------------ choosing the inputs


@dataclass(frozen=True)
class Sample:
    """Seeded choices of variants from one store, made before any timing on a separate
    reader, so choosing them warms nothing a timing reads."""

    store: Path

    def rows(self, count: int, chrom: str | None, seed: int) -> np.ndarray:
        import ogld

        reader = ogld.open(self.store)._reader
        start, stop = reader._ranges[chrom] if chrom else (0, int(reader._stops[-1]))
        return np.sort(np.random.default_rng(seed).choice(np.arange(start, stop), count, False))

    def alids(self, rows: np.ndarray) -> list[str]:
        import ogld

        return [str(a) for a in ogld.open(self.store)._reader.alids_of(rows)]

    def near_pairs(self, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """`count` chr22 pairs within near_bp: a random row, and a random later row within it."""
        import ogld

        reader = ogld.open(self.store)._reader
        c = reader._chromosome("22")
        rng = np.random.default_rng(seed)
        local = rng.choice(len(c.position) - 1, count, replace=False)
        stop = np.searchsorted(c.position, c.position[local] + 3_000_000, side="right")
        other = np.array([rng.integers(i + 1, s) for i, s in zip(local, stop, strict=True)])
        return local + c.start, other + c.start


def analyses(names: list[str] | None) -> list[Path]:
    found = sorted(p for p in ANALYSES.iterdir() if p.is_dir())
    return [p for p in found if names is None or p.name in names]


# ------------------------------------------------------------------ the plans


def clump_each(b: Bench, store: str, args: argparse.Namespace, chrom: str | None) -> None:
    """Each Analysis at each r², cold (a fresh open per Analysis) and then warm."""
    for analysis in analyses(args.analyses):
        alids, z = hits(analysis, chrom)
        if not alids:
            continue
        for r2 in args.r2:
            meta = {"case": "clump", "analysis": analysis.name, "r2": r2}
            opened = b.cold(store, clump_query(alids, z, r2), **meta)
            if not args.cold_only:
                b.again(store, opened, clump_query(alids, z, r2), "warm", **meta)


def three_states(b: Bench, store: str, query: Query, **meta: Any) -> None:
    opened = b.cold(store, query, **meta)
    b.again(store, opened, query, "open", **meta)
    b.again(store, opened, query, "warm", **meta)


def regions(
    b: Bench,
    store: str,
    widths: tuple[int, ...],
    chrom: str = "22",
    centres: tuple[int, ...] = CENTRES,
) -> None:
    """Each region `width` Mb wide at each centre on `chrom`, cold and then warm. A region off
    chr22 records its chromosome; #8's chr22 records carry none."""
    for width in widths:
        for centre in centres:
            meta = {"case": "region", "width_mb": width, "centre": centre}
            meta |= {} if chrom == "22" else {"chrom": chrom}
            query = region_query(chrom, centre, width * 500_000)
            opened = b.cold(store, query, **meta)
            b.again(store, opened, query, "warm", **meta)
            del opened


def scattered(
    b: Bench, store: str, sample: Sample, chroms: tuple[str | None, ...], seeds: int
) -> None:
    for chrom in chroms:
        for seed in range(SEED, SEED + seeds):
            alids = sample.alids(sample.rows(500, chrom, seed))
            meta = {"case": "scattered", "chrom": chrom or "genome", "seed": seed}
            three_states(b, store, matrix_query(alids), **meta)


def proxies(b: Bench, store: str, sample: Sample, seeds: int) -> None:
    for count in (1, 10, 100):
        for seed in range(SEED, SEED + (10 if count == 1 else seeds)):
            alids = sample.alids(sample.rows(count, None, seed))
            meta = {"case": "proxies", "variants": count, "seed": seed}
            three_states(b, store, proxies_query(alids), **meta)


def single_pairs(b: Bench, store: str, sample: Sample) -> None:
    first, second = sample.near_pairs(20, SEED)
    for k, (i, j) in enumerate(zip(first.tolist(), second.tolist(), strict=True)):
        query = pairs_query(np.array([i]), np.array([j]))
        meta = {"case": "pair", "pair": k}
        if k < 5:
            three_states(b, store, query, **meta)
            continue
        opened = b.cold(store, query, **meta)
        b.again(store, opened, query, "open", **meta)


def step1(b: Bench, args: argparse.Namespace) -> None:
    sample = Sample(STORES / EUR)
    cases = set(args.case or ("proxies", "scattered", "region", "clump"))
    if "proxies" in cases:
        proxies(b, EUR, sample, args.seeds)
    # #8 drew lists on chr22 and genome-wide; another chromosome (#31: chr11) is drawn alone.
    lists = ("22", None) if args.chrom == "22" else (args.chrom,)
    if "scattered" in cases:
        scattered(b, EUR, sample, lists, args.seeds)
    if "region" in cases:
        regions(b, EUR, tuple(args.widths), args.chrom, tuple(args.centres))
    if "clump" in cases:
        clump_each(b, EUR, args, None)


def step2(b: Bench, args: argparse.Namespace) -> None:
    """Each case on every layout in turn, so the node's load drifts over all of them alike.
    The layouts hold the same data (`build_layouts.py --check`), so one sample serves all."""
    stores = [f"chr22-{layout}.store" for layout in args.layout]
    sample = Sample(LAYOUT_DIR / stores[0])
    cases: dict[str, Callable[[str], None]] = {
        "pair": lambda store: single_pairs(b, store, sample),
        "scattered": lambda store: scattered(b, store, sample, ("22",), args.seeds),
        "region": lambda store: regions(b, store, (3,)),
        "clump": lambda store: clump_each(b, store, args, "22"),
    }
    for name in args.case or cases:
        for store in stores:
            cases[name](store)


PLANS: Final = {"step1": (step1, STORES), "step2": (step2, LAYOUT_DIR)}


@contextlib.contextmanager
def hosted(root: Path, target: Target) -> Iterator[Host | None]:
    if target.latency_ms is None:
        yield None
        return
    host = Host(root)
    try:
        host.control(f"latency/{target.latency_ms:g}")
        yield host
    finally:
        host.close()


def environment(args: argparse.Namespace) -> dict[str, Any]:
    """What a run ran on: the software, the thread bounds and the node's load when it began."""
    import aiohttp
    import fsspec
    import zarr

    import ogld

    commit = subprocess.run(
        ["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    return {
        "kind": "environment",
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": sys.argv[1:],
        "commit": commit,
        # The ogld a run imported: this checkout's, or another put first on PYTHONPATH, as #31
        # timed the tip's reader beside its own.
        "ogld_source": str(Path(ogld.__file__).resolve().parent),
        "versions": {
            "ogld": ogld.__version__,
            "zarr": zarr.__version__,
            "numpy": np.__version__,
            "fsspec": fsspec.__version__,
            "aiohttp": aiohttp.__version__,
        },
        "zarr_config": {k: zarr.config.get(k) for k in ("threading", "async", "array")},
        "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "cpus": os.cpu_count(),
        "load": os.getloadavg(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("plan", choices=sorted(PLANS))
    parser.add_argument("--target", nargs="+", type=Target.parse, default=[Target("local", None)])
    parser.add_argument("--case", nargs="+", help="the plan's cases to run; all if omitted")
    parser.add_argument("--layout", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--analyses", nargs="+", help="the Analyses to clump; all 37 if omitted")
    parser.add_argument("--r2", nargs="+", type=float, default=[0.01, 0.001])
    parser.add_argument("--cold-only", action="store_true", help="clump cold only, not warm")
    parser.add_argument("--chrom", default="22", help="step 1's chromosome for lists and regions")
    parser.add_argument(
        "--centres", nargs="+", type=int, default=list(CENTRES), help="step 1's region centres"
    )
    parser.add_argument(
        "--widths", nargs="+", type=int, default=[3, 10], help="step 1's region widths, Mb"
    )
    parser.add_argument(
        "--seeds", type=int, default=3, help="draws of 500 scattered, 10 and 100 proxy variants"
    )
    parser.add_argument("--zarr-threads", type=int, default=6)
    parser.add_argument(
        "--reader",
        choices=["ogld", "concurrent"],
        default="ogld",
        help="ogld's reader as it is, or with the benchmark's concurrent chunk-run reads",
    )
    parser.add_argument(
        "--cache-mib", type=int, help="the chunk cache's budget; ogld's 256 MiB if omitted"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    instrument(args.zarr_threads, args.reader, args.cache_mib)
    plan, root = PLANS[args.plan]
    with args.out.open("a") as out:
        out.write(json.dumps(environment(args)) + "\n")
        for target in args.target:
            with hosted(root, target) as host:
                plan(Bench(root, target, host, out, args.plan, args.reader), args)


if __name__ == "__main__":
    main()
