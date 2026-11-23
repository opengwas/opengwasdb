#!/usr/bin/env python3
"""Time LD Store reads from local disk and over HTTP from a static host (ogld#8).

Two plans, each a list of cases timed on each target:

- `step1`: the published HGDP+1kGP EUR store (k = 10, layout C). Clumping each of
  opengwas/opengwasdb#403's 37 Analyses' 5e-8 tiers at r² 0.01 and 0.001 within 3 Mb; region
  matrices 3 Mb and 10 Mb wide, the second reading the Far Pairs for its pairs beyond
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

Where things are comes from their owners, never from here: the published EUR store is
`build_hgdp1kgp.store_path("EUR")` under `$OPENGWASDB_LD_STORES`, #403's Analyses are under
`$OPENGWASDB_LD_403_ANALYSES` and the layouts' builds under `$OPENGWASDB_LD_LAYOUTS`, each with its
default in `scripts/test_panel/roots.py`. Each run's first record says which commit's code ran and,
through `test_panel.checkout.problem`, whether the checkout could be cited as that code.

Threads: zarr's pool is `--zarr-threads` (6), BLAS 1; the static host and the proxy run one
event loop and at most two worker threads each. So a timing uses at most 8 threads busy at once
over HTTP, and 7 locally. Every record carries the node's load average before and after.

`--cache-mib` sets the chunk cache's budget, and `--without` turns off ogld#32's reader levers,
each as it was before that ticket, from outside `ogld/`: `concurrency` (each run of chunks its
own Zarr call, one after another), `indexes` (shard indexes unkept), `count` (string chunks
counted one string at a time) and `cache` (the 256 MiB budget). All four off is the reader
ogld#8 measured, and reproduces its request counts. ogld#8's `--reader concurrent`, a prototype
of concurrent fetching that copied the cache's keeping of chunks, is gone: the cache fetches
concurrently itself (ogld#32).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import subprocess
import sys
import threading
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
#: `scripts/`, where the owners of the recipe's facts and of the roots live.
SCRIPTS: Final = HERE.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
recipe = importlib.import_module("build_hgdp1kgp")
roots = importlib.import_module("test_panel.roots")
checkout = importlib.import_module("test_panel.checkout")

#: The cases each plan times, in the order it times them.
CASES: Final = {
    "step1": ("proxies", "scattered", "region", "clump"),
    "step2": ("pair", "scattered", "region", "clump"),
}
#: The region centres on chr22: inside its band, each with 5 Mb either side on the chromosome.
CENTRES: Final = (25_000_000, 32_000_000, 40_000_000)
SEED: Final = 8

Query = Callable[[Any], dict[str, Any]]


# ------------------------------------------------------------------ what a timing read


@dataclass
class Tally:
    """What the client read during one timing: zarr store reads, chunk-cache reads, and the
    seconds spent loading chromosomes and scattering the Far Pairs into a matrix."""

    gets: Counter[str] = field(default_factory=Counter)
    get_bytes: Counter[str] = field(default_factory=Counter)
    cache_reads: Counter[str] = field(default_factory=Counter)
    chunks: Counter[str] = field(default_factory=Counter)
    decoded_bytes: Counter[str] = field(default_factory=Counter)
    seconds: Counter[str] = field(default_factory=Counter)

    def as_json(self) -> dict[str, Any]:
        return {name: dict(value) for name, value in vars(self).items()}


TALLY = Tally()
_TALLY_LOCK = threading.Lock()


def _record_get(key: str, out: Any) -> None:
    name = http_static.array_of(key)
    with _TALLY_LOCK:
        TALLY.gets[name] += 1
        TALLY.get_bytes[name] += 0 if out is None else len(out)


def _counting_get(original: Callable[..., Any]) -> Callable[..., Any]:
    async def get(self: Any, key: str, prototype: Any = None, byte_range: Any = None) -> Any:
        out = await original(self, key, prototype, byte_range)
        _record_get(key, out)
        return out

    return get


def _counting_get_sync(original: Callable[..., Any]) -> Callable[..., Any]:
    def get(self: Any, key: str, *, prototype: Any = None, byte_range: Any = None) -> Any:
        out = original(self, key, prototype=prototype, byte_range=byte_range)
        _record_get(key, out)
        return out

    return get


def _tally_run(path: str, first: int, stop: int, size: int, itemsize: int) -> None:
    """Count one chunk-cache read of chunks first … stop − 1 of the array at `path`."""
    name = http_static.array_of(path + "/c/0")
    TALLY.cache_reads[name] += 1
    TALLY.chunks[name] += stop - first
    TALLY.decoded_bytes[name] += (stop - first) * size * itemsize


def _counting_run(original: Callable[..., Any]) -> Callable[..., Any]:
    """Count each run of chunks the cache fetched and keeps (`ChunkCache._keep_run`, ogld#32)."""

    def keep_run(self: Any, path: str, first: int, stop: int, size: int, data: Any) -> Any:
        _tally_run(path, first, stop, size, data.dtype.itemsize)
        return original(self, path, first, stop, size, data)

    return keep_run


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


#: How `instrument` replaces an attribute: `setattr` in a run, a test's `monkeypatch.setattr`.
Patch = Callable[[Any, str, Any], None]


#: The reader levers ogld#32 added, each of which `--without` turns off from outside `ogld/`.
LEVERS: Final = ("concurrency", "indexes", "count", "cache")
#: The chunk cache's budget before ogld#32, in MiB.
CACHE_MIB_BEFORE: Final = 256


def _sequential_chunks(self: Any, array: Any, wanted: np.ndarray, size: int) -> list[np.ndarray]:
    """`ChunkCache._chunks` as it was before ogld#32 (`--without concurrency`): each run of
    neighbouring missing chunks in its own Zarr call, one after another."""
    found = self._kept_chunks(array.path, wanted.tolist())
    missing = np.array([k for k in wanted.tolist() if k not in found], dtype=np.int64)
    for run in np.split(missing, np.flatnonzero(np.diff(missing) != 1) + 1):
        if len(run):
            first, stop = int(run[0]), int(run[-1]) + 1
            data = np.asarray(array[first * size : stop * size])
            found.update(self._keep_run(array.path, first, stop, size, data))
    return [found[k] for k in wanted.tolist()]


def _counted_per_string(chunk: np.ndarray) -> int:
    """`location._counted` as it was before ogld#32 (`--without count`): each string's UTF-8
    bytes, one string at a time, by the same rule and constants."""
    import math

    from opengwasdb.ld.store import location

    if chunk.dtype.kind != "T":
        return int(chunk.nbytes)
    utf8 = (len(s.encode("utf-8")) for s in chunk.tolist())
    long = [n for n in utf8 if n > location._INLINE]
    arena = math.ceil(location._GROWTH * (sum(long) + location._HEADER * len(long)))
    return int(chunk.nbytes) + arena + location._ARENA_FIXED


async def _index_passed(self: Any, key: str, prototype: Any, byte_range: Any = None) -> Any:
    """`_KeptIndexes.get` passing every read through, as before ogld#32 (`--without indexes`)."""
    return await self._store.get(key, prototype, byte_range)


def _index_passed_sync(
    self: Any, key: str, *, prototype: Any = None, byte_range: Any = None
) -> Any:
    return self._store.get_sync(key, prototype=prototype, byte_range=byte_range)


def without(levers: list[str], cache_mib: int | None, patch: Patch = setattr) -> int | None:
    """Turn `levers` off, each as it was before ogld#32, from outside `ogld/`: the cache's
    budget in MiB that leaves, None for ogld's default."""
    from opengwasdb.ld.store import location

    if "concurrency" in levers:
        patch(location.ChunkCache, "_chunks", _sequential_chunks)
    if "indexes" in levers:
        patch(location._KeptIndexes, "get", _index_passed)
        patch(location._KeptIndexes, "get_sync", _index_passed_sync)
    if "count" in levers:
        patch(location, "_counted", _counted_per_string)
    if "cache" in levers and cache_mib is None:
        return CACHE_MIB_BEFORE
    return cache_mib


def instrument(zarr_threads: int | None, cache_mib: int | None, patch: Patch = setattr) -> None:
    """Bound zarr's threads and count what each timing reads. The counting wraps the store's
    reads from outside ogld: the benchmark measures the reader, it does not change it."""
    import zarr
    from zarr.storage import FsspecStore, LocalStore

    from opengwasdb.ld.query import clumping
    from opengwasdb.ld.store import location
    from opengwasdb.ld.store.location import ChunkCache
    from opengwasdb.ld.store.reader import StoreReader

    if zarr_threads is not None:
        zarr.config.set({"threading.max_workers": zarr_threads})
    if cache_mib is not None:
        # A reader makes its cache with the module's budget when it opens (`ChunkCache()`).
        patch(location, "CACHE_BYTES", cache_mib << 20)
    _wrap(LocalStore, "get", _counting_get, patch)
    _wrap(LocalStore, "get_sync", _counting_get_sync, patch)
    _wrap(FsspecStore, "get", _counting_get, patch)
    _wrap(ChunkCache, "_keep_run", _counting_run, patch)
    _wrap(StoreReader, "_chromosome", _loading, patch)
    for owner, name, label in (
        (StoreReader, "_scatter_far", "far_scatter"),
        # The string reads and the cache's per-string count, the 7% of #14's profile.
        (StoreReader, "rows_of", "rows_of"),
        (StoreReader, "variants", "variants"),
        (StoreReader, "loci_of", "loci_of"),
        (StoreReader, "partner_row", "partner_row"),
        (clumping, "_component", "greedy"),
        (location, "_counted", "counted"),
    ):
        _wrap(owner, name, lambda original, label=label: _timing(original, label), patch)


def _wrap(owner: Any, name: str, wrapper: Callable[[Any], Any], patch: Patch) -> None:
    """Replace `owner.name` with `wrapper` of it: how the benchmark counts, from outside."""
    patch(owner, name, wrapper(getattr(owner, name)))


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
    without: tuple[str, ...] = ()

    def location(self, store: str) -> str:
        if self.host is None:
            return str(self.root / store)
        return f"{self.host.url}/{store}/"

    def timed(
        self, fn: Callable[[], dict[str, Any]], store_of: Callable[[], Any], **meta: Any
    ) -> dict[str, Any]:
        """Run `fn` once, timed, and write its record: what it read, the load around it, and
        the budget of the chunk cache the store `store_of` gives read through."""
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
            "without": list(self.without),
            "cache_bytes": int(store_of()._reader.cache_bytes),
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
            held["store"] = opened(self.location(store))
            took = time.perf_counter() - start
            return {"open_seconds": took, **query(held["store"])}

        self.timed(run, lambda: held["store"], store=store, state="cold", **meta)
        return held["store"]

    def again(self, store: str, held: Any, query: Query, state: str, **meta: Any) -> None:
        """Run `query` on the opened store `held`: `open` with its chunk cache emptied and the
        pages evicted, `warm` as the last run left it."""
        if state == "open":
            # Emptied in place: the store's wrapper keeps shard indexes in this same cache.
            cache = held._reader._cache
            with cache._lock:
                cache._kept.clear()
                cache._bytes = 0
            evict(self.root / store)
        self.timed(lambda: query(held), lambda: held, store=store, state=state, **meta)


def opened(location: str | Path) -> Any:
    """The LD Store at `location`, opened as a caller opens one."""
    import opengwasdb.ld

    return opengwasdb.ld.open(location)


def published() -> Path:
    """The published HGDP+1kGP EUR store, where the recipe that built it puts it (ogld#7)."""
    path: Path = recipe.store_path("EUR")
    return path


def layouts() -> Path:
    """Where `build_layouts.py` built EUR chr22 in each layout."""
    path: Path = roots.root("OPENGWASDB_LD_LAYOUTS")
    return path


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
        reader = opened(self.store)._reader
        start, stop = reader._ranges[chrom] if chrom else (0, int(reader._stops[-1]))
        return np.sort(np.random.default_rng(seed).choice(np.arange(start, stop), count, False))

    def alids(self, rows: np.ndarray) -> list[str]:
        return [str(a) for a in opened(self.store)._reader.alids_of(rows)]

    def near_pairs(self, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """`count` chr22 pairs within the store's near_bp: a random row, and a random later
        row within it, by the Near Band's own stop (`geometry.stop_within`, spec §4)."""
        from opengwasdb.ld.format.geometry import stop_within

        reader = opened(self.store)._reader
        c = reader._chromosome("22")
        rng = np.random.default_rng(seed)
        local = rng.choice(len(c.position) - 1, count, replace=False)
        stop = stop_within(c.position, reader.provenance["near_bp"])[local]
        other = np.array([rng.integers(i + 1, s) for i, s in zip(local, stop, strict=True)])
        return local + c.start, other + c.start


def analyses(names: list[str] | None) -> list[Path]:
    """#403's Analyses under `$OPENGWASDB_LD_403_ANALYSES`: all of them, or those `names`, every
    one of which must be there. A name that is not is refused, never dropped, so a run cannot
    time fewer Analyses than it was asked for."""
    found = sorted(p for p in roots.root("OPENGWASDB_LD_403_ANALYSES").iterdir() if p.is_dir())
    if names is None:
        return found
    unknown = sorted(set(names) - {p.name for p in found})
    if unknown:
        raise SystemExit(
            f"no such Analyses under $OPENGWASDB_LD_403_ANALYSES: {', '.join(unknown)}"
        )
    return [p for p in found if p.name in names]


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
    store = published().name
    sample = Sample(published())
    cases = set(args.case or CASES["step1"])
    if "proxies" in cases:
        proxies(b, store, sample, args.seeds)
    # #8 drew lists on chr22 and genome-wide; another chromosome (#31: chr11) is drawn alone.
    lists = ("22", None) if args.chrom == "22" else (args.chrom,)
    if "scattered" in cases:
        scattered(b, store, sample, lists, args.seeds)
    if "region" in cases:
        regions(b, store, tuple(args.widths), args.chrom, tuple(args.centres))
    if "clump" in cases:
        clump_each(b, store, args, None)


def step2(b: Bench, args: argparse.Namespace) -> None:
    """Each case on every layout in turn, so the node's load drifts over all of them alike.
    The layouts hold the same data (`build_layouts.py --check`), so one sample serves all."""
    stores = [f"chr22-{layout}.store" for layout in args.layout]
    sample = Sample(layouts() / stores[0])
    cases: dict[str, Callable[[str], None]] = {
        "pair": lambda store: single_pairs(b, store, sample),
        "scattered": lambda store: scattered(b, store, sample, ("22",), args.seeds),
        "region": lambda store: regions(b, store, (3,)),
        "clump": lambda store: clump_each(b, store, args, "22"),
    }
    for name in args.case or CASES["step2"]:
        for store in stores:
            cases[name](store)


#: Each plan, and the directory its stores are in (and that the static host serves).
PLANS: Final = {
    "step1": (step1, lambda: published().parent),
    "step2": (step2, layouts),
}


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


def code_state(installed: str) -> dict[str, Any]:
    """Which commit's code a run ran, and why the checkout cannot be cited as that code ("" when
    it can): `test_panel.checkout.problem`, the one guard `make_pcs.py` and the realdata tier
    also use. It refuses an installed ogld that does not name HEAD, a changed tracked file under
    `ogld/` or `scripts/`, and an untracked `.py` there."""
    repository = SCRIPTS.parent
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    problem: str = checkout.problem(repository, installed)
    return {"commit": commit, "installed": installed, "clean": not problem, "problem": problem}


def environment(args: argparse.Namespace) -> dict[str, Any]:
    """What a run ran on: the code, the software, the thread bounds and the node's load when it
    began."""
    import aiohttp
    import fsspec
    import zarr

    import opengwasdb.ld

    code = code_state(opengwasdb.ld.__version__)
    return {
        "kind": "environment",
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": sys.argv[1:],
        "commit": code["commit"],
        "code": code,
        # The ogld a run imported: this checkout's, or another put first on PYTHONPATH, as #31
        # timed the tip's reader beside its own.
        "ogld_source": str(Path(opengwasdb.ld.__file__).resolve().parent),
        "versions": {
            "ogld": opengwasdb.ld.__version__,
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
        "--cache-mib", type=int, help="the chunk cache's budget; ogld's default if omitted"
    )
    parser.add_argument(
        "--without",
        nargs="+",
        choices=LEVERS,
        default=[],
        help="ogld#32's reader levers to turn off, each as it was before that ticket",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    checked(args)
    instrument(args.zarr_threads, without(args.without, args.cache_mib))
    plan, root_of = PLANS[args.plan]
    root = root_of()
    with args.out.open("a") as out:
        out.write(json.dumps(environment(args)) + "\n")
        for target in args.target:
            with hosted(root, target) as host:
                bench = Bench(root, target, host, out, args.plan, "ogld", tuple(args.without))
                plan(bench, args)


def checked(args: argparse.Namespace) -> None:
    """Refuse, before anything is timed, a case the plan does not have or an Analysis that is
    not there: a name dropped would time less than was asked for, and say nothing."""
    unknown = sorted(set(args.case or ()) - set(CASES[args.plan]))
    if unknown:
        known = ", ".join(CASES[args.plan])
        raise SystemExit(f"{args.plan} has no case {', '.join(unknown)}; its cases are {known}")
    analyses(args.analyses)


if __name__ == "__main__":
    main()
