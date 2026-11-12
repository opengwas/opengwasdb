#!/usr/bin/env python3
"""Render `latency.py`'s records as the tables of `docs/benchmarks/latency-tables.md` (ogld#8).

    pixi run -e dev python scripts/bench/report.py docs/benchmarks/results/*.jsonl \
        --layouts /data/opengwasdb/work/ogld/scratch/8/layouts > docs/benchmarks/latency-tables.md

Every number in the tables is a record's, aggregated as each table says: a sum over the 37
Analyses for clumping, a median over its repeats for everything else. Each row gives the node's
1-minute load average across its records, before and after, as the records carry it.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Record = dict[str, Any]
STATES = ("cold", "open", "warm")
TARGET_ORDER = ("local", "http0", "http10", "http50")


def read(paths: Iterable[Path]) -> tuple[list[Record], list[Record]]:
    """The timing records and the environment records of every file."""
    timings, environments = [], []
    for path in paths:
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record.get("kind") == "environment":
                environments.append(record)
            else:
                timings.append({**record, "reader": reader_label(record)})
    return timings, environments


def reader_label(record: Record) -> str:
    """The reader a record ran: ogld's or the concurrent prototype, and its cache if not ogld's
    256 MiB."""
    cache = record.get("cache_bytes", 1 << 28)
    label = str(record.get("reader", "ogld"))
    return label if cache == 1 << 28 else f"{label}, cache {cache >> 20:,} MiB"


def seconds(value: float) -> str:
    if value < 1:
        return f"{value * 1000:.0f} ms" if value >= 0.01 else f"{value * 1000:.1f} ms"
    return f"{value:.2f} s" if value < 100 else f"{value:,.0f} s"


def megabytes(value: float) -> str:
    return f"{value / 1e6:,.1f}" if value >= 1e5 else f"{value / 1e6:.3f}"


def wire_or_client(record: Record, key: str) -> float:
    """Requests or bytes: the proxy's count over HTTP, the client's zarr reads locally."""
    if record["wire"] is not None:
        return float(record["wire"][key])
    client = record["client"]["gets" if key == "requests" else "get_bytes"]
    return float(sum(client.values()))


def array_bytes(record: Record, prefix: str) -> float:
    if record["wire"] is not None:
        found = record["wire"]["bytes_by_array"]
    else:
        found = record["client"]["get_bytes"]
    return float(sum(v for k, v in found.items() if k.startswith(prefix)))


def index_row_bytes(record: Record) -> float:
    """The bytes of Proxy Index rows read: `partner` and `r`, not the `indptr` a load reads."""
    return array_bytes(record, "index/partner") + array_bytes(record, "index/r")


def load_range(records: list[Record]) -> str:
    values = [r["load_before"][0] for r in records] + [r["load_after"][0] for r in records]
    return f"{min(values):.0f}–{max(values):.0f}"


def order_key(target: str) -> tuple[int, str]:
    return (TARGET_ORDER.index(target) if target in TARGET_ORDER else 99, target)


def grouped(records: list[Record], *keys: str) -> dict[tuple[Any, ...], list[Record]]:
    out: dict[tuple[Any, ...], list[Record]] = defaultdict(list)
    for record in records:
        out[tuple(record.get(k) for k in keys)].append(record)
    return out


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def median_of(records: list[Record], value: Callable[[Record], float]) -> float:
    return statistics.median(value(r) for r in records)


def chunk_count(record: Record, prefix: str = "") -> float:
    return float(sum(v for k, v in record["client"]["chunks"].items() if k.startswith(prefix)))


def cache_reads(record: Record) -> float:
    return float(sum(record["client"]["cache_reads"].values()))


def spent(record: Record, label: str) -> float:
    return float(record["client"]["seconds"].get(label, 0.0))


# ------------------------------------------------------------------ step 1


def clump_rows(records: list[Record]) -> list[list[str]]:
    """One row per reader, r², state and target: sums over the Analyses timed."""
    groups = grouped(records, "reader", "r2", "state", "target")

    def sort_key(item: tuple[tuple[Any, ...], list[Record]]) -> tuple[Any, ...]:
        reader, r2, state, target = item[0]
        return (reader, -r2, STATES.index(state), order_key(target))

    return [clump_row(key, found) for key, found in sorted(groups.items(), key=sort_key)]


def clump_row(key: tuple[Any, ...], found: list[Record]) -> list[str]:
    reader, r2, state, target = key
    total = sum(r["seconds"] for r in found)
    leads = sum(r["result"]["leads"] for r in found)
    return [
        reader,
        f"{r2:g}",
        state,
        target,
        str(len(found)),
        f"{sum(r['result']['hits'] for r in found):,}",
        f"{leads:,}",
        seconds(total),
        f"{1000 * total / max(leads, 1):.1f}",
        f"{sum(cache_reads(r) for r in found):,.0f}",
        f"{sum(chunk_count(r, 'near') for r in found):,.0f}",
        megabytes(sum(wire_or_client(r, "bytes") for r in found)),
        f"{sum(wire_or_client(r, 'requests') for r in found):,.0f}",
        load_range(found),
    ]


CLUMP_HEADER = [
    "reader",
    "r²",
    "state",
    "target",
    "Analyses",
    "hits",
    "leads",
    "time, sum",
    "ms per lead",
    "chunk-cache reads",
    "Near Band chunks decoded",
    "MB read",
    "requests",
    "load",
]


def median_rows(
    records: list[Record], keys: tuple[str, ...], columns: list[Callable[[list[Record]], str]]
) -> list[list[str]]:
    rows = []
    groups = grouped(records, *keys)

    def sort_key(item: tuple[tuple[Any, ...], list[Record]]) -> tuple[Any, ...]:
        key = item[0]
        return tuple(column_key(k, v) for k, v in zip(keys, key, strict=True))

    for key, found in sorted(groups.items(), key=sort_key):
        rows.append([str(v) for v in key] + [column(found) for column in columns])
    return rows


def column_key(name: str, value: Any) -> tuple[Any, ...]:
    """Targets in latency order, states cold to warm, numbers as numbers, the rest as text."""
    if name == "target":
        return order_key(value)
    if name == "state":
        return (STATES.index(value), "")
    if isinstance(value, int | float):
        return (value, "")
    return (float("inf"), str(value))


def common_columns() -> list[Callable[[list[Record]], str]]:
    return [
        lambda f: str(len(f)),
        lambda f: seconds(median_of(f, lambda r: r["seconds"])),
        lambda f: (
            f"{seconds(min(r['seconds'] for r in f))}–{seconds(max(r['seconds'] for r in f))}"
        ),
        lambda f: megabytes(median_of(f, lambda r: wire_or_client(r, "bytes"))),
        lambda f: f"{median_of(f, lambda r: wire_or_client(r, 'requests')):,.0f}",
        lambda f: f"{median_of(f, cache_reads):,.0f}",
    ]


COMMON_HEADER = ["runs", "time, median", "range", "MB read", "requests", "chunk-cache reads"]


def region_columns() -> list[Callable[[list[Record]], str]]:
    return [
        lambda f: f"{median_of(f, lambda r: r['result']['variants']):,.0f}",
        *common_columns(),
        lambda f: megabytes(median_of(f, lambda r: array_bytes(r, "near/r"))),
        lambda f: megabytes(median_of(f, index_row_bytes)),
        lambda f: seconds(median_of(f, lambda r: spent(r, "far_scatter"))),
        load_range,
    ]


REGION_HEADER = [
    "variants",
    *COMMON_HEADER,
    "MB of `near/r`",
    "MB of index rows",
    "far-pair scatter",
    "load",
]


def list_columns() -> list[Callable[[list[Record]], str]]:
    return [
        *common_columns(),
        lambda f: f"{median_of(f, lambda r: chunk_count(r, 'near')):,.0f}",
        lambda f: f"{median_of(f, lambda r: chunk_count(r, 'index')):,.0f}",
        lambda f: seconds(median_of(f, lambda r: spent(r, "far_scatter"))),
        lambda f: seconds(median_of(f, lambda r: spent(r, "chromosome_loads"))),
        load_range,
    ]


LIST_HEADER = [
    *COMMON_HEADER,
    "Near Band chunks",
    "index chunks",
    "far-pair scatter",
    "chromosome loads",
    "load",
]


def step1(records: list[Record]) -> list[str]:
    of = grouped(records, "case")
    out = ["## Step 1: the published EUR store (× 511, k = 10, layout C)", ""]
    out += ["### Summary: time per target, ogld's reader", ""]
    out += summary(records, STEP1_READS, "ogld")
    out += ["### Clumping #403's 37 Analyses, 5e-8 tier, 3 Mb window", ""]
    out += [table(CLUMP_HEADER, clump_rows(of.get(("clump",), []))), ""]
    out += ["### Region matrices on chr22, centred at 25, 32 and 40 Mb", ""]
    keys = ("reader", "width_mb", "target", "state")
    rows = median_rows(of.get(("region",), []), keys, region_columns())
    out += [table(["reader", "width, Mb", "target", "state", *REGION_HEADER], rows), ""]
    out += ["### A matrix of 500 scattered variants (3 seeds)", ""]
    keys = ("reader", "chrom", "target", "state")
    rows = median_rows(of.get(("scattered",), []), keys, list_columns())
    out += [table(["reader", "where", "target", "state", *LIST_HEADER], rows), ""]
    out += ["### Proxies at r² 0.8 within 10 Mb (10 draws of 1 variant, 3 of 10 and of 100)", ""]
    keys = ("reader", "variants", "target", "state")
    rows = median_rows(of.get(("proxies",), []), keys, list_columns())
    out += [table(["reader", "variants", "target", "state", *LIST_HEADER], rows), ""]
    return out


# ------------------------------------------------------------------ the summary


@dataclass(frozen=True)
class Read:
    """One row of a summary: which records, what names a draw, and sum or median."""

    label: str
    keep: Callable[[Record], bool]
    draw: str
    summed: bool


def case_is(case: str, **fields: Any) -> Callable[[Record], bool]:
    return lambda r: r.get("case") == case and all(r.get(k) == v for k, v in fields.items())


STEP1_READS = [
    Read("clump, r² 0.01", case_is("clump", r2=0.01), "analysis", True),
    Read("clump, r² 0.001", case_is("clump", r2=0.001), "analysis", True),
    Read("region 3 Mb", case_is("region", width_mb=3), "centre", False),
    Read("region 10 Mb", case_is("region", width_mb=10), "centre", False),
    Read("500 scattered, chr22", case_is("scattered", chrom="22"), "seed", False),
    Read("500 scattered, genome", case_is("scattered", chrom="genome"), "seed", False),
    Read("proxies, 1 variant", case_is("proxies", variants=1), "seed", False),
    Read("proxies, 10 variants", case_is("proxies", variants=10), "seed", False),
    Read("proxies, 100 variants", case_is("proxies", variants=100), "seed", False),
]


def aggregate(found: list[Record], summed: bool, value: Callable[[Record], float]) -> float:
    return sum(value(r) for r in found) if summed else median_of(found, value)


def summary_row(read: Read, label: str, by_target: dict[str, list[Record]]) -> list[str]:
    """Each target's time over the draws given, and the bytes and requests of the first."""
    cells = [
        seconds(aggregate(by_target[t], read.summed, lambda r: r["seconds"]))
        if by_target.get(t)
        else "—"
        for t in TARGET_ORDER
    ]
    first = next(by_target[t] for t in TARGET_ORDER if by_target.get(t))
    data = aggregate(first, read.summed, lambda r: wire_or_client(r, "bytes"))
    requests = aggregate(first, read.summed, lambda r: wire_or_client(r, "requests"))
    return [label, *cells, megabytes(data), f"{requests:,.0f}"]


def by_target(records: list[Record]) -> dict[str, list[Record]]:
    found: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        found[r["target"]].append(r)
    return found


def only(found: dict[str, list[Record]], draw: str, kept: set[Any]) -> dict[str, list[Record]]:
    """Each target's records of the draws `kept`."""
    return {t: [r for r in records if r[draw] in kept] for t, records in found.items()}


def summary_rows(read: Read, state: str, records: list[Record]) -> list[list[str]]:
    """The read's row over every draw, for the targets that timed them all; and, when the
    slower targets timed fewer, a row over the draws every target timed."""
    found = by_target(records)
    draws = {t: {r[read.draw] for r in each} for t, each in found.items()}
    every = set().union(*draws.values())
    common = set.intersection(*draws.values())
    how = "sum" if read.summed else "median"
    complete = {t: each for t, each in found.items() if draws[t] == every}
    rows = [summary_row(read, f"{read.label}, {state}, {how} of {len(every)}", complete)]
    if common and common != every:
        label = f"{read.label}, {state}, {how} of {len(common)}"
        rows.append(summary_row(read, label, only(found, read.draw, common)))
    return rows


def read_rows(read: Read, records: list[Record]) -> list[list[str]]:
    rows: list[list[str]] = []
    for state in STATES:
        in_state = [r for r in records if r["state"] == state]
        if in_state:
            rows += summary_rows(read, state, in_state)
    return rows


def summary(records: list[Record], reads: list[Read], reader: str) -> list[str]:
    mine = [r for r in records if r["reader"] == reader]
    rows = [row for read in reads for row in read_rows(read, list(filter(read.keep, mine)))]
    header = ["read", *TARGET_ORDER, "MB read", "requests"]
    return [table(header, rows), ""]


# ------------------------------------------------------------------ step 2


def layout_of(record: Record) -> str:
    return str(record["store"]).removeprefix("chr22-").removesuffix(".store")


def builds(directory: Path | None) -> list[str]:
    if directory is None:
        return []
    rows = []
    for path in sorted(directory.glob("chr22-*.time.json")):
        b = json.loads(path.read_text())
        rows.append(
            [
                b["layout"],
                f"{b['near_r'][0]:,}",
                f"{b['near_r'][1]:,}",
                f"{b['near_r_bytes']:,}",
                str(b["near_r_files"]),
                f"{b['store_bytes']:,}",
                seconds(b["wall_seconds"]),
                seconds(b["cpu_seconds"]),
                f"{b['peak_rss_bytes'] / 2**30:.2f}",
                f"{b['load_before'][0]:.0f}–{b['load_after'][0]:.0f}",
            ]
        )
    header = [
        "layout",
        "inner chunk",
        "shard",
        "`near/22/r` bytes",
        "shard files",
        "store bytes",
        "build wall",
        "build CPU",
        "peak RSS, GiB",
        "load",
    ]
    return ["### Size on disk and build, chr22", "", table(header, rows), ""]


def layout_cell(found: list[Record], summed: bool) -> str:
    if not found:
        return "—"
    if summed:
        total = sum(r["seconds"] for r in found)
        data = sum(wire_or_client(r, "bytes") for r in found)
        requests = sum(wire_or_client(r, "requests") for r in found)
    else:
        total = median_of(found, lambda r: r["seconds"])
        data = median_of(found, lambda r: wire_or_client(r, "bytes"))
        requests = median_of(found, lambda r: wire_or_client(r, "requests"))
    return f"{seconds(total)} · {megabytes(data)} MB · {requests:,.0f} req"


def step2_case(records: list[Record], name: str, summed: bool) -> list[list[str]]:
    rows = []
    groups = grouped(records, "reader", "target", "state")
    for (reader, target, state), found in sorted(
        groups.items(), key=lambda kv: (kv[0][0], order_key(kv[0][1]), STATES.index(kv[0][2]))
    ):
        by_layout = defaultdict(list)
        for r in found:
            by_layout[layout_of(r)].append(r)
        cells = [layout_cell(by_layout.get(x, []), summed) for x in ("A", "B", "C")]
        rows.append([name, reader, target, state, *cells, load_range(found)])
    return rows


def step2(records: list[Record], layouts: Path | None) -> list[str]:
    out = ["## Step 2: EUR chr22 in layouts A, B and C (× 511, k = 10)", "", *builds(layouts)]
    of = grouped(records, "case")
    rows: list[list[str]] = []
    for r2 in (0.01, 0.001):
        clumps = [r for r in of.get(("clump",), []) if r["r2"] == r2]
        rows += step2_case(clumps, f"clump chr22 hits, r² {r2:g} (sum)", summed=True)
    rows += step2_case(of.get(("region",), []), "3 Mb region (median of 3)", summed=False)
    rows += step2_case(of.get(("scattered",), []), "500 scattered (median of 3)", summed=False)
    rows += step2_case(of.get(("pair",), []), "single pair (median)", summed=False)
    header = ["read", "reader", "target", "state", "A", "B", "C", "load"]
    out += ["### Reads, each cell time · MB read · requests", "", table(header, rows), ""]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--layouts", type=Path, help="build_layouts.py's output directory")
    args = parser.parse_args()
    timings, environments = read(args.results)
    by_step = grouped(timings, "step")
    lines = ["# LD Store read latency: tables (ogld#8)", ""]
    lines += ["Generated by `scripts/bench/report.py` from `docs/benchmarks/results/`.", ""]
    lines += [f"Runs: {len(environments)}; records: {len(timings)}.", ""]
    lines += step1(by_step.get(("step1",), []))
    lines += step2(by_step.get(("step2",), []), args.layouts)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
