#!/usr/bin/env python3
"""ogld#32's reads against ogld#8's: requests and seconds per read, before and after.

    pixi run -e dev python scripts/bench/deltas.py \
        --before docs/benchmarks/results/step1-*.jsonl.gz \
        --after docs/benchmarks/results/32/step1-*.jsonl.gz \
        --levers docs/benchmarks/results/32/lever-*.jsonl.gz > docs/benchmarks/fast-reader-tables.md

Records are read and aggregated by `report.py`, as ogld#8's tables are: a sum over the Analyses
for clumping, a median over the draws for everything else. A read is compared on the draws both
sides timed at a target, so each cell compares the same Analyses, centres and seeds. "Before"
is ogld#8's reader as it ran there (its 256 MiB cache, not the prototype); "after" is every
lever on. The lever table times each lever alone, the others off (`latency.py --without`).
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import report

Record = dict[str, Any]
#: The levers as `latency.py --without` names them, and what turning one on alone is called.
LEVERS = ("concurrency", "indexes", "count", "cache")


def step1(records: Iterable[Record], label: str) -> list[Record]:
    """The step-1 records of one reader, every lever on or every lever as ogld#8 ran it."""
    return [r for r in records if r["step"] == "step1" and r["reader"] == label]


def common(before: list[Record], after: list[Record], draw: str) -> tuple[list[Record], ...]:
    """Each side's records of the draws both sides timed."""
    both = {r[draw] for r in before} & {r[draw] for r in after}
    return [r for r in before if r[draw] in both], [r for r in after if r[draw] in both]


def cell(read: report.Read, before: list[Record], after: list[Record]) -> str:
    """Seconds before → after, and how many times faster, over the draws both timed."""
    if not before or not after:
        return "—"
    old = report.aggregate(before, read.summed, lambda r: r["seconds"])
    new = report.aggregate(after, read.summed, lambda r: r["seconds"])
    return f"{report.seconds(old)} → {report.seconds(new)} ({old / new:.1f}×)"


def requests(read: report.Read, records: list[Record]) -> str:
    """The read's requests as the first target with records counted them."""
    found = report.by_target(records)
    first = next((found[t] for t in report.TARGET_ORDER if found.get(t)), [])
    if not first:
        return "—"
    counted = report.aggregate(first, read.summed, lambda r: report.wire_or_client(r, "requests"))
    return f"{counted:,.0f}"


def at(records: list[Record], target: str) -> list[Record]:
    return [r for r in records if r["target"] == target]


def delta_row(read: report.Read, state: str, old: list[Record], new: list[Record]) -> list[str]:
    """One read in one state: the requests of the first target both timed, and each target's
    seconds before and after, over the draws both timed there; empty if no target has both."""
    pairs = [common(at(old, t), at(new, t), read.draw) for t in ("local", "http10", "http50")]
    cells = [cell(read, b, a) for b, a in pairs]
    counted = next(((b, a) for b, a in pairs if b), None)
    if counted is None:
        return []
    how = "sum" if read.summed else "median"
    draws = len({r[read.draw] for r in counted[1]})
    change = f"{requests(read, counted[0])} → {requests(read, counted[1])}"
    return [f"{read.label}, {state}, {how} of {draws}", change, *cells]


def in_state(records: list[Record], read: report.Read, state: str) -> list[Record]:
    return [r for r in records if read.keep(r) and r["state"] == state]


def delta_rows(before: list[Record], after: list[Record]) -> list[list[str]]:
    """One row per read and state: requests and seconds per target, before and after."""
    rows = (
        delta_row(read, state, in_state(before, read, state), in_state(after, read, state))
        for read in report.STEP1_READS
        for state in report.STATES
    )
    return [row for row in rows if row]


def lever_label(record: Record) -> str:
    off = set(record.get("without", []))
    on = [lever for lever in LEVERS if lever not in off]
    return {0: "none (ogld#8's reader)", len(LEVERS): "all four"}.get(len(on), " + ".join(on))


LEVER_READS = [
    report.Read("500 scattered, chr22", report.case_is("scattered", chrom="22"), "seed", False),
    report.Read(
        "500 scattered, genome", report.case_is("scattered", chrom="genome"), "seed", False
    ),
    report.Read("clump GCST90319613, r² 0.01", report.case_is("clump", r2=0.01), "analysis", True),
]


def lever_rows(records: list[Record]) -> list[list[str]]:
    """One row per lever setting, read and state: requests, bytes and seconds at 10 ms."""
    rows = []
    order = {lever_label({"without": list(LEVERS)}): -1, lever_label({}): 9}
    for read in LEVER_READS:
        for state in report.STATES:
            found = [r for r in records if read.keep(r) and r["state"] == state]
            groups = report.grouped(found, "lever")
            for (lever,), each in sorted(groups.items(), key=lambda g: order.get(g[0][0], 0)):
                time = report.aggregate(each, read.summed, lambda r: r["seconds"])
                data = report.aggregate(
                    each, read.summed, lambda r: report.wire_or_client(r, "bytes")
                )
                rows.append(
                    [
                        f"{read.label}, {state}",
                        lever,
                        requests(read, each),
                        report.megabytes(data),
                        report.seconds(time),
                        report.load_range(each),
                    ]
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--before", nargs="+", type=Path, required=True)
    parser.add_argument("--after", nargs="+", type=Path, required=True)
    parser.add_argument("--levers", nargs="+", type=Path, default=[])
    args = parser.parse_args()
    before = step1(report.read(args.before)[0], "ogld")
    after = [r for r in report.read(args.after)[0] if not r.get("without")]
    after = step1(after, after[0]["reader"])  # as `report.read` labelled it
    out = [
        "# The fast reader's tables (ogld#32)",
        "",
        "_Rendered by `scripts/bench/deltas.py` from `results/32/` and ogld#8's step-1 records in",
        "`results/`; see [`fast-reader.md`](fast-reader.md). Seconds before → after, and how many",
        "times faster._",
        "",
        "## ogld#8's step-1 reads, before and after ogld#32",
        "",
    ]
    header = ["read", "requests", "local", "+10 ms", "+50 ms"]
    out += [report.table(header, delta_rows(before, after)), ""]
    if args.levers:
        # Each file is one lever setting; read alone, so `report.read`'s rule that a later run
        # supersedes an earlier one's same read does not drop the other settings.
        levers = [
            {**r, "lever": lever_label(r)} for path in args.levers for r in report.read([path])[0]
        ]
        out += ["## Each lever alone, at +10 ms", ""]
        header = ["read", "levers on", "requests", "MB read", "seconds", "load"]
        out += [report.table(header, lever_rows(levers)), ""]
    print("\n".join(out))


if __name__ == "__main__":
    main()
