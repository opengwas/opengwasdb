#!/usr/bin/env python3
"""One command that reproduces epic #307's timing evidence.

The epic's tables were assembled by hand from step records, phase lines, live
sampling and two one-off reader timings. This harness derives all of it again
from the same inputs and writes one JSON artifact holding:

* every step of the Stores named with `--store`, with its wall time, its
  `log_phase` phases (start, end, span, and the minute-rounded `done in` value
  beside them), its progress phases with the pace each achieved **outside** the
  nightly I/O window, and the count of lines that looked like a phase or
  progress line and did not parse;
* the reader A/B (`--source`): the current reader, a control reader exported
  with `git archive` from `--control-revision` (the epic measured at
  `d405de0`, and both readers changed after it), and the parse floor --
  interleaved, in fresh interpreters, an even number of rounds of at least six
  so that head and control each lead half of them;
* the comparison: every figure in the epic's tables beside its measured value,
  whether the record reproduces it, and the reason when it does not.

Windows are compared on the log lines' own clock. `--window START/END` is one
interval; `--daily-window HH:MM-HH:MM --tz ZONE --until DATE` repeats the
nightly snapshot over every night the records span, which is what keeps an
overnight phase from being reported as the step's clean rate. The daily window
is an assumption read off 22 Oct's observation, so `--daily-from` starts it
on the first October night and the September build is not flagged by a window
nobody observed; the artifact says so.

Usage (the sources are read-only and both sides are I/O bound, so take the
host's shared heavy-job lock around it):

    flock -w 14400 <lock> pixi run -e dev python \\
        benchmarks/measure_epic307_evidence.py \\
        --store OGS-00009=/data/opengwasdb/stores/OGS-00009 \\
        --store OGS-00016=/data/opengwasdb/stores/OGS-00016 \\
        --store OGS-00017=/data/opengwasdb/stores/OGS-00017 \\
        --store OGS-00012=/data/opengwasdb/stores/OGS-00012 \\
        --store OGS-00013=/data/opengwasdb/stores/OGS-00013 \\
        --store OGS-00009-sept=/data/opengwasdb/stores/OGS-00009/replaced/20261021T223815Z \\
        --store OGS-00010-retired=/data/opengwasdb/work/retired/OGS-00010 \\
        --window 2026-10-22T01:06/2026-10-22T16:10 \\
        --daily-window 01:06-16:10 --tz Europe/London \\
        --daily-from 2026-10-14 --until 2026-11-02 \\
        --source opengwasdb.finngen-r13=/data/.../finngen_R13_AB1_ACTINOMYCOSIS.gz \\
        --source opengwasdb.gwas-vcf=/data/opengwasdb/raw/gwas-vcf/ukb-b/ukb-b-10001.vcf.gz \\
        --control-revision d405de0 --repetitions 6 \\
        --output docs/benchmark-output/opengwasdb_epic307_evidence.json

The stores and sources are read, never written. Re-run it; never hand-edit a
number.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

from benchmarks import _reader_ab
from benchmarks._artifact import (
    add_labelled_store_option,
    export_revision,
    labelled_stores,
    provenance,
    write_artifact,
)
from benchmarks._epic307_figures import compare, evidence
from benchmarks._step_records import (
    Window,
    daily_windows,
    first_date,
    measure_store,
    parse_window,
    store_span,
)

DEFAULT_OUTPUT = Path("docs/benchmark-output/opengwasdb_epic307_evidence.json")

#: The daily window's basis, recorded in the artifact so the assumption travels
#: with the numbers it flags.
DAILY_WINDOW_BASIS = (
    "assumed to repeat nightly from the 22 Oct observation (about 01:06 to about 16:10 BST, "
    "judging by when Pass 2 slowed and when the z band-write recovered), applied back to "
    "--daily-from because OGS-00016's October build slows from its first night in the same "
    "hours; September's records are not covered, the epic having read them as clean. The "
    "backup stopped covering /data/opengwasdb on 2 Nov 2026, which is why --until is "
    "exclusive there"
)

STORE_NOTE = (
    "These are timing baselines: each Store's steps, phases and clean rates, re-derived from "
    "its own records. The store bytes are not measured, and a Store re-encoded after its build "
    "still has the build's records."
)


def _clock(value: str) -> date:
    return date.fromisoformat(value)


REPO = Path(__file__).resolve().parents[1]


def _control(revision: str | None, into: Path) -> dict[str, Any] | None:
    """The tree the `reader_base` side imports: `git archive` of the revision.

    The fingerprint is the archive's, computed here from the commit, not read
    off whatever tree a person left on disk.
    """
    if revision is None:
        return None
    return export_revision(revision, into / "control", repo=REPO)


def _windows(
    args: argparse.Namespace, stores: Sequence[dict[str, Any]], until: date | None
) -> list[Window]:
    """The one-off windows plus one window per night of the records' span."""
    windows = [parse_window(text) for text in args.window]
    if args.daily_window:
        if not args.tz or until is None:
            raise SystemExit("--daily-window needs --tz and --until")
        start = first_date(record for store in stores for record in store["records"])
        if start is None:
            raise SystemExit("--daily-window needs at least one record with a timestamp")
        if start >= until:
            raise SystemExit(f"--until {until} is not after the first record's date {start}")
        if args.daily_from:
            start = _clock(args.daily_from)
        if start >= until:
            raise SystemExit(f"--until {until} is not after the first night, {start}")
        windows += daily_windows(args.daily_window, args.tz, first=start, until=until)
    return windows


def _host() -> dict[str, Any]:
    """The load and uptime readings the finish rules ask every cited timing for."""
    load = os.getloadavg()
    uptime = Path("/proc/uptime").read_text().split()[0]
    return {
        "load_1m": round(load[0], 2),
        "load_5m": round(load[1], 2),
        "load_15m": round(load[2], 2),
        "uptime_seconds": float(uptime),
        "hostname": os.uname().nodename,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    add_labelled_store_option(parser, help="a Store Release directory whose records/ to parse")
    parser.add_argument(
        "--window",
        action="append",
        default=[],
        metavar="START/END",
        help="one window in log-clock time, e.g. 2026-10-22T01:06/2026-10-22T16:10; repeatable",
    )
    parser.add_argument(
        "--daily-window",
        metavar="HH:MM-HH:MM",
        help="the nightly window's clock times, repeated over every night the records span",
    )
    parser.add_argument("--tz", help="the zone --daily-window's clock times are read in")
    parser.add_argument(
        "--daily-from",
        help="the first night --daily-window applies to (YYYY-MM-DD); default: the night of the "
        "first record. Name the night the window was observed, so earlier Stores are not flagged",
    )
    parser.add_argument(
        "--until",
        help="the last night to expand --daily-window over, exclusive (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="CAPABILITY=PATH",
        help=f"a source file for the reader A/B, capability one of {sorted(_reader_ab.SOURCES)}",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=_reader_ab.MIN_ROUNDS,
        help="rounds per source: an even number, so head and control each lead half of them",
    )
    parser.add_argument(
        "--source-repetitions",
        action="append",
        default=[],
        metavar="CAPABILITY=N",
        help="rounds for one source, when its samples are cheap enough to afford more",
    )
    parser.add_argument(
        "--control-revision",
        help="a commit of this repository to export with `git archive` as the control "
        "reader_base imports (the epic's d405de0)",
    )
    return parser


def _child_parser() -> argparse.ArgumentParser:
    """The one measurement a child interpreter runs, and nothing else."""
    parser = argparse.ArgumentParser(description="one reader A/B sample")
    parser.add_argument("--capability", required=True)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--side", required=True)
    parser.add_argument(
        "--module-path",
        type=Path,
        help="a tree to import opengwasdb from, ahead of this checkout",
    )
    return parser


def _committed_artifact(output: Path) -> bool:
    """Whether `HEAD` already holds a file at the artifact's path.

    The path is resolved first, so an absolute `--output` and a relative one
    are the same question, and a path outside this work tree is not held by it.
    A run outside a git work tree has no commit to name and is refused.

    The artifact names the commit it was made at, so it has to be produced at a
    commit that does not yet contain it: re-running at the artifact's own commit
    would record a revision whose tree holds a *different* artifact, and the
    `git archive <commit>` a reader checks would not be the code that ran. The
    run is refused rather than allowed to overwrite its own provenance.
    """
    top = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=False
    )
    if top.returncode != 0:
        raise SystemExit(
            "the artifact names the commit it was made at, and this is not a git work tree"
        )
    root = Path(top.stdout.strip()).resolve()
    try:
        relative = output.resolve().relative_to(root).as_posix()
    except ValueError:
        return False
    found = subprocess.run(
        ["git", "cat-file", "-e", f"HEAD:{relative}"], cwd=root, capture_output=True, check=False
    )
    return found.returncode == 0


def _child_main(argv: Sequence[str]) -> int:
    args = _child_parser().parse_args(argv)
    _reader_ab.child(args.capability, args.path, args.side, args.module_path)
    return 0


def _sources(entries: list[str]) -> list[tuple[str, Path]]:
    """`CAPABILITY=PATH` entries, parsed by the shared `LABEL=PATH` helper."""
    sources = labelled_stores(entries)
    unknown = [capability for capability, _ in sources if capability not in _reader_ab.SOURCES]
    if unknown:
        raise SystemExit(
            f"unknown Source Reader Capability {unknown}; known: {sorted(_reader_ab.SOURCES)}"
        )
    return sources


def _source_repetitions(entries: Sequence[str]) -> dict[str, int]:
    """`CAPABILITY=N` overrides, refusing a source this run does not measure."""
    rounds: dict[str, int] = {}
    for entry in entries:
        capability, _, count = entry.partition("=")
        if capability not in _reader_ab.SOURCES or not count.isdigit():
            raise SystemExit(f"--source-repetitions needs CAPABILITY=N, got {entry!r}")
        if int(count) < 2:
            raise SystemExit(f"--source-repetitions {capability}: at least two rounds")
        rounds[capability] = int(count)
    return rounds


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "--child":
        return _child_main(raw[1:])
    args = _parser().parse_args(raw)
    if args.repetitions < 2 or args.repetitions % 2:
        raise SystemExit(
            "--repetitions must be an even number of at least 2: the sides are interleaved "
            "and head and control each lead half the rounds"
        )
    if _committed_artifact(args.output):
        raise SystemExit(
            f"{args.output} is already committed; commit the harness first and run from that "
            "commit, so the artifact names a revision that does not already hold it"
        )
    until = _clock(args.until) if args.until else None
    sources = _sources(args.source)
    rounds = _source_repetitions(args.source_repetitions)
    host_start = _host()
    stores = [
        measure_store(path, label=label) for label, path in labelled_stores(args.store)
    ]
    windows = _windows(args, stores, until)
    with tempfile.TemporaryDirectory(prefix="epic307-control-") as scratch:
        control = _control(args.control_revision, Path(scratch))
        reader_ab = [
            _reader_ab.measure_source(
                capability,
                path,
                repetitions=rounds.get(capability, args.repetitions),
                control=control,
            )
            for capability, path in sources
        ]
    spans = [store_span(store["records"]) for store in stores]
    payload = {
        "artifact": "epic #307 evidence: step timings, window flags, reader A/B, epic comparison",
        "generator": "benchmarks/measure_epic307_evidence.py",
        "argv": list(sys.argv[1:] if argv is None else argv),
        **provenance(),
        "host": {
            "start": host_start,
            "end": _host(),
            "note": "the load at the run's start and end; the reader A/B records a load per sample",
        },
        "method_note": (
            "A step's wall_seconds is its record's elapsed_seconds and its phases need not sum "
            "to it. A phase's span comes from its two log lines; the log's own `done in` value "
            "is minute-rounded, so it is recorded beside the span, not instead of it. Windows "
            "are half-open intervals on the log lines' clock (host local time). A phase's clean "
            "rate is pooled over the progress intervals that fall entirely outside every window; "
            "the pace on the last line before a window is recorded beside it."
        ),
        "windows": {
            "one_off": [window.to_dict() for window in windows if window.date is None],
            "daily": (
                None
                if not args.daily_window
                else {
                    "text": args.daily_window,
                    "tz": args.tz,
                    "from": min(window.date for window in windows if window.date),
                    "until": args.until,
                    "basis": DAILY_WINDOW_BASIS,
                    "nights": sum(1 for window in windows if window.date),
                }
            ),
            "intervals": [window.to_dict() for window in windows],
        },
        "control": (
            None
            if control is None
            else {
                "revision": control["revision"],
                "fingerprint": control["opengwasdb_fingerprint"],
                "fingerprint_basis": control["fingerprint_basis"],
            }
        ),
        "stores_note": STORE_NOTE,
        "stores": [
            {
                "label": store["label"],
                "path": store["path"],
                "store_id": store["store_id"],
                "span": _span_dict(spans[index]),
                "steps": [record.to_dict(windows) for record in store["records"]],
            }
            for index, store in enumerate(stores)
        ],
        "reader_ab": reader_ab,
        "comparison": compare(evidence(stores, windows, reader_ab)),
    }
    write_artifact(args.output, payload)
    _summarise(payload)
    return 0


def _span_dict(span: tuple[datetime | None, datetime | None]) -> dict[str, str | None]:
    first, last = span
    return {
        "first": first.isoformat() if first else None,
        "last": last.isoformat() if last else None,
    }


def _summarise(payload: dict[str, Any]) -> None:
    for store in payload["stores"]:
        for step in store["steps"]:
            wall = "no wall time" if step["wall_seconds"] is None else f"{step['wall_seconds']}s"
            unparsed = step["unparsed"]["lines"]
            flag = f", {unparsed} unparsed line(s)" if unparsed else ""
            print(
                f"{store['label']} {step['step']}: {wall}, {len(step['phases'])} phase(s), "
                f"{len(step['progress'])} progress phase(s){flag}"
            )
    for source in payload["reader_ab"]:
        medians = ", ".join(f"{side} {seconds}s" for side, seconds in source["medians"].items())
        print(f"reader A/B {source['label']}: {medians} ({source['n_rows']} rows)")
    figures = payload["comparison"]["figures"]
    counts: dict[str, int] = {}
    for figure in figures:
        counts[figure["verdict"]] = counts.get(figure["verdict"], 0) + 1
    summary = ", ".join(f"{count} {verdict}" for verdict, count in sorted(counts.items()))
    print(f"comparison: {len(figures)} epic figures — {summary}")


if __name__ == "__main__":
    raise SystemExit(main())
