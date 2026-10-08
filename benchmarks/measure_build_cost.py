#!/usr/bin/env python3
"""Build each #249 pilot with the 0.1.0 and the 0.2.0 builders, and record the cost.

ADR 0041's rule is that a rebuild from source, not a stamp or a conversion, is
what proves a builder; this harness measures what that rebuild costs. For each
pilot it runs the **same argv** once under the pre-upgrade checkout (Zarr v2,
format 0.1.0) and once under the 0.2.0 checkout (Zarr v3 sharded), under
`/usr/bin/time -v`, and records wall time, peak RSS and the 1-minute load before
and after. A build starts only while the 1-minute load is below `--max-load`, so
another worker's heavy job cannot masquerade as the cost of the change.

Each side is a **command prefix** plus a working directory, so the harness does
not know or care how a checkout is entered:

  pixi run -e dev python benchmarks/measure_build_cost.py \
      --head-cwd .                      --head-cmd 'pixi run -e dev opengwasdb' \
      --base-cwd /tmp/.../base-src      --base-cmd 'pixi run -e dev opengwasdb' \
      --work /data/opengwasdb/work/epic240/249 \
      --pilot dense --pilot ragged-besd --pilot ragged-ssf --pilot hybrid \
      --output docs/benchmark-output/opengwasdb_build_cost_epic240_249.json

`--pilot` is repeatable; `--order base,head` (default: alternate per pilot)
decides which side runs first. The output is a JSON artifact recording every
argv, cwd, wall second, peak RSS in **KiB** (as `/usr/bin/time -v` reports it)
and GiB, the **guard environment** each step inherited, and the load window.
`--merge` combines run artifacts (in run order) into one committed file, so the
artifact is the harness's own output rather than hand-assembled. Re-run it; never
hand-edit the numbers.

The documented #249 commands:

  pixi run -e dev python benchmarks/measure_build_cost.py --work W \
      --pilot dense --pilot ragged-besd --pilot ragged-ssf --output /tmp/small.json
  pixi run -e dev python benchmarks/measure_build_cost.py --work W \
      --pilot hybrid --output /tmp/hybrid.json
  pixi run -e dev python benchmarks/measure_build_cost.py --merge /tmp/small.json \
      --merge /tmp/hybrid.json --output docs/benchmark-output/build_cost.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks._artifact import commit

TIME_BIN = "/usr/bin/time"
KIB_PER_GIB = 1024 * 1024
#: The guard switch the harness inherits; recorded per step so an artifact says
#: which configuration produced each number (unset = the production path).
GUARD_ENV = "OPEN_GWASDB_REQUIRE_WHOLE_SHARD_WRITES"
#: Constant notes every run records, so a merged artifact is reproducible from the
#: documented commands rather than annotated by hand.
PRODUCTION_CONFIG_NOTE = (
    "Every build inherits this process's environment; `guard_env` on each step is "
    f"{GUARD_ENV} at run time (null = unset = the production configuration)."
)
TIMING_NOTE = (
    "Each build starts only while the 1-minute load is below `load_threshold_1m` "
    "(`load_1m_before`); the end load can be higher from other tenants. Wall time and "
    "peak RSS are `/usr/bin/time -v` on the CLI process: `Maximum resident set size` "
    "is the largest single process in the pixi -> python -> forked-worker tree, not "
    "their sum, and is reported in KiB and GiB. Re-run with "
    "benchmarks/measure_build_cost.py; never hand-edit."
)
_ARTIFACT_KEYS = (
    "artifact",
    "commit",
    "base_cwd",
    "base_revision",
    "head_cwd",
    "head_revision",
    "load_threshold_1m",
    "n_workers_note",
    "production_config",
    "timing_note",
)
_ELAPSED = re.compile(r"Elapsed \(wall clock\) time.*?\):\s*([0-9:.]+)")
_MAXRSS = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")

#: The analyses manifests the registered releases were built from. Every one is
#: read-only and outside this worktree.
OGS = Path("/data/opengwasdb/stores")


def _dense_argv(out: Path) -> list[str]:
    return [
        "build-dense-vcf",
        str(OGS / "OGS-00008/work/analyses.tsv"),
        str(out),
        "--store-id",
        "OGS-00008",
        "--release-id",
        "OGS-00008",
        "--source-reader-capability",
        "opengwasdb.gwas-vcf",
        "--source-assembly",
        "hg19",
        "--n-workers",
        "4",
    ]


def _ragged_besd_argv(out: Path) -> list[str]:
    return [
        "build-ragged-besd",
        "/data/opengwasdb/raw/eqtlgen/pilot-10",
        str(out),
        "--store-id",
        "OGS-00001",
        "--release-id",
        "OGS-00001",
        "--analyses",
        str(OGS / "OGS-00001/work/analyses.tsv"),
        "--source-build",
        "hg19",
        "--tissue",
        "whole_blood",
    ]


def _ragged_ssf_argv(out: Path) -> list[str]:
    # `source_file` in the manifest is absolute, so the filtered directory only
    # has to exist; it is the same raw tree the registered release read.
    return [
        "build-ragged-ssf",
        str(OGS / "OGS-00007/work/analyses.tsv"),
        "/data/opengwasdb/raw/ebi-sun-pqtl-10/filtered",
        str(out),
        "--store-id",
        "OGS-00007",
        "--release-id",
        "OGS-00007",
        "--stored-effect-scale",
        "sd",
    ]


def _hybrid_argv(out: Path) -> list[str]:
    # OGS-00004's registered argv, from stores/OGS-00004/records/build.json: the
    # two-pass `--reference-panel` form (the single-pass flag is #255, which this
    # checkout does not have), no `--n-workers` (the build's own default).
    return [
        "build-hybrid",
        str(OGS / "OGS-00004/work/analyses.tsv"),
        str(out),
        "--store-id",
        "OGS-00004",
        "--release-id",
        "OGS-00004",
        "--reference-panel",
        "/data/opengwasdb/reference/alid-panel/EUR-variants.tsv.gz",
        "--source-reader-capability",
        "opengwasdb.gwas-ssf",
        "--source-assembly",
        "hg38",
    ]


#: pilot -> the argv tail a build appends to the side's command prefix.
PILOTS: dict[str, Any] = {
    "dense": _dense_argv,
    "ragged-besd": _ragged_besd_argv,
    "ragged-ssf": _ragged_ssf_argv,
    "hybrid": _hybrid_argv,
}


def parse_time_v(stderr: str) -> tuple[float, int]:
    """Wall seconds and peak RSS (KiB) from `/usr/bin/time -v` output."""
    elapsed = _ELAPSED.search(stderr)
    maxrss = _MAXRSS.search(stderr)
    if elapsed is None or maxrss is None:
        raise ValueError("/usr/bin/time -v did not report elapsed time and max RSS")
    parts = [float(p) for p in elapsed.group(1).split(":")]
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds, int(maxrss.group(1))


def _one_minute_load() -> float:
    """The node's 1-minute load average, the quiet-window signal."""
    return os.getloadavg()[0]


def wait_for_quiet_load(
    max_load: float,
    poll_seconds: float,
    max_wait: float,
    load_fn: Callable[[], float] = _one_minute_load,
) -> float:
    """Block until the 1-minute load is below `max_load`; return the reading.

    Fails loudly at the deadline rather than starting a build on a busy node,
    which would report another job's contention as this change's cost.
    """
    deadline = time.monotonic() + max_wait
    while True:
        load = load_fn()
        if load < max_load:
            return load
        if time.monotonic() >= deadline:
            raise SystemExit(
                f"1-minute load stayed at or above {max_load} for "
                f"{max_wait / 60:.0f} min; not starting a build on a busy node"
            )
        time.sleep(poll_seconds)


def run_step(
    *,
    side: str,
    pilot: str,
    command: list[str],
    cwd: Path,
    out: Path,
    log_path: Path,
    max_load: float,
    poll_seconds: float,
    max_wait: float,
) -> dict[str, Any]:
    """Run one build under `/usr/bin/time -v`, gated on the 1-minute load."""
    start_load = wait_for_quiet_load(max_load, poll_seconds, max_wait)
    argv = [TIME_BIN, "-v", *command]
    started = datetime.now(UTC)
    with log_path.open("wb") as log:
        completed = subprocess.run(argv, cwd=cwd, stdout=log, stderr=log, check=False)
    stderr = log_path.read_text(encoding="utf-8", errors="replace")
    seconds, maxrss_kib = parse_time_v(stderr)
    end_load = os.getloadavg()[0]
    return {
        "pilot": pilot,
        "side": side,
        "cwd": str(cwd),
        "argv": command,
        "store": str(out),
        "log": str(log_path),
        "exit_code": completed.returncode,
        "started_at": started.isoformat(),
        "guard_env": os.environ.get(GUARD_ENV),
        "load_1m_before": round(start_load, 2),
        "load_1m_after": round(end_load, 2),
        "wall_seconds": round(seconds, 2),
        "wall_hms": _hms(seconds),
        "peak_rss_kib": maxrss_kib,
        "peak_rss_gib": round(maxrss_kib / KIB_PER_GIB, 2),
    }


def _hms(seconds: float) -> str:
    whole = int(round(seconds))
    return f"{whole // 3600}:{(whole % 3600) // 60:02d}:{whole % 60:02d}"


def _side_prefix(text: str) -> list[str]:
    parts = shlex.split(text)
    if not parts:
        raise SystemExit("a --base-cmd/--head-cmd prefix cannot be empty")
    return parts


def _checkout_revision(cwd: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def build_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    """The (pilot, side, argv, cwd, store, log) steps, in the run order."""
    sides = {
        "base": (args.base_cmd_list, args.base_cwd),
        "head": (args.head_cmd_list, args.head_cwd),
    }
    order = [side for side in args.order.split(",") if side in args.sides_list]
    unknown = [side for side in args.sides_list if side not in ("base", "head")]
    if unknown or not order:
        raise SystemExit(f"--sides must name base and/or head once each, got {args.sides!r}")
    steps: list[dict[str, Any]] = []
    for index, pilot in enumerate(args.pilot):
        if pilot not in PILOTS:
            raise SystemExit(f"unknown pilot {pilot!r}; choose from {sorted(PILOTS)}")
        pilot_order = order if index % 2 == 0 else list(reversed(order))
        for side in pilot_order:
            prefix, cwd = sides[side]
            out = args.work / f"{pilot}-{side}{args.suffix}.opengwasdb"
            log_path = args.work / f"{pilot}-{side}{args.suffix}.log"
            steps.append(
                {
                    "pilot": pilot,
                    "side": side,
                    "command": [*prefix, *PILOTS[pilot](out)],
                    "cwd": cwd,
                    "store": out,
                    "log": log_path,
                }
            )
    return steps


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-cmd", default="pixi run -e dev opengwasdb")
    parser.add_argument("--head-cmd", default="pixi run -e dev opengwasdb")
    parser.add_argument("--base-cwd", type=Path, default=Path("/tmp/epic240/247/base-src"))
    parser.add_argument("--head-cwd", type=Path, default=Path.cwd())
    parser.add_argument("--work", type=Path, help="where each build's store and log go")
    parser.add_argument("--pilot", action="append", default=[])
    parser.add_argument("--order", default="base,head")
    parser.add_argument(
        "--sides",
        default="base,head",
        help="which sides to build; `head` alone runs the guard-on proof builds",
    )
    parser.add_argument(
        "--suffix", default="", help="appended to each store/log name, e.g. -guard"
    )
    parser.add_argument("--max-load", type=float, default=3.0)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--max-wait-min", type=float, default=120.0)
    parser.add_argument(
        "--merge",
        type=Path,
        action="append",
        default=[],
        help="merge these run artifacts, in order, into --output; builds nothing",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def merge_artifacts(paths: list[Path]) -> dict[str, Any]:
    """Combine run artifacts, in run order, through the harness itself.

    No step is reordered or renamed: the merged artifact's step list is the runs'
    concatenation, which is the order the documented commands run them in.  The
    base and head revisions and the commit must agree, so a merge cannot mix two
    code versions into one before/after table; the same pilot and side must not
    appear twice, and every step must have the guard unset, because the merged
    file is the timed, production-configuration artifact (the guard-on proof
    builds are a separate run).
    """
    merged: dict[str, Any] | None = None
    runs: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        missing_notes = [
            key for key in ("production_config", "timing_note") if key not in payload
        ]
        if missing_notes:
            raise SystemExit(
                f"{path}: no {missing_notes}; a run from before the harness recorded "
                "them cannot be merged, so only a fresh re-run can be"
            )
        if merged is None:
            merged = {key: payload[key] for key in _ARTIFACT_KEYS if key in payload}
            merged["steps"] = []
            merged["runs"] = []
        for key in ("commit", "base_revision", "head_revision"):
            if payload.get(key) != merged.get(key):
                raise SystemExit(
                    f"{path}: {key} is {payload.get(key)!r}, the first run's is "
                    f"{merged.get(key)!r}; refusing to merge two code versions"
                )
        merged["steps"].extend(payload["steps"])
        runs.append(
            {
                "path": str(path),
                "measured_at": payload.get("measured_at"),
                "n_steps": len(payload["steps"]),
            }
        )
    if merged is None:
        raise SystemExit("--merge needs at least one artifact")
    _refuse_a_bad_timed_merge(merged["steps"])
    merged["runs"] = runs
    merged["measured_at"] = datetime.now(UTC).isoformat()
    return merged


def _refuse_a_bad_timed_merge(steps: list[dict[str, Any]]) -> None:
    """A timed before/after has one step per (pilot, side), no guard, and a state."""
    pairs = [(step["pilot"], step["side"]) for step in steps]
    duplicates = sorted({pair for pair in pairs if pairs.count(pair) > 1})
    if duplicates:
        raise SystemExit(f"duplicate pilot/side step(s) in the merge: {duplicates}")
    unstated = [pair for pair, step in zip(pairs, steps, strict=True) if "guard_env" not in step]
    if unstated:
        raise SystemExit(
            f"step(s) {unstated} record no {GUARD_ENV}; a missing key is not "
            "'unset', so only a fresh re-run can be merged"
        )
    guarded = [step for step in steps if step["guard_env"] is not None]
    if guarded:
        raise SystemExit(
            f"{len(guarded)} step(s) ran with {GUARD_ENV} set; the merged artifact is "
            "the timed production-configuration one, so the guard must be unset"
        )


def main() -> int:
    args = _parser().parse_args()
    if args.merge:
        artifact = merge_artifacts(args.merge)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.output} ({len(artifact['steps'])} steps from {len(args.merge)} runs)")
        return 0
    if args.work is None:
        raise SystemExit("--work is required unless --merge is given")
    args.pilot = args.pilot or list(PILOTS)
    args.work.mkdir(parents=True, exist_ok=True)
    args.base_cmd_list = _side_prefix(args.base_cmd)
    args.head_cmd_list = _side_prefix(args.head_cmd)
    args.sides_list = [side for side in args.sides.split(",") if side]
    steps = build_plan(args)
    results = [
        run_step(
            side=step["side"],
            pilot=step["pilot"],
            command=step["command"],
            cwd=step["cwd"],
            out=step["store"],
            log_path=step["log"],
            max_load=args.max_load,
            poll_seconds=args.poll_seconds,
            max_wait=args.max_wait_min * 60,
        )
        for step in steps
    ]
    artifact = {
        "artifact": "opengwasdb build cost, epic #240 / #249",
        "commit": commit(),
        "measured_at": datetime.now(UTC).isoformat(),
        "base_cwd": str(args.base_cwd),
        "base_revision": _checkout_revision(args.base_cwd),
        "head_cwd": str(args.head_cwd),
        "head_revision": _checkout_revision(args.head_cwd),
        "load_threshold_1m": args.max_load,
        "n_workers_note": (
            "dense uses --n-workers 4 on both sides; ragged and hybrid use the "
            "builder's own default, which the registered OGS-00004 argv also left unset"
        ),
        "production_config": PRODUCTION_CONFIG_NOTE,
        "timing_note": TIMING_NOTE,
        "steps": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    failed = [step for step in results if step["exit_code"] != 0]
    if failed:
        print(f"{len(failed)} build(s) exited non-zero; see the log paths", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
