#!/usr/bin/env python3
"""Wall time of the top-hit index rebuild, the format-3.0 pass #144 left untimed.

Issue #144 asked for the fit, the candidate measurement, the rewrite **and the
top-hit index rebuild** charged separately. The first three were measured on a
column slice of the FinnGen R13 pilot
(`benchmarks/measure_se_migration_phases.py`); the rebuild was declared out of
scope there because it is a store-level pass, not a pass over the `se` plane.

That left the migration's headline unexplained. A full FinnGen R13 pilot-20
migration took 3,863 s over 424,612,300 cells. Scaling the slice's 195.8 s of
plane passes to the full plane accounts for somewhere between ~420 s (if
`rewrite.exceptions` is per-row-chunk and roughly independent of the column
count) and ~980 s (if every phase is linear in cells). The remainder --
2,900-3,400 s, or 75-89% of the migration -- is unaccounted, and the rebuild is
the only pass left to hold it. This measures it instead of inferring it.

The rebuild is charged in two phases by `build_top_hit_indexes` itself:
`top_hits.scan` (decode every `z` cell through the store's codec, keep the
candidates, and decode their `se`/`eaf`) and `top_hits.write` (rank and write
the tiers). Measuring through the production function, rather than
reimplementing its body here, is deliberate: a benchmark that reimplements what
it measures drifts away from it silently.

Runs on a reflinked copy, so the source release is untouched.

Usage:

    measure_top_hit_rebuild.py STORE --into COPY
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import zarr

from opengwasdb.encoding.timing import PhaseTimer
from opengwasdb.layouts.dense.top_hits import build_top_hit_indexes
from opengwasdb.model.manifest import StoreManifest

DEFAULT_OUTPUT = Path("docs/benchmark-output/opengwasdb_top_hit_rebuild.json")


def _commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def _reflink_copy(source: Path, destination: Path) -> None:
    """Share extents where the filesystem can, so the copy is near-free."""
    if destination.exists():
        raise SystemExit(f"{destination}: already exists; refusing to overwrite")
    subprocess.run(["cp", "-a", "--reflink=auto", str(source), str(destination)], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("store", type=Path)
    parser.add_argument("--into", type=Path, required=True, help="reflink here and rebuild there")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    print(f"Copying {args.store} -> {args.into}", flush=True)
    _reflink_copy(args.store, args.into)

    manifest = StoreManifest.load(args.into)
    root = zarr.open_group(str(args.into / "data.zarr"), mode="r")
    rows, analyses = (int(n) for n in root["z"].shape)

    timer = PhaseTimer()
    started = time.perf_counter()
    build_top_hit_indexes(args.into, encoding=manifest.encoding, timer=timer)
    wall = time.perf_counter() - started

    print(f"Top-hit rebuild: {wall:.1f}s", flush=True)
    print(timer.format_report(), flush=True)

    result = {
        "issue": "144",
        "store": str(args.store),
        "measured_on_copy": str(args.into),
        "store_format": manifest.format_version,
        "encoding": manifest.encoding.to_manifest(),
        "commit": _commit(),
        "measured_at": datetime.now(UTC).isoformat(),
        "rows": rows,
        "analyses": analyses,
        "cells": rows * analyses,
        "wall_seconds": round(wall, 3),
        "phases": [
            {"phase": name, "seconds": round(seconds, 3), "share": round(share, 5)}
            for name, seconds, share in timer.report()
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
