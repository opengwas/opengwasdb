#!/usr/bin/env python3
"""Run one #252 Hybrid extra shape in one column, recording the environment.

The seven #242 shapes anchor their PheWAS on an on-panel variant, so they never
exercise the Ragged Overflow's PheWAS path; this runs the two shapes #252 added
(`phewas_off_axis`, `bulk_overflow_heavy`) through #252's own one-shape probe and
writes one JSON line per repetition, each carrying the interpreter, the Zarr
version, the `opengwasdb` path and fingerprint, the commit, and the sha256 of the
probe and this runner. That is what lets an artifact say which code and which
environment produced column `a-2.18-0.1.0` (#250 review r2, major 3).

Run one column per invocation, from whichever environment that column needs:

    # this code, Zarr 3 (run from the worktree)
    pixi run -e dev python benchmarks/ogs00011_extra_shapes.py \
        --column b-code-0.1.0 --store /data/opengwasdb/stores/OGS-00011/store.opengwasdb \
        --shape phewas_off_axis --reps 5 \
        --output /tmp/epic240/250/extra-shapes.jsonl

    # 745796c's package under Zarr 2.18, layered ahead of this tree
    PYTHONPATH=/tmp/epic240/250/base-layered:$PWD \
        /tmp/epic240/244/base-src/.pixi/envs/dev/bin/python \
        benchmarks/ogs00011_extra_shapes.py \
        --column a-2.18-0.1.0 --store ... --shape phewas_off_axis --reps 5 --output ...

Aggregate with `scripts/build_ogs00011_extra_shapes.py`.
"""

from __future__ import annotations

# The runner records the environment it ran in, so it needs the interpreter and
# host beside the probe's numbers.
import argparse
import hashlib
import json
import os
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path

import zarr

from benchmarks import ogs00011_ab
from benchmarks._artifact import commit, package_fingerprint
from benchmarks._quiet import wait_for_quiet


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _environment() -> dict:
    path, fingerprint = package_fingerprint()
    probe = Path(ogs00011_ab.__file__).resolve()
    runner = Path(__file__).resolve()
    return {
        "python": sys.executable,
        "zarr": zarr.__version__,
        "hostname": socket.gethostname(),
        "opengwasdb_path": path,
        "opengwasdb_fingerprint": fingerprint,
        "commit": commit(),
        "probe_path": str(probe),
        "probe_sha256": _sha256(probe),
        "runner_path": str(runner),
        "runner_sha256": _sha256(runner),
        "measured_at": datetime.now(UTC).isoformat(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", required=True)
    ap.add_argument("--shape", required=True)
    ap.add_argument("--column", required=True, help="the label this run writes into every record")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--limit", type=float, default=1500.0)
    ap.add_argument("--max-start-load", type=float, default=3.0)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    waited, gave_up = wait_for_quiet(args.max_start_load)
    print(
        f"{args.column}/{args.shape}: waited {waited:.0f}s for load < {args.max_start_load} "
        f"({os.getloadavg()[0]:.2f}); gave_up={gave_up}",
        flush=True,
    )
    environment = _environment() | {
        "column": args.column,
        "waited_for_load_s": round(waited, 1),
        "waited_timed_out": gave_up,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        for rep in range(1, args.reps + 1):
            record = ogs00011_ab.measure_one_shape(args.store, args.shape, args.limit) | environment
            record["rep"] = rep
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(f"  rep {rep}: {record['elapsed_ms']:.0f} ms", flush=True)


if __name__ == "__main__":
    main()
