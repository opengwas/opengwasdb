#!/usr/bin/env python3
"""Interleaved before/after A/B for the OGS-00011 variant-side query shapes (#252).

Shape by shape: run the shape on the *before* tree, then on the *after* tree,
back to back, so load drift hits both sides equally. Each side runs in a fresh
interpreter that imports its own tree's `opengwasdb`, so the code measured is
the tree's, and records elapsed, peak RSS (sampled on a background thread, as
`benchmarks/_rss.py` does), the 1/5/15-minute loads at the run's start and end,
a sha256 of the six returned arrays, and whether the per-shape time limit was
hit. The same script, in `--identity` mode, hashes a lighter public-query set on
one tree so a caller can compare two trees' spot answers.

This is the committed entry point that produced
`docs/benchmark-output/opengwasdb_ogs00011_hybrid_252_ab.json` and
`docs/benchmark-output/opengwasdb_252_spot_identity.json`. The before tree was
the branch base `f168ef1` checked out detached; the after tree was this branch's
HEAD. Both trees' commit and `opengwasdb` source fingerprint are recorded in the
artifact, so a number cannot be attributed to code that did not produce it.

Usage (driver, from the after tree; `--before-tree` is a read-only checkout of
the base):

  pixi run -e dev python benchmarks/ogs00011_ab.py \
      --store /data/opengwasdb/stores/OGS-00011/store.opengwasdb \
      --before-tree /tmp/252-before --after-tree "$PWD" \
      --limit 300 --output /tmp/epic240/252/ogs00011_ab.json

Internal modes (`--one-shape`, `--one-identity`) are how the driver re-invokes
itself in a tree; they print one JSON record and are not meant to be called by
hand.
"""

from __future__ import annotations

# Driver and one-shape probe for the OGS-00011 before/after A/B.
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

#: The order the two trees are run in for every shape.
SIDES = ("before", "after")
SHAPES = [
    "bulk",
    "phewas",
    "regional",
    "regional_one_analysis",
    "tophits",
    "random_lookup_10_variants_100_analyses",
    "random_lookup_100_variants_10_analyses",
    "phewas_off_axis",
    "bulk_overflow_heavy",
]
REGION_SHAPES = ("regional", "regional_one_analysis")


class _Timeout(Exception):
    pass


def _raise(_signum, _frame):
    raise _Timeout


def _loads() -> list[float]:
    with open("/proc/loadavg") as fh:
        parts = fh.read().split()
    return [float(parts[0]), float(parts[1]), float(parts[2])]


def _digest(result: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for key in ("variant_index", "analysis_index", "z", "se", "eaf"):
        arr = np.ascontiguousarray(result[key])
        h.update(key.encode())
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    status = np.asarray(result["association_status"], dtype=object)
    h.update(b"association_status")
    h.update(str(len(status)).encode())
    for value in status:
        h.update(str(value).encode())
        h.update(b"\x00")
    return h.hexdigest()


def _tree_on_path(tree: Path) -> None:
    sys.path.insert(0, str(tree))


def _one_shape(args: argparse.Namespace) -> None:
    """Run one shape once with an alarm and an RSS sampler; print one JSON line."""
    import benchmarks.benchmark_ogs00011_hybrid as harness
    from benchmarks._rss import RssSampler, rss_mb
    from opengwasdb.query import query_store

    limit = args.limit
    # A plain handle, not a `with` block: the shape's callable holds the query,
    # and closing it before the shape runs reads an empty store (review round 1).
    query = query_store(args.store)
    try:
        fn = harness._patterns(query, Path(args.store))[args.shape]
        load_start = _loads()
        baseline_mb = rss_mb()
        previous = signal.signal(signal.SIGALRM, _raise)
        signal.setitimer(signal.ITIMER_REAL, limit)
        sampler = RssSampler()
        start = time.perf_counter()
        try:
            with sampler:
                result = fn()
            timed_out = False
        except _Timeout:
            result = None
            timed_out = True
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        load_end = _loads()
    finally:
        query.close()
    record: dict[str, object] = {
        "shape": args.shape,
        "limit_s": limit,
        "elapsed_ms": round(elapsed_ms, 3),
        "timed_out": timed_out,
        "baseline_mb": round(baseline_mb, 1),
        "peak_mb": round(sampler.peak_mb, 1),
        "load_start": load_start,
        "load_end": load_end,
    }
    if result is not None:
        record["result_count"] = int(len(result["z"]))
        record["sha256"] = _digest(result)
    print(json.dumps(record), flush=True)


def _one_identity(args: argparse.Namespace) -> None:
    """Hash a fixed public-query set on this tree; print one JSON line."""
    from opengwasdb.query import query_store

    with query_store(args.store) as query:
        table = query.variants_table()
        variants = sorted((int(k), str(v["alid"])) for k, v in table.items())
        analyses = [str(row["analysis_id"]) for _, row in sorted(query.analyses_table().items())]
        hybrid = hasattr(query, "_shared_is_on_panel")
        if hybrid:
            off = next((v for v in variants if not query._shared_is_on_panel(v[0])), variants[0])
        else:
            off = variants[0]
        chrom, pos, _ea, _oa = variants[0][1].split(":")
        window = (chrom, max(0, int(pos) - 50_000), int(pos) + 50_000)
        queries = {
            "phewas_first": lambda: query.phewas(variants[0][1]),
            "phewas_off_panel": lambda: query.phewas(off[1]),
            "range_small": lambda: query.range_phewas(*window),
            "lookup_off_panel": lambda: query.lookup([off[1]], analyses[:3]),
            "lookup_small": lambda: query.lookup([v[1] for v in variants[:3]], analyses[:3]),
        }
        digests = {}
        for name, call in queries.items():
            result = call()
            digests[name] = {"sha256": _digest(result), "rows": int(len(result["z"]))}
    print(
        json.dumps(
            {
                "hybrid": hybrid,
                "n_variants": len(variants),
                "n_analyses": len(analyses),
                "queries": digests,
            }
        ),
        flush=True,
    )


def _run(tree: Path, args: argparse.Namespace, shape: str | None = None) -> dict:
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--one-shape" if shape is not None else "--one-identity",
        "--store",
        str(args.store),
        "--limit",
        str(args.limit),
    ]
    if shape is not None:
        argv += ["--shape", shape]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree)
    out = subprocess.run(argv, cwd=tree, env=env, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"{shape or 'identity'} on {tree} failed:\n{out.stdout}\n{out.stderr}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def _provenance(tree: Path) -> dict[str, str]:
    code = (
        "import json;"
        "from benchmarks._artifact import provenance;"
        "print(json.dumps(provenance()))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree)
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=tree, env=env, capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def _driver(args: argparse.Namespace) -> None:
    trees = {"before": Path(args.before_tree).resolve(), "after": Path(args.after_tree).resolve()}
    if args.identity:
        artifact: dict[str, object] = {
            "store": str(args.store),
            "note": (
                "Spot identity: the same public query surface and selection run against "
                "each tree, hashed with sha256 over variant_index/analysis_index/z/se/eaf "
                "and association_status, order included. Not a quiet-node measurement."
            ),
            "provenance": {side: _provenance(tree) for side, tree in trees.items()},
            "trees": {side: str(tree) for side, tree in trees.items()},
            "queries": {},
        }
        for side in SIDES:
            artifact["queries"][side] = _run(trees[side], args)
        print(json.dumps(artifact, indent=2) + "\n")
        if args.output:
            Path(args.output).write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
            print(f"Wrote {args.output}")
        return

    artifact = {
        "store": str(args.store),
        "shape_time_limit_s": args.limit,
        "region_shape_time_limit_s": args.limit,
        "note": (
            "Before/after interleaved shape by shape (before then after, back to back). "
            "NOT a quiet-node measurement: the machine is shared with other epic-240 "
            "lanes. Each run's start and end 1/5/15-minute loads are recorded, so load "
            "drift hits both sides equally. Both trees' commit and opengwasdb source "
            "fingerprint are recorded."
        ),
        "provenance": {side: _provenance(tree) for side, tree in trees.items()},
        "trees": {side: str(tree) for side, tree in trees.items()},
        "shapes": {},
    }
    for shape in args.shapes.split(","):
        shape = shape.strip()
        if not shape:
            continue
        before = _run(trees["before"], args, shape)
        after = _run(trees["after"], args, shape)
        artifact["shapes"][shape] = {"before": before, "after": after}
        print(
            f"{shape:44s} before={before['elapsed_ms']:10.1f} ms "
            f"({before['peak_mb']:8.1f} MB)  after={after['elapsed_ms']:10.1f} ms "
            f"({after['peak_mb']:8.1f} MB)  same={before.get('sha256') == after.get('sha256')}",
            flush=True,
        )
    if args.output:
        Path(args.output).write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--before-tree")
    ap.add_argument("--after-tree")
    ap.add_argument("--limit", type=float, default=300.0)
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--one-shape", dest="one_shape", action="store_true")
    ap.add_argument("--one-identity", dest="one_identity", action="store_true")
    ap.add_argument("--shape")
    ap.add_argument("--tree")
    args = ap.parse_args()
    if args.tree:
        _tree_on_path(Path(args.tree))
    if args.one_shape:
        _one_shape(args)
        return
    if args.one_identity:
        _one_identity(args)
        return
    if not (args.before_tree and args.after_tree):
        raise SystemExit("--before-tree and --after-tree are required in driver mode")
    _driver(args)


if __name__ == "__main__":
    main()
