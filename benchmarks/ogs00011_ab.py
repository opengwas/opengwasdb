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
from collections.abc import Callable
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

#: The probes whose selection is known to return rows on OGS-00011, with the
#: floor the committed artifact measured. A count below the floor is a changed
#: selection, a changed store or an empty read, and the run must fail rather
#: than publish it. Every shape is in exactly one of this table and
#: `EMPTY_SHAPES`; a shape in neither is refused, so a new shape cannot be added
#: without deciding which it is.
MIN_COUNTS = {
    "bulk": 5_000_000,
    "phewas": 3_000,
    "regional": 8_000_000,
    "regional_one_analysis": 1_000,
    "tophits": 6_000,
    "random_lookup_100_variants_10_analyses": 1,
    "phewas_off_axis": 1,
    "bulk_overflow_heavy": 85_000_000,
}

#: The shapes whose random selection is intentionally empty on OGS-00011. They
#: must return exactly zero rows; a non-zero count means the selection moved.
EMPTY_SHAPES = frozenset({"random_lookup_10_variants_100_analyses"})

#: Per-store, per-query identity minima, keyed by the store directory name. A
#: zero is an explicit allowance (that probe is known to return nothing on that
#: store); a positive value is a floor. Every query the identity probe runs must
#: appear here, so one non-empty query can never make an empty one pass.
IDENTITY_MIN_COUNTS: dict[str, dict[str, int]] = {
    "OGS-00001": {
        "phewas_first": 1,
        "phewas_off_panel": 1,
        "range_small": 100,
        "lookup_off_panel": 1,
        "lookup_small": 3,
    },
    "OGS-00006": {
        "phewas_first": 1,
        "phewas_off_panel": 1,
        "range_small": 1,
        "lookup_off_panel": 0,
        "lookup_small": 0,
    },
    "OGS-00004": {
        "phewas_first": 1,
        "phewas_off_panel": 1,
        "range_small": 50,
        "lookup_off_panel": 1,
        "lookup_small": 1,
    },
}


def _select_shapes(raw: str) -> list[str]:
    """The shape names a run will measure; an empty selection is refused."""
    selected = [name.strip() for name in raw.split(",") if name.strip()]
    if not selected:
        raise SystemExit("no shapes selected: an A/B run with no shapes is not evidence")
    return selected


def _require_floor(selected: list[str]) -> None:
    """Refuse a run whose shapes are all in `EMPTY_SHAPES`: it proves nothing."""
    if not any(name in MIN_COUNTS for name in selected):
        raise SystemExit(
            "every selected shape is in EMPTY_SHAPES; at least one shape with a "
            "MIN_COUNTS floor is required"
        )


def _check_shape(name: str, record: dict) -> None:
    """Refuse a shape that timed out, is unknown, or did not return its rows."""
    if record.get("timed_out"):
        raise SystemExit(
            f"{name}: hit the {record.get('limit_s')}s limit; a timed-out run is not evidence"
        )
    count = record.get("result_count")
    if name in EMPTY_SHAPES:
        if count != 0:
            raise SystemExit(f"{name}: whitelisted as empty but returned {count} rows")
        return
    if name not in MIN_COUNTS:
        raise SystemExit(
            f"{name}: no expected count; add it to MIN_COUNTS or EMPTY_SHAPES"
        )
    if count is None or count < MIN_COUNTS[name]:
        raise SystemExit(
            f"{name}: expected at least {MIN_COUNTS[name]} rows, got {count}"
        )


def _check_pair(name: str, before: dict, after: dict) -> None:
    """Refuse a before/after pair whose counts or answers differ."""
    _check_shape(name, before)
    _check_shape(name, after)
    if before.get("result_count") != after.get("result_count"):
        raise SystemExit(
            f"{name}: before {before.get('result_count')} rows, after "
            f"{after.get('result_count')} rows"
        )
    if not before.get("sha256") or not after.get("sha256"):
        raise SystemExit(f"{name}: a side returned no answer to hash")
    if before["sha256"] != after["sha256"]:
        raise SystemExit(f"{name}: before and after answers differ")


def _check_identity_side(side: str, record: dict, minima: dict[str, int]) -> None:
    """Refuse an identity run whose queries are missing or below their floors.

    Per query, not summed: a store with one non-empty probe and one that should
    have returned rows but did not is not evidence (review round 3).
    """
    for name, result in record["queries"].items():
        if name not in minima:
            raise SystemExit(f"identity {name}: no expected minimum for this query")
        if int(result["rows"]) < minima[name]:
            raise SystemExit(
                f"identity {name} on {side}: expected at least {minima[name]} rows, "
                f"got {result['rows']}"
            )
    missing = set(minima) - set(record["queries"])
    if missing:
        raise SystemExit(f"identity on {side}: no result for {sorted(missing)}")


def _check_identity_pair(name: str, before: dict, after: dict) -> None:
    """Refuse an identity query whose count or hash differs between the trees."""
    if before["rows"] != after["rows"] or before["sha256"] != after["sha256"]:
        raise SystemExit(
            f"identity {name}: before {before['rows']} rows/{before['sha256'][:12]} "
            f"!= after {after['rows']} rows/{after['sha256'][:12]}"
        )


class _Timeout(Exception):
    pass


def _raise(_signum, _frame):
    raise _Timeout


def _loads() -> list[float]:
    with open("/proc/loadavg") as fh:
        parts = fh.read().split()
    return [float(parts[0]), float(parts[1]), float(parts[2])]


def _digest(result: dict[str, np.ndarray], *, canonical: bool = False) -> str:
    """A sha256 over the six arrays, optionally in canonical row order.

    `canonical=True` sorts by `(variant_index, analysis_index)` first, for a
    comparison where the two answers may be grouped differently: the facade
    makes no ordering guarantee (ADR 0033) and the variant index returns a
    region variant-major where the scan returns flat-CSR order (ADR 0060).
    """
    keys = ("variant_index", "analysis_index", "z", "se", "eaf")
    status = np.asarray(result["association_status"], dtype=object)
    order = None
    if canonical and len(result["z"]):
        order = np.lexsort(
            (np.asarray(result["analysis_index"]), np.asarray(result["variant_index"]))
        )
    h = hashlib.sha256()
    for key in keys:
        arr = np.asarray(result[key])
        if order is not None:
            arr = arr[order]
        arr = np.ascontiguousarray(arr)
        h.update(key.encode())
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    if order is not None:
        status = status[order]
    h.update(b"association_status")
    h.update(str(len(status)).encode())
    for value in status:
        h.update(str(value).encode())
        h.update(b"\x00")
    return h.hexdigest()


def _tree_on_path(tree: Path) -> None:
    sys.path.insert(0, str(tree))


def measure_one_shape(
    store: str,
    shape: str,
    limit: float,
    *,
    before_timing: Callable[[], dict] | None = None,
    canonical_digest: bool = False,
    warm_index: bool = False,
) -> dict:
    """Run one shape once with an alarm and an RSS sampler; return its record.

    Split out of `_one_shape` so the committed extras runner
    (`benchmarks/ogs00011_extra_shapes.py`) can call it in-process and add the
    environment record beside it (#250 review r2, major 3).

    `before_timing` runs after the store is open and the shape is built, and
    immediately before `load_start` is sampled and the clock starts; its
    fields join the record. The runner passes its load gate here, because
    opening OGS-00011 itself raises the load: a gate before the open let a
    repetition start at 3.54 after passing at 2.72 (#250 round 3).
    """
    import benchmarks.benchmark_ogs00011_hybrid as harness
    from benchmarks._rss import RssSampler, rss_mb
    from opengwasdb.query import query_store

    # A plain handle, not a `with` block: the shape's callable holds the query,
    # and closing it before the shape runs reads an empty store (review round 1).
    query = query_store(store)
    if warm_index:
        # Read the variant index's exception tables before the clock starts: the
        # scan side reads the equivalent table when its reader opens, so timing
        # the first indexed decode cold would compare two different spans.
        reader = getattr(query, "_by_variant", None)
        if reader is not None:
            reader.warm()
    try:
        fn = harness._patterns(query, Path(store))[shape]
        gate = before_timing() if before_timing is not None else {}
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
        "shape": shape,
        "limit_s": limit,
        "elapsed_ms": round(elapsed_ms, 3),
        "timed_out": timed_out,
        "baseline_mb": round(baseline_mb, 1),
        "peak_mb": round(sampler.peak_mb, 1),
        "load_start": load_start,
        "load_end": load_end,
        **gate,
    }
    if result is not None:
        record["result_count"] = int(len(result["z"]))
        record["sha256"] = _digest(result, canonical=canonical_digest)
        record["canonical_digest"] = canonical_digest
    return record


def _open_eager_tables() -> None:
    """Make the exception/overflow tables eager (`read`) instead of windowed (`open`).

    The named bulk controls compare two **windowed** arms, so a
    windowed-versus-eager slowdown cancels in them and they cannot detect the
    regression they were added for.  This restores 144f335's behaviour so the
    windowed arm can be compared against it directly (review round 4,
    finding 5).
    """
    from opengwasdb.encoding import SparseExactTable

    SparseExactTable.open = classmethod(lambda cls, group: cls.read(group))


def _one_shape(args: argparse.Namespace) -> None:
    """Run one shape once with an alarm and an RSS sampler; print one JSON line."""
    if getattr(args, "eager_tables", False):
        _open_eager_tables()
    gate = None
    max_load = getattr(args, "max_start_load", 0.0) or 0.0
    if max_load > 0:
        from benchmarks._quiet import wait_for_quiet

        def gate() -> dict:
            waited, gave_up = wait_for_quiet(max_load)
            return {"gate_waited_s": round(waited, 1), "gate_gave_up": gave_up}

    record = measure_one_shape(
        args.store,
        args.shape,
        args.limit,
        before_timing=gate,
        canonical_digest=bool(getattr(args, "canonical_identity", False)),
        warm_index=bool(getattr(args, "warm_index", False)),
    )
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


def _tree_provenance(trees: dict[str, Path]) -> dict[str, dict[str, str]]:
    return {side: _provenance(tree) for side, tree in trees.items()}


def _identity_aggregate(args: argparse.Namespace) -> None:
    """Run the identity probe on each store against both trees; write the aggregate.

    This is the committed producer of
    `docs/benchmark-output/opengwasdb_252_spot_identity.json`: one entry per
    store, each query with its before/after count and hash, and `all_identical`
    true only when every one matches.
    """
    trees = {"before": Path(args.before_tree).resolve(), "after": Path(args.after_tree).resolve()}
    artifact: dict[str, object] = {
        "note": (
            "Spot identity for #252: the same public query surface and selection run "
            "against each tree through the committed benchmarks/ogs00011_ab.py "
            "--identity, hashed with sha256 over variant_index/analysis_index/z/se/eaf "
            "and association_status, order included. Not a quiet-node measurement."
        ),
        "provenance": _tree_provenance(trees),
        "trees": {side: str(tree) for side, tree in trees.items()},
        "stores": {},
    }
    all_identical = True
    for store in args.identity_store:
        name = Path(store).parent.name or str(store)
        if name not in IDENTITY_MIN_COUNTS:
            raise SystemExit(
                f"identity for {name}: no per-query minima; add the store to "
                "IDENTITY_MIN_COUNTS"
            )
        minima = IDENTITY_MIN_COUNTS[name]
        single = argparse.Namespace(**vars(args))
        single.store = store
        sides = {side: _run(trees[side], single) for side in SIDES}
        for side in SIDES:
            _check_identity_side(side, sides[side], minima)
        entry: dict[str, object] = {
            "hybrid": bool(sides["after"]["hybrid"]),
            "n_variants": int(sides["after"]["n_variants"]),
            "n_analyses": int(sides["after"]["n_analyses"]),
            "minima": dict(minima),
            "queries": {},
        }
        for query in sides["after"]["queries"]:
            before, after = sides["before"]["queries"][query], sides["after"]["queries"][query]
            _check_identity_pair(query, before, after)
            entry["queries"][query] = {
                "before": before, "after": after,
                "identical": before["rows"] == after["rows"]
                and before["sha256"] == after["sha256"],
            }
        artifact["stores"][name] = entry
        print(f"{name}: {len(entry['queries'])} queries identical", flush=True)
    artifact["all_identical"] = all_identical
    if args.output:
        Path(args.output).write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(json.dumps(artifact, indent=2))


def _driver(args: argparse.Namespace) -> None:
    trees = {"before": Path(args.before_tree).resolve(), "after": Path(args.after_tree).resolve()}
    if args.identity:
        if args.identity_store:
            _identity_aggregate(args)
            return
        artifact: dict[str, object] = {
            "store": str(args.store),
            "note": (
                "Spot identity: the same public query surface and selection run against "
                "each tree, hashed with sha256 over variant_index/analysis_index/z/se/eaf "
                "and association_status, order included. Not a quiet-node measurement."
            ),
            "provenance": _tree_provenance(trees),
            "trees": {side: str(tree) for side, tree in trees.items()},
            "queries": {},
        }
        for side in SIDES:
            record = _run(trees[side], args)
            name = Path(args.store).parent.name or str(args.store)
            if name not in IDENTITY_MIN_COUNTS:
                raise SystemExit(
                    f"identity for {name}: no per-query minima; add the store to "
                    "IDENTITY_MIN_COUNTS"
                )
            _check_identity_side(side, record, IDENTITY_MIN_COUNTS[name])
            artifact["queries"][side] = record
        for query in artifact["queries"]["after"]["queries"]:
            _check_identity_pair(
                query,
                artifact["queries"]["before"]["queries"][query],
                artifact["queries"]["after"]["queries"][query],
            )
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
        "provenance": _tree_provenance(trees),
        "trees": {side: str(tree) for side, tree in trees.items()},
        "shapes": {},
    }
    selected = _select_shapes(args.shapes)
    _require_floor(selected)
    for shape in selected:
        before = _run(trees["before"], args, shape)
        after = _run(trees["after"], args, shape)
        _check_pair(shape, before, after)
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
    ap.add_argument("--store")
    ap.add_argument("--identity-store", action="append", default=[])
    ap.add_argument("--before-tree")
    ap.add_argument("--after-tree")
    ap.add_argument("--limit", type=float, default=300.0)
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--one-shape", dest="one_shape", action="store_true")
    ap.add_argument(
        "--eager-tables",
        dest="eager_tables",
        action="store_true",
        help="read exception/overflow tables eagerly (144f335's behaviour)",
    )
    ap.add_argument("--one-identity", dest="one_identity", action="store_true")
    ap.add_argument(
        "--max-start-load",
        type=float,
        default=0.0,
        help="--one-shape: wait for the 1-minute load below this before timing (0 = off)",
    )
    ap.add_argument(
        "--canonical-identity",
        dest="canonical_identity",
        action="store_true",
        help="--one-shape: hash the answer in canonical (variant, analysis) row order",
    )
    ap.add_argument(
        "--warm-index",
        dest="warm_index",
        action="store_true",
        help="--one-shape: read the variant index's exception tables before timing",
    )
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
