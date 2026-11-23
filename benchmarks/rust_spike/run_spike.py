#!/usr/bin/env python3
"""Time scattered Dense lookups (and the harness's other shapes) under each #477 candidate.

Candidate kinds
---------------
``python-fused``          the Python reader as shipped: fused pipeline, one worker, Blosc threads on
                          (``store_reader/arrays.py``).
``python-zarrs``          the same reader under ``zarrs.ZarrsCodecPipeline`` (zarrs-python).
``python-zarr-pool16``    the fused pipeline with zarr's own pool at 16 workers, Blosc threads on.
``python-threadpool16``   the reader's ``grid.read`` over up to 16 row partitions on a 16-thread
                          ``ThreadPoolExecutor`` (the same inner chunks ``main.rs`` reads),
                          numcodecs' Blosc threads off.
``rust:N``                ``benchmarks/rust_spike`` (the ``zarrs`` crate) with the chunk groups on
                          N threads; zarrs' own codec concurrency is capped at 1 and
                          ``RAYON_NUM_THREADS`` set to N (both recorded as set, not read back).

What runs is asserted, not assumed.  A child sets its zarr/numcodecs configuration *after* importing
the Store Reader (whose array seam sets the configuration at import), reads the effective pipeline
class, ``max_workers`` and Blosc thread setting back from zarr and numcodecs, and aborts if they are
not the ones the kind names.

Measurement
-----------
Every measurement is a fresh process: its first call is the *cold* call (interpreter new, arrays
just opened, nothing decoded yet; the OS page cache stays warm and cannot be dropped without
root, so "cold" is not a first-touch disk read), followed by ``--warm`` warm calls.
Rounds are interleaved:
for each (shape, stage), round r runs every candidate once before round r+1 starts, so a change in
host load lands on all candidates alike.  Stages separate the ALID-to-row step
(``axis_resolution``) from the decode (``decode``: the cell read with rows and columns given)
and the whole ``lookup`` (``end_to_end``).  The other harness shapes run end to end only.

Agreement: the reference is the first ``python-fused`` run, which must be listed first; every
other run's first-call digests (and, for Rust, its cold process's too) must equal the reference's
for the same shape and stage, or the run aborts.  A Python process digests its first and its
last (timed) call; a Rust process digests its cold call and its untimed warm-up call.

Commands
--------
See ``docs/benchmark-output/rust_spike_ogs00009.md`` for the exact ``cargo build``,
``rechunk.py`` and ``--run`` commands.  ``summarise_spike.py`` turns this script's JSON into
the tables.  The selection
file is committed beside the JSON (``docs/benchmark-output/rust_spike_selection.json``).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
STAGES = ("axis_resolution", "decode", "end_to_end")
KINDS = (
    "python-fused",
    "python-zarrs",
    "python-zarr-pool16",
    "python-threadpool16",
    "rust",
)
POOL = 16
ZARRS_PIPELINE = "zarrs.ZarrsCodecPipeline"
FUSED = "zarr.core.codec_pipeline.FusedCodecPipeline"
# Expected effective configuration per Python kind: (config path, pipeline class name,
# max_workers, numcodecs Blosc threads).
EXPECTED: dict[str, tuple[str, str, int | None, bool]] = {
    "python-fused": (FUSED, "FusedCodecPipeline", 1, True),
    "python-zarrs": (ZARRS_PIPELINE, "ZarrsCodecPipeline", None, True),
    "python-zarr-pool16": (FUSED, "FusedCodecPipeline", POOL, True),
    "python-threadpool16": (FUSED, "FusedCodecPipeline", 1, False),
}


def shape_name(n_variants: int, n_analyses: int) -> str:
    return f"random_lookup_{n_variants}_variants_{n_analyses}_analyses"


def selection_for(
    selection: dict[str, Any], n_variants: int, n_analyses: int
) -> tuple[list[str], list[str]]:
    """The harness's slicing (`common_query_patterns`): first n variants, first m Analyses."""
    return (
        list(selection["random_alids"][:n_variants]),
        list(selection["random_analyses"][:n_analyses]),
    )


def digest_rows(values: Any) -> str:
    return hashlib.sha256(np.asarray(values, dtype="int64").tobytes()).hexdigest()


# --------------------------------------------------------------------------- child


def _threadpool_read(
    q: Any, rows: np.ndarray, cols: list[int], pool: Any, cells_cls: Any
) -> dict[str, Any]:
    """`grid.read` over up to 16 row partitions on a thread pool, merged into the lookup's order.

    One `grid.read` per (row block, column block) group costs ~28 ms of fixed zarr/event-loop
    overhead each (390 groups in the 10x100 shape), which would measure that overhead rather than
    the decode, so each task takes whole row blocks times every requested column.  The chunks
    decoded
    are the ones `main.rs` reads.  Results come back Analysis-major per task; the merge sorts them
    into the whole read's order (columns outer, rows inner), so the arrays are the ones one
    `grid.read` would return.
    """
    array = q._grid._z.array
    row_chunk = int(array.chunks[0])
    rows = np.asarray(rows, dtype=np.int64)
    col_array = np.asarray(cols, dtype=np.int64)
    # Rows sharing a row block stay together, so the tasks decode exactly the inner chunks
    # `main.rs` reads (no chunk twice); the blocks are dealt round-robin to POOL buckets.
    by_block: dict[int, list[int]] = {}
    for row in rows.tolist():
        by_block.setdefault(row // row_chunk, []).append(row)
    buckets: list[list[int]] = [[] for _ in range(min(POOL, len(by_block)))]
    for i, block_rows in enumerate(by_block.values()):
        buckets[i % len(buckets)].extend(block_rows)
    results = list(
        pool.map(lambda b: q._grid.read(cells_cls.cross(b, cols), observed_only=False), buckets)
    )
    row_pos = {int(r): i for i, r in enumerate(rows.tolist())}
    col_pos = {int(c): i for i, c in enumerate(col_array.tolist())}
    names = list(results[0].keys()) if results else []
    merged = {n: np.concatenate([np.asarray(r[n]) for r in results]) for n in names}
    key = np.array(
        [
            col_pos[int(a)] * len(rows) + row_pos[int(v)]
            for v, a in zip(merged["variant_index"], merged["analysis_index"], strict=True)
        ],
        dtype=np.int64,
    )
    order = np.argsort(key, kind="stable")
    return {n: v[order] for n, v in merged.items()}


def _child(args: argparse.Namespace) -> None:
    """One fresh-process measurement of a Python candidate; prints one JSON line."""
    # Import everything the Store Reader pulls in FIRST: its array seam sets the zarr
    # configuration at import, and anything set before that is overwritten.
    import numcodecs.blosc
    import zarr

    import opengwasdb
    import opengwasdb.query
    import opengwasdb.store_reader.arrays
    from benchmarks import _query_shapes
    from benchmarks.benchmark_store_comparison import result_digests
    from opengwasdb.store_reader.associations.cells import Cells

    if not Path(opengwasdb.__file__).resolve().is_relative_to(REPO):
        raise SystemExit(f"opengwasdb imported from {opengwasdb.__file__}, not {REPO}")
    path, pipeline_class, workers, blosc_threads = EXPECTED[args.kind]
    zarrs_version = None
    if args.kind == "python-zarrs":
        import zarrs

        zarrs_version = zarrs.__version__
    zarr.config.set({"codec_pipeline.path": path})
    if workers is not None:
        zarr.config.set({"codec_pipeline.max_workers": workers})
    numcodecs.blosc.use_threads = blosc_threads

    selection = json.loads(Path(args.selection).read_text())
    alids, analyses = selection_for(selection, args.n_variants, args.n_analyses)
    opened = time.perf_counter()
    q = opengwasdb.query.query_store(args.store)
    open_ms = (time.perf_counter() - opened) * 1000.0

    pool = None
    if args.kind == "python-threadpool16":
        from concurrent.futures import ThreadPoolExecutor

        pool = ThreadPoolExecutor(max_workers=POOL)

    def analysis_columns() -> list[int]:
        cols = []
        for aid in analyses:
            index = q._analysis_index(aid)
            if index is None:
                raise SystemExit(f"Analysis {aid} is not in {args.store}")
            cols.append(index)
        return cols

    if args.shape in {shape_name(10, 100), shape_name(100, 10)}:
        rows = q._variant_axis.indices_by_identifiers(alids)
        if len(rows) != len(alids):
            raise SystemExit("an ALID did not resolve; the selection is for another release")
        cols = analysis_columns()

        def axis() -> dict[str, np.ndarray]:
            r = q._variant_axis.indices_by_identifiers(alids)
            return {
                "rows": np.asarray(r, dtype="int64"),
                "cols": np.asarray(analysis_columns(), dtype="int64"),
            }

        def decode() -> Any:
            if pool is None:
                return q._grid.read(Cells.cross(rows, cols), observed_only=False)
            return _threadpool_read(q, rows, cols, pool, Cells)

        def end_to_end() -> Any:
            if pool is None:
                return q.lookup(alids, analyses)
            return _threadpool_read(
                q, q._variant_axis.indices_by_identifiers(alids), analysis_columns(), pool, Cells
            )

        fn = {"axis_resolution": axis, "decode": decode, "end_to_end": end_to_end}[args.stage]
    else:
        if args.kind != "python-fused" or args.stage != "end_to_end":
            raise SystemExit(
                "the harness's other shapes are timed end to end with python-fused only"
            )
        region = selection["region"]
        patterns = _query_shapes.common_query_patterns(
            q,
            exposure=selection["exposure_analysis_id"],
            phewas_alid=selection["phewas_alid"],
            region=(region["chrom"], region["start"], region["end"]),
            random_alids=list(selection["random_alids"]),
            random_analyses=list(selection["random_analyses"]),
        )
        fn = patterns[args.shape]

    effective: dict[str, Any] = {}
    times: list[float] = []
    first_digests: dict[str, str] = {}
    last_digests: dict[str, str] = {}
    count = 0
    for call in range(1 + args.warm):
        started = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - started) * 1000.0)
        arrays = dict(result.items()) if hasattr(result, "items") else dict(result)
        digests = (
            {k: digest_rows(v) for k, v in arrays.items()}
            if args.stage == "axis_resolution"
            else result_digests(arrays)
        )
        if call == 0:
            first_digests = digests
            count = int(len(arrays["z"])) if "z" in arrays else int(len(arrays["rows"]))
            # Read back what actually ran, after the first call has opened every array.
            effective = {
                "config_path": zarr.config.get("codec_pipeline.path"),
                "config_max_workers": zarr.config.get("codec_pipeline.max_workers"),
                "array_pipeline_class": type(q._grid._z.array._async_array.codec_pipeline).__name__,
                "blosc_use_threads": bool(numcodecs.blosc.use_threads),
                "blosc_nthreads_env": os.environ.get("BLOSC_NTHREADS"),
            }
        last_digests = digests
    if (
        effective["config_path"] != path
        or effective["array_pipeline_class"] != pipeline_class
        or (workers is not None and effective["config_max_workers"] != workers)
        or effective["blosc_use_threads"] != blosc_threads
    ):
        raise SystemExit(
            f"{args.kind}: effective configuration {effective} is not the one the kind names "
            f"({path}, {pipeline_class}, max_workers={workers}, blosc_threads={blosc_threads})"
        )
    print(
        json.dumps(
            {
                "cold_ms": times[0],
                "warm_ms": times[1:],
                "open_ms": open_ms,
                "digests_first": first_digests,
                "digests_last": last_digests,
                "result_count": count,
                "effective_config": effective,
                "zarr": zarr.__version__,
                "numcodecs": numcodecs.__version__,
                "zarrs_python": zarrs_version,
                "threadpool": POOL if pool else None,
            }
        )
    )


# -------------------------------------------------------------------------- parent


def _run_cmd(cmd: list[str], cwd: str, env: dict[str, str] | None = None) -> tuple[str, float]:
    started = time.perf_counter()
    done = subprocess.run(
        cmd, cwd=cwd, env={**os.environ, **(env or {})}, capture_output=True, text=True
    )
    wall_ms = (time.perf_counter() - started) * 1000.0
    if done.returncode:
        raise SystemExit(f"{' '.join(cmd[:6])} failed ({done.returncode}):\n{done.stderr[-3000:]}")
    return done.stdout.strip().splitlines()[-1], wall_ms


def _rust_arrays_to_result(measurement: dict[str, Any]) -> dict[str, np.ndarray]:
    from opengwasdb.store_reader.associations.record import STATUS_LABELS

    dtypes = {"int32": "<i4", "float32": "<f4", "int8": "i1"}
    out = {
        name: np.frombuffer(base64.b64decode(p["base64"]), dtype=dtypes[p["dtype"]])
        for name, p in measurement["arrays"].items()
    }
    status = out.pop("association_status_int8")
    out["association_status"] = STATUS_LABELS[status.astype(np.int64)]
    return out


def _zarrs_crate_version(lock: Path) -> str:
    match = re.search(r'name = "zarrs"\nversion = "([^"]+)"', lock.read_text())
    if not match:
        raise SystemExit(f"no zarrs entry in {lock}")
    return match.group(1)


def _array_sizes(store: str) -> dict[str, Any]:
    """Apparent bytes per top-level array of data.zarr (files only; symlinks followed)."""
    root = Path(store) / "data.zarr"
    out: dict[str, int] = {}
    for entry in sorted(root.iterdir()):
        if entry.name == "zarr.json":
            continue
        target = entry.resolve()
        total = (
            sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
            if target.is_dir()
            else 0
        )
        out[entry.name] = total
    return {"apparent_bytes": out, "total": sum(out.values())}


def _hardware(store: str) -> dict[str, Any]:
    def sh(cmd: str) -> str:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()

    return {
        "lscpu": sh("lscpu"),
        "free_g": sh("free -g"),
        "filesystem": sh(f"df -T {store}"),
        "mount": sh(f"findmnt -T {store} -o SOURCE,FSTYPE,OPTIONS"),
        "kernel": platform.platform(),
    }


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True
    ).stdout.strip()


def _parent(args: argparse.Namespace) -> None:
    from benchmarks._quiet import wait_for_quiet
    from benchmarks.benchmark_store_comparison import result_digests
    from opengwasdb.query import query_store

    args.selection = Path(args.selection).resolve()  # children run from the scratch directory
    selection = json.loads(args.selection.read_text())
    runs = []
    for spec in args.run:
        name, _, rest = spec.partition("=")
        kind, _, store = rest.partition("@")
        if not (name and kind and store) or kind.split(":")[0] not in KINDS:
            raise SystemExit(f"--run wants NAME=KIND@STORE with KIND in {KINDS}, got {spec!r}")
        runs.append((name, kind, store))
    if not runs or runs[0][1] != "python-fused":
        raise SystemExit("the first --run must be python-fused: it is the agreement reference")
    reference_name = runs[0][0]

    manifest = json.loads((Path(args.base_store) / "manifest.json").read_text())["encoding"]
    q = query_store(args.base_store)
    lookup_shapes = [
        shape_name(s["n_variants"], s["n_analyses"]) for s in selection["random_lookup_shapes"]
    ]
    axis_cache: dict[str, dict[str, Any]] = {}
    for shape in selection["random_lookup_shapes"]:
        name = shape_name(shape["n_variants"], shape["n_analyses"])
        alids, analyses = selection_for(selection, shape["n_variants"], shape["n_analyses"])
        rows = [int(x) for x in q._variant_axis.indices_by_identifiers(alids)]
        cols = []
        for aid in analyses:
            index = q._analysis_index(aid)
            if index is None:
                raise SystemExit(f"Analysis {aid} is not in {args.base_store}")
            cols.append(int(index))
        wide = [a for a in alids if len(a.encode()) > 32]
        wide_rows = [int(q._variant_axis.indices_by_identifiers([a])[0]) for a in wide]
        axis_cache[name] = {
            "alids": alids,
            "analyses": analyses,
            "rows": rows,
            "cols": cols,
            "wide_rows": wide_rows,
        }
    q.close()

    crate = _zarrs_crate_version(REPO / "benchmarks/rust_spike/Cargo.lock")
    scratch = Path(args.scratch)
    out: dict[str, Any] = {
        "tool": "benchmarks/rust_spike/run_spike.py",
        "ticket": 477,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": sys.argv,
        "commit": _git("rev-parse", "HEAD"),
        "tree_clean": not _git("status", "--porcelain", "--", "benchmarks/rust_spike"),
        "base_store": args.base_store,
        "selection": json.loads(json.dumps(selection)),
        "rounds": args.rounds,
        "warm_per_process": args.warm,
        "max_start_load": args.max_start_load,
        "zarrs_crate": crate,
        "reference_run": reference_name,
        "hardware": _hardware(args.base_store),
        "store_sizes": {s: _array_sizes(s) for s in sorted({r[2] for r in runs})},
        "cold_definition": (
            "the first call of a fresh process: arrays just opened, nothing decoded; the OS page "
            "cache is warm and cannot be dropped without root, so this is not a first-touch disk "
            "read"
        ),
        "columns": [],
    }
    reference: dict[tuple[str, str], dict[str, str]] = {}
    extra_shapes = list(args.extra_shape)

    plan = [(s, st, args.rounds) for s in lookup_shapes for st in STAGES]
    plan += [(s, "end_to_end", args.extra_rounds) for s in extra_shapes]
    for shape, stage, rounds in plan:
        extra = shape in extra_shapes
        active = [r for r in runs if (not extra) or r[1] == "python-fused"]
        columns = {
            name: {
                "run": name,
                "kind": kind,
                "store": store,
                "shape": shape,
                "stage": stage,
                "rounds": [],
            }
            for name, kind, store in active
        }
        for rnd in range(rounds):
            for name, kind, store in active:
                waited, timed_out = wait_for_quiet(args.max_start_load, timeout_s=1800.0)
                record: dict[str, Any] = {
                    "round": rnd,
                    "load_1m_at_start": os.getloadavg()[0],
                    "waited_for_quiet_s": waited,
                    "quiet_wait_timed_out": timed_out,
                }
                base = kind.split(":")[0]
                if base == "rust":
                    threads = kind.partition(":")[2] or "16"
                    ax = axis_cache[shape]
                    request = {
                        "store_path": store,
                        "shape": shape,
                        "stage": stage,
                        "alids": ax["alids"],
                        "analysis_ids": ax["analyses"],
                        "rows": ax["rows"],
                        "cols": ax["cols"],
                        "wide_rows": ax["wide_rows"],
                        "z_scale": float(manifest["z"]["scale"]),
                        "se_step": float(manifest["se"]["residual_range"]) / 127.0,
                        "eaf_step": float(manifest["eaf"]["residual_range"]) / 127.0,
                        "reps": args.warm,
                    }
                    request_path = scratch / "request.json"
                    request_path.write_text(json.dumps(request))
                    env = {"SPIKE_THREADS": threads, "RAYON_NUM_THREADS": threads}
                    cold_line, _ = _run_cmd(
                        [args.rust_binary, str(request_path), "cold"], str(scratch), env
                    )
                    warm_line, _ = _run_cmd(
                        [args.rust_binary, str(request_path), "warm"], str(scratch), env
                    )
                    cold, warm = json.loads(cold_line), json.loads(warm_line)
                    if cold["zarrs_crate"] != crate or warm["zarrs_crate"] != crate:
                        raise SystemExit(
                            "the Rust binary was built against a different zarrs than Cargo.lock "
                            "names"
                        )
                    record.update(
                        cold_ms=cold["times_ms"][0],
                        warm_ms=warm["times_ms"],
                        result_count=warm["result_count"],
                        effective_config={
                            "threads": warm["threads"],
                            "zarrs_codec_concurrent_target": warm["zarrs_codec_concurrent_target"],
                            "rayon_num_threads_env": warm["rayon_num_threads_env"],
                            "wide_alid_fallback": warm["wide_alid_fallback"],
                        },
                    )
                    if warm["threads"] != int(threads):
                        raise SystemExit("the Rust binary ran a different thread count than asked")
                    digests = (
                        None
                        if stage == "axis_resolution"
                        else {
                            "first": result_digests(_rust_arrays_to_result(cold)),
                            "last": result_digests(_rust_arrays_to_result(warm)),
                        }
                    )
                else:
                    cmd = [sys.executable]
                    if base == "python-zarrs":
                        cmd = [
                            "pixi",
                            "run",
                            "--manifest-path",
                            f"{args.zarrs_python}/pixi.toml",
                            "python",
                        ]
                    cmd += [
                        str(Path(__file__).resolve()),
                        "--child",
                        "--kind",
                        base,
                        "--store",
                        store,
                        "--selection",
                        str(args.selection),
                        "--shape",
                        shape,
                        "--stage",
                        stage,
                        "--warm",
                        str(args.extra_warm if extra else args.warm),
                    ]
                    if not extra:
                        n_variants, n_analyses = (int(x) for x in re.findall(r"\d+", shape)[:2])
                        cmd += ["--n-variants", str(n_variants), "--n-analyses", str(n_analyses)]
                    else:
                        cmd += ["--n-variants", "100", "--n-analyses", "100"]
                    line, _ = _run_cmd(cmd, str(scratch), {"PYTHONPATH": str(REPO)})
                    child = json.loads(line)
                    record.update(
                        cold_ms=child["cold_ms"],
                        warm_ms=child["warm_ms"],
                        open_ms=child["open_ms"],
                        result_count=child["result_count"],
                        effective_config=child["effective_config"],
                        versions={
                            k: child[k] for k in ("zarr", "numcodecs", "zarrs_python", "threadpool")
                        },
                    )
                    digests = {"first": child["digests_first"], "last": child["digests_last"]}
                if digests is not None:
                    key = (shape, stage)
                    if name == reference_name and key not in reference:
                        reference[key] = digests["first"]
                    if key not in reference:
                        raise SystemExit("the reference run must be measured first in every round")
                    for which, got in digests.items():
                        if got != reference[key]:
                            differing = sorted(
                                k
                                for k in set(got) | set(reference[key])
                                if got.get(k) != reference[key].get(k)
                            )
                            raise SystemExit(
                                f"{name} ({which} call) disagrees with {reference_name} on "
                                f"{shape}/{stage}: {differing}; refusing to publish timings for "
                                f"a wrong answer"
                            )
                    record["digests"] = digests["first"]
                    record["digests_equal_reference"] = True
                columns[name]["rounds"].append(record)
            print(
                f"{shape} {stage} round {rnd}: "
                + ", ".join(f"{n} {columns[n]['rounds'][-1]['warm_ms'][-1]:.1f}" for n in columns),
                flush=True,
            )
        out["columns"].extend(columns.values())
        Path(args.output).write_text(json.dumps(out, indent=1) + "\n")
    out["agreement"] = (
        f"every digested process (cold and warm, Rust and Python) equals {reference_name}'s "
        f"first call "
        "for the same shape and stage; otherwise the run aborts and writes nothing further"
    )
    Path(args.output).write_text(json.dumps(out, indent=1) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--child", action="store_true")
    p.add_argument("--kind")
    p.add_argument("--store")
    p.add_argument("--shape")
    p.add_argument("--n-variants", type=int)
    p.add_argument("--n-analyses", type=int)
    p.add_argument("--stage")
    p.add_argument("--selection", type=Path)
    p.add_argument("--warm", type=int, default=3, help="warm calls per fresh process")
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--extra-rounds", type=int, default=5)
    p.add_argument(
        "--extra-warm", type=int, default=1, help="warm calls per process for the extra shapes"
    )
    p.add_argument(
        "--extra-shape",
        action="append",
        default=[],
        help="a harness shape name (python-fused only)",
    )
    p.add_argument("--output", type=Path)
    p.add_argument("--base-store")
    p.add_argument("--run", action="append", default=[], help="NAME=KIND@STORE")
    p.add_argument("--rust-binary")
    p.add_argument(
        "--zarrs-python", help="directory holding the scratch pixi.toml with zarrs-python"
    )
    p.add_argument("--scratch", help="neutral working directory for children")
    p.add_argument("--max-start-load", type=float, default=3.0)
    args = p.parse_args()
    if args.child:
        _child(args)
    else:
        if args.rounds < 5:
            raise SystemExit("at least 5 rounds (5 cold calls per column)")
        _parent(args)


if __name__ == "__main__":
    main()
