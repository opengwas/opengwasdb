#!/usr/bin/env python3
"""Time scattered Dense lookups on the same selection under each #477 candidate.

Candidates (each on the harness's two ``random_lookup_*`` shapes, seed-0
selection taken from the committed selection file):

* ``python-fused``  the Python reader as it is (fused codec pipeline, ``store/arrays.py``);
* ``python-zarrs``  the same reader with zarrs-python's codec pipeline;
* ``rust[:N]``      ``benchmarks/rust_spike`` (the ``zarrs`` crate), a minimal Dense lookup, chunk groups on N threads (default 16);
* any of the three on a re-chunked copy of the lookup arrays (smaller inner chunk).

Each (candidate, store, shape, stage) is one fresh process: its first call is
the *cold* call (a fresh interpreter, arrays opened, nothing decoded; the OS
page cache cannot be dropped without root and stays warm), then ``--reps``
warm calls.  Stages separate the ALID-to-row step from the decode:

* ``axis_resolution``  ALIDs and Analysis ids to row and column indices;
* ``decode``           the orthogonal cell read plus decoding, rows/cols given;
* ``end_to_end``       ``StoreQuery.lookup`` (resolution + decode + record).

Every candidate's result digests are compared with the first python-fused
run; a mismatch stops the run before any timing is written.

Usage (parent, dev environment, repository root)::

    pixi run -e dev python benchmarks/rust_spike/run_spike.py \\
        --selection SELECTION.json --output OUT.json --rust-binary BIN \\
        --zarrs-python 'PIXI_DIR' --run NAME=KIND@STORE [--run ...]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
STAGES = ("axis_resolution", "decode", "end_to_end")
KINDS = ("python-fused", "python-zarrs", "rust")
ZARRS_PIPELINE = "zarrs.ZarrsCodecPipeline"


def _shape_name(n_variants: int, n_analyses: int) -> str:
    return f"random_lookup_{n_variants}_variants_{n_analyses}_analyses"


def _selection_for(selection: dict[str, Any], shape: dict[str, int]) -> tuple[list[str], list[str]]:
    """The harness's slicing: variants first n, analyses first m (cf. common_query_patterns)."""
    return (
        list(selection["random_alids"][: shape["n_variants"]]),
        list(selection["random_analyses"][: shape["n_analyses"]]),
    )


def _summary(times: list[float]) -> dict[str, float]:
    ordered = sorted(times)
    return {
        "median_ms": ordered[len(ordered) // 2],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
    }


# --------------------------------------------------------------------------- child


def _child(args: argparse.Namespace) -> None:
    """One fresh-process measurement of a Python candidate; prints one JSON line."""
    import opengwasdb

    if not Path(opengwasdb.__file__).resolve().is_relative_to(REPO):
        raise SystemExit(f"opengwasdb imported from {opengwasdb.__file__}, not {REPO}")
    import zarr

    if args.kind == "python-zarrs":
        import zarrs

        zarr.config.set({"codec_pipeline.path": ZARRS_PIPELINE})
        zarrs_version = zarrs.__version__ if hasattr(zarrs, "__version__") else "unknown"
    else:
        zarrs_version = None
    from benchmarks.benchmark_store_comparison import result_digests
    from opengwasdb.associations.cells import Cells
    from opengwasdb.query import query_store

    selection = json.loads(Path(args.selection).read_text())
    shape = {"n_variants": args.n_variants, "n_analyses": args.n_analyses}
    alids, analyses = _selection_for(selection, shape)
    opened = time.perf_counter()
    q = query_store(args.store)
    open_ms = (time.perf_counter() - opened) * 1000.0
    pipeline = type(q._grid._group["z"]._async_array.codec_pipeline).__name__ if hasattr(
        q._grid, "_group"
    ) else "unknown"
    rows = q._variant_axis.indices_by_identifiers(alids)
    cols = [i for aid in analyses if (i := q._analysis_index(aid)) is not None]

    def axis() -> dict[str, np.ndarray]:
        r = q._variant_axis.indices_by_identifiers(alids)
        c = [i for aid in analyses if (i := q._analysis_index(aid)) is not None]
        return {"rows": np.asarray(r, dtype="int64"), "cols": np.asarray(c, dtype="int64")}

    calls = {
        "axis_resolution": axis,
        "decode": lambda: q._grid.read(Cells.cross(rows, cols), observed_only=False),
        "end_to_end": lambda: q.lookup(alids, analyses),
    }
    fn = calls[args.stage]
    times: list[float] = []
    first: dict[str, np.ndarray] | None = None
    for call in range(1 + args.reps):
        started = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - started) * 1000.0)
        if call == 0:
            first = dict(result.items()) if hasattr(result, "items") else dict(result)
    assert first is not None
    digests = (
        {k: _digest_rows(v) for k, v in first.items()}
        if args.stage == "axis_resolution"
        else result_digests(first)
    )
    print(
        json.dumps(
            {
                "cold_ms": times[0],
                "warm_ms": times[1:],
                "open_ms": open_ms,
                "digests": digests,
                "result_count": int(len(first["z"])) if "z" in first else int(len(first["rows"])),
                "codec_pipeline": pipeline,
                "zarr": zarr.__version__,
                "zarrs_python": zarrs_version,
            }
        )
    )


def _digest_rows(values: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.asarray(values, dtype="int64").tobytes()).hexdigest()


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
    from opengwasdb.associations.record import STATUS_LABELS

    dtypes = {"int32": "<i4", "float32": "<f4", "int8": "i1"}
    out: dict[str, np.ndarray] = {}
    for name, packed in measurement["arrays"].items():
        out[name] = np.frombuffer(base64.b64decode(packed["base64"]), dtype=dtypes[packed["dtype"]])
    status = out.pop("association_status_int8")
    out["association_status"] = STATUS_LABELS[status.astype(np.int64)]
    return out


def _hardware(store: str) -> dict[str, Any]:
    def sh(cmd: str) -> str:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()

    return {
        "lscpu": sh("lscpu"),
        "free_g": sh("free -g"),
        "filesystem": sh(f"df -T {store}"),
        "mount": sh(f"findmnt -T {store} -o SOURCE,FSTYPE,OPTIONS"),
        "kernel": platform.platform(),
        "page_cache_note": (
            "cold = first call of a fresh process; the OS page cache is warm and cannot be "
            "dropped without root, so cold figures are not first-touch disk numbers"
        ),
    }


def _parent(args: argparse.Namespace) -> None:
    from benchmarks._quiet import wait_for_quiet
    from benchmarks.benchmark_store_comparison import result_digests
    from opengwasdb.query import query_store

    selection = json.loads(Path(args.selection).read_text())
    shapes = selection["random_lookup_shapes"]
    runs = [r.split("=", 1) for r in args.run]
    runs = [(name, *spec.split("@", 1)) for name, spec in runs]
    for _, kind, _ in runs:
        if kind.split(":")[0] not in KINDS:
            raise SystemExit(f"unknown candidate kind {kind!r}")
    base_store = args.base_store
    manifest = json.loads((Path(base_store) / "manifest.json").read_text())["encoding"]
    q = query_store(base_store)
    axis_cache: dict[str, dict[str, Any]] = {}
    for shape in shapes:
        name = _shape_name(shape["n_variants"], shape["n_analyses"])
        alids, analyses = _selection_for(selection, shape)
        rows = [int(x) for x in q._variant_axis.indices_by_identifiers(alids)]
        cols = [i for aid in analyses if (i := q._analysis_index(aid)) is not None]
        wide = [a for a in alids if len(a.encode()) > 32]
        wide_rows = [int(q._variant_axis.indices_by_identifiers([a])[0]) for a in wide]
        axis_cache[name] = {
            "alids": alids, "analyses": analyses, "rows": rows, "cols": cols,
            "wide_rows": wide_rows,
        }
    q.close()

    zarrs_python = args.zarrs_python
    reference: dict[tuple[str, str], dict[str, str]] = {}
    out: dict[str, Any] = {
        "tool": "benchmarks/rust_spike/run_spike.py",
        "ticket": 477,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "base_store": base_store,
        "selection": str(args.selection),
        "reps": args.reps,
        "max_start_load": args.max_start_load,
        "hardware": _hardware(base_store),
        "runs": [],
    }
    for label, kind, store in runs:
        record: dict[str, Any] = {"name": label, "kind": kind, "store": store, "columns": []}
        for shape in shapes:
            shape_name = _shape_name(shape["n_variants"], shape["n_analyses"])
            ax = axis_cache[shape_name]
            for stage in STAGES:
                waited, timed_out = wait_for_quiet(args.max_start_load, timeout_s=1800.0)
                load = os.getloadavg()
                column: dict[str, Any] = {
                    "shape": shape_name,
                    "stage": stage,
                    "load_1m_at_start": load[0],
                    "waited_for_quiet_s": waited,
                    "quiet_wait_timed_out": timed_out,
                }
                if kind.split(":")[0] == "rust":
                    # "rust:N" decodes chunk groups on N threads; plain "rust" is 16.
                    threads = kind.partition(":")[2] or "16"
                    column["rust_threads"] = int(threads)
                    rust_env = {"SPIKE_THREADS": threads}
                    request = {
                        "store_path": store,
                        "shape": shape_name,
                        "stage": stage,
                        "alids": ax["alids"],
                        "analysis_ids": ax["analyses"],
                        "rows": ax["rows"],
                        "cols": ax["cols"],
                        "wide_rows": ax["wide_rows"],
                        "z_scale": float(manifest["z"]["scale"]),
                        "se_step": float(manifest["se"]["residual_range"]) / 127.0,
                        "eaf_step": float(manifest["eaf"]["residual_range"]) / 127.0,
                        "reps": args.reps,
                    }
                    request_path = Path(args.scratch) / "request.json"
                    request_path.write_text(json.dumps(request))
                    cold, cold_wall = _run_cmd(
                        [args.rust_binary, str(request_path), "cold"], cwd=args.scratch, env=rust_env
                    )
                    warm, warm_wall = _run_cmd(
                        [args.rust_binary, str(request_path), "warm"], cwd=args.scratch, env=rust_env
                    )
                    cold_m, warm_m = json.loads(cold), json.loads(warm)
                    column["cold_ms"] = cold_m["times_ms"][0]
                    column["warm_ms"] = warm_m["times_ms"]
                    column["process_wall_ms"] = {"cold": cold_wall, "warm": warm_wall}
                    column["result_count"] = warm_m["result_count"]
                    column["zarrs_crate"] = warm_m["zarrs_crate"]
                    column["wide_alid_fallback"] = warm_m["wide_alid_fallback"]
                    column["codec_pipeline"] = "zarrs crate"
                    if stage == "axis_resolution":
                        digests: dict[str, str] | None = None
                    else:
                        digests = result_digests(_rust_arrays_to_result(warm_m))
                else:
                    env = {"PYTHONPATH": str(REPO)}
                    if kind == "python-zarrs":
                        cmd = ["pixi", "run", "--manifest-path", f"{zarrs_python}/pixi.toml", "python"]
                        cwd = args.scratch
                    else:
                        cmd = [sys.executable]
                        cwd = args.scratch
                    cmd += [
                        str(Path(__file__).resolve()),
                        "--child",
                        "--kind", kind,
                        "--store", store,
                        "--selection", str(args.selection),
                        "--n-variants", str(shape["n_variants"]),
                        "--n-analyses", str(shape["n_analyses"]),
                        "--stage", stage,
                        "--reps", str(args.reps),
                    ]
                    line, wall = _run_cmd(cmd, cwd=cwd, env=env)
                    child = json.loads(line)
                    column.update(
                        {
                            "cold_ms": child["cold_ms"],
                            "warm_ms": child["warm_ms"],
                            "open_ms": child["open_ms"],
                            "process_wall_ms": wall,
                            "result_count": child["result_count"],
                            "codec_pipeline": child["codec_pipeline"],
                            "zarr": child["zarr"],
                            "zarrs_python": child["zarrs_python"],
                        }
                    )
                    digests = child["digests"]
                if digests is not None:
                    key = (shape_name, stage)
                    if key not in reference:
                        reference[key] = digests
                    elif digests != reference[key]:
                        differing = sorted(
                            k for k in set(digests) | set(reference[key])
                            if digests.get(k) != reference[key].get(k)
                        )
                        raise SystemExit(
                            f"{label} disagrees with the first run on {shape_name}/{stage}: "
                            f"{differing}; refusing to publish timings for a wrong answer"
                        )
                    column["digests_equal_reference"] = True
                    column["digests"] = digests
                column["cold_summary"] = {"ms": column["cold_ms"]}
                column["warm_summary"] = _summary(column["warm_ms"])
                record["columns"].append(column)
                print(
                    f"{label:24s} {shape_name:44s} {stage:16s} cold {column['cold_ms']:9.2f} "
                    f"warm med {column['warm_summary']['median_ms']:9.2f} ms "
                    f"(load {load[0]:.1f})",
                    flush=True,
                )
        out["runs"].append(record)
        Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    out["agreement"] = "every digested column equal to the first run's (else the run aborts)"
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--child", action="store_true")
    p.add_argument("--kind")
    p.add_argument("--store")
    p.add_argument("--n-variants", type=int)
    p.add_argument("--n-analyses", type=int)
    p.add_argument("--stage")
    p.add_argument("--selection", type=Path)
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--output", type=Path)
    p.add_argument("--base-store")
    p.add_argument("--run", action="append", default=[], help="NAME=KIND@STORE")
    p.add_argument("--rust-binary")
    p.add_argument("--zarrs-python", help="directory holding the scratch pixi.toml with zarrs-python")
    p.add_argument("--scratch", help="neutral working directory for children")
    p.add_argument("--max-start-load", type=float, default=3.0)
    args = p.parse_args()
    if args.child:
        _child(args)
    else:
        if args.reps < 5:
            raise SystemExit("at least 5 reps")
        _parent(args)


if __name__ == "__main__":
    main()
