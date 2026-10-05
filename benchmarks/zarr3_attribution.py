"""Which #244 read lever does what: OGS-00009 query shapes per zarr configuration.

Each configuration runs in a fresh process, under the environment of the
checkout it measures, and the configurations are interleaved inside each round,
so a noisy minute on a shared node lands on every configuration rather than on
one. This is the attribution table in ADR 0056 and in #244's Stage A re-run
comment, and the worker-count measurement that chose `max_workers = 1`.

A configuration is `NAME=LABEL:CHECKOUT:PYTHON`. The child puts CHECKOUT first
on `sys.path`, so the code under test is that checkout's, and PYTHON is that
checkout's environment (zarr 2.18 for the base). LABEL is `<pipeline>[+bt]`:

  pipeline  default | asis | fused | fused_mw1 | fused_mw8 (`_zarr3_levers.PIPELINES`)
  +bt       numcodecs.blosc.use_threads = True after import

The label is applied after the package is imported, so a checkout that
configures zarr at import time is overridden only where the label says so;
`asis` applies nothing and measures the checkout as committed. Each child
records the settings that actually took effect, so a label that did not take
hold is visible rather than silently mislabelled.

Output: one JSON line per process appended to `<output-dir>/attribution.jsonl`,
`{"config", "round", "load_1m_before", "load_1m_after", "result"}`, and each
configuration's stderr in `<output-dir>/<config>.err`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any


def _levers() -> ModuleType:
    """`_zarr3_levers`, loaded from beside this script.

    The child runs under the measured checkout's environment, whose `benchmarks`
    has no copy of the module, so `from benchmarks import _zarr3_levers` would
    fail there.
    """
    spec = importlib.util.spec_from_file_location(
        "_zarr3_levers", Path(__file__).with_name("_zarr3_levers.py")
    )
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load _zarr3_levers.py from beside this script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _time_shapes(
    shapes: dict[str, Any], reps: int, levers: ModuleType
) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    med: dict[str, float] = {}
    p95: dict[str, float] = {}
    rows: dict[str, int] = {}
    for name, fn in shapes.items():
        n = reps if name != "bulk" else 2
        first = fn()
        rows[name] = len(next(iter(first.values()))) if first else 0
        samples = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1000.0)
        med[name] = round(statistics.median(samples), 2)
        p95[name] = round(levers.p95(samples), 2)
    return med, p95, rows


def child(store: Path, selection_path: Path, label: str, reps: int, with_bulk: bool) -> None:
    """Time the shapes once in this process and print one JSON line."""
    sys.path.insert(0, str(Path.cwd()))
    import zarr

    levers = _levers()
    pipeline, _, bt = label.partition("+")
    # The package and the shared shapes are imported before the label applies,
    # as in the measured runs: the label overrides an import-time configuration.
    importlib.import_module("benchmarks._query_shapes")
    import numcodecs.blosc

    from opengwasdb.query import query_store

    if zarr.__version__.startswith("3") and pipeline != "asis":
        zarr.config.set(levers.PIPELINES[pipeline])
    if bt == "bt":
        numcodecs.blosc.use_threads = True  # undo zarr 3's import-time False

    selection = levers.load_selection(selection_path)
    effective: dict[str, object] = {"use_threads": numcodecs.blosc.use_threads}
    with query_store(store) as q:
        shapes = levers.lever_shapes(q, selection, with_bulk=with_bulk)
        med, p95, rows = _time_shapes(shapes, reps, levers)
        if zarr.__version__.startswith("3"):
            z = zarr.open_group(str(store / "data.zarr"), mode="r")["z"]
            effective["pipeline"] = type(z._async_array.codec_pipeline).__name__
            effective["codec_pipeline.max_workers"] = zarr.config.get(
                "codec_pipeline.max_workers", None
            )
    effective["use_threads_after"] = numcodecs.blosc.use_threads
    result = {
        "label": label,
        "zarr": zarr.__version__,
        "cwd": str(Path.cwd()),
        "effective": effective,
        "ms": med,
        "p95_ms": p95,
        "rows": rows,
    }
    print(json.dumps(result))


def _load_1m() -> float:
    return float(Path("/proc/loadavg").read_text().split()[0])


def _parse_config(text: str) -> tuple[str, str, Path, Path]:
    name, _, rest = text.partition("=")
    parts = rest.split(":")
    if not name or len(parts) != 3:
        raise SystemExit(f"--config {text!r}: expected NAME=LABEL:CHECKOUT:PYTHON")
    label, checkout, python = parts
    return name, label, Path(checkout).resolve(), Path(python).resolve()


def drive(args: argparse.Namespace) -> None:
    """Run every configuration once per round, interleaved, one process each."""
    configs = [_parse_config(text) for text in args.config]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "attribution.jsonl"
    env = {**os.environ, "PYTHONNOUSERSITE": "1"}
    print(f"start: load {_load_1m()}", flush=True)
    for round_no in range(1, args.rounds + 1):
        for name, label, checkout, python in configs:
            before = _load_1m()
            with open(args.output_dir / f"{name}.err", "a", encoding="utf-8") as err:
                proc = subprocess.run(
                    [
                        str(python),
                        str(Path(__file__).resolve()),
                        "--child",
                        str(args.store.resolve()),
                        str(args.selection.resolve()),
                        label,
                        str(args.reps),
                        "1" if args.with_bulk else "0",
                    ],
                    cwd=checkout,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=err,
                    text=True,
                )
            lines = proc.stdout.strip().splitlines()
            result = json.loads(lines[-1]) if proc.returncode == 0 and lines else None
            record = {
                "config": name,
                "round": round_no,
                "load_1m_before": before,
                "load_1m_after": _load_1m(),
                "result": result,
            }
            with open(out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
            print(f"{name} round {round_no} done (load {before} -> {record['load_1m_after']})")
            if result is None:
                raise SystemExit(f"{name} round {round_no} failed; see {name}.err")
    print(f"end: load {_load_1m()}; wrote {out}", flush=True)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        store, selection, label, reps, with_bulk = sys.argv[2:7]
        child(Path(store), Path(selection), label, int(reps), with_bulk == "1")
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--store", type=Path, required=True, help="the Store Release to read")
    ap.add_argument(
        "--selection",
        type=Path,
        required=True,
        help="a #242 store-comparison artifact whose `selection` block gives the queries",
    )
    ap.add_argument(
        "--config",
        action="append",
        required=True,
        help="NAME=LABEL:CHECKOUT:PYTHON; repeat, in the order to interleave",
    )
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--reps", type=int, default=7, help="timed reps per shape (bulk: 2)")
    ap.add_argument("--with-bulk", action="store_true", help="also time one whole Analysis")
    ap.add_argument("--output-dir", type=Path, required=True)
    drive(ap.parse_args())


if __name__ == "__main__":
    main()
