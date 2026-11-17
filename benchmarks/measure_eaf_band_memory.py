#!/usr/bin/env python3
"""What one residual `eaf` band worker holds, as a function of the band's size (#320).

The Dense build caps the encode's pool by a quarter of the host's available
memory, and it needs a per-band figure to divide that budget by. The figure is
**fixed plus per cell** rather than one byte-per-cell rate: a worker forked from
the build inherits the parent's resident set, and only the band's own cells
scale with the band. This harness measures that shape on a real release:

* a **real band** of `OGS-00009`, decoded from its stored `int8` plane,
  baseline, exception table and panel frequencies into the `float32` grid the
  build stages -- so the band carries real NaNs and real exception cells;
* the same at a **tenth** the rows, the second point the fit needs;
* each measurement runs the production worker, `_encode_eaf_band`, in a forked
  child, and reads that child's `VmHWM` from `/proc/<pid>/status` while it is
  alive. `ru_maxrss` cannot be used: it is a lifetime high-water mark that also
  counts the launcher's footprint, which a forked worker never adds.

The band is staged into `--scratch` by a separate process, so the measuring
process never holds the decode.

  pixi run -e dev python benchmarks/measure_eaf_band_memory.py \\
      --store /data/opengwasdb/stores/OGS-00009/store.opengwasdb \\
      --scratch /data/opengwasdb/work/orchestrate/opengwasdb-307/scratch/320/eaf-band \\
      --repeats 2 \\
      --out docs/benchmark-output/opengwasdb_320_eaf_band/eaf_band_memory.json
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from benchmarks._artifact import artifact_header, host_load, write_artifact
from opengwasdb.encoding import (
    EafExceptionTable,
    StoreCodec,
    StoreEncoding,
    positions_row_band,
)
from opengwasdb.layouts.dense.build_vcf import (
    _eaf_row_band,
    _EafBandContext,
    _encode_eaf_band,
)

#: Rows of the second band, as a fraction of the production band.
_SMALLER_BAND_FRACTION = 10


def _vm_hwm_bytes(pid: int) -> int:
    """A live process's `VmHWM` in bytes; `VmHWM` is its own RSS high-water mark."""
    text = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    raise SystemExit(f"/proc/{pid}/status carries no VmHWM; the peak cannot be reported")


def _vm_rss_bytes(pid: int) -> int:
    text = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise SystemExit(f"/proc/{pid}/status carries no VmRSS")


def _plan(store: Path) -> StoreEncoding:
    manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
    return StoreEncoding.from_manifest(manifest["encoding"])


def _stage_grid(store: Path, rows: int, destination: Path) -> dict[str, Any]:
    """Decode `rows` variants of the release's stored frequencies into a `float32` grid.

    The build's staging plane holds the frequency a source reported -- exactly,
    or as the residual coding quantises it -- with NaN where no Analysis
    reported one, and the panel's frequency on imputed cells. This reconstructs
    that from what the release stores, so the band a worker is measured on is
    the band a build of it would hand over.
    """
    root = zarr.open_group(str(store / "data.zarr"), mode="r")
    n_analyses = int(root["eaf"].shape[1])
    plan = _plan(store)
    codec = StoreCodec(plan, eaf_exceptions=EafExceptionTable.open(root))
    codes = np.asarray(root["eaf"][0:rows], dtype=np.int8)
    baseline = np.repeat(
        np.asarray(root["eaf_baseline"][0:rows], dtype=np.float32)[:, None],
        n_analyses,
        axis=1,
    )
    imputed = (
        np.asarray(root["imputed"][0:rows], dtype=bool)
        if plan.eaf.reference and "imputed" in root
        else None
    )
    reference = (
        np.repeat(
            np.asarray(root["eaf_reference"][0:rows], dtype=np.float32)[:, None],
            n_analyses,
            axis=1,
        )
        if imputed is not None
        else None
    )
    grid = codec.decode_eaf(
        codes,
        baseline=baseline,
        positions=positions_row_band(0, n_analyses),
        imputed=imputed,
        reference=reference,
    )
    exceptions = int(np.count_nonzero(codes == -127))
    absent = int(np.count_nonzero(np.isnan(grid)))
    del codes, baseline, imputed, reference
    gc.collect()

    if destination.exists():
        raise SystemExit(f"{destination}: already exists; refusing to overwrite")
    array = zarr.open_array(str(destination), mode="w", zarr_format=3, shape=grid.shape,
                           dtype="float32", chunks=grid.shape)
    array[:] = grid
    written = grid.shape
    del grid, array
    gc.collect()
    return {"rows": rows, "shape": list(written), "exceptions": exceptions, "absent": absent}


def _encode_in_child(context: _EafBandContext, done: Any, release: Any) -> None:
    """The production band worker, held alive so its parent can read `VmHWM`."""
    _encode_eaf_band(context, 0)
    done.set()
    release.wait()


def _measure(grid_path: Path, rows: int, n_analyses: int, store: Path) -> dict[str, Any]:
    """One forked worker's peak RSS on the staged band."""
    grid = zarr.open_array(str(grid_path), mode="r")
    codec = StoreCodec(_plan(store))
    context = _EafBandContext(
        staging=grid, codec=codec, n_analyses=n_analyses, band_rows=rows, n_variants=rows
    )
    fork = multiprocessing.get_context("fork")
    done, release = fork.Event(), fork.Event()
    started = time.monotonic()
    child = fork.Process(target=_encode_in_child, args=(context, done, release))
    child.start()
    if not done.wait(timeout=1800):
        child.kill()
        raise SystemExit("the band worker did not finish inside 1800 s")
    elapsed = time.monotonic() - started
    peak, rss = _vm_hwm_bytes(child.pid), _vm_rss_bytes(child.pid)
    release.set()
    child.join(timeout=120)
    return {
        "rows": rows,
        "cells": rows * n_analyses,
        "encode_seconds": round(elapsed, 2),
        "rss_at_peak_bytes": rss,
        "vm_hwm_bytes": peak,
    }


def _fit(points: list[dict[str, Any]]) -> dict[str, Any]:
    """Least squares `fixed + per_cell x cells` over the measured bands."""
    cells = np.array([point["cells"] for point in points], dtype=np.float64)
    peak = np.array([point["vm_hwm_bytes"] for point in points], dtype=np.float64)
    per_cell, fixed = np.polyfit(cells, peak, 1)
    predicted = fixed + per_cell * cells
    return {
        "fixed_bytes": int(round(fixed)),
        "per_cell_bytes": round(float(per_cell), 3),
        "residual_bytes": [int(round(value)) for value in peak - predicted],
        "residual_fraction": [round(float(value), 5) for value in (peak - predicted) / peak],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", type=Path, required=True, help="a Dense release root")
    parser.add_argument("--scratch", type=Path, required=True, help="where the staged bands go")
    parser.add_argument("--repeats", type=int, default=2, help="workers per band size")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", nargs=2, metavar=("ROWS", "DESTINATION"), help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.stage is not None:
        rows, destination = int(args.stage[0]), Path(args.stage[1])
        print(json.dumps(_stage_grid(args.store, rows, destination)))
        return 0

    header = artifact_header("benchmarks/measure_eaf_band_memory.py", sys.argv[1:])
    root = zarr.open_group(str(args.store / "data.zarr"), mode="r")
    n_analyses = int(root["eaf"].shape[1])
    shard_rows = int(root["eaf"].shards[0])
    band_rows = _eaf_row_band(shard_rows, n_analyses)
    sizes = [band_rows, max(1, band_rows // _SMALLER_BAND_FRACTION)]
    args.scratch.mkdir(parents=True, exist_ok=True)

    staged: list[dict[str, Any]] = []
    points: list[dict[str, Any]] = []
    for rows in sizes:
        grid_path = args.scratch / f"grid-{rows}.zarr"
        produced = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--store",
                str(args.store),
                "--scratch",
                str(args.scratch),
                "--out",
                str(args.out),
                "--stage",
                str(rows),
                str(grid_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if produced.returncode != 0:
            raise SystemExit(f"staging {rows} rows failed:\n{produced.stdout}\n{produced.stderr}")
        staged.append(json.loads(produced.stdout.strip().splitlines()[-1]))
        for _ in range(args.repeats):
            points.append(_measure(grid_path, rows, n_analyses, args.store))

    fit = _fit(points)
    payload = {
        **header,
        "store": str(args.store),
        "n_analyses": n_analyses,
        "band_rule": {"shard_rows": shard_rows, "band_rows": band_rows},
        "staged_bands": staged,
        "workers": points,
        "fit": fit,
        "host_load_end": host_load(),
    }
    write_artifact(args.out, payload)
    for point in points:
        cells = point["cells"]
        estimate = fit["fixed_bytes"] + fit["per_cell_bytes"] * cells
        print(
            f"{point['rows']:>8} rows  {cells/1e6:>7.1f}M cells  "
            f"peak {point['vm_hwm_bytes']/1024**3:>6.2f} GiB  "
            f"estimate {estimate/1024**3:>6.2f} GiB  {point['encode_seconds']:>6.2f} s"
        )
    print(
        f"fit: fixed {fit['fixed_bytes']/1024**2:.0f} MiB + "
        f"{fit['per_cell_bytes']:.1f} B a cell"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
