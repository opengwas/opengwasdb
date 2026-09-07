#!/usr/bin/env python3
"""Phase-level timing of the format-3.0 SE passes on a real EAF-bearing plane.

Issue #144 asked which of the four passes over a Dense `se` plane -- the
per-Analysis fit, the candidate measurement, the rewrite, and the top-hit
index rebuild -- dominates the 3,863 s a full FinnGen R13 pilot migration took.
Issue #146 then bounded the measurement to a sample of row chunks, which
changes that split.

This benchmark measures the three `optimise_dense_se_joint` passes (fit,
measurement, rewrite+count) on **a column slice of a real store**: the first
`--analyses` Analyses of a Dense format-2.0 release, decoded from its stored
arrays, chunked exactly as the store chunks them. The top-hit rebuild is a
store-level pass (it re-reads the variant axis and every Analysis's retained
hits) and is not part of the plane rewrite, so it is reported separately when
a full migration runs; here it is out of scope and stated as such.

The slice keeps real values, real chunking and the real compressor, so the
per-phase seconds transfer to the full plane at the recorded cell count. Two
runs are reported: the pre-#146 exhaustive survey (`max-chunks 0` = no bound)
and the #146 sampled survey (`max-chunks 64`), with whether the two chose the
same encoding. Numbers are committed, never hand-edited (CONTRIBUTING).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import zarr

from opengwasdb.encoding.measure import SeMeasurementRecord
from opengwasdb.encoding.plan import EafEncoding, SeEncoding, StoreEncoding, ZEncoding
from opengwasdb.encoding.planes import DenseEafPlane
from opengwasdb.encoding.se import optimise_dense_se_joint
from opengwasdb.encoding.timing import PhaseTimer
from opengwasdb.model.manifest import StoreManifest


def _preliminary() -> StoreEncoding:
    return StoreEncoding(
        z=ZEncoding("int16_fixed", scale=1024),
        se=SeEncoding("float16"),
        eaf=EafEncoding("float32"),
    )


def _scratch_group(root: zarr.Group, n_analyses: int, tmp: Path) -> zarr.Group:
    """A fresh Dense plane holding the real store's first `n_analyses` columns."""
    se_src = root["se"]
    n_rows, _ = se_src.shape
    row_chunk = int(se_src.chunks[0])
    compressor = se_src.compressor
    group = zarr.open_group(str(tmp / "dense.zarr"), mode="w")
    group.create_dataset(
        "se",
        shape=(n_rows, n_analyses),
        chunks=(row_chunk, n_analyses),
        compressor=compressor,
        dtype="float32",
    )
    group.create_dataset(
        "eaf",
        shape=(n_rows, n_analyses),
        chunks=(row_chunk, n_analyses),
        compressor=compressor,
        dtype="float32",
    )
    group.create_dataset(
        "z",
        shape=(n_rows, n_analyses),
        chunks=(row_chunk, n_analyses),
        compressor=compressor,
        dtype="float16",
    )
    return group


def _fill(group: zarr.Group, root: zarr.Group, store_encoding, n_analyses: int) -> None:
    """Decode the store's se/eaf planes column-sliced into the scratch group.

    The scratch plane is decoded `float32`/`float32` -- including EAF, whose
    stored bytes on a 2.0 store are int8 logit residuals against a baseline
    and must be decoded, not copied, to be usable as frequencies.
    """
    se_src = root["se"]
    n_rows, _ = se_src.shape
    row_chunk = int(se_src.chunks[0])
    eaf_plane = DenseEafPlane.open(root, store_encoding)
    for r0 in range(0, n_rows, row_chunk):
        r1 = min(r0 + row_chunk, n_rows)
        group["se"][r0:r1] = np.asarray(se_src[r0:r1, :n_analyses], dtype=np.float32)
        decoded = eaf_plane.band(r0, r1)
        group["eaf"][r0:r1] = np.asarray(decoded[:, :n_analyses], dtype=np.float32)


def _run(group: zarr.Group, preliminary: StoreEncoding, cap: int, tmp: Path) -> dict:
    timer = PhaseTimer()
    record = SeMeasurementRecord()
    started = time.perf_counter()
    selected, _ = optimise_dense_se_joint(
        group, preliminary, timer=timer, record=record, measure_max_chunks=cap
    )
    wall = time.perf_counter() - started
    phases = [
        {"phase": name, "seconds": round(seconds, 3), "share": round(share, 5)}
        for name, seconds, share in timer.report()
    ]
    return {
        "wall_seconds": round(wall, 3),
        "phases": phases,
        "se": selected.se.to_manifest(),
        "measurement": record.to_manifest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("store", type=Path)
    parser.add_argument("--analyses", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/benchmark-output/opengwasdb_se_migration_phases.json"),
    )
    parser.add_argument("--scratch", type=Path, default=Path("/tmp/ogdb-se-phases"))
    args = parser.parse_args()

    manifest = StoreManifest.load(args.store)
    store_root = zarr.open_group(str(args.store / "data.zarr"), mode="r")
    n_rows, n_total = store_root["se"].shape
    n_analyses = min(args.analyses, int(n_total))

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    cells = n_rows * n_analyses
    print(
        f"Slicing {n_rows:,} rows x {n_analyses} of {n_total} analyses ({cells:,} cells)",
        flush=True,
    )

    results = {}
    store_encoding = manifest.encoding
    for cap, name in ((0, "exhaustive"), (64, "sampled")):
        tmp = args.scratch / name
        if tmp.exists():
            import shutil

            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        group = _scratch_group(store_root, n_analyses, tmp)
        started = time.perf_counter()
        _fill(group, store_root, store_encoding, n_analyses)
        fill_seconds = time.perf_counter() - started
        results[name] = _run(group, _preliminary(), cap if cap else 10**9, tmp)
        results[name]["column_slice_fill_seconds"] = round(fill_seconds, 3)

    payload = {
        "issue": "144/146",
        "store": str(args.store),
        "store_format": manifest.format_version,
        "commit": commit,
        "measured_at": datetime.now(UTC).isoformat(),
        "rows": int(n_rows),
        "analyses_total": int(n_total),
        "analyses_sliced": int(n_analyses),
        "cells_sliced": int(cells),
        "top_hits_phase": (
            "not measured here; a store-level pass run by migrate_store_to_format_3.py"
        ),
        "exhaustive": results["exhaustive"],
        "sampled": results["sampled"],
        "same_encoding": results["exhaustive"]["se"] == results["sampled"]["se"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
