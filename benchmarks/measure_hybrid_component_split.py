#!/usr/bin/env python3
"""Count each Analysis's associations in a Hybrid store's two components.

A Hybrid Store Release partitions every Analysis's associations disjointly
between the Dense Component (variants on the reference axis) and the Ragged
Overflow Component (the Analysis's off-axis variants). How an Analysis splits
between them is what decides which query path serves it: a whole-Analysis read
touches both, a PheWAS or lookup of an off-axis variant scans the whole
Overflow. Nothing in the store records the split, so this measures it:

* Ragged Overflow: the Analysis's CSR row length, read from `offsets`;
* Dense Component: the cells of the Analysis's column whose `z` code is not
  `Z_MISSING`, counted by a parallel scan of the whole `z` plane.

Each Analysis's two counts are set against the build's own record of what it
retained (`provenance.info_score.analyses[].associations_retained`). That
record counts rows before rows naming the same variant collapse into one cell,
so a store may hold fewer associations than it retained, never more. More is a
store that invented associations and fails the run; fewer is recorded per
Analysis as `n_not_stored`, so a large shortfall is visible rather than hidden
behind a plausible-looking split.

Usage:
  pixi run -e dev python benchmarks/measure_hybrid_component_split.py \
      --store /data/opengwasdb/stores/OGS-00011/store.opengwasdb \
      --output docs/benchmark-output/opengwasdb_ogs00011_component_split.tsv
"""

from __future__ import annotations

# Measures how each OGS-00011 Analysis splits between the Hybrid components.
import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import zarr

from benchmarks._artifact import provenance
from opengwasdb.encoding import Z_MISSING

# Row blocks span whole chunks, so no chunk is decompressed by two workers.
ROWS_PER_TASK_CHUNKS = 10


def _count_block(args: tuple[str, int, int]) -> np.ndarray:
    plane, start, stop = args
    z = zarr.open_array(plane, mode="r")
    return np.count_nonzero(z[start:stop, :] != Z_MISSING, axis=0).astype(np.int64)


def dense_counts(store: Path, n_workers: int) -> np.ndarray:
    plane = store / "dense" / "data.zarr" / "z"
    z = zarr.open_array(str(plane), mode="r")
    step = z.chunks[0] * ROWS_PER_TASK_CHUNKS
    tasks = [(str(plane), s, min(s + step, z.shape[0])) for s in range(0, z.shape[0], step)]
    total = np.zeros(z.shape[1], dtype=np.int64)
    started = time.perf_counter()
    with ProcessPoolExecutor(n_workers) as pool:
        for done, counts in enumerate(pool.map(_count_block, tasks, chunksize=4), start=1):
            total += counts
            if done % 200 == 0 or done == len(tasks):
                print(f"dense scan {done}/{len(tasks)} blocks "
                      f"({time.perf_counter() - started:.0f} s)", flush=True)
    return total


def ragged_counts(store: Path) -> np.ndarray:
    offsets = zarr.open_array(str(store / "data.zarr" / "ragged" / "offsets"), mode="r")[:]
    return np.diff(offsets.astype(np.int64))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--n-workers", type=int, default=32)
    args = ap.parse_args()

    manifest = json.loads((args.store / "manifest.json").read_text())
    retained = {
        row["analysis_id"]: row.get("associations_retained")
        for row in manifest["provenance"]["info_score"]["analyses"]
    }
    with open(args.store / "analyses.tsv", newline="") as fh:
        analyses = list(csv.DictReader(fh, delimiter="\t"))

    ragged = ragged_counts(args.store)
    dense = dense_counts(args.store, args.n_workers)
    if not len(ragged) == len(dense) == len(analyses):
        raise SystemExit(
            f"component widths disagree: dense {len(dense)}, ragged {len(ragged)}, "
            f"analyses.tsv {len(analyses)}"
        )

    invented = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as fh:
        out = csv.writer(fh, delimiter="\t", lineterminator="\n")
        out.writerow(["analysis_index", "analysis_id", "analysis_label", "sample_size",
                      "stored_effect_scale", "n_hits_5e8", "n_dense", "n_ragged", "n_total",
                      "dense_fraction", "associations_retained", "n_not_stored"])
        for row in analyses:
            i = int(row["analysis_index"])
            n_total = int(dense[i] + ragged[i])
            expected = retained.get(row["analysis_id"])
            if expected is not None and n_total > int(expected):
                invented.append((row["analysis_id"], n_total, int(expected)))
            out.writerow([i, row["analysis_id"], row["analysis_label"], row["sample_size"],
                          row["stored_effect_scale"], row["n_hits_5e8"],
                          int(dense[i]), int(ragged[i]), n_total,
                          f"{dense[i] / n_total:.6f}" if n_total else "",
                          "" if expected is None else expected,
                          "" if expected is None else int(expected) - n_total])

    retained_total = sum(int(v) for v in retained.values() if v is not None)
    summary = {
        "store": str(args.store),
        "n_analyses": len(analyses),
        "n_dense_associations": int(dense.sum()),
        "n_ragged_associations": int(ragged.sum()),
        "n_retained_by_build": retained_total,
        "n_not_stored": retained_total - int(dense.sum() + ragged.sum()),
        "n_stored_above_retained": len(invented),
        **provenance(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if invented:
        raise SystemExit(f"{len(invented)} Analysis(es) store more than the build retained, "
                         f"e.g. {invented[:3]}")


if __name__ == "__main__":
    main()
