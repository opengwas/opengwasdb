"""Does the Blosc-thread lever move the SE encoding plan? Real OGS-00009 data (#244).

The plan (`optimise_dense_se`, ADR 0037) is chosen from compressed sizes taken
with numcodecs' Blosc. With `n_workers=1` those sizes are measured in the
parent, where #244 makes Blosc multi-threaded; with `n_workers>1` they are
measured in forked workers, which numcodecs always runs single-threaded. This
stages a real slice the way a Dense build stages it -- a float32 `se` plane and
a float32 `eaf` plane -- runs the production chooser, and prints the plan, a
hash of the stored codes and the bytes of the rewritten plane. Every label and
worker count must print the same plan, hash and bytes.

  off   numcodecs.blosc.use_threads = False (zarr 3's default)
  on    numcodecs.blosc.use_threads = True
  asis  whatever the checkout's seam sets

Run from the checkout under test, which is put first on `sys.path`. The fixture
stores cannot answer this: they have too few cells for the residual SE path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

#: Below this many finite SE cells the chooser's answer would not be meaningful.
MIN_FINITE_CELLS = 1_000_000


def stage(store: Path, scratch: Path, start: int, n_rows: int) -> tuple[Path, int]:
    """Write the slice's decoded `se` and `eaf` as float32 planes; returns (group, finite SE)."""
    from opengwasdb.query import query_store
    from opengwasdb.store import arrays
    from opengwasdb.store.arrays import ArrayRole

    with query_store(store) as q:
        rows = np.arange(start, start + n_rows, dtype=np.int64)
        se = np.asarray(q._se.rows(rows), dtype=np.float32)
        eaf = np.asarray(q._eaf.band(start, start + n_rows), dtype=np.float32)
    finite = int(np.isfinite(se).sum())
    if finite <= MIN_FINITE_CELLS:
        raise SystemExit(f"{finite:,} finite SE cells: the slice must carry real SE values")
    group = arrays.open_group_for_write(scratch, "w")
    for name, plane in (("se", se), ("eaf", eaf)):
        arrays.create_array(
            group,
            name,
            ArrayRole.DENSE_STATISTIC_PLANE,
            data=plane,
            hint=(1000, 1000),
            fill_value=np.nan,
        )
    return scratch, finite


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("label", choices=("off", "on", "asis"))
    ap.add_argument("n_workers", type=int)
    ap.add_argument("--store", type=Path, required=True, help="OGS-00009's store.opengwasdb")
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--start", type=int, default=6_500_000)
    ap.add_argument("--rows", type=int, default=50_000)
    args = ap.parse_args()
    sys.path.insert(0, str(Path.cwd()))
    logging.basicConfig(level=logging.WARNING)
    import numcodecs.blosc

    from opengwasdb.encoding import optimise_dense_se
    from opengwasdb.encoding.plan import StoreEncoding
    from opengwasdb.store import arrays

    if args.label != "asis":
        numcodecs.blosc.use_threads = args.label == "on"
    path = args.scratch / f"se-plan-{args.label}-n{args.n_workers}.zarr"
    path, finite = stage(args.store, path, args.start, args.rows)
    encoding = StoreEncoding.from_manifest(
        {
            "version": 3,
            "z": {"kind": "int16_fixed", "scale": 1024},
            "se": {"kind": "float16"},
            "eaf": {"kind": "float32"},
        }
    )
    t0 = time.perf_counter()
    selected = optimise_dense_se(arrays.open_group(path, "r+"), encoding, n_workers=args.n_workers)
    elapsed = time.perf_counter() - t0
    stored = np.asarray(arrays.open_group(path)["se"][:])
    chunk_files = sorted(p for p in (path / "se").iterdir() if not p.name.startswith("."))
    result = {
        "label": args.label,
        "n_workers": args.n_workers,
        "use_threads": numcodecs.blosc.use_threads,
        "finite_se_cells": finite,
        "plan": selected.to_manifest(),
        "stored_dtype": str(stored.dtype),
        "stored_sha": hashlib.sha256(stored.tobytes()).hexdigest()[:16],
        "se_chunk_bytes": sum(p.stat().st_size for p in chunk_files),
        "se_chunk_files": len(chunk_files),
        "s": round(elapsed, 1),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
