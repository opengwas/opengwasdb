#!/usr/bin/env python3
"""Measure, read-only, what removing the Hybrid duplication costs a backfill (ADR 0102 §4, #463).

    hybrid_backfill.py RELEASE OUT_JSON

For a Hybrid Store Release, the Dense Component (`dense/`) carries its own variant table and
identifier sidecars. Under #472 it would read them through `dense/dense_to_shared.npy` and the
shared (top-level) sidecars instead. This script times the work a rewrite would do to prove that
is lossless, and counts the bytes dropped:

  1. derive the Dense Component's ALID and rsID search indexes from the shared ones and
     dense_to_shared, and check they equal the Dense Component's own (exact array equality);
  2. walk both variant tables and check each Dense Component row equals the shared row
     dense_to_shared names, on every column but variant_index (this covers long ALIDs,
     which the S32 index leaves out, and source_alid, which no sidecar carries yet).

Imports numpy only, not opengwasdb.
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def timed(record, name, fn):
    t0 = time.perf_counter()
    out = fn()
    record[name] = round(time.perf_counter() - t0, 2)
    return out


def derive_index(shared_keys, shared_rows, d2s):
    """The Dense Component's (key, dense row) index, from the shared one and dense_to_shared."""
    pos = np.searchsorted(d2s, shared_rows)
    safe = np.minimum(pos, d2s.size - 1)
    on = d2s[safe] == shared_rows
    return shared_keys[on], pos[on].astype(np.int32)


def index_check(rel, which, seconds, d2s):
    shared_keys = np.load(rel / f"variant_{which}_bytes.npy")
    shared_rows = np.load(rel / f"variant_{which}_rows.npy")
    own_keys = np.load(rel / "dense" / f"variant_{which}_bytes.npy")
    own_rows = np.load(rel / "dense" / f"variant_{which}_rows.npy")
    keys, rows = timed(
        seconds, f"derive_{which}_index", lambda: derive_index(shared_keys, shared_rows, d2s)
    )
    same = bool(
        keys.shape == own_keys.shape
        and np.array_equal(keys, own_keys.astype(keys.dtype))
        and np.array_equal(rows, own_rows)
    )
    return {"own_entries": int(own_keys.size), "derived_entries": int(keys.size), "equal": same}


def lines(path):
    with gzip.open(path, "rt", newline="\n") as fh:
        for line in fh:
            if not line.startswith("#"):
                yield line.rstrip("\n").split("\t")


def table_walk(rel, d2s):
    """Each Dense Component row against the shared row dense_to_shared names."""
    shared = lines(rel / "variants.tsv.gz")
    dense = lines(rel / "dense" / "variants.tsv.gz")
    mismatches = long_alids = n = 0
    examples = []
    cursor = -1
    row = None
    for i, drow in enumerate(dense):
        target = int(d2s[i])
        while cursor < target:
            row = next(shared)
            cursor += 1
        if int(drow[2]) != i or int(row[2]) != target:
            raise SystemExit(f"variant_index column out of step at dense row {i}")
        a = drow[:2] + drow[3:]
        b = row[:2] + row[3:]
        if a != b:
            mismatches += 1
            if len(examples) < 5:
                examples.append({"dense": drow, "shared": row})
        if len(drow[5]) > 32:
            long_alids += 1
        n += 1
    return {
        "dense_rows": n, "mismatches": mismatches, "long_alids": long_alids, "examples": examples
    }


DROPPED = [
    "variants.tsv.gz", "variants.tsv.gz.tbi", "variant_offsets.npy",
    "variant_alid_bytes.npy", "variant_alid_rows.npy",
    "variant_rsid_bytes.npy", "variant_rsid_rows.npy", "analyses.tsv",
]


def main():
    rel, out = Path(sys.argv[1]), Path(sys.argv[2])
    seconds: dict[str, float] = {}
    d2s = timed(
        seconds, "load_dense_to_shared", lambda: np.load(rel / "dense" / "dense_to_shared.npy")
    )
    result = {
        "release": str(rel),
        "dense_rows": int(d2s.size),
        "shared_rows": int(np.load(rel / "variant_offsets.npy", mmap_mode="r").shape[0]),
        "dense_to_shared_strictly_ascending": bool((np.diff(d2s) > 0).all()),
        "alid_index": index_check(rel, "alid", seconds, d2s),
        "rsid_index": index_check(rel, "rsid", seconds, d2s),
    }
    result["table"] = timed(seconds, "table_walk", lambda: table_walk(rel, d2s))
    sizes = {name: os.stat(rel / "dense" / name).st_size for name in DROPPED}
    result["dropped_bytes"] = sizes
    result["dropped_total"] = sum(sizes.values())
    result["seconds"] = seconds
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "table"}, indent=1))
    print(json.dumps({k: v for k, v in result["table"].items() if k != "examples"}))


if __name__ == "__main__":
    main()
