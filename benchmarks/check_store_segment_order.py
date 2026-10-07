#!/usr/bin/env python3
"""Check one store-format rule over registered stores: segment ordering (#252).

`variant_index` must be non-decreasing within every Analysis's segment of a
standalone Ragged CSR or a Hybrid Overflow CSR (spec §11, §20). This is the
release-evidence runner for that rule: it runs **only** the ordering check --
not the whole of `validate_store` -- over every Ragged and Hybrid Store Release
under a root, and reports per Store the segments checked, the entries read, the
wall time and the peak RSS, so a one-time full pass is bounded and honest.

Usage:
  pixi run -e dev python benchmarks/check_store_segment_order.py \
      --root /data/opengwasdb/stores \
      --output /tmp/epic240/252/segment_order.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from benchmarks._rss import RssSampler, rss_mb
from opengwasdb.model.enums import PrimaryStorageLayout
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.store.arrays import open_group
from opengwasdb.validation.validate import _segment_order_errors


def _layouts(root: Path) -> list[tuple[str, str]]:
    """Every `(name, layout)` under `root` whose layout has a Ragged CSR."""
    found: list[tuple[str, str]] = []
    for store_root in sorted(root.glob("OGS-*")):
        store = store_root / "store.opengwasdb"
        if not store.is_dir():
            continue
        # A manifest this build cannot read is a failure, not a store to skip:
        # an unreadable Ragged or Hybrid release would otherwise go unchecked.
        manifest = StoreManifest.load(store)
        layout = manifest.primary_layout
        if layout in (PrimaryStorageLayout.RAGGED, PrimaryStorageLayout.HYBRID):
            found.append((store_root.name, layout.value))
    return found


def _check_one(store: Path) -> dict[str, object]:
    """Run the ordering rule on one store's Ragged CSR; return its record."""
    group_path = store / "data.zarr" / "ragged"
    baseline_mb = rss_mb()
    with RssSampler(interval=0.02) as sampler:
        started = time.perf_counter()
        root = open_group(group_path)
        offsets = np.asarray(root["offsets"][:], dtype=np.int64)
        n_assoc = int(offsets[-1]) if len(offsets) else 0
        errors: list[str] = []
        _segment_order_errors(root, offsets, n_assoc, errors, "data.zarr/ragged")
        elapsed = time.perf_counter() - started
    return {
        "segments_checked": max(len(offsets) - 1, 0),
        "entries_read": n_assoc,
        "wall_s": round(elapsed, 3),
        "baseline_mb": round(baseline_mb, 1),
        "peak_mb": round(sampler.peak_mb, 1),
        "delta_mb": round(sampler.peak_mb - baseline_mb, 1),
        "ok": not errors,
        "errors": errors,
    }


def _run_one(store: Path) -> dict[str, object]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
    out = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--one-store", str(store)],
        capture_output=True,
        text=True,
        env=env,
    )
    if out.returncode != 0:
        raise SystemExit(f"{store} failed:\n{out.stdout}\n{out.stderr}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/data/opengwasdb/stores"))
    ap.add_argument("--one-store", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    if args.one_store is not None:
        record = _check_one(args.one_store)
        record["store"] = str(args.one_store)
        print(json.dumps(record), flush=True)
        return

    artifact: dict[str, object] = {
        "rule": "variant_index is non-decreasing within every Analysis's segment (spec §11, §20)",
        "root": str(args.root),
        "note": (
            "Runs only the segment-ordering rule, not the whole of validate_store. "
            "Peak RSS is the process peak while the rule ran, including opening the "
            "store's offsets; the array is read in 1,000,000-cell windows and never "
            "materialised."
        ),
        "stores": {},
    }
    for name, layout in _layouts(args.root):
        store = args.root / name / "store.opengwasdb"
        record = _run_one(store)
        record["layout"] = layout
        artifact["stores"][name] = record
        print(
            f"{name:12s} {layout:7s} segments={record['segments_checked']:>8} "
            f"entries={record['entries_read']:>12,} wall={record['wall_s']:8.3f}s "
            f"peak={record['peak_mb']:9.1f}MB ok={record['ok']}",
            flush=True,
        )
    artifact["all_ok"] = all(
        store.get("ok", True) for store in artifact["stores"].values()
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    if not artifact["all_ok"]:
        raise SystemExit("a store failed the segment-ordering rule")


if __name__ == "__main__":
    main()
