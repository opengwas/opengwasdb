#!/usr/bin/env python3
"""How many times does a Ragged build rewrite each association-sequence shard?

#249 investigates the write amplification of the 1-D Ragged sequences. A Zarr
v3 shard is one file, so a write that covers part of it reads, re-encodes and
rewrites the whole shard. With the decided 50,000,000-element sequence shard
and the 4,194,304-cell flush region, a build rewrote every full shard about
twelve times -- invisible on the small registered pilots, which fit in one shard.

This script builds a **synthetic** Ragged component in process (no registered
Store is opened) under `opengwasdb.store.arrays.count_shard_writes`, so the real
bytes the storage layer was handed are attributable to each array, and reports
per array: shard count, bytes written, final bytes on disk, the ratio (write
amplification) and the most writes any one shard received. `--legacy-step`
reproduces the pre-#249 region step on the same code path by replacing
`sequence_region_step`; the two runs together are the fix's before/after.

  pixi run -e dev python benchmarks/measure_write_amplification.py \
      --work /tmp/epic240/249/amp-fixed --output /tmp/epic240/249/amp-fixed.json
  pixi run -e dev python benchmarks/measure_write_amplification.py \
      --legacy-step --work /tmp/epic240/249/amp-legacy \
      --output /tmp/epic240/249/amp-legacy.json
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks._artifact import commit
from opengwasdb.encoding import EncodingMeasurements, StoreEncoding
from opengwasdb.layouts.ragged import zarr_csr
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRWriter
from opengwasdb.store.arrays import (
    ASSOCIATION_SEQUENCE_CHUNK,
    RAGGED_SEQUENCE_SHARD_ELEMENTS,
    count_shard_writes,
)


def _synthetic_writer(
    total_cells: int, n_analyses: int, n_variants: int, seed: int
) -> RaggedCSRWriter:
    """A writer holding `total_cells` associations across `n_analyses` Analyses.

    Indices are sorted within an Analysis, as a real builder's are, and the EAF
    column is present so the residual `eaf`/`se` planes are written too.
    """
    rng = np.random.default_rng(seed)
    per_analysis = total_cells // n_analyses
    writer = RaggedCSRWriter(n_variants)
    for _ in range(n_analyses):
        writer.add_analysis(
            np.sort(rng.integers(0, n_variants, size=per_analysis)).astype(np.int32),
            rng.standard_normal(per_analysis).astype(np.float32),
            np.abs(rng.standard_normal(per_analysis)).astype(np.float32),
            rng.random(per_analysis).astype(np.float32),
        )
    return writer


def _plan(writer: RaggedCSRWriter, n_analyses: int) -> StoreEncoding:
    """The two-pass plan decision a Ragged builder makes (ADR 0037)."""
    eaf = writer.eaf_measurements()
    preliminary = StoreEncoding.decide(EncodingMeasurements(n_analyses=n_analyses, eaf=eaf))
    return StoreEncoding.decide(
        EncodingMeasurements(
            n_analyses=n_analyses, eaf=eaf, se=writer.se_measurements(preliminary)
        )
    )


def _final_bytes(root: Path) -> dict[str, int]:
    """On-disk bytes per array under the store's `ragged` group."""
    sizes: dict[str, int] = {}
    for array_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        sizes[array_dir.name] = sum(f.stat().st_size for f in array_dir.rglob("*") if f.is_file())
    return sizes


def _array_report(
    name: str, counts: dict[str, int], written: int, final: int
) -> dict[str, Any]:
    return {
        "shards": len(counts),
        "shard_writes": sum(counts.values()),
        "max_writes_one_shard": max(counts.values(), default=0),
        "bytes_written": written,
        "final_bytes": final,
        "amplification": round(written / final, 3) if final else None,
    }


def measure(args: argparse.Namespace) -> dict[str, Any]:
    if args.legacy_step:
        # The pre-#249 step: the region alone, ignoring the shard. Same code
        # path, same arrays; only the write granularity differs.
        def legacy_step(total: int, region_cells: int) -> int:
            return max(1, int(region_cells))

        zarr_csr.sequence_region_step = legacy_step
    store = args.work / "data.zarr"
    if store.exists():
        raise SystemExit(f"{store} exists; remove it or choose another --work")
    writer = _synthetic_writer(args.total_cells, args.n_analyses, args.n_variants, args.seed)
    encoding = _plan(writer, args.n_analyses)
    started = time.monotonic()
    with count_shard_writes() as recorder:
        writer.flush(args.work, encoding, region_cells=args.region_cells)
    wall = time.monotonic() - started
    peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    written = recorder.bytes_written_by_array()
    counts = recorder.chunk_writes()
    final = _final_bytes(store / "ragged")
    arrays = {
        name: _array_report(name, counts.get(name, {}), written.get(name, 0), final.get(name, 0))
        for name in sorted(set(final) | set(counts))
    }
    total_written = sum(written.values())
    total_final = sum(final.values())
    return {
        "artifact": "Ragged sequence write amplification, epic #240 / #249",
        "commit": commit(),
        "measured_at": datetime.now(UTC).isoformat(),
        "legacy_step": bool(args.legacy_step),
        "shard_elements": RAGGED_SEQUENCE_SHARD_ELEMENTS,
        "inner_chunk": ASSOCIATION_SEQUENCE_CHUNK,
        "region_cells": args.region_cells,
        "total_cells": args.total_cells,
        "n_analyses": args.n_analyses,
        "n_variants": args.n_variants,
        "encoding": encoding.to_manifest(),
        "wall_seconds": round(wall, 2),
        "peak_rss_kib": peak_rss_kib,
        "peak_rss_gib": round(peak_rss_kib / (1024 * 1024), 2),
        "total_bytes_written": total_written,
        "total_final_bytes": total_final,
        "total_amplification": round(total_written / total_final, 3) if total_final else None,
        "arrays": arrays,
        "note": (
            "bytes_written counts every byte handed to the storage layer while the "
            "component flushed; final_bytes is what is on disk afterwards. A shard "
            "written once has amplification near 1; a max_writes_one_shard above 1 "
            "is the read-modify-write this ticket fixes."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-cells", type=int, default=160_000_000)
    parser.add_argument("--n-analyses", type=int, default=32)
    parser.add_argument("--n-variants", type=int, default=20_000_000)
    parser.add_argument("--region-cells", type=int, default=1 << 22)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--legacy-step",
        action="store_true",
        help="reproduce the pre-#249 region step (region only, not the shard)",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    payload = measure(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}: amplification {payload['total_amplification']}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
