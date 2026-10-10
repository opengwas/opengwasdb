#!/usr/bin/env python3
"""Index-build cost per store for #252: wall time, peak RSS and disk (ruling c).

ADR 0060 estimates the index's storage, build time and query cost; this harness
replaces the build half with measurements. For each store it rebuilds the
variant-centric index from the finished Analysis-sorted planes with
`add_variant_index(force=True)`, sampling peak RSS on a background thread and
recording the wall time and the bytes the new `ragged/by_variant/` group adds.

Each store is measured in one process after a warm-up open, so the sampled peak
is that store's build and not a previous store's. The artifact is re-run, never
hand-edited:

    pixi run -e dev python benchmarks/measure_variant_index_cost.py \
        --store OGS-00001=/data/opengwasdb/work/epic252/OGS-00001-0.2.0 \
        --store OGS-00011=/data/opengwasdb/work/epic252/OGS-00011-0.2.0 \
        --output docs/benchmark-output/opengwasdb_252_index_cost.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from benchmarks._artifact import add_labelled_store_option, labelled_stores, provenance
from benchmarks._rss import RssSampler, rss_mb
from opengwasdb.layouts.ragged.by_variant import BY_VARIANT_GROUP, add_variant_index
from opengwasdb.layouts.ragged.zarr_csr import RAGGED_ZARR_PATH


def _directory_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def measure_one(label: str, store: Path) -> dict:
    """Rebuild one store's index and record wall time, peak RSS and disk."""
    index_dir = store / RAGGED_ZARR_PATH / BY_VARIANT_GROUP
    before = _directory_bytes(index_dir) if index_dir.exists() else 0
    baseline = rss_mb()
    with RssSampler() as sampler:
        started = time.perf_counter()
        result = add_variant_index(store, force=True)
        elapsed = time.perf_counter() - started
    peak = max(sampler.peak_mb, rss_mb())
    after = _directory_bytes(index_dir)
    return {
        "store": str(store),
        "n_axis": result.n_axis,
        "n_rows": result.n_rows,
        "build_seconds": round(elapsed, 1),
        "peak_rss_gib": round(peak / 1024.0, 3),
        "baseline_rss_gib": round(baseline / 1024.0, 3),
        "index_bytes": after,
        "index_gib": round(after / 2**30, 3),
        "previous_index_gib": round(before / 2**30, 3),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_labelled_store_option(parser)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    stores = labelled_stores(args.store)
    rows = [{"label": label, **measure_one(label, path)} for label, path in stores]
    artifact = {
        "harness": "benchmarks/measure_variant_index_cost.py",
        **provenance(),
        "stores": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "stores": [row["label"] for row in rows]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
