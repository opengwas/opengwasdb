#!/usr/bin/env python3
"""Time and peak RSS of the variant-index validation alone (#252, round 2).

The full `validate_store` on OGS-00011 is #254's problem -- it decodes whole
Dense and Overflow planes -- so this measures the #252 addition on its own:
`_validate_variant_index` (presence, span, counts, ordering, layout, digest) on
one component.

    pixi run -e dev python benchmarks/measure_252_validate_index.py \
        --store /data/opengwasdb/work/epic252/OGS-00011-0.2.0 \
        --output docs/benchmark-output/opengwasdb_252_validate_index.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from benchmarks._artifact import provenance, write_artifact
from benchmarks._rss import RssSampler, rss_mb
from opengwasdb.store import open_store
from opengwasdb.store.arrays import open_group
from opengwasdb.validation import validate as v


def measure(store_path: Path) -> dict:
    store = open_store(store_path)
    ragged = store_path / "data.zarr" / "ragged"
    n_assoc = int(np.asarray(open_group(ragged)["offsets"][:], dtype=np.int64)[-1])
    n_axis = int(store.manifest.provenance["n_variants"])
    errors: list[str] = []
    baseline = rss_mb()
    with RssSampler() as sampler:
        started = time.perf_counter()
        v._validate_variant_index(
            store,
            ragged,
            n_assoc,
            errors,
            component_label="data.zarr/ragged",
            n_axis=n_axis,
        )
        elapsed = time.perf_counter() - started
    peak = max(sampler.peak_mb, rss_mb())
    return {
        "store": str(store_path),
        "component": "data.zarr/ragged",
        "n_assoc": n_assoc,
        "n_axis": n_axis,
        "seconds": round(elapsed, 1),
        "errors": errors[:5],
        "baseline_mb": round(baseline, 1),
        "peak_mb": round(peak, 1),
        "peak_gib": round(peak / 1024.0, 3),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = {
        "harness": "benchmarks/measure_252_validate_index.py",
        **provenance(),
        **measure(args.store),
    }
    write_artifact(args.output, artifact)
    return 0


if __name__ == "__main__":
    sys.exit(main())
