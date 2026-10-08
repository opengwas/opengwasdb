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
amplification) and the most writes any one shard received.

The fixture's plan is built explicitly as `se int8_residual` + `eaf
int8_residual`, and the cells are drawn so the residual branches and every side
table are exercised: frequencies sit on a per-variant baseline with small noise
(the residual EAF plane), standard errors follow the MAF model (the residual SE
fit), and a handful of cells per Analysis are exact SE exceptions (0.0),
out-of-range z (|z| = 100, the z overflow table) and out-of-range EAF (1.0,
the EAF exception table).

`--legacy-step` reproduces the pre-#249 region step on the same code path by
replacing `sequence_region_step`; the two runs together are the fix's
before/after, and `--combine` assembles them into the committed envelope through
the script rather than by hand.

  pixi run -e dev python benchmarks/measure_write_amplification.py \
      --legacy-step --work /tmp/epic240/249/amp-legacy --output /tmp/epic240/249/amp-legacy.json
  pixi run -e dev python benchmarks/measure_write_amplification.py \
      --work /tmp/epic240/249/amp-fixed --output /tmp/epic240/249/amp-fixed.json
  pixi run -e dev python benchmarks/measure_write_amplification.py \
      --combine /tmp/epic240/249/amp-legacy.json /tmp/epic240/249/amp-fixed.json \
      --output docs/benchmark-output/opengwasdb_ragged_write_amplification_epic240_249.json
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks._artifact import commit
from opengwasdb.encoding.plan import EafEncoding, SeEncoding, StoreEncoding, ZEncoding
from opengwasdb.layouts.ragged import zarr_csr
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRWriter
from opengwasdb.store.arrays import (
    ASSOCIATION_SEQUENCE_CHUNK,
    RAGGED_SEQUENCE_SHARD_ELEMENTS,
    count_shard_writes,
)

#: The guard switch, recorded per run so the artifact says which configuration
#: each number came from (the fixed run is guard-on, the legacy run guard-off).
GUARD_ENV = "OPEN_GWASDB_REQUIRE_WHOLE_SHARD_WRITES"
#: Exact `se` exceptions (0.0) and EAF exceptions (1.0), and out-of-range z, per
#: Analysis, so every side table the residual plan can write is non-empty.
_N_EXCEPTIONS = 4
_N_OVERFLOW = 2
_OVERFLOW_Z = 100.0
#: The MAF model the residual SE fit expects, as the other Ragged fixtures use.
_SE_INTERCEPT = -3.0
_SE_SLOPE = -0.5


def _plan() -> StoreEncoding:
    """The residual plan every run uses, so the residual branches are reached."""
    return StoreEncoding(
        z=ZEncoding("int16_fixed", scale=1024),
        se=SeEncoding("int8_residual", 0.5),
        eaf=EafEncoding("int8_residual", 0.5),
    )


def _synthetic_writer(
    total_cells: int, n_analyses: int, n_variants: int, seed: int
) -> RaggedCSRWriter:
    """A writer holding `total_cells` associations across `n_analyses` Analyses.

    Indices are sorted within an Analysis, as a real builder's are.  Frequencies
    are a per-variant truth with small noise, so the residual EAF plane is small;
    standard errors follow the MAF model, so the residual SE fit converges; and
    the planted exceptions make the exception/overflow tables non-empty.
    """
    rng = np.random.default_rng(seed)
    per_analysis = total_cells // n_analyses
    truth = rng.uniform(0.05, 0.95, n_variants).astype(np.float32)
    writer = RaggedCSRWriter(n_variants)
    for _ in range(n_analyses):
        vi = np.sort(rng.integers(0, n_variants, size=per_analysis)).astype(np.int32)
        eaf = np.clip(truth[vi] + rng.normal(0, 0.002, per_analysis), 1e-4, 1 - 1e-4)
        x = np.log(2 * eaf * (1 - eaf))
        se = np.exp(_SE_INTERCEPT + _SE_SLOPE * x + rng.normal(0, 0.01, per_analysis))
        z = rng.standard_normal(per_analysis)
        for offset in range(_N_EXCEPTIONS):
            se[offset] = 0.0
            eaf[offset] = 1.0
        for offset in range(_N_OVERFLOW):
            z[per_analysis - 1 - offset] = _OVERFLOW_Z
        writer.add_analysis(
            vi,
            z.astype(np.float32),
            se.astype(np.float32),
            eaf.astype(np.float32),
        )
    return writer


def _final_bytes(root: Path) -> dict[str, int]:
    """On-disk bytes per array under the store's `ragged` group."""
    sizes: dict[str, int] = {}
    for array_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        sizes[array_dir.name] = sum(f.stat().st_size for f in array_dir.rglob("*") if f.is_file())
    return sizes


def _array_report(name: str, counts: dict[str, int], written: int, final: int) -> dict[str, Any]:
    return {
        "shards": len(counts),
        "shard_writes": sum(counts.values()),
        "max_writes_one_shard": max(counts.values(), default=0),
        "bytes_written": written,
        "final_bytes": final,
        "amplification": round(written / final, 3) if final else None,
    }


def _status_kib(key: str) -> int:
    """One `Vm*` figure from `/proc/self/status`, in KiB (`0` if absent)."""
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{key}:"):
                return int(line.split()[1])
    except OSError:
        pass
    return 0


def _reset_peak_rss() -> None:
    """Reset VmHWM (`5` to `clear_refs`), so VmHWM afterwards is the flush's peak.

    `ru_maxrss` is the process's lifetime high-water mark, set while the fixture
    was generated and the plan chosen, so it cannot see the flush's own working
    set.  After the reset, VmHWM minus the RSS before the flush is what the
    flush itself added.
    """
    try:
        Path("/proc/self/clear_refs").write_text("5\n", encoding="utf-8")
    except OSError:
        pass


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
    encoding = _plan()
    rss_before_kib = _status_kib("VmRSS")
    _reset_peak_rss()
    started = time.monotonic()
    with count_shard_writes() as recorder:
        writer.flush(args.work, encoding, region_cells=args.region_cells)
    wall = time.monotonic() - started
    lifetime_peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    flush_peak_kib = max(0, _status_kib("VmHWM") - rss_before_kib)

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
        "guard_env": os.environ.get(GUARD_ENV),
        "legacy_step": bool(args.legacy_step),
        "shard_elements": RAGGED_SEQUENCE_SHARD_ELEMENTS,
        "inner_chunk": ASSOCIATION_SEQUENCE_CHUNK,
        "region_cells": args.region_cells,
        "total_cells": args.total_cells,
        "n_analyses": args.n_analyses,
        "n_variants": args.n_variants,
        "encoding": encoding.to_manifest(),
        "wall_seconds": round(wall, 2),
        "flush_peak_rss_kib": flush_peak_kib,
        "flush_peak_rss_gib": round(flush_peak_kib / (1024 * 1024), 3),
        "lifetime_peak_rss_kib": lifetime_peak_kib,
        "lifetime_peak_rss_gib": round(lifetime_peak_kib / (1024 * 1024), 2),
        "total_bytes_written": total_written,
        "total_final_bytes": total_final,
        "total_amplification": round(total_written / total_final, 3) if total_final else None,
        "arrays": arrays,
        "note": (
            "bytes_written counts every byte handed to the storage layer while the "
            "component flushed; final_bytes is what is on disk afterwards. A shard "
            "written once has amplification near 1; a max_writes_one_shard above 1 "
            "is the read-modify-write this ticket fixes. `flush_peak_rss_kib` is "
            "VmHWM after resetting it minus VmRSS before the flush, i.e. the "
            "flush's own added peak; `lifetime_peak_rss_kib` is `ru_maxrss` and "
            "predates the flush."
        ),
    }


def combine_artifacts(legacy_path: Path, fixed_path: Path) -> dict[str, Any]:
    """Assemble the before/after envelope from two runs, through the script.

    The envelope is the committed artifact, so re-running the two documented
    commands and this one reproduces the file byte for byte (apart from
    `measured_at`); nothing is copied or annotated by hand.
    """
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    fixed = json.loads(fixed_path.read_text(encoding="utf-8"))
    if legacy["legacy_step"] is not True or fixed["legacy_step"] is not False:
        raise SystemExit("--combine wants the legacy run first and the fixed run second")
    return {
        "artifact": "Ragged sequence write amplification before/after #249",
        "commit": legacy["commit"],
        "measured_at": datetime.now(UTC).isoformat(),
        "shard_elements": legacy["shard_elements"],
        "inner_chunk": legacy["inner_chunk"],
        "region_cells": legacy["region_cells"],
        "fixture": {
            "total_cells": legacy["total_cells"],
            "n_analyses": legacy["n_analyses"],
            "n_variants": legacy["n_variants"],
            "encoding": legacy["encoding"],
        },
        "runs": [
            {"path": str(legacy_path), "legacy_step": True},
            {"path": str(fixed_path), "legacy_step": False},
        ],
        "legacy_step": legacy,
        "fixed_step": fixed,
        "note": (
            "Both runs are the re-runnable output of "
            "benchmarks/measure_write_amplification.py; the legacy run replaces "
            "sequence_region_step with the pre-#249 region-only step on the same "
            "code path. The envelope itself is written by --combine. Never "
            "hand-edited."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work", type=Path, help="where the synthetic component is built")
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
    parser.add_argument(
        "--combine",
        type=Path,
        nargs=2,
        metavar=("LEGACY_JSON", "FIXED_JSON"),
        help="assemble the committed envelope from the two runs; builds nothing",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.combine:
        payload = combine_artifacts(args.combine[0], args.combine[1])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.output}: combined envelope")
        return 0
    if args.work is None:
        raise SystemExit("--work is required unless --combine is given")
    args.work.mkdir(parents=True, exist_ok=True)
    payload = measure(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}: amplification {payload['total_amplification']}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
