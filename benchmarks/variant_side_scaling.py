#!/usr/bin/env python3
"""Time and peak-RSS scaling for the variant-side shapes, indexed vs scanned (#252).

Every variant-side shape is measured on one store, twice: once answered from the
variant-centric index (`ragged/by_variant/`, ADR 0060) and once from the step-3
scan. Both sides therefore come from **one store copy and one code tree**, so a
difference is the index and not a store or a build. The scan side is produced
in-process by dropping the query's index handle, as in the identity harness.

Each side runs in a fresh interpreter that samples peak RSS on a background
thread (`benchmarks/_rss.py`), so memory is the shape's own, not a previous
shape's high-water mark. Inside the probe the work is split into three phases by
wrapping the reader:

* **match** -- locating the answer's rows (`variant_positions` /
  `segment_positions` / `rows_for_variant*`);
* **read** -- decoding them (the reader's `*_slice`/`*_at`/`decode` calls, which
  is where zarr decompresses);
* **gather** -- everything else (result assembly, the `ancestry` concat, status).

Shape by shape the phases separate asymptotics from constant factors: a scan's
match phase grows with the store, an index's does not.

    pixi run -e dev python benchmarks/variant_side_scaling.py \
        --store OGS-00001=/data/opengwasdb/work/epic252/OGS-00001-0.2.0 \
        --store OGS-00011=/data/opengwasdb/work/epic252/OGS-00011-0.2.0 \
        --output docs/benchmark-output/opengwasdb_252_scaling.json

A store without the index is refused: the indexed side would be the scan.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from benchmarks._artifact import add_labelled_store_option, labelled_stores, provenance
from benchmarks._query_shapes import probe_variant_alid
from benchmarks._rss import RssSampler, rss_mb
from opengwasdb.layouts.ragged.by_variant import ByVariantReader
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRReader
from opengwasdb.query import query_store

#: The TCF7L2 1 Mb window the committed OGS-00011 benchmark uses (rs7903146).
REGION = ("10", 112_500_000, 113_500_000)

_MATCH_METHODS = ("variant_positions", "segment_positions")
_READ_METHODS = (
    "z_slice",
    "se_slice",
    "eaf_slice_read",
    "z_at",
    "se_at",
    "eaf_at_read",
    "variant_index_at",
)


class PhaseTimer:
    """Wall-clock seconds accumulated per phase, installed over the readers."""

    def __init__(self) -> None:
        self.times: dict[str, float] = {"match": 0.0, "read": 0.0}


_timer = PhaseTimer()
_PHASES_INSTALLED = False


def _wrap(timer: PhaseTimer, owner: type, names: tuple[str, ...], phase: str) -> None:
    for name in names:
        original = getattr(owner, name, None)
        if original is None:
            continue

        def wrapper(self: Any, *args: Any, __original: Any = original, **kwargs: Any) -> Any:
            started = perf_counter()
            try:
                return __original(self, *args, **kwargs)
            finally:
                timer.times[phase] += perf_counter() - started

        setattr(owner, name, wrapper)


def _install_phases() -> None:
    global _PHASES_INSTALLED
    if _PHASES_INSTALLED:
        return
    _wrap(_timer, RaggedCSRReader, _MATCH_METHODS, "match")
    _wrap(_timer, RaggedCSRReader, _READ_METHODS, "read")
    _wrap(_timer, ByVariantReader, ("rows_for_variant", "rows_for_variant_range"), "match")
    _wrap(_timer, ByVariantReader, ("decode",), "read")
    _PHASES_INSTALLED = True


def _calls(query: Any) -> dict[str, Callable[[], dict[str, np.ndarray]]]:
    on = probe_variant_alid(query, off_panel=False)
    off = probe_variant_alid(query, off_panel=True)
    analyses = [str(row["analysis_id"]) for _, row in sorted(query.analyses_table().items())][:10]
    calls: dict[str, Callable[[], dict[str, np.ndarray]]] = {
        "range_phewas": lambda: query.range_phewas(*REGION),
    }
    if on is not None:
        calls["phewas"] = lambda: query.phewas(on)
    if off is not None:
        calls["phewas_off_panel"] = lambda: query.phewas(off)
    if analyses and on is not None:
        calls["lookup_10x10"] = lambda: query.lookup([on], analyses)
    if analyses and off is not None:
        calls["lookup_off_panel"] = lambda: query.lookup([off], analyses)
    return calls


def _probe(store: Path, shape: str, *, indexed: bool, reps: int = 3) -> dict[str, Any]:
    """One shape on one side, in this process, with phases and peak RSS.

    A warm-up runs first; each timed repetition waits for the 1-minute load to
    fall below 3, so a contended node is recorded rather than silently timed.
    The reported time and each phase are the median over the repetitions; peak
    RSS is the maximum the sampler saw across them.
    """
    from benchmarks._quiet import wait_for_quiet

    _install_phases()
    query = query_store(store)
    if not indexed:
        query._by_variant = None
    try:
        call = _calls(query)[shape]
        call()  # warm-up
        _timer.times = {"match": 0.0, "read": 0.0}
        baseline = rss_mb()
        totals: list[float] = []
        matches: list[float] = []
        reads: list[float] = []
        result: dict[str, np.ndarray] = {}
        with RssSampler() as sampler:
            for _ in range(reps):
                wait_for_quiet(3.0)
                _timer.times = {"match": 0.0, "read": 0.0}
                started = perf_counter()
                result = call()
                totals.append(perf_counter() - started)
                matches.append(_timer.times["match"])
                reads.append(_timer.times["read"])
        peak = max(sampler.peak_mb, rss_mb())
    finally:
        query.close()
    total = float(np.median(totals))
    match = float(np.median(matches))
    read = float(np.median(reads))
    return {
        "baseline_mb": round(baseline, 1),
        "peak_mb": round(peak, 1),
        "delta_mb": round(peak - baseline, 1),
        "result_count": int(len(result["z"])),
        "reps": reps,
        "elapsed_s": round(total, 4),
        "match_s": round(match, 4),
        "read_s": round(read, 4),
        "gather_s": round(max(0.0, total - match - read), 4),
    }


def measure(store: Path, shapes: list[str], *, reps: int = 3) -> dict[str, Any]:
    shapes_out: dict[str, Any] = {}
    for shape in shapes:
        sides: dict[str, Any] = {}
        for side, indexed in (("indexed", True), ("scanned", False)):
            sides[side] = _probe(store, shape, indexed=indexed, reps=reps)
        if sides["indexed"]["result_count"] != sides["scanned"]["result_count"]:
            raise SystemExit(
                f"{store}:{shape}: indexed {sides['indexed']['result_count']} rows "
                f"!= scanned {sides['scanned']['result_count']}"
            )
        shapes_out[shape] = sides
    return shapes_out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_labelled_store_option(parser)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--shapes",
        default="phewas,range_phewas,lookup_10x10",
        help="comma-separated shape names",
    )
    parser.add_argument("--reps", type=int, default=3, help="repetitions per side")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from opengwasdb.layouts.ragged.by_variant import has_variant_index

    shapes = [name.strip() for name in args.shapes.split(",") if name.strip()]
    stores = labelled_stores(args.store)
    for _label, path in stores:
        if not has_variant_index(path):
            raise SystemExit(f"{path}: no variant index; run `ogdb build-variant-index` first")
    artifact = {
        "harness": "benchmarks/variant_side_scaling.py",
        **provenance(),
        "stores": {
            label: {"store": str(path), "shapes": measure(path, shapes, reps=args.reps)}
            for label, path in stores
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "stores": [label for label, _ in stores]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
