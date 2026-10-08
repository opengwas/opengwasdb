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
        --store /data/opengwasdb/work/epic252/OGS-00001 \
        --output docs/benchmark-output/opengwasdb_252_scaling_OGS-00001.json

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

from benchmarks._artifact import provenance
from benchmarks._rss import RssSampler, rss_mb
from opengwasdb.layouts.ragged.by_variant import ByVariantReader
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRReader
from opengwasdb.query import query_store

REGION = ("1", 114_000_000, 115_000_000)

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


def _first_alid(query: Any, *, off_panel: bool | None = None) -> str | None:
    table = query.variants_table()
    if not table:
        return None
    indices = np.sort(np.array(list(table), dtype=np.int64))
    mask = getattr(query, "_on_panel_mask", None)
    if off_panel is not None and mask is not None:
        on_panel = np.asarray(mask(indices))
        indices = indices[~on_panel] if off_panel else indices[on_panel]
    return str(table[int(indices[0])]["alid"]) if len(indices) else None


def _calls(query: Any) -> dict[str, Callable[[], dict[str, np.ndarray]]]:
    alid = _first_alid(query)
    off = _first_alid(query, off_panel=True)
    analyses = [str(row["analysis_id"]) for _, row in sorted(query.analyses_table().items())][:10]
    calls: dict[str, Callable[[], dict[str, np.ndarray]]] = {
        "range_phewas": lambda: query.range_phewas(*REGION),
    }
    if alid is not None:
        calls["phewas"] = lambda: query.phewas(alid)
    if off is not None:
        calls["phewas_off_panel"] = lambda: query.phewas(off)
    if analyses and alid is not None:
        calls["lookup_10x10"] = lambda: query.lookup([alid], analyses)
    if analyses and off is not None:
        calls["lookup_off_panel"] = lambda: query.lookup([off], analyses)
    return calls


def _probe(store: Path, shape: str, *, indexed: bool) -> dict[str, Any]:
    """One shape on one side, in this process, with phases and peak RSS."""
    _install_phases()
    query = query_store(store)
    if not indexed:
        query._by_variant = None
    try:
        call = _calls(query)[shape]
        _timer.times = {"match": 0.0, "read": 0.0}
        baseline = rss_mb()
        with RssSampler() as sampler:
            started = perf_counter()
            result = call()
            total = perf_counter() - started
        peak = max(sampler.peak_mb, rss_mb())
    finally:
        query.close()
    match = _timer.times["match"]
    read = _timer.times["read"]
    return {
        "baseline_mb": round(baseline, 1),
        "peak_mb": round(peak, 1),
        "delta_mb": round(peak - baseline, 1),
        "result_count": int(len(result["z"])),
        "elapsed_s": round(total, 4),
        "match_s": round(match, 4),
        "read_s": round(read, 4),
        "gather_s": round(max(0.0, total - match - read), 4),
    }


def measure(store: Path, shapes: list[str]) -> dict[str, Any]:
    shapes_out: dict[str, Any] = {}
    for shape in shapes:
        sides: dict[str, Any] = {}
        for side, indexed in (("indexed", True), ("scanned", False)):
            sides[side] = _probe(store, shape, indexed=indexed)
        if sides["indexed"]["result_count"] != sides["scanned"]["result_count"]:
            raise SystemExit(
                f"{store}:{shape}: indexed {sides['indexed']['result_count']} rows "
                f"!= scanned {sides['scanned']['result_count']}"
            )
        shapes_out[shape] = sides
    return shapes_out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--shapes",
        default="phewas,range_phewas,lookup_10x10",
        help="comma-separated shape names",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from opengwasdb.layouts.ragged.by_variant import has_variant_index

    if not has_variant_index(args.store):
        raise SystemExit(f"{args.store}: no variant index; run `ogdb build-variant-index` first")
    shapes = [name.strip() for name in args.shapes.split(",") if name.strip()]
    artifact = {
        "harness": "benchmarks/variant_side_scaling.py",
        "store": str(args.store),
        **provenance(),
        "shapes": measure(args.store, shapes),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "shapes": shapes}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
