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
from opengwasdb.encoding import DenseEafPlane, DenseSePlane, DenseZPlane
from opengwasdb.layouts.ragged.by_variant import ByVariantReader
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRReader
from opengwasdb.query import query_store

#: The TCF7L2 1 Mb window the committed OGS-00011 benchmark uses (rs7903146).
#: A store that does not hold it gets one around the first variant it does hold,
#: so no cell measures an empty range (review round 1, finding 4).
TCF7L2_REGION = ("10", 112_500_000, 113_500_000)

_OVERFLOW_MATCH = ("variant_positions", "segment_positions")
_OVERFLOW_READ = (
    "z_slice",
    "se_slice",
    "eaf_slice_read",
    "z_at",
    "se_at",
    "eaf_at_read",
    "variant_index_at",
)
_DENSE_READ = (
    "band",
    "column",
    "row",
    "rows",
    "block",
    "points",
    "read_points",
    "read_band",
    "read_row",
    "read_rows",
    "read_block",
    "read_column",
)


class PhaseTimer:
    """Wall-clock seconds accumulated per phase, installed over the readers."""

    def __init__(self) -> None:
        self.times: dict[str, float] = {}
        self.reset()

    def reset(self) -> None:
        self.times = {"dense": 0.0, "overflow_match": 0.0, "overflow_read": 0.0}


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
    _wrap(_timer, RaggedCSRReader, _OVERFLOW_MATCH, "overflow_match")
    _wrap(_timer, RaggedCSRReader, _OVERFLOW_READ, "overflow_read")
    _wrap(_timer, ByVariantReader, ("rows_for_variant", "rows_for_variant_range"), "overflow_match")
    _wrap(_timer, ByVariantReader, ("decode",), "overflow_read")
    for plane in (DenseZPlane, DenseSePlane, DenseEafPlane):
        _wrap(_timer, plane, _DENSE_READ, "dense")
    _PHASES_INSTALLED = True


def _store_inputs(query: Any) -> dict[str, Any]:
    """A region and lookup selections this store actually holds (finding 4).

    The region is the committed TCF7L2 window for a store that holds it, else a
    1 Mb window around the first variant the index covers; the lookup selections
    are `(variant, analyses)` pairs taken from an indexed variant's block, so a
    lookup measures a non-empty answer rather than a miss.
    """
    axis = query._variant_axis
    table = query.analyses_table()
    reader = getattr(query, "_by_variant", None)
    n = int(axis.n_variants)
    region = TCF7L2_REGION
    narrow_alids: list[str] = []
    narrow_analyses: list[str] = []
    wide_alids: list[str] = []
    wide_analyses: list[str] = []
    if n:
        step = max(1, n // 256)
        for index in range(0, n, step):
            if reader is not None:
                start, end = reader.rows_for_variant(index)
                if end <= start:
                    continue
                block = [int(a) for a in np.asarray(reader._analysis_index[start:end])]
            else:
                block = []
            alid = str(axis.by_index(index).alid)
            if not narrow_alids:
                record = axis.by_index(index)
                region = (
                    record.chromosome,
                    max(1, int(record.position) - 500_000),
                    int(record.position) + 500_000,
                )
                narrow_analyses = [str(table[a]["analysis_id"]) for a in block[:5]]
            if len(narrow_alids) < 10:
                narrow_alids.append(alid)
            if not wide_analyses and len(block) >= 10:
                wide_alids.append(alid)
                wide_analyses = [str(table[a]["analysis_id"]) for a in block[:50]]
            if len(narrow_alids) >= 10 and wide_analyses:
                break
    return {
        "region": region,
        "narrow": (narrow_alids, narrow_analyses),
        "wide": (wide_alids, wide_analyses),
    }


def _calls(query: Any, inputs: dict[str, Any]) -> dict[str, Callable[[], dict[str, np.ndarray]]]:
    on = probe_variant_alid(query, off_panel=False)
    off = probe_variant_alid(query, off_panel=True)
    region = inputs["region"]
    narrow_alids, narrow_analyses = inputs["narrow"]
    wide_alids, wide_analyses = inputs["wide"]
    calls: dict[str, Callable[[], dict[str, np.ndarray]]] = {
        "range_phewas": lambda: query.range_phewas(*region),
    }
    if on is not None:
        calls["phewas"] = lambda: query.phewas(on)
    if off is not None:
        calls["phewas_off_panel"] = lambda: query.phewas(off)
    if narrow_alids and narrow_analyses:
        calls["lookup_10_variants"] = lambda: query.lookup(narrow_alids, narrow_analyses)
    if wide_alids and wide_analyses:
        calls["lookup_50_analyses"] = lambda: query.lookup(wide_alids, wide_analyses)
    return calls


def _probe(
    store: Path,
    shape: str,
    *,
    indexed: bool,
    inputs: dict[str, Any],
    reps: int = 3,
    max_load: float = 3.0,
) -> dict[str, Any]:
    """One shape on one side, in this process, with phases and peak RSS.

    A warm-up runs first; each timed repetition waits for the 1-minute load to
    fall below `max_load` (0 disables), so a contended node is recorded rather
    than silently timed -- and a test can turn the wait off.  The reported time
    and each phase are the median over the repetitions; peak RSS is the maximum
    the sampler saw across them.  The phases split a Hybrid shape into the Dense
    Component's own window read and the Overflow's match/read, so the dominant
    half is attributed rather than guessed.
    """
    from benchmarks._quiet import wait_for_quiet

    _install_phases()
    query = query_store(store)
    if not indexed:
        query._by_variant = None
    try:
        call = _calls(query, inputs)[shape]
        call()  # warm-up
        _timer.reset()
        baseline = rss_mb()
        totals: list[float] = []
        phases: dict[str, list[float]] = {"dense": [], "overflow_match": [], "overflow_read": []}
        result: dict[str, np.ndarray] = {}
        with RssSampler() as sampler:
            for _ in range(reps):
                wait_for_quiet(max_load)
                _timer.reset()
                started = perf_counter()
                result = call()
                totals.append(perf_counter() - started)
                for key in phases:
                    phases[key].append(_timer.times[key])
        peak = max(sampler.peak_mb, rss_mb())
    finally:
        query.close()
    total = float(np.median(totals))
    overflow_match = float(np.median(phases["overflow_match"]))
    overflow_read = float(np.median(phases["overflow_read"]))
    # Everything not attributed to the Overflow reader is the Dense side (a
    # Hybrid's Dense Component read) plus result assembly.  Reporting the
    # remainder rather than the wrapped Dense calls keeps the phases additive:
    # a Dense plane's public method calls others, so summing them double-counts.
    dense = max(0.0, total - overflow_match - overflow_read)
    return {
        "baseline_mb": round(baseline, 1),
        "peak_mb": round(peak, 1),
        "delta_mb": round(peak - baseline, 1),
        "result_count": int(len(result["z"])),
        "reps": reps,
        "elapsed_s": round(total, 4),
        "dense_s": round(dense, 4),
        "overflow_match_s": round(overflow_match, 4),
        "overflow_read_s": round(overflow_read, 4),
    }


def _content_digest(result: dict[str, np.ndarray]) -> str:
    """A sha256 over the answer in canonical `(variant, analysis)` row order."""
    import hashlib

    order = np.lexsort(
        (np.asarray(result["analysis_index"]), np.asarray(result["variant_index"]))
    )
    h = hashlib.sha256()
    for key in ("variant_index", "analysis_index", "z", "se", "eaf", "association_status"):
        arr = np.asarray(result[key])[order]
        h.update(key.encode())
        h.update(str(arr.dtype).encode())
        if arr.dtype == object:
            for value in arr:
                h.update(str(value).encode())
        else:
            h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def measure(
    store: Path,
    shapes: list[str],
    *,
    reps: int = 3,
    max_load: float = 3.0,
    allow_unindexed: bool = False,
) -> dict[str, Any]:
    """Every shape, indexed and scanned (scanned only when unindexed)."""
    from opengwasdb.layouts.ragged.by_variant import has_variant_index

    indexed = has_variant_index(store)
    if not indexed and not allow_unindexed:
        raise SystemExit(f"{store}: no variant index; pass --allow-unindexed to scan it")
    probe = query_store(store)
    try:
        inputs = _store_inputs(probe)
    finally:
        probe.close()
    sides_for = ("indexed", "scanned") if indexed else ("scanned",)
    shapes_out: dict[str, Any] = {}
    for shape in shapes:
        probe = query_store(store)
        try:
            available = shape in _calls(probe, inputs)
        finally:
            probe.close()
        if not available:
            shapes_out[shape] = {"skipped": "no non-empty selection on this store"}
            print(f"{store.name}:{shape}: skipped (no non-empty selection)", flush=True)
            continue
        sides: dict[str, Any] = {}
        digests: dict[str, str] = {}
        for side in sides_for:
            record = _probe(
                store,
                shape,
                indexed=side == "indexed",
                inputs=inputs,
                reps=reps,
                max_load=max_load,
            )
            sides[side] = record
        if indexed:
            query = query_store(store)
            try:
                call = _calls(query, inputs)[shape]
                digests["indexed"] = _content_digest(call())
                query._by_variant = None
                digests["scanned"] = _content_digest(call())
            finally:
                query.close()
            if digests["indexed"] != digests["scanned"]:
                raise SystemExit(
                    f"{store}:{shape}: indexed content {digests['indexed'][:12]} "
                    f"!= scanned {digests['scanned'][:12]}"
                )
            sides["content_sha256"] = digests["indexed"]
        shapes_out[shape] = sides
    return shapes_out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_labelled_store_option(parser)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--shapes",
        default="phewas,range_phewas,lookup_10_variants,lookup_50_analyses",
        help="comma-separated shape names",
    )
    parser.add_argument("--reps", type=int, default=3, help="repetitions per side")
    parser.add_argument(
        "--allow-unindexed",
        action="store_true",
        help="measure the scan only, for a store that carries no variant index",
    )
    parser.add_argument(
        "--max-start-load",
        type=float,
        default=3.0,
        help="wait for the 1-minute load below this before each repetition (0 disables)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    shapes = [name.strip() for name in args.shapes.split(",") if name.strip()]
    stores = labelled_stores(args.store)
    artifact = {
        "harness": "benchmarks/variant_side_scaling.py",
        **provenance(),
        "shapes": shapes,
        "stores": {
            label: {
                "store": str(path),
                "shapes": measure(
                    path,
                    shapes,
                    reps=args.reps,
                    max_load=args.max_start_load,
                    allow_unindexed=args.allow_unindexed,
                ),
            }
            for label, path in stores
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "stores": [label for label, _ in stores]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
