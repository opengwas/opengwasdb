#!/usr/bin/env python3
"""The #242 seven-shape identity for stores whose hard-coded selection is absent.

`benchmarks/benchmark_store_comparison.py` resolves one selection from
OGS-00009's own exposure (`ukb-b-17805`, the chr19 APOE region) and applies it to
every store, which is what makes its timings comparable. OGS-00001, OGS-00004,
OGS-00007 and OGS-00008 do not hold that Analysis, so for them the selection is
resolved from each store's own axes instead: a strongest genome-wide top hit (or,
without one, the Analysis with the most rows), that hit's 1 Mb window, and up to
100 ALIDs spread across the variant axis. Every other part of the contract is the
#242 one -- the same seven shapes, the same `result_digests` hashing (dtype,
shape, order, values, NaN positions) and the same `assert_identical`, so the two
harnesses cannot disagree about what "identical" means.

A shape whose reference-store result is **empty** is reported under
`shapes_skipped` with its reason rather than silently dropped, so an identity run
cannot pass by running nothing.

  pixi run -e dev python benchmarks/benchmark_store_identity.py \
      head=/data/.../249/dense-head.opengwasdb converted=/data/.../OGS-00008 \
      --output /tmp/epic240/249/identity-OGS-00008.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks._artifact import provenance, write_artifact
from benchmarks.benchmark_store_comparison import assert_identical, result_digests
from opengwasdb.query import query_store

#: The seven shapes #242 compares, in report order.  Fixed here so a run cannot
#: quietly drop one; `select_shapes` returns exactly these names.
SHAPE_NAMES: tuple[str, ...] = (
    "analysis",
    "phewas",
    "regional",
    "regional_one_analysis",
    "top_hits",
    "random_lookup_10_variants_100_analyses",
    "random_lookup_100_variants_10_analyses",
)

#: The genome-wide threshold a top hit is taken at, and the relaxed one for a
#: store with no genome-wide hit (so the PheWAS shape still has a variant).
GENOME_WIDE = 5e-8
RELAXED = 5e-4


def _analyses(q: Any) -> list[str]:
    table = q.analyses_table()
    return [table[index]["analysis_id"] for index in sorted(table)]


def resolve_selection(q: Any, aids: list[str]) -> tuple[str, Any, float, str]:
    """An exposure, a PheWAS variant and a threshold, from the store's own axes.

    Prefers the strongest genome-wide top hit, which the `phewas` and `top_hits`
    shapes are most meaningful against.  A store with none falls back to the
    Analysis holding the most associations and that Analysis's first variant, at
    the relaxed threshold, so the run still covers every shape.
    """
    top = q.top_hits(threshold=GENOME_WIDE)
    if len(top["z"]):
        keep = int(np.argmax(np.abs(np.asarray(top["z"], dtype=np.float64))))
        analysis_index = int(top["analysis_index"][keep])
        variant_index = int(top["variant_index"][keep])
        analysis = q.analyses_table()[analysis_index]["analysis_id"]
        return analysis, q._variant_axis.by_index(variant_index), GENOME_WIDE, (
            "strongest genome-wide top hit"
        )
    best, best_n = aids[0], -1
    for aid in aids:
        n = len(q.analysis(aid)["z"])
        if n > best_n:
            best, best_n = aid, n
    result = q.analysis(best)
    variant_index = int(result["variant_index"][0])
    return best, q._variant_axis.by_index(variant_index), RELAXED, (
        f"no genome-wide top hit; Analysis {best} has the most associations"
    )


def select_shapes(
    q: Any, exposure: str, record: Any, threshold: float
) -> dict[str, Callable[[], dict[str, Any]]]:
    """The seven shape closures, against the resolved selection."""
    aids = _analyses(q)
    region = (record.chromosome, max(1, record.position - 500_000), record.position + 500_000)
    axis = q._variant_axis
    # Up to 100 ALIDs spread across the axis, read from the axis rather than by
    # resolving every association of an Analysis (millions of BGZF seeks).
    step = max(1, int(axis.n_variants) // 100)
    variants: list[str] = []
    for index in range(0, int(axis.n_variants), step):
        rec = axis.by_index(index)
        if rec is not None:
            variants.append(rec.alid)
        if len(variants) >= 100:
            break
    # A small sample of the window's own variants for the one-window/one-Analysis
    # shape; the whole window would be thousands of ALIDs and is not the point.
    regional: list[str] = []
    for index in np.asarray(axis.range_indices(*region)).tolist()[:50]:
        rec = axis.by_index(int(index))
        if rec is not None:
            regional.append(rec.alid)
    if not regional:
        regional = [record.alid]
    return {
        "analysis": lambda: q.analysis(exposure),
        "phewas": lambda: q.phewas(record.alid),
        "regional": lambda: q.range_phewas(*region),
        "regional_one_analysis": lambda: q.lookup(regional, [exposure]),
        "top_hits": lambda: q.top_hits(analysis_id=exposure, threshold=threshold),
        "random_lookup_10_variants_100_analyses": lambda: q.lookup(variants[:10], aids),
        "random_lookup_100_variants_10_analyses": lambda: q.lookup(variants[:100], aids[:10]),
    }


def measure(label: str, path: str, *, reference: bool) -> tuple[dict, dict, dict, dict]:
    """Run the seven shapes on one store; return digests, skips, counts and meta."""
    q = query_store(path)
    try:
        aids = _analyses(q)
        exposure, record, threshold, how = resolve_selection(q, aids)
        shapes = select_shapes(q, exposure, record, threshold)
        results: dict[str, dict] = {}
        counts: dict[str, int] = {}
        skipped: dict[str, str] = {}
        for name, run_shape in shapes.items():
            started = time.perf_counter()
            result = run_shape()
            counts[name] = int(len(result["z"]))
            print(
                f"  {label} {name:46s} {time.perf_counter() - started:8.1f}s "
                f"rows={counts[name]}",
                flush=True,
            )
            if len(result["z"]) == 0 and reference:
                skipped[name] = "reference result empty; the shape proves nothing"
                continue
            results[name] = result_digests(result)
        meta = {
            "label": label,
            "path": str(path),
            "exposure": exposure,
            "phewas_alid": record.alid if record is not None else None,
            "exposure_reason": how,
            "threshold": threshold,
            "n_analyses": len(aids),
            "n_variants": int(q._variant_axis.n_variants),
            "shapes_run": sorted(results),
            "shapes_skipped": skipped,
        }
        return results, skipped, counts, meta
    finally:
        q.close()


def _spec(text: str) -> tuple[str, Path]:
    label, separator, path = text.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError(f"a store wants LABEL=PATH, got {text!r}")
    if not Path(path).is_dir():
        raise argparse.ArgumentTypeError(f"not a directory: {path!r}")
    return label, Path(path)


def run(specs: list[tuple[str, Path]]) -> dict[str, Any]:
    if len(specs) < 2:
        raise SystemExit("need a reference store and at least one store to compare")
    reference_label, reference_path = specs[0]
    reference, skipped, counts, reference_meta = measure(
        reference_label, str(reference_path), reference=True
    )
    if not reference:
        raise SystemExit(
            f"every shape was empty on the reference store {reference_label}; "
            "the comparison would prove nothing"
        )
    comparisons = []
    for label, path in specs[1:]:
        other, _other_skipped, _other_counts, other_meta = measure(
            label, str(path), reference=False
        )
        assert_identical(reference_label, reference, label, other)
        comparisons.append(other_meta)
    return {
        **provenance(),
        "artifact": "seven-shape Store identity, epic #240",
        "reference": reference_meta,
        "comparisons": comparisons,
        "shapes": list(SHAPE_NAMES),
        "shapes_skipped": skipped,
        "rows": counts,
        "identical": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("store", nargs="+", type=_spec, metavar="LABEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    payload = run(args.store)
    write_artifact(args.output, payload)
    print(json.dumps({"shapes_skipped": payload["shapes_skipped"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
