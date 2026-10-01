"""Shared query-shape construction and RSS probing for the full-scale benchmarks.

The ukb-b (OGS-00009), Reference-Completed (OGS-00010) and FinnGen R13
(OGS-00016) benchmarks time the same seven query shapes on the same
fresh-interpreter RSS probe, so their reports are comparable. Holding the shape
construction and the probe contract in one place is what keeps them from
drifting into almost-the-same queries: a selection, a seed or a shape renamed
in one harness would otherwise silently change what the others measured
(issue #241). #240's store-comparison harness builds on this module.

The harness-specific half stays with each script: which Store Release to open,
which Analysis and region drive the shapes, and the extra argv the fresh
interpreter needs. `benchmarks/_rss.py` owns the probe mechanics; this module
owns what is probed and how the argv that re-invokes a probe is shaped.
"""

from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks._rss import sample_query
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.query import query_store

# The random-lookup shapes keep their historical sizes and seed. The committed
# OGS-00009 and OGS-00016 artifacts were measured with `default_rng(0)` drawing
# this many variants and Analyses, so changing either would make a re-run
# incomparable with them rather than a reproduction of them (issue #241).
RANDOM_AXIS_SIZE = 100
LOOKUP_NARROW_AXIS_SIZE = 10
GENOME_WIDE = 5e-8

# What the two random-lookup shapes query, in the order the reports describe
# them. Recorded in the OGS-00009 and OGS-00016 artifacts, so it lives with the
# shapes rather than being restated per harness (issue #241).
RANDOM_LOOKUP_SHAPES = [
    {"n_variants": LOOKUP_NARROW_AXIS_SIZE, "n_analyses": RANDOM_AXIS_SIZE},
    {"n_variants": RANDOM_AXIS_SIZE, "n_analyses": LOOKUP_NARROW_AXIS_SIZE},
]


def resolve_axis_selections(
    axis: Any,
    analyses: dict[int, dict[str, Any]],
    n_variants: int,
    n_analyses: int,
) -> tuple[list[str], list[str]]:
    """The random-lookup variant ALIDs and Analysis ids the shapes query.

    Draw order is part of the measurement: variants first, then Analyses, from
    one `default_rng(0)`. Callers pass the axis sizes explicitly rather than
    reading them off `axis`, because the OGS-00010 probe re-derives the
    OGS-00009 selections against the *source* release's recorded sizes.
    """
    rng = np.random.default_rng(0)
    random_alids = [
        record.alid
        for record in (
            axis.by_index(int(v))
            for v in rng.choice(n_variants, size=RANDOM_AXIS_SIZE, replace=False)
        )
        if record is not None
    ]
    random_analyses = [
        analyses[int(a)]["analysis_id"]
        for a in rng.choice(n_analyses, size=RANDOM_AXIS_SIZE, replace=False)
    ]
    return random_alids, random_analyses


def regional_alids(axis: Any, region: tuple[str, int, int]) -> list[str]:
    """The ALIDs of one chromosome window, in axis order."""
    rows = axis.range_indices(*region)
    return [
        record.alid
        for record in (axis.by_index(int(row)) for row in rows)
        if record is not None
    ]


def common_query_patterns(
    q: Any,
    *,
    exposure: str,
    phewas_alid: str,
    region: tuple[str, int, int],
    random_alids: list[str],
    random_analyses: list[str],
) -> dict[str, Callable[[], dict[str, np.ndarray]]]:
    """The seven shapes every full-scale dense benchmark times.

    The region is resolved against `q` here, but the random selections are
    captured rather than drawn, so the OGS-00010 harness can pass the ALIDs it
    re-derived from the source release.
    """
    regional = regional_alids(q._variant_axis, region)
    return {
        "bulk": lambda: q.analysis(exposure),
        "phewas": lambda: q.phewas(phewas_alid),
        "regional": lambda: q.range_phewas(*region),
        "regional_one_analysis": lambda: q.lookup(regional, [exposure]),
        "tophits": lambda: q.top_hits(analysis_id=exposure, threshold=GENOME_WIDE),
        "random_lookup_10_variants_100_analyses": lambda: q.lookup(
            random_alids[:LOOKUP_NARROW_AXIS_SIZE], random_analyses
        ),
        "random_lookup_100_variants_10_analyses": lambda: q.lookup(
            random_alids, random_analyses[:LOOKUP_NARROW_AXIS_SIZE]
        ),
    }


def build_query_patterns(
    q: Any,
    analyses: dict[int, dict[str, Any]],
    n_variants: int,
    n_analyses: int,
    *,
    exposure: str,
    phewas_alid: str,
    region: tuple[str, int, int],
) -> dict[str, Callable[[], dict[str, np.ndarray]]]:
    """The common shapes for a harness that draws its selections from `q` itself."""
    random_alids, random_analyses = resolve_axis_selections(
        q._variant_axis, analyses, n_variants, n_analyses
    )
    return common_query_patterns(
        q,
        exposure=exposure,
        phewas_alid=phewas_alid,
        region=region,
        random_alids=random_alids,
        random_analyses=random_analyses,
    )


def measure_shape_rss(patterns: dict[str, Callable[[], Any]], shape: str) -> dict[str, float]:
    """Run one already-built shape in this interpreter and return its RSS record.

    `patterns` is dropped before the sampler starts: holding the other shapes'
    closures (and the selections they captured) would charge their memory to
    this shape's baseline.
    """
    fn = patterns[shape]
    del patterns
    gc.collect()
    record = sample_query(fn)
    record["query"] = shape
    return record


def emit_rss_probe(
    args: argparse.Namespace,
    measure: Callable[[argparse.Namespace, str], dict[str, float]],
) -> bool:
    """Handle the internal `--rss-shape` re-invocation, in a fresh interpreter.

    The parent re-runs the script once per shape because a query's intermediate
    buffers are freed before it returns, so its peak RSS cannot be read in the
    parent after another shape has run. Returns True when the probe record was
    printed and the caller should stop.
    """
    if not args.rss_shape:
        return False
    print(json.dumps(measure(args, args.rss_shape)))
    return True


def add_common_args(ap: argparse.ArgumentParser) -> None:
    """The argv every full-scale dense benchmark accepts."""
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--skip-rss", action="store_true", help="skip the per-shape RSS probes")
    ap.add_argument(
        "--rss-shape", default=None, help="internal: measure one shape's RSS and exit"
    )


def open_benchmark_store(store: Path) -> tuple[Any, StoreManifest]:
    """Open a Store Release and its manifest together, for a normal (non-probe) run."""
    return query_store(store), StoreManifest.load(store)


def start_benchmark(
    parser: argparse.ArgumentParser,
    measure: Callable[[argparse.Namespace, str], dict[str, float]],
) -> tuple[argparse.Namespace, Any, StoreManifest]:
    """Parse argv, honour an internal RSS probe, then open the Store Release.

    A probe re-invocation prints its one JSON record and exits before a store is
    opened; a normal run gets the parsed args, the query handle and the manifest
    back. Harnesses whose argv has no pre-store mode (the OGS-00009 and FinnGen
    benchmarks) use this directly; the ukb-b harness, which has a top-hits
    experiment to handle first, calls the two steps itself.
    """
    args = parser.parse_args()
    if emit_rss_probe(args, measure):
        raise SystemExit(0)
    q, plan = open_benchmark_store(args.store)
    return args, q, plan
