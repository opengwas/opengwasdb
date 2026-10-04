"""Shared pieces of the #244 read-lever evidence behind ADR 0056.

The scripts that measured zarr-python 3's read levers (Blosc threads, the fused
codec pipeline at one worker, top-hit arrays opened once) run the same query
shapes in the same order, set zarr's pipeline by the same labels, and read the
same JSON-lines outputs. Holding those here keeps a label or a shape order from
meaning different things in two scripts.

`benchmarks/zarr3_attribution.py` times a checkout in a child process that runs
under *that checkout's* environment -- the zarr 2.18 base, or `708d179` from
before the levers -- where `benchmarks` holds no copy of this module. The child
loads it from beside the script instead, so this module imports nothing at
module level beyond the standard library, and `lever_shapes` imports the shared
query shapes from whichever checkout is first on `sys.path`.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

#: Where #244's committed outputs live, relative to the repository root. The
#: analysis scripts read their inputs from here by default.
OUTPUTS = Path("docs/benchmark-output/opengwasdb_zarr3_read_levers")
#: The #242 zarr 2.18 baseline: the 2.18 timings, peaks and query selection.
ZARR2_BASELINE = Path("docs/benchmark-output/opengwasdb_store_comparison_ogs00009_zarr2.json")

FUSED_PIPELINE = "zarr.core.codec_pipeline.FusedCodecPipeline"

#: zarr runtime configuration per pipeline label. `asis` sets nothing, so the
#: checkout's own configuration (the seam's, from #244 on) is what is measured.
PIPELINES: dict[str, dict[str, object]] = {
    "default": {},
    "asis": {},
    "fused": {"codec_pipeline.path": FUSED_PIPELINE},
    "fused_mw1": {"codec_pipeline.path": FUSED_PIPELINE, "codec_pipeline.max_workers": 1},
    "fused_mw8": {"codec_pipeline.path": FUSED_PIPELINE, "codec_pipeline.max_workers": 8},
}

#: The order the shapes were timed in. Each runs after the previous one in the
#: same process, so the order is part of the measurement.
SHAPE_ORDER = (
    "phewas",
    "tophits",
    "regional_one_analysis",
    "regional",
    "rand_10x100",
    "rand_100x10",
    "bulk",
)

#: The order the #242 harness and its reports list the shapes in.
HARNESS_ORDER = (
    "bulk",
    "phewas",
    "regional",
    "regional_one_analysis",
    "tophits",
    "rand_10x100",
    "rand_100x10",
)

#: The short names these outputs use -> the names `benchmarks/_query_shapes.py`
#: and the #242 harness artifacts use.
HARNESS_NAME = {
    "rand_10x100": "random_lookup_10_variants_100_analyses",
    "rand_100x10": "random_lookup_100_variants_10_analyses",
}


def load_selection(path: Path) -> dict[str, Any]:
    """The query selection a #242 store-comparison artifact recorded.

    The committed `opengwasdb_store_comparison_ogs00009_zarr2.json` carries the
    one every #244 lever measurement used.
    """
    selection: dict[str, Any] = json.loads(path.read_text())["selection"]
    return selection


def selection_region(selection: dict[str, Any]) -> tuple[str, int, int]:
    region = selection["region"]
    return (str(region["chrom"]), int(region["start"]), int(region["end"]))


def lever_shapes(
    q: Any, selection: dict[str, Any], *, with_bulk: bool
) -> dict[str, Callable[[], dict[str, Any]]]:
    """The harness's query shapes, keyed by short name, in `SHAPE_ORDER`.

    Built by the shared `common_query_patterns`, so these are the #242 harness's
    queries exactly; only the order and the names differ.
    """
    from benchmarks import _query_shapes

    common = _query_shapes.common_query_patterns(
        q,
        exposure=selection["exposure_analysis_id"],
        phewas_alid=selection["phewas_alid"],
        region=selection_region(selection),
        random_alids=selection["random_alids"],
        random_analyses=selection["random_analyses"],
    )
    names = [name for name in SHAPE_ORDER if with_bulk or name != "bulk"]
    return {name: common[HARNESS_NAME.get(name, name)] for name in names}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def say(**fields: object) -> None:
    """Print one JSON line and flush, so a hang leaves every line before it."""
    print(json.dumps(fields), flush=True)


def p95(samples: list[float]) -> float:
    """The nearest-rank 95th percentile the lever runs recorded."""
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]


def attribution_runs(path: Path, config: str) -> list[dict[str, Any]]:
    """The per-process results one attribution config recorded, failing on a gap."""
    runs = []
    for line in read_jsonl(path):
        if line["result"] is None:
            raise SystemExit(f"{path}: a run produced no result: {line}")
        if line["config"] == config:
            runs.append(line["result"])
    if not runs:
        raise SystemExit(f"{path}: no runs of config {config!r}")
    return runs


def attribution_medians(path: Path, config: str) -> dict[str, float]:
    """Per shape, the median of the per-process medians for one config."""
    runs = attribution_runs(path, config)
    return {shape: statistics.median(r["ms"][shape] for r in runs) for shape in runs[0]["ms"]}


def harness_rows(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """A #242 harness artifact's first store: timings and memory by short name."""
    short = {v: k for k, v in HARNESS_NAME.items()}
    store = json.loads(path.read_text())["stores"][0]
    return {
        "time": {short.get(t["query"], t["query"]): t for t in store["timings"]},
        "mem": {short.get(m["query"], m["query"]): m for m in store.get("memory", [])},
    }
