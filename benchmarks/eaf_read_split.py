"""How much of each Dense query is the duplicate EAF read? (#253)

A Dense query on a release with residual SE reads the `eaf` plane twice: once
inside SE decoding, because residual SE is predicted from the frequency, and
once for the result's `eaf` column. This times each harness shape and splits the
time spent inside `DenseEafPlane` reads by caller: a call from `DenseSePlane`
decodes SE; any other caller is the result column, the read #253 removes.

    pixi run -e dev python benchmarks/eaf_read_split.py \\
        --store /data/opengwasdb/stores/OGS-00009/store.opengwasdb \\
        --selection docs/benchmark-output/opengwasdb_store_comparison_ogs00009_zarr2.json \\
        --output /tmp/eaf_split_1.json

Run it once per fresh process; #253 quotes two. Top hits reads no plane on a
release with a current top-hit index, so it is not timed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from benchmarks import _zarr3_levers as levers

#: The shapes, in the order timed, by the names the output uses.
SPLIT_SHAPES = {
    "phewas": "phewas",
    "regional_one_analysis": "regional_one_analysis",
    "random_10x100": "rand_10x100",
    "random_100x10": "rand_100x10",
    "regional": "regional",
    "bulk": "bulk",
}

#: Seconds inside `DenseEafPlane` reads since the last reset, by caller.
SPENT: defaultdict[str, float] = defaultdict(float)


def instrument() -> None:
    """Wrap `DenseEafPlane.points` and `.band` to charge their time to their caller."""
    from opengwasdb.encoding import planes

    for name in ("points", "band"):
        original = getattr(planes.DenseEafPlane, name)

        def wrapped(self: Any, *a: Any, _orig: Callable[..., Any] = original, **k: Any) -> Any:
            caller = sys._getframe(1).f_code.co_qualname
            t0 = time.perf_counter()
            try:
                return _orig(self, *a, **k)
            finally:
                key = "se_decode" if caller.startswith("DenseSePlane") else "result_eaf"
                SPENT[key] += time.perf_counter() - t0

        setattr(planes.DenseEafPlane, name, wrapped)


def split_one(fn: Callable[[], object], reps: int) -> dict[str, float]:
    fn()
    tot, se, res = [], [], []
    for _ in range(reps):
        SPENT.clear()
        t0 = time.perf_counter()
        fn()
        tot.append(time.perf_counter() - t0)
        se.append(SPENT["se_decode"])
        res.append(SPENT["result_eaf"])

    def ms(xs: list[float]) -> float:
        return round(statistics.median(xs) * 1000, 2)

    return {
        "total_ms": ms(tot),
        "eaf_in_se_decode_ms": ms(se),
        "eaf_for_result_ms": ms(res),
        "result_eaf_share": round(statistics.median(res) / statistics.median(tot), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--selection", type=Path, default=levers.ZARR2_BASELINE)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--shapes", nargs="+", choices=list(SPLIT_SHAPES), default=list(SPLIT_SHAPES))
    args = ap.parse_args()
    from opengwasdb.query import query_store

    instrument()
    selection = levers.load_selection(args.selection)
    out: dict[str, dict[str, float]] = {}
    with query_store(args.store) as q:
        shapes = levers.lever_shapes(q, selection, with_bulk=True)
        for name in args.shapes:
            out[name] = split_one(shapes[SPLIT_SHAPES[name]], 7 if name != "bulk" else 3)
            print(name, out[name], flush=True)
    args.output.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
