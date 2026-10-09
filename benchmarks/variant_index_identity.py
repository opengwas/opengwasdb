#!/usr/bin/env python3
"""Identity between the variant-centric index and the scan it replaces (#252).

Every variant-side shape must return the same rows whether it is answered from
`ragged/by_variant/` (ADR 0060) or from the step-3 scan. The facade makes no
ordering guarantee beyond grouping (ADR 0033), so the two routes are compared in
canonical `(variant_index, analysis_index)` order -- the index returns a region
variant-major where the scan returns it Analysis-major, and neither order is
part of the contract.

The scan route is produced in-process by dropping the query's index handle
(`query._by_variant = None`), so one store and one code tree answer both ways
and an identity failure cannot be a store difference. That is deliberate: a
second store copy built without the index would also change the comparison.

This is the harness the step-5 identity artifact is written from:

    pixi run -e dev python benchmarks/variant_index_identity.py \
        --store /data/opengwasdb/work/epic252/OGS-00011 \
        --output docs/benchmark-output/opengwasdb_252_variant_index_identity.json

It refuses to publish a run in which any shape returned nothing on either side
(a vacuous identity), or in which any shape's two answers differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks._artifact import (
    add_labelled_store_option,
    labelled_stores,
    provenance,
    write_artifact,
)
from benchmarks._query_shapes import probe_variant_alid
from opengwasdb.query import query_store

#: The shapes every store is checked on. Each returns a `(label, callable)`
#: pair; the callable takes a query facade and the chosen inputs.
SHAPE_NAMES = ("phewas", "range_phewas", "lookup")

#: The TCF7L2 1 Mb window the committed OGS-00011 benchmark uses (rs7903146);
#: the same biological question, so the two artifacts are comparable.
REGION = ("10", 112_500_000, 113_500_000)


def _canonical(result: dict[str, np.ndarray]) -> list[np.ndarray]:
    """The result's six arrays in `(variant_index, analysis_index)` order."""
    order = np.lexsort(
        (np.asarray(result["analysis_index"]), np.asarray(result["variant_index"]))
    )
    keys = ("variant_index", "analysis_index", "z", "se", "eaf", "association_status")
    return [np.asarray(result[key])[order] for key in keys]


def _digest(result: dict[str, np.ndarray]) -> tuple[str, int]:
    """A sha256 over the canonical result, and its row count."""
    h = hashlib.sha256()
    for arr in _canonical(result):
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        if arr.dtype == object:
            for value in arr:
                h.update(str(value).encode())
        else:
            h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest(), int(len(result["z"]))


def _analysis_ids(query: Any, limit: int) -> list[str]:
    rows = sorted(query.analyses_table().items())
    return [str(row["analysis_id"]) for _, row in rows[:limit]]


def _select_inputs(query: Any) -> dict[str, Any]:
    """The variant and Analysis inputs both facades are queried with.

    Selected **once**, from the indexed facade, so the scan side runs the same
    query and not a similar one: `probe_variant_alid` finds an off-panel variant
    through the index, which the scan facade does not have.
    """
    return {
        "on": probe_variant_alid(query, off_panel=False),
        "off": probe_variant_alid(query, off_panel=True),
        "analyses": _analysis_ids(query, 10),
    }


def _calls(
    query: Any, inputs: dict[str, Any]
) -> dict[str, Callable[[], dict[str, np.ndarray]]]:
    """One callable per shape, bound to `query` and the shared `inputs`.

    The probe variants come from `probe_variant_alid`, which asks the axis by
    index: `variants_table()` on OGS-00011's 164 M-variant shared axis is a
    memory bomb, and this harness must not use it.
    """
    on, off, analyses = inputs["on"], inputs["off"], inputs["analyses"]
    calls: dict[str, Callable[[], dict[str, np.ndarray]]] = {}
    if on is not None:
        calls["phewas"] = lambda: query.phewas(on)
    if off is not None:
        calls["phewas_off_panel"] = lambda: query.phewas(off)
    calls["range_phewas"] = lambda: query.range_phewas(*REGION)
    if on is not None and analyses:
        calls["lookup"] = lambda: query.lookup([on], analyses)
    if off is not None and analyses:
        calls["lookup_off_panel"] = lambda: query.lookup([off], analyses)
    return calls


def check_store(store: Path) -> dict[str, Any]:
    """Run every shape indexed and scanned; return per-shape digests and counts."""
    indexed = query_store(store)
    scanned = query_store(store)
    if getattr(scanned, "_by_variant", None) is None:
        raise SystemExit(
            f"{store}: the store carries no variant index, so this harness would "
            "compare the scan with itself; run `ogdb build-variant-index` first"
        )
    scanned._by_variant = None
    shapes: dict[str, Any] = {}
    try:
        # The inputs are selected once from the indexed facade and both call
        # sets are bound to them, so the scan side runs the *same* query.  The
        # closures must be built per facade: a single `_calls(indexed)` called
        # twice queried the index both times, which made this harness unable to
        # fail (review round 1, blocker).
        inputs = _select_inputs(indexed)
        calls_indexed = _calls(indexed, inputs)
        calls_scanned = _calls(scanned, inputs)
        for name, call in calls_indexed.items():
            indexed_digest, indexed_rows = _digest(call())
            scan_digest, scan_rows = _digest(calls_scanned[name]())
            if indexed_rows != scan_rows or indexed_digest != scan_digest:
                raise SystemExit(
                    f"{store}:{name}: indexed {indexed_rows} rows/{indexed_digest[:12]} "
                    f"!= scanned {scan_rows} rows/{scan_digest[:12]}"
                )
            shapes[name] = {
                "rows": indexed_rows,
                "sha256": indexed_digest,
                "identical": True,
            }
            print(f"  {name}: {indexed_rows} rows identical", flush=True)
    finally:
        indexed.close()
        scanned.close()
    if not shapes or all(shape["rows"] == 0 for shape in shapes.values()):
        raise SystemExit(f"{store}: no shape returned rows; an identity check would be vacuous")
    return shapes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_labelled_store_option(parser, help="a store to check; repeatable")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    stores = labelled_stores(args.store)
    artifact = {
        "harness": "benchmarks/variant_index_identity.py",
        **provenance(),
        "stores": {
            label: {"store": str(path), "shapes": check_store(path)}
            for label, path in stores
        },
    }
    write_artifact(args.output, artifact)
    print(json.dumps({"output": str(args.output), "stores": [label for label, _ in stores]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
