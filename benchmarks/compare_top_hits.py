#!/usr/bin/env python3
"""#312's read-only proof: inline versus standalone Top-Hit Index arrays and attributes.

Recovered from #322's scratch harness. Counts hits in each 1,024-Analysis
column band (ADR 0058), including bands the small inline/standalone fixture
never reached. This command never rebuilds an index or writes either Store.

    compare_top_hits.py INLINE_STORE STANDALONE_STORE --output OUT.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from benchmarks._artifact import provenance
from benchmarks.compare_store_releases import compare_values, discover_arrays
from opengwasdb.store.open import open_store

PREFIX = "data.zarr/top_hits/"
COLUMN_BAND = 1024


def tier_arrays(root: Path) -> dict[str, Path]:
    return {rel: root / rel for rel in discover_arrays(root) if rel.startswith(PREFIX)}


def _compare_arrays(inline: dict[str, Path], standalone: dict[str, Path]) -> dict[str, Any]:
    arrays = {}
    for rel in sorted(set(inline) | set(standalone)):
        if rel not in inline or rel not in standalone:
            arrays[rel] = {"only": "inline" if rel in inline else "standalone"}
            continue
        report = compare_values(inline[rel], standalone[rel])
        array = zarr.open_array(inline[rel], mode="r")
        arrays[rel] = {"shape": list(array.shape), "dtype": str(array.dtype), **report}
    return arrays


def _tier_records(inline: Any, standalone: Any) -> dict[str, Any]:
    left = set(inline.group_keys())
    right = set(standalone.group_keys()) if standalone is not None else set()
    records = {}
    for tier in sorted(left | right):
        attrs_i = dict(inline[tier].attrs) if tier in left else None
        attrs_s = dict(standalone[tier].attrs) if tier in right else None
        record = {"attrs_equal": attrs_i == attrs_s, "attrs": attrs_i}
        if tier in left and "analysis_index" in inline[tier]:
            analyses = np.asarray(inline[f"{tier}/analysis_index"][:])
            record["hits"] = int(analyses.size)
            record["hits_by_column_band"] = np.bincount(analyses // COLUMN_BAND).tolist()
        records[tier] = record
    return records


def compare_top_hits(inline: Path, standalone: Path) -> dict[str, Any]:
    """Compare the same decoded arrays and attributes as the #312 fixture proof."""
    arrays_i, arrays_s = tier_arrays(inline), tier_arrays(standalone)
    if not arrays_i:
        raise SystemExit("the inline release holds no Top-Hit Index: nothing would be proven")
    arrays = _compare_arrays(arrays_i, arrays_s)
    root_i = open_store(inline).arrays(mode="r")["top_hits"]
    right = open_store(standalone).arrays(mode="r")
    root_s = right["top_hits"] if "top_hits" in right else None
    group_attrs_equal = root_s is not None and dict(root_i.attrs) == dict(root_s.attrs)
    tiers = _tier_records(root_i, root_s)
    differing = sorted(
        rel
        for rel, row in arrays.items()
        if row.get("only") or not row.get("comparable") or row.get("differing_cells")
    )
    return {
        "inline": str(inline),
        "standalone": str(standalone),
        "identical": not differing
        and group_attrs_equal
        and all(tier["attrs_equal"] for tier in tiers.values()),
        "n_arrays": {"inline": len(arrays_i), "standalone": len(arrays_s)},
        "arrays_differing": differing,
        "group_attrs_equal": group_attrs_equal,
        "group_attrs": dict(root_i.attrs),
        "tiers": tiers,
        "arrays": arrays,
    }


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("inline", type=Path)
    parser.add_argument("standalone", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(raw)
    if any(
        args.output.resolve().is_relative_to(store.resolve())
        for store in (args.inline, args.standalone)
    ):
        raise SystemExit("the comparison report must be outside both input Stores")
    result = compare_top_hits(args.inline, args.standalone)
    result.update({"generator": "benchmarks/compare_top_hits.py", "argv": raw, **provenance()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("identical", "n_arrays", "arrays_differing", "group_attrs_equal")
            }
        )
    )
    for tier, record in result["tiers"].items():
        print(tier, record.get("hits"), record.get("hits_by_column_band"), record["attrs_equal"])
    return 0 if result["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
