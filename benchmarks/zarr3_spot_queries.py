"""Do existing Store Releases answer identically after a change? Spot queries, hashed (#244).

  record   run five spot queries on one Store Release (one Analysis genome-wide,
           a PheWAS, a 200 kb region, a lookup and the top hits) and hash every
           returned array: dtype, shape and values, so a re-ordered, re-typed or
           changed answer differs
  compare  compare two records, one line per store: identical, or the keys that differ

Record under each environment, then compare; #244 recorded OGS-00009 (Dense),
OGS-00001 (Ragged) and OGS-00004 (Hybrid) under zarr 2.18 and zarr 3:

    (cd /path/to/base && pixi run -e dev python /path/to/this/benchmarks/zarr3_spot_queries.py \\
        record /data/opengwasdb/stores/OGS-00001/store.opengwasdb /tmp/spot-base-OGS-00001.json)
    pixi run -e dev python benchmarks/zarr3_spot_queries.py \\
        record /data/opengwasdb/stores/OGS-00001/store.opengwasdb /tmp/spot-head-OGS-00001.json
    pixi run -e dev python benchmarks/zarr3_spot_queries.py \\
        compare /tmp/spot-base-OGS-00001.json /tmp/spot-head-OGS-00001.json

`record` imports nothing from `benchmarks`, so it runs under any checkout's
environment; `opengwasdb` is whichever that environment installs.

`--shapes` limits the record to named queries. A Ragged or Hybrid store's
phewas, regional and top-hit-scan shapes run through #252's variant-side paths,
so its identity record names only the Analysis-side shapes (`--shapes analysis
lookup` for Ragged, `--shapes analysis` for Hybrid) rather than paying an O(n)
scan for an answer #253 does not change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SHAPES = ["analysis", "phewas", "regional", "lookup", "top_hits"]


def _hash_array(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(str(values.shape).encode())
    if values.dtype.kind in "OUSV":
        # `tobytes` on an object or string array hashes pointers, which differ run
        # to run; the values themselves are what must agree.
        digest.update(repr(values.tolist()).encode())
    else:
        digest.update(values.tobytes())
    return digest.hexdigest()[:16]


def _hash_result(result: dict[str, Any]) -> dict[str, object]:
    return {
        "n_rows": len(next(iter(result.values()))) if result else 0,
        "arrays": {key: _hash_array(np.asarray(value)) for key, value in sorted(result.items())},
    }


def _hits(analyses: dict[int, dict[str, Any]], index: int) -> int:
    try:
        return int(analyses[index].get("n_hits_5e8") or 0)
    except (TypeError, ValueError):
        return 0


def record(store_path: Path, out_path: Path, shapes: list[str] | None = None) -> None:
    from opengwasdb.query import query_store

    shapes = SHAPES if shapes is None else shapes
    rec: dict[str, object] = {"store": str(store_path)}
    with query_store(store_path) as q:
        analyses = q.analyses_table()
        variants = q.variants_table()
        rec["n_analyses"] = len(analyses)
        rec["n_variants"] = len(variants)
        if not (analyses and variants):
            raise SystemExit("the store has no axes; the spot queries would prove nothing")
        # The Analysis with the most genome-wide hits, so `top_hits` returns rows.
        exposure_index = max(sorted(analyses), key=lambda i: _hits(analyses, i))
        exposure = analyses[exposure_index]["analysis_id"]
        rec["exposure_n_hits_5e8"] = _hits(analyses, exposure_index)
        variant = variants[0]
        identifier = variant["alid"]
        chromosome = str(variant["chromosome"])
        start = max(0, int(variant["position"]) - 100_000)
        end = int(variant["position"]) + 100_000
        others = [analyses[index]["analysis_id"] for index in list(analyses)[:3]]
        rec["exposure"] = exposure
        rec["identifier"] = identifier
        queries = {
            "analysis": lambda: q.analysis(exposure),
            "phewas": lambda: q.phewas(identifier),
            "regional": lambda: q.range_phewas(chromosome, start, end),
            "lookup": lambda: q.lookup([identifier], [exposure, *others]),
            "top_hits": lambda: q.top_hits(analysis_id=exposure, threshold=5e-8),
        }
        for name in shapes:
            rec[name] = _hash_result(queries[name]())
    text = json.dumps(rec, indent=2, sort_keys=True)
    out_path.write_text(text + "\n", encoding="utf-8")
    print(text)


def compare(base_path: Path, head_path: Path) -> int:
    base = json.loads(base_path.read_text())
    head = json.loads(head_path.read_text())
    shapes = [name for name in SHAPES if name in base]
    if not all(base[name]["n_rows"] > 0 for name in shapes if name != "lookup"):
        raise SystemExit("the base record has an empty shape; identity would prove nothing")
    differ = sorted(k for k in set(base) | set(head) if base.get(k) != head.get(k))
    summary = {
        "store": base["store"],
        "identical": not differ,
        "differing_keys": differ,
        "rows": {name: base[name]["n_rows"] for name in shapes},
        "arrays_per_shape": {name: len(base[name]["arrays"]) for name in shapes},
    }
    print(json.dumps(summary))
    return 0 if not differ else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("store", type=Path)
    rec.add_argument("output", type=Path)
    rec.add_argument(
        "--shapes",
        nargs="+",
        choices=SHAPES,
        default=SHAPES,
        help="which spot queries to run; a Ragged or Hybrid store's variant-side "
        "shapes are #252's, so its identity record names only the Analysis-side ones",
    )
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("base", type=Path)
    cmp_.add_argument("head", type=Path)
    args = ap.parse_args()
    if args.command == "record":
        record(args.store, args.output, args.shapes)
        return 0
    return compare(args.base, args.head)


if __name__ == "__main__":
    raise SystemExit(main())
