#!/usr/bin/env python3
"""Per-column disagreement between a Hybrid's two variant tables (ADR 0102 §4, #463, #481).

    hybrid_columns.py RELEASE OUT_JSON

Read-only. Walks the Dense Component's `dense/variants.tsv.gz` beside the shared
`variants.tsv.gz`, comparing each Dense Component row with the shared row
`dense/dense_to_shared.npy` names, column by column (all but `variant_index`). Counts the
disagreements per column and per pattern, and whether each `rsid` or `source_alid`
disagreement is the Dense Component's `.` against a shared value.
"""

from __future__ import annotations

import gzip
import json
import sys
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

COLS = ["chromosome", "position", "variant_index", "effect_allele", "other_allele", "alid",
        "rsid", "source_alid"]


def lines(path: Path) -> Iterator[list[str]]:
    with gzip.open(path, "rt", newline="\n") as fh:
        for line in fh:
            if not line.startswith("#"):
                yield line.rstrip("\n").split("\t")


def compare(rel: Path) -> dict[str, Any]:
    d2s = np.load(rel / "dense" / "dense_to_shared.npy")
    shared = lines(rel / "variants.tsv.gz")
    per_col: Counter[str] = Counter()
    patterns: Counter[str] = Counter()
    examples: dict[str, Any] = {}
    dot = {"rsid": 0, "source_alid": 0}
    other = {"rsid": 0, "source_alid": 0}
    cursor, s = -1, [""] * len(COLS)
    for i, d in enumerate(lines(rel / "dense" / "variants.tsv.gz")):
        while cursor < int(d2s[i]):
            s, cursor = next(shared), cursor + 1
        bad = [c for j, c in enumerate(COLS) if j != 2 and d[j] != s[j]]
        if not bad:
            continue
        key = "+".join(bad)
        patterns[key] += 1
        examples.setdefault(key, {"dense": d, "shared": s})
        per_col.update(bad)
        for c in ("rsid", "source_alid"):
            if c in bad:
                j = COLS.index(c)
                if d[j] == "." and s[j] != ".":
                    dot[c] += 1
                else:
                    other[c] += 1
    return {
        "per_column": dict(per_col), "patterns": dict(patterns), "examples": examples,
        "dense_dot_where_shared_set": dot, "other_disagreements": other,
    }


def main() -> None:
    rel, out = Path(sys.argv[1]), Path(sys.argv[2])
    t0 = time.perf_counter()
    result = compare(rel)
    result["seconds"] = round(time.perf_counter() - t0, 1)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "examples"}, indent=1))


if __name__ == "__main__":
    main()
