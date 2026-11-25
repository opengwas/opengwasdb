#!/usr/bin/env python3
"""ADR 0102's size table, generated from `inventory.py` output (#463). Apparent MB = bytes / 1e6.

    size_table.py INVENTORY_JSON[:dense] ...

Each argument is one column: an inventory JSON, or `JSON:dense` for that Hybrid release's nested
Dense Component. A component a release lacks prints as an em dash.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

Release = dict[str, Any]
DROPPED_TIER = ("abs_z", "p_value")
ABBREV = {"reference_completed": "RC", "observed_only": "OO"}


def file_bytes(r: Release, name: str) -> int | None:
    entry = r["entries"].get(name)
    return None if entry is None else int(entry["bytes"])


def file_pair(r: Release, a: str, b: str) -> int | None:
    x, y = file_bytes(r, a), file_bytes(r, b)
    return None if x is None and y is None else (x or 0) + (y or 0)


def arrays(r: Release, pred: Callable[[str], bool]) -> int | None:
    hits = [m["bytes"] for k, m in r["entries"]["data.zarr"]["arrays"].items() if pred(k)]
    return sum(hits) if hits else None


def leaf(k: str) -> str:
    return k.split("/")[-1]


def plain(k: str) -> bool:
    return not k.startswith(("top_hits/", "indexed_subsets/", "ragged/by_variant/"))


def side_table(k: str) -> bool:
    name = leaf(k)
    return name == "se_coefficients" or "_exception_" in name or "_overflow_" in name


def sqlite_table(r: Release, table: str) -> int | None:
    sq = r["entries"].get("index.sqlite", {}).get("sqlite", {})
    meta = sq.get("tables", {}).get(table)
    if meta is None:
        return None
    index = sum(v or 0 for k, v in sq.get("indexes", {}).items() if k.endswith(f"_{table}_1"))
    return (meta["bytes"] or 0) + index


ROWS: list[tuple[str, Callable[[Release], int | None]]] = [
    ("`manifest.json`", lambda r: file_bytes(r, "manifest.json")),
    ("`analyses.tsv`", lambda r: file_bytes(r, "analyses.tsv")),
    ("`overview.html`", lambda r: file_bytes(r, "overview.html")),
    ("`regions.tsv`", lambda r: file_bytes(r, "regions.tsv")),
    ("`index.sqlite` (whole file)", lambda r: file_bytes(r, "index.sqlite")),
    ("&nbsp;&nbsp;`completion_quality` (table + its index)",
     lambda r: sqlite_table(r, "completion_quality")),
    ("&nbsp;&nbsp;`metadata` (table + its index)", lambda r: sqlite_table(r, "metadata")),
    ("&nbsp;&nbsp;`variant_aliases` (table + its index)",
     lambda r: sqlite_table(r, "variant_aliases")),
    ("`variants.tsv.gz`", lambda r: file_bytes(r, "variants.tsv.gz")),
    ("`variants.tsv.gz.tbi`", lambda r: file_bytes(r, "variants.tsv.gz.tbi")),
    ("`variant_offsets.npy`", lambda r: file_bytes(r, "variant_offsets.npy")),
    ("`variant_alid_bytes.npy` + `_rows.npy`",
     lambda r: file_pair(r, "variant_alid_bytes.npy", "variant_alid_rows.npy")),
    ("`variant_rsid_bytes.npy` + `_rows.npy`",
     lambda r: file_pair(r, "variant_rsid_bytes.npy", "variant_rsid_rows.npy")),
    ("`dense_to_shared.npy`", lambda r: file_bytes(r, "dense_to_shared.npy")),
    ("`data.zarr` `z`", lambda r: arrays(r, lambda k: plain(k) and leaf(k) == "z")),
    ("`data.zarr` `se`", lambda r: arrays(r, lambda k: plain(k) and leaf(k) == "se")),
    ("`data.zarr` `eaf`", lambda r: arrays(r, lambda k: plain(k) and leaf(k) == "eaf")),
    ("`data.zarr` `eaf_baseline`",
     lambda r: arrays(r, lambda k: plain(k) and leaf(k) == "eaf_baseline")),
    ("`data.zarr` `se_coefficients`, exception and overflow tables",
     lambda r: arrays(r, lambda k: plain(k) and side_table(k))),
    ("`data.zarr` `eaf_reference`",
     lambda r: arrays(r, lambda k: plain(k) and leaf(k) == "eaf_reference")),
    ("`data.zarr` `imputed`", lambda r: arrays(r, lambda k: plain(k) and leaf(k) == "imputed")),
    ("`data.zarr` `on_panel`", lambda r: arrays(r, lambda k: leaf(k) == "on_panel")),
    ("`data.zarr` `ragged/offsets`, `ragged/variant_index`",
     lambda r: arrays(r, lambda k: k in ("ragged/offsets", "ragged/variant_index"))),
    ("`data.zarr` `ragged/by_variant/*`",
     lambda r: arrays(r, lambda k: k.startswith("ragged/by_variant/"))),
    ("`data.zarr` `top_hits/*` read arrays",
     lambda r: arrays(r, lambda k: k.startswith("top_hits/") and leaf(k) not in DROPPED_TIER)),
    ("`data.zarr` `top_hits/*/abs_z`",
     lambda r: arrays(r, lambda k: k.startswith("top_hits/") and leaf(k) == "abs_z")),
    ("`data.zarr` `top_hits/*/p_value`",
     lambda r: arrays(r, lambda k: k.startswith("top_hits/") and leaf(k) == "p_value")),
    ("`data.zarr` `indexed_subsets/hm3`",
     lambda r: arrays(r, lambda k: k.startswith("indexed_subsets/"))),
    ("`data.zarr` group `zarr.json`", lambda r: int(r["entries"]["data.zarr"]["group_meta_bytes"])),
]


def fmt(b: int | None) -> str:
    if b is None:
        return "—"
    mb = b / 1e6
    if b == 0:
        return "0"
    return f"{mb:,.0f}" if mb >= 100 else (f"{mb:.1f}" if mb >= 0.1 else f"{mb:.3f}")


def column(arg: str) -> tuple[str, Release, int]:
    path, _, sub = arg.partition(":")
    inv = json.loads(Path(path).read_text())
    store = Path(inv["path"]).parent.name
    label = f"{store} {inv['primary_layout'].capitalize()} {ABBREV.get(inv['completion'], '?')}"
    if inv["primary_layout"] == "hybrid":
        label += " dense/" if sub else " top"
    release = inv["entries"][sub] if sub else inv
    return label, release, int(release["bytes"] if sub else inv["total_bytes"])


def main() -> None:
    cols = [column(a) for a in sys.argv[1:]]
    print("| Component | " + " | ".join(label for label, _, _ in cols) + " |")
    print("|---|" + "---:|" * len(cols))
    for name, fn in ROWS:
        print(f"| {name} | " + " | ".join(fmt(fn(r)) for _, r, _ in cols) + " |")
    print("| **Total** | " + " | ".join(fmt(total) for _, _, total in cols) + " |")


if __name__ == "__main__":
    main()
