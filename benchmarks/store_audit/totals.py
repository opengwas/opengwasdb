#!/usr/bin/env python3
"""Per-release totals for ADR 0102 (#463): the planes, the accelerators, and what 0.3.0 removes.

    totals.py INVENTORY_JSON ... [--output OUT.json]

"Removed" is gross: the top-hit tiers' `abs_z` and `p_value`, `variant_offsets.npy`, an
Observed-Only release's whole `index.sqlite`, and a Hybrid Dense Component's own variant table,
sidecars, `analyses.tsv` and `overview.html`. It nets off neither the sidecars 0.3.0 adds nor the
32-byte ALID index it replaces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

#: The Dense Component files a 0.3.0 Hybrid no longer carries (ADR 0102 §4, #472).
HYBRID_DUPLICATION = (
    "variants.tsv.gz", "variants.tsv.gz.tbi", "variant_offsets.npy",
    "variant_alid_bytes.npy", "variant_alid_rows.npy",
    "variant_rsid_bytes.npy", "variant_rsid_rows.npy", "analyses.tsv", "overview.html",
)


def group_sum(release: dict[str, Any], pred: Any) -> int:
    arrays = release["entries"]["data.zarr"]["arrays"]
    return sum(m["bytes"] for k, m in arrays.items() if pred(k))


def component_totals(release: dict[str, Any]) -> dict[str, int]:
    e = release["entries"]
    tables = e.get("index.sqlite", {}).get("sqlite", {}).get("tables", {})
    quality = tables.get("completion_quality")
    return {
        "top_hits": group_sum(release, lambda k: k.startswith("top_hits/")),
        "top_hits_abs_z_p_value": group_sum(
            release,
            lambda k: k.startswith("top_hits/") and k.split("/")[-1] in ("abs_z", "p_value"),
        ),
        "indexed_subsets": group_sum(release, lambda k: k.startswith("indexed_subsets/")),
        "planes_z_se_eaf": group_sum(
            release,
            lambda k: k.split("/")[-1] in ("z", "se", "eaf")
            and not k.startswith(("top_hits/", "indexed_subsets/", "ragged/by_variant/")),
        ),
        "variant_offsets": e["variant_offsets.npy"]["bytes"],
        "variants_tsv": e["variants.tsv.gz"]["bytes"],
        "tbi": e["variants.tsv.gz.tbi"]["bytes"],
        "index_sqlite": e["index.sqlite"]["bytes"],
        "completion_quality_rows": quality["rows"] if quality else 0,
    }


def removed(t: dict[str, int]) -> int:
    gone = t["top_hits_abs_z_p_value"] + t["variant_offsets"]
    return gone if t["completion_quality_rows"] else gone + t["index_sqlite"]


def release_row(inv: dict[str, Any]) -> dict[str, Any]:
    t = component_totals(inv)
    row: dict[str, Any] = {
        "layout": inv["primary_layout"], "completion": inv["completion"],
        "format": inv["format_version"], "total": inv["total_bytes"], "top": t,
    }
    drop = removed(t)
    if "dense" in inv["entries"]:
        dense = inv["entries"]["dense"]
        d = component_totals(dense)
        row["dense"] = d
        row["hybrid_duplication"] = sum(dense["entries"][n]["bytes"] for n in HYBRID_DUPLICATION)
        drop += row["hybrid_duplication"] + d["top_hits_abs_z_p_value"]
        if not d["completion_quality_rows"]:
            drop += d["index_sqlite"]
    row["removed_at_030"] = drop
    row["removed_pct"] = round(100 * drop / inv["total_bytes"], 2)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("inventories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = {}
    for path in args.inventories:
        inv = json.loads(path.read_text())
        name, row = Path(inv["path"]).parent.name, release_row(inv)
        out[name] = row
        t = row["top"]
        print(
            name, row["layout"], row["completion"], row["format"],
            f"total={row['total']:,}", f"removed={row['removed_at_030']:,} ({row['removed_pct']}%)",
            f"planes={t['planes_z_se_eaf']:,} ({100 * t['planes_z_se_eaf'] / row['total']:.2f}%)",
            f"hm3={100 * t['indexed_subsets'] / row['total']:.2f}%",
            f"top_hits={t['top_hits']:,} abs_z+p_value={t['top_hits_abs_z_p_value']:,}",
            f"hybrid_duplication={row.get('hybrid_duplication', 0):,}",
        )
    if args.output:
        args.output.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
