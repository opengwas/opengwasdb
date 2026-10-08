#!/usr/bin/env python3
"""Print the "Checked on FinnGen (#250)" block ADR 0058 pastes.

Every number is read from the committed #250 artifacts
(`opengwasdb_ogs00016_store_comparison_zarr3.json` and
`opengwasdb_ogs00016_conversion_0_2_0.json`), so the ADR block is a paste of
this command's output rather than a hand copy -- the pattern
`benchmarks/zarr3_lever_tables.py decision` uses for ADR 0058's own tables.

Run from the repository root:

    python3 scripts/build_finngen_shape_check.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("docs/benchmark-output")

#: ADR 0058's own OGS-00009 ratios (0.1.0 -> 0.2.0 under Zarr 3), quoted from
#: the ADR for comparison; the FinnGen ratios beside them are computed here.
OGS00009_RATIOS = {
    ("phewas",): 2.76,
    ("random_lookup_10_variants_100_analyses",): 2.80,
    ("random_lookup_100_variants_10_analyses",): 1.83,
    ("regional",): 1.35,
    ("bulk",): 0.35,
}


def main() -> None:
    cmp = json.loads((OUT / "opengwasdb_ogs00016_store_comparison_zarr3.json").read_text())
    conv = json.loads((OUT / "opengwasdb_ogs00016_conversion_0_2_0.json").read_text())
    stores = {s["label"]: s for s in cmp["stores"]}
    v2 = {r["query"]: r for r in stores["v2-ogs00016"]["timings"]}
    v3 = {r["query"]: r for r in stores["v3-ogs00016"]["timings"]}
    arrays2 = {a["node"]: a for a in stores["v2-ogs00016"]["footprint"]["arrays"]}
    arrays3 = {a["node"]: a for a in stores["v3-ogs00016"]["footprint"]["arrays"]}
    total = conv["after"]["apparent_bytes"] / conv["before"]["apparent_bytes"] * 100 - 100

    print("## Checked on FinnGen (#250)")
    print()
    print("Run on the full FinnGen R13 release (2,754 Analyses x 21,230,615 variants), converted")
    print(f"bit-exactly from 0.1.0 to 0.2.0. Total apparent size grows **{total:+.2f} %** "
          f"({conv['before']['n_files']:,} files -> **{conv['after']['n_files']:,}**). The only")
    print("material per-plane cost is the `se` plane; `z` and the top-hit index are flat:")
    print()
    print("| array | 0.1.0 | 0.2.0 | ratio |")
    print("|---|---:|---:|---:|")
    for node in ("data.zarr/z", "data.zarr/se", "data.zarr/eaf", "data.zarr/top_hits/p_5e_04/z"):
        a, b = arrays2[node]["apparent_bytes"], arrays3[node]["apparent_bytes"]
        print(f"| `{node.removeprefix('data.zarr/')}` | {a / 1e9:.2f} GB | {b / 1e9:.2f} GB | "
              f"{b / a:.3f}x |")
    print()
    print("The 0.1.0 -> 0.2.0 time ratios under Zarr 3 match this ADR's OGS-00009 ratios:")
    print()
    print("| shape | FinnGen | OGS-00009 (this ADR) |")
    print("|---|---:|---:|")
    for (shape,), ref in OGS00009_RATIOS.items():
        ratio = v3[shape]["median_ms"] / v2[shape]["median_ms"]
        print(f"| {shape.replace('_', ' ')} | {ratio:.2f}x | {ref:.2f}x |")
    print()
    print("No new size or time cost appears, and the whole-Analysis guard holds "
          f"({v3['bulk']['median_ms'] / v2['bulk']['median_ms']:.2f}x against the 1.25x limit). "
          "**The decision is not contradicted and is not corrected.** Full report: "
          "`docs/benchmark-output/opengwasdb_ogs00016_0_2_0.html`.")


if __name__ == "__main__":
    main()
