#!/usr/bin/env python3
"""ADR 0106 (#418): `n_loci_5e8` against `plink2 --clump` for #403's 37 EUR Analyses.

    plink_compare.py SWEEP.json --ld-store EUR=PATH --output OUT.json [--r2 R2 --kb KB]

#403's plink2 counts (`eur_sweep.repro.json`, tag `locus-leads-prototype`): `--clump-unphased`,
no PC Adjustment, kb 10,000, r2 0.01 and 0.001, p1 = p2 = 5e-8, on the HGDP+1kGP EUR pgen, over
each Analysis's observed 5e-8 rows that matched that panel. Here: the step's compute path on the
same Analyses' published tiers, read only, against the EUR Genotype Store (PC-adjusted, k = 10).

By default at the stored configuration, r2 = the store's t_far and kb 10,000. `--r2` and `--kb`
measure another configuration, to tell the threshold's effect from the PC Adjustment's: the step
stores only the default, so this sets the module's window and the mapped store's r2 for the run.
A Store the reader cannot open is skipped and named.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import opengwasdb.clumps as clumps
from opengwasdb.store import open_store
from opengwasdb.store_reader.analyses import read_analyses
from opengwasdb.store_reader.format import UnsupportedFormatVersion

STORES = Path("/data/opengwasdb/stores")


def store_rows(store: str, analyses: list[dict[str, Any]], location: str,
               r2: float | None) -> list[dict[str, Any]]:
    release = open_store(STORES / store / "store.opengwasdb")
    ids = {r["analysis_id"]: int(r["analysis_index"])
           for r in read_analyses(release.analyses_path).rows}
    n = len(clumps.ancestries(release))
    tiers = clumps.Tiers.read(release, clumps.tier_components(release, n))
    opened = clumps.open_ld_stores({"EUR": location},
                                   reference_assembly=release.manifest.reference_assembly)
    if r2 is not None:
        opened["EUR"].identity["r2"] = r2
    try:
        wanted = [ids[a["analysis"]["analysis_id"]] for a in analyses]
        result = clumps.compute_clumps(release, opened, tiers=tiers, analyses=wanted)
    finally:
        clumps.close_ld_stores(opened)
    rows = []
    for a in analyses:
        index = ids[a["analysis"]["analysis_id"]]
        part = tiers.slice(index)
        lead = np.concatenate([result.clumps[i].lead[x:y] for i, x, y in part.parts])
        plink = a["configs"][0]["by_r2"]
        rows.append({
            "store": store, "analysis_id": a["analysis"]["analysis_id"],
            "present": int((lead >= 0).sum()), "absent": int((lead == clumps.ABSENT).sum()),
            "ambiguous": int((lead == clumps.AMBIGUOUS).sum()),
            "plink_rows": a["rows"]["shared_set"], "n_loci_5e8": result.n_loci[index],
            "plink_r2_0.01": plink["0.01"]["index_variants"],
            "plink_r2_0.001": plink["0.001"]["index_variants"],
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sweep", type=Path)
    parser.add_argument("--ld-store", required=True)
    parser.add_argument("--r2", type=float)
    parser.add_argument("--kb", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    location = args.ld_store.split("=", 1)[1]
    if args.kb is not None:
        clumps.KB = args.kb  # a measured configuration, never a stored one
    by_store: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for analysis in json.loads(args.sweep.read_text())["analyses"]:
        by_store[analysis["store"]].append(analysis)
    rows: list[dict[str, Any]] = []
    skipped = []
    for store, analyses in sorted(by_store.items()):
        try:
            rows += store_rows(store, analyses, location, args.r2)
        except (UnsupportedFormatVersion, clumps.ClumpRefusal) as exc:  # named, not hidden
            skipped.append({"store": store, "analyses": len(analyses),
                            "reason": f"{type(exc).__name__}: {exc}"[:300]})
    loci = np.array([r["n_loci_5e8"] for r in rows], dtype=float)
    plink = np.array([r["plink_r2_0.01"] for r in rows], dtype=float)
    summary = {
        "configuration": {"r2": args.r2 if args.r2 is not None else "t_far",
                          "kb": clumps.KB},
        "analyses": len(rows), "skipped": skipped,
        "n_loci_5e8": int(loci.sum()), "plink_r2_0.01": int(plink.sum()),
        "plink_r2_0.001": int(sum(r["plink_r2_0.001"] for r in rows)),
        "ratio_to_plink_0.01": {"pooled": round(float(loci.sum() / plink.sum()), 3),
                                "median": round(float(np.median(loci / plink)), 3)},
        "analyses_fewer_equal_more_than_plink_0.01": [
            int((loci < plink).sum()), int((loci == plink).sum()), int((loci > plink).sum())],
        "rows_present": sum(r["present"] for r in rows),
        "plink_rows": sum(r["plink_rows"] for r in rows),
        "rows_absent": sum(r["absent"] for r in rows),
        "rows_ambiguous": sum(r["ambiguous"] for r in rows),
    }
    args.output.write_text(json.dumps({"summary": summary, "analyses": rows}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
