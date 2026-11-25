#!/usr/bin/env python3
"""ADR 0106 (#418): the step's compute path on a published Store's 5e-8 tier, against the
prototype's records, read only.

    parity.py STORE PROTOTYPE.jsonl --ld-store EUR=PATH --scratch DIR --output OUT.json

`opengwasdb clump` refuses a Store of any format but 0.3.0 (ADR 0106 §10), and OGS-00009 and
OGS-00027 have no 0.3.0 release until opengwas/opengwasdb-stores#498 rebuilds them, so this
runs `compute_clumps`, the step without its writes and its format refusal, on their published
0.2.0 tiers. Every Analysis of both is EUR, as the prototype clumped them
(`/data/opengwasdb/work/clump-62/time_clumps.py --all`). Nothing is written to the Store.

The record gives the leads against the prototype's, the time against the prototype's, the
arrays' size as the step would store them (written under DIR, never into the Store), and the
loci lost entirely: runs of an Analysis's absent rows, consecutive ones less than G apart, with
no row the LD Store holds within G of the run, at G = 250 kb and 1 Mb (a reporting choice).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from opengwasdb.clumps import (
    ABSENT,
    ClumpResult,
    Tiers,
    ancestries,
    close_ld_stores,
    compute_clumps,
    open_ld_stores,
    tier_components,
)
from opengwasdb.store import open_store
from opengwasdb.store.arrays import create_array, open_group_for_write
from opengwasdb.store_reader.layout import ArrayRole
from opengwasdb.store_reader.top_hits import CLUMPS_ARRAYS
from opengwasdb.top_hits import TOP_HIT_CHUNK_SIZE

GAPS = {"250kb": 250_000, "1mb": 1_000_000}


def lost(lead: np.ndarray, chrom: np.ndarray, position: np.ndarray, gap: int) -> int:
    """Runs of absent rows with no present row within `gap` of the run."""
    count = 0
    for c in np.unique(chrom[lead == ABSENT]):
        on = chrom == c
        absent = np.sort(position[on & (lead == ABSENT)])
        present = np.sort(position[on & (lead >= 0)])
        for run in np.split(absent, np.flatnonzero(np.diff(absent) >= gap) + 1):
            k = np.searchsorted(present, run[0] - gap)
            count += int(k == len(present) or present[k] > run[-1] + gap)
    return count


def per_analysis(tiers: Tiers, result: ClumpResult, index: int) -> dict[str, int] | None:
    part = tiers.slice(index)
    if not len(part):
        return None
    lead = np.concatenate([result.clumps[i].lead[a:b] for i, a, b in part.parts])
    chrom = tiers.identity.chromosome[part.identity]
    position = tiers.identity.position[part.identity]
    row = {"present": int((lead >= 0).sum()), "absent": int((lead == ABSENT).sum()),
           "leads": int(result.n_loci[index] or 0)}
    row.update({f"lost_{name}": lost(lead, chrom, position, gap) for name, gap in GAPS.items()})
    return row


def stored_bytes(result: ClumpResult, scratch: Path) -> int:
    group = open_group_for_write(scratch, "w", zarr_format=3)
    for c, clumped in enumerate(result.clumps):
        g = group.create_group(f"c{c}")
        values = {"lead": clumped.lead, "r": clumped.r}
        for name, dtype in CLUMPS_ARRAYS.items():
            create_array(g, name, ArrayRole.TOP_HIT_INDEX, data=values[name], dtype=dtype,
                         hint=TOP_HIT_CHUNK_SIZE)
    return sum(f.stat().st_size for f in scratch.rglob("*") if f.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("store", type=Path)
    parser.add_argument("prototype", type=Path)
    parser.add_argument("--ld-store", required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ancestry, location = args.ld_store.split("=", 1)

    release = open_store(args.store)
    started = time.perf_counter()
    tiers = Tiers.read(release, tier_components(release, len(ancestries(release))))
    read_s = time.perf_counter() - started
    opened = open_ld_stores({ancestry: location},
                            reference_assembly=release.manifest.reference_assembly)
    try:
        started = time.perf_counter()
        result = compute_clumps(release, opened, tiers=tiers)
        clump_s = time.perf_counter() - started
    finally:
        close_ld_stores(opened)

    rows = {i: r for i in range(len(result.n_loci)) if (r := per_analysis(tiers, result, i))}
    proto = {rec["analysis_index"]: rec for rec in map(json.loads, args.prototype.read_text()
                                                       .splitlines())}
    compared = sorted(set(rows) & {k for k, v in proto.items() if "leads" in v})
    with_present = [r for r in rows.values() if r["present"] > 0]
    n_rows = sum(c.n_rows for c in result.components)
    size = stored_bytes(result, args.scratch)
    summary: dict[str, Any] = {
        "store": args.store.parent.name,
        "format_version": release.manifest.format_version,
        "analyses_with_rows": len(rows),
        "compared_with_prototype": len(compared),
        "n_loci_5e8": sum(r["leads"] for r in rows.values()),
        "prototype_leads": sum(proto[k]["leads"] for k in compared),
        "analyses_whose_leads_differ": sum(rows[k]["leads"] != proto[k]["leads"]
                                           for k in compared),
        "analyses_whose_present_or_absent_rows_differ": sum(
            (rows[k]["present"], rows[k]["absent"]) != (proto[k]["present"], proto[k]["absent"])
            for k in compared),
        "markers": result.markers(),
        "analyses_every_observed_row_absent": sum(r["present"] == 0 for r in rows.values()),
        "loci_lost": {name: sum(r[f"lost_{name}"] for r in rows.values()) for name in GAPS},
        "loci_lost_in_analyses_with_a_present_row": {
            name: sum(r[f"lost_{name}"] for r in with_present) for name in GAPS},
        "seconds": {"read_tiers_and_identity": round(read_s, 1), "clump": round(clump_s, 1),
                    "total": round(read_s + clump_s, 1)},
        "prototype_seconds": {
            "clump": round(sum(v.get("clump_s", 0) for v in proto.values()), 1),
            "identity": round(sum(v.get("identity_s", 0) for v in proto.values()), 1)},
        "tier_rows": n_rows,
        "stored_bytes": size,
        "bytes_per_row": round(size / n_rows, 3),
    }
    args.output.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
