#!/usr/bin/env python3
"""Does a Hybrid `lookup` by rsID find on-panel cells its Dense Component's table lost? (#481)

    rsid_lookup_probe.py RELEASE RSID ALID

Read-only. RSID and ALID name the same variant. `phewas` resolves through the shared variant
table; the Hybrid `lookup` resolves through the nested Dense Component's own rsID index
(ADR 0102 §4). Prints the row counts of both queries by both identifiers, over the first five
Analyses `phewas` returns.
"""

from __future__ import annotations

import csv
import sys

import opengwasdb
from opengwasdb.query import query_store


def main() -> None:
    release, rsid, alid = sys.argv[1:4]
    print("opengwasdb from", opengwasdb.__file__)
    query = query_store(release)
    with open(f"{release}/analyses.tsv", newline="") as fh:
        rows = csv.DictReader(fh, delimiter="\t")
        by_index = {int(r["analysis_index"]): r["analysis_id"] for r in rows}
    phewas = query.phewas(rsid)
    seen = dict.fromkeys(int(i) for i in phewas["analysis_index"].tolist())
    ids = [by_index[i] for i in seen][:5]
    print("phewas(rsid) rows:", phewas["variant_index"].size, "first Analyses:", ids)
    print("phewas(alid) rows:", query.phewas(alid)["variant_index"].size)
    print("lookup([alid], those Analyses) rows:", query.lookup([alid], ids)["variant_index"].size)
    print("lookup([rsid], those Analyses) rows:", query.lookup([rsid], ids)["variant_index"].size)


if __name__ == "__main__":
    main()
