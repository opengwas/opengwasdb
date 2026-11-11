#!/usr/bin/env python3
"""Write each HGDP+1kGP panel's keep file: its unrelated samples (ADR 0002, decision 11).

The owner decided (10 November 2026, epic opengwas/ogld#1) that an LD Store is built from
unrelated samples only. The HGDP+1kGP members files include 1kGP's trios: EUR has 197
first-degree pairs among its 771 samples, which put chance cross-chromosome LD at 1.76x what n
predicts; without the relatives it is 1.07x. Every panel uses gnomAD v3.1.2's maximal set of
unrelated samples (`relatedness_inference.related` false in its sample metadata), which leaves
no first- or second-degree pair.

A panel's samples are the ones `scripts/make_pcs.py` made its PC File for, the members present
in v2, read from that PC File by the build's own reader (`ogld.build.inputs.pc_file_iids`, so a
missing or empty one is refused); the keep file is those that are unrelated, in the PC File's
order, under plink2's `#IID` header. A sample the metadata lacks is refused, naming it. The
same inputs write the same bytes. Expected counts: EUR 665, AFR 608, EAS 724, SAS 672, AMR 341.

    pixi run -e dev python scripts/make_keeps.py [--panel EUR ...]

Writes `<out>/<panel>.keep` (default `$OGLD_KEEP`, `/data/opengwasdb/work/ogld/keep`). The PC
Files are not changed: they were computed on unrelated samples with the relatives projected.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from test_panel.roots import META_NAME, root
from test_panel.tools import relatedness

from ogld.build.inputs import pc_file_iids
from ogld.format import InputError

PANELS = ("EUR", "AFR", "EAS", "SAS", "AMR")


def unrelated(pc_file: Path, meta: Path) -> list[str]:
    """A panel's samples that gnomAD v3.1.2 does not mark related, in the PC File's order."""
    try:
        iids = pc_file_iids(pc_file, pc_file.stem)
    except InputError as refused:
        raise SystemExit(str(refused)) from refused
    related = relatedness(set(iids), meta)
    return [iid for iid in iids if not related[iid]]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--pcs", type=Path, help="the PC Files' directory; default: $OGLD_PCS")
    parser.add_argument("--meta", type=Path, help=f"default: $OGLD_REFERENCE/{META_NAME}")
    parser.add_argument("--out", type=Path, help="default: $OGLD_KEEP")
    parser.add_argument("--panel", action="append", choices=PANELS, help="default: all five")
    args = parser.parse_args(argv)
    pcs = args.pcs or root("OGLD_PCS")
    meta = args.meta or root("OGLD_REFERENCE") / META_NAME
    out = args.out or root("OGLD_KEEP")
    for panel in args.panel or PANELS:
        kept = unrelated(pcs / f"{panel}.eigenvec", meta)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{panel}.keep").write_text("#IID\n" + "".join(f"{iid}\n" for iid in kept))
        print(panel, len(kept))


if __name__ == "__main__":
    main()
