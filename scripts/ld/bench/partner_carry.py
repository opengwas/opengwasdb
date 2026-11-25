#!/usr/bin/env python3
"""The Partner Rows' peak carry, measured on the Test Store (partner-rows §6.1; opengwas/ogld#67).

The carry holds the pairs whose later row has not been written yet: each pair's earlier row, its
later row and its code, 10 bytes (`assembly.PAIR_BYTES`), with its destination implicit in the
bucket it sits in. What bounds it is the pairs spanning one row within `far_bp` plus one bucket of
`CARRY_ROWS` rows, not the chromosome's length, so it is the number to watch on a whole genome.

    pixi run -e dev python scripts/ld/bench/partner_carry.py [--out partner_carry.json]

It writes the committed Test Store again at `--t-partner 0.01`, its rows in blocks of
`--block-rows` (16 by default: a frame's worth), into a temporary directory, wrapping
`PartnerRowsStream.close` to read each chromosome's `peak_carry` before it closes, and prints, per
chromosome, the peak's pairs and bytes. Written whole, as the committed store is, the carry is 0:
every pair's later row is in the block being written, so nothing is ever held over. Nothing is
committed.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from opengwasdb.ld.store.assembly import PAIR_BYTES, PartnerRowsStream

#: `scripts/`, where the Test Store's own writer lives.
SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
make_test_store = importlib.import_module("make_test_store")

#: The Test Store's Partner Threshold and scale, as the committed store records them.
T_PARTNER, SCALE = 0.01, 511


def written(out: Path, block_rows: int) -> None:
    """The Test Store at `out`, each chromosome's rows cut in blocks of `block_rows`."""
    from ogquery.ld.format.geometry import stop_within

    from opengwasdb.ld.format.geometry import chromosome_ranges
    from opengwasdb.ld.store import StoreWriter

    vs = make_test_store.variant_set()
    ranges = chromosome_ranges(vs.variants.chrom)
    writer = StoreWriter.create(
        out,
        variants=vs.variants,
        encoding=make_test_store.REncoding(SCALE),
        near_bp=make_test_store.NEAR_BP,
        far_bp=make_test_store.FAR_BP,
        n=make_test_store.panel_data().panel().n_samples,
        k=10,
        t_partner=T_PARTNER,
    )
    started = time.perf_counter()
    for chrom, r in make_test_store.r_matrices(vs, 10, ranges).items():
        first, stop = ranges[chrom]
        position = vs.variants.position[first:stop]
        for at in range(first, stop, block_rows):
            last = min(at + block_rows, stop)
            far_stop = int(stop_within(position, make_test_store.FAR_BP)[last - first - 1])
            writer.write(writer.cut(at, r[at - first : last - first, at - first : far_stop]))
    writer.finish(
        make_test_store.sourced(
            10, vs, make_test_store.pc_diagnostic(vs, 10), make_test_store.build_record(started)
        )
    )


def measured(out: Path, block_rows: int) -> dict[str, Any]:
    """Write the Test Store at `out` in blocks and return each chromosome's peak carry."""
    found: list[dict[str, Any]] = []
    real = PartnerRowsStream.close

    def sampled(self: PartnerRowsStream) -> int:
        pairs, held = self.peak_carry
        found.append(
            {
                "chrom": self.chrom,
                "peak_pairs": pairs,
                "peak_bytes": held,
                "pair_bytes": PAIR_BYTES,
            }
        )
        return real(self)

    PartnerRowsStream.close = sampled
    try:
        written(out, block_rows)
    finally:
        PartnerRowsStream.close = real
    return {
        "t_partner": T_PARTNER,
        "scale": SCALE,
        "block_rows": block_rows,
        "chromosomes": found,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, help="write the JSON here as well as printing it")
    parser.add_argument(
        "--block-rows", type=int, default=16, help="rows a block of the write holds"
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as scratch:
        found = measured(Path(scratch) / "store", args.block_rows)
    found["peak_pairs"] = max(entry["peak_pairs"] for entry in found["chromosomes"])
    found["peak_bytes"] = max(entry["peak_bytes"] for entry in found["chromosomes"])
    text = json.dumps(found, indent=1)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
