#!/usr/bin/env python3
"""Each chromosome's Far Pairs, and the chunks a list matrix's far read touches (ogld#8).

Before ogld#31, `pair_r` read the Proxy Index row of every listed variant once a list spanned
more than near_bp (spec §10). This measured what a separate Far Pairs component, the index's
entries more than near_bp apart in their earlier row, would save, and its output,
`docs/benchmarks/results/far_pairs.json`, is the evidence `docs/benchmarks/latency.md`,
`docs/benchmarks/far-pairs.md` and spec §11 cite for ogld#31's `far/`. It is kept so those
figures can be made again. Per chromosome it counts the Far Pairs from the Proxy Index by the
store's own rule (`_beyond_near`, which `far/` is written by), and for the benchmark's
500-variant draws (`latency.Sample`, seeds 8–10) the distinct chunks of one array each would
read: the index's as the store chunks it, and a Far Pairs array chunked as `ogld.store.layout`
chunks the index. Far Pairs cluster in a few rows (long-range LD), and a row with none touches
no chunk, so the saving in chunks is set by how many listed rows hold any, not by the average a
row.

    pixi run -e dev python scripts/bench/far_pairs.py [--out far_pairs.json] [--chrom 6 11]

It reads every chromosome's `index/<c>/partner` once, a block of rows at a time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import latency
import numpy as np

from ogld.format.geometry import indptr_of, owners, spans
from ogld.store.assembly import _beyond_near
from ogld.store.layout import SIDE

#: Index entries read at a time.
_BLOCK = 1 << 25


def far_lengths(c: Any, near_bp: int) -> np.ndarray:
    """Each row's Far Pairs: its index entries to a later row more than near_bp away."""
    out = np.zeros(len(c.position), dtype=np.int64)
    ip = c.index_indptr
    a = 0
    while a < len(c.position):
        b = int(np.searchsorted(ip, ip[a] + _BLOCK, side="right")) - 1
        b = max(b, a + 1)
        partner = np.asarray(c.index_partner[ip[a] : ip[b]]).astype(np.int64)
        owner = owners(np.diff(ip[a : b + 1]), np.arange(a, b, dtype=np.int64))
        far = (partner > owner) & _beyond_near(c.position, owner, partner, near_bp)
        out += np.bincount(owner[far], minlength=len(out))
        a = b
    return out


def chunks_touched(indptr: np.ndarray, local: np.ndarray, size: int) -> int:
    """The distinct inner chunks of `size` elements that the CSR rows `local` span."""
    first, stop = indptr[local], indptr[local + 1]
    some = stop > first
    lo, hi = first[some] // size, (stop[some] - 1) // size
    return len(np.unique(spans(lo, hi - lo + 1)))


def touched(c: Any, far_indptr: np.ndarray, rows: np.ndarray) -> dict[str, int]:
    """The chunks of one array each that the global `rows` on chromosome `c` read: the index's,
    and a Far Pairs array's."""
    local = rows[(rows >= c.start) & (rows < c.stop)] - c.start
    return {
        "index_chunks": chunks_touched(c.index_indptr, local, int(c.index_partner.chunks[0])),
        "far_pairs_chunks": chunks_touched(far_indptr, local, SIDE[0]),
    }


def chromosome(
    reader: Any, sample: Any, name: str, near_bp: int, genome: list[np.ndarray]
) -> dict[str, Any]:
    """One chromosome's Far Pairs, and the chunks its own 500-row draws and its share of the
    genome-wide draws `genome` read."""
    c = reader._chromosome(name)
    far = far_lengths(c, near_bp)
    far_indptr = indptr_of(far)
    seeds = range(latency.SEED, latency.SEED + 3)
    draws = [
        {"seed": seed, **touched(c, far_indptr, sample.rows(500, name, seed))} for seed in seeds
    ]
    shares = [
        {"seed": seed, **touched(c, far_indptr, rows)}
        for seed, rows in zip(seeds, genome, strict=True)
    ]
    return {
        "chrom": name,
        "rows": len(c.position),
        "index_entries": int(c.index_indptr[-1]),
        "far_pairs": int(far_indptr[-1]),
        "far_pairs_per_row": float(far_indptr[-1] / len(c.position)),
        "rows_with_far_pairs": int((far > 0).sum()),
        "far_pairs_chunks": -(-int(far_indptr[-1]) // SIDE[0]),
        "draws_of_500": draws,
        "genome_draws_of_500": shares,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--chrom", nargs="+", help="the chromosomes to count; all if omitted")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    latency.instrument(6, "ogld", None)
    store = latency.published()
    sample = latency.Sample(store)
    reader = latency.opened(store)._reader
    near_bp = reader.provenance["near_bp"]
    names = args.chrom or list(reader._ranges)
    unknown = sorted(set(names) - set(reader._ranges))
    if unknown:
        raise SystemExit(f"{store} has no chromosome {', '.join(unknown)}")
    # The genome-wide draws of step 1's 500 scattered variants, each chromosome's share counted
    # with that chromosome.
    genome = [sample.rows(500, None, seed) for seed in range(latency.SEED, latency.SEED + 3)]
    found = []
    for name in names:
        found.append(chromosome(reader, sample, name, near_bp, genome))
        print(json.dumps(found[-1]), flush=True)
    counts = reader.provenance["counts"]
    summary = {"store": str(store), "near_bp": near_bp, "counts": counts, "chromosomes": found}
    args.out.write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
