"""One Locus Lead per LD block (#409).

The lead of a block is its strongest row in the Locus Lead ranking (`abs_z`
descending, then Store Variant Index ascending). Blocks are half-open,
`[start, end)`, so a row at an endpoint two blocks share belongs to the block
that starts there. A row outside every block is never dropped: each stretch
between blocks (before a chromosome's first, between two, after its last, or a
whole chromosome without blocks) holds one lead of its own, and the rows and
stretches are counted. `DEFINITION` says so in the output.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .rule import rule_order

DEFINITION = {
    "blocks": "the rows of the LD-region file for the ancestry, each half-open [start, end), "
              "so a row at a shared endpoint belongs to the block that starts there",
    "lead": "the block's strongest row: abs_z descending, then Store Variant Index ascending",
    "outside": "a row outside every block belongs to the stretch between blocks it falls in "
               "(before a chromosome's first block, between two, after its last, or the whole "
               "of a chromosome without blocks); each stretch holds one lead, and those rows "
               "and stretches are counted",
}


@dataclass(frozen=True)
class Blocks:
    """One ancestry's blocks, per chromosome: starts and ends in position order."""

    ancestry: str
    starts: dict[str, np.ndarray]
    ends: dict[str, np.ndarray]

    def count(self) -> int:
        return sum(s.size for s in self.starts.values())

    def unit_of(self, chromosome: str, position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Each row's block or stretch: the index of the last block starting at or before it,
        and whether it lies inside that block."""
        starts = self.starts.get(chromosome, np.empty(0, dtype="int64"))
        ends = self.ends.get(chromosome, np.empty(0, dtype="int64"))
        at = np.searchsorted(starts, position, side="right") - 1
        inside = (at >= 0) & (position < ends[np.clip(at, 0, None)] if ends.size else False)
        return at, np.asarray(inside, dtype=bool)


@dataclass(frozen=True)
class BlockLeads:
    lead_of: np.ndarray
    outside_rows: int
    outside_stretches: int


def read_blocks(path: Path, ancestry: str) -> Blocks:
    """The blocks of `ancestry` in a `chr start end ancestry` file; overlaps are refused."""
    with path.open() as handle:
        rows = [r for r in csv.DictReader(handle, delimiter="\t") if r["ancestry"] == ancestry]
    if not rows:
        raise ValueError(f"{path}: no {ancestry} block")
    starts: dict[str, list[int]] = {}
    ends: dict[str, list[int]] = {}
    for row in rows:
        starts.setdefault(row["chr"], []).append(int(row["start"]))
        ends.setdefault(row["chr"], []).append(int(row["end"]))
    out_starts, out_ends = {}, {}
    for chromosome, begin in starts.items():
        order = np.argsort(begin, kind="stable")
        s, e = np.array(begin)[order], np.array(ends[chromosome])[order]
        if np.any(e <= s) or np.any(s[1:] < e[:-1]):
            raise ValueError(f"{path}: {ancestry} blocks on chromosome {chromosome} overlap or "
                             "are empty")
        out_starts[chromosome], out_ends[chromosome] = s, e
    return Blocks(ancestry, out_starts, out_ends)


def block_leads(chromosome: np.ndarray, position: np.ndarray, abs_z: np.ndarray,
                variant_index: np.ndarray, blocks: Blocks) -> BlockLeads:
    """Each row's lead: the strongest row of its block, or of its stretch between blocks."""
    position = np.asarray(position, dtype="int64")
    unit = np.empty(position.size, dtype=object)
    outside = np.zeros(position.size, dtype=bool)
    for label in np.unique(chromosome):
        on = chromosome == label
        at, inside = blocks.unit_of(str(label), position[on])
        unit[on] = [f"{label}:{'block' if i else 'after'}:{a}"
                    for a, i in zip(at.tolist(), inside.tolist(), strict=True)]
        outside[on] = ~inside
    lead_of = np.full(position.size, -1, dtype="int64")
    lead: dict[str, int] = {}
    for row in rule_order(abs_z, variant_index).tolist():
        lead_of[row] = lead.setdefault(unit[row], row)
    return BlockLeads(lead_of, int(outside.sum()), len({u for u in unit[outside]}))
