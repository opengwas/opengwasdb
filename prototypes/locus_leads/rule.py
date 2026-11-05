"""The Locus Lead rule (#62's proposal, "The rule").

Take the rows in descending abs_z, ties broken by ascending Store Variant Index,
and accept each as a Locus Lead unless an already-accepted lead on the same
chromosome lies within W bp *and* D cM of it. D splits signals across a
recombination hotspot that a bp window would merge; W stops cM that barely
moves over megabases (centromeres, inversions) from merging both arms.

Ranking is on abs_z and never on p: a float64 two-sided p loses precision from
|z| 37.54, where it turns subnormal, and is 0 from 38.50, which would tie the
strongest rows of a big locus.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import numpy as np


def rule_order(abs_z: np.ndarray, variant_index: np.ndarray) -> np.ndarray:
    """Row order the rule visits: abs_z descending, then Store Variant Index ascending."""
    order: np.ndarray = np.lexsort((np.asarray(variant_index), -np.asarray(abs_z)))
    return order


def locus_leads(
    chromosome: np.ndarray,
    position: np.ndarray,
    cm: np.ndarray,
    abs_z: np.ndarray,
    variant_index: np.ndarray,
    *,
    d_cm: float,
    w_bp: int,
) -> np.ndarray:
    """For each row, the row of its Locus Lead: itself when it is one.

    A row within reach of several accepted leads is assigned to the strongest
    of them, the one accepted first.
    """
    cm = np.asarray(cm, dtype="float64")
    if not np.all(np.isfinite(cm)):
        raise ValueError("every row needs a finite cM; rows without one are not ranked")
    lead_of = np.full(cm.shape, -1, dtype="int64")
    accepted: dict[str, _Leads] = {}
    for row in rule_order(abs_z, variant_index):
        leads = accepted.setdefault(str(chromosome[row]), _Leads())
        lead = leads.within(int(position[row]), float(cm[row]), d_cm=d_cm, w_bp=w_bp)
        if lead is None:
            leads.add(int(row), int(position[row]), float(cm[row]))
            lead = int(row)
        lead_of[row] = lead
    return lead_of


@dataclass
class _Leads:
    """One chromosome's accepted leads, kept in position order for the W search."""

    position: list[int] = field(default_factory=list)
    entries: list[tuple[int, int, float]] = field(default_factory=list)  # (rank, row, cM)

    def add(self, row: int, position: int, cm: float) -> None:
        at = bisect.bisect_left(self.position, position)
        self.position.insert(at, position)
        self.entries.insert(at, (len(self.entries), row, cm))

    def within(self, position: int, cm: float, *, d_cm: float, w_bp: int) -> int | None:
        lo = bisect.bisect_left(self.position, position - w_bp)
        hi = bisect.bisect_right(self.position, position + w_bp)
        near = [(rank, row) for rank, row, lead_cm in self.entries[lo:hi]
                if abs(lead_cm - cm) <= d_cm]
        return min(near)[1] if near else None
