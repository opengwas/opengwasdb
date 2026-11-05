"""Where Locus Leads and plink2's index variants disagree.

Both methods partition the same rows: the rule assigns each row to a Locus
Lead, plink2 to the index variant whose clump holds it, and each assigns a row
only to a stronger one. A row both make a lead agrees. Otherwise:
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFINITIONS = {
    "spurious_split": "a Locus Lead that plink2 places inside a stronger index variant's clump",
    "lost_signal": "a plink2 index variant that the rule merges into a stronger Locus Lead",
    "spurious_split_rate": "spurious splits / Locus Leads",
    "lost_signal_rate": "lost signals / plink2 index variants",
}


@dataclass(frozen=True)
class Classification:
    rows: int
    locus_leads: int
    index_variants: int
    agreed: int
    spurious_splits: list[tuple[int, int]]  # (Locus Lead, index variant whose clump holds it)
    lost_signals: list[tuple[int, int]]  # (index variant, Locus Lead it merged into)

    @property
    def spurious_split_rate(self) -> float:
        return len(self.spurious_splits) / self.locus_leads if self.locus_leads else float("nan")

    @property
    def lost_signal_rate(self) -> float:
        return len(self.lost_signals) / self.index_variants if self.index_variants else float("nan")

    def counts(self) -> dict[str, int]:
        return {"rows": self.rows, "locus_leads": self.locus_leads,
                "index_variants": self.index_variants, "agreed": self.agreed,
                "spurious_splits": len(self.spurious_splits),
                "lost_signals": len(self.lost_signals)}


def classify(lead_of: np.ndarray, index_of: np.ndarray) -> Classification:
    """Compare the rule's assignment (`lead_of`) with plink2's (`index_of`), row by row."""
    rows = np.arange(lead_of.size)
    lead = lead_of == rows
    index = index_of == rows
    split = np.flatnonzero(lead & ~index)
    lost = np.flatnonzero(index & ~lead)
    return Classification(
        rows=int(rows.size), locus_leads=int(lead.sum()), index_variants=int(index.sum()),
        agreed=int(np.sum(lead & index)),
        spurious_splits=[(int(row), int(index_of[row])) for row in split],
        lost_signals=[(int(row), int(lead_of[row])) for row in lost],
    )
