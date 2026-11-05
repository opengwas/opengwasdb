"""Spurious splits and lost signals (prototypes/locus_leads, #402).

A spurious split is a Locus Lead that plink2 places inside a stronger index
variant's clump; a lost signal is a plink2 index variant that the rule merges
into a stronger Locus Lead.
"""

from __future__ import annotations

import numpy as np
import pytest
from prototype_modules import load_prototype

compare = load_prototype("locus_leads", "compare")


def test_spurious_splits_and_lost_signals_on_a_hand_worked_example() -> None:
    # Nine rows, numbered in the rule's order, so a lower row is stronger.
    # The rule's leads are 0, 2, 4, 5 and 7; plink2's index variants 0, 3, 4 and 6.
    # Row 8 is neither, and the two methods assign it to different rows.
    lead_of = np.array([0, 0, 2, 2, 4, 5, 5, 7, 7])
    index_of = np.array([0, 0, 0, 3, 4, 4, 6, 6, 6])
    assert np.all(lead_of <= np.arange(9)) and np.all(index_of <= np.arange(9))

    result = compare.classify(lead_of, index_of)

    # Leads 2, 5 and 7 sit in the clumps of 0, 4 and 6: three spurious splits.
    # Index variants 3 and 6 merge into leads 2 and 5: two lost signals.
    # Rows 0 and 4 are both a lead and an index variant; rows 1 and 8 neither.
    assert result.spurious_splits == [(2, 0), (5, 4), (7, 6)]
    assert result.lost_signals == [(3, 2), (6, 5)]
    assert result.counts() == {
        "rows": 9, "locus_leads": 5, "index_variants": 4, "agreed": 2,
        "spurious_splits": 3, "lost_signals": 2,
    }
    assert result.spurious_split_rate == pytest.approx(3 / 5)
    assert result.lost_signal_rate == pytest.approx(2 / 4)
