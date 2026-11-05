"""Matching Store rows to genotype-panel rows (prototypes/locus_leads, #402).

A Store ALID spells its alleles sorted, a pvar row in REF/ALT order, and the
pvar ID column truncates long alleles; so rows match on chromosome, position
and allele set, read from the REF and ALT columns.
"""

from __future__ import annotations

import numpy as np
from prototype_modules import load_prototype

panel = load_prototype("locus_leads", "panel")

LONG = "T" + "TG" * 60  # 121 bases; the panel's IDs truncate past 100

# One chromosome of a panel. 300 holds an insertion and a deletion with the
# same allele set; 600 and 700 hold different variants whose IDs collide
# after truncation.
CHR22 = panel.PanelVariants(
    position=np.array([100, 200, 300, 300, 400, 600, 700]),
    variant_id=np.array(["22:100:C:T", "22:200:G:A", "22:300:C:CAATA", "22:300:CAATA:C",
                         "22:400:A:G", "22:x:T:TTG", "22:x:T:TTG"]),
    ref=np.array(["C", "G", "C", "CAATA", "A", "T", "G"]),
    alt=np.array(["T", "A", "CAATA", "C", "G", LONG, "C"]),
)


def test_store_rows_match_on_allele_set_and_every_unmatched_row_is_counted() -> None:
    rows = {
        "same orientation": ("22", 100, "C", "T"),
        "swapped": ("22", 200, "A", "G"),
        "different alleles": ("22", 400, "A", "C"),
        "absent": ("22", 500, "A", "G"),
        "insertion or deletion": ("22", 300, "C", "CAATA"),
        "truncated id": ("22", 600, LONG, "T"),
        "off panel": ("X", 100, "C", "T"),
    }
    chromosome, position, allele_1, allele_2 = (np.array(c) for c in zip(*rows.values(),
                                                                         strict=True))
    assert len(rows) == 7 and set(chromosome) == {"22", "X"}

    match = panel.match_to_panel(chromosome, position, allele_1, allele_2, {"22": CHR22})

    assert list(match.panel_id) == ["22:100:C:T", "22:200:G:A", "", "", "", "", ""]
    assert list(match.status) == [
        "matched", "matched", "allele_mismatch", "absent_at_position",
        "ambiguous", "ambiguous", "off_panel_chromosome",
    ]
    assert match.counts() == {
        "matched": 2, "allele_mismatch": 1, "absent_at_position": 1,
        "ambiguous": 2, "off_panel_chromosome": 1,
    }
