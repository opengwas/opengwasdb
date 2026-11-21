"""Liftover loss classes against a tiny chain and reference (#93)."""

from pathlib import Path

import numpy as np

from ogld.build.importing.lift import SourceVariant, far_distance, lift_variants

FIXTURE = Path(__file__).parent / "data" / "panukb"


def test_lift_allele_mismatch_unmapped_and_collision_are_counted() -> None:
    rows = [
        SourceVariant(0, "22", 1, "A", "C", 0.25),
        SourceVariant(1, "22", 2, "A", "G", 0.1),
        SourceVariant(2, "22", 2, "A", "G", 0.2),
        SourceVariant(3, "22", 80, "A", "G", 0.1),  # chain gap
        SourceVariant(4, "22", 101, "A", "C", 0.1),  # reverse REF = T, not A
        SourceVariant(5, "22", 102, "T", "G", 0.1),  # reverse REF = A
    ]
    table, losses = lift_variants(rows, FIXTURE / "tiny.chain", FIXTURE / "tiny.fa")
    assert len(table) == 2
    assert losses == {
        "liftover_identity_collision": 2,
        "liftover_missing_or_ambiguous": 1,
        "allele_mismatch": 1,
    }
    assert table.alid.tolist() == ["22:1:A:C", "22:59:A:C"]
    assert table.sign.tolist() == [-1, -1]
    assert table.negative_strand.tolist() == [False, True]


def test_complete_radius_uses_lifted_not_source_distances() -> None:
    # Target distance 9,000,001 joins a pair outside the published source band.
    assert far_distance(np.array([1, 10_000_002]), np.array([1, 9_000_002])) == 9_000_000
