"""Liftover loss classes against a tiny chain and reference (#93)."""

from pathlib import Path

import numpy as np
import pytest

from ogld.build.importing.lift import (
    SourceVariant,
    far_distance,
    lift_variants,
    variant_chromosomes,
)

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


def test_tsv_liftover_input_streams_one_chromosome_before_later_bad_row(tmp_path) -> None:
    tsv = tmp_path / "variants.tsv"
    tsv.write_text(
        'locus\talleles\tAF\tidx\n1:1\t["A","C"]\t0.1\t0\n'
        '2:1\t["A","C"]\t0.2\t1\nX:1\t["A","C"]\t0.1\t2\n'
    )
    chromosomes = variant_chromosomes(tsv)
    name, rows = next(chromosomes)
    assert name == "1"
    assert rows == [SourceVariant(0, "1", 1, "A", "C", 0.1)]
    with pytest.raises(ValueError, match="unsupported chromosome X"):
        next(chromosomes)


def test_complete_radius_uses_lifted_not_source_distances() -> None:
    # Target distance 9,000,001 joins a pair outside the published source band.
    assert far_distance(np.array([1, 10_000_002]), np.array([1, 9_000_002])) == 9_000_000
