"""Liftover loss classes against a tiny chain and reference (#93)."""

from pathlib import Path

import numpy as np
import pytest

from ogld.build.importing.lift import (
    Lifted,
    SourceVariant,
    drop_displaced,
    far_distance,
    lift_variants,
    variant_chromosomes,
)
from ogld.format import NEAR_BP
from ogld.store import Variants

FIXTURE = Path(__file__).parent / "data" / "panukb"


def lifted(source: list[int], target: list[int]) -> Lifted:
    positions = np.array(target, dtype=np.int64)
    count = len(positions)
    return Lifted(
        Variants(
            np.array(["1"] * count),
            positions,
            np.array([f"1:{position}:A:C" for position in positions]),
            np.array(["A"] * count),
            np.array(["C"] * count),
            np.full(count, 0.25),
        ),
        np.arange(count, dtype=np.int64),
        np.array(source, dtype=np.int64),
        np.ones(count, dtype=np.int64),
        np.zeros(count, dtype=bool),
    )


def test_displaced_segment_is_dropped_to_reach_target() -> None:
    table = lifted(
        [1, 500_000, 1_000_000, 11_000_100, 11_000_200, 12_000_000],
        [1, 500_000, 1_000_000, 2_000_000, 2_000_100, 12_000_000],
    )
    assert far_distance(table.source_position, table.variants.position) < NEAR_BP
    kept, dropped = drop_displaced("1", table, target=8_000_000, cap=10_000)
    assert dropped == 2
    assert kept.index.tolist() == [0, 1, 2, 5]
    assert kept.source_position.tolist() == [1, 500_000, 1_000_000, 12_000_000]
    assert far_distance(kept.source_position, kept.variants.position) >= 8_000_000


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
    assert table.index.tolist() == [0, 5]
    assert table.source_position.tolist() == [1, 102]
    assert table.variants.position.tolist() == [1, 59]
    assert table.variants.a1_frequency.tolist() == [0.75, 0.9]


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


def test_complete_chromosome_loses_nothing() -> None:
    table = lifted([1, 2_000_000, 12_000_000], [1, 2_000_000, 12_000_000])
    assert far_distance(table.source_position, table.variants.position) >= 8_000_000
    kept, dropped = drop_displaced("1", table, target=8_000_000, cap=10_000)
    assert dropped == 0
    assert kept.index.tolist() == table.index.tolist()
    np.testing.assert_array_equal(kept.source_position, table.source_position)
    np.testing.assert_array_equal(kept.variants.position, table.variants.position)


def test_displaced_segment_over_cap_is_refused_with_its_spans() -> None:
    table = lifted(
        [1, 500_000, 1_000_000, 11_000_100, 11_000_200, 12_000_000],
        [1, 500_000, 1_000_000, 2_000_000, 2_000_100, 12_000_000],
    )
    assert far_distance(table.source_position, table.variants.position) < NEAR_BP
    with pytest.raises(ValueError, match="displaced segment") as refusal:
        drop_displaced("1", table, target=8_000_000, cap=1)
    message = str(refusal.value)
    assert "chromosome 1" in message
    assert "GRCh37 11000100-11000200" in message
    assert "GRCh38 2000000-2000100" in message
    assert "target 8000000" in message


def test_a_real_near_band_hole_keeps_the_near_band_refusal() -> None:
    table = lifted(
        [1, 500_000, 1_000_000, 11_000_100, 11_000_200, 12_000_000],
        [1, 500_000, 1_000_000, 2_000_000, 2_000_100, 12_000_000],
    )
    assert far_distance(table.source_position, table.variants.position) <= NEAR_BP
    with pytest.raises(ValueError, match="chromosome 1: liftover leaves holes in the Near Band"):
        drop_displaced("1", table, target=8_000_000, cap=0)


def test_a_short_radius_with_a_complete_near_band_names_the_radius_and_target() -> None:
    table = lifted([1, 500_000, 16_000_000, 16_000_100], [1, 500_000, 6_000_000, 6_000_100])
    reached = far_distance(table.source_position, table.variants.position)
    assert NEAR_BP < reached < 8_000_000
    with pytest.raises(ValueError, match="short of target") as refusal:
        drop_displaced("1", table, target=8_000_000, cap=0)
    message = str(refusal.value)
    assert "Near Band" not in message
    assert f"chromosome 1: liftover reaches a complete radius of {reached}" in message
    assert "target 8000000" in message


def test_equal_segments_tie_drops_the_one_with_the_lowest_grch37_position() -> None:
    table = lifted([1, 500_000, 11_000_100, 11_000_200], [1, 500_000, 2_000_000, 2_000_100])
    assert far_distance(table.source_position, table.variants.position) < NEAR_BP
    first, dropped = drop_displaced("1", table, target=8_000_000, cap=10_000)
    again, _ = drop_displaced("1", table, target=8_000_000, cap=10_000)
    assert dropped == 2
    assert first.index.tolist() == again.index.tolist() == [2, 3]


def test_complete_radius_uses_lifted_not_source_distances() -> None:
    # Target distance 9,000,001 joins a pair outside the published source band.
    assert far_distance(np.array([1, 10_000_002]), np.array([1, 9_000_002])) == 9_000_000
