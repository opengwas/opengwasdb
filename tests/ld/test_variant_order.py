"""The engine's variant order, as `ogld.identity.variant_order_key` sorts it (spec §2)."""

from __future__ import annotations

import pytest

from ogld.identity import orient, variant_order_key


def test_variants_sort_by_chromosome_then_position_then_alid() -> None:
    records = [
        ("X", 5, "A", "C"),
        ("22", 9, "A", "G"),
        ("2", 90, "A", "G"),
        ("22", 9, "A", "C"),
        ("MT", 1, "A", "G"),
        ("2", 100, "C", "G"),
        ("Y", 5, "A", "C"),
    ]
    ordered = sorted(records, key=lambda r: variant_order_key(orient(*r)))
    assert [(c, p, alt) for c, p, _, alt in ordered] == [
        ("2", 90, "G"),
        ("2", 100, "G"),
        ("22", 9, "C"),  # 22:9:A:C before 22:9:A:G: ALID ascending at one position
        ("22", 9, "G"),
        ("X", 5, "C"),
        ("Y", 5, "C"),
        ("MT", 1, "G"),
    ]


def test_a_chromosome_the_engine_does_not_rank_has_no_key() -> None:
    with pytest.raises(KeyError):
        variant_order_key(orient("HLA-A", 5, "A", "C"))
