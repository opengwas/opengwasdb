"""The engine's variant order, as `ogld.identity.order_key` and `variant_order_key` sort it
(spec §2)."""

from __future__ import annotations

import pytest

from ogld.identity import order_key, orient, variant_order_key

RECORDS = [
    ("X", 5, "A", "C"),
    ("22", 9, "A", "G"),
    ("2", 90, "A", "G"),
    ("22", 9, "A", "C"),
    ("MT", 1, "A", "G"),
    ("2", 100, "C", "G"),
    ("Y", 5, "A", "C"),
]
#: The engine's order of RECORDS, as (chromosome, position, ALT).
ENGINE_ORDER = [
    ("2", 90, "G"),
    ("2", 100, "G"),
    ("22", 9, "C"),  # 22:9:A:C before 22:9:A:G: ALID ascending at one position
    ("22", 9, "G"),
    ("X", 5, "C"),
    ("Y", 5, "C"),
    ("MT", 1, "G"),
]


def test_variants_sort_by_chromosome_then_position_then_alid() -> None:
    ordered = sorted(RECORDS, key=lambda r: variant_order_key(orient(*r)))
    assert [(c, p, alt) for c, p, _, alt in ordered] == ENGINE_ORDER


def test_the_order_of_chromosome_position_and_alid_alone_is_the_engine_s() -> None:
    # What a clumping's result holds, with no alleles to re-form an identity from (ogld#14).
    fields = {(o.chrom, o.position, o.alid): r for r in RECORDS for o in [orient(*r)]}
    ordered = [fields[f] for f in sorted(fields, key=lambda f: order_key(*f))]
    assert [(c, p, alt) for c, p, _, alt in ordered] == ENGINE_ORDER


def test_order_key_ranks_10_after_9_and_2_where_text_would_not() -> None:
    assert order_key("10", 5, "10:5:A:C") > order_key("9", 50, "9:50:A:C")
    assert order_key("10", 5, "10:5:A:C") > order_key("2", 50, "2:50:A:C")


def test_a_chromosome_the_engine_does_not_rank_has_no_key() -> None:
    with pytest.raises(KeyError):
        variant_order_key(orient("HLA-A", 5, "A", "C"))
    with pytest.raises(KeyError):
        order_key("HLA-A", 5, "HLA-A:5:A:C")


def test_order_key_reads_a_chr_prefix_as_the_engine_does() -> None:
    assert order_key("chr1", 5, "1:5:A:C") == order_key("1", 5, "1:5:A:C")
    assert order_key("chrX", 5, "X:5:A:C") > order_key("chr22", 9, "22:9:A:C")
