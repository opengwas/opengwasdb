"""The engine's variant order, as an LD build sorts its Variant Set by it
(`opengwasdb.variants.axis.alid_sort_key` over sorted-spelling ALIDs), and a chromosome's place in
it, `opengwasdb.variants.normalise.chromosome_rank` (ld-store §2)."""

from __future__ import annotations

from opengwasdb.variants.axis import alid_sort_key
from opengwasdb.variants.normalise import chromosome_rank
from opengwasdb.variants.sorted_spelling import sorted_identity

RECORDS = [
    ("X", 5, "A", "C"),
    ("22", 9, "A", "G"),
    ("2", 90, "A", "G"),
    ("22", 9, "A", "C"),
    ("MT", 1, "A", "G"),
    ("2", 100, "C", "G"),
    ("Y", 5, "A", "C"),
    ("22", 9, "A", "AT"),
]
#: The engine's order of RECORDS, as (chromosome, position, ALT).
ENGINE_ORDER = [
    ("2", 90, "G"),
    ("2", 100, "G"),
    ("22", 9, "AT"),  # 22:9:A:AT before 22:9:A:C: `AT` sorts before `C`
    ("22", 9, "C"),  # 22:9:A:C before 22:9:A:G: ALID ascending at one position
    ("22", 9, "G"),
    ("X", 5, "C"),
    ("Y", 5, "C"),
    ("MT", 1, "G"),
]


def test_variants_sort_by_chromosome_then_position_then_alid() -> None:
    ordered = sorted(RECORDS, key=lambda r: alid_sort_key(sorted_identity(*r).alid))
    assert [(c, p, alt) for c, p, _, alt in ordered] == ENGINE_ORDER


def test_the_order_ranks_10_after_9_and_2_where_text_would_not() -> None:
    assert alid_sort_key("10:5:A:C") > alid_sort_key("9:50:A:C")
    assert alid_sort_key("10:5:A:C") > alid_sort_key("2:50:A:C")


def test_a_sorted_alid_s_bytewise_order_at_a_position_is_the_engine_s() -> None:
    # A clumping orders a tie by its ALID string; the engine by its alleles. They agree because
    # ':' sorts below every base.
    alids = [
        sorted_identity("22", 9, ref, alt).alid
        for ref, alt in [
            ("A", "AT"),
            ("A", "C"),
            ("AT", "C"),
            ("A", "ACGT"),
            ("AA", "C"),
            ("CAG", "C"),
        ]
    ]
    assert sorted(alids) == sorted(alids, key=alid_sort_key)


def test_a_chromosome_s_rank_is_the_first_field_of_its_order() -> None:
    # The Variant Set's chromosomes and validation's ranges are ordered by this alone, with no
    # position or ALID made up to order them by (review of epic #1).
    names = ["X", "22", "2", "MT", "10", "Y", "9", "1"]
    assert sorted(names, key=chromosome_rank) == ["1", "2", "9", "10", "22", "X", "Y", "MT"]
    for name in names:
        assert chromosome_rank(name) == alid_sort_key(f"{name}:5:A:C")[0][0]
    assert chromosome_rank("chrX") == chromosome_rank("X")
