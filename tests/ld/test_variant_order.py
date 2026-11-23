"""The engine's variant order, as an LD build sorts its Variant Set by it
(`opengwasdb.variants.axis.alid_sort_key` over sorted-spelling ALIDs), and a chromosome's place in
it, `opengwasdb.variants.normalise.chromosome_rank` (ld-store §2)."""

from __future__ import annotations

import numpy as np
import pytest

from opengwasdb.ld.query.results import Clumps
from opengwasdb.store_reader.identity import AUTOSOMES, VariantNormalisationError
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


def test_a_clumping_s_leads_take_the_order_of_their_fields_on_a_tie() -> None:
    # What a clumping's result holds, with no alleles to re-form an identity from (ogld#14).
    found = [sorted_identity(*r) for r in RECORDS]
    clumps = Clumps(
        alid=np.array([o.alid for o in found]),
        chrom=np.array([o.chrom for o in found]),
        position=np.array([o.position for o in found], dtype=np.int64),
        score=np.ones(len(found)),
        lead=np.array([o.alid for o in found]),
        r=np.ones(len(found)),
        absent=(),
        r2=0.1,
        window_bp=1e6,
    )
    by_alid = {o.alid: r for o, r in zip(found, RECORDS, strict=True)}
    assert [(by_alid[a][0], by_alid[a][1], by_alid[a][3]) for a in clumps.leads()] == ENGINE_ORDER


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


def test_a_chromosome_the_engine_does_not_rank_is_refused() -> None:
    for label in ("HLA-A", "chr 22", "-1"):
        with pytest.raises(VariantNormalisationError, match="no place in the variant order"):
            chromosome_rank(label)
    with pytest.raises(VariantNormalisationError, match="unsupported chromosome"):
        chromosome_rank("27")


def test_the_rank_reads_a_chr_prefix_as_the_engine_does() -> None:
    assert chromosome_rank("chr1") == chromosome_rank("1")
    assert chromosome_rank("chrX") > chromosome_rank("chr22")


def test_a_chromosome_s_rank_is_the_first_field_of_its_order() -> None:
    # The Variant Set's chromosomes and validation's ranges are ordered by this alone, with no
    # position or ALID made up to order them by (review of epic #1).
    names = ["X", "22", "2", "MT", "10", "Y", "9", "1"]
    assert sorted(names, key=chromosome_rank) == ["1", "2", "9", "10", "22", "X", "Y", "MT"]
    for name in names:
        assert chromosome_rank(name) == alid_sort_key(f"{name}:5:A:C")[0][0]
    assert chromosome_rank("chrX") == chromosome_rank("X")


def test_the_autosomes_are_1_to_22_in_the_engine_s_order() -> None:
    assert tuple(str(n) for n in range(1, 23)) == AUTOSOMES
    assert sorted(AUTOSOMES, key=chromosome_rank) == list(AUTOSOMES)
