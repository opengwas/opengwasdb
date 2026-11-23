"""ogld#3's contract: the engine's ALID and A1, so r is signed as the engine signs a statistic.

An LD Store spells ALIDs sorted until #462 (`opengwasdb.variants.sorted_spelling`): ALID is
`chr:pos:A1:A2`, A1 the alphabetically first allele after upper-casing, an indel's too. These are
literals of that rule. Nothing is trimmed or left-aligned (ADR 0089). The conformance test against
the engine's committed answers is `tests/ld/test_sorted_spelling_conformance.py`.
"""

from __future__ import annotations

import pytest
from support import panel_data

from opengwasdb.store_reader.identity import normalise_chromosome
from opengwasdb.variants.sorted_spelling import sorted_alid_locus, sorted_identity


@pytest.mark.parametrize(
    ("ref", "alt", "alid", "a1_is_alt"),
    [
        ("G", "A", "22:16050075:A:G", True),
        ("A", "G", "22:16050075:A:G", False),
        ("T", "C", "22:16050075:C:T", True),
    ],
)
def test_a1_is_the_alphabetically_first_allele(
    ref: str, alt: str, alid: str, a1_is_alt: bool
) -> None:
    o = sorted_identity("22", 16050075, ref, alt)
    assert o.alid == alid
    assert (o.a1, o.a2) == tuple(alid.split(":")[2:])
    assert o.a1_is_alt is a1_is_alt


def test_an_indel_is_spelt_sorted_whichever_allele_is_the_reference() -> None:
    # The engine spells this deletion `1:827587:GGGGCC:G` (ADR 0079); an LD Store, until #462,
    # spells it as it spells the insertion, so its sign is set by ALT, not by which is longer.
    deletion = sorted_identity("1", 827587, "GGGGCC", "G")
    insertion = sorted_identity("1", 827587, "G", "GGGGCC")
    assert deletion.alid == insertion.alid == "1:827587:G:GGGGCC"
    assert (deletion.a1_is_alt, insertion.a1_is_alt) == (True, False)


def test_a_lower_case_base_is_the_same_base() -> None:
    # opengwas/opengwasdb#360.
    lower = sorted_identity("22", 100, "g", "a")
    assert lower.alid == "22:100:A:G"
    assert lower == sorted_identity("22", 100, "G", "A")


def test_an_alid_is_never_truncated() -> None:
    # The engine's |S32 index drops ALIDs over 32 bytes; an LD Store stores them whole (ld-store
    # §2).
    long = "A" + "CGT" * 15
    o = sorted_identity("21", 5_000_000, long, "A")
    assert len(o.alid) > 32
    assert o.alid == f"21:5000000:A:{long}"


def test_identical_alleles_are_refused() -> None:
    with pytest.raises(ValueError, match="allele"):
        sorted_identity("22", 100, "A", "a")


@pytest.mark.parametrize(
    ("label", "canonical"),
    [
        ("22", "22"),
        ("chr22", "22"),
        ("X", "X"),
        ("chrX", "X"),
        ("23", "X"),
        ("M", "MT"),
        ("chrM", "MT"),
    ],
)
def test_chromosome_labels_are_the_engine_s(label: str, canonical: str) -> None:
    assert normalise_chromosome(label) == canonical


def test_an_unsupported_chromosome_is_refused() -> None:
    with pytest.raises(ValueError, match="chr"):
        normalise_chromosome("27")


@pytest.mark.parametrize(
    ("alid", "locus"),
    [
        ("22:16050075:A:G", ("22", 16050075)),
        ("X:100:A:ACGT", ("X", 100)),
        ("rs123", None),
        ("22:abc:A:G", None),
        ("", None),
    ],
)
def test_an_alid_names_its_locus(alid: str, locus: tuple[str, int] | None) -> None:
    assert sorted_alid_locus(alid) == locus


def test_every_test_panel_record_is_oriented_consistently() -> None:
    sorted_identity("22", 1, "A", "G")
    v = panel_data().variants()
    assert len(v.pos) > 1000, "the Test Panel must be loaded for this to mean anything"
    flipped = 0
    for chrom, pos, ref, alt in zip(v.chrom, v.pos, v.ref, v.alt, strict=True):
        o = sorted_identity(str(chrom), int(pos), str(ref), str(alt))
        assert o.alid == f"{o.chrom}:{o.position}:{o.a1}:{o.a2}"
        assert o.a1 < o.a2
        if len(ref) == len(alt) == 1:
            assert o.a1_is_alt == (o.a1 == str(alt).upper())
        flipped += not o.a1_is_alt
    # About half of a real panel's records have A1 = REF, so both signs are exercised.
    assert 0 < flipped < len(v.pos)
