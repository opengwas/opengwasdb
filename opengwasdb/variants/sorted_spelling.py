"""The sorted spelling of an ALID: both alleles sorted, an indel's too.

The engine spells an indel reference first (`normalise.alid_alleles`, ADR 0079) and keeps the
sorted spelling for reading a GWAS Store of format 0.2.0 or 0.2.1
(`normalise.sorted_alid_alleles`). Every LD Store is spelt sorted: its format did not move with
ADR 0079, and its move is #462 (owner, 23 November 2026, on #456). So this package owns both
spellings, and this module is the sorted one's record identity, as an LD build forms it and an
LD Store reader parses it. It forms the ALID with `sorted_alid_alleles` and judges a record by
`orient_to_canonical`, so it has no rule of its own beyond the spelling.

Sorted and reference-first spell one ALID for every pair except an indel whose reference allele
is longer and sorts after its alternate: a deletion, `GGGGCC>G`, is `1:827587:GGGGCC:G`
reference first and `1:827587:G:GGGGCC` sorted, the spelling of the insertion `G>GGGGCC` too.
A sorted Store holds the deletion under the sorted ALID, unless the insertion is also at that
site, when it keeps the insertion there (`resolve_sorted_reversed_indels`), and r there is
signed against the allele the reference-first spelling puts second. So a lookup that receives
the reference-first spelling refuses (`ReferenceFirstIndel`), naming the sorted ALID, rather than
report it Absent (owner, on #456). The reverse case cannot be told apart: a reference-first
insertion ALID is also a sorted ALID, and at a site that holds only the deletion it finds the
deletion's row (#462).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from opengwasdb.variants.normalise import (
    CANONICAL_CHROMOSOMES,
    VALID_BASES,
    VariantNormalisationError,
    normalise_allele,
    orient_to_canonical,
    sorted_alid_alleles,
)

#: A position an LD Store's int32 column holds (ld-store spec §2), spelt with at most this many
#: digits.
_MAX_POSITION: Final = 2**31 - 1
_MAX_POSITION_DIGITS: Final = 10


class DuplicateRecord(VariantNormalisationError):
    """Records that share a sorted ALID and are not the two halves of one reversed indel:
    identical REF and ALT, a third record, or two SNVs (`resolve_sorted_reversed_indels`)."""

    reason = "duplicate_record"


class ReferenceFirstIndel(VariantNormalisationError):
    """An ALID that spells an indel reference first where ALIDs are spelt sorted.

    Its alleles differ in length and its A1 sorts after its A2: a deletion whose REF sorts after
    its ALT, which no sorted spelling forms. A sorted Store would hold that variant as
    `chr:pos:A2:A1`. That row is this deletion unless the panel also has the insertion at that
    site, and either way r there is signed against the other allele, A2 here (#462).
    """

    reason = "indel_spelt_reference_first"


@dataclass(frozen=True)
class SortedIdentity:
    """A biallelic record's identity in the sorted spelling (ld-store spec §2, §3).

    `position` is the one the ALID carries, and the Variant Set is ordered by it. `a1_is_alt`
    says which allele genotype dosages count: r computed from ALT dosages is r for A1 when it is
    True, and must be negated when it is False.
    """

    chrom: str
    position: int
    alid: str
    a1: str
    a2: str
    a1_is_alt: bool


def sorted_identity(chrom: str, position: int, ref: str, alt: str) -> SortedIdentity:
    """The sorted-spelling identity of the record `ref`/`alt` at `chrom`:`position`.

    Judged by `orient_to_canonical`, with `ref` declared the reference: the chromosome, then
    ALT, then REF, then the position, then identical alleles, each refused with that
    function's `VariantNormalisationError`. A1 is the allele `sorted_alid_alleles` puts first.
    Nothing is trimmed, left-aligned or truncated.
    """
    variant = orient_to_canonical(chrom, position, alt, ref, other_is_reference=True).variant
    a1, a2 = sorted_alid_alleles(variant.effect_allele, variant.other_allele)
    alt_allele = normalise_allele(alt)
    return SortedIdentity(
        chrom=variant.chromosome,
        position=variant.position,
        alid=f"{variant.chromosome}:{variant.position}:{a1}:{a2}",
        a1=a1,
        a2=a2,
        a1_is_alt=alt_allele == a1,
    )


def sorted_alid_locus(alid: str) -> tuple[str, int] | None:
    """The chromosome and position a sorted-spelling ALID names, or None when it is not one.

    The strict parser, for an LD Store's lookup: an ALID either is one exactly as such a Store
    spells it or names nothing there. `opengwasdb.variants.axis.parse_canonical_alid` is the
    lenient one, for the engine's GWAS Store and variant-list reads: it normalises a chromosome
    alias and an allele's case, takes the alleles in either order, and has no int32 or digit
    limit. Both exist because an LD lookup must find nothing under a spelling its Store never
    wrote, while the engine resolves an identifier as it is given.

    A reader finds an ALID by this locus, then compares whole ALID strings there, so a malformed
    one is Absent rather than an error, and this never raises. Such an ALID is
    `chrom:position:A1:A2`: a chromosome `1`–`22`, `X`, `Y` or `MT`, a position of at most 10
    ASCII digits, positive, unpadded and within int32, and two distinct alleles over A, C, G, T
    with A1 sorted before A2. A reference-first indel ALID is not one
    (`spells_indel_reference_first`).
    """
    parts = alid.split(":")
    if len(parts) != 4:
        return None
    chrom, pos, a1, a2 = parts
    if not (chrom in CANONICAL_CHROMOSOMES and _is_canonical_position(pos)):
        return None
    if not (_is_allele(a1) and _is_allele(a2) and a1 < a2):
        return None
    return chrom, int(pos)


def spells_indel_reference_first(alid: str) -> bool:
    """Whether `alid` is a well-formed ALID that only the reference-first spelling forms: an
    indel whose A1 sorts after its A2, which `sorted_alid_locus` would accept with its alleles
    swapped."""
    parts = alid.split(":")
    if len(parts) != 4:
        return False
    chrom, pos, a1, a2 = parts
    if len(a1) == len(a2) or not a1 > a2:
        return False
    return sorted_alid_locus(f"{chrom}:{pos}:{a2}:{a1}") is not None


def refuse_reference_first_indel(alid: str, where: str) -> None:
    """Raise `ReferenceFirstIndel` when `alid` spells an indel reference first; `where` names
    the Store or reader that spells ALIDs sorted."""
    if spells_indel_reference_first(alid):
        chrom, pos, a1, a2 = alid.split(":")
        raise ReferenceFirstIndel(
            f"{alid} spells an indel reference first (ADR 0079), but {where} spells indels "
            f"sorted, so it would hold this variant as {chrom}:{pos}:{a2}:{a1}. That row is "
            "this deletion unless the panel also has the insertion at that site; either way r "
            f"there is signed against the other allele, {a2}, not {a1} (#462)"
        )


def resolve_sorted_reversed_indels(
    records: Sequence[tuple[str, int, str, str]],
) -> tuple[list[int], list[int]]:
    """Which `(chrom, position, ref, alt)` records a sorted Store keeps, and which it drops.

    The sorted spelling gives an insertion and a deletion at one position one ALID (`T/TA` and
    `TA/T` are both `T:TA`). The record whose REF the ALID spells first is kept, so the kept
    record is the one the reference-first spelling names by the same ALID, and the other is
    dropped, counted as `variants_dropped.duplicate_reversed_indel` (build-inputs §6). This is
    the rule ADR 0072 D5 gave GWAS Stores, retired there by ADR 0079 and kept here because a
    sorted Store still needs it. Returns the indices of the kept and of the dropped, each in
    input order. Raises `DuplicateRecord` for records that no split of one site gives:
    identical REF and ALT, a third record, or two SNVs. A record `sorted_identity` refuses
    raises its own error.
    """
    seen: dict[str, dict[bool, int]] = {}
    for index, (chrom, position, ref, alt) in enumerate(records):
        identity = sorted_identity(chrom, position, ref, alt)
        by_side = seen.setdefault(identity.alid, {})
        _require_a_split(identity, index, by_side)
        by_side[identity.a1_is_alt] = index
    # A1 = REF exactly when A1 is not the ALT; the record spelling REF second is dropped.
    dropped = sorted(by_side[True] for by_side in seen.values() if len(by_side) == 2)
    lost = set(dropped)
    return [i for i in range(len(records)) if i not in lost], dropped


def _require_a_split(identity: SortedIdentity, index: int, by_side: dict[bool, int]) -> None:
    """Refuse a record that no split of one site gives beside the records already at its ALID."""
    snv = len(identity.a1) == 1 and len(identity.a2) == 1
    if identity.a1_is_alt in by_side or (snv and by_side):
        raise DuplicateRecord(
            f"record {index} is not a split of one site: it shares {identity.alid} with record "
            f"{next(iter(by_side.values()))}"
        )


def _is_canonical_position(text: str) -> bool:
    if not (text.isascii() and text.isdigit()) or len(text) > _MAX_POSITION_DIGITS:
        return False
    return text == str(int(text)) and 0 < int(text) <= _MAX_POSITION


def _is_allele(text: str) -> bool:
    return bool(text) and VALID_BASES.issuperset(text)


__all__ = [
    "DuplicateRecord",
    "ReferenceFirstIndel",
    "SortedIdentity",
    "refuse_reference_first_indel",
    "resolve_sorted_reversed_indels",
    "sorted_alid_locus",
    "sorted_identity",
    "spells_indel_reference_first",
]
