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

# The sorted spelling's read side -- parsing a sorted ALID and refusing a reference-first indel --
# is the LD Store reader's, and its one owner is ogquery (opengwas/ogquery#2). This module keeps
# the record identity a build forms, and re-exports the read names where a caller imports them.
from ogquery.alid import ReferenceFirstIndel as ReferenceFirstIndel
from ogquery.alid import refuse_reference_first_indel as refuse_reference_first_indel
from ogquery.alid import sorted_alid_locus as sorted_alid_locus
from ogquery.alid import spells_indel_reference_first as spells_indel_reference_first

from opengwasdb.store_reader.identity import (
    VariantNormalisationError,
    identity_allele,
    normalise_allele,
)
from opengwasdb.variants.normalise import orient_to_canonical, sorted_alid_alleles


class DuplicateRecord(VariantNormalisationError):
    """Records that share a sorted ALID and are not the two halves of one reversed indel:
    identical REF and ALT, a third record, or two SNVs (`resolve_sorted_reversed_indels`)."""

    reason = "duplicate_record"


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


@dataclass(frozen=True)
class SortedLookup:
    """How a GWAS Store's ALID is looked up in a Store spelt sorted (ADR 0106 §6).

    `alid` is the sorted spelling to look up. `swapped` says it puts the other allele first, so
    an r the sorted Store gives for this variant is signed to the GWAS Store's A2 and is negated
    to sign it to its A1. `reversed_indel` is, for an indel, the other indel of the same sequence
    at the same position, spelt as the GWAS Store spells it, reference first: the two share
    `alid`, and a sorted Store keeps one record for both (`resolve_sorted_reversed_indels`). It
    is None for alleles of one length, whose sorted spelling no other variant shares.
    """

    alid: str
    swapped: bool
    reversed_indel: str | None


def sorted_lookup(alid: str) -> SortedLookup:
    """The sorted spelling of a GWAS Store's ALID (`alid`), for a lookup in a sorted Store.

    A Store from format 0.2.2 spells an indel reference first (ADR 0079), and an LD Store sorts
    both alleles (`sorted_alid_alleles`); every other pair is spelt the same by both. A deletion
    whose reference sorts after its alternate is spelt with its alleles swapped (`swapped`). An
    ALID that is not four fields is its own spelling: a sorted Store holds no such ALID, and the
    lookup finds it Absent.
    """
    parts = alid.split(":")
    if len(parts) != 4:
        return SortedLookup(alid, False, None)
    chrom, pos, a1, a2 = parts
    first, second = sorted_alid_alleles(a1, a2)
    spelt = f"{chrom}:{pos}:{first}:{second}"
    swapped = first != identity_allele(a1)
    reversed_indel = f"{chrom}:{pos}:{a2}:{a1}" if len(a1) != len(a2) else None
    return SortedLookup(spelt, swapped, reversed_indel)


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
