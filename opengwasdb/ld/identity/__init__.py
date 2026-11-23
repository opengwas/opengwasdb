"""Variant identity: the engine's ALID and A1, so r is signed as the engine signs it (layer 0).

ogld must form exactly the engine's ALID (opengwas/opengwasdb `CONTEXT.md`, _Variant
Identity_) and sign r against its A1. A mismatch is silent: it flips the sign of r for
some variants, the allele-flip error fine-mapping diagnostics exist to catch. This module
is a copy of the engine's rules at `1ab2b09`, held to the engine's own answers by the
fixture in `tests/ld/data/identity/` (ADR 0089). The engine upper-cases, validates A, C, G, T
and sorts the two alleles. It does not trim or left-align, though its glossary says it does.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

#: The autosomes, `1`–`22`, in the engine's order: the chromosomes a build reads (build-inputs §2).
AUTOSOMES: Final = tuple(str(number) for number in range(1, 23))
_VALID_BASES = frozenset("ACGT")
_MISSING_CHROMOSOMES = frozenset({"", ".", "NA", "NAN", "NONE"})
_AUTOSOME_SET = frozenset(AUTOSOMES)
_SEX_AND_MT = frozenset({"X", "Y", "MT"})
_CHROMOSOME_ALIASES = {
    "23": "X",
    "X": "X",
    "24": "Y",
    "Y": "Y",
    "25": "MT",
    "26": "MT",
    "M": "MT",
    "MT": "MT",
}
#: A position an int32 column holds (ld-store §2), spelt with at most this many digits.
_MAX_POSITION = 2**31 - 1
_MAX_POSITION_DIGITS = 10


class IdentityError(ValueError):
    """A record or label the engine's identity function refuses.

    `reason` is the engine's own: `chromosome_missing` or `allele_invalid`
    (opengwas/opengwasdb#365), and empty for every other refusal the engine makes (an
    unsupported chromosome number, a bad position, identical alleles). `duplicate_record` is
    ogld's, for two records that cannot both be one variant's split
    (`resolve_reversed_indels`).
    """

    def __init__(self, message: str, reason: str = "") -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class Orientation:
    """A genotype record's engine identity (ld-store §2, §3).

    `position` is the position the ALID carries, after any normalisation the engine
    applies, and the one the Variant Set is ordered by. `a1_is_alt` says which allele the
    genotype dosages count: r computed from ALT dosages is r for A1 when it is True, and
    must be negated when it is False.
    """

    chrom: str
    position: int
    alid: str
    a1: str
    a2: str
    a1_is_alt: bool


def normalise_chromosome(label: str) -> str:
    """The engine's canonical chromosome label (`1`–`22`, `X`, `Y`, `MT`; no `chr` prefix).

    Raises IdentityError for a label the engine refuses: a missing one, or a number outside
    `1`–`26`. LD Store groups are named by this label (ld-store §1), and a query's `chrom`
    passes through it (ogld#3). Like the engine, anything else passes through, trimmed of a
    `chr` prefix only: a contig such as `HLA-A`, and `-1`, `+22`, `22.0` or `chr 22` (which
    keeps its space). The build's autosomes-only rule (build-inputs §2) refuses those, not
    this; `alid_locus` accepts only `1`–`22`, `X`, `Y` and `MT`.
    """
    chrom = str(label).strip()
    if chrom.lower().startswith("chr"):
        chrom = chrom[3:]
    canonical = _CHROMOSOME_ALIASES.get(chrom.upper())
    if canonical is not None:
        return canonical
    if chrom in _AUTOSOME_SET:
        return chrom
    if chrom.upper() in _MISSING_CHROMOSOMES:
        raise IdentityError(f"chromosome is missing: {label!r}", "chromosome_missing")
    if chrom.isdigit():
        raise IdentityError(f"unsupported chromosome {label!r}")
    return chrom


def _identity_allele(allele: str) -> str:
    value = str(allele).strip().upper()
    if not value:
        raise IdentityError("allele is missing", "allele_invalid")
    if not _is_allele(value):
        raise IdentityError(f"unsupported allele {value!r}", "allele_invalid")
    return value


def orient(chrom: str, position: int, ref: str, alt: str) -> Orientation:
    """The engine's identity of the biallelic record `ref`/`alt` at `chrom`:`position` (ogld#3).

    A1 is the allele the engine puts first: alphabetically first after upper-casing. The
    engine does no trimming or left alignment (ADR 0089), and neither does this. Raises
    IdentityError for a record the engine cannot identify, judged in the engine's order: the
    chromosome, then ALT, then REF (the engine's effect allele is the ALT), then the position
    (an integer above 0), then identical alleles. Never truncates an allele (ld-store §2).
    """
    canonical_chrom = normalise_chromosome(chrom)
    alt_allele = _identity_allele(alt)
    ref_allele = _identity_allele(ref)
    try:
        pos = int(position)
    except (TypeError, ValueError) as exc:
        raise IdentityError(f"invalid position {position!r}") from exc
    if pos <= 0:
        raise IdentityError(f"invalid position {position!r}")
    if ref_allele == alt_allele:
        raise IdentityError("effect and other alleles are identical")
    a1, a2 = sorted((ref_allele, alt_allele))
    return Orientation(
        chrom=canonical_chrom,
        position=pos,
        alid=f"{canonical_chrom}:{pos}:{a1}:{a2}",
        a1=a1,
        a2=a2,
        a1_is_alt=alt_allele == a1,
    )


def alid_locus(alid: str) -> tuple[str, int] | None:
    """The chromosome and position an ALID names, or None when it is not an ALID (ld-store §2).

    A reader finds an ALID by this locus, then compares full ALID strings at that
    position, so a malformed one is Absent rather than an error, and this never raises. An
    ALID is `chrom:position:A1:A2`: a chromosome label of the format (`1`–`22`, `X`, `Y`, `MT`,
    ld-store §2), a position of at most 10 ASCII digits, positive, unpadded and within int32, and
    two distinct alleles over A, C, G, T with A1 sorted before A2.
    """
    parts = alid.split(":")
    if len(parts) != 4:
        return None
    chrom, pos, a1, a2 = parts
    if not (_is_canonical_chromosome(chrom) and _is_canonical_position(pos)):
        return None
    if not (_is_allele(a1) and _is_allele(a2) and a1 < a2):
        return None
    return chrom, int(pos)


_CHROMOSOME_RANK = {name: rank for rank, name in enumerate((*AUTOSOMES, "X", "Y", "MT"), start=1)}


def chromosome_rank(chrom: str) -> int:
    """A chromosome's place in the engine's order, `1`–`22`, then `X`, `Y`, `MT` (ld-store §2): the
    first field of `order_key`, so chromosomes are ordered by it alone, with no position or ALID
    made up to call `order_key` with. The label passes through `normalise_chromosome` first, so
    `chr1` ranks as `1`; a chromosome off that list has no rank and raises KeyError."""
    return _CHROMOSOME_RANK[normalise_chromosome(chrom)]


def order_key(chrom: str, position: int, alid: str) -> tuple[int, int, str]:
    """The engine's variant order, as a sort key: chromosome (`chromosome_rank`), then
    position ascending, then ALID ascending (bytewise: an ALID is ASCII) at one position
    (ld-store §2). A chromosome with no rank raises KeyError.

    It takes the three fields the order is defined on, so a caller holding them, as a
    clumping's result does, orders by the engine's rule without re-forming an identity
    from the ALID's alleles (ogld#14)."""
    return chromosome_rank(chrom), int(position), alid


def variant_order_key(o: Orientation) -> tuple[int, int, str]:
    """The engine's variant order of an identified record: `order_key` of its fields."""
    return order_key(o.chrom, o.position, o.alid)


def _is_canonical_chromosome(label: str) -> bool:
    return label in _AUTOSOME_SET or label in _SEX_AND_MT


def _is_canonical_position(text: str) -> bool:
    if not (text.isascii() and text.isdigit()) or len(text) > _MAX_POSITION_DIGITS:
        return False
    return text == str(int(text)) and 0 < int(text) <= _MAX_POSITION


def _is_allele(text: str) -> bool:
    return bool(text) and _VALID_BASES.issuperset(text)


def resolve_reversed_indels(
    records: Sequence[tuple[str, int, str, str]],
) -> tuple[list[int], list[int]]:
    """Which of `(chrom, position, ref, alt)` records a Variant Set keeps, and which it drops.

    An insertion and a deletion at one position form one ALID, because the ALID sorts its
    alleles (`T/TA` and `TA/T` are both `T:TA`). Following the engine (opengwas/opengwasdb ADR
    0072, `duplicate_reversed_indel`), the record whose REF the ALID spells first (REF = A1) is
    kept and the other is dropped, to be counted as `variants_dropped.duplicate_reversed_indel`
    (build-inputs §6). Returns the indices into `records` of the kept and of the dropped, each in
    input order. Raises IdentityError, reason `duplicate_record`, for two records that no split of
    one site gives: identical REF and ALT, or two SNVs sharing an ALID. Any record the engine
    refuses raises its own IdentityError.
    """
    seen: dict[str, dict[bool, int]] = {}
    for index, (chrom, position, ref, alt) in enumerate(records):
        o = orient(chrom, position, ref, alt)
        by_orientation = seen.setdefault(o.alid, {})
        _require_a_split(o, index, by_orientation)
        by_orientation[o.a1_is_alt] = index
    # A1 = REF exactly when A1 is not the ALT; the record spelling REF second is dropped.
    dropped = sorted(by_o[True] for by_o in seen.values() if len(by_o) == 2)
    return [i for i in range(len(records)) if i not in set(dropped)], dropped


def _require_a_split(o: Orientation, index: int, by_orientation: dict[bool, int]) -> None:
    """Refuse a record that no split of one site gives beside the records already at its ALID."""
    snv = len(o.a1) == 1 and len(o.a2) == 1
    if o.a1_is_alt in by_orientation or (snv and by_orientation):
        raise IdentityError(
            f"record {index} is not a split of one site: it shares {o.alid} with record "
            f"{next(iter(by_orientation.values()))}",
            "duplicate_record",
        )


__all__ = [
    "AUTOSOMES",
    "IdentityError",
    "Orientation",
    "alid_locus",
    "chromosome_rank",
    "normalise_chromosome",
    "order_key",
    "orient",
    "resolve_reversed_indels",
    "variant_order_key",
]
