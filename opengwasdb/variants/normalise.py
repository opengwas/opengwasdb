"""Canonical variant identity and association orientation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

VALID_BASES = frozenset("ACGT")
#: A one-base allele in either case: the cell nearly every row holds, checked
#: without a call.
_ONE_BASE = frozenset("ACGTacgt")
#: The spellings a source uses for a chromosome it does not report, compared
#: upper-cased (issue #365).
_MISSING_CHROMOSOMES = frozenset({"", ".", "NA", "NAN", "NONE"})
_AUTOSOMES = frozenset(str(number) for number in range(1, 23))
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


class VariantNormalisationError(ValueError):
    """Raised when a source row cannot be represented as a canonical variant."""


class MalformedIdentity(VariantNormalisationError):
    """A source row whose identity no ALID can spell (issue #365).

    Its chromosome is missing, or an allele lies outside the ALID alphabet.
    Every Source Reader drops such a row, and Association Admission counts and
    logs it under ``reason``. Any other `VariantNormalisationError` is the
    reader's own to handle.
    """

    reason = ""


class MissingChromosome(MalformedIdentity):
    """The source gives no chromosome: an empty cell or a missing-value token."""

    reason = "chromosome_missing"


class InvalidAllele(MalformedIdentity):
    """An allele is empty, or holds a character other than A, C, G or T."""

    reason = "allele_invalid"


@dataclass
class IdentityFaults:
    """The rows one stream dropped for a malformed identity: ``(source key, reason)``.

    A source key is ``chromosome:position:ref:alt`` as the source spelt them.
    """

    rows: list[tuple[str, str]] = field(default_factory=list)

    def record(self, key: str, fault: MalformedIdentity) -> None:
        self.rows.append((key, fault.reason))


@dataclass(frozen=True)
class CanonicalVariant:
    """Store-local canonical variant identity before variant-index assignment."""

    chromosome: str
    position: int
    effect_allele: str
    other_allele: str

    @property
    def alid(self) -> str:
        return f"{self.chromosome}:{self.position}:{self.effect_allele}:{self.other_allele}"


@dataclass(frozen=True)
class Orientation:
    """How a source association maps to canonical ALID orientation."""

    variant: CanonicalVariant
    flipped: bool


def normalise_chromosome(chromosome: str) -> str:
    """Return the canonical label for a supported physical chromosome.

    The canonical non-autosomal labels are explicitly ``X``, ``Y`` and ``MT``;
    numeric PLINK-style spellings are aliases, not separate identities (ADR 0052).
    """

    chrom = str(chromosome).strip()
    if chrom.lower().startswith("chr"):
        chrom = chrom[3:]
    canonical = _CHROMOSOME_ALIASES.get(chrom.upper())
    if canonical is not None:
        return canonical
    if chrom in _AUTOSOMES:
        return chrom
    if chrom.upper() in _MISSING_CHROMOSOMES:
        raise MissingChromosome(f"chromosome is missing: {chromosome!r}")
    if chrom.isdigit():
        raise VariantNormalisationError(f"unsupported chromosome {chromosome!r}")
    return chrom


def identity_allele(allele: str) -> str:
    """An allele as a variant's identity is assigned from it: upper-cased (spec §4).

    Bases are case-insensitive in VCF and GWAS-SSF, so ``a`` is the base ``A``.
    Every routing key and ALID, and every choice of A1 -- which sets the sign a
    statistic is stored with -- is made from alleles spelt this way, on every
    build path and in extraction (issue #360). A source's own spelling is kept
    only as provenance: a Variant Reference's source keys, and the rsid
    harvest, which compares the names a site's spellings carry.

    This is the rule. `identity_alleles` is its batch form, for Pass 2's
    per-association keys, and `identity_base` its byte-level fast form, for a
    reader's one-base cells; a contract test holds all three to one answer
    over every byte an allele can hold.
    """
    return allele.upper()


def identity_alleles(alleles: Iterable[str]) -> list[str]:
    """`identity_allele` over a batch."""
    return list(map(identity_allele, alleles))


def identity_base(cell: bytes) -> str | None:
    """`normalise_allele` of a cell holding one base, at byte speed, else ``None``.

    The fast form of `identity_allele` for a reader's per-row hot path: the
    mask ``0xDF`` clears ASCII's case bit, and only the ALID alphabet is
    returned, so every other cell -- an indel, an empty cell, any other byte
    -- is left to `normalise_allele`.
    """
    stripped = cell.strip()
    if len(stripped) != 1:
        return None
    base = chr(stripped[0] & 0xDF)
    return base if base in VALID_BASES else None


def canonical_alid(chromosome: str, position: int | str, first: str, second: str) -> str:
    """The ALID of the variant whose alleles are ``first`` and ``second``, either way round.

    ``chr:pos:A1:A2``: each allele spelt as `identity_allele` spells it, A1 the
    alphabetically first. The one place an ALID is formed from raw alleles.
    """
    a1, a2 = sorted((identity_allele(first), identity_allele(second)))
    return f"{chromosome}:{position}:{a1}:{a2}"


def normalise_allele(allele: str) -> str:
    """Upper-case (`identity_allele`) and validate a simple v0.1 allele string."""

    value = identity_allele(str(allele).strip())
    if not value:
        raise InvalidAllele("allele is missing")
    if any(base not in VALID_BASES for base in value):
        raise InvalidAllele(f"unsupported allele {value!r}")
    return value


def allele_fault(*alleles: str | None) -> InvalidAllele | None:
    """Why the first of ``alleles`` outside the ALID alphabet is, or ``None``.

    `normalise_allele`'s rule, for a reader that forms no `Orientation`: an
    allele is judged as identity spells it (`identity_allele`), so ``a`` is in
    the alphabet. A one-base cell costs a set lookup. A missing cell (``None``)
    is empty.
    """
    for allele in alleles:
        if allele in _ONE_BASE:
            continue
        try:
            normalise_allele(allele or "")
        except InvalidAllele as fault:
            return fault
    return None


def identity_fault(chromosome: str, *alleles: str | None) -> MalformedIdentity | None:
    """Why a row's identity is malformed (#365), or ``None``, in
    `orient_to_canonical`'s order: the chromosome, then the alleles.

    A chromosome refused for another reason (an unsupported number) is not this
    rule's, and neither are the row's alleles then: `orient_to_canonical`
    refuses the row for its chromosome first.
    """
    try:
        normalise_chromosome(chromosome)
    except MalformedIdentity as fault:
        return fault
    except VariantNormalisationError:
        return None
    return allele_fault(*alleles)


def orient_to_canonical(
    chromosome: str,
    position: int | str,
    source_effect_allele: str,
    source_other_allele: str,
) -> Orientation:
    """Create an ALID with alphabetically first allele as canonical effect allele.

    If the source effect allele is not the canonical A1, signed statistics must
    be negated by the caller.
    """

    # A malformed identity is judged before the position (#365), so a row is
    # dropped for the same reason whichever reader forms it.
    chrom = normalise_chromosome(chromosome)
    effect = normalise_allele(source_effect_allele)
    other = normalise_allele(source_other_allele)
    try:
        pos = int(position)
    except (TypeError, ValueError) as exc:
        raise VariantNormalisationError(f"invalid position {position!r}") from exc
    if pos <= 0:
        raise VariantNormalisationError(f"invalid position {position!r}")

    if effect == other:
        raise VariantNormalisationError("effect and other alleles are identical")

    a1, a2 = sorted((effect, other))
    variant = CanonicalVariant(chromosome=chrom, position=pos, effect_allele=a1, other_allele=a2)
    return Orientation(variant=variant, flipped=(effect != a1))


def oriented_alid(
    chromosome: str, position: int, effect: str, other: str
) -> tuple[str, bool] | None:
    """`orient_to_canonical`'s ALID and flip, for a row whose identity is already judged.

    For a reader's per-row hot path: ``chromosome`` is `normalise_chromosome`'s
    and ``position`` positive, and both alleles have passed `allele_fault`, so
    `identity_allele` spells them as `normalise_allele` would. ``None`` where
    the two are one allele, which `orient_to_canonical` refuses. A contract
    test holds the two to one answer.
    """
    snv = _SNV_ORIENTATION.get((effect, other))
    if snv is not None:
        first, second, flipped = snv
        return f"{chromosome}:{position}:{first}:{second}", flipped
    first, second = identity_allele(effect), identity_allele(other)
    if first == second:
        return None
    if first < second:
        return f"{chromosome}:{position}:{first}:{second}", False
    return f"{chromosome}:{position}:{second}:{first}", True


#: `oriented_alid`'s answer for every pair of distinct one-base alleles, in
#: either case: ``(A1, A2, flipped)``, formed by `identity_allele`.
_SNV_ORIENTATION: dict[tuple[str, str], tuple[str, str, bool]] = {
    (effect, other): (
        min(identity_allele(effect), identity_allele(other)),
        max(identity_allele(effect), identity_allele(other)),
        identity_allele(effect) > identity_allele(other),
    )
    for effect in _ONE_BASE
    for other in _ONE_BASE
    if identity_allele(effect) != identity_allele(other)
}


def orient_or_drop(
    chromosome: str | None,
    position: int | str | None,
    source_effect_allele: str | None,
    source_other_allele: str | None,
    faults: IdentityFaults | None = None,
) -> Orientation | None:
    """`orient_to_canonical`, or ``None`` for a row that names no variant.

    A malformed identity is recorded in ``faults`` under its source key
    (``chromosome:position:other:effect``, the source's ref then alt), for
    Association Admission to count and log (#365); every other refusal stays
    the reader's own silent drop. A cell a truncated row does not reach is
    ``None``, which is an empty cell: a missing chromosome or allele.
    """
    chromosome, position = chromosome or "", "" if position is None else position
    source_effect_allele = source_effect_allele or ""
    source_other_allele = source_other_allele or ""
    try:
        return orient_to_canonical(
            chromosome, position, source_effect_allele, source_other_allele
        )
    except MalformedIdentity as fault:
        if faults is not None:
            key = f"{chromosome}:{position}:{source_other_allele}:{source_effect_allele}"
            faults.record(key, fault)
        return None
    except VariantNormalisationError:
        return None


def chromosome_sort_key(chromosome: str) -> tuple[int, str]:
    """Sort chromosomes in natural human order where possible."""

    chrom = normalise_chromosome(chromosome)
    if chrom.isdigit():
        return (int(chrom), chrom)
    special = {"X": 23, "Y": 24, "MT": 25}
    return (special.get(chrom.upper(), 1000), chrom)
