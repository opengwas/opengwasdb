"""Canonical variant identity and association orientation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

VALID_BASES = frozenset("ACGT")
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
    if not chrom:
        raise VariantNormalisationError("chromosome is missing")
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
        raise VariantNormalisationError("allele is missing")
    if any(base not in VALID_BASES for base in value):
        raise VariantNormalisationError(f"unsupported allele {value!r}")
    return value


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

    chrom = normalise_chromosome(chromosome)
    try:
        pos = int(position)
    except (TypeError, ValueError) as exc:
        raise VariantNormalisationError(f"invalid position {position!r}") from exc
    if pos <= 0:
        raise VariantNormalisationError(f"invalid position {position!r}")

    effect = normalise_allele(source_effect_allele)
    other = normalise_allele(source_other_allele)
    if effect == other:
        raise VariantNormalisationError("effect and other alleles are identical")

    a1, a2 = sorted((effect, other))
    variant = CanonicalVariant(chromosome=chrom, position=pos, effect_allele=a1, other_allele=a2)
    return Orientation(variant=variant, flipped=(effect != a1))


def chromosome_sort_key(chromosome: str) -> tuple[int, str]:
    """Sort chromosomes in natural human order where possible."""

    chrom = normalise_chromosome(chromosome)
    if chrom.isdigit():
        return (int(chrom), chrom)
    special = {"X": 23, "Y": 24, "MT": 25}
    return (special.get(chrom.upper(), 1000), chrom)
