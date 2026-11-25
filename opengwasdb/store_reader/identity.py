"""How a variant's identity is spelt: its chromosome label and its alleles (spec §4).

The read-side rules of variant identity, which every reader of a Store Release
applies to what it is asked for and `opengwasdb.variants.normalise` builds the
canonical identity of a source row on (#464): a chromosome's canonical label
(`normalise_chromosome`, ADR 0052), an allele's spelling (`identity_allele`,
`normalise_allele`), and the errors that refuse one.
"""

from __future__ import annotations

from typing import Final

VALID_BASES = frozenset("ACGT")

#: The spellings a source uses for a chromosome it does not report, compared
#: upper-cased (issue #365).
_MISSING_CHROMOSOMES = frozenset({"", ".", "NA", "NAN", "NONE"})

#: The autosomes, `1`-`22`, in the variant order (`chromosome_rank`).
AUTOSOMES: Final = tuple(str(number) for number in range(1, 23))
_AUTOSOMES = frozenset(AUTOSOMES)
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

_PRIMARY_NON_AUTOSOMES = frozenset({"X", "Y", "MT"})

#: Every canonical chromosome label an ALID carries: the autosomes, `X`, `Y` and `MT`
#: (`chromosome_rank` ranks exactly these).
CANONICAL_CHROMOSOMES: Final = frozenset((*AUTOSOMES, *_PRIMARY_NON_AUTOSOMES))


class VariantNormalisationError(ValueError):
    """Raised when a source row cannot be represented as a canonical variant."""


class UnsupportedChromosome(VariantNormalisationError):
    """A chromosome no ALID can spell: a number outside 1-26 (ADR 0052). `fields` is what a
    reader of the Store Reader contract compares (#474); the message is for people."""

    def __init__(self, chromosome: str) -> None:
        self.chromosome = chromosome
        super().__init__(f"unsupported chromosome {chromosome!r}")

    @property
    def fields(self) -> dict[str, object]:
        return {"chromosome": self.chromosome}


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
        raise UnsupportedChromosome(chromosome)
    return chrom


def is_primary_contig(chromosome: str) -> bool:
    """Whether ``chromosome`` is a primary-assembly contig: 1-22, X, Y or MT.

    The labels spec §4 makes canonical, and their numeric aliases (`23`-`26`,
    `M`), with or without a `chr` prefix. An alt, random or unplaced contig is
    not, and neither is a label no ALID can spell. This is the one definition:
    every path that lifts a source variant onto the release's assembly asks it
    of the contig the lift lands on (#376).
    """
    try:
        label = normalise_chromosome(chromosome)
    except VariantNormalisationError:
        return False
    return label in _AUTOSOMES or label in _PRIMARY_NON_AUTOSOMES


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


def normalise_allele(allele: str) -> str:
    """Upper-case (`identity_allele`) and validate a simple v0.1 allele string."""

    value = identity_allele(str(allele).strip())
    if not value:
        raise InvalidAllele("allele is missing")
    if any(base not in VALID_BASES for base in value):
        raise InvalidAllele(f"unsupported allele {value!r}")
    return value
