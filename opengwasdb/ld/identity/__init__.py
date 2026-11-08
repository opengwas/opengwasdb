"""Variant identity: the engine's ALID and A1, so r is signed as the engine signs it (layer 0).

ogld must form exactly the engine's ALID (opengwas/opengwasdb `CONTEXT.md`, _Variant
Identity_) and sign r against its A1. A mismatch is silent: it flips the sign of r for
some variants, the allele-flip error fine-mapping diagnostics exist to catch. ogld#3
decides whether this imports the engine's function or keeps a copy held to it by a
conformance test; either way these signatures are the seam the rest of ogld calls.
ADR 0003 records the decision: a copy, held to the engine by `tests/data/identity/`.
"""

from __future__ import annotations

from dataclasses import dataclass

_VALID_BASES = frozenset("ACGT")
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


@dataclass(frozen=True)
class Orientation:
    """A genotype record's engine identity (spec §2, §3).

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

    Raises ValueError for a label the engine does not support. LD Store groups are named
    by this label (spec §1), and a query's `chrom` passes through it (ogld#3). Like the
    engine, a non-numeric label that is no alias (a contig such as `HLA-A`) passes through
    unchanged; the build's autosomes-only rule (build-inputs §2) refuses it, not this.
    """
    chrom = str(label).strip()
    if chrom.lower().startswith("chr"):
        chrom = chrom[3:]
    canonical = _CHROMOSOME_ALIASES.get(chrom.upper())
    if canonical is not None:
        return canonical
    if chrom in _AUTOSOMES:
        return chrom
    if chrom.upper() in _MISSING_CHROMOSOMES:
        raise ValueError(f"chromosome is missing: {label!r}")
    if chrom.isdigit():
        raise ValueError(f"unsupported chromosome {label!r}")
    return chrom


def _identity_allele(allele: str) -> str:
    value = str(allele).strip().upper()
    if not value:
        raise ValueError("allele is missing")
    if any(base not in _VALID_BASES for base in value):
        raise ValueError(f"unsupported allele {value!r}")
    return value


def orient(chrom: str, position: int, ref: str, alt: str) -> Orientation:
    """The engine's identity of the biallelic record `ref`/`alt` at `chrom`:`position` (ogld#3).

    A1 is the allele the engine puts first: alphabetically first after upper-casing. The
    engine does no trimming or left alignment (ADR 0003), and neither does this; the build's
    bcftools step left-aligns its input (build-inputs §3). Raises ValueError for a record the
    engine cannot identify: a missing or unsupported chromosome, an allele that is empty or
    outside A, C, G, T, a position below 1, or identical alleles. Judged in that order, as the
    engine judges them. Never truncates an allele (spec §2).
    """
    canonical_chrom = normalise_chromosome(chrom)
    ref_allele = _identity_allele(ref)
    alt_allele = _identity_allele(alt)
    if int(position) <= 0:
        raise ValueError(f"invalid position {position!r}")
    if ref_allele == alt_allele:
        raise ValueError("effect and other alleles are identical")
    a1, a2 = sorted((ref_allele, alt_allele))
    pos = int(position)
    return Orientation(
        chrom=canonical_chrom,
        position=pos,
        alid=f"{canonical_chrom}:{pos}:{a1}:{a2}",
        a1=a1,
        a2=a2,
        a1_is_alt=alt_allele == a1,
    )


def alid_locus(alid: str) -> tuple[str, int] | None:
    """The chromosome and position an ALID names, or None when it is not an ALID (spec §2).

    A reader finds an ALID by this locus, then compares full ALID strings at that
    position, so a malformed one is Absent rather than an error (ogld#3). An ALID is
    `chrom:position:A1:A2` with a canonical chromosome label, a position spelt as a
    positive integer without padding, and two alleles over A, C, G, T.
    """
    parts = alid.split(":")
    if len(parts) != 4:
        return None
    chrom, pos, a1, a2 = parts
    if not (_is_canonical_chromosome(chrom) and _is_canonical_position(pos)):
        return None
    if not (_is_allele(a1) and _is_allele(a2)):
        return None
    return chrom, int(pos)


def _is_canonical_chromosome(label: str) -> bool:
    try:
        return normalise_chromosome(label) == label
    except ValueError:
        return False


def _is_canonical_position(text: str) -> bool:
    return text.isascii() and text.isdigit() and text == str(int(text)) and int(text) > 0


def _is_allele(text: str) -> bool:
    return bool(text) and all(base in _VALID_BASES for base in text)


__all__ = ["Orientation", "alid_locus", "normalise_chromosome", "orient"]
