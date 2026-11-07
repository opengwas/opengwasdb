"""Variant identity: the engine's ALID and A1, so r is signed as the engine signs it (layer 0).

ogld must form exactly the engine's ALID (opengwas/opengwasdb `CONTEXT.md`, _Variant
Identity_) and sign r against its A1. A mismatch is silent: it flips the sign of r for
some variants, the allele-flip error fine-mapping diagnostics exist to catch. ogld#3
decides whether this imports the engine's function or keeps a copy held to it by a
conformance test; either way these signatures are the seam the rest of ogld calls.
"""

from __future__ import annotations

from dataclasses import dataclass


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
    by this label (spec §1), and a query's `chrom` passes through it (ogld#3).
    """
    raise NotImplementedError("ogld#3")


def orient(chrom: str, position: int, ref: str, alt: str) -> Orientation:
    """The engine's identity of the biallelic record `ref`/`alt` at `chrom`:`position` (ogld#3).

    A1 is the allele the engine puts first: alphabetically first after upper-casing,
    trimming and left alignment. Raises ValueError for a record the engine cannot
    identify, such as identical alleles. Never truncates an allele (spec §2).
    """
    raise NotImplementedError("ogld#3")


def alid_locus(alid: str) -> tuple[str, int] | None:
    """The chromosome and position an ALID names, or None when it is not an ALID (spec §2).

    A reader finds an ALID by this locus, then compares full ALID strings at that
    position, so a malformed one is Absent rather than an error (ogld#3).
    """
    raise NotImplementedError("ogld#3")


__all__ = ["Orientation", "alid_locus", "normalise_chromosome", "orient"]
