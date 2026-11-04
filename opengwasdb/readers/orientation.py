"""How a Source Reader signs a statistic by an allele swap (spec §5, ADR 0036).

A Source Reader stores every association on the canonical A1, the lesser of a
variant's two alleles as `opengwasdb.variants.normalise` spells them. When the
source's effect allele is A2 the row is *flipped*: its z is negated and its
effect-allele frequency becomes ``1 - af``, so the frequency stays the stored
effect allele's. A reverse-strand lift can swap A1 and A2 again, and the same
rule re-orients the row.

This module is the only code that applies that rule. `swap_one` is its per-row
form, for the association stream a Dense or Hybrid build routes row by row;
`swap` and `a1_frequencies` are its array forms, for a batch of rows. It never
decides *whether* a row is flipped, which is the identity rule's, and it never
changes how a source spells an allele (ADR 0075).
"""

from __future__ import annotations

import numpy as np

from opengwasdb.variants.normalise import identity_allele

__all__ = ["a1_frequencies", "is_palindromic", "swap", "swap_one"]

# Strand-ambiguous allele pairs: unalignable without strand info, so excluded.
_PALINDROMES = frozenset({frozenset({"A", "T"}), frozenset({"C", "G"})})


def swap_one(flipped: bool, z: float, eaf: float | None) -> tuple[float, float | None]:
    """One row's ``(z, eaf)`` on A1: negated and complemented when ``flipped``.

    ``eaf`` is the frequency of the row's own effect allele, ``None`` where the
    source reports none, which stays ``None``: a frequency is never invented.
    A NaN or infinite z is negated like any other, so a row Association
    Admission will drop keeps the value it would have had.
    """
    if not flipped:
        return z, eaf
    return -z, None if eaf is None else 1.0 - eaf


def swap(
    flipped: np.ndarray, z: np.ndarray, eaf: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """`swap_one` over parallel arrays; ``eaf`` is NaN where a row has none."""
    return np.where(flipped, -z, z), a1_frequencies(flipped, eaf)


def a1_frequencies(flipped: np.ndarray, af: np.ndarray) -> np.ndarray:
    """Each row's frequency of its A1, from the frequency of its own effect allele.

    The frequency half of `swap`, for a caller that reads frequencies alone:
    ancestry assignment, the resolver's MAF filter and a BESD file set's
    shared `.esi` frequencies. NaN stays NaN.
    """
    oriented: np.ndarray = np.where(flipped, 1.0 - af, af)
    return oriented


def is_palindromic(ref: str, alt: str) -> bool:
    """True for a strand-ambiguous A/T or C/G pair, in either case.

    A frequency read at such a site cannot be put on a strand, so the
    annotation projections leave it out.
    """
    return frozenset({identity_allele(ref), identity_allele(alt)}) in _PALINDROMES
