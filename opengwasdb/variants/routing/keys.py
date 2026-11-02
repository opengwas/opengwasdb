"""How a build spells a source coordinate when it routes it (spec §4).

A Routing Key is the string a source coordinate is matched by, in Pass 1's
harvest and in every Pass 2 lookup. A Dense build keys a coordinate by the
source's own spelling; a Hybrid build upper-cases the alleles, since a Source
Reader keeps the source's spelling, an ALID is upper-case, and case is not an
orientation change. The harvested names and the associations are keyed by
the same function, so a name is looked up exactly as its association is
routed (issue #255).
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np

#: A source's own variant identity, exactly as a SourceReader streams it.
SourceKey = tuple[str, int, str, str]


class RoutingKey(StrEnum):
    """The spelling a build routes a source coordinate by."""

    #: The source's own allele spelling (Dense).
    EXACT = "exact"
    #: The alleles upper-cased, in the source's order (Hybrid).
    CASE_FOLDED = "case_folded"

    def of(self, site: SourceKey) -> str:
        """``site`` as a ``chrom:pos:ref:alt`` routing key."""
        chrom, pos, ref, alt = site
        if self is RoutingKey.CASE_FOLDED:
            return f"{chrom}:{pos}:{ref.upper()}:{alt.upper()}"
        return f"{chrom}:{pos}:{ref}:{alt}"

    def origin(self, site: SourceKey) -> str:
        """The source-build ALID ``site`` names: its alleles sorted, spelt as routed."""
        chrom, pos, ref, alt = site
        if self is RoutingKey.CASE_FOLDED:
            ref, alt = ref.upper(), alt.upper()
        a1, a2 = sorted((ref, alt))
        return f"{chrom}:{pos}:{a1}:{a2}"

    @property
    def folds(self) -> bool:
        """Whether two different coordinates can share one key."""
        return self is RoutingKey.CASE_FOLDED

    def encode_batch(
        self, chroms: list[str], poss: list[int], refs: list[str], alts: list[str]
    ) -> np.ndarray:
        """A batch's routing keys as one byte-string array, vectorised.

        Byte-for-byte the keys `of` writes, so a batch looked up in a
        `RouteTable` matches the keys the table was built with. ``pos``
        int->bytes goes through numpy (``np.int64(10).astype('S') == b'10'``),
        not per-element Python string work.
        """
        if self is RoutingKey.CASE_FOLDED:
            refs = [ref.upper() for ref in refs]
            alts = [alt.upper() for alt in alts]
        key = np.char.add(np.asarray(chroms, dtype="S"), b":")
        key = np.char.add(key, np.asarray(poss, dtype=np.int64).astype("S"))
        key = np.char.add(key, b":")
        key = np.char.add(key, np.asarray(refs, dtype="S"))
        key = np.char.add(key, b":")
        return np.char.add(key, np.asarray(alts, dtype="S"))
