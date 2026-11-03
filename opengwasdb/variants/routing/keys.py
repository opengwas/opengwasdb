"""How a build spells a source coordinate when it routes it (spec §4).

A Routing Key is the string a source coordinate is matched by, in Pass 1's
harvest and in every Pass 2 lookup: ``chrom:pos:ref:alt``, its alleles spelt
as identity is assigned from them (`identity_allele`, upper-cased) and kept in
the source's order. Case is not an orientation change, so two spellings of
one site share a key on every build (issue #360). The harvested names and the
associations are keyed by the same functions, so a name is looked up exactly
as its association is routed (issue #255).
"""

from __future__ import annotations

import numpy as np

from opengwasdb.variants.normalise import canonical_alid, identity_allele, identity_alleles

#: A source's own variant identity, exactly as a SourceReader streams it.
SourceKey = tuple[str, int, str, str]


def routing_key(site: SourceKey) -> str:
    """``site`` as a ``chrom:pos:ref:alt`` routing key."""
    chrom, pos, ref, alt = site
    return f"{chrom}:{pos}:{identity_allele(ref)}:{identity_allele(alt)}"


def source_alid(site: SourceKey) -> str:
    """The source-build ALID ``site`` names: its own coordinate, as an ALID."""
    chrom, pos, ref, alt = site
    return canonical_alid(chrom, pos, ref, alt)


def encode_routing_keys(
    chroms: list[str], poss: list[int], refs: list[str], alts: list[str]
) -> np.ndarray:
    """A batch's routing keys as one byte-string array, vectorised.

    Byte-for-byte the keys `routing_key` writes, so a batch looked up in a
    `RouteTable` matches the keys the table was built with. ``pos``
    int->bytes goes through numpy (``np.int64(10).astype('S') == b'10'``),
    not per-element Python string work.
    """
    key = np.char.add(np.asarray(chroms, dtype="S"), b":")
    key = np.char.add(key, np.asarray(poss, dtype=np.int64).astype("S"))
    key = np.char.add(key, b":")
    key = np.char.add(key, np.asarray(identity_alleles(refs), dtype="S"))
    key = np.char.add(key, b":")
    return np.char.add(key, np.asarray(identity_alleles(alts), dtype="S"))
