"""How a build spells a source coordinate when it routes it (spec §4).

A Routing Key is the string a source coordinate is matched by, in Pass 1's
harvest and in every Pass 2 lookup: ``chrom:pos:ref:alt``, its alleles spelt
as identity is assigned from them (`identity_allele`, upper-cased) and kept in
the source's order. Case is not an orientation change, so two spellings of
one site share a key on every build (issue #360). The harvested names and the
associations are keyed by the same functions, so a name is looked up exactly
as its association is routed (issue #255).

Pass 2 matches a key as one integer (`IntegerKeyCodec`, issue #316): the
string search it replaced spent about a third of a FinnGen worker's time
building byte keys and comparing them in Python.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from opengwasdb.variants.normalise import canonical_alid, identity_allele, identity_alleles

#: A source's own variant identity, exactly as a SourceReader streams it.
SourceKey = tuple[str, int, str, str]

#: Hashes a Routing Key's bytes. Only the low `_HASH_BITS` bits of what it
#: returns reach the integer key.
KeyHash = Callable[[bytes], int]


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


def key_hash(key: bytes) -> int:
    """The hash `IntegerKeyCodec` takes of a Routing Key it cannot spell exactly.

    CRC-32, because it is the same in every process and every run: `hash()` is
    salted per interpreter, so a table built under one salt would find no row
    for a key hashed under another, and each such association would be dropped
    as off its axis rather than refused. Looked up when a table is built, not
    bound as a default, so a test can make every key collide.
    """
    return zlib.crc32(key)


# One integer key, high bit first: chromosome code (16) | position (30) |
# hashed (1) | alleles (17). A one-base SNV's alleles fill the low four bits
# exactly, ref then alt; any other row's 17 bits are its key's hash.
_HASH_BITS = 17
_HASHED = 1 << _HASH_BITS
_POSITION_SHIFT = _HASH_BITS + 1
_POSITION_BITS = 30
_CHROMOSOME_SHIFT = _POSITION_SHIFT + _POSITION_BITS
#: The most chromosome labels one table can code.
MAX_CHROMOSOMES = 1 << (64 - _CHROMOSOME_SHIFT)
#: A position the field cannot hold is written as this, and the row hashed.
_POSITION_LIMIT = (1 << _POSITION_BITS) - 1

#: Each byte's two-bit base, in either case; 4 for any other byte.
_BASE_CODE = np.full(256, 4, dtype=np.uint64)
for _code, _base in enumerate(b"ACGT"):
    _BASE_CODE[_base] = _BASE_CODE[_base | 0x20] = _code


@dataclass(frozen=True)
class IntegerKeys:
    """One batch's Routing Keys as integers.

    ``values`` is each row's key, meaningful only where ``known``: a row whose
    chromosome the table does not hold matches nothing. ``hashed`` holds, in
    row order, the known rows whose integer is a hash of their Routing Key,
    and ``full`` those Routing Keys, which a match must equal.
    """

    values: np.ndarray
    known: np.ndarray
    hashed: np.ndarray
    full: list[bytes]


class _ChromosomeCodes(dict[str, int]):
    def __missing__(self, chromosome: str) -> int:
        return -1


class IntegerKeyCodec:
    """Spells a Routing Key as a ``uint64`` that orders and narrows a search.

    A key holds its chromosome's code and its position exactly. A one-base SNV
    of A, C, G or T, in either case, holds its alleles exactly too, so equal
    integers are equal Routing Keys. Every other key -- an indel, a key of any
    length, one whose position the field cannot hold -- holds a hash of its
    whole Routing Key instead, and a bit no exact key sets: equal integers then
    only say two keys may be equal, and the caller compares the ``full`` keys.
    """

    def __init__(self, chromosomes: Iterable[str], hash_key: KeyHash) -> None:
        codes = _ChromosomeCodes()
        for chromosome in sorted(set(chromosomes)):
            codes[chromosome] = len(codes)
        if len(codes) > MAX_CHROMOSOMES:
            raise ValueError(
                f"a route table names {len(codes)} chromosomes; its integer key "
                f"codes at most {MAX_CHROMOSOMES}"
            )
        self._codes = codes
        self._hash = hash_key

    def encode(
        self,
        chroms: Sequence[str],
        poss: Sequence[int],
        refs: Sequence[str],
        alts: Sequence[str],
    ) -> IntegerKeys:
        """The integer keys of one batch of ``chrom:pos:ref:alt`` coordinates."""
        n = len(chroms)
        chromosome = np.fromiter(map(self._codes.__getitem__, chroms), dtype=np.int64, count=n)
        position = np.asarray(poss, dtype=np.int64).reshape(n)
        known = chromosome >= 0
        held = (position >= 0) & (position <= _POSITION_LIMIT)
        ref, alt = _one_base_codes(refs), _one_base_codes(alts)
        exact = held & (ref < 4) & (alt < 4)
        hashed = np.flatnonzero(known & ~exact)
        full = [
            routing_key((chroms[i], poss[i], refs[i], alts[i])).encode()
            for i in hashed.tolist()
        ]
        alleles = (ref << np.uint64(2)) | alt
        mask = _HASHED - 1
        alleles[hashed] = np.fromiter(
            (_HASHED | (value & mask) for value in map(self._hash, full)),
            dtype=np.uint64,
            count=len(full),
        )
        values = (
            (np.maximum(chromosome, 0).astype(np.uint64) << np.uint64(_CHROMOSOME_SHIFT))
            | (np.where(held, position, _POSITION_LIMIT).astype(np.uint64)
               << np.uint64(_POSITION_SHIFT))
            | alleles
        )
        return IntegerKeys(values, known, hashed, full)


def _one_base_codes(alleles: Sequence[str]) -> np.ndarray:
    """Each allele's two-bit base where it is one of A, C, G or T in either
    case, else 4.

    The alleles are joined and read as bytes once, rather than one Python call
    an allele. A batch that is not ASCII, or whose alleles hold the separator,
    is given no exact allele, so each of its keys is hashed and confirmed in
    full instead.
    """
    n = len(alleles)
    codes = np.full(n, 4, dtype=np.uint64)
    if n == 0:
        return codes
    joined = "\n".join(alleles)
    if not joined.isascii():
        return codes
    data = np.frombuffer(joined.encode("ascii"), dtype=np.uint8)
    ends = np.flatnonzero(data == ord("\n"))
    if len(ends) != n - 1:
        return codes
    starts = np.concatenate(([0], ends + 1))
    one = np.concatenate((ends, [len(data)])) - starts == 1
    codes[one] = _BASE_CODE[data[starts[one]]]
    return codes


def is_hashed(values: np.ndarray) -> np.ndarray:
    """Whether each integer key holds a hash, so equal integers need not be
    equal Routing Keys."""
    hashed: np.ndarray = (values & np.uint64(_HASHED)) != 0
    return hashed
