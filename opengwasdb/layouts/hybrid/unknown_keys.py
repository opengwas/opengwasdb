"""One ``uint64`` per Pass 2 off-reference source key (ticket #218).

A Hybrid build with ``--variant-reference`` discovers variants the reference
never named only while Pass 2 streams the sources, so each worker spills its
Analysis's off-reference associations to a per-column file keyed by the raw
source coordinate. Carrying that key as a pickled ``dtype=object`` array of
strings forced every post-Pass-2 consumer to unpickle it and walk it in Python
one key at a time, which is why ``_unknown_key_assembly`` and the
``.unk`` → ``.ovf`` fold both ran at about 18 MiB/s on one core (issue #217).

This module gives each key a fixed-width ``uint64`` instead:

* an SNV with one-base A/C/G/T alleles packs *losslessly* -- chromosome
  (5 bits), position (28 bits), ref (2 bits), alt (2 bits) -- so the raw key can
  be reconstructed exactly;
* every other key (indels, multi-base or unusual alleles) hashes into the
  remaining 63 bits and carries its raw string in a small per-column side file,
  because liftover and canonicalisation still need it;
* the top bit tags which region a value is in, so a packed SNV can never collide
  with a hashed key.

Two distinct raw keys that hash to the same value are a silent wrong answer --
one association would be merged into the other -- so the encoder refuses them
loudly, naming both keys. A hash value outside the 63-bit region is likewise
refused rather than truncated to fit.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "EncodedKeys",
    "MAX_POSITION",
    "UnknownKeyEncodingError",
    "decode_keys",
    "decode_spill",
    "encode_key",
    "encode_keys",
    "hashed_lookup",
    "is_hashed",
    "pack_key",
    "unpack_key",
]

HashFn = Callable[[str], int]

# Packed SNV layout, low bit first: ref (2) | alt (2) | position (28) |
# chromosome (5). Bit 63 is the tag; a packed value never reaches it.
_POSITION_BITS = 28
_ALLELE_BITS = 2
_CHROMOSOME_BITS = 5
MAX_POSITION = (1 << _POSITION_BITS) - 1
_REF_SHIFT = 0
_ALT_SHIFT = _REF_SHIFT + _ALLELE_BITS
_POSITION_SHIFT = _ALT_SHIFT + _ALLELE_BITS
_CHROMOSOME_SHIFT = _POSITION_SHIFT + _POSITION_BITS
_CHROMOSOME_MASK = (1 << _CHROMOSOME_BITS) - 1

# Bit 63 separates hashed keys from packed SNVs.
HASH_TAG = 1 << 63
HASH_MASK = HASH_TAG - 1

_ALLELE_CODES = {"A": 0, "C": 1, "G": 2, "T": 3}
_ALLELE_LABELS = ("A", "C", "G", "T")

# Canonical labels after #216/ADR 0052: autosomes 1-22, then X, Y and MT.
_CHROMOSOME_CODES = {
    **{str(number): number for number in range(1, 23)},
    "X": 23,
    "Y": 24,
    "MT": 25,
}
_CHROMOSOME_LABELS = [""] * (1 << _CHROMOSOME_BITS)
for _label, _code in _CHROMOSOME_CODES.items():
    _CHROMOSOME_LABELS[_code] = _label

# Object arrays for vectorised packed decoding (see `_decode_packed`).
_CHROMOSOME_LABEL_ARRAY = np.array(_CHROMOSOME_LABELS, dtype=object)
_ALLELE_LABEL_ARRAY = np.array(_ALLELE_LABELS, dtype=object)


class UnknownKeyEncodingError(ValueError):
    """Raised when an off-reference key cannot be encoded or decoded faithfully."""


def _stable_hash(key: str) -> int:
    """A deterministic 63-bit hash of a raw key string.

    Deterministic across processes and runs (unlike ``hash()``, which is salted
    per interpreter and would make a spill undecodable by the parent after a
    fork), and wide enough that a collision is not a practical concern. A
    collision is still detected rather than merged.
    """
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") & HASH_MASK


def _split_key(key: str) -> tuple[str, int, str, str] | None:
    """``(chromosome, position, ref, alt)`` for a ``chrom:pos:ref:alt`` key."""
    parts = key.split(":")
    if len(parts) != 4:
        return None
    chromosome, position, ref, alt = parts
    if not (position.isascii() and position.isdigit()):
        return None
    return chromosome, int(position), ref, alt


def pack_key(chromosome: str, position: int, ref: str, alt: str) -> int | None:
    """The losslessly packed value for an SNV, or ``None`` when it is not one.

    ``None`` is the instruction to hash. A key is packable only when its
    chromosome is a canonical label after #216, its position fits the 28-bit
    field, and both alleles are one base of uppercase A/C/G/T -- everything else
    has to keep its raw string to round-trip exactly.
    """
    chromosome_code = _CHROMOSOME_CODES.get(chromosome)
    if chromosome_code is None or not 0 < position <= MAX_POSITION:
        return None
    ref_code = _ALLELE_CODES.get(ref)
    alt_code = _ALLELE_CODES.get(alt)
    if ref_code is None or alt_code is None:
        return None
    return (
        (chromosome_code << _CHROMOSOME_SHIFT)
        | (position << _POSITION_SHIFT)
        | (ref_code << _REF_SHIFT)
        | (alt_code << _ALT_SHIFT)
    )


def unpack_key(value: int) -> str:
    """Reconstruct the raw key a packed SNV value was built from."""
    if value & HASH_TAG:
        raise UnknownKeyEncodingError(
            f"value {value} is a hashed key and has no packed decoding"
        )
    chromosome = _CHROMOSOME_LABELS[(value >> _CHROMOSOME_SHIFT) & _CHROMOSOME_MASK]
    position = (value >> _POSITION_SHIFT) & MAX_POSITION
    ref = _ALLELE_LABELS[(value >> _REF_SHIFT) & 0b11]
    alt = _ALLELE_LABELS[(value >> _ALT_SHIFT) & 0b11]
    return f"{chromosome}:{position}:{ref}:{alt}"


def _hashed_value(key: str, hash_fn: HashFn) -> int:
    value = int(hash_fn(key))
    if not 0 <= value <= HASH_MASK:
        raise UnknownKeyEncodingError(
            f"hash of off-reference key {key!r} is {value}, outside the 63-bit hash "
            f"region [0, {HASH_MASK}]; refusing to truncate it"
        )
    return value | HASH_TAG


def _resolve_hash_fn(hash_fn: HashFn | None) -> HashFn:
    # Resolve the module default here, not as a default argument value, so a
    # test can inject a colliding hash function through the module attribute.
    return _stable_hash if hash_fn is None else hash_fn


def encode_key(key: str, *, hash_fn: HashFn | None = None) -> int:
    """Encode one raw off-reference key as a ``uint64``."""
    resolved = _resolve_hash_fn(hash_fn)
    parts = _split_key(key)
    if parts is not None:
        packed = pack_key(*parts)
        if packed is not None:
            return packed
    return _hashed_value(key, resolved)


@dataclass(frozen=True)
class EncodedKeys:
    """A column's encoded keys, plus the side-file half of the hashed ones."""

    values: np.ndarray  # uint64, one per input key
    hashed_index: np.ndarray  # int64, positions whose value is hashed
    hashed_raw: list[str]  # raw keys for those positions, in order

    def hashed_lookup(self) -> dict[int, str]:
        """``{encoded value: raw key}`` for the hashed half."""
        return {
            int(self.values[position]): raw
            for position, raw in zip(self.hashed_index.tolist(), self.hashed_raw, strict=True)
        }


def encode_keys(
    keys: Sequence[str], *, hash_fn: HashFn | None = None
) -> EncodedKeys:
    """Encode one column's raw keys, refusing any hash collision.

    The collision check is per column because the spill's last-wins dedup is:
    two distinct raw keys sharing a value would otherwise collapse into one
    association without a trace.
    """
    resolved = _resolve_hash_fn(hash_fn)
    values = np.empty(len(keys), dtype=np.uint64)
    hashed_index: list[int] = []
    hashed_raw: list[str] = []
    seen: dict[int, str] = {}
    for index, key in enumerate(keys):
        value = encode_key(key, hash_fn=resolved)
        values[index] = value
        if value & HASH_TAG:
            existing = seen.get(value)
            if existing is None:
                seen[value] = key
            elif existing != key:
                raise UnknownKeyEncodingError(
                    f"hash collision between off-reference keys {existing!r} and {key!r} "
                    f"(both encode to {value}); refusing to merge them"
                )
            hashed_index.append(index)
            hashed_raw.append(key)
    return EncodedKeys(
        values=values,
        hashed_index=np.array(hashed_index, dtype=np.int64),
        hashed_raw=hashed_raw,
    )


def is_hashed(values: np.ndarray) -> np.ndarray:
    """A bool mask saying which encoded values live in the hash region."""
    mask: np.ndarray = (values & np.uint64(HASH_TAG)) != np.uint64(0)
    return mask


def hashed_lookup(
    values: np.ndarray, hashed_index: np.ndarray, hashed_raw: Sequence[str]
) -> dict[int, str]:
    """``{encoded value: raw key}`` for a spill's hashed rows, validated.

    The side file is the only place a hashed key's raw string exists, so it must
    name every tagged row in ``values`` exactly once. A short, duplicated or
    misplaced entry raises rather than decoding to a shorter, plausible key list
    or a wrong key.
    """
    if len(hashed_index) != len(hashed_raw):
        raise UnknownKeyEncodingError(
            f"side file has {len(hashed_raw)} raw key(s) for {len(hashed_index)} hashed row(s)"
        )
    positions = np.asarray(hashed_index, dtype=np.int64)
    tagged = np.flatnonzero(is_hashed(values))
    if len(positions) != len(tagged) or not np.array_equal(np.sort(positions), tagged):
        raise UnknownKeyEncodingError(
            f"side file names {len(positions)} hashed row(s) but the spill has {len(tagged)}; "
            "the raw keys cannot be placed"
        )
    lookup: dict[int, str] = {}
    for position, raw in zip(positions.tolist(), hashed_raw, strict=True):
        value = int(values[position])
        existing = lookup.get(value)
        if existing is not None and existing != raw:
            raise UnknownKeyEncodingError(
                f"hash collision between off-reference keys {existing!r} and {raw!r} "
                f"(both encode to {value}); refusing to merge them"
            )
        lookup[value] = raw
    return lookup


def _decode_packed(values: np.ndarray) -> list[str]:
    """Decode a uint64 array of packed SNVs, field extraction vectorised.

    Only the final string formatting is per key; the bit-shuffling and label
    lookups run in numpy, which is what keeps the scratch-read pass from being
    slower than the pickled form it replaced (issue #218).
    """
    chromosome_codes = (
        (values >> np.uint64(_CHROMOSOME_SHIFT)) & np.uint64(_CHROMOSOME_MASK)
    ).astype(np.int64)
    positions = ((values >> np.uint64(_POSITION_SHIFT)) & np.uint64(MAX_POSITION)).astype(
        np.int64
    )
    ref_codes = ((values >> np.uint64(_REF_SHIFT)) & np.uint64(0b11)).astype(np.int64)
    alt_codes = ((values >> np.uint64(_ALT_SHIFT)) & np.uint64(0b11)).astype(np.int64)
    chromosomes = _CHROMOSOME_LABEL_ARRAY[chromosome_codes]
    refs = _ALLELE_LABEL_ARRAY[ref_codes]
    alts = _ALLELE_LABEL_ARRAY[alt_codes]
    return [
        f"{chromosome}:{position}:{ref}:{alt}"
        for chromosome, position, ref, alt in zip(
            chromosomes.tolist(),
            positions.tolist(),
            refs.tolist(),
            alts.tolist(),
            strict=True,
        )
    ]


def decode_spill(
    values: np.ndarray, hashed_index: np.ndarray, hashed_raw: Sequence[str]
) -> tuple[list[str], dict[int, str]]:
    """Decode a spill to raw keys and its hashed-value lookup in one pass.

    The lookup travels back because a build-wide collision check has to see every
    column's hashed keys without re-reading the side files (issue #218 review).
    """
    hashed = hashed_lookup(values, hashed_index, hashed_raw)
    decoded: list[str] = [""] * len(values)
    packed_positions = np.flatnonzero(~is_hashed(values))
    if len(packed_positions):
        packed_raw = _decode_packed(values[packed_positions])
        for position, raw in zip(packed_positions.tolist(), packed_raw, strict=True):
            decoded[position] = raw
    for position, raw in zip(hashed_index.tolist(), hashed_raw, strict=True):
        decoded[int(position)] = raw
    return decoded, hashed


def decode_keys(
    values: np.ndarray, hashed_index: np.ndarray, hashed_raw: Sequence[str]
) -> list[str]:
    """Decode a spilled key array back to raw keys, in row order.

    Packed SNVs are reconstructed from their bits; hashed keys come from the
    per-column side file. Every row must decode -- a missing or misplaced side
    entry fails loudly rather than producing a shorter, plausible key list.
    """
    raw_keys, _ = decode_spill(values, hashed_index, hashed_raw)
    return raw_keys
