"""Sorted global off-reference key table (ticket #222).

Pass 2 discovers variants the variant reference never named, one column per
Analysis. Resolving them to a shared Variant Index used to build two global
Python dicts keyed by the raw source coordinate -- one insert per association,
about 15 billion inserts on OGS-00011 -- and then look every association up in
them one at a time. This module replaces both dicts with a sorted ``uint64`` key
array (the encoding from #218) aligned with each key's final shared Variant
Index.

The build-wide distinct keys come from ``key_runs``' memory-bounded merge;
here they are resolved -- liftover and canonicalisation once per *distinct*
key -- into a :class:`KeyTable` the ``.unk`` → ``.ovf`` fold binary-searches
rather than walking a dict. :class:`AlidIndex` maps ALIDs onto the shared axis
without a ``str -> int`` dict over it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from opengwasdb.build.liftover import build_liftover_lookup
from opengwasdb.layouts.hybrid.key_runs import HG19, HG38
from opengwasdb.layouts.hybrid.unknown_keys import (
    UnknownKeyEncodingError,
    is_hashed,
    packed_alids,
    packed_fields,
)

__all__ = [
    "HG19",
    "HG38",
    "AlidIndex",
    "DistinctKeys",
    "KeyTable",
    "ResolvedKeys",
    "resolve_keys",
]


@dataclass(frozen=True)
class DistinctKeys:
    """The build's distinct off-reference keys, ready to resolve.

    ``values`` is sorted ascending; ``assembly_bits`` parallels it (the OR of
    every declaring assembly's bit). ``hashed_values`` is its hash-region
    subset, sorted, and ``hashed_raw`` the raw key of each -- read from the
    side files once the merge is done, because liftover and canonicalisation
    need the string.
    """

    values: np.ndarray
    assembly_bits: np.ndarray
    hashed_values: np.ndarray
    hashed_raw: list[str]


@dataclass(frozen=True)
class ResolvedKeys:
    """Distinct off-reference keys that resolved to an hg38 ALID.

    ``keys`` stays sorted so the fold can ``searchsorted`` it. ``alids`` and
    ``origins`` parallel it; ``origins`` is the canonical source coordinate,
    which is what ``hg38_to_source`` records before its collision blanking.
    """

    keys: np.ndarray
    alids: list[str]
    origins: list[str]


@dataclass(frozen=True)
class KeyTable:
    """A sorted ``uint64`` key array aligned with the shared Variant Index.

    The fold binary-searches ``keys``; a key that is absent failed resolution
    (two assemblies, or a failed lift) and its associations are dropped.
    """

    keys: np.ndarray
    shared_index: np.ndarray

    def lookup(self, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(shared index, matched)`` for each query key, vectorised.

        A missing key gets index ``0`` and ``matched=False``; callers must use
        the mask, never the index, for a miss.
        """
        if len(self.keys) == 0:
            return np.zeros(len(keys), dtype=np.int64), np.zeros(len(keys), dtype=bool)
        position = np.searchsorted(self.keys, keys)
        clipped = np.minimum(position, len(self.keys) - 1)
        matched = self.keys[clipped] == keys
        return self.shared_index[clipped], matched


_UINT64_MASK = (1 << 64) - 1


class AlidIndex:
    """ALID -> axis position as a sorted hash array, not a Python dict.

    Mapping an ALID to its index on the shared axis used to build a
    ``str -> int`` dict over every shared variant -- tens of millions of
    entries resident through consolidation. This keys the axis by the
    process-local string hash (sorted ``uint64``) and binary-searches it: 16
    bytes per ALID, plus a pointer to the axis list the caller already holds.

    A hash hit is confirmed by comparing the axis string with the query, so a
    query absent from the axis fails loudly even if its hash happens to match
    another ALID's. Two axis ALIDs sharing one hash would make the search
    ambiguous; in that (astronomically rare) case the index falls back to a
    dict rather than return another ALID's position.
    """

    def __init__(self, axis: Sequence[str]) -> None:
        self._axis = np.asarray(axis, dtype=object)
        self._fallback: dict[str, int] | None = None
        hashes = _string_hashes(axis)
        order = np.argsort(hashes, kind="stable")
        self._hashes = hashes[order]
        self._positions = order.astype(np.int64)
        if bool((self._hashes[1:] == self._hashes[:-1]).any()):
            self._fallback = {alid: index for index, alid in enumerate(axis)}

    def lookup(self, alids: Sequence[str]) -> np.ndarray:
        """The axis position of each ALID, in input order; absent ALIDs raise."""
        if not len(alids):
            return np.empty(0, dtype=np.int64)
        if self._fallback is not None:
            fallback = self._fallback
            return np.fromiter(
                (fallback[alid] for alid in alids), dtype=np.int64, count=len(alids)
            )
        if not len(self._hashes):
            raise UnknownKeyEncodingError(f"ALID {alids[0]!r} is absent from an empty axis")
        queries = _string_hashes(alids)
        clipped = np.minimum(np.searchsorted(self._hashes, queries), len(self._hashes) - 1)
        positions = self._positions[clipped]
        found = self._axis[positions] == np.asarray(alids, dtype=object)
        if not bool(found.all()):
            missing = alids[int(np.flatnonzero(~found)[0])]
            raise UnknownKeyEncodingError(
                f"ALID {missing!r} is absent from the shared axis; refusing to guess its index"
            )
        return positions


def _string_hashes(strings: Sequence[str]) -> np.ndarray:
    """Each string's process-local hash as ``uint64``."""
    return np.fromiter(
        (hash(value) & _UINT64_MASK for value in strings), dtype=np.uint64, count=len(strings)
    )


def _canonical_source_key(key: str) -> str:
    """The ALID a source key names before any liftover (alleles sorted)."""
    chrom, position, ref, alt = key.split(":")
    a1, a2 = sorted((ref, alt))
    return f"{chrom}:{int(position)}:{a1}:{a2}"


def _hashed_raw_list(queries: np.ndarray, merged: DistinctKeys) -> list[str]:
    """The side-file raw keys for hashed values, in ``queries`` order."""
    if len(queries) == 0:
        return []
    if len(merged.hashed_values) == 0:
        raise UnknownKeyEncodingError(
            "a hashed off-reference key has no raw side-file key; refusing to resolve it"
        )
    side = np.searchsorted(merged.hashed_values, queries)
    clipped = np.minimum(side, len(merged.hashed_values) - 1)
    if not np.array_equal(merged.hashed_values[clipped], queries):
        raise UnknownKeyEncodingError(
            "a hashed off-reference key has no raw side-file key; refusing to resolve it"
        )
    return [merged.hashed_raw[int(index)] for index in side]


def _origins(
    values: np.ndarray,
    alids: Sequence[str | None],
    is_hg19: np.ndarray,
    hashed: np.ndarray,
    merged: DistinctKeys,
) -> list[str | None]:
    """The canonical source coordinate of every distinct key.

    An hg38 key's canonical coordinate *is* its ALID, so ``alids`` is reused
    rather than recomputed; only the hg19 keys need their own hg19 coordinate
    (which liftover then moves away from).
    """
    origins: list[str | None] = list(alids)
    packed_positions = np.flatnonzero(is_hg19 & ~hashed)
    if len(packed_positions):
        packed = packed_alids(values[packed_positions])
        for position, origin in zip(packed_positions.tolist(), packed, strict=True):
            origins[position] = origin
    hashed_positions = np.flatnonzero(is_hg19 & hashed)
    raws = _hashed_raw_list(values[hashed_positions], merged)
    for position, raw in zip(hashed_positions.tolist(), raws, strict=True):
        origins[position] = _canonical_source_key(raw)
    return origins


def _hg38_alids(
    values: np.ndarray, bits: np.ndarray, hashed: np.ndarray, merged: DistinctKeys
) -> list[str | None]:
    """Canonical ALIDs for the hg38-declared keys (``None`` for hg19 keys)."""
    alids: list[str | None] = [None] * len(values)
    hg38 = bits == HG38
    packed_positions = np.flatnonzero(hg38 & ~hashed)
    if len(packed_positions):
        packed = packed_alids(values[packed_positions])
        for position, alid in zip(packed_positions.tolist(), packed, strict=True):
            alids[position] = alid
    hashed_positions = np.flatnonzero(hg38 & hashed)
    raws = _hashed_raw_list(values[hashed_positions], merged)
    for position, raw in zip(hashed_positions.tolist(), raws, strict=True):
        alids[position] = _canonical_source_key(raw)
    return alids


def _lift_hg19(
    values: np.ndarray,
    is_hg19: np.ndarray,
    hashed: np.ndarray,
    merged: DistinctKeys,
    alids: list[str | None],
    *,
    liftover_failure_threshold: float,
    chain_file: str | Path | None,
) -> None:
    """Fill the hg19-declared keys' ALIDs in place, dropping failed lifts."""
    packed_positions = np.flatnonzero(is_hg19 & ~hashed)
    hashed_positions = np.flatnonzero(is_hg19 & hashed)
    tuples: list[tuple[str, int, str, str]] = []
    if len(packed_positions):
        chroms, positions, refs, alts = packed_fields(values[packed_positions])
        tuples.extend(zip(chroms, positions, refs, alts, strict=True))
    tuples.extend(
        _split_source_key(raw) for raw in _hashed_raw_list(values[hashed_positions], merged)
    )
    if not tuples:
        return
    lifted = build_liftover_lookup(
        tuples,
        from_build="hg19",
        to_build="hg38",
        failure_threshold=liftover_failure_threshold,
        chain_file=chain_file,
    )
    positions = np.concatenate([packed_positions, hashed_positions])
    for position, tup in zip(positions.tolist(), tuples, strict=True):
        alids[position] = lifted.get(tup)


def _split_source_key(key: str) -> tuple[str, int, str, str]:
    chrom, position, ref, alt = key.split(":")
    return chrom, int(position), ref, alt


def resolve_keys(
    merged: DistinctKeys,
    *,
    liftover_failure_threshold: float,
    chain_file: str | Path | None,
) -> ResolvedKeys:
    """Resolve build-wide distinct keys to hg38 ALIDs, once per distinct key.

    A key declared under both assemblies has no single locus and is dropped; an
    hg19 key that fails liftover is dropped with its associations, exactly as
    the dict-based resolution did. Packed SNVs resolve from their bits; only the
    hashed keys touch their side-file string.
    """
    bits = merged.assembly_bits
    keep = (bits == HG19) | (bits == HG38)
    values, bits = merged.values[keep], bits[keep]
    if len(values) == 0:
        return ResolvedKeys(values, [], [])
    hashed = is_hashed(values)
    alids = _hg38_alids(values, bits, hashed, merged)
    is_hg19 = bits == HG19
    if bool(is_hg19.any()):
        _lift_hg19(
            values,
            is_hg19,
            hashed,
            merged,
            alids,
            liftover_failure_threshold=liftover_failure_threshold,
            chain_file=chain_file,
        )
    origins = _origins(values, alids, is_hg19, hashed, merged)
    return _drop_unresolved(values, alids, origins)


def _drop_unresolved(
    values: np.ndarray, alids: list[str | None], origins: list[str | None]
) -> ResolvedKeys:
    """Keep the keys that resolved; ``alids``/``origins`` are reused, not copied,
    when every key did (the usual case -- they are tens of millions long)."""
    resolved = np.array([alid is not None for alid in alids], dtype=bool)
    if bool(resolved.all()) and all(origin is not None for origin in origins):
        return ResolvedKeys(
            keys=values,
            alids=cast(list[str], alids),
            origins=cast(list[str], origins),
        )
    kept = np.flatnonzero(resolved)
    resolved_alids: list[str] = []
    resolved_origins: list[str] = []
    for position in kept.tolist():
        alid = alids[position]
        origin = origins[position]
        assert alid is not None  # narrowed by `resolved` above
        assert origin is not None
        resolved_alids.append(alid)
        resolved_origins.append(origin)
    return ResolvedKeys(
        keys=values[kept],
        alids=resolved_alids,
        origins=resolved_origins,
    )
