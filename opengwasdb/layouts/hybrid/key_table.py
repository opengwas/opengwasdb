"""Sorted global off-reference key table (ticket #222).

Pass 2 discovers variants the variant reference never named, one column per
Analysis. Resolving them to a shared Variant Index used to build two global
Python dicts keyed by the raw source coordinate -- one insert per association,
about 15 billion inserts on OGS-00011 -- and then look every association up in
them one at a time. This module replaces both dicts with a sorted ``uint64`` key
array (the encoding from #218) aligned with each key's final shared Variant
Index.

Workers reduce a chunk of columns to sorted distinct keys, each carrying the
assemblies that declared it; the parent merges those with a vectorised sort and
reduction, never a per-key Python loop. Liftover and canonicalisation then run
once per *distinct* key, and the ``.unk`` → ``.ovf`` fold binary-searches the
table rather than walking a dict.

The build-wide hash guarantee from #218 survives the rewrite: merging still
compares every hashed key's raw string, so two distinct keys sharing one hash
fail the build naming both, wherever in the manifest they sit.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from opengwasdb.build.liftover import build_liftover_lookup
from opengwasdb.layouts.hybrid.unknown_keys import (
    UnknownKeyEncodingError,
    is_hashed,
    packed_alids,
    packed_fields,
)

__all__ = [
    "HG19",
    "HG38",
    "ChunkKeys",
    "KeyTable",
    "ResolvedKeys",
    "merge_chunks",
    "resolve_keys",
]

# One bit per source assembly, OR-ed when a key is declared more than once.
# A key declared under both has no single physical locus and is dropped.
HG19 = 1
HG38 = 2


@dataclass(frozen=True)
class ChunkKeys:
    """One worker's distinct off-reference keys, after reducing its columns.

    ``values`` is sorted ascending; ``assembly_bits`` parallels it.
    ``hashed_values``/``hashed_raw`` are its hash-region subset, carried so the
    parent can resolve and collision-check them without re-reading a spill.
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


def _run_starts(sorted_values: np.ndarray) -> np.ndarray:
    """A bool mask marking the first element of each equal-value run."""
    starts = np.empty(len(sorted_values), dtype=bool)
    if len(sorted_values) == 0:
        return starts
    starts[0] = True
    np.not_equal(sorted_values[1:], sorted_values[:-1], out=starts[1:])
    return starts


def _merge_values(chunks: Sequence[ChunkKeys]) -> tuple[np.ndarray, np.ndarray]:
    """Distinct values with the OR of every declaring assembly's bit."""
    if not chunks:
        return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.int8)
    values = np.concatenate([chunk.values for chunk in chunks])
    bits = np.concatenate([chunk.assembly_bits for chunk in chunks])
    order = np.argsort(values, kind="stable")
    values, bits = values[order], bits[order]
    starts = _run_starts(values)
    return values[starts], np.bitwise_or.reduceat(bits, np.flatnonzero(starts))


def _hashed_collision(values: np.ndarray, raw: np.ndarray, mismatch: np.ndarray) -> str:
    """Name both raw keys of the first hashed-value collision found."""
    bad = int(np.flatnonzero(mismatch)[0])
    start = bad
    while start > 0 and values[start - 1] == values[bad]:
        start -= 1
    return (
        f"hash collision between off-reference keys {raw[start]!r} and {raw[bad]!r} "
        f"(both encode to {int(values[bad])}); refusing to merge them"
    )


def _merge_hashed(chunks: Sequence[ChunkKeys]) -> tuple[np.ndarray, list[str]]:
    """Distinct hashed values and their raw keys, refusing a collision.

    A hash is a function of the raw string alone, so two distinct raw keys that
    hash to one value would silently become one variant. Comparing the raw keys
    of a value's occurrences is the only way to tell the two apart; this is the
    build-wide half of the per-column check ``encode_keys`` already makes.
    """
    if not chunks:
        return np.empty(0, dtype=np.uint64), []
    values = np.concatenate([chunk.hashed_values for chunk in chunks])
    raw_list = list(itertools.chain.from_iterable(chunk.hashed_raw for chunk in chunks))
    if len(values) == 0:
        return values, []
    order = np.argsort(values, kind="stable")
    values = values[order]
    raw = np.asarray(raw_list, dtype=object)[order]
    starts = _run_starts(values)
    first = raw[starts]
    mismatch = raw != first[np.cumsum(starts) - 1]
    if bool(mismatch.any()):
        raise UnknownKeyEncodingError(_hashed_collision(values, raw, mismatch))
    return values[starts], first.tolist()


def merge_chunks(chunks: Sequence[ChunkKeys]) -> ChunkKeys:
    """Merge sorted per-chunk distinct keys into one build-wide set.

    Vectorised: ``np.argsort`` + ``reduceat`` for the values and assemblies,
    and a run comparison for the hashed collision check. Nothing here loops
    per key in Python.
    """
    values, bits = _merge_values(chunks)
    hashed_values, hashed_raw = _merge_hashed(chunks)
    return ChunkKeys(
        values=values,
        assembly_bits=bits,
        hashed_values=hashed_values,
        hashed_raw=hashed_raw,
    )


def _canonical_source_key(key: str) -> str:
    """The ALID a source key names before any liftover (alleles sorted)."""
    chrom, position, ref, alt = key.split(":")
    a1, a2 = sorted((ref, alt))
    return f"{chrom}:{int(position)}:{a1}:{a2}"


def _hashed_raw_list(queries: np.ndarray, merged: ChunkKeys) -> list[str]:
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
    merged: ChunkKeys,
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
    values: np.ndarray, bits: np.ndarray, hashed: np.ndarray, merged: ChunkKeys
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
    merged: ChunkKeys,
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
    merged: ChunkKeys,
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
    resolved = np.array([alid is not None for alid in alids], dtype=bool)
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
