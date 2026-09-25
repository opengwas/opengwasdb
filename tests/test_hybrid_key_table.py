"""Unit tests for the sorted off-reference key table (ticket #222).

The table replaces two per-association Python dicts with a sorted ``uint64``
key array; these tests pin the vectorised merge's assembly OR, its build-wide
hash guarantee, the lookup the fold uses, and the assembly/liftover rules that
decide which keys resolve. They exercise the module directly because that is
where the reduction order lives.
"""

from __future__ import annotations

import numpy as np
import pytest

from opengwasdb.layouts.hybrid.key_table import (
    HG19,
    HG38,
    ChunkKeys,
    KeyTable,
    merge_chunks,
    resolve_keys,
)
from opengwasdb.layouts.hybrid.unknown_keys import (
    HASH_TAG,
    UnknownKeyEncodingError,
    encode_key,
)


def _hashed(raw: str, value: int) -> ChunkKeys:
    """One hashed key in its own chunk, with a value the caller controls."""
    return ChunkKeys(
        values=np.array([value], dtype=np.uint64),
        assembly_bits=np.array([HG19], dtype=np.int8),
        hashed_values=np.array([value], dtype=np.uint64),
        hashed_raw=[raw],
    )


def _packed(key: str, bit: int) -> ChunkKeys:
    """One packed SNV key in its own chunk, declared under ``bit``."""
    value = np.uint64(encode_key(key))
    return ChunkKeys(
        values=np.array([value], dtype=np.uint64),
        assembly_bits=np.array([bit], dtype=np.int8),
        hashed_values=np.array([], dtype=np.uint64),
        hashed_raw=[],
    )


def test_merge_chunks_ors_the_declaring_assemblies() -> None:
    """The same key in an hg19 chunk and an hg38 chunk carries both bits, which
    is what the two-assembly drop in ``resolve_keys`` keys off."""
    merged = merge_chunks([_packed("1:5:A:G", HG19), _packed("1:5:A:G", HG38)])
    assert len(merged.values) == 1, "the fixture must be one key declared twice"
    assert int(merged.assembly_bits[0]) == (HG19 | HG38)


def test_merge_chunks_repeats_one_assembly_without_widening_it() -> None:
    merged = merge_chunks([_packed("1:5:A:G", HG38), _packed("1:5:A:G", HG38)])
    assert len(merged.values) == 1
    assert int(merged.assembly_bits[0]) == HG38


def test_merge_chunks_detects_a_cross_chunk_hash_collision() -> None:
    """Two distinct raw keys forced onto one hash must fail the build-wide
    merge naming both, wherever in the manifest they sit (#218 review)."""
    value = int(HASH_TAG | 7)
    with pytest.raises(UnknownKeyEncodingError) as excinfo:
        merge_chunks([_hashed("1:5:A:AT", value), _hashed("1:6:A:GA", value)])
    message = str(excinfo.value)
    assert "1:5:A:AT" in message
    assert "1:6:A:GA" in message


def test_merge_chunks_does_not_call_a_repeated_key_a_collision() -> None:
    value = int(HASH_TAG | 7)
    merged = merge_chunks([_hashed("1:5:A:AT", value), _hashed("1:5:A:AT", value)])
    assert len(merged.hashed_values) == 1
    assert merged.hashed_raw == ["1:5:A:AT"]


def test_key_table_lookup_reports_matches_and_misses() -> None:
    """The fold trusts only ``matched``; a miss must not silently read a
    neighbour's shared index."""
    table = KeyTable(
        keys=np.array([1, 3, 5], dtype=np.uint64),
        shared_index=np.array([10, 11, 12], dtype=np.int64),
    )
    index, matched = table.lookup(np.array([3, 4, 1], dtype=np.uint64))
    np.testing.assert_array_equal(matched, [True, False, True])
    np.testing.assert_array_equal(index[matched], [11, 10])


def test_key_table_lookup_of_an_empty_table_is_all_misses() -> None:
    table = KeyTable(
        keys=np.empty(0, dtype=np.uint64), shared_index=np.empty(0, dtype=np.int64)
    )
    index, matched = table.lookup(np.array([1, 2], dtype=np.uint64))
    np.testing.assert_array_equal(matched, [False, False])
    np.testing.assert_array_equal(index, [0, 0])


def test_resolve_keys_drops_a_key_declared_on_two_assemblies() -> None:
    merged = merge_chunks([_packed("1:5:A:G", HG19), _packed("1:5:A:G", HG38)])
    resolved = resolve_keys(merged, liftover_failure_threshold=1.0, chain_file=None)
    assert len(resolved.keys) == 0


def test_resolve_keys_canonicalises_an_hg38_packed_snv() -> None:
    resolved = resolve_keys(
        _packed("1:5:T:A", HG38), liftover_failure_threshold=1.0, chain_file=None
    )
    assert resolved.alids == ["1:5:A:T"]
    assert resolved.origins == ["1:5:A:T"]


def test_resolve_keys_keeps_two_hg38_keys_on_one_alid() -> None:
    """A:G and G:A are two raw keys naming one physical SNV; both resolve, both
    carry the same ALID, and the provenance collision is left for the caller."""
    merged = merge_chunks([_packed("1:5:A:G", HG38), _packed("1:5:G:A", HG38)])
    resolved = resolve_keys(merged, liftover_failure_threshold=1.0, chain_file=None)
    assert len(resolved.keys) == 2
    assert resolved.alids == ["1:5:A:G", "1:5:A:G"]
