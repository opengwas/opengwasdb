"""Unit tests for the sorted off-reference key table (ticket #222).

The table replaces two per-association Python dicts with a sorted ``uint64``
key array; these tests pin the vectorised merge's assembly OR, its build-wide
hash guarantee, the lookup the fold uses, and the assembly/liftover rules that
decide which keys resolve. They exercise the module directly because that is
where the reduction order lives.
"""

from __future__ import annotations

import weakref
from collections.abc import Iterator, Sequence

import numpy as np
import pytest

from opengwasdb.layouts.hybrid import key_table
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


def _random_chunk(rng: np.random.Generator, universe: int, size: int) -> ChunkKeys:
    """Distinct keys drawn from a small universe so chunks overlap heavily,
    with a hashed subset whose raw key is a function of its value (no collision)."""
    values = np.unique(rng.integers(0, universe, size=size).astype(np.uint64))
    hashed_values = values[values % 5 == 0] | np.uint64(HASH_TAG)
    values = np.unique(np.concatenate([values, hashed_values]))
    return ChunkKeys(
        values=values,
        assembly_bits=rng.choice([HG19, HG38], size=len(values)).astype(np.int8),
        hashed_values=hashed_values,
        hashed_raw=[f"raw-{int(value)}" for value in hashed_values.tolist()],
    )


def test_merge_key_stream_matches_the_all_at_once_merge() -> None:
    """The incremental fold must be the same reduction as one big merge: same
    distinct keys, same OR of assemblies, same hashed raw keys."""
    rng = np.random.default_rng(222)
    chunks = [_random_chunk(rng, 400, 150) for _ in range(13)]
    expected = merge_chunks(chunks)
    assert int((expected.assembly_bits == (HG19 | HG38)).sum()) > 0, (
        "the fixture must hold keys declared on both assemblies across chunks"
    )
    assert len(expected.hashed_values) > 0
    streamed = key_table.merge_key_stream(iter(chunks))
    np.testing.assert_array_equal(streamed.values, expected.values)
    np.testing.assert_array_equal(streamed.assembly_bits, expected.assembly_bits)
    np.testing.assert_array_equal(streamed.hashed_values, expected.hashed_values)
    assert streamed.hashed_raw == expected.hashed_raw


def test_merge_key_stream_of_nothing_is_empty() -> None:
    merged = key_table.merge_key_stream(iter([]))
    assert len(merged.values) == 0
    assert merged.hashed_raw == []


def test_merge_key_stream_detects_a_hash_collision_across_distant_chunks() -> None:
    value = int(HASH_TAG | 7)
    filler = [_packed(f"1:{position}:A:G", HG38) for position in range(10, 16)]
    with pytest.raises(UnknownKeyEncodingError, match="hash collision"):
        key_table.merge_key_stream(
            iter([_hashed("1:5:A:AT", value), *filler, _hashed("1:6:A:GA", value)])
        )


def _tracked(chunks: list[ChunkKeys], live_counts: list[int]) -> Iterator[ChunkKeys]:
    """Yield ``chunks``, recording before each how many earlier ones survive.

    Only the stream holds a strong reference to a chunk after yielding it, so a
    surviving earlier chunk is one the consumer is still retaining.
    """
    alive: list[weakref.ref[np.ndarray]] = []
    while chunks:
        live_counts.append(sum(ref() is not None for ref in alive))
        chunk = chunks.pop(0)
        alive.append(weakref.ref(chunk.values))
        yield chunk
        del chunk


def test_merge_key_stream_releases_each_chunk_once_folded() -> None:
    """Review round 1 blocker: ``merge_chunks(list(stream))`` retained every
    worker result at once. The fold may keep at most one unmerged chunk (the
    binary counter's lowest level) while the next arrives."""
    rng = np.random.default_rng(7)
    chunks = [_random_chunk(rng, 10_000, 2_000) for _ in range(32)]
    live_counts: list[int] = []
    merged = key_table.merge_key_stream(_tracked(chunks, live_counts))
    assert len(live_counts) == 32, "the fixture must stream every chunk"
    assert len(merged.values) > 2_000
    assert max(live_counts) <= 1, f"earlier chunks retained: {live_counts}"


def _length_hashes(strings: Sequence[str]) -> np.ndarray:
    """A deliberately colliding hash: every string of one length shares it."""
    return np.array([len(value) for value in strings], dtype=np.uint64)


def test_alid_index_returns_axis_positions_in_query_order() -> None:
    axis = ["1:5:A:G", "1:9:C:T", "2:1:A:C", "X:3:G:T"]
    index = key_table.AlidIndex(axis)
    np.testing.assert_array_equal(
        index.lookup(["X:3:G:T", "1:5:A:G", "2:1:A:C"]), [3, 0, 2]
    )
    assert index.lookup([]).tolist() == []


def test_alid_index_refuses_an_absent_alid() -> None:
    with pytest.raises(UnknownKeyEncodingError, match="1:7:A:G"):
        key_table.AlidIndex(["1:5:A:G", "1:9:C:T"]).lookup(["1:5:A:G", "1:7:A:G"])


def test_alid_index_refuses_an_absent_alid_whose_hash_matches_the_axis(monkeypatch) -> None:
    """A hash hit alone must not be trusted: an ALID off the axis that happens
    to share another's hash would otherwise silently take that ALID's index."""
    monkeypatch.setattr(key_table, "_string_hashes", _length_hashes)
    index = key_table.AlidIndex(["1:5:A:G", "10:5:A:G"])
    assert index.lookup(["10:5:A:G"]).tolist() == [1], "the fixture's hashes must be distinct"
    with pytest.raises(UnknownKeyEncodingError, match="11:5:A:G"):
        index.lookup(["11:5:A:G"])


def test_alid_index_resolves_axis_alids_that_share_a_hash(monkeypatch) -> None:
    monkeypatch.setattr(key_table, "_string_hashes", _length_hashes)
    axis = ["1:5:A:G", "1:9:C:T", "10:5:A:G"]
    np.testing.assert_array_equal(
        key_table.AlidIndex(axis).lookup(["1:9:C:T", "10:5:A:G", "1:5:A:G"]), [1, 2, 0]
    )
