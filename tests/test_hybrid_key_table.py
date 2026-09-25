"""Unit tests for the sorted off-reference key table (ticket #222).

The table replaces two per-association Python dicts with a sorted ``uint64``
key array; these tests pin the lookup the fold uses, the assembly/liftover
rules that decide which distinct keys resolve, and the ALID index that maps
them onto the shared axis without a dict. The build-wide merge that produces
the distinct keys is ``test_hybrid_key_runs``'.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from opengwasdb.layouts.hybrid import key_table
from opengwasdb.layouts.hybrid.key_table import (
    HG19,
    HG38,
    DistinctKeys,
    KeyTable,
    resolve_keys,
)
from opengwasdb.layouts.hybrid.unknown_keys import (
    UnknownKeyEncodingError,
    encode_key,
    is_hashed,
)


def _distinct(*declared: tuple[str, int]) -> DistinctKeys:
    """Distinct keys from ``(raw key, assembly bits)`` pairs, as the merge
    would hand them to ``resolve_keys``: sorted, hashed raw keys attached."""
    by_value = {encode_key(raw): (raw, bits) for raw, bits in declared}
    values = np.array(sorted(by_value), dtype=np.uint64)
    hashed = values[is_hashed(values)]
    return DistinctKeys(
        values=values,
        assembly_bits=np.array([by_value[v][1] for v in values.tolist()], dtype=np.int8),
        hashed_values=hashed,
        hashed_raw=np.array(
            [by_value[v][0] for v in hashed.tolist()], dtype=key_table.RAW_KEY_DTYPE
        ),
    )


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
    resolved = resolve_keys(
        _distinct(("1:5:A:G", HG19 | HG38), ("1:6:A:G", HG38)),
        liftover_failure_threshold=1.0,
        chain_file=None,
    )
    assert resolved.alids == ["1:6:A:G"], "only the single-assembly key may resolve"


def test_resolve_keys_canonicalises_an_hg38_packed_snv() -> None:
    resolved = resolve_keys(
        _distinct(("1:5:T:A", HG38)), liftover_failure_threshold=1.0, chain_file=None
    )
    assert resolved.alids == ["1:5:A:T"]
    assert resolved.origins == ["1:5:A:T"]


def test_resolve_keys_canonicalises_an_hg38_hashed_key_from_its_raw_string() -> None:
    distinct = _distinct(("1:5:CT:C", HG38))
    assert len(distinct.hashed_values) == 1, "the fixture must be a hashed key"
    resolved = resolve_keys(distinct, liftover_failure_threshold=1.0, chain_file=None)
    assert resolved.alids == ["1:5:C:CT"]


def test_resolve_keys_keeps_two_hg38_keys_on_one_alid() -> None:
    """A:G and G:A are two raw keys naming one physical SNV; both resolve, both
    carry the same ALID, and the provenance collision is left for the caller."""
    resolved = resolve_keys(
        _distinct(("1:5:A:G", HG38), ("1:5:G:A", HG38)),
        liftover_failure_threshold=1.0,
        chain_file=None,
    )
    assert len(resolved.keys) == 2
    assert resolved.alids == ["1:5:A:G", "1:5:A:G"]


def test_canonical_raw_keys_verify_detects_hash_collision_naming_both_keys() -> None:
    canonical = key_table.CanonicalRawKeys(
        values=np.array([100, 200], dtype=np.uint64),
        raw=np.array(["1:100:A:AT", "1:200:C:CT"], dtype=key_table.RAW_KEY_DTYPE),
    )
    # Matching keys pass
    canonical.verify(
        column=0,
        values=np.array([100, 200], dtype=np.uint64),
        raws=np.array(["1:100:A:AT", "1:200:C:CT"], dtype=key_table.RAW_KEY_DTYPE),
    )
    # Colliding raw key on value 200 fails naming both
    with pytest.raises(UnknownKeyEncodingError) as excinfo:
        canonical.verify(
            column=1,
            values=np.array([100, 200], dtype=np.uint64),
            raws=np.array(["1:100:A:AT", "1:200:C:CA"], dtype=key_table.RAW_KEY_DTYPE),
        )
    assert "'1:200:C:CT'" in str(excinfo.value)
    assert "'1:200:C:CA'" in str(excinfo.value)
    assert "200" in str(excinfo.value)


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
    index = key_table.AlidIndex(axis)
    np.testing.assert_array_equal(
        index.lookup(["1:9:C:T", "10:5:A:G", "1:5:A:G"]), [1, 2, 0]
    )
    assert index.n_bucketed == 2


def test_alid_index_collision_allocates_only_colliding_buckets_never_whole_axis(
    monkeypatch,
) -> None:
    """Review round 2 major: AlidIndex must handle only colliding buckets,
    never allocate a whole-axis dict."""
    axis = [f"1:{i}:A:G" for i in range(50)]

    def _mock_hashes(strings: Sequence[str]) -> np.ndarray:
        hashes = []
        for s in strings:
            if s in ("1:3:A:G", "1:7:A:G"):
                hashes.append(42)
            else:
                hashes.append(1000 + int(s.split(":")[1]))
        return np.array(hashes, dtype=np.uint64)

    monkeypatch.setattr(key_table, "_string_hashes", _mock_hashes)
    index = key_table.AlidIndex(axis)
    assert index.n_bucketed == 2
    assert not hasattr(index, "_fallback")
    assert set(index._bucket_members.keys()) == {"1:3:A:G", "1:7:A:G"}
    queries = ["1:7:A:G", "1:0:A:G", "1:3:A:G", "1:49:A:G"]
    np.testing.assert_array_equal(index.lookup(queries), [7, 0, 3, 49])

    def _mock_hashes_with_absent(strings: Sequence[str]) -> np.ndarray:
        hashes = []
        for s in strings:
            if s in ("1:3:A:G", "1:7:A:G", "1:absent:A:G"):
                hashes.append(42)
            else:
                hashes.append(1000 + int(s.split(":")[1]))
        return np.array(hashes, dtype=np.uint64)

    monkeypatch.setattr(key_table, "_string_hashes", _mock_hashes_with_absent)
    with pytest.raises(UnknownKeyEncodingError, match="1:absent:A:G"):
        index.lookup(["1:absent:A:G"])


def test_drop_unresolved_keeps_only_resolved_keys_in_step() -> None:
    values = np.array([1, 2, 3, 4], dtype=np.uint64)
    resolved = key_table._drop_unresolved(
        values, ["a", None, "c", "d"], ["oa", None, "oc", "od"]
    )
    assert resolved.keys.tolist() == [1, 3, 4]
    assert resolved.alids == ["a", "c", "d"]
    assert resolved.origins == ["oa", "oc", "od"]


def test_drop_unresolved_reuses_the_lists_when_every_key_resolved() -> None:
    """The lists are tens of millions long at OGS-00011 scale; a copy when
    nothing is dropped doubles the resolve peak (ticket #222)."""
    alids: list[str | None] = ["a", "b"]
    origins: list[str | None] = ["oa", "ob"]
    resolved = key_table._drop_unresolved(np.array([1, 2], dtype=np.uint64), alids, origins)
    assert resolved.alids is alids
    assert resolved.origins is origins
