"""Unit tests for the uint64 off-reference key encoding (ticket #218).

The encoding is the whole point of the ticket: a lossless packed SNV, a tagged
hash for everything else, and a raw-key side file only for the hashed keys.
These tests pin the round-trip, the tag separation, and the two loud refusals
(collision, out-of-range hash) the spill format depends on.
"""

from __future__ import annotations

import numpy as np
import pytest

from opengwasdb.layouts.hybrid import unknown_keys
from opengwasdb.layouts.hybrid.build import _read_unknown_side_file, _spill_hybrid_column
from opengwasdb.layouts.hybrid.unknown_keys import (
    HASH_TAG,
    MAX_POSITION,
    UnknownKeyEncodingError,
    decode_keys,
    encode_key,
    encode_keys,
    hashed_lookup,
    is_hashed,
    unpack_key,
)

CANONICAL_CHROMOSOMES = [str(number) for number in range(1, 23)] + ["X", "Y", "MT"]

# Keys that cannot be packed: indels, multi-base or unusual alleles, an
# aliased/lower-case chromosome, a position outside the 28-bit field, and a
# contig label outside the canonical set.
NON_PACKABLE_KEYS = [
    "1:100:A:AT",
    "1:100:A:ATCG",
    "1:100:a:G",
    "chr1:100:A:G",
    "1:100:A:C:G",
    "1:100:.:G",
    "1:0:A:G",
    f"1:{MAX_POSITION + 1}:A:G",
    "KI270728.1:5:A:G",
]


@pytest.mark.parametrize("chromosome", CANONICAL_CHROMOSOMES)
def test_packed_snv_round_trips_at_the_maximum_position(chromosome: str) -> None:
    """Every canonical #216 label packs at the largest 28-bit position, and
    decodes back to exactly the raw key the worker saw."""
    for ref, alt in (("A", "C"), ("C", "G"), ("G", "T"), ("T", "A")):
        key = f"{chromosome}:{MAX_POSITION}:{ref}:{alt}"
        value = encode_key(key)
        assert value & HASH_TAG == 0, key
        assert unpack_key(value) == key


def test_packed_snv_round_trips_at_position_one() -> None:
    """Position 1 is inside the field; position 0 is not (below)."""
    key = "1:1:A:G"
    assert unpack_key(encode_key(key)) == key


def test_packed_values_stay_below_the_hash_tag() -> None:
    keys = [f"{chromosome}:10:A:C" for chromosome in CANONICAL_CHROMOSOMES]
    values = encode_keys(keys).values
    assert bool((values < np.uint64(HASH_TAG)).all())
    assert not is_hashed(values).any()


@pytest.mark.parametrize("key", NON_PACKABLE_KEYS)
def test_non_packable_keys_are_hashed_and_round_trip(key: str) -> None:
    encoded = encode_keys([key])
    assert is_hashed(encoded.values)[0], key
    assert encode_key(key) >= HASH_TAG
    assert decode_keys(encoded.values, encoded.hashed_index, encoded.hashed_raw) == [key]


def test_mixed_column_round_trips_in_row_order() -> None:
    keys = ["1:5:A:C", "1:6:A:AT", "X:7:G:T", "1:8:AA:A"]
    encoded = encode_keys(keys)

    assert encoded.values.dtype == np.uint64
    assert encoded.hashed_index.tolist() == [1, 3]
    assert encoded.hashed_raw == ["1:6:A:AT", "1:8:AA:A"]
    assert decode_keys(encoded.values, encoded.hashed_index, encoded.hashed_raw) == keys


def test_non_ascii_position_digits_are_hashed_not_lossily_parsed() -> None:
    """``str.isdigit()`` accepts non-ASCII digits that ``int()`` parses, so
    accepting them would pack the key under a different position than the raw
    key names. They must hash and keep their raw string instead."""
    key = "1:\uff11\uff12:A:G"  # fullwidth 12
    encoded = encode_keys([key])
    assert is_hashed(encoded.values)[0]
    assert decode_keys(encoded.values, encoded.hashed_index, encoded.hashed_raw) == [key]


def test_hash_collision_is_refused_naming_both_keys(monkeypatch) -> None:
    """Two distinct hashed keys sharing a value must fail the build loudly,
    not silently merge one association into the other."""
    monkeypatch.setattr(unknown_keys, "_stable_hash", lambda key: 12345)
    with pytest.raises(UnknownKeyEncodingError) as excinfo:
        encode_keys(["1:1:A:AT", "1:1:A:GA"])

    message = str(excinfo.value)
    assert "1:1:A:AT" in message
    assert "1:1:A:GA" in message


def test_the_same_key_twice_is_not_a_collision() -> None:
    encoded = encode_keys(["1:1:A:AT", "1:1:A:AT"])
    assert decode_keys(encoded.values, encoded.hashed_index, encoded.hashed_raw) == [
        "1:1:A:AT",
        "1:1:A:AT",
    ]


@pytest.mark.parametrize("value", [1 << 63, (1 << 63) + 5, -1])
def test_hash_outside_the_63_bit_region_is_refused_not_truncated(
    monkeypatch, value: int
) -> None:
    monkeypatch.setattr(unknown_keys, "_stable_hash", lambda key: value)
    with pytest.raises(UnknownKeyEncodingError, match="outside the 63-bit hash region"):
        encode_key("1:1:A:AT")


def test_missing_side_entry_fails_loudly() -> None:
    encoded = encode_keys(["1:5:A:AT"])
    with pytest.raises(UnknownKeyEncodingError, match="side file"):
        decode_keys(encoded.values, encoded.hashed_index, [])


def test_hashed_lookup_rejects_a_short_side_file() -> None:
    encoded = encode_keys(["1:5:A:AT", "1:6:A:GA"])
    with pytest.raises(UnknownKeyEncodingError, match="hashed row"):
        hashed_lookup(encoded.values, encoded.hashed_index[:1], encoded.hashed_raw[:1])


def test_hashed_lookup_rejects_a_duplicated_row() -> None:
    encoded = encode_keys(["1:5:A:AT", "1:6:A:GA"])
    with pytest.raises(UnknownKeyEncodingError, match="hashed row"):
        hashed_lookup(encoded.values, np.array([0, 0]), encoded.hashed_raw)


def test_hashed_lookup_rejects_a_packed_row_name() -> None:
    values = np.array([encode_key("1:5:A:C"), encode_key("1:6:A:AT")], dtype=np.uint64)
    with pytest.raises(UnknownKeyEncodingError, match="hashed row"):
        hashed_lookup(values, np.array([0]), ["1:6:A:AT"])


def test_hashed_lookup_rejects_two_swapped_side_entries() -> None:
    """Swapping two genuine raw keys keeps every tagged row covered, so only
    re-deriving each key's encoding can prove the side file pairs the right key
    with the right row. Without that check the spill decodes to the two keys in
    the wrong order, attaching each row's statistics to the other variant."""
    encoded = encode_keys(["1:5:A:AT", "1:6:A:GA"])
    swapped = list(reversed(encoded.hashed_raw))

    with pytest.raises(UnknownKeyEncodingError, match="encodes to"):
        hashed_lookup(encoded.values, encoded.hashed_index, swapped)
    with pytest.raises(UnknownKeyEncodingError, match="encodes to"):
        decode_keys(encoded.values, encoded.hashed_index, swapped)


def test_unpacking_a_hashed_value_is_refused() -> None:
    with pytest.raises(UnknownKeyEncodingError, match="hashed key"):
        unpack_key(encode_key("1:5:A:AT"))


def test_spill_is_uint64_plus_float32_with_a_raw_side_file(tmp_path) -> None:
    """The on-disk spill is the new format: a uint64 key array, float32
    statistics, and a side file only for the hashed rows. It loads without
    pickle."""
    keys = ["1:5:A:C", "1:6:A:AT"]
    encoded = encode_keys(keys)
    stats = {
        "z": np.array([1.5, -2.5], dtype=np.float32),
        "se": np.array([0.1, 0.2], dtype=np.float32),
        "eaf": np.array([0.3, np.nan], dtype=np.float32),
    }
    dense = (
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.float32),
    )
    _spill_hybrid_column(
        tmp_path,
        0,
        dense,
        dense,
        (
            encoded.values,
            stats["z"],
            stats["se"],
            stats["eaf"],
            encoded.hashed_index,
            encoded.hashed_raw,
        ),
    )

    with np.load(tmp_path / "0.unk.npz") as data:  # allow_pickle defaults to False
        assert data["keys"].dtype == np.uint64
        assert data["z"].dtype == np.float32
        assert data["se"].dtype == np.float32
        assert data["eaf"].dtype == np.float32
        assert data["hashed_index"].tolist() == [1]
        assert decode_keys(
            data["keys"], data["hashed_index"], _read_unknown_side_file(tmp_path, 0)
        ) == keys
    assert (tmp_path / "0.unk.raw").read_text(encoding="utf-8") == "1:6:A:AT\n"


def test_a_fully_packed_spill_writes_no_side_file(tmp_path) -> None:
    encoded = encode_keys(["1:5:A:C"])
    dense = (
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.float32),
    )
    float_stats = np.array([1.0], dtype=np.float32)
    _spill_hybrid_column(
        tmp_path,
        0,
        dense,
        dense,
        (encoded.values, float_stats, float_stats, float_stats, encoded.hashed_index, []),
    )
    assert (tmp_path / "0.unk.npz").exists()
    assert not (tmp_path / "0.unk.raw").exists()


def test_validation_names_the_row_whose_raw_key_does_not_encode_to_it() -> None:
    """The re-encode check is one array comparison (ticket #222); the error must
    still name the offending row and key, not the first row checked."""
    raw = ["1:5:A:AT", "1:6:A:GA", "1:7:A:TA", "1:8:A:CA"]
    encoded = encode_keys(raw)
    assert encoded.hashed_index.tolist() == [0, 1, 2, 3], "every row must be hashed"
    stale = [*raw[:2], "1:99:A:TA", raw[3]]
    with pytest.raises(UnknownKeyEncodingError, match=r"row 2 names raw key '1:99:A:TA'"):
        unknown_keys.validated_hashed_values(encoded.values, encoded.hashed_index, stale)


def test_packed_alids_are_built_blockwise_without_losing_a_block(monkeypatch) -> None:
    """``packed_alids`` builds tens of millions of strings a bounded block at a
    time (ticket #222); every block, including a short last one, must land."""
    keys = [f"1:{position}:G:A" for position in range(1, 8)]
    values = np.array([encode_key(key) for key in keys], dtype=np.uint64)
    expected = [f"1:{position}:A:G" for position in range(1, 8)]
    assert unknown_keys.packed_alids(values) == expected
    monkeypatch.setattr(unknown_keys, "_ALID_BLOCK", 3)
    assert unknown_keys.packed_alids(values) == expected
