"""An off-reference row is encoded from its fields, not from a key string (#318).

`RouteTable` keeps each unmatched row's chromosome, position and alleles, and a
Hybrid off-reference spill encodes them with `encode_fields`. The spill must be
the one `encode_keys` wrote from each row's Routing Key, byte for byte.
"""

from __future__ import annotations

import numpy as np
import pytest

from opengwasdb.layouts.hybrid import unknown_keys
from opengwasdb.layouts.hybrid.unknown_keys import (
    MAX_POSITION,
    UnknownKeyEncodingError,
    encode_fields,
    encode_keys,
)
from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers.interface import ReaderAssociation
from opengwasdb.variants.routing import RouteTable
from opengwasdb.variants.routing.keys import routing_key

#: Rows meeting every way a key packs or hashes: case, indels, every canonical
#: chromosome kind, labels no packing holds (one with a colon), and positions
#: at, past and short of the packed field.
ROWS = [
    ("1", 100, "A", "G"),
    ("1", 100, "a", "g"),
    ("22", 5, "T", "c"),
    ("X", 7, "C", "A"),
    ("Y", 8, "G", "T"),
    ("MT", 9, "A", "T"),
    ("1", 200, "AT", "A"),
    ("2", 300, "a", "acgt"),
    ("GL000192.1", 10, "A", "G"),
    ("HLA-A*01:01", 11, "C", "T"),
    ("1", MAX_POSITION, "A", "C"),
    ("1", MAX_POSITION + 1, "A", "C"),
    ("1", 2**40, "G", "A"),
    ("3", 400, "ACG", "TGA"),
]


def _keys(rows: list[tuple[str, int, str, str]]) -> unknown_keys.EncodedKeys:
    return encode_keys([routing_key(row) for row in rows])


def _equal(a: unknown_keys.EncodedKeys, b: unknown_keys.EncodedKeys) -> bool:
    return (
        a.values.dtype == b.values.dtype
        and a.values.tobytes() == b.values.tobytes()
        and a.hashed_index.tolist() == b.hashed_index.tolist()
        and a.hashed_raw == b.hashed_raw
    )


def test_the_rows_both_pack_and_hash() -> None:
    encoded = _keys(ROWS)
    assert 0 < len(encoded.hashed_raw) < len(ROWS)


@pytest.mark.parametrize("as_arrays", [False, True], ids=["lists", "arrays"])
def test_fields_encode_as_their_routing_keys_do(as_arrays: bool) -> None:
    chromosome, position, ref, alt = (list(column) for column in zip(*ROWS, strict=True))
    if as_arrays:
        fields = (
            np.array(chromosome, dtype=object), np.array(position, dtype=np.int64),
            np.array(ref, dtype=object), np.array(alt, dtype=object),
        )
        assert _equal(encode_fields(*fields), _keys(ROWS))
    else:
        assert _equal(encode_fields(chromosome, position, ref, alt), _keys(ROWS))
    empty = encode_fields([], [], [], [])
    assert _equal(empty, encode_keys([]))


def test_a_hash_collision_is_refused_from_fields_as_from_keys() -> None:
    rows = [("1", 1, "A", "AT"), ("1", 1, "A", "GA")]
    with pytest.raises(UnknownKeyEncodingError, match="hash collision"):
        encode_keys([routing_key(row) for row in rows], hash_fn=lambda key: 7)
    columns = [list(column) for column in zip(*rows, strict=True)]
    with pytest.raises(UnknownKeyEncodingError, match="hash collision"):
        encode_fields(*columns, hash_fn=lambda key: 7)


def test_routing_keeps_an_unmatched_rows_fields_and_spells_its_key_on_demand() -> None:
    table = RouteTable.build({("1", 100, "A", "G"): "1:100:A:G"}, (), [["1:100:A:G"]],
                             keep_unmatched=True)
    associations = [
        ReaderAssociation(c, p, r, a, 1.0, 0.1, StoredEffectScale.SD) for c, p, r, a in ROWS
    ]
    unmatched = table.route_column(associations).unmatched
    expected = [row for row in ROWS if routing_key(row) != "1:100:A:G"]
    assert len(unmatched) == len(expected) > 10
    assert (list(unmatched.chromosome), list(unmatched.ref)) == (
        [row[0] for row in expected], [row[2] for row in expected],
    )
    assert unmatched.keys == [routing_key(row) for row in expected]
    assert _equal(
        encode_fields(unmatched.chromosome, unmatched.position, unmatched.ref, unmatched.alt),
        _keys(expected),
    )
