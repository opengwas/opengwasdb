"""One column's off-reference keys encoded one at a time: the oracle `encode_fields`
is held to (ticket #218, #318).

A Hybrid build's Pass 2 encodes a batch's fields directly (`encode_fields`) and never
forms a Routing Key per row. These tests still want the slow, obvious answer beside it
-- each key through `encode_key`, a hashed key named in the side file, and two keys
that share a hash refused -- so they hold it here rather than in the package, which
has no caller for it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from opengwasdb.layouts.hybrid.unknown_keys import (
    HASH_TAG,
    EncodedKeys,
    HashFn,
    UnknownKeyEncodingError,
    encode_key,
)


def encode_keys(keys: Sequence[str], *, hash_fn: HashFn | None = None) -> EncodedKeys:
    """Encode one column's raw keys, refusing any hash collision.

    The collision check is per column because the spill resolves a column's
    repeated values as one association's duplicates (ADR 0072, D5): two
    distinct raw keys sharing a value would otherwise be judged as one
    association without a trace.
    """
    values = np.empty(len(keys), dtype=np.uint64)
    hashed_index: list[int] = []
    hashed_raw: list[str] = []
    seen: dict[int, str] = {}
    for index, key in enumerate(keys):
        value = encode_key(key, hash_fn=hash_fn)
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
