"""ogld#42: `rows_of` reads the ALID index, and answers exactly what it answered before.

The oracle is the store's own variant table, read here into a dictionary: the test compares
`rows_of` with that, never with a second search. What is asked: every ALID of the Test Store,
an ALID it does not hold, a malformed one, and (on the crafted store) one 33 bytes wide. The
index is also broken on purpose — a build that truncates, one whose entries are in no order,
and one missing an entry — because a reader that trusts it answers another variant's row, or
none, and the answers must not move.
"""

from __future__ import annotations

import tracemalloc
from pathlib import Path

import numpy as np
import zarr
from store_copies import copy_store

from ogquery.ld.store import StoreReader
from opengwasdb.ld.store import layout


def oracle(store: Path) -> dict[str, int]:
    """Each ALID the store holds, by its Variant Set row, from its variant table alone."""
    recorded = np.asarray(zarr.open_group(store, mode="r")["variants/alid"][:]).tolist()
    return {str(alid): row for row, alid in enumerate(recorded)}


# ------------------------------------------------------------------ what a lookup holds


DIGITS = 12


def fabricated_keys(entries: int) -> tuple[np.ndarray, np.ndarray]:
    """`entries` keys of a fixed shape, ascending, and the rows they name.

    A key is `22:<12 digits, zero padded>:A:C`, NUL padded to the slot: zero padding makes
    lexicographic order numeric order, so the keys are built a column at a time and sorted already.
    """
    keys = np.zeros((entries, layout.ALID_INDEX_WIDTH), dtype=np.uint8)
    keys[:, 0:3] = np.frombuffer(b"22:", dtype=np.uint8)
    numbers = np.arange(entries, dtype=np.int64)
    for at in range(DIGITS):
        keys[:, 3 + at] = ord("0") + (numbers // 10 ** (DIGITS - 1 - at)) % 10
    keys[:, 3 + DIGITS] = ord(":")
    keys[:, 4 + DIGITS : 7 + DIGITS] = np.frombuffer(b"A:C", dtype=np.uint8)
    return keys, np.arange(entries, dtype=np.int32) % 1_000


def peak_lookup(tmp_path: Path, name: str, entries: int) -> tuple[int, int]:
    """Peak traced bytes while a lookup searches an index of `entries` keys, and the index's size.

    The store's own index is replaced by `entries` keys, at a store's own chunk and shard shapes,
    and the reader's cache is set to nothing, so what is traced is the lookup's transient and not
    the chunks it keeps: the same measurement the reviews made, which peaked at 548 MB before the
    search read a shard at a time, and at 50 MB when it kept the previous shard alive.
    """
    store = copy_store(tmp_path / name)
    keys, rows = fabricated_keys(entries)
    root = zarr.open_group(store, mode="r+")
    group = root[layout.ALID_INDEX]
    assert isinstance(group, zarr.Group)
    del group["alid"]
    del group["row"]
    layout.put_numbers(group, "alid", keys.reshape(-1), layout.ALID_KEY_FILL, layout.ALID_KEYS)
    layout.put_numbers(group, "row", rows, layout.ALID_ROW_FILL, layout.ALID_ROWS)
    reader = StoreReader.open(store, cache_bytes=0)
    index = reader._variant_table._index()
    assert int(index.alid.chunks[0]) == layout.ALID_KEYS[0], "a store's inner chunks"
    if entries * layout.ALID_INDEX_WIDTH > layout.ALID_KEYS[1]:
        assert int(index.alid.shards[0]) == layout.ALID_KEYS[1], "a store's shards"
    present = [f"22:{i:0{DIGITS}d}:A:C" for i in range(0, entries, entries // 100)]
    tracemalloc.start()
    try:
        hits = reader._variant_table._searched(layout.alid_keys(present)[0])
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert (hits >= 0).all(), "every key it was given is in the index"
    return peak, keys.nbytes


def test_a_long_lookup_holds_one_shard_of_keys_and_not_the_index(tmp_path: Path) -> None:
    # Three times the keys, the same transient: the index is read a shard at a time, in key order,
    # and the shard just read is let go before the next is fetched, so nothing a lookup holds grows
    # with the store (ogld#42's reviews measured a whole read of EUR's keys held twice, 548 MB, and
    # a loop that kept the previous shard alive, 50.4 MB at EUR's shapes).
    # Both are more than a shard, so the two measurements have the same shard: what differs is the
    # number of them.
    short, _ = peak_lookup(tmp_path, "short", 600_000)
    long, size = peak_lookup(tmp_path, "long", 1_100_000)
    shard = layout.ALID_KEYS[1]
    assert size > 2 * shard, "an index of three shards"
    # Two shards of keys — the one being decoded and the one being joined — plus 8 MiB, half a
    # shard, of slack for the queries, the rows found and the allocator. A bound any looser could
    # hide a third shard, which is what the loop kept alive before fix round 3.
    bound = 2 * shard + (8 << 20)
    assert long < bound, f"{long:,} bytes held, for an index of {size:,} in shard {shard:,}"
    # The index is 16 MB larger; what the lookup holds grew by less than a quarter of that, where
    # holding the table itself would have grown by all of it, twice.
    assert long - short < size // 4, (short, long)
