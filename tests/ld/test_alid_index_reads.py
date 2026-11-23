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
import pytest
import zarr
from store_copies import TEST_STORE, copy_store

from opengwasdb.ld.store import StoreReader, layout
from opengwasdb.ld.store.variants import _index_pays


def oracle(store: Path) -> dict[str, int]:
    """Each ALID the store holds, by its Variant Set row, from its variant table alone."""
    recorded = np.asarray(zarr.open_group(store, mode="r")["variants/alid"][:]).tolist()
    return {str(alid): row for row, alid in enumerate(recorded)}


def test_every_alid_of_the_test_store_is_its_own_row() -> None:
    rows = oracle(TEST_STORE)
    found = StoreReader.open(TEST_STORE).rows_of(list(rows))
    assert found.tolist() == [rows[alid] for alid in rows], "every ALID, in table order"


def test_a_list_in_any_order_and_with_repeats_reads_the_same_rows() -> None:
    rows = oracle(TEST_STORE)
    some = [*sorted(rows)[::37], "nothing", "22:1:A:G", ""]
    asked = list(reversed(some)) + some[:3]
    found = StoreReader.open(TEST_STORE).rows_of(asked)
    assert found.tolist() == [rows.get(alid, -1) for alid in asked]


def test_an_absent_a_malformed_and_a_wider_alid_are_not_this_store_s_rows() -> None:
    reader = StoreReader.open(TEST_STORE)
    wide = [
        alid for alid in oracle(TEST_STORE) if len(alid.encode("utf-8")) > layout.ALID_INDEX_WIDTH
    ]
    assert len(wide) == 8, "the Test Store holds ALIDs wider than the slot"
    absent = "22:16050075:A:T"
    assert absent not in oracle(TEST_STORE)
    found = reader.rows_of([absent, "nonsense", "22:100:a:G", *wide])
    assert found.tolist() == [-1, -1, -1, *[oracle(TEST_STORE)[alid] for alid in wide]]
    # The store's ALIDs wider than the slot are found, by the exact scan, at their position.
    assert (found[3:] >= 0).all()


def test_an_alid_at_the_slot_s_width_is_found(tmp_path: Path) -> None:
    # A slot-width ALID is indexed like any other: `alid_index.keys` refuses an index whose key is
    # not its row's full ALID, so a truncating build is a store validation fails, not one the
    # reader second-guesses (ADR 0096).
    store = copy_store(tmp_path / "store")
    rows = oracle(store)
    alid = next(a for a in rows if len(a.encode("utf-8")) == layout.ALID_INDEX_WIDTH)
    assert StoreReader.open(store).rows_of([alid]).tolist() == [rows[alid]]


def test_an_index_missing_an_entry_falls_back_to_the_scan(tmp_path: Path) -> None:
    # Its key replaced by one no ALID has, so the entry answers nothing: the ALID it named is
    # found by the exact scan at its position, as it was before the index.
    store = copy_store(tmp_path / "store")
    rows = oracle(store)
    root = zarr.open_group(store, mode="r+")
    keys = np.asarray(root[f"{layout.ALID_INDEX}/alid"][:]).reshape(-1, layout.ALID_INDEX_WIDTH)
    entries = np.asarray(root[f"{layout.ALID_INDEX}/row"][:])
    gone = sorted(rows)[40]
    assert len(gone.encode("utf-8")) < layout.ALID_INDEX_WIDTH, "a slot-width ALID is indexed too"
    lost = np.flatnonzero(entries == rows[gone])
    assert len(lost) == 1
    keys[lost] = np.frombuffer(b"22:99999999:A:C".ljust(32, b"\0"), dtype=np.uint8)
    root[f"{layout.ALID_INDEX}/alid"][:] = keys.ravel()
    found = StoreReader.open(store).rows_of(sorted(rows)[:60])
    assert found.tolist() == [rows[alid] for alid in sorted(rows)[:60]]


def test_an_unsorted_index_answers_the_same_rows(tmp_path: Path) -> None:
    store = copy_store(tmp_path / "store")
    rows = oracle(store)
    root = zarr.open_group(store, mode="r+")
    keys = np.asarray(root[f"{layout.ALID_INDEX}/alid"][:]).reshape(-1, layout.ALID_INDEX_WIDTH)
    entries = np.asarray(root[f"{layout.ALID_INDEX}/row"][:])
    order = np.random.default_rng(7).permutation(len(entries))
    root[f"{layout.ALID_INDEX}/alid"][:] = keys[order].ravel()
    root[f"{layout.ALID_INDEX}/row"][:] = entries[order]
    asked = [*sorted(rows)[:100], "22:16050075:A:T"]
    assert StoreReader.open(store).rows_of(asked).tolist() == [rows.get(a, -1) for a in asked]


def test_a_truncated_index_answers_the_same_rows(tmp_path: Path) -> None:
    store = copy_store(tmp_path / "store")
    rows = oracle(store)
    root = zarr.open_group(store, mode="r+")
    alids = sorted(rows)
    keys = np.array(alids, dtype=f"|S{layout.ALID_INDEX_WIDTH}")
    root[f"{layout.ALID_INDEX}/alid"][:] = keys.view(np.uint8).ravel()
    root[f"{layout.ALID_INDEX}/row"][:] = np.array([rows[a] for a in alids], dtype=np.int32)
    asked = [*alids[:100], "22:16050075:A:T"]
    assert StoreReader.open(store).rows_of(asked).tolist() == [rows.get(a, -1) for a in asked]


@pytest.mark.parametrize("alids", [[], ["nonsense"]])
def test_a_list_with_nothing_to_look_up_reads_no_index(tmp_path: Path, alids: list[str]) -> None:
    store = copy_store(tmp_path / "store")
    for chunk in (store / layout.ALID_INDEX / "alid").rglob("c/*"):
        chunk.unlink()
    reader = StoreReader.open(store)
    assert reader.rows_of(alids).tolist() == [-1] * len(alids)


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


def test_the_index_is_read_only_for_a_list_that_pays_for_it() -> None:
    # The crossover is a list naming a `_INDEX_SHARE`th of the Variant Set's entries or more, and
    # never fewer than one; the edge is a literal, not the constant that derives it (ogld#42's
    # review: 4,000 of EUR's 8.59 M is where the measurement puts it).
    assert not _index_pays(1, 4_300), "one ALID is one fewer than the threshold of 4,300 entries"
    assert _index_pays(2, 4_300), "two ALIDs are the threshold of 4,300 entries"
    # A store small enough that one ALID is a share of it asks for the index at one.
    assert _index_pays(1, 1_000)
    assert _index_pays(3_975, 8_546_809), "EUR's own threshold"
    assert not _index_pays(3_974, 8_546_809)
