"""How an LD Store's Zarr arrays are laid out (spec §6): the one place that creates them.

Every array is Zarr v3 with `sharding_indexed`. The chunk and shard shapes, the codec chain
and the fill value of each kind of array are the spec's table, kept here as data so the
writer cannot create an array the table does not describe, and the reader's tests can name
the fill values it promises.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import numpy.typing as npt
import zarr
from zarr.codecs import ZstdCodec

from ogld.format import REncoding

#: (inner chunk, shard) in elements, for every array but the Near Band's `r`.
SIDE: Final = (16_384, 1_048_576)
#: The Near Band's `r`: layout C (spec §6; ADR 0006).
NEAR_R: Final = (262_144, 67_108_864)

#: The ALID search index (spec §2, §6; ADR 0010): the variant table's ALIDs in one array a search
#: can run over. A key is an ALID's UTF-8 bytes, NUL right-padded to `ALID_INDEX_WIDTH`; an ALID
#: wider than that is left out of the index and found by an exact scan at its position, never
#: truncated into it: two indels at one position that agree over the slot would then share a key,
#: and one would answer to the other's name (opengwasdb issue #127, its ALID width).
ALID_INDEX_WIDTH: Final = 32
#: The index's group and its two arrays, by their path.
ALID_INDEX: Final = "variants/alid_index"
ALID_INDEX_ALID: Final = f"{ALID_INDEX}/alid"
ALID_INDEX_ROW: Final = f"{ALID_INDEX}/row"
#: Their (inner chunk, shard) in elements, as `SIDE` is. A key is 32 bytes, so an inner chunk of
#: 262,144 bytes holds 8,192 keys and a 16 MiB shard 131,072: the whole index is 275 MB on EUR, and
#: one search reads the chunks on its path. `docs/benchmarks/alid-index.md` measured this against
#: `SIDE`'s 16 KiB chunks, whose 16,693 of them made a whole read 4.6 s against 0.2 s.
ALID_KEYS: Final = (262_144, 16_777_216)
ALID_ROWS: Final = (262_144, 1_048_576)
#: The keys' fill value: no ALID is empty, so a whole key of NUL bytes is one no store holds. It is
#: 0 because every padding byte is NUL too, so a key is judged whole rather than byte by byte
#: (`refused_if_lost`), and a lost chunk reads as keys of no bytes.
ALID_KEY_FILL: Final = 0
#: The rows': a row is never negative.
ALID_ROW_FILL: Final = -1

#: Each array's fill value: one a valid LD Store never holds (spec §6). The `r` arrays' is
#: their encoding's reserved code and the strings' is ""; both are set where those arrays are
#: made.
INT64_FILL: Final = -1
PARTNER_FILL: Final = -1
POSITION_FILL: Final = 0
FREQUENCY_FILL: Final = float("nan")


def never_data(path: str, values: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    """Where `values` of the array at `path`, such as `index/22/partner`, are ones no valid LD
    Store holds there (spec §2–§6). The array's kind is its name, the path's last component.

    Each array's fill value must be such a value, because a chunk the store lost reads as its
    fill value: then every read can tell a lost chunk from data and refuse it. The reader
    refuses an array whose fill value is not, and `envelope.fill` fails it (spec §12).
    """
    v = np.asarray(values)
    name = path.rsplit("/", 1)[-1]
    if path == ALID_INDEX_ALID:
        # A key is judged 32 bytes at a time, never byte by byte: a padding byte is 0, which its
        # fill value is too, so no single byte can be the one no store holds. `refused_if_lost`
        # judges a whole key; the fill must be 0, or a lost chunk would read as a run of keys.
        return np.asarray(v == ALID_KEY_FILL)
    if name == "r":
        # Every admitted encoding shares the container and its reserved code (ADR 0004), so a
        # lost chunk is told from data before the store's own encoding is known.
        return np.asarray(v == REncoding.reserved_code)
    if name in {"indptr", "partner", "row"}:
        return np.asarray(v < 0)
    if name == "position":
        return np.asarray(v <= 0)
    if name in {"alid", "a1", "a2"}:
        return np.asarray(v == "")
    if name == "a1_frequency":
        with np.errstate(invalid="ignore"):
            return np.asarray(~((v > 0) & (v < 1)))
    raise KeyError(f"{name} is not an array of an LD Store (spec §1)")


def shown_fill(fill: object) -> str:
    """A fill value as a message names it: a string quoted, so an empty one shows."""
    return f'"{fill}"' if isinstance(fill, str) else str(fill)


def refused_if_lost(array: zarr.Array[Any], values: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """`values` read from `array`, unless one is a value no LD Store holds there, as its fill
    value is: that is a chunk the store lost, and decoding it would be inventing data."""
    if array.path == ALID_INDEX_ALID:
        return _refused_if_lost_keys(array, values)
    lost = never_data(array.path, values)
    if lost.any():
        raise ValueError(
            f"{array.path} read {int(lost.sum())} element(s) no LD Store holds there, as its "
            f"fill value {shown_fill(array.fill_value)} is: a chunk the store lost, never data "
            "(spec §3, §6)"
        )
    return values


def _refused_if_lost_keys(array: zarr.Array[Any], values: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """The keys `values` of `variants/alid_index/alid`, unless one is all NUL bytes: no ALID is
    empty, so that is a chunk the store lost rather than a key (spec §6)."""
    keys = np.asarray(values, dtype=np.uint8).reshape(-1, ALID_INDEX_WIDTH)
    lost = ~keys.any(axis=1)
    if lost.any():
        raise ValueError(
            f"{array.path} read {int(lost.sum())} key(s) of no bytes at all, as its fill value "
            f"{shown_fill(array.fill_value)} is: a chunk the store lost, never data (spec §6)"
        )
    return values


def alid_keys(
    alids: npt.ArrayLike, first: int = 0
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.int64]]:
    """The key of each ALID of `alids` that fits the slot, and its row, in the order given (spec
    §2; ADR 0010). An ALID wider than `ALID_INDEX_WIDTH` bytes has no key, and its row is not
    left out of the count here: `derive_alid_index` counts them from `n_variants` and the keys.

    `first` is the Variant Set row of `alids[0]`, so a block of a store's table gives the rows
    the whole table's index has. The padding is numpy's own `|S32` cast, which is the engine's
    (ADR 0010), and nothing is truncated: every ALID that reaches it fits.
    """
    text = np.asarray(alids)
    # UTF-8 first, so a key is the ALID's bytes however wide the characters are; the cast to the
    # slot then only pads, because the wider ones are already gone.
    encoded = np.char.encode(text, "utf-8") if len(text) else np.empty(0, dtype="S1")
    lengths = np.strings.str_len(encoded) if len(text) else np.empty(0, dtype=np.int64)
    fits = lengths <= ALID_INDEX_WIDTH
    keys = np.array(encoded[fits], dtype=f"|S{ALID_INDEX_WIDTH}")
    return (
        keys.view(np.uint8).reshape(-1, ALID_INDEX_WIDTH),
        np.flatnonzero(fits).astype(np.int64) + first,
    )


def ordered_index(
    keys: npt.NDArray[np.uint8], rows: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.int64]]:
    """`keys` and `rows` in the engine's order (ADR 0010): by key bytewise, ties by row, which
    is what its stable sort of the padded keys gives."""
    order = np.argsort(keys.view(f"|S{ALID_INDEX_WIDTH}").ravel(), kind="stable")
    return keys[order], rows[order]


def alid_index(
    alids: npt.ArrayLike,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.int64]]:
    """The ALID index of one Variant Set's `alids`: its keys, and the row of each, in the
    engine's order (spec §2, §6; ADR 0010). The one place the width, the padding and the sort
    are written; a store writes what this gives, and validation computes it too."""
    return ordered_index(*alid_keys(alids))


def put_alid_index(
    index: zarr.Group, keys: npt.NDArray[np.uint8], rows: npt.NDArray[np.int64]
) -> None:
    """Write the index's two arrays into the group `index`: the keys' bytes, and each entry's
    Variant Set row (spec §6)."""
    put_numbers(index, "alid", keys.reshape(-1), ALID_KEY_FILL, ALID_KEYS)
    put_numbers(index, "row", rows.astype(np.int32), ALID_ROW_FILL, ALID_ROWS)


#: The keys of one record of `variants.attrs["chromosomes"]` (spec §2).
CHROMOSOME_KEYS: Final = ("name", "row_start", "row_stop")


def chromosomes_record(ranges: dict[str, tuple[int, int]]) -> list[dict[str, Any]]:
    """The `variants` group's `chromosomes` attribute for these ranges, in order (spec §2)."""
    return [
        dict(zip(CHROMOSOME_KEYS, (name, start, stop), strict=True))
        for name, (start, stop) in ranges.items()
    ]


def _is_row(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _chromosome_entry(entry: object) -> tuple[str, int, int]:
    """One `chromosomes` entry as (name, row_start, row_stop), or ValueError."""
    if not isinstance(entry, dict) or set(entry) != set(CHROMOSOME_KEYS):
        raise ValueError(f"a chromosomes entry does not have exactly the keys {CHROMOSOME_KEYS}")
    name, start, stop = (entry[key] for key in CHROMOSOME_KEYS)
    if not (isinstance(name, str) and _is_row(start) and _is_row(stop) and 0 <= start <= stop):
        raise ValueError(f"a chromosomes entry is not a name and rows 0 <= start <= stop: {entry}")
    return name, start, stop


def chromosomes_listed(record: object) -> list[tuple[str, int, int]]:
    """The (name, row_start, row_stop) of each entry of a `chromosomes` attribute, in order and
    with repeats kept, so a rule can see them. A record that is not a list of entries with
    exactly these keys, a string name and integer bounds 0 ≤ start ≤ stop raises ValueError."""
    if not isinstance(record, list):
        raise ValueError("the chromosomes attribute is not a list")
    return [_chromosome_entry(entry) for entry in record]


def clipped(length: int, inner: int, shard: int) -> tuple[int, int]:
    """The inner chunk and shard of an array of `length`: an inner chunk longer than the array
    is clipped to it, and a shard to the smallest whole number of inner chunks covering it."""
    inner = min(inner, max(1, length))
    shard = min(shard, -(-max(1, length) // inner) * inner)
    return inner, shard


def put_numbers(
    group: zarr.Group,
    name: str,
    data: npt.NDArray[np.generic],
    fill: float | int,
    shape: tuple[int, int] = SIDE,
) -> zarr.Array[Any]:
    """Create the numeric array `name` (bytes, then zstd level 3) and write all of `data`."""
    array = new_numbers(group, name, len(data), data.dtype, fill, shape)
    if len(data):
        array[:] = data
    return array


def new_numbers(
    group: zarr.Group,
    name: str,
    length: int,
    dtype: npt.DTypeLike,
    fill: float | int,
    shape: tuple[int, int] = SIDE,
) -> zarr.Array[Any]:
    """A numeric array `name` of `length` with nothing written: every chunk reads `fill`."""
    inner, shard = clipped(length, *shape)
    return group.create_array(
        name,
        shape=(length,),
        dtype=dtype,
        chunks=(inner,),
        shards=(shard,),
        compressors=[ZstdCodec(level=3)],
        fill_value=fill,
    )


def put_strings(group: zarr.Group, name: str, data: npt.NDArray[np.str_]) -> zarr.Array[Any]:
    """Create the string array `name` (vlen-utf8, then zstd level 0) and write all of `data`."""
    inner, shard = clipped(len(data), *SIDE)
    array = group.create_array(
        name,
        shape=data.shape,
        dtype=str,
        chunks=(inner,),
        shards=(shard,),
        fill_value="",
    )
    if len(data):
        array[:] = data
    return array


def new_r_array(
    group: zarr.Group, name: str, length: int, shape: tuple[int, int], encoding: REncoding
) -> zarr.Array[Any]:
    """An `r` array of `length` in `encoding`'s dtype with nothing written: every chunk reads
    the encoding's reserved code (spec §3, §6)."""
    inner, shard = clipped(length, *shape)
    return group.create_array(
        name,
        shape=(length,),
        dtype=encoding.dtype,
        chunks=(inner,),
        shards=(shard,),
        compressors=[ZstdCodec(level=3)],
        fill_value=encoding.reserved_code,
    )
