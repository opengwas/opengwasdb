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

from ogld.format import ENCODING, RESERVED_CODE

#: The dtype of every `r` array: the encoding's, never a literal (spec §3).
R_DTYPE: Final = np.dtype(ENCODING["r_dtype"])

#: (inner chunk, shard) in elements, for every array but the Near Band's `r`.
SIDE: Final = (16_384, 1_048_576)
#: The Near Band's `r`: layout C, provisional until ogld#8 settles it (spec §6).
NEAR_R: Final = (262_144, 67_108_864)

#: Each array's fill value: one a valid LD Store never holds (spec §6). The `r` arrays' is
#: RESERVED_CODE and the strings' is ""; both are set where those arrays are made.
INT64_FILL: Final = -1
PARTNER_FILL: Final = -1
POSITION_FILL: Final = 0
FREQUENCY_FILL: Final = float("nan")


def never_data(name: str, values: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    """Where `values` of the array called `name` (its last path component) are ones no valid
    LD Store holds there (spec §2–§6).

    Each array's fill value must be such a value, because a chunk the store lost reads as its
    fill value: then every read can tell a lost chunk from data and refuse it. The reader
    refuses an array whose fill value is not, and `envelope.fill` fails it (spec §12).
    """
    v = np.asarray(values)
    if name == "r":
        return np.asarray(v == RESERVED_CODE)
    if name in {"indptr", "partner"}:
        return np.asarray(v < 0)
    if name == "position":
        return np.asarray(v <= 0)
    if name in {"alid", "a1", "a2"}:
        return np.asarray(v == "")
    if name == "a1_frequency":
        with np.errstate(invalid="ignore"):
            return np.asarray(~((v > 0) & (v < 1)))
    raise KeyError(f"{name} is not an array of an LD Store (spec §1)")


def spans(first: npt.ArrayLike, length: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """first[i], first[i] + 1, … for length[i] elements each, back to back: the elements of
    CSR rows, or the columns of Near Band rows (spec §4, §5)."""
    count = np.asarray(length, dtype=np.int64)
    offset = np.repeat(np.cumsum(count) - count, count)
    start = np.repeat(np.asarray(first, dtype=np.int64), count)
    return start + (np.arange(int(count.sum()), dtype=np.int64) - offset)


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
    inner, shard = clipped(len(data), *shape)
    array = group.create_array(
        name,
        shape=data.shape,
        dtype=data.dtype,
        chunks=(inner,),
        shards=(shard,),
        compressors=[ZstdCodec(level=3)],
        fill_value=fill,
    )
    if len(data):
        array[:] = data
    return array


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
    group: zarr.Group, name: str, length: int, shape: tuple[int, int]
) -> zarr.Array[Any]:
    """An `r` array of `length` with nothing written: every chunk reads the reserved code."""
    inner, shard = clipped(length, *shape)
    return group.create_array(
        name,
        shape=(length,),
        dtype=R_DTYPE,
        chunks=(inner,),
        shards=(shard,),
        compressors=[ZstdCodec(level=3)],
        fill_value=RESERVED_CODE,
    )
