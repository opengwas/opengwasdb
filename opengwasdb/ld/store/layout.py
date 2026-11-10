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
