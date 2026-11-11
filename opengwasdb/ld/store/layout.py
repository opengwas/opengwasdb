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
#: The Near Band's `r`: layout C, provisional until ogld#8 settles it (spec §6).
NEAR_R: Final = (262_144, 67_108_864)

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
    if name == "r":
        # Every admitted encoding shares the container and its reserved code (ADR 0004), so a
        # lost chunk is told from data before the store's own encoding is known.
        return np.asarray(v == REncoding.reserved_code)
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


def shown_fill(fill: object) -> str:
    """A fill value as a message names it: a string quoted, so an empty one shows."""
    return f'"{fill}"' if isinstance(fill, str) else str(fill)


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


def put_codes(
    group: zarr.Group,
    name: str,
    codes: npt.NDArray[np.generic],
    encoding: REncoding,
    shape: tuple[int, int] = SIDE,
) -> zarr.Array[Any]:
    """Create the `r` array `name` in `encoding`'s dtype and fill value, and write `codes`."""
    array = new_r_array(group, name, len(codes), shape, encoding)
    if len(codes):
        array[:] = codes
    return array
