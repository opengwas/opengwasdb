"""How an LD Store's Zarr arrays are laid out (ld-store §6): the one place that creates them.

Every array is Zarr v3 with `sharding_indexed`, except the Partner Rows' two: those are one chunk
each with the `bytes` codec alone, because their bytes are the frames themselves and a reader
addresses one frame's byte range (partner-rows §5.1; ADR 0097). The chunk and shard shapes, the
codec chain and the fill value of each kind of array are the spec's table, kept here as data so the
writer cannot create an array the table does not describe, and the reader's tests can name the fill
values it promises.
"""

from __future__ import annotations
import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from os import stat_result
from pathlib import Path
from typing import Any, Final
import numpy as np
import numpy.typing as npt
import zarr
from zarr.codecs import ZstdCodec
from zarr.storage import LocalStore, StorePath, WrapperStore
from ogquery.ld.format import InputError
from opengwasdb.ld.format import REncoding

# The read side of this module is ogquery's (opengwas/ogquery#2): re-exported by name.
from ogquery.ld.store.layout import (
    alid_companion as alid_companion,
    alid_fits as alid_fits,
    ALID_INDEX as ALID_INDEX,
    ALID_INDEX_ALID as ALID_INDEX_ALID,
    ALID_INDEX_ROW as ALID_INDEX_ROW,
    ALID_INDEX_WIDTH as ALID_INDEX_WIDTH,
    ALID_KEY_DTYPE as ALID_KEY_DTYPE,
    ALID_KEY_FILL as ALID_KEY_FILL,
    alid_keys as alid_keys,
    ALID_MMAP as ALID_MMAP,
    ALID_MMAP_ATTR as ALID_MMAP_ATTR,
    alid_mmap_error as alid_mmap_error,
    alid_mmap_generation as alid_mmap_generation,
    ALID_MMAP_KEYS as ALID_MMAP_KEYS,
    ALID_MMAP_MANIFEST as ALID_MMAP_MANIFEST,
    alid_mmap_manifest as alid_mmap_manifest,
    ALID_MMAP_PAYLOADS as ALID_MMAP_PAYLOADS,
    alid_mmap_pointer as alid_mmap_pointer,
    ALID_MMAP_ROW_DTYPE as ALID_MMAP_ROW_DTYPE,
    ALID_MMAP_ROWS as ALID_MMAP_ROWS,
    ALID_MMAP_SCHEMA as ALID_MMAP_SCHEMA,
    alid_mmap_source_change as alid_mmap_source_change,
    alid_mmap_sources as alid_mmap_sources,
    AlidCompanion as AlidCompanion,
    AlidPayload as AlidPayload,
    CHROMOSOME_KEYS as CHROMOSOME_KEYS,
    chromosomes_listed as chromosomes_listed,
    chunk_file as chunk_file,
    chunk_key as chunk_key,
    file_identity as file_identity,
    file_sha256 as file_sha256,
    FRAMES as FRAMES,
    FRAMES_INDEX as FRAMES_INDEX,
    json_bytes as json_bytes,
    local_path as local_path,
    local_root as local_root,
    never_data as never_data,
    PARTNERS as PARTNERS,
    partners_path as partners_path,
    refused_if_lost as refused_if_lost,
    shown_fill as shown_fill,
)


#: (inner chunk, shard) in elements, for every array but the Near Band's `r`.
SIDE: Final = (16_384, 1_048_576)


#: The Near Band's `r`: layout C (ld-store §6; ADR 0092).
NEAR_R: Final = (262_144, 67_108_864)


def _create_array(group: zarr.Group, name: str, **spec: Any) -> zarr.Array[Any]:
    """`group.create_array`, the one call every array here is made by, with how it is written
    fixed: a chunk holding only the fill value is left out and reads as the fill value, as zarr's
    default had it for every Store ogld built. `import opengwasdb` sets zarr's process-wide
    `array.write_empty_chunks` to True for the GWAS Store (opengwasdb/store/arrays.py); an LD
    Store or Genotype Store array does not inherit it (#455)."""
    return group.create_array(name, config={"write_empty_chunks": False}, **spec)


#: Their (inner chunk, shard) in elements, as `SIDE` is. A key is 32 bytes, so an inner chunk of
#: 262,144 bytes holds 8,192 keys and a 16 MiB shard 524,288: the whole index is 275 MB on EUR, and
#: a search reads it one shard at a time. `docs/benchmarks/ld/alid-index.md` measured this against
#: `SIDE`'s 16 KiB chunks, whose 16,693 of them made a whole read 4.6 s against 0.2 s.
ALID_KEYS: Final = (262_144, 16_777_216)


ALID_ROWS: Final = (262_144, 1_048_576)


#: The rows': a row is never negative.
ALID_ROW_FILL: Final = -1


#: Each array's fill value: one a valid LD Store never holds (ld-store §6). The `r` arrays' is
#: their encoding's reserved code and the strings' is ""; both are set where those arrays are
#: made.
INT64_FILL: Final = -1


PARTNER_FILL: Final = -1


POSITION_FILL: Final = 0


FREQUENCY_FILL: Final = float("nan")


#: The file the frames are appended to until the group is closed, when it becomes the `frames`
#: chunk: the array's shape is the whole file's, so it is not known until the last frame (§5.1).
FRAMES_BUILDING: Final = "frames.bin"


#: The frames' fill value: 0, because a frame is a zstd frame and 0 begins none (ld-store §3). A
#: chunk a store lost is refused by the frame's own checksum and length, not by a fill value.
FRAMES_FILL: Final = 0


#: The offsets': the largest uint64, which no offset of a whole number of bytes is.
FRAMES_INDEX_FILL: Final = int(np.iinfo(np.uint64).max)


#: The dtype the offsets are written and hashed in: little-endian whatever this machine is,
#: because their chunk file is the offsets' bytes (partner-rows §5.1, §5.3). The writer, the
#: reader and the rule that hashes the chunk all take it from here.
FRAMES_INDEX_DTYPE: Final = np.dtype("<u8")


def ordered_index(
    keys: npt.NDArray[np.uint8], rows: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.int64]]:
    """`keys` and `rows` in the engine's order (ADR 0096): by key bytewise, ties by row, which
    is what its stable sort of the padded keys gives."""
    order = np.argsort(keys.view(f"|S{ALID_INDEX_WIDTH}").ravel(), kind="stable")
    return keys[order], rows[order]


def alid_index(
    alids: npt.ArrayLike,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.int64]]:
    """The ALID index of one Variant Set's `alids`: its keys, and the row of each, in the
    engine's order (ld-store §2, §6; ADR 0096). The one place the width, the padding and the sort
    are written; a store writes what this gives, and validation computes it too."""
    return ordered_index(*alid_keys(alids))


def put_alid_index(
    index: zarr.Group, keys: npt.NDArray[np.uint8], rows: npt.NDArray[np.int64]
) -> None:
    """Write the index's two arrays into the group `index`: the keys' bytes, and each entry's
    Variant Set row (ld-store §6)."""
    put_numbers(index, "alid", keys.reshape(-1), ALID_KEY_FILL, ALID_KEYS)
    put_numbers(index, "row", rows.astype(np.int32), ALID_ROW_FILL, ALID_ROWS)


def chromosomes_record(ranges: dict[str, tuple[int, int]]) -> list[dict[str, Any]]:
    """The `variants` group's `chromosomes` attribute for these ranges, in order (ld-store §2)."""
    return [
        dict(zip(CHROMOSOME_KEYS, (name, start, stop), strict=True))
        for name, (start, stop) in ranges.items()
    ]


def clipped(length: int, inner: int, shard: int) -> tuple[int, int]:
    """The inner chunk and shard of an array of `length`: an inner chunk longer than the array
    is clipped to it, and a shard to the smallest whole number of inner chunks covering it."""
    inner = min(inner, max(1, length))
    shard = min(shard, -(-max(1, length) // inner) * inner)
    return inner, shard


def new_frames_array(group: zarr.Group, length: int) -> zarr.Array[Any]:
    """The `frames` array: `uint8`, one chunk of its whole length, the `bytes` codec alone and no
    compressor, because its bytes are already zstd frames (partner-rows §5.1)."""
    if length < 1:
        raise ValueError("a Partner Rows' frames are at least one frame's bytes, never none")
    return _create_array(
        group,
        FRAMES,
        shape=(length,),
        dtype=np.uint8,
        chunks=(length,),
        compressors=[],
        fill_value=FRAMES_FILL,
    )


def new_frames_index(group: zarr.Group, length: int) -> zarr.Array[Any]:
    """The `frames_idx` array: the F + 1 uint64 byte offsets of the frames, one chunk of its
    whole length, little-endian in a `bytes` codec alone so the file is the offsets (§5.1)."""
    return _create_array(
        group,
        FRAMES_INDEX,
        shape=(length,),
        dtype=FRAMES_INDEX_DTYPE,
        chunks=(length,),
        compressors=[],
        fill_value=FRAMES_INDEX_FILL,
    )


def local_directory(store_path: StorePath) -> Path:
    """The local directory a store path names, refused for a store that is not local: the Partner
    Rows' frames are appended to a file, which a remote store has none of (partner-rows §5.1)."""
    return local_root(store_path) / store_path.path


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
    return _create_array(
        group,
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
    array = _create_array(
        group,
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
    the encoding's reserved code (ld-store §3, §6)."""
    inner, shard = clipped(length, *shape)
    return _create_array(
        group,
        name,
        shape=(length,),
        dtype=encoding.dtype,
        chunks=(inner,),
        shards=(shard,),
        compressors=[ZstdCodec(level=3)],
        fill_value=encoding.reserved_code,
    )
