"""The numeric decoder's seam: the same chunks as zarr, not another LD encoding (#40)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from store_copies import TEST_STORE
from zarr.codecs import BytesCodec, Crc32cCodec, GzipCodec, ShardingCodec, ZstdCodec

from ogld.store import location


@pytest.mark.parametrize("endian", ["little", "big"])
def test_numeric_chunks_equal_zarr(tmp_path: Path, endian: str) -> None:
    # Deliberately non-palindromic bytes, multiple shards, and a padded final chunk.
    data = np.arange(103, dtype=np.int32) * 257 - 1000
    array = zarr.create_array(
        store=tmp_path / "numbers",
        shape=data.shape,
        dtype=data.dtype,
        chunks=(32,),
        shards=(64,),
        serializer=BytesCodec(endian=endian),
        compressors=[ZstdCodec(level=3)],
        fill_value=-1,
    )
    array[:] = data
    assert location._NumericDecoder.of(array) is not None
    cache = location.ChunkCache()
    np.testing.assert_array_equal(cache.take(array, np.arange(len(data))), array[:])


def test_every_test_store_array_equals_zarr() -> None:
    root = zarr.open_group(TEST_STORE, mode="r")
    arrays = [
        (name, node) for name, node in root.members(max_depth=None) if isinstance(node, zarr.Array)
    ]
    assert len(arrays) > 20
    for _, array in arrays:
        decoded = location._NumericDecoder.of(array)
        assert (decoded is None) == (array.dtype.kind == "T"), array.path
        coords = np.arange(array.shape[0])
        np.testing.assert_array_equal(location.ChunkCache().take(array, coords), array[:])


@pytest.mark.parametrize("chain", ["gzip", "index", "start", "unsharded"])
def test_unknown_chains_fall_back_to_zarr(tmp_path: Path, chain: str) -> None:
    shard = ShardingCodec(
        chunk_shape=(8,),
        codecs=[BytesCodec(), GzipCodec()] if chain == "gzip" else [BytesCodec(), ZstdCodec()],
        index_codecs=[BytesCodec()] if chain == "index" else [BytesCodec(), Crc32cCodec()],
        index_location="start" if chain == "start" else "end",
    )
    array = zarr.create_array(
        store=tmp_path / "a",
        shape=(23,),
        dtype="int32",
        chunks=(32,),
        serializer=BytesCodec() if chain == "unsharded" else shard,
        compressors=None,
        fill_value=-1,
    )
    array[:] = np.arange(23) * 17
    assert location._NumericDecoder.of(array) is None
    np.testing.assert_array_equal(location.ChunkCache().take(array, np.arange(23)), array[:])


def numbers(tmp_path: Path) -> zarr.Array[Any]:
    array = zarr.create_array(
        store=tmp_path / "r",
        shape=(29,),
        dtype="int32",
        chunks=(8,),
        shards=(32,),
        compressors=[ZstdCodec()],
        fill_value=-1,
    )
    # Chunk 1 is elided, not a missing shard: the other chunks are really there.
    data = np.arange(29, dtype=np.int32) * 17
    data[8:16] = -1
    array[:] = data
    assert (tmp_path / "r/c/0").exists()
    return array


def test_an_empty_inner_chunk_reads_the_fill_value(tmp_path: Path) -> None:
    array = numbers(tmp_path)
    assert np.all(array[8:16] == -1)
    np.testing.assert_array_equal(location.ChunkCache().take(array, np.arange(29)), array[:])


@pytest.mark.parametrize("damage", ["chunk", "crc", "short", "short_frame"])
def test_corruption_raises_as_zarr_does(tmp_path: Path, damage: str) -> None:
    array = numbers(tmp_path)
    path = tmp_path / "r/c/0"
    data = bytearray(path.read_bytes())
    if damage == "chunk":
        data[0:4] = b"bad!"
    elif damage == "crc":
        data[-1] ^= 1
    elif damage == "short":
        data = data[:5]
    else:
        # A valid zstd frame with too few decoded elements is corrupt as a Zarr chunk.
        # Replacing its index entry avoids testing a checksum or bad-magic failure instead.
        import numcodecs

        crc = numcodecs.CRC32C()
        index = np.frombuffer(crc.decode(data[-68:]), dtype="<u8").copy().reshape(4, 2)
        short = numcodecs.Zstd().encode(np.array([17], dtype=np.int32))
        at = len(data) - 68
        index[0] = (at, len(short))
        data = data[:-68] + short + crc.encode(index).tobytes()
    path.write_bytes(data)
    with pytest.raises((ValueError, RuntimeError)):
        array[:]
    with pytest.raises((ValueError, RuntimeError)):
        location.ChunkCache().take(array, np.arange(29))
