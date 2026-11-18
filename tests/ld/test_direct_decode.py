"""The numeric decoder's seam: the same chunks as zarr, not another LD encoding (#40)."""

from __future__ import annotations

import asyncio
import threading
import tracemalloc
import weakref
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from store_copies import TEST_STORE, resharded
from zarr.abc.store import RangeByteRequest, SuffixByteRequest
from zarr.codecs import BytesCodec, Crc32cCodec, GzipCodec, ShardingCodec, ZstdCodec
from zarr.core.metadata.v3 import RectilinearChunkGridMetadata

from ogld.store import decoding, location


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
    assert decoding.NumericDecoder.of(array) is not None
    cache = location.ChunkCache()
    np.testing.assert_array_equal(cache.take(array, np.arange(len(data))), array[:])


def test_big_endian_shard_index_equals_zarr(tmp_path: Path) -> None:
    shard = ShardingCodec(
        chunk_shape=(8,),
        codecs=[BytesCodec(endian="big"), ZstdCodec()],
        index_codecs=[BytesCodec(endian="big"), Crc32cCodec()],
    )
    array = zarr.create_array(
        store=tmp_path / "a",
        shape=(19,),
        dtype="int32",
        chunks=(16,),
        serializer=shard,
        compressors=None,
        fill_value=-1,
    )
    array[:] = np.arange(19) * 257
    assert decoding.NumericDecoder.of(array) is not None
    np.testing.assert_array_equal(location.ChunkCache().take(array, np.arange(19)), array[:])


def test_non_regular_or_multidimensional_arrays_are_not_recognised(tmp_path: Path) -> None:
    matrix = zarr.create_array(store=tmp_path / "matrix", shape=(8, 8), dtype="int32")
    assert decoding.NumericDecoder.of(matrix) is None
    array = numbers(tmp_path)
    meta = replace(
        array.metadata,
        shape=(32,),
        chunk_grid=RectilinearChunkGridMetadata(chunk_shapes=((16, 16),)),
    )
    rectilinear = zarr.Array(zarr.AsyncArray(meta, array.store_path))
    assert decoding.NumericDecoder.of(rectilinear) is None


def test_synchronous_index_reads_keep_the_same_budget(tmp_path: Path) -> None:
    array = numbers(tmp_path)
    cache = location.ChunkCache(256)
    store = location._KeptIndexes(array.store, cache)
    expected = (tmp_path / "r/c/0").read_bytes()
    for _ in range(2):
        got = store.get_sync("c/0", byte_range=SuffixByteRequest(68))
        assert got is not None
        assert got.to_bytes() == expected[-68:]
    assert cache._bytes == 68
    assert store.get_sync("c/missing", byte_range=SuffixByteRequest(68)) is None
    prefix = store.get_sync("c/0", byte_range=RangeByteRequest(0, 4))
    assert prefix is not None
    assert prefix.to_bytes() == expected[:4]


def test_a_sparse_warm_read_does_not_join_whole_chunks(tmp_path: Path) -> None:
    # 8 MiB kept, one value per chunk returned: a full concatenate is not a read's result.
    size = 65536
    data = np.arange(32 * size, dtype=np.int32)
    array = zarr.create_array(
        store=tmp_path / "a",
        shape=data.shape,
        dtype=data.dtype,
        chunks=(size,),
        shards=(size * 8,),
        compressors=[ZstdCodec()],
    )
    array[:] = data
    cache = location.ChunkCache()
    coords = np.arange(32) * size + 17
    cache.take(array, coords)
    tracemalloc.start()
    try:
        got = cache.take(array, coords[::-1])
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    np.testing.assert_array_equal(got, data[coords[::-1]])
    assert peak < data.nbytes // 2, f"{peak} bytes held to return {got.nbytes}"


def test_decoded_chunks_transfer_their_memory_to_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    array = numbers(tmp_path)
    fetched = location._fetched
    held: list[np.ndarray] = []

    def recording(*args: Any) -> Any:
        result = fetched(*args)
        held.extend(result)
        return result

    monkeypatch.setattr(location, "_fetched", recording)
    cache = location.ChunkCache()
    cache.take(array, np.arange(29))
    kept = [entry[0] for key, entry in cache._kept.items() if isinstance(key, tuple)]
    assert len(kept) == 4
    assert all(any(np.shares_memory(chunk, source) for source in held) for chunk in kept)


def test_cancelled_decode_drains_its_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    decoder = decoding.NumericDecoder(1, 1, np.dtype("i4"), np.dtype("<u8"))

    def held(*args: Any) -> None:
        started.set()
        assert release.wait(2), "the test must release its own worker"
        finished.set()

    monkeypatch.setattr(decoding.NumericDecoder, "_into", held)

    async def run() -> None:
        task = asyncio.create_task(decoder.decode(b"x", 0, [(0, 0, 1)], [None]))
        while not started.is_set():
            await asyncio.sleep(0.001)
        task.cancel()
        try:
            await asyncio.sleep(0.01)
            assert not task.done(), "a cancelled read must still own its worker's buffers"
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()

    asyncio.run(run())


def test_every_test_store_array_equals_zarr() -> None:
    root = zarr.open_group(TEST_STORE, mode="r")
    arrays = [
        (name, node) for name, node in root.members(max_depth=None) if isinstance(node, zarr.Array)
    ]
    assert len(arrays) > 20
    for _, array in arrays:
        decoded = decoding.NumericDecoder.of(array)
        assert (decoded is None) == (array.dtype.kind == "T" or array.shards is None), array.path
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
    assert decoding.NumericDecoder.of(array) is None
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


def test_noncontiguous_physical_offsets_equal_zarr(tmp_path: Path) -> None:
    import numcodecs

    array = numbers(tmp_path)
    array[:] = np.arange(29, dtype=np.int32) * 17
    path = tmp_path / "r/c/0"
    data = path.read_bytes()
    crc = numcodecs.CRC32C()
    index = np.frombuffer(crc.decode(data[-68:]), dtype="<u8").copy().reshape(4, 2)
    # Logical chunks 0,1 now live at physical chunks 0,2, with an unwanted chunk between.
    index[[1, 2]] = index[[2, 1]]
    path.write_bytes(data[:-68] + crc.encode(index).tobytes())
    coords = np.arange(16)
    np.testing.assert_array_equal(location.ChunkCache().take(array, coords), array[:16])


def test_an_elided_near_band_inner_chunk_is_refused(tmp_path: Path) -> None:
    import ogld

    path = resharded(tmp_path / "store", 512, 4)
    store = ogld.open(path)
    start, _ = store._reader._ranges["22"]
    names = store._reader.alids_of(np.array([start, start + 1])).tolist()
    assert np.isfinite(store.matrix(names).r).all()
    array = zarr.open_array(path / "near/22/r", mode="r+")
    array[:512] = array.fill_value
    assert (path / "near/22/r/c/0").exists(), "the shard must still hold its other chunks"
    with pytest.raises(ValueError, match="chunk the store lost"):
        ogld.open(path).matrix(names)


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


def test_unknown_frame_size_uses_the_validating_decode() -> None:
    import numcodecs

    source = np.arange(8, dtype=np.int32)
    codec = numcodecs.Zstd()
    encoded = codec.encode(source)
    assert encoded[4] == 0x20, "one-byte content size in a single-segment frame"
    # Replace the one-byte size with a window descriptor, making a valid unknown-size frame.
    frame = memoryview(encoded[:4] + b"\x00\x58" + encoded[6:])
    assert decoding._frame_size(frame) is None
    decoder = decoding.NumericDecoder(8, 1, source.dtype, np.dtype("<u8"))
    np.testing.assert_array_equal(decoder._frame(codec, frame), source)
    with pytest.raises(ValueError, match="expected"):
        decoding.NumericDecoder(9, 1, source.dtype, np.dtype("<u8"))._frame(codec, frame)
    larger = codec.encode(np.arange(100, dtype=np.int32))
    assert decoding._frame_size(memoryview(larger)) == 400, "two-byte size includes +256"


def test_unknown_size_native_frames_do_not_double_the_flight_buffer() -> None:
    import numcodecs

    source = np.arange(65536, dtype=np.int32)
    codec = numcodecs.Zstd()
    encoded = codec.encode(source)
    assert encoded[4] & 0xE0 == 0xA0, "single segment with four-byte content size"
    frame = memoryview(encoded[:4] + bytes([encoded[4] & 0x1F, 0x58]) + encoded[9:])
    assert decoding._frame_size(frame) is None
    decoder = decoding.NumericDecoder(len(source), 1, source.dtype, np.dtype("<u8"))
    tracemalloc.start()
    try:
        got = decoder._frame(codec, frame)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    np.testing.assert_array_equal(got, source)
    assert peak < source.nbytes * 1.5, "native fallback must not copy the validated raw buffer"


def test_completed_worker_does_not_retain_released_buffers(monkeypatch: pytest.MonkeyPatch) -> None:
    retained: list[Any] = []
    futures: list[Future[Any]] = []
    references: list[weakref.ReferenceType[np.ndarray]] = []

    class Executor:
        def submit(self, call: Any) -> Future[Any]:
            retained.append(call)
            future: Future[Any] = Future()
            futures.append(future)
            future.set_result(call())
            return future

    def job() -> Any:
        data = np.empty(1024, dtype=np.uint8)
        references.append(weakref.ref(data))
        return lambda: data.nbytes

    monkeypatch.setattr(decoding, "_DECODE_POOL", Executor())
    asyncio.run(decoding.on_pool(job()))
    assert len(retained) == 1, "the executor deliberately keeps its completed work item"
    assert references[0]() is None, "completed jobs must not retain released byte credits"
    value = asyncio.run(decoding.on_pool(lambda: np.zeros(1024)))
    reference = weakref.ref(value)
    del value
    assert len(futures) == 2, "completed Futures are deliberately kept too"
    assert reference() is None, "a completed I/O Future must not retain its returned range"
