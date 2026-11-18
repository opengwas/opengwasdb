"""Bounded warming must not change answers or exceed either byte budget (ogld#40)."""

from __future__ import annotations

import asyncio
import os
import tracemalloc
import weakref
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from store_copies import TEST_STORE, resharded, store_reads

import ogld
from ogld.store import decoding, location


def test_prefetch_fits_or_declines_without_reading(tmp_path: Path) -> None:
    root = resharded(tmp_path / "store", 512, 4)
    cache = location.ChunkCache(1 << 20)
    array = location.open_group(root, cache=cache)["near/22/r"]
    assert isinstance(array, zarr.Array)
    assert cache.prefetch(array, [0, 1024], [512, 512])
    assert (array.path, 0) in cache._kept
    assert (array.path, 2) in cache._kept
    assert cache._bytes <= cache.budget
    tiny = location.ChunkCache(1)
    assert not tiny.prefetch(array, [0], [512])
    assert not tiny._kept


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


@pytest.mark.parametrize("missing", [False, True])
def test_prefetch_corruption_and_missing_shards(tmp_path: Path, missing: bool) -> None:
    root = resharded(tmp_path / "store", 512, 4)
    chunk = root / "near/22/r/c/0"
    if missing:
        chunk.unlink()
    else:
        data = chunk.read_bytes()
        chunk.write_bytes(b"bad!" + data[4:])
    cache = location.ChunkCache(1 << 20)
    array = location.open_group(root, cache=cache)["near/22/r"]
    assert isinstance(array, zarr.Array)
    if missing:
        assert cache.prefetch(array, [0], [512])
        assert (cache.take(array, [0, 511]) == array.fill_value).all()
    else:
        with pytest.raises((RuntimeError, ValueError)):
            cache.prefetch(array, [0], [512])


def test_empty_unsupported_and_key_limited_prefetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = resharded(tmp_path / "store", 512, 4)
    cache = location.ChunkCache(1 << 20)
    group = location.open_group(root, cache=cache)
    near, names = group["near/22/r"], group["variants/alid"]
    assert isinstance(near, zarr.Array)
    assert isinstance(names, zarr.Array)
    assert cache.prefetch(near, [], [])
    assert not cache.prefetch(names, [0], [1])
    monkeypatch.setattr(location, "_PREFETCH_CHUNKS", 1)
    assert not cache.prefetch(near, [0, 1024], [512, 512])
    assert cache._bytes == 0


def test_proxies_read_the_partner_rows_and_nothing_of_near_or_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`rows_of` answers from the ALID index, and from 0.2.0 a variant's proxies are its Partner
    Rows row: no `near/` array of any kind, and none of its metadata (partner-rows §8.3, §8.4;
    ld-store §10). The frames are the one chunk file it reads."""
    names = ogld.open(TEST_STORE)._reader.alids_of(np.array([500, 520])).tolist()
    preads: list[int] = []
    original = os.pread

    def recording_pread(fd: int, length: int, offset: int) -> bytes:
        preads.append(length)
        return original(fd, length, offset)

    monkeypatch.setattr(os, "pread", recording_pread)
    store = ogld.open(TEST_STORE)
    with store_reads() as keys:
        assert (store._reader.rows_of(names) >= 0).all()
        store.proxies(names, r2=0.8, kb=3000)
    assert preads, "a local frame is one pread of the chunk file (partner-rows §7)"
    offending = [key for key in keys if key.startswith("near/")]
    assert offending == [], f"proxies read {offending}"


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


def test_encoded_credit_bounds_actual_buffers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(location, "_FETCH_BYTES", 32)
    budget = location._EncodedBudget()
    held: dict[int, np.ndarray] = {}
    peak = 0

    async def job() -> None:
        nonlocal peak
        await budget.acquire(16)
        data = np.empty(16, dtype=np.uint8)
        held[id(data)] = data
        peak = max(peak, sum(a.nbytes for a in held.values()))
        await asyncio.sleep(0.001)
        held.pop(id(data))
        del data
        await budget.release(16)

    async def run() -> None:
        await asyncio.gather(*(job() for _ in range(10)))

    asyncio.run(run())
    assert peak == 32
    assert budget.held == 0


def test_oversized_sparse_read_streams_without_pinning_a_chromosome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bound, size = 1 << 19, 65536
    array = zarr.create_array(
        store=tmp_path / "a", shape=(32 * size,), dtype="int32", chunks=(size,), shards=(8 * size,)
    )
    array[:] = np.arange(32 * size, dtype=np.int32)
    monkeypatch.setattr(location, "_FETCH_BYTES", bound)
    coords = np.arange(32) * size + 17
    cache = location.ChunkCache(0)
    tracemalloc.start()
    try:
        got = cache.take(array, coords)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    np.testing.assert_array_equal(got, coords)
    assert peak < 4 * bound, f"{peak} bytes pinned for {got.nbytes} requested bytes"
    assert cache._bytes == 0


def test_sliding_windows_and_zero_cache_equal_whole_chromosome(tmp_path: Path) -> None:
    root = resharded(tmp_path / "store", 512, 4)
    store = ogld.open(root)
    rows = np.r_[np.arange(0, 50, 5), np.arange(500, 550, 5)]
    names = store._reader.alids_of(rows).tolist()
    score = np.random.default_rng(40).random(len(rows))
    expected = store.clump(names, score, r2=0.01, kb=3000)
    assert (expected.lead != expected.alid).any(), "the fixture must have joined members"
    for budget in [0, 4096]:
        got = ogld.open(root, cache_bytes=budget).clump(names, score, r2=0.01, kb=3000)
        for name, column in expected.columns().items():
            np.testing.assert_array_equal(got.columns()[name], column)
