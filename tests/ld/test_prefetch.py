"""Bounded warming must not change answers or exceed either byte budget (ogld#40)."""

from __future__ import annotations

import asyncio
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
import zarr
from store_copies import resharded

import ogld
from ogld.store import location


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
    for budget in [0, 4096]:
        got = ogld.open(root, cache_bytes=budget).clump(names, score, r2=0.01, kb=3000)
        for name, column in expected.columns().items():
            np.testing.assert_array_equal(got.columns()[name], column)
