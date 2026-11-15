"""Compressed-byte credits cover real ranges through store I/O and zstd (#40)."""

from __future__ import annotations

import asyncio
import threading
import weakref
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from zarr.abc.store import RangeByteRequest
from zarr.storage import LocalStore

from ogld.store import decoding, location


class Ranges:
    def __init__(self) -> None:
        self.references: list[weakref.ReferenceType[np.ndarray]] = []
        self.sizes: list[int] = []
        self.peak = 0
        self.lock = threading.Lock()

    def held(self) -> int:
        return sum(a.nbytes for ref in self.references if (a := ref()) is not None)

    def record(self, buffer: Any) -> None:
        data = buffer.as_numpy_array()
        with self.lock:
            self.references.append(weakref.ref(data))
            self.sizes.append(data.nbytes)
            self.peak = max(self.peak, self.held())


def test_prefetch_credit_bounds_real_compressed_ranges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    size, bound = 32, 3072
    root = zarr.open_group(tmp_path / "store", mode="w")
    array = root.create_array(
        "r", shape=(48 * size,), chunks=(size,), shards=(16 * size,), dtype="i4"
    )
    source = np.random.default_rng(40).integers(-(1 << 30), 1 << 30, array.shape, dtype=np.int32)
    array[:] = source
    cache = location.ChunkCache(1 << 20)
    array = location.open_group(tmp_path / "store", cache=cache)["r"]
    assert isinstance(array, zarr.Array)
    measured = Ranges()
    _observe_ranges(monkeypatch, measured)
    monkeypatch.setattr(location, "_FETCH_BYTES", bound)
    wanted = np.array([0, 15, 16, 31, 32, 47]) * size
    assert cache.prefetch(array, wanted, np.full(len(wanted), size))
    coords = (wanted[:, None] + np.arange(size)).ravel()
    np.testing.assert_array_equal(cache.take(array, coords), source[coords])
    assert len(measured.sizes) == 3, "three shards must compete for the same allowance"
    assert all(bound // 2 < n <= bound for n in measured.sizes), "two ranges cannot fit together"
    assert measured.peak == max(measured.sizes) <= bound
    assert measured.held() == 0, "no completed I/O Future or decode job may retain a range"


def _observe_ranges(monkeypatch: pytest.MonkeyPatch, measured: Ranges) -> None:
    getter, decode, release = (
        LocalStore.get_sync,
        decoding.NumericDecoder.decode,
        location._EncodedBudget.release,
    )

    def recording(self: Any, key: str, **kwargs: Any) -> Any:
        result = getter(self, key, **kwargs)
        if isinstance(kwargs.get("byte_range"), RangeByteRequest) and result is not None:
            measured.record(result)
        return result

    async def held_decode(self: Any, *args: Any) -> None:
        await asyncio.sleep(0.02)  # Make competing shard requests overlap if credits are broken.
        await decode(self, *args)

    async def released(self: Any, size: int) -> None:
        assert measured.held() == 0, "actual compressed buffers outlived their credit"
        await release(self, size)

    monkeypatch.setattr(LocalStore, "get_sync", recording)
    monkeypatch.setattr(decoding.NumericDecoder, "decode", held_decode)
    monkeypatch.setattr(location._EncodedBudget, "release", released)
