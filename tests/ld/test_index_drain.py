"""A cancelled suffix waiter still owns its shared index I/O until it finishes (#40)."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest
import zarr
from store_copies import resharded
from zarr.abc.store import SuffixByteRequest
from zarr.core.buffer import default_buffer_prototype
from zarr.storage import LocalStore

from ogld.store import decoding, location


async def reached(event: threading.Event) -> None:
    async with asyncio.timeout(2):
        while not event.is_set():
            await asyncio.sleep(0.001)


@pytest.mark.parametrize("cause", ["cancel", "sibling_error"])
def test_prefetch_drains_shared_index_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cause: str
) -> None:
    root = resharded(tmp_path / "store", 512, 4)
    array = location.open_group(root, cache=location.ChunkCache())["near/22/r"]
    assert isinstance(array, zarr.Array)
    decoder = decoding.NumericDecoder.of(array)
    assert decoder is not None
    started, failed, release = threading.Event(), threading.Event(), threading.Event()
    getter = LocalStore.get_sync

    def held(self: Any, key: str, **kwargs: Any) -> Any:
        if isinstance(kwargs.get("byte_range"), SuffixByteRequest):
            if key.endswith("/1"):
                started.set()
                assert release.wait(2), "the test must release its own suffix worker"
            elif cause == "sibling_error":
                assert started.wait(2), "the sibling must already own index I/O"
                failed.set()
                raise OSError("failed sibling suffix")
        return getter(self, key, **kwargs)

    monkeypatch.setattr(LocalStore, "get_sync", held)

    async def run() -> None:
        task = asyncio.create_task(location._prefetched(array, decoder, [0, 4]))
        try:
            await reached(started)
            if cause == "cancel":
                task.cancel()
            else:
                await reached(failed)
            await asyncio.sleep(0.02)
            assert not task.done(), "prefetch returned while its shared suffix worker was blocked"
        finally:
            release.set()
            result = (await asyncio.gather(task, return_exceptions=True))[0]
            await asyncio.gather(*array.store._pending.values(), return_exceptions=True)
        expected = asyncio.CancelledError if cause == "cancel" else OSError
        assert isinstance(result, expected)
        assert not array.store._pending

    asyncio.run(run())


@pytest.mark.parametrize("fails", [False, True])
def test_cancelled_index_waiter_does_not_cancel_another_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fails: bool
) -> None:
    root = resharded(tmp_path / "store", 512, 4)
    store = location._KeptIndexes(LocalStore(root, read_only=True), location.ChunkCache())
    started, release = threading.Event(), threading.Event()
    getter, calls = LocalStore.get_sync, []

    def held(self: Any, key: str, **kwargs: Any) -> Any:
        calls.append(key)
        started.set()
        assert release.wait(2), "the test must release its own suffix worker"
        if fails:
            raise OSError("shared suffix failed")
        return getter(self, key, **kwargs)

    monkeypatch.setattr(LocalStore, "get_sync", held)

    async def run() -> None:
        request = ("near/22/r/c/0", default_buffer_prototype(), SuffixByteRequest(68))
        first = asyncio.create_task(store.get(*request))
        second = asyncio.create_task(store.get(*request))
        try:
            await reached(started)
            first.cancel()
            await asyncio.sleep(0.01)
            first.cancel()  # Repeated cancellation must not reach the shared fetch either.
            await asyncio.sleep(0.01)
            assert not first.done(), "the cancelled waiter must drain, not abandon, shared I/O"
            assert not second.done()
        finally:
            release.set()
            a, b = await asyncio.gather(first, second, return_exceptions=True)
            await asyncio.gather(*store._pending.values(), return_exceptions=True)
        assert isinstance(a, asyncio.CancelledError)
        if fails:
            assert isinstance(b, OSError)
        else:
            assert b.to_bytes() == (root / request[0]).read_bytes()[-68:]
        assert calls == [request[0]], "both readers must share exactly one suffix request"
        assert not store._pending

    asyncio.run(run())
