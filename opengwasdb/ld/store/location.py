"""Where an LD Store is, and how its elements are read (spec §1, §10).

A store is a local path or an HTTP(S) URL, opened as one Zarr hierarchy. A reader and
validation open it the same way, so a store served from a static host that honours byte
ranges reads exactly as it does from disk. zarr opens a URL through fsspec's HTTP filesystem
(aiohttp). Nothing here lists a directory: every node is found by its own metadata document,
which a static host serves like any other file.

Every element a reader reads goes through a `ChunkCache`. It reads whole inner chunks, which
is what Zarr fetches and decodes in any case, keeps the decoded chunks and the shard indexes
up to a byte budget, and serves each request from them. So a query that reads a few pairs at a
time, as clumping does once per lead, fetches each chunk once, from disk or over HTTP. A read's
missing runs of chunks are fetched together, a bounded batch at a time (ogld#32). How a store is
read is decided here, in one place, which ogld#8 and ogld#32 measured over HTTP.
"""

from __future__ import annotations

import asyncio
import math
import os
import threading
from collections import OrderedDict
from collections.abc import Iterator
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import zarr
from zarr.abc.store import ByteRequest, Store, SuffixByteRequest
from zarr.core.buffer import Buffer, BufferPrototype, default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage import StorePath, WrapperStore

from ogld.format import InputError
from ogld.format.geometry import spans

#: The decoded inner chunks and shard indexes a reader keeps unless its caller says otherwise,
#: in bytes: 8,192 of layout C's 512 KiB Near Band chunks. ogld#8 measured repeated reads
#: warm at this budget, and not at 256 MiB, which held 512 (owner, 12 November 2026; ogld#32).
CACHE_BYTES: Final = 1 << 32
#: The decoded bytes one read has in flight at once, beside at most zarr's `async.concurrency`
#: (10 by default) pieces of runs: two of layout C's 128 MiB Near Band shards (ogld#32).
_FETCH_BYTES: Final = 1 << 28


def open_group(location: str | os.PathLike[str], *, cache: ChunkCache | None = None) -> zarr.Group:
    """The root group of the hierarchy at `location`, read-only, whatever its Zarr format,
    with every shard index its arrays read kept in `cache` when one is given (ogld#32).

    A location with no Zarr group at all, or whose root metadata will not parse, is an
    `InputError` naming it. A Zarr v2 group opens,
    so validation can say it is not v3 (spec §12, `envelope.arrays`) rather than not there.
    """
    root = _opened(location)
    if cache is None:
        return root
    # zarr chose the store for `location`, as it chooses for any location; the same group is
    # put on that store, wrapped, from the metadata already read, so wrapping reads nothing.
    kept = _KeptIndexes(root.store, cache)
    return zarr.Group(zarr.AsyncGroup(root.metadata, StorePath(kept, root.path)))


def _opened(location: str | os.PathLike[str]) -> zarr.Group:
    """`open_group` of `location` as zarr opens it, its failures named."""
    try:
        return zarr.open_group(str(location), mode="r")
    except FileNotFoundError as missing:
        raise InputError(f"no LD Store at {location}: there is no Zarr group there") from missing
    except (ValueError, TypeError) as unreadable:
        raise InputError(
            f"{location}'s metadata cannot be read as a Zarr group: "
            f"{type(unreadable).__name__}: {unreadable}"
        ) from unreadable


class ChunkCache:
    """Decoded inner chunks of one store's 1-D arrays, read whole and kept to `budget` bytes,
    the least recently used dropped first.

    Each kept chunk owns its memory, copied out of the read that fetched it (a view would keep
    that whole read alive), and counts an upper bound on what is held (`_counted`), so the
    budget is a bound on the memory kept. One lock guards every lookup, insertion and eviction,
    so threads sharing a reader keep one copy of a chunk they both missed, and the count stays
    the sum of what is kept. Reads, and counting, run outside it.
    """

    def __init__(self, budget: int | None = None) -> None:
        if budget is not None and (
            isinstance(budget, bool) or not isinstance(budget, int) or budget < 0
        ):
            raise InputError(f"cache_bytes is a count of bytes, an int ≥ 0, not {budget!r}")
        self._budget = CACHE_BYTES if budget is None else budget
        self._bytes = 0
        self._kept: OrderedDict[tuple[str, int], tuple[npt.NDArray[Any], int]] = OrderedDict()
        self._lock = threading.Lock()

    def take(self, array: zarr.Array[Any], coords: npt.ArrayLike) -> npt.NDArray[Any]:
        """The elements of a 1-D array at `coords`, in that order, from the inner chunks that
        hold them: each read whole, once, the missing ones fetched together (`_chunks`)."""
        at = np.asarray(coords, dtype=np.int64)
        if len(at) == 0:
            return np.asarray(array[0:0])
        size = int(array.chunks[0])
        which = at // size
        wanted = np.unique(which)
        chunks = self._chunks(array, wanted, size)
        start = np.concatenate([[0], np.cumsum([len(c) for c in chunks])])
        joined = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        found: npt.NDArray[Any] = joined[
            start[np.searchsorted(wanted, which)] + (at - which * size)
        ]
        return found

    def runs(
        self, array: zarr.Array[Any], first: npt.ArrayLike, length: npt.ArrayLike
    ) -> list[npt.NDArray[Any]]:
        """The runs first[i] … first[i] + length[i] − 1 of a 1-D array, each as one slice of
        the inner chunks that hold it, read whole, once, as `take` reads them.

        A run in one chunk is a read-only view of the kept chunk, and one across chunks is
        joined; no coordinate is built per element. So a region's Near Band rows read as
        contiguous slices, one per row (spec §10; ogld#14)."""
        start = np.asarray(first, dtype=np.int64)
        count = np.asarray(length, dtype=np.int64)
        size = int(array.chunks[0])
        some = count > 0
        lo = start[some] // size
        hi = (start[some] + count[some] - 1) // size
        wanted = np.unique(spans(lo, hi - lo + 1))
        chunks = dict(zip(wanted.tolist(), self._chunks(array, wanted, size), strict=True))
        empty = np.asarray(array[0:0])
        return [
            _run(chunks, s, n, size) if n else empty
            for s, n in zip(start.tolist(), count.tolist(), strict=True)
        ]

    def _chunks(
        self, array: zarr.Array[Any], wanted: npt.NDArray[np.int64], size: int
    ) -> list[npt.NDArray[Any]]:
        """The decoded chunks `wanted`, in order: kept ones from memory, the rest fetched
        together, a batch of runs at a time (ogld#32)."""
        found = self._kept_chunks(array.path, wanted.tolist())
        missing = [k for k in wanted.tolist() if k not in found]
        per_shard = (array.shards or array.chunks)[0] // size
        most = max(1, _FETCH_BYTES // (size * array.dtype.itemsize))
        concurrency = int(zarr.config.get("async.concurrency"))
        for batch in _batches(_pieces(missing, per_shard, most), concurrency, most):
            for (first, stop), data in zip(batch, _fetched(array, batch, size), strict=True):
                found.update(self._keep_run(array.path, first, stop, size, data))
        return [found[k] for k in wanted.tolist()]

    def _kept_chunks(self, path: str, wanted: list[int]) -> dict[int, npt.NDArray[Any]]:
        """Those of the chunks `wanted` that are kept, each marked as just used."""
        found: dict[int, npt.NDArray[Any]] = {}
        with self._lock:
            for k in wanted:
                kept = self._kept.get((path, k))
                if kept is not None:
                    found[k] = kept[0]
                    self._kept.move_to_end((path, k))
        return found

    def _keep_run(
        self, path: str, first: int, stop: int, size: int, data: npt.NDArray[Any]
    ) -> dict[int, npt.NDArray[Any]]:
        """Chunks first … stop − 1 of the array at `path`, fetched as `data`, each copied out
        and kept."""
        copies = [
            data[(k - first) * size : (k - first + 1) * size].copy() for k in range(first, stop)
        ]
        read = {first + i: (chunk, _counted(chunk)) for i, chunk in enumerate(copies)}
        with self._lock:
            return {k: self._keep((path, k), *entry) for k, entry in read.items()}

    def index(self, key: str, size: int) -> bytes | None:
        """The kept index of the shard at store key `key`, its last `size` bytes, if kept,
        marked as just used. A shard index is kept beside the chunks, as (`key`, −`size`)."""
        with self._lock:
            kept = self._kept.get((key, -size))
            if kept is None:
                return None
            self._kept.move_to_end((key, -size))
        return kept[0].tobytes()

    def keep_index(self, key: str, size: int, data: bytes) -> bytes:
        """Keep the index `data` of the shard at store key `key`, its last `size` bytes, in
        the budget as a chunk is kept, and give it back."""
        held = np.frombuffer(data, dtype=np.uint8).copy()
        with self._lock:
            self._keep((key, -size), held, held.nbytes)
        return data

    def _keep(
        self, key: tuple[str, int], chunk: npt.NDArray[Any], counted: int
    ) -> npt.NDArray[Any]:
        """Keep `chunk`, or the copy another thread kept first; the caller holds the lock."""
        kept = self._kept.get(key)
        if kept is not None:
            self._kept.move_to_end(key)
            return kept[0]
        self._kept[key] = (chunk, counted)
        self._bytes += counted
        while self._bytes > self._budget and self._kept:
            _, (_, dropped) = self._kept.popitem(last=False)
            self._bytes -= dropped
        return chunk


def _pieces(missing: list[int], per_shard: int, most: int) -> list[tuple[int, int]]:
    """The runs of neighbouring chunks among `missing`, ascending, each cut where a shard ends
    and at `most` chunks: (first, stop) of each piece, one shard's chunks each."""
    pieces: list[tuple[int, int]] = []
    for run in np.split(np.asarray(missing, dtype=np.int64), _breaks(missing)):
        first, stop = (int(run[0]), int(run[-1]) + 1) if len(run) else (0, 0)
        while first < stop:
            end = min(stop, (first // per_shard + 1) * per_shard, first + most)
            pieces.append((first, end))
            first = end
    return pieces


def _breaks(missing: list[int]) -> npt.NDArray[np.intp]:
    """Where a run of neighbouring chunks ends among the ascending `missing`."""
    return np.flatnonzero(np.diff(np.asarray(missing, dtype=np.int64)) != 1) + 1


def _batches(
    pieces: list[tuple[int, int]], concurrency: int, most: int
) -> Iterator[list[tuple[int, int]]]:
    """`pieces` in order, as batches of at most `concurrency` pieces and `most` chunks: what
    one read has in flight at once (spec §10; ogld#32)."""
    batch: list[tuple[int, int]] = []
    held = 0
    for first, stop in pieces:
        if batch and (len(batch) == concurrency or held + stop - first > most):
            yield batch
            batch, held = [], 0
        batch.append((first, stop))
        held += stop - first
    if batch:
        yield batch


def _fetched(
    array: zarr.Array[Any], pieces: list[tuple[int, int]], size: int
) -> list[npt.NDArray[Any]]:
    """The elements of each piece of `array`'s chunks, fetched together on zarr's event loop,
    where `zarr.Array` runs each read it is asked for."""

    async def together() -> list[Any]:
        reads = (array.async_array.getitem(slice(a * size, b * size)) for a, b in pieces)
        return list(await asyncio.gather(*reads))

    return [np.asarray(data) for data in sync(together())]


class _KeptIndexes(WrapperStore[Store]):
    """A store whose shard indexes a `ChunkCache` keeps, in its budget (spec §6, §10; ogld#32).

    zarr reads a shard's index before the chunks a read needs from it, and again on every
    read: over HTTP, a round trip before the one that fetches the chunks. Spec §6 puts an LD
    Store's shard index at the shard's end, which zarr reads as a suffix of the shard, and
    reads nothing else that way. So each suffix read is answered from the cache when it holds
    it, and kept when it is fetched; every other read passes through. A missing shard is not
    kept, and reads as zarr reads it, its array's fill value, which the reader refuses.

    A read fetches its runs together, so two runs of one shard can both miss its index at once.
    The first fetches it and the second awaits that fetch (`_pending`), so the index is fetched
    once. zarr runs every asynchronous read on its one event loop, so `_pending` is only ever
    touched from that loop's thread.
    """

    def __init__(self, store: Store, cache: ChunkCache) -> None:
        super().__init__(store)
        self._cache = cache
        self._pending: dict[tuple[str, int], asyncio.Future[bytes | None]] = {}

    def _with_store(self, store: Store) -> _KeptIndexes:
        return type(self)(store, self._cache)

    async def get(
        self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        if not isinstance(byte_range, SuffixByteRequest):
            return await self._store.get(key, prototype, byte_range)
        kept = self._cache.index(key, byte_range.suffix)
        if kept is None:
            kept = await self._shared(key, prototype, byte_range)
        return None if kept is None else prototype.buffer.from_bytes(kept)

    async def _shared(
        self, key: str, prototype: BufferPrototype, byte_range: SuffixByteRequest
    ) -> bytes | None:
        """The index at `key`, from the fetch of it in flight if there is one, else fetched
        and kept; None for a missing shard."""
        pending = self._pending.get((key, byte_range.suffix))
        if pending is None:
            pending = asyncio.ensure_future(self._fetched(key, prototype, byte_range))
            self._pending[(key, byte_range.suffix)] = pending
            pending.add_done_callback(lambda _: self._pending.pop((key, byte_range.suffix)))
        return await asyncio.shield(pending)

    async def _fetched(
        self, key: str, prototype: BufferPrototype, byte_range: SuffixByteRequest
    ) -> bytes | None:
        fetched = await self._store.get(key, prototype, byte_range)
        if fetched is None:
            return None
        return self._cache.keep_index(key, byte_range.suffix, fetched.to_bytes())

    def get_sync(
        self,
        key: str,
        *,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        if not isinstance(byte_range, SuffixByteRequest):
            return super().get_sync(key, prototype=prototype, byte_range=byte_range)
        kept = self._cache.index(key, byte_range.suffix)
        if kept is None:
            fetched = super().get_sync(key, prototype=prototype, byte_range=byte_range)
            if fetched is None:
                return None
            kept = self._cache.keep_index(key, byte_range.suffix, fetched.to_bytes())
        return (prototype or default_buffer_prototype()).buffer.from_bytes(kept)


def _run(
    chunks: dict[int, npt.NDArray[Any]], start: int, length: int, size: int
) -> npt.NDArray[Any]:
    """Elements start … start + length − 1 from the decoded `chunks` of `size` elements each."""
    stop = start + length
    pieces = [
        chunks[k][max(start, k * size) - k * size : min(stop, (k + 1) * size) - k * size]
        for k in range(start // size, (stop - 1) // size + 1)
    ]
    run = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
    view = run.view()
    view.flags.writeable = False
    return view


#: numpy's variable-width strings (StringDType, what zarr returns for a string array) keep a
#: string of at most 15 UTF-8 bytes inside the array, and a longer one in an arena beside it:
#: its bytes after a size header of at most 8, in an arena that grows by a quarter at a time,
#: plus the array's own allocator, measured at 328 bytes. `nbytes` counts none of the arena.
_INLINE: Final = 15
_HEADER: Final = 8
_GROWTH: Final = 1.25
_ARENA_FIXED: Final = 512
#: UTF-8 takes at most 4 bytes a character. The count proves a string ASCII, a byte a
#: character, over at most its first `_ASCII_CHECKED` characters, in one fixed-width copy of
#: the chunk that many bytes wide: 4 MiB for 16,384 strings (ogld#32).
_UTF8_MOST: Final = 4
_ASCII_CHECKED: Final = 256


def _counted(chunk: npt.NDArray[Any]) -> int:
    """An upper bound on the bytes `chunk` holds: a numeric chunk's buffer, and for a string
    chunk its buffer and arena. Measured against tracemalloc on 72 chunks of 1–16,384 strings
    of 1–20,000 characters, ASCII and not, the count was never below what a chunk held.

    No string is visited in Python (ogld#32): numpy counts each string's characters, and an
    ASCII chunk's strings take a byte each, so it counts their bytes exactly. Any other string
    counts 4 bytes a character, UTF-8's most: an upper bound, and no ALID or allele the engine
    forms needs it."""
    if chunk.dtype.kind != "T":
        return int(chunk.nbytes)
    characters = np.strings.str_len(chunk).astype(np.int64)
    utf8 = characters * _bytes_per_character(chunk, characters)
    long = utf8[utf8 > _INLINE]
    arena = math.ceil(_GROWTH * (int(long.sum()) + _HEADER * len(long))) + _ARENA_FIXED
    return int(chunk.nbytes) + arena


def _bytes_per_character(
    chunk: npt.NDArray[Any], characters: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64] | int:
    """1 for each string of `chunk` proved ASCII, and `_UTF8_MOST` for every other.

    numpy casts variable-width strings to fixed-width bytes only when every character it keeps
    is ASCII, and raises otherwise; the cast keeps each string's first `width` characters. So
    one cast proves every string of at most `width` characters ASCII, or proves nothing."""
    width = min(int(characters.max(initial=0)), _ASCII_CHECKED)
    try:
        chunk.astype(np.dtype(f"S{max(width, 1)}"))
    except UnicodeEncodeError:
        return _UTF8_MOST
    return np.where(characters <= width, 1, _UTF8_MOST)
