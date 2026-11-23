"""Where an LD Store is, and how its elements are read (ld-store §1, §10).

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
read is decided here, in one place: ogld#8 measured it (`docs/benchmarks/ld/latency.md`), and
ogld#32 tuned it (`docs/benchmarks/ld/fast-reader.md`).

`OpenedStore` is the one object an opened store's facts and its cache are held in, because a
reader and the `VariantTable` it composes are two halves of one store and not two stores
(opengwas/ogld#94): a replaced cache is replaced for both. The reads every other read is made of are
here beside it — `elements_of` for an array's elements at coordinates, `runs_of` for its ragged
runs, both refused if a chunk was lost, `array_indptr` for an offset array read whole, and
`kept_array` for an array whose fill value no valid store can hold (ld-store §6).
"""

from __future__ import annotations

import asyncio
import math
import os
import threading
import weakref
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
import numpy.typing as npt
import zarr
from zarr.abc.store import ByteRequest, RangeByteRequest, Store, SuffixByteRequest
from zarr.core.buffer import Buffer, BufferPrototype, default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage import LocalStore, StorePath, WrapperStore

from opengwasdb.ld.format import InputError, Provenance
from opengwasdb.ld.format.geometry import spans
from opengwasdb.ld.store import layout
from opengwasdb.ld.store.decoding import DecodedChunk, NumericDecoder, on_pool
from opengwasdb.ld.store.layout import never_data, refused_if_lost, shown_fill

#: The decoded inner chunks and shard indexes a reader keeps unless its caller says otherwise,
#: in bytes: 8,192 of layout C's 512 KiB Near Band chunks. ogld#8 measured repeated reads
#: warm at this budget, and not at 256 MiB, which held 512 (owner, 12 November 2026; ogld#32).
CACHE_BYTES: Final = 1 << 32
#: The bytes, as the cache counts memory, one read has in flight at once, in at most zarr's
#: `async.concurrency` (10 by default) pieces of runs: two of layout C's 128 MiB Near Band
#: shards. The bound is per read: reads from several threads each hold a batch (ogld#32).
#: A string chunk is planned at the most a string has held so far (`ChunkCache._chunk_bytes`).
_FETCH_BYTES: Final = 1 << 28


def open_group(location: str | os.PathLike[str], *, cache: ChunkCache | None = None) -> zarr.Group:
    """The root group of the hierarchy at `location`, read-only, whatever its Zarr format,
    with every shard index its arrays read kept in `cache` when one is given (ogld#32).

    A location with no Zarr group at all, or whose root metadata will not parse, is an
    `InputError` naming it. A Zarr v2 group opens,
    so validation can say it is not v3 (ld-store §12, `envelope.arrays`) rather than not there.
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


class FrameSource(Protocol):
    """The bytes of one array's chunk file, read one range at a time (partner-rows §7).

    A reader takes frames through this seam so a local store and a remote one differ in one place:
    `PreadSource` reads a file it holds open, and `RangeSource` asks the store, which over HTTP is
    one Range request.
    """

    def read(self, start: int, stop: int) -> bytes:
        """Bytes [start, stop) of the chunk file, which may be fewer when the file ends early."""
        ...

    def close(self) -> None:
        """Release whatever the source holds, a local file's descriptor included."""
        ...


class PreadSource:
    """A local chunk file read at an offset with `os.pread` (partner-rows §5.4, §7).

    The descriptor is opened by the first read and kept for the reader's life, so a frame is one
    `os.pread` and no zarr call: a local store pays no event-loop hop a frame
    (`scripts/ld/bench/frame_read.py` measures both paths on one file). Opening it lazily is what
    keeps a refusal's context: a file that is missing or unopenable refuses the read that needed it,
    naming the chromosome and the frame the caller asked for, as a remote store's empty Range does.

    `close` marks the source closed, waits for the reads in flight, and only then closes the
    descriptor, so no read ever runs on a closed descriptor — and so never on the descriptor a later
    `open` reused for another file. A read after close refuses. Reads do not serialise each other:
    only the count of them is locked, and the `pread` runs outside the lock.

    A source, and the reader that owns it, is not fork-safe: a process worker opens its own reader,
    because a descriptor and a lock are not inherited across `fork` (partner-rows §7; ogld#71).
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._state = threading.Condition()
        self._descriptor: int | None = None
        self._active = 0
        self._closed = False
        self._finalizer: weakref.finalize[Any, Any] | None = None

    def read(self, start: int, stop: int) -> bytes:
        with self._state:
            if self._closed:
                raise ValueError(f"{self._path} is closed: a source that is closed is not read")
            descriptor = self._descriptor if self._descriptor is not None else self._opened()
            self._active += 1
        try:
            return self._pread(descriptor, start, stop)
        finally:
            with self._state:
                self._active -= 1
                if not self._active:
                    self._state.notify_all()

    def _opened(self) -> int:
        """The descriptor, opened once; the caller holds the state's lock."""
        try:
            descriptor = os.open(self._path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        except OSError as failed:
            raise ValueError(f"{self._path} cannot be opened: {failed}") from failed
        self._descriptor = descriptor
        # A reader dropped without being closed still releases its descriptor: `opengwasdb.ld.open`
        # has no
        # close of its own, so a caller that opens a store per request would otherwise leak one.
        self._finalizer = weakref.finalize(self, _closed, descriptor)
        return descriptor

    def _pread(self, descriptor: int, start: int, stop: int) -> bytes:
        """One positional read on `descriptor`, which no close can reach while it is counted."""
        try:
            return os.pread(descriptor, stop - start, start)
        except OSError as failed:
            raise ValueError(
                f"{self._path}'s bytes [{start}, {stop}) could not be read: {failed}"
            ) from failed

    def close(self) -> None:
        """Refuse later reads, wait for those in flight, and close the descriptor (§7)."""
        with self._state:
            self._closed = True
            while self._active:
                self._state.wait()
            finalizer, self._finalizer = self._finalizer, None
        if finalizer is not None:
            finalizer()


def _closed(descriptor: int) -> None:
    """Close a descriptor, once: the finalizer's, for a source that was never closed."""
    with suppress(OSError):
        os.close(descriptor)


class RangeSource:
    """A chunk file fetched through the array's own store path, one range a request (§7).

    Over HTTP this is one Range request (partner-rows §10); for a store on this disk it is the
    store's own read, which is why `frame_source` gives a local store a `PreadSource` instead. The
    key is the array's metadata's, never `c/0` assumed (`layout.chunk_key`).
    """

    def __init__(self, array: zarr.Array[Any]) -> None:
        self._array = array
        self._key = layout.chunk_key(array)

    def read(self, start: int, stop: int) -> bytes:
        store = self._array.store_path.store
        try:
            fetched = sync(
                store.get(self._key, default_buffer_prototype(), RangeByteRequest(start, stop))
            )
        except ValueError:
            raise
        except Exception as failed:
            if _unsatisfiable(failed):
                # A range that begins past the file's end is unsatisfiable, and the file ends
                # there: the read's bytes are none, as a local `pread` past the end answers. A
                # probe past the last byte is how a store's chunk file is measured (partner-rows
                # §5.1).
                return b""
            raise ValueError(
                f"{self._key} could not be read as bytes [{start}, {stop}): "
                f"{type(failed).__name__}: {failed}"
            ) from failed
        if fetched is None:
            raise ValueError(
                f"{self._key} is not there: a one-chunk array's chunk file the store lost, never a "
                "short read"
            )
        return bytes(fetched.to_bytes())

    def close(self) -> None:
        """Nothing is held: each read is the store's own."""


def _unsatisfiable(error: BaseException) -> bool:
    """Whether a store refused a range that begins past its file's end: HTTP answers one with 416
    (RFC 9110 §15.5.17), and an empty body may answer it too, which reads as no bytes already."""
    seen: BaseException | None = error
    while seen is not None:
        if getattr(seen, "status", None) == 416:
            return True
        seen = seen.__cause__
    return False


def frame_source(array: zarr.Array[Any]) -> FrameSource:
    """The source a one-chunk array's frames are read from (partner-rows §7): `os.pread` on a local
    store's file, held open, and the store's own range request for one that is not on this disk.

    A local store is the one whose chunk file `layout.chunk_file` can name; for any other, that
    resolution is refused and the store answers the range. The path is never assumed to be `c/0`:
    the array's metadata and chunk key encoding say where its chunk is.
    """
    try:
        path = layout.chunk_file(array)
    except ValueError:
        return RangeSource(array)
    return PreadSource(path)


@dataclass
class OpenedStore:
    """An opened LD Store as both halves of its reader see it (ld-store §7, §10; ogld#94).

    The group, the location a refusal names (ld-store §9), the provenance the reads judge rows and
    entries by, and the one chunk cache every read goes through. A reader and the `VariantTable`
    it composes hold this same object rather than copies, so replacing any of the four replaces it
    for both halves: `StoreReader._cache` is this `cache`, and the benchmark harness swaps it to
    measure what a read holds. One store therefore keeps one chunk budget, and a query's positions
    and its pairs are kept under it (ld-store §10).
    """

    root: zarr.Group
    location: str
    provenance: Provenance
    cache: ChunkCache


@dataclass(frozen=True)
class _ShardIndex:
    """The key a shard's index is kept under beside the chunks: the shard's store key, and the
    bytes of its index, which zarr reads as the shard's suffix (ld-store §6). A chunk is kept under
    (array path, chunk number); the two kinds of key are never equal."""

    key: str
    size: int


#: What the cache keeps each thing under: a chunk, or a shard's index.
_Key = tuple[str, int] | _ShardIndex


class ChunkCache:
    """Decoded inner chunks of one store's 1-D arrays, read whole and kept to `budget` bytes,
    the least recently used dropped first.

    Each kept chunk owns its memory, adopted from a direct decode or copied out of a fallback
    run (a view would keep that whole run alive), and counts an upper bound on what is held
    (`_counted`), so the budget bounds the memory kept. One lock guards lookup, insertion
    and eviction, so threads sharing a reader keep one copy of a chunk they both missed.
    The count stays
    the sum of what is kept. Reads, and counting, run outside it.
    """

    def __init__(self, budget: int | None = None) -> None:
        if budget is not None and (
            isinstance(budget, bool) or not isinstance(budget, int) or budget < 0
        ):
            raise InputError(f"cache_bytes is a count of bytes, an int ≥ 0, not {budget!r}")
        self._budget = CACHE_BYTES if budget is None else budget
        self._bytes = 0
        self._kept: OrderedDict[_Key, tuple[npt.NDArray[Any], int]] = OrderedDict()
        #: The most bytes an element has held, as counted, in any chunk of each array read.
        self._rates: dict[str, float] = {}
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
        most = max(1, _FETCH_BYTES // (self._chunk_bytes(array, size) or 1))
        if len(wanted) > most:
            return self._take_bounded(array, at, which, wanted, most)
        chunks = self._chunks(array, wanted, size)
        if len(chunks) == 1:
            single: npt.NDArray[Any] = chunks[0][at - wanted[0] * size]
            return single
        # Joining full cached chunks copied gigabytes per sparse lead even on a cache hit
        # (ogld#40). Gather only requested values, then undo the grouping's permutation.
        order = np.argsort(which, kind="stable")
        offsets = (at % size)[order]
        starts = np.searchsorted(which[order], wanted)
        stops = np.append(starts[1:], len(at))
        grouped = np.concatenate(
            [chunk[offsets[a:b]] for chunk, a, b in zip(chunks, starts, stops, strict=True)]
        )
        found = np.empty_like(grouped)
        found[order] = grouped
        return found

    def _take_bounded(
        self,
        array: zarr.Array[Any],
        at: npt.NDArray[np.int64],
        which: npt.NDArray[np.int64],
        wanted: npt.NDArray[np.int64],
        most: int,
    ) -> npt.NDArray[Any]:
        order = np.argsort(which, kind="stable")
        ordered = which[order]
        result = np.empty(len(at), dtype=array.dtype)
        for start in range(0, len(wanted), most):
            numbers = wanted[start : start + most]
            a = np.searchsorted(ordered, numbers[0])
            b = np.searchsorted(ordered, numbers[-1], side="right")
            asked = order[a:b]
            result[asked] = self.take(array, at[asked])
        return result

    def runs(
        self, array: zarr.Array[Any], first: npt.ArrayLike, length: npt.ArrayLike
    ) -> list[npt.NDArray[Any]]:
        """The runs first[i] … first[i] + length[i] − 1 of a 1-D array, each as one slice of
        the inner chunks that hold it, read whole, once, as `take` reads them.

        A run in one chunk is a read-only view of the kept chunk, and one across chunks is
        joined; no coordinate is built per element. So a region's Near Band rows read as
        contiguous slices, one per row (ld-store §10; ogld#14)."""
        start = np.asarray(first, dtype=np.int64)
        count = np.asarray(length, dtype=np.int64)
        size = int(array.chunks[0])
        some = count > 0
        lo = start[some] // size
        hi = (start[some] + count[some] - 1) // size
        wanted = np.unique(spans(lo, hi - lo + 1))
        chunks = dict(zip(wanted.tolist(), self._chunks(array, wanted, size), strict=True))
        # A run of no elements is an empty array of the array's own dtype, built here rather than
        # read from the store: one per call is a read of the store's metadata at best, and a
        # caller that reads one chunk at a time makes a call per chunk (ogld#42).
        empty = np.empty(0, dtype=array.dtype)
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
        found.update(
            self._fetched_chunks(array, [k for k in wanted.tolist() if k not in found], size)
        )
        return [found[k] for k in wanted.tolist()]

    def _fetched_chunks(
        self, array: zarr.Array[Any], missing: list[int], size: int
    ) -> dict[int, npt.NDArray[Any]]:
        """The chunks `missing`, ascending, fetched a batch at a time and kept."""
        found: dict[int, npt.NDArray[Any]] = {}
        if missing and self._chunk_bytes(array, size) is None:
            # A string array none of whose chunks has been counted: one alone, to count it.
            found.update(self._fetch_batch(array, [(missing[0], missing[0] + 1)], size))
            missing = missing[1:]
        for batch in self._planned(array, missing, size):
            found.update(self._fetch_batch(array, batch, size))
        return found

    def _planned(
        self, array: zarr.Array[Any], missing: list[int], size: int
    ) -> Iterator[list[tuple[int, int]]]:
        """The batches `missing` is fetched in: runs cut where a shard ends, at most zarr's
        `async.concurrency` pieces and `_FETCH_BYTES` as one chunk is planned to hold."""
        per_shard = (array.shards or array.chunks)[0] // size
        most = max(1, _FETCH_BYTES // (self._chunk_bytes(array, size) or 1))
        decoder = NumericDecoder.of(array)
        if decoder is not None:
            # Reserve trimming the padded tail, or non-native byte-order conversion.
            most = max(1, most - 1 if decoder.dtype.isnative else most // 2)
        concurrency = int(zarr.config.get("async.concurrency"))
        return _batches(_pieces(missing, per_shard, most), concurrency, most)

    def _chunk_bytes(self, array: zarr.Array[Any], size: int) -> int | None:
        """What one chunk of `array` is planned to hold in flight: a numeric chunk's buffer,
        which is what it holds. A string chunk's size is not known until it is decoded, so it
        is planned at the most a string has held, as the cache counts it (`_counted`), in any
        chunk of that array counted so far; None before any is."""
        if array.dtype.kind != "T":
            return size * int(array.dtype.itemsize)
        with self._lock:
            rate = self._rates.get(array.path)
        return None if rate is None else math.ceil(rate * size)

    def _fetch_batch(
        self, array: zarr.Array[Any], batch: list[tuple[int, int]], size: int
    ) -> dict[int, npt.NDArray[Any]]:
        """The chunks of one batch, fetched together and kept. Each piece is let go once its
        chunks are copied out, and none outlives the call, so a read holds one batch at a
        time."""
        pieces = _fetched(array, batch, size)
        found: dict[int, npt.NDArray[Any]] = {}
        for first, stop in batch:
            if pieces and isinstance(pieces[0], DecodedChunk):
                for k in range(first, stop):
                    found.update(self._keep_run(array.path, k, k + 1, size, pieces.pop(0)))
            else:
                found.update(self._keep_run(array.path, first, stop, size, pieces.pop(0)))
        return found

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
        """Chunks first … stop − 1, adopted if owned by the decoder, otherwise copied out.

        The ownership token is only made for one chunk, so no neighbouring data is retained.
        """
        copies = (
            [np.asarray(data)]
            if isinstance(data, DecodedChunk)
            else [
                data[(k - first) * size : (k - first + 1) * size].copy() for k in range(first, stop)
            ]
        )
        read = {first + i: (chunk, _counted(chunk)) for i, chunk in enumerate(copies)}
        rate = max(counted / max(1, len(chunk)) for chunk, counted in read.values())
        with self._lock:
            self._rates[path] = max(rate, self._rates.get(path, 0.0))
            return {k: self._keep((path, k), *entry) for k, entry in read.items()}

    @property
    def budget(self) -> int:
        """The bytes this cache keeps at most (ld-store §10)."""
        return self._budget

    def index(self, shard: _ShardIndex) -> bytes | None:
        """The kept index of `shard`, if kept, marked as just used."""
        with self._lock:
            kept = self._kept.get(shard)
            if kept is None:
                return None
            self._kept.move_to_end(shard)
        return kept[0].tobytes()

    def keep_index(self, shard: _ShardIndex, data: bytes) -> bytes:
        """Keep `shard`'s index `data` in the budget, as a chunk is kept, and give it back."""
        held = np.frombuffer(data, dtype=np.uint8).copy()
        with self._lock:
            self._keep(shard, held, held.nbytes)
        return data

    def _keep(self, key: _Key, chunk: npt.NDArray[Any], counted: int) -> npt.NDArray[Any]:
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


def elements_of(
    array: zarr.Array[Any], cache: ChunkCache, coords: npt.NDArray[np.int64]
) -> npt.NDArray[Any]:
    """The elements of `array` at `coords`, through the chunk cache; a lost chunk refused."""
    return refused_if_lost(array, cache.take(array, coords))


def runs_of(
    array: zarr.Array[Any],
    cache: ChunkCache,
    first: npt.NDArray[np.int64],
    length: npt.NDArray[np.int64],
) -> list[npt.NDArray[Any]]:
    """The ragged runs of `array` (ld-store §10), each one slice, through the chunk cache;
    a lost chunk refused."""
    return [refused_if_lost(array, run) for run in cache.runs(array, first, length)]


def array_indptr(array: zarr.Array[Any], cache: ChunkCache) -> npt.NDArray[np.int64]:
    """An offset array read whole, refused if a chunk of it was lost."""
    data = cache.runs(array, [0], [array.shape[0]])[0]
    return refused_if_lost(array, data).astype(np.int64, copy=False)


def kept_array(root: zarr.Group, path: str) -> zarr.Array[Any]:
    """An array, refused unless a chunk it lost would read as a value no store holds there."""
    array = _array(root, path)
    if not never_data(path, array.fill_value).all():
        raise ValueError(
            f"{path} has the fill value {shown_fill(array.fill_value)}, which valid data can hold "
            "there: a chunk it lost would read as data (ld-store §6, envelope.fill)"
        )
    return array


def _array(root: zarr.Group, path: str) -> zarr.Array[Any]:
    """The array at `path`; one the store lacks is refused, naming it, never read as empty."""
    try:
        member = root[path]
    except KeyError:
        raise ValueError(
            f"{path} is missing: every LD Store holds it (ld-store §1, envelope.arrays)"
        ) from None
    assert isinstance(member, zarr.Array)
    return member


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
    one read has in flight at once (ld-store §10; ogld#32)."""
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
    where `zarr.Array` runs each read it is asked for. A piece that fails cancels the others
    of its batch, rather than leaving them fetching, and its own error is raised."""

    decoder = NumericDecoder.of(array) if isinstance(array, zarr.Array) else None

    async def together() -> list[Any]:
        try:
            async with asyncio.TaskGroup() as group:
                reads = [
                    group.create_task(
                        decoder.read(array, a, b)
                        if decoder is not None
                        else array.async_array.getitem(slice(a * size, b * size))
                    )
                    for a, b in pieces
                ]
        except BaseExceptionGroup as failed:
            raise failed.exceptions[0] from failed
        return [read.result() for read in reads]

    fetched = sync(together())
    if decoder is not None:
        return [chunk for piece in fetched for chunk in piece]
    return [np.asarray(data) for data in fetched]


class _KeptIndexes(WrapperStore[Store]):
    """A store whose shard indexes a `ChunkCache` keeps, in its budget (ld-store §6, §10; ogld#32).

    zarr reads a shard's index before the chunks a read needs from it, and again on every
    read: over HTTP, a round trip before the one that fetches the chunks. ld-store §6 puts an LD
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
        self._pending: dict[_ShardIndex, asyncio.Future[bytes | None]] = {}

    def _with_store(self, store: Store) -> _KeptIndexes:
        return type(self)(store, self._cache)

    async def get(
        self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        shard, kept = self._looked_up(key, byte_range)
        if shard is None:
            return await self._get(key, prototype, byte_range)
        if kept is None:
            kept = await self._shared(shard, prototype)
        return _buffer(prototype, kept)

    async def _get(
        self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None
    ) -> Buffer | None:
        if type(self._store) is LocalStore:
            # Its public sync getter has the same missing-file semantics as get(). Sharing
            # the pool prevents the one default asyncio helper from serialising range I/O.
            result: Buffer | None = await on_pool(
                partial(self._store.get_sync, key, prototype=prototype, byte_range=byte_range)
            )
            return result
        return await self._store.get(key, prototype, byte_range)

    def get_sync(
        self,
        key: str,
        *,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        shard, kept = self._looked_up(key, byte_range)
        if shard is None:
            return super().get_sync(key, prototype=prototype, byte_range=byte_range)
        if kept is None:
            fetched = super().get_sync(key, prototype=prototype, byte_range=byte_range)
            kept = self._kept_from(shard, fetched)
        return _buffer(prototype, kept)

    def _looked_up(
        self, key: str, byte_range: ByteRequest | None
    ) -> tuple[_ShardIndex | None, bytes | None]:
        """The shard index a read of `key` asks for, None for any other read, and that index
        as the cache keeps it, None if it does not."""
        if not isinstance(byte_range, SuffixByteRequest):
            return None, None
        shard = _ShardIndex(key, byte_range.suffix)
        return shard, self._cache.index(shard)

    def _kept_from(self, shard: _ShardIndex, fetched: Buffer | None) -> bytes | None:
        """`shard`'s index as fetched, kept; None, and nothing kept, for a missing shard."""
        return None if fetched is None else self._cache.keep_index(shard, fetched.to_bytes())

    async def _shared(self, shard: _ShardIndex, prototype: BufferPrototype) -> bytes | None:
        """`shard`'s index, from the fetch of it in flight if there is one, else fetched and
        kept; None for a missing shard."""
        pending = self._pending.get(shard)
        if pending is None:
            pending = asyncio.ensure_future(self._fetched(shard, prototype))
            self._pending[shard] = pending
            pending.add_done_callback(lambda _: self._pending.pop(shard))
        try:
            return await asyncio.shield(pending)
        except asyncio.CancelledError:
            # Sharing protects other readers, not permission to outlive this read (ADR 0095).
            # Consume a failure while draining, but preserve this waiter's cancellation.
            drained = asyncio.gather(pending, return_exceptions=True)
            while not drained.done():
                with suppress(asyncio.CancelledError):
                    await asyncio.shield(drained)
            raise

    async def _fetched(self, shard: _ShardIndex, prototype: BufferPrototype) -> bytes | None:
        fetched = await self._get(shard.key, prototype, SuffixByteRequest(shard.size))
        return self._kept_from(shard, fetched)


def _buffer(prototype: BufferPrototype | None, kept: bytes | None) -> Buffer | None:
    """A shard index as zarr asks for it, None for a missing shard."""
    if kept is None:
        return None
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

    The cast to fixed-width bytes `width` wide raises on any non-ASCII character. numpy 2.4
    checks every character of a string, even those past `width` that the copy drops; the count
    relies only on the characters the copy keeps. So one cast that does not raise proves every
    string of at most `width` characters ASCII, and a string longer than that counts at UTF-8's
    most, as one that raises counts every string."""
    width = min(int(characters.max(initial=0)), _ASCII_CHECKED)
    try:
        chunk.astype(np.dtype(f"S{max(width, 1)}"))
    except UnicodeEncodeError:
        return _UTF8_MOST
    return np.where(characters <= width, 1, _UTF8_MOST)
