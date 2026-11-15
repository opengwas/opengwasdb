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
read is decided here, in one place: ogld#8 measured it (`docs/benchmarks/latency.md`), and
ogld#32 tuned it (`docs/benchmarks/fast-reader.md`).
"""

from __future__ import annotations

import asyncio
import importlib
import math
import os
import threading
from collections import OrderedDict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt
import zarr
from zarr.abc.store import ByteRequest, RangeByteRequest, Store, SuffixByteRequest
from zarr.codecs import BytesCodec, Crc32cCodec, ShardingCodec, ZstdCodec
from zarr.core.buffer import Buffer, BufferPrototype, default_buffer_prototype
from zarr.core.metadata.v3 import ArrayV3Metadata, RegularChunkGridMetadata
from zarr.core.sync import sync
from zarr.storage import LocalStore, StorePath, WrapperStore

from ogld.format import InputError
from ogld.format.geometry import spans

#: The decoded inner chunks and shard indexes a reader keeps unless its caller says otherwise,
#: in bytes: 8,192 of layout C's 512 KiB Near Band chunks. ogld#8 measured repeated reads
#: warm at this budget, and not at 256 MiB, which held 512 (owner, 12 November 2026; ogld#32).
CACHE_BYTES: Final = 1 << 32
#: The bytes, as the cache counts memory, one read has in flight at once, in at most zarr's
#: `async.concurrency` (10 by default) pieces of runs: two of layout C's 128 MiB Near Band
#: shards. The bound is per read: reads from several threads each hold a batch (ogld#32).
#: A string chunk is planned at the most a string has held so far (`ChunkCache._chunk_bytes`).
_FETCH_BYTES: Final = 1 << 28
#: Bound planning metadata even for a store re-chunked to tiny elements (ADR 0009).
_PREFETCH_CHUNKS: Final = 1 << 16


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


@dataclass(frozen=True)
class _ShardIndex:
    """The key a shard's index is kept under beside the chunks: the shard's store key, and the
    bytes of its index, which zarr reads as the shard's suffix (spec §6). A chunk is kept under
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

    def prefetch(self, array: zarr.Array[Any], first: npt.ArrayLike, length: npt.ArrayLike) -> bool:
        """Warm all candidate chunks only if their buffers and indexes fit the cache.

        Planning merges chunk intervals without materialising a coordinate per pair. A
        refusal fetches nothing; the reader then warms bounded positional windows instead.
        """
        decoder = _NumericDecoder.of(array)
        if decoder is None:
            return False
        wanted = _prefetch_plan(array, decoder, first, length, self._budget)
        if wanted is None:
            return False
        kept = self._kept_chunks(array.path, wanted)
        missing = [k for k in wanted if k not in kept]
        most = max(1, _FETCH_BYTES // (decoder.size * array.dtype.itemsize) // 2)
        for start in range(0, len(missing), most):
            numbers = missing[start : start + most]
            chunks = sync(_prefetched(array, decoder, numbers))
            for k, chunk in zip(numbers, chunks, strict=True):
                self._keep_run(array.path, k, k + 1, decoder.size, chunk)
        return True

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
        decoder = _NumericDecoder.of(array)
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
            if pieces and isinstance(pieces[0], _DecodedChunk):
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
            if isinstance(data, _DecodedChunk)
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
        """The bytes this cache keeps at most (spec §10)."""
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


def _prefetch_plan(
    array: zarr.Array[Any],
    decoder: _NumericDecoder,
    first: npt.ArrayLike,
    length: npt.ArrayLike,
    budget: int,
) -> list[int] | None:
    lo, hi = _chunk_intervals(first, length, decoder.size)
    count = int((hi - lo + 1).sum())
    # Counting every shard between the ends is conservative for sparse runs.
    shards = int(hi[-1] // decoder.per_shard - lo[0] // decoder.per_shard + 1) if len(lo) else 0
    needed = count * decoder.size * array.dtype.itemsize + shards * (decoder.per_shard * 16 + 4)
    if needed > budget or count > _PREFETCH_CHUNKS:
        return None
    wanted: list[int] = spans(lo, hi - lo + 1).tolist()
    return wanted


def _chunk_intervals(
    first: npt.ArrayLike, length: npt.ArrayLike, size: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    start, count = np.asarray(first, dtype=np.int64), np.asarray(length, dtype=np.int64)
    some = count > 0
    lo, hi = start[some] // size, (start[some] + count[some] - 1) // size
    order = np.argsort(lo, kind="stable")
    lo, hi = lo[order], np.maximum.accumulate(hi[order])
    begins = np.r_[0, np.flatnonzero(lo[1:] > hi[:-1] + 1) + 1]
    ends = np.r_[begins[1:] - 1, len(lo) - 1]
    return (lo[begins], hi[ends]) if len(lo) else (lo, hi)


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
    where `zarr.Array` runs each read it is asked for. A piece that fails cancels the others
    of its batch, rather than leaving them fetching, and its own error is raised."""

    decoder = _NumericDecoder.of(array) if isinstance(array, zarr.Array) else None

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


# Local range I/O and numeric decoding share seven workers, not two nested pools.
# Zarr's separately configured helper handles fallback codecs (ADR 0009, round 2).
# numcodecs is a runtime dependency without type stubs, as in build.inputs.
_NUMCODECS = importlib.import_module("numcodecs")
_DECODE_WORKERS: Final = 7
_DECODE_POOL = ThreadPoolExecutor(max_workers=_DECODE_WORKERS, thread_name_prefix="ogld_decode")


def _reset_decode_pool() -> None:
    """A fork inherits no live executor threads; it must not submit to their old queue."""
    global _DECODE_POOL
    _DECODE_POOL = ThreadPoolExecutor(max_workers=_DECODE_WORKERS, thread_name_prefix="ogld_decode")


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_decode_pool)


type _ChunkRanges = list[tuple[int, int, int]]
type _Decoded = list[npt.NDArray[Any] | None]


class _DecodedChunk(np.ndarray[Any, np.dtype[Any]]):
    """Ownership token: one decoded chunk's buffer, never a view of a multi-chunk run.

    Only the direct decoder constructs these; the cache can adopt them without copying,
    unlike a zarr run whose views retain uncounted neighbouring chunks (ADR 0009).
    """


@dataclass(frozen=True)
class _NumericDecoder:
    """Only the declared bytes/zstd chain, with a bytes/CRC32C suffix index (ADR 0009).

    No zarr private codec or index API is called. Offsets are absolute uint64 byte offsets,
    paired with lengths, and both all-ones means absent, as sharding_indexed specifies.
    """

    size: int
    per_shard: int
    dtype: np.dtype[Any]
    index_dtype: np.dtype[Any]

    @classmethod
    def of(cls, array: zarr.Array[Any]) -> _NumericDecoder | None:
        """Recognise the exact codec chain, not a path or a writer's default shape."""
        shard = _numeric_shard(array)
        if shard is None:
            return None
        inner, index = shard.codecs, shard.index_codecs
        if not (_bytes_chain(inner, ZstdCodec) and _bytes_chain(index, Crc32cCodec)):
            return None
        if array.shards is None:
            return None
        size = int(array.chunks[0])
        return cls(
            size,
            int(array.shards[0]) // size,
            array.dtype.newbyteorder(_endian(inner[0])),
            np.dtype("u8").newbyteorder(_endian(index[0])),
        )

    async def read(self, array: zarr.Array[Any], first: int, stop: int) -> list[npt.NDArray[Any]]:
        """One planned piece: the index, only its chunks' ranges, then bounded decode jobs."""
        key = array.metadata.encode_chunk_key((first // self.per_shard,))
        path = array.store_path / key
        prototype = default_buffer_prototype()
        index = await path.get(prototype, SuffixByteRequest(self.per_shard * 16 + 4))
        out: _Decoded = [None] * (stop - first)
        if index is not None:
            entries = self._index(index.to_bytes())[
                first % self.per_shard : first % self.per_shard + stop - first
            ]
            for lo, hi, chunks in _byte_runs(entries):
                fetched = await path.get(prototype, RangeByteRequest(lo, hi))
                if fetched is not None:
                    await self._decode(fetched.to_bytes(), lo, chunks, out)
        return [self._owned(array, first + k, chunk) for k, chunk in enumerate(out)]

    def _owned(
        self, array: zarr.Array[Any], k: int, chunk: npt.NDArray[Any] | None
    ) -> npt.NDArray[Any]:
        length = min(self.size, int(array.shape[0]) - k * self.size)
        if chunk is None:
            chunk = np.full(length, array.fill_value, dtype=array.dtype)
        elif length != self.size:
            # A truncated view would keep the padded tail outside the counted budget.
            chunk = chunk[:length].copy()
        return chunk.view(_DecodedChunk)

    def _index(self, data: bytes) -> npt.NDArray[np.uint64]:
        expected = self.per_shard * 16 + 4
        if len(data) != expected:
            raise ValueError(
                f"shard index is truncated or corrupt: {len(data)} bytes, expected {expected}"
            )
        raw = _NUMCODECS.CRC32C().decode(data)
        return np.frombuffer(raw, dtype=self.index_dtype).reshape(self.per_shard, 2)

    async def _decode(
        self, data: bytes | memoryview[int], offset: int, chunks: _ChunkRanges, out: _Decoded
    ) -> None:
        """At most seven jobs per range. Drain running jobs even if another piece fails.

        Cancelling a future cannot stop zstd on a worker. Shielding and draining keeps that
        worker's buffers inside this read's lifetime and in-flight bound (ADR 0009).
        """
        jobs = [
            asyncio.create_task(
                _on_pool(partial(self._into, data, offset, chunks[i::_DECODE_WORKERS], out))
            )
            for i in range(min(_DECODE_WORKERS, len(chunks)))
        ]
        pending = asyncio.gather(*jobs, return_exceptions=True)
        try:
            results = await asyncio.shield(pending)
        except asyncio.CancelledError:
            await pending
            raise
        for result in results:
            if isinstance(result, BaseException):
                raise result

    def _into(
        self, data: bytes | memoryview[int], offset: int, chunks: _ChunkRanges, out: _Decoded
    ) -> None:
        codec = _NUMCODECS.Zstd()
        for k, start, length in chunks:
            frame = memoryview(data)[start - offset : start - offset + length]
            out[k] = self._frame(codec, frame)

    def _frame(self, codec: Any, frame: memoryview) -> npt.NDArray[Any]:
        expected = self.size * self.dtype.itemsize
        declared = _frame_size(frame)
        if declared is None:
            raw = codec.decode(frame)
            if len(raw) != expected:
                raise ValueError(f"decoded chunk has {len(raw)} bytes, expected {expected}")
            return np.frombuffer(raw, dtype=self.dtype).astype(
                self.dtype.newbyteorder("="), copy=False
            )
        if declared != expected:
            raise ValueError(f"decoded chunk declares {declared} bytes, expected {expected}")
        out = np.empty(self.size, dtype=self.dtype.newbyteorder("="))
        codec.decode(frame, out=out)
        if not self.dtype.isnative:
            out.byteswap(inplace=True)
        return out


class _EncodedBudget:
    """Compressed prefetch ranges: one shared allowance, not one allowance per task."""

    def __init__(self) -> None:
        self.held = 0
        self.condition = asyncio.Condition()

    async def acquire(self, size: int) -> None:
        async with self.condition:
            await self.condition.wait_for(
                lambda: self.held == 0 or self.held + size <= _FETCH_BYTES
            )
            self.held += size

    async def release(self, size: int) -> None:
        async with self.condition:
            self.held -= size
            self.condition.notify_all()


async def _prefetched(
    array: zarr.Array[Any], decoder: _NumericDecoder, numbers: list[int]
) -> list[npt.NDArray[Any]]:
    budget = _EncodedBudget()
    semaphore = asyncio.Semaphore(int(zarr.config.get("async.concurrency")))
    out: _Decoded = [None] * len(numbers)
    groups = np.split(
        np.arange(len(numbers)), np.flatnonzero(np.diff(np.array(numbers) // decoder.per_shard)) + 1
    )
    try:
        async with asyncio.TaskGroup() as tasks:
            for group in groups:
                if len(group):
                    tasks.create_task(
                        _prefetch_shard(array, decoder, numbers, group, out, budget, semaphore)
                    )
    except BaseExceptionGroup as failed:
        raise failed.exceptions[0] from failed
    return [decoder._owned(array, k, chunk) for k, chunk in zip(numbers, out, strict=True)]


async def _prefetch_shard(
    array: zarr.Array[Any],
    decoder: _NumericDecoder,
    numbers: list[int],
    group: npt.NDArray[np.int64],
    out: _Decoded,
    budget: _EncodedBudget,
    semaphore: asyncio.Semaphore,
) -> None:
    keys = np.array(numbers, dtype=np.int64)[group]
    path = array.store_path / array.metadata.encode_chunk_key((int(keys[0]) // decoder.per_shard,))
    async with semaphore:
        index = await path.get(
            default_buffer_prototype(), SuffixByteRequest(decoder.per_shard * 16 + 4)
        )
    if index is None:
        return
    entries = decoder._index(index.to_bytes())[keys % decoder.per_shard]
    for lo, hi, chunks in _byte_runs(entries, gap=1 << 19):
        await budget.acquire(hi - lo)
        try:
            async with semaphore:
                data = await path.get(default_buffer_prototype(), RangeByteRequest(lo, hi))
            if data is not None:
                view = memoryview(data.as_numpy_array()).cast("B")
                places = [(int(group[k]), a, n) for k, a, n in chunks]
                await decoder._decode(view, lo, places, out)
                del view
            del data
        finally:
            await budget.release(hi - lo)


def _frame_size(frame: memoryview) -> int | None:
    """RFC 8878 §3.1.1: only a known content size permits destination-only decoding.

    numcodecs accepts short output when given a larger destination. Unknown-size, invalid
    or truncated headers therefore retain the allocating decode and exact-length check.
    """
    if len(frame) < 6 or bytes(frame[:4]) != b"\x28\xb5\x2f\xfd":
        return None
    descriptor = frame[4]
    single = (descriptor >> 5) & 1
    length = (0, 2, 4, 8)[descriptor >> 6] or single
    start = 5 + (not single) + (0, 1, 2, 4)[descriptor & 3]
    if descriptor & 8 or not length or len(frame) < start + length:
        return None
    return int.from_bytes(frame[start : start + length], "little") + (256 if length == 2 else 0)


async def _on_pool(call: Any) -> Any:
    """A cancelled read still owns its worker's buffers until that worker returns."""

    result: list[Any] = []

    def run() -> None:
        nonlocal call
        try:
            result.append(call())
        finally:
            # Neither a retained work item nor its completed Future may keep buffers
            # after their byte credit is released: the Future's result is always None.
            call = None

    pending = asyncio.get_running_loop().run_in_executor(_DECODE_POOL, run)
    try:
        await asyncio.shield(pending)
    except asyncio.CancelledError:
        await pending
        result.clear()
        raise
    return result.pop()


def _numeric_shard(array: zarr.Array[Any]) -> ShardingCodec | None:
    meta = array.metadata
    if not isinstance(meta, ArrayV3Metadata) or array.ndim != 1:
        return None
    if not isinstance(meta.chunk_grid, RegularChunkGridMetadata):
        return None
    if array.dtype.kind not in "iuf" or len(meta.codecs) != 1:
        return None
    shard = meta.codecs[0]
    return shard if type(shard) is ShardingCodec and shard.index_location == "end" else None


def _bytes_chain(codecs: tuple[Any, ...], last: type[Any]) -> bool:
    return (
        len(codecs) == 2
        and type(codecs[0]) is BytesCodec
        and type(codecs[1]) is last
        and codecs[0].endian in {"little", "big"}
    )


def _endian(codec: Any) -> Literal["<", ">"]:
    return "<" if codec.endian == "little" else ">"


def _byte_runs(
    entries: npt.NDArray[np.uint64],
    *,
    gap: int = 0,
) -> Iterator[tuple[int, int, list[tuple[int, int, int]]]]:
    """Merge only physically adjacent ranges, never read intervening unwanted chunks."""
    absent = np.iinfo(np.uint64).max
    chunks = sorted(
        ((k, int(a), int(b)) for k, (a, b) in enumerate(entries) if (a, b) != (absent, absent)),
        key=lambda entry: entry[1],
    )
    run: list[tuple[int, int, int]] = []
    lo, hi = 0, 0
    for k, start, length in chunks:
        if run and _split_range(lo, hi, start, length, gap):
            yield lo, hi, run
            run = []
        if not run:
            lo = hi = start
        hi = max(hi, start + length)
        run.append((k, start, length))
    if run:
        yield lo, hi, run


def _split_range(lo: int, hi: int, start: int, length: int, gap: int) -> bool:
    return start > hi + gap or bool(gap and start + length - lo > 1 << 24)


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
            result: Buffer | None = await _on_pool(
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
            # Sharing protects other readers, not permission to outlive this read (ADR 0009).
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
