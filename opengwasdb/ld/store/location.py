"""Where an LD Store is, and how its elements are read (spec §1, §10).

A store is a local path or an HTTP(S) URL, opened as one Zarr hierarchy. A reader and
validation open it the same way, so a store served from a static host that honours byte
ranges reads exactly as it does from disk. zarr opens a URL through fsspec's HTTP filesystem
(aiohttp). Nothing here lists a directory: every node is found by its own metadata document,
which a static host serves like any other file.

Every element a reader or a rule reads goes through a `ChunkCache`. It reads whole inner
chunks, which is what Zarr fetches and decodes in any case, keeps the decoded chunks up to a
byte budget, and serves each request from them. So a query that reads a few pairs at a time,
as clumping does once per lead, fetches each chunk once, from disk or over HTTP. How a store
is read is decided here, in one place: ogld#8 measured it (`docs/benchmarks/latency.md`), and
ogld#32 tunes it.
"""

from __future__ import annotations

import math
import os
import threading
from collections import OrderedDict
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import zarr

from ogld.format import InputError
from ogld.format.geometry import spans

#: Decoded inner chunks a reader keeps, in bytes. At layout C a Near Band chunk is 512 KiB.
CACHE_BYTES: Final = 1 << 28


def open_group(location: str | os.PathLike[str]) -> zarr.Group:
    """The root group of the hierarchy at `location`, read-only, whatever its Zarr format.

    A location with no Zarr group at all, or whose root metadata will not parse, is an
    `InputError` naming it. A Zarr v2 group opens,
    so validation can say it is not v3 (spec §12, `envelope.arrays`) rather than not there.
    """
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
        self._budget = CACHE_BYTES if budget is None else budget
        self._bytes = 0
        self._kept: OrderedDict[tuple[str, int], tuple[npt.NDArray[Any], int]] = OrderedDict()
        self._lock = threading.Lock()

    def take(self, array: zarr.Array[Any], coords: npt.ArrayLike) -> npt.NDArray[Any]:
        """The elements of a 1-D array at `coords`, in that order, from the inner chunks that
        hold them: each read whole, once, and runs of neighbouring chunks in one read."""
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
        """The decoded chunks `wanted`, in order: kept ones from memory, the rest read."""
        found = self._kept_chunks(array.path, wanted.tolist())
        missing = np.array([k for k in wanted.tolist() if k not in found], dtype=np.int64)
        for run in np.split(missing, np.flatnonzero(np.diff(missing) != 1) + 1):
            if len(run):
                found.update(self._read_run(array, int(run[0]), int(run[-1]) + 1, size))
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

    def _read_run(
        self, array: zarr.Array[Any], first: int, stop: int, size: int
    ) -> dict[int, npt.NDArray[Any]]:
        """Chunks first … stop − 1, read in one request, each copied out and kept."""
        data = np.asarray(array[first * size : stop * size])
        copies = [
            data[(k - first) * size : (k - first + 1) * size].copy() for k in range(first, stop)
        ]
        read = {first + i: (chunk, _counted(chunk)) for i, chunk in enumerate(copies)}
        with self._lock:
            return {k: self._keep((array.path, k), *entry) for k, entry in read.items()}

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


def _counted(chunk: npt.NDArray[Any]) -> int:
    """An upper bound on the bytes `chunk` holds: a numeric chunk's buffer, and for a string
    chunk its buffer and arena. Measured against tracemalloc on 72 chunks of 1–16,384 strings
    of 1–20,000 characters, ASCII and not, the count was never below what a chunk held."""
    if chunk.dtype.kind != "T":
        return int(chunk.nbytes)
    long = [n for n in (len(s.encode("utf-8")) for s in chunk.tolist()) if n > _INLINE]
    arena = math.ceil(_GROWTH * (sum(long) + _HEADER * len(long))) + _ARENA_FIXED
    return int(chunk.nbytes) + arena
