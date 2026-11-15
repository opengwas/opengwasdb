"""The narrow numeric codec backend behind location's bounded reads (ADR 0009; ogld#40).

This module owns codec recognition, shard/frame parsing, decoded-buffer ownership and the
single executor shared by local range I/O and decoding. Location owns transport wrapping,
cache accounting, prefetch planning and byte credits; this backend never chooses those limits.
"""

from __future__ import annotations

import asyncio
import importlib
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt
import zarr
from zarr.abc.store import RangeByteRequest, SuffixByteRequest
from zarr.codecs import BytesCodec, Crc32cCodec, ShardingCodec, ZstdCodec
from zarr.core.buffer import default_buffer_prototype
from zarr.core.metadata.v3 import ArrayV3Metadata, RegularChunkGridMetadata

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
type Decoded = list[npt.NDArray[Any] | None]


class DecodedChunk(np.ndarray[Any, np.dtype[Any]]):
    """Ownership token: one decoded chunk's buffer, never a view of a multi-chunk run.

    Only the direct decoder constructs these; the cache can adopt them without copying,
    unlike a zarr run whose views retain uncounted neighbouring chunks (ADR 0009).
    """


@dataclass(frozen=True)
class NumericDecoder:
    """Only the declared bytes/zstd chain, with a bytes/CRC32C suffix index (ADR 0009).

    No zarr private codec or index API is called. Offsets are absolute uint64 byte offsets,
    paired with lengths, and both all-ones means absent, as sharding_indexed specifies.
    """

    size: int
    per_shard: int
    dtype: np.dtype[Any]
    index_dtype: np.dtype[Any]

    @classmethod
    def of(cls, array: zarr.Array[Any]) -> NumericDecoder | None:
        """Recognise the exact codec chain, not a path or a writer's default shape."""
        shard = _numeric_shard(array)
        if shard is None:
            return None
        inner, index = shard.codecs, shard.index_codecs
        if not (
            _bytes_chain(inner, ZstdCodec, array.dtype)
            and _bytes_chain(index, Crc32cCodec, np.dtype("u8"))
        ):
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

    @property
    def index_size(self) -> int:
        """The uint64 offset/length table and its CRC32C, counted and fetched alike."""
        return self.per_shard * 16 + 4

    async def read(self, array: zarr.Array[Any], first: int, stop: int) -> list[npt.NDArray[Any]]:
        """One planned piece: the index, only its chunks' ranges, then bounded decode jobs."""
        key = array.metadata.encode_chunk_key((first // self.per_shard,))
        path = array.store_path / key
        prototype = default_buffer_prototype()
        index = await path.get(prototype, SuffixByteRequest(self.index_size))
        out: Decoded = [None] * (stop - first)
        if index is not None:
            entries = self._index(index.to_bytes())[
                first % self.per_shard : first % self.per_shard + stop - first
            ]
            for lo, hi, chunks in _byte_runs(entries):
                fetched = await path.get(prototype, RangeByteRequest(lo, hi))
                if fetched is not None:
                    await self.decode(fetched.to_bytes(), lo, chunks, out)
        return [self.owned(array, first + k, chunk) for k, chunk in enumerate(out)]

    def ranges(
        self, index: bytes, keys: npt.NDArray[np.int64], *, gap: int
    ) -> Iterator[tuple[int, int, _ChunkRanges]]:
        """Validate an index and locate selected chunks; the caller chooses coalescing policy."""
        return _byte_runs(self._index(index)[keys], gap=gap)

    def owned(
        self, array: zarr.Array[Any], k: int, chunk: npt.NDArray[Any] | None
    ) -> npt.NDArray[Any]:
        """Give the cache exactly one chunk's buffer, filling absence or trimming padding."""
        length = min(self.size, int(array.shape[0]) - k * self.size)
        if chunk is None:
            chunk = np.full(length, array.fill_value, dtype=array.dtype)
        elif length != self.size:
            # A truncated view would keep the padded tail outside the counted budget.
            chunk = chunk[:length].copy()
        return chunk.view(DecodedChunk)

    def _index(self, data: bytes) -> npt.NDArray[np.uint64]:
        expected = self.index_size
        if len(data) != expected:
            raise ValueError(
                f"shard index is truncated or corrupt: {len(data)} bytes, expected {expected}"
            )
        raw = _NUMCODECS.CRC32C().decode(data)
        return np.frombuffer(raw, dtype=self.index_dtype).reshape(self.per_shard, 2)

    async def decode(
        self, data: bytes | memoryview[int], offset: int, chunks: _ChunkRanges, out: Decoded
    ) -> None:
        """At most seven jobs per range. Drain running jobs even if another piece fails.

        Cancelling a future cannot stop zstd on a worker. Shielding and draining keeps that
        worker's buffers inside this read's lifetime and in-flight bound (ADR 0009).
        """
        jobs = [
            asyncio.create_task(
                on_pool(partial(self._into, data, offset, chunks[i::_DECODE_WORKERS], out))
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
        self, data: bytes | memoryview[int], offset: int, chunks: _ChunkRanges, out: Decoded
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


async def on_pool(call: Any) -> Any:
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


def _bytes_chain(codecs: tuple[Any, ...], last: type[Any], dtype: np.dtype[Any]) -> bool:
    """Bytes then `last`. A one-byte element has no byte order, and zarr declares none for it
    (ADR 0010); every wider one must declare the order it was written in (ADR 0009)."""
    return (
        len(codecs) == 2
        and type(codecs[0]) is BytesCodec
        and type(codecs[1]) is last
        and (
            codecs[0].endian in {"little", "big"}
            or (dtype.itemsize == 1 and codecs[0].endian is None)
        )
    )


def _endian(codec: Any) -> Literal["<", ">"]:
    return ">" if codec.endian == "big" else "<"


def _byte_runs(
    entries: npt.NDArray[np.uint64],
    *,
    gap: int = 0,
) -> Iterator[tuple[int, int, _ChunkRanges]]:
    """Coalesce physical ranges, reading intervening bytes only within the caller's gap."""
    absent = np.iinfo(np.uint64).max
    chunks = sorted(
        ((k, int(a), int(b)) for k, (a, b) in enumerate(entries) if (a, b) != (absent, absent)),
        key=lambda entry: entry[1],
    )
    run: _ChunkRanges = []
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
