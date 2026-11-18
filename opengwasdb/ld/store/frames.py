"""The Partner Rows' frames and their group (partner-rows §5.1, §5.2; ogld#67).

The Partner Rows hold, for each variant, every partner within `far_bp` whose code reaches the
Partner Floor within `near_bp` or the far code floor beyond it, in both variants' rows. They are
grouped `ROWS_PER_FRAME` rows to a zstd frame, and one chromosome's frames are the bytes of a
single-chunk `uint8` array, so a row is one positional read and one decompress (spec §5.4).

This module owns both directions of that codec and the group around it:

- `encode_frame` and `decode_frame` write and read a frame's byte layout (§5.2): k, the rows'
  lengths, the partner deltas and the codes, the two arrays byte-shuffled, as one zstd frame at
  level 3 with the content checksum and the content size set in its header. `Frame` is a decoded
  frame, and `Frame.row` gives one row's partners and codes. #68 validates frames through them
  and #70 reads rows with them; neither parses a payload of its own.
- `FramesGroup` is one chromosome's `partners/<chrom>/` group: its two arrays, its attributes and
  the chunk file they point at. A frame is compressed on a bounded pool and appended to the chunk
  file as it is finished, in frame order, so the bytes do not depend on the pool's size or on the
  order its workers finish in. The array's shape is not known until the chromosome ends, so the
  group is a building file until `close`, which creates the metadata at its final shape and puts
  the file where that metadata says its chunk is.

A frame is never read short: a checksum that fails, a payload that is not exactly
4 + 4k + 6E bytes, or a k that is not the rows the frame is due raises `ValueError` naming the
chromosome and the frame (§5.4). A caller that got rows from a frame got all of them.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import shutil
import struct
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
import zarr
from zarr.abc.store import RangeByteRequest
from zarr.core.buffer import default_buffer_prototype
from zarr.storage import LocalStore

from ogld.format import Codes, InputError, REncoding
from ogld.format.geometry import indptr_of, owners
from ogld.store import layout
from ogld.store.location import FrameSource, frame_source

#: Rows a frame holds (partner-rows §1, §5.2: G = 16, the smallest G of those equally fast). A
#: reader takes it from the group's `rows_per_frame` attribute, never from here.
ROWS_PER_FRAME: Final = 16
#: A frame's zstd level (§5.2), recorded in the group's `zstd_level`.
ZSTD_LEVEL: Final = 3

#: The payload's arrays' wire dtypes (§5.2), little-endian whatever this machine is: the field
#: widths are bytes on disk. The codes are the encoding's container, which every admitted encoding
#: shares (ADR 0004), so their wire dtype is `REncoding.dtype` with a little-endian order.
_LITTLE_ROW: Final = np.dtype("<u4")
_LITTLE_DELTA: Final = np.dtype("<i4")
_LITTLE_CODES: Final = REncoding.dtype.newbyteorder("<")
_LITTLE_OFFSET: Final = np.dtype("<u8")

# numcodecs is a runtime dependency without type stubs, as in decoding and build.inputs.
_NUMCODECS = importlib.import_module("numcodecs")

#: One zstd frame at level 3 with its content checksum, which is what a frame's bytes are (§5.2).
_ZSTD: Final = _NUMCODECS.Zstd(level=ZSTD_LEVEL, checksum=True)

#: Bytes `FramesGroup.close` reads back through zarr at each end and in the middle of the chunk
#: file: enough to prove the metadata resolves the file it wrote, without a second pass over it.
_READ_BACK: Final = 64
#: The largest frames file `FramesGroup.close` decodes whole through zarr. A one-chunk array's
#: slice read fetches and decodes the whole chunk, so a whole-chromosome read-back would double a
#: build's peak memory (EUR's frames are 14.8 GB); above this, the metadata, the file's hash and
#: zarr's own range reads at both ends and the middle stand in for the decode, and the tests hold
#: the decode itself on the Test Store, whose frames are tens of kilobytes.
_DECODE_LIMIT: Final = 1 << 20


def zstd_version() -> str:
    """The zstd library's version, which a frame's bytes depend on, so a store records it (§7).

    It is `numcodecs`' zstd, the one this module compresses with; a build that recorded another
    did not write these frames.
    """
    version = getattr(_NUMCODECS.zstd, "__version__", None)
    if not isinstance(version, str):
        raise ValueError(
            "numcodecs does not say which zstd it holds, and a frame's bytes depend on it (spec §7)"
        )
    return version


@dataclass(frozen=True)
class Frame:
    """One decoded frame: the rows [first, first + rows) of chromosome `chrom` (§5.2).

    `indptr` holds the frame's entry offsets, `rows + 1` of them, so row i's partners are
    `partners[indptr[i]:indptr[i + 1]]`, ascending and each its chromosome-local row. `number` is
    the frame's own index, so a refusal can name it.
    """

    chrom: str
    number: int
    first: int
    indptr: npt.NDArray[np.int64]
    partners: npt.NDArray[np.int32]
    codes: Codes

    @property
    def rows(self) -> int:
        """The rows this frame holds, its k (§5.2)."""
        return len(self.indptr) - 1

    @property
    def entries(self) -> int:
        """The entries it holds, E."""
        return int(self.indptr[-1])

    def row(self, local_row: int) -> tuple[npt.NDArray[np.int32], Codes]:
        """The partners of chromosome-local row `local_row` and their codes, or a refusal naming
        the frame when the row is not one of its own."""
        if not self.first <= local_row < self.first + self.rows:
            raise ValueError(
                f"row {local_row} is not in frame {self.number} of chromosome {self.chrom}, "
                f"which holds rows {self.first}–{self.first + self.rows - 1} (spec §5.2)"
            )
        at = local_row - self.first
        span = slice(int(self.indptr[at]), int(self.indptr[at + 1]))
        return self.partners[span], self.codes[span]


def _where(chrom: str, number: int) -> str:
    return f"frame {number} of chromosome {chrom}"


def _shuffled(values: npt.NDArray[Any]) -> bytes:
    """`values` as planes: every element's byte 0, then every element's byte 1, and so on (§5.2).

    Deltas are small, so their high planes are almost all 0 or 0xFF and compress to nearly nothing.
    """
    planes = np.ascontiguousarray(values).view("u1").reshape(-1, values.dtype.itemsize)
    return planes.T.tobytes()


def _unshuffled(raw: bytes, dtype: np.dtype[Any]) -> npt.NDArray[Any]:
    """The elements of `raw`'s planes, in order: the inverse of `_shuffled` (§5.2)."""
    width = dtype.itemsize
    planes = np.frombuffer(raw, "u1").reshape(width, -1)
    return planes.T.copy().ravel().view(dtype)


def _row_of_planes(
    raw: bytes, *, offset: int, elements: int, start: int, count: int, dtype: np.dtype[Any]
) -> npt.NDArray[Any]:
    """Elements [start, start + count) of the byte-shuffled array at `offset`, gathered from its
    planes: the inverse of `_shuffled` (§5.2) for one row of a frame, without unshuffling the rows
    a reader did not ask for. `elements` is the array's whole length, which is each plane's stride.
    """
    planes = np.empty((dtype.itemsize, count), dtype=np.uint8)
    for byte in range(dtype.itemsize):
        at = offset + byte * elements + start
        planes[byte] = np.frombuffer(raw[at : at + count], dtype=np.uint8)
    return planes.T.copy().ravel().view(dtype)


def _lengths(
    rows: Sequence[tuple[npt.ArrayLike, npt.ArrayLike]], first: int, where: str
) -> npt.NDArray[np.uint32]:
    """Each row's entry count, refused unless its codes are exactly as many as its partners.

    A code is its pair's, so a row of one count and a row of another cannot be a frame, and a
    compensating pair of rows would attach one row's code to another's partner: each row is judged
    on its own, before anything is concatenated or narrowed, and the refusal names the row (§5.2).
    """
    counts = []
    for at, (partners, codes) in enumerate(rows):
        held, coded = len(np.asarray(partners)), len(np.asarray(codes))
        if held != coded:
            raise ValueError(
                f"{where}, row {first + at}: {held} partner(s) and {coded} code(s), and a row's "
                "codes are its pairs' (spec §5.2)"
            )
        counts.append(held)
    return np.array(counts, dtype=_LITTLE_ROW)


def _stacked(
    lengths: npt.NDArray[np.uint32], rows: Sequence[tuple[npt.ArrayLike, npt.ArrayLike]]
) -> tuple[npt.NDArray[np.int64], Codes]:
    """The frame's partners and codes, row after row, as the payload holds them: the partners as
    int64 so their deltas cannot wrap, the codes in the encoding's container."""
    if not int(lengths.sum()):
        return np.empty(0, np.int64), np.empty(0, REncoding.dtype)
    partners = np.concatenate([np.asarray(row, dtype=np.int64) for row, _ in rows])
    codes = np.concatenate([np.asarray(codes).astype(_LITTLE_CODES) for _, codes in rows])
    return partners, codes.astype(REncoding.dtype)


def _checked(first: int, lengths: npt.NDArray[np.uint32], partners: npt.NDArray[np.int64]) -> None:
    """Refuse partners that are not strictly ascending within their row, that are negative, or
    that name their own row: a row a reader could not binary-search, or that holds its diagonal."""
    starts = np.cumsum(lengths)[lengths > 0] - lengths[lengths > 0]
    row = owners(lengths) + first
    after_first = np.ones(len(partners), dtype=bool)
    after_first[starts] = False
    out_of_order = after_first[1:] & ~(np.diff(partners) > 0)
    for found, what in (
        (partners < 0, "a negative partner"),
        (partners == row, "its own row"),
        (np.concatenate([[False], out_of_order]), "partners that do not ascend"),
    ):
        at = np.flatnonzero(found)
        if at.size:
            raise ValueError(f"row {int(row[at[0]])} holds {what} (spec §3, §5.2)")


def _deltas(
    first: int, lengths: npt.NDArray[np.uint32], partners: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    """Each row's partners as deltas: m₀ − l first, then mᵢ − mᵢ₋₁ (§5.2)."""
    deltas = np.empty(len(partners), dtype=np.int64)
    deltas[1:] = partners[1:] - partners[:-1]
    nonempty = np.flatnonzero(lengths)
    starts = np.cumsum(lengths)[nonempty] - lengths[nonempty]
    deltas[starts] = partners[starts] - (first + nonempty)
    bounds = np.iinfo(_LITTLE_DELTA)
    if len(deltas) and (deltas.min() < bounds.min or deltas.max() > bounds.max):
        raise ValueError(
            "a Partner Rows delta does not fit int32: the frame's partners are not one "
            "chromosome's rows (spec §5.2)"
        )
    return deltas.astype(_LITTLE_DELTA)


def encode_frame(
    first: int,
    rows: Sequence[tuple[npt.ArrayLike, npt.ArrayLike]],
    *,
    chrom: str,
    number: int,
) -> bytes:
    """One frame's bytes: `rows`, the first of them chromosome-local row `first` (§5.2).

    `rows` holds one (partners, codes) pair per row, its partners ascending and chromosome-local,
    and each code the pair's Stored r. The payload is k, the lengths, the byte-shuffled int32
    deltas and the byte-shuffled codes, compressed as one zstd frame at level 3 with the content
    checksum and the content size in its header, so a frame is checked on the way back in. A row
    whose codes are not as many as its partners, or whose partners are not ascending from its own
    row, is refused naming the chromosome, the frame and the row.
    """
    where = _where(chrom, number)
    if not rows:
        raise ValueError(f"{where} holds no rows: a frame holds at least one (spec §5.2)")
    lengths = _lengths(rows, first, where)
    partners, codes = _stacked(lengths, rows)
    try:
        _checked(first, lengths, partners)
    except ValueError as refused:
        raise ValueError(f"{where}: {refused}") from refused
    payload = (
        struct.pack("<I", len(lengths))
        + lengths.astype(_LITTLE_ROW).tobytes()
        + _shuffled(_deltas(first, lengths, partners))
        + _shuffled(codes)
    )
    return bytes(_ZSTD.encode(payload))


#: A zstd frame's magic number, little-endian as it lies on disk (RFC 8878 §3.1.1).
ZSTD_MAGIC: Final = b"\x28\xb5\x2f\xfd"


def _header(raw: bytes, where: str) -> None:
    """Refuse a frame whose zstd header does not declare its content checksum and its content size
    (partner-rows §5.2; RFC 8878 §3.1.1.1).

    The checksum and the content size are what make a frame verifiable: a decoder that verifies
    neither accepts a damaged frame as data, and configuring the compressor with them does not
    require an incoming frame to carry them. So both are read from the header descriptor and
    refused when absent, before anything is decoded.
    """
    if len(raw) < 5 or not raw.startswith(ZSTD_MAGIC):
        raise ValueError(
            f"{where} does not begin a zstd frame: its first bytes are "
            f"{raw[:5].hex() or 'none'} (partner-rows §5.2)"
        )
    descriptor = raw[4]
    size_flag, single_segment = descriptor >> 6, descriptor & 0b0010_0000
    if not descriptor & 0b0000_0100:
        raise ValueError(
            f"{where} declares no content checksum: a frame's header carries one, or the frame "
            "cannot be verified (partner-rows §5.2)"
        )
    if size_flag == 0 and not single_segment:
        raise ValueError(
            f"{where} declares no content size: a frame's header carries one, or its payload "
            "cannot be checked against it (partner-rows §5.2)"
        )


def _payload(raw: bytes, where: str) -> bytes:
    """`raw`'s decompressed payload, or a refusal naming `where` when its zstd frame or its
    checksum fails (§5.4)."""
    _header(raw, where)
    try:
        return bytes(_ZSTD.decode(raw))
    except Exception as failed:
        raise ValueError(
            f"{where} does not decompress: its zstd frame is damaged, or its content checksum "
            f"fails ({failed}). A frame that cannot be checked is never read as a short row "
            "(spec §5.4)"
        ) from failed


def _counted(payload: bytes, where: str, rows: int) -> tuple[int, npt.NDArray[np.int64]]:
    """The payload's k and lengths, refused when k is not `rows` or the payload's length is not
    exactly 4 + 4k + 6E (§5.2)."""
    if len(payload) < 4:
        raise ValueError(f"{where} holds {len(payload)} bytes: too few for its k (spec §5.2)")
    k = int(np.frombuffer(payload, _LITTLE_ROW, 1)[0])
    if k != rows:
        raise ValueError(f"{where} holds {k} rows, not the {rows} it is due (spec §5.2)")
    if len(payload) < 4 + 4 * k:
        raise ValueError(
            f"{where} holds {len(payload)} bytes: too few for k = {k} rows' lengths (spec §5.2)"
        )
    lengths = np.frombuffer(payload, _LITTLE_ROW, k, 4).astype(np.int64)
    entries = int(lengths.sum())
    if len(payload) != 4 + 4 * k + 6 * entries:
        raise ValueError(
            f"{where}'s payload is {len(payload)} bytes, and k = {k} with {entries} entries "
            f"needs exactly {4 + 4 * k + 6 * entries} = 4 + 4k + 6E (spec §5.2)"
        )
    return k, lengths


def _partners(
    first: int, lengths: npt.NDArray[np.int64], deltas: npt.NDArray[np.int64]
) -> npt.NDArray[np.int32]:
    """The deltas as each row's chromosome-local partners: a cumulative sum reset at each row's
    first entry (§5.2)."""
    partners = np.cumsum(deltas, dtype=np.int64)
    nonempty = np.flatnonzero(lengths)
    starts = np.cumsum(lengths)[nonempty] - lengths[nonempty]
    before = np.where(starts > 0, partners[np.maximum(starts - 1, 0)], 0)
    partners += np.repeat(first + nonempty - before, lengths[nonempty])
    bounds = np.iinfo(np.int32)
    if len(partners) and (partners.min() < 0 or partners.max() > bounds.max):
        raise ValueError("a Partner Rows frame decodes to a partner that is not a row (spec §5.2)")
    return partners.astype(np.int32)


def decode_frame(raw: bytes, *, chrom: str, number: int, first: int, rows: int) -> Frame:
    """The frame `raw` holds, which must be rows [first, first + `rows`) of chromosome `chrom`.

    The chromosome, the frame's number and the rows it is due are the caller's: they are what a
    refusal names, and what a payload is checked against, so a frame is never read as a short or a
    borrowed row (§5.4). A wrong k, a payload that is not exactly 4 + 4k + 6E bytes and a failed
    checksum all raise `ValueError` naming the chromosome and the frame.
    """
    return FramePayload.parse(raw, chrom=chrom, number=number, first=first, rows=rows).decode()


@dataclass(frozen=True)
class FramePayload:
    """One frame's decompressed payload, with the rows' lengths that address it (§5.2).

    A reader keeps this rather than every decoded row: the decompress is the expensive half of a
    frame, and a row's own bytes are a gather from each of the payload's planes. `row` does that
    gather and decodes one row (the row-local unshuffle, §7), so reading one row of a frame costs
    one row's work and not the other fifteen; `decode` unshuffles the whole payload and gives the
    `Frame` that validation and the frames' own tests read.
    """

    chrom: str
    number: int
    first: int
    lengths: npt.NDArray[np.int64]
    payload: bytes

    @classmethod
    def parse(cls, raw: bytes, *, chrom: str, number: int, first: int, rows: int) -> FramePayload:
        """The payload `raw` holds — one zstd frame, its checksum verified — checked as the frame
        of rows [first, first + `rows`) of chromosome `chrom` (§5.2, §5.4)."""
        where = _where(chrom, number)
        held = _payload(raw, where)
        _, lengths = _counted(held, where, rows)
        return cls(chrom, number, first, lengths.astype(np.int64), held)

    @property
    def rows(self) -> int:
        """The rows this frame holds, its k (§5.2)."""
        return len(self.lengths)

    @property
    def entries(self) -> int:
        """The entries it holds, E."""
        return int(self.lengths.sum())

    @property
    def bytes(self) -> int:
        """What keeping it holds: the payload and the offsets into it."""
        return len(self.payload) + int(self.lengths.nbytes)

    def decode(self) -> Frame:
        """The whole frame, every row's partners and codes (§5.2): one unshuffle of each plane."""
        head = 4 + 4 * self.rows
        entries = self.entries
        deltas = _unshuffled(self.payload[head : head + 4 * entries], _LITTLE_DELTA)
        codes = _unshuffled(self.payload[head + 4 * entries :], _LITTLE_CODES)
        return Frame(
            self.chrom,
            self.number,
            self.first,
            indptr_of(self.lengths),
            _partners(self.first, self.lengths, deltas.astype(np.int64)),
            codes.astype(REncoding.dtype),
        )

    def row(self, local_row: int) -> tuple[npt.NDArray[np.int32], Codes]:
        """Row `local_row`'s partners and codes, decoded from the payload's planes alone (§5.2).

        The deltas are gathered from their planes for this row's entries and cumulated from its own
        row, so the rest of the frame is never unshuffled: one row costs one row's bytes.
        """
        if not self.first <= local_row < self.first + self.rows:
            raise ValueError(
                f"row {local_row} is not in frame {self.number} of chromosome {self.chrom}, "
                f"which holds rows {self.first}–{self.first + self.rows - 1} (spec §5.2)"
            )
        at = local_row - self.first
        start = int(self.lengths[:at].sum())
        count = int(self.lengths[at])
        if not count:
            return np.empty(0, np.int32), np.empty(0, REncoding.dtype)
        head = 4 + 4 * self.rows
        entries = self.entries
        deltas = _row_of_planes(
            self.payload,
            offset=head,
            elements=entries,
            start=start,
            count=count,
            dtype=_LITTLE_DELTA,
        ).astype(np.int64)
        codes = _row_of_planes(
            self.payload,
            offset=head + 4 * entries,
            elements=entries,
            start=start,
            count=count,
            dtype=_LITTLE_CODES,
        ).astype(REncoding.dtype)
        partners = np.cumsum(deltas) + local_row
        bounds = np.iinfo(np.int32)
        if partners.min() < 0 or partners.max() > bounds.max:
            raise ValueError(
                f"frame {self.number} of chromosome {self.chrom} decodes to a partner that is not "
                "a row (spec §5.2)"
            )
        return partners.astype(np.int32), codes


# ---------------------------------------------------------------- reading (§7)

#: One row decoded: its partners, chromosome-local and ascending, and their codes (§5.2).
_Row = tuple[npt.NDArray[np.int32], Codes]

#: The frames a reader keeps unless its caller says otherwise, in bytes (§7): 256 MiB, which holds
#: about 3,400 of a panel store's frames (a median frame's payload is some 77 KB) and all 58 of the
#: Test Store's. It is the frame cache's own budget, beside `location.CACHE_BYTES`, which
#: keeps the Near Band's inner chunks for `matrix` and `region`.
FRAME_CACHE_BYTES: Final = 1 << 28


def _frame_bytes(frame: FramePayload) -> int:
    """What keeping a frame holds: its decompressed payload and the rows' lengths (§7)."""
    return frame.bytes


class FrameCache:
    """Decoded frames kept by one reader, byte-bounded, the least recently used dropped first
    (partner-rows §7).

    It is separate from `location.ChunkCache`, which keeps the Near Band's inner chunks for
    `matrix` and `region`: a frame is not a chunk of a Zarr array, and a request that reads one
    never reads the other. Neighbouring leads often share a frame, and repeated Analyses on one
    reader hit it, so a frame decoded once serves every row taken from it.

    One lock guards lookup, insertion and eviction, so the count stays the sum of what is kept and
    a frame another thread kept first is the one given back. Decoding runs outside it: two threads
    that miss the same frame may each decode it and keep one copy, which costs a duplicate decode
    and never a different row (partner-rows §9.2 needs no single-flight for that).
    """

    def __init__(self, budget: int | None = None) -> None:
        if budget is not None and (
            isinstance(budget, bool) or not isinstance(budget, int) or budget < 0
        ):
            raise InputError(f"frame_cache_bytes is a count of bytes, an int ≥ 0, not {budget!r}")
        self._budget = FRAME_CACHE_BYTES if budget is None else budget
        self._bytes = 0
        self._kept: OrderedDict[tuple[str, int], tuple[FramePayload, int]] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def budget(self) -> int:
        """The bytes this cache keeps at most (§7)."""
        return self._budget

    @property
    def held(self) -> int:
        """The bytes it keeps now, as it counts them: never more than the budget."""
        with self._lock:
            return self._bytes

    def take(self, chrom: str, number: int) -> FramePayload | None:
        """Frame `number` of `chrom` if it is kept, marked as just used; None otherwise."""
        key = (chrom, number)
        with self._lock:
            kept = self._kept.get(key)
            if kept is None:
                return None
            self._kept.move_to_end(key)
        return kept[0]

    def keep(self, frame: FramePayload) -> FramePayload:
        """Keep `frame` in the budget, or the copy another thread kept first, and give it back.

        The least recently used frames are dropped until the count is within the budget, a frame
        larger than the whole budget included: a cache keeps no more than it says it does.
        """
        key = (frame.chrom, frame.number)
        counted = _frame_bytes(frame)
        with self._lock:
            kept = self._kept.get(key)
            if kept is not None:
                self._kept.move_to_end(key)
                return kept[0]
            self._kept[key] = (frame, counted)
            self._bytes += counted
            while self._bytes > self._budget and self._kept:
                _, (_, dropped) = self._kept.popitem(last=False)
                self._bytes -= dropped
            return frame


def _member(group: zarr.Group, name: str) -> zarr.Array[Any]:
    """The array `name` of a Partner Rows group, refused naming it when the group holds none."""
    path = f"{group.path}/{name}"
    try:
        member = group[name]
    except KeyError:
        raise ValueError(
            f"{path} is missing: a Partner Rows group holds it (partner-rows §5.1)"
        ) from None
    if not isinstance(member, zarr.Array):
        raise ValueError(
            f"{path} is not an array: a Partner Rows group holds two (partner-rows §5.1)"
        )
    return member


def _attribute(attributes: Mapping[str, Any], name: str, path: str) -> int:
    """One of a Partner Rows group's integer attributes, refused naming it when it is not there."""
    value = attributes.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{path} records no {name}: a Partner Rows group holds it (partner-rows §5.3)"
        )
    return value


def _offsets(index: zarr.Array[Any], frames: int, size: int, chrom: str) -> npt.NDArray[np.uint64]:
    """A chromosome's frame offsets, read whole and checked before any frame is (partner-rows §7).

    They are the group's `frames` + 1 byte offsets of one chunk: 0 first, strictly ascending, and
    ending at the `frames_bytes` the group records, so a frame's range is inside the file the
    metadata names. An offsets chunk the store lost reads as its fill value, which the last of these
    checks refuses.
    """
    path = layout.partners_path(chrom)
    if index.shape != (frames + 1,) or index.chunks != index.shape:
        raise ValueError(
            f"{path}/frames_idx has shape {index.shape} and chunks {index.chunks}, not one chunk "
            f"of the {frames + 1} offsets of {frames} frames (partner-rows §5.1)"
        )
    offsets = np.asarray(index[:], dtype=np.uint64)
    if (offsets > size).any() or int(offsets[0]) != 0 or int(offsets[-1]) != size:
        raise ValueError(
            f"chromosome {chrom}'s frames_idx runs from {int(offsets[0])} to {int(offsets[-1])}, "
            f"and the group's frames_bytes is {size}: the offsets must start at 0, ascend and end "
            "at the file's length (partner-rows §5.2)"
        )
    steps = np.diff(offsets.astype(np.int64))
    if len(steps) and (steps <= 0).any():
        raise ValueError(
            f"chromosome {chrom}'s frames_idx does not ascend at offset "
            f"{int(np.flatnonzero(steps <= 0)[0])}: every frame holds bytes (partner-rows §5.2)"
        )
    return offsets


class PartnerFrames:
    """One chromosome's Partner Rows, opened to be read a row or a frame at a time (§5.1, §7).

    Opening reads what a chromosome's first use needs and nothing else: the group's attributes and
    the `frames_idx` chunk whole (§7 step 1), whose offsets are checked against the group's `frames`
    and `frames_bytes` before any frame is read, and the `frames` array's metadata, which names the
    one chunk file the frames are byte ranges of.

    A frame is one read through `location`'s frame source — `os.pread` on a local file's kept
    descriptor, one Range request for a store that is not on this disk — and one
    `FramePayload.parse`, the store's one frame decoder, so a frame that fails its header, its
    checksum, its k or its payload length is refused naming the chromosome and the frame, and never
    read as a short row (§5.4). Each frame parsed is kept in the reader's `FrameCache`, so a row of
    a frame another row already fetched costs no read, and a row costs only its own gather from the
    payload's planes (§7).

    A source is not fork-safe and is not inherited across `fork` (its lock and descriptor are not):
    a process worker opens its own reader, and so its own sources (partner-rows §9; ogld#71).
    """

    def __init__(self, group: zarr.Group, chrom: str, cache: FrameCache) -> None:
        path = layout.partners_path(chrom)
        attributes = group.attrs
        self._chrom = chrom
        self._cache = cache
        self._rows = _attribute(attributes, "rows", path)
        self._per_frame = _attribute(attributes, "rows_per_frame", path)
        frames = _attribute(attributes, "frames", path)
        size = _attribute(attributes, "frames_bytes", path)
        due = -(-self._rows // self._per_frame) if self._per_frame > 0 else -1
        if self._rows < 1 or self._per_frame < 1:
            raise ValueError(
                f"{path} holds {self._rows} rows in frames of {self._per_frame}: a Partner Rows "
                "group holds at least one row, in a frame of at least one (partner-rows §5.2)"
            )
        if frames != due:
            raise ValueError(
                f"{path} records {frames} frames for {self._rows} rows in frames of "
                f"{self._per_frame}: it holds {due} (partner-rows §5.2)"
            )
        self._frames = frames
        self._offsets = _offsets(_member(group, layout.FRAMES_INDEX), frames, size, chrom)
        self._array = _member(group, layout.FRAMES)
        if (
            self._array.shape != (size,)
            or self._array.chunks != self._array.shape
            or self._array.dtype != np.uint8
        ):
            raise ValueError(
                f"{path}/{layout.FRAMES} is shape {self._array.shape}, chunks "
                f"{self._array.chunks} and {self._array.dtype}, not one chunk of the {size} bytes "
                "of frames the group records (partner-rows §5.1)"
            )
        try:
            self._source: FrameSource = frame_source(self._array)
        except ValueError as unreadable:
            raise ValueError(f"{path}: {unreadable}") from unreadable

    @property
    def rows(self) -> int:
        """The rows this chromosome's Partner Rows hold, the group's `rows` (§5.3)."""
        return self._rows

    def frame(self, number: int) -> FramePayload:
        """Frame `number`: from the cache, or one byte-range read and one decompress (§7)."""
        if not 0 <= number < self._frames:
            raise ValueError(
                f"frame {number} is not one of chromosome {self._chrom}'s {self._frames} frames "
                "(partner-rows §5.2)"
            )
        kept = self._cache.take(self._chrom, number)
        if kept is not None:
            return kept
        return self._cache.keep(self._read(number, number)[0])

    def close(self) -> None:
        """Release the chunk file this chromosome's frames are read from (§7): a local store's
        descriptor, closed by the reader that owns it."""
        self._source.close()

    def row(self, local_row: int) -> tuple[npt.NDArray[np.int32], Codes]:
        """Row `local_row`'s partners and codes, decoded from the frame that holds it (§7 step 3).

        The row is gathered from the frame's planes and its deltas cumulated, so a row costs one
        row's work: the frame's other rows are never unshuffled to answer it. The arrays are new
        ones, owned by the caller, and the cached payload they came from is not changed by them.
        """
        if not 0 <= local_row < self._rows:
            raise ValueError(
                f"row {local_row} is not in chromosome {self._chrom}, which holds {self._rows} "
                "rows (partner-rows §5.2)"
            )
        return self.frame(local_row // self._per_frame).row(local_row)

    def rows_of(
        self, local_rows: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int32], Codes]:
        """Each of `local_rows`' rows, back to back, as (indptr, partners, codes), in the order
        asked for and with a repeated row sliced again (§7).

        The frames the rows need that are not kept are read in runs of physically adjacent frames,
        one byte-range read a run: frames that do not touch are never coalesced, because the bytes
        between them would be read for nothing (§9.2).
        """
        at = np.asarray(local_rows, dtype=np.int64)
        outside = at[(at < 0) | (at >= self._rows)]
        if len(outside):
            raise ValueError(
                f"rows {outside[:5].tolist()} are not in chromosome {self._chrom}, which holds "
                f"{self._rows} rows (partner-rows §5.2)"
            )
        rows = self._decoded(at)
        indptr = indptr_of(np.array([len(partners) for partners, _ in rows], dtype=np.int64))
        if not rows:
            return indptr, np.empty(0, np.int32), np.empty(0, REncoding.dtype)
        return (
            indptr,
            np.concatenate([partners for partners, _ in rows]),
            np.concatenate([codes for _, codes in rows]),
        )

    def _decoded(self, at: npt.NDArray[np.int64]) -> list[_Row]:
        """Each of `at`'s rows decoded, in that order (§7): from the cache where it holds the
        frame, and from the runs of frames just read where it does not."""
        pieces, missing = self._from_cache(at)
        self._from_runs(at, pieces, missing)
        rows = [piece for piece in pieces if piece is not None]
        assert len(rows) == len(at), "every row asked for is decoded"
        return rows

    def _from_cache(
        self, at: npt.NDArray[np.int64]
    ) -> tuple[list[_Row | None], dict[int, list[int]]]:
        """Each row's slice of its frame where the cache holds it, and the rows it does not, by the
        frame each needs."""
        pieces: list[_Row | None] = [None] * len(at)
        missing: dict[int, list[int]] = {}
        which = at // self._per_frame
        for i, (row, number) in enumerate(zip(at.tolist(), which.tolist(), strict=True)):
            kept = self._cache.take(self._chrom, number)
            if kept is None:
                missing.setdefault(number, []).append(i)
            else:
                pieces[i] = kept.row(row)
        return pieces, missing

    def _from_runs(
        self, at: npt.NDArray[np.int64], pieces: list[_Row | None], missing: dict[int, list[int]]
    ) -> None:
        """Decode the rest from the runs of adjacent frames just read, then offer those payloads to
        the cache: a run is decoded while it is owned, so no budget can evict it between the read
        and the row, and only one run's payloads are held at a time beside the result (§9.2)."""
        for run in _runs(sorted(missing)):
            held = self._read(run[0], run[-1])
            for number, payload in zip(run, held, strict=True):
                for i in missing[number]:
                    pieces[i] = payload.row(int(at[i]))
            for payload in held:
                self._cache.keep(payload)

    def _read(self, first: int, last: int) -> list[FramePayload]:
        """Frames [first, last] as one read of the chunk file, parsed and handed back unkept (§9.2).

        They are physically adjacent, so one read serves them all; the bytes of a frame are its own
        slice of what was read, and each is parsed on its own, so a frame that fails its header,
        its checksum, its k or its payload length is refused naming itself. The caller owns the
        payloads it gets: a bulk read decodes its rows from them before the cache is offered any,
        so a run is never read twice for want of a budget.
        """
        start, stop = int(self._offsets[first]), int(self._offsets[last + 1])
        try:
            raw = self._source.read(start, stop)
        except ValueError as unread:
            raise ValueError(f"{_run(first, last, self._chrom)}: {unread}") from unread
        if len(raw) != stop - start:
            raise ValueError(
                f"{_run(first, last, self._chrom)} holds {len(raw)} of the {stop - start} bytes at "
                f"[{start}, {stop}): a short read is a damaged chunk file, never a short row "
                "(partner-rows §5.4)"
            )
        return [
            FramePayload.parse(
                raw[int(self._offsets[number]) - start : int(self._offsets[number + 1]) - start],
                chrom=self._chrom,
                number=number,
                first=number * self._per_frame,
                rows=min(self._per_frame, self._rows - number * self._per_frame),
            )
            for number in range(first, last + 1)
        ]


def _runs(numbers: list[int]) -> Iterator[list[int]]:
    """`numbers`, ascending and distinct, as the runs of physically adjacent frames among them:
    each a read of its own, and none bridging a frame that was not asked for (§9.2)."""
    run: list[int] = []
    for number in [*numbers, -1]:
        if run and number != run[-1] + 1:
            yield run
            run = []
        if number >= 0:
            run.append(number)


def _run(first: int, last: int, chrom: str) -> str:
    """A frame, or the run of frames one read served, as a refusal names it."""
    return _where(chrom, first) if first == last else f"frames {first}–{last} of chromosome {chrom}"


def open_partner_frames(root: zarr.Group, chrom: str, cache: FrameCache) -> PartnerFrames:
    """Chromosome `chrom`'s Partner Rows of `root`, opened for reading, refused naming the group
    when the store holds none for it (partner-rows §5.1)."""
    path = layout.partners_path(chrom)
    try:
        member = root[path]
    except KeyError:
        raise ValueError(
            f"{path} is missing: this LD Store holds no Partner Rows for chromosome {chrom} "
            "(partner-rows §5.1)"
        ) from None
    if not isinstance(member, zarr.Group):
        raise ValueError(
            f"{path} is not a group: a Partner Rows component is one (partner-rows §5.1)"
        )
    return PartnerFrames(member, chrom, cache)


class FramesGroup:
    """One chromosome's Partner Rows group, written a frame at a time (§5.1, §5.3; ogld#67).

    `add` takes a frame's rows, compresses them on a pool of `workers` threads and appends the
    bytes, in frame order, to the group's chunk file. Frames are appended strictly in order, so a
    chromosome's bytes are the same at any number of workers and however its rows were blocked.

    The `frames` array's length is the whole file's, which is not known until the last row, so the
    group is built in a temporary directory beside the store's other chromosomes — a dot-prefixed
    name, which `close` renames into `partners/<chrom>/` in one step — and until then nothing of
    it is a node of the store's hierarchy. `close` creates both arrays' metadata at their final
    shape, puts the file where that metadata's chunk key names, checks that zarr reads it back as
    the frames appended, sets the group's attributes and only then publishes it, so a chromosome
    is either whole or not there at all (issue #67).

    A group that fails — a frame that will not compress, a row count short, a chunk file zarr
    cannot read back — is poisoned: it removes its temporary directory, shuts its pool down, and
    every later `add` or `close` raises, naming the first failure. A component that is not whole
    is never finished and never published.

    One writer per store: concurrent writers are not supported, and the temporary group and the
    rename assume it. A temporary group another run left behind — a crash — is removed when its
    chromosome is written again.
    """

    def __init__(self, root: zarr.Group, chrom: str, rows: int, *, workers: int = 1) -> None:
        """The group of chromosome `chrom`, whose Variant Set has `rows` rows, written on
        `workers` compressor threads."""
        if rows < 1:
            raise ValueError(f"chromosome {chrom} has no rows, so it has no Partner Rows")
        if workers < 1:
            raise ValueError(
                f"workers counts compressor threads, so it must be at least 1: {workers}"
            )
        self._chrom = chrom
        self._rows = rows
        self._root = root
        # `partners/` is a node of the store's own hierarchy (spec §1): its metadata is what keeps
        # zarr's `members()` from warning that the component is not one.
        partner_rows = root.require_group(layout.PARTNERS)
        beside = layout.local_directory(partner_rows.store_path)
        self._final = beside / chrom
        if self._final.exists():
            raise FileExistsError(
                f"{layout.partners_path(chrom)} exists: a chromosome's Partner Rows are written "
                "once"
            )
        # A writer that crashed left a group of this chromosome at `.<chrom>.tmp-*`, which a
        # zarr listing warns about: none can be live, because a store has one writer (see below).
        for stale in beside.iterdir():
            if stale.name.startswith(f".{chrom}.tmp-"):
                shutil.rmtree(stale, ignore_errors=True)
        self._directory = Path(tempfile.mkdtemp(prefix=f".{chrom}.tmp-", dir=beside))
        # `mkdtemp` made the directory, so `w` only fills it; no other writer can see it.
        self._group = zarr.open_group(str(self._directory), mode="w", zarr_format=3)
        self._building = self._directory / layout.FRAMES_BUILDING
        self._file = self._building.open("wb")
        self._sha = hashlib.sha256()
        self._offsets = [0]
        self._bytes = 0
        self._entries = 0
        self._added = 0
        self._submitted = 0
        self._ahead = 2 * workers
        self._pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
        self._pending: list[Future[bytes]] = []
        self._failure: BaseException | None = None
        self._published = False
        self._closed = False

    @property
    def entries(self) -> int:
        """The entries appended so far, twice the pairs the rows hold (§5.3)."""
        return self._entries

    def add(self, rows: Sequence[tuple[npt.ArrayLike, npt.ArrayLike]]) -> None:
        """Append `rows` as one frame: the rows that follow the last frame appended.

        A frame holds `ROWS_PER_FRAME` rows, except a chromosome's last, which holds those left.
        """
        self._live()
        try:
            self._add(rows)
        except BaseException as failed:
            self._poison(failed)
            raise

    def _add(self, rows: Sequence[tuple[npt.ArrayLike, npt.ArrayLike]]) -> None:
        if rows and self._added % ROWS_PER_FRAME:
            raise ValueError(
                f"frame {self._submitted - 1} of chromosome {self._chrom} is still open: a frame "
                f"holds {ROWS_PER_FRAME} rows (spec §5.2)"
            )
        due = min(ROWS_PER_FRAME, self._rows - self._added)
        if len(rows) != due:
            raise ValueError(
                f"frame {self._submitted} of chromosome {self._chrom} holds {due} rows, not "
                f"{len(rows)} (spec §5.2)"
            )
        self._entries += sum(len(np.asarray(partners)) for partners, _ in rows)
        self._pending.append(self._start(self._added, rows))
        self._added += len(rows)
        if len(self._pending) > self._ahead:
            self._append()

    def _start(
        self, first: int, rows: Sequence[tuple[npt.ArrayLike, npt.ArrayLike]]
    ) -> Future[bytes]:
        """The frame's bytes, being compressed on the pool or made here when there is none.

        The frame is numbered where it is submitted: with a pool several are pending at once, and
        a refusal from one of them must name its own frame, not the one the offsets have reached.
        """
        number = self._submitted
        self._submitted += 1
        if self._pool is not None:
            return self._pool.submit(encode_frame, first, rows, chrom=self._chrom, number=number)
        done: Future[bytes] = Future()
        done.set_result(encode_frame(first, rows, chrom=self._chrom, number=number))
        return done

    def _append(self) -> None:
        """Append the frames' oldest bytes, waiting for its compression if it is still running.

        The future is taken before it is dropped: a frame that failed stays in the queue, so the
        group cannot go on as if it had been written.
        """
        raw = self._pending[0].result()
        self._pending.pop(0)
        self._file.write(raw)
        self._sha.update(raw)
        self._bytes += len(raw)
        self._offsets.append(self._bytes)

    def close(self) -> int:
        """Finish the group: its two arrays, its attributes and its chunk file, publishes it, and
        returns the entries it holds (§5.1, §5.3).

        Refuses a chromosome whose rows are not all written, and refuses a chunk file zarr cannot
        read back as the frames appended, before anything is published: a component that is not
        whole is one no provenance claims.
        """
        self._live()
        try:
            self._close()
        except BaseException as failed:
            self._poison(failed)
            raise
        self._closed = True
        return self._entries

    def _close(self) -> None:
        if self._added != self._rows:
            raise ValueError(
                f"chromosome {self._chrom}'s Partner Rows hold {self._added} of {self._rows} "
                "rows: a store assembly fault"
            )
        while self._pending:
            self._append()
        self._file.close()
        if self._pool is not None:
            self._pool.shutdown()
        offsets = self._offsets_array()
        frames_array = self._written()
        index = layout.new_frames_index(self._group, len(offsets))
        index[:] = offsets
        self._read_back(frames_array, offsets)
        self._group.attrs.update(
            {
                "rows": self._rows,
                "rows_per_frame": ROWS_PER_FRAME,
                "frames": len(offsets) - 1,
                "entries": self._entries,
                "frames_bytes": self._bytes,
                "frames_sha256": self._sha.hexdigest(),
                "frames_idx_sha256": _digest(layout.chunk_file(index)),
                "zstd_level": ZSTD_LEVEL,
            }
        )
        self._publish()

    def _live(self) -> None:
        """Refuse a group that already failed, naming its first failure, or one already closed."""
        if self._failure is not None:
            raise ValueError(
                f"chromosome {self._chrom}'s Partner Rows failed and are never finished: "
                f"{self._failure!r}"
            ) from self._failure
        if self._closed:
            raise ValueError(f"chromosome {self._chrom}'s Partner Rows are already closed")

    def _poison(self, failed: BaseException) -> None:
        """Keep the first failure, and leave nothing of a component that cannot be whole."""
        self._failure = failed
        self._file.close()
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
        shutil.rmtree(self._final if self._published else self._directory, ignore_errors=True)

    def _publish(self) -> None:
        """Make the whole group `partners/<chrom>/` in one rename, and refuse a store zarr cannot
        read it back from.

        A target that is there now, empty or not, is refused rather than replaced: `os.rename`
        would silently move this group over an empty directory. The check is as late as it can be,
        but a store is written by one writer at a time, so nothing else makes one meanwhile.
        """
        if self._final.exists():
            raise FileExistsError(
                f"{layout.partners_path(self._chrom)} exists: a chromosome's Partner Rows are "
                "written once, by one writer"
            )
        os.rename(self._directory, self._final)
        self._published = True
        published = zarr.open_group(str(self._final), mode="r")
        held = published["frames"]
        assert isinstance(held, zarr.Array)
        if held.shape != (self._bytes,) or published.attrs.get("frames_bytes") != self._bytes:
            raise ValueError(
                f"the published {layout.partners_path(self._chrom)} does not hold its frames: a "
                "store assembly fault"
            )

    def _offsets_array(self) -> npt.NDArray[np.uint64]:
        """The frames' byte offsets, F + 1 of them, refused unless they start at 0, ascend
        strictly and end at the file's length (§5.2, §5.3)."""
        offsets = np.array(self._offsets, dtype=_LITTLE_OFFSET)
        if offsets[0] != 0 or (np.diff(offsets) <= 0).any() or int(offsets[-1]) != self._bytes:
            raise ValueError(
                f"chromosome {self._chrom}'s frames do not tile its {self._bytes} bytes: a store "
                "assembly fault"
            )
        return offsets

    def _written(self) -> zarr.Array[Any]:
        """Create the `frames` array at the file's length and move the appended bytes to where its
        chunk key says they are, never assuming the name (§5.1)."""
        array = layout.new_frames_array(self._group, self._bytes)
        target = layout.chunk_file(array)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self._building, target)
        return array

    def _read_back(self, array: zarr.Array[Any], offsets: npt.NDArray[np.uint64]) -> None:
        """Refuse unless zarr reads the chunk file back as what was appended: the array is one
        chunk of every byte and holds nothing else, the file at the path its metadata gives is
        those bytes, whole, and zarr's own reads of it agree with the file's.

        A one-chunk array's slice read decodes the whole chunk, so the decode itself is checked
        whole only on a frames file within `_DECODE_LIMIT` bytes, and otherwise by the metadata,
        the hash and zarr's range reads. The tests hold the whole decode on the Test Store."""
        self._one_chunk(array)
        path = self._file_is_the_frames(array, offsets)
        self._ranges_agree(array, path)
        if self._bytes <= _DECODE_LIMIT:
            self._decodes_to(array, path)

    def _one_chunk(self, array: zarr.Array[Any]) -> None:
        """The array is one chunk of every byte the frames hold, and of nothing else."""
        if array.shape != (self._bytes,) or array.chunks != (self._bytes,):
            raise ValueError(
                f"chromosome {self._chrom}'s frames array has shape {array.shape} and chunks "
                f"{array.chunks}, not one chunk of its {self._bytes} bytes (spec §5.1)"
            )

    def _file_is_the_frames(self, array: zarr.Array[Any], offsets: npt.NDArray[np.uint64]) -> Path:
        """The file the array's metadata resolves, which holds the frames appended, whole."""
        path = layout.chunk_file(array)
        if path.stat().st_size != self._bytes or _digest(path) != self._sha.hexdigest():
            raise ValueError(
                f"{path} is not the frames appended to chromosome {self._chrom}'s Partner Rows: "
                "a store assembly fault"
            )
        if int(offsets[-1]) != path.stat().st_size:
            raise ValueError(f"chromosome {self._chrom}'s frames idx does not end at its file")
        return path

    def _ranges_agree(self, array: zarr.Array[Any], path: Path) -> None:
        """zarr's own reads of the file's first, middle and last bytes are the file's."""
        for at in sorted({0, self._bytes // 2, max(0, self._bytes - _READ_BACK)}):
            span = slice(at, min(self._bytes, at + _READ_BACK))
            try:
                read = _ranged(array, span)
            except ValueError as unread:
                raise ValueError(f"chromosome {self._chrom}: {unread}") from unread
            with path.open("rb") as held:
                held.seek(span.start)
                if read != held.read(span.stop - span.start):
                    raise ValueError(
                        f"zarr reads chromosome {self._chrom}'s frames back differently from the "
                        f"chunk file's own bytes at {span.start} (spec §5.1)"
                    )

    def _decodes_to(self, array: zarr.Array[Any], path: Path) -> None:
        """zarr decodes the whole chunk to the file's bytes, which it does read whole: bounded by
        `_DECODE_LIMIT`."""
        if bytes(np.asarray(array[:]).tobytes()) != path.read_bytes():
            raise ValueError(
                f"chromosome {self._chrom}'s frames do not decode to the chunk file's bytes "
                "(spec §5.1)"
            )


def _ranged(array: zarr.Array[Any], span: slice) -> bytes:
    """Bytes `span` of a one-chunk array's chunk file, read through the array's own store path and
    chunk key, as a reader's range request does (partner-rows §7)."""
    store = cast(LocalStore, array.store_path.store)
    key = array.store_path.path + "/" + array.metadata.encode_chunk_key((0,))
    fetched = store.get_sync(
        key,
        prototype=default_buffer_prototype(),
        byte_range=RangeByteRequest(span.start, span.stop),
    )
    if fetched is None:
        raise ValueError(f"{key} is not there: a frames chunk file a store lost")
    return bytes(fetched.to_bytes())


def _digest(path: Path) -> str:
    """A file's SHA-256, read in blocks so a chromosome's frames are never held whole."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
