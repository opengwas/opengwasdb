"""The Partner Rows' frames and their group (partner-rows §5.1, §5.2; ogld#67).

The Partner Rows hold, for each variant, every partner within `far_bp` whose code reaches the
Partner Floor within `near_bp` or the far code floor beyond it, in both variants' rows. They are
grouped `ROWS_PER_FRAME` rows to a zstd frame, and one chromosome's frames are the bytes of a
single-chunk `uint8` array, so a row is one positional read and one decompress (partner-rows §5.4).

This module owns both directions of that codec and the group around it:

- `encode_frame` and `decode_frame` write and read a frame's byte layout (§5.2): k, the rows'
  lengths, the partner deltas and the codes, the two arrays byte-shuffled, as one zstd frame at
  level 3 with the content checksum and the content size set in its header. `Frame` is a decoded
  frame, and `Frame.row` gives one row's partners and codes. opengwas/ogld#68 validates frames
  through them and opengwas/ogld#70 reads rows with them; neither parses a payload of its own.
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
from typing import Any, Final
import numpy as np
import numpy.typing as npt
import zarr
from zarr.abc.store import RangeByteRequest
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage import LocalStore
from ogquery.ld.format import Codes, InputError
from opengwasdb.ld.format import REncoding
from ogquery.ld.format.geometry import indptr_of, owners
from opengwasdb.ld.store import layout
from ogquery.ld.store.location import FrameSource, frame_source

# The read side of this module is ogquery's (opengwas/ogquery#2): re-exported by name.
from ogquery.ld.store.frames import (
    Frame as Frame,
    FRAME_CACHE_BYTES as FRAME_CACHE_BYTES,
    FrameCache as FrameCache,
    FramePayload as FramePayload,
    open_partner_frames as open_partner_frames,
    PartnerFrames as PartnerFrames,
    ZSTD_LEVEL as ZSTD_LEVEL,
    ZSTD_MAGIC as ZSTD_MAGIC,
)
from ogquery.ld.store.frames import _LITTLE_CODES
from ogquery.ld.store.frames import _LITTLE_DELTA
from ogquery.ld.store.frames import _LITTLE_ROW
from ogquery.ld.store.frames import _NUMCODECS
from ogquery.ld.store.frames import _ZSTD
from ogquery.ld.store.frames import _where


#: Rows a frame holds (partner-rows §1, §5.2: G = 16, the smallest G of those equally fast). A
#: reader takes it from the group's `rows_per_frame` attribute, never from here.
ROWS_PER_FRAME: Final = 16


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
            "numcodecs does not say which zstd it holds, and a frame's bytes depend on it "
            "(ld-store §7)"
        )
    return version


def _shuffled(values: npt.NDArray[Any]) -> bytes:
    """`values` as planes: every element's byte 0, then every element's byte 1, and so on (§5.2).

    Deltas are small, so their high planes are almost all 0 or 0xFF and compress to nearly nothing.
    """
    planes = np.ascontiguousarray(values).view("u1").reshape(-1, values.dtype.itemsize)
    return planes.T.tobytes()


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
                "codes are its pairs' (partner-rows §5.2)"
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
            raise ValueError(f"row {int(row[at[0]])} holds {what} (ld-store §3, partner-rows §5.2)")


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
            "chromosome's rows (partner-rows §5.2)"
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
        raise ValueError(f"{where} holds no rows: a frame holds at least one (partner-rows §5.2)")
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


def decode_frame(raw: bytes, *, chrom: str, number: int, first: int, rows: int) -> Frame:
    """The frame `raw` holds, which must be rows [first, first + `rows`) of chromosome `chrom`.

    The chromosome, the frame's number and the rows it is due are the caller's: they are what a
    refusal names, and what a payload is checked against, so a frame is never read as a short or a
    borrowed row (§5.4). A wrong k, a payload that is not exactly 4 + 4k + 6E bytes and a failed
    checksum all raise `ValueError` naming the chromosome and the frame.
    """
    return FramePayload.parse(raw, chrom=chrom, number=number, first=first, rows=rows).decode()


def _beside(root: zarr.Group, container: Path | None) -> Path:
    """The directory a chromosome's group is built in and published into: `container`, or the
    store's own `partners/` group, which this makes a node of its hierarchy (ld-store §1,
    partner-rows §5.1).

    A derivation supplies a plain directory and makes it a node only once every chromosome is in
    place, so a run killed part-way leaves the store it was (opengwas/ogld#69)."""
    if container is None:
        return layout.local_directory(root.require_group(layout.PARTNERS).store_path)
    return container


def _remove_stale(beside: Path, chrom: str) -> None:
    """Remove what a writer that crashed left of this chromosome: a group at `.<chrom>.tmp-*`
    beside where it is published, which a zarr listing warns about. None can be live, because a
    store has one writer (see `FramesGroup`)."""
    for stale in beside.iterdir():
        if stale.name.startswith(f".{chrom}.tmp-"):
            shutil.rmtree(stale, ignore_errors=True)


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
    is either whole or not there at all (issue opengwas/ogld#67).

    A group that fails — a frame that will not compress, a row count short, a chunk file zarr
    cannot read back — is poisoned: it removes its temporary directory, shuts its pool down, and
    every later `add` or `close` raises, naming the first failure. A component that is not whole
    is never finished and never published.

    One writer per store: concurrent writers are not supported, and the temporary group and the
    rename assume it. A temporary group another run left behind — a crash — is removed when its
    chromosome is written again.

    `container` is the directory the group is built in and published into; left out, it is the
    store's `partners/` group, which this then creates. `opengwasdb ld derive-partners` gives a
    plain directory that is not yet a node of the store's hierarchy, so a derivation killed part-way
    leaves the store it was: a `partners/` node is one validation judges and a reader trusts
    (partner-rows §5.1, §6.2; issue opengwas/ogld#69).
    """

    def __init__(
        self,
        root: zarr.Group,
        chrom: str,
        rows: int,
        *,
        workers: int = 1,
        container: Path | None = None,
    ) -> None:
        """The group of chromosome `chrom`, whose Variant Set has `rows` rows, written on
        `workers` compressor threads, into `container` when one is given."""
        if rows < 1:
            raise ValueError(f"chromosome {chrom} has no rows, so it has no Partner Rows")
        if workers < 1:
            raise ValueError(
                f"workers counts compressor threads, so it must be at least 1: {workers}"
            )
        self._chrom = chrom
        self._rows = rows
        self._root = root
        beside = _beside(root, container)
        self._final = beside / chrom
        if self._final.exists():
            raise FileExistsError(
                f"{layout.partners_path(chrom)} exists: a chromosome's Partner Rows are written "
                "once"
            )
        _remove_stale(beside, chrom)
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
                f"holds {ROWS_PER_FRAME} rows (partner-rows §5.2)"
            )
        due = min(ROWS_PER_FRAME, self._rows - self._added)
        if len(rows) != due:
            raise ValueError(
                f"frame {self._submitted} of chromosome {self._chrom} holds {due} rows, not "
                f"{len(rows)} (partner-rows §5.2)"
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
        offsets = np.array(self._offsets, dtype=layout.FRAMES_INDEX_DTYPE)
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
                f"{array.chunks}, not one chunk of its {self._bytes} bytes (partner-rows §5.1)"
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
                        f"chunk file's own bytes at {span.start} (partner-rows §5.1)"
                    )

    def _decodes_to(self, array: zarr.Array[Any], path: Path) -> None:
        """zarr decodes the whole chunk to the file's bytes, which it does read whole: bounded by
        `_DECODE_LIMIT`."""
        if bytes(np.asarray(array[:]).tobytes()) != path.read_bytes():
            raise ValueError(
                f"chromosome {self._chrom}'s frames do not decode to the chunk file's bytes "
                "(partner-rows §5.1)"
            )


def _ranged(array: zarr.Array[Any], span: slice) -> bytes:
    """Bytes `span` of a one-chunk array's chunk file, read through the array's own store path and
    chunk key, as a reader's range request does (partner-rows §7).

    A local store's own synchronous read is taken as it is; any other store — an HTTP one, which
    has no synchronous read of its own — is read through zarr's async API, one Range request.
    """
    store = array.store_path.store
    key = layout.chunk_key(array)
    request = RangeByteRequest(span.start, span.stop)
    prototype = default_buffer_prototype()
    if isinstance(store, LocalStore):
        fetched = store.get_sync(key, prototype=prototype, byte_range=request)
    else:
        fetched = sync(store.get(key, prototype=prototype, byte_range=request))
    if fetched is None:
        raise ValueError(f"{key} is not there: a frames chunk file a store lost")
    return bytes(fetched.to_bytes())


def _digest(path: Path) -> str:
    """A file's SHA-256, read in blocks so a chromosome's frames are never held whole."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
