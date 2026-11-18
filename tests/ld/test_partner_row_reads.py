"""A variant's Partner Rows row, read through a frame cache (partner-rows §5, §7; ogld#70).

Every expected entry is reconstructed here from the store's own Near Band and Far Pairs with plain
zarr and numpy, by partner-rows §3's rule written out in this file: every pair within `near_bp` at
the Partner Floor, every Far Pair beyond it, and each kept pair in both variants' rows.
`partner_row` reads none of that — it reads one frame's byte range from `partners/<chrom>/frames` —
so a second layout of the same codes is held to the codes, not to itself.

The damage cases rewrite a copy of the committed Test Store, whose frames are #67's; nothing here
writes into `tests/data/`. Each is held on both transport paths — a local directory and the same
directory over HTTP — because §7 gives them one read.
"""

from __future__ import annotations

import contextlib
import http.server
import json
import os
import shutil
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cache, partial
from pathlib import Path
from typing import Any

import numcodecs
import numpy as np
import numpy.typing as npt
import pytest
import zarr
from contract.support import RangeHandler
from store_copies import PARTNER_FLOOR, TEST_STORE
from zarr.abc.store import RangeByteRequest

from ogld.format import InputError
from ogld.store import frames
from ogld.store.reader import StoreReader

#: The Test Store's distances (spec §7). Its scale and Partner Floor are `tests/store_copies.py`'s,
#: the one independent owner of its encoding (ownership.md, "Test oracles"), so the reconstruction
#: below still depends on nothing in `frames.py`.
NEAR_BP, FAR_BP = 3_000_000, 10_000_000
CHROMOSOMES = ("21", "22")
#: The frames the Test Store's chr22 has, and the one the damage cases damage.
CHR22_FRAMES = 37
CHR22_ROWS = 588
DAMAGED_FRAME = 5

#: zstd at a frame's own level, with the content checksum its header carries (§5.2).
_ZSTD = numcodecs.Zstd(level=frames.ZSTD_LEVEL, checksum=True)

Pairs = tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int16]]


def store() -> zarr.Group:
    return zarr.open_group(str(TEST_STORE), mode="r")


def chromosomes(group: zarr.Group) -> dict[str, tuple[int, int]]:
    listed: list[dict[str, Any]] = group["variants"].attrs["chromosomes"]
    return {entry["name"]: (entry["row_start"], entry["row_stop"]) for entry in listed}


def near_pairs(group: zarr.Group, chrom: str) -> Pairs:
    """Every Near Band pair of one chromosome, from spec §4's address rule written out here: the
    code of pair (l, m) is at `r[indptr[l] + (m − l − 1)]`, so its address gives its pair."""
    held = group["near"][chrom]
    indptr = np.asarray(held["indptr"][:], dtype=np.int64)
    codes = np.asarray(held["r"][:], dtype=np.int16)
    lengths = np.diff(indptr)
    row = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
    offset = np.arange(len(codes), dtype=np.int64) - np.repeat(indptr[:-1], lengths)
    return row, row + 1 + offset, codes


def far_pairs(group: zarr.Group, chrom: str) -> Pairs:
    """Every Far Pair of one chromosome, each in its earlier row: (row, later row, code)."""
    held = group["far"][chrom]
    indptr = np.asarray(held["indptr"][:], dtype=np.int64)
    rows = np.repeat(np.arange(len(indptr) - 1, dtype=np.int64), np.diff(indptr))
    return rows, np.asarray(held["partner"][:], dtype=np.int64), np.asarray(held["r"][:], np.int16)


@cache
def expected(chrom: str) -> Pairs:
    """Each row's partners and codes as partner-rows §3 requires them: every Near Band pair at the
    Partner Floor and every Far Pair, in both variants' rows, ascending within a row."""
    group = store()
    near_lo, near_hi, near_codes = near_pairs(group, chrom)
    far_lo, far_hi, far_codes = far_pairs(group, chrom)
    start, stop = chromosomes(group)[chrom]
    reaches = np.abs(near_codes) >= PARTNER_FLOOR
    lo = np.concatenate([near_lo[reaches], near_hi[reaches], far_lo, far_hi])
    hi = np.concatenate([near_hi[reaches], near_lo[reaches], far_hi, far_lo])
    codes = np.concatenate([near_codes[reaches], near_codes[reaches], far_codes, far_codes])
    order = np.lexsort((hi, lo))
    lo, hi, codes = lo[order], hi[order], codes[order]
    assert len(lo) > 1_000, "the Test Store's Partner Rows must hold many entries"
    assert (hi < lo).any(), "a row holds upstream partners"
    assert (hi > lo).any(), "and downstream ones"
    indptr = np.concatenate([[0], np.cumsum(np.bincount(lo, minlength=stop - start))]).astype(
        np.int64
    )
    return indptr, hi, codes


@pytest.mark.parametrize("chrom", CHROMOSOMES)
def test_every_row_is_the_near_band_and_the_far_pairs_at_their_floors(chrom: str) -> None:
    reader = StoreReader.open(TEST_STORE)
    indptr, partners, codes = expected(chrom)
    read = 0
    for local_row in range(len(indptr) - 1):
        got_partners, got_codes = reader.partner_row(chrom, local_row)
        span = slice(int(indptr[local_row]), int(indptr[local_row + 1]))
        np.testing.assert_array_equal(got_partners, partners[span], f"row {local_row}'s partners")
        np.testing.assert_array_equal(got_codes, codes[span], f"row {local_row}'s codes")
        read += len(got_partners)
    assert read == len(partners), "every entry the rows hold was read"


@contextlib.contextmanager
def serving(directory: Path) -> Iterator[tuple[str, list[tuple[str, str | None]]]]:
    """`directory` over HTTP, with the path and `Range` header of every request answered."""
    seen: list[tuple[str, str | None]] = []

    class Counting(RangeHandler):
        def send_head(self) -> Any:
            seen.append((self.path, self.headers.get("Range")))
            return super().send_head()

    handler = partial(Counting, directory=str(directory))
    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        worker = threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True)
        worker.start()
        yield f"http://127.0.0.1:{server.server_address[1]}/", seen
        server.shutdown()
        worker.join()


def sampled(chrom: str) -> list[int]:
    """The rows an HTTP pass reads: every seventh row, and the first and last row of every frame, so
    every frame of the chromosome is read at both its ends without reading all of its rows. The
    frames are the store's own, from its `rows_per_frame`, never a constant (§5.2)."""
    held = store()["partners"][chrom]
    rows, per_frame = int(held.attrs["rows"]), int(held.attrs["rows_per_frame"])
    firsts = range(0, rows, per_frame)
    lasts = [min(first + per_frame - 1, rows - 1) for first in firsts]
    return sorted({*range(0, rows, 7), *firsts, *lasts})


def test_rows_over_http_are_the_near_band_and_the_far_pairs_at_their_floors() -> None:
    with serving(TEST_STORE) as (url, _):
        reader = StoreReader.open(url)
        for chrom in CHROMOSOMES:
            indptr, partners, codes = expected(chrom)
            rows = sampled(chrom)
            assert len(rows) > 100, "the sample must touch most frames and rows"
            for local_row in rows:
                span = slice(int(indptr[local_row]), int(indptr[local_row + 1]))
                got_partners, got_codes = reader.partner_row(chrom, local_row)
                np.testing.assert_array_equal(got_partners, partners[span])
                np.testing.assert_array_equal(got_codes, codes[span])


def test_a_row_decoded_alone_is_the_row_of_the_whole_frame() -> None:
    """The row-local gather and the whole-frame decode are two readings of one payload, so they
    must give the same row for every frame of the committed store: the row-local one is what a read
    uses, and the whole-frame one is what validation reads (§7)."""
    group = store()["partners"]
    for chrom in CHROMOSOMES:
        held = group[chrom]
        rows, per_frame = int(held.attrs["rows"]), int(held.attrs["rows_per_frame"])
        offsets = np.asarray(held["frames_idx"][:], dtype=np.int64)
        raw = np.asarray(held["frames"][:], dtype=np.uint8).tobytes()
        assert len(offsets) > 3, "the Test Store's chromosome holds several frames"
        for number in range(len(offsets) - 1):
            first = number * per_frame
            payload = frames.FramePayload.parse(
                raw[offsets[number] : offsets[number + 1]],
                chrom=chrom,
                number=number,
                first=first,
                rows=min(per_frame, rows - first),
            )
            whole = payload.decode()
            for local_row in range(first, first + payload.rows):
                alone_partners, alone_codes = payload.row(local_row)
                frame_partners, frame_codes = whole.row(local_row)
                np.testing.assert_array_equal(alone_partners, frame_partners)
                np.testing.assert_array_equal(alone_codes, frame_codes)


Served = tuple[str, Path]


@pytest.fixture(params=["local", "http"])
def test_store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Served]:
    """A copy of the committed Test Store, as a local path and over HTTP: both are one read path
    (§7), so each damage case is held on both."""
    copied = tmp_path / "store"
    shutil.copytree(TEST_STORE, copied)
    if request.param == "local":
        yield str(copied), copied
    else:
        with serving(copied) as (url, _):
            yield url, copied


@dataclass
class Reads:
    """The chunk-file reads one reader made, however they were served (§7): `os.pread` calls on a
    local store's kept descriptor, and byte ranges asked of the store on a remote one."""

    preads: list[tuple[int, int]] = field(default_factory=list)
    ranges: list[tuple[str, int, int]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.preads) + len(self.ranges)


@contextlib.contextmanager
def counted_reads(reader: StoreReader, monkeypatch: pytest.MonkeyPatch) -> Iterator[Reads]:
    """Count a reader's chunk-file reads on both transport paths, so a count is a count either way:
    `os.pread(length, offset)` for a local store, a `RangeByteRequest(key, start, stop)` for one
    that is not. Opening a chromosome's group, its `frames_idx` chunk and any whole-chunk read are
    none of them: only a frame's bytes are."""
    held = reader._root.store  # the store every array of the reader shares
    reads = Reads()
    original_get = held.get
    original_pread = os.pread

    async def get(key: str, prototype: Any, byte_range: Any = None) -> Any:
        if isinstance(byte_range, RangeByteRequest):
            reads.ranges.append((key, byte_range.start, byte_range.end))
        return await original_get(key, prototype, byte_range)

    def pread(fd: int, length: int, offset: int) -> bytes:
        reads.preads.append((length, offset))
        return original_pread(fd, length, offset)

    monkeypatch.setattr(held, "get", get)
    monkeypatch.setattr(os, "pread", pread)
    yield reads


def frames_read(seen: list[tuple[str, str | None]]) -> list[tuple[str, str | None]]:
    """The requests that fetched a frame's bytes: a `Range` read of a `frames` chunk file."""
    return [
        (path, header)
        for path, header in seen
        if "/partners/" in path and "/frames/c/" in path and header is not None
    ]


def frame_range(store_root: Path, chrom: str, number: int) -> tuple[int, int]:
    """The byte range frame `number` of `chrom` occupies in the group's chunk file (§5.2)."""
    offsets = np.fromfile(
        store_root / "partners" / chrom / "frames_idx" / "c" / "0", dtype="<u8"
    ).astype(np.int64)
    return int(offsets[number]), int(offsets[number + 1])


def test_one_uncached_row_is_one_positional_read_and_a_kept_frame_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = StoreReader.open(TEST_STORE)
    start, stop = frame_range(TEST_STORE, "22", 6)
    with counted_reads(reader, monkeypatch) as reads:
        reader.partner_row("22", 100)
        assert reads.preads == [(stop - start, start)], (
            "one uncached row is one pread of the frame that holds it, at its offset"
        )
        assert reads.ranges == [], "a local store's frames take no store call and no event loop"
        assert len(reader.partner_row("22", 100)[0]) > 0
        assert reads.total == 1, "the kept frame is read no more"
        reader.partner_row("22", 0)
        assert reads.total == 2, "a row in another frame is another read"


def test_one_uncached_row_over_http_is_one_range_request() -> None:
    with serving(TEST_STORE) as (url, seen):
        reader = StoreReader.open(url)
        reader.partner_row("22", 100)
        assert len(frames_read(seen)) == 1, "one uncached row is one Range request"
        assert len(reader.partner_row("22", 100)[0]) > 0
        assert len(frames_read(seen)) == 1, "the kept frame is fetched no more"


def test_a_bulk_read_coalesces_only_physically_adjacent_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = StoreReader.open(TEST_STORE)
    start = frame_range(TEST_STORE, "22", 0)[0]
    stop = frame_range(TEST_STORE, "22", 1)[1]
    with counted_reads(reader, monkeypatch) as reads:
        indptr, partners, codes = reader.partner_rows("22", np.array([0, 16]))
        assert reads.preads == [(stop - start, start)], "two adjacent frames, one read"
    assert len(indptr) == 3, "one indptr entry a row, and one more"
    for i, local_row in enumerate((0, 16)):
        want_partners, want_codes = reader.partner_row("22", local_row)
        np.testing.assert_array_equal(partners[indptr[i] : indptr[i + 1]], want_partners)
        np.testing.assert_array_equal(codes[indptr[i] : indptr[i + 1]], want_codes)
    fresh = StoreReader.open(TEST_STORE)
    with counted_reads(fresh, monkeypatch) as reads:
        fresh.partner_rows("22", np.array([0, 80]))
        assert reads.total == 2, "frames 0 and 5 are not adjacent, so each is a read of its own"


class Decompressions:
    """Counts a frame's decompressions, and does what `_ZSTD` does."""

    def __init__(self, codec: Any) -> None:
        self._codec = codec
        self.count = 0

    def decode(self, raw: bytes) -> Any:
        self.count += 1
        return self._codec.decode(raw)

    def encode(self, payload: bytes) -> Any:
        return self._codec.encode(payload)


def test_a_bulk_read_decodes_its_rows_from_the_run_it_fetched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bulk read answers from the bytes it just read, not from the cache it hopes still holds
    them: with no cache at all, two adjacent rows are one read and each frame one decompress."""
    reader = StoreReader.open(TEST_STORE, frame_cache_bytes=0)
    counting = Decompressions(frames._ZSTD)
    monkeypatch.setattr(frames, "_ZSTD", counting)
    with counted_reads(reader, monkeypatch) as reads:
        indptr, partners, codes = reader.partner_rows("22", np.array([0, 16]))
    assert reads.total == 1, "two adjacent frames, one read, even with no cache"
    assert counting.count == 2, "each frame decompressed once, not once a row"
    assert len(indptr) == 3
    for i, local_row in enumerate((0, 16)):
        want_partners, want_codes = reader.partner_row("22", local_row)
        np.testing.assert_array_equal(partners[indptr[i] : indptr[i + 1]], want_partners)
        np.testing.assert_array_equal(codes[indptr[i] : indptr[i + 1]], want_codes)


def test_a_bulk_read_answers_each_row_asked_for_in_order() -> None:
    reader = StoreReader.open(TEST_STORE)
    asked = np.array([33, 3, 33, 500])
    indptr, partners, codes = reader.partner_rows("22", asked)
    assert len(indptr) == len(asked) + 1, "one indptr entry a row, and one more"
    for i, local_row in enumerate(asked):
        want_partners, want_codes = reader.partner_row("22", int(local_row))
        np.testing.assert_array_equal(partners[indptr[i] : indptr[i + 1]], want_partners)
        np.testing.assert_array_equal(codes[indptr[i] : indptr[i + 1]], want_codes)


def test_four_threads_first_use_read_the_offsets_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The chromosome's first use is one load, however many threads make it (§7): the offsets chunk
    is read once and one descriptor is opened, not one per thread."""
    reader = StoreReader.open(TEST_STORE)
    held = reader._root.store
    seen: list[str] = []
    original = held.get

    async def get(key: str, prototype: Any, byte_range: Any = None) -> Any:
        if "frames_idx/c/" in key:
            seen.append(key)
        return await original(key, prototype, byte_range)

    monkeypatch.setattr(held, "get", get)
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda _: reader.partner_row("22", 100), range(4)))
    assert len(seen) == 1, f"four first calls read the offsets chunk {len(seen)} times"
    for partners, codes in rows[1:]:
        np.testing.assert_array_equal(partners, rows[0][0])
        np.testing.assert_array_equal(codes, rows[0][1])


def test_a_reader_closes_the_chunk_file_it_kept_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local frame read keeps one descriptor for the reader's life, and the reader closes it
    (§7): closed twice is closed once."""
    closed: list[int] = []
    original = os.close
    monkeypatch.setattr(os, "close", lambda fd: closed.append(fd) or original(fd))
    reader = StoreReader.open(TEST_STORE)
    reader.partner_row("22", 100)
    reader.partner_row("21", 100)
    assert closed == [], "an open reader keeps its descriptors"
    reader.close()
    assert len(closed) == 2, "one descriptor a chromosome read, closed with the reader"
    reader.close()
    assert len(closed) == 2, "closing a closed reader closes nothing"


def test_four_threads_reading_one_frame_get_equal_rows() -> None:
    reader = StoreReader.open(TEST_STORE)
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda _: reader.partner_row("22", 100), range(8)))
    assert len(rows[0][0]) > 0, "the row must hold entries"
    for partners, codes in rows[1:]:
        np.testing.assert_array_equal(partners, rows[0][0])
        np.testing.assert_array_equal(codes, rows[0][1])


def test_the_frame_cache_keeps_no_more_than_its_bytes() -> None:
    budget = 8_000
    reader = StoreReader.open(TEST_STORE, frame_cache_bytes=budget)
    assert reader.frame_cache_bytes == budget, "the frame cache has its own budget"
    assert reader.cache_bytes != budget, "and it is not the chunk cache's"
    for local_row in range(CHR22_ROWS):
        reader.partner_row("22", local_row)
    held = reader._frame_cache.held
    assert 0 < held <= budget, f"a frame cache of {budget} bytes held {held}"


def test_a_frame_cache_of_no_bytes_keeps_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    reader = StoreReader.open(TEST_STORE, frame_cache_bytes=0)
    with counted_reads(reader, monkeypatch) as reads:
        reader.partner_row("22", 100)
        reader.partner_row("22", 100)
        assert reads.total == 2, "nothing kept, so the second read fetches the frame again"
    assert reader._frame_cache.held == 0


# ---------------------------------------------------------------- refusing damage


def frame_bytes(store_root: Path, chrom: str, number: int) -> bytes:
    held = (store_root / "partners" / chrom / "frames" / "c" / "0").read_bytes()
    start, stop = frame_range(store_root, chrom, number)
    return held[start:stop]


def payload(raw: bytes) -> bytes:
    return bytes(_ZSTD.decode(raw))


def restamped(store_root: Path, chrom: str, number: int, raw: bytes) -> None:
    """`chrom`'s frames with frame `number` replaced by `raw`, its `frames_idx` and its group's
    `frames_bytes` rewritten to match: a store whose frame is damaged and whose metadata is right,
    so the frame's own checks are what refuse it (§5.4)."""
    directory = store_root / "partners" / chrom
    kept = (directory / "frames" / "c" / "0").read_bytes()
    offsets = np.fromfile(directory / "frames_idx" / "c" / "0", dtype="<u8").astype(np.int64)
    pieces = [
        raw if at == number else kept[int(offsets[at]) : int(offsets[at + 1])]
        for at in range(len(offsets) - 1)
    ]
    written = [0]
    for piece in pieces:
        written.append(written[-1] + len(piece))
    (directory / "frames" / "c" / "0").write_bytes(b"".join(pieces))
    (directory / "frames_idx" / "c" / "0").write_bytes(np.array(written, dtype="<u8").tobytes())
    meta = json.loads((directory / "frames" / "zarr.json").read_text())
    meta["shape"] = [written[-1]]
    meta["chunk_grid"]["configuration"]["chunk_shape"] = [written[-1]]
    (directory / "frames" / "zarr.json").write_text(json.dumps(meta))
    group = json.loads((directory / "zarr.json").read_text())
    group["attributes"]["frames_bytes"] = written[-1]
    (directory / "zarr.json").write_text(json.dumps(group))


def test_a_flipped_byte_in_a_frame_is_refused_naming_the_chromosome_and_the_frame(
    test_store: Served,
) -> None:
    location, copied = test_store
    raw = bytearray(frame_bytes(copied, "22", DAMAGED_FRAME))
    # The last bytes are zstd's content checksum, so the payload itself still decodes: only the
    # checksum the frame's header carries can refuse this one (§5.2).
    raw[-1] ^= 0xFF
    restamped(copied, "22", DAMAGED_FRAME, bytes(raw))
    with pytest.raises(ValueError, match=rf"frame {DAMAGED_FRAME} of chromosome 22"):
        StoreReader.open(location).partner_row("22", DAMAGED_FRAME * 16 + 1)


def test_a_short_payload_is_refused_naming_the_chromosome_and_the_frame(
    test_store: Served,
) -> None:
    """A payload one entry shorter than its lengths say, recompressed with a valid checksum: the one
    thing wrong with it is that it is short (§5.2)."""
    location, copied = test_store
    raw = frame_bytes(copied, "22", DAMAGED_FRAME)
    held = payload(raw)
    shorter = held[:-6]
    assert len(shorter) == len(held) - 6, "one entry's bytes are 4 of a delta and 2 of a code"
    restamped(copied, "22", DAMAGED_FRAME, bytes(_ZSTD.encode(shorter)))
    with pytest.raises(ValueError, match=rf"frame {DAMAGED_FRAME} of chromosome 22.*4 \+ 4k \+ 6E"):
        StoreReader.open(location).partner_row("22", DAMAGED_FRAME * 16 + 1)


def test_an_oversized_payload_is_refused_naming_the_chromosome_and_the_frame(
    test_store: Served,
) -> None:
    location, copied = test_store
    raw = frame_bytes(copied, "22", DAMAGED_FRAME)
    restamped(copied, "22", DAMAGED_FRAME, bytes(_ZSTD.encode(payload(raw) + b"\x00")))
    with pytest.raises(ValueError, match=rf"frame {DAMAGED_FRAME} of chromosome 22.*4 \+ 4k \+ 6E"):
        StoreReader.open(location).partner_row("22", DAMAGED_FRAME * 16 + 1)


def test_a_frame_without_a_content_checksum_is_refused_naming_the_chromosome_and_the_frame(
    test_store: Served,
) -> None:
    """A frame compressed without its content checksum cannot be verified, so it is refused before
    it is decoded (§5.2): the header must declare the checksum."""
    location, copied = test_store
    raw = frame_bytes(copied, "22", DAMAGED_FRAME)
    unchecked = numcodecs.Zstd(level=frames.ZSTD_LEVEL, checksum=False).encode(payload(raw))
    restamped(copied, "22", DAMAGED_FRAME, bytes(unchecked))
    with pytest.raises(
        ValueError, match=rf"frame {DAMAGED_FRAME} of chromosome 22.*content checksum"
    ):
        StoreReader.open(location).partner_row("22", DAMAGED_FRAME * 16 + 1)


def test_a_frame_without_a_content_size_is_refused_naming_the_chromosome_and_the_frame(
    test_store: Served,
) -> None:
    """A frame whose header declares no content size is refused too (§5.2). Its descriptor has the
    content-size flag and the single-segment flag cleared; the check reads the header alone, so the
    declaration is what refuses it."""
    location, copied = test_store
    raw = bytearray(frame_bytes(copied, "22", DAMAGED_FRAME))
    assert raw[4] >> 6, "the committed frame declares its content size"
    raw[4] &= ~((0b11 << 6) | (1 << 5))
    restamped(copied, "22", DAMAGED_FRAME, bytes(raw))
    with pytest.raises(ValueError, match=rf"frame {DAMAGED_FRAME} of chromosome 22.*content size"):
        StoreReader.open(location).partner_row("22", DAMAGED_FRAME * 16 + 1)


def test_a_frame_of_the_wrong_k_is_refused_naming_the_chromosome_and_the_frame(
    test_store: Served,
) -> None:
    location, copied = test_store
    held = payload(frame_bytes(copied, "22", DAMAGED_FRAME))
    wrong = int(np.frombuffer(held, "<u4", 1)[0]) + 1
    k = np.array([wrong], "<u4").tobytes()
    restamped(copied, "22", DAMAGED_FRAME, bytes(_ZSTD.encode(k + held[4:])))
    with pytest.raises(
        ValueError, match=rf"frame {DAMAGED_FRAME} of chromosome 22 holds {wrong} rows"
    ):
        StoreReader.open(location).partner_row("22", DAMAGED_FRAME * 16 + 1)


def test_a_truncated_frames_file_is_refused_naming_the_chromosome_and_the_frame(
    test_store: Served,
) -> None:
    location, copied = test_store
    held = copied / "partners" / "22" / "frames" / "c" / "0"
    held.write_bytes(held.read_bytes()[:-64])
    with pytest.raises(
        ValueError, match=rf"frame {CHR22_FRAMES - 1} of chromosome 22 holds \d+ of the \d+ bytes"
    ):
        StoreReader.open(location).partner_row("22", CHR22_ROWS - 1)


def test_a_frames_idx_that_does_not_end_at_frames_bytes_is_refused(test_store: Served) -> None:
    location, copied = test_store
    directory = copied / "partners" / "22"
    group = json.loads((directory / "zarr.json").read_text())
    group["attributes"]["frames_bytes"] += 1
    (directory / "zarr.json").write_text(json.dumps(group))
    with pytest.raises(ValueError, match=r"chromosome 22.*frames_idx.*frames_bytes"):
        StoreReader.open(location).partner_row("22", 0)


def test_a_store_whose_partner_rows_group_is_missing_is_refused(test_store: Served) -> None:
    location, copied = test_store
    shutil.rmtree(copied / "partners" / "22")
    with pytest.raises(ValueError, match=r"partners/22 is missing"):
        StoreReader.open(location).partner_row("22", 0)
    reader = StoreReader.open(location)
    assert len(reader.partner_row("21", 5)[0]) > 0, "chr21's rows are intact"


def test_a_row_outside_its_chromosome_is_refused() -> None:
    reader = StoreReader.open(TEST_STORE)
    with pytest.raises(ValueError, match=r"row 588 is not in chromosome 22"):
        reader.partner_row("22", CHR22_ROWS)
    with pytest.raises(ValueError, match=r"no chromosome 23"):
        reader.partner_row("23", 0)


#: The provenance a 0.2.0 store adds for the Partner Rows (spec §7; partner-rows §5.3): a 0.1.0
#: store records none of it, because it holds no Partner Rows.
PARTNER_KEYS = ("t_partner", "partner_code_floor")


def test_a_0_1_0_store_is_refused_naming_the_missing_component(tmp_path: Path) -> None:
    copied = tmp_path / "store"
    shutil.copytree(TEST_STORE, copied)
    root = copied / "zarr.json"
    provenance = json.loads(root.read_text())
    recorded = provenance["attributes"]["ld_store"]
    recorded["format_version"] = "0.1.0"
    for key in PARTNER_KEYS:
        del recorded[key]
    del recorded["software"]["zstd"]
    del recorded["counts"]["partner_entries"]
    root.write_text(json.dumps(provenance))
    reader = StoreReader.open(copied)
    assert reader.provenance["format_version"] == "0.1.0", "the copy is a 0.1.0 store"
    with pytest.raises(InputError, match=r"Partner Rows"):
        reader.partner_row("22", 0)
