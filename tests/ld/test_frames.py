"""The Partner Rows' frames and their group (partner-rows §5.1, §5.2; ogld#67).

The frame's payload is written out here as a literal, byte by byte, so the encoder and the decoder
are held to the spec's layout rather than to each other. Two rows of 16 are a frame: row 5 holds
partners 3 and 7 at codes 52 and −90, row 6 holds partner 8 at code 120. Its deltas are m₀ − l for
each row's first partner (−2 and 2) and mᵢ − mᵢ₋₁ after that (4), and both planes are byte-shuffled.

The group's tests read the chunk file back at the path the array's own chunk key encoding gives,
never a name written here, as the spec says a reader must (partner-rows §5.1).
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Sequence
from pathlib import Path

import numcodecs
import numpy as np
import pytest
import zarr

from ogld.format import REncoding
from ogld.store import frames

#: The frame's decompressed payload, field by field (spec §5.2), little-endian:
#: k = 2, lengths (2, 1), then the partner deltas (−2, 4, 2) and the codes (52, −90, 120),
#: each array stored as planes: every element's byte 0, then every element's byte 1, and so on.
K = bytes.fromhex("02000000")
LENGTHS = bytes.fromhex("0200000001000000")
DELTAS = bytes.fromhex("fe0402ff0000ff0000ff0000")
CODES = bytes.fromhex("34a67800ff00")
PAYLOAD = K + LENGTHS + DELTAS + CODES

#: What those bytes stand for: (partners, codes) per row, the frame's first row being 5.
ROWS = [
    (np.array([3, 7], dtype=np.int32), np.array([52, -90], dtype=np.int16)),
    (np.array([8], dtype=np.int32), np.array([120], dtype=np.int16)),
]
FIRST, NUMBER, ROWS_DUE = 5, 3, 2
ZSTD = numcodecs.Zstd(level=3, checksum=True)


def compressed(payload: bytes = PAYLOAD) -> bytes:
    """`payload` as a frame's bytes: one zstd frame, level 3, with its checksum."""
    return ZSTD.encode(payload)


def decoded(raw: bytes, rows: int = ROWS_DUE) -> frames.Frame:
    return frames.decode_frame(raw, chrom="22", number=NUMBER, first=FIRST, rows=rows)


def encoded(first: int, rows: list[tuple[np.ndarray, np.ndarray]], number: int = NUMBER) -> bytes:
    return frames.encode_frame(first, rows, chrom="22", number=number)


def test_the_spec_s_payload_bytes_are_what_the_decoder_reads() -> None:
    frame = decoded(compressed())
    assert frame.rows == ROWS_DUE
    assert [frame.row(FIRST + i)[0].tolist() for i in range(ROWS_DUE)] == [[3, 7], [8]]
    assert [frame.row(FIRST + i)[1].tolist() for i in range(ROWS_DUE)] == [[52, -90], [120]]


def test_the_encoder_writes_the_spec_s_payload_bytes() -> None:
    assert ZSTD.decode(encoded(FIRST, ROWS)) == PAYLOAD


def test_a_frame_is_one_zstd_frame_with_its_checksum_and_its_content_size() -> None:
    raw = encoded(FIRST, ROWS)
    assert raw[:4] == b"\x28\xb5\x2f\xfd", "a zstd frame's magic number"
    descriptor = raw[4]
    assert descriptor & 0x04, "the content checksum is set (spec §5.2)"
    assert descriptor >> 6 or descriptor & 0x20, "the content size is in the header (spec §5.2)"
    assert ZSTD.decode(raw) == PAYLOAD


def test_a_frame_of_empty_rows_is_still_a_frame() -> None:
    empty = [(np.empty(0, np.int32), np.empty(0, REncoding.dtype))] * 3
    frame = frames.decode_frame(encoded(0, empty), chrom="22", number=0, first=0, rows=3)
    assert frame.entries == 0
    assert [len(frame.row(i)[0]) for i in range(3)] == [0, 0, 0]


@pytest.mark.parametrize("how", ["a flipped byte", "a truncated payload", "a wrong k"])
def test_a_damaged_frame_is_refused_naming_its_chromosome_and_frame(how: str) -> None:
    if how == "a flipped byte":
        raw = bytearray(encoded(FIRST, ROWS))
        raw[len(raw) // 2] ^= 0xFF
        raw = bytes(raw)
    elif how == "a truncated payload":
        raw = compressed(PAYLOAD[:-2])
    else:
        raw = compressed(bytes.fromhex("03000000") + LENGTHS + DELTAS + CODES)
    with pytest.raises(ValueError, match="frame 3") as refused:
        decoded(raw)
    assert "chromosome 22" in str(refused.value), refused.value
    assert f"frame {NUMBER}" in str(refused.value), refused.value


def test_a_row_outside_the_frame_is_refused() -> None:
    frame = decoded(compressed())
    for row in (FIRST - 1, FIRST + ROWS_DUE):
        with pytest.raises(ValueError, match="frame 3") as refused:
            frame.row(row)
        assert f"frame {NUMBER}" in str(refused.value), refused.value


# ------------------------------------------------------------------ the group


def frame_rows(rows: int, pairs: Sequence[tuple[int, int]]) -> list[tuple[np.ndarray, np.ndarray]]:
    """`rows` rows of the Partner Rows' shape holding `pairs` both ways, each at its own code: a
    fixture of the component's own shape: the group's tests hold a store's rows, not a
    codec's."""
    held: list[dict[int, int]] = [{} for _ in range(rows)]
    for code, (lo, hi) in enumerate(pairs, start=1):
        held[lo][hi] = code
        held[hi][lo] = code
    return [
        (
            np.array(sorted(row), dtype=np.int32),
            np.array([row[m] for m in sorted(row)], dtype=np.int16),
        )
        for row in held
    ]


#: Two frames' worth of rows: 20 rows, so frame 0 holds rows 0–15 and frame 1 rows 16–19.
PAIRS = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (4, 7), (16, 19)]
TWO_FRAMES = frame_rows(20, PAIRS)


def written(path: Path, rows: int) -> frames.FramesGroup:
    """A chromosome's Partner Rows group with `TWO_FRAMES` appended, as the emitter does."""
    group = frames.FramesGroup(open_store(path), "22", rows)
    for first in range(0, rows, frames.ROWS_PER_FRAME):
        group.add(TWO_FRAMES[first : first + frames.ROWS_PER_FRAME])
    return group


def open_store(path: Path) -> zarr.Group:
    return zarr.open_group(str(path), mode="w-", zarr_format=3)


def chunk_file(array: zarr.Array) -> Path:
    """Where the array's one chunk is written, from its own metadata (spec §5.1): never assumed."""
    key = array.metadata.encode_chunk_key((0,))
    return Path(array.store_path.store.root) / array.store_path.path / key


def test_the_two_arrays_hold_one_chunk_each_and_the_chunk_file_is_the_frames(
    tmp_path: Path,
) -> None:
    whole, second = encoded(0, TWO_FRAMES[:16], 0), encoded(16, TWO_FRAMES[16:], 1)
    raw = whole + second
    entries = written(tmp_path / "store", rows=20).close()
    assert entries == 2 * len(PAIRS), "each pair's code is held in both variants' rows"
    held = zarr.open_group(str(tmp_path / "store"), mode="r")["partners/22"]
    assert held.attrs["rows"] == 20
    assert held.attrs["rows_per_frame"] == 16
    assert held.attrs["frames"] == 2
    assert held.attrs["entries"] == entries
    assert held.attrs["frames_bytes"] == len(raw)
    assert held.attrs["zstd_level"] == 3
    assert held.attrs["frames_sha256"] == hashlib.sha256(raw).hexdigest()
    frames_array, index = held["frames"], held["frames_idx"]
    assert frames_array.dtype == np.uint8
    assert frames_array.shape == (len(raw),)
    assert frames_array.chunks == (len(raw),), "one chunk, the whole array"
    assert [codec["name"] for codec in frames_array.metadata.to_dict()["codecs"]] == ["bytes"], (
        "bytes only"
    )
    assert frames_array.fill_value == 0
    assert index.dtype == np.uint64
    assert index.shape == (3,), "F + 1 offsets"
    assert index.chunks == (3,), "one chunk, the whole array"
    np.testing.assert_array_equal(
        np.asarray(index[:]), [0, len(whole), len(raw)], "the frames' byte offsets"
    )
    held_index = hashlib.sha256(chunk_file(index).read_bytes()).hexdigest()
    assert held.attrs["frames_idx_sha256"] == held_index
    assert chunk_file(frames_array).read_bytes() == raw, "the chunk file is the raw frames"
    np.testing.assert_array_equal(np.asarray(frames_array[:]), np.frombuffer(raw, np.uint8))


def test_a_frame_read_back_through_zarr_is_the_frame_written(tmp_path: Path) -> None:
    raw = encoded(0, TWO_FRAMES[:16], 0) + encoded(16, TWO_FRAMES[16:], 1)
    written(tmp_path / "store", rows=20).close()
    held = zarr.open_group(str(tmp_path / "store"), mode="r")["partners/22"]
    offsets = np.asarray(held["frames_idx"][:])
    array = held["frames"]
    for number in range(len(offsets) - 1):
        span = slice(int(offsets[number]), int(offsets[number + 1]))
        assert np.asarray(array[span]).tobytes() == raw[span]
        first = number * 16
        rows = min(16, 20 - first)
        frame = frames.decode_frame(raw[span], chrom="22", number=number, first=first, rows=rows)
        counted = sum(len(TWO_FRAMES[row][0]) for row in range(first, first + rows))
        assert frame.rows == rows
        assert frame.entries == counted, "the entries the fixture holds, counted here"
        for row in range(first, first + rows):
            partners, codes = frame.row(row)
            np.testing.assert_array_equal(partners, TWO_FRAMES[row][0], err_msg=str(row))
            np.testing.assert_array_equal(codes, TWO_FRAMES[row][1], err_msg=str(row))


def test_zarr_lists_the_partner_rows_as_a_node_with_no_warning(tmp_path: Path) -> None:
    written(tmp_path / "store", rows=20).close()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        members = zarr.open_group(str(tmp_path / "store"), mode="r").members(max_depth=None)
    warned = [str(w.message) for w in caught]
    assert not warned, f"plain files beside a hierarchy warn; a node does not: {warned}"
    listed = [str(path) for path, _ in members]
    assert "partners/22/frames" in listed, listed
    assert "partners/22/frames_idx" in listed, listed


def test_a_group_that_is_not_closed_has_no_arrays_yet(tmp_path: Path) -> None:
    group = frames.FramesGroup(open_store(tmp_path / "store"), "22", rows=16)
    group.add(TWO_FRAMES[:16])
    assert not (tmp_path / "store" / "partners" / "22" / "frames").exists(), (
        "the array's shape is unknown until the chromosome ends, so its node is not made"
    )
    group.close()


def test_a_frame_of_more_rows_than_a_frame_holds_is_refused(tmp_path: Path) -> None:
    group = frames.FramesGroup(open_store(tmp_path / "store"), "22", rows=40)
    with pytest.raises(ValueError, match=str(frames.ROWS_PER_FRAME)):
        group.add(TWO_FRAMES[:17])


#: Forty rows, frames of 16, 16 and 8.
FORTY = frame_rows(40, PAIRS)


def test_a_failed_compression_poisons_the_group_and_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A frame that never compressed is not a frame the chromosome holds: the group is poisoned, so
    # a second close cannot publish a component short of one, and no file is left behind.
    real = frames.encode_frame
    made = {"frames": 0}

    def failing(
        first: int,
        rows: list[tuple[np.ndarray, np.ndarray]],
        *,
        chrom: str,
        number: int,
    ) -> bytes:
        made["frames"] += 1
        if made["frames"] == 1:
            raise RuntimeError("the compressor died")
        return real(first, rows, chrom=chrom, number=number)

    monkeypatch.setattr(frames, "encode_frame", failing)
    group = frames.FramesGroup(open_store(tmp_path / "store"), "22", rows=32, workers=4)
    # Two frames, the first of them never compressed: before this was refused, the second close
    # published rows=32 with frames=1, a component short of a frame that looked whole.
    group.add(FORTY[:16])
    group.add(FORTY[16:32])
    with pytest.raises(RuntimeError, match="the compressor died"):
        group.close()
    with pytest.raises(ValueError, match="compressor died") as refused:
        group.close()
    assert "chromosome 22" in str(refused.value), refused.value
    partners = tmp_path / "store" / "partners"
    assert sorted(path.name for path in partners.iterdir()) == ["zarr.json"], "no group, no temp"


def test_a_failed_frame_in_line_poisons_the_group_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(
        first: int,
        rows: list[tuple[np.ndarray, np.ndarray]],
        *,
        chrom: str,
        number: int,
    ) -> bytes:
        raise RuntimeError("bad partners")

    monkeypatch.setattr(frames, "encode_frame", failing)
    group = frames.FramesGroup(open_store(tmp_path / "store"), "22", rows=16)
    with pytest.raises(RuntimeError, match="bad partners"):
        group.add(FORTY[:16])
    with pytest.raises(ValueError, match="bad partners"):
        group.add(FORTY[:16])
    with pytest.raises(ValueError, match="bad partners"):
        group.close()


def test_a_group_that_fails_before_its_attributes_is_never_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The whole group is built beside the store and published with one rename: a failure after the
    # frames and before the attributes leaves no `partners/<chrom>/` at all (issue #67).
    group = frames.FramesGroup(open_store(tmp_path / "store"), "22", rows=16)
    group.add(FORTY[:16])

    def failing(group_: zarr.Group, length: int) -> zarr.Array:
        raise RuntimeError("the offsets could not be written")

    monkeypatch.setattr(frames.layout, "new_frames_index", failing)
    with pytest.raises(RuntimeError, match="offsets"):
        group.close()
    partners = tmp_path / "store" / "partners"
    assert not (partners / "22").exists(), "a half-built chromosome is never published"
    assert sorted(path.name for path in partners.iterdir()) == ["zarr.json"], "nor left behind"


def test_the_group_is_published_whole_and_a_second_one_is_refused(tmp_path: Path) -> None:
    raw = encoded(0, TWO_FRAMES[:16], 0) + encoded(16, TWO_FRAMES[16:], 1)
    written(tmp_path / "store", rows=20).close()
    partners = tmp_path / "store" / "partners"
    published = zarr.open_group(str(partners / "22"), mode="r")
    assert published.attrs["frames"] == 2
    assert published.attrs["frames_bytes"] == len(raw)
    assert published["frames"].shape == (len(raw),)
    assert (partners / "22" / "zarr.json").is_file(), "the group's own metadata moved with it"
    assert not [path for path in partners.iterdir() if path.name.startswith(".")], (
        "a dot-prefixed leftover would make zarr's members() warn"
    )
    again = zarr.open_group(str(tmp_path / "store"), mode="r+")
    with pytest.raises(FileExistsError, match="partners/22 exists"):
        frames.FramesGroup(again, "22", rows=20)


@pytest.mark.parametrize(
    "rows",
    [
        [(np.array([], np.int32), np.array([7], dtype=REncoding.dtype))],
        [(np.array([3], np.int32), np.array([], dtype=REncoding.dtype))],
        [
            (np.array([3], np.int32), np.array([7], dtype=REncoding.dtype)),
            (np.array([4, 5], np.int32), np.array([8], dtype=REncoding.dtype)),
        ],
    ],
)
def test_a_row_of_partners_and_codes_of_other_counts_is_refused_naming_it(
    rows: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    # A code is its pair's, so a row of one count and a row of another cannot be a frame: refused
    # per row, before anything is concatenated or narrowed, and the row is named (§5.2).
    with pytest.raises(ValueError, match="codes are its pairs") as refused:
        encoded(0, rows)
    assert "chromosome 22" in str(refused.value), refused.value
    assert f"frame {NUMBER}" in str(refused.value), refused.value
    assert "row 0" in str(refused.value) or "row 1" in str(refused.value), refused.value


def test_a_target_that_appeared_before_the_rename_is_refused_not_replaced(tmp_path: Path) -> None:
    # One writer per store, and a chromosome's group is written once: a target that is there when
    # the group is published, empty or not, is refused rather than replaced (issue #67).
    group = frames.FramesGroup(open_store(tmp_path / "store"), "22", rows=16)
    group.add(FORTY[:16])
    target = tmp_path / "store" / "partners" / "22"
    target.mkdir()
    with pytest.raises(FileExistsError, match="partners/22 exists"):
        group.close()
    assert target.is_dir(), "the target is still the caller's"
    assert not list(target.iterdir()), "and nothing replaced it"
    assert not [path for path in target.parent.iterdir() if path.name.startswith(".")], (
        "nor left a temporary group behind"
    )


def test_a_stale_temporary_group_of_the_same_chromosome_is_removed(tmp_path: Path) -> None:
    # A writer that crashed leaves `partners/.<chrom>.tmp-*`, which zarr's `members()` warns about:
    # it is removed when that chromosome is written again, and no other chromosome's is touched.
    partners = tmp_path / "store" / "partners"
    open_store(tmp_path / "store")
    partners.mkdir()
    crashed = partners / ".22.tmp-crashed"
    crashed.mkdir()
    (crashed / "frames.bin").write_bytes(b"a frame a crashed writer left")
    elsewhere = partners / ".21.tmp-live"
    elsewhere.mkdir()
    group = frames.FramesGroup(zarr.open_group(str(tmp_path / "store"), mode="r+"), "22", rows=16)
    assert not crashed.exists(), "a crash's leftover is removed before the chromosome is written"
    assert elsewhere.exists(), "and another chromosome's temp group is left alone"
    group.add(FORTY[:16])
    group.close()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        members = zarr.open_group(str(tmp_path / "store"), mode="r").members(max_depth=None)
    assert [str(path) for path, _ in members if "partners/22" in str(path)], "the group is there"
    assert "Object at .21.tmp-live" in str([str(w.message) for w in caught]), (
        "the other chromosome's leftover still warns, which is why a store holding one is refused"
    )


def test_a_refusal_from_a_pending_frame_names_its_own_number(tmp_path: Path) -> None:
    # With eight compressors three frames are pending at once; a row of the second must be named
    # frame 1, not the frame the appended offsets have reached (issue #67's review, round 2).
    rows = frame_rows(48, [*PAIRS, (21, 24), (21, 30)])
    partners, codes = rows[21]
    rows[21] = (partners, codes[:-1])
    group = frames.FramesGroup(open_store(tmp_path / "store"), "22", rows=48, workers=8)
    for first in range(0, 48, 16):
        group.add(rows[first : first + 16])
    with pytest.raises(ValueError, match="frame 1 of chromosome 22, row 21") as refused:
        group.close()
    assert "2 partner(s) and 1 code(s)" in str(refused.value), refused.value


@pytest.mark.parametrize(
    ("partners", "what"),
    [([9, 7], "partners that do not ascend"), ([7, 7], "partners that do not ascend")],
)
def test_partners_that_do_not_ascend_are_refused_naming_the_row(
    partners: list[int], what: str
) -> None:
    # Strictly ascending, never the row: refused with the chromosome, the frame and the row, so a
    # malformed emitter names where it went wrong rather than raising a bare TypeError.
    rows = frame_rows(48, [*PAIRS, (5, 7), (5, 9)])
    rows[5] = (np.array(partners, np.int32), np.array([1, 2], dtype=REncoding.dtype))
    with pytest.raises(ValueError, match=what) as refused:
        encoded(0, rows[:16], 0)
    assert "frame 0 of chromosome 22" in str(refused.value), refused.value
    assert f"row 5 holds {what}" in str(refused.value), refused.value


def test_a_group_closed_with_rows_missing_is_refused(tmp_path: Path) -> None:
    group = frames.FramesGroup(open_store(tmp_path / "store"), "22", rows=20)
    group.add(TWO_FRAMES[:16])
    with pytest.raises(ValueError, match="16 of 20 rows"):
        group.close()
