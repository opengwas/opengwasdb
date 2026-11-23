"""What the Partner Rows tests read the Test Store with: the pairs it holds, from ld-store §4's
address rule written out here, and its Partner Rows decoded frame by frame."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
import zarr
from store_copies import TEST_STORE

from opengwasdb.ld.store import frames

Pairs = tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int16]]


def store() -> zarr.Group:
    return zarr.open_group(str(TEST_STORE), mode="r")


def chromosomes(group: zarr.Group) -> dict[str, tuple[int, int]]:
    listed: list[dict[str, Any]] = group["variants"].attrs["chromosomes"]
    return {entry["name"]: (entry["row_start"], entry["row_stop"]) for entry in listed}


def near_pairs(group: zarr.Group, chrom: str) -> Pairs:
    """Every Near Band pair of one chromosome, from ld-store §4's address rule written out here:
    r(l, m) is at `r[indptr[l] + (m − l − 1)]`, so a code's address gives its pair."""
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


def decoded_frames(held: zarr.Group, chrom: str) -> Iterator[tuple[int, frames.Frame]]:
    """Each frame of one chromosome's Partner Rows group, decoded, with its first row."""
    rows, per_frame = int(held.attrs["rows"]), int(held.attrs["rows_per_frame"])
    offsets = np.asarray(held["frames_idx"][:], dtype=np.int64)
    raw = np.asarray(held["frames"][:], dtype=np.uint8).tobytes()
    for number in range(len(offsets) - 1):
        first = number * per_frame
        yield (
            first,
            frames.decode_frame(
                raw[offsets[number] : offsets[number + 1]],
                chrom=chrom,
                number=number,
                first=first,
                rows=min(per_frame, rows - first),
            ),
        )
