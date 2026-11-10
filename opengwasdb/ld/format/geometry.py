"""Where each row of an LD Store lies, and where its pairs are stored (spec §2, §4, §5).

Pure functions of positions, chromosome labels and the two distances: no Zarr and no I/O.
Which rows a chromosome has, which later variants fall within `near_bp` or `far_bp` of a
row, at what address of a Near Band's `r` a pair is held, and how a ragged set of ranges
becomes flat coordinates all live here, once. Every layer above calls these, and none
recomputes a stop or an address of its own: a second copy is a second place for an
off-by-one, and an off-by-one here is a wrong r that looks like a right one.

Positions are one chromosome's, ascending, in row order (spec §2). A row's *stop* is the
index just past its last variant within a distance, so it counts the row itself and any
variant at the same position, and a variant exactly `distance` bp away is inside.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

# ------------------------------------------------------------------ chromosomes (spec §2)


def chromosome_ranges(chrom: npt.ArrayLike) -> dict[str, tuple[int, int]]:
    """Each chromosome's [row_start, row_stop), in order of appearance; a chromosome that
    reappears after another breaks the Variant Set's order (spec §2)."""
    labels = np.asarray(chrom, dtype=np.str_)
    ranges: dict[str, tuple[int, int]] = {}
    if len(labels) == 0:
        return ranges
    edges = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate([[0], edges])
    stops = np.concatenate([edges, [len(labels)]])
    for start, stop in zip(starts, stops, strict=True):
        name = str(labels[start])
        if name in ranges:
            raise ValueError(f"chromosome {name} is not contiguous in the Variant Set (spec §2)")
        ranges[name] = (int(start), int(stop))
    return ranges


def listed_ranges(ranges: dict[str, tuple[int, int]]) -> list[dict[str, Any]]:
    """The ranges as `variants.attrs["chromosomes"]` records them (spec §2)."""
    return [
        {"name": name, "row_start": start, "row_stop": stop}
        for name, (start, stop) in ranges.items()
    ]


def ranges_listed(listed: Sequence[dict[str, Any]]) -> dict[str, tuple[int, int]]:
    """The ranges a `chromosomes` attribute records, in its order; `listed_ranges` inverted."""
    return {str(c["name"]): (int(c["row_start"]), int(c["row_stop"])) for c in listed}


def chromosome_of(ranges: dict[str, tuple[int, int]], row: int) -> str:
    """The chromosome holding global `row`; a row of no chromosome raises."""
    for name, (start, stop) in ranges.items():
        if start <= row < stop:
            return name
    raise ValueError(f"row {row} is in no chromosome's range")


def chromosome_index(stops: npt.ArrayLike, rows: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """For each global row, the index of its chromosome among those whose row_stop are
    `stops` (ascending). A row at or past the last stop gets an index one past the end."""
    return np.searchsorted(np.asarray(stops), np.asarray(rows), side="right").astype(np.int64)


# ------------------------------------------------------------------ stops (spec §4, §5)


def _stop(position: npt.ArrayLike, distance: int) -> npt.NDArray[np.int64]:
    at = np.asarray(position, dtype=np.int64)
    return np.searchsorted(at, at + distance, side="right").astype(np.int64)


def near_stop(position: npt.ArrayLike, near_bp: int) -> npt.NDArray[np.int64]:
    """For each row, the index just past the last variant with p_m − p_l ≤ near_bp (spec §4).
    Exactly near_bp away is within; one more is not."""
    return _stop(position, near_bp)


def far_stop(position: npt.ArrayLike, far_bp: int) -> npt.NDArray[np.int64]:
    """For each row, the index just past the last variant with p_m − p_l ≤ far_bp (spec §5)."""
    return _stop(position, far_bp)


def near_start(
    position: npt.ArrayLike, near_bp: int, rows: npt.ArrayLike | None = None
) -> npt.NDArray[np.int64]:
    """For each of `rows` (all of them when None), the index of the first variant with
    p_l − p_m ≤ near_bp: where the Near Band partners before it begin (spec §4). Exactly
    near_bp away is within."""
    at = np.asarray(position, dtype=np.int64)
    of = at if rows is None else at[np.asarray(rows, dtype=np.int64)]
    return np.searchsorted(at, of - near_bp, side="left").astype(np.int64)


def near_row_lengths(position: npt.ArrayLike, near_bp: int) -> npt.NDArray[np.int64]:
    """The length of each Near Band row of one chromosome: len_l = |{m > l : p_m − p_l ≤ near_bp}|
    (spec §4). `position` is the chromosome's positions in row order, ascending."""
    return (near_stop(position, near_bp) - np.arange(len(np.asarray(position))) - 1).astype(
        np.int64
    )


def position_rows(
    position: npt.ArrayLike, start: npt.ArrayLike, stop: npt.ArrayLike
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """The rows [lo, hi) with start ≤ position ≤ stop, as indices into `position` (spec §2).
    Both ends are inclusive."""
    at = np.asarray(position, dtype=np.int64)
    lo = np.searchsorted(at, np.asarray(start), side="left")
    hi = np.searchsorted(at, np.asarray(stop), side="right")
    return lo.astype(np.int64), hi.astype(np.int64)


# ------------------------------------------------------------------ addresses (spec §4)


def near_address(
    indptr: npt.ArrayLike, lo: npt.ArrayLike, hi: npt.ArrayLike
) -> npt.NDArray[np.int64]:
    """The index in `near/<chrom>/r` of the pair of local rows lo < hi: row lo's first code
    is at indptr[lo], and its codes run over lo + 1, lo + 2, … (spec §4)."""
    low = np.asarray(lo, dtype=np.int64)
    return np.asarray(indptr, dtype=np.int64)[low] + (np.asarray(hi, dtype=np.int64) - low - 1)


# ------------------------------------------------------------------ ragged ranges


def spans(first: npt.ArrayLike, length: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """first[i], first[i] + 1, … for length[i] elements each, back to back: the elements of
    CSR rows, or the columns of Near Band rows (spec §4, §5)."""
    count = np.asarray(length, dtype=np.int64)
    offset = np.repeat(np.cumsum(count) - count, count)
    start = np.repeat(np.asarray(first, dtype=np.int64), count)
    return start + (np.arange(int(count.sum()), dtype=np.int64) - offset)


def spans_at(
    first: npt.ArrayLike, length: npt.ArrayLike, at: npt.ArrayLike
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """The inverse of `spans`: for elements `at` of `spans(first, length)`, the span i each
    lies in and its value, first[i] + its place in the span, without building the spans. For
    a block of Near Band rows that is each code's row and its later row (spec §4)."""
    count = np.asarray(length, dtype=np.int64)
    ends = np.cumsum(count)
    where = np.asarray(at, dtype=np.int64)
    span = np.searchsorted(ends, where, side="right")
    value = np.asarray(first, dtype=np.int64)[span] + (where - (ends[span] - count[span]))
    return span.astype(np.int64), value


# ------------------------------------------------------------------ bounded iteration


def windows(start: int, stop: int, step: int) -> Iterator[tuple[int, int]]:
    """[lo, hi) windows of at most `step` that tile [start, stop)."""
    for lo in range(start, stop, step):
        yield lo, min(stop, lo + step)


def blocks(*limits: tuple[npt.NDArray[np.int64], int]) -> Iterator[tuple[int, int]]:
    """Row ranges [a, b) holding at most `limit` elements of each array an `indptr` delimits,
    for each (indptr, limit) of `limits`, or one row when that row alone holds more."""
    rows, a = len(limits[0][0]) - 1, 0
    while a < rows:
        b = min(int(np.searchsorted(p, p[a] + most, side="right")) - 1 for p, most in limits)
        b = min(max(b, a + 1), rows)
        yield a, b
        a = b


def row_pairs(
    first: npt.NDArray[np.int64],
    stop: npt.NDArray[np.int64],
    per_step: int,
) -> Iterator[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]:
    """The pairs (i, j) with first[i] ≤ j < stop[i], a bounded number at a time: whole rows
    until `per_step` pairs, or one row when it alone holds more."""
    length = np.maximum(stop - first, 0)
    ends = np.cumsum(length)
    start = 0
    while start < len(length):
        before = int(ends[start] - length[start])
        end = max(int(np.searchsorted(ends, before + per_step, side="right")), start + 1)
        if ends[end - 1] > before:
            yield (
                np.repeat(np.arange(start, end), length[start:end]),
                spans(first[start:end], length[start:end]),
            )
        start = end
