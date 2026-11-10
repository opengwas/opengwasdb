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

from collections.abc import Iterator

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


def chromosome_index(stops: npt.ArrayLike, rows: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """For each global row, the index of its chromosome among those whose row_stop are
    `stops` (ascending). A row outside [0, V), V being the last stop, is no row of the
    Variant Set and raises: −1 is what an Absent ALID reads as, never the first chromosome."""
    edge = np.asarray(stops, dtype=np.int64)
    at = np.asarray(rows, dtype=np.int64)
    size = int(edge[-1]) if len(edge) else 0
    bad = (at < 0) | (at >= size)
    if bad.any():
        raise ValueError(f"row {int(at[bad].ravel()[0])} is not in the Variant Set of {size} rows")
    return np.searchsorted(edge, at, side="right").astype(np.int64)


def local_rows(rows: npt.ArrayLike, start: int) -> npt.NDArray[np.int64]:
    """Global rows as rows of the chromosome that starts at global row `start` (spec §2)."""
    return np.asarray(rows, dtype=np.int64) - start


def global_rows(local: npt.ArrayLike, start: int) -> npt.NDArray[np.int64]:
    """Rows of the chromosome that starts at global row `start` as global rows (spec §2)."""
    return np.asarray(local, dtype=np.int64) + start


# ------------------------------------------------------------------ stops (spec §4, §5)


def stop_within(position: npt.ArrayLike, distance: int) -> npt.NDArray[np.int64]:
    """For each row, the index just past the last variant with p_m − p_l ≤ distance: pass
    `near_bp` for the Near Band's stop (spec §4) and `far_bp` for the Proxy Index's (spec §5).
    Exactly `distance` away is within; one more is not. It counts the row itself, and
    any variant at the same position."""
    at = np.asarray(position, dtype=np.int64)
    return np.searchsorted(at, at + distance, side="right").astype(np.int64)


def start_within(
    position: npt.ArrayLike, distance: int, rows: npt.ArrayLike | None = None
) -> npt.NDArray[np.int64]:
    """For each of `rows` (all of them when None), the index of the first variant with
    p_l − p_m ≤ distance: where the partners before it begin (spec §4). Exactly `distance`
    away is within."""
    at = np.asarray(position, dtype=np.int64)
    of = at if rows is None else at[np.asarray(rows, dtype=np.int64)]
    return np.searchsorted(at, of - distance, side="left").astype(np.int64)


def within(gap: npt.ArrayLike, distance: int) -> npt.NDArray[np.bool_]:
    """Whether positions `gap` bp apart, either way round, are within `distance`: the edge
    the stops use, so exactly `distance` is within and one more is not."""
    return np.asarray(np.abs(np.asarray(gap, dtype=np.int64)) <= distance)


def near_row_lengths(position: npt.ArrayLike, near_bp: int) -> npt.NDArray[np.int64]:
    """The length of each Near Band row of one chromosome: len_l = |{m > l : p_m − p_l ≤ near_bp}|
    (spec §4). `position` is the chromosome's positions in row order, ascending."""
    return (stop_within(position, near_bp) - np.arange(len(np.asarray(position))) - 1).astype(
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


def owners(length: npt.ArrayLike, labels: npt.ArrayLike | None = None) -> npt.NDArray[np.int64]:
    """The owner of each element of the ragged ranges `length` describes, back to back: i
    repeated length[i] times, or labels[i] when `labels` are given. The owners of `spans`."""
    count = np.asarray(length, dtype=np.int64)
    who = np.arange(len(count), dtype=np.int64) if labels is None else np.asarray(labels)
    return np.repeat(who, count).astype(np.int64)


def indptr_of(length: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """The CSR `indptr` of rows of `length` elements each: 0, then the running total."""
    return np.concatenate([[0], np.cumsum(np.asarray(length, dtype=np.int64))]).astype(np.int64)


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


def near_pair_of(
    indptr: npt.ArrayLike, at: npt.ArrayLike
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """The inverse of `near_address`: the pair (lo, hi) of rows, as indices into the rows
    `indptr` delimits, stored at each address `at` of a Near Band's `r` (spec §4). `indptr`
    may be a slice of rows, its first entry the address of its first row's first code."""
    edges = np.asarray(indptr, dtype=np.int64)
    lo, hi = spans_at(
        np.arange(1, len(edges), dtype=np.int64), np.diff(edges), np.asarray(at) - edges[0]
    )
    return lo, hi


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
