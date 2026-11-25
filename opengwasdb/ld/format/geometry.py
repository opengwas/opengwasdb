"""Where each row of an LD Store lies, and where its pairs are stored (ld-store §2, §4, §5).

Pure functions of positions, chromosome labels and the two distances: no Zarr and no I/O.
Which rows a chromosome has, which later variants fall within `near_bp` or `far_bp` of a
row, at what address of a Near Band's `r` a pair is held, and how a ragged set of ranges
becomes flat coordinates all live here, once. Every layer above calls these, and none
recomputes a stop or an address of its own: a second copy is a second place for an
off-by-one, and an off-by-one here is a wrong r that looks like a right one.

Positions are one chromosome's, ascending, in row order (ld-store §2). A row's *stop* is the
index just past its last variant within a distance, so it counts the row itself and any
variant at the same position, and a variant exactly `distance` bp away is inside.
"""

from __future__ import annotations
from collections.abc import Iterable, Iterator
import numpy as np
import numpy.typing as npt

# The read side of this module is ogquery's (opengwas/ogquery#2): re-exported by name.
from ogquery.ld.format.geometry import (
    blocks as blocks,
    chromosome_index as chromosome_index,
    global_rows as global_rows,
    indptr_of as indptr_of,
    local_rows as local_rows,
    near_address as near_address,
    near_row_lengths as near_row_lengths,
    owners as owners,
    position_rows as position_rows,
    row_pairs as row_pairs,
    spans as spans,
    stop_within as stop_within,
    windows as windows,
    within as within,
)


def chromosome_ranges(chrom: npt.ArrayLike) -> dict[str, tuple[int, int]]:
    """Each chromosome's [row_start, row_stop), in order of appearance; a chromosome that
    reappears after another breaks the Variant Set's order (ld-store §2). `chrom` may be fixed-width
    (`<U`) or numpy's variable-width `StringDType`, which is what zarr returns."""
    labels = np.asarray(chrom)
    if labels.dtype.kind not in "UT":  # bytes and objects become str; U and StringDType are
        labels = labels.astype(np.str_)  # already text
    ranges: dict[str, tuple[int, int]] = {}
    if len(labels) == 0:
        return ranges
    edges = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate([[0], edges])
    stops = np.concatenate([edges, [len(labels)]])
    for start, stop in zip(starts, stops, strict=True):
        name = str(labels[start])
        if name in ranges:
            raise ValueError(
                f"chromosome {name} is not contiguous in the Variant Set (ld-store §2)"
            )
        ranges[name] = (int(start), int(stop))
    return ranges


def chromosome_ranges_from_counts(
    chromosomes: Iterable[tuple[str, int]],
) -> dict[str, tuple[int, int]]:
    """Ranges of distinct, nonempty closed chromosomes in their supplied order.

    Joining immutable chromosomes already knows their counts. Expanding those to
    genome-wide labels merely to recover their edges would waste memory (ADR 0101).
    """
    ranges: dict[str, tuple[int, int]] = {}
    total = 0
    for name, count in chromosomes:
        if name in ranges:
            raise ValueError(
                f"chromosome {name} is not contiguous in the Variant Set (ld-store §2)"
            )
        if type(count) is not int or count <= 0:
            raise ValueError(f"chromosome {name} count must be a positive integer, not {count!r}")
        ranges[name] = (total, total + count)
        total += count
    return ranges


def spans_at(
    first: npt.ArrayLike, length: npt.ArrayLike, at: npt.ArrayLike
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """The inverse of `spans`: for elements `at` of `spans(first, length)`, the span i each
    lies in and its value, first[i] + its place in the span, without building the spans. For
    a block of Near Band rows that is each code's row and its later row (ld-store §4)."""
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
    `indptr` delimits, stored at each address `at` of a Near Band's `r` (ld-store §4). `indptr`
    may be a slice of rows, its first entry the address of its first row's first code."""
    edges = np.asarray(indptr, dtype=np.int64)
    lo, hi = spans_at(
        np.arange(1, len(edges), dtype=np.int64), np.diff(edges), np.asarray(at) - edges[0]
    )
    return lo, hi
