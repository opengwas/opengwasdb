"""One chromosome's LD, tile by tile: the Near Band exactly, the Proxy Index at t_far.

A tile is a block of rows against every later variant within `far_bp`: one matrix product of
unit vectors (`ogld.build.adjust`). From it come the tile's Near Band rows, every code
(spec §4), and its upper-triangle Proxy Index entries, the pairs within `far_bp` whose code
reaches the far code floor (spec §3, §5), one quantisation for both, so the index's near half
carries the band's own codes (`index.near`, spec §12). Tiles are computed on `threads` worker
threads and written in row order, so the store is the same for any thread count.

The far code floor is judged on codes, not on r: a pair is kept exactly when |code| ≥ the
floor. Only pairs with |r| × 32767 ≥ floor − 1 are quantised to judge, which keeps every pair
whose code can reach the floor, since rounding moves r × 32767 by at most ½.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from ogld.format import R_SCALE, near_row_lengths, quantise

#: Rows a tile computes: a tile is TILE_ROWS × (variants within far_bp) float64.
TILE_ROWS: Final = 256


@dataclass(frozen=True)
class Band:
    """One chromosome's geometry: for each local row l, the first row past near_bp, past
    far_bp and past `change_bp` (exclusive stops), from its positions (spec §4)."""

    near_stop: npt.NDArray[np.int64]
    far_stop: npt.NDArray[np.int64]
    change_stop: npt.NDArray[np.int64]

    @classmethod
    def of(cls, position: npt.NDArray[np.int64], near_bp: int, far_bp: int, change_bp: int) -> Band:
        rows = np.arange(len(position), dtype=np.int64)
        near = rows + 1 + near_row_lengths(position, near_bp)
        far = np.searchsorted(position, position + far_bp, side="right").astype(np.int64)
        change = np.searchsorted(position, position + change_bp, side="right").astype(np.int64)
        return cls(near, far, change)


@dataclass(frozen=True)
class Inputs:
    """What every tile of one chromosome reads.

    `units` are the PC-adjusted unit vectors, `unadjusted` the k = 0 ones (None at k = 0, where
    the two are one), and `outside` marks the variants outside the Long-Range LD Regions, whose
    pairs within `change_bp` the PC diagnostic's change in r is measured on.
    """

    units: npt.NDArray[np.float64]
    unadjusted: npt.NDArray[np.float64] | None
    outside: npt.NDArray[np.bool_]
    band: Band
    floor: int


@dataclass(frozen=True)
class Tile:
    """One tile's rows `first` … : their Near Band row lengths and codes, their Proxy Index
    entries (local rows lo < hi, codes), and the change in r within `change_bp`."""

    first: int
    lengths: npt.NDArray[np.int64]
    codes: npt.NDArray[np.int16]
    lo: npt.NDArray[np.int64]
    hi: npt.NDArray[np.int64]
    entry_codes: npt.NDArray[np.int16]
    change: npt.NDArray[np.float64]


def cells(
    starts: npt.NDArray[np.int64], stops: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """For rows t = 0, 1, …, the columns starts[t] … stops[t] − 1, row after row, as (t, column)."""
    lengths = stops - starts
    t = np.repeat(np.arange(len(starts), dtype=np.int64), lengths)
    offset = np.arange(int(lengths.sum()), dtype=np.int64) - np.repeat(
        np.cumsum(lengths) - lengths, lengths
    )
    return t, np.repeat(starts, lengths) + offset


def ragged(
    block: npt.NDArray[np.float64],
    first: int,
    starts: npt.NDArray[np.int64],
    stops: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    """Row t's columns starts[t] … stops[t] − 1 of `block`, whose column 0 is row `first`,
    row after row."""
    t, column = cells(starts, stops)
    gathered: npt.NDArray[np.float64] = block[t, column - first]
    return gathered


def tile(inputs: Inputs, first: int, last: int) -> Tile:
    """Rows `first` … `last − 1` of one chromosome (local rows)."""
    band = inputs.band
    stop = int(band.far_stop[last - 1])
    r = inputs.units[first:last] @ inputs.units[first:stop].T
    rows = np.arange(first, last, dtype=np.int64)
    codes = quantise(ragged(r, first, rows + 1, band.near_stop[first:last]))
    lo, hi, entry_codes = _entries(r, first, band.far_stop, inputs.floor)
    return Tile(
        first,
        band.near_stop[first:last] - rows - 1,
        codes,
        lo,
        hi,
        entry_codes,
        _change(inputs, r, first, last),
    )


def _entries(
    r: npt.NDArray[np.float64], first: int, far_stop: npt.NDArray[np.int64], floor: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int16]]:
    """The tile's pairs lo < hi within far_bp whose code reaches `floor` (spec §5)."""
    t, c = np.nonzero(np.abs(r) * R_SCALE >= floor - 1)
    lo, hi = first + t.astype(np.int64), first + c.astype(np.int64)
    inside = (hi > lo) & (hi < far_stop[lo])
    t, c, lo, hi = t[inside], c[inside], lo[inside], hi[inside]
    codes = quantise(r[t, c])
    reach = np.abs(codes.astype(np.int64)) >= floor
    return lo[reach], hi[reach], codes[reach]


def _change(
    inputs: Inputs, r: npt.NDArray[np.float64], first: int, last: int
) -> npt.NDArray[np.float64]:
    """Adjusted minus unadjusted r for the tile's pairs within `change_bp`, both variants outside
    the Long-Range LD Regions; at k = 0 the two are one and every change is 0."""
    starts = np.arange(first + 1, last + 1, dtype=np.int64)
    stops = inputs.band.change_stop[first:last]
    t, partner = cells(starts, stops)
    clear = inputs.outside[first + t] & inputs.outside[partner]
    after = r[t[clear], partner[clear] - first]
    if inputs.unadjusted is None:
        return np.zeros_like(after)
    u0 = inputs.unadjusted
    r0 = u0[first:last] @ u0[first : int(stops.max())].T
    before: npt.NDArray[np.float64] = r0[t[clear], partner[clear] - first]
    return after - before


def tiles(inputs: Inputs, threads: int) -> Iterator[Tile]:
    """Every tile of the chromosome, in row order, computed on `threads` threads."""
    n = len(inputs.units)
    spans = [(first, min(first + TILE_ROWS, n)) for first in range(0, n, TILE_ROWS)]
    if threads <= 1:
        yield from (tile(inputs, first, last) for first, last in spans)
        return
    with ThreadPoolExecutor(max_workers=threads) as pool:
        yield from _in_order(pool, lambda span: tile(inputs, *span), spans, 2 * threads)


def _in_order(
    pool: ThreadPoolExecutor,
    work: Callable[[tuple[int, int]], Tile],
    spans: list[tuple[int, int]],
    ahead: int,
) -> Iterator[Tile]:
    """`work` of each span, in order, with at most `ahead` in flight: bounded memory."""
    pending: deque[Future[Tile]] = deque()
    for span in spans:
        pending.append(pool.submit(work, span))
        if len(pending) >= ahead:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


@dataclass(frozen=True)
class Entries:
    """A chromosome's Proxy Index entries, each pair once: local rows lo < hi and its code."""

    lo: npt.NDArray[np.int64]
    hi: npt.NDArray[np.int64]
    codes: npt.NDArray[np.int16]

    @classmethod
    def of(
        cls,
        lo: list[npt.NDArray[np.int64]],
        hi: list[npt.NDArray[np.int64]],
        codes: list[npt.NDArray[np.int16]],
    ) -> Entries:
        return cls(np.concatenate(lo), np.concatenate(hi), np.concatenate(codes))

    def both_directions(
        self, rows: int
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int16]]:
        """The Proxy Index in CSR form over `rows` local rows: each pair in both its rows, with
        one code, partners ascending within a row (spec §5)."""
        row = np.concatenate([self.lo, self.hi])
        partner = np.concatenate([self.hi, self.lo])
        codes = np.concatenate([self.codes, self.codes])
        order = np.lexsort((partner, row))
        indptr = np.concatenate([[0], np.cumsum(np.bincount(row, minlength=rows))])
        return indptr.astype(np.int64), partner[order], codes[order]
