"""One chromosome's r, tile by tile, from a window of prepared genotypes (ADR 0013).

A tile is a block of rows against every later variant within `far_bp`, computed by
`ogld.genotypes.r`. The build hands the product to the store, `StoreWriter.cut`,
which quantises, thresholds and counts (ogld#17). The build keeps only its own diagnostic, the
change in r from PC Adjustment within `change_bp`. Tiles are computed on `threads` worker
threads and written in row order, so the store is the same for any thread count.

Prepared genotypes are held for a window of rows, never the whole chromosome (`Window`). The
window runs from the first row of the oldest tile in flight to the far stop of the newest, plus
the genotype block being read (`UNIT_ROWS` rows).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from ogld.format.geometry import near_row_lengths, owners, spans, stop_within, windows
from ogld.genotypes import Block, r
from ogld.store import Cut

#: Rows a tile computes: a tile is TILE_ROWS × (variants within far_bp) float64.
TILE_ROWS: Final = 256
#: Rows of genotypes read and adjusted at a time into the genotype window: 8,192 rows of
#: 665 samples are 44 MB, where `ogld.build.inputs.CHUNK_RECORDS` rows would be 350 MB.
UNIT_ROWS: Final = 8_192

_Units = npt.NDArray[np.float64]
#: One block: (first local row, prepared adjusted genotypes, intercept-only or None).
GenotypeBlock = tuple[int, Block, Block | None]


@dataclass(frozen=True)
class Band:
    """One chromosome's geometry for the build: for each local row l, the first row past far_bp
    and past `change_bp` (exclusive stops), and how many later rows lie within `change_bp`, all
    from `ogld.format.geometry` (spec §5; §7's change in r)."""

    change_length: npt.NDArray[np.int64]
    far_stop: npt.NDArray[np.int64]
    change_stop: npt.NDArray[np.int64]

    @classmethod
    def of(cls, position: npt.NDArray[np.int64], far_bp: int, change_bp: int) -> Band:
        return cls(
            near_row_lengths(position, change_bp),
            stop_within(position, far_bp),
            stop_within(position, change_bp),
        )

    def reach(self) -> int:
        """The most rows one tile spans: from its first row to its last row's far stop."""
        first = np.arange(0, len(self.far_stop), TILE_ROWS)
        last = np.minimum(first + TILE_ROWS, len(self.far_stop)) - 1
        return int((self.far_stop[last] - first).max()) if len(first) else 0


class Window:
    """A window of prepared genotypes, adjusted and (at k > 0) intercept-only.

    Blocks arrive in row order and their calls, projections, norms and signs are buffered
    together. When full, rows still needed move to fresh arrays. An in-flight tile keeps
    its views of the old arrays, so nothing it reads is ever overwritten (ogld#17).
    """

    def __init__(self, blocks: Iterator[GenotypeBlock], capacity: int) -> None:
        self._blocks = blocks
        self._capacity = capacity
        self._units: Block | None = None
        self._unadjusted: Block | None = None
        self._base = 0
        self._end = 0

    def reach(self, stop: int, keep: int) -> None:
        """Read blocks until rows below `stop` are held, keeping the rows from `keep` on. A block
        must start where the last ended: a gap or an overlap would leave rows of the array that are
        not the chromosome's."""
        while self._end < stop:
            first, units, unadjusted = next(self._blocks)
            if first != self._end:
                raise ValueError(
                    f"a block of genotypes starts at row {first}, not {self._end}: the window "
                    "holds a chromosome's rows in order, with none missing (a builder fault)"
                )
            if first + len(units) - self._base > self._capacity:
                self._move(keep)
            self._units = self._put(self._units, units)
            if unadjusted is not None:
                self._unadjusted = self._put(self._unadjusted, unadjusted)
            self._end = first + len(units)

    def _put(self, held: Block | None, block: Block) -> Block:
        """Write `block` at the window's end, making the buffer at the first block."""
        if held is None:
            held = _allocate(block, self._capacity)
        _copy(held, self._end - self._base, block)
        return held

    def _move(self, keep: int) -> None:
        """Move rows `keep` … end to the start of new arrays."""
        live = slice(keep - self._base, self._end - self._base)
        if self._units is not None:
            self._units = _fresh(self._units, live, self._capacity)
        if self._unadjusted is not None:
            self._unadjusted = _fresh(self._unadjusted, live, self._capacity)
        self._base = keep

    def rows(self, first: int, stop: int, unadjusted: bool = False) -> Block:
        """Rows `first` … `stop − 1`, a view; the unadjusted ones when asked. Rows the window does
        not hold are refused: the array's other rows are some other variants' vectors."""
        held = self._unadjusted if unadjusted else self._units
        if held is None or not self._base <= first <= stop <= self._end:
            kind = "unadjusted " if unadjusted else ""
            raise ValueError(
                f"rows {first}–{stop - 1} are not in the window of {kind}genotypes, which holds "
                f"rows {self._base}–{self._end - 1}{'' if held is not None else ' (none at k = 0)'}"
            )
        return held[first - self._base : stop - self._base]

    @property
    def unadjusted(self) -> bool:
        return self._unadjusted is not None


def _allocate(block: Block, capacity: int) -> Block:
    return Block(
        np.empty((capacity, block.calls.shape[1])),
        np.empty((capacity, block.projection.shape[1])),
        np.empty(capacity),
        np.empty(capacity),
    )


def _copy(out: Block, at: int, block: Block) -> None:
    end = at + len(block)
    out.calls[at:end] = block.calls
    out.projection[at:end] = block.projection
    out.norm[at:end] = block.norm
    out.sign[at:end] = block.sign


def _fresh(old: Block, live: slice, capacity: int) -> Block:
    new = _allocate(old, capacity)
    _copy(new, 0, old[live])
    return new


@dataclass(frozen=True)
class Inputs:
    """What every tile of one chromosome reads besides its genotypes: its geometry, which
    variants lie outside the Long-Range LD Regions (whose pairs within `change_bp` the PC
    diagnostic's change in r is measured on), its first global row, and the store's `cut`."""

    band: Band
    outside: npt.NDArray[np.bool_]
    start: int
    cut: Callable[[int, _Units], Cut]


@dataclass(frozen=True)
class Work:
    """One tile's rows `first` … `last − 1` (local) and the genotypes it reads: its own rows
    and those up to its far stop, and, at k > 0, the unadjusted ones up to `change_bp`."""

    first: int
    last: int
    rows: Block
    span: Block
    unadjusted: tuple[Block, Block] | None


def _work(window: Window, band: Band, first: int, last: int) -> Work:
    stop = int(band.far_stop[last - 1])
    unadjusted = None
    if window.unadjusted:
        reach = int(band.change_stop[first:last].max())
        unadjusted = (window.rows(first, last, True), window.rows(first, reach, True))
    return Work(first, last, window.rows(first, last), window.rows(first, stop), unadjusted)


def _tile(inputs: Inputs, work: Work) -> tuple[Cut, npt.NDArray[np.float64]]:
    """A tile's cut and its change in r."""
    values = r(work.rows, work.span)
    return inputs.cut(inputs.start + work.first, values), _change(inputs, work, values)


def change_pairs(
    band: Band, outside: npt.NDArray[np.bool_], first: int, last: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
    """The pairs within `change_bp` of rows `first` … `last − 1`, row after row, as (t, partner,
    clear): t the row's offset from `first`, partner its row, and clear when both variants lie
    outside the Long-Range LD Regions. The one definition of the PC diagnostic's 100 kb pairs
    (spec §7): the tiles measure the change on them, and the build refuses a Variant Set with
    none before it writes (`diagnostic.require_change_pairs`)."""
    starts = np.arange(first + 1, last + 1, dtype=np.int64)
    length = band.change_length[first:last]
    t, partner = owners(length), spans(starts, length)
    return t, partner, outside[first + t] & outside[partner]


def _change(inputs: Inputs, work: Work, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Adjusted minus unadjusted r for the tile's pairs within `change_bp`, both variants outside
    the Long-Range LD Regions; at k = 0 the two are one and every change is 0."""
    t, partner, clear = change_pairs(inputs.band, inputs.outside, work.first, work.last)
    t, column = t[clear], partner[clear] - work.first
    after = values[t, column]
    if work.unadjusted is None:
        return np.zeros_like(after)
    rows, span = work.unadjusted
    return after - r(rows, span)[t, column]


Tiled = tuple[Cut, npt.NDArray[np.float64]]


def tiles(inputs: Inputs, units: Iterator[GenotypeBlock], threads: int) -> Iterator[Tiled]:
    """Every tile of the chromosome, in row order, computed on `threads` threads, each as its
    store cut and its change in r. At most 2 × `threads` tiles are in flight, and prepared
    genotypes are held for those tiles' window only."""
    n = len(inputs.band.far_stop)
    ahead = 2 * threads if threads > 1 else 1
    window = Window(units, min(n, inputs.band.reach() + ahead * TILE_ROWS + UNIT_ROWS))
    pending: deque[tuple[int, Future[Tiled]]] = deque()
    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        for first, last in windows(0, n, TILE_ROWS):
            keep = pending[0][0] if pending else first
            window.reach(int(inputs.band.far_stop[last - 1]), keep)
            work = _work(window, inputs.band, first, last)
            pending.append((first, pool.submit(_tile, inputs, work)))
            if len(pending) >= ahead:
                yield pending.popleft()[1].result()
        while pending:
            yield pending.popleft()[1].result()
