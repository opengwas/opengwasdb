"""What goes in an LD Store, from the r its producer supplies (spec §3–§7, §12; ogld#17).

`StoreWriter` is the seam a producer calls. This module, behind it, owns what goes in a store
(owner, 10 November 2026; `docs/architecture/ownership.md`, "Provenance and what goes in a
store"). It handles:

- quantising r at the store's encoding, through `ogld.format.REncoding` (spec §3);
- the Near Band's row lengths and `indptr`, from the positions (spec §4);
- the threshold at the far code floor, judged on codes, for every pair within far_bp (§3, §5);
- the Far Pairs: the pairs at that floor beyond near_bp, once each, in the earlier row (spec §5;
  ADR 0007), from the pairs a cut already keeps, with no second pass over r;
- the Partner Rows: every pair within far_bp at its distance's floor, in both variants' rows, 16
  rows to a zstd frame (spec §5; partner-rows §5, §6.1; ADR 0011);
- their derivation for a store built before them, from that store's own Near Band and Far Pairs,
  through the same emitter a build feeds (partner-rows §6.2; ogld#69);
- the counts, and the rest of the provenance a store derives from itself (spec §7).

Validation at `finish` runs every rule of spec §12 through `validate_group`. A producer supplies r
for blocks of rows in order, and at `finish` it supplies what only the source knows
(`ogld.format.Sourced`).

**Memory is bounded by the block and the far_bp window, never the chromosome.** These are
measured on chr6 of lane #5's whole-EUR store (602,466 rows, 6,826,805,033 Near Band codes).
What is held:

- **The Near Band carry:** less than one shard of `r`, 67,108,864 codes, so 128 MiB.
- **The Partner Rows carry:** the pairs whose later row is not yet written. At most these are
  the pairs spanning one row within far_bp, plus one bucket of `CARRY_ROWS` rows. At the MHC
  that is 33,869,608 pairs at 10 B, 323 MiB.
- **The Partner Rows' frames:** at most `2 × workers` compressed frames in flight, and one frame
  of rows, 16 rows' partners, before it is compressed.
- **The Far Pairs output:** less than one shard each of `partner` and `r`, 6 MiB. Nothing is
  carried: a row's Far Pairs are all in its own block.
- **Per-row arrays:** each `indptr`, n_c + 1 int64.
- **A cut in flight,** between a producer's worker and `write`: at most 31 MiB for a 256-row
  block.
"""

from __future__ import annotations

import fcntl
import json
import multiprocessing
import os
import shutil
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
import zarr

from ogld.format import (
    FORMAT_VERSION,
    T_FAR_Q,
    Codes,
    Counts,
    InputError,
    Provenance,
    REncoding,
    Sourced,
    claims_partner_rows,
    t_far,
)
from ogld.format.geometry import (
    blocks,
    chromosome_index,
    indptr_of,
    near_pair_of,
    near_row_lengths,
    owners,
    spans_at,
    stop_within,
    windows,
    within,
)
from ogld.store import frames, layout
from ogld.store.location import ChunkCache, open_group
from ogld.store.validation import (
    PARTNER_RULES,
    ValidationFailed,
    require_alid_mmap,
    validate_group,
)

#: Proxy Index entries `derive_far_pairs` reads at once, or one row's when it holds more: 8 Mi
#: entries are 48 MiB of `partner` and `r`, and about 200 MiB as they are judged. That function is
#: the only code left that reads the component (#72): no store this package writes holds one.
DERIVE_ENTRIES: Final = 1 << 23
#: Variant Set rows `derive_alid_index` reads at once: 1,048,576 ALIDs, whose keys are 32 MiB.
ALID_ROWS: Final = 1 << 20
#: The bytes one carried pair holds: its earlier row, its later row and its code — the row's two
#: 4-byte integers and its code's 2 — the destination being its bucket (partner-rows §6.1).
PAIR_BYTES: Final = 10
#: The rows of one bucket of the Partner Rows' carried pairs. Pairs are carried by the bucket
#: their later row falls in, so a bucket is freed once its rows are written. Only the pairs of
#: one partly written bucket are held beyond those spanning the row being written.
CARRY_ROWS: Final = 256

_Rows = npt.NDArray[np.int32]


@dataclass(frozen=True)
class Cut:
    """A block of rows cut from its producer's r: what `StoreWriter.write` writes.

    `first` is the block's first global row and `rows` how many it holds, all on one chromosome.
    `near` holds their Near Band codes, row after row (spec §4). `lo`, `hi` and `codes` are their
    Far Pairs, as local rows lo < hi: each partner hi more than near_bp and at most far_bp from
    row lo whose code `REncoding.stored_mask` keeps, once, with its one code (spec §3, §5),
    ascending by (lo, hi). `far` marks those more than near_bp apart — every one of them, and it
    is kept so a reader of a Cut can name the half it reads. `far_candidates`
    counts the block's pairs more than near_bp and at most far_bp apart, all judged, and
    `far_pairs` those of them kept (spec §7). `by` is the `Cutter` that cut it, so a cut is
    written only by its own store.

    A producer neither builds nor reads one: it hands what `StoreWriter.cut` returned to
    `StoreWriter.write`.
    """

    first: int
    rows: int
    near: Codes
    lo: _Rows
    hi: _Rows
    codes: Codes
    far: npt.NDArray[np.bool_]
    far_candidates: int
    by: Cutter

    @property
    def far_pairs(self) -> int:
        return int(self.far.sum())


class Cutter:
    """What every block of one store is cut by, fixed when the store is created: the Variant
    Set's positions and chromosome ranges, near_bp and far_bp, the encoding, and t_far with its
    far code floor and t_partner with its Partner Floor (spec §3–§5; partner-rows §3). Nothing in
    it changes after `create`, so blocks are cut on any thread, in any order."""

    def __init__(
        self,
        position: npt.NDArray[np.int64],
        ranges: dict[str, tuple[int, int]],
        distances: tuple[int, int],
        encoding: REncoding,
        *,
        n: int,
        k: int,
        t_partner: float,
    ) -> None:
        """Blocks are thresholded as they arrive, so t_far for `n` samples and `k` covariates, and
        its floor, are fixed here, before the first; an n and k with no t_far raise here. So is
        `t_partner` and its Partner Floor, and a t_partner above t_far raises here, naming both.
        The positions are copied and made read-only, so no caller can change what a cut reads."""
        self.t_far = t_far(n, k)
        self.floor = encoding.far_code_floor(self.t_far)
        self.t_partner = t_partner
        self.partner_floor = encoding.partner_code_floor(t_partner, self.t_far)
        self.encoding = encoding
        self.n, self.k = n, k
        self.near_bp, self.far_bp = distances
        self._least = encoding.least_r(self.floor)
        self._position = np.array(position, dtype=np.int64)
        self._position.flags.writeable = False
        self._ranges = ranges
        self._names = list(ranges)
        self._stops = [stop for _, stop in ranges.values()]

    def chromosome(self, row: int) -> tuple[str, int, int]:
        """The chromosome of global `row`, its first row and its stop; a row outside the
        Variant Set raises."""
        name = self._names[int(chromosome_index(self._stops, [row])[0])]
        return (name, *self._ranges[name])

    def positions(self, name: str) -> npt.NDArray[np.int64]:
        start, stop = self._ranges[name]
        return self._position[start:stop]

    def cut(self, first: int, r: npt.NDArray[np.floating[Any]]) -> Cut:
        """`StoreWriter.cut`'s work, which says what `r` holds.

        For each row l of the block it reads r(l, m) only for l < m < far stop(l). Every m within
        near_bp is quantised, in order, for the Near Band. Beyond near_bp a pair is quantised,
        to judge it on its code, only when its |r| can reach the floor. A value that is not a
        number is such a pair, so quantising refuses it rather than let it pass as below the
        floor. So a pair the producer did not measure is refused, never stored and never read as
        a Bound (spec §8).
        """
        name, start, stop = self.chromosome(first)
        if r.ndim != 2 or len(r) == 0:
            raise ValueError(f"r for a block of rows is a matrix with a row each, not {r.shape}")
        rows, width = r.shape
        if first + rows > stop:
            raise ValueError(
                f"rows {first}–{first + rows - 1} run past the end of chromosome {name} at "
                f"row {stop - 1}: a block of rows is on one chromosome"
            )
        local = first - start
        near_stop, far_stop = self._stops_of(name, local, rows, width)
        near = self._near(first, r, near_stop)
        lo, hi, codes = self._pairs(first, r, near_stop, far_stop)
        return Cut(
            first,
            rows,
            near,
            (local + lo).astype(np.int32),
            (local + hi).astype(np.int32),
            codes,
            _beyond_near(self.positions(name), local + lo, local + hi, self.near_bp),
            int((far_stop - near_stop).sum()),
            self,
        )

    def _stops_of(
        self, name: str, local: int, rows: int, width: int
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """The near and far stops of the block's rows, counted from its first row; refuse an r
        that is not exactly as wide as its last row's far stop (spec §4, §5)."""
        position = self.positions(name)
        reach = position[local : local + width + 1]
        far_stop = stop_within(reach, self.far_bp)[:rows]
        if far_stop[-1] != width:
            last = local + rows - 1
            want = int(stop_within(position[last:], self.far_bp)[0]) + last - local
            raise ValueError(
                f"r for rows {local}–{last} of chromosome {name} has {width} columns; it needs "
                f"{want} columns from its first row, to the last variant within far_bp = "
                f"{self.far_bp:,} of its last row (spec §5)"
            )
        return stop_within(reach, self.near_bp)[:rows], far_stop

    def _near(
        self, first: int, r: npt.NDArray[np.floating[Any]], near_stop: npt.NDArray[np.int64]
    ) -> Codes:
        """The block's Near Band codes, row after row. Each row's run is one slice: five times
        faster than a fancy index (lane #5)."""
        runs = [r[t, t + 1 : stop] for t, stop in enumerate(near_stop)]
        values = np.concatenate(runs)
        lengths = near_stop - np.arange(1, len(near_stop) + 1)

        def pair(at: int) -> tuple[int, int]:
            t, m = spans_at(np.arange(1, len(near_stop) + 1), lengths, [at])
            return first + int(t[0]), first + int(m[0])

        return self._quantised(values, pair)

    def _pairs(
        self,
        first: int,
        r: npt.NDArray[np.floating[Any]],
        near_stop: npt.NDArray[np.int64],
        far_stop: npt.NDArray[np.int64],
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], Codes]:
        """The block's Far Pairs beyond near_bp: (t, c), near stop(t) ≤ c < far stop(t), that
        `REncoding.stored_mask` keeps at their distance, as offsets from its first row, ascending
        by (t, c), with their codes. The Near Band's own pairs are `_near`'s, so no pair inside
        near_bp is quantised, given a distance or masked here. Only those whose |r| is not below
        the encoding's `least_r(floor)` are quantised, which keeps every pair whose code can reach
        the floor; NaN is never below it. The test is two comparisons, not |r|, which would copy
        the block's r: 133 MB a worker on chr6's widest tile."""
        below = (r > -self._least) & (r < self._least)
        t, c = np.nonzero(~below)
        del below
        # Only the far half: [cut.far] is all that reads lo, hi and codes, and the Near Band
        # holds every pair inside near_bp.
        inside = (c > t) & (c >= near_stop[t]) & (c < far_stop[t])
        t, c = t[inside], c[inside]
        codes = self._quantised(r[t, c], lambda at: (first + int(t[at]), first + int(c[at])))
        name, start, _ = self.chromosome(first)
        position = self.positions(name)
        keep = self.encoding.stored_mask(
            position[first - start + t] - position[first - start + c],
            codes,
            near_bp=self.near_bp,
            far_bp=self.far_bp,
            far_code_floor=self.floor,
        )
        return t[keep], c[keep], codes[keep]

    def _quantised(
        self, values: npt.NDArray[np.floating[Any]], pair: Callable[[int], tuple[int, int]]
    ) -> Codes:
        """`values` as the encoding's codes. The encoding refuses one that is not a correlation;
        the refusal then names its pair, found by halving."""
        try:
            return self.encoding.quantise(values)
        except ValueError as refused:
            lo, hi = 0, len(values)
            while hi - lo > 1:
                mid = (lo + hi) // 2
                try:
                    self.encoding.quantise(values[lo:mid])
                    lo = mid
                except ValueError:
                    hi = mid
            row, partner = pair(lo)
            raise ValueError(
                f"r({row}, {partner}) = {values[lo]} is not a correlation: a pair within far_bp "
                "that was not measured, or a fault in computing it. Nothing is stored for it, "
                "and it is never read as a Bound (spec §3, §8)"
            ) from refused


class ShardCarry:
    """Values written a whole shard at a time, less than one carried to the next call.

    Zarr writes a shard without reading it back only when a write covers it whole (zarr 3.4.2,
    `_is_complete_shard_write`), so `put(at, values)` is called with one whole shard at a time,
    and what is left is `rest` until its array ends. The one loop both streams share.
    """

    def __init__(
        self, shard: int, dtype: npt.DTypeLike, put: Callable[[int, npt.NDArray[Any]], None]
    ) -> None:
        self._buffer = np.empty(max(1, shard), dtype=dtype)
        self._held = 0
        self.written = 0
        self._put = put

    def add(self, values: npt.NDArray[Any]) -> None:
        shard, at = len(self._buffer), 0
        while at < len(values):
            take = min(shard - self._held, len(values) - at)
            self._buffer[self._held : self._held + take] = values[at : at + take]
            self._held += take
            at += take
            if self._held == shard:
                self._put(self.written, self._buffer)
                self.written += shard
                self._held = 0

    @property
    def total(self) -> int:
        """Every value added: those written and those carried."""
        return self.written + self._held

    def rest(self) -> npt.NDArray[Any]:
        """The values carried, less than a shard."""
        return self._buffer[: self._held]


class Appender:
    """A 1-D array whose length is known only at its end, written a whole shard at a time.

    The array is created at one shard when the first fills and grows by `resize`. An array under
    one shard is made at its end, at its clipped shape (spec §6). Either way its shard files and
    metadata are those of the array written whole (checked on zarr 3.4.2).
    """

    def __init__(
        self, make: Callable[[int], zarr.Array[Any]], dtype: npt.DTypeLike, shard: int
    ) -> None:
        self._make = make
        self._array: zarr.Array[Any] | None = None
        self._carry = ShardCarry(shard, dtype, self._put)

    def add(self, values: npt.NDArray[Any]) -> None:
        self._carry.add(values)

    def _put(self, at: int, values: npt.NDArray[Any]) -> None:
        if self._array is None:
            self._array = self._make(len(values))
        else:
            self._array.resize((at + len(values),))
        self._array[at : at + len(values)] = values

    def close(self) -> None:
        """Write what is carried, making or growing the array to its length."""
        total, rest = self._carry.total, self._carry.rest()
        if self._array is None:
            self._array = self._make(total)
        elif len(rest):
            self._array.resize((total,))
        if len(rest):
            self._array[self._carry.written : total] = rest


class NearBandStream:
    """One chromosome's Near Band, written as its blocks arrive (spec §4, §6).

    This is lane #4's measured design (comment on ogld#17). `r`'s length, the sum of the row
    lengths, is known from the positions before any code arrives. So `indptr` is written whole,
    and `r` is created at its full length, when the chromosome's first block arrives. Codes
    are written in whole shards of `r` (`ShardCarry`), and less than one shard is carried to the
    next block. The last shard is written when the last row arrives.
    """

    def __init__(self, group: zarr.Group, lengths: npt.NDArray[np.int64], encoding: REncoding):
        indptr = indptr_of(lengths)
        layout.put_numbers(group, "indptr", indptr, layout.INT64_FILL)
        self._total = int(indptr[-1])
        self._array = layout.new_r_array(group, "r", self._total, layout.NEAR_R, encoding)
        shard = layout.clipped(self._total, *layout.NEAR_R)[1]
        self._carry = ShardCarry(min(shard, self._total), encoding.dtype, self._put)

    def _put(self, at: int, codes: Codes) -> None:
        self._array[at : at + len(codes)] = codes

    def add(self, codes: Codes) -> None:
        """Append the next rows' codes, writing each shard they complete."""
        self._carry.add(codes)

    def close(self) -> None:
        """Write the carried codes; refuse a band whose codes are not the sum of its lengths."""
        if self._carry.total != self._total:
            raise ValueError(
                f"the Near Band holds {self._carry.total} codes of {self._total}: a store "
                "assembly fault"
            )
        rest = self._carry.rest()
        if len(rest):
            self._array[self._carry.written :] = rest


_Pairs = tuple[_Rows, _Rows, Codes]


def _entries(group: zarr.Group, encoding: REncoding) -> tuple[Appender, Appender]:
    """A sparse component's `partner` and `r` (spec §5, §6), each grown a whole shard at a time."""
    side = layout.SIDE
    partner = Appender(
        lambda n: layout.new_numbers(group, "partner", n, np.int32, layout.PARTNER_FILL, side),
        np.int32,
        side[1],
    )
    codes = Appender(
        lambda n: layout.new_r_array(group, "r", n, side, encoding), encoding.dtype, side[1]
    )
    return partner, codes


def _by_later_row(pairs: list[_Pairs]) -> _Pairs:
    """Pieces whose earlier rows ascend from piece to piece, each sorted by (hi, lo), as one
    array sorted by (hi, lo): a stable sort by hi keeps lo ascending within each hi."""
    lo, hi, codes = (np.concatenate([p[i] for p in pairs]) for i in range(3))
    order = np.argsort(hi, kind="stable")
    return lo[order], hi[order], codes[order]


@dataclass(frozen=True)
class Rows:
    """The rows one block of a sparse component completes: their entry counts, and their
    partners and codes row after row, each row's partners in ascending order."""

    first: int
    counts: npt.NDArray[np.int64]
    partner: _Rows
    codes: Codes

    @property
    def last(self) -> int:
        """The row after the block's last."""
        return self.first + len(self.counts)


class RowCarry:
    """The block-at-a-time row carry the Partner Rows are written by (spec §5; partner-rows
    §6.1).

    A pair (lo, hi) goes into row lo as the block arrives, because both rows are the store's, and
    into row hi only when hi's block does. So row l's entries are the pairs whose earlier row is in
    its own block, which are ascending in partner already, and those carried to it from earlier
    blocks. A pair whose later row is still to come is carried, in buckets of `CARRY_ROWS` rows by
    that later row: at most the pairs spanning one row within far_bp and one bucket, however long
    the chromosome.

    `complete` gives each row's partners in ascending order, as partner-rows §3 requires, so a
    reader's binary search and a cumulative sum need no case for the row.
    The pairs handed to it are ascending by (lo, hi): the order a cut's Near Band codes and its
    kept pairs arrive in, once the caller has interleaved the two by row.
    """

    def __init__(self, rows: int) -> None:
        #: Each row's entries once written; −1 until then, so a row never written is never read as
        #: a row with no partners.
        self._counts = np.full(rows, -1, dtype=np.int64)
        self._next = 0
        self._carried: dict[int, list[_Pairs]] = {}
        #: The pairs carried now, and the most ever: counted as they are kept and released, never
        #: by walking the pieces, so a block costs what it carries and nothing per live piece.
        self._held = 0
        self._peak = 0

    @property
    def peak(self) -> tuple[int, int]:
        """The most pairs the carry held at once, and their bytes: what bounds a chromosome's
        memory however long it is (partner-rows §6.1)."""
        return self._peak, self._peak * PAIR_BYTES

    @property
    def next_row(self) -> int:
        """The first row not yet completed."""
        return self._next

    def complete(self, rows: int, lo: _Rows, hi: _Rows, codes: Codes) -> Rows:
        """Complete the next `rows` rows from the pairs (lo, hi) of their own blocks, and carry
        the pairs whose later row is still to come."""
        a, b = self._next, self._next + rows
        closing = hi < b
        below = [*self._take(b), (lo[closing], hi[closing], codes[closing])]
        lo_below, hi_below, codes_below = _by_later_row(below)
        self._carry(lo[~closing], hi[~closing], codes[~closing])
        owner = np.concatenate([hi_below, lo])
        order = np.argsort(owner, kind="stable")
        counts = np.bincount(owner - a, minlength=rows)
        self._counts[a:b] = counts
        self._next = b
        self._peak = max(
            self._peak, sum(len(piece[0]) for held in self._carried.values() for piece in held)
        )
        return Rows(
            a,
            counts,
            np.concatenate([lo_below, hi])[order].astype(np.int32),
            np.concatenate([codes_below, codes])[order],
        )

    def _carry(self, lo: _Rows, hi: _Rows, codes: Codes) -> None:
        """Keep pairs whose later row is still to come, sorted by (hi, lo), in the bucket of
        that row: a copy each, so a bucket's memory goes when it is written."""
        if not len(hi):
            return
        lo, hi, codes = _by_later_row([(lo, hi, codes)])
        self._held += len(hi)
        bucket = hi // CARRY_ROWS
        edges = np.flatnonzero(np.diff(bucket)) + 1
        for part in np.split(np.arange(len(hi)), edges):
            key = int(bucket[part[0]])
            span = slice(int(part[0]), int(part[-1]) + 1)
            piece = (lo[span].copy(), hi[span].copy(), codes[span].copy())
            self._carried.setdefault(key, []).append(piece)

    def _take(self, b: int) -> list[_Pairs]:
        """The carried pairs whose later row is below `b`, removed from the carry, bucket by
        bucket in order."""
        taken: list[_Pairs] = []
        for key in sorted(k for k in self._carried if k * CARRY_ROWS < b):
            pieces = self._carried.pop(key)
            if (key + 1) * CARRY_ROWS <= b:
                taken += pieces
                self._held -= sum(len(piece[1]) for piece in pieces)
                continue
            taken += self._split(key, pieces, b)
        return taken

    def _split(self, key: int, pieces: list[_Pairs], b: int) -> list[_Pairs]:
        """The part of a bucket only partly taken, releasing it; the rest stays carried."""
        taken: list[_Pairs] = []
        kept: list[_Pairs] = []
        for lo, hi, codes in pieces:
            at = int(np.searchsorted(hi, b))
            taken.append((lo[:at], hi[:at], codes[:at]))
            self._held -= at
            if at < len(hi):
                kept.append((lo[at:], hi[at:], codes[at:]))
        if kept:
            self._carried[key] = kept
        return taken

    def close(self) -> None:
        """Refuse a component whose rows were not all written, or whose carry is left over."""
        if self._carried or (self._counts < 0).any():
            raise ValueError("a sparse component has rows not written: a store assembly fault")


class PartnerRowsStream:
    """One chromosome's Partner Rows, written a frame at a time (§5; partner-rows §6.1).

    It is fed by (a block of rows, its Near Band codes, its Far Pairs) rather than by a `Cut`
    alone, so that #69's derivation can feed the same emitter from a store's own Near Band and Far
    Pairs and produce a build's bytes. The Near Band's codes become (lo, hi) pairs through the
    band's row lengths, and each is kept at the Partner Floor; the Far Pairs are already the pairs
    beyond near_bp at the far code floor, so they are kept as they come. Both are handed to
    `RowCarry`, whose rows then hold every partner on both sides, each at its pair's one code.

    The rows are assembled into frames of `frames.ROWS_PER_FRAME`, which `frames.FramesGroup`
    compresses on a bounded pool and appends in frame order, so the bytes do not depend on the
    block size or the pool's size.
    """

    def __init__(
        self,
        root: zarr.Group,
        chrom: str,
        lengths: npt.NDArray[np.int64],
        encoding: REncoding,
        *,
        partner_floor: int,
        far_floor: int,
        near_bp: int,
        far_bp: int,
        workers: int = 1,
        container: Path | None = None,
    ) -> None:
        """The Partner Rows of chromosome `chrom`, whose Near Band rows are `lengths` codes long
        and whose pairs are kept at the floors `REncoding.partner_mask` chooses: the Partner
        Floor within `near_bp`, the far code floor beyond it to `far_bp`. `container` is the
        directory the group is published into, the store's `partners/` group by default; a
        derivation gives a directory that is not yet one (#69)."""
        self._chrom = chrom
        self._encoding = encoding
        self._indptr = indptr_of(lengths)
        self._partner_floor = partner_floor
        self._far_floor = far_floor
        self._near_bp = near_bp
        self._far_bp = far_bp
        self._rows = RowCarry(len(lengths))
        self._group = frames.FramesGroup(
            root, chrom, len(lengths), workers=workers, container=container
        )
        self._pending: list[tuple[_Rows, Codes]] = []

    @property
    def chrom(self) -> str:
        """The chromosome these rows belong to."""
        return self._chrom

    @property
    def peak_carry(self) -> tuple[int, int]:
        """The most pairs the emitter's carry held at once, and their bytes (partner-rows §6.1)."""
        return self._rows.peak

    def add(self, rows: int, near: Codes, lo: _Rows, hi: _Rows, codes: Codes) -> None:
        """Write the next `rows` rows, keeping the block's Near Band codes `near` (row after row)
        at the Partner Floor and every pair of `lo`, `hi`, `codes` (its Far Pairs)."""
        # A code no store holds, which is what a lost chunk reads as, is refused rather than
        # compared against the floor: a fill value would silently drop its pair.
        self._encoding.require_codes(near)
        self._encoding.require_codes(codes)
        first = self._rows.next_row
        at = np.arange(self._indptr[first], self._indptr[first + rows], dtype=np.int64)
        if len(near) != len(at):
            raise ValueError(
                f"a block of {rows} rows of chromosome {self._chrom} carries {len(near)} Near "
                f"Band codes, not its band's {len(at)}: a store assembly fault"
            )
        # The floor is judged on the code alone, so only the codes that reach it need their pair:
        # the address rule runs on those, and never on a whole block. On EUR chr22 that is 5.0% of
        # the band's codes, and computing the other 95%'s pairs and discarding them was 37 s of
        # the chromosome's 41 s of selection (ogld#69). The near codes are all within near_bp and
        # the Far Pairs all beyond it, so each group is judged at its own side's boundary; the one
        # `REncoding.partner_mask` makes the floor choice for both.
        near_keep = self._encoding.partner_mask(
            np.full(len(near), self._near_bp, dtype=np.int64),
            near,
            near_bp=self._near_bp,
            far_bp=self._far_bp,
            partner_code_floor=self._partner_floor,
            far_code_floor=self._far_floor,
        )
        far_keep = self._encoding.partner_mask(
            np.full(len(codes), self._far_bp, dtype=np.int64),
            codes,
            near_bp=self._near_bp,
            far_bp=self._far_bp,
            partner_code_floor=self._partner_floor,
            far_code_floor=self._far_floor,
        )
        near_lo, near_hi = near_pair_of(self._indptr[first : first + rows + 1], at[near_keep])
        # The Near Band's kept pairs come first and every one of them is closer than the Far Pairs
        # of its row, so a stable sort by the earlier row gives the (lo, hi) order `RowCarry` needs.
        rows_lo = np.concatenate([near_lo + first, lo[far_keep]]).astype(np.int32)
        order = np.argsort(rows_lo, kind="stable")
        rows_hi = np.concatenate([near_hi + first, hi[far_keep]]).astype(np.int32)
        block = self._rows.complete(
            rows,
            rows_lo[order],
            rows_hi[order],
            np.concatenate([near[near_keep], codes[far_keep]])[order],
        )
        self._rows_in(block)

    def _rows_in(self, block: Rows) -> None:
        """Keep the block's rows until a frame holds its own, then append that frame."""
        edges = np.cumsum(block.counts)
        starts = edges - block.counts
        for start, stop in zip(starts, edges, strict=True):
            self._pending.append(
                (block.partner[int(start) : int(stop)], block.codes[int(start) : int(stop)])
            )
            if len(self._pending) == frames.ROWS_PER_FRAME:
                self._frame()

    def _frame(self) -> None:
        self._group.add(self._pending)
        self._pending = []

    def close(self) -> int:
        """Append the last frame and finish the group; return the chromosome's entries: every
        pair's code in both variants' rows (§5.3)."""
        self._rows.close()
        if self._pending:
            self._frame()
        return self._group.close()


class FarPairsStream:
    """One chromosome's Far Pairs, written row by row as blocks arrive (spec §5; ADR 0007).

    Row l holds every pair (l, m), m > l, more than near_bp and at most far_bp apart whose code
    reaches the far code floor, ascending in m: the pairs beyond near_bp once each, in their
    earlier row (spec §5; ADR 0007). They all come from row l's own block, so each block's rows
    are complete when it arrives, and nothing is carried. `partner` and `r` grow shard by shard
    (`Appender`). `indptr` is held, n_c + 1 int64, and written at the end.
    """

    def __init__(self, group: zarr.Group, rows: int, encoding: REncoding) -> None:
        #: Each row's pairs once written; −1 until then, so a row never written is never read as a
        #: row with no Far Pairs.
        self._counts = np.full(rows, -1, dtype=np.int64)
        self._group = group
        self._next = 0
        self._partner, self._codes = _entries(group, encoding)

    def add(self, rows: int, lo: _Rows, hi: _Rows, codes: Codes) -> None:
        """Write the next `rows` rows: their Far Pairs (lo, hi), lo among those rows, ascending
        by (lo, hi), with their codes."""
        a, b = self._next, self._next + rows
        if len(lo) and not (a <= int(lo[0]) and int(lo[-1]) < b):
            raise ValueError(
                f"Far Pairs of rows {int(lo[0])}–{int(lo[-1])} written as rows {a}–{b - 1}: "
                "a store assembly fault"
            )
        self._counts[a:b] = np.bincount(lo - a, minlength=rows)
        self._partner.add(hi)
        self._codes.add(codes)
        self._next = b

    def close(self) -> None:
        """Write `indptr` and the last pairs. A row not written would be a row the Far Pairs
        lack, read as one with none, and is refused."""
        if (self._counts < 0).any():
            raise ValueError("the Far Pairs have rows not written: a store assembly fault")
        layout.put_numbers(self._group, "indptr", indptr_of(self._counts), layout.INT64_FILL)
        self._partner.close()
        self._codes.close()


def _beyond_near(
    position: npt.NDArray[np.int64], lo: npt.ArrayLike, hi: npt.ArrayLike, near_bp: int
) -> npt.NDArray[np.bool_]:
    """Which of one chromosome's local pairs (lo[k], hi[k]) lie more than near_bp apart: of the
    pairs the store keeps at the far code floor, its Far Pairs (spec §3, §5). The one test the
    store writes them by, from a cut and from an index alike (ADR 0007); exactly near_bp apart is
    within, the edge `ogld.format.geometry.within` holds."""
    at = np.asarray(position, dtype=np.int64)
    return ~within(at[np.asarray(hi)] - at[np.asarray(lo)], near_bp)


def derive_far_pairs(
    index: zarr.Group,
    far: zarr.Group,
    position: npt.NDArray[np.int64],
    near_bp: int,
    encoding: REncoding,
) -> int:
    """Write one chromosome's Far Pairs into the empty group `far`, from its Proxy Index group
    `index`, for a store written before they came back (ADR 0007), and return how many. They are
    the index's entries beyond near_bp (`_beyond_near`) in each pair's earlier row (spec §5), read
    a block of rows at a time and written in row order through `FarPairsStream`, as a store being
    built writes them, so the arrays are a build's, byte for byte. The index is not changed. An
    index element no store holds, which is what a lost chunk reads as, is refused, naming its
    array, rather than read as a row without Far Pairs.

    This is the one place left in `ogld/` that reads the Proxy Index (#72), and it has a live
    caller: `/data/opengwasdb/work/ogld/realdata/EUR-k0.store` is a readable 0.1.0 store that
    holds an index and no `far/`, and #73 derives its Far Pairs before its Partner Rows. A store
    this package writes never needs it, and no other code path names the component's arrays
    (`tests/test_proxy_index_removed.py`).
    """
    indptr = _whole(index, "indptr")
    stream = FarPairsStream(far, len(indptr) - 1, encoding)
    written = 0
    for a, b in blocks((indptr, DERIVE_ENTRIES)):
        span = slice(int(indptr[a]), int(indptr[b]))
        partner, codes = _whole(index, "partner", span), _whole(index, "r", span)
        owner = owners(np.diff(indptr[a : b + 1]), np.arange(a, b, dtype=np.int64))
        kept = (partner > owner) & _beyond_near(position, owner, partner, near_bp)
        stream.add(b - a, owner[kept].astype(np.int32), partner[kept].astype(np.int32), codes[kept])
        written += int(kept.sum())
    stream.close()
    return written


def derive_alid_index(variants: zarr.Group, index: zarr.Group) -> int:
    """Write the ALID search index into `index` from `variants`' own ALID column (spec §2, §6;
    ADR 0010), and return how many ALIDs are too wide for its slot, counted from that column.

    The keys, their padding and their order are `layout.alid_index`'s: a store writes what the
    rule gives, whether it is being built or derived from the table it already has, and whether
    its own column is read here or handed over by its producer. An ALID wider than the slot is
    left out rather than truncated into it, so a lookup can never answer another variant's name
    (opengwasdb issue #127); such an ALID stays reachable by the exact scan at its position.

    The column is read a block of rows at a time, so nothing holds the table's strings whole:
    the keys take 32 bytes a variant, read at the end to sort them. The count is the table's own,
    never `n_variants` less the keys' length: a count inferred from the index would agree with any
    index, including one with an entry too many (spec §7; ogld#42's review).
    """
    alid = variants["alid"]
    assert isinstance(alid, zarr.Array)
    rows = int(alid.shape[0])
    keys: list[npt.NDArray[np.uint8]] = []
    at: list[npt.NDArray[np.int64]] = []
    wide = 0
    for lo, hi in windows(0, rows, ALID_ROWS):
        alids = np.asarray(alid[lo:hi])
        wide += int((~layout.alid_fits(alids)).sum())
        block, first = layout.alid_keys(alids, first=lo)
        keys.append(block)
        at.append(first)
    ordered, entries = layout.ordered_index(
        np.concatenate(keys) if keys else np.empty((0, layout.ALID_INDEX_WIDTH), np.uint8),
        np.concatenate(at) if at else np.empty(0, np.int64),
    )
    layout.put_alid_index(index, ordered, entries)
    return wide


def _whole(group: zarr.Group, name: str, span: slice = slice(None)) -> npt.NDArray[Any]:
    """Elements `span` of the array `name`, refused where one is a value no store holds there;
    an `indptr` or a `partner` as int64."""
    array = group[name]
    assert isinstance(array, zarr.Array)
    values = layout.refused_if_lost(array, np.asarray(array[span]))
    return values if name == "r" else values.astype(np.int64)


@dataclass
class Tally:
    """The counts provenance records (spec §7), added up block by block."""

    near_pairs: int = 0
    far_candidates: int = 0
    far_pairs: int = 0
    partner_entries: int = 0

    def add(self, cut: Cut) -> None:
        """Count a cut: its Near Band codes, and its pairs judged and kept beyond near_bp. The
        Partner Rows' entries are their frames', counted when a chromosome is closed (§5.3)."""
        self.near_pairs += len(cut.near)
        self.far_candidates += cut.far_candidates
        self.far_pairs += cut.far_pairs

    def counts(self, alids_left_out: int = 0) -> Counts:
        return {
            "near_pairs": self.near_pairs,
            "far_candidates": self.far_candidates,
            "far_pairs": self.far_pairs,
            "partner_entries": self.partner_entries,
            "alids_left_out": alids_left_out,
        }


class Chromosome:
    """One chromosome being written: its Near Band, its Far Pairs and its Partner Rows
    (spec §4, §5; partner-rows §5, §6.1)."""

    def __init__(
        self, root: zarr.Group, cutter: Cutter, name: str, *, compress_workers: int = 1
    ) -> None:
        self.name = name
        position = cutter.positions(name)
        lengths = near_row_lengths(position, cutter.near_bp)
        self.near = NearBandStream(root.require_group(f"near/{name}"), lengths, cutter.encoding)
        self.far = FarPairsStream(root.require_group(f"far/{name}"), len(position), cutter.encoding)
        self.partners = PartnerRowsStream(
            root,
            name,
            lengths,
            cutter.encoding,
            partner_floor=cutter.partner_floor,
            far_floor=cutter.floor,
            near_bp=cutter.near_bp,
            far_bp=cutter.far_bp,
            workers=compress_workers,
        )

    def add(self, cut: Cut) -> None:
        self.near.add(cut.near)
        self.far.add(cut.rows, cut.lo[cut.far], cut.hi[cut.far], cut.codes[cut.far])
        self.partners.add(cut.rows, cut.near, cut.lo[cut.far], cut.hi[cut.far], cut.codes[cut.far])

    def close(self) -> int:
        """Finish all three, and return the chromosome's Partner Rows entries (§5.3)."""
        self.near.close()
        self.far.close()
        return self.partners.close()


_PRODUCER_KEYS: Final = frozenset(Sourced.__required_keys__)


def _only_what_the_producer_knows(sourced: Sourced) -> dict[str, Any]:
    """`sourced` as a mutable dict, refused unless it is exactly the producer's keys: a key the
    store derives is refused, because the store would record its own instead, and one no reader
    knows is refused, because it would be stored unread and unchecked (spec §7)."""
    given: dict[str, Any] = dict(sourced)
    derived_keys = set(Provenance.__required_keys__) - _PRODUCER_KEYS
    ours = sorted(set(given) & derived_keys)
    if ours:
        raise ValueError(
            f"{', '.join(ours)}: the store derives these itself; a producer gives only "
            f"{sorted(_PRODUCER_KEYS)} (spec §7)"
        )
    unknown = sorted(set(given) - _PRODUCER_KEYS)
    if unknown:
        raise ValueError(
            f"{', '.join(unknown)}: not a key of `Sourced`; spec §7 lets a store hold other keys, "
            "but this writer records only the producer's and its own"
        )
    missing = sorted(_PRODUCER_KEYS - set(given))
    if missing:
        raise ValueError(f"provenance lacks {', '.join(missing)}, which only its producer knows")
    return given


def _with_zstd(software: object) -> dict[str, str]:
    """The producer's `software` versions with `zstd` added: a frame's bytes depend on the zstd
    that made them, so a store records it, and refuses a producer's other version (spec §7)."""
    told = dict(cast(dict[str, str], software))
    mine = frames.zstd_version()
    if told.setdefault("zstd", mine) != mine:
        raise ValueError(
            f"provenance records zstd {told['zstd']!r}, but the Partner Rows' frames were "
            f"compressed with {mine!r}: a frame's bytes depend on it (spec §7)"
        )
    return told


def provenance(
    sourced: Sourced, cutter: Cutter, n_variants: int, tally: Tally, *, alids_left_out: int = 0
) -> Provenance:
    """The store's provenance, in spec §7's order. It is the producer's `sourced` plus what the
    store derives from itself: `format_version`, `encoding`, `n_variants`, `near_bp`, `far_bp`,
    `t_far`, `t_far_q`, `far_code_floor`, `t_partner`, `partner_code_floor` and `counts`, the
    count of ALIDs the search index left out and of Partner Rows entries among them (ADR 0010;
    partner-rows §5.3). `software` is the producer's list of versions, which the store adds `zstd`
    to, the library that compressed its Partner Rows' frames (§7).

    Spec §7 lets a store hold other keys, which a reader ignores. This writer records none: a key
    `Sourced` does not name is refused, whether it is one the store derives, which a producer may
    not give, or one no reader knows, which would otherwise be stored unread and unchecked. A
    `sourced` that lacks a key of the producer's, or records an n or k other than the threshold's,
    is refused too."""
    given = _only_what_the_producer_knows(sourced)
    said = (sourced["panel"]["n"], sourced["adjustment"]["k"])
    if said != (cutter.n, cutter.k):
        raise ValueError(
            f"provenance records n = {said[0]} and k = {said[1]}, but the store was "
            f"thresholded at t_far for n = {cutter.n} and k = {cutter.k} (spec §3)"
        )
    given["software"] = _with_zstd(given["software"])
    derived: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "encoding": cutter.encoding.record(),
        "n_variants": n_variants,
        "near_bp": cutter.near_bp,
        "far_bp": cutter.far_bp,
        "t_far": cutter.t_far,
        "t_far_q": T_FAR_Q,
        "far_code_floor": cutter.floor,
        "t_partner": cutter.t_partner,
        "partner_code_floor": cutter.partner_floor,
        "counts": tally.counts(alids_left_out),
    }
    order = Provenance.__annotations__
    return cast(Provenance, {key: derived.get(key, given.get(key)) for key in order})


# ---------------------------------------------------------------- deriving Partner Rows (ogld#69)


@dataclass(frozen=True)
class _Schedule:
    """One chromosome's derivation, and what every chromosome of its store shares
    (partner-rows §6.2, §6.3): the job a process worker is handed.

    `start` and `stop` are the chromosome's rows in the Variant Set, so a worker needs nothing
    from the parent but this. A worker opens the store itself rather than inheriting a group, a
    chunk cache or a decoder's pool: none is fork-safe (partner-rows §9; ogld#70).
    """

    store: Path
    chrom: str
    start: int
    stop: int
    near_bp: int
    far_bp: int
    floor: int
    far_floor: int
    encoding: REncoding
    threads: int

    @property
    def rows(self) -> int:
        """The chromosome's rows in the Variant Set."""
        return self.stop - self.start


@dataclass(frozen=True)
class _Plan:
    """What one `derive-partners` run reads from its store before it writes anything."""

    store: Path
    encoding: REncoding
    near_bp: int
    far_bp: int
    floor: int
    far_floor: int
    recorded: dict[str, Any]
    ranges: dict[str, tuple[int, int]]
    named: tuple[str, ...]
    missing: tuple[str, ...]
    claimed: bool

    def present(self, chrom: str) -> bool:
        """Whether `chrom`'s Partner Rows are published: a group at `partners/<chrom>/`."""
        return (self.store / layout.PARTNERS / chrom).is_dir()

    @property
    def complete(self) -> bool:
        """Whether every chromosome of the Variant Set has its Partner Rows: what the provenance
        claims, and the only state in which it may (partner-rows §6.2)."""
        return all(self.present(chrom) for chrom in self.ranges)

    @property
    def whole_after(self) -> bool:
        """Whether this run would leave every chromosome with its Partner Rows, which is what
        `--drop-index` needs before it does anything (partner-rows §6.2)."""
        return all(self.present(chrom) or chrom in self.named for chrom in self.ranges)


def derive_partners(
    location: str | os.PathLike[str],
    *,
    t_partner: float,
    threads: int = 1,
    processes: int = 1,
    chromosomes: Sequence[str] | None = None,
    drop_index: bool = False,
) -> Provenance | None:
    """Add the Partner Rows to the existing LD Store at `location`, from its own Near Band and
    Far Pairs, and return the provenance that claims them (partner-rows §6.2; ADR 0011, D6).

    `t_partner` is the **Partner Threshold**, a required input with no default, and never above
    the store's recorded `t_far`. Chromosomes are derived `processes` at a time, largest first,
    each on `threads` compressor threads, and `chromosomes` names the ones to derive (all of them
    by default). The rules run on `threads` too: on EUR's genome they read the whole Near Band and
    every frame, and that is the larger half of a derivation's time (ogld#69).

    The provenance is written once, at the end, and only when every chromosome of the Variant Set
    has its Partner Rows and the ten rules of `PARTNER_RULES` pass: a run that derives some
    chromosomes and not all leaves the provenance as it was and returns None, and an interrupted
    run leaves the store the one it was, since `partners/` is not a node of its hierarchy until
    then. A store whose provenance already claims its Partner Rows is refused as nothing to do,
    unless `chromosomes` names ones it lacks.

    `drop_index` removes a Proxy Index the store still holds — a leftover of a store derived
    before #72 — and its count, a step of its own (partner-rows §6.2): it does so only for a store
    whose Partner Rows pass those rules, and never for one whose derivation is incomplete. A store
    this package writes holds no `index/`, so there it is nothing to do.
    """
    store = Path(location)
    root = _writable(store)
    plan = _plan(root, store, t_partner, chromosomes)
    _refuse_writes(plan)
    if drop_index and not plan.whole_after:
        raise InputError(_incomplete(plan))
    if plan.missing:
        _derive(plan, threads=threads, processes=processes)
    elif plan.claimed:
        _nothing_to_do(plan, drop_index)
    else:
        plan = _settle(root, plan, t_partner, threads, processes)
    if not plan.complete:
        return None
    made = _claim(root, plan, t_partner, threads)
    return _drop_index(root, store, made) if drop_index else made


def _settle(root: zarr.Group, plan: _Plan, t_partner: float, threads: int, processes: int) -> _Plan:
    """An unclaimed component — a whole `partners/` that no provenance claims, which a run killed
    between publishing it and writing the provenance leaves — is completed or rebuilt, never
    trusted (partner-rows §6.2).

    The ten rules decide which: they read the component alone, so a component they accept is one
    provenance write away from being the store's, and one they refuse is removed and derived again
    from the band. Nothing reads it in the meantime: a reader and validation both key on the
    provenance, never on a group being there (ogld#69).
    """
    if _rules_pass(root, plan, t_partner, threads):
        return plan
    _remove_component(plan.store)
    fresh = _plan(root, plan.store, t_partner, tuple(plan.ranges))
    _derive(fresh, threads=threads, processes=processes)
    return fresh


def _rules_pass(root: zarr.Group, plan: _Plan, t_partner: float, threads: int) -> bool:
    """Whether the ten rules accept the component of a store that does not claim it yet, which
    only their answer can say: `partners/` is made a node first, since a 0.2.0 store's component is
    one (`partners.files`), and the claim that would make it the store's is not written here.

    The residue's own metadata is part of what they judge: a group that cannot say how many
    entries it holds — no `entries`, or metadata that will not parse — is a component that cannot
    be claimed, so it is a **rejection** and `_settle` derives it again. A failure to read the
    store's *source* — its variant table, its Near Band, its Far Pairs — is the store's own, and
    propagates instead (ogld#69).
    """
    _publish(root)
    try:
        made = _claimed(plan, t_partner)
    except InputError:
        return False
    findings = validate_group(root, made, PARTNER_RULES, threads=threads)
    return not any(findings.values())


def _remove_component(store: Path) -> None:
    """Remove every chromosome's Partner Rows, and the directory that holds them: an unclaimed
    component the rules refused, which a rerun derives again (partner-rows §6.2)."""
    component = store / layout.PARTNERS
    shutil.rmtree(component, ignore_errors=False)
    component.mkdir()


def _nothing_to_do(plan: _Plan, drop_index: bool) -> None:
    """Refuse a run whose store already has every Partner Rows' chromosome, unless it is the
    Proxy Index's removal that was asked for."""
    if plan.claimed and not drop_index:
        raise InputError(
            f"{plan.store} already records Partner Rows at t_partner = "
            f"{plan.recorded['t_partner']!r} over {len(plan.ranges)} chromosome(s): there is "
            "nothing to derive, and --chromosomes names one it lacks"
        )


def _claim(root: zarr.Group, plan: _Plan, t_partner: float, threads: int) -> Provenance:
    """Make the derived component the store's: the ten rules pass, `partners/` becomes a node of
    the hierarchy, and the provenance that describes it is written last (partner-rows §6.2, §10).

    That last write is **the claim**, and it is one write: the root's metadata is replaced whole, so
    a run killed before it leaves a store whose provenance says 0.1.0 — a store that does not have
    the Partner Rows, whatever directories exist (§6.2). Nothing reads a component the provenance
    does not claim, so the node comes first and the rules judge the component as the store will
    hold it: a 0.2.0 store's Partner Rows are a node of its hierarchy (`partners.files`), and only
    the claim that follows makes them the store's.

    What a killed run left unpublished goes first: it is this run's own, no writer is live beside
    it, and `partners.files` refuses a store that holds one (§5.1).
    """
    _clear_temporaries(plan.store / layout.PARTNERS)
    made = _claimed(plan, t_partner)
    _refuse_temporaries(plan.store)
    _publish(root)
    findings = validate_group(root, made, PARTNER_RULES, threads=threads)
    if any(findings.values()):
        raise ValidationFailed(findings)
    _record(root, made)
    return cast(Provenance, made)


def _publish(root: zarr.Group) -> None:
    """Make `partners/` a node of the store's hierarchy, before the claim that activates it."""
    root.require_group(layout.PARTNERS)


def _record(root: zarr.Group, made: dict[str, Any]) -> None:
    """The claim: the provenance, in one replaced metadata write (partner-rows §6.2)."""
    root.attrs.update({"ld_store": made})


def _writable(store: Path) -> zarr.Group:
    """The store's root group, opened to be written: a derivation writes `partners/` and the
    provenance, and nothing else, and neither until every chromosome is in place (§6.2)."""
    open_group(store)  # names a location that is no Zarr group at all, as a reader does
    try:
        return zarr.open_group(str(store), mode="r+")
    except (OSError, ValueError) as unwritable:
        raise InputError(f"{store} cannot be written to: {unwritable}") from unwritable


def _refuse_writes(plan: _Plan) -> None:
    """Refuse, before anything is written, a store whose writable paths are not plainly the
    store's own: a derivation writes the root metadata, `partners/` and every group and temporary
    in it, and a symbolic link at any of them puts a write, or a deletion, in the directory the
    link names — even one inside the store, which a cleanup would then empty (ogld#69).

    It also refuses a `zarr.*.partial` a store already holds, in the root or in `partners/`: zarr
    writes a metadata document through one (`<path>.<uuid>.partial`, opened `wb` with no
    `O_NOFOLLOW`) and then replaces the document with it, so a link at that name would be written
    through and published as the store's metadata.

    A store's read-only components may be links: the scratch stores #71 and #73 read are a copied
    `zarr.json` beside links to the published store's `near/`, `far/`, `variants/` and `index/`
    (a published 0.1.0 store holds that last one until its rollout drops it).

    This check runs before the first write and is not a lock: a concurrent writer to the same store
    could still plant one between the check and the write. A store has one writer (as §6.1's carry
    and `FramesGroup` assume), and a derivation is it.
    """
    for path in _written(plan):
        if path.is_symlink():
            raise InputError(
                f"{path} is a symbolic link: a derivation writes only what is plainly inside the "
                f"store at {plan.store.resolve()}, and never through a link (partner-rows §6.2)"
            )
    _refuse_temporaries(plan.store)


def _refuse_temporaries(store: Path) -> None:
    """Refuse a store holding a metadata temporary, which zarr would write through and publish.

    It is checked again immediately before the claim, so a temporary that appeared while the
    derivation ran is refused too (ogld#69)."""
    for directory in (store, store / layout.PARTNERS):
        if not directory.is_dir() or directory.is_symlink():
            continue
        for temporary in sorted(directory.glob("zarr.*.partial")):
            raise InputError(
                f"{temporary} is a temporary a metadata write left behind or was planted at, and "
                "zarr writes through it: remove it before deriving into this store "
                "(partner-rows §6.2)"
            )


def _written(plan: _Plan) -> Iterator[Path]:
    """The paths a run owns: the root metadata, `partners/`, and every entry of it — the groups it
    derives or claims, and the temporaries it removes. A store is derived into, so its component
    is the store's own, and a link at any of these would put it somewhere else."""
    store = plan.store
    yield store / "zarr.json"
    component = store / layout.PARTNERS
    yield component
    if component.is_dir() and not component.is_symlink():
        yield from component.iterdir()


def _plan(
    root: zarr.Group, store: Path, t_partner: float, chromosomes: Sequence[str] | None
) -> _Plan:
    """What `derive_partners` reads before it writes: the store's encoding, its distances, the
    threshold's floor, and the chromosomes that lack Partner Rows, largest first (§6.3)."""
    recorded = root.attrs.get("ld_store")
    if not isinstance(recorded, dict):
        raise InputError(f"{store} has no ld_store provenance: it is not a finished LD Store")
    encoding = REncoding.of(recorded.get("encoding"))
    floor = encoding.partner_code_floor(t_partner, _number(recorded, "t_far", store))
    ranges = _ranges(root)
    named = _named(chromosomes, ranges, store)
    claimed = _claimed_at(recorded, t_partner, floor, store)
    missing = tuple(
        sorted(
            (chrom for chrom in named if not (store / layout.PARTNERS / chrom).is_dir()),
            key=lambda chrom: ranges[chrom][1] - ranges[chrom][0],
            reverse=True,
        )
    )
    return _Plan(
        store,
        encoding,
        _integer(recorded, "near_bp", store),
        _integer(recorded, "far_bp", store),
        floor,
        _integer(recorded, "far_code_floor", store),
        recorded,
        ranges,
        named,
        missing,
        claimed,
    )


def _named(
    chromosomes: Sequence[str] | None, ranges: dict[str, tuple[int, int]], store: Path
) -> tuple[str, ...]:
    """The chromosomes a run was asked for, in the order asked, each one of the store's."""
    if chromosomes is None:
        return tuple(ranges)
    unknown = [chrom for chrom in chromosomes if chrom not in ranges]
    if unknown:
        raise InputError(
            f"{', '.join(unknown)}: not a chromosome of {store}, whose Variant Set is on "
            f"{', '.join(ranges)}"
        )
    return tuple(dict.fromkeys(chromosomes))


def _claimed_at(recorded: dict[str, Any], t_partner: float, floor: int, store: Path) -> bool:
    """Whether the provenance claims Partner Rows already, and at this floor (spec §13;
    partner-rows §6.2). The claim is `ogld.format.claims_partner_rows`, one rule for the reader,
    validation and this derivation: a store that records 0.1.0 is unclaimed whatever keys it
    carries, so a run settles its residue instead of refusing it as nothing to derive. A store
    that does claim them at another floor is refused rather than given frames at a second: a row
    would hold two floors."""
    if not claims_partner_rows(recorded):
        return False
    said, at = recorded.get("t_partner"), recorded.get("partner_code_floor")
    if said != t_partner or at != floor:
        raise InputError(
            f"{store} records Partner Rows at t_partner = {said!r} (code floor {at!r}), not "
            f"{t_partner!r} (code floor {floor}): deriving at a second floor would put two in "
            "one store's rows (partner-rows §3)"
        )
    return True


def _ranges(root: zarr.Group) -> dict[str, tuple[int, int]]:
    """Each chromosome's rows in the Variant Set, as the `variants` group records them (spec §2)."""
    variants = root.get("variants")
    if not isinstance(variants, zarr.Group):
        raise InputError("the store has no variants group: it is not an LD Store")
    try:
        listed = layout.chromosomes_listed(variants.attrs.get("chromosomes"))
    except ValueError as malformed:
        raise InputError(f"the chromosomes ranges are malformed: {malformed}") from malformed
    return {name: (start, stop) for name, start, stop in listed}


def _number(recorded: dict[str, Any], key: str, store: Path) -> float:
    """A provenance number, refused for a missing one with the store that lacks it named."""
    value = recorded.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise InputError(f"{store} records no {key}, which is what a derivation thresholds at")
    return float(value)


def _integer(recorded: dict[str, Any], key: str, store: Path) -> int:
    """A provenance integer, refused for a missing one with the store that lacks it named."""
    value = recorded.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise InputError(f"{store} records no {key}, which is what a derivation reads the band by")
    return value


def _incomplete(plan: _Plan) -> str:
    """Why a Proxy Index a store holds is not removed: its Partner Rows are not all there."""
    absent = [chrom for chrom in plan.ranges if not plan.present(chrom)]
    return (
        f"{plan.store} is missing the Partner Rows of {len(absent)} chromosome(s), "
        f"{', '.join(absent)}: --drop-index removes the Proxy Index only from a store whose "
        "Partner Rows are whole (partner-rows §6.2)"
    )


def _derive(
    plan: _Plan,
    *,
    threads: int,
    processes: int,
    work: Callable[[_Schedule], int] | None = None,
) -> None:
    """Derive the chromosomes `plan` lacks, largest first, on `processes` worker processes
    (partner-rows §6.3). One process derives them one after another; more take one each.

    `work` is the unit one worker runs, `derive_partner_rows` unless it is given: it is a
    parameter so that the pool's own contract — a child process a chromosome, and a worker's
    failure stopping the run and publishing nothing — is held by a test and not by a reading of
    this function (ogld#69).
    """
    run = derive_partner_rows if work is None else work
    if threads < 1 or processes < 1:
        raise InputError(
            f"threads and processes count workers, so each is at least 1, not {threads} and "
            f"{processes}"
        )
    component = plan.store / layout.PARTNERS
    component.mkdir(exist_ok=True)
    jobs = [
        _Schedule(
            plan.store,
            chrom,
            *plan.ranges[chrom],
            plan.near_bp,
            plan.far_bp,
            plan.floor,
            plan.far_floor,
            plan.encoding,
            threads,
        )
        for chrom in plan.missing
    ]
    if processes == 1:
        for job in jobs:
            run(job)
        return
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(processes, len(jobs)), mp_context=context) as pool:
        list(pool.map(run, jobs))


def derive_partner_rows(job: _Schedule) -> int:
    """Write one chromosome's Partner Rows from its store's Near Band and Far Pairs, and return
    them: every pair's code in both variants' rows, twice the pairs (partner-rows §5.3, §6.2).

    The Near Band is read a block of rows at a time through the direct decoder (ADR 0009), and the
    Far Pairs as their `indptr` whole and their partner and code arrays one block's worth at a
    time. Both are handed to the emitter a build hands its cuts, so the frames are a build's, byte
    for byte (§6.1). `FramesGroup` publishes the group by one rename, so the chromosome is whole or
    absent however the run ends (§5.1).
    """
    root = open_group(job.store)
    cache = ChunkCache(0)
    lengths = near_row_lengths(
        _slice(cache, _array(root, "variants/position"), job.start, job.rows).astype(np.int64),
        job.near_bp,
    )
    far = root.get(f"far/{job.chrom}")
    if not isinstance(far, zarr.Group):
        raise InputError(f"{job.store} has no far/{job.chrom}: the store holds no Far Pairs")
    stream = PartnerRowsStream(
        root,
        job.chrom,
        lengths,
        job.encoding,
        partner_floor=job.floor,
        far_floor=job.far_floor,
        near_bp=job.near_bp,
        far_bp=job.far_bp,
        workers=job.threads,
        container=job.store / layout.PARTNERS,
    )
    _fed(stream, cache, _array(root, f"near/{job.chrom}/r"), far, indptr_of(lengths))
    return stream.close()


def _fed(
    stream: PartnerRowsStream,
    cache: ChunkCache,
    near: zarr.Array[Any],
    far: zarr.Group,
    indptr: npt.NDArray[np.int64],
) -> None:
    """Feed `stream` the chromosome's rows a block at a time: the block's Near Band codes, and its
    rows' Far Pairs as their earlier row and partner (partner-rows §6.1, §6.2)."""
    rows = len(indptr) - 1
    far_indptr = _slice(cache, _array(far, "indptr"), 0, rows + 1)
    partner, codes = _array(far, "partner"), _array(far, "r")
    for a, b in blocks((indptr, DERIVE_ENTRIES)):
        first, last = int(indptr[a]), int(indptr[b])
        at, many = int(far_indptr[a]), int(far_indptr[b] - far_indptr[a])
        stream.add(
            b - a,
            _slice(cache, near, first, last - first),
            owners(np.diff(far_indptr[a : b + 1]), np.arange(a, b, dtype=np.int64)).astype(
                np.int32
            ),
            _slice(cache, partner, at, many),
            _slice(cache, codes, at, many),
        )


def _array(group: zarr.Group, path: str) -> zarr.Array[Any]:
    """The array at `path` of `group` or of the store, refused where the store has none: a 0.1.0
    store is derived from its Near Band and its Far Pairs, and a store without them has no
    Partner Rows to derive (partner-rows §6.2)."""
    found = group.get(path)
    if not isinstance(found, zarr.Array):
        raise InputError(
            f"{path} is missing: a store is derived from its own Near Band and Far Pairs"
        )
    return found


def _slice(cache: ChunkCache, array: zarr.Array[Any], first: int, length: int) -> npt.NDArray[Any]:
    """Elements `first` … `first + length − 1` of `array`, read through the direct decoder, and
    refused where one is a value no store holds there: a chunk the store lost (ADR 0009)."""
    return layout.refused_if_lost(array, cache.runs(array, [first], [length])[0])


def _clear_temporaries(component: Path) -> None:
    """Remove the groups a killed run never renamed into place: an entry of `partners/` whose
    name begins with a dot is one, and none can be live while a run derives (partner-rows §5.1,
    §6.2). A store that keeps one is refused by `partners.files`, so it is removed, not claimed
    over."""
    for stale in component.iterdir():
        if not stale.name.startswith("."):
            continue
        if stale.is_symlink() or not stale.is_dir():
            # A link is removed, never followed: what it names is not this store's (ogld#69).
            stale.unlink(missing_ok=True)
        else:
            shutil.rmtree(stale, ignore_errors=False)


def _claimed(plan: _Plan, t_partner: float) -> dict[str, Any]:
    """`plan.recorded` as the store's provenance once its Partner Rows are claimed: format
    0.2.0's four keys, the count over every chromosome's own `entries`, and no other change
    (partner-rows §5.3; spec §7)."""
    made = dict(plan.recorded)
    made["format_version"] = FORMAT_VERSION
    made["t_partner"] = t_partner
    made["partner_code_floor"] = plan.floor
    made["counts"] = {
        **cast(dict[str, Any], made["counts"]),
        "partner_entries": _partner_entries(plan),
    }
    made["software"] = {**cast(dict[str, Any], made["software"]), "zstd": frames.zstd_version()}
    return made


def _removed(index: Path) -> None:
    """Remove the Proxy Index a store holds at `index`: the link itself when it is one, and never
    what a link names. A removal that fails raises, naming the path."""
    try:
        if index.is_symlink() or not index.is_dir():
            index.unlink(missing_ok=True)
        else:
            shutil.rmtree(index)
    except OSError as failed:
        raise InputError(
            f"{index} could not be removed ({failed}): its count is kept, and the Proxy Index is "
            "still the store's (partner-rows §6.2)"
        ) from failed


def _partner_entries(plan: _Plan) -> int:
    """Every chromosome's Partner Rows entries, from each published group's own attribute: twice
    the pairs the component holds (partner-rows §5.3). A group without it is refused rather than
    counted as none, which would claim a component smaller than the one in the store."""
    total = 0
    for chrom in plan.ranges:
        try:
            group = zarr.open_group(str(plan.store / layout.PARTNERS / chrom), mode="r")
        except (OSError, ValueError) as unreadable:
            raise InputError(
                f"{layout.partners_path(chrom)} cannot be read as a group ({unreadable}): its "
                "Partner Rows are not a component a store claims (partner-rows §5.3)"
            ) from unreadable
        entries = group.attrs.get("entries")
        if not isinstance(entries, int) or isinstance(entries, bool):
            raise InputError(
                f"{layout.partners_path(chrom)} records no entries: its Partner Rows are not a "
                "group a build or a derivation wrote (partner-rows §5.3)"
            )
        total += entries
    return total


def _drop_index(root: zarr.Group, store: Path, recorded: Provenance) -> Provenance:
    """Remove a Proxy Index a store still holds, whose Partner Rows hold every entry it had
    (partner-rows §6.2; ADR 0011, D1): its `index/` group and its `counts.index_entries`.

    From #72 a store this package writes has none, and no rule expects one: the removal is what
    finishes a store derived by an earlier `ogld derive-partners`, and a store without one is left
    as it is. The count goes with the arrays, in that order, and only after they are gone: a
    removal that fails leaves the count in place and raises, naming the path, and a run killed
    between the two leaves a count with no arrays, which validation reads as an extra provenance
    key and ignores (spec §7).

    A scratch store's `index/` is a link to a published one, which is read, never removed: the link
    goes and the directory it names stays (partner-rows §6.2).
    """
    _removed(store / "index")
    counts = dict(cast(dict[str, Any], recorded["counts"]))
    counts.pop("index_entries", None)
    made: dict[str, Any] = dict(recorded)
    made["counts"] = counts
    root.attrs.update({"ld_store": made})
    return cast(Provenance, made)


def derive_alid_mmap(path: Path) -> int:
    """Publish the LD Store's optional local companion (ADR 0012)."""
    return derive_variant_companion(path, provenance_key="ld_store")


def derive_variant_companion(path: Path, *, provenance_key: str) -> int:
    """Publish an optional local companion from validated canonical arrays (ADR 0012).

    Serialize cooperating derivations. A private partial is fsynced before its immutable
    generation rename; replacing root metadata publishes the pointer last. A retry cleans a
    partial or validates/reuses a complete generation, never overwrites a published one.
    """
    directory = path / layout.ALID_MMAP
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _derive_alid_mmap(path, directory, provenance_key)


def _derive_alid_mmap(path: Path, directory: Path, provenance_key: str) -> int:
    root = zarr.open_group(path, mode="r")
    rules = ("alid_index.count", "alid_index.rows", "alid_index.order", "alid_index.keys")
    findings = validate_group(root, root.attrs.get(provenance_key), rules=rules)
    if any(findings.values()):
        raise ValidationFailed(findings)
    existing = layout.alid_companion(root)
    if existing is not None:
        return existing.count
    sources = layout.alid_mmap_sources(path)
    partial = directory / ".partial"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir()
    count = _write_alid_payloads(root, partial)
    payloads = {
        name: layout.file_sha256(partial / name)
        for name in (payload.name for payload in layout.ALID_MMAP_PAYLOADS)
    }
    raw = layout.json_bytes(layout.alid_mmap_manifest(count, sources, payloads))
    _durable_bytes(partial / layout.ALID_MMAP_MANIFEST, raw)
    _sync_directory(partial)
    pointer = layout.alid_mmap_pointer(raw)
    destination = directory / layout.alid_mmap_generation(pointer)
    _install_alid_generation(partial, destination, payloads)
    _sync_directory(directory)
    changed = layout.alid_mmap_source_change(sources, layout.alid_mmap_sources(path))
    if changed is not None:
        raise layout.alid_mmap_error(
            destination / layout.ALID_MMAP_MANIFEST,
            f"canonical source {path / changed}: changed during companion derivation",
        )
    # Integrity hashes cannot establish correspondence. This same bounded comparison is
    # `alid_index.count`'s, and must pass before the publication point (ADR 0012, round 2).
    require_alid_mmap(root, destination, count)
    document = json.loads((path / "zarr.json").read_bytes())
    document["attributes"][layout.ALID_MMAP_ATTR] = pointer
    temporary = path / ".alid-mmap-root.partial"
    _durable_bytes(temporary, layout.json_bytes(document))
    os.replace(temporary, path / "zarr.json")
    _sync_directory(path)
    return count


def _install_alid_generation(partial: Path, destination: Path, payloads: dict[str, str]) -> None:
    if destination.exists():
        for name in (*payloads, layout.ALID_MMAP_MANIFEST):
            if layout.file_sha256(destination / name) != layout.file_sha256(partial / name):
                raise layout.alid_mmap_error(destination / name, "unpublished generation differs")
        shutil.rmtree(partial)
    else:
        partial.rename(destination)


def _write_alid_payloads(root: zarr.Group, partial: Path) -> int:
    rows = root[layout.ALID_INDEX_ROW]
    assert isinstance(rows, zarr.Array)
    count = int(rows.shape[0])
    for payload in layout.ALID_MMAP_PAYLOADS:
        array = root[payload.source]
        assert isinstance(array, zarr.Array)
        with (partial / payload.name).open("wb") as stream:
            for lo, hi in windows(0, count, ALID_ROWS):
                values = layout.refused_if_lost(
                    array, np.asarray(array[lo * payload.width : hi * payload.width])
                )
                stream.write(values.astype(payload.dtype, copy=False).tobytes())
            stream.flush()
            os.fsync(stream.fileno())
    return count


def _durable_bytes(path: Path, raw: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
