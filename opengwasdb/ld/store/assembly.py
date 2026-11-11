"""What goes in an LD Store, from the r its producer supplies (spec §3–§7, §12; ogld#17).

`StoreWriter` is the seam a producer calls. This module, behind it, owns what goes in a store
(owner, 10 November 2026; `docs/architecture/ownership.md`, "Provenance and what goes in a
store"). It handles:

- quantising r at the store's encoding, through `ogld.format.REncoding` (spec §3);
- the Near Band's row lengths and `indptr`, from the positions (spec §4);
- the threshold at the far code floor, judged on codes, for every pair within far_bp (§3, §5);
- both directions of the Proxy Index, partners ascending, one code per pair (spec §5);
- the counts, and the rest of the provenance a store derives from itself (spec §7).

Validation at `finish` stays as it was, all 18 rules of spec §12 through `validate_group`. A
producer supplies r for blocks of rows in order, and at `finish` it supplies what only the
source knows (`ogld.format.Sourced`).

**Memory is bounded by the block and the far_bp window, never the chromosome.** These are
measured on chr6 of lane #5's whole-EUR store (602,466 rows, 6,826,805,033 Near Band codes,
731,767,394 Proxy Index entries). What is held:

- **The Near Band carry:** less than one shard of `r`, 67,108,864 codes, so 128 MiB.
- **The Proxy Index carry:** the pairs whose later row is not yet written. At most these are
  the pairs spanning one row within far_bp, plus one bucket of `CARRY_ROWS` rows. At the MHC
  that is 33,869,608 pairs at 10 B, 323 MiB.
- **The Proxy Index output:** less than one shard each of `partner` and `r`, 6 MiB.
- **Per-row arrays:** each `indptr`, n_c + 1 int64.
- **A cut in flight,** between a producer's worker and `write`: at most 31 MiB for a 256-row
  block.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
import zarr

from ogld.format import (
    FORMAT_VERSION,
    T_FAR_Q,
    Codes,
    Counts,
    Provenance,
    REncoding,
    Sourced,
    t_far,
)
from ogld.format.geometry import (
    chromosome_index,
    indptr_of,
    near_row_lengths,
    spans_at,
    stop_within,
)
from ogld.store import layout

#: The rows of one bucket of the Proxy Index's carried pairs. Pairs are carried by the bucket
#: their later row falls in, so a bucket is freed once its rows are written. Only the pairs of
#: one partly written bucket are held beyond those spanning the row being written.
CARRY_ROWS: Final = 256

_Rows = npt.NDArray[np.int32]


@dataclass(frozen=True)
class Cut:
    """A block of rows cut from its producer's r: what `StoreWriter.write` writes.

    `first` is the block's first global row and `rows` how many it holds, all on one chromosome.
    `near` holds their Near Band codes, row after row (spec §4). `lo`, `hi` and `codes` are their
    Proxy Index pairs, as local rows lo < hi: each partner hi within far_bp of row lo whose code
    reaches the far code floor, once, with its one code (spec §3, §5), ascending by (lo, hi).
    `far_candidates` counts the block's pairs more than near_bp and at most far_bp apart, all
    judged, and `far_pairs` those of them kept (spec §7). `by` is the `Cutter` that cut it, so a
    cut is written only by its own store.

    A producer neither builds nor reads one: it hands what `StoreWriter.cut` returned to
    `StoreWriter.write`.
    """

    first: int
    rows: int
    near: Codes
    lo: _Rows
    hi: _Rows
    codes: Codes
    far_candidates: int
    far_pairs: int
    by: Cutter


class Cutter:
    """What every block of one store is cut by, fixed when the store is created: the Variant
    Set's positions and chromosome ranges, near_bp and far_bp, the encoding, and t_far with its
    far code floor (spec §3–§5). Nothing in it changes after `create`, so blocks are cut on any
    thread, in any order."""

    def __init__(
        self,
        position: npt.NDArray[np.int64],
        ranges: dict[str, tuple[int, int]],
        distances: tuple[int, int],
        encoding: REncoding,
        *,
        n: int,
        k: int,
    ) -> None:
        """Blocks are thresholded as they arrive, so t_far for `n` samples and `k` covariates, and
        its floor, are fixed here, before the first; an n and k with no t_far raise here. The
        positions are copied and made read-only, so no caller can change what a cut reads."""
        self.t_far = t_far(n, k)
        self.floor = encoding.far_code_floor(self.t_far)
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
        lo, hi, codes = self._pairs(first, r, far_stop)
        return Cut(
            first,
            rows,
            near,
            (local + lo).astype(np.int32),
            (local + hi).astype(np.int32),
            codes,
            int((far_stop - near_stop).sum()),
            int((hi >= near_stop[lo]).sum()),
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
        self, first: int, r: npt.NDArray[np.floating[Any]], far_stop: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], Codes]:
        """The block's pairs (t, c), t < c < far stop(t), whose code reaches the floor, as
        offsets from its first row, ascending by (t, c), with their codes. Only those whose |r|
        is not below the encoding's `least_r(floor)` are quantised, which keeps every pair whose
        code can reach the floor; NaN is never below it. The test is two comparisons, not |r|,
        which would copy the block's r: 133 MB a worker on chr6's widest tile."""
        below = (r > -self._least) & (r < self._least)
        t, c = np.nonzero(~below)
        del below
        inside = (c > t) & (c < far_stop[t])
        t, c = t[inside], c[inside]
        codes = self._quantised(r[t, c], lambda at: (first + int(t[at]), first + int(c[at])))
        keep = np.abs(codes.astype(np.int32)) >= self.floor
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


def _by_later_row(pairs: list[_Pairs]) -> _Pairs:
    """Pieces whose earlier rows ascend from piece to piece, each sorted by (hi, lo), as one
    array sorted by (hi, lo): a stable sort by hi keeps lo ascending within each hi."""
    lo, hi, codes = (np.concatenate([p[i] for p in pairs]) for i in range(3))
    order = np.argsort(hi, kind="stable")
    return lo[order], hi[order], codes[order]


class ProxyIndexStream:
    """One chromosome's Proxy Index, both directions, written row by row as blocks arrive (§5).

    Row l holds every pair (m, l) with m < l, which come from earlier blocks or its own, then
    every pair (l, m) with m > l, which come from its own block. So row l is complete when its
    own block arrives, and each block's rows are written then: partners below ascending, then
    partners above, each with its pair's one code. A block's pairs whose later row is still to
    come are carried, in buckets of `CARRY_ROWS` rows by that later row. That is at most the
    pairs spanning one row within far_bp and one bucket, however long the chromosome.

    `partner` and `r` grow shard by shard (`Appender`). `indptr` is held, n_c + 1 int64, and
    written at the end.
    """

    def __init__(self, group: zarr.Group, rows: int, encoding: REncoding) -> None:
        #: Each row's entries once written; −1 until then, so a row never written is never read as
        #: a row with no partners.
        self._counts = np.full(rows, -1, dtype=np.int64)
        self._group = group
        self._next = 0
        self._carried: dict[int, list[_Pairs]] = {}
        shard = layout.SIDE[1]
        side = layout.SIDE
        self._partner = Appender(
            lambda n: layout.new_numbers(group, "partner", n, np.int32, layout.PARTNER_FILL, side),
            np.int32,
            shard,
        )
        self._codes = Appender(
            lambda n: layout.new_r_array(group, "r", n, side, encoding), encoding.dtype, shard
        )

    def add(self, cut: Cut) -> None:
        """Write the rows of `cut`, which continue the rows already written, and carry its
        pairs whose later row is still to come."""
        a, b = self._next, self._next + cut.rows
        closing = cut.hi < b
        below = [*self._take(b), (cut.lo[closing], cut.hi[closing], cut.codes[closing])]
        lo, hi, codes = _by_later_row(below)
        self._carry(cut.lo[~closing], cut.hi[~closing], cut.codes[~closing])
        owner = np.concatenate([hi, cut.lo])
        partner = np.concatenate([lo, cut.hi])
        order = np.argsort(owner, kind="stable")
        self._counts[a:b] = np.bincount(owner - a, minlength=b - a)
        self._partner.add(partner[order])
        self._codes.add(np.concatenate([codes, cut.codes])[order])
        self._next = b

    def _carry(self, lo: _Rows, hi: _Rows, codes: Codes) -> None:
        """Keep pairs whose later row is still to come, sorted by (hi, lo), in the bucket of
        that row: a copy each, so a bucket's memory goes when it is written."""
        if not len(hi):
            return
        lo, hi, codes = _by_later_row([(lo, hi, codes)])
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
                continue
            kept = []
            for lo, hi, codes in pieces:
                at = int(np.searchsorted(hi, b))
                taken.append((lo[:at], hi[:at], codes[:at]))
                if at < len(hi):
                    kept.append((lo[at:], hi[at:], codes[at:]))
            if kept:
                self._carried[key] = kept
        return taken

    def close(self) -> None:
        """Write `indptr` and the last entries. A row not written, or a carried pair left over,
        would be a row the index lacks, and is refused."""
        if self._carried or (self._counts < 0).any():
            raise ValueError("the Proxy Index has rows not written: a store assembly fault")
        layout.put_numbers(self._group, "indptr", indptr_of(self._counts), layout.INT64_FILL)
        self._partner.close()
        self._codes.close()


@dataclass
class Tally:
    """The counts provenance records (spec §7), added up block by block."""

    near_pairs: int = 0
    far_candidates: int = 0
    far_pairs: int = 0
    index_entries: int = 0

    def add(self, cut: Cut) -> None:
        """Count a cut: its Near Band codes, its pairs judged and kept beyond near_bp, and two
        Proxy Index entries for each pair it keeps, one in each of its rows (spec §5)."""
        self.near_pairs += len(cut.near)
        self.far_candidates += cut.far_candidates
        self.far_pairs += cut.far_pairs
        self.index_entries += 2 * len(cut.codes)

    def counts(self) -> Counts:
        return {
            "near_pairs": self.near_pairs,
            "far_candidates": self.far_candidates,
            "far_pairs": self.far_pairs,
            "index_entries": self.index_entries,
        }


class Chromosome:
    """One chromosome being written: its Near Band and its Proxy Index."""

    def __init__(self, root: zarr.Group, cutter: Cutter, name: str) -> None:
        self.name = name
        position = cutter.positions(name)
        self.near = NearBandStream(
            root.require_group(f"near/{name}"),
            near_row_lengths(position, cutter.near_bp),
            cutter.encoding,
        )
        self.index = ProxyIndexStream(
            root.require_group(f"index/{name}"), len(position), cutter.encoding
        )

    def add(self, cut: Cut) -> None:
        self.near.add(cut.near)
        self.index.add(cut)

    def close(self) -> None:
        """Finish both."""
        self.near.close()
        self.index.close()


_PRODUCER_KEYS: Final = frozenset(Sourced.__required_keys__)


def provenance(sourced: Sourced, cutter: Cutter, n_variants: int, tally: Tally) -> Provenance:
    """The store's provenance, in spec §7's order. It is the producer's `sourced` plus what the
    store derives from itself: `format_version`, `encoding`, `n_variants`, `near_bp`, `far_bp`,
    `t_far`, `t_far_q`, `far_code_floor` and `counts`.

    Spec §7 lets a store hold other keys, which a reader ignores. This writer records none: a key
    `Sourced` does not name is refused, whether it is one the store derives, which a producer may
    not give, or one no reader knows, which would otherwise be stored unread and unchecked. A
    `sourced` that lacks a key of the producer's, or records an n or k other than the threshold's,
    is refused too."""
    given = dict(sourced)
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
    said = (sourced["panel"]["n"], sourced["adjustment"]["k"])
    if said != (cutter.n, cutter.k):
        raise ValueError(
            f"provenance records n = {said[0]} and k = {said[1]}, but the store was "
            f"thresholded at t_far for n = {cutter.n} and k = {cutter.k} (spec §3)"
        )
    derived: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "encoding": cutter.encoding.record(),
        "n_variants": n_variants,
        "near_bp": cutter.near_bp,
        "far_bp": cutter.far_bp,
        "t_far": cutter.t_far,
        "t_far_q": T_FAR_Q,
        "far_code_floor": cutter.floor,
        "counts": tally.counts(),
    }
    order = Provenance.__annotations__
    return cast(Provenance, {key: derived.get(key, given.get(key)) for key in order})
