"""Reading an LD Store: the seam every query is built on (spec §1–§6, §10; ogld#4).

`StoreReader` is the store layer's whole read interface. It hides Zarr, the chunk and
shard shapes, chromosome-local rows and the int16 codes: callers pass and receive global
Variant Set rows and Stored r as float64. Every unstored pair it reports is NaN, never a
number, and the meaning of a NaN is the format's to say (`ogld.format.answer_kinds`).

Reads are bounded (spec §10). Opening reads the provenance and the chromosome ranges, and
nothing else. A chromosome's positions and its two `indptr` arrays are read the first time
a request touches it, and kept. Every other read goes through the reader's `ChunkCache`,
which fetches only the inner chunks holding the elements a request needs, from disk or over
HTTP alike, and keeps them, so a query that reads a few pairs at a time fetches each once.

A chunk the store lost reads as its array's fill value, and spec §6 chose each fill value
as one a valid LD Store never holds. So every read refuses that value, naming the array,
rather than decode it: a lost Near Band chunk is a refusal, never a run of r = 0.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
import zarr

from ogld.format import (
    ENCODING,
    FORMAT_VERSION,
    RESERVED_CODE,
    InputError,
    Provenance,
    dequantise,
    judge_window,
)
from ogld.format.geometry import (
    blocks,
    chromosome_index,
    global_rows,
    indptr_of,
    local_rows,
    near_address,
    owners,
    position_rows,
    row_pairs,
    spans,
    stop_within,
    windows,
    within,
)
from ogld.identity import alid_locus
from ogld.store.layout import chromosomes_listed, never_data, shown_fill
from ogld.store.location import ChunkCache, open_group
from ogld.store.tables import Partners, Variants

#: Each chromosome's arrays (spec §1), in the order `_Chromosome` takes them.
_CHROMOSOME_ARRAYS: Final = (
    ("near", "indptr"),
    ("near", "r"),
    ("index", "indptr"),
    ("index", "partner"),
    ("index", "r"),
)
#: The variant table's columns (spec §2).
_VARIANT_COLUMNS: Final = ("position", "alid", "a1", "a2", "a1_frequency")
#: The pairs one step of `pair_r` resolves at once, which bounds its working memory.
_PAIRS_PER_STEP: Final = 1 << 22


@dataclass(frozen=True)
class _Chromosome:
    """One chromosome: its rows, the positions and `indptr` arrays a reader keeps (spec §10),
    and handles on its code arrays, read element by element as requests need them."""

    name: str
    start: int
    stop: int
    position: npt.NDArray[np.int64]
    near_indptr: npt.NDArray[np.int64]
    near_r: zarr.Array[Any]
    index_indptr: npt.NDArray[np.int64]
    index_partner: zarr.Array[Any]
    index_r: zarr.Array[Any]


@dataclass(frozen=True)
class _Placed:
    """Ascending, distinct local rows of one chromosome, the row and column of each in a
    square result, and that result: what `pair_r` and `region_r` write their pairs into."""

    local: npt.NDArray[np.int64]
    at: npt.NDArray[np.int64]
    out: npt.NDArray[np.float64]


class StoreReader:
    """An opened LD Store, local or over HTTP (spec §2, §10).

    Opening reads the provenance and the chromosome ranges and refuses a store whose
    `format_version` or `encoding` this package does not implement (spec §13), naming
    it, before anything is decoded.
    """

    _root: zarr.Group
    _provenance: Provenance
    _ranges: dict[str, tuple[int, int]]
    _names: list[str]
    _stops: npt.NDArray[np.int64]
    _loaded: dict[str, _Chromosome]
    _cache: ChunkCache
    _columns: dict[str, zarr.Array[Any]]

    def _attach(self, root: zarr.Group, provenance: Provenance) -> None:
        """Keep the provenance and the chromosome ranges of an opened, checked store."""
        self._root = root
        self._provenance = provenance
        listed = chromosomes_listed(_group(root, "variants").attrs["chromosomes"])
        self._ranges = {name: (start, stop) for name, start, stop in listed}
        self._names = list(self._ranges)
        self._stops = np.array([stop for _, stop in self._ranges.values()], dtype=np.int64)
        self._loaded = {}
        self._cache = ChunkCache()
        # Each handle is opened once: opening one reads its metadata document, which over
        # HTTP is a request.
        self._columns = {name: _kept_array(root, f"variants/{name}") for name in _VARIANT_COLUMNS}

    @classmethod
    def open(cls, location: str | os.PathLike[str]) -> StoreReader:
        """Open the LD Store at a local path or an HTTP(S) URL (ogld#4)."""
        root = open_group(location)
        recorded = root.attrs.get("ld_store")
        if root.metadata.zarr_format != 3:
            raise InputError(
                f"{location} is a Zarr v{root.metadata.zarr_format} hierarchy; an LD Store is "
                "Zarr v3 (spec §1)"
            )
        if not isinstance(recorded, dict):
            raise InputError(
                f"{location} has no ld_store provenance: it is not a finished LD Store"
            )
        if recorded.get("format_version") != FORMAT_VERSION:
            raise InputError(
                f"{location} is LD Store format {recorded.get('format_version')!r}; "
                f"this ogld reads {FORMAT_VERSION!r} only (spec §13)"
            )
        if recorded.get("encoding") != ENCODING:
            raise InputError(
                f"{location} records the encoding {recorded.get('encoding')!r}; "
                f"this ogld reads {ENCODING!r} only (spec §3)"
            )
        reader = cls()
        reader._attach(root, cast(Provenance, recorded))
        return reader

    @property
    def provenance(self) -> Provenance:
        """The root group's `ld_store` attribute, as recorded (spec §7)."""
        return self._provenance

    def rows_of(self, alids: Sequence[str]) -> npt.NDArray[np.int64]:
        """The Variant Set row of each ALID, in order; −1 where the ALID is Absent (spec §2).

        A malformed ALID is Absent too: it names no variant in the store. Each ALID's
        position is found in its chromosome's positions, and only the ALID strings at
        that position are read and compared, all in one read.
        """
        asked: list[int] = []
        candidates: list[npt.NDArray[np.int64]] = []
        for chrom, (which, at) in _loci(alids, self._ranges).items():
            c = self._chromosome(chrom)
            lo, hi = position_rows(c.position, at, at)
            asked.extend(owners(hi - lo, which).tolist())
            candidates.append(global_rows(spans(lo, hi - lo), c.start))
        found = np.full(len(alids), -1, dtype=np.int64)
        if not asked:
            return found
        rows = np.concatenate(candidates)
        names = self._strings("alid", rows)
        query = np.array([alids[i] for i in asked], dtype=np.str_)
        hit = names == query
        found[np.array(asked, dtype=np.int64)[hit]] = rows[hit]
        return found

    def rows_in(self, chrom: str, start: int, stop: int) -> npt.NDArray[np.int64]:
        """The rows on `chrom` with start ≤ position ≤ stop, ascending (spec §2)."""
        if alid_locus(f"{chrom}:1:A:C") is None:
            raise InputError(
                f"{chrom!r} is not a canonical chromosome label (1–22, X, Y, MT; no 'chr' prefix)"
            )
        if chrom not in self._ranges:
            return np.empty(0, dtype=np.int64)
        c = self._chromosome(chrom)
        lo, hi = position_rows(c.position, start, stop)
        return global_rows(np.arange(int(lo), int(hi), dtype=np.int64), c.start)

    def alids_of(self, rows: npt.NDArray[np.int64]) -> npt.NDArray[np.str_]:
        """The ALID of each of `rows`, in the order given (spec §2): the one string column a
        query names its rows by, read without the others (ogld#14)."""
        return self._strings("alid", self._checked(rows))

    def variants(self, rows: npt.NDArray[np.int64]) -> Variants:
        """The Variant Set's columns at `rows`, in the order given (spec §2)."""
        rows = self._checked(rows)
        position = np.empty(len(rows), dtype=np.int64)
        for c, members in self._by_chromosome(rows):
            position[members] = c.position[local_rows(rows[members], c.start)]
        names = np.array(self._names, dtype=np.str_)
        chrom = names[chromosome_index(self._stops, rows)]
        frequency = self._read(self._columns["a1_frequency"], rows)
        return Variants(
            chrom=chrom,
            position=position,
            alid=self._strings("alid", rows),
            a1=self._strings("a1", rows),
            a2=self._strings("a2", rows),
            a1_frequency=frequency.astype(np.float64),
        )

    def pair_r(self, rows: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        """The square matrix of Stored r among `rows`, in the order given (spec §8, §10).

        Pairs within near_bp come from the Near Band, resolved as `r_of_pairs` resolves
        them; farther ones from the Proxy Index rows of the listed variants, each entry that
        is a listed pair scattered into the matrix, rather than searched for pair by pair.
        The diagonal is 1. Every pair the store does not hold, the Bound and Not Measured
        alike, is NaN. The matrix is written in place, so `matrix` holds one float64 copy of
        its result (ogld#14); a row asked for twice is copied from its first.
        """
        rows = self._checked(rows)
        out = np.full((len(rows), len(rows)), np.nan)
        np.fill_diagonal(out, 1.0)
        source = np.arange(len(rows), dtype=np.int64)
        for c, members in self._by_chromosome(rows):
            local, first, back = np.unique(
                local_rows(rows[members], c.start), return_index=True, return_inverse=True
            )
            placed = _Placed(local, members[first], out)
            self._near_square(c, placed)
            self._scatter_far(c, placed)
            source[members] = placed.at[back]
        repeat = np.flatnonzero(source != np.arange(len(rows)))
        # Every row first, then every column: a column copied before its rows would miss them.
        out[repeat] = out[source[repeat]]
        out[:, repeat] = out[:, source[repeat]]
        return out

    def partners(
        self,
        rows: npt.NDArray[np.int64],
        *,
        r2: float,
        window_bp: float,
        among: npt.NDArray[np.int64] | None = None,
    ) -> Partners:
        """Each of `rows`' partners at r̂² ≥ r2 within `window_bp`, both sides, with their
        Stored r and distance (spec §9, §10; ogld#14): the one read `proxies` and `clump` share.

        A partner is another variant on the row's chromosome with distance d ≤ window_bp and
        r̂² ≥ r2, judged on the Stored r in float64 (§3); never the row itself. Row i's
        partners are `partner[indptr[i]:indptr[i + 1]]`, ascending, with `r` and `distance`
        at the same offsets. Both edges are inclusive, as `proxies` asks; `clump` narrows them
        to its own, r̂² > r2 and d < W, because the edges are each query's (§9), not the
        reader's. `among`, ascending distinct global rows, keeps only the partners in it:
        `clump` passes the hits still unclumped, so it resolves only the pairs it needs.

        A request `ogld.format.judge_window` refuses is refused here, by it, with its
        `Refusal`: a window past far_bp, or past near_bp below t_far, where a short answer would
        read as a complete one (§9). The reader never decides that rule itself. Of an answered
        window it chooses the stored part, so no query branches on t_far:

        - r2 ≥ t_far: each row's Proxy Index row, both sides in one piece, and no Near Band
          chunk. The index holds every pair within far_bp at r̂² ≥ t_far (§5), so it misses
          none.
        - r2 < t_far: the Near Band alone, which holds every pair within near_bp (§4). Without
          `among`, each row's Near Band row up to the window and its column within it; with
          `among`, one element of `near/<c>/r` per pair of a row and a member of `among` within
          the window, and nothing else.

        Memory: 48 bytes per kept entry, its partner, r and distance held once as read and
        once in the result while they are put in order. Beside them, one block at a time as
        it is read and judged: at most `_PAIRS_PER_STEP` entries (or one row's, if more) of at
        most 72 bytes each, measured at 30 for the index and 53–70 for the Near Band, and some
        600 bytes of bookkeeping a block. `tests/test_partner_reads.py` holds the bound.
        """
        p = self._provenance
        judge_window(
            r2=r2, window_bp=window_bp, t_far=p["t_far"], near_bp=p["near_bp"], far_bp=p["far_bp"]
        ).raise_if_refused()
        index = r2 >= p["t_far"]
        # d ≤ W is d ≤ ⌊W⌋ for a whole d; the window is answered, so ⌊W⌋ is within the part.
        reach = math.floor(window_bp)
        listed = None if among is None else self._listed(among)
        read: _Read
        estimate: _Estimate = partial(_window_rows, reach=reach)
        if index:
            read, estimate = self._index_entries, _index_length
        elif listed is None:
            read = partial(self._band_entries, reach=reach)
        else:
            read = partial(self._listed_entries, reach=reach, listed=listed)

        def judge(
            gap: npt.NDArray[np.int64], partner: npt.NDArray[np.int64], r: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.bool_]:
            kept = within(gap, reach) & (r * r >= r2)
            return kept if listed is None else kept & _member(partner, listed)

        return self._gather(rows, (read, estimate), judge)

    def r_of_pairs(
        self, first: npt.NDArray[np.int64], second: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float64]:
        """The Stored r of each pair (first[k], second[k]), as one array (spec §8, §10; ogld#14).

        `first` and `second` broadcast against each other, so one row in `first` is one row
        against many. A row with itself is 1. Every pair the store does not hold, the Bound,
        Not Measured and a pair across chromosomes alike, is NaN. A pair within near_bp is one
        element of the Near Band; a pair farther apart is looked up in the Proxy Index row of
        its earlier variant. `partners` (below t_far, with `among`) and `pair_r` resolve
        their pairs as it does, so a pair is resolved in one place.

        Memory: the result, 8 bytes a pair, and one step of at most `_PAIRS_PER_STEP` pairs
        at a time, about 60 bytes each (58 measured), with the index rows of that step's
        earlier variants. The broadcast is never copied, so one row against many holds its
        result and one step. `tests/test_partner_reads.py` holds the bound.
        """
        # Views of the broadcast, never copies: one row against many is one row, repeated by a
        # stride of 0, and each step reads its own slice of them.
        each, other = np.broadcast_arrays(np.asarray(first, np.int64), np.asarray(second, np.int64))
        if each.ndim != 1:
            raise ValueError(f"pairs are one-dimensional lists of rows, not of shape {each.shape}")
        out = np.full(len(each), np.nan)
        for s, e in windows(0, len(out), _PAIRS_PER_STEP):
            out[s:e] = self._step_r(self._checked(each[s:e]), self._checked(other[s:e]))
        return out

    def _step_r(
        self, a: npt.NDArray[np.int64], b: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float64]:
        """The Stored r of one step of global pairs, NaN across chromosomes."""
        out = np.full(len(a), np.nan)
        which = chromosome_index(self._stops, a)
        same = which == chromosome_index(self._stops, b)
        for index in np.unique(which[same]):
            c = self._chromosome(self._names[index])
            at = np.flatnonzero(same & (which == index))
            out[at] = self._local_r(c, local_rows(a[at], c.start), local_rows(b[at], c.start))
        return out

    def region_r(self, start: int, stop: int) -> npt.NDArray[np.float64]:
        """The square matrix of Stored r among the contiguous rows [start, stop) of one
        chromosome, in row order (spec §8, §10; ogld#14): the read `region` makes.

        Each row's Near Band pairs inside the region are one contiguous slice of
        `near/<c>/r`, from the row's first code to its last partner before `stop`, never a
        gather pair by pair, and never the row's partners past the region. The region's pairs
        beyond near_bp come from its rows' Proxy Index rows, each entry inside the region
        scattered into the matrix. The diagonal is 1 and every unstored pair is NaN. A span
        that leaves its chromosome or the Variant Set raises; start == stop is 0 × 0.

        Memory: the n × n float64 result, and one block of rows at a time, at most
        `_PAIRS_PER_STEP` codes, with their index rows.
        """
        c = self._span(start, stop)
        out = np.full((stop - start, stop - start), np.nan)
        np.fill_diagonal(out, 1.0)
        if c is None:
            return out
        local = local_rows(np.arange(start, stop, dtype=np.int64), c.start)
        # Each row's partners inside the region and within near_bp: its stop among the
        # region's own positions, so a row's slice ends at the region's last row.
        length = stop_within(c.position[local], self._provenance["near_bp"]) - (
            np.arange(len(local)) + 1
        )
        first = c.near_indptr[local]
        for a, b in blocks((indptr_of(length), _PAIRS_PER_STEP)):
            for i, codes in enumerate(self._runs(c.near_r, first[a:b], length[a:b]), start=a):
                r = dequantise(codes)
                out[i, i + 1 : i + 1 + len(r)] = r
                out[i + 1 : i + 1 + len(r), i] = r
        self._scatter_far(c, _Placed(local, np.arange(len(local), dtype=np.int64), out))
        return out

    def index(self, rows: npt.NDArray[np.int64]) -> Partners:
        """Each row's Proxy Index entries: every partner within far_bp, either side, with
        r̂² ≥ t_far, and its Stored r (spec §5). One sparse row per variant (§10). The stored
        part as it is: no query calls it, `partners` chooses it."""
        return self._gather(rows, (self._index_entries, _index_length))

    def band(self, rows: npt.NDArray[np.int64]) -> Partners:
        """Each row's Near Band partners on both sides, with their Stored r (spec §4, §10).

        The downstream half is the row; the upstream half is a column, one element in
        each row of the preceding near_bp. It is the slow read the Proxy Index exists to
        avoid. The stored part as it is: no query calls it, `partners` chooses it.
        """
        near_bp = self._provenance["near_bp"]
        part = (partial(self._band_entries, reach=near_bp), partial(_window_rows, reach=near_bp))
        return self._gather(rows, part)

    # ------------------------------------------------------------ resolving pairs

    def _local_r(
        self, c: _Chromosome, a: npt.NDArray[np.int64], b: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float64]:
        """The Stored r of the local pairs (a[k], b[k]) of one chromosome, NaN unstored."""
        out = np.full(len(a), np.nan)
        for s, e in windows(0, len(a), _PAIRS_PER_STEP):
            lo, hi = np.minimum(a[s:e], b[s:e]), np.maximum(a[s:e], b[s:e])
            out[s:e] = self._resolved(c, lo, hi)
        return out

    def _resolved(
        self, c: _Chromosome, lo: npt.NDArray[np.int64], hi: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float64]:
        """The Stored r of local pairs lo ≤ hi (spec §8, Figure 2): 1 for a row with itself,
        the Near Band within near_bp, the earlier row's Proxy Index row out to far_bp, and
        NaN for the Bound and Not Measured."""
        p = self._provenance
        gap = c.position[hi] - c.position[lo]
        near = within(gap, p["near_bp"]) & (lo != hi)
        far = np.flatnonzero(~within(gap, p["near_bp"]) & within(gap, p["far_bp"]))
        r = np.where(lo == hi, 1.0, np.nan)
        codes = self._read(c.near_r, near_address(c.near_indptr, lo[near], hi[near]))
        r[near] = dequantise(codes)
        if len(far):
            held, codes = self._index_codes(c, lo[far], hi[far])
            r[far[held]] = dequantise(codes[held])
        return r

    def _near_square(self, c: _Chromosome, placed: _Placed) -> None:
        """Write the Stored r of every pair of the placed rows within near_bp into the square."""
        local, at, out = placed.local, placed.at, placed.out
        near = stop_within(c.position[local], self._provenance["near_bp"])
        first = np.arange(1, len(local) + 1, dtype=np.int64)
        for i, j in row_pairs(first, near, _PAIRS_PER_STEP):
            r = self._resolved(c, local[i], local[j])
            out[at[i], at[j]] = r
            out[at[j], at[i]] = r

    def _scatter_far(self, c: _Chromosome, placed: _Placed) -> None:
        """Write the Far Pairs among the placed rows into the square: each earlier row's
        Proxy Index entries that are later placed rows beyond near_bp. Pairs beyond near_bp
        that the index lacks stay NaN: the Bound, or Not Measured."""
        local, at, out = placed.local, placed.at, placed.out
        near_bp = self._provenance["near_bp"]
        position = c.position[local]
        if len(local) == 0 or within(position[-1] - position[0], near_bp):
            return
        length = c.index_indptr[local + 1] - c.index_indptr[local]
        for a, b in blocks((indptr_of(length), _PAIRS_PER_STEP)):
            partner, codes, n = self._index_rows(c, local[a:b])
            owner = owners(n, np.arange(a, b, dtype=np.int64))
            j = np.minimum(np.searchsorted(local, partner), len(local) - 1)
            keep = (local[j] == partner) & (j > owner)
            keep &= ~within(position[j] - position[owner], near_bp)
            r = dequantise(codes[keep])
            out[at[owner[keep]], at[j[keep]]] = r
            out[at[j[keep]], at[owner[keep]]] = r

    def _index_codes(
        self, c: _Chromosome, lo: npt.NDArray[np.int64], hi: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[Any]]:
        """For local pairs lo < hi beyond near_bp: whether the Proxy Index holds each, as
        the Far Pair it then is, and its code there; read from each lo's index row."""
        held_by = np.unique(lo)
        partner, codes, length = self._index_rows(c, held_by)
        size = c.stop - c.start
        keys = owners(length, held_by) * size + partner
        order = np.argsort(keys, kind="stable")
        keys, codes = keys[order], codes[order]
        wanted = lo * size + hi
        if len(keys) == 0:
            return np.full(len(wanted), False), np.full(len(wanted), RESERVED_CODE, codes.dtype)
        at = np.minimum(np.searchsorted(keys, wanted), len(keys) - 1)
        return keys[at] == wanted, codes[at]

    def _index_rows(
        self, c: _Chromosome, local: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[Any], npt.NDArray[np.int64]]:
        """The Proxy Index rows of `local` rows, back to back: partners, codes, row lengths."""
        first = c.index_indptr[local]
        length = c.index_indptr[local + 1] - first
        coords = spans(first, length)
        partner = self._read(c.index_partner, coords).astype(np.int64)
        return partner, self._read(c.index_r, coords), length

    # ------------------------------------------------------------ partners

    def _index_entries(self, c: _Chromosome, local: npt.NDArray[np.int64]) -> _RowEntries:
        """Each local row's Proxy Index row: lengths, partners and Stored r (spec §5)."""
        partner, codes, length = self._index_rows(c, local)
        return length, partner, dequantise(codes)

    def _band_entries(
        self, c: _Chromosome, local: npt.NDArray[np.int64], *, reach: int
    ) -> _RowEntries:
        """Each local row's Near Band partners within `reach` ≤ near_bp on both sides: the
        row up to `reach`, and the column, one element in each earlier row within it."""
        lo, hi = position_rows(c.position, c.position[local] - reach, c.position[local] + reach)
        # Each row's span runs from its first upstream partner to its last downstream one,
        # the row itself included once; dropping it leaves both halves, ascending.
        span, owner = spans(lo, hi - lo), owners(hi - lo, local)
        partner, owner = span[span != owner], owner[span != owner]
        lo_pair, hi_pair = np.minimum(owner, partner), np.maximum(owner, partner)
        codes = self._read(c.near_r, near_address(c.near_indptr, lo_pair, hi_pair))
        return hi - lo - 1, partner, dequantise(codes)

    def _listed_entries(
        self,
        c: _Chromosome,
        local: npt.NDArray[np.int64],
        *,
        reach: int,
        listed: npt.NDArray[np.int64],
    ) -> _RowEntries:
        """Each local row's pairs with the `listed` rows within `reach` ≤ near_bp, both sides,
        resolved one Near Band element each: one row against many, and nothing else."""
        lo, hi = position_rows(c.position, c.position[local] - reach, c.position[local] + reach)
        a = np.searchsorted(listed, global_rows(lo, c.start))
        b = np.searchsorted(listed, global_rows(hi, c.start))
        partner = local_rows(listed[spans(a, b - a)], c.start)
        which = owners(b - a)
        other = partner != local[which]
        partner, which = partner[other], which[other]
        length = np.bincount(which, minlength=len(local)).astype(np.int64)
        return length, partner, self._local_r(c, local[which], partner)

    def _blocks(
        self,
        c: _Chromosome,
        local: npt.NDArray[np.int64],
        part: tuple[_Read, _Estimate],
        judge: _Judge | None,
    ) -> Iterator[tuple[int, int, _Kept]]:
        """The entries a part's read gives `local` rows, a block of rows [a, b) at a time, by
        the part's estimate of each row's entries: each block's partners as global rows, with
        Stored r and distance, and kept only where `judge` says, when one is given. So what is
        held before it is judged stays near `_PAIRS_PER_STEP` entries, and no block is joined
        to another."""
        read, estimate = part
        for a, b in blocks((indptr_of(estimate(c, local)), _PAIRS_PER_STEP)):
            length, partner, r = read(c, local[a:b])
            gap = c.position[partner] - c.position[owners(length, local[a:b])]
            np.abs(gap, out=gap)
            partner = global_rows(partner, c.start)
            if judge is not None:
                kept = judge(gap, partner, r)
                length = np.bincount(owners(length)[kept], minlength=b - a).astype(np.int64)
                partner, r, gap = partner[kept], r[kept], gap[kept]
            yield a, b, (length, partner, r, gap)

    def _gather(
        self,
        rows: npt.NDArray[np.int64],
        part: tuple[_Read, _Estimate],
        judge: _Judge | None = None,
    ) -> Partners:
        """The partners a part's read gives each of `rows`, in CSR form in the order asked.

        Each block is held as read until every row's length is known, then put in its place
        in the result and let go: the kept entries are held twice only while they are put in
        order (see `partners`)."""
        rows = self._checked(rows)
        length = np.empty(len(rows), dtype=np.int64)
        found: list[tuple[npt.NDArray[np.int64], _Kept]] = []
        for c, members in self._by_chromosome(rows):
            for a, b, kept in self._blocks(c, local_rows(rows[members], c.start), part, judge):
                length[members[a:b]] = kept[0]
                found.append((members[a:b], kept))
        return _placed_in_order(length, found)

    def _listed(self, among: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        """`among`, if it is ascending, distinct rows of the Variant Set."""
        listed = self._checked(among)
        if (np.diff(listed) <= 0).any():
            raise ValueError("among must be ascending, distinct Variant Set rows")
        return listed

    def _span(self, start: int, stop: int) -> _Chromosome | None:
        """The chromosome holding the rows [start, stop), or None when the span is empty."""
        size = int(self._stops[-1]) if len(self._stops) else 0
        if not 0 <= start <= stop <= size:
            raise ValueError(f"rows [{start}, {stop}) are not a span of the {size}-row Variant Set")
        if start == stop:
            return None
        first, last = chromosome_index(self._stops, np.array([start, stop - 1]))
        if first != last:
            raise ValueError(f"rows [{start}, {stop}) leave their chromosome: a region is on one")
        return self._chromosome(self._names[int(first)])

    def _runs(
        self, array: zarr.Array[Any], first: npt.NDArray[np.int64], length: npt.NDArray[np.int64]
    ) -> list[npt.NDArray[Any]]:
        """The ragged runs of `array` (spec §10), each one slice, through the chunk cache;
        a lost chunk refused."""
        return [_refused_if_lost(array, run) for run in self._cache.runs(array, first, length)]

    def _chromosome(self, name: str) -> _Chromosome:
        """One chromosome's kept arrays, read on first use (spec §10). Each of its five arrays'
        fill values is judged before any element of the chromosome is decoded."""
        if name not in self._loaded:
            start, stop = self._ranges[name]
            near_indptr, near_r, index_indptr, index_partner, index_r = (
                _kept_array(self._root, f"{kind}/{name}/{array}")
                for kind, array in _CHROMOSOME_ARRAYS
            )
            position = self._columns["position"]
            self._loaded[name] = _Chromosome(
                name,
                start,
                stop,
                _refused_if_lost(position, np.asarray(position[start:stop])).astype(np.int64),
                _indptr(near_indptr),
                near_r,
                _indptr(index_indptr),
                index_partner,
                index_r,
            )
        return self._loaded[name]

    def _by_chromosome(
        self, rows: npt.NDArray[np.int64]
    ) -> Iterator[tuple[_Chromosome, npt.NDArray[np.int64]]]:
        """Each chromosome `rows` touch, with the positions in `rows` that are on it."""
        which = chromosome_index(self._stops, rows)
        for index in np.unique(which):
            yield self._chromosome(self._names[index]), np.flatnonzero(which == index)

    def _strings(self, name: str, rows: npt.NDArray[np.int64]) -> npt.NDArray[np.str_]:
        """The string column `name` at `rows`; an empty string is a lost chunk's, never data."""
        # zarr hands back numpy StringDType, which casts to a fixed width only via a list.
        values = np.array(self._read(self._columns[name], rows).tolist(), dtype=np.str_)
        return values

    def _read(self, array: zarr.Array[Any], coords: npt.NDArray[np.int64]) -> npt.NDArray[Any]:
        """The elements of `array` at `coords`, through the chunk cache; a lost chunk refused."""
        return _refused_if_lost(array, self._cache.take(array, coords))

    def _checked(self, rows: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        """`rows`, if every one is a row of the Variant Set. An Absent ALID is −1 from `rows_of`
        and is no row: read as an index it would be the last variant."""
        rows = np.asarray(rows, dtype=np.int64)
        size = int(self._stops[-1]) if len(self._stops) else 0
        bad = (rows < 0) | (rows >= size)
        if bad.any():
            raise ValueError(
                f"rows {rows[bad][:5].tolist()} are outside the Variant Set of "
                f"{size} rows: an Absent ALID is not a row (spec §8)"
            )
        return rows


#: What a stored part's read gives some rows: row lengths, partners (local rows), Stored r.
_RowEntries = tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]
#: A block's kept entries: row lengths, partners (global rows), Stored r and distance.
_Kept = tuple[
    npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.int64]
]
#: Which of a block's entries `partners` keeps, from their distance, partner and Stored r.
_Judge = Callable[
    [npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]], npt.NDArray[np.bool_]
]
#: A stored part's read of some local rows of one chromosome, and its estimate of each row's
#: entries before they are judged, which sizes the blocks `partners` reads.
_Read = Callable[["_Chromosome", npt.NDArray[np.int64]], _RowEntries]
_Estimate = Callable[["_Chromosome", npt.NDArray[np.int64]], npt.NDArray[np.int64]]


def _index_length(c: _Chromosome, local: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Each local row's Proxy Index entries."""
    return c.index_indptr[local + 1] - c.index_indptr[local]


def _window_rows(
    c: _Chromosome, local: npt.NDArray[np.int64], *, reach: int
) -> npt.NDArray[np.int64]:
    """How many rows lie within `reach` of each local row, on both sides, itself included."""
    lo, hi = position_rows(c.position, c.position[local] - reach, c.position[local] + reach)
    return hi - lo


def _placed_in_order(
    length: npt.NDArray[np.int64], found: list[tuple[npt.NDArray[np.int64], _Kept]]
) -> Partners:
    """The CSR result of rows of `length` entries, from blocks of rows held in `found`: each
    block is copied to its place, then dropped, so it is not held beside the whole result."""
    indptr = indptr_of(length)
    partner = np.empty(int(indptr[-1]), dtype=np.int64)
    r = np.empty(int(indptr[-1]), dtype=np.float64)
    distance = np.empty(int(indptr[-1]), dtype=np.int64)
    while found:
        members, (lengths, partners, values, gaps) = found.pop()
        where = spans(indptr[members], lengths)
        partner[where], r[where], distance[where] = partners, values, gaps
    return Partners(indptr, partner, r, distance)


def _member(rows: npt.NDArray[np.int64], listed: npt.NDArray[np.int64]) -> npt.NDArray[np.bool_]:
    """Whether each of `rows` is in `listed`, which is ascending and distinct."""
    at = np.searchsorted(listed, rows)
    found = at < len(listed)
    found[found] = listed[at[found]] == rows[found]
    return found


def _loci(
    alids: Sequence[str], ranges: dict[str, tuple[int, int]]
) -> dict[str, tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]:
    """Per chromosome of the store, which of `alids` name a locus on it, and its position."""
    by_chrom: dict[str, tuple[list[int], list[int]]] = {}
    for i, alid in enumerate(alids):
        locus = alid_locus(alid)
        if locus is not None and locus[0] in ranges:
            which, at = by_chrom.setdefault(locus[0], ([], []))
            which.append(i)
            at.append(locus[1])
    return {
        chrom: (np.array(which, dtype=np.int64), np.array(at, dtype=np.int64))
        for chrom, (which, at) in by_chrom.items()
    }


def _refused_if_lost(array: zarr.Array[Any], values: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """`values` read from `array`, unless one is a value no LD Store holds there, as its fill
    value is: that is a chunk the store lost, and decoding it would be inventing data."""
    lost = never_data(array.path, values)
    if lost.any():
        raise ValueError(
            f"{array.path} read {int(lost.sum())} element(s) no LD Store holds there, as its "
            f"fill value {shown_fill(array.fill_value)} is: a chunk the store lost, never data "
            "(spec §3, §6)"
        )
    return values


def _group(root: zarr.Group, path: str) -> zarr.Group:
    member = root[path]
    assert isinstance(member, zarr.Group)
    return member


def _array(root: zarr.Group, path: str) -> zarr.Array[Any]:
    member = root[path]
    assert isinstance(member, zarr.Array)
    return member


def _kept_array(root: zarr.Group, path: str) -> zarr.Array[Any]:
    """An array, refused unless a chunk it lost would read as a value no store holds there."""
    array = _array(root, path)
    if not never_data(path, array.fill_value).all():
        raise ValueError(
            f"{path} has the fill value {shown_fill(array.fill_value)}, which valid data can hold "
            "there: a chunk it lost would read as data (spec §6, envelope.fill)"
        )
    return array


def _indptr(array: zarr.Array[Any]) -> npt.NDArray[np.int64]:
    return _refused_if_lost(array, np.asarray(array[:]).astype(np.int64))
