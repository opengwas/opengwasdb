"""Reading an LD Store: the seam every query is built on (spec §1–§6, §10; ogld#4).

`StoreReader` is the store layer's whole read interface. It hides Zarr, the chunk and
shard shapes, chromosome-local rows and the codes, which it decodes at the scale the store
records (`ogld.format.REncoding`; ADR 0004): callers pass and receive global Variant Set rows
and Stored r as float64. Every unstored pair it reports is NaN, never a
number, and the meaning of a NaN is the format's to say (`ogld.format.answer_kinds`).

Reads are bounded (spec §10). Opening reads the provenance and the chromosome ranges, and
nothing else. A chromosome's positions and its Near Band's `indptr` are read on first touch;
its Proxy Index's `indptr` only when the index is used, and its Far Pairs' `indptr` when a
request needs pairs beyond near_bp. Each is kept. Every other read goes through `ChunkCache`,
which fetches only the inner chunks holding the elements a request needs, from disk or over
HTTP alike, and keeps them, so a query that reads a few pairs at a time fetches each once.

A chunk the store lost reads as its array's fill value, and spec §6 chose each fill value
as one a valid LD Store never holds. So every read refuses that value, naming the array,
rather than decode it: a lost Near Band chunk is a refusal, never a run of r = 0.

From format 0.2.0 a variant's Partner Rows row is a read of its own (`partner_row`; partner-rows
§7): one frame's byte range out of `partners/<chrom>/frames`, one decode, and the row's slice of
the decoded frame, which the reader's frame cache keeps. It reads no Near Band chunk, no Proxy
Index entry and no shard index, and a store that holds no Partner Rows is refused naming the
component.
"""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import cached_property, partial
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
import zarr

from ogld.format import READABLE_FORMATS, Codes, InputError, Provenance, REncoding, judge_window
from ogld.format.geometry import (
    blocks,
    chromosome_index,
    global_rows,
    indptr_of,
    local_rows,
    near_address,
    near_row_lengths,
    owners,
    position_rows,
    row_pairs,
    spans,
    stop_within,
    windows,
    within,
)
from ogld.identity import alid_locus
from ogld.store import layout
from ogld.store.frames import FrameCache, PartnerFrames, open_partner_frames
from ogld.store.layout import chromosomes_listed, never_data, refused_if_lost, shown_fill
from ogld.store.location import ChunkCache, open_group
from ogld.store.tables import Partners, Variants

#: Each chromosome's arrays a request touching it opens (spec §1), in the order `_Chromosome`
#: takes them. Its Far Pairs are opened apart, when a request first needs them (`_far_pairs`).
_CHROMOSOME_ARRAYS: Final = (
    ("near", "indptr"),
    ("near", "r"),
    ("index", "indptr"),
    ("index", "partner"),
    ("index", "r"),
)
#: A sparse component's arrays, `index/<chrom>/` and `far/<chrom>/` alike (spec §5).
_SPARSE_ARRAYS: Final = ("indptr", "partner", "r")
#: The variant table's columns (spec §2).
_VARIANT_COLUMNS: Final = ("position", "alid", "a1", "a2", "a1_frequency")
#: The pairs one step of `pair_r` resolves at once, which bounds its working memory.
_PAIRS_PER_STEP: Final = 1 << 22
#: The most component threads one request runs (partner-rows §9.2). Four is where the prototype was
#: fastest (#65: 1.051 s at G = 8 and 1.067 s at G = 16); eight is slower, 5.753 s and 5.109 s,
#: because a component's per-lead Python work holds the GIL while only the reads and the decode
#: release it. The pool's size is the reader's, so no query signature carries it.
COMPONENT_WORKERS_MAX: Final = 4
#: How long a list of ALIDs makes the index worth reading: a `_INDEX_SHARE`th of the Variant Set's
#: entries or more (ogld#42). The whole index costs about what the exact scan's ALID string chunks
#: do at about 4,000 of EUR's 8.59 M ALIDs, and below that a list's loci touch fewer chunks than the
#: index holds; `docs/benchmarks/alid-index.md` has both sides of it. A small store's list is a
#: large share of it, so a test store's lookups read the index.
_INDEX_SHARE: Final = 2_150


def _index_pays(count: int, entries: int) -> bool:
    """Whether `count` keys are worth reading the whole index of `entries` (spec §10; ADR 0010):
    a list naming a `_INDEX_SHARE`th of them or more, and never fewer than one."""
    return count >= max(1, entries // _INDEX_SHARE)


@dataclass(frozen=True)
class _Sparse:
    """A sparse component: handles opened and fill-checked together, indptr loaded only on
    use. Below t_far, clumping must not pay to decode an index it never reads (ogld#40)."""

    source: zarr.Array[Any]
    cache: ChunkCache
    partner: zarr.Array[Any]
    r: zarr.Array[Any]

    @cached_property
    def indptr(self) -> npt.NDArray[np.int64]:
        return _indptr(self.source, self.cache)


@dataclass(frozen=True)
class _Loci:
    """One chromosome's Variant Set rows and positions, and nothing of its LD components (#71).

    A query that reads only the Partner Rows still needs each hit's chromosome, position and ALID
    to order it and measure distances, and must not read `near/<chrom>/indptr` or open the Proxy
    Index's arrays to get them (partner-rows §8; ld-store §10). This is that much, from
    `variants/position` alone; `_Chromosome` is it plus the Near Band and the Proxy Index, for the
    reads that resolve stored pairs.
    """

    name: str
    start: int
    stop: int
    position: npt.NDArray[np.int64]


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
    index: _Sparse


@dataclass(frozen=True)
class _AlidIndex:
    """The ALID search index (spec §2; ADR 0010): one NUL-padded 32-byte key an entry, sorted, and
    the Variant Set row each entry names. Keys are bytes, so a search compares bytes."""

    alid: zarr.Array[Any]
    row: zarr.Array[Any]


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
    it, before anything is decoded. Every code is decoded with the store's own encoding, so
    one process reads stores of different scales side by side.

    A reader is not fork-safe and is not inherited across `fork`: it holds locks and, once a
    chromosome's Partner Rows are read, a chunk file's descriptor. A process worker opens its own
    reader (partner-rows §9; ogld#71).
    """

    _root: zarr.Group
    _provenance: Provenance
    _encoding: REncoding
    _ranges: dict[str, tuple[int, int]]
    _names: list[str]
    _stops: npt.NDArray[np.int64]
    _loaded: dict[str, _Chromosome]
    _located: dict[str, _Loci]
    _far: dict[str, _Sparse]
    _cache: ChunkCache
    _frame_cache: FrameCache
    _partner_frames: dict[str, PartnerFrames]
    _loci_lock: threading.Lock
    _partner_lock: threading.Lock
    _query_workers: int
    _closed: bool
    _columns: dict[str, zarr.Array[Any]]
    _index_arrays: _AlidIndex | None

    def _attach(
        self,
        root: zarr.Group,
        provenance: Provenance,
        encoding: REncoding,
        cache: ChunkCache,
        frames: FrameCache,
    ) -> None:
        """Keep the provenance, the encoding and the chromosome ranges of an opened, checked
        store, and the two caches its reads go through: decoded Near Band chunks, and decoded
        Partner Rows frames (partner-rows §7), and the component threads it may run (§9.2)."""
        self._root = root
        self._provenance = provenance
        self._encoding = encoding
        self._cache = cache
        self._frame_cache = frames
        self._query_workers = COMPONENT_WORKERS_MAX
        listed = chromosomes_listed(_group(root, "variants").attrs["chromosomes"])
        self._ranges = {name: (start, stop) for name, start, stop in listed}
        self._names = list(self._ranges)
        self._stops = np.array([stop for _, stop in self._ranges.values()], dtype=np.int64)
        self._loaded = {}
        self._located = {}
        self._far = {}
        self._partner_frames = {}
        # A chromosome's first use is one load, however many threads make it: its positions are
        # read once and both caches hold that one array (#71), and its Partner Rows' offsets chunk
        # is read once and one chunk-file descriptor is opened (partner-rows §7).
        self._loci_lock = threading.Lock()
        self._partner_lock = threading.Lock()
        self._closed = False
        # Each handle is opened once: opening one reads its metadata document, which over
        # HTTP is a request.
        self._columns = {name: _kept_array(root, f"variants/{name}") for name in _VARIANT_COLUMNS}
        self._index_arrays = None

    @classmethod
    def open(
        cls,
        location: str | os.PathLike[str],
        *,
        cache_bytes: int | None = None,
        frame_cache_bytes: int | None = None,
    ) -> StoreReader:
        """Open the LD Store at a local path or an HTTP(S) URL (ogld#4), keeping at most
        `cache_bytes` of its Near Band chunks (spec §10; the default if None, ogld#32) and
        `frame_cache_bytes` of its decoded Partner Rows frames (partner-rows §7; its own default
        if None). The two are separate budgets, because the two reads serve different queries."""
        cache = ChunkCache(cache_bytes)
        frame_cache = FrameCache(frame_cache_bytes)
        root = open_group(location, cache=cache)
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
        if recorded.get("format_version") not in READABLE_FORMATS:
            raise InputError(
                f"{location} is LD Store format {recorded.get('format_version')!r}; "
                f"this ogld reads {' and '.join(READABLE_FORMATS)} only (spec §13)"
            )
        try:
            encoding = REncoding.of(recorded.get("encoding"))
        except InputError as unimplemented:
            raise InputError(f"{location}: {unimplemented}") from unimplemented
        reader = cls()
        reader._attach(root, cast(Provenance, recorded), encoding, cache, frame_cache)
        return reader

    @property
    def cache_bytes(self) -> int:
        """The bytes of chunks and shard indexes this reader keeps at most (spec §10)."""
        return self._cache.budget

    @property
    def frame_cache_bytes(self) -> int:
        """The bytes of decoded Partner Rows frames this reader keeps at most (partner-rows §7)."""
        return self._frame_cache.budget

    def _set_query_workers(self, requested: int) -> None:
        """Set the threads one `clump` may run its components on (partner-rows §9.2).

        The size of that pool is the reader's, not a caller's setting: this is the private seam
        #71's tests use, and it holds the cap, so asking for eight gives `COMPONENT_WORKERS_MAX`.
        """
        if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
            raise InputError(
                f"the component thread count is a whole number of at least 1: {requested!r}"
            )
        self._query_workers = min(requested, COMPONENT_WORKERS_MAX)

    def close(self) -> None:
        """Close the reader: after it returns no Partner Rows read is answered, and every chunk file
        it kept open is closed (partner-rows §7).

        A first use that is racing `close` cannot publish a source that outlives it: it holds the
        chromosome lock `close` waits on and re-checks the closed state under it, so either the
        close closes what was published, or the first use finds the reader closed and refuses. A
        read already in flight finishes on the file it opened, because a source closes only after
        its reads have finished (`PreadSource.close`). Closing twice is closing once.
        """
        with self._partner_lock:
            # Marked closed and emptied under the same lock a first use takes, so no first use can
            # publish a source between the two: one that runs after this sees the closed reader and
            # refuses, and one that ran before is in the list closed below.
            self._closed = True
            frames, self._partner_frames = list(self._partner_frames.values()), {}
        for partner in frames:
            partner.close()

    def __enter__(self) -> StoreReader:
        return self

    def __exit__(self, kind: object, value: object, traceback: object) -> None:
        self.close()

    @property
    def provenance(self) -> Provenance:
        """The root group's `ld_store` attribute, as recorded (spec §7)."""
        return self._provenance

    @property
    def encoding(self) -> REncoding:
        """The encoding the store records, which every code it reads is decoded with (§3)."""
        return self._encoding

    def rows_of(self, alids: Sequence[str]) -> npt.NDArray[np.int64]:
        """The Variant Set row of each ALID, in order; −1 where the ALID is Absent (spec §2).

        A malformed ALID is Absent too: it names no variant in the store. A long list is searched
        for in the store's ALID index (ADR 0010), which holds every ALID of the table that fits its
        32-byte slot, sorted, so one search answers it without reading the variant table at all.
        Every ALID the index does not hold — absent, too wide for the slot, or one a broken build
        left out — takes the exact scan at its own position, which is what found it before the
        index, and a list too short for the index to pay for takes it for every ALID. So the index
        can only make this read faster, never change its answer; and a store whose index could
        answer falsely is one validation fails, as it fails a store with any other broken array.
        """
        found = np.full(len(alids), -1, dtype=np.int64)
        if not len(alids):
            return found
        loci = _loci(alids, self._ranges)
        if not loci:
            return found
        scanned = np.ones(len(alids), dtype=bool)
        self._from_index(alids, loci, found, scanned)
        rest = np.flatnonzero(scanned)
        if len(rest):
            found[rest] = self._scanned(alids, rest)
        return found

    def _from_index(
        self,
        alids: Sequence[str],
        loci: dict[str, tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]],
        found: npt.NDArray[np.int64],
        scanned: npt.NDArray[np.bool_],
    ) -> None:
        """Fill `found` from the ALID index for every ALID it can answer, and clear `scanned`
        there; what it cannot answer stays for the exact scan (spec §2, §10; ADR 0010).

        Which ALIDs the index holds a key for is `layout.alid_fits`' question, not this method's:
        an ALID wider than the slot has no key and is scanned, and a store whose key for one row
        is another row's — truncated into the slot, or invented — is one validation fails
        (`alid_index.keys`). A list too short to pay for the index asks for nothing.
        """
        asked = [i for which, _ in loci.values() for i in which.tolist()]
        keys, fitting = layout.alid_keys([alids[i] for i in asked])
        if not len(keys) or not _index_pays(len(keys), self._entries()):
            return
        hits = self._searched(keys)
        where = hits >= 0
        found[np.array(asked, dtype=np.int64)[fitting[where]]] = self._read(
            self._index().row, hits[where]
        )
        scanned[np.array(asked, dtype=np.int64)[fitting[where]]] = False

    def _entries(self) -> int:
        """How many entries the Variant Set's index holds: its rows less the ALIDs too wide for its
        slot, which the store's provenance counts (spec §7). A store written before the index
        records no count and has no index to read either, so a search refuses it by name."""
        counts = self._provenance["counts"]
        return int(self._provenance["n_variants"]) - int(counts.get("alids_left_out", 0))

    def _scanned(self, alids: Sequence[str], which: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        """The rows of the ALIDs at `which`, in that order: the exact scan of the variant table
        at each one's own position (spec §2), which is how `rows_of` found every ALID before the
        index. Only the ALID strings at those positions are read, in one read per chromosome."""
        found = np.full(len(which), -1, dtype=np.int64)
        chosen = [alids[int(i)] for i in which]
        asked: list[int] = []
        candidates: list[npt.NDArray[np.int64]] = []
        for chrom, (at, positions) in _loci(chosen, self._ranges).items():
            c = self._chromosome_loci(chrom)
            lo, hi = position_rows(c.position, positions, positions)
            asked.extend(owners(hi - lo, at).tolist())
            candidates.append(global_rows(spans(lo, hi - lo), c.start))
        if not asked:
            return found
        rows = np.concatenate(candidates)
        names = self._strings("alid", rows)
        query = np.array([chosen[i] for i in asked], dtype=np.str_)
        hit = names == query
        found[np.array(asked, dtype=np.int64)[hit]] = rows[hit]
        return found

    def _searched(self, keys: npt.NDArray[np.uint8]) -> npt.NDArray[np.int64]:
        """Where each key's entry lies in the ALID index, or −1 where the index holds none.

        The index is read one shard at a time — the unit the store itself reads and fetches — in key
        order, and each read's keys are searched for the queries that fall within them. So the table
        is never held whole beside the chunk cache: a lookup's transient is one shard of keys — 16
        MiB on EUR, released before the next is fetched — and the queries' own keys, where reading
        the array in one run holds all 273 MB of EUR's keys twice (spec §10; ogld#42's reviews).
        """
        index = self._index()
        count = int(index.row.shape[0])
        found = np.full(len(keys), -1, dtype=np.int64)
        if not count or not len(keys):
            return found
        width = layout.ALID_INDEX_WIDTH
        # One shard's keys, from the array's own shapes: a store written with other ones is read
        # one of its shards at a time too. An array with no shards reads one inner chunk.
        shard = int((index.alid.shards or index.alid.chunks)[0])
        per = max(1, shard // width)
        order = np.argsort(keys.view(f"|S{width}").ravel(), kind="stable")
        wanted = keys[order].view(f"|S{width}").ravel()
        chunk = np.empty(0, dtype=f"|S{width}")
        for first in range(0, count, per):
            # Let the shard just searched go before the next is fetched and joined: a lookup holds
            # one shard's keys, not the one before it as well (ogld#42's fix round 3).
            del chunk
            chunk = self._chunk(index, first, min(first + per, count))
            a = int(np.searchsorted(wanted, chunk[0], "left"))
            b = int(np.searchsorted(wanted, chunk[-1], "right"))
            if b <= a:
                continue
            at = np.minimum(np.searchsorted(chunk, wanted[a:b]), len(chunk) - 1)
            same = chunk[at] == wanted[a:b]
            found[order[a:b][same]] = first + at[same]
        return found

    def _chunk(self, index: _AlidIndex, first: int, stop: int) -> npt.NDArray[Any]:
        """The keys of the index's entries [first, stop) — one read, one shard at most — refused if
        one of its chunks is lost."""
        width = layout.ALID_INDEX_WIDTH
        length = np.array([(stop - first) * width])
        run = self._runs(index.alid, np.array([first * width]), length)[0]
        keys = refused_if_lost(index.alid, run)
        return keys.view(f"|S{width}").ravel()

    def _index(self) -> _AlidIndex:
        """The ALID index's arrays, opened on first use (spec §10): opening reads their
        metadata, and nothing reads them until a search does."""
        if self._index_arrays is None:
            self._index_arrays = _AlidIndex(
                _kept_array(self._root, layout.ALID_INDEX_ALID),
                _kept_array(self._root, layout.ALID_INDEX_ROW),
            )
        return self._index_arrays

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

    def chromosome_range(self, chrom: str) -> tuple[int, int]:
        """The Variant Set rows chromosome `chrom` occupies, `[start, stop)` (spec §2).

        It is what turns a Partner Rows row's chromosome-local partner into a Variant Set row
        (partner-rows §7), so a component split and a greedy can work in the store's own rows. A
        label the store does not hold is refused naming it, as `partner_row` refuses it.
        """
        try:
            return self._ranges[chrom]
        except KeyError:
            raise ValueError(
                f"no chromosome {chrom} in this LD Store, whose Variant Set is on "
                f"{', '.join(self._names)} (spec §2)"
            ) from None

    def alids_of(self, rows: npt.NDArray[np.int64]) -> npt.NDArray[np.str_]:
        """The ALID of each of `rows`, in the order given (spec §2): the one string column a
        query names its rows by, read without the others (ogld#14)."""
        return self._strings("alid", self._checked(rows))

    def loci_of(
        self, rows: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.str_], npt.NDArray[np.int64]]:
        """Chromosomes and positions, without allele columns clump never returns (ogld#40).

        It reads the variant table's positions and nothing of the LD components: `clump` and
        `proxies` call it, and from 0.2.0 their only LD read is the Partner Rows (#71). This is a
        store-layer read, not a public query or a new caller setting. Geometry and chromosome label
        dtypes stay identical to the full variant-table read.
        """
        rows = self._checked(rows)
        position = np.empty(len(rows), dtype=np.int64)
        for c, members in self._loci_by_chromosome(rows):
            position[members] = c.position[local_rows(rows[members], c.start)]
        names = np.array(self._names, dtype=np.str_)
        return names[chromosome_index(self._stops, rows)], position

    def variants(self, rows: npt.NDArray[np.int64]) -> Variants:
        """The Variant Set's columns at `rows`, in the order given (spec §2)."""
        rows = self._checked(rows)
        chrom, position = self.loci_of(rows)
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
        them; farther ones from the Far Pairs rows of the listed variants, each pair whose
        partner is listed scattered into the matrix, rather than searched for pair by pair. No
        Proxy Index entry is read (ADR 0007).
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

    def clump_warmer(
        self, rows: npt.NDArray[np.int64], *, r2: float, window_bp: float
    ) -> Callable[[np.int64], None]:
        """Warm candidate chunks together; oversized chromosomes use positional windows.

        The reader owns candidate geometry and selection of the stored part. Warming is
        optional: a missed chunk still goes through the ordinary bounded read (ADR 0009).

        **No query calls this since #71**: `clump` reads the Partner Rows and its own frame
        cache, so the Near Band's warm windows warm nothing a clump reads. It is kept, tested
        and unused until #72 removes it with the Near Band partner reads it serves.
        """
        if r2 >= self._provenance["t_far"] or not len(rows) or self.cache_bytes == 0:
            return lambda row: None
        c, _ = next(self._by_chromosome(rows))
        local = np.unique(local_rows(rows, c.start))
        reach = math.floor(window_bp)
        if self._prefetch_hits(c, local, reach):
            return lambda row: None
        return _ClumpWindows(self, c, local, reach).warm

    def _prefetch_hits(self, c: _Chromosome, local: npt.NDArray[np.int64], reach: int) -> bool:
        if len(local) < 2:
            return True
        stop = stop_within(c.position[local], reach)
        at = np.flatnonzero(stop > np.arange(len(local)) + 1)
        first = near_address(c.near_indptr, local[at], local[at + 1])
        length = local[stop[at] - 1] - local[at + 1] + 1
        return self._cache.prefetch(c.near_r, first, length)

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
        part = self._part(index=index, reach=reach, listed=listed)
        return self._gather(rows, part, partial(_judged, r2=r2, reach=reach, listed=listed))

    def _part(
        self, *, index: bool, reach: int, listed: npt.NDArray[np.int64] | None
    ) -> tuple[_Read, _Estimate]:
        """The stored part `partners` reads, with its estimate of each row's entries: the
        Proxy Index rows, the Near Band rows and columns, or the Near Band pairs with a list."""
        if index:
            return self._index_entries, _index_length
        estimate = partial(_window_rows, reach=reach)
        if listed is None:
            return partial(self._band_entries, reach=reach), estimate
        return partial(self._listed_entries, reach=reach, listed=listed), estimate

    def r_of_pairs(
        self, first: npt.NDArray[np.int64], second: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float64]:
        """The Stored r of each pair (first[k], second[k]), as one array (spec §8, §10; ogld#14).

        `first` and `second` broadcast against each other, so one row in `first` is one row
        against many. A row with itself is 1. Every pair the store does not hold, the Bound,
        Not Measured and a pair across chromosomes alike, is NaN. A pair within near_bp is one
        element of the Near Band; a pair farther apart is looked up in the Far Pairs row of
        its earlier variant (spec §8). `partners` (below t_far, with `among`) and `pair_r`
        resolve their pairs as it does, so a pair is resolved in one place.

        Memory: the result, 8 bytes a pair, and one step of at most `_PAIRS_PER_STEP` pairs
        at a time, about 60 bytes each (58 measured), with the Far Pairs rows of that step's
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
        beyond near_bp come from its rows' Far Pairs rows, each pair inside the region scattered
        into the matrix (ADR 0007). The diagonal is 1 and every unstored pair is NaN. A span
        that leaves its chromosome or the Variant Set raises; start == stop is 0 × 0.

        Memory: the n × n float64 result, and one block of rows at a time, at most
        `_PAIRS_PER_STEP` codes, with their Far Pairs rows.
        """
        c = self._span(start, stop)
        out = np.full((stop - start, stop - start), np.nan)
        np.fill_diagonal(out, 1.0)
        if c is None:
            return out
        local = local_rows(np.arange(start, stop, dtype=np.int64), c.start)
        # Each row's partners inside the region and within near_bp: its Near Band row length
        # among the region's own positions, so a row's slice ends at the region's last row.
        length = near_row_lengths(c.position[local], self._provenance["near_bp"])
        first = c.near_indptr[local]
        for a, b in blocks((indptr_of(length), _PAIRS_PER_STEP)):
            for i, codes in enumerate(self._runs(c.near_r, first[a:b], length[a:b]), start=a):
                r = self._encoding.dequantise(codes)
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

    # ------------------------------------------------------------ partner rows

    def require_partner_rows(self) -> float:
        """The Partner Threshold this store recorded (spec §7; partner-rows §3).

        From format 0.2.0 `clump` and `proxies` read the Partner Rows and nothing else, so a
        store that holds none — every 0.1.0 store — is refused here, naming the missing component
        and the way to derive it, before any read. It is never answered from the Near Band: that
        is the fallback the Partner Rows exist to replace (partner-rows §8.1, §8.4).
        """
        if "t_partner" not in self._provenance:
            raise InputError(
                f"this LD Store is format {self._provenance['format_version']}, which has no "
                f"Partner Rows ({layout.PARTNERS}/): clump and proxies read them, so it is refused "
                "naming the missing component; a store built before 0.2.0 derives them with "
                "`ogld derive-partners` (spec §13; partner-rows §5.1)"
            )
        return float(self._provenance["t_partner"])

    def partner_row(self, chrom: str, local_row: int) -> tuple[npt.NDArray[np.int32], Codes]:
        """One variant's Partner Rows row: every partner within far_bp at its distance's floor,
        each with its code, chromosome-local and ascending (partner-rows §7).

        This is the read `clump` and `proxies` make from #71: one frame's byte range, one decode,
        and the row's slice of the decoded frame, which the reader's frame cache keeps, so a row of
        a frame fetched before costs no read (§7 steps 2–3). The partners are chromosome-local rows,
        as the frame holds them; a code is the stored pair's, judged by the query's threshold and
        never turned into a number here (§4).

        A store that holds no Partner Rows — every 0.1.0 store — is refused naming the component,
        and so is a frame that fails its checksum, its k or its payload length, naming the
        chromosome and the frame. A row is never answered from a damaged frame, nor a short one.
        """
        return self._frames(chrom).row(local_row)

    def partner_rows(
        self, chrom: str, local_rows: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int32], Codes]:
        """Each of `local_rows`' Partner Rows rows, in the order asked, as (indptr, partners,
        codes) (partner-rows §7): row i's entries are `partners[indptr[i]:indptr[i + 1]]`.

        A request for several rows coalesces only frames that are physically adjacent, one
        byte-range read for the run, and never merges frames with bytes between them (§9.2). Rows a
        frame already served cost no read, and a row asked for twice is answered twice.
        """
        return self._frames(chrom).rows_of(local_rows)

    def _frames(self, chrom: str) -> PartnerFrames:
        """One chromosome's Partner Rows, opened on first use (spec §10; partner-rows §7).

        Opening one reads its group's attributes, its `frames_idx` chunk and its `frames` metadata:
        a request that never asks for a Partner Rows row — `matrix`, `region` — reads none of them.
        """
        if self._closed:
            raise ValueError(
                "this reader is closed: its Partner Rows reads are over (partner-rows §7)"
            )
        frames = self._partner_frames.get(chrom)
        if frames is not None:
            return frames
        if chrom not in self._ranges:
            raise ValueError(
                f"no chromosome {chrom} in this LD Store, whose Variant Set is on "
                f"{', '.join(self._names)} (spec §2)"
            )
        # A 0.1.0 store records none of the Partner Rows' keys, because it holds no Partner Rows
        # (spec §13): the read is refused naming the component, never answered from another one.
        self.require_partner_rows()
        with self._partner_lock:
            if self._closed:
                raise ValueError(
                    "this reader is closed: its Partner Rows reads are over (partner-rows §7)"
                )
            frames = self._partner_frames.get(chrom)
            if frames is None:
                frames = open_partner_frames(self._root, chrom, self._frame_cache)
                self._partner_frames[chrom] = frames
        return frames

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
        the Near Band within near_bp, the earlier row's Far Pairs row out to far_bp, and NaN
        for the Bound and Not Measured."""
        p = self._provenance
        gap = c.position[hi] - c.position[lo]
        near = within(gap, p["near_bp"]) & (lo != hi)
        far = np.flatnonzero(~within(gap, p["near_bp"]) & within(gap, p["far_bp"]))
        r = np.where(lo == hi, 1.0, np.nan)
        codes = self._read(c.near_r, near_address(c.near_indptr, lo[near], hi[near]))
        r[near] = self._encoding.dequantise(codes)
        if len(far):
            held, codes = self._far_codes(c, lo[far], hi[far])
            r[far[held]] = self._encoding.dequantise(codes[held])
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
        """Write the Far Pairs among the placed rows into the square: each placed row's Far
        Pairs whose partner is placed. Pairs beyond near_bp that the Far Pairs lack stay NaN:
        the Bound, or Not Measured. A Far Pair never answers a pair within near_bp, which is
        the Near Band's (spec §8)."""
        local, at, out = placed.local, placed.at, placed.out
        near_bp = self._provenance["near_bp"]
        position = c.position[local]
        if len(local) == 0 or within(position[-1] - position[0], near_bp):
            return
        far = self._far_pairs(c)
        length = far.indptr[local + 1] - far.indptr[local]
        for a, b in blocks((indptr_of(length), _PAIRS_PER_STEP)):
            partner, codes, n = self._rows(far, local[a:b])
            owner = owners(n, np.arange(a, b, dtype=np.int64))
            j = np.minimum(np.searchsorted(local, partner), len(local) - 1)
            keep = (local[j] == partner) & ~within(position[j] - position[owner], near_bp)
            r = self._encoding.dequantise(codes[keep])
            out[at[owner[keep]], at[j[keep]]] = r
            out[at[j[keep]], at[owner[keep]]] = r

    def _far_codes(
        self, c: _Chromosome, lo: npt.NDArray[np.int64], hi: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[Any]]:
        """For local pairs lo < hi beyond near_bp: whether each is a Far Pair, and its code;
        read from each lo's Far Pairs row (spec §8)."""
        held_by = np.unique(lo)
        partner, codes, length = self._rows(self._far_pairs(c), held_by)
        size = c.stop - c.start
        keys = owners(length, held_by) * size + partner
        order = np.argsort(keys, kind="stable")
        keys, codes = keys[order], codes[order]
        wanted = lo * size + hi
        if len(keys) == 0:
            reserved = self._encoding.reserved_code
            return np.full(len(wanted), False), np.full(len(wanted), reserved, codes.dtype)
        at = np.minimum(np.searchsorted(keys, wanted), len(keys) - 1)
        return keys[at] == wanted, codes[at]

    def _rows(
        self, part: _Sparse, local: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[Any], npt.NDArray[np.int64]]:
        """The rows of `local` rows in a sparse component, back to back: partners, codes, row
        lengths."""
        first = part.indptr[local]
        length = part.indptr[local + 1] - first
        coords = spans(first, length)
        partner = self._read(part.partner, coords).astype(np.int64)
        return partner, self._read(part.r, coords), length

    # ------------------------------------------------------------ partners

    def _index_entries(self, c: _Chromosome, local: npt.NDArray[np.int64]) -> _RowEntries:
        """Each local row's Proxy Index row: lengths, partners and Stored r (spec §5)."""
        partner, codes, length = self._rows(c.index, local)
        return length, partner, self._encoding.dequantise(codes)

    def _band_entries(
        self, c: _Chromosome, local: npt.NDArray[np.int64], *, reach: int
    ) -> _RowEntries:
        """Each local row's Near Band partners within `reach` ≤ near_bp on both sides: the
        row up to `reach`, and the column, one element in each earlier row within it."""
        lo, hi = _window(c, local, reach)
        # Each row's span runs from its first upstream partner to its last downstream one,
        # the row itself included once; dropping it leaves both halves, ascending.
        span, owner = spans(lo, hi - lo), owners(hi - lo, local)
        partner, owner = span[span != owner], owner[span != owner]
        lo_pair, hi_pair = np.minimum(owner, partner), np.maximum(owner, partner)
        codes = self._read(c.near_r, near_address(c.near_indptr, lo_pair, hi_pair))
        return hi - lo - 1, partner, self._encoding.dequantise(codes)

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
        lo, hi = _window(c, local, reach)
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
        return [refused_if_lost(array, run) for run in self._cache.runs(array, first, length)]

    def _chromosome_loci(self, name: str) -> _Loci:
        """One chromosome's rows and positions, read on first use and nothing else (#71).

        Positions are the variant table's, one slice a chromosome, so an ALID lookup, a clump and
        a proxies query read no `near/<chrom>/indptr` and open none of the Proxy Index's arrays.
        A chromosome's positions are loaded once, however many threads are cold on it: the check is
        repeated under the lock, as a first use of the Partner Rows is (§7), so no caller publishes
        a second copy of a chromosome-sized array.
        """
        loci = self._located.get(name)
        if loci is not None:
            return loci
        with self._loci_lock:
            return self._loaded_loci(name)

    def _loaded_loci(self, name: str) -> _Loci:
        """`_chromosome_loci` with the lock held: the one load a chromosome's positions get."""
        loci = self._located.get(name)
        if loci is None:
            start, stop = self._ranges[name]
            position = self._columns["position"]
            loci = _Loci(
                name,
                start,
                stop,
                self._runs(position, np.array([start]), np.array([stop - start]))[0].astype(
                    np.int64
                ),
            )
            self._located[name] = loci
        return loci

    def _chromosome(self, name: str) -> _Chromosome:
        """One chromosome's kept arrays, read on first use (spec §10). Each of its five arrays'
        fill values is judged before any element of the chromosome is decoded.

        It takes its positions from the published `_Loci`, and publishes it under the same lock, so
        `_loaded[c].position is _located[c].position` however the two paths interleave.
        """
        loaded = self._loaded.get(name)
        if loaded is not None:
            return loaded
        with self._loci_lock:
            if name in self._loaded:
                return self._loaded[name]
            loci = self._loaded_loci(name)
            near_indptr, near_r, index_indptr, index_partner, index_r = (
                _kept_array(self._root, f"{kind}/{name}/{array}")
                for kind, array in _CHROMOSOME_ARRAYS
            )
            assembled = _Chromosome(
                loci.name,
                loci.start,
                loci.stop,
                loci.position,
                _indptr(near_indptr, self._cache),
                near_r,
                _Sparse(index_indptr, self._cache, index_partner, index_r),
            )
            self._loaded[name] = assembled
            return assembled

    def _far_pairs(self, c: _Chromosome) -> _Sparse:
        """One chromosome's Far Pairs, opened and their `indptr` kept the first time a request
        needs its pairs beyond near_bp (spec §10). `proxies` and `clump` never do, so a
        chromosome they touch reads no more than it did before the Far Pairs. Each array's fill
        value is judged before any of its elements is decoded."""
        if c.name not in self._far:
            indptr, partner, r = (
                _kept_array(self._root, f"far/{c.name}/{array}") for array in _SPARSE_ARRAYS
            )
            self._far[c.name] = _Sparse(indptr, self._cache, partner, r)
        return self._far[c.name]

    def _by_chromosome(
        self, rows: npt.NDArray[np.int64]
    ) -> Iterator[tuple[_Chromosome, npt.NDArray[np.int64]]]:
        """Each chromosome `rows` touch, with the positions in `rows` that are on it."""
        which = chromosome_index(self._stops, rows)
        for index in np.unique(which):
            yield self._chromosome(self._names[index]), np.flatnonzero(which == index)

    def _loci_by_chromosome(
        self, rows: npt.NDArray[np.int64]
    ) -> Iterator[tuple[_Loci, npt.NDArray[np.int64]]]:
        """`_by_chromosome` without the LD components: each chromosome `rows` touch, its rows and
        positions, and the positions in `rows` that are on it (#71)."""
        which = chromosome_index(self._stops, rows)
        for index in np.unique(which):
            yield self._chromosome_loci(self._names[index]), np.flatnonzero(which == index)

    def _strings(self, name: str, rows: npt.NDArray[np.int64]) -> npt.NDArray[np.str_]:
        """The string column `name` at `rows`; an empty string is a lost chunk's, never data."""
        # zarr hands back numpy StringDType, which casts to a fixed width only via a list.
        values = np.array(self._read(self._columns[name], rows).tolist(), dtype=np.str_)
        return values

    def _read(self, array: zarr.Array[Any], coords: npt.NDArray[np.int64]) -> npt.NDArray[Any]:
        """The elements of `array` at `coords`, through the chunk cache; a lost chunk refused."""
        return refused_if_lost(array, self._cache.take(array, coords))

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


class _ClumpWindows:
    """One sliding positional warm range, never a second cache or chromosome-sized pin."""

    def __init__(
        self, reader: StoreReader, c: _Chromosome, local: npt.NDArray[np.int64], reach: int
    ) -> None:
        self.reader, self.c, self.local, self.reach = reader, c, local, reach
        self.position = c.position[local]
        self.low, self.high = 1, 0

    def warm(self, row: np.int64) -> None:
        centre = int(self.c.position[int(row) - self.c.start])
        if self.low <= centre <= self.high:
            return
        radius = self.reach
        while True:
            a, b = position_rows(self.position, centre - radius, centre + radius)
            if (
                self.reader._prefetch_hits(self.c, self.local[int(a) : int(b)], self.reach)
                or radius == 0
            ):
                self.low, self.high = centre - radius, centre + radius
                return
            radius //= 2


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
    return c.index.indptr[local + 1] - c.index.indptr[local]


def _window(
    c: _Chromosome, local: npt.NDArray[np.int64], reach: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """For each local row, the rows [lo, hi) within `reach` bp of it on both sides, itself
    included: the window `partners` reads in the Near Band."""
    return position_rows(c.position, c.position[local] - reach, c.position[local] + reach)


def _window_rows(
    c: _Chromosome, local: npt.NDArray[np.int64], *, reach: int
) -> npt.NDArray[np.int64]:
    """How many rows lie within `reach` of each local row, on both sides, itself included."""
    lo, hi = _window(c, local, reach)
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


def _judged(
    gap: npt.NDArray[np.int64],
    partner: npt.NDArray[np.int64],
    r: npt.NDArray[np.float64],
    *,
    r2: float,
    reach: int,
    listed: npt.NDArray[np.int64] | None,
) -> npt.NDArray[np.bool_]:
    """Which entries `partners` keeps: d ≤ reach and r̂² ≥ r2, the inclusive edges every
    query narrows from (spec §9), and in `listed` when there is one."""
    kept = within(gap, reach) & (r * r >= r2)
    return kept if listed is None else kept & _member(partner, listed)


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


def _group(root: zarr.Group, path: str) -> zarr.Group:
    member = root[path]
    assert isinstance(member, zarr.Group)
    return member


def _array(root: zarr.Group, path: str) -> zarr.Array[Any]:
    """The array at `path`; one the store lacks is refused, naming it, never read as empty."""
    try:
        member = root[path]
    except KeyError:
        raise ValueError(
            f"{path} is missing: every LD Store holds it (spec §1, envelope.arrays)"
        ) from None
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


def _indptr(array: zarr.Array[Any], cache: ChunkCache) -> npt.NDArray[np.int64]:
    data = cache.runs(array, [0], [array.shape[0]])[0]
    return refused_if_lost(array, data).astype(np.int64, copy=False)
