"""Reading an LD Store: the seam every query is built on (ld-store §1–§6, §10; ogld#4).

`StoreReader` is the store layer's whole read interface. It hides Zarr, the chunk and
shard shapes, chromosome-local rows and the codes, which it decodes at the scale the store
records (`opengwasdb.ld.format.REncoding`; ADR 0090): callers pass and receive global Variant Set
rows
and Stored r as float64. Every unstored pair it reports is NaN, never a
number, and the meaning of a NaN is the format's to say (`opengwasdb.ld.format.answer_kinds`).

The variant half is `VariantTable`'s (`opengwasdb/ld/store/variants.py`), which this reader composes
and
delegates to: the rows of a list of ALIDs, a region's rows, a chromosome's range, the columns at
some rows, and the loci a query orders and measures by (#94). One class owns ALID lookup, so a
second reader composes the same one and #87's change to `rows_of` lands once for both (owner, on
epic #18, 21 November 2026). What is below is the LD half: pairs, matrices, regions and the
Partner Rows.

Reads are bounded (ld-store §10). Opening reads the provenance and the chromosome ranges, and
nothing else. A chromosome's positions and its Near Band's `indptr` are read on first touch,
and its Far Pairs' `indptr` when a request needs pairs beyond near_bp. Each is kept. Every
other read goes through `ChunkCache`,
which fetches only the inner chunks holding the elements a request needs, from disk or over
HTTP alike, and keeps them, so a query that reads a few pairs at a time fetches each once.

A chunk the store lost reads as its array's fill value, and ld-store §6 chose each fill value
as one a valid LD Store never holds. So every read refuses that value, naming the array,
rather than decode it: a lost Near Band chunk is a refusal, never a run of r = 0.

From format 0.2.0 a variant's Partner Rows row is a read of its own (`partner_row`; partner-rows
§7): one frame's byte range out of `partners/<chrom>/frames`, one decode, and the row's slice of
the decoded frame, which the reader's frame cache keeps. It reads no Near Band chunk and no shard
index, and a store that holds no Partner Rows is refused naming the component.

No read here opens a Proxy Index: format 0.2.0 stores hold none (#72), and a 0.2.0 store that
still carries a leftover `index/` answers exactly as one without it does (ld-store §12).
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from functools import cached_property, partial
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
import zarr

from opengwasdb.ld.format import (
    READABLE_FORMATS,
    Codes,
    InputError,
    Provenance,
    REncoding,
    claims_partner_rows,
)
from opengwasdb.ld.format.geometry import (
    blocks,
    chromosome_index,
    indptr_of,
    local_rows,
    near_address,
    near_row_lengths,
    owners,
    row_pairs,
    spans,
    stop_within,
    windows,
    within,
)
from opengwasdb.ld.store import layout
from opengwasdb.ld.store.frames import FrameCache, PartnerFrames, open_partner_frames
from opengwasdb.ld.store.location import (
    ChunkCache,
    OpenedStore,
    array_indptr,
    elements_of,
    kept_array,
    open_group,
    runs_of,
)
from opengwasdb.ld.store.tables import Variants
from opengwasdb.ld.store.variants import VariantTable
from opengwasdb.ld.store.workers import COMPONENT_WORKERS_MAX, ComponentWorkers

#: Each chromosome's arrays a request touching it opens (ld-store §1), in the order `_Chromosome`
#: takes them. Its Far Pairs are opened apart, when a request first needs them (`_far_pairs`).
_CHROMOSOME_ARRAYS: Final = (
    ("near", "indptr"),
    ("near", "r"),
)
#: A sparse component's arrays, as `far/<chrom>/` holds them (ld-store §5).
_SPARSE_ARRAYS: Final = ("indptr", "partner", "r")
#: The pairs one step of `pair_r` resolves at once, which bounds its working memory.
_PAIRS_PER_STEP: Final = 1 << 22


@dataclass(frozen=True)
class _Sparse:
    """A sparse component, the Far Pairs: handles opened and fill-checked together, indptr loaded
    only on use, so a request that needs no far pair decodes nothing of it (ogld#40)."""

    source: zarr.Array[Any]
    cache: ChunkCache
    partner: zarr.Array[Any]
    r: zarr.Array[Any]

    @cached_property
    def indptr(self) -> npt.NDArray[np.int64]:
        return array_indptr(self.source, self.cache)


@dataclass(frozen=True)
class _Chromosome:
    """One chromosome: its rows, the positions and `indptr` arrays a reader keeps (ld-store §10),
    and handles on its code arrays, read element by element as requests need them."""

    name: str
    start: int
    stop: int
    position: npt.NDArray[np.int64]
    near_indptr: npt.NDArray[np.int64]
    near_r: zarr.Array[Any]


@dataclass(frozen=True)
class _Placed:
    """Ascending, distinct local rows of one chromosome, the row and column of each in a
    square result, and that result: what `pair_r` and `region_r` write their pairs into."""

    local: npt.NDArray[np.int64]
    at: npt.NDArray[np.int64]
    out: npt.NDArray[np.float64]


class StoreReader(ComponentWorkers):
    """An opened LD Store, local or over HTTP (ld-store §2, §10).

    Opening reads the provenance and the chromosome ranges and refuses a store whose
    `format_version` or `encoding` this package does not implement (ld-store §13), naming
    it, before anything is decoded. Every code is decoded with the store's own encoding, so
    one process reads stores of different scales side by side.

    It composes a `VariantTable` for the variant half (#94) and keeps the LD half itself: the
    Near Band, the Far Pairs, the Partner Rows and the codes it decodes. Every public variant
    read is a delegate, so a query that names one reader is answered by one owner's code.

    A reader is not fork-safe and is not inherited across `fork`: it holds locks and, once a
    chromosome's Partner Rows are read, a chunk file's descriptor. A process worker opens its own
    reader (partner-rows §9; ogld#71).
    """

    _opened: OpenedStore
    _encoding: REncoding
    _variant_table: VariantTable
    _ranges: dict[str, tuple[int, int]]
    _names: list[str]
    _stops: npt.NDArray[np.int64]
    _loaded: dict[str, _Chromosome]
    _far: dict[str, _Sparse]
    _frame_cache: FrameCache
    _partner_frames: dict[str, PartnerFrames]
    _loaded_lock: threading.Lock
    _partner_lock: threading.Lock
    _query_workers: int
    _closed: bool

    def _attach(self, opened: OpenedStore, encoding: REncoding, frames: FrameCache) -> None:
        """Keep the opened store, the encoding and the variant table of a checked store, and the
        second cache its reads go through: decoded Partner Rows frames (partner-rows §7), and the
        component threads it may run (§9.2).

        The table takes the same `OpenedStore` this reader holds, so one store is opened once
        (#94) and the group, the location a refusal names, the provenance and the one chunk cache
        are one object's for both halves. Its chromosome record is held here too, as the same
        objects and never a copy, because the LD reads walk it per step: a range, a label and a
        stop.
        """
        self._opened = opened
        self._encoding = encoding
        self._frame_cache = frames
        self._query_workers = COMPONENT_WORKERS_MAX
        self._variant_table = VariantTable(opened)
        self._ranges = self._variant_table.ranges
        self._names = self._variant_table.names
        self._stops = self._variant_table.stops
        self._loaded = {}
        self._far = {}
        self._partner_frames = {}
        # A chromosome's first use is one load, however many threads make it: its positions are
        # loaded once, by the variant table, under the table's lock, and the Near Band's `indptr`
        # this reader loads beside them is one load under this one; its Partner Rows' offsets chunk
        # is read once and one chunk-file descriptor is opened (partner-rows §7).
        self._loaded_lock = threading.Lock()
        self._partner_lock = threading.Lock()
        self._closed = False

    @classmethod
    def open(
        cls,
        location: str | os.PathLike[str],
        *,
        cache_bytes: int | None = None,
        frame_cache_bytes: int | None = None,
    ) -> StoreReader:
        """Open the LD Store at a local path or an HTTP(S) URL (ogld#4), keeping at most
        `cache_bytes` of its Near Band chunks (ld-store §10; the default if None, ogld#32) and
        `frame_cache_bytes` of its decoded Partner Rows frames (partner-rows §7; its own default
        if None). The two are separate budgets, because the two reads serve different queries."""
        return cls.from_opened(
            _opened_store(location, cache_bytes), frame_cache_bytes=frame_cache_bytes
        )

    @classmethod
    def from_opened(
        cls, opened: OpenedStore, *, frame_cache_bytes: int | None = None
    ) -> StoreReader:
        """An LD reader over a root a dispatcher opened and named an LD Store (#91): the root is
        read once, so an open costs one metadata read."""
        try:
            encoding = REncoding.of(opened.provenance.get("encoding"))
        except InputError as unimplemented:
            raise InputError(f"{opened.location}: {unimplemented}") from unimplemented
        reader = cls()
        reader._attach(opened, encoding, FrameCache(frame_cache_bytes))
        return reader

    # The store's facts and its chunk cache are the `OpenedStore` the variant table holds too, so
    # these read and, for the cache, replace it for both halves (#94). The benchmark harness swaps
    # `_cache` to measure what a read holds; a copy here would leave the variant reads behind.
    @property
    def _root(self) -> zarr.Group:
        return self._opened.root

    @_root.setter
    def _root(self, root: zarr.Group) -> None:
        self._opened.root = root

    @property
    def _location(self) -> str:
        return self._opened.location

    @_location.setter
    def _location(self, location: str) -> None:
        self._opened.location = location

    @property
    def _provenance(self) -> Provenance:
        return self._opened.provenance

    @_provenance.setter
    def _provenance(self, provenance: Provenance) -> None:
        self._opened.provenance = provenance

    @property
    def _cache(self) -> ChunkCache:
        return self._opened.cache

    @_cache.setter
    def _cache(self, cache: ChunkCache) -> None:
        self._opened.cache = cache

    @property
    def cache_bytes(self) -> int:
        """The bytes of chunks and shard indexes this reader keeps at most (ld-store §10)."""
        return self._cache.budget

    @property
    def frame_cache_bytes(self) -> int:
        """The bytes of decoded Partner Rows frames this reader keeps at most (partner-rows §7)."""
        return self._frame_cache.budget

    def request(self) -> AbstractContextManager[object]:
        """Held for one public query; an LD Store has nothing to enter, its codes are stored."""
        return nullcontext()

    def close(self) -> None:
        """Close the reader: after it returns no Partner Rows read is answered, and every chunk file
        it kept open is closed (partner-rows §7). The variant table's optional ALID mappings are
        released too (ADR 0098), after any concurrent mapped lookup finishes.

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
        self._variant_table.close()

    def __enter__(self) -> StoreReader:
        return self

    def __exit__(self, kind: object, value: object, traceback: object) -> None:
        self.close()

    @property
    def location(self) -> str:
        """Where the store was opened from, as the caller named it: what a refusal names (§9), so
        a user with two stores knows which one declined to answer."""
        return self._location

    @property
    def provenance(self) -> Provenance:
        """The root group's `ld_store` attribute, as recorded (ld-store §7)."""
        return self._provenance

    @property
    def encoding(self) -> REncoding:
        """The encoding the store records, which every code it reads is decoded with (§3)."""
        return self._encoding

    # ------------------------------------------------------- the variant table's reads
    #
    # These are the read interface a query names (owner, 10 November 2026), and their one owner is
    # `opengwasdb.ld.store.VariantTable` (#94): the reader answers them through the table it
    # composes, so a
    # second reader and this one cannot drift apart on what an ALID, a region or a chromosome means.

    def rows_of(self, alids: Sequence[str]) -> npt.NDArray[np.int64]:
        """The Variant Set row of each ALID, in order; −1 where the ALID is Absent (ld-store §2).

        The table's search: the ALID index (ADR 0096) for a long list, the validated local
        companion where one is published (ADR 0098), and the exact scan for what neither answers.
        A malformed ALID is Absent too. See `VariantTable.rows_of`.
        """
        return self._variant_table.rows_of(alids)

    def rows_in(self, chrom: str, start: int, stop: int) -> npt.NDArray[np.int64]:
        """The rows on `chrom` with start ≤ position ≤ stop, ascending (ld-store §2).

        The rows are the variant table's answer; the chromosome is opened here first, as this has
        always been the reader's first touch of a chromosome that `region` will read from its Near
        Band, so each of its arrays' fill values is judged before any element is decoded (ld-store
        §6). `loci_of` is the variant read that must open none of them (#71).
        """
        if chrom in self._ranges:
            self._chromosome(chrom)
        return self._variant_table.rows_in(chrom, start, stop)

    def chromosome_range(self, chrom: str) -> tuple[int, int]:
        """The Variant Set rows chromosome `chrom` occupies, `[start, stop)` (ld-store §2), which is
        what turns a Partner Rows row's local partner into a Variant Set row (partner-rows §7)."""
        return self._variant_table.chromosome_range(chrom)

    def alids_of(self, rows: npt.NDArray[np.int64]) -> npt.NDArray[np.str_]:
        """The ALID of each of `rows`, in the order given (ld-store §2): the one string column a
        query names its rows by, read without the others (ogld#14)."""
        return self._variant_table.alids_of(rows)

    def loci_of(
        self, rows: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.str_], npt.NDArray[np.int64]]:
        """Chromosomes and positions, without allele columns clump never returns (ogld#40): the
        variant table's positions, and nothing of the LD components (#71)."""
        return self._variant_table.loci_of(rows)

    def variants(self, rows: npt.NDArray[np.int64]) -> Variants:
        """The Variant Set's columns at `rows`, in the order given (ld-store §2)."""
        return self._variant_table.variants(rows)

    def pair_r(self, rows: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        """The square matrix of Stored r among `rows`, in the order given (ld-store §8, §10).

        Pairs within near_bp come from the Near Band, resolved as `r_of_pairs` resolves
        them; farther ones from the Far Pairs rows of the listed variants, each pair whose
        partner is listed scattered into the matrix, rather than searched for pair by pair
        (ADR 0093).
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

    def r_of_pairs(
        self, first: npt.NDArray[np.int64], second: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float64]:
        """The Stored r of each pair (first[k], second[k]), as one array (ld-store §8, §10;
        ogld#14).

        `first` and `second` broadcast against each other, so one row in `first` is one row
        against many. A row with itself is 1. Every pair the store does not hold, the Bound,
        Not Measured and a pair across chromosomes alike, is NaN. A pair within near_bp is one
        element of the Near Band; a pair farther apart is looked up in the Far Pairs row of
        its earlier variant (ld-store §8). `partners` (below t_far, with `among`) and `pair_r`
        resolve their pairs as it does, so a pair is resolved in one place.

        Memory: the result, 8 bytes a pair, and one step of at most `_PAIRS_PER_STEP` pairs
        at a time, about 60 bytes each (58 measured), with the Far Pairs rows of that step's
        earlier variants. The broadcast is never copied, so one row against many holds its
        result and one step. `tests/ld/test_partner_reads.py` holds the bound.
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
        chromosome, in row order (ld-store §8, §10; ogld#14): the read `region` makes.

        Each row's Near Band pairs inside the region are one contiguous slice of
        `near/<c>/r`, from the row's first code to its last partner before `stop`, never a
        gather pair by pair, and never the row's partners past the region. The region's pairs
        beyond near_bp come from its rows' Far Pairs rows, each pair inside the region scattered
        into the matrix (ADR 0093). The diagonal is 1 and every unstored pair is NaN. A span
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

    # ------------------------------------------------------------ partner rows
    def require_partner_rows(self) -> float:
        """The Partner Threshold this store recorded (ld-store §7; partner-rows §3).

        From format 0.2.0 `clump` and `proxies` read the Partner Rows and nothing else, so a
        store that holds none — every 0.1.0 store — is refused here, naming the store, the missing
        component and the way to derive it, before any read. It is never answered from the Near
        Band: that is the fallback the Partner Rows exist to replace (partner-rows §8.1, §8.4).

        The recorded **format version** decides, and the key alone does not: ld-store §7 lets a
        store hold other keys, and a 0.1.0 store's extra `t_partner` is no claim to a component
        whose frames nothing has judged. Reading one would answer a query from an unclaimed residue
        — a wrong answer that looks like a right one (ogld#69).
        """
        if not self.has_partner_rows:
            raise InputError(
                f"this LD Store at {self._location} is format "
                f"{self._provenance['format_version']}, which has no "
                f"Partner Rows ({layout.PARTNERS}/): clump and proxies read them, so it is refused "
                "naming the missing component; a store built before 0.2.0 derives them with "
                "`ogld derive-partners` (ld-store §13; partner-rows §5.1)"
            )
        return float(self._provenance["t_partner"])

    @property
    def has_partner_rows(self) -> bool:
        """Whether the store's provenance claims Partner Rows, which is
        `opengwasdb.ld.format.claims_partner_rows`: the recorded format version is 0.2.0 or later
        **and**
        the Partner Threshold is recorded, which is what a derivation writes in its one claim
        (partner-rows §6.2; ld-store §13). The reader, validation and the derivation ask that one
        rule, so a store one of them calls claimed is one they all do.

        A version this package reads (0.1.0 or 0.2.0, refused at open otherwise) that is below
        0.2.0 has none, whatever keys or groups a store carries: they are a residue nothing has
        claimed (ogld#69).
        """
        return claims_partner_rows(self._provenance)

    def clump_rows(
        self, chrom: str, rows: npt.NDArray[np.int64], window: float
    ) -> AbstractContextManager[Callable[[int], tuple[npt.NDArray[np.int32], Codes]]]:
        """The request-local row-source seam `clump` reads a component's partners through (#91).

        `clump` asks this for a callable instead of calling `partner_row` itself, so a reader
        whose pairs are not materialised can answer from the component's hits alone. For an LD
        Store the seam adds nothing: its Partner Rows are stored, so the callable is the same
        `partner_row`, with no read of its own and no change to an answer or a timing. How
        another kind restricts it is that reader's to say.
        """
        return nullcontext(partial(self.partner_row, chrom))

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
        """One chromosome's Partner Rows, opened on first use (ld-store §10; partner-rows §7).

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
                f"{', '.join(self._names)} (ld-store §2)"
            )
        # A 0.1.0 store records none of the Partner Rows' keys, because it holds no Partner Rows
        # (ld-store §13): the read is refused naming the component, never answered from another one.
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
        """The Stored r of local pairs lo ≤ hi (ld-store §8, Figure 2): 1 for a row with itself,
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
        the Near Band's (ld-store §8)."""
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
        read from each lo's Far Pairs row (ld-store §8)."""
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
        """The ragged runs of `array` (ld-store §10), each one slice, through the chunk cache;
        a lost chunk refused. One read, which the variant table makes too (`runs_of`)."""
        return runs_of(array, self._cache, first, length)

    def _chromosome(self, name: str) -> _Chromosome:
        """One chromosome's kept arrays, read on first use (ld-store §10). Each of its three arrays'
        fill values is judged before any element of the chromosome is decoded.

        It takes its positions from the table's published `Loci`, so the LD half and the variant
        half keep the one position array: `_loaded[c].position is table.loci(c).position` however
        the two paths interleave, because the table publishes it under its own lock and this
        publishes the chromosome under this reader's.
        """
        loaded = self._loaded.get(name)
        if loaded is not None:
            return loaded
        with self._loaded_lock:
            if name in self._loaded:
                return self._loaded[name]
            loci = self._variant_table.loci(name)
            near_indptr, near_r = (
                kept_array(self._root, f"{kind}/{name}/{array}")
                for kind, array in _CHROMOSOME_ARRAYS
            )
            assembled = _Chromosome(
                loci.name,
                loci.start,
                loci.stop,
                loci.position,
                array_indptr(near_indptr, self._cache),
                near_r,
            )
            self._loaded[name] = assembled
            return assembled

    def _far_pairs(self, c: _Chromosome) -> _Sparse:
        """One chromosome's Far Pairs, opened and their `indptr` kept the first time a request
        needs its pairs beyond near_bp (ld-store §10). `proxies` and `clump` never do, because they
        read the Partner Rows, so a chromosome they touch pays nothing for the Far Pairs. Each
        array's fill value is judged before any of its elements is decoded."""
        if c.name not in self._far:
            indptr, partner, r = (
                kept_array(self._root, f"far/{c.name}/{array}") for array in _SPARSE_ARRAYS
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

    def _read(self, array: zarr.Array[Any], coords: npt.NDArray[np.int64]) -> npt.NDArray[Any]:
        """The elements of `array` at `coords`, through the chunk cache; a lost chunk refused.
        One read, which the variant table makes too (`elements_of`)."""
        return elements_of(array, self._cache, coords)

    def _checked(self, rows: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        """`rows`, if every one is a row of the Variant Set: the table's guard. An Absent ALID is
        −1 from `rows_of` and is no row (ld-store §8; `VariantTable.checked_rows`)."""
        return self._variant_table.checked_rows(rows)


def open_variant_table(
    location: str | os.PathLike[str], *, cache_bytes: int | None = None
) -> VariantTable:
    """The variant table of the LD Store at `location`, and nothing of its LD (ogld#94).

    A reader that composes a table hands it the group it opened, so `StoreReader` does not come
    through here: this is for a caller that wants the variant table alone, which is how a test
    holds it to `StoreReader`'s own answers. It refuses what `StoreReader.open` refuses — a Zarr
    v3 hierarchy, a finished store, a format this package reads (ld-store §1, §13) — and says
    nothing about the encoding, which no variant read decodes a code with.
    """
    return VariantTable(_opened_store(location, cache_bytes))


def _opened_store(location: str | os.PathLike[str], cache_bytes: int | None) -> OpenedStore:
    """An opened store this package can read: its group, the location naming it, its `ld_store`
    provenance and the one chunk cache its reads go through, refused naming the store and what is
    wrong with it (ld-store §1, §7, §13). It reads the root's metadata and nothing else, so it is
    the same work whether a reader or a bare variant table follows."""
    cache = ChunkCache(cache_bytes)
    return opened_ld(open_group(location, cache=cache), str(location), cache)


def opened_ld(root: zarr.Group, location: str, cache: ChunkCache) -> OpenedStore:
    """`OpenedStore` for an already opened root, checked as an LD Store (ld-store §1, §7, §13).

    The kind dispatcher (`opengwasdb.ld.query.kind`) opens a root once and hands it here or to the
    Genotype Store reader, so one open is one metadata read whichever kind it holds (#91).
    """
    if root.metadata.zarr_format != 3:
        raise InputError(
            f"{location} is a Zarr v{root.metadata.zarr_format} hierarchy; an LD Store is "
            "Zarr v3 (ld-store §1)"
        )
    recorded = root.attrs.get("ld_store")
    if not isinstance(recorded, dict):
        raise InputError(f"{location} has no ld_store provenance: it is not a finished LD Store")
    if recorded.get("format_version") not in READABLE_FORMATS:
        raise InputError(
            f"{location} is LD Store format {recorded.get('format_version')!r}; "
            f"this ogld reads {' and '.join(READABLE_FORMATS)} only (ld-store §13)"
        )
    return OpenedStore(root, location, cast(Provenance, recorded), cache)
