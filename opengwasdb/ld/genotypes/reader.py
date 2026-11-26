"""On-demand LD from pinned pgens, composing the canonical VariantTable (ADR 0100).

`GenotypeReader` satisfies ogquery's public `ogquery.ld.Reader` protocol structurally, as
ogquery's `StoreReader` does: it composes the same `OpenedStore` and `VariantTable` (ogquery's,
opengwas/ogquery#2) and holds its own computed-LD half, rather than inheriting the LD Store
reader's. It stays in opengwasdb because it needs pgenlib and threadpoolctl (ogquery ADR 0002).
No decoded calls survive a request. Adjustment arrays are read-only mmaps, shared by worker
processes through the page cache.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from ogquery.ld.format import Codes, InputError, Provenance
from ogquery.ld.format.geometry import global_rows, local_rows, position_rows
from ogquery.ld.store.location import ChunkCache, OpenedStore, open_group
from ogquery.ld.store.tables import Variants
from ogquery.ld.store.variants import VariantTable
from ogquery.ld.store.workers import ComponentWorkers
from threadpoolctl import threadpool_limits

from opengwasdb.ld.format import REncoding
from opengwasdb.ld.genotypes.adjust import Block, r
from opengwasdb.ld.genotypes.layout import KEY, VERSION, Fileset, mapped_arrays, open_filesets
from opengwasdb.ld.genotypes.pgen import Genotypes, import_pgenlib

_READ_ROWS = 8192
_BLAS_LOCK = threading.Lock()
_BLAS_CONTROL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ogld_blas")
_blas_users = 0
_blas_control: AbstractContextManager[object] | None = None


def _limit_blas() -> AbstractContextManager[object]:
    control: AbstractContextManager[object] = threadpool_limits(limits=1, user_api="blas")
    control.__enter__()
    return control


@contextmanager
def _one_blas_thread() -> Iterator[None]:
    global _blas_users, _blas_control
    # The pool is process-wide, and threadpoolctl retains a controlling thread. Both
    # changes run on that same thread; restoration waits for the last active request.
    with _BLAS_LOCK:
        if not _blas_users:
            _blas_control = _BLAS_CONTROL.submit(_limit_blas).result()
        _blas_users += 1
    try:
        yield
    finally:
        with _BLAS_LOCK:
            _blas_users -= 1
            if not _blas_users:
                assert _blas_control is not None
                _BLAS_CONTROL.submit(_blas_control.__exit__, None, None, None).result()
                _blas_control = None


#: The pgen handles kept idle across decodes. A handle keeps the file pages its decode touched,
#: so keeping all 22 EUR pgens' handles added ~830 MB to a whole-genome clump's peak RSS; eight
#: keeps the reuse that matters (a chromosome's components, a chr22 request) within the bound.
_HANDLES_KEPT = 8


class _PgenHandles:
    """Pgen readers leased one to a decoding thread at a time, at most `_HANDLES_KEPT` kept.

    A pgen handle takes ~16 ms to build on the EUR pgens; keeping the handles a request's decodes
    already made is a handle cache, never a decoded call cache. At most one thread holds a handle
    at a time (pgenlib readers are not shared). The idle list is bounded because a handle also
    keeps the pgen file pages its decode touched: all 22 at once measurably grew a clump's peak
    RSS, so the least recently returned handle is closed when the bound is reached. `close`
    closes every idle handle, and a handle leased when `close` runs is closed when its lease ends.
    """

    def __init__(self, filesets: list[Fileset], *, keep: int = _HANDLES_KEPT) -> None:
        if keep < 1:
            raise ValueError(f"at least one pgen handle must be kept idle, not {keep}")
        self._filesets = filesets
        self._keep = keep
        self._idle: list[tuple[int, Genotypes]] = []
        self._lock = threading.Lock()
        self._closed = False

    @contextmanager
    def lease(self, index: int) -> Iterator[Genotypes]:
        with self._lock:
            if self._closed:
                raise InputError(
                    f"{self._filesets[index].pgen}: its Genotype Store reader is closed"
                )
            handle = None
            for at, (fileset, idle) in enumerate(self._idle):
                if fileset == index:
                    handle = idle
                    del self._idle[at]
                    break
        if handle is None:
            handle = Genotypes(self._filesets[index])
        try:
            yield handle
        finally:
            self._release(index, handle)

    def _release(self, index: int, handle: Genotypes) -> None:
        evicted = None
        with self._lock:
            if not self._closed:
                if len(self._idle) >= self._keep:
                    _, evicted = self._idle.pop(0)
                self._idle.append((index, handle))
                keep = True
            else:
                keep = False
        if evicted is not None:
            evicted.close()
        if not keep:
            handle.close()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            idle, self._idle = self._idle, []
        for _, handle in idle:
            handle.close()


class GenotypeReader(ComponentWorkers):
    """The query reader interface with computed LD; it composes the variant table and store.

    The variant half is `VariantTable`'s, exactly as `StoreReader` composes it (opengwas/ogld#94),
    and the adjustment data and pgen handles are this class's. A pgen handle serves one decoding
    thread at a time and is kept for the reader's lifetime (pgenlib handles are not shared
    concurrently); `_PgenHandles` leases it and `close()` closes it.
    """

    _opened: OpenedStore
    _encoding: REncoding
    _variant_table: VariantTable
    _ranges: dict[str, tuple[int, int]]
    _filesets: list[Fileset]
    _handles: _PgenHandles
    _arrays: dict[str, npt.NDArray[Any]]
    _query_workers: int
    _closed: bool

    @classmethod
    def open(
        cls, location: str | os.PathLike[str], *, cache_bytes: int | None = None
    ) -> GenotypeReader:
        """Open the Genotype Store at a local path, refusing an LD Store or another kind."""
        cache = ChunkCache(cache_bytes)
        root = open_group(location, cache=cache)
        record = root.attrs.get(KEY)
        provenance = cast(Provenance, record if isinstance(record, dict) else {})
        return cls.from_opened(OpenedStore(root, str(Path(location)), provenance, cache))

    @classmethod
    def from_opened(cls, opened: OpenedStore) -> GenotypeReader:
        """A reader over a root a dispatcher has already opened and named a Genotype Store."""
        root = opened.root
        record = root.attrs.get(KEY)
        if not isinstance(record, dict) or "ld_store" in root.attrs:
            raise InputError(
                f"{opened.location}: not a Genotype Store: missing {KEY} or found ld_store"
            )
        if root.metadata.zarr_format != 3 or record.get("format_version") != VERSION:
            raise InputError(f"{opened.location}: unsupported Genotype Store format")
        import_pgenlib()
        path = Path(opened.location)
        reader = cls()
        reader._opened = opened
        reader._encoding = REncoding.of(record.get("encoding"))
        reader._variant_table = VariantTable(opened)
        reader._ranges = reader._variant_table.ranges
        reader._filesets = open_filesets(path, record["filesets"])
        reader._handles = _PgenHandles(reader._filesets)
        reader._arrays = mapped_arrays(path, record["arrays"])
        reader._query_workers = 1
        reader._closed = False
        return reader

    def request(self) -> AbstractContextManager[object]:
        """One public query holds the one-BLAS-thread guard for its whole duration."""
        return _one_blas_thread()

    def close(self) -> None:
        """Close the reader, the pgen handles it kept, and the ALID mappings its table kept."""
        self._closed = True
        self._handles.close()
        self._variant_table.close()
        self._arrays = {}

    @property
    def location(self) -> str:
        """Where the store was opened from, as the caller named it."""
        return self._opened.location

    @property
    def provenance(self) -> Provenance:
        """The root group's `genotype_store` attribute, as recorded (ld-store §7)."""
        return self._opened.provenance

    @property
    def encoding(self) -> REncoding:
        """The encoding the store records, which every code it reads is decoded with (§3)."""
        return self._encoding

    # ------------------------------------------------------- the variant table's reads
    #
    # As in `StoreReader`, their one owner is `ogquery.ld.store.VariantTable` (#94): both readers
    # answer through the table they compose, so neither drifts on an ALID, a region or a locus.

    def rows_of(self, alids: Sequence[str]) -> npt.NDArray[np.int64]:
        return self._variant_table.rows_of(alids)

    def rows_in(self, chrom: str, start: int, stop: int) -> npt.NDArray[np.int64]:
        return self._variant_table.rows_in(chrom, start, stop)

    def region_rows(self, chrom: str, start: int, stop: int) -> npt.NDArray[np.int64]:
        """`rows_in`'s rows, for `region` (ogquery's `Reader.region_rows`; ogquery ADR 0002).

        An LD Store's reader opens its chromosome's Near Band here, before deciding a region holds
        no variant. A Genotype Store has no Near Band: it computes a region's r from the pgens, so
        its region's rows are the variant table's, as `rows_in` reads them."""
        return self.rows_in(chrom, start, stop)

    def chromosome_range(self, chrom: str) -> tuple[int, int]:
        return self._variant_table.chromosome_range(chrom)

    def alids_of(self, rows: npt.NDArray[np.int64]) -> npt.NDArray[np.str_]:
        return self._variant_table.alids_of(rows)

    def loci_of(
        self, rows: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.str_], npt.NDArray[np.int64]]:
        return self._variant_table.loci_of(rows)

    def variants(self, rows: npt.NDArray[np.int64]) -> Variants:
        return self._variant_table.variants(rows)

    # ------------------------------------------------------------ the computed LD half

    def require_partner_rows(self) -> float:
        """The Partner Threshold this sidecar recorded (ld-store §7; partner-rows §3)."""
        return float(self.provenance["t_partner"])

    def _data(self, rows: npt.NDArray[np.int64]) -> Block:
        if self._closed:
            raise InputError(f"{self.location}: Genotype Store reader is closed")
        a = self._arrays
        files = a["fileset"][rows]
        sources = np.unique(files)
        if len(sources) == 1:
            calls = self._calls(int(sources[0]), a["pgen_row"][rows])
        else:
            calls = np.empty((len(rows), a["basis"].shape[0]), dtype=np.float64)
            for index in sources:
                at = np.flatnonzero(files == index)
                with self._handles.lease(int(index)) as genotypes:
                    calls[at] = genotypes.rows(a["pgen_row"][rows[at]])
        # Completeness was checked by Variant Set selection. Inputs are pinned on open and
        # immutable thereafter, as for LD arrays; rescanning every dosage costs a full memory
        # pass without adding integrity beyond that contract (ADR 0100).
        return Block(calls, a["projection"][rows], a["norm"][rows], a["sign"][rows])

    def _calls(self, fileset: int, rows: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        source = self._filesets[fileset]
        if len(rows) < _READ_ROWS:
            with self._handles.lease(fileset) as genotypes:
                return genotypes.rows(rows)
        calls = np.empty((len(rows), len(source.subset)), dtype=np.float64)

        def read(bounds: tuple[int, int]) -> None:
            lo, hi = bounds
            with self._handles.lease(fileset) as genotypes:
                genotypes.rows_into(rows[lo:hi], calls[lo:hi])

        bounds = [(lo, min(lo + _READ_ROWS, len(rows))) for lo in range(0, len(rows), _READ_ROWS)]
        # Native pgen decoding releases the GIL. Four disjoint output slices avoid both the
        # serial 95 ms dosage decode and a concatenation-sized copy measured in opengwas/ogld#91.
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="ogld_pgen") as pool:
            list(pool.map(read, bounds))
        return calls

    def pair_r(self, rows: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        rows = self._variant_table.checked_rows(rows)
        out = np.full((len(rows), len(rows)), np.nan)
        chrom, position = self.loci_of(rows)
        for name in np.unique(chrom):
            at = np.flatnonzero(chrom == name)
            block = self._data(rows[at])
            for first in range(0, len(at), 512):
                stop = min(first + 512, len(at))
                values = self._values(
                    r(block[first:stop], block), position[at[first:stop]], position[at]
                )
                if len(at) == len(rows):
                    out[first:stop] = values
                else:
                    out[np.ix_(at[first:stop], at)] = values
        np.fill_diagonal(out, 1.0)
        return out

    def _values(
        self,
        correlations: npt.NDArray[np.float64],
        row_position: npt.NDArray[np.int64],
        span_position: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float64]:
        codes = self.encoding.quantise(correlations)
        values = self.encoding.dequantise(codes)
        distance = np.abs(row_position[:, None] - span_position)
        p = self.provenance
        stored = self.encoding.stored_mask(
            distance,
            codes,
            near_bp=p["near_bp"],
            far_bp=p["far_bp"],
            far_code_floor=p["far_code_floor"],
        )
        values[~stored] = np.nan
        return values

    def region_r(self, start: int, stop: int) -> npt.NDArray[np.float64]:
        rows = self._variant_table.checked_rows(np.arange(start, stop, dtype=np.int64))
        chrom, position = self.loci_of(rows)
        if len(rows) and chrom[0] != chrom[-1]:
            raise InputError(f"{self.location}: a region cannot cross chromosomes")
        out = np.empty((len(rows), len(rows)), dtype=np.float64)
        block = self._data(rows)
        for first in range(0, len(rows), 128):
            last = min(first + 128, len(rows))
            raw = r(block[first:last], block[first:])
            # Small stripes keep quantisation's temporary arrays in cache. Compute each
            # off-diagonal tile only once, then mirror its codes like the LD Store
            # (opengwas/ogld#91).
            for lo in range(first, last, 16):
                hi = min(lo + 16, last)
                out[lo:hi, first:] = self._values(
                    raw[lo - first : hi - first], position[lo:hi], position[first:]
                )
            out[first:last, :first] = out[:first, first:last].T
            diagonal = out[first:last, first:last]
            lower = np.tril_indices(last - first, k=-1)
            diagonal[lower] = diagonal.T[lower]
        np.fill_diagonal(out, 1.0)
        return out

    def partner_row(self, chrom: str, row: int) -> tuple[npt.NDArray[np.int32], Codes]:
        start, _ = self.chromosome_range(chrom)
        lead = int(global_rows(row, start))
        position = int(self.loci_of(np.array([lead], dtype=np.int64))[1][0])
        far = self.provenance["far_bp"]
        rows = self.rows_in(chrom, position - far, position + far)
        return self._partners(start, lead, rows, self._data(rows))

    def partner_rows(
        self, chrom: str, local_rows: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int32], Codes]:
        pairs = [self.partner_row(chrom, int(row)) for row in local_rows]
        indptr = np.r_[0, np.cumsum([len(pair[0]) for pair in pairs])].astype(np.int64)
        if not pairs:
            return indptr, np.empty(0, dtype=np.int32), self.encoding.quantise(np.empty(0))
        return indptr, np.concatenate([a for a, _ in pairs]), np.concatenate([b for _, b in pairs])

    @contextmanager
    def clump_rows(
        self, chrom: str, rows: npt.NDArray[np.int64], window: float
    ) -> Iterator[Callable[[int], tuple[npt.NDArray[np.int32], Codes]]]:
        start, _ = self.chromosome_range(chrom)
        hits = global_rows(rows, start)
        positions = self.loci_of(hits)[1]
        # This is component-local working memory, released when its greedy ends, not a call cache.
        # No non-hit is decoded and no eliminated lead's correlation row is computed.
        block = self._data(hits)

        def partners(row: int) -> tuple[npt.NDArray[np.int32], Codes]:
            lead = int(global_rows(row, start))
            at = self._locate(hits, lead)
            lo, hi = position_rows(positions, positions[at] - window, positions[at] + window)
            first, last = int(lo), int(hi)
            return self._partners(
                start, lead, hits[first:last], block[first:last], positions[first:last]
            )

        yield partners

    def _locate(self, rows: npt.NDArray[np.int64], lead: int) -> int:
        """The index of `lead` in the ascending `rows` of its window, refusing loudly if absent.

        The window comes from the store, so the lead is in it; a store whose positions and rows
        disagree must fail here, naming the ALID, rather than read the neighbouring variant's
        row as the lead's (ld-store §2).
        """
        lo, hi = position_rows(rows, lead, lead)
        if int(hi) - int(lo) != 1:
            alid = str(self.alids_of(np.array([lead], dtype=np.int64))[0])
            raise InputError(f"{self.location}: the lead {alid} is not among its window's rows")
        return int(lo)

    def _partners(
        self,
        start: int,
        lead: int,
        rows: npt.NDArray[np.int64],
        block: Block,
        position: npt.NDArray[np.int64] | None = None,
    ) -> tuple[npt.NDArray[np.int32], Codes]:
        at = self._locate(rows, lead)
        codes = self.encoding.quantise(r(block[at : at + 1], block))[0]
        if position is None:
            position = self.loci_of(rows)[1]
        distance = np.abs(position - position[at])
        p = self.provenance
        keep = (rows != lead) & self.encoding.partner_mask(
            distance,
            codes,
            near_bp=p["near_bp"],
            far_bp=p["far_bp"],
            partner_code_floor=p["partner_code_floor"],
            far_code_floor=p["far_code_floor"],
        )
        return local_rows(rows[keep], start).astype(np.int32), codes[keep]
