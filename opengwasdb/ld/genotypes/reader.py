"""On-demand LD from pinned pgens, composed with the canonical VariantTable (ADR 0015)."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from ogld.format import Codes, InputError, Provenance, REncoding
from ogld.genotypes.adjust import Block, r
from ogld.genotypes.layout import KEY, VERSION, Fileset, mapped_arrays, open_filesets
from ogld.genotypes.pgen import Genotypes, import_pgenlib
from ogld.store import StoreReader
from ogld.store.location import ChunkCache, OpenedStore, open_group
from ogld.store.variants import VariantTable

_READ_ROWS = 8192
_BLAS_LOCK = threading.Lock()
_BLAS_CONTROL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ogld_blas")
_blas_users = 0
_blas_control: AbstractContextManager[object] | None = None


def _limit_blas() -> AbstractContextManager[object]:
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as missing:
        raise InputError(
            "genotype requests need threadpoolctl from pip install 'ogld[build]'"
        ) from missing
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


class GenotypeReader(StoreReader):
    """The query reader interface with computed LD; inherited variant reads delegate only.

    No decoded calls survive a request. Adjustment arrays are read-only mmaps, shared by
    worker processes through the page cache. Pgen handles are request-local (pgenlib handles
    are not shared between concurrent requests).
    """

    _filesets: list[Fileset]
    _arrays: dict[str, npt.NDArray[Any]]

    @classmethod
    def open(
        cls,
        location: str | os.PathLike[str],
        *,
        frame_cache_bytes: int | None = None,
        cache_bytes: int | None = None,
    ) -> GenotypeReader:
        path = Path(location)
        cache = ChunkCache(cache_bytes)
        root = open_group(location, cache=cache)
        recorded = root.attrs.get(KEY)
        if not isinstance(recorded, dict) or "ld_store" in root.attrs:
            raise InputError(f"{location}: not a Genotype Store: missing {KEY} or found ld_store")
        if root.metadata.zarr_format != 3 or recorded.get("format_version") != VERSION:
            raise InputError(f"{location}: unsupported Genotype Store format")
        import_pgenlib()
        reader = cls()
        reader._opened = OpenedStore(root, str(location), cast(Provenance, recorded), cache)
        reader._encoding = REncoding.of(recorded.get("encoding"))
        reader._variant_table = VariantTable(reader._opened)
        reader._ranges = reader._variant_table.ranges
        reader._filesets = open_filesets(path, recorded["filesets"])
        reader._arrays = mapped_arrays(path, recorded["arrays"])
        reader._query_workers = 1
        reader._closed = False
        return reader

    def close(self) -> None:
        self._closed = True
        self._variant_table.close()
        self._arrays = {}

    def rows_in(self, chrom: str, start: int, stop: int) -> npt.NDArray[np.int64]:
        return self._variant_table.rows_in(chrom, start, stop)

    def require_partner_rows(self) -> float:
        return self.provenance["t_partner"]

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
                with Genotypes(self._filesets[int(index)]) as genotypes:
                    calls[at] = genotypes.rows(a["pgen_row"][rows[at]])
        # Completeness was checked by Variant Set selection. Inputs are pinned on open and
        # immutable thereafter, as for LD arrays; rescanning every dosage costs a full memory
        # pass without adding integrity beyond that contract (ADR 0015).
        return Block(calls, a["projection"][rows], a["norm"][rows], a["sign"][rows])

    def _calls(self, fileset: int, rows: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        source = self._filesets[fileset]
        if len(rows) < _READ_ROWS:
            with Genotypes(source) as genotypes:
                return genotypes.rows(rows)
        calls = np.empty((len(rows), len(source.subset)), dtype=np.float64)

        def read(bounds: tuple[int, int]) -> None:
            lo, hi = bounds
            with Genotypes(source) as genotypes:
                genotypes.rows_into(rows[lo:hi], calls[lo:hi])

        bounds = [(lo, min(lo + _READ_ROWS, len(rows))) for lo in range(0, len(rows), _READ_ROWS)]
        # Native pgen decoding releases the GIL. Four disjoint output slices avoid both the
        # serial 95 ms dosage decode and a concatenation-sized copy measured in #91.
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="ogld_pgen") as pool:
            list(pool.map(read, bounds))
        return calls

    def pair_r(self, rows: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        rows = self._variant_table.checked_rows(rows)
        out = np.full((len(rows), len(rows)), np.nan)
        chrom, position = self.loci_of(rows)
        with _one_blas_thread():
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
        with _one_blas_thread():
            block = self._data(rows)
            for first in range(0, len(rows), 128):
                last = min(first + 128, len(rows))
                raw = r(block[first:last], block[first:])
                # Small stripes keep quantisation's temporary arrays in cache. Compute each
                # off-diagonal tile only once, then mirror its codes like the LD Store (#91).
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
        lead = start + row
        position = int(self.loci_of(np.array([lead], dtype=np.int64))[1][0])
        far = self.provenance["far_bp"]
        rows = self.rows_in(chrom, position - far, position + far)
        with _one_blas_thread():
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
        hits = rows + start
        positions = self.loci_of(hits)[1]
        # This is component-local working memory, released when its greedy ends, not a call cache.
        # No non-hit is decoded and no eliminated lead's correlation row is computed.
        with _one_blas_thread():
            block = self._data(hits)

            def partners(row: int) -> tuple[npt.NDArray[np.int32], Codes]:
                lead = row + start
                at = int(np.searchsorted(hits, lead))
                lo = int(np.searchsorted(positions, positions[at] - window))
                hi = int(np.searchsorted(positions, positions[at] + window, side="right"))
                return self._partners(start, lead, hits[lo:hi], block[lo:hi], positions[lo:hi])

            yield partners

    def _partners(
        self,
        start: int,
        lead: int,
        rows: npt.NDArray[np.int64],
        block: Block,
        position: npt.NDArray[np.int64] | None = None,
    ) -> tuple[npt.NDArray[np.int32], Codes]:
        at = int(np.searchsorted(rows, lead))
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
        return (rows[keep] - start).astype(np.int32), codes[keep]
