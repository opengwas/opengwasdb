"""On-demand LD from pinned pgens, composed with the canonical VariantTable (ADR 0015)."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from ogld.format import Codes, InputError, Provenance, REncoding
from ogld.genotypes.adjust import Block, r
from ogld.genotypes.pgen import Genotypes, import_pgenlib
from ogld.genotypes.sidecar import KEY, VERSION, Fileset, mapped_arrays, open_filesets
from ogld.store import StoreReader
from ogld.store.location import ChunkCache, OpenedStore, open_group
from ogld.store.variants import VariantTable


@contextmanager
def _one_blas_thread() -> Iterator[None]:
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as missing:
        raise InputError(
            "genotype requests need threadpoolctl from pip install 'ogld[build]'"
        ) from missing
    with threadpool_limits(limits=1, user_api="blas"):
        yield


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
        calls = np.empty((len(rows), a["basis"].shape[0]), dtype=np.float64)
        files = a["fileset"][rows]
        for index in np.unique(files):
            at = np.flatnonzero(files == index)
            with Genotypes(self._filesets[int(index)]) as genotypes:
                calls[at] = genotypes.rows(a["pgen_row"][rows[at]])
        if (calls < 0).any() or not np.isfinite(calls).all():
            raise InputError(f"{self.location}: missing or invalid genotype calls")
        return Block(calls, a["projection"][rows], a["norm"][rows], a["sign"][rows])

    def pair_r(self, rows: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        rows = self._variant_table.checked_rows(rows)
        out = np.full((len(rows), len(rows)), np.nan)
        chrom, position = self.loci_of(rows)
        p = self.provenance
        with _one_blas_thread():
            for name in np.unique(chrom):
                at = np.flatnonzero(chrom == name)
                block = self._data(rows[at])
                for first in range(0, len(at), 128):
                    stop = min(first + 128, len(at))
                    codes = self.encoding.quantise(r(block[first:stop], block))
                    values = self.encoding.dequantise(codes)
                    distance = np.abs(position[at[first:stop], None] - position[at])
                    missing = (distance > p["far_bp"]) | (
                        (distance > p["near_bp"]) & (np.abs(codes) < p["far_code_floor"])
                    )
                    values[missing] = np.nan
                    out[np.ix_(at[first:stop], at)] = values
        np.fill_diagonal(out, 1.0)
        return out

    def region_r(self, start: int, stop: int) -> npt.NDArray[np.float64]:
        return self.pair_r(np.arange(start, stop, dtype=np.int64))

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
        floor = np.where(distance <= p["near_bp"], p["partner_code_floor"], p["far_code_floor"])
        keep = (rows != lead) & (distance <= p["far_bp"]) & (np.abs(codes) >= floor)
        return (rows[keep] - start).astype(np.int32), codes[keep]
