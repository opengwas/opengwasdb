"""On-demand LD from pinned pgens, composed with the canonical VariantTable (ADR 0015)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from threadpoolctl import threadpool_limits

from ogld.format import InputError, Provenance, REncoding
from ogld.genotypes.adjust import Block, r
from ogld.genotypes.pgen import Genotypes, import_pgenlib
from ogld.genotypes.sidecar import KEY, VERSION, Fileset, mapped_arrays, open_filesets
from ogld.store import StoreReader
from ogld.store.location import ChunkCache, OpenedStore, open_group
from ogld.store.variants import VariantTable


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
        cache_bytes: int | None = None,
        frame_cache_bytes: int | None = None,
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
        with threadpool_limits(limits=1, user_api="blas"):
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
