"""Reading an LD Store: the seam every query is built on (spec §1–§6, §10; ogld#4).

`StoreReader` is the store layer's whole read interface. It hides Zarr, the chunk and
shard shapes, chromosome-local rows and the int16 codes: callers pass and receive global
Variant Set rows and Stored r as float64. Every unstored pair it reports is NaN, never a
number, and the meaning of a NaN is the format's to say (`ogld.format.answer_kinds`).

This is the walking skeleton's version (ogld#12): it reads whole chromosomes into memory.
ogld#4 replaces the internals with bounded reads, without changing a signature.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import zarr

from ogld.format import ENCODING, FORMAT_VERSION, Provenance, dequantise
from ogld.identity import alid_locus
from ogld.store.layout import R_DTYPE
from ogld.store.tables import Partners, Variants


@dataclass(frozen=True)
class _Chromosome:
    """One chromosome's Variant Set rows and its arrays, read whole."""

    name: str
    start: int
    stop: int
    near_indptr: npt.NDArray[np.int64]
    near_r: npt.NDArray[np.int16]
    index_indptr: npt.NDArray[np.int64]
    index_partner: npt.NDArray[np.int32]
    index_r: npt.NDArray[np.int16]


class StoreReader:
    """An opened LD Store, local or over HTTP (spec §1).

    Opening reads the provenance and the chromosome ranges and refuses a store whose
    `format_version` or `encoding` this package does not implement (spec §13), naming
    it, before anything is decoded.
    """

    _root: zarr.Group
    _provenance: Provenance
    _ranges: dict[str, tuple[int, int]]
    _columns: dict[str, npt.NDArray[Any]]
    _chrom: npt.NDArray[Any]
    _loaded: dict[str, _Chromosome]

    def _attach(self, root: zarr.Group, provenance: Provenance) -> None:
        """Read the Variant Set and chromosome ranges of an opened, checked store."""
        self._root = root
        self._provenance = provenance
        variants = root["variants"]
        assert isinstance(variants, zarr.Group)
        listed = cast(list[dict[str, Any]], variants.attrs["chromosomes"])
        self._ranges = {str(c["name"]): (int(c["row_start"]), int(c["row_stop"])) for c in listed}
        self._columns = {name: _read(variants, name) for name in _VARIANT_COLUMNS}
        self._chrom = np.empty(len(self._columns["position"]), dtype=object)
        for name, (start, stop) in self._ranges.items():
            self._chrom[start:stop] = name
        self._loaded = {}

    @classmethod
    def open(cls, location: str | os.PathLike[str]) -> StoreReader:
        """Open the LD Store at a local path or an HTTP(S) URL (ogld#4)."""
        if str(location).startswith(("http://", "https://")):
            raise NotImplementedError("ogld#4")
        root = zarr.open_group(str(location), mode="r", zarr_format=3)
        recorded = root.attrs.get("ld_store")
        if not isinstance(recorded, dict):
            raise ValueError(
                f"{location} has no ld_store provenance: it is not a finished LD Store"
            )
        if recorded.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"{location} is LD Store format {recorded.get('format_version')!r}; "
                f"this ogld reads {FORMAT_VERSION!r} only (spec §13)"
            )
        if recorded.get("encoding") != ENCODING:
            raise ValueError(
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

        A malformed ALID is Absent too: it names no variant in the store.
        """
        position = self._columns["position"]
        names = self._columns["alid"]
        found = np.full(len(alids), -1, dtype=np.int64)
        for i, alid in enumerate(alids):
            locus = alid_locus(alid)
            if locus is None or locus[0] not in self._ranges:
                continue
            chrom, at = locus
            start, stop = self._ranges[chrom]
            lo = start + int(np.searchsorted(position[start:stop], at, side="left"))
            hi = start + int(np.searchsorted(position[start:stop], at, side="right"))
            for row in range(lo, hi):
                if names[row] == alid:
                    found[i] = row
                    break
        return found

    def rows_in(self, chrom: str, start: int, stop: int) -> npt.NDArray[np.int64]:
        """The rows on `chrom` with start ≤ position ≤ stop, ascending (spec §2)."""
        if chrom not in self._ranges:
            return np.empty(0, dtype=np.int64)
        first, last = self._ranges[chrom]
        position = self._columns["position"][first:last]
        lo = int(np.searchsorted(position, start, side="left"))
        hi = int(np.searchsorted(position, stop, side="right"))
        return np.arange(first + lo, first + hi, dtype=np.int64)

    def variants(self, rows: npt.NDArray[np.int64]) -> Variants:
        """The Variant Set's columns at `rows`, in the order given (spec §2)."""
        c = self._columns
        return Variants(
            chrom=self._chrom[rows].astype(np.str_),
            position=c["position"][rows].astype(np.int64),
            alid=c["alid"][rows],
            a1=c["a1"][rows],
            a2=c["a2"][rows],
            a1_frequency=c["a1_frequency"][rows].astype(np.float64),
        )

    def _chromosome(self, name: str) -> _Chromosome:
        if name not in self._loaded:
            start, stop = self._ranges[name]
            near = self._root[f"near/{name}"]
            index = self._root[f"index/{name}"]
            assert isinstance(near, zarr.Group)
            assert isinstance(index, zarr.Group)
            self._loaded[name] = _Chromosome(
                name,
                start,
                stop,
                _array(near, "indptr", np.int64),
                _array(near, "r", R_DTYPE),
                _array(index, "indptr", np.int64),
                _array(index, "partner", np.int32),
                _array(index, "r", R_DTYPE),
            )
        return self._loaded[name]

    def pair_r(self, rows: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        """The square matrix of Stored r among `rows`, in the order given (spec §8, §10).

        Pairs within near_bp come from the Near Band, farther ones from the Proxy Index.
        The diagonal is 1. Every pair the store does not hold, the Bound and Not Measured
        alike, is NaN.
        """
        near_bp, far_bp = self._provenance["near_bp"], self._provenance["far_bp"]
        position = self._columns["position"].astype(np.int64)
        n = len(rows)
        out = np.full((n, n), np.nan)
        for name in dict.fromkeys(self._chrom[rows]):
            chrom = self._chromosome(str(name))
            members = np.flatnonzero(self._chrom[rows] == name)
            local = rows[members] - chrom.start
            a, b = np.meshgrid(np.arange(len(members)), np.arange(len(members)), indexing="ij")
            lo = np.minimum(local[a], local[b])
            hi = np.maximum(local[a], local[b])
            distance = np.abs(position[rows[members]][a] - position[rows[members]][b])
            near = (lo < hi) & (distance <= near_bp)
            codes = np.full(a.shape, -1, dtype=np.int64)
            codes[near] = chrom.near_r[chrom.near_indptr[lo[near]] + (hi[near] - lo[near] - 1)]
            stored = near.copy()
            far = (lo < hi) & (distance > near_bp) & (distance <= far_bp)
            held = _index_codes(chrom, lo[far], hi[far])
            codes[far] = held
            stored[far] = held != _NOT_IN_INDEX
            block = np.full(a.shape, np.nan)
            block[stored] = dequantise(codes[stored])
            block[lo == hi] = 1.0
            out[np.ix_(members, members)] = block
        return out

    def index(self, rows: npt.NDArray[np.int64]) -> Partners:
        """Each row's Proxy Index entries: every partner within far_bp, either side, with
        r̂² ≥ t_far, and its Stored r (spec §5). One sparse row per variant (§10)."""
        indptr = [0]
        partner: list[npt.NDArray[np.int64]] = []
        codes: list[npt.NDArray[np.int16]] = []
        for row in rows:
            chrom = self._chromosome(str(self._chrom[row]))
            local = int(row) - chrom.start
            lo, hi = chrom.index_indptr[local], chrom.index_indptr[local + 1]
            partner.append(chrom.index_partner[lo:hi].astype(np.int64) + chrom.start)
            codes.append(chrom.index_r[lo:hi])
            indptr.append(indptr[-1] + int(hi - lo))
        flat = np.concatenate(partner) if partner else np.empty(0, dtype=np.int64)
        held = np.concatenate(codes) if codes else np.empty(0, dtype=R_DTYPE)
        return Partners(np.array(indptr, dtype=np.int64), flat, dequantise(held))

    def band(self, rows: npt.NDArray[np.int64]) -> Partners:
        """Each row's Near Band partners on both sides, with their Stored r (spec §4, §10).

        The downstream half is the row; the upstream half is a column, one element in
        each row of the preceding near_bp. It is the slow read the Proxy Index exists to
        avoid, used only by `proxies` at an r2 below t_far.
        """
        raise NotImplementedError("ogld#4")


_NOT_IN_INDEX = np.iinfo(np.int64).min
_VARIANT_COLUMNS = ("position", "alid", "a1", "a2", "a1_frequency")


def _index_codes(
    chrom: _Chromosome, lo: npt.NDArray[np.int64], hi: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    """The Proxy Index's code for each pair (lo, hi) of local rows, or `_NOT_IN_INDEX`."""
    size = chrom.stop - chrom.start
    rows = np.repeat(np.arange(size, dtype=np.int64), np.diff(chrom.index_indptr))
    keys = rows * size + chrom.index_partner.astype(np.int64)
    wanted = lo * size + hi
    at = np.minimum(np.searchsorted(keys, wanted), max(len(keys) - 1, 0))
    if len(keys) == 0:
        return np.full(len(wanted), _NOT_IN_INDEX, dtype=np.int64)
    hit = keys[at] == wanted
    return np.where(hit, chrom.index_r[at].astype(np.int64), _NOT_IN_INDEX)


def _array(group: zarr.Group, name: str, dtype: Any) -> npt.NDArray[Any]:
    member = group[name]
    assert isinstance(member, zarr.Array)
    return np.asarray(member[:]).astype(dtype, copy=False)


def _read(group: zarr.Group, name: str) -> npt.NDArray[Any]:
    member = group[name]
    assert isinstance(member, zarr.Array)
    values = np.asarray(member[:])
    return np.array(values.tolist(), dtype=np.str_) if name in {"alid", "a1", "a2"} else values
