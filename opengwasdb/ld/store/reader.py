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

import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
import zarr

from ogld.format import ENCODING, FORMAT_VERSION, RESERVED_CODE, InputError, Provenance, dequantise
from ogld.identity import alid_locus
from ogld.store.location import ChunkCache, open_group
from ogld.store.tables import Partners, Variants

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


class StoreReader:
    """An opened LD Store, local or over HTTP (spec §1).

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

    def _attach(self, root: zarr.Group, provenance: Provenance) -> None:
        """Keep the provenance and the chromosome ranges of an opened, checked store."""
        self._root = root
        self._provenance = provenance
        listed = cast(list[dict[str, Any]], _group(root, "variants").attrs["chromosomes"])
        self._ranges = {str(c["name"]): (int(c["row_start"]), int(c["row_stop"])) for c in listed}
        self._names = list(self._ranges)
        self._stops = np.array([stop for _, stop in self._ranges.values()], dtype=np.int64)
        self._loaded = {}
        self._cache = ChunkCache()

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
            lo = np.searchsorted(c.position, at, side="left")
            hi = np.searchsorted(c.position, at, side="right")
            asked.extend(np.repeat(which, hi - lo).tolist())
            candidates.append(c.start + _spans(lo, hi - lo))
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
        lo = int(np.searchsorted(c.position, start, side="left"))
        hi = int(np.searchsorted(c.position, stop, side="right"))
        return np.arange(c.start + lo, c.start + hi, dtype=np.int64)

    def variants(self, rows: npt.NDArray[np.int64]) -> Variants:
        """The Variant Set's columns at `rows`, in the order given (spec §2)."""
        rows = self._checked(rows)
        position = np.empty(len(rows), dtype=np.int64)
        for c, members in self._by_chromosome(rows):
            position[members] = c.position[rows[members] - c.start]
        names = np.array(self._names, dtype=np.str_)
        chrom = names[np.searchsorted(self._stops, rows, side="right")]
        frequency = self._cache.take(_array(self._root, "variants/a1_frequency"), rows)
        _refuse_fill(np.isnan(frequency), "variants/a1_frequency", "NaN")
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

        Pairs within near_bp come from the Near Band, farther ones from the Proxy Index
        row of the earlier variant. The diagonal is 1. Every pair the store does not hold,
        the Bound and Not Measured alike, is NaN.
        """
        rows = self._checked(rows)
        out = np.full((len(rows), len(rows)), np.nan)
        for c, members in self._by_chromosome(rows):
            local, back = np.unique(rows[members] - c.start, return_inverse=True)
            block = self._block(c, local)
            out[np.ix_(members, members)] = block[np.ix_(back, back)]
        return out

    def _block(self, c: _Chromosome, local: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        """The Stored r among one chromosome's ascending, distinct local rows (NaN unstored)."""
        at = c.position[local]
        near = np.searchsorted(at, at + self._provenance["near_bp"], side="right")
        far = np.searchsorted(at, at + self._provenance["far_bp"], side="right")
        block = np.full((len(local), len(local)), np.nan)
        np.fill_diagonal(block, 1.0)
        first = np.arange(1, len(local) + 1, dtype=np.int64)
        for i, j in _pairs(first, near):
            codes = self._codes(c.near_r, c.near_indptr[local[i]] + (local[j] - local[i] - 1))
            block[i, j] = block[j, i] = dequantise(codes)
        for i, j in _pairs(near, far):
            held, codes = self._index_codes(c, local[i], local[j])
            block[i[held], j[held]] = block[j[held], i[held]] = dequantise(codes[held])
        return block

    def _index_codes(
        self, c: _Chromosome, lo: npt.NDArray[np.int64], hi: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[Any]]:
        """For local pairs lo < hi beyond near_bp: whether the Proxy Index holds each, as
        the Far Pair it then is, and its code there; read from each lo's index row."""
        owners = np.unique(lo)
        partner, codes, length = self._index_rows(c, owners)
        size = c.stop - c.start
        keys = np.repeat(owners, length) * size + partner
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
        coords = _spans(first, length)
        partner = self._cache.take(c.index_partner, coords).astype(np.int64)
        _refuse_fill(partner < 0, f"index/{c.name}/partner", "-1")
        return partner, self._codes(c.index_r, coords), length

    def index(self, rows: npt.NDArray[np.int64]) -> Partners:
        """Each row's Proxy Index entries: every partner within far_bp, either side, with
        r̂² ≥ t_far, and its Stored r (spec §5). One sparse row per variant (§10)."""

        def entries(c: _Chromosome, local: npt.NDArray[np.int64]) -> _Entries:
            partner, codes, length = self._index_rows(c, local)
            return length, partner, codes

        return self._gather(rows, entries)

    def band(self, rows: npt.NDArray[np.int64]) -> Partners:
        """Each row's Near Band partners on both sides, with their Stored r (spec §4, §10).

        The downstream half is the row; the upstream half is a column, one element in
        each row of the preceding near_bp. It is the slow read the Proxy Index exists to
        avoid, used only by `proxies` at an r2 below t_far.
        """
        near_bp = self._provenance["near_bp"]

        def entries(c: _Chromosome, local: npt.NDArray[np.int64]) -> _Entries:
            first = np.searchsorted(c.position, c.position[local] - near_bp, side="left")
            length = (local - first) + (c.near_indptr[local + 1] - c.near_indptr[local])
            # Each row's span runs from its first upstream partner to its last downstream one,
            # the row itself included once; dropping it leaves both halves, ascending.
            span = _spans(first, length + 1)
            owner = np.repeat(local, length + 1)
            partner, owner = span[span != owner], owner[span != owner]
            lo, hi = np.minimum(owner, partner), np.maximum(owner, partner)
            return length, partner, self._codes(c.near_r, c.near_indptr[lo] + (hi - lo - 1))

        return self._gather(rows, entries)

    def _gather(
        self,
        rows: npt.NDArray[np.int64],
        entries: Callable[[_Chromosome, npt.NDArray[np.int64]], _Entries],
    ) -> Partners:
        """The partners `entries` gives each of `rows`, as global rows, in CSR form."""
        rows = self._checked(rows)
        length = np.empty(len(rows), dtype=np.int64)
        found: list[tuple[npt.NDArray[np.int64], _Entries]] = []
        for c, members in self._by_chromosome(rows):
            got = entries(c, rows[members] - c.start)
            length[members] = got[0]
            found.append((members, (got[0], got[1] + c.start, got[2])))
        indptr = np.concatenate([[0], np.cumsum(length)]).astype(np.int64)
        partner = np.empty(int(indptr[-1]), dtype=np.int64)
        r = np.empty(int(indptr[-1]), dtype=np.float64)
        for members, (lengths, partners, codes) in found:
            where = _spans(indptr[members], lengths)
            partner[where] = partners
            r[where] = dequantise(codes)
        return Partners(indptr, partner, r)

    def _chromosome(self, name: str) -> _Chromosome:
        """One chromosome's kept arrays, read on first use (spec §10)."""
        if name not in self._loaded:
            start, stop = self._ranges[name]
            position = np.asarray(_array(self._root, "variants/position")[start:stop])
            _refuse_fill(position <= 0, "variants/position", "0")
            near_r, index_r = (
                _r_array(self._root, f"{kind}/{name}/r") for kind in ("near", "index")
            )
            self._loaded[name] = _Chromosome(
                name,
                start,
                stop,
                position.astype(np.int64),
                _indptr(self._root, f"near/{name}/indptr"),
                near_r,
                _indptr(self._root, f"index/{name}/indptr"),
                _array(self._root, f"index/{name}/partner"),
                index_r,
            )
        return self._loaded[name]

    def _by_chromosome(
        self, rows: npt.NDArray[np.int64]
    ) -> Iterator[tuple[_Chromosome, npt.NDArray[np.int64]]]:
        """Each chromosome `rows` touch, with the positions in `rows` that are on it."""
        which = np.searchsorted(self._stops, rows, side="right")
        for index in np.unique(which):
            yield self._chromosome(self._names[index]), np.flatnonzero(which == index)

    def _strings(self, name: str, rows: npt.NDArray[np.int64]) -> npt.NDArray[np.str_]:
        """The string column `name` at `rows`; an empty string is a lost chunk's, never data."""
        # zarr hands back numpy StringDType, which casts to a fixed width only via a list.
        values = np.array(
            self._cache.take(_array(self._root, f"variants/{name}"), rows).tolist(), dtype=np.str_
        )
        _refuse_fill(values == "", f"variants/{name}", '""')
        return values

    def _codes(self, array: zarr.Array[Any], coords: npt.NDArray[np.int64]) -> npt.NDArray[Any]:
        """The codes of an `r` array at `coords`; the reserved code is a lost chunk, refused."""
        codes = self._cache.take(array, coords)
        if (codes == RESERVED_CODE).any():
            raise ValueError(
                f"{array.path} read the reserved code {RESERVED_CODE} where a pair is stored: a "
                "chunk the store lost, never a correlation (spec §3)"
            )
        return codes

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


#: What `_gather` asks for each chromosome: row lengths, partners (local rows) and codes.
_Entries = tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[Any]]


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


def _spans(first: npt.NDArray[np.int64], length: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """first[i], first[i] + 1, … for length[i] elements each, back to back."""
    length = np.asarray(length, dtype=np.int64)
    total = int(length.sum())
    offset = np.repeat(np.cumsum(length) - length, length)
    return np.repeat(np.asarray(first, dtype=np.int64), length) + (np.arange(total) - offset)


def _pairs(
    first: npt.NDArray[np.int64], stop: npt.NDArray[np.int64]
) -> Iterator[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]:
    """The pairs (i, j) with first[i] ≤ j < stop[i], a bounded number at a time."""
    length = np.maximum(stop - first, 0)
    ends = np.cumsum(length)
    start = 0
    while start < len(length):
        before = int(ends[start] - length[start])
        end = max(int(np.searchsorted(ends, before + _PAIRS_PER_STEP, side="right")), start + 1)
        if ends[end - 1] > before:
            yield (
                np.repeat(np.arange(start, end), length[start:end]),
                _spans(first[start:end], length[start:end]),
            )
        start = end


def _refuse_fill(bad: npt.NDArray[np.bool_], name: str, fill: str) -> None:
    """Refuse values that are their array's fill value, which no valid LD Store holds."""
    if bad.any():
        raise ValueError(
            f"{name} read {int(bad.sum())} element(s) as its fill value {fill}, which no LD "
            "Store holds: a chunk the store lost, never data (spec §6)"
        )


def _group(root: zarr.Group, path: str) -> zarr.Group:
    member = root[path]
    assert isinstance(member, zarr.Group)
    return member


def _array(root: zarr.Group, path: str) -> zarr.Array[Any]:
    member = root[path]
    assert isinstance(member, zarr.Array)
    return member


def _r_array(root: zarr.Group, path: str) -> zarr.Array[Any]:
    """An `r` array, refused unless a chunk it lost would read as the reserved code."""
    array = _array(root, path)
    if array.fill_value != RESERVED_CODE:
        raise ValueError(
            f"{path} has the fill value {array.fill_value}, not {RESERVED_CODE}: a chunk it "
            "lost would read as a correlation (spec §6, envelope.fill)"
        )
    return array


def _indptr(root: zarr.Group, path: str) -> npt.NDArray[np.int64]:
    values = np.asarray(_array(root, path)[:]).astype(np.int64)
    _refuse_fill(values < 0, path, "-1")
    return values
