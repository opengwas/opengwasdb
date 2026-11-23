"""The variant table and the ALID lookup: the read every query begins with (ld-store §2, §10).

`VariantTable` is the half of a reader that names variants: the Variant Set's rows, their
chromosomes, positions, ALIDs and alleles (`variants/`), the search that turns a list of ALIDs
into rows — the ALID Index (ADR 0096) or the local companion over it (ADR 0098) — and the
chromosome-local loci the LD half needs to place a row. `StoreReader` composes one and delegates
to it, and the Genotype Store reader composes the same class over its sidecar, so ALID lookup
has one owner (owner, on epic opengwas/ogld#18, 21 November 2026; ogld#94). `rows_of`, `rows_in`,
`chromosome_range`, `alids_of`, `loci_of` and `variants` are that owner's reads.

A table opens no store: its reader opens the group, refuses what it cannot read (ld-store §13) and
hands the table the `opengwasdb.ld.store.location.OpenedStore` both halves hold — the group, the
location
that names the store in a refusal, the recorded provenance and the one chunk cache every read goes
through. So one store is opened once however many halves read it, a replaced cache is replaced for
both, and only the companion's mapping is the table's own to release (`close`). A caller that
wants the table and nothing else opens it with the store layer's own opener,
`opengwasdb.ld.store.reader.open_variant_table` (#94).
"""

from __future__ import annotations

import mmap
import os
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import zarr

from opengwasdb.ld.format import InputError, Provenance
from opengwasdb.ld.format.geometry import (
    chromosome_index,
    global_rows,
    local_rows,
    owners,
    position_rows,
    spans,
)
from opengwasdb.ld.identity import alid_locus
from opengwasdb.ld.store import layout
from opengwasdb.ld.store.layout import chromosomes_listed, refused_if_lost
from opengwasdb.ld.store.location import OpenedStore, elements_of, kept_array, runs_of
from opengwasdb.ld.store.tables import Variants

#: The variant table's columns (ld-store §2).
_VARIANT_COLUMNS: Final = ("position", "alid", "a1", "a2", "a1_frequency")
#: How long a list of ALIDs makes the index worth reading: a `_INDEX_SHARE`th of the Variant Set's
#: entries or more (ogld#42). The whole index costs about what the exact scan's ALID string chunks
#: do at about 4,000 of EUR's 8.59 M ALIDs, and below that a list's loci touch fewer chunks than the
#: index holds; `docs/benchmarks/ld/alid-index.md` has both sides of it. A small store's list is a
#: large share of it, so a test store's lookups read the index.
_INDEX_SHARE: Final = 2_150


def _index_pays(count: int, entries: int) -> bool:
    """Whether `count` keys are worth reading the whole index of `entries` (ld-store §10; ADR 0096):
    a list naming a `_INDEX_SHARE`th of them or more, and never fewer than one."""
    return count >= max(1, entries // _INDEX_SHARE)


@dataclass(frozen=True)
class Loci:
    """One chromosome's Variant Set rows and positions, and nothing of its LD components
    (opengwas/ogld#71).

    A query that reads only the Partner Rows still needs each hit's chromosome, position and ALID
    to order it and measure distances, and must not read `near/<chrom>/indptr` to get them
    (partner-rows §8; ld-store §10). This is that much, from `variants/position` alone; the
    reader's `_Chromosome` is it plus the Near Band, for the reads that resolve stored pairs.
    """

    name: str
    start: int
    stop: int
    position: npt.NDArray[np.int64]


@dataclass(frozen=True)
class _AlidIndex:
    """The ALID search index (ld-store §2; ADR 0096): one NUL-padded 32-byte key an entry, sorted,
    and the Variant Set row each entry names. Keys are bytes, so a search compares bytes."""

    alid: zarr.Array[Any]
    row: zarr.Array[Any]


class _MappedAlids:
    """One immutable companion bundle, owned by its table (ADR 0098).

    Lookup returns independent row arrays; release serializes
    with lookup so no borrowed numpy view survives a mapping. No pool or global cache here.
    """

    def __init__(self, companion: layout.AlidCompanion) -> None:
        self._path = companion.path
        self._lock = threading.Lock()
        self._maps: list[mmap.mmap] = []
        self._closed = False
        self._keys = self._rows = np.empty(0)
        try:
            self._keys = self._map(companion, layout.ALID_MMAP_KEYS, layout.ALID_KEY_DTYPE)
            self._rows = self._map(
                companion, layout.ALID_MMAP_ROWS, np.dtype(layout.ALID_MMAP_ROW_DTYPE)
            )
        except BaseException:
            # Tracebacks can retain this half-built bundle; cleanup cannot wait for collection.
            self.release()
            raise

    def _map(
        self, companion: layout.AlidCompanion, name: str, dtype: np.dtype[Any]
    ) -> npt.NDArray[Any]:
        path = companion.path / name
        try:
            with path.open("rb") as stream:
                actual = layout.file_identity(os.fstat(stream.fileno()))
                if actual != dict(companion.identity)[name]:
                    raise layout.alid_mmap_error(path, "changed before mapping")
                if not companion.count:
                    return np.empty(0, dtype=dtype)
                mapped = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
            self._maps.append(mapped)
            return np.frombuffer(mapped, dtype=dtype)
        except (OSError, ValueError) as failed:
            raise layout.alid_mmap_error(path, failed) from failed

    def lookup(self, keys: npt.NDArray[Any]) -> npt.NDArray[np.int64]:
        with self._lock:
            if self._closed:
                raise layout.alid_mmap_error(
                    self._path / layout.ALID_MMAP_KEYS, "mapping has been released"
                )
            found = np.full(len(keys), -1, dtype=np.int64)
            if len(self._keys):
                at = np.minimum(np.searchsorted(self._keys, keys), len(self._keys) - 1)
                same = self._keys[at] == keys
                found[same] = self._rows[at[same]]
            return found

    def release(self) -> None:
        """Discard views before closing mappings, serialized with lookup (ADR 0098)."""
        with self._lock:
            self._closed = True
            self._keys = self._rows = np.empty(0)
            for mapped in self._maps:
                mapped.close()
            self._maps.clear()


def _mapped_keys(
    alids: Sequence[str],
) -> tuple[npt.NDArray[Any], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Encode query bytes without parsing ALIDs or truncating a name into another (ADR 0098)."""
    encoded: list[bytes] = []
    fitting: list[int] = []
    wide: list[int] = []
    for i, alid in enumerate(alids):
        try:
            key = alid.encode("utf-8")
        except UnicodeEncodeError:
            continue
        if b"\0" in key:
            continue
        if len(key) > layout.ALID_INDEX_WIDTH:
            wide.append(i)
        else:
            encoded.append(key)
            fitting.append(i)
    return (
        np.array(encoded, dtype=layout.ALID_KEY_DTYPE),
        np.array(fitting, dtype=np.int64),
        np.array(wide, dtype=np.int64),
    )


class VariantTable:
    """An LD Store's variant table and ALID lookup, over a group its reader opened (ld-store §2).

    It keeps what a variant read needs and nothing of the LD Store's LD: the chromosome record,
    the `variants/` columns, a chromosome's positions once it is first touched (ld-store §10), the
    ALID index on first search (ADR 0096) and the optional local companion on first fitting
    lookup (ADR 0098). The companion's lifetime is the table's: `close` releases it, and a
    lookup that finds it closed refuses rather than answering from the canonical index.
    """

    _opened: OpenedStore
    _ranges: dict[str, tuple[int, int]]
    _names: list[str]
    _stops: npt.NDArray[np.int64]
    _located: dict[str, Loci]
    _loci_lock: threading.Lock
    _columns: dict[str, zarr.Array[Any]]
    _index_arrays: _AlidIndex | None
    _alid_mmap: _MappedAlids | None
    _alid_lock: threading.Lock
    _alid_checked: bool
    _alid_error: str | None
    _has_alid_mmap: bool
    _closed: bool

    def __init__(self, opened: OpenedStore) -> None:
        """Take the variant table of a store its reader has opened and checked: the chromosome
        ranges and the columns' handles (each opened once, because opening one reads its metadata
        document, which over HTTP is a request). The table opens no group of its own, so the reader
        that composes it opens the store once, and it holds the reader's `OpenedStore` rather than
        a copy of it, so the one chunk cache and the location a refusal names are one object's
        (opengwas/ogld#94)."""
        self._opened = opened
        root = opened.root
        listed = chromosomes_listed(_group(root, "variants").attrs["chromosomes"])
        self._ranges = {name: (start, stop) for name, start, stop in listed}
        self._names = list(self._ranges)
        self._stops = np.array([stop for _, stop in self._ranges.values()], dtype=np.int64)
        self._located = {}
        # A chromosome's first use is one load, however many threads make it (opengwas/ogld#71).
        self._loci_lock = threading.Lock()
        self._columns = {name: kept_array(root, f"variants/{name}") for name in _VARIANT_COLUMNS}
        self._index_arrays = None
        self._alid_lock = threading.Lock()
        self._alid_checked = False
        self._alid_error = None
        self._alid_mmap = None
        self._has_alid_mmap = (
            layout.local_path(root) is not None and layout.ALID_MMAP_ATTR in root.attrs
        )
        self._closed = False

    def close(self) -> None:
        """Release the optional ALID companion's mappings (ADR 0098), after any concurrent mapped
        lookup finishes, and refuse a later one: closing before a lookup attaches nothing, and
        closing twice is closing once. The store's group and its chunk cache are its reader's."""
        with self._alid_lock:
            self._closed = True
            if self._alid_mmap is not None:
                self._alid_mmap.release()

    @property
    def location(self) -> str:
        """Where the store was opened from, as the caller named it: what a refusal names (§9), so
        a user with two stores knows which one declined to answer."""
        return self._opened.location

    @property
    def provenance(self) -> Provenance:
        """The root group's `ld_store` attribute, as recorded (ld-store §7)."""
        return self._opened.provenance

    @property
    def ranges(self) -> dict[str, tuple[int, int]]:
        """Each chromosome's Variant Set rows, `[start, stop)`, in the store's recorded order
        (ld-store §2), kept as they were opened."""
        return self._ranges

    @property
    def names(self) -> list[str]:
        """The store's chromosome labels, in the order `ranges` holds them."""
        return self._names

    @property
    def stops(self) -> npt.NDArray[np.int64]:
        """Each chromosome's `row_stop`, which turns a Variant Set row into its chromosome."""
        return self._stops

    def loci(self, chrom: str) -> Loci:
        """One chromosome's rows and positions, read on first use and nothing else
        (opengwas/ogld#71).

        Positions are the variant table's, one slice a chromosome, so an ALID lookup, a clump and
        a proxies query read no `near/<chrom>/indptr`. A chromosome's positions are loaded once,
        however many threads are cold on it: the check is repeated under the lock, as a first use
        of the Partner Rows is (partner-rows §7), so no caller publishes a second copy of a
        chromosome-sized array.
        """
        loci = self._located.get(chrom)
        if loci is not None:
            return loci
        with self._loci_lock:
            return self._loaded_loci(chrom)

    def rows_of(self, alids: Sequence[str]) -> npt.NDArray[np.int64]:
        """The Variant Set row of each ALID, in order; −1 where the ALID is Absent (ld-store §2).

        A malformed ALID is Absent too: it names no variant in the store. A long list is searched
        for in the store's ALID index (ADR 0096), which holds every ALID of the table that fits its
        32-byte slot, sorted, so one search answers it without reading the variant table at all.
        A validated local companion instead answers fitting hits and misses by mmap without
        parsing loci or reading columns (ADR 0098); only wider names take the exact scan.
        Without that companion, every ALID the index does not hold — absent, too wide for the
        slot, or one a broken build left out — takes the exact scan at its own position. A list
        too short for the index to pay for takes that scan for every ALID. So the index
        can only make this read faster, never change its answer; and a store whose index could
        answer falsely is one validation fails, as it fails a store with any other broken array.
        """
        found = np.full(len(alids), -1, dtype=np.int64)
        if not len(alids):
            return found
        mapped = self._mapped_rows(alids)
        if mapped is not None:
            return mapped
        loci = _loci(alids, self._ranges)
        if not loci:
            return found
        scanned = np.ones(len(alids), dtype=bool)
        self._from_index(alids, loci, found, scanned)
        rest = np.flatnonzero(scanned)
        if len(rest):
            found[rest] = self._scanned(alids, rest)
        return found

    def _mapped_alids(self) -> _MappedAlids | None:
        """Verify/attach once, with sticky failure and close serialized on the same lock."""
        with self._alid_lock:
            if self._alid_error is not None:
                raise InputError(self._alid_error)
            if self._closed:
                raise self._alid_refusal("reader has been released")
            if not self._alid_checked:
                try:
                    companion = layout.alid_companion(self._opened.root)
                    self._alid_mmap = None if companion is None else _MappedAlids(companion)
                except Exception as failed:
                    # Keep text, not a traceback retaining the half-built table/mappings.
                    refusal = self._alid_refusal(failed)
                    self._alid_error = str(refusal)
                    raise refusal from failed
                self._alid_checked = True
            return self._alid_mmap

    def _alid_refusal(self, reason: object) -> InputError:
        return InputError(
            f"{self.location}: ALID companion "
            f"{self._opened.root.attrs.get(layout.ALID_MMAP_ATTR)!r}, "
            f"{layout.ALID_MMAP_MANIFEST}: {reason}"
        )

    def _mapped_rows(self, alids: Sequence[str]) -> npt.NDArray[np.int64] | None:
        if not self._has_alid_mmap:
            return None
        keys, fitting, wide = _mapped_keys(alids)
        found = np.full(len(alids), -1, dtype=np.int64)
        if len(fitting) or self._alid_error is not None or self._closed:
            mapped = self._mapped_alids()
            if mapped is None:
                return None
            found[fitting] = mapped.lookup(keys)
        if len(wide):
            found[wide] = self._scanned(alids, wide)
        return found

    def _from_index(
        self,
        alids: Sequence[str],
        loci: dict[str, tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]],
        found: npt.NDArray[np.int64],
        scanned: npt.NDArray[np.bool_],
    ) -> None:
        """Fill `found` from the ALID index for every ALID it can answer, and clear `scanned`
        there; what it cannot answer stays for the exact scan (ld-store §2, §10; ADR 0096).

        Which ALIDs the index holds a key for is `layout.alid_fits`' question, not this method's:
        an ALID wider than the slot has no key and is scanned, and a store whose key for one row
        is another row's — truncated into the slot, or invented — is one validation fails
        (`alid_index.keys`). A list too short to pay for the index asks for nothing.
        """
        asked = [i for which, _ in loci.values() for i in which.tolist()]
        keys, fitting = layout.alid_keys([alids[i] for i in asked])
        if not len(keys) or not _index_pays(len(keys), self._entries()):
            return
        if self._opened.root.get(layout.ALID_INDEX) is None:
            return
        hits = self._searched(keys)
        where = hits >= 0
        found[np.array(asked, dtype=np.int64)[fitting[where]]] = elements_of(
            self._index().row, self._opened.cache, hits[where]
        )
        scanned[np.array(asked, dtype=np.int64)[fitting[where]]] = False

    def _entries(self) -> int:
        """How many entries the Variant Set's index holds: its rows less the ALIDs too wide for its
        slot, which the store's provenance counts (ld-store §7). A store written before the index
        records no count; `_from_index` keeps that store on the exact-scan fallback (ADR 0098)."""
        counts = self._opened.provenance["counts"]
        return int(self._opened.provenance["n_variants"]) - int(counts.get("alids_left_out", 0))

    def _scanned(self, alids: Sequence[str], which: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        """The rows of the ALIDs at `which`, in that order: the exact scan of the variant table
        at each one's own position (ld-store §2), which is how `rows_of` found every ALID before the
        index. Only the ALID strings at those positions are read, in one read per chromosome."""
        found = np.full(len(which), -1, dtype=np.int64)
        chosen = [alids[int(i)] for i in which]
        asked: list[int] = []
        candidates: list[npt.NDArray[np.int64]] = []
        for chrom, (at, positions) in _loci(chosen, self._ranges).items():
            c = self.loci(chrom)
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
        the array in one run holds all 273 MB of EUR's keys twice (ld-store §10; ogld#42's reviews).
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
        order = np.argsort(keys.view(layout.ALID_KEY_DTYPE).ravel(), kind="stable")
        wanted = keys[order].view(layout.ALID_KEY_DTYPE).ravel()
        chunk = np.empty(0, dtype=layout.ALID_KEY_DTYPE)
        for first in range(0, count, per):
            # Let the shard just searched go before the next is fetched and joined: a lookup holds
            # one shard's keys, not the one before it as well.
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
        run = runs_of(index.alid, self._opened.cache, np.array([first * width]), length)[0]
        keys = refused_if_lost(index.alid, run)
        return keys.view(layout.ALID_KEY_DTYPE).ravel()

    def _index(self) -> _AlidIndex:
        """The ALID index's arrays, opened on first use (ld-store §10): opening reads their
        metadata, and nothing reads them until a search does."""
        if self._index_arrays is None:
            self._index_arrays = _AlidIndex(
                kept_array(self._opened.root, layout.ALID_INDEX_ALID),
                kept_array(self._opened.root, layout.ALID_INDEX_ROW),
            )
        return self._index_arrays

    def rows_in(self, chrom: str, start: int, stop: int) -> npt.NDArray[np.int64]:
        """The rows on `chrom` with start ≤ position ≤ stop, ascending (ld-store §2)."""
        if alid_locus(f"{chrom}:1:A:C") is None:
            raise InputError(
                f"{chrom!r} is not a canonical chromosome label (1–22, X, Y, MT; no 'chr' prefix)"
            )
        if chrom not in self._ranges:
            return np.empty(0, dtype=np.int64)
        c = self.loci(chrom)
        lo, hi = position_rows(c.position, start, stop)
        return global_rows(np.arange(int(lo), int(hi), dtype=np.int64), c.start)

    def chromosome_range(self, chrom: str) -> tuple[int, int]:
        """The Variant Set rows chromosome `chrom` occupies, `[start, stop)` (ld-store §2).

        It is what turns a Partner Rows row's chromosome-local partner into a Variant Set row
        (partner-rows §7), so a component split and a greedy can work in the store's own rows. A
        label the store does not hold is refused naming it, as `partner_row` refuses it.
        """
        try:
            return self._ranges[chrom]
        except KeyError:
            raise ValueError(
                f"no chromosome {chrom} in this LD Store, whose Variant Set is on "
                f"{', '.join(self._names)} (ld-store §2)"
            ) from None

    def alids_of(self, rows: npt.NDArray[np.int64]) -> npt.NDArray[np.str_]:
        """The ALID of each of `rows`, in the order given (ld-store §2): the one string column a
        query names its rows by, read without the others (ogld#14)."""
        return self._strings("alid", self.checked_rows(rows))

    def loci_of(
        self, rows: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.str_], npt.NDArray[np.int64]]:
        """Chromosomes and positions, without allele columns clump never returns (ogld#40).

        It reads the variant table's positions and nothing of the LD components: `clump` and
        `proxies` call it, and from 0.2.0 their only LD read is the Partner Rows (opengwas/ogld#71).
        This is a store-layer read, not a public query or a new caller setting. Geometry and
        chromosome label dtypes stay identical to the full variant-table read.
        """
        rows = self.checked_rows(rows)
        position = np.empty(len(rows), dtype=np.int64)
        for c, members in self._loci_by_chromosome(rows):
            position[members] = c.position[local_rows(rows[members], c.start)]
        names = np.array(self._names, dtype=np.str_)
        return names[chromosome_index(self._stops, rows)], position

    def variants(self, rows: npt.NDArray[np.int64]) -> Variants:
        """The Variant Set's columns at `rows`, in the order given (ld-store §2)."""
        rows = self.checked_rows(rows)
        chrom, position = self.loci_of(rows)
        frequency = elements_of(self._columns["a1_frequency"], self._opened.cache, rows)
        return Variants(
            chrom=chrom,
            position=position,
            alid=self._strings("alid", rows),
            a1=self._strings("a1", rows),
            a2=self._strings("a2", rows),
            a1_frequency=frequency.astype(np.float64),
        )

    def checked_rows(self, rows: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        """`rows`, if every one is a row of the Variant Set. An Absent ALID is −1 from `rows_of`
        and is no row: read as an index it would be the last variant."""
        rows = np.asarray(rows, dtype=np.int64)
        size = int(self._stops[-1]) if len(self._stops) else 0
        bad = (rows < 0) | (rows >= size)
        if bad.any():
            raise ValueError(
                f"rows {rows[bad][:5].tolist()} are outside the Variant Set of "
                f"{size} rows: an Absent ALID is not a row (ld-store §8)"
            )
        return rows

    def _loaded_loci(self, name: str) -> Loci:
        """`loci` with the lock held: the one load a chromosome's positions get."""
        loci = self._located.get(name)
        if loci is None:
            start, stop = self._ranges[name]
            position = self._columns["position"]
            loci = Loci(
                name,
                start,
                stop,
                runs_of(position, self._opened.cache, np.array([start]), np.array([stop - start]))[
                    0
                ].astype(np.int64),
            )
            self._located[name] = loci
        return loci

    def _loci_by_chromosome(
        self, rows: npt.NDArray[np.int64]
    ) -> Iterator[tuple[Loci, npt.NDArray[np.int64]]]:
        """Each chromosome `rows` touch, its rows and positions, and the positions in `rows` that
        are on it (opengwas/ogld#71), without opening any of its LD components."""
        which = chromosome_index(self._stops, rows)
        for index in np.unique(which):
            yield self.loci(self._names[index]), np.flatnonzero(which == index)

    def _strings(self, name: str, rows: npt.NDArray[np.int64]) -> npt.NDArray[np.str_]:
        """The string column `name` at `rows`; an empty string is a lost chunk's, never data."""
        # zarr hands back numpy StringDType, which casts to a fixed width only via a list.
        values = np.array(
            elements_of(self._columns[name], self._opened.cache, rows).tolist(), dtype=np.str_
        )
        return values


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
