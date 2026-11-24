"""The variable-length ALID sidecar: every row's ALID, and every ALID's row (format 0.3.0).

Four plain `.npy` files beside the Store Variant Table (ADR 0105, spec §1, §4):

- ``variant_alid_utf8.npy`` (``uint8``): every row's ALID, UTF-8, concatenated in
  row order;
- ``variant_alid_offsets.npy`` (``uint64``, one more than the rows): row ``i``'s
  ALID is ``utf8[offsets[i]:offsets[i + 1]]``, so naming a row is two reads, never
  an inversion of an index (#465);
- ``variant_alid_keys.npy`` (``|S24``): each ALID's first `KEY_WIDTH` bytes, sorted
  by the full ALID bytes;
- ``variant_alid_order.npy`` (``int32``): the row each key belongs to.

Finding an ALID's row is a `searchsorted` over the keys, vectorised for a batch, and
an exact compare on the full bytes wherever a key is not the whole ALID. Every ALID
is indexed, however long: the 0.2 series' fixed-width ``|S32`` index left out each
ALID wider than its slot (115,043 on OGS-00010, #127) and resolved those by a scan
of the Variant Table, which a 0.3.0 reader does not need.

This module imports numpy and the standard library only, so the Store Reader, the
validator and every writer share one statement of the layout. `write_alid_sidecar` is
the sidecar's one writer: a build reaches it through `write_variant_axis` (the axis's
writer), and the rewrite to 0.3.0 (`opengwasdb.store.rewrite`) calls it directly.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

from opengwasdb.store_reader.npy_sidecar import (
    encode_utf8_offsets,
    offsets_fault,
    open_sidecar_arrays,
    slice_texts,
)

ALID_UTF8_FILENAME = "variant_alid_utf8.npy"
ALID_OFFSETS_FILENAME = "variant_alid_offsets.npy"
ALID_KEYS_FILENAME = "variant_alid_keys.npy"
ALID_ORDER_FILENAME = "variant_alid_order.npy"

#: The four files, in the order `encode_alids` returns them.
ALID_SIDECAR_FILENAMES: tuple[str, ...] = (
    ALID_UTF8_FILENAME,
    ALID_OFFSETS_FILENAME,
    ALID_KEYS_FILENAME,
    ALID_ORDER_FILENAME,
)

#: Bytes of each ALID the sorted key holds. A key is a search aid, never an identity:
#: an ALID at least this long is confirmed on its full bytes. ALID length is mean 14.9
#: and p99 21 across the FinnGen, GWAS Catalog and metabolome pilots (`alid.ALID_WIDTH`),
#: so nearly every key is its whole ALID and nearly every search ends at the key.
KEY_WIDTH = 24
KEY_DTYPE = np.dtype(f"S{KEY_WIDTH}")
ORDER_DTYPE = np.dtype("<i4")

#: Rows a structural check reads at a time. `_keys_of` builds two `int64` arrays of
#: rows x `KEY_WIDTH` (19 MB each at 100,000 rows) and a boolean one of 2.4 MB.
_WINDOW = 100_000

#: What `AlidSidecar.search` returns for an ALID the release does not hold.
ABSENT = -1


class DuplicateAlidError(ValueError):
    """Two rows of one Variant Index carry the same ALID."""


@dataclass(frozen=True)
class EncodedAlids:
    """The four arrays of a sidecar, before they are written."""

    utf8: np.ndarray
    offsets: np.ndarray
    keys: np.ndarray
    order: np.ndarray


def encode_alids(alids: Sequence[str]) -> EncodedAlids:
    """The sidecar for `alids`, one per row in row order. Refuses an empty or repeated ALID."""
    encoded = [alid.encode("utf-8") for alid in alids]
    n_rows = len(encoded)
    if n_rows > np.iinfo(ORDER_DTYPE).max:
        raise ValueError(f"{n_rows} rows do not fit the sidecar's {ORDER_DTYPE} row order")
    utf8, offsets = encode_utf8_offsets(encoded)
    lengths = np.diff(offsets)
    if n_rows and int(lengths.min()) == 0:
        raise ValueError(f"row {int(np.argmin(lengths))} has an empty ALID")
    # The cast truncates each ALID to its key, which is what a key is; the full bytes
    # settle every tie (`_sorted_keys`) and every search (`AlidSidecar.rows_of`).
    keys, order = _sorted_keys(encoded)
    # Only equal keys can hide equal ALIDs, and each tied run is now in full-byte order.
    repeated = next(_unrising_pairs(keys, lambda slot: encoded[int(order[slot])]), None)
    if repeated is not None:
        raise DuplicateAlidError(
            f"ALID {repeated[0].decode('utf-8')!r} names more than one row; a Variant "
            "Index holds each canonical variant once"
        )
    return EncodedAlids(utf8=utf8, offsets=offsets, keys=keys, order=order)


def _sorted_keys(encoded: list[bytes]) -> tuple[np.ndarray, np.ndarray]:
    """The keys in full-byte order, and each key's row: sorted by key, then each tied
    run by its full bytes."""
    keys_by_row = np.array(encoded, dtype=KEY_DTYPE) if encoded else np.empty(0, KEY_DTYPE)
    order = np.argsort(keys_by_row, kind="stable").astype(ORDER_DTYPE)
    keys = keys_by_row[order]
    for start, stop in _tied_runs(keys):
        order[start:stop] = sorted(order[start:stop].tolist(), key=encoded.__getitem__)
    return keys, order


def _tied_runs(keys: np.ndarray) -> Iterator[tuple[int, int]]:
    """Each run of two or more equal adjacent keys, as ``[start, stop)``."""
    if len(keys) < 2:
        return
    tied = np.flatnonzero(keys[1:] == keys[:-1])
    if not tied.size:
        return
    breaks = np.flatnonzero(np.diff(tied) != 1)
    starts = np.concatenate(([tied[0]], tied[breaks + 1]))
    ends = np.concatenate((tied[breaks], [tied[-1]]))
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        yield start, end + 2


def _unrising_pairs(
    keys: np.ndarray, full_at_slot: Callable[[int], bytes]
) -> Iterator[tuple[bytes, bytes]]:
    """Each pair of neighbours sharing a key whose full ALIDs do not strictly rise.

    The one walk of the tied runs: the encoder reads a repeat in it as a duplicate it
    refuses, and the structural check reads either fault in it as a sidecar it fails.
    """
    for start, stop in _tied_runs(keys):
        run = [full_at_slot(slot) for slot in range(start, stop)]
        for before, after in zip(run, run[1:], strict=False):
            if not before < after:
                yield before, after


def write_alid_sidecar(store_path: str | Path, alids: Sequence[str]) -> None:
    """Write the four sidecar files for `alids` (one per row, in row order) into `store_path`."""
    encoded = encode_alids(alids)
    store = Path(store_path)
    arrays = (encoded.utf8, encoded.offsets, encoded.keys, encoded.order)
    for name, array in zip(ALID_SIDECAR_FILENAMES, arrays, strict=True):
        np.save(store / name, array)


class AlidSidecar:
    """A release's ALID sidecar, memory-mapped: each row's ALID, and each ALID's row.

    One of the two ALID indexes a Variant Axis opens (`opengwasdb.store_reader.alid_index`);
    it holds every ALID, so it never answers that one is unindexed.
    """

    FILENAMES: ClassVar[tuple[str, ...]] = ALID_SIDECAR_FILENAMES

    def __init__(
        self, utf8: np.ndarray, offsets: np.ndarray, keys: np.ndarray, order: np.ndarray
    ) -> None:
        self.utf8 = utf8
        self.offsets = offsets
        self.keys = keys
        self.order = order
        self._text = memoryview(np.asarray(utf8))

    @classmethod
    def open(cls, store_path: str | Path) -> AlidSidecar | None:
        """The sidecar in `store_path`; None when it holds none of the four files.

        A directory holding some of them and not the others is refused: a half-written
        sidecar answers some lookups and not others, which reads as absence.
        """
        arrays = open_sidecar_arrays(store_path, ALID_SIDECAR_FILENAMES, "ALID sidecar")
        return None if arrays is None else cls(*arrays)

    @property
    def n_rows(self) -> int:
        return len(self.offsets) - 1

    def alid(self, row: int) -> str:
        """Row `row`'s ALID."""
        start, stop = int(self.offsets[row]), int(self.offsets[row + 1])
        return str(self._text[start:stop], "utf-8")

    def alids(self, rows: np.ndarray) -> np.ndarray:
        """The ALIDs of `rows`, as an object array in `rows`' order; None for a row past
        the sidecar's end, which names no ALID there."""
        rows = np.asarray(rows, dtype=np.int64)
        out = np.full(rows.size, None, dtype=object)
        held = np.flatnonzero((rows >= 0) & (rows < self.n_rows))
        out[held] = slice_texts(self._text, self.offsets, rows[held])
        return out

    def search(self, alids: Sequence[str]) -> np.ndarray:
        """Each ALID's row, or `ABSENT`: the ALID index's question (`rows_of`)."""
        return self.rows_of(alids)

    def unindexed_rows(self, n_rows: int) -> np.ndarray:
        """The rows the index leaves out: none."""
        return np.empty(0, dtype=np.int64)

    def problems(self, table_rows: int, expected_indexed: int) -> Iterator[str]:
        """Its structure (`sidecar_problems`), and one row per row of the table: every
        row is indexed, so `expected_indexed` is `table_rows` and adds nothing."""
        found = list(sidecar_problems(self))
        yield from found
        if not found and self.n_rows != table_rows:
            yield (
                f"the ALID sidecar names {self.n_rows} rows but variants.tsv.gz has "
                f"{table_rows}"
            )

    def rows_of(self, alids: Sequence[str]) -> np.ndarray:
        """Each ALID's row, ``-1`` where the release holds none, in `alids`' order."""
        found = np.full(len(alids), ABSENT, dtype=np.int64)
        if not len(alids) or not len(self.keys):
            return found
        encoded = [alid.encode("utf-8") for alid in alids]
        queries = np.array(encoded, dtype=KEY_DTYPE)
        lo = np.searchsorted(self.keys, queries, side="left")
        hi = np.searchsorted(self.keys, queries, side="right")
        lengths = np.fromiter((len(item) for item in encoded), dtype=np.int64, count=len(encoded))
        # A query shorter than the key is its whole key, and only a stored ALID as short
        # has the same key: that one is the match, and there is no other (no duplicates).
        whole = (hi > lo) & (lengths < KEY_WIDTH)
        found[whole] = np.asarray(self.order)[lo[whole]]
        for i in np.flatnonzero((hi > lo) & (lengths >= KEY_WIDTH)).tolist():
            found[i] = self._row_in_run(int(lo[i]), int(hi[i]), encoded[i])
        return found

    def _row_in_run(self, lo: int, hi: int, query: bytes) -> int:
        """The row among slots ``[lo, hi)``, which share one key and run in full-byte
        order, whose ALID is `query`; -1 if none is."""
        at = bisect_left(range(lo, hi), query, key=self.alid_bytes_at_slot)
        if at < hi - lo and self.alid_bytes_at_slot(lo + at) == query:
            return int(self.order[lo + at])
        return ABSENT

    def alid_bytes_at_slot(self, slot: int) -> bytes:
        """The full ALID, as bytes, of the key at `slot` of the sorted keys."""
        row = int(self.order[slot])
        return bytes(self._text[int(self.offsets[row]) : int(self.offsets[row + 1])])


def sidecar_problems(sidecar: AlidSidecar) -> Iterator[str]:
    """What is wrong with the sidecar's own structure, if anything.

    Offsets start at 0, rise with every row and end at the byte array's end; the order
    names every row exactly once, so every ALID is indexed; each key is its row's
    ALID's first `KEY_WIDTH` bytes; and the keys run in the order of the full ALID
    bytes, strictly, so no ALID names two rows. Whether each row's ALID is the one the
    Variant Table gives it is the validator's comparison (`variant_table.alid_sidecar`).
    """
    n_rows = sidecar.n_rows
    fault = offsets_fault(
        sidecar.offsets,
        len(sidecar.utf8),
        names=(ALID_OFFSETS_FILENAME, ALID_UTF8_FILENAME),
        rising=("row", "every row has an ALID"),
    )
    if fault:
        yield fault
        return
    if len(sidecar.keys) != n_rows or len(sidecar.order) != n_rows:
        yield (
            f"variant_alid_keys.npy has {len(sidecar.keys)} keys and variant_alid_order.npy "
            f"{len(sidecar.order)} rows, for {n_rows} ALIDs"
        )
        return
    yield from _order_problems(sidecar)


def _order_problems(sidecar: AlidSidecar) -> Iterator[str]:
    order = np.asarray(sidecar.order, dtype=np.int64)
    n_rows = sidecar.n_rows
    if order.size and (int(order.min()) < 0 or int(order.max()) >= n_rows):
        yield f"variant_alid_order.npy names a row outside 0..{n_rows - 1}"
        return
    unindexed = int(np.count_nonzero(np.bincount(order, minlength=n_rows) == 0))
    if unindexed:
        yield (
            f"variant_alid_order.npy leaves {unindexed} row(s) out: an ALID the index lacks "
            "answers its own lookup with nothing"
        )
        return
    yield from _key_problems(sidecar, order)


def _key_problems(sidecar: AlidSidecar, order: np.ndarray) -> Iterator[str]:
    wrong = next(_wrong_keys(sidecar, order), None)
    if wrong is not None:
        yield wrong
        return
    yield from _unordered_keys(sidecar)


def _wrong_keys(sidecar: AlidSidecar, order: np.ndarray) -> Iterator[str]:
    """Each window's first key that is not its row's ALID's start, if any."""
    keys = sidecar.keys
    for start in range(0, len(order), _WINDOW):
        rows = order[start : start + _WINDOW]
        expected = _keys_of(sidecar, rows)
        wrong = np.flatnonzero(np.asarray(keys[start : start + _WINDOW]) != expected)
        if wrong.size:
            row = int(rows[wrong[0]])
            yield (
                f"variant_alid_keys.npy holds a key for row {row} that is not the start of "
                f"its ALID {sidecar.alid(row)!r} ({wrong.size} such key(s) in rows "
                f"{start}..{start + len(rows) - 1})"
            )


def _unordered_keys(sidecar: AlidSidecar) -> Iterator[str]:
    """The first place the keys, or the full ALIDs sharing one key, are not strictly rising."""
    keys = sidecar.keys
    if len(keys) > 1:
        falling = np.flatnonzero(np.asarray(keys[1:]) < np.asarray(keys[:-1]))
        if falling.size:
            yield f"variant_alid_keys.npy is not sorted at slot {int(falling[0]) + 1}"
            return
    fault = next(_unrising_pairs(np.asarray(keys), sidecar.alid_bytes_at_slot), None)
    if fault is not None:
        before, after = fault
        word = "names more than one row" if before == after else "is out of order"
        yield f"ALID {after.decode('utf-8', 'replace')!r} {word} in variant_alid_keys.npy"


def _keys_of(sidecar: AlidSidecar, rows: np.ndarray) -> np.ndarray:
    """The key each of `rows` should have: its ALID's first `KEY_WIDTH` bytes."""
    starts = np.asarray(sidecar.offsets[rows], dtype=np.int64)
    lengths = np.asarray(sidecar.offsets[rows + 1], dtype=np.int64) - starts
    columns = np.arange(KEY_WIDTH, dtype=np.int64)
    within = columns[None, :] < lengths[:, None]
    at = np.where(within, starts[:, None] + columns[None, :], 0)
    utf8 = np.asarray(sidecar.utf8)
    if not utf8.size:
        return np.empty(len(rows), dtype=KEY_DTYPE)
    gathered = np.where(within, utf8[at], 0).astype(np.uint8)
    return np.ascontiguousarray(gathered).view(KEY_DTYPE).reshape(len(rows))
