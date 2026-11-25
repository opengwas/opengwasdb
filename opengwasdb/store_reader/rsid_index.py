"""A Store Release's rsID index: each rsID's rows, and, inverted, each row's rsID.

Two plain `.npy` files beside the Store Variant Table (issue #109, spec §1, §4):

- ``variant_rsid_bytes.npy`` (``|S24``): every indexed rsID, sorted;
- ``variant_rsid_rows.npy`` (``<i4``): the row each one names.

An rsID that names several rows (a multi-allelic site) is in it once per row, in
ascending row order. A row whose table cell is blank is not in it. A row is in it at
most once, because the Variant Table gives a row one rsID cell.

**An rsID's rows** are a binary search of the sorted names (`rows_of`).

**A row's rsID** (`rsids`) inverts the index (`npy_sidecar.invert_rows`), as
`alid_index.FixedWidthAlidIndex` inverts the 0.2 series' ALID pair. On format 0.3.0
this is how the Store Reader names a row's rsID without reading `variants.tsv.gz`
(#469). Until it is inverted, a call asking for few rows scans the rows array in
chunks instead, which allocates nothing the size of the axis. The index inverts once
what the scans have cost would pass what one inversion costs (`_scan_cost`), and keeps
the inverse for every later call. Measured on OGS-00004 (23,726,262 rows, 20,008,480
indexed; `scratch/469/invert_cost.txt`): a scan takes 0.04 s for one row and 0.25 s for
32, and slows sharply past that (1.2 s for 128); the inversion takes 1.0 s and keeps
95 MB, after which any number of rows is a gather.

Both paths refuse an index that names a row twice, which would give that row whichever
rsID came last, and one that names a row outside the axis.

**Threads.** Two threads asking at once may both scan, or both invert: each builds the
same inverse, the second assignment replaces the first, and nothing is lost but the
work and, for a moment, a second copy of the inverse. No lock is held.

This module imports numpy and the standard library only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from opengwasdb.store_reader.alid import RSID_DTYPE, is_indexable_rsid
from opengwasdb.store_reader.npy_sidecar import invert_rows, open_pair_or_none

VARIANT_RSID_BYTES_FILENAME = "variant_rsid_bytes.npy"
VARIANT_RSID_ROWS_FILENAME = "variant_rsid_rows.npy"

#: The rsID index's two files, names then rows.
RSID_INDEX_FILENAMES: tuple[str, str] = (VARIANT_RSID_BYTES_FILENAME, VARIANT_RSID_ROWS_FILENAME)

#: More distinct rows than this in one call are never scanned: past it a scan slows
#: sharply (numpy's `isin` changes method), and one call would cost an inversion.
_SCANNED_ROWS = 32

#: What one scan costs, as a share of one inversion (measured on OGS-00004, above):
#: reading the rows array once, and each distinct row asked on top of that.
_SCAN_COST_PER_CALL = 0.04
_SCAN_COST_PER_ROW = 0.0066

#: Entries of the rows array one step of the scan compares at once.
_SCAN_CHUNK = 1 << 20


class MissingRsidIndex(FileNotFoundError):
    """A release whose `format_version` names its rows' rsIDs from the rsID index lacks it."""


class MalformedRsidIndex(ValueError):
    """An rsID index that cannot name a row's rsID: two files of unequal length, or a
    row named twice or outside the axis."""


def _scan_cost(n_wanted: int) -> float:
    """What scanning for `n_wanted` distinct rows costs, as a share of one inversion."""
    return _SCAN_COST_PER_CALL + _SCAN_COST_PER_ROW * n_wanted


class RsidIndex:
    """A release's rsID index, memory-mapped."""

    def __init__(self, names: np.ndarray, rows: np.ndarray) -> None:
        if len(names) != len(rows):
            raise MalformedRsidIndex(
                f"{VARIANT_RSID_BYTES_FILENAME} has {len(names)} entries but "
                f"{VARIANT_RSID_ROWS_FILENAME} has {len(rows)}"
            )
        self.names = names
        self.rows = rows
        self._slot_of_row: np.ndarray | None = None  # row -> entry, once inverted
        self._scanned = 0.0  # what the scans so far have cost, in inversions

    @classmethod
    def open(cls, store_path: str | Path) -> RsidIndex | None:
        """The index in `store_path`; None unless both of its files are there.

        A release built before issue #109 has neither, and resolves no rsID by name.
        """
        arrays = open_pair_or_none(store_path, RSID_INDEX_FILENAMES)
        return None if arrays is None else cls(*arrays)

    def __len__(self) -> int:
        return len(self.names)

    @property
    def inverted(self) -> bool:
        """Whether the index has been inverted, so naming any row is a gather."""
        return self._slot_of_row is not None

    def rows_of(self, rsid: str) -> np.ndarray:
        """Every row `rsid` names, ascending, as int32; none for a name the index cannot
        hold (`is_indexable_rsid`: blank, or wider than its slot, which would be
        truncated into a false match) or does not."""
        if not len(self.names) or not is_indexable_rsid(rsid):
            return np.empty(0, dtype="int32")
        query = np.array(rsid, dtype=RSID_DTYPE)
        lo = int(np.searchsorted(self.names, query, side="left"))
        hi = int(np.searchsorted(self.names, query, side="right"))
        return np.asarray(self.rows[lo:hi], dtype="int32")

    def rsids(self, rows: np.ndarray, n_rows: int) -> np.ndarray:
        """Each of `rows`' rsID, in `rows`' order, as an object array; None for a row the
        index does not name. `n_rows` is the axis's length: a row outside it is refused."""
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        out = np.full(rows.size, None, dtype=object)
        if not rows.size:
            return out
        outside = (rows < 0) | (rows >= n_rows)
        if outside.any():
            raise IndexError(
                f"row {int(rows[np.flatnonzero(outside)[0]])} is outside the axis of {n_rows}"
            )
        slots = self._slots(rows, n_rows)
        named = np.flatnonzero(slots >= 0)
        if named.size:
            names = np.asarray(self.names[slots[named]])
            out[named] = [bytes(name).decode("utf-8") for name in names]
        return out

    def _slots(self, rows: np.ndarray, n_rows: int) -> np.ndarray:
        """Each row's entry in the index, -1 for a row it does not name: scanned while
        the scans so far and this one cost less than one inversion, else inverted."""
        wanted = np.unique(rows)
        cost = _scan_cost(wanted.size)
        if (
            self._slot_of_row is None
            and wanted.size <= _SCANNED_ROWS
            and self._scanned + cost <= 1.0
        ):
            self._scanned += cost
            return self._scanned_slots(rows, wanted, n_rows)
        slots: np.ndarray = self._inverse(n_rows)[rows].astype(np.int64)
        return slots

    def _scanned_slots(self, rows: np.ndarray, wanted: np.ndarray, n_rows: int) -> np.ndarray:
        """`_slots` by a chunked scan of the rows array for the few `wanted` rows, which
        refuses what the inversion refuses: a row named twice, or outside the axis."""
        slot_of = dict.fromkeys(wanted.tolist(), -1)
        for lo in range(0, len(self.rows), _SCAN_CHUNK):
            block = np.asarray(self.rows[lo : lo + _SCAN_CHUNK])
            _refuse_outside(block, n_rows)
            for at in np.flatnonzero(np.isin(block, wanted)).tolist():
                row = int(block[at])
                if slot_of[row] >= 0:
                    raise MalformedRsidIndex(f"the rsID index names row {row} twice")
                slot_of[row] = lo + at
        return np.fromiter((slot_of[int(row)] for row in rows), dtype=np.int64, count=rows.size)

    def _inverse(self, n_rows: int) -> np.ndarray:
        """Each row's entry, -1 for a row not named: the index inverted, once."""
        if self._slot_of_row is not None:
            return self._slot_of_row
        rows = np.asarray(self.rows, dtype=np.int64)
        _refuse_outside(rows, n_rows)
        inverse = invert_rows(rows, n_rows, np.int32)
        entries = np.arange(rows.size, dtype=np.int32)
        if not np.array_equal(inverse[rows], entries):
            twice = int(rows[np.flatnonzero(inverse[rows] != entries)[0]])
            raise MalformedRsidIndex(f"the rsID index names row {twice} twice")
        self._slot_of_row = inverse
        return inverse


def _refuse_outside(rows: np.ndarray, n_rows: int) -> None:
    """Refuse index rows outside an axis of `n_rows`."""
    if rows.size and (int(rows.min()) < 0 or int(rows.max()) >= n_rows):
        raise MalformedRsidIndex(
            f"the rsID index names rows {int(rows.min())}..{int(rows.max())}, outside "
            f"the axis of {n_rows}"
        )
