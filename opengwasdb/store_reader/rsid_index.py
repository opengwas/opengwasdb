"""A Store Release's rsID index: each rsID's rows, and, inverted, each row's rsID.

Two plain `.npy` files beside the Store Variant Table (issue #109, spec §1, §4):

- ``variant_rsid_bytes.npy`` (``|S24``): every indexed rsID, sorted;
- ``variant_rsid_rows.npy`` (``<i4``): the row each one names.

An rsID that names several rows (a multi-allelic site) is in it once per row, in
ascending row order. A row whose table cell is blank is not in it. A row is in it at
most once, because the Variant Table gives a row one rsID cell.

**An rsID's rows** are a binary search of the sorted names (`rows_of`).

**A row's rsID** (`rsids`) inverts the index, as `alid_index.FixedWidthAlidIndex`
inverts the 0.2 series' ALID pair. On format 0.3.0 this is how the Store Reader names a
row's rsID without reading `variants.tsv.gz` (#469). The inversion is sized to what is
asked. Up to `_SCANNED_ROWS` rows are found by a chunked scan of the rows array, which
allocates nothing the size of the axis. More rows invert the whole index once, into
one `int32` per row of the axis, and keep that for every later call. Measured on
OGS-00004 (23,726,262 rows, 20,008,480 indexed): the scan takes 0.04 s for one row and
0.25 s for 32; the inversion takes 1.0 s and keeps 95 MB, after which any number of rows
is a gather (`scratch/469/invert_cost.txt`). The inversion refuses an index that names a row
twice, which would give that row whichever rsID came last.

This module imports numpy and the standard library only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from opengwasdb.store_reader.alid import RSID_DTYPE, is_indexable_rsid

RSID_BYTES_FILENAME = "variant_rsid_bytes.npy"
RSID_ROWS_FILENAME = "variant_rsid_rows.npy"

#: The rsID index's two files, names then rows.
RSID_INDEX_FILENAMES: tuple[str, str] = (RSID_BYTES_FILENAME, RSID_ROWS_FILENAME)

#: Up to this many distinct rows, `rsids` scans the rows array rather than inverting it.
_SCANNED_ROWS = 32

#: Entries of the rows array one step of the scan compares at once.
_SCAN_CHUNK = 1 << 20


class MissingRsidIndex(FileNotFoundError):
    """A release whose `format_version` names its rows' rsIDs from the rsID index lacks it."""


class MalformedRsidIndex(ValueError):
    """An rsID index that cannot name a row's rsID: two files of unequal length, or a
    row named twice or outside the axis."""


class RsidIndex:
    """A release's rsID index, memory-mapped."""

    def __init__(self, names: np.ndarray, rows: np.ndarray) -> None:
        if len(names) != len(rows):
            raise MalformedRsidIndex(
                f"{RSID_BYTES_FILENAME} has {len(names)} entries but {RSID_ROWS_FILENAME} "
                f"has {len(rows)}"
            )
        self.names = names
        self.rows = rows
        self._slot_of_row: np.ndarray | None = None  # row -> entry, built on first wide ask

    @classmethod
    def open(cls, store_path: str | Path) -> RsidIndex | None:
        """The index in `store_path`; None unless both of its files are there.

        A release built before issue #109 has neither, and resolves no rsID by name.
        """
        names_path, rows_path = (Path(store_path) / name for name in RSID_INDEX_FILENAMES)
        if not (names_path.exists() and rows_path.exists()):
            return None
        return cls(np.load(names_path, mmap_mode="r"), np.load(rows_path, mmap_mode="r"))

    def __len__(self) -> int:
        return len(self.names)

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
        """Each row's entry in the index, -1 for a row it does not name."""
        wanted = np.unique(rows)
        if self._slot_of_row is None and wanted.size <= _SCANNED_ROWS:
            return self._scanned_slots(rows, wanted)
        slots: np.ndarray = self._inverse(n_rows)[rows].astype(np.int64)
        return slots

    def _scanned_slots(self, rows: np.ndarray, wanted: np.ndarray) -> np.ndarray:
        """`_slots` by a chunked scan of the rows array for the few `wanted` rows."""
        slot_of = dict.fromkeys(wanted.tolist(), -1)
        for lo in range(0, len(self.rows), _SCAN_CHUNK):
            block = np.asarray(self.rows[lo : lo + _SCAN_CHUNK])
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
        if rows.size and (int(rows.min()) < 0 or int(rows.max()) >= n_rows):
            raise MalformedRsidIndex(
                f"the rsID index names rows {int(rows.min())}..{int(rows.max())}, outside "
                f"the axis of {n_rows}"
            )
        entries = np.arange(rows.size, dtype=np.int32)
        inverse = np.full(n_rows, -1, dtype=np.int32)
        inverse[rows] = entries
        if not np.array_equal(inverse[rows], entries):
            twice = int(rows[np.flatnonzero(inverse[rows] != entries)[0]])
            raise MalformedRsidIndex(f"the rsID index names row {twice} twice")
        self._slot_of_row = inverse
        return inverse
