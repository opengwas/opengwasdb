"""A Store Release's ALID index: the one place that knows there are two.

A Variant Axis asks its ALID index three things: the row of each ALID, the ALID of
each row, and which rows the index leaves out. Two indexes answer them:

- `AlidSidecar` (format 0.3.0, ADR 0105): the variable-length sidecar, which holds
  every ALID;
- `FixedWidthAlidIndex` (the 0.2 and 0.1 series): `variant_alid_bytes.npy` /
  `variant_alid_rows.npy`, which leaves out each ALID wider than `ALID_WIDTH` (#127).
  For such an ALID it answers `UNINDEXED`, and the caller reads the Variant Table.

The choice between them is made here and only here, by one key: the release's
`format_version` (`alid_index_kind`). The Variant Axis opens that kind and no other
(`open_alid_index`), and refuses a release whose files for it are missing; a stray
sidecar beside a 0.2 release's pair is never read (`envelope.closed` reports it). The
envelope and its required entries ask the same function for the kind's file names.
#476 deletes `FixedWidthAlidIndex`, its entry in `ALID_INDEX_KINDS`, the branch in
`alid_index_kind`, and `UNINDEXED` with the Variant Axis's scan for it.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import ClassVar, Protocol

import numpy as np

from opengwasdb.store_reader.alid import ALID_DTYPE, is_indexable_alid
from opengwasdb.store_reader.alid_sidecar import ABSENT, AlidSidecar
from opengwasdb.store_reader.format import carries_alid_sidecar

__all__ = [
    "ABSENT",
    "AlidIndexKind",
    "MissingAlidIndex",
    "ALID_INDEX_FILENAMES",
    "ALID_INDEX_KINDS",
    "UNINDEXED",
    "AlidIndex",
    "FixedWidthAlidIndex",
    "alid_index_kind",
    "open_alid_index",
]

#: What `AlidIndex.search` returns for an ALID the index cannot be asked about: one
#: wider than the 0.2 series' fixed slot. Only the Variant Table can say where it is.
UNINDEXED = -2


class MissingAlidIndex(FileNotFoundError):
    """A release's `format_version` carries an ALID index whose files are not there."""


class AlidIndex(Protocol):
    """What a Variant Axis, and the validator, ask of a release's ALID index."""

    FILENAMES: ClassVar[tuple[str, ...]]

    def search(self, alids: Sequence[str]) -> np.ndarray:
        """Each ALID's row, `ABSENT`, or `UNINDEXED`, in `alids`' order."""
        ...

    def alids(self, rows: np.ndarray) -> np.ndarray:
        """Each row's ALID as an object array; None for a row the index leaves out."""
        ...

    def unindexed_rows(self, n_rows: int) -> np.ndarray:
        """The rows, of `n_rows`, whose ALID the index leaves out."""
        ...

    def problems(self, table_rows: int, expected_indexed: int) -> Iterator[str]:
        """What is wrong with the index for a Variant Table of `table_rows` rows, of which
        `expected_indexed` should be indexed; nothing if it is sound."""
        ...


class AlidIndexKind(Protocol):
    """One kind of ALID index, as a class: its files, and how to open it."""

    @property
    def FILENAMES(self) -> tuple[str, ...]:
        ...

    def open(self, store_path: str | Path) -> AlidIndex | None:
        """The index in `store_path`; None when none of its files is there."""
        ...


#: Keys one collision window compares: 1,000,000 32-byte keys is 32 MB.
_KEY_WINDOW = 1_000_000


class FixedWidthAlidIndex:
    """The 0.2 series' ALID index: `|S32` keys sorted, and each key's row (#127, #029)."""

    FILENAMES: ClassVar[tuple[str, ...]] = ("variant_alid_bytes.npy", "variant_alid_rows.npy")

    def __init__(self, keys: np.ndarray, rows: np.ndarray) -> None:
        self.keys = keys
        self.rows = rows
        self._inverse: np.ndarray | None = None  # row -> key slot, built on first `alids`

    @classmethod
    def open(cls, store_path: str | Path) -> FixedWidthAlidIndex | None:
        keys_path, rows_path = (Path(store_path) / name for name in cls.FILENAMES)
        if not (keys_path.exists() and rows_path.exists()):
            return None
        return cls(np.load(keys_path, mmap_mode="r"), np.load(rows_path, mmap_mode="r"))

    def search(self, alids: Sequence[str]) -> np.ndarray:
        found = np.full(len(alids), ABSENT, dtype=np.int64)
        indexable = np.array([is_indexable_alid(alid) for alid in alids], dtype=bool)
        found[~indexable] = UNINDEXED
        at = np.flatnonzero(indexable)
        if not (at.size and len(self.keys)):
            return found
        # Never cast an over-wide ALID into the slot: truncated, it would match another
        # variant's key (#127). Only the indexable ones are asked.
        queries = np.array([alids[int(i)] for i in at], dtype=ALID_DTYPE)
        slot = np.minimum(np.searchsorted(self.keys, queries), len(self.keys) - 1)
        hit = np.asarray(self.keys)[slot] == queries
        found[at[hit]] = np.asarray(self.rows, dtype=np.int64)[slot[hit]]
        return found

    def alids(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        out = np.full(rows.size, None, dtype=object)
        if not rows.size:
            return out
        slot = self._inverse_of(int(rows.max()) + 1)[rows]
        indexed = slot >= 0
        out[indexed] = [bytes(b).decode("utf-8") for b in np.asarray(self.keys)[slot[indexed]]]
        return out

    def unindexed_rows(self, n_rows: int) -> np.ndarray:
        indexed = np.zeros(n_rows, dtype=bool)
        indexed[np.asarray(self.rows, dtype=np.int64)] = True
        return np.flatnonzero(~indexed)

    def problems(self, table_rows: int, expected_indexed: int) -> Iterator[str]:
        """One key per ALID that fits the slot, and no key shared by two variants."""
        # One entry per *indexable* ALID, not per variant: an ALID too wide for the
        # slot is left out rather than truncated into a key it shares with another
        # variant (issue #127). Those resolve by exact scan.
        if len(self.keys) != expected_indexed:
            yield (
                f"variant_alid_bytes.npy has {len(self.keys)} entries but "
                f"variants.tsv.gz has {expected_indexed} ALID(s) narrow enough to index "
                f"(of {table_rows} rows)"
            )
        collisions = self.repeated_keys()
        if collisions:
            yield (
                f"variant_alid_bytes.npy holds {collisions} key(s) shared by more than "
                "one variant: a lookup by one variant's ALID returns another's row. "
                "This is what silently truncating an over-wide ALID produced before "
                "issue #127 — rebuild the store"
            )

    def repeated_keys(self) -> int:
        """How many keys equal the key before them, read a window at a time so the
        index -- 387 MB on OGS-00006 -- is never held whole."""
        n_keys = len(self.keys)
        repeats = 0
        for lo in range(0, max(n_keys - 1, 0), _KEY_WINDOW):
            window = np.array(self.keys[lo : lo + _KEY_WINDOW + 1])
            repeats += int(np.count_nonzero(window[1:] == window[:-1]))
        return repeats

    def _inverse_of(self, at_least: int) -> np.ndarray:
        """Each row's key slot, -1 for a row left out: the pair inverted, once."""
        rows = np.asarray(self.rows, dtype=np.int64)
        size = max(at_least, int(rows.max()) + 1 if rows.size else 0)
        if self._inverse is None or len(self._inverse) < size:
            inverse = np.full(size, -1, dtype=np.int64)
            inverse[rows] = np.arange(rows.size, dtype=np.int64)
            self._inverse = inverse
        return self._inverse


#: The ALID indexes a release can carry, newest first.
ALID_INDEX_KINDS: tuple[AlidIndexKind, ...] = (AlidSidecar, FixedWidthAlidIndex)

#: Every file any ALID index is stored in.
ALID_INDEX_FILENAMES: frozenset[str] = frozenset(
    name for kind in ALID_INDEX_KINDS for name in kind.FILENAMES
)


def alid_index_kind(format_version: str) -> AlidIndexKind:
    """The ALID index a release of `format_version` carries (spec §1, §4): the one key."""
    return AlidSidecar if carries_alid_sidecar(format_version) else FixedWidthAlidIndex


def open_alid_index(store_path: str | Path, format_version: str) -> AlidIndex:
    """The ALID index a release of `format_version` carries, opened from `store_path`.

    Only that kind is opened: what other files lie beside it is the envelope's
    question, never the reader's. Every readable version carries an index, so a
    release whose index files are missing is refused rather than read by scanning.
    """
    kind = alid_index_kind(format_version)
    opened = kind.open(store_path)
    if opened is None:
        raise MissingAlidIndex(
            f"{store_path}: format_version {format_version} carries an ALID index in "
            f"{', '.join(kind.FILENAMES)}, which is missing; the release is incomplete. "
            "Rebuild it, or rewrite it from an intact source (spec §1, §4)"
        )
    return opened
