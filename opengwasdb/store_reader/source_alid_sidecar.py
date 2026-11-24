"""The `source_alid` sidecar: each row's pre-liftover ALID, in row order (format 0.3.0).

A row's `source_alid` is the ALID its source spelt it by, on the source's Reference
Assembly: the provenance link back to the dataset a lifted row came from (#470, ADR
0105 §8). Every build writes an unlifted row's `source_alid` as its own ALID; a row
two source variants collided onto, and a row Reference Completion added from the
panel, have none. The 0.2 series holds it in the Variant Table's eighth column only.

Three plain `.npy` files beside the Store Variant Table (spec §1, §4):

- ``variant_source_alid_state.npy`` (``|u1``, one per row): `NONE` (0), the row has no
  `source_alid`; `OWN_ALID` (1), it is the row's own ALID, which the ALID sidecar
  holds; `OTHER` (2), it is another ALID, held here;
- ``variant_source_alid_utf8.npy`` (``uint8``): each `OTHER` row's `source_alid`,
  UTF-8, concatenated in row order;
- ``variant_source_alid_offsets.npy`` (``<u8``, one more than the `OTHER` rows): the
  ``k``-th `OTHER` row's is ``utf8[offsets[k]:offsets[k + 1]]``.

Most rows of a release built on its own assembly name their own ALID. Stored whole
they would repeat the ALID sidecar's text: 489 MB on OGS-00003's 21.3 M rows, 1.34
times its Variant Table (ADR 0105 §8). The state costs one byte a row, and only a
lifted row's text is stored, whole however long.

An `OTHER` row's text is its rank among the `OTHER` rows. The reader finds it by a
binary search of the `OTHER` rows' numbers, which it gathers once, on first use, from
the state (`SourceAlidSidecar.lifted_rows`): an array the size of the lifted rows, not
of the axis, and nothing on disk. A cumulative count over every row would be an 8-byte
array a row, which is what the layout avoids.

This module imports numpy and the standard library only, as every codec does. It does
not read the ALID sidecar: a caller that wants an `OWN_ALID` row's text passes the
ALIDs in. `write_source_alid_sidecar` is the sidecar's one writer: a build or a
completion reaches it through `write_variant_axis`, and the rewrite to 0.3.0
(`opengwasdb.store.rewrite`) calls it directly.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

from opengwasdb.store_reader.npy_sidecar import holds_sidecar, open_sidecar_arrays

SOURCE_ALID_STATE_FILENAME = "variant_source_alid_state.npy"
SOURCE_ALID_UTF8_FILENAME = "variant_source_alid_utf8.npy"
SOURCE_ALID_OFFSETS_FILENAME = "variant_source_alid_offsets.npy"

#: The three files, in the order `encode_source_alids` returns them.
SOURCE_ALID_SIDECAR_FILENAMES: tuple[str, ...] = (
    SOURCE_ALID_STATE_FILENAME,
    SOURCE_ALID_UTF8_FILENAME,
    SOURCE_ALID_OFFSETS_FILENAME,
)

#: A row's state: no `source_alid`, its own ALID, or another ALID held in the text.
NONE, OWN_ALID, OTHER = 0, 1, 2

STATE_DTYPE = np.dtype("|u1")
UTF8_DTYPE = np.dtype("uint8")
OFFSET_DTYPE = np.dtype("<u8")

#: Rows a scan or a comparison reads at a time: `rows_of` gathers a window's values
#: of one length into a rows x length array of `int64` positions.
_WINDOW = 65_536

#: Characters a Variant Table cell cannot hold, so a `source_alid` holding one could
#: never agree with the table it restates: the column and row separators, and NUL.
_UNWRITABLE = ("\t", "\n", "\r", "\x00")

#: The table's blank cell. A `source_alid` spelt so would read back from the table as
#: none and from the sidecar as ``"."``: one value, two answers.
_TABLE_BLANK = "."

_WHAT = "source_alid sidecar"

#: Each row's ALID, for the rows given: what an `OWN_ALID` row's `source_alid` is.
OwnAlids = Callable[[np.ndarray], np.ndarray]

#: The row each ALID names, -1 where none: how an `OWN_ALID` row is found by value.
RowsOfAlids = Callable[[Sequence[str]], np.ndarray]


class UnencodableSourceAlidError(ValueError):
    """A `source_alid` the sidecar and the Variant Table could not both hold as given."""


class MissingSourceAlidSidecar(FileNotFoundError):
    """A release's `format_version` carries a `source_alid` sidecar whose files are not there."""


class MalformedSourceAlidSidecar(ValueError):
    """A sidecar that cannot be read as it stands: offsets that do not start at 0 or end
    at the byte array's end, an array not of its dtype, a state outside 0-2, or offsets
    that do not number the `OTHER` rows plus one."""


@dataclass(frozen=True)
class EncodedSourceAlids:
    """The three arrays of a sidecar, before they are written."""

    state: np.ndarray
    utf8: np.ndarray
    offsets: np.ndarray


def state_of(value: str | None, alid: str) -> int:
    """The state of a row whose ALID is `alid` and whose `source_alid` is `value`."""
    if not value:
        return NONE
    return OWN_ALID if value == alid else OTHER


def encode_source_alids(values: Sequence[str | None], alids: Sequence[str]) -> EncodedSourceAlids:
    """The sidecar for `values`, one per row in row order, beside each row's ALID in
    `alids`; None or ``""`` is none.

    Refuses a value the table could not hold as the sidecar does (`_UNWRITABLE`), the
    table's blank ``"."``, and one that is not text UTF-8 can spell (a lone surrogate).
    """
    if len(values) != len(alids):
        raise ValueError(f"{len(values)} source_alids for {len(alids)} rows: one each per row")
    state = np.fromiter(
        (state_of(value, alid) for value, alid in zip(values, alids, strict=True)),
        dtype=STATE_DTYPE,
        count=len(values),
    )
    encoded = [
        _encoded(int(row), values[int(row)]) for row in np.flatnonzero(state == OTHER).tolist()
    ]
    for row in np.flatnonzero(state == OWN_ALID).tolist():
        _encoded(int(row), values[int(row)])  # refused alike, though not stored
    lengths = np.fromiter((len(item) for item in encoded), dtype=OFFSET_DTYPE, count=len(encoded))
    offsets = np.zeros(len(encoded) + 1, dtype=OFFSET_DTYPE)
    np.cumsum(lengths, out=offsets[1:])
    utf8 = np.frombuffer(b"".join(encoded), dtype=UTF8_DTYPE)
    return EncodedSourceAlids(state=state, utf8=utf8, offsets=offsets)


def _encoded(row: int, value: str | None) -> bytes:
    assert value, "only a row with a source_alid is encoded"
    if value == _TABLE_BLANK or any(char in value for char in _UNWRITABLE):
        raise UnencodableSourceAlidError(
            f"row {row}'s source_alid {value!r} cannot be written: the Variant Table would "
            "read it as another value than the sidecar holds"
        )
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise UnencodableSourceAlidError(
            f"row {row}'s source_alid {value!r} is not text UTF-8 can spell"
        ) from exc


def write_source_alid_sidecar(
    store_path: str | Path, values: Sequence[str | None], alids: Sequence[str]
) -> None:
    """Write the three sidecar files for `values` (one per row, in row order, beside
    each row's ALID in `alids`) into `store_path`."""
    encoded = encode_source_alids(values, alids)
    store = Path(store_path)
    arrays = (encoded.state, encoded.utf8, encoded.offsets)
    for name, array in zip(SOURCE_ALID_SIDECAR_FILENAMES, arrays, strict=True):
        np.save(store / name, array)


class SourceAlidSidecar:
    """A release's `source_alid` sidecar, memory-mapped: each row's state, and each
    lifted row's `source_alid`."""

    FILENAMES: ClassVar[tuple[str, ...]] = SOURCE_ALID_SIDECAR_FILENAMES

    def __init__(self, state: np.ndarray, utf8: np.ndarray, offsets: np.ndarray) -> None:
        """Refuses offsets that cannot slice the bytes (`MalformedSourceAlidSidecar`):
        read as they stand, they would name a row by another row's bytes, or by bytes
        past the end. Three reads of the offsets; whether they number the `OTHER` rows
        is checked when those are first gathered (`lifted_rows`)."""
        self.state = state
        self.utf8 = utf8
        self.offsets = offsets
        fault = _slicing_fault(state, utf8, offsets)
        if fault:
            raise MalformedSourceAlidSidecar(f"the {_WHAT} cannot be read: {fault}")
        self._text = memoryview(np.asarray(utf8))
        self._lifted: np.ndarray | None = None
        self._others: np.ndarray | None = None

    @classmethod
    def open(cls, store_path: str | Path) -> SourceAlidSidecar | None:
        """The sidecar in `store_path`; None when it holds none of its files. A directory
        holding some and not the others is refused (`npy_sidecar.holds_sidecar`)."""
        arrays = open_sidecar_arrays(store_path, SOURCE_ALID_SIDECAR_FILENAMES, _WHAT)
        return None if arrays is None else cls(*arrays)

    @staticmethod
    def held_in(store_path: str | Path) -> bool:
        """Whether `store_path` holds the sidecar's files, without reading them; a
        partial set is refused."""
        return holds_sidecar(store_path, SOURCE_ALID_SIDECAR_FILENAMES, _WHAT)

    @property
    def n_rows(self) -> int:
        return len(self.state)

    def lifted_rows(self) -> np.ndarray:
        """The `OTHER` rows, ascending: the `k`-th holds the `k`-th text. Gathered once,
        from the state; refused unless the offsets number them plus one, or if a state
        is outside 0-2."""
        if self._lifted is None:
            fault = _state_fault(self.state) or _count_fault(self.state, self.offsets)
            if fault:
                raise MalformedSourceAlidSidecar(f"the {_WHAT} cannot be read: {fault}")
            self._lifted = self._other_rows()
        return self._lifted

    def _other_rows(self) -> np.ndarray:
        """The `OTHER` rows, ascending, whatever else the state holds."""
        if self._others is None:
            self._others = np.flatnonzero(np.asarray(self.state) == OTHER)
        return self._others

    def values(self, rows: np.ndarray | Sequence[int], own_alids: OwnAlids) -> np.ndarray:
        """Each row's `source_alid`, as an object array in `rows`' order; None for a
        row that has none. `own_alids` gives an `OWN_ALID` row's ALID. A row outside
        the sidecar is refused, never read as none."""
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        if rows.size and (int(rows.min()) < 0 or int(rows.max()) >= self.n_rows):
            outside = int(rows[(rows < 0) | (rows >= self.n_rows)][0])
            raise IndexError(f"row {outside} is outside the {_WHAT}'s {self.n_rows} rows")
        lifted = self.lifted_rows()
        state = np.asarray(self.state[rows])
        out = np.full(rows.size, None, dtype=object)
        own = state == OWN_ALID
        if own.any():
            out[own] = own_alids(rows[own])
        other = np.flatnonzero(state == OTHER)
        ranks = np.searchsorted(lifted, rows[other])
        out[other] = self.texts(ranks)
        return out

    def texts(self, ranks: np.ndarray) -> list[str]:
        """The `source_alid` of each of the `OTHER` rows at `ranks` among them."""
        ranks = np.asarray(ranks, dtype=np.int64)
        starts = np.asarray(self.offsets[ranks], dtype=np.int64).tolist()
        stops = np.asarray(self.offsets[ranks + 1], dtype=np.int64).tolist()
        text = self._text
        return [str(text[a:b], "utf-8") for a, b in zip(starts, stops, strict=True)]

    def sourced(self, start: int, stop: int) -> int:
        """How many rows in ``[start, stop)`` have a `source_alid`."""
        return int(np.count_nonzero(np.asarray(self.state[start : min(stop, self.n_rows)])))

    def differing(
        self, start: int, values: Sequence[str | None], alids: Sequence[str]
    ) -> list[int]:
        """The rows from `start`, one per given value (None is none) and ALID, whose
        stored state, or whose stored text for a lifted row, is another, ascending.

        The one comparison of the sidecar against the Variant Table: the rewrite's
        read-back and the validator's `variant_table.source_alid_sidecar` both make it.
        A row past the sidecar's end is not compared; `n_rows` says the sidecar is short.
        """
        held = max(min(start + len(values), self.n_rows) - start, 0)
        stored = np.asarray(self.state[start : start + held])
        expected = np.fromiter(
            (state_of(values[k], alids[k]) for k in range(held)), dtype=np.int64, count=held
        )
        wrong = set(np.flatnonzero(stored != expected).tolist())
        both = np.flatnonzero((stored == OTHER) & (expected == OTHER))
        wrong |= self._wrong_texts(start, both, values)
        return [start + k for k in sorted(wrong)]

    def _wrong_texts(self, start: int, both: np.ndarray, values: Sequence[str | None]) -> set[int]:
        """Of the lifted rows `start + both`, those whose text is not `values`'. Read
        without `lifted_rows`' checks, so a malformed sidecar is compared, not refused:
        a row whose rank the offsets do not hold is wrong."""
        ranks = np.searchsorted(self._other_rows(), start + both)
        held = ranks < len(self.offsets) - 1
        wrong = {int(k) for k in both[~held].tolist()}
        texts = self.texts(ranks[held])
        return wrong | {
            int(k) for k, text in zip(both[held].tolist(), texts, strict=True) if text != values[k]
        }

    def rows_of(self, wanted: Iterable[str], rows_of_alids: RowsOfAlids) -> dict[str, int]:
        """The first row whose `source_alid` is each of `wanted`; one no row has is
        absent. An `OWN_ALID` row is found by its ALID (`rows_of_alids`, the ALID
        index's search); an `OTHER` row by one scan of the text, a window at a time,
        that stops once each is found."""
        queries = [value for value in set(wanted) if value]
        found = self._own_rows_of(queries, rows_of_alids)
        lifted = self.lifted_rows()
        for value, rank in self._lifted_ranks_of(queries):
            row = int(lifted[rank])
            found[value] = min(found.get(value, row), row)
        return found

    def _own_rows_of(self, queries: list[str], rows_of_alids: RowsOfAlids) -> dict[str, int]:
        """The row of each query that is an `OWN_ALID` row's ALID."""
        rows = np.asarray(rows_of_alids(queries)).tolist()
        return {
            value: int(row)
            for value, row in zip(queries, rows, strict=True)
            if row >= 0 and int(self.state[row]) == OWN_ALID
        }

    def _lifted_ranks_of(self, queries: list[str]) -> Iterator[tuple[str, int]]:
        """Each query's first rank among the `OTHER` rows' texts: one scan, a window at a
        time, that stops once each is found."""
        remaining = {value.encode("utf-8"): value for value in queries}
        n_lifted = len(self.lifted_rows())
        for start in range(0, n_lifted, _WINDOW):
            if not remaining:
                return
            for rank, query in self._matches(start, min(start + _WINDOW, n_lifted), remaining):
                yield remaining.pop(query), rank

    def _matches(
        self, start: int, stop: int, wanted: dict[bytes, str]
    ) -> Iterator[tuple[int, bytes]]:
        """Each wanted value's first text in ranks ``[start, stop)``, as ``(rank, value)``."""
        bounds = np.asarray(self.offsets[start : stop + 1], dtype=np.int64)
        chunk = np.asarray(self.utf8[int(bounds[0]) : int(bounds[-1])])
        first: dict[bytes, int] = {}
        for length in sorted({len(query) for query in wanted}):
            queries = [query for query in wanted if len(query) == length]
            for rank, value in _ranks_holding(chunk, bounds, length, queries):
                first.setdefault(value, start + rank)
        for value, rank in sorted(first.items(), key=lambda item: item[1]):
            yield rank, value

    def problems(self, table_rows: int) -> Iterator[str]:
        """Its structure, and one row per row of the table.

        Each array is 1-d of its dtype; every state is 0, 1 or 2; the offsets number
        the `OTHER` rows plus one and rise with each, so text is held for those rows
        alone and none is empty. That they start at 0 and end at the byte array's end
        is checked at open. Whether each row's state and text are the table's is
        `differing`'s.
        """
        fault = (
            _dtype_fault(self.state, self.utf8, self.offsets)
            or _state_fault(self.state)
            or _count_fault(self.state, self.offsets)
        )
        if fault:
            yield fault
            return
        falling = _first_not_rising(self.offsets)
        if falling is not None:
            yield (
                f"{SOURCE_ALID_OFFSETS_FILENAME} does not rise at lifted row {falling}: "
                "each holds a source_alid"
            )
            return
        if self.n_rows != table_rows:
            yield f"the {_WHAT} holds {self.n_rows} rows but variants.tsv.gz has {table_rows}"


def _ranks_holding(
    chunk: np.ndarray, bounds: np.ndarray, length: int, queries: list[bytes]
) -> Iterator[tuple[int, bytes]]:
    """Each text of a window, ascending, that is one of `queries`, all `length` bytes
    long: ``(rank within the window, value)``. `chunk` is the window's bytes and
    `bounds` its texts' offsets."""
    ranks = np.flatnonzero(np.diff(bounds) == length)
    if not ranks.size:
        return
    at = (bounds[ranks] - bounds[0])[:, None] + np.arange(length, dtype=np.int64)
    stored = np.ascontiguousarray(chunk[at]).view(f"S{length}").reshape(ranks.size)
    hits = np.flatnonzero(np.isin(stored, np.array(queries, dtype=f"S{length}")))
    wanted = set(queries)
    for k in hits.tolist():
        value = chunk[at[k]].tobytes()  # exact: an `S` compare ignores trailing NULs
        if value in wanted:
            yield int(ranks[k]), value


def _slicing_fault(state: np.ndarray, utf8: np.ndarray, offsets: np.ndarray) -> str:
    """Why `offsets` cannot slice `utf8` at all, or ``""``."""
    fault = _dtype_fault(state, utf8, offsets)
    if fault:
        return fault
    if not len(offsets):
        return (
            f"{SOURCE_ALID_OFFSETS_FILENAME} is empty; it holds one more entry than the "
            "lifted rows"
        )
    first, last = int(offsets[0]), int(offsets[-1])
    if first != 0 or last != len(utf8):
        return (
            f"{SOURCE_ALID_OFFSETS_FILENAME} runs from {first} to {last}, not from 0 to the "
            f"{len(utf8)} bytes of {SOURCE_ALID_UTF8_FILENAME}"
        )
    return ""


def _dtype_fault(state: np.ndarray, utf8: np.ndarray, offsets: np.ndarray) -> str:
    """The first array not 1-d of the dtype the layout gives it, or ``""``."""
    expected = (
        (SOURCE_ALID_STATE_FILENAME, state, STATE_DTYPE),
        (SOURCE_ALID_UTF8_FILENAME, utf8, UTF8_DTYPE),
        (SOURCE_ALID_OFFSETS_FILENAME, offsets, OFFSET_DTYPE),
    )
    for name, array, dtype in expected:
        if array.dtype != dtype or array.ndim != 1:
            return f"{name} is {array.dtype} of {array.ndim} dimension(s), not 1-d {dtype}"
    return ""


def _state_fault(state: np.ndarray) -> str:
    """The first row whose state is not 0, 1 or 2, or ``""``."""
    for start in range(0, len(state), _WINDOW * 16):
        block = np.asarray(state[start : start + _WINDOW * 16])
        bad = np.flatnonzero(block > OTHER)
        if bad.size:
            row = start + int(bad[0])
            return f"{SOURCE_ALID_STATE_FILENAME} holds state {int(block[bad[0]])} at row {row}"
    return ""


def _count_fault(state: np.ndarray, offsets: np.ndarray) -> str:
    """Why the offsets do not number the `OTHER` rows plus one, or ``""``."""
    n_other = int(np.count_nonzero(np.asarray(state) == OTHER))
    if len(offsets) != n_other + 1:
        return (
            f"{SOURCE_ALID_OFFSETS_FILENAME} has {len(offsets)} offsets for {n_other} lifted "
            "row(s); it holds one more than they"
        )
    return ""


def _first_not_rising(offsets: np.ndarray) -> int | None:
    """The first lifted row whose end offset is not above its start, a window at a time."""
    n = len(offsets) - 1
    for start in range(0, n, _WINDOW * 16):
        block = np.asarray(offsets[start : min(start + _WINDOW * 16, n) + 1])
        flat = np.flatnonzero(block[1:] <= block[:-1])
        if flat.size:
            return start + int(flat[0])
    return None
