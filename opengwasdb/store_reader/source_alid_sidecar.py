"""The `source_alid` sidecar: each row's pre-liftover ALID, in row order (format 0.3.0).

A row's `source_alid` is the ALID its source spelt it by, on the source's Reference
Assembly: the provenance link back to the dataset a lifted row came from (#470, ADR
0105 §8). Every build writes an unlifted row's `source_alid` as its own ALID; a row
two source variants collided onto, and a row Reference Completion added from the
panel, have none. The 0.2 series holds it in the Variant Table's eighth column only.

Two plain `.npy` files beside the Store Variant Table (spec §1, §4):

- ``variant_source_alid_utf8.npy`` (``uint8``): every row's `source_alid`, UTF-8,
  concatenated in row order;
- ``variant_source_alid_offsets.npy`` (``<u8``, one more than the rows): row ``i``'s
  is ``utf8[offsets[i]:offsets[i + 1]]``, and an empty slice is none.

It is the ALID sidecar's row-to-text half (`opengwasdb.store_reader.alid_sidecar`)
without its search keys: nothing looks a row up by its `source_alid` but the
validator's source-fidelity join, which scans (`SourceAlidSidecar.rows_of`). A value is
stored whole, however long.

This module imports numpy and the standard library only, as every codec does.
`write_source_alid_sidecar` is the sidecar's one writer: a build or a completion
reaches it through `write_variant_axis`, and the rewrite to 0.3.0
(`opengwasdb.store.rewrite`) calls it directly.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

from opengwasdb.store_reader.npy_sidecar import holds_sidecar, open_sidecar_arrays

SOURCE_ALID_UTF8_FILENAME = "variant_source_alid_utf8.npy"
SOURCE_ALID_OFFSETS_FILENAME = "variant_source_alid_offsets.npy"

#: The two files, in the order `encode_source_alids` returns them.
SOURCE_ALID_SIDECAR_FILENAMES: tuple[str, ...] = (
    SOURCE_ALID_UTF8_FILENAME,
    SOURCE_ALID_OFFSETS_FILENAME,
)

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


class UnencodableSourceAlidError(ValueError):
    """A `source_alid` the sidecar and the Variant Table could not both hold as given."""


class MissingSourceAlidSidecar(FileNotFoundError):
    """A release's `format_version` carries a `source_alid` sidecar whose files are not there."""


class MalformedSourceAlidSidecar(ValueError):
    """A sidecar whose offsets cannot slice its bytes: they do not start at 0, or do not
    end at the byte array's end, or an array is not of its dtype."""


@dataclass(frozen=True)
class EncodedSourceAlids:
    """The two arrays of a sidecar, before they are written."""

    utf8: np.ndarray
    offsets: np.ndarray


def encode_source_alids(values: Sequence[str | None]) -> EncodedSourceAlids:
    """The sidecar for `values`, one per row in row order; None or ``""`` is none.

    Refuses a value the table could not hold as the sidecar does (`_UNWRITABLE`), the
    table's blank ``"."``, and one that is not text UTF-8 can spell (a lone surrogate).
    """
    encoded = [_encoded(row, value) for row, value in enumerate(values)]
    lengths = np.fromiter((len(item) for item in encoded), dtype=OFFSET_DTYPE, count=len(encoded))
    offsets = np.zeros(len(encoded) + 1, dtype=OFFSET_DTYPE)
    np.cumsum(lengths, out=offsets[1:])
    utf8 = np.frombuffer(b"".join(encoded), dtype=UTF8_DTYPE)
    return EncodedSourceAlids(utf8=utf8, offsets=offsets)


def _encoded(row: int, value: str | None) -> bytes:
    if not value:
        return b""
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


def write_source_alid_sidecar(store_path: str | Path, values: Sequence[str | None]) -> None:
    """Write the two sidecar files for `values` (one per row, in row order) into `store_path`."""
    encoded = encode_source_alids(values)
    store = Path(store_path)
    arrays = (encoded.utf8, encoded.offsets)
    for name, array in zip(SOURCE_ALID_SIDECAR_FILENAMES, arrays, strict=True):
        np.save(store / name, array)


class SourceAlidSidecar:
    """A release's `source_alid` sidecar, memory-mapped: each row's `source_alid`."""

    FILENAMES: ClassVar[tuple[str, ...]] = SOURCE_ALID_SIDECAR_FILENAMES

    def __init__(self, utf8: np.ndarray, offsets: np.ndarray) -> None:
        """Refuses offsets that cannot slice the bytes (`MalformedSourceAlidSidecar`):
        read as they stand, they would name a row by another row's bytes, or by bytes
        past the end. Three reads of the offsets; that they never fall is the
        validator's (`problems`)."""
        self.utf8 = utf8
        self.offsets = offsets
        fault = _slicing_fault(utf8, offsets)
        if fault:
            raise MalformedSourceAlidSidecar(f"the {_WHAT} cannot be read: {fault}")
        self._text = memoryview(np.asarray(utf8))

    @classmethod
    def open(cls, store_path: str | Path) -> SourceAlidSidecar | None:
        """The sidecar in `store_path`; None when it holds neither file. A directory
        holding one and not the other is refused (`npy_sidecar.holds_sidecar`)."""
        arrays = open_sidecar_arrays(store_path, SOURCE_ALID_SIDECAR_FILENAMES, _WHAT)
        return None if arrays is None else cls(*arrays)

    @staticmethod
    def held_in(store_path: str | Path) -> bool:
        """Whether `store_path` holds the sidecar's files, without reading them; a
        partial set is refused."""
        return holds_sidecar(store_path, SOURCE_ALID_SIDECAR_FILENAMES, _WHAT)

    @property
    def n_rows(self) -> int:
        return len(self.offsets) - 1

    def values(self, rows: np.ndarray | Sequence[int]) -> np.ndarray:
        """Each row's `source_alid`, as an object array in `rows`' order; None for a
        row that has none. A row outside the sidecar is refused, never read as none."""
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        if rows.size and (int(rows.min()) < 0 or int(rows.max()) >= self.n_rows):
            outside = int(rows[(rows < 0) | (rows >= self.n_rows)][0])
            raise IndexError(f"row {outside} is outside the {_WHAT}'s {self.n_rows} rows")
        starts = np.asarray(self.offsets[rows], dtype=np.int64).tolist()
        stops = np.asarray(self.offsets[rows + 1], dtype=np.int64).tolist()
        out = np.full(rows.size, None, dtype=object)
        text = self._text
        out[:] = [
            str(text[a:b], "utf-8") if b > a else None
            for a, b in zip(starts, stops, strict=True)
        ]
        return out

    def sourced(self, start: int, stop: int) -> int:
        """How many rows in ``[start, stop)`` have a `source_alid`."""
        bounds = np.asarray(self.offsets[start : min(stop, self.n_rows) + 1], dtype=np.int64)
        return int(np.count_nonzero(np.diff(bounds)))

    def differing(self, start: int, values: Sequence[str | None]) -> list[int]:
        """The rows from `start`, one per given value (None is none), whose stored
        `source_alid` is another, ascending.

        The one comparison of the sidecar against the Variant Table: the rewrite's
        read-back and the validator's `variant_table.source_alid_sidecar` both make it.
        A row past the sidecar's end is not compared; `n_rows` says the sidecar is short.
        """
        held = max(min(start + len(values), self.n_rows) - start, 0)
        stored = self.values(np.arange(start, start + held, dtype=np.int64))
        return [start + k for k in range(held) if stored[k] != (values[k] or None)]

    def rows_of(self, wanted: Iterable[str]) -> dict[str, int]:
        """The first row whose `source_alid` is each of `wanted`; one no row has is
        absent. One scan, a window of rows at a time, that stops once each is found."""
        remaining = {value.encode("utf-8"): value for value in wanted if value}
        found: dict[str, int] = {}
        for start in range(0, self.n_rows, _WINDOW):
            if not remaining:
                break
            for row, query in self._matches(start, min(start + _WINDOW, self.n_rows), remaining):
                found[remaining.pop(query)] = row
        return found

    def _matches(
        self, start: int, stop: int, wanted: dict[bytes, str]
    ) -> Iterator[tuple[int, bytes]]:
        """Each wanted value's first row in ``[start, stop)``, as ``(row, value)``."""
        bounds = np.asarray(self.offsets[start : stop + 1], dtype=np.int64)
        chunk = np.asarray(self.utf8[int(bounds[0]) : int(bounds[-1])])
        first: dict[bytes, int] = {}
        for length in sorted({len(query) for query in wanted}):
            queries = [query for query in wanted if len(query) == length]
            for row, value in _rows_holding(chunk, bounds, length, queries):
                first.setdefault(value, start + row)
        for value, row in sorted(first.items(), key=lambda item: item[1]):
            yield row, value

    def problems(self, table_rows: int) -> Iterator[str]:
        """Its structure, and one row per row of the table.

        Each array is 1-d of its dtype, and the offsets never fall; that they start at 0
        and end at the byte array's end is checked at open. Whether each row's value is
        the table's is `differing`'s.
        """
        fault = _dtype_fault(self.utf8, self.offsets)
        if fault:
            yield fault
            return
        falling = _first_falling(self.offsets)
        if falling is not None:
            yield f"{SOURCE_ALID_OFFSETS_FILENAME} falls at row {falling}"
            return
        if self.n_rows != table_rows:
            yield f"the {_WHAT} holds {self.n_rows} rows but variants.tsv.gz has {table_rows}"


def _rows_holding(
    chunk: np.ndarray, bounds: np.ndarray, length: int, queries: list[bytes]
) -> Iterator[tuple[int, bytes]]:
    """Each row of a window, ascending, whose value is one of `queries`, all `length`
    bytes long: ``(row within the window, value)``. `chunk` is the window's bytes and
    `bounds` its rows' offsets."""
    rows = np.flatnonzero(np.diff(bounds) == length)
    if not rows.size:
        return
    at = (bounds[rows] - bounds[0])[:, None] + np.arange(length, dtype=np.int64)
    stored = np.ascontiguousarray(chunk[at]).view(f"S{length}").reshape(rows.size)
    hits = np.flatnonzero(np.isin(stored, np.array(queries, dtype=f"S{length}")))
    wanted = set(queries)
    for k in hits.tolist():
        value = chunk[at[k]].tobytes()  # exact: an `S` compare ignores trailing NULs
        if value in wanted:
            yield int(rows[k]), value


def _slicing_fault(utf8: np.ndarray, offsets: np.ndarray) -> str:
    """Why `offsets` cannot slice `utf8` at all, or ``""``."""
    fault = _dtype_fault(utf8, offsets)
    if fault:
        return fault
    if not len(offsets):
        return f"{SOURCE_ALID_OFFSETS_FILENAME} is empty; it holds one more entry than the rows"
    first, last = int(offsets[0]), int(offsets[-1])
    if first != 0 or last != len(utf8):
        return (
            f"{SOURCE_ALID_OFFSETS_FILENAME} runs from {first} to {last}, not from 0 to the "
            f"{len(utf8)} bytes of {SOURCE_ALID_UTF8_FILENAME}"
        )
    return ""


def _dtype_fault(utf8: np.ndarray, offsets: np.ndarray) -> str:
    """The first array not 1-d of the dtype the layout gives it, or ``""``."""
    expected = (
        (SOURCE_ALID_UTF8_FILENAME, utf8, UTF8_DTYPE),
        (SOURCE_ALID_OFFSETS_FILENAME, offsets, OFFSET_DTYPE),
    )
    for name, array, dtype in expected:
        if array.dtype != dtype or array.ndim != 1:
            return f"{name} is {array.dtype} of {array.ndim} dimension(s), not 1-d {dtype}"
    return ""


def _first_falling(offsets: np.ndarray) -> int | None:
    """The first row whose end offset is below its start, read a window at a time."""
    n_rows = len(offsets) - 1
    for start in range(0, n_rows, _WINDOW * 16):
        block = np.asarray(offsets[start : min(start + _WINDOW * 16, n_rows) + 1])
        falling = np.flatnonzero(block[1:] < block[:-1])
        if falling.size:
            return start + int(falling[0])
    return None
