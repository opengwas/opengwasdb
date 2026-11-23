"""The position sidecar: every row's position, and each chromosome's rows (format 0.3.0).

Three plain `.npy` files beside the Store Variant Table (ADR 0105 §6, spec §1, §4):

- ``variant_position.npy`` (``<u4``): every row's position, in row order;
- ``variant_position_chromosomes.npy`` (``|S<n>``): each chromosome the release
  holds, once, in row order -- which is the canonical chromosome order, because a
  Variant Index is (`variants.axis.alid_sort_key`);
- ``variant_position_chromosome_offsets.npy`` (``<u8``, one more than the
  chromosomes): chromosome ``k``'s rows are ``offsets[k]:offsets[k + 1]``.

Rows are in genomic order, so within one chromosome's rows the positions never fall.
A range is the chromosome's row slice and two `searchsorted` calls inside it: rows
``[lo + left(start), lo + right(end))``, read from a memory map, with no tabix and no
pysam. A 0.2 release answers a range through its tabix index instead
(`variant_axis.VariantAxis.range_indices`).

This module imports numpy and the standard library only, so the Store Reader, the
validator and every writer share one statement of the layout. `write_position_sidecar`
is the sidecar's one writer: a build reaches it through `write_variant_axis`, and the
rewrite to 0.3.0 (`opengwasdb.store.rewrite`) calls it directly. The canonical
chromosome order is not stated here: a writer or the validator passes it in as
`chromosome_key` (`opengwasdb.variants.normalise.chromosome_sort_key`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from opengwasdb.store_reader.npy_sidecar import holds_sidecar, open_sidecar_arrays

POSITION_FILENAME = "variant_position.npy"
CHROMOSOMES_FILENAME = "variant_position_chromosomes.npy"
CHROMOSOME_OFFSETS_FILENAME = "variant_position_chromosome_offsets.npy"

#: The three files, in the order `encode_positions` returns them.
POSITION_SIDECAR_FILENAMES: tuple[str, ...] = (
    POSITION_FILENAME,
    CHROMOSOMES_FILENAME,
    CHROMOSOME_OFFSETS_FILENAME,
)

#: A position is 1-based and at most 2**32 - 1; the longest human chromosome is
#: 248,956,422 bases (GRCh38 chromosome 1).
POSITION_DTYPE = np.dtype("<u4")
ROW_OFFSET_DTYPE = np.dtype("<u8")
POSITION_MAX = int(np.iinfo(POSITION_DTYPE).max)

#: How a writer or the validator orders chromosomes: a sort key over a label.
ChromosomeKey = Callable[[str], Any]


class MissingPositionSidecar(FileNotFoundError):
    """A release's `format_version` carries a position sidecar whose files are not there."""


class MalformedPositionSidecar(ValueError):
    """A sidecar whose chromosomes cannot slice its rows: offsets that do not number the
    chromosomes plus one, a chromosome named twice, or a label that is not UTF-8."""


class UnorderedPositionsError(ValueError):
    """The rows handed to the writer are not in genomic order, so no slice holds a chromosome."""


@dataclass(frozen=True)
class EncodedPositions:
    """The three arrays of a sidecar, before they are written."""

    positions: np.ndarray
    chromosomes: np.ndarray
    offsets: np.ndarray


def encode_positions(
    chromosomes: Sequence[str],
    positions: Sequence[int],
    *,
    chromosome_key: ChromosomeKey | None = None,
) -> EncodedPositions:
    """The sidecar for rows of `chromosomes` and `positions`, one each per row in row order.

    Refuses rows a range could not be answered from: a chromosome whose rows are not
    contiguous, a position that falls within a chromosome, one outside ``1..2**32-1``,
    an empty label, and, given `chromosome_key`, chromosomes out of its order.
    """
    n_rows = len(chromosomes)
    if len(positions) != n_rows:
        raise ValueError(f"{len(positions)} positions for {n_rows} chromosomes: one each per row")
    values = np.asarray(positions, dtype=np.int64).reshape(n_rows)
    if n_rows and (int(values.min()) < 1 or int(values.max()) > POSITION_MAX):
        bad = int(np.flatnonzero((values < 1) | (values > POSITION_MAX))[0])
        raise ValueError(
            f"row {bad} is at position {int(values[bad])}, outside 1..{POSITION_MAX}"
        )
    labels, starts = _runs(chromosomes)
    fault = _first_fault(labels, starts, values, chromosome_key)
    if fault:
        raise UnorderedPositionsError(f"{fault}; rows must be in genomic order")
    offsets = np.append(starts, n_rows).astype(ROW_OFFSET_DTYPE)
    return EncodedPositions(
        positions=values.astype(POSITION_DTYPE),
        chromosomes=_label_array(labels),
        offsets=offsets,
    )


def _runs(chromosomes: Sequence[str]) -> tuple[list[str], np.ndarray]:
    """Each run of equal adjacent labels: its label and its first row."""
    if not len(chromosomes):
        return [], np.empty(0, dtype=np.int64)
    labels = np.asarray(chromosomes, dtype=object)
    starts = np.concatenate(([0], np.flatnonzero(labels[1:] != labels[:-1]) + 1))
    return [str(labels[i]) for i in starts.tolist()], starts.astype(np.int64)


def _first_fault(
    labels: list[str],
    starts: np.ndarray,
    positions: np.ndarray,
    chromosome_key: ChromosomeKey | None,
) -> str:
    """Why rows with these runs and positions are not in genomic order, or ``""``."""
    return (
        _label_fault(labels, starts)
        or _order_fault(labels, chromosome_key)
        or _position_fault(starts, positions)
    )


def _label_fault(labels: list[str], starts: np.ndarray) -> str:
    """An empty label, or a chromosome whose rows are not contiguous; else ``""``."""
    if any(not label for label in labels):
        return "a row has an empty chromosome"
    seen: set[str] = set()
    for label, start in zip(labels, starts.tolist(), strict=True):
        if label in seen:
            return f"chromosome {label!r} resumes at row {start} after another chromosome's rows"
        seen.add(label)
    return ""


def _order_fault(labels: list[str], chromosome_key: ChromosomeKey | None) -> str:
    """The first chromosome out of `chromosome_key`'s order, if one is given; else ``""``."""
    if chromosome_key is None:
        return ""
    keys = [chromosome_key(label) for label in labels]
    for k in range(1, len(keys)):
        if not keys[k - 1] < keys[k]:
            return (
                f"chromosome {labels[k]!r} follows {labels[k - 1]!r}, out of the "
                "canonical chromosome order"
            )
    return ""


def _position_fault(starts: np.ndarray, positions: np.ndarray) -> str:
    """The first row whose position falls within its chromosome; else ``""``."""
    falling = _falling_rows(positions, starts)
    if falling.size:
        row = int(falling[0])
        return (
            f"row {row} is at position {int(positions[row])}, before row {row - 1} at "
            f"{int(positions[row - 1])} on the same chromosome"
        )
    return ""


def _falling_rows(positions: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Each row whose position is below the one before it on the same chromosome."""
    if len(positions) < 2:
        return np.empty(0, dtype=np.int64)
    falling = np.flatnonzero(np.asarray(positions[1:]) < np.asarray(positions[:-1])) + 1
    return falling[~np.isin(falling, starts)]


def _label_array(labels: list[str]) -> np.ndarray:
    encoded = [label.encode("utf-8") for label in labels]
    width = max((len(item) for item in encoded), default=1)
    return np.array(encoded, dtype=f"S{max(width, 1)}")


def write_position_sidecar(
    store_path: str | Path,
    chromosomes: Sequence[str],
    positions: Sequence[int],
    *,
    chromosome_key: ChromosomeKey | None,
) -> None:
    """Write the three sidecar files for rows of `chromosomes` and `positions` into
    `store_path`. `chromosome_key` is the canonical chromosome order the rows are in;
    None says the caller holds them to it otherwise."""
    encoded = encode_positions(chromosomes, positions, chromosome_key=chromosome_key)
    store = Path(store_path)
    arrays = (encoded.positions, encoded.chromosomes, encoded.offsets)
    for name, array in zip(POSITION_SIDECAR_FILENAMES, arrays, strict=True):
        np.save(store / name, array)


class PositionSidecar:
    """A release's position sidecar, memory-mapped: each row's position, each chromosome's rows."""

    FILENAMES: ClassVar[tuple[str, ...]] = POSITION_SIDECAR_FILENAMES

    def __init__(self, positions: np.ndarray, chromosomes: np.ndarray, offsets: np.ndarray) -> None:
        """Refuses chromosomes that cannot slice the rows (`MalformedPositionSidecar`):
        read as they stand, they would answer one chromosome's ranges from another's
        rows, or from none. A check of the chromosomes alone, a few dozen; the rows'
        order is the validator's (`sidecar_problems`)."""
        self.positions = positions
        self.chromosomes = chromosomes
        self.offsets = offsets
        self.labels: list[str] = _labels(chromosomes)
        bounds = np.asarray(offsets, dtype=np.int64).tolist()
        fault = _slicing_fault(self.labels, len(bounds))
        if fault:
            raise MalformedPositionSidecar(f"the position sidecar cannot be read: {fault}")
        self._slices: dict[str, tuple[int, int]] = {
            label: (lo, hi)
            for label, lo, hi in zip(self.labels, bounds[:-1], bounds[1:], strict=True)
        }

    @classmethod
    def open(cls, store_path: str | Path) -> PositionSidecar | None:
        """The sidecar in `store_path`; None when it holds none of the three files.

        A directory holding some of them and not the others is refused
        (`npy_sidecar.holds_sidecar`).
        """
        arrays = open_sidecar_arrays(store_path, POSITION_SIDECAR_FILENAMES, _WHAT)
        return None if arrays is None else cls(*arrays)

    @staticmethod
    def held_in(store_path: str | Path) -> bool:
        """Whether `store_path` holds the sidecar's files, without reading them; a
        partial set is refused."""
        return holds_sidecar(store_path, POSITION_SIDECAR_FILENAMES, _WHAT)

    @property
    def n_rows(self) -> int:
        return len(self.positions)

    def rows_in(self, chromosome: str, start: int, end: int) -> np.ndarray:
        """The rows on `chromosome` at a position in ``[start, end]``, ascending.

        `chromosome` is matched as given: the caller normalises it. None on a
        chromosome the release lacks, and none when `end` is below `start`.
        """
        bounds = self._slices.get(chromosome)
        lo_value, hi_value = max(int(start), 1), min(int(end), POSITION_MAX)
        if bounds is None or hi_value < lo_value:
            return np.empty(0, dtype=np.int64)
        lo, hi = bounds
        block = self.positions[lo:hi]
        first = int(np.searchsorted(block, POSITION_DTYPE.type(lo_value), side="left"))
        last = int(np.searchsorted(block, POSITION_DTYPE.type(hi_value), side="right"))
        return np.arange(lo + first, lo + last, dtype=np.int64)

    def chromosomes_of(self, rows: np.ndarray | Sequence[int]) -> list[str | None]:
        """Each row's chromosome, as its slice names it; None for a row in no slice."""
        rows = np.asarray(rows, dtype=np.int64)
        offsets = np.asarray(self.offsets, dtype=np.int64)
        labels = self.labels
        at = np.searchsorted(offsets, rows, side="right") - 1
        held = (at >= 0) & (at < len(labels)) & (rows < (offsets[-1] if offsets.size else 0))
        return [labels[k] if ok else None for k, ok in zip(at.tolist(), held.tolist(), strict=True)]

    def placements(self, rows: np.ndarray | Sequence[int]) -> list[tuple[str | None, int | None]]:
        """Each row's ``(chromosome, position)``; ``(None, None)`` past the sidecar's end."""
        rows = np.asarray(rows, dtype=np.int64)
        chromosomes = self.chromosomes_of(rows)
        held = rows[rows < self.n_rows]
        positions = dict(zip(held.tolist(), np.asarray(self.positions[held]).tolist(), strict=True))
        return [
            (chromosome, positions.get(row)) if chromosome is not None else (None, None)
            for row, chromosome in zip(rows.tolist(), chromosomes, strict=True)
        ]

    def misplaced(
        self, start: int, chromosomes: Sequence[str], positions: Sequence[int]
    ) -> list[int]:
        """The rows from `start`, one per given chromosome and position, that this
        sidecar places elsewhere, ascending.

        The one comparison of the sidecar against the Variant Table: the rewrite's
        read-back and the validator's `variant_table.position_sidecar` both make it.
        A row past the sidecar's end is not compared; `n_rows` says the sidecar is short.
        """
        held = max(min(start + len(chromosomes), self.n_rows) - start, 0)
        stored = np.asarray(self.positions[start : start + held], dtype=np.int64)
        given = np.asarray(positions[:held], dtype=np.int64)
        named = self.chromosomes_of(np.arange(start, start + held))
        return [
            start + k
            for k in range(held)
            if stored[k] != given[k] or named[k] != chromosomes[k]
        ]

    def problems(
        self, table_rows: int, *, chromosome_key: ChromosomeKey | None
    ) -> Iterator[str]:
        """Its structure (`sidecar_problems`), and one row per row of the table."""
        found = list(sidecar_problems(self, chromosome_key=chromosome_key))
        yield from found
        if not found and self.n_rows != table_rows:
            yield (
                f"the position sidecar holds {self.n_rows} rows but variants.tsv.gz has "
                f"{table_rows}"
            )


def sidecar_problems(
    sidecar: PositionSidecar, *, chromosome_key: ChromosomeKey | None
) -> Iterator[str]:
    """What is wrong with the sidecar's own structure, if anything.

    Each array has its dtype; the offsets start at 0, rise with every chromosome and
    end at the row count; each chromosome is named once, and in `chromosome_key`'s
    order when one is given; and every position is in ``1..2**32-1`` and never falls
    within a chromosome. Whether each row's chromosome and position are the ones the
    Variant Table gives it is the validator's comparison (`variant_table.position_sidecar`).
    """
    fault = _dtype_problem(sidecar) or _offset_problem(sidecar)
    if fault:
        yield fault
        return
    positions = np.asarray(sidecar.positions)
    if positions.size and int(positions.min()) < 1:
        yield f"{POSITION_FILENAME} holds position 0 at row {int(np.argmin(positions))}"
        return
    starts = np.asarray(sidecar.offsets, dtype=np.int64)[:-1]
    fault = _first_fault(sidecar.labels, starts, positions, chromosome_key)
    if fault:
        yield f"the position sidecar is not in genomic order: {fault}"


def _offset_problem(sidecar: PositionSidecar) -> str:
    """Why the chromosome offsets do not slice the rows into the chromosomes, or ``""``.

    That they number the chromosomes plus one is checked at open (`_slicing_fault`).
    """
    offsets = np.asarray(sidecar.offsets, dtype=np.int64)
    if int(offsets[0]) != 0 or int(offsets[-1]) != sidecar.n_rows:
        return (
            f"{CHROMOSOME_OFFSETS_FILENAME} runs from {int(offsets[0])} to {int(offsets[-1])}, "
            f"not from 0 to the {sidecar.n_rows} rows of {POSITION_FILENAME}"
        )
    if np.any(offsets[1:] <= offsets[:-1]):
        at = int(np.flatnonzero(offsets[1:] <= offsets[:-1])[0])
        return f"{CHROMOSOME_OFFSETS_FILENAME} does not rise at chromosome {at}: each holds a row"
    return ""


_WHAT = "position sidecar"


def _labels(chromosomes: np.ndarray) -> list[str]:
    """Each stored chromosome label, decoded. A label that is not UTF-8 is refused,
    never replaced: a replaced label would match no chromosome a caller asks for."""
    labels = []
    for k, label in enumerate(np.asarray(chromosomes).tolist()):
        try:
            labels.append(bytes(label).decode("utf-8"))
        except (TypeError, UnicodeDecodeError) as exc:
            raise MalformedPositionSidecar(
                f"the position sidecar cannot be read: {CHROMOSOMES_FILENAME}'s chromosome "
                f"{k} is not UTF-8 bytes ({label!r})"
            ) from exc
    return labels


def _slicing_fault(labels: list[str], n_offsets: int) -> str:
    """Why chromosomes `labels`, with `n_offsets` row offsets, cannot slice the rows;
    else ``""``. Only the chromosomes are read."""
    if n_offsets != len(labels) + 1:
        return (
            f"{CHROMOSOME_OFFSETS_FILENAME} has {n_offsets} offsets for {len(labels)} "
            "chromosome(s); it holds one more than the chromosomes"
        )
    seen: set[str] = set()
    for label in labels:
        if label in seen:
            return f"{CHROMOSOMES_FILENAME} names chromosome {label!r} twice"
        seen.add(label)
    return ""


def _dtype_problem(sidecar: PositionSidecar) -> str:
    """The first array not of the dtype the layout gives it, or ``""``."""
    expected = (
        (POSITION_FILENAME, sidecar.positions, POSITION_DTYPE),
        (CHROMOSOME_OFFSETS_FILENAME, sidecar.offsets, ROW_OFFSET_DTYPE),
    )
    for name, array, dtype in expected:
        if array.dtype != dtype or array.ndim != 1:
            return f"{name} is {array.dtype} of {array.ndim} dimension(s), not 1-d {dtype}"
    if sidecar.chromosomes.dtype.kind != "S" or sidecar.chromosomes.ndim != 1:
        return f"{CHROMOSOMES_FILENAME} is {sidecar.chromosomes.dtype}, not 1-d bytes"
    return ""
