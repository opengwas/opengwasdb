"""The columnar engine behind a Source Reader's association batches (#318, #319).

Pass 2 reads a tabular source's associations as columns, a block of the
decompressed file at a time. The per-row reader is the reference: a block holds
exactly its associations, in order, bit for bit. This module is what every such
reader shares -- reading a source in blocks of whole lines, handing a block it
cannot read as columns to the per-row reader, saying so in the log, and the
identity cells and association columns a block is judged and built from. What
differs between formats (which columns an association reads, how its statistics
are formed, what a row's identity is declared to be) is a `Format`, which
`opengwasdb.readers.finngen` and `opengwasdb.readers.gwas_ssf` each provide.

The columnar parse reads only what it can show it reads as `csv` does. A block
with a quote, a carriage return or a NUL, or any byte outside ASCII, is one
`csv` would split or decode otherwise than at tabs and newlines, so it and the
rest of the source are handed to the per-row reader. Within a block, a row the
columns cannot decide is read by the per-row rules from its own cells.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import IO, Any, Protocol

import numpy as np
import pandas as pd

from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers.interface import (
    AssociationBatch,
    ImputationScoreStatus,
    ReaderAssociation,
    StreamTally,
)
from opengwasdb.stats import FrequencyTally
from opengwasdb.variants.normalise import VariantNormalisationError

#: Associations a batch holds once a source is handed to the per-row reader.
ROW_BATCH = 250_000

#: The cells a statistic reads as missing: `tabular.parse_finite_float`'s and
#: `stats.parse_af`'s set, which a test holds this to.
MISSING_STATISTICS = ("", ".", "NA", "NaN", "None", "nan")

#: How pandas reads a block: every line one row, every field as written.
PANDAS = {
    "sep": "\t",
    "header": None,
    "index_col": False,
    "engine": "c",
    "quoting": csv.QUOTE_NONE,
    "skip_blank_lines": False,
    "keep_default_na": False,
    "low_memory": False,
}

#: The largest position an int64 column holds; a row's beyond it keeps its
#: Python int, as the per-row reader's association does.
INT64_MAX = np.iinfo(np.int64).max


class Format(Protocol):
    """What a tabular format gives the engine to read one source."""

    @property
    def log(self) -> logging.Logger:
        """Where the engine logs."""
        ...

    @property
    def label(self) -> str:
        """How the engine names a source in the log."""
        ...

    @property
    def crlf(self) -> bool:
        """Whether a line ending in CR LF is read as plain text, as `csv` reads it."""
        ...

    def block_bytes(self) -> int:
        """Decompressed bytes one block reads."""
        ...

    def block_batches(
        self, lines: bytes, offset: int, source: Source
    ) -> Iterator[AssociationBatch]:
        """One block of whole lines, starting at byte ``offset``, as batches."""
        ...

    def per_row(
        self, text: io.TextIOBase, names: list[str] | None, source: Source
    ) -> Iterator[AssociationBatch]:
        """``text`` read by the per-row reader and batched; ``names`` are the
        header's, already read, or ``None`` when ``text`` starts at the header."""
        ...


# ── the header and the bytes `csv` reads otherwise than between tabs ─────────

#: The bytes `csv` reads otherwise than as text between tabs and newlines.
_SPECIAL = ((b'"', "a quote"), (b"\r", "a carriage return"), (b"\x00", "a NUL byte"))
_NON_ASCII = re.compile(rb"[\x80-\xff]")
_STRAY_CR = re.compile(rb"\r(?!\n)")


def _found(data: bytes, crlf: bool) -> list[tuple[int, str]]:
    """Each kind of special byte ``data`` holds, at its first place."""
    found = [
        (at, name)
        for byte, name in _SPECIAL
        if not (crlf and byte == b"\r") and (at := data.find(byte)) >= 0
    ]
    stray = _STRAY_CR.search(data) if crlf else None
    if stray is not None:
        found.append((stray.start(), "a carriage return that ends no line"))
    return found


def first_special(
    data: bytes, *, ascii_only: bool = True, crlf: bool = False
) -> tuple[int, str] | None:
    """Where ``data`` first holds a byte `csv` reads otherwise than by splitting
    at tabs and newlines -- a quote, a carriage return, a NUL, or (with
    ``ascii_only``) any byte outside ASCII -- and what it is; ``None`` for none.

    With ``crlf`` a carriage return that ends a line with its line feed is plain,
    as `csv` reads it, and only one that does not is special."""
    found = _found(data, crlf)
    if ascii_only and not data.isascii():
        outside = _NON_ASCII.search(data)
        if outside is not None:
            found.append((outside.start(), "a byte outside ASCII"))
    return min(found) if found else None


def header_names(header: bytes, *, crlf: bool = False) -> list[str] | str:
    """A header line's names, as `csv` reads a line it splits only at tabs, or
    why it cannot be read so. With ``crlf`` the line may end in CR LF."""
    if not header.endswith(b"\n") or len(header) < 2:
        return "its header is not one whole line"
    special = first_special(header, ascii_only=False, crlf=crlf)
    if special is not None:
        return f"its header holds {special[1]}"
    try:
        return header.rstrip(b"\r\n").decode("utf-8").split("\t") if crlf else (
            header[:-1].decode("utf-8").split("\t")
        )
    except UnicodeDecodeError:
        return "its header is not UTF-8"


def without_blank_lines(lines: bytes) -> bytes:
    """``lines`` less every empty line, which `csv.DictReader` skips."""
    lines = lines.lstrip(b"\n")
    # `find`, not `in`: on a block of real text it is the quicker scan.
    while lines.find(b"\n\n") >= 0:
        lines = lines.replace(b"\n\n", b"\n")
    return lines


class _Resumed(io.RawIOBase):
    """The bytes a block parse already read from a source, then the rest of it,
    as the blocks ``rest`` yields."""

    def __init__(self, head: bytes, rest: Iterator[bytes]) -> None:
        super().__init__()
        self._pending = memoryview(head)
        self._rest = rest

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        while not self._pending:
            block = next(self._rest, None)
            if block is None:
                return 0
            self._pending = memoryview(block)
        size = min(len(buffer), len(self._pending))
        memoryview(buffer).cast("B")[:size] = self._pending[:size]
        self._pending = self._pending[size:]
        return size


def text_after(head: bytes, rest: Iterator[bytes]) -> io.TextIOWrapper:
    """``head``, then the rest of the source, as text the per-row reader reads."""
    source = io.BufferedReader(_Resumed(head, rest))
    return io.TextIOWrapper(source, encoding="utf-8", newline="")


# ── one source read a block at a time ────────────────────────────────────────


def association_batches(
    fh: IO[bytes],
    path: object,
    tally: StreamTally,
    scale: StoredEffectScale,
    start: Callable[[bytes], tuple[Format, list[str] | None, str | None]],
) -> Iterator[AssociationBatch]:
    """A source's associations, a block of whole lines at a time (`Source`).

    ``start`` reads the header line to the source's `Format` and either
    ``(names, None)``, where the source can be read as columns, or
    ``(None, reason)``, where the per-row reader must read it throughout.
    """
    header = fh.readline()
    fmt, names, reason = start(header)
    if not header:
        return
    source = Source(path, fh, tally, scale, fmt)
    if reason is not None:
        yield from source.per_row(header, None, 0, reason)
    else:
        yield from source.blocks(names or [], len(header))
    source.summarise()


@dataclass
class Source:
    """One source read a block at a time, and what the read fills.

    A line split across two reads is carried to the next block, and a last line
    with no newline is read as if it had one, as `csv` reads it. From the first
    block that is not plain, the rest goes to the per-row reader. The rows read
    the same either way; the per-row reader is about five times slower, so the
    first place a source is read row by row is logged, by byte offset, and once
    the source is read one more line says how much of it was (`summarise`).
    """

    path: object
    fh: IO[bytes]
    tally: StreamTally
    scale: StoredEffectScale
    fmt: Format
    warned: bool = False
    #: Blocks parsed as columns, and blocks read row by row: one pandas
    #: refused, the block that sent the rest of the source row by row, and
    #: every read after it.
    column_blocks: int = 0
    per_row_blocks: int = 0
    #: Rows of the column-parsed blocks that were read from their own line.
    undecided: int = 0

    def warn(self, message: str, *args: object) -> None:
        """Log ``message`` the first time this source falls back, and never again."""
        if not self.warned:
            self.fmt.log.warning(self.fmt.label + " %s: " + message, self.path, *args)
            self.warned = True

    def summarise(self) -> None:
        """Say, once the source is read, how much of it the per-row rules read."""
        if self.per_row_blocks or self.undecided:
            self.fmt.log.warning(
                self.fmt.label + " %s: %d of %d blocks read row by row, and %d rows of the "
                "others read from their own line",
                self.path, self.per_row_blocks, self.column_blocks + self.per_row_blocks,
                self.undecided,
            )

    def per_row(
        self, head: bytes, names: list[str] | None, offset: int, reason: str
    ) -> Iterator[AssociationBatch]:
        """``head`` and the rest of the source, from byte ``offset``, row by row."""
        self.warn("read row by row from byte %d on, since %s", offset, reason)
        text = text_after(head, self._rest())
        yield from self.fmt.per_row(text, names, self)

    def _rest(self) -> Iterator[bytes]:
        """The source's remaining blocks, each counted as read row by row."""
        while block := self.fh.read(self.fmt.block_bytes()):
            self.per_row_blocks += 1
            yield block

    def blocks(self, names: list[str], offset: int) -> Iterator[AssociationBatch]:
        """The source after its header, which ends at byte ``offset``."""
        carry = b""
        while block := self.fh.read(self.fmt.block_bytes()):
            data = carry + block
            cut = data.rfind(b"\n") + 1
            lines, carry = data[:cut], data[cut:]
            special = first_special(lines, crlf=self.fmt.crlf)
            if special is not None:
                at, what = special
                self.per_row_blocks += 1
                yield from self.per_row(data, names, offset + at, f"it holds {what}")
                return
            yield from self.fmt.block_batches(self._lf(lines), offset, self)
            offset += cut
        special = first_special(carry, crlf=self.fmt.crlf)
        if special is not None:
            at, what = special
            self.per_row_blocks += 1
            yield from self.per_row(carry, names, offset + at, f"it holds {what}")
        elif carry:
            yield from self.fmt.block_batches(self._lf(carry) + b"\n", offset, self)

    def _lf(self, lines: bytes) -> bytes:
        """``lines`` with each CR LF read as the line feed it ends in, where the
        format reads one so."""
        return lines.replace(b"\r\n", b"\n") if self.fmt.crlf else lines


def block_batches(
    lines: bytes,
    offset: int,
    source: Source,
    read: Callable[[bytes], Any],
    build: Callable[[Any, bytes], AssociationBatch],
    per_row_names: list[str],
) -> Iterator[AssociationBatch]:
    """One block's associations, as one batch, or none where it holds none.

    ``lines`` start at byte ``offset`` of the source. ``read`` parses them as
    columns and ``build`` judges the parse and forms the batch.
    """
    lines = without_blank_lines(lines)
    if not lines:
        return
    try:
        block = read(lines)
    except pd.errors.ParserError as refused:
        # pandas refuses a block none of whose lines reaches the header's last
        # column; `csv` pads each, and reads it.
        source.warn("the block at byte %d is read row by row: pandas refused it (%s)",
                    offset, str(refused).strip())
        source.per_row_blocks += 1
        text = io.TextIOWrapper(io.BytesIO(lines), encoding="ascii", newline="")
        yield from source.fmt.per_row(text, per_row_names, source)
        return
    source.column_blocks += 1
    batch = build(block, lines)
    if len(batch):
        yield batch


# ── a block's identity cells ─────────────────────────────────────────────────


@dataclass(frozen=True)
class Cells:
    """One identity column of a block, each distinct cell judged once.

    ``verbatim`` is each distinct cell as the source spells it, ``value`` its
    value under the rule (``""`` where the rule refuses it) and ``ok`` whether
    the rule accepts it; ``codes`` says which distinct cell each row holds.
    """

    verbatim: np.ndarray
    value: np.ndarray
    ok: np.ndarray
    codes: np.ndarray

    @classmethod
    def of(cls, column: pd.Series, rule: Callable[[str], str]) -> Cells:
        verbatim = column.cat.categories.to_numpy(dtype=object)
        value = np.full(len(verbatim), "", dtype=object)
        ok = np.zeros(len(verbatim), dtype=bool)
        for index, cell in enumerate(verbatim.tolist()):
            try:
                value[index] = rule(cell)
            except VariantNormalisationError:
                continue
            ok[index] = True
        codes = column.cat.codes.to_numpy()
        if codes.min(initial=0) < 0:
            raise ValueError(f"block column {column.name!r} read a cell as missing")
        return cls(verbatim, value, ok, codes)

    def rows(self, of: np.ndarray) -> np.ndarray:
        """``of`` (one entry per distinct cell) for each row."""
        selected: np.ndarray = of[self.codes]
        return selected


def orientation(ref: Cells, alt: Cells) -> tuple[np.ndarray, np.ndarray]:
    """Per row: whether `alt`, the effect allele, is A2, so z is negated
    (`alid_alleles`: an indel spelt reference first, any other pair sorted),
    and whether the two alleles are one."""
    spelt = np.unique(np.concatenate((ref.value, alt.value)))
    ref_rank = ref.rows(np.searchsorted(spelt, ref.value))
    alt_rank = alt.rows(np.searchsorted(spelt, alt.value))
    indel = is_indel(ref, alt)
    return indel | (ref_rank < alt_rank), ref_rank == alt_rank


def is_indel(ref: Cells, alt: Cells) -> np.ndarray:
    """Per row: whether the two normalised alleles differ in length."""
    lengths = np.vectorize(len, otypes=[np.int64])
    different: np.ndarray = ref.rows(lengths(ref.value)) != alt.rows(lengths(alt.value))
    return different


@dataclass(frozen=True)
class Identity:
    """A block's identity columns, judged: chromosome, the two alleles (``ref``
    the other allele, ``alt`` the effect allele), and the position.

    ``flipped`` and ``same`` are per row: whether the effect allele is A2, and
    whether the alleles are one. ``position`` is ``None`` where pandas read
    some cell otherwise than as `int()` reads it.
    """

    chromosome: Cells
    ref: Cells
    alt: Cells
    flipped: np.ndarray
    same: np.ndarray
    position: np.ndarray | None

    @classmethod
    def read(
        cls, frame: pd.DataFrame, chromosome: int, ref: int, alt: int, position: int,
        normalise_chromosome: Callable[[str], str], normalise_allele: Callable[[str], str],
    ) -> Identity:
        refs = Cells.of(frame[ref], normalise_allele)
        alts = Cells.of(frame[alt], normalise_allele)
        return cls(
            Cells.of(frame[chromosome], normalise_chromosome),
            refs,
            alts,
            *orientation(refs, alts),
            numbers(frame, position, "i"),
        )

    def __len__(self) -> int:
        return len(self.chromosome.codes)

    def valid(self) -> np.ndarray:
        """The rows whose identity every rule accepts; none where the position
        column could not be read as `int()` reads it."""
        if self.position is None:
            return np.zeros(len(self), dtype=bool)
        valid: np.ndarray = (
            self.chromosome.rows(self.chromosome.ok)
            & self.ref.rows(self.ref.ok)
            & self.alt.rows(self.alt.ok)
            & (self.position > 0)
            & ~self.same
        )
        return valid

    def columns(
        self, z: np.ndarray, se: np.ndarray, af: np.ndarray, kept: np.ndarray
    ) -> Columns:
        """The block's association columns: ``z`` (as reported) negated and
        ``af`` complemented where the effect allele is A2, and an ``af`` outside
        (0, 1) made absent. ``kept`` are the rows that hold an association."""
        in_range = (af > 0) & (af < 1)
        return Columns(
            chromosome=self.chromosome.rows(self.chromosome.value),
            # A copy: pandas' own column is read-only, and an undecided row's
            # position is placed in it.
            position=(
                np.zeros(len(self), dtype=np.int64) if self.position is None
                else self.position.copy()
            ),
            ref=self.ref.rows(self.ref.verbatim),
            alt=self.alt.rows(self.alt.verbatim),
            z=np.where(self.flipped, -z, z),
            se=se,
            eaf=np.where(in_range, np.where(self.flipped, 1.0 - af, af), np.nan),
            kept=kept.copy(),
        )


def made_absent(af: np.ndarray, decided: np.ndarray, tally: FrequencyTally) -> None:
    """Count the ``decided`` rows whose frequency was reported but is not inside
    (0, 1), as `parse_af` counts it made absent (D7)."""
    outside = ~((af > 0) & (af < 1)) & ~np.isnan(af)
    tally.made_absent += int(np.count_nonzero(decided & outside))


def numbers(frame: pd.DataFrame, index: int | None, kind: str) -> np.ndarray | None:
    """A numeric column, or ``None`` where pandas read some cell other than as
    `int()` (``kind`` ``"i"``) or `float()` (``"f"``) reads it. An absent
    statistic column is all missing."""
    if index is None:
        return np.full(len(frame), np.nan)
    column = frame[index]
    if column.dtype.kind != kind:
        return None
    values: np.ndarray = column.to_numpy(dtype=np.int64 if kind == "i" else np.float64)
    return values


def line_cells(
    lines: bytes, reads: dict[str, int], rows: np.ndarray
) -> Iterator[tuple[int, dict[str, str | None]]]:
    """Each of ``rows``' cells by header name, read from its own line.

    The block is plain, so a line's cells are what lies between its tabs, as
    `csv` reads them, and a cell a short line does not reach is ``None``, as a
    `csv.DictReader` row holds it.
    """
    ends = np.flatnonzero(np.frombuffer(lines, dtype=np.uint8) == ord("\n"))
    starts = np.concatenate(([0], ends[:-1] + 1))
    for row in rows.tolist():
        cells = lines[starts[row] : ends[row]].decode("ascii").split("\t")
        yield row, {
            name: cells[index] if index < len(cells) else None for name, index in reads.items()
        }


# ── a block's association columns ────────────────────────────────────────────


@dataclass
class Columns:
    """A block's association columns, an entry a row, and the rows that hold one."""

    chromosome: np.ndarray
    position: np.ndarray
    ref: np.ndarray
    alt: np.ndarray
    z: np.ndarray
    se: np.ndarray
    eaf: np.ndarray
    kept: np.ndarray

    def place(self, rows: Iterable[tuple[int, ReaderAssociation | None]]) -> None:
        """Put each undecided row's association, where it has one, in its place."""
        for row, association in rows:
            if association is None:
                continue
            if association.position > INT64_MAX and self.position.dtype != object:
                self.position = self.position.astype(object)
            self.chromosome[row] = association.chromosome
            self.position[row] = association.position
            self.ref[row], self.alt[row] = association.ref, association.alt
            self.z[row], self.se[row] = association.z, association.se
            self.eaf[row] = np.nan if association.eaf is None else association.eaf
            self.kept[row] = True

    def batch(self) -> AssociationBatch:
        kept = self.kept
        held = int(np.count_nonzero(kept))
        return AssociationBatch(
            chromosome=self.chromosome[kept],
            position=self.position[kept],
            ref=self.ref[kept],
            alt=self.alt[kept],
            z=self.z[kept],
            se=self.se[kept],
            eaf=self.eaf[kept],
            imputation_score=np.full(held, np.nan),
            imputation_status=np.full(held, ImputationScoreStatus.UNDECLARED, dtype=object),
        )

