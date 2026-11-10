"""FinnGen R13 tabular summary-statistics SourceReader.

FinnGen publishes one tab-delimited, bgzip-compressed file per endpoint on
GRCh38.  ``ref`` is the reference allele and ``alt`` the effect allele; the
reader therefore converts ``beta / sebeta`` to the package-wide canonical A1
orientation while leaving the source REF/ALT labels intact on streamed
records. Every row declares its ``ref`` the reference, so an indel's ALID
spells it first (ADR 0079).

Pass 2 reads the associations as columns, a block of the file at a time
(`FinnGenR13Reader.stream_association_batches`, #318), held bit for bit to the
per-row reader (`_source_row`), which stays the reference.
"""

from __future__ import annotations

import csv
import gzip
import io
import logging
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import opengwasdb.readers.interface as interface
import opengwasdb.readers.tabular as tabular
from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers.interface import (
    AssociationBatch,
    ImputationScoreStatus,
    ReaderAssociation,
    RegionsFile,
    SiteMetrics,
    SourceVariant,
    batched,
)
from opengwasdb.readers.rows import ReaderBatch, stream_associations, stream_batches
from opengwasdb.stats import FrequencyTally
from opengwasdb.variants.normalise import (
    IdentityFaults,
    VariantNormalisationError,
    normalise_allele,
    normalise_chromosome,
    orient_or_drop,
)

FINNGEN_R13_CAPABILITY = "opengwasdb.finngen-r13"

log = logging.getLogger(__name__)
_FINNGEN_CHROMOSOMES = {
    str(chromosome).encode(): normalise_chromosome(str(chromosome))
    for chromosome in range(1, 24)
}
_FINNGEN_ALLELE_PAIRS = {
    (bytes((ref,)), bytes((alt,))): (chr(ref), chr(alt))
    for ref in b"ACGTacgt"
    for alt in b"ACGTacgt"
    if (ref & 0xDF) != (alt & 0xDF)
}


def _first_rsid(value: str | None) -> str:
    """FinnGen's `rsids` column is comma-separated where dbSNP names one
    position more than once. The Store Variant Table has one rsid per row, so
    take the first and leave the rest unrecorded rather than inventing a
    multi-value convention no reader or query path understands (issue #109).
    """
    if not value:
        return ""
    first = value.split(",")[0].strip()
    return first if first.startswith("rs") else ""


def _opener(path: str | Path) -> Callable[..., Any]:
    """How a FinnGen source is opened: by its name, `.gz` or `.bgz` as gzip."""
    return gzip.open if str(path).endswith((".gz", ".bgz")) else open


def _iter_rows(
    path: str | Path,
    tally: interface.StreamTally | None = None,
    *,
    infinite_effects: bool = True,
) -> Iterator[tabular.SourceRow]:
    """Every row naming a variant. ``infinite_effects`` keeps an infinite
    `beta` as the row's effect, for Association Admission to drop (D4); the
    metrics projection reads it as absent, as every metrics projection does."""
    parse_beta = tabular.parse_effect if infinite_effects else tabular.parse_finite_float
    with _opener(path)(path, "rt", encoding="utf-8", newline="") as fh:
        yield from _source_rows(
            csv.DictReader(fh, delimiter="\t"), tally or interface.StreamTally(), parse_beta
        )


def _source_rows(
    rows: Iterable[Mapping[str, str | None]],
    tally: interface.StreamTally,
    parse_beta: Callable[[str | None], float | None],
) -> Iterator[tabular.SourceRow]:
    """`_source_row` over rows read as mappings of header names to cells."""
    faults, frequencies = tally.faults, tally.frequencies
    for row in rows:
        source_row = _source_row(row, faults, frequencies, parse_beta)
        if source_row is not None:
            yield source_row


def _source_row(
    row: Mapping[str, str | None],
    faults: IdentityFaults,
    frequencies: FrequencyTally,
    parse_beta: Callable[[str | None], float | None],
) -> tabular.SourceRow | None:
    """One row as the per-row reader reads it, or ``None`` for one naming no variant.

    The reference every faster FinnGen path is held to (#318). ``row`` maps
    the header's names to the row's cells, ``None`` for a cell a short row
    does not reach, as `csv.DictReader` gives it.
    """
    # R13 uses #chrom. Accepting chrom as well preserves compatibility
    # with older captured FinnGen releases without changing semantics.
    chromosome = row.get("#chrom", row.get("chrom"))
    ref = row.get("ref")
    alt = row.get("alt")
    # A row too short to reach an allele has a missing one, which
    # `faults` names (#365); the None checks are the type checker's.
    orientation = orient_or_drop(
        chromosome, row.get("pos"), alt, ref, faults,
        declaration=tabular.EVERY_ROW_REFERENCE, row=row,
    )
    if orientation is None or ref is None or alt is None:
        return None
    return tabular.SourceRow(
        chromosome=orientation.variant.chromosome,
        position=orientation.variant.position,
        ref=ref,
        alt=alt,
        alid=orientation.variant.alid,
        flipped=orientation.flipped,
        beta=parse_beta(row.get("beta")),
        se=tabular.parse_positive_float(row.get("sebeta")),
        af_alt=tabular.parse_af(row.get("af_alt"), frequencies),
        rsid=_first_rsid(row.get("rsids")),
        # SAIGE's two-sided p, for the resolver's effect check (ADR 0069).
        # It never stands in for an unusable `sebeta` here.
        p_value=tabular.parse_finite_float(row.get("pval")),
    )


def _projection_indexes(header: bytes) -> tuple[int, int, int, int, int | None, int] | None:
    columns = header.rstrip(b"\r\n").split(b"\t")
    indexes = {name: index for index, name in enumerate(columns)}
    chromosome = indexes.get(b"#chrom", indexes.get(b"chrom"))
    if chromosome is None:
        return None
    try:
        position = indexes[b"pos"]
        ref = indexes[b"ref"]
        alt = indexes[b"alt"]
    except KeyError:
        return None
    rsid_index = indexes.get(b"rsids")
    last_selected = max(
        chromosome,
        position,
        ref,
        alt,
        rsid_index if rsid_index is not None else 0,
    )
    split_limit = last_selected + (last_selected < len(columns) - 1)
    return chromosome, position, ref, alt, rsid_index, split_limit


def _projected_rsid(row: list[bytes], index: int | None) -> str:
    if index is None or index >= len(row):
        return ""
    value = row[index]
    if value.startswith(b"rs") and value[-1] not in b" \t\r\n\v\f":
        comma = value.find(b",")
        if comma < 0:
            return value.decode("utf-8")
    if value.startswith(b'"'):
        decoded = next(csv.reader([value.decode("utf-8")], delimiter="\t"))[0]
        value = decoded.encode()
    first = value.split(b",", 1)[0].strip()
    return first.decode("utf-8") if first.startswith(b"rs") else ""


def _required_projection(
    path: str | Path, header: bytes
) -> tuple[int, int, int, int, int | None, int]:
    if b'"' in header:
        fields = next(csv.reader([header.decode("utf-8")], delimiter="\t"))
        header = b"\t".join(value.encode() for value in fields)
    projection = _projection_indexes(header)
    if projection is not None:
        return projection
    raise ValueError(
        f"{path}: missing required variant columns; expected #chrom/chrom, pos, ref, alt"
    )


def _fallback_variant(
    row: list[bytes],
    line: bytes,
    chromosome: int,
    position: int,
    ref: int,
    alt: int,
    rsid: int | None,
    identifier: str,
) -> tuple[SourceVariant, ...]:
    if b'"' in line:
        decoded = next(csv.reader([line.decode("utf-8")], delimiter="\t"))
        row = [value.encode() for value in decoded]
        identifier = _projected_rsid(row, rsid)
    variant = tabular.project_source_variant(
        row[chromosome],
        row[position],
        row[ref],
        row[alt],
        identifier,
        other_is_reference=True,
    )
    return () if variant is None else (variant,)


def _iter_variants(path: str | Path) -> Iterator[SourceVariant]:
    """Run FinnGen's hot identity path while preserving full-row semantics.

    Keeping this source-specific loop avoids a general dispatcher on each of
    21 million production rows; byte-parity tests guard it against drift from
    the reference full-row parser.
    """
    with _opener(path)(path, "rb") as fh:
        projection = _required_projection(path, fh.readline())
        chromosome, position, ref, alt, rsid, split_limit = projection
        last_required = max(chromosome, position, ref, alt)
        for line in fh:
            row = line.split(b"\t", split_limit)
            if len(row) <= last_required:
                continue
            row[-1] = row[-1].rstrip(b"\r\n")
            identifier = ""
            try:
                identifier = _projected_rsid(row, rsid)
                projected_chromosome = _FINNGEN_CHROMOSOMES[row[chromosome].strip()]
                projected_position = int(row[position])
                if projected_position <= 0:
                    raise ValueError
                projected_ref, projected_alt = _FINNGEN_ALLELE_PAIRS[(row[ref], row[alt])]
            except (KeyError, UnicodeDecodeError, ValueError):
                yield from _fallback_variant(
                    row, line, chromosome, position, ref, alt, rsid, identifier
                )
                continue
            variant = object.__new__(SourceVariant)
            object.__setattr__(
                variant,
                "__dict__",
                {
                    "rsid": identifier,
                    "alt": projected_alt,
                    "ref": projected_ref,
                    "position": projected_position,
                    "chromosome": projected_chromosome,
                },
            )
            yield variant


def stream_full_row_variants(path: str | Path) -> Iterator[SourceVariant]:
    """Reference variant semantics retained for parity benchmarks (issue #179)."""
    yield from tabular.stream_variants(_iter_rows(path))


# ── Association batches (#318) ───────────────────────────────────────────────
#
# Pass 2 reads a FinnGen source's associations as columns, a block of the
# decompressed file at a time. The per-row reader (`_source_row`) is the
# reference: a block holds exactly its associations, in order, bit for bit.
#
# The columnar parse reads only what it can show it reads as `csv` does. A
# block with a quote, a carriage return or a NUL, or any byte outside ASCII, is
# one `csv` would split or decode otherwise than at tabs and newlines, so it
# and the rest of the source are handed to the per-row reader. Within a block,
# a row the columns cannot decide -- an identity the rules refuse, a position
# or statistic pandas did not read as `int()` or `float()` would, a frequency
# whose absence the tally must explain -- is read by `_source_row` from its
# own cells.

#: Decompressed bytes one block reads, 32 MiB: what bounds a Pass 2 worker's
#: batch, which holds one block and the line the read before it split. R13's
#: 21,326,102-row ACTINOMYCOSIS file reads as 63 batches, the largest of
#: 346,947 rows, and the read's sampled resident memory peaked at 501 MiB
#: (ADR 0076 gives `ru_maxrss` beside it).
_BLOCK_BYTES = 1 << 25

#: Associations a batch holds once a source is handed to the per-row reader.
_ROW_BATCH = 250_000

#: The cells a statistic reads as missing: `tabular.parse_finite_float`'s and
#: `stats.parse_af`'s set, which a test holds this to.
_MISSING_STATISTICS = ("", ".", "NA", "NaN", "None", "nan")

#: How pandas reads a block: every line one row, every field as written.
_PANDAS = {
    "sep": "\t",
    "header": None,
    "index_col": False,
    "engine": "c",
    "quoting": csv.QUOTE_NONE,
    "skip_blank_lines": False,
    "keep_default_na": False,
    "low_memory": False,
}


@dataclass(frozen=True)
class _Plan:
    """Where a source's header puts each cell an association reads.

    ``names`` are the header's names as `csv` reads a header with no quote.
    A repeated name is its last column, as a `csv.DictReader` row holds it.
    """

    names: list[str]
    reads: dict[str, int]

    @classmethod
    def of(cls, header: bytes) -> _Plan | str:
        """The plan, or why the columnar parse leaves the source to the per-row
        reader: a header `csv` reads otherwise than by splitting at tabs, one
        that is not UTF-8, or one naming no chromosome, position or allele."""
        names = _header_names(header)
        if isinstance(names, str):
            return names
        at = {name: index for index, name in enumerate(names)}
        chromosome = "#chrom" if "#chrom" in at else "chrom"
        reads = {name: at[name] for name in (chromosome, *_READ) if name in at}
        missing = _missing_identity(reads)
        if missing:
            return f"its header names no {', '.join(missing)} column"
        return cls(names, reads)

    @property
    def chromosome(self) -> int:
        return self.reads["#chrom"] if "#chrom" in self.reads else self.reads["chrom"]


def _missing_identity(reads: dict[str, int]) -> list[str]:
    """The identity columns a header's ``reads`` lack, by the names it may use."""
    missing = [name for name in ("pos", "ref", "alt") if name not in reads]
    if "#chrom" not in reads and "chrom" not in reads:
        missing.insert(0, "#chrom or chrom")
    return missing


def _header_names(header: bytes) -> list[str] | str:
    """A header line's names, as `csv` reads a line it splits only at tabs, or
    why it cannot be read so."""
    if not header.endswith(b"\n") or len(header) < 2:
        return "its header is not one whole line"
    special = _first_special(header, ascii_only=False)
    if special is not None:
        return f"its header holds {special[1]}"
    try:
        return header[:-1].decode("utf-8").split("\t")
    except UnicodeDecodeError:
        return "its header is not UTF-8"


#: The cells an association reads beyond its chromosome.
_READ = ("pos", "ref", "alt", "beta", "sebeta", "af_alt")


#: The bytes `csv` reads otherwise than as text between tabs and newlines.
_SPECIAL = ((b'"', "a quote"), (b"\r", "a carriage return"), (b"\x00", "a NUL byte"))
_NON_ASCII = re.compile(rb"[\x80-\xff]")


def _first_special(data: bytes, *, ascii_only: bool = True) -> tuple[int, str] | None:
    """Where ``data`` first holds a byte `csv` reads otherwise than by splitting
    at tabs and newlines -- a quote, a carriage return, a NUL, or (with
    ``ascii_only``) any byte outside ASCII -- and what it is; ``None`` for none."""
    found = [(at, name) for byte, name in _SPECIAL if (at := data.find(byte)) >= 0]
    if ascii_only and not data.isascii():
        outside = _NON_ASCII.search(data)
        if outside is not None:
            found.append((outside.start(), "a byte outside ASCII"))
    return min(found) if found else None


def _without_blank_lines(lines: bytes) -> bytes:
    """``lines`` less every empty line, which `csv.DictReader` skips."""
    lines = lines.lstrip(b"\n")
    # `find`, not `in`: on a block of real FinnGen text it is the quicker scan.
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


def _association_batches(
    path: str | Path, tally: interface.StreamTally, scale: StoredEffectScale
) -> Iterator[AssociationBatch]:
    """A source's associations, a block of whole lines at a time (`_Source`)."""
    with _opener(path)(path, "rb") as fh:
        source = _Source(path, fh, tally, scale)
        header = fh.readline()
        plan = _Plan.of(header)
        if not header:
            return
        if isinstance(plan, str):
            yield from source.per_row(header, None, 0, plan)
        else:
            yield from source.blocks(plan, len(header))
        source.summarise()


@dataclass
class _Source:
    """One FinnGen source read a block at a time, and what the read fills.

    A line split across two reads is carried to the next block, and a last line
    with no newline is read as if it had one, as `csv` reads it. From the first
    block that is not plain, the rest goes to the per-row reader. The rows read
    the same either way; the per-row reader is about five times slower, so the
    first place a source is read row by row is logged, by byte offset, and once
    the source is read one more line says how much of it was (`summarise`).
    """

    path: str | Path
    fh: io.BufferedIOBase
    tally: interface.StreamTally
    scale: StoredEffectScale
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
            log.warning("FinnGen source %s: " + message, self.path, *args)
            self.warned = True

    def summarise(self) -> None:
        """Say, once the source is read, how much of it the per-row rules read."""
        if self.per_row_blocks or self.undecided:
            log.warning(
                "FinnGen source %s: %d of %d blocks read row by row, and %d rows of the "
                "others read from their own line",
                self.path, self.per_row_blocks, self.column_blocks + self.per_row_blocks,
                self.undecided,
            )

    def per_row(
        self, head: bytes, names: list[str] | None, offset: int, reason: str
    ) -> Iterator[AssociationBatch]:
        """``head`` and the rest of the source, from byte ``offset``, row by row."""
        self.warn("read row by row from byte %d on, since %s", offset, reason)
        text = _text_after(head, self._rest())
        yield from _per_row_batches(text, names, self.tally, self.scale)

    def _rest(self) -> Iterator[bytes]:
        """The source's remaining blocks, each counted as read row by row."""
        while block := self.fh.read(_BLOCK_BYTES):
            self.per_row_blocks += 1
            yield block

    def blocks(self, plan: _Plan, offset: int) -> Iterator[AssociationBatch]:
        """The source after its header, which ends at byte ``offset``."""
        carry = b""
        while block := self.fh.read(_BLOCK_BYTES):
            data = carry + block
            cut = data.rfind(b"\n") + 1
            lines, carry = data[:cut], data[cut:]
            special = _first_special(lines)
            if special is not None:
                at, what = special
                self.per_row_blocks += 1
                yield from self.per_row(data, plan.names, offset + at, f"it holds {what}")
                return
            yield from _block_batches(lines, offset, plan, self)
            offset += cut
        special = _first_special(carry)
        if special is not None:
            at, what = special
            self.per_row_blocks += 1
            yield from self.per_row(carry, plan.names, offset + at, f"it holds {what}")
        elif carry:
            yield from _block_batches(carry + b"\n", offset, plan, self)


def _text_after(head: bytes, rest: Iterator[bytes]) -> io.TextIOWrapper:
    """``head``, then the rest of the source, as text the per-row reader reads."""
    source = io.BufferedReader(_Resumed(head, rest))
    return io.TextIOWrapper(source, encoding="utf-8", newline="")


def _per_row_batches(
    text: io.TextIOBase,
    names: list[str] | None,
    tally: interface.StreamTally,
    scale: StoredEffectScale,
) -> Iterator[AssociationBatch]:
    """Source text read by the per-row reader, and batched.

    ``names`` are the header's, already read; ``None`` when ``text`` starts at
    the header.
    """
    rows = csv.DictReader(text, fieldnames=names, delimiter="\t")
    associations = stream_associations(_source_rows(rows, tally, tabular.parse_effect), scale)
    yield from batched(associations, rows=_ROW_BATCH)


def _block_batches(
    lines: bytes, offset: int, plan: _Plan, source: _Source
) -> Iterator[AssociationBatch]:
    """One block's associations, as one batch, or none where it holds none.

    ``lines`` start at byte ``offset`` of the source.
    """
    lines = _without_blank_lines(lines)
    if not lines:
        return
    try:
        block = _Block.read(lines, plan)
    except pd.errors.ParserError as refused:
        # pandas refuses a block none of whose lines reaches the header's last
        # column; `csv` pads each, and reads it.
        source.warn("the block at byte %d is read row by row: pandas refused it (%s)",
                    offset, str(refused).strip())
        source.per_row_blocks += 1
        text = io.TextIOWrapper(io.BytesIO(lines), encoding="ascii", newline="")
        yield from _per_row_batches(text, plan.names, source.tally, source.scale)
        return
    source.column_blocks += 1
    batch = block.associations(lines, plan, source)
    if len(batch):
        yield batch


@dataclass(frozen=True)
class _Cells:
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
    def of(cls, column: pd.Series, rule: Callable[[str], str]) -> _Cells:
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
            raise ValueError(f"FinnGen block column {column.name!r} read a cell as missing")
        return cls(verbatim, value, ok, codes)

    def rows(self, of: np.ndarray) -> np.ndarray:
        """``of`` (one entry per distinct cell) for each row."""
        selected: np.ndarray = of[self.codes]
        return selected


def _orientation(ref: _Cells, alt: _Cells) -> tuple[np.ndarray, np.ndarray]:
    """Per row: whether `alt`, the effect allele, is A2, so z is negated
    (`alid_alleles`: an indel spelt reference first, any other pair sorted),
    and whether the two alleles are one."""
    spelt = np.unique(np.concatenate((ref.value, alt.value)))
    ref_rank = ref.rows(np.searchsorted(spelt, ref.value))
    alt_rank = alt.rows(np.searchsorted(spelt, alt.value))
    lengths = np.vectorize(len, otypes=[np.int64])
    indel = ref.rows(lengths(ref.value)) != alt.rows(lengths(alt.value))
    return indel | (ref_rank < alt_rank), ref_rank == alt_rank


def _numbers(frame: pd.DataFrame, index: int | None, kind: str) -> np.ndarray | None:
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


@dataclass(frozen=True)
class _Block:
    """One block's identity cells and numbers, before any row is decided."""

    chromosome: _Cells
    ref: _Cells
    alt: _Cells
    #: Per row: whether the effect allele is A2, and whether the alleles are one.
    flipped: np.ndarray
    same: np.ndarray
    position: np.ndarray | None
    beta: np.ndarray | None
    se: np.ndarray | None
    af: np.ndarray | None

    @classmethod
    def read(cls, lines: bytes, plan: _Plan) -> _Block:
        reads = plan.reads
        statistics = [reads[name] for name in ("beta", "sebeta", "af_alt") if name in reads]
        frame = pd.read_csv(
            io.BytesIO(lines),
            names=range(len(plan.names)),
            usecols=sorted(reads.values()),
            dtype={reads[name]: "category" for name in ("ref", "alt")}
            | {plan.chromosome: "category"},
            na_values={index: list(_MISSING_STATISTICS) for index in statistics},
            # `float()` is correctly rounded and pandas' default converter is not.
            float_precision="round_trip",
            **_PANDAS,
        )
        ref = _Cells.of(frame[reads["ref"]], normalise_allele)
        alt = _Cells.of(frame[reads["alt"]], normalise_allele)
        return cls(
            _Cells.of(frame[plan.chromosome], normalise_chromosome),
            ref,
            alt,
            *_orientation(ref, alt),
            _numbers(frame, reads["pos"], "i"),
            _numbers(frame, reads.get("beta"), "f"),
            _numbers(frame, reads.get("sebeta"), "f"),
            _numbers(frame, reads.get("af_alt"), "f"),
        )

    def decided(self, plan: _Plan) -> np.ndarray:
        """The rows the columns decide: every identity rule passed, and every
        number read as the per-row reader reads it. A frequency read as
        missing is left to the row, since only its cell says whether the
        tally counts it."""
        rows = len(self.chromosome.codes)
        if self.position is None or self.beta is None or self.se is None or self.af is None:
            return np.zeros(rows, dtype=bool)
        decided: np.ndarray = (
            self.chromosome.rows(self.chromosome.ok)
            & self.ref.rows(self.ref.ok)
            & self.alt.rows(self.alt.ok)
            & (self.position > 0)
            & ~self.same
        )
        if "af_alt" in plan.reads:
            decided &= ~np.isnan(self.af)
        return decided

    def associations(self, lines: bytes, plan: _Plan, source: _Source) -> AssociationBatch:
        """The block's associations: the decided rows from the columns, the
        rest from `_source_row`, in the block's order."""
        decided = self.decided(plan)
        columns = _Columns.of(self, decided, source.tally)
        undecided = np.flatnonzero(~decided)
        if len(undecided):
            source.undecided += len(undecided)
            columns.place(_undecided(lines, plan, undecided, source.tally, source.scale))
        return columns.batch()


#: The largest position an int64 column holds; a row's beyond it keeps its
#: Python int, as the per-row reader's association does.
_INT64_MAX = np.iinfo(np.int64).max


@dataclass
class _Columns:
    """A block's association columns, an entry a row, and the rows that hold one."""

    chromosome: np.ndarray
    position: np.ndarray
    ref: np.ndarray
    alt: np.ndarray
    z: np.ndarray
    se: np.ndarray
    eaf: np.ndarray
    kept: np.ndarray

    @classmethod
    def of(cls, block: _Block, decided: np.ndarray, tally: interface.StreamTally) -> _Columns:
        """The decided rows' associations, by `_source_row`'s rules as arrays:
        z is beta/se where both are reported (else NaN), negated when the
        effect allele is A2; SE is the reported one where positive and finite;
        the frequency is the effect allele's where inside (0, 1), complemented
        with z, and any other reported one is counted as made absent (D7)."""
        rows = len(decided)
        missing = np.full(rows, np.nan)
        beta = missing if block.beta is None else block.beta
        se = missing if block.se is None else block.se
        af = missing if block.af is None else block.af
        se_ok = np.isfinite(se) & (se > 0)
        reported = np.divide(
            beta, se, out=np.full(rows, np.nan), where=decided & ~np.isnan(beta) & se_ok
        )
        af_ok = (af > 0) & (af < 1)
        tally.frequencies.made_absent += int(np.count_nonzero(decided & ~np.isnan(af) & ~af_ok))
        return cls(
            chromosome=block.chromosome.rows(block.chromosome.value),
            # A copy: pandas' own column is read-only, and an undecided row's
            # position is placed in it.
            position=(
                np.zeros(rows, dtype=np.int64) if block.position is None else block.position.copy()
            ),
            ref=block.ref.rows(block.ref.verbatim),
            alt=block.alt.rows(block.alt.verbatim),
            z=np.where(block.flipped, -reported, reported),
            se=np.where(se_ok, se, np.nan),
            eaf=np.where(af_ok, np.where(block.flipped, 1.0 - af, af), np.nan),
            kept=decided.copy(),
        )

    def place(self, rows: Iterable[tuple[int, ReaderAssociation | None]]) -> None:
        """Put each undecided row's association, where it has one, in its place."""
        for row, association in rows:
            if association is None:
                continue
            if association.position > _INT64_MAX and self.position.dtype != object:
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


def _undecided(
    lines: bytes,
    plan: _Plan,
    rows: np.ndarray,
    tally: interface.StreamTally,
    scale: StoredEffectScale,
) -> Iterator[tuple[int, ReaderAssociation | None]]:
    """Each of ``rows``, read by `_source_row` from its own line.

    The block is plain, so a line's cells are what lies between its tabs, as
    `csv` reads them, and a cell a short line does not reach is ``None``, as a
    `csv.DictReader` row holds it.
    """
    ends = np.flatnonzero(np.frombuffer(lines, dtype=np.uint8) == ord("\n"))
    starts = np.concatenate(([0], ends[:-1] + 1))
    faults, frequencies = tally.faults, tally.frequencies
    for row in rows.tolist():
        cells = lines[starts[row] : ends[row]].decode("ascii").split("\t")
        cell = {name: cells[index] if index < len(cells) else None
                for name, index in plan.reads.items()}
        source_row = _source_row(cell, faults, frequencies, tabular.parse_effect)
        if source_row is None:
            yield row, None
            continue
        yield row, next(stream_associations([source_row], scale))


@dataclass(frozen=True)
class FinnGenR13Reader:
    """Reader for one FinnGen R13 endpoint summary-statistics file."""

    path: str | Path
    stored_effect_scale: StoredEffectScale = StoredEffectScale.LOG_OR

    def stream_associations(
        self, tally: interface.StreamTally | None = None
    ) -> Iterator[ReaderAssociation]:
        yield from tabular.stream_associations(
            _iter_rows(self.path, tally), self.stored_effect_scale
        )

    def stream_association_batches(
        self, tally: interface.StreamTally | None = None
    ) -> Iterator[AssociationBatch]:
        """`stream_associations`, a block of the file at a time, as columns
        (#318): what Pass 2 routes. The same associations bit for bit, and
        the same ``tally``; see `_association_batches`."""
        yield from _association_batches(
            self.path, tally or interface.StreamTally(), self.stored_effect_scale
        )

    def stream_batches(self, tally: interface.StreamTally | None = None) -> Iterator[ReaderBatch]:
        yield from stream_batches(_iter_rows(self.path, tally))

    def stream_metric_chunks(self) -> Iterator[tabular.MetricsChunk]:
        """The resolver's blocks, from the full-row parser: FinnGen has no
        column-projected scan, and its rows read the same."""
        yield from tabular.metrics_chunks_from_rows(
            _iter_rows(self.path, infinite_effects=False)
        )

    def stream_variants(self) -> Iterator[SourceVariant]:
        yield from _iter_variants(self.path)

    def extract_at_sites(
        self, alids: Iterable[str], *, region: str | None = None, regions_file: RegionsFile = None
    ) -> dict[str, SiteMetrics]:
        return tabular.extract_at_sites(_iter_rows(self.path), alids, region=region)
