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
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import opengwasdb.readers.columnar as columnar
import opengwasdb.readers.interface as interface
import opengwasdb.readers.tabular as tabular
from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers.interface import (
    AssociationBatch,
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
# decompressed file at a time (`opengwasdb.readers.columnar` reads the blocks
# and falls back to the per-row reader where it must). The per-row reader
# (`_source_row`) is the reference: a block holds exactly its associations, in
# order, bit for bit.

#: Decompressed bytes one block reads, 32 MiB: what bounds a Pass 2 worker's
#: batch, which holds one block and the line the read before it split. R13's
#: 21,326,102-row ACTINOMYCOSIS file reads as 63 batches, the largest of
#: 346,947 rows, and the read's sampled resident memory peaked at 501 MiB
#: (ADR 0076 gives `ru_maxrss` beside it).
_BLOCK_BYTES = 1 << 25

_MISSING_STATISTICS = columnar.MISSING_STATISTICS

#: The cells an association reads beyond its chromosome.
_READ = ("pos", "ref", "alt", "beta", "sebeta", "af_alt")


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
        names = columnar.header_names(header)
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


def _association_batches(
    path: str | Path, tally: interface.StreamTally, scale: StoredEffectScale
) -> Iterator[AssociationBatch]:
    """A source's associations, a block of whole lines at a time."""

    def start(header: bytes) -> tuple[columnar.Format, list[str] | None, str | None]:
        found = _Plan.of(header)
        if isinstance(found, str):
            return _Format(None), None, found
        return _Format(found), found.names, None

    with _opener(path)(path, "rb") as fh:
        yield from columnar.association_batches(fh, path, tally, scale, start)


@dataclass(frozen=True)
class _Format:
    """FinnGen's columns, to the engine; ``plan`` is ``None`` for a source the
    per-row reader reads throughout."""

    plan: _Plan | None
    log: logging.Logger = log
    label: str = "FinnGen source"

    def block_bytes(self) -> int:
        return _BLOCK_BYTES

    def per_row(
        self, text: io.TextIOBase, names: list[str] | None, source: columnar.Source
    ) -> Iterator[AssociationBatch]:
        yield from _per_row_batches(text, names, source.tally, source.scale)

    def block_batches(
        self, lines: bytes, offset: int, source: columnar.Source
    ) -> Iterator[AssociationBatch]:
        assert self.plan is not None
        yield from _block_batches(lines, offset, self.plan, source)


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
    yield from batched(associations, rows=columnar.ROW_BATCH)


def _block_batches(
    lines: bytes, offset: int, plan: _Plan, source: columnar.Source
) -> Iterator[AssociationBatch]:
    """One block's associations, as one batch, or none where it holds none.

    ``lines`` start at byte ``offset`` of the source.
    """
    yield from columnar.block_batches(
        lines,
        offset,
        source,
        lambda text: _Block.read(text, plan),
        lambda block, text: block.associations(text, plan, source),
        plan.names,
    )


@dataclass(frozen=True)
class _Block:
    """One block's identity cells and numbers, before any row is decided."""

    chromosome: columnar.Cells
    ref: columnar.Cells
    alt: columnar.Cells
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
            **columnar.PANDAS,
        )
        ref = columnar.Cells.of(frame[reads["ref"]], normalise_allele)
        alt = columnar.Cells.of(frame[reads["alt"]], normalise_allele)
        return cls(
            columnar.Cells.of(frame[plan.chromosome], normalise_chromosome),
            ref,
            alt,
            *columnar.orientation(ref, alt),
            columnar.numbers(frame, reads["pos"], "i"),
            columnar.numbers(frame, reads.get("beta"), "f"),
            columnar.numbers(frame, reads.get("sebeta"), "f"),
            columnar.numbers(frame, reads.get("af_alt"), "f"),
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

    def associations(
        self, lines: bytes, plan: _Plan, source: columnar.Source
    ) -> AssociationBatch:
        """The block's associations: the decided rows from the columns, the
        rest from `_source_row`, in the block's order."""
        decided = self.decided(plan)
        columns = self.columns(decided, source.tally)
        undecided = np.flatnonzero(~decided)
        if len(undecided):
            source.undecided += len(undecided)
            columns.place(_undecided(lines, plan, undecided, source.tally, source.scale))
        return columns.batch()

    def columns(self, decided: np.ndarray, tally: interface.StreamTally) -> columnar.Columns:
        """The decided rows' associations, by `_source_row`'s rules as arrays:
        z is beta/se where both are reported (else NaN), negated when the
        effect allele is A2; SE is the reported one where positive and finite;
        the frequency is the effect allele's where inside (0, 1), complemented
        with z, and any other reported one is counted as made absent (D7)."""
        rows = len(decided)
        missing = np.full(rows, np.nan)
        beta = missing if self.beta is None else self.beta
        se = missing if self.se is None else self.se
        af = missing if self.af is None else self.af
        se_ok = np.isfinite(se) & (se > 0)
        reported = np.divide(
            beta, se, out=np.full(rows, np.nan), where=decided & ~np.isnan(beta) & se_ok
        )
        af_ok = (af > 0) & (af < 1)
        tally.frequencies.made_absent += int(np.count_nonzero(decided & ~np.isnan(af) & ~af_ok))
        return columnar.Columns(
            chromosome=self.chromosome.rows(self.chromosome.value),
            # A copy: pandas' own column is read-only, and an undecided row's
            # position is placed in it.
            position=(
                np.zeros(rows, dtype=np.int64) if self.position is None else self.position.copy()
            ),
            ref=self.ref.rows(self.ref.verbatim),
            alt=self.alt.rows(self.alt.verbatim),
            z=np.where(self.flipped, -reported, reported),
            se=np.where(se_ok, se, np.nan),
            eaf=np.where(af_ok, np.where(self.flipped, 1.0 - af, af), np.nan),
            kept=decided.copy(),
        )


def _undecided(
    lines: bytes,
    plan: _Plan,
    rows: np.ndarray,
    tally: interface.StreamTally,
    scale: StoredEffectScale,
) -> Iterator[tuple[int, ReaderAssociation | None]]:
    """Each of ``rows``, read by `_source_row` from its own line."""
    faults, frequencies = tally.faults, tally.frequencies
    for row, cell in columnar.line_cells(lines, plan.reads, rows):
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
