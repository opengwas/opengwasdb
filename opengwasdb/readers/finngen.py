"""FinnGen R13 tabular summary-statistics SourceReader.

FinnGen publishes one tab-delimited, bgzip-compressed file per endpoint on
GRCh38.  ``ref`` is the reference allele and ``alt`` the effect allele; the
reader therefore converts ``beta / sebeta`` to the package-wide canonical A1
orientation while leaving the source REF/ALT labels intact on streamed
records, each declaring ``ref`` the reference.
"""

from __future__ import annotations

import csv
import gzip
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import opengwasdb.readers.interface as interface
import opengwasdb.readers.tabular as tabular
from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers.interface import (
    ReaderAssociation,
    RegionsFile,
    SiteMetrics,
    SourceVariant,
)
from opengwasdb.readers.rows import ReaderBatch, stream_batches
from opengwasdb.variants.normalise import (
    normalise_chromosome,
    orient_or_drop,
)

FINNGEN_R13_CAPABILITY = "opengwasdb.finngen-r13"
#: FinnGen's `ref` is the GRCh38 reference allele, on every row.
_REFERENCE = interface.ReferenceAllele.REF
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
    tally = tally or interface.StreamTally()
    faults, frequencies = tally.faults, tally.frequencies
    opener = gzip.open if str(path).endswith((".gz", ".bgz")) else open
    with opener(path, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            # R13 uses #chrom. Accepting chrom as well preserves compatibility
            # with older captured FinnGen releases without changing semantics.
            chromosome = row.get("#chrom", row.get("chrom"))
            ref = row.get("ref")
            alt = row.get("alt")
            # A row too short to reach an allele has a missing one, which
            # `faults` names (#365); the None checks are the type checker's.
            orientation = orient_or_drop(
                chromosome, row.get("pos"), alt, ref, faults, other_is_reference=True
            )
            if orientation is None or ref is None or alt is None:
                continue
            yield tabular.SourceRow(
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
    opener = gzip.open if str(path).endswith((".gz", ".bgz")) else open
    with opener(path, "rb") as fh:
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


@dataclass(frozen=True)
class FinnGenR13Reader:
    """Reader for one FinnGen R13 endpoint summary-statistics file."""

    path: str | Path
    stored_effect_scale: StoredEffectScale = StoredEffectScale.LOG_OR

    def stream_associations(
        self, tally: interface.StreamTally | None = None
    ) -> Iterator[ReaderAssociation]:
        yield from tabular.stream_associations(
            _iter_rows(self.path, tally), self.stored_effect_scale, _REFERENCE
        )

    def stream_batches(self, tally: interface.StreamTally | None = None) -> Iterator[ReaderBatch]:
        yield from stream_batches(_iter_rows(self.path, tally), _REFERENCE)

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
