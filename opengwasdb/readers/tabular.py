"""Shared mechanics for canonical tabular summary-statistics readers."""

from __future__ import annotations

import csv
import gzip
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers.gwas_vcf import is_palindromic
from opengwasdb.readers.interface import ReaderAssociation, SiteMetrics, SourceVariant
from opengwasdb.stats import parse_af
from opengwasdb.variants.normalise import (
    VariantNormalisationError,
    normalise_allele,
    normalise_chromosome,
)

_MISSING = {"", ".", "NA", "NaN", "nan", "None"}
_VALID_DISTINCT_ALLELE_CODES = frozenset(
    (ref, alt) for ref in b"ACGT" for alt in b"ACGT" if ref != alt
)
_CHROMOSOME_23_SENTINELS = (b"\0", b"23")
_SINGLE_ALLELE_TEXT = {bytes((code,)): chr(code) for code in b"ACGTacgt"}


def parse_finite_float(value: str | None) -> float | None:
    if value is None or value.strip() in _MISSING:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def parse_positive_float(value: str | None) -> float | None:
    parsed = parse_finite_float(value)
    return parsed if parsed is not None and parsed > 0.0 else None


__all__ = [
    "TabularRow",
    "VariantProjectionColumns",
    "extract_at_sites",
    # Re-exported: the readers built on this module import their column
    # parsing from here, but the allele-frequency rule itself is shared with
    # every other source path (ADR 0036), so it lives in opengwasdb.stats.
    "parse_af",
    "parse_finite_float",
    "parse_positive_float",
    "project_source_variant",
    "stream_projected_variants",
    "stream_associations",
    "stream_variants",
]


@dataclass(frozen=True)
class VariantProjectionColumns:
    """Provider column names needed by a variant-only tabular scan."""

    chromosome: tuple[bytes, ...]
    position: bytes
    ref: bytes
    alt: bytes
    aliases: tuple[bytes, ...] = ()


@dataclass(frozen=True)
class _ResolvedProjection:
    chromosome: int
    position: int
    ref: int
    alt: int
    first_alias: int | None
    second_alias: int | None
    last_required: int
    split_limit: int


def _normalise_projected_chromosome(value: bytes, chromosome_23_is_x: bool) -> str:
    stripped = value.strip()
    if chromosome_23_is_x and stripped == b"23":
        return "X"
    if stripped.isdigit():
        return stripped.decode("ascii")
    return normalise_chromosome(value.decode("utf-8"))


def _validate_projected_alleles(ref_bytes: bytes, alt_bytes: bytes) -> tuple[str, str]:
    ref = ref_bytes.decode("utf-8")
    alt = alt_bytes.decode("utf-8")
    ref_code = ref_bytes[0] & 0xDF if len(ref_bytes) == 1 else 0
    alt_code = alt_bytes[0] & 0xDF if len(alt_bytes) == 1 else 0
    if ref_code in b"ACGT" and alt_code in b"ACGT" and ref_code != alt_code:
        return ref, alt
    if normalise_allele(ref) == normalise_allele(alt):
        raise VariantNormalisationError("effect and other alleles are identical")
    return ref, alt


def project_source_variant(
    chromosome_bytes: bytes,
    position_bytes: bytes,
    ref_bytes: bytes,
    alt_bytes: bytes,
    rsid: str,
    *,
    chromosome_23_is_x: bool = False,
) -> SourceVariant | None:
    """Validate projected identity fields without constructing an orientation.

    Numeric chromosomes and single-base alleles are the common production path.
    Less common labels and long alleles fall back to the same normalizers used
    by full association parsing, while parity tests protect the observable
    reader contract (issue #179).
    """
    try:
        stripped_chromosome = chromosome_bytes.strip()
        if stripped_chromosome == _CHROMOSOME_23_SENTINELS[chromosome_23_is_x]:
            chromosome = "X"
        elif stripped_chromosome.isdigit():
            chromosome = stripped_chromosome.decode("ascii")
        else:
            chromosome = _normalise_projected_chromosome(
                chromosome_bytes, chromosome_23_is_x
            )
        position = int(position_bytes)
        if position <= 0:
            raise VariantNormalisationError(f"invalid position {position_bytes!r}")
        ref_code = ref_bytes[0] & 0xDF if len(ref_bytes) == 1 else 0
        alt_code = alt_bytes[0] & 0xDF if len(alt_bytes) == 1 else 0
        if (ref_code, alt_code) in _VALID_DISTINCT_ALLELE_CODES:
            ref = _SINGLE_ALLELE_TEXT[ref_bytes]
            alt = _SINGLE_ALLELE_TEXT[alt_bytes]
        else:
            ref, alt = _validate_projected_alleles(ref_bytes, alt_bytes)
    except (UnicodeDecodeError, ValueError, VariantNormalisationError):
        return None

    # Frozen dataclass initialization performs five guarded assignments; at
    # 21M rows that costs seconds. Validation is complete, so one dictionary
    # update preserves the exact public value without five guarded writes.
    variant = object.__new__(SourceVariant)
    object.__setattr__(
        variant,
        "__dict__",
        {
            "chromosome": chromosome,
            "position": position,
            "ref": ref,
            "alt": alt,
            "rsid": rsid,
        },
    )
    return variant


def _first_named_index(indexes: dict[bytes, int], names: tuple[bytes, ...]) -> int | None:
    for name in names:
        if name in indexes:
            return indexes[name]
    return None


def _alias_pair(
    indexes: dict[bytes, int], names: tuple[bytes, ...]
) -> tuple[int | None, int | None]:
    aliases = [indexes.get(name) for name in names[:2]]
    aliases += [None] * (2 - len(aliases))
    return aliases[0], aliases[1]


def _resolve_projection(
    header: list[bytes], columns: VariantProjectionColumns
) -> _ResolvedProjection | None:
    indexes = {name: index for index, name in enumerate(header)}
    chromosome = _first_named_index(indexes, columns.chromosome)
    position = indexes.get(columns.position)
    ref = indexes.get(columns.ref)
    alt = indexes.get(columns.alt)
    if chromosome is None or position is None or ref is None or alt is None:
        return None
    aliases = _alias_pair(indexes, columns.aliases)
    selected = (chromosome, position, ref, alt) + tuple(
        index for index in aliases if index is not None
    )
    last_selected = max(selected)
    return _ResolvedProjection(
        chromosome=chromosome,
        position=position,
        ref=ref,
        alt=alt,
        first_alias=aliases[0],
        second_alias=aliases[1],
        last_required=max(chromosome, position, ref, alt),
        split_limit=last_selected + (last_selected < len(header) - 1),
    )


def _fallback_rsid(
    row: list[bytes], first_alias: int | None, second_alias: int | None
) -> str:
    first = (
        row[first_alias].decode("utf-8").strip()
        if first_alias is not None and first_alias < len(row)
        else ""
    )
    if first.startswith("rs"):
        return first
    if second_alias is not None and second_alias < len(row):
        second = row[second_alias].decode("utf-8").strip()
        if second.startswith("rs"):
            return second
    return ""


def _required_projection(
    path: str | Path,
    header_line: bytes,
    columns: VariantProjectionColumns,
) -> _ResolvedProjection:
    header = header_line.rstrip(b"\r\n").split(b"\t")
    if b'"' in header_line:
        fields = next(csv.reader([header_line.decode("utf-8")], delimiter="\t"))
        header = [field.encode("utf-8") for field in fields]
    projection = _resolve_projection(header, columns)
    if projection is not None:
        return projection
    expected = (
        b"/".join(columns.chromosome),
        columns.position,
        columns.ref,
        columns.alt,
    )
    names = ", ".join(name.decode("ascii") for name in expected)
    raise ValueError(f"{path}: missing required variant columns; expected {names}")


def _projected_variant(
    row: list[bytes],
    projection: _ResolvedProjection,
    chromosome_23_is_x: bool,
) -> tuple[SourceVariant, ...]:
    try:
        rsid = _fallback_rsid(
            row, projection.first_alias, projection.second_alias
        )
    except UnicodeDecodeError:
        return ()
    variant = project_source_variant(
        row[projection.chromosome],
        row[projection.position],
        row[projection.ref],
        row[projection.alt],
        rsid,
        chromosome_23_is_x=chromosome_23_is_x,
    )
    return () if variant is None else (variant,)


def stream_projected_variants(
    path: str | Path,
    columns: VariantProjectionColumns,
    *,
    chromosome_23_is_x: bool = False,
) -> Iterator[SourceVariant]:
    """Stream header-indexed identity fields without parsing statistics."""
    opener = gzip.open if str(path).endswith((".gz", ".bgz")) else open
    with opener(path, "rb") as fh:
        projection = _required_projection(path, fh.readline(), columns)
        last_required = projection.last_required
        split_limit = projection.split_limit
        for line in fh:
            row = line.split(b"\t", split_limit)
            if b'"' in line:
                fields = next(csv.reader([line.decode("utf-8")], delimiter="\t"))
                row = [field.encode("utf-8") for field in fields]
            row[-1] = row[-1].rstrip(b"\r\n")
            if len(row) <= last_required:
                continue
            yield from _projected_variant(
                row, projection, chromosome_23_is_x
            )


@dataclass(frozen=True)
class TabularRow:
    """Format-neutral row after source-specific column mapping."""

    chromosome: str
    position: int
    ref: str
    alt: str
    alid: str
    flipped: bool
    beta: float | None
    se: float | None
    af_alt: float | None
    rsid: str = ""  # the source's own identifier for this row; "" when it names none


def stream_associations(
    rows: Iterable[TabularRow], stored_effect_scale: StoredEffectScale
) -> Iterator[ReaderAssociation]:
    for row in rows:
        if row.beta is None or row.se is None:
            continue
        z = row.beta / row.se
        eaf = row.af_alt
        if row.flipped:
            z = -z
            # `af_alt` is the frequency of the source's alt allele; flipping
            # swapped which allele is stored as the effect one (ADR 0036).
            eaf = None if eaf is None else 1.0 - eaf
        yield ReaderAssociation(
            chromosome=row.chromosome,
            position=row.position,
            ref=row.ref,
            alt=row.alt,
            z=z,
            se=row.se,
            stored_effect_scale=stored_effect_scale,
            eaf=eaf,
        )


def stream_variants(rows: Iterable[TabularRow]) -> Iterator[SourceVariant]:
    for row in rows:
        yield SourceVariant(
            chromosome=row.chromosome,
            position=row.position,
            ref=row.ref,
            alt=row.alt,
            rsid=row.rsid,
        )


def extract_at_sites(
    rows: Iterable[TabularRow], alids: Iterable[str]
) -> dict[str, SiteMetrics]:
    wanted = alids if isinstance(alids, (set, frozenset, dict)) else set(alids)
    if not wanted:
        return {}
    result: dict[str, SiteMetrics] = {}
    for row in rows:
        if row.alid not in wanted or row.af_alt is None or row.se is None:
            continue
        if is_palindromic(row.ref, row.alt):
            continue
        result[row.alid] = SiteMetrics(
            af=(1.0 - row.af_alt) if row.flipped else row.af_alt,
            se=row.se,
        )
    return result
