"""Shared mechanics for canonical tabular summary-statistics readers."""

from __future__ import annotations

import csv
import gzip
import math
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers.gwas_vcf import is_palindromic
from opengwasdb.readers.interface import ReaderAssociation, SiteMetrics, SourceVariant
from opengwasdb.stats import parse_af
from opengwasdb.variants.normalise import (
    VALID_BASES,
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
    "MetricsProjectionColumns",
    "TabularMetricsRow",
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
    "stream_projected_metrics",
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
class MetricsProjectionColumns:
    """Provider column names needed by a metrics-only tabular scan (issue #207).

    Identity columns are required. `frequency`, `beta` and `standard_error` are
    optional on purpose: a harmonised file that reports `odds_ratio` instead of
    `beta` still names every variant, and ancestry assignment reads only
    frequencies -- refusing the whole file over a column one of the two stages
    never reads would throw away the other stage's evidence with it.
    """

    chromosome: tuple[bytes, ...]
    position: bytes
    ref: bytes
    alt: bytes
    frequency: bytes
    beta: bytes
    standard_error: bytes


@dataclass(frozen=True)
class TabularMetricsRow:
    """One source row's variant identity and the statistics both pre-build
    annotation stages read (issue #207).

    The parent of :class:`TabularRow`, which adds the source's own identifier for
    the row and nothing else. The split is not cosmetic: the one-pass resolver
    reads a genome-wide file per Analysis and never looks at an rsid, so the
    metrics projection does not pay for the identifier columns or for decoding
    them -- and a row shape that could not name an rsid is not a row shape
    anything else should be handed.

    ``af_alt`` is the source's own frequency for its own effect allele -- not
    A1-oriented, which is why ``flipped`` travels beside it (ADR 0036). ``ref``
    and ``alt`` are the source's own labels, verbatim.
    """

    chromosome: str
    position: int
    ref: str
    alt: str
    alid: str
    flipped: bool
    af_alt: float | None
    beta: float | None
    se: float | None


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


def _header_identity(
    header: list[bytes],
    columns: VariantProjectionColumns | MetricsProjectionColumns,
) -> tuple[dict[bytes, int], tuple[int, int, int, int]] | None:
    """A header's column index and its four required identity column indexes.

    ``None`` when the header does not name all four, which is the one thing both
    projections cannot proceed without; everything else each reads is its own
    business. Sharing this is what keeps "which columns a scan needs" stated
    once rather than once per projection.
    """
    indexes = {name: index for index, name in enumerate(header)}
    chromosome = _first_named_index(indexes, columns.chromosome)
    position = indexes.get(columns.position)
    ref = indexes.get(columns.ref)
    alt = indexes.get(columns.alt)
    if chromosome is None or position is None or ref is None or alt is None:
        return None
    return indexes, (chromosome, position, ref, alt)


def _resolve_projection(
    header: list[bytes], columns: VariantProjectionColumns
) -> _ResolvedProjection | None:
    found = _header_identity(header, columns)
    if found is None:
        return None
    indexes, (chromosome, position, ref, alt) = found
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


def _header_cells(header_line: bytes) -> list[bytes]:
    """A header line's column names, unquoting a quoted header (issue #179)."""
    header = header_line.rstrip(b"\r\n").split(b"\t")
    if b'"' in header_line:
        fields = next(csv.reader([header_line.decode("utf-8")], delimiter="\t"))
        header = [field.encode("utf-8") for field in fields]
    return header


def _missing_columns(path: str | Path, expected: tuple[bytes, ...]) -> ValueError:
    """The loud failure for a file that does not name the columns a scan needs."""
    names = ", ".join(name.decode("ascii") for name in expected)
    return ValueError(f"{path}: missing required variant columns; expected {names}")


def _required_projection(
    path: str | Path,
    header_line: bytes,
    columns: VariantProjectionColumns,
) -> _ResolvedProjection:
    projection = _resolve_projection(_header_cells(header_line), columns)
    if projection is not None:
        return projection
    raise _missing_columns(
        path,
        (b"/".join(columns.chromosome), columns.position, columns.ref, columns.alt),
    )


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


# Chromosome labels `normalise_chromosome` returns unchanged, so the hot path
# can skip the string work for the labels production data actually uses.
# Anything else -- `chr1`, a contig name, `23` -- falls back to it, which is
# also what keeps this map from silently becoming the rule.
_FAST_CHROMOSOMES: dict[bytes, str] = {
    **{str(number).encode(): str(number) for number in range(1, 23)},
    b"X": "X",
    b"x": "X",
    b"Y": "Y",
    b"y": "Y",
    b"M": "M",
    b"m": "M",
    b"MT": "MT",
    b"mt": "MT",
}


@dataclass(frozen=True)
class _ResolvedMetricsProjection:
    chromosome: int
    position: int
    ref: int
    alt: int
    frequency: int | None
    beta: int | None
    standard_error: int | None
    last_identity: int
    split_limit: int


def _metrics_projection_indexes(
    header: list[bytes], columns: MetricsProjectionColumns
) -> _ResolvedMetricsProjection | None:
    found = _header_identity(header, columns)
    if found is None:
        return None
    indexes, (chromosome, position, ref, alt) = found
    optional = (
        indexes.get(columns.frequency),
        indexes.get(columns.beta),
        indexes.get(columns.standard_error),
    )
    last_identity = max(chromosome, position, ref, alt)
    last_selected = max([last_identity, *(index for index in optional if index is not None)])
    return _ResolvedMetricsProjection(
        chromosome=chromosome,
        position=position,
        ref=ref,
        alt=alt,
        frequency=optional[0],
        beta=optional[1],
        standard_error=optional[2],
        last_identity=last_identity,
        split_limit=last_selected + (last_selected < len(header) - 1),
    )


def _required_metrics_projection(
    path: str | Path, header_line: bytes, columns: MetricsProjectionColumns
) -> _ResolvedMetricsProjection:
    projection = _metrics_projection_indexes(_header_cells(header_line), columns)
    if projection is None:
        raise _missing_columns(
            path,
            (b"/".join(columns.chromosome), columns.position, columns.ref, columns.alt),
        )
    return projection


def _metrics_cell(row: list[bytes], index: int | None) -> bytes:
    """One cell, or ``b""`` when its column is absent or the row stops short.

    A short row is how a truncated file presents itself. Reading the unsplit
    remainder `split` leaves in the final slot instead would turn a missing cell
    into a plausible one.
    """
    if index is None or index >= len(row):
        return b""
    return row[index]


def _metrics_float(
    cell: bytes, parse: Callable[[str | None], float | None]
) -> float | None:
    """A cell parsed by the shared per-format rule, absent cells included."""
    return parse(cell.decode("utf-8") if cell else None)


def _fast_allele(cell: bytes) -> str | None:
    """The normalised allele when the cell is one base, else ``None``."""
    stripped = cell.strip()
    if len(stripped) != 1:
        return None
    base = chr(stripped[0] & 0xDF)
    return base if base in VALID_BASES else None


def _metrics_allele(cell: bytes) -> str:
    """One normalised allele, via the single-base fast path where it applies.

    `normalise_allele` remains the rule: the fast path covers exactly the cells
    it would return unchanged (one base, either case, stripped), and everything
    else -- indels, empty cells, non-ACGT -- is handed to it.
    """
    fast = _fast_allele(cell)
    if fast is not None:
        return fast
    return normalise_allele(cell.decode("utf-8"))


def _metrics_chromosome(cell: bytes) -> str:
    """One normalised chromosome label, fast path first."""
    fast = _FAST_CHROMOSOMES.get(cell.strip())
    return fast if fast is not None else normalise_chromosome(cell.decode("utf-8"))


def _metrics_identity(
    row: list[bytes], projection: _ResolvedMetricsProjection
) -> tuple[str, int, str, str, str, bool] | None:
    """``(chromosome, position, ref, alt, alid, flipped)``, or ``None``.

    ``None`` is the row naming no canonical variant at all -- an unusable
    position or allele pair -- which is the same row `_iter_rows` drops. A cell
    that is not decodable is *not* such a row: it raises, as the full-row parser
    raises, rather than quietly shrinking the file's evidence.
    """
    ref_cell = row[projection.ref]
    alt_cell = row[projection.alt]
    try:
        position = int(row[projection.position])
        if position <= 0:
            return None
        chromosome = _metrics_chromosome(row[projection.chromosome])
        effect = _metrics_allele(alt_cell)
        other = _metrics_allele(ref_cell)
    except (ValueError, VariantNormalisationError):
        return None
    if effect == other:
        return None
    a1, a2 = (effect, other) if effect < other else (other, effect)
    alid = f"{chromosome}:{position}:{a1}:{a2}"
    ref = ref_cell.decode("utf-8")
    alt = alt_cell.decode("utf-8")
    return chromosome, position, ref, alt, alid, effect != a1


def _project_metrics_row(
    row: list[bytes], projection: _ResolvedMetricsProjection
) -> TabularMetricsRow | None:
    identity = _metrics_identity(row, projection)
    if identity is None:
        return None
    chromosome, position, ref, alt, alid, flipped = identity
    return TabularMetricsRow(
        chromosome=chromosome,
        position=position,
        ref=ref,
        alt=alt,
        alid=alid,
        flipped=flipped,
        af_alt=_metrics_float(_metrics_cell(row, projection.frequency), parse_af),
        beta=_metrics_float(_metrics_cell(row, projection.beta), parse_finite_float),
        se=_metrics_float(_metrics_cell(row, projection.standard_error), parse_positive_float),
    )


def _metrics_fields(line: bytes, split_limit: int) -> list[bytes]:
    """One line's cells, without splitting a tail nothing reads.

    `split` leaves the unsplit remainder in the final slot, so every cell before
    `split_limit` is clean. A line carrying a quote is re-split through the csv
    parser, the only thing that unquotes a cell -- the same fallback the variant
    projection uses (issue #179).
    """
    row = line.split(b"\t", split_limit)
    if b'"' in line:
        row = [
            cell.encode("utf-8")
            for cell in next(csv.reader([line.decode("utf-8")], delimiter="\t"))
        ]
    row[-1] = row[-1].rstrip(b"\r\n")
    return row


def stream_projected_metrics(
    path: str | Path, columns: MetricsProjectionColumns
) -> Iterator[TabularMetricsRow]:
    """Stream every row's variant identity and statistics, column-projected.

    The hot path for issue #207's one-pass resolver: a genome-wide file is read
    once per Analysis and only its identity and statistics columns are ever
    wanted, so they are indexed in the header and read by position rather than
    materialising a dict per row. Rows naming no usable variant are dropped, as
    the full-row parser drops them; everything else is yielded in file order.
    """
    opener = gzip.open if str(path).endswith((".gz", ".bgz")) else open
    with opener(path, "rb") as fh:
        projection = _required_metrics_projection(path, fh.readline(), columns)
        for line in fh:
            row = _metrics_fields(line, projection.split_limit)
            if len(row) <= projection.last_identity:
                continue
            projected = _project_metrics_row(row, projection)
            if projected is not None:
                yield projected


@dataclass(frozen=True)
class TabularRow(TabularMetricsRow):
    """Format-neutral row after source-specific column mapping.

    :class:`TabularMetricsRow` plus the source's own identifier for the row --
    blank when the source records none, never fabricated.
    """

    rsid: str = ""


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
