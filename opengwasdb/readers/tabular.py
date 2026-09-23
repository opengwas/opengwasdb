"""Shared mechanics for canonical tabular summary-statistics readers."""

from __future__ import annotations

import csv
import gzip
import math
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers.effect_source import (
    EffectSourceKind,
    UnsignedZScoreError,
    derive_z_score_effect,
    resolve_effect_source,
    resolve_sample_size_column,
)
from opengwasdb.readers.gwas_vcf import is_palindromic
from opengwasdb.readers.interface import ReaderAssociation, SiteMetrics, SourceVariant
from opengwasdb.stats import parse_af
from opengwasdb.variants.normalise import (
    VALID_BASES,
    VariantNormalisationError,
    normalise_allele,
    normalise_chromosome,
)

if TYPE_CHECKING:
    # Only ever a signature annotation here; the value itself comes from the
    # caller's resolved effect source (issue #215).
    from opengwasdb.readers.effect_source import EffectSource

_MISSING = {"", ".", "NA", "NaN", "nan", "None"}
_VALID_DISTINCT_ALLELE_CODES = frozenset(
    (ref, alt) for ref in b"ACGT" for alt in b"ACGT" if ref != alt
)
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
    "DEFAULT_CHUNK_ROWS",
    "MetricsChunk",
    "metrics_chunks_from_rows",
    "project_source_variant",
    "stream_projected_metric_chunks",
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

    Identity columns are required. `frequency` and `standard_error` are
    optional on purpose: a harmonised file that reports `odds_ratio` instead of
    `beta` still names every variant, and ancestry assignment reads only
    frequencies -- refusing the whole file over a column one of the two stages
    never reads would throw away the other stage's evidence with it.

    The effect column is deliberately *not* declared here. It is resolved from
    the header by :func:`opengwasdb.readers.effect_source.resolve_effect_source`
    (issue #213), because `beta` and `odds_ratio` are two permitted spellings
    of the same thing and the file -- not the provider declaration -- says which
    one it used.
    """

    chromosome: tuple[bytes, ...]
    position: bytes
    ref: bytes
    alt: bytes
    frequency: bytes
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


def _normalise_projected_chromosome(value: bytes) -> str:
    stripped = value.strip()
    if stripped in _FAST_AUTOSOMES:
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
) -> SourceVariant | None:
    """Validate projected identity fields without constructing an orientation.

    Numeric chromosomes and single-base alleles are the common production path.
    Less common labels and long alleles fall back to the same normalizers used
    by full association parsing, while parity tests protect the observable
    reader contract (issue #179).
    """
    try:
        chromosome = _normalise_projected_chromosome(chromosome_bytes)
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
    )
    return () if variant is None else (variant,)


def stream_projected_variants(
    path: str | Path,
    columns: VariantProjectionColumns,
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
            yield from _projected_variant(row, projection)


# The common unprefixed spellings can skip decoding while returning exactly
# the same explicit canonical labels as `normalise_chromosome` (ADR 0052).
_FAST_AUTOSOMES = frozenset(str(number).encode() for number in range(1, 23))
_FAST_CHROMOSOME_SPELLINGS = (
    *(str(number) for number in range(1, 27)),
    "X",
    "x",
    "Y",
    "y",
    "M",
    "m",
    "MT",
    "mt",
)
_FAST_CHROMOSOMES: dict[bytes, str] = {
    spelling.encode(): normalise_chromosome(spelling)
    for spelling in _FAST_CHROMOSOME_SPELLINGS
}


@dataclass(frozen=True)
class _ResolvedMetricsProjection:
    chromosome: int
    position: int
    ref: int
    alt: int
    frequency: int | None
    effect: int | None
    effect_source: EffectSource | None
    standard_error: int | None
    sample_size: int | None
    last_identity: int
    split_limit: int

    @property
    def effect_kind(self) -> EffectSourceKind | None:
        """The resolved effect kind, for call sites that need only that."""
        return None if self.effect_source is None else self.effect_source.kind


def _metrics_projection_indexes(
    header: list[bytes], columns: MetricsProjectionColumns
) -> _ResolvedMetricsProjection | None:
    found = _header_identity(header, columns)
    if found is None:
        return None
    indexes, (chromosome, position, ref, alt) = found
    effect_source = resolve_effect_source(header)
    effect = (
        None
        if effect_source is None
        else indexes.get(effect_source.column_name.encode("utf-8"))
    )
    frequency = indexes.get(columns.frequency)
    standard_error = indexes.get(columns.standard_error)
    # The per-row sample size is only read when a z-score derivation needs it;
    # a beta/odds_ratio file never pays for the lookup or its ambiguity rule.
    sample_size: int | None = None
    if effect_source is not None and effect_source.kind is EffectSourceKind.Z_SCORE:
        sample_size_name = resolve_sample_size_column(header)
        if sample_size_name is not None:
            sample_size = indexes.get(sample_size_name.encode("utf-8"))
    last_identity = max(chromosome, position, ref, alt)
    optional = (frequency, effect, standard_error, sample_size)
    last_selected = max([last_identity, *(index for index in optional if index is not None)])
    return _ResolvedMetricsProjection(
        chromosome=chromosome,
        position=position,
        ref=ref,
        alt=alt,
        frequency=frequency,
        effect=effect,
        effect_source=effect_source,
        standard_error=standard_error,
        sample_size=sample_size,
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
    beta, se = _projected_effect(row, projection)
    return TabularMetricsRow(
        chromosome=chromosome,
        position=position,
        ref=ref,
        alt=alt,
        alid=alid,
        flipped=flipped,
        af_alt=_metrics_float(_metrics_cell(row, projection.frequency), parse_af),
        beta=beta,
        se=se,
    )


def _projected_effect(
    row: list[bytes], projection: _ResolvedMetricsProjection
) -> tuple[float | None, float | None]:
    """The row's ``(beta, se)``, deriving both from a signed z where that is the source.

    A z-score derivation consumes the row's own EAF and sample size and yields
    `(None, None)` for any row it cannot honestly derive -- an EAF outside
    `(0, 1)`, a non-positive or absent N -- exactly as an unusable beta drops
    the row (issue #215).
    """
    source = projection.effect_source
    if source is not None and source.kind is EffectSourceKind.Z_SCORE:
        derived = derive_z_score_effect(
            _metrics_float(_metrics_cell(row, projection.effect), parse_finite_float),
            _metrics_float(_metrics_cell(row, projection.frequency), parse_af),
            _metrics_float(_metrics_cell(row, projection.sample_size), parse_positive_float),
        )
        return derived if derived is not None else (None, None)
    return (
        _effect_beta(_metrics_cell(row, projection.effect), projection.effect_kind),
        _metrics_float(_metrics_cell(row, projection.standard_error), parse_positive_float),
    )


def _effect_beta(cell: bytes, kind: EffectSourceKind | None) -> float | None:
    """The beta the row carries, on the scale the association stream expects.

    An `odds_ratio` is read as its log, and only a positive finite value has
    one; anything else is absent, exactly as an unusable `beta` is. The
    standard error is not touched by the transform -- GWAS-SSF reports it on
    the log scale already (issue #213).
    """
    if kind is EffectSourceKind.ODDS_RATIO:
        raw = _metrics_float(cell, parse_positive_float)
        return math.log(raw) if raw is not None else None
    return _metrics_float(cell, parse_finite_float)


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


def require_signed_z_score(path: str | Path, column_name: str) -> None:
    """Refuse a z-score column that carries no negative value.

    A signed z has both directions; a `|z|` or chi-square statistic has the same
    magnitude with no sign, and reading it as signed would give every derived
    effect the wrong direction (issue #215). The column is read until a negative
    value proves it signed -- before any row is yielded, because a prefix cannot
    prove a column is signed and a partially consumed stream must not look
    plausible. A column with no negative value is read to the end and refused.

    A column with no usable value at all is not *unsigned* -- it is unusable,
    and every row drops for that reason. Only a column that has values and none
    of them negative is refused.

    This is the file-reading half of the guard; the scale policy that decides
    *whether* a z-score source is admissible at all lives with the reader
    (`GwasSsfReader`), which is where `stored_effect_scale` is known.
    """
    frames = pd.read_csv(
        path,
        sep="\t",
        usecols=[column_name],
        # The same missing-value vocabulary and float rule the projections use,
        # so a value this guard sees is a value they would see.
        keep_default_na=False,
        na_values={column_name: _MISSING_TOKENS},
        chunksize=DEFAULT_CHUNK_ROWS,
        engine="c",
    )
    seen = False
    with frames:
        for frame in frames:
            column = frame[column_name]
            values = (
                column.to_numpy(dtype="float64", copy=False)
                if column.dtype.kind == "f"
                else _floats_from_text(column)
            )
            finite = np.isfinite(values)
            if not finite.any():
                continue
            seen = True
            if bool((values[finite] < 0.0).any()):
                return
    if seen:
        raise UnsignedZScoreError(
            f"{path}: z-score column {column_name!r} carries no negative value; "
            "an unsigned statistic is not a signed z"
        )


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


#: The largest position an ``int64`` column can carry. A source naming a larger
#: one is refused rather than wrapped: the row-wise projection has no such
#: bound, so this is the one coordinate the two paths disagree about, and it is
#: eleven orders of magnitude beyond the longest human chromosome.
_MAX_POSITION = 2**63 - 1

#: Single-base codes for the strand-ambiguity test, keyed by the *verbatim*
#: upper-cased cell, because that is what `is_palindromic` is given row-wise.
_PALINDROME_CODES = {"A": 1, "T": 2, "C": 3, "G": 4}
_PALINDROME_PAIRS = ((1, 2), (2, 1), (3, 4), (4, 3))

_STATISTIC_FIELDS = ("frequency", "effect", "standard_error", "sample_size")
_PROJECTED_FIELDS = ("chromosome", "position", "ref", "alt", *_STATISTIC_FIELDS)
_MISSING_TOKENS = sorted(_MISSING)

#: Rows per block. Peak memory is a block's ALID strings rather than the whole
#: source, so this is the knob that bounds a resolver worker's footprint: 50,000
#: costs about 23 MB against a genome-wide source where 1,000,000 costs 359 MB,
#: and measures fractionally *faster* for it (issue #209).
DEFAULT_CHUNK_ROWS = 50_000


@dataclass(frozen=True)
class MetricsChunk:
    """A block of projected source rows held as columns (issue #209).

    The column-oriented counterpart of :class:`TabularMetricsRow`: the same
    projection over many rows at once, so the resolver's per-row Python cost --
    a frozen dataclass, an f-string ALID and six parse calls for every row of a
    genome-wide file -- becomes a handful of array operations per block. Only
    rows naming a usable canonical variant are present, which is exactly the set
    `stream_projected_metrics` yields, in the same order.

    `palindromic` travels instead of the source's own `ref`/`alt` labels because
    deciding strand ambiguity is the only thing those labels are read for. It is
    computed from the *verbatim* labels, as `is_palindromic` is given them
    row-wise -- not from the normalised alleles, which differ for a
    whitespace-padded cell. Matching the row-wise answer matters more than
    matching the tidier one.

    Absent statistics are `NaN`, never `0.0`. Each of `af_alt`, `beta` and `se`
    carries the usability rule its row-wise `parse_*` counterpart applies, so a
    value present here is one the row-wise path would also have reported.
    """

    alid: np.ndarray
    flipped: np.ndarray
    palindromic: np.ndarray
    af_alt: np.ndarray
    beta: np.ndarray
    se: np.ndarray

    def __len__(self) -> int:
        return int(self.alid.shape[0])


def _projected_column_names(
    header: list[bytes], projection: _ResolvedMetricsProjection
) -> dict[str, str]:
    """Each projected field's own header name; fields the file lacks are absent.

    `read_csv` selects columns by name, so a header that repeats a projected
    name would let pandas mangle one of them and hand back one column's values
    under another column's name -- a wrong answer with nothing to see. It raises
    here instead.
    """
    names: dict[str, str] = {}
    for field_name in _PROJECTED_FIELDS:
        index: int | None = getattr(projection, field_name)
        if index is None:
            continue
        cell = header[index]
        if header.count(cell) != 1:
            raise ValueError(
                f"projected column {cell.decode('utf-8', 'replace')!r} appears "
                f"{header.count(cell)} times in the header; it must be unique"
            )
        names[field_name] = cell.decode("utf-8")
    return names


def _category_arrays(
    column: pd.Series, normalise: Callable[[str], str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-row normalised label, single-base code, and validity.

    Normalisation runs once per *category* rather than once per row, which is the
    whole reason the identity columns are read as categoricals: a genome-wide
    source has millions of rows and a few tens of thousands of distinct allele
    strings. The rule applied is the shared `normalise_*` function itself, not a
    re-spelling of it, so the two projections cannot drift apart.
    """
    categories = column.cat.categories
    normalised = np.empty(len(categories), dtype=object)
    base = np.zeros(len(categories), dtype=np.int8)
    valid = np.zeros(len(categories), dtype=bool)
    for index, value in enumerate(categories):
        text = str(value)
        base[index] = _PALINDROME_CODES.get(text.upper(), 0)
        try:
            normalised[index] = normalise(text)
        except VariantNormalisationError:
            normalised[index] = ""
            continue
        valid[index] = True
    codes = column.cat.codes.to_numpy()
    if codes.min(initial=0) < 0:
        # `keep_default_na=False` is what makes this unreachable: a cell the
        # source omits arrives as the empty category, which `normalise_*`
        # rejects on its own. Indexing with -1 would instead silently pick the
        # last category, which is a wrong allele wearing a right one's name.
        raise ValueError(
            f"{column.name!r} has a missing category despite keep_default_na=False"
        )
    return normalised[codes], base[codes], valid[codes]


def _positions_from_text(column: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """`int()` per cell, for a column that is not wholly integer literals."""
    position = np.zeros(len(column), dtype="int64")
    valid = np.zeros(len(column), dtype=bool)
    for index, cell in enumerate(column.to_numpy(dtype=object)):
        try:
            value = int(cell)
        except (TypeError, ValueError):
            continue
        if 0 < value <= _MAX_POSITION:
            position[index] = value
            valid[index] = True
    return position, valid


def _position_arrays(column: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Per-row position and validity, under `int()`'s own grammar.

    `to_numeric` reports an integer dtype exactly when every cell is a plain
    integer literal, which is what a `base_pair_location` column ordinarily is
    and is all the checking that case needs. Anything else -- one `1e5`, one
    `100.5`, one empty cell -- falls back to `int()` per cell, because a float
    parser accepts positions the row-wise projection drops and would put a
    variant at a coordinate no row of the source names.
    """
    converted = pd.to_numeric(column, errors="coerce")
    if converted.dtype.kind not in "iu":
        return _positions_from_text(column)
    position = converted.to_numpy(dtype="int64")
    return position, (position > 0) & (position <= _MAX_POSITION)


def _floats_from_text(column: pd.Series) -> np.ndarray:
    """`float()` per cell -- the row-wise rule -- for a column pandas left as text.

    Reached only when a statistic column carries a token that is neither a
    number nor one of `_MISSING`, which is a malformed source rather than an
    ordinary one. `to_numeric` would be the vectorised answer and is not used:
    it is a digit less accurate than `float()` on a long decimal, and a
    statistic that differs from the row-wise projection in its last place is
    exactly the difference nothing downstream would report.
    """
    values = np.full(len(column), np.nan)
    for index, cell in enumerate(column.to_numpy(dtype=object)):
        try:
            values[index] = float(cell)
        except (TypeError, ValueError):
            continue
    return values


def _statistic_array(
    frame: pd.DataFrame, name: str | None, usable: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """One statistic column as float64, every unusable value `NaN`.

    A column the file does not carry is all-`NaN` rather than missing, which is
    what the row-wise projection reports for it too: `_metrics_cell` hands an
    absent column the same empty cell it hands a short row.
    """
    if name is None:
        return np.full(len(frame), np.nan)
    column = frame[name]
    values = (
        column.to_numpy(dtype="float64", copy=False)
        if column.dtype.kind == "f"
        else _floats_from_text(column)
    )
    return np.where(usable(values), values, np.nan)


def _palindromic(effect: np.ndarray, other: np.ndarray) -> np.ndarray:
    """`is_palindromic`, over two arrays of single-base codes."""
    ambiguous = np.zeros(effect.shape, dtype=bool)
    for first, second in _PALINDROME_PAIRS:
        ambiguous |= (effect == first) & (other == second)
    return ambiguous


def _effect_usable(effect_kind: EffectSourceKind | None) -> Callable[[np.ndarray], np.ndarray]:
    """The row-wise usability rule for the resolved effect kind, as an array rule.

    `parse_positive_float` is what an `odds_ratio` is read through row-wise, so
    the blocked path applies the same `> 0` bound before the log; a `beta` is
    only required to be finite.
    """
    if effect_kind is EffectSourceKind.ODDS_RATIO:
        return lambda values: np.isfinite(values) & (values > 0.0)
    return np.isfinite


def _z_score_arrays(
    frame: pd.DataFrame, names: dict[str, str]
) -> tuple[np.ndarray, np.ndarray]:
    """``(beta, se)`` derived from a signed z, EAF and per-row N (issue #215).

    The usability rules mirror `derive_z_score_effect`: a finite z, an EAF in
    `(0, 1)` and a positive N. Every unusable input is already `NaN`, so the
    expression is `NaN` for that row and `beta`/`se` stay absent -- never a
    substituted frequency or a study-level N.
    """
    z = _statistic_array(frame, names.get("effect"), np.isfinite)
    af = _statistic_array(
        frame, names.get("frequency"), lambda v: np.isfinite(v) & (v > 0.0) & (v < 1.0)
    )
    n = _statistic_array(frame, names.get("sample_size"), lambda v: np.isfinite(v) & (v > 0.0))
    # A `NaN` anywhere propagates through the whole expression; `errstate` only
    # keeps numpy from warning about it, it does not change the answer.
    with np.errstate(invalid="ignore", divide="ignore"):
        se = 1.0 / np.sqrt(2.0 * af * (1.0 - af) * (n + z * z))
        beta = z * se
    usable = np.isfinite(se)
    return np.where(usable, beta, np.nan), np.where(usable, se, np.nan)


def _projected_effect_arrays(
    frame: pd.DataFrame, names: dict[str, str], effect_source: EffectSource | None
) -> tuple[np.ndarray, np.ndarray]:
    """One block's ``(beta, se)`` arrays, from the resolved effect source."""
    kind = None if effect_source is None else effect_source.kind
    if kind is EffectSourceKind.Z_SCORE:
        return _z_score_arrays(frame, names)
    beta = _statistic_array(frame, names.get("effect"), _effect_usable(kind))
    if kind is EffectSourceKind.ODDS_RATIO:
        # Every value that survived `_effect_usable` is positive, so the only
        # `NaN`s entering `log` are already-absent ones, which stay absent.
        beta = np.log(beta)
    se = _statistic_array(
        frame, names.get("standard_error"), lambda v: np.isfinite(v) & (v > 0.0)
    )
    return beta, se


def _projected_chunk(
    frame: pd.DataFrame, names: dict[str, str], effect_source: EffectSource | None
) -> MetricsChunk:
    """One block's projection, reduced to the rows naming a canonical variant."""
    chromosome, _, chromosome_ok = _category_arrays(
        frame[names["chromosome"]], normalise_chromosome
    )
    effect, effect_base, effect_ok = _category_arrays(frame[names["alt"]], normalise_allele)
    other, other_base, other_ok = _category_arrays(frame[names["ref"]], normalise_allele)
    position, position_ok = _position_arrays(frame[names["position"]])

    keep = chromosome_ok & position_ok & effect_ok & other_ok & (effect != other)
    lower = np.where(effect < other, effect, other)
    upper = np.where(effect < other, other, effect)
    text = np.where(position_ok, position, 0).astype(str).astype(object)
    beta, se = _projected_effect_arrays(frame, names, effect_source)
    return MetricsChunk(
        alid=(chromosome + ":" + text + ":" + lower + ":" + upper)[keep],
        flipped=(effect != lower)[keep],
        palindromic=_palindromic(effect_base, other_base)[keep],
        af_alt=_statistic_array(
            frame, names.get("frequency"), lambda v: np.isfinite(v) & (v >= 0.0) & (v <= 1.0)
        )[keep],
        beta=beta[keep],
        se=se[keep],
    )


def _metric_frame_reader(
    path: str | Path, names: dict[str, str], chunk_rows: int
) -> pd.io.parsers.TextFileReader:
    """The chunked pandas reader for one metrics projection.

    The position keeps its source text: pandas would happily read `1e5` and
    `100.5` as numbers, and `int()` -- the rule the row-wise projection applies
    -- rejects both, so the literal is the only thing that can be checked
    against it. Missing values are `_MISSING` and nothing else, per column:
    pandas' own default token list would turn a chromosome spelled `NA` -- which
    the row-wise projection keeps -- into a dropped row, while a statistic has
    exactly the missing spellings `parse_finite_float` accepts.
    """
    dtypes: dict[str, str] = {names["position"]: "str"}
    dtypes.update(
        dict.fromkeys((names[field] for field in ("chromosome", "ref", "alt")), "category")
    )
    return pd.read_csv(
        path,
        sep="\t",
        usecols=list(names.values()),
        dtype=dtypes,
        keep_default_na=False,
        na_values={
            names[field]: _MISSING_TOKENS for field in _STATISTIC_FIELDS if field in names
        },
        # `float()` is correctly rounded and pandas' default converter is not;
        # on a real source that is a one-in-a-hundred-thousand row whose
        # frequency differs from the row-wise projection in its last place.
        float_precision="round_trip",
        chunksize=chunk_rows,
        engine="c",
    )


def stream_projected_metric_chunks(
    path: str | Path, columns: MetricsProjectionColumns, *, chunk_rows: int = DEFAULT_CHUNK_ROWS
) -> Iterator[MetricsChunk]:
    """Stream the projection `stream_projected_metrics` produces, by block.

    Row-for-row and field-for-field identical to the row-wise projection --
    `tests/test_projected_metric_chunks.py` holds that bar -- and about twice as
    quick on a genome-wide source, because allele and chromosome normalisation
    run once per distinct string rather than once per row, and the statistics
    are parsed a column at a time (issue #209).

    Two differences are deliberate. A byte that is not valid UTF-8 fails the
    whole block, where the row-wise path drops only that row when the byte lands
    in an identity cell: both refuse to invent a value, and `resolve_analysis`
    turns this one into a per-Analysis error rather than a quietly shorter file.
    A header that repeats a projected column name raises, for the reason
    `_projected_column_names` gives.

    `chunk_rows` bounds memory, not semantics: the projection of a source does
    not depend on how it is blocked, which is what the parity tests assert.
    """
    opener = gzip.open if str(path).endswith((".gz", ".bgz")) else open
    with opener(path, "rb") as fh:
        header_line = fh.readline()
    projection = _required_metrics_projection(path, header_line, columns)
    names = _projected_column_names(_header_cells(header_line), projection)
    frames = _metric_frame_reader(path, names, chunk_rows)
    with frames:
        for frame in frames:
            yield _projected_chunk(frame, names, projection.effect_source)


def metrics_chunks_from_rows(
    rows: Iterable[TabularMetricsRow], *, chunk_rows: int = DEFAULT_CHUNK_ROWS
) -> Iterator[MetricsChunk]:
    """Block a row-wise projection, for a reader that has no blocked one.

    The resolver accumulates from :class:`MetricsChunk` and only that, so there
    is one accumulation to get right rather than a fast one and a slow one that
    can disagree. A reader that can only yield rows pays for the blocking and
    gets the same answer (issue #209).
    """
    batch: list[TabularMetricsRow] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= chunk_rows:
            yield _chunk_of_rows(batch)
            batch = []
    if batch:
        yield _chunk_of_rows(batch)


def _chunk_of_rows(rows: list[TabularMetricsRow]) -> MetricsChunk:
    return MetricsChunk(
        alid=np.array([row.alid for row in rows], dtype=object),
        flipped=np.array([row.flipped for row in rows], dtype=bool),
        palindromic=np.array(
            [is_palindromic(row.ref, row.alt) for row in rows], dtype=bool
        ),
        af_alt=np.array([_or_nan(row.af_alt) for row in rows], dtype="float64"),
        beta=np.array([_or_nan(row.beta) for row in rows], dtype="float64"),
        se=np.array([_or_nan(row.se) for row in rows], dtype="float64"),
    )


def _or_nan(value: float | None) -> float:
    return math.nan if value is None else value


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
