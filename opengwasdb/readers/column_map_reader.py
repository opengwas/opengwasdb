"""The generic tabular reader: any tabular collection, read through its Column Map.

A Column Map (`opengwasdb.readers.column_map`) declares which provider column
holds each field and which conventions the values follow. This module reads a
provider file through one, by presenting the file to the GWAS-SSF reader as the
GWAS-SSF columns the map says it holds -- in the stream, not on disk (issue
#281, ADR 0063). Every rule that reader carries (effect resolution, recovery of
a standard error, z-score derivation, the projections' parity) therefore holds
for every mapped collection, and there is one tabular reader rather than one
per format.

The translation has two parts:

- **The header**, always: each declared column is renamed to its GWAS-SSF
  name, and every other column is renamed out of the reader's reach, so an
  undeclared provider column that happens to be called `beta` or `n` is never
  read. A declared column a file lacks is a loud failure, as is a field the
  file names twice.
- **The rows**, only when a convention needs it: deriving the other allele from
  `ref`/`alt`/`effect_allele`, splitting a `lower,upper` interval, reading an
  exactly-integral position written in scientific notation, and refusing a
  statistic that is neither a number nor a declared missing value. A map with
  none of these -- both built-ins -- passes every data byte through untouched.
"""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import IO, Any

from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers.column_map import (
    AlleleConvention,
    ColumnMap,
    ColumnMapError,
    PositionNotation,
)
from opengwasdb.readers.effect_source import EffectSource
from opengwasdb.readers.gwas_ssf import _VARIANT_COLUMNS, GwasSsfReader
from opengwasdb.readers.interface import (
    ImputationScoreDeclaration,
    ReaderAssociation,
    SiteMetrics,
    SourceVariant,
)
from opengwasdb.readers.tabular import (
    DEFAULT_CHUNK_ROWS,
    MetricsChunk,
    TabularMetricsRow,
    open_binary,
    stream_projected_variants,
)

__all__ = ["ColumnMapReader", "MappedSource"]

#: Each non-allele field's GWAS-SSF column, the name the reader resolves it by.
_CANONICAL = {
    "chromosome": "chromosome",
    "position": "base_pair_location",
    "beta": "beta",
    "odds_ratio": "odds_ratio",
    "z_score": "z_score",
    "standard_error": "standard_error",
    "frequency": "effect_allele_frequency",
    "p_value": "p_value",
    "ci_lower": "ci_lower",
    "ci_upper": "ci_upper",
    "sample_size": "n",
    "rsid": "rsid",
    "variant_id": "variant_id",
}
#: How each allele convention's fields are presented. A declared allele field
#: absent here (MVP's `alt`) is read by the row translation, not by name.
_ALLELE_NAMES: dict[AlleleConvention, dict[str, str]] = {
    AlleleConvention.EFFECT_AND_OTHER: {
        "effect_allele": "effect_allele", "other_allele": "other_allele",
    },
    AlleleConvention.REF_ALT_EFFECT_IS_ALT: {"alt": "effect_allele", "ref": "other_allele"},
    AlleleConvention.REF_ALT_AND_EFFECT_ALLELE: {
        "effect_allele": "effect_allele", "ref": "other_allele",
    },
}
_STATISTIC_FIELDS = (
    "beta", "odds_ratio", "z_score", "standard_error", "frequency", "p_value",
    "ci_lower", "ci_upper", "sample_size",
)
_SCIENTIFIC_POSITION = re.compile(rb"[0-9]+(\.[0-9]+)?[eE]\+?[0-9]+")
#: Lines translated per chunk handed on: large enough that the generator's own
#: cost vanishes beside the translation.
_LINES_PER_CHUNK = 4096


@dataclass(frozen=True)
class _RowPlan:
    """What the row translation does to each data line of one file."""

    source: str
    width: int
    names: tuple[str, ...]
    missing: frozenset[bytes] | None
    statistics: tuple[int, ...]
    alleles: tuple[int, int, int] | None
    position: int | None
    interval: int | None
    #: Declared by the map whenever `interval` is set; never a default.
    interval_separator: bytes | None

    @property
    def needed(self) -> bool:
        return bool(
            self.statistics or self.alleles or self.position is not None
            or self.interval is not None
        )


@dataclass(frozen=True)
class MappedSource:
    """One provider file, presented in GWAS-SSF's columns by its Column Map.

    `passthrough` names provider columns read under their own name: the column
    a declared imputation score names, which the manifest declares per Analysis
    rather than the map per collection.
    """

    path: str | Path
    column_map: ColumnMap
    passthrough: tuple[str, ...] = ()

    def __str__(self) -> str:
        return str(self.path)

    def open_binary(self) -> IO[bytes]:
        raw = open_binary(self.path)
        try:
            header, plan = self._translate_header(raw.readline())
        except BaseException:
            raw.close()
            raise
        body = _translated_rows(raw, plan) if plan.needed else _chunks(raw)
        return io.BufferedReader(_ChunkStream(_prepend(header, body), raw.close), 1 << 20)

    def _translate_header(self, header_line: bytes) -> tuple[bytes, _RowPlan]:
        cells = [cell.strip() for cell in _cells(header_line.rstrip(b"\r\n"))]
        names = [cell.decode("utf-8") for cell in cells]
        matched = _match_fields(names, self.column_map, str(self))
        _check_unmapped(names, matched, self.column_map, self.passthrough, str(self))
        presented = self._presented_names(names, matched)
        plan = self._row_plan(names, matched)
        return ("\t".join([*presented, *plan.names]) + "\n").encode("utf-8"), plan

    def _presented_names(self, names: list[str], matched: dict[str, int]) -> list[str]:
        presented = [f"__unmapped_{index}__" for index in range(len(names))]
        for index, name in enumerate(names):
            if name in self.passthrough:
                presented[index] = name
        renames = {**_CANONICAL, **_ALLELE_NAMES[self.column_map.conventions.alleles]}
        for field, index in matched.items():
            if field in renames:
                presented[index] = renames[field]
        _refuse_passthrough_collision(presented, self.passthrough, str(self))
        return presented

    def _row_plan(self, names: list[str], matched: dict[str, int]) -> _RowPlan:
        conventions = self.column_map.conventions
        missing = conventions.missing_values
        scientific = conventions.position_notation is PositionNotation.INTEGER_OR_EXACT_SCIENTIFIC
        return _RowPlan(
            source=str(self),
            width=len(names),
            names=("ci_lower", "ci_upper") if "confidence_interval" in matched else (),
            missing=None if missing is None else frozenset(t.encode() for t in missing),
            statistics=_checked_statistics(matched, missing),
            alleles=_derived_allele_indexes(matched, conventions.alleles),
            position=matched["position"] if scientific else None,
            interval=matched.get("confidence_interval"),
            interval_separator=_encoded(conventions.confidence_interval_separator),
        )


def _encoded(separator: str | None) -> bytes | None:
    return None if separator is None else separator.encode()


def _checked_statistics(
    matched: dict[str, int], missing: tuple[str, ...] | None
) -> tuple[int, ...]:
    """The statistic cells a closed missing-value vocabulary checks; none under the legacy rule."""
    if missing is None:
        return ()
    return tuple(matched[field] for field in _STATISTIC_FIELDS if field in matched)


def _derived_allele_indexes(
    matched: dict[str, int], alleles: AlleleConvention
) -> tuple[int, int, int] | None:
    """`(effect_allele, ref, alt)` when the other allele is derived from them."""
    if alleles is not AlleleConvention.REF_ALT_AND_EFFECT_ALLELE:
        return None
    return matched["effect_allele"], matched["ref"], matched["alt"]


def _match_fields(names: list[str], column_map: ColumnMap, source: str) -> dict[str, int]:
    """Each declared field's header index; a field named twice is refused.

    Names are compared without surrounding whitespace, so a padded duplicate
    (`beta ` beside `beta`, the GCST006329 shape) is two columns named one
    thing, refused rather than half-read (ADR 0049).
    """
    matched: dict[str, int] = {}
    for field, spellings in column_map.field_spellings().items():
        hits = [index for index, name in enumerate(names) if name in spellings]
        if len(hits) > 1:
            found = ", ".join(repr(names[index]) for index in hits)
            raise ColumnMapError(
                f"{source}: field {field!r} is named {len(hits)} times ({found}); "
                f"Column Map {column_map.name!r} cannot tell which to read"
            )
        if hits:
            matched[field] = hits[0]
        elif field in column_map.columns:
            raise ColumnMapError(
                f"{source}: Column Map {column_map.name!r} declares {field!r} as "
                f"{' or '.join(map(repr, spellings))}, which the file does not carry"
            )
    return matched


def _check_unmapped(
    names: list[str],
    matched: dict[str, int],
    column_map: ColumnMap,
    passthrough: tuple[str, ...],
    source: str,
) -> None:
    if column_map.conventions.unmapped_columns != "refuse":
        return
    known = {*matched.values()}
    unmapped = [
        name
        for index, name in enumerate(names)
        if index not in known and name not in column_map.ignored_columns
        and name not in passthrough
    ]
    if unmapped:
        raise ColumnMapError(
            f"{source}: column(s) {', '.join(map(repr, unmapped))} are neither mapped nor "
            f"ignored by Column Map {column_map.name!r}"
        )


def _refuse_passthrough_collision(
    presented: list[str], passthrough: tuple[str, ...], source: str
) -> None:
    collided = sorted(name for name in passthrough if presented.count(name) > 1)
    if collided:
        raise ColumnMapError(
            f"{source}: declared column(s) {', '.join(collided)} collide with a mapped field"
        )


def _cells(line: bytes) -> list[bytes]:
    """One line's tab-separated cells, unquoting through csv when it carries a quote."""
    if b'"' not in line:
        return line.split(b"\t")
    fields = next(csv.reader([line.decode("utf-8")], delimiter="\t"))
    return [cell.encode("utf-8") for cell in fields]


def _join(cells: list[bytes], quoted: bool) -> bytes:
    """One translated line, re-quoted when its source was, so a quoted tab stays one cell."""
    if not quoted:
        return b"\t".join(cells) + b"\n"
    buffer = io.StringIO()
    csv.writer(buffer, delimiter="\t", lineterminator="\n").writerow(
        cell.decode("utf-8") for cell in cells
    )
    return buffer.getvalue().encode("utf-8")


def _translated_rows(lines: Iterable[bytes], plan: _RowPlan) -> Iterator[bytes]:
    batch: list[bytes] = []
    for number, line in enumerate(lines, start=2):
        batch.append(_translate_row(line, plan, number))
        if len(batch) >= _LINES_PER_CHUNK:
            yield b"".join(batch)
            batch = []
    if batch:
        yield b"".join(batch)


def _translate_row(line: bytes, plan: _RowPlan, number: int) -> bytes:
    body = line.rstrip(b"\r\n")
    if not body:
        return line
    cells = _cells(body)
    if len(cells) > plan.width:
        raise ColumnMapError(
            f"{plan.source}: line {number} has {len(cells)} fields; the header has {plan.width}"
        )
    cells += [b""] * (plan.width - len(cells))
    where = f"{plan.source}: line {number}"
    if plan.alleles is not None:
        cells[plan.alleles[1]] = _other_allele(cells, plan.alleles, where)
    if plan.position is not None:
        cells[plan.position] = _position(cells[plan.position])
    for index in plan.statistics:
        cells[index] = _statistic(cells[index], plan.missing, where)
    extra = [] if plan.interval is None else _interval(cells[plan.interval], plan, where)
    return _join([*cells, *extra], quoted=b'"' in body)


def _other_allele(cells: list[bytes], indexes: tuple[int, int, int], where: str) -> bytes:
    """The allele of `ref`/`alt` that the effect allele is not, compared verbatim.

    An effect allele naming neither fails the whole file, as dbGaP MVP's GWAS-SSF
    rewrite does: an allele pair the file cannot orient is not a row to skip.
    """
    effect, ref, alt = (cells[index] for index in indexes)
    if effect == alt:
        return ref
    if effect == ref:
        return alt
    raise ColumnMapError(
        f"{where}: effect allele {effect.decode()!r} is neither ref {ref.decode()!r} "
        f"nor alt {alt.decode()!r}"
    )


def _position(cell: bytes) -> bytes:
    """An exactly integral `2.4e+07` as `24000000`; every other cell unchanged."""
    if _SCIENTIFIC_POSITION.fullmatch(cell):
        value = Decimal(cell.decode("ascii"))
        if value == value.to_integral_value():
            return str(int(value)).encode("ascii")
    return cell


def _is_missing(cell: bytes, missing: frozenset[bytes] | None) -> bool:
    """Whether a stripped statistic cell is missing under the map's vocabulary."""
    if missing is not None:
        return cell in missing
    try:
        float(cell)
    except ValueError:
        return True
    return False


def _statistic(cell: bytes, missing: frozenset[bytes] | None, where: str) -> bytes:
    """A declared missing token as an empty cell; a number unchanged; else refused."""
    stripped = cell.strip()
    if _is_missing(stripped, missing):
        return b""
    try:
        float(stripped)
    except ValueError:
        raise ColumnMapError(
            f"{where}: {cell.decode('utf-8', 'replace')!r} is neither a number nor a "
            "declared missing value"
        ) from None
    return cell


def _interval(cell: bytes, plan: _RowPlan, where: str) -> list[bytes]:
    """A `lower<sep>upper` cell as its two bounds; a missing interval as two empty cells."""
    expected = plan.interval_separator
    if expected is None:
        raise ColumnMapError(f"{where}: the Column Map declares no confidence_interval_separator")
    stripped = cell.strip()
    if _is_missing(stripped, plan.missing):
        return [b"", b""]
    lower, separator, upper = stripped.partition(expected)
    if not separator:
        raise ColumnMapError(
            f"{where}: confidence interval {cell.decode('utf-8', 'replace')!r} is not a "
            f"lower{expected.decode()}upper pair"
        )
    return [_statistic(lower, plan.missing, where), _statistic(upper, plan.missing, where)]


def _chunks(raw: IO[bytes]) -> Iterator[bytes]:
    while chunk := raw.read(1 << 20):
        yield chunk


def _prepend(first: bytes, rest: Iterator[bytes]) -> Iterator[bytes]:
    yield first
    yield from rest


class _ChunkStream(io.RawIOBase):
    """A readable byte stream over an iterator of byte chunks."""

    def __init__(self, chunks: Iterator[bytes], close_source: Callable[[], None]) -> None:
        super().__init__()
        self._chunks = chunks
        self._pending = memoryview(b"")
        self._close_source = close_source

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        while not self._pending:
            chunk = next(self._chunks, None)
            if chunk is None:
                return 0
            self._pending = memoryview(chunk)
        target = memoryview(buffer).cast("B")
        size = min(len(target), len(self._pending))
        target[:size] = self._pending[:size]
        self._pending = self._pending[size:]
        return size

    def close(self) -> None:
        if not self.closed:
            self._close_source()
        super().close()


@dataclass(frozen=True)
class ColumnMapReader:
    """SourceReader for one file of a collection described by a Column Map.

    The reading rules are `GwasSsfReader`'s, applied to the map's translation of
    the file, so this offers every stream that reader does -- associations,
    variants, the one-pass resolver's metrics and blocked metrics, and site
    extraction -- with the same answers. The rsid separator is the one
    convention the variant projection applies itself, because FinnGen's `rsids`
    lists every dbSNP name at a position and the first is the row's.
    """

    path: str | Path
    column_map: ColumnMap
    stored_effect_scale: StoredEffectScale = StoredEffectScale.SD
    chunk_rows: int = DEFAULT_CHUNK_ROWS
    imputation_score_declaration: ImputationScoreDeclaration | None = None

    def _source(self) -> MappedSource:
        declaration = self.imputation_score_declaration
        passthrough = () if declaration is None else (declaration.column_name,)
        return MappedSource(self.path, self.column_map, passthrough)

    def _reader(self) -> GwasSsfReader:
        return GwasSsfReader(
            self._source(),
            self.stored_effect_scale,
            self.chunk_rows,
            self.imputation_score_declaration,
        )

    @property
    def effect_source(self) -> EffectSource | None:
        """The effect column this file resolves to, named as the map presents it."""
        return self._reader().effect_source

    @property
    def imputation_score_column(self) -> str | None:
        return self._reader().imputation_score_column

    def stream_associations(self) -> Iterator[ReaderAssociation]:
        yield from self._reader().stream_associations()

    def stream_variants(self) -> Iterator[SourceVariant]:
        separator = self.column_map.conventions.rsid_separator
        columns = replace(_VARIANT_COLUMNS, alias_separator=separator)
        yield from stream_projected_variants(self._source(), columns)

    def stream_metrics(self) -> Iterator[TabularMetricsRow]:
        yield from self._reader().stream_metrics()

    def stream_metric_chunks(self) -> Iterator[MetricsChunk]:
        yield from self._reader().stream_metric_chunks()

    def extract_at_sites(self, alids: Iterable[str]) -> dict[str, SiteMetrics]:
        return self._reader().extract_at_sites(alids)

