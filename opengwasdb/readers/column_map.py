"""The Column Map: one Source Collection's tabular format, declared as data (issue #281).

A Column Map tells the generic tabular reader which provider column holds each
field it reads and which conventions the collection's values follow (stores
ADR 0036, ADR 0063). It replaces a reader written per format, or a rewrite of
every provider file into GWAS-SSF before it can be read.

Every way a map can be wrong is refused here, before any source is opened:

- a key, field or convention this module does not know;
- a required field or a convention left undeclared -- nothing is defaulted;
- a convention the reader knows but cannot honour, which is refused with the
  reason rather than read as something nearby.

A map is named by a file path, or by the name of a built-in map (`gwas-ssf`,
`finngen-r13`), which reproduce the two dedicated tabular readers and ship
inside this package. Its `sha256` is the checksum of the file's bytes, so a
Resolution Record and a build can name exactly which map read a source.
"""

from __future__ import annotations

import hashlib
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from importlib import resources
from pathlib import Path
from typing import Any

from opengwasdb.build.liftover import normalise_build

__all__ = [
    "BUILTIN_COLUMN_MAPS",
    "COLUMN_MAP_CAPABILITY",
    "AlleleConvention",
    "ColumnMap",
    "ColumnMapError",
    "Conventions",
    "PositionNotation",
    "as_column_map",
    "check_mapped_rows",
    "load_column_map",
    "mapped_row_build",
]

#: The Source Reader Capability of a collection read through a Column Map.
COLUMN_MAP_CAPABILITY = "opengwasdb.column-map"

#: Built-in maps, by the name a caller uses for them.
BUILTIN_COLUMN_MAPS: tuple[str, ...] = ("finngen-r13", "gwas-ssf")

_FORMAT_VERSION = 1


class ColumnMapError(ValueError):
    """A Column Map that cannot be read as declared, or a source it cannot read."""


class AlleleConvention(StrEnum):
    """Which columns name the alleles, and which allele the effect refers to."""

    #: `effect_allele` and `other_allele` columns (GWAS-SSF).
    EFFECT_AND_OTHER = "effect_and_other"
    #: `ref` and `alt` columns; the effect is the `alt` allele's (FinnGen).
    REF_ALT_EFFECT_IS_ALT = "ref_alt_effect_is_alt"
    #: `ref`, `alt` and an `effect_allele` column naming one of them; the other
    #: allele is whichever of the two it does not name (dbGaP MVP).
    REF_ALT_AND_EFFECT_ALLELE = "ref_alt_and_effect_allele"


class PositionNotation(StrEnum):
    """How the position column writes a base-pair coordinate."""

    #: Decimal integers only; anything else is not a position.
    INTEGER = "integer"
    #: Decimal integers, or scientific notation that is exactly an integer
    #: (`2.4e+07`): R writes a round position that way.
    INTEGER_OR_EXACT_SCIENTIFIC = "integer_or_exact_scientific"


#: Every field a map may declare. The identity fields are required according to
#: the allele convention; the rest are optional.
_FIELDS = frozenset(
    {
        "chromosome", "position", "effect_allele", "other_allele", "ref", "alt",
        "beta", "odds_ratio", "z_score", "standard_error", "frequency", "p_value",
        "ci_lower", "ci_upper", "confidence_interval", "sample_size", "rsid", "variant_id",
    }
)
_EFFECT_FIELDS = ("beta", "odds_ratio", "z_score")
_ALLELE_FIELDS: dict[AlleleConvention, tuple[str, ...]] = {
    AlleleConvention.EFFECT_AND_OTHER: ("effect_allele", "other_allele"),
    AlleleConvention.REF_ALT_EFFECT_IS_ALT: ("ref", "alt"),
    AlleleConvention.REF_ALT_AND_EFFECT_ALLELE: ("ref", "alt", "effect_allele"),
}
_ALL_ALLELE_FIELDS = frozenset({"effect_allele", "other_allele", "ref", "alt"})

_TOP_LEVEL_REQUIRED = ("column_map_version", "name", "columns", "conventions")
_TOP_LEVEL_OPTIONAL = ("description", "optional_columns", "ignored_columns")

#: Conventions every map declares.
_ALWAYS_DECLARED = (
    "missing_values", "unmapped_columns", "alleles", "position_notation",
    "chromosome_spelling", "genome_build",
)
#: Conventions about one column, declared exactly when one of its fields is.
_COLUMN_CONVENTIONS: dict[str, tuple[str, ...]] = {
    "frequency_allele": ("frequency",),
    "p_value_form": ("p_value",),
    "confidence_interval_separator": ("confidence_interval",),
    "rsid_separator": ("rsid", "variant_id"),
}

#: The values a closed-vocabulary convention may take and the reader honours.
_HONOURED: dict[str, frozenset[str]] = {
    "unmapped_columns": frozenset({"ignore", "refuse"}),
    "alleles": frozenset(AlleleConvention),
    "position_notation": frozenset(PositionNotation),
    "chromosome_spelling": frozenset({"adr-0052"}),
    "frequency_allele": frozenset({"effect_allele"}),
    "p_value_form": frozenset({"p"}),
}
#: Values the reader understands and will not read, each with the reason. A
#: map declaring one is refused rather than read as the nearest honoured value.
_UNHONOURED: dict[str, dict[str, str]] = {
    "chromosome_spelling": {
        "plink": (
            "PLINK's 25 = XY pseudo-autosomal code has no OpenGWASDB chromosome "
            "identity, and ADR 0052 reads 25 as MT; resolve XY rows upstream"
        ),
    },
    "frequency_allele": {
        "other_allele": "a frequency of the other allele is not read; no reader inverts it yet",
    },
    "p_value_form": {
        "neg_log10_p": "a -log10 p-value is not read; no reader converts it yet",
    },
}


@dataclass(frozen=True)
class Conventions:
    """A collection's declared conventions, validated.

    `missing_values` is `None` when the map declares `"any_non_numeric"` -- the
    rule GWAS-SSF and FinnGen have always been read under, where any statistic
    that is not a number is missing. A tuple is the closed list of missing
    tokens, and any other non-numeric statistic is then a loud failure.
    `genome_build` is `None` for `"per_analysis"`: the manifest declares it.
    """

    missing_values: tuple[str, ...] | None
    unmapped_columns: str
    alleles: AlleleConvention
    position_notation: PositionNotation
    chromosome_spelling: str
    genome_build: str | None
    frequency_allele: str | None = None
    p_value_form: str | None = None
    confidence_interval_separator: str | None = None
    rsid_separator: str | None = None


@dataclass(frozen=True)
class ColumnMap:
    """One Source Collection's columns and conventions (issue #281).

    `columns` must be present in every file the map reads; `optional_columns`
    are read when a file carries them. Each field maps to the provider's
    enumerated spellings of that column, of which a file may carry one.
    """

    name: str
    origin: str
    sha256: str
    columns: dict[str, tuple[str, ...]]
    optional_columns: dict[str, tuple[str, ...]]
    ignored_columns: tuple[str, ...]
    conventions: Conventions

    def field_spellings(self) -> dict[str, tuple[str, ...]]:
        """Every declared field's spellings, required and optional alike."""
        return {**self.columns, **self.optional_columns}

    def fingerprint(self) -> dict[str, str]:
        """What a Resolution Record or a checkpoint records about this map."""
        return {"name": self.name, "sha256": self.sha256}

    @property
    def spec(self) -> str:
        """What `load_column_map` loads this map from again, from any directory:
        a built-in's bare name, or a file's absolute path."""
        return self.origin.removeprefix("builtin:")


def load_column_map(spec: str | Path) -> ColumnMap:
    """Load and validate a Column Map named by a file path or a built-in name.

    A string is a built-in name exactly when it is one of
    `BUILTIN_COLUMN_MAPS`; anything else is a path. A built-in name that is
    also a file in the working directory is refused as ambiguous rather than
    resolved either way.
    """
    if isinstance(spec, str) and spec in BUILTIN_COLUMN_MAPS:
        if Path(spec).exists():
            raise ColumnMapError(
                f"Column Map {spec!r} is ambiguous: it names a built-in map and a file; "
                f"pass ./{spec} for the file"
            )
        text = resources.files("opengwasdb.readers").joinpath(f"column_maps/{spec}.toml")
        return parse_column_map(text.read_bytes(), origin=f"builtin:{spec}")
    path = Path(spec)
    if not path.is_file():
        builtins = ", ".join(BUILTIN_COLUMN_MAPS)
        raise ColumnMapError(
            f"Column Map {str(spec)!r} is neither a file nor a built-in map ({builtins})"
        )
    # Absolute, so `spec` reloads this file from any directory -- and a file
    # named like a built-in (`./gwas-ssf`) is never reloaded as the built-in.
    return parse_column_map(path.read_bytes(), origin=str(path.resolve()))


def as_column_map(spec: ColumnMap | str | Path | None) -> ColumnMap | None:
    """A caller's Column Map argument, loaded when it names one."""
    if spec is None or isinstance(spec, ColumnMap):
        return spec
    return load_column_map(spec)


def parse_column_map(content: bytes, *, origin: str) -> ColumnMap:
    """Validate one Column Map's bytes; `origin` names it in every refusal."""
    try:
        document = tomllib.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ColumnMapError(f"Column Map {origin}: not valid UTF-8 TOML: {exc}") from exc
    try:
        column_map = _column_map(document, origin, hashlib.sha256(content).hexdigest())
    except ColumnMapError as exc:
        raise ColumnMapError(f"Column Map {origin}: {exc}") from None
    if origin.startswith("builtin:") and column_map.name != origin.removeprefix("builtin:"):
        raise ColumnMapError(f"Column Map {origin}: built-in map names itself {column_map.name!r}")
    return column_map


def _column_map(document: dict[str, Any], origin: str, sha256: str) -> ColumnMap:
    _require_keys(document, _TOP_LEVEL_REQUIRED, _TOP_LEVEL_OPTIONAL, "top-level key")
    if document["column_map_version"] != _FORMAT_VERSION:
        raise ColumnMapError(
            f"unsupported column_map_version {document['column_map_version']!r}; "
            f"this reader reads version {_FORMAT_VERSION}"
        )
    name = _text(document["name"], "name")
    columns = _field_table(document["columns"], "columns")
    optional = _field_table(document.get("optional_columns", {}), "optional_columns")
    ignored = _spellings(document.get("ignored_columns", []), "ignored_columns", allow_empty=True)
    conventions = _conventions(document["conventions"], {*columns, *optional})
    _check_fields(columns, optional, conventions.alleles)
    _check_spellings_distinct({**columns, **optional}, ignored)
    return ColumnMap(name, origin, sha256, columns, optional, ignored, conventions)


def _require_keys(
    table: Mapping[str, Any], required: tuple[str, ...], optional: tuple[str, ...], what: str
) -> None:
    unknown = sorted(set(table) - set(required) - set(optional))
    if unknown:
        raise ColumnMapError(f"unknown {what}(s): {', '.join(unknown)}")
    missing = [key for key in required if key not in table]
    if missing:
        raise ColumnMapError(f"undeclared {what}(s): {', '.join(missing)}")


def _text(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ColumnMapError(f"{what} must be a non-empty string, got {value!r}")
    return value


def _spellings(value: object, what: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    """A column's spellings: one name, or a list of the names it may carry."""
    names = [value] if isinstance(value, str) else value
    if not isinstance(names, list) or (not names and not allow_empty):
        raise ColumnMapError(f"{what} must be a column name or a list of them, got {value!r}")
    spellings = tuple(_text(name, what) for name in names)
    if len(set(spellings)) != len(spellings):
        raise ColumnMapError(f"{what} repeats a spelling: {list(spellings)!r}")
    return spellings


def _field_table(table: object, section: str) -> dict[str, tuple[str, ...]]:
    if not isinstance(table, dict):
        raise ColumnMapError(f"[{section}] must be a table")
    unknown = sorted(set(table) - _FIELDS)
    if unknown:
        raise ColumnMapError(f"unknown field(s) in [{section}]: {', '.join(unknown)}")
    return {field: _spellings(value, f"{section}.{field}") for field, value in table.items()}


def _check_fields(
    columns: dict[str, tuple[str, ...]],
    optional: dict[str, tuple[str, ...]],
    alleles: AlleleConvention,
) -> None:
    """The field set the allele convention needs, and nothing contradicting it."""
    both = sorted(set(columns) & set(optional))
    if both:
        raise ColumnMapError(f"field(s) declared both required and optional: {', '.join(both)}")
    for field in ("chromosome", "position", *_ALLELE_FIELDS[alleles]):
        if field not in columns:
            where = "declared optional" if field in optional else "undeclared"
            raise ColumnMapError(f"required field {field!r} is {where} in [columns]")
    extra = sorted((_ALL_ALLELE_FIELDS - set(_ALLELE_FIELDS[alleles])) & {*columns, *optional})
    if extra:
        raise ColumnMapError(
            f"allele field(s) {', '.join(extra)} are not read under alleles = {alleles.value!r}"
        )
    _check_optional_fields({*columns, *optional})


def _check_optional_fields(declared: set[str]) -> None:
    if not declared & set(_EFFECT_FIELDS):
        effects = ", ".join(_EFFECT_FIELDS)
        raise ColumnMapError(f"no effect column declared; declare one of {effects}")
    if ("ci_lower" in declared) != ("ci_upper" in declared):
        raise ColumnMapError("ci_lower and ci_upper are declared together or not at all")
    if "confidence_interval" in declared and "ci_lower" in declared:
        raise ColumnMapError("declare confidence_interval or ci_lower/ci_upper, not both")


def _check_spellings_distinct(fields: dict[str, tuple[str, ...]], ignored: tuple[str, ...]) -> None:
    """One provider column is one field: never two, and never both read and ignored."""
    owner: dict[str, str] = {}
    for field, spellings in sorted(fields.items()):
        for spelling in spellings:
            if spelling in owner:
                raise ColumnMapError(
                    f"provider column {spelling!r} is mapped to both {owner[spelling]} and {field}"
                )
            owner[spelling] = field
    read_and_ignored = sorted(set(ignored) & set(owner))
    if read_and_ignored:
        raise ColumnMapError(f"column(s) both mapped and ignored: {', '.join(read_and_ignored)}")


def _conventions(table: object, declared_fields: set[str]) -> Conventions:
    if not isinstance(table, dict):
        raise ColumnMapError("[conventions] must be a table")
    required = (*_ALWAYS_DECLARED, *_column_conventions(declared_fields))
    _require_keys(table, required, (), "convention")
    for key in table:
        if key in _HONOURED:
            _honoured(key, table[key])
    return Conventions(
        missing_values=_missing_values(table["missing_values"]),
        unmapped_columns=table["unmapped_columns"],
        alleles=AlleleConvention(table["alleles"]),
        position_notation=PositionNotation(table["position_notation"]),
        chromosome_spelling=table["chromosome_spelling"],
        genome_build=_genome_build(table["genome_build"]),
        frequency_allele=table.get("frequency_allele"),
        p_value_form=table.get("p_value_form"),
        confidence_interval_separator=_separator(
            table, "confidence_interval_separator", may_be_none=False
        ),
        rsid_separator=_separator(table, "rsid_separator"),
    )


def _column_conventions(declared_fields: set[str]) -> tuple[str, ...]:
    """The column conventions this field set needs; one about no column is unknown."""
    return tuple(
        key for key, fields in _COLUMN_CONVENTIONS.items() if declared_fields & set(fields)
    )


def _honoured(key: str, value: object) -> None:
    if isinstance(value, str) and value in _HONOURED[key]:
        return
    reason = _UNHONOURED.get(key, {}).get(value) if isinstance(value, str) else None
    if reason is not None:
        raise ColumnMapError(f"{key} = {value!r}: the reader cannot honour it: {reason}")
    allowed = ", ".join(sorted(_HONOURED[key]))
    raise ColumnMapError(f"{key} = {value!r} is not a known value; expected one of {allowed}")


def _missing_values(value: object) -> tuple[str, ...] | None:
    if value == "any_non_numeric":
        return None
    if not isinstance(value, list) or not all(isinstance(token, str) for token in value):
        raise ColumnMapError(
            f"missing_values must be a list of tokens or 'any_non_numeric', got {value!r}"
        )
    return tuple(token.strip() for token in value)


def _genome_build(value: object) -> str | None:
    if value == "per_analysis":
        return None
    try:
        return normalise_build(_text(value, "genome_build"))
    except ValueError as exc:
        raise ColumnMapError(f"genome_build {value!r}: {exc}; or 'per_analysis'") from exc


def _separator(table: Mapping[str, Any], key: str, *, may_be_none: bool = True) -> str | None:
    """A one-character separator; `'none'` only where a cell may hold one value.

    An interval column always holds two bounds, so it has no `'none'`: a map
    that maps one must say what separates them rather than be given a comma.
    """
    if key not in table:
        return None
    value = table[key]
    if value == "none" and may_be_none:
        return None
    if not isinstance(value, str) or len(value) != 1 or value in "\t\r\n":
        allowed = "one character or 'none'" if may_be_none else "one character"
        raise ColumnMapError(f"{key} must be {allowed}, got {value!r}")
    return value


def mapped_row_build(
    declared: str | None, fallback: str | None, column_map: ColumnMap, context: str
) -> str:
    """The source build a mapped row is read in (issue #281).

    The one rule resolution and Store Release Build both apply: the row's own
    build, else the caller's explicit fallback, else the map's `genome_build`.
    A `per_analysis` map leaves the build to the manifest, so a row that
    declares none, with no fallback, is refused -- never read as hg19 because
    that is a builder's default.
    """
    if declared:
        return declared
    build = fallback or column_map.conventions.genome_build
    if build is None:
        raise ColumnMapError(
            f"{context} declares no source build, and Column Map {column_map.name!r} "
            "declares genome_build = 'per_analysis'; declare the row's build or a CLI fallback"
        )
    return build


def check_mapped_rows(
    rows: Sequence[tuple[str, str, str]], column_map: ColumnMap | None, context: str
) -> None:
    """Refuse a manifest whose rows and Column Map do not agree (issue #281).

    `rows` are `(analysis_id, source_reader_capability, source build)`. A row
    read through a map needs one; a map no row reads through is a run that
    looks mapped and is not; and a row declaring another build than its map
    declares is two authorities disagreeing, which is refused, never resolved.
    """
    mapped = [row for row in rows if row[1] == COLUMN_MAP_CAPABILITY]
    _require_map_and_rows(len(mapped), column_map, context)
    declared = None if column_map is None else column_map.conventions.genome_build
    if column_map is None or declared is None:
        return
    for analysis_id, _capability, build in mapped:
        if normalise_build(build) != declared:
            raise ColumnMapError(
                f"{context}: analysis {analysis_id!r} declares source build {build!r} but "
                f"Column Map {column_map.name!r} declares genome_build {declared!r}"
            )


def _require_map_and_rows(n_mapped: int, column_map: ColumnMap | None, context: str) -> None:
    if n_mapped and column_map is None:
        raise ColumnMapError(
            f"{context}: {n_mapped} Analyses declare source_reader_capability "
            f"{COLUMN_MAP_CAPABILITY!r}; supply their Column Map with --column-map"
        )
    if column_map is not None and not n_mapped:
        raise ColumnMapError(
            f"{context}: Column Map {column_map.name!r} was supplied but no Analysis "
            f"declares source_reader_capability {COLUMN_MAP_CAPABILITY!r}"
        )
