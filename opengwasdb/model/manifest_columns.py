"""Manifest column alias resolution (issue #170, #172, #177; ADR 0034).

Canonical ``analyses.tsv`` column names (ADR 0034) are the standard across all
manifests and builders. Legacy builder-manifest names remain supported for a
deprecation window (issue #170, #172). When both canonical and legacy spellings
are present in a manifest, the canonical name always wins.
"""

from __future__ import annotations

from collections.abc import Container, Mapping
from dataclasses import dataclass
from pathlib import Path

#: Canonical ``analyses.tsv`` names -> tuple of legacy aliases in precedence order.
#: A manifest may use canonical or legacy spellings; when both are present the
#: canonical name wins.
MANIFEST_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "analysis_id": ("trait_id",),
    "source_file": ("file_path", "filtered_file"),
    "analysis_label": ("trait_name",),
    "sample_size": ("n",),
}


def manifest_column(
    fieldnames: Container[str],
    canonical: str,
) -> str | None:
    """Resolve `canonical`'s column name, or None when neither spelling is present.

    Returns the canonical ``analyses.tsv`` name when present in `fieldnames`,
    otherwise the first matching legacy alias, or None when absent.
    """
    if canonical in fieldnames:
        return canonical
    for legacy in MANIFEST_COLUMN_ALIASES.get(canonical, ()):
        if legacy in fieldnames:
            return legacy
    return None


def required_manifest_column(
    fieldnames: Container[str],
    canonical: str,
    manifest_path: str | Path | None = None,
) -> str:
    """Resolve `canonical`'s column name, raising ValueError when neither the
    canonical spelling nor any legacy alias is present."""
    column = manifest_column(fieldnames, canonical)
    if column is not None:
        return column
    aliases = MANIFEST_COLUMN_ALIASES.get(canonical, ())
    loc = f"manifest {manifest_path} is missing" if manifest_path else "missing"
    if len(aliases) == 1:
        raise ValueError(
            f"{loc} required column: {canonical!r} (legacy name {aliases[0]!r})"
        )
    if len(aliases) > 1:
        legacy_str = ", ".join(repr(a) for a in aliases)
        raise ValueError(
            f"{loc} required column: {canonical!r} (legacy aliases: {legacy_str})"
        )
    raise ValueError(f"{loc} required column: {canonical!r}")


def require_columns(
    fieldnames: Container[str],
    manifest_path: str | Path | None = None,
    *columns: str,
) -> None:
    """Raise ValueError when any of `columns` is absent from `fieldnames`."""
    loc = f"manifest {manifest_path} is missing" if manifest_path else "missing"
    for column in columns:
        if column not in fieldnames:
            raise ValueError(f"{loc} required column: {column!r}")


def manifest_trait_name(
    row: Mapping[str, str],
    label_col: str | None,
    trait_id: str,
    *,
    fallback_on_blank: bool = False,
) -> str:
    """The row's analysis_label, or its analysis_id when the manifest names no
    label column (``_read_manifest``'s legacy ``trait_name`` default).

    When `fallback_on_blank` is False (the default, preserving Dense and
    ADR 0034 semantics), a present-but-blank label column preserves ``""``;
    when True (Ancestry's legacy fallback), a blank label falls back to
    `trait_id`.
    """
    if label_col is None:
        return trait_id
    val = row.get(label_col, trait_id)
    if fallback_on_blank and not val:
        return trait_id
    return val


def manifest_n(row: Mapping[str, str], sample_size_col: str | None) -> int:
    """The row's sample_size, or 0 when the manifest names no sample-size
    column (``_read_manifest``'s legacy ``n`` default)."""
    return int((row.get(sample_size_col) or 0) if sample_size_col is not None else 0)


@dataclass(frozen=True)
class ManifestColumns:
    """The four alias-bearing manifest columns, resolved to real names."""

    analysis_id: str
    source_file: str
    analysis_label: str | None
    sample_size: str | None


def resolve_manifest_columns(
    fieldnames: Container[str],
    manifest_path: str | Path | None = None,
) -> ManifestColumns:
    """Resolve the alias-bearing columns, raising when a required one is absent."""
    return ManifestColumns(
        analysis_id=required_manifest_column(fieldnames, "analysis_id", manifest_path),
        source_file=required_manifest_column(fieldnames, "source_file", manifest_path),
        analysis_label=manifest_column(fieldnames, "analysis_label"),
        sample_size=manifest_column(fieldnames, "sample_size"),
    )
