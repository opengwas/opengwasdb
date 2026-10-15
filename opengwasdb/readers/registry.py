"""Source Reader Capability resolution (issue #19; opengwasdb-stores ADR-0009).

The single, documented point that maps a Source Collection's
`source_reader_capability` string to a concrete `SourceReader`. Existing
builders do not yet call through this resolver -- wiring them is out of
scope here (see `opengwasdb.readers.interface`'s module docstring); this
ticket only introduces the seam.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers.column_map import COLUMN_MAP_CAPABILITY, ColumnMap
from opengwasdb.readers.column_map_reader import ColumnMapReader
from opengwasdb.readers.finngen import FINNGEN_R13_CAPABILITY, FinnGenR13Reader
from opengwasdb.readers.gwas_ssf import GWAS_SSF_CAPABILITY, GwasSsfReader
from opengwasdb.readers.gwas_vcf import GWAS_VCF_CAPABILITY, GwasVcfReader
from opengwasdb.readers.interface import ImputationScoreDeclaration, SourceReader


def _needs_column_map(path: str | Path, _scale: StoredEffectScale) -> SourceReader:
    """`opengwasdb.column-map` without its map: `resolve_reader` builds a mapped
    reader itself, so reaching this factory means no map was given."""
    raise ValueError(f"{COLUMN_MAP_CAPABILITY!r} source {path} needs a Column Map")


_READERS: dict[str, Callable[[str | Path, StoredEffectScale], SourceReader]] = {
    COLUMN_MAP_CAPABILITY: _needs_column_map,
    FINNGEN_R13_CAPABILITY: FinnGenR13Reader,
    GWAS_VCF_CAPABILITY: GwasVcfReader,
    GWAS_SSF_CAPABILITY: GwasSsfReader,
}


def known_capabilities() -> tuple[str, ...]:
    """Return all registered source reader capabilities in sorted order."""
    return tuple(sorted(_READERS))


def resolve_reader(
    capability: str, path: str | Path, stored_effect_scale: StoredEffectScale,
    *, imputation_score_declaration: ImputationScoreDeclaration | None = None,
    column_map: ColumnMap | None = None,
) -> SourceReader:
    """Return a SourceReader for `path`, given its Source Collection's
    `source_reader_capability` string.

    `stored_effect_scale` is the Analysis's manifest-resolved effect scale
    (issue #17): every reader needs it, since no Source Format's own header
    is authoritative for it.

    `column_map` is the Column Map an `opengwasdb.column-map` Analysis is read
    through (issue #281). It is required for that capability and refused for
    every other, so a map can neither be forgotten nor silently not used.

    Raises ValueError for an unknown capability, naming the ones this
    package knows about.
    """
    if column_map is not None:
        return _column_map_reader(
            capability, path, stored_effect_scale, imputation_score_declaration, column_map
        )
    try:
        reader_factory = _READERS[capability]
    except KeyError:
        known = ", ".join(known_capabilities()) or "(none registered)"
        raise ValueError(
            f"unknown source reader capability {capability!r}; known: {known}"
        ) from None
    # Built first, so a capability that cannot be read as given (a mapped one
    # without its map) says so before anything else is said about it.
    reader = reader_factory(path, stored_effect_scale)
    if imputation_score_declaration is None:
        return reader
    if capability != GWAS_SSF_CAPABILITY:
        raise ValueError(f"imputation score declaration unsupported by {capability!r}")
    return GwasSsfReader(
        path, stored_effect_scale, imputation_score_declaration=imputation_score_declaration
    )


def _column_map_reader(
    capability: str,
    path: str | Path,
    stored_effect_scale: StoredEffectScale,
    imputation_score_declaration: ImputationScoreDeclaration | None,
    column_map: ColumnMap,
) -> ColumnMapReader:
    if capability != COLUMN_MAP_CAPABILITY:
        raise ValueError(
            f"a Column Map reads only {COLUMN_MAP_CAPABILITY!r} sources, not {capability!r}"
        )
    return ColumnMapReader(
        path,
        column_map,
        stored_effect_scale,
        imputation_score_declaration=imputation_score_declaration,
    )
