"""The build manifest: one row per Analysis, read once for every builder that takes one.

The Dense and Hybrid builders and Variant Reference extraction all read the
same manifest the same way: canonical analyses.tsv names or their legacy
aliases, each row's Source Reader capability and source assembly with their
per-release defaults, the SD standardisation its original_sd_method implies,
and the Analytical and Attribution Metadata the builders copy into
analyses.tsv. A row is validated as it is read, so a bad manifest fails
before any source is opened.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from opengwasdb.build.liftover import normalise_build
from opengwasdb.model.analyses import Analysis, PassthroughMetadata
from opengwasdb.model.enums import OriginalSdMethod, StoredEffectScale
from opengwasdb.model.manifest_columns import (
    ManifestColumns,
    manifest_n,
    manifest_trait_name,
    require_columns,
    resolve_manifest_columns,
)
from opengwasdb.readers.column_map import (
    COLUMN_MAP_CAPABILITY,
    ColumnMap,
    MappedRowDeclaration,
    as_column_map,
    check_mapped_rows,
    mapped_row_build,
)
from opengwasdb.readers.gwas_vcf import GWAS_VCF_CAPABILITY
from opengwasdb.readers.registry import SourceSpec, known_capabilities

__all__ = [
    "ManifestRow",
    "apply_se_divisor",
    "column_map_provenance",
    "manifest_row_to_analysis",
    "read_manifest",
]


@dataclass(frozen=True)
class ManifestRow:
    """One Analysis's row: its source, how to read it, and the metadata copied from it."""

    trait_id: str
    file_path: str
    trait_name: str
    n: int
    stored_effect_scale: str
    se_divisor: float  # divides original_se to standardise to SD units (issue #18); 1.0 = no-op
    source_reader_capability: str  # resolves to a SourceReader (issue #20); GWAS_VCF_CAPABILITY
    # when the manifest omits the column -- the only format this builder has ever supported.
    source_assembly: str  # normalised "hg19"/"hg38" (issue #85); "hg19" when the manifest omits
    # the column -- every source this builder read before GWAS-SSF (#84) was hg19 GWAS-VCF.
    original_sd: str  # raw manifest value ("" when the sd_method tier carries no magnitude);
    # not part of PassthroughMetadata because this builder *uses* it (se_divisor), rather
    # than only copying it.
    assigned_ancestry: str  # optional manifest column (issue #22); "" when the manifest omits it
    # or carries no assignment for this row -- e.g. a Catalogue subset's kept rows (already
    # filtered to one target ancestry) stamp this in verbatim; a bare manifest has no column.
    # Shared-core analyses.tsv columns the manifest supplies and this builder only copies
    # (issues #86, #83) -- sample-size interpretation and counts, Original Effect Scale,
    # ancestry-assignment method and proportions, Attribution. Blank when absent; the shared
    # builder must not infer their meaning. One type, shared with Ragged, so a column added
    # to the contract reaches both layouts.
    metadata: PassthroughMetadata = field(default_factory=PassthroughMetadata)
    # Trait-ontology columns (ADR 0034, issue #68): optional manifest columns, resolved
    # directly rather than via a phenotype_id/phenotype_label intermediary. "" when the
    # manifest omits them -- never fabricated. Not passthrough-shared: Ragged resolves its
    # own (a gene ID, ADR 0035) rather than copying a manifest column.
    trait_ontology_id: str = ""
    trait_ontology_label: str = ""
    # The Column Map an `opengwasdb.column-map` row is read through (issue
    # #281); `None` for every other capability. Carried on the row so every
    # pass that resolves this row's reader resolves the same one.
    column_map: ColumnMap | None = None

    @property
    def source(self) -> SourceSpec:
        """The row's source, as Variant Routing and the Pass 2 workers read it."""
        return SourceSpec(
            analysis_id=self.trait_id,
            path=self.file_path,
            capability=self.source_reader_capability,
            assembly=self.source_assembly,
            stored_effect_scale=StoredEffectScale(self.stored_effect_scale),
            column_map=self.column_map,
        )


def apply_se_divisor(se: np.ndarray, se_divisor: float) -> np.ndarray:
    """Continuous-trait phenotype-SD standardisation (issue #18):
    ``stored_se = original_se / sd``. Shared by the dense and hybrid builders,
    both of which resolve an association stream to ``(index, z, se)`` and need
    the same no-op-when-1.0 divide applied to ``se`` before spilling."""
    if se_divisor == 1.0:
        return se
    return se / np.float32(se_divisor)


def manifest_row_to_analysis(row: ManifestRow) -> Analysis:
    """A manifest row's Analytical + Attribution Metadata (issue #22, ADR
    0034), shared by the dense and hybrid manifest builders.

    Optional fields are passed through verbatim and remain blank when absent.
    In particular, the shared builder cannot infer how ancestry was assigned:
    the manifest producer owns that fact (issue #86).
    """
    return row.metadata.applied_to(
        Analysis(
            analysis_id=row.trait_id,
            analysis_label=row.trait_name,
            trait_ontology_id=row.trait_ontology_id,
            trait_ontology_label=row.trait_ontology_label,
            stored_effect_scale=row.stored_effect_scale,
            assigned_ancestry=row.assigned_ancestry,
            sample_size=str(row.n) if row.n else "",
            original_sd=row.original_sd,
        )
    )


# original_sd_method tiers that carry an actual phenotype-SD magnitude to divide
# by (ADR-0029 methods 2-5). declared_standardised implies sd=1 (no magnitude
# recorded); binary_trait is not SD-scale at all -- neither rescales.
_SD_RESCALE_METHODS = frozenset(
    {
        OriginalSdMethod.SOURCE_PROVIDED,
        OriginalSdMethod.ESTIMATED_FROM_SOURCE_MAF,
        OriginalSdMethod.ESTIMATED_FROM_REFERENCE_MAF,
        OriginalSdMethod.ESTIMATED_FROM_BETA_DISTRIBUTION,
    }
)


def _parse_source_assembly(
    row: Mapping[str, str], fallback: str, trait_id: str, manifest_path: str | Path
) -> str:
    """Resolve and validate source_assembly for one manifest row."""
    raw = row.get("source_assembly") or fallback
    try:
        return normalise_build(raw)
    except ValueError as exc:
        raise ValueError(
            f"manifest {manifest_path}: analysis {trait_id!r} has invalid "
            f"source_assembly {raw!r}"
        ) from exc


def _parse_reader_capability(
    row: Mapping[str, str], fallback: str, trait_id: str, manifest_path: str | Path
) -> str:
    """Resolve and validate source_reader_capability for one manifest row."""
    capability = row.get("source_reader_capability") or fallback
    if capability not in known_capabilities():
        known = ", ".join(known_capabilities()) or "(none registered)"
        raise ValueError(
            f"manifest {manifest_path}: analysis {trait_id!r} has unknown "
            f"source_reader_capability {capability!r}; known: {known}"
        )
    return capability


def _parse_original_sd(
    row: Mapping[str, str], trait_id: str, manifest_path: str | Path
) -> float:
    """Validate original_sd_method and original_sd, returning se_divisor."""
    sd_method_raw = row["original_sd_method"]
    try:
        sd_method = OriginalSdMethod(sd_method_raw)
    except ValueError as exc:
        raise ValueError(
            f"manifest {manifest_path}: analysis {trait_id!r} has invalid "
            f"original_sd_method {sd_method_raw!r}"
        ) from exc
    if sd_method is OriginalSdMethod.UNAVAILABLE:
        raise ValueError(
            f"manifest {manifest_path}: analysis {trait_id!r} has "
            "original_sd_method='unavailable' -- its phenotype SD could not be "
            "established upstream, so the build cannot standardise its effects (issue #18)"
        )
    se_divisor = 1.0
    original_sd_raw = row.get("original_sd", "")
    if sd_method in _SD_RESCALE_METHODS:
        try:
            se_divisor = float(original_sd_raw)
        except ValueError as exc:
            raise ValueError(
                f"manifest {manifest_path}: analysis {trait_id!r} has "
                f"original_sd_method={sd_method.value!r} but original_sd "
                f"{original_sd_raw!r} is not a valid number"
            ) from exc
        if not se_divisor > 0:
            raise ValueError(
                f"manifest {manifest_path}: analysis {trait_id!r} has "
                f"non-positive original_sd {original_sd_raw!r}"
            )
    elif original_sd_raw:
        raise ValueError(
            f"manifest {manifest_path}: analysis {trait_id!r} has "
            f"original_sd_method={sd_method.value!r}, which carries no SD "
            f"magnitude, but original_sd={original_sd_raw!r} was supplied"
        )
    return se_divisor


def _parse_manifest_row(
    row: Mapping[str, str],
    cols: ManifestColumns,
    manifest_path: str | Path,
    fallback_assembly: str,
    fallback_capability: str,
) -> ManifestRow:
    """Parse and validate one manifest row into a ManifestRow."""
    trait_id = row[cols.analysis_id]
    scale = row["stored_effect_scale"]
    try:
        StoredEffectScale(scale)
    except ValueError as exc:
        raise ValueError(
            f"manifest {manifest_path}: analysis {trait_id!r} has invalid "
            f"stored_effect_scale {scale!r}"
        ) from exc
    assembly = _parse_source_assembly(row, fallback_assembly, trait_id, manifest_path)
    capability = _parse_reader_capability(row, fallback_capability, trait_id, manifest_path)
    se_divisor = _parse_original_sd(row, trait_id, manifest_path)
    return ManifestRow(
        trait_id=trait_id,
        file_path=row[cols.source_file],
        trait_name=manifest_trait_name(row, cols.analysis_label, trait_id),
        n=manifest_n(row, cols.sample_size),
        stored_effect_scale=scale,
        se_divisor=se_divisor,
        source_reader_capability=capability,
        source_assembly=assembly,
        original_sd=row.get("original_sd", ""),
        assigned_ancestry=row.get("assigned_ancestry") or "",
        metadata=PassthroughMetadata.from_manifest_row(row),
        trait_ontology_id=row.get("trait_ontology_id") or "",
        trait_ontology_label=(
            row.get("trait_ontology_label") or row.get("trait_ontology_name") or ""
        ),
    )


def _resolve_manifest_defaults(
    default_capability: str | None, default_assembly: str | None
) -> tuple[str, str]:
    """Validate and resolve fallback capability and assembly."""
    if default_capability is not None:
        if not default_capability or default_capability not in known_capabilities():
            known = ", ".join(known_capabilities()) or "(none registered)"
            raise ValueError(
                f"unknown source reader capability {default_capability!r}; known: {known}"
            )
    if default_assembly is not None:
        default_assembly = normalise_build(default_assembly)
    return default_assembly or "hg19", default_capability or GWAS_VCF_CAPABILITY


def read_manifest(
    manifest_path: str | Path,
    *,
    default_source_reader_capability: str | None = None,
    default_source_assembly: str | None = None,
    column_map: ColumnMap | str | Path | None = None,
) -> list[ManifestRow]:
    """Read the build manifest with canonical names, aliases, and option defaults.

    Accepts canonical ``analyses.tsv`` names (``analysis_id``/``source_file``/
    ``analysis_label``/``sample_size``, ADR 0034, #170) or legacy aliases
    (``trait_id``/``file_path``/``trait_name``/``n``). An empty manifest is refused
    here, with the path a caller needs. Requires ``stored_effect_scale``
    (#17) and ``original_sd_method`` (#18; ``original_sd`` required for rescaling tiers).

    Optional ``source_reader_capability`` (#20, #174) and ``source_assembly`` (#85, #174)
    fall back to CLI defaults (or GWAS-VCF/hg19). Note harmonised SSF is already hg38 and
    must declare so or coordinates liftover twice (issue #85 double-liftover warning).

    ``column_map`` is the Column Map ``opengwasdb.column-map`` rows are read
    through (#281). Its declared genome build stands in for a row omitting
    ``source_assembly`` when no CLI default is given, and must agree with every
    such row's build otherwise.
    """
    fallback_assembly, fallback_capability = _resolve_manifest_defaults(
        default_source_reader_capability, default_source_assembly
    )
    with open(manifest_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    require_columns(fieldnames, manifest_path, "stored_effect_scale", "original_sd_method")
    cols = resolve_manifest_columns(fieldnames, manifest_path)
    loaded_map = as_column_map(column_map)
    parsed = [
        _with_column_map(
            _parse_manifest_row(row, cols, manifest_path, fallback_assembly, fallback_capability),
            row, loaded_map, default_source_assembly, f"manifest {manifest_path}",
        )
        for row in rows
    ]
    check_mapped_rows(
        [
            MappedRowDeclaration(
                row.trait_id, row.source_reader_capability, row.source_assembly
            )
            for row in parsed
        ],
        loaded_map,
        f"manifest {manifest_path}",
    )
    if not parsed:
        raise ValueError(f"manifest {manifest_path} contains no rows")
    return parsed


def column_map_provenance(rows: Iterable[ManifestRow]) -> dict[str, str] | None:
    """The Column Map a build's mapped rows were read through, as a store records it.

    `{"name", "sha256"}` -- the fingerprint a Resolution Record carries -- or
    `None` when no row was mapped, so an unmapped store's provenance is
    unchanged (#281). `read_manifest` gives every mapped row the same map.
    """
    for row in rows:
        if row.column_map is not None:
            return row.column_map.fingerprint()
    return None


def _with_column_map(
    row: ManifestRow,
    raw: Mapping[str, str],
    column_map: ColumnMap | None,
    default_assembly: str | None,
    context: str,
) -> ManifestRow:
    """A mapped row with its Column Map, in the build `mapped_row_build` gives it.

    The row's own build is re-derived through the shared rule rather than taken
    from `_parse_manifest_row`, which falls back to hg19 -- a default a mapped
    row must never inherit (#281).
    """
    if row.source_reader_capability != COLUMN_MAP_CAPABILITY or column_map is None:
        return row
    declared = row.source_assembly if raw.get("source_assembly") else None
    build = mapped_row_build(
        declared, default_assembly, column_map, f"{context}: analysis {row.trait_id!r}"
    )
    return replace(row, column_map=column_map, source_assembly=normalise_build(build))
