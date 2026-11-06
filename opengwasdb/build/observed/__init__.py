"""Observed-Only build pipeline."""

from pathlib import Path

from opengwasdb.admission.declared import declared_rules
from opengwasdb.admission.rules import AnalysisRules
from opengwasdb.build.liftover import normalise_build
from opengwasdb.layouts.build_manifest import ManifestRow, manifest_row_to_analysis, read_manifest
from opengwasdb.layouts.dense import (
    DEFAULT_CHUNK_SHAPE,
    AnalysisSource,
    DenseBuildResult,
    build_dense_observed_store,
)
from opengwasdb.model.enums import AssociationCoverage
from opengwasdb.readers.association_tsv import ASSOCIATION_TSV_CAPABILITY
from opengwasdb.readers.interface import StreamTally
from opengwasdb.readers.registry import open_source


def build_dense_observed_from_manifest(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    store_id: str,
    release_id: str,
    reference_assembly: str = "GRCh37",
    chunk_shape: tuple[int, int] = DEFAULT_CHUNK_SHAPE,
    overwrite: bool = False,
    association_coverage: AssociationCoverage | str = AssociationCoverage.GWAS,
    admission_record: str | Path | None = None,
) -> DenseBuildResult:
    """Build a Dense Observed-Only Store from a build manifest of association TSVs.

    The manifest is the one every builder takes (`read_manifest`): a row per
    Analysis, naming its association TSV (`opengwasdb.association-tsv`) and
    declaring what the Store records of it. This builder does not lift, so every
    row's source assembly must be ``reference_assembly``, which a row that omits
    the column takes.
    """
    reference_build = normalise_build(reference_assembly)
    rows = read_manifest(
        manifest_path,
        default_source_reader_capability=ASSOCIATION_TSV_CAPABILITY,
        default_source_assembly=reference_build,
    )
    _refuse_unreadable_rows(rows, reference_build, reference_assembly, manifest_path)
    rules = declared_rules(
        manifest_path, {row.trait_id: row.source_reader_capability for row in rows}
    )
    sources = [_read_analysis(row, rules[row.trait_id]) for row in rows]
    return build_dense_observed_store(
        sources,
        output_path,
        store_id=store_id,
        release_id=release_id,
        reference_assembly=reference_assembly,
        chunk_shape=chunk_shape,
        overwrite=overwrite,
        association_coverage=association_coverage,
        admission_record=admission_record,
    )


def _refuse_unreadable_rows(
    rows: list[ManifestRow],
    reference_build: str,
    reference_assembly: str,
    manifest_path: str | Path,
) -> None:
    """Refuse a manifest row this builder cannot honour, before any source is read."""
    for row in rows:
        if row.source_reader_capability != ASSOCIATION_TSV_CAPABILITY:
            raise ValueError(
                f"manifest {manifest_path}: analysis {row.trait_id!r} names "
                f"source_reader_capability {row.source_reader_capability!r}, which build-dense "
                f"does not read; it reads {ASSOCIATION_TSV_CAPABILITY!r}"
            )
        if row.source_assembly != reference_build:
            raise ValueError(
                f"manifest {manifest_path}: analysis {row.trait_id!r} is {row.source_assembly}, "
                f"but build-dense does not lift: it writes its sources as they are, as "
                f"{reference_assembly} (--reference-assembly)"
            )


def _read_analysis(row: ManifestRow, rules: AnalysisRules) -> AnalysisSource:
    """One manifest row's association TSV, read through its Source Reader."""
    tally = StreamTally()
    batches = list(open_source(row.source).stream_batches(tally))
    if not batches and not tally.faults.rows:
        raise ValueError(f"analysis {row.trait_id!r}: {row.file_path} holds no associations")
    return AnalysisSource(manifest_row_to_analysis(row), rules, row.se_divisor, batches, tally)


__all__ = ["build_dense_observed_from_manifest", "build_dense_observed_store"]
