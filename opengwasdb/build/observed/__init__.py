"""Observed-Only build pipeline."""

from pathlib import Path

from opengwasdb.build.source import read_normalised_associations, stream_normalised_associations
from opengwasdb.layouts.dense import DenseBuildResult, build_dense_observed_store
from opengwasdb.model.enums import AssociationCoverage


def build_dense_observed_from_sources(
    source_paths: list[str | Path],
    output_path: str | Path,
    *,
    store_id: str,
    release_id: str,
    reference_assembly: str,
    overwrite: bool = False,
    association_coverage: AssociationCoverage | str = AssociationCoverage.GWAS,
) -> DenseBuildResult:
    records = stream_normalised_associations(source_paths)
    return build_dense_observed_store(
        records,
        output_path,
        store_id=store_id,
        release_id=release_id,
        reference_assembly=reference_assembly,
        overwrite=overwrite,
        association_coverage=association_coverage,
    )


__all__ = [
    "build_dense_observed_from_sources",
    "build_dense_observed_store",
    "read_normalised_associations",
    "stream_normalised_associations",
]
