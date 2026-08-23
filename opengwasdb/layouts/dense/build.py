"""Dense Observed-Only writer."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numcodecs import Blosc

from opengwasdb.build.source import NormalisedAssociation
from opengwasdb.encoding import (
    EncodingMeasurements,
    StoreCodec,
    StoreEncoding,
    ZOverflowBuilder,
)
from opengwasdb.index import create_lookup_indexes, initialise_schema, set_metadata
from opengwasdb.layouts.dense.constants import (
    DEFAULT_CHUNK_SHAPE,
    DEFAULT_COMPRESSOR,
    DEFAULT_DTYPE,
)
from opengwasdb.layouts.dense.overview import write_overview_html
from opengwasdb.layouts.dense.top_hits import build_top_hit_indexes, read_top_hit_counts
from opengwasdb.model.analyses import Analysis, analyses_table_from_records, write_analyses
from opengwasdb.model.enums import AssociationCoverage, CompletionState, PrimaryStorageLayout
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.store.open import CURRENT_FORMAT_VERSION, OpenGWASDBStore, StagedRelease
from opengwasdb.variants import (
    VARIANT_AXIS_FORMAT,
    VARIANT_OFFSETS_FILENAME,
    VARIANT_TABIX_FILENAME,
    VARIANT_TABLE_FILENAME,
    CanonicalVariant,
    chromosome_sort_key,
    write_variant_axis,
)


@dataclass(frozen=True)
class DenseBuildResult:
    """Paths and dimensions for a built Dense Observed-Only store."""

    output_path: Path
    n_variants: int
    n_analyses: int


def add_hit_counts(store_path: str | Path, analyses: list[Analysis]) -> list[Analysis]:
    """Add per-Analysis Top-Hit Counts from ``store_path``'s already-built
    top-hit index onto whatever each ``Analysis`` already carries (ADR 0032).

    Blank ``n_hits_*`` fields act as zero, so this both sets counts from
    scratch (a fresh build's analyses start blank) and accumulates a Hybrid
    store's Dense Component and Ragged Overflow Component contributions onto
    the same Analysis when called once per component -- the two partition an
    Analysis's associations disjointly (CONTEXT.md, Query Component).
    """
    counts = read_top_hit_counts(store_path, len(analyses))
    return [
        replace(
            a,
            n_hits_5e8=str(int(a.n_hits_5e8 or 0) + counts["n_hits_5e8"][i]),
            n_hits_5e6=str(int(a.n_hits_5e6 or 0) + counts["n_hits_5e6"][i]),
            n_hits_5e4=str(int(a.n_hits_5e4 or 0) + counts["n_hits_5e4"][i]),
        )
        for i, a in enumerate(analyses)
    ]


def write_analyses_tsv(output_path: Path, analyses: list[Analysis]) -> Path:
    """Write ``analyses.tsv`` + ``overview.html`` -- the sole source of truth
    for Analytical Metadata (ADR 0030, issue #22; unified schema, ADR 0034),
    replacing the old SQLite ``analyses`` table and its own generated
    human-browsable rendering.
    """
    output_path = Path(output_path)
    table = analyses_table_from_records(analyses)
    path = write_analyses(output_path / "analyses.tsv", table)
    write_overview_html(output_path, table)
    return path


def build_dense_observed_store(
    records: list[NormalisedAssociation],
    output_path: str | Path,
    *,
    store_id: str,
    release_id: str,
    reference_assembly: str,
    chunk_shape: tuple[int, int] = DEFAULT_CHUNK_SHAPE,
    dtype: str = DEFAULT_DTYPE,
    overwrite: bool = False,
) -> DenseBuildResult:
    """Write a Dense Observed-Only Store Release from normalised associations."""

    if not records:
        raise ValueError("cannot build a store with no association records")

    out = Path(output_path)
    with OpenGWASDBStore.staging(out, overwrite=overwrite) as staged:
        variants = _collect_variants(records)
        analyses = _collect_analyses(records)
        variant_index = {variant.alid: i for i, variant in enumerate(variants)}
        analysis_index = {analysis.analysis_id: i for i, analysis in enumerate(analyses)}

        # z is accumulated in float32 and quantised once, by the codec, on the
        # way to disk -- never pre-rounded into the stored dtype here.
        z = np.full((len(variants), len(analyses)), np.nan, dtype=np.float32)
        se = np.full((len(variants), len(analyses)), np.nan, dtype=dtype)

        seen_cells: set[tuple[int, int]] = set()
        for record in records:
            row = variant_index[record.variant.alid]
            col = analysis_index[record.analysis_id]
            cell = (row, col)
            if cell in seen_cells:
                raise ValueError(
                    f"duplicate association for variant {record.variant.alid} "
                    f"and analysis {record.analysis_id}"
                )
            seen_cells.add(cell)
            z[row, col] = record.z
            se[row, col] = record.se

        # The encoding plan is decided once, here, and read back from the
        # manifest by everything downstream (ADR 0037, issue #119).
        encoding = StoreEncoding.decide(EncodingMeasurements(n_analyses=len(analyses)))
        rsid_by_alid = _first_rsids_by_alid(records)
        _write_manifest(
            staged, store_id, release_id, reference_assembly, records, chunk_shape, dtype,
            encoding,
        )
        write_variant_axis(staged.path, variants, rsid_by_alid)
        _write_index(staged, variants, analyses, records, chunk_shape, dtype)
        _write_zarr(staged, z, se, chunk_shape, dtype, encoding)
        build_top_hit_indexes(staged.path, encoding=encoding)
        write_analyses_tsv(staged.path, add_hit_counts(staged.path, analyses))
        return DenseBuildResult(output_path=out, n_variants=len(variants), n_analyses=len(analyses))


def _collect_variants(records: list[NormalisedAssociation]) -> list[CanonicalVariant]:
    by_alid = {record.variant.alid: record.variant for record in records}
    return sorted(
        by_alid.values(),
        key=lambda variant: (
            chromosome_sort_key(variant.chromosome),
            variant.position,
            variant.effect_allele,
            variant.other_allele,
        ),
    )


def _collect_analyses(records: list[NormalisedAssociation]) -> list[Analysis]:
    by_id: dict[str, Analysis] = {}
    for record in records:
        existing = by_id.get(record.analysis_id)
        current = Analysis(
            analysis_id=record.analysis_id,
            analysis_label=record.analysis_label or "",
            trait_ontology_id=record.trait_ontology_id or "",
            trait_ontology_label=record.trait_ontology_label or "",
            license=record.license or "",
            publication_doi=record.publication_doi or "",
            publication_pmid=record.publication_pmid or "",
            consortium=record.consortium or "",
            first_author=record.first_author or "",
            stored_effect_scale=record.stored_effect_scale.value,
        )
        if existing is None:
            by_id[record.analysis_id] = current
            continue
        if existing.stored_effect_scale != current.stored_effect_scale:
            raise ValueError(f"analysis {record.analysis_id} has mixed stored_effect_scale values")
    return [by_id[key] for key in sorted(by_id)]


def _write_manifest(
    staged: StagedRelease,
    store_id: str,
    release_id: str,
    reference_assembly: str,
    records: list[NormalisedAssociation],
    chunk_shape: tuple[int, int],
    dtype: str,
    encoding: StoreEncoding,
) -> None:
    manifest = StoreManifest(
        encoding=encoding,
        store_id=store_id,
        release_id=release_id,
        format_version=CURRENT_FORMAT_VERSION,
        primary_layout=PrimaryStorageLayout.DENSE,
        association_coverage=AssociationCoverage.FULL,
        completion_state=CompletionState.OBSERVED_ONLY,
        reference_assembly=reference_assembly,
        created_at=datetime.now(UTC).isoformat(),
        provenance={
            "builder": "opengwasdb.v0.1_dense_observed",
            "source_record_count": len(records),
            "dense": {
                "statistic_arrays": ["z", "se"],
                "se_dtype": dtype,
                "chunk_shape": list(chunk_shape),
                "compressor": DEFAULT_COMPRESSOR,
                "top_hit_thresholds": [5e-8, 5e-6, 5e-4],
                "variant_axis": {
                    "format": VARIANT_AXIS_FORMAT,
                    "table": VARIANT_TABLE_FILENAME,
                    "tabix_index": VARIANT_TABIX_FILENAME,
                    "row_offsets": VARIANT_OFFSETS_FILENAME,
                },
            },
        },
    )
    staged.write_manifest(manifest)


def _write_index(
    staged: StagedRelease,
    variants: list[CanonicalVariant],
    analyses: list[Analysis],
    records: list[NormalisedAssociation],
    chunk_shape: tuple[int, int],
    dtype: str,
) -> None:
    with staged.index_connection() as connection:
        initialise_schema(connection)
        set_metadata(connection, "schema_version", 2)
        set_metadata(connection, "n_variants", len(variants))
        set_metadata(connection, "n_analyses", len(analyses))
        set_metadata(
            connection,
            "dense",
            {
                "se_dtype": dtype,
                "chunk_shape": list(chunk_shape),
                "compressor": DEFAULT_COMPRESSOR,
                "variant_axis": {
                    "format": VARIANT_AXIS_FORMAT,
                    "table": VARIANT_TABLE_FILENAME,
                    "tabix_index": VARIANT_TABIX_FILENAME,
                    "row_offsets": VARIANT_OFFSETS_FILENAME,
                },
            },
        )
        # `variant_aliases` is no longer the rsid lookup path -- `write_variant_axis`
        # writes an rsid search index every layout shares (issue #109), and
        # `VariantAxis.by_identifier` consults that first. The table is still written
        # here, and still read as a fallback, so stores built before that index keep
        # resolving rsids; nothing new depends on it.
        variant_indices = {variant.alid: i for i, variant in enumerate(variants)}
        aliases: set[tuple[str, int]] = set()
        for record in records:
            if record.rsid:
                aliases.add((record.rsid, variant_indices[record.variant.alid]))
        connection.executemany(
            "INSERT OR IGNORE INTO variant_aliases(alias, variant_index) VALUES (?, ?)",
            sorted(aliases),
        )
        connection.commit()


def _first_rsids_by_alid(records: list[NormalisedAssociation]) -> dict[str, str]:
    rsid_by_alid: dict[str, str] = {}
    for record in records:
        if record.rsid:
            rsid_by_alid.setdefault(record.variant.alid, record.rsid)
    return rsid_by_alid


def _write_zarr(
    staged: StagedRelease,
    z: np.ndarray,
    se: np.ndarray,
    chunk_shape: tuple[int, int],
    dtype: str,
    encoding: StoreEncoding,
) -> None:
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    # Clip chunk shape to array dimensions so zarr's declared shape matches what
    # is physically stored — oversized chunks cause zarr to allocate a large
    # decompression buffer even when the array is narrower than chunk_shape[1].
    effective_chunks = (min(chunk_shape[0], z.shape[0]), min(chunk_shape[1], z.shape[1]))
    root = staged.arrays(mode="w")
    # Flat C-order position is exactly `flatnonzero` over the whole grid, which
    # is what the overflow table keys on.
    codec = StoreCodec(encoding)
    overflow = ZOverflowBuilder()
    codes = codec.encode_z(z, positions=np.flatnonzero, overflow=overflow)
    root.create_dataset(
        "z", data=codes, chunks=effective_chunks, compressor=compressor, dtype=codec.z_dtype
    )
    root.create_dataset("se", data=se, chunks=effective_chunks, compressor=compressor, dtype=dtype)
    overflow.table().write(root)
    root.attrs["layout"] = "dense"
    root.attrs["completion_state"] = "observed_only"
    root.attrs["compressor"] = DEFAULT_COMPRESSOR
    root.attrs["chunk_shape"] = list(effective_chunks)
