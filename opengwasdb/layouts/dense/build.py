"""Dense Observed-Only writer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from opengwasdb import admission
from opengwasdb.admission.rules import AnalysisRules
from opengwasdb.encoding import (
    EncodingMeasurements,
    StoreEncoding,
    eaf_baseline_from_grid,
    fit_se_grid,
    measure_eaf_grid,
)
from opengwasdb.encoding.association_codec import (
    AssociationCodec,
    CellValues,
    Placement,
    log_se_out_of_range,
)
from opengwasdb.encoding.planes import write_encoded_planes
from opengwasdb.index import initialise_schema, set_metadata
from opengwasdb.layouts.build_manifest import apply_se_divisor
from opengwasdb.layouts.dense.constants import (
    DEFAULT_CHUNK_SHAPE,
    DEFAULT_DTYPE,
    dense_index_metadata,
    dense_layout_records,
    dense_provenance_block,
)
from opengwasdb.layouts.dense.top_hits import build_top_hit_indexes
from opengwasdb.layouts.release_manifest import observed_only_release_manifest
from opengwasdb.model.analyses import Analysis, analyses_table_from_records, write_analyses
from opengwasdb.model.enums import (
    AssociationCoverage,
    EafOrientationOutcome,
    EafScope,
    PrimaryStorageLayout,
    coverage_for_layout,
)
from opengwasdb.readers.interface import StreamTally
from opengwasdb.readers.rows import ReaderBatch
from opengwasdb.store.arrays import ArrayRole, compressor
from opengwasdb.store.open import OpenGWASDBStore, StagedRelease
from opengwasdb.top_hits import TOP_HIT_THRESHOLDS, add_hit_counts
from opengwasdb.variants import (
    VARIANT_AXIS_FORMAT,
    VARIANT_OFFSETS_FILENAME,
    VARIANT_TABIX_FILENAME,
    VARIANT_TABLE_FILENAME,
    CanonicalVariant,
    chromosome_sort_key,
    write_variant_axis,
)
from opengwasdb.variants.routing import collapse_duplicates


@dataclass(frozen=True)
class DenseBuildResult:
    """Paths and dimensions for a built Dense Observed-Only store."""

    output_path: Path
    n_variants: int
    n_analyses: int


def write_analyses_tsv(output_path: Path, analyses: list[Analysis]) -> Path:
    """Write ``analyses.tsv`` + ``overview.html`` -- the sole source of truth
    for Analytical Metadata (ADR 0030, issue #22; unified schema, ADR 0034),
    replacing the old SQLite ``analyses`` table and its own generated
    human-browsable rendering.
    """
    output_path = Path(output_path)
    table = analyses_table_from_records(analyses)
    path = write_analyses(output_path / "analyses.tsv", table)
    # Imported here rather than at module scope: `opengwasdb.overview` reads
    # Indexed Variant Subsets through `layouts.dense.indexed_subsets`, which
    # runs this package's `__init__`, which imports this module -- a
    # module-scope import of `write_overview_html` closes that cycle (#57).
    from opengwasdb.overview import write_overview_html

    write_overview_html(output_path, table)
    return path


@dataclass(frozen=True)
class NormalisedAssociation:
    """One association of one Analysis, as the build holds it between reading and
    writing: the canonical variant, and the statistics on its A1."""

    analysis_id: str
    variant: CanonicalVariant
    z: float
    se: float
    rsid: str | None = None
    eaf: float | None = None


@dataclass(frozen=True)
class AnalysisSource:
    """One Analysis as a build reads it: what its manifest row declares, and what
    its Source Reader yielded.

    ``analysis`` is the row's Analytical and Attribution Metadata, ``rules`` its
    declared INFO and MAF filters (D2), and ``se_divisor`` the phenotype SD that
    standardises its standard errors (#18). ``tally`` names the rows the reader
    left out as malformed (#365) and the frequencies it made absent (D7).
    """

    analysis: Analysis
    rules: AnalysisRules
    se_divisor: float
    batches: Sequence[ReaderBatch]
    tally: StreamTally


def build_dense_observed_store(
    sources: Sequence[AnalysisSource],
    output_path: str | Path,
    *,
    store_id: str,
    release_id: str,
    reference_assembly: str,
    chunk_shape: tuple[int, int] = DEFAULT_CHUNK_SHAPE,
    dtype: str = DEFAULT_DTYPE,
    overwrite: bool = False,
    association_coverage: AssociationCoverage | str = AssociationCoverage.GWAS,
    admission_record: str | Path | None = None,
) -> DenseBuildResult:
    """Write a Dense Observed-Only Store Release from its Analyses' associations.

    The Store's columns are the ``sources``, in order. ``association_coverage``
    is a build input (issue #298), and a Dense matrix can carry only ``gwas``:
    every cell it holds is a (variant, Analysis) the source either reported or
    did not, so a narrower coverage has no representation here
    (`coverage_for_layout`, ADR 0064). Each Analysis's records pass its
    Admission gate, which writes ``admission_record`` (beside the Store unless
    named, #357), and logs there the rows its reader left out as malformed
    (#365).
    """
    coverage = coverage_for_layout(association_coverage, PrimaryStorageLayout.DENSE)

    _refuse_unbuildable(sources)

    out = Path(output_path)
    with OpenGWASDBStore.staging(out, overwrite=overwrite) as staged:
        directory = admission.admission_record_for(out, admission_record)
        records = _admitted_records(sources, directory)
        variants = _collect_variants(records)
        analyses = _collect_analyses(records, sources)
        z, se, eaf = _scatter_records(records, variants, analyses)

        # The encoding plan is decided once, here, and read back from the
        # manifest by everything downstream (ADR 0037, issue #119).
        encoding, coefficients = _decide_encoding(z, se, eaf, chunk_shape)
        rsid_by_alid = _first_rsids_by_alid(records)
        _write_manifest(
            staged,
            store_id,
            release_id,
            reference_assembly,
            records,
            (len(variants), len(analyses)),
            chunk_shape,
            dtype,
            encoding,
            coverage,
        )
        write_variant_axis(staged.path, variants, rsid_by_alid)
        _write_index(staged, variants, analyses, records, chunk_shape)
        _write_zarr(staged, (z, se, eaf), coefficients, chunk_shape, encoding, analyses)
        build_top_hit_indexes(staged.path, encoding=encoding)
        write_analyses_tsv(staged.path, add_hit_counts(staged.path, analyses))
        return DenseBuildResult(output_path=out, n_variants=len(variants), n_analyses=len(analyses))


def _refuse_unbuildable(sources: Sequence[AnalysisSource]) -> None:
    """Refuse an Analysis named twice, and a build with no association at all."""
    ids = [source.analysis.analysis_id for source in sources]
    if len(set(ids)) != len(ids):
        repeated = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"Analyses named more than once: {', '.join(repeated)}")
    if not any(batch for source in sources for batch in source.batches):
        raise ValueError("cannot build a store with no association records")


def _records_of(source: AnalysisSource) -> list[NormalisedAssociation]:
    """One Analysis's associations, in the order its reader yielded them."""
    analysis_id = source.analysis.analysis_id
    records = []
    for batch in source.batches:
        for alid, z, se, eaf, rsid in zip(
            batch.alid.tolist(), batch.z.tolist(), batch.se.tolist(),
            batch.eaf.tolist(), batch.rsid.tolist(), strict=True,
        ):
            chromosome, position, a1, a2 = alid.split(":")
            records.append(
                NormalisedAssociation(
                    analysis_id=analysis_id,
                    variant=CanonicalVariant(chromosome, int(position), a1, a2),
                    z=z,
                    se=se,
                    rsid=rsid or None,
                    eaf=None if np.isnan(eaf) else eaf,
                )
            )
    return records


def _admitted_records(
    sources: Sequence[AnalysisSource], directory: Path
) -> list[NormalisedAssociation]:
    """The records each Analysis's Admission gate keeps, in source order, with
    each Analysis's standard errors standardised (#18).

    Analyses are judged in the order of ``sources``, which is the order the
    Store's columns take (`_collect_analyses`), each under the filters its
    manifest row declares (D2). An Analysis's records for one variant resolve as
    routing resolves an axis's (`collapse_duplicates`): agreeing ones collapse
    to the first, conflicting ones are all dropped and logged.
    """
    records, positions = _gathered(sources)
    record_ = admission.AdmissionRecord.open(
        directory, [source.analysis.analysis_id for source in sources], command="build-dense"
    )
    keep = np.zeros(len(records), dtype=bool)
    for source, at in zip(sources, positions, strict=True):
        with record_.gate(source.rules) as gate:
            gate.streamed(source.tally)
            if at:
                keep[at] = _admitted(records, at, gate)
    record_.finish()
    kept = [record for record, judged in zip(records, keep.tolist(), strict=True) if judged]
    return _standardised(kept, sources)


def _gathered(
    sources: Sequence[AnalysisSource],
) -> tuple[list[NormalisedAssociation], list[list[int]]]:
    """Every Analysis's records in ``sources`` order, and where each Analysis's are."""
    records: list[NormalisedAssociation] = []
    positions: list[list[int]] = []
    for source in sources:
        own = _records_of(source)
        positions.append(list(range(len(records), len(records) + len(own))))
        records.extend(own)
    return records, positions


def _standardised(
    records: list[NormalisedAssociation], sources: Sequence[AnalysisSource]
) -> list[NormalisedAssociation]:
    """``records`` with each Analysis's standard errors divided by its phenotype SD (#18)."""
    divisor = {s.analysis.analysis_id: s.se_divisor for s in sources}
    return [
        record if divisor[record.analysis_id] == 1.0
        else replace(
            record,
            se=float(
                apply_se_divisor(
                    np.array([record.se], dtype=np.float32), divisor[record.analysis_id]
                )[0]
            ),
        )
        for record in records
    ]


def _admitted(
    records: list[NormalisedAssociation], positions: list[int], gate: admission.Gate
) -> np.ndarray:
    """Which of one Analysis's records, at ``positions``, its gate and its
    duplicates' resolution keep."""
    keep = gate.source_rows(_source_rows([records[i] for i in positions]))
    kept = [position for position, judged in zip(positions, keep.tolist(), strict=True) if judged]
    keep[keep] = _resolved_duplicates([records[i] for i in kept], gate)
    return keep


def _resolved_duplicates(records: list[NormalisedAssociation], gate: admission.Gate) -> np.ndarray:
    """Which of one Analysis's admitted records survive its duplicates' resolution."""
    number: dict[str, int] = {}
    codes = np.array(
        [number.setdefault(r.variant.alid, len(number)) for r in records], dtype=np.int64
    )
    found = collapse_duplicates(
        codes,
        np.array([r.z for r in records], dtype=np.float64),
        np.array([r.se for r in records], dtype=np.float64),
    )
    gate.duplicates([records[i].variant.alid for i in found.dropped.tolist()], found.collapsed)
    keep = np.zeros(len(records), dtype=bool)
    keep[found.keep] = True
    return keep


def _source_rows(records: list[NormalisedAssociation]) -> admission.SourceRows:
    """One Analysis's records as the batch its gate judges, each named by its ALID."""
    alids = [record.variant.alid for record in records]
    return admission.SourceRows(
        eaf=np.array([np.nan if r.eaf is None else r.eaf for r in records], dtype=np.float64),
        se=np.array([r.se for r in records], dtype=np.float64),
        z=np.array([r.z for r in records], dtype=np.float64),
        keys_at=lambda at: [alids[i] for i in at.tolist()],
    )


def _scatter_records(
    records: list[NormalisedAssociation],
    variants: list[CanonicalVariant],
    analyses: list[Analysis],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lay the records out as three parallel `float32` grids.

    `z`, `se` and `eaf` are all accumulated at full precision and quantised
    once, by the codec, on the way to disk — never pre-rounded into a stored
    dtype here, which would make an "exact" exception a rounded one.
    """
    variant_index = {variant.alid: i for i, variant in enumerate(variants)}
    analysis_index = {analysis.analysis_id: i for i, analysis in enumerate(analyses)}
    shape = (len(variants), len(analyses))
    z, se, eaf = (np.full(shape, np.nan, dtype=np.float32) for _ in range(3))
    seen_cells: set[tuple[int, int]] = set()
    for record in records:
        row = variant_index[record.variant.alid]
        col = analysis_index[record.analysis_id]
        if (row, col) in seen_cells:
            raise ValueError(
                f"duplicate association for variant {record.variant.alid} "
                f"and analysis {record.analysis_id}"
            )
        seen_cells.add((row, col))
        z[row, col] = record.z
        se[row, col] = record.se
        if record.eaf is not None:
            eaf[row, col] = record.eaf
    return z, se, eaf


def _decide_encoding(
    z: np.ndarray, se: np.ndarray, eaf: np.ndarray, chunk_shape: tuple[int, int]
) -> tuple[StoreEncoding, np.ndarray]:
    """Measure both planes and settle the store's plan, in the required order.

    `se` is fitted against *decoded* EAF, not the source's: the residual coding
    has to predict from the frequencies a reader will actually get back, so the
    EAF plan is decided and round-tripped first (ADR 0037 §3).
    """
    n_analyses = eaf.shape[1]
    eaf_measured = measure_eaf_grid(eaf)
    preliminary = StoreEncoding.decide(
        EncodingMeasurements(n_analyses=n_analyses, eaf=eaf_measured)
    )
    baseline = eaf_baseline_from_grid(eaf) if preliminary.eaf.is_residual else None
    decoded_eaf = (
        AssociationCodec(preliminary, eaf_baseline=baseline)
        .round_trip(CellValues(z, se, eaf), Placement.band(0, *eaf.shape), where="build-dense")
        .eaf
    )
    coefficients, se_measured = fit_se_grid(
        se,
        decoded_eaf,
        compressor=compressor(),
        chunks=(min(chunk_shape[0], se.shape[0]), min(chunk_shape[1], se.shape[1])),
    )
    return (
        StoreEncoding.decide(
            EncodingMeasurements(n_analyses=n_analyses, eaf=eaf_measured, se=se_measured)
        ),
        coefficients,
    )


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


def _eaf_columns(reports_frequencies: bool) -> tuple[str, str]:
    """`eaf_scope` and `eaf_orientation` for one Analysis of a general source.

    `unverified`, never `passed`: this builder takes records straight from a
    source and has no reference panel to check an orientation against, and a
    frequency column that has never been checked is indistinguishable from one
    reported against the other allele (spec §9.1, issue #115).
    """
    if not reports_frequencies:
        return EafScope.ABSENT.value, ""
    return EafScope.ASSOCIATION.value, EafOrientationOutcome.UNVERIFIED.value


def _collect_analyses(
    records: list[NormalisedAssociation], sources: Sequence[AnalysisSource]
) -> list[Analysis]:
    """The Analyses with a kept record, in ``sources`` order, each with the
    frequency scope of its records."""
    with_eaf = {record.analysis_id for record in records if record.eaf is not None}
    kept = {record.analysis_id for record in records}
    analyses = []
    for source in sources:
        analysis_id = source.analysis.analysis_id
        if analysis_id in kept:
            eaf_scope, eaf_orientation = _eaf_columns(analysis_id in with_eaf)
            analyses.append(
                replace(source.analysis, eaf_scope=eaf_scope, eaf_orientation=eaf_orientation)
            )
    return analyses


def _write_manifest(
    staged: StagedRelease,
    store_id: str,
    release_id: str,
    reference_assembly: str,
    records: list[NormalisedAssociation],
    shape: tuple[int, int],
    chunk_shape: tuple[int, int],
    dtype: str,
    encoding: StoreEncoding,
    association_coverage: AssociationCoverage,
) -> None:
    manifest = observed_only_release_manifest(
        PrimaryStorageLayout.DENSE,
        store_id=store_id,
        release_id=release_id,
        encoding=encoding,
        association_coverage=association_coverage,
        reference_assembly=reference_assembly,
        provenance={
            "builder": "opengwasdb.v0.1_dense_observed",
            "source_record_count": len(records),
            "eaf_orientation": {
                "analyses": [
                    {
                        "analysis_id": analysis_id,
                        "outcome": EafOrientationOutcome.UNVERIFIED.value,
                        "note": "general Dense source supplied no orientation reference",
                    }
                    for analysis_id in sorted(
                        {record.analysis_id for record in records if record.eaf is not None}
                    )
                ]
            },
            "dense": {
                **dense_provenance_block(shape, hint=chunk_shape, se_dtype=encoding.se.dtype),
                "top_hit_thresholds": list(TOP_HIT_THRESHOLDS),
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
) -> None:
    with staged.index_connection() as connection:
        initialise_schema(connection)
        set_metadata(connection, "schema_version", 2)
        set_metadata(connection, "n_variants", len(variants))
        set_metadata(connection, "n_analyses", len(analyses))
        set_metadata(
            connection,
            "dense",
            dense_index_metadata(
                (len(variants), len(analyses)),
                hint=chunk_shape,
                variant_axis={
                    "format": VARIANT_AXIS_FORMAT,
                    "table": VARIANT_TABLE_FILENAME,
                    "tabix_index": VARIANT_TABIX_FILENAME,
                    "row_offsets": VARIANT_OFFSETS_FILENAME,
                },
            ),
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
    grids: tuple[np.ndarray, np.ndarray, np.ndarray],
    se_coefficients: np.ndarray,
    chunk_shape: tuple[int, int],
    encoding: StoreEncoding,
    analyses: list[Analysis],
) -> None:
    """Encode the `(z, se, eaf)` grid once, write each plane with its side
    arrays, and log each Analysis's SEs above `float16`'s range."""
    z, se, eaf = grids
    root = staged.arrays(mode="w")
    baseline = eaf_baseline_from_grid(eaf) if encoding.eaf.is_residual else None
    coefficients = se_coefficients if encoding.se.is_residual else None
    codec = AssociationCodec(encoding, eaf_baseline=baseline, se_coefficients=coefficients)
    encoded = codec.encode(
        CellValues(z, se, None if encoding.eaf.is_absent else eaf),
        Placement.band(0, *z.shape),
        where="build-dense",
    )
    write_encoded_planes(
        root,
        encoding,
        encoded,
        role=ArrayRole.DENSE_STATISTIC_PLANE,
        compressor=compressor(),
        chunks=chunk_shape,
        eaf_baseline=baseline,
        se_coefficients=coefficients,
    )
    root.attrs["layout"] = "dense"
    root.attrs["completion_state"] = "observed_only"
    root.attrs.update(dense_layout_records(z.shape, hint=chunk_shape))
    log_se_out_of_range(
        encoded.storable.se_out_of_range,
        [analysis.analysis_id for analysis in analyses],
        where="build-dense",
    )
