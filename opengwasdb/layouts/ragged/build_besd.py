"""Build a Ragged Observed-Only Store from BESD files.

``build_ragged_from_besd`` is a thin orchestrator over five private phases,
each a cohesive stage of the build: source reading (ESI/EPI), the canonical
variant axis (with inline hg19→hg38 liftover), Analysis metadata, CSR
ingestion and encoding, and the indexes + manifest output. The split exists so
no single function carries the whole build's branching (issue #130); the
fail-loud rules each phase applies live beside the code that applies them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from opengwasdb.build.liftover import build_liftover_lookup
from opengwasdb.encoding import EncodingMeasurements, StoreEncoding
from opengwasdb.layouts.dense.build import add_hit_counts
from opengwasdb.layouts.ragged.analyses import molecular_analysis
from opengwasdb.layouts.ragged.besd_reader import (
    BESDReader,
    ProbeRecord,
    SnpRecord,
    read_epi,
    read_esi,
)
from opengwasdb.layouts.ragged.top_hits import build_ragged_top_hit_indexes
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRWriter
from opengwasdb.model.analyses import Analysis, write_analysis_records
from opengwasdb.model.enums import (
    AssociationCoverage,
    CompletionState,
    PrimaryStorageLayout,
)
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.store.open import CURRENT_FORMAT_VERSION, OpenGWASDBStore, StagedRelease
from opengwasdb.variants.axis import (
    VARIANT_AXIS_FORMAT,
    VARIANT_TABIX_FILENAME,
    VARIANT_TABLE_FILENAME,
    write_variant_axis,
)
from opengwasdb.variants.normalise import (
    CanonicalVariant,
    VariantNormalisationError,
    chromosome_sort_key,
    normalise_chromosome,
    orient_to_canonical,
)

_REFERENCE_ASSEMBLY = "GRCh38"
# Build names whose coordinates already are (or claim) hg38 — no liftover.
_BUILDS_ALREADY_HG38 = ("hg38", "grch38", "b38", "38")


@dataclass(frozen=True)
class RaggedBuildResult:
    output_path: Path
    n_variants: int
    n_analyses: int
    n_associations: int


def build_ragged_from_besd(
    besd_prefix: str | Path,
    output_path: str | Path,
    *,
    store_id: str,
    release_id: str,
    tissue: str | None = None,
    source_build: str = "hg38",
    overwrite: bool = False,
) -> RaggedBuildResult:
    """Build a Ragged Observed-Only Store from BESD files.

    besd_prefix: path without extension (.esi, .epi, .besd are appended).
    source_build: genome assembly of the input BESD ("hg38" or "hg19").
    When source_build is "hg19", SNP coordinates are lifted over to hg38 inline.
    """
    prefix = Path(besd_prefix)
    out = Path(output_path)
    with OpenGWASDBStore.staging(out, overwrite=overwrite) as staged:
        # Phase 1 — source reading (ESI/EPI); Phase 2 — canonical variant axis
        # (liftover first: a row that fails the lift never reaches the axis,
        # so no association can sit at a wrong coordinate).
        snps, probes = _read_sources(prefix)
        lifted = _lifted_coordinates(snps, source_build)
        variants, rsid_by_alid, esi_to_variant = _canonical_variants(snps, lifted)
        print(f"Canonical variants: {len(variants)} (from {len(snps)} ESI entries)")
        write_variant_axis(staged.path, variants, rsid_by_alid)

        # Phase 3 — Analysis metadata from the EPI records. No `analyses` table
        # (ADR 0034, issue #69): analyses.tsv below is the sole source of
        # truth; the empty index file lets Reference Completion add
        # completion_quality.
        analyses = _analysis_records(probes, tissue)
        staged.index_connection().close()
        # Phase 4 — CSR ingestion and the encoding plan it decides.
        csr, encoding = _ingest_besd(prefix, probes, esi_to_variant, len(variants), staged.path)
        # Phase 5 — indexes and manifest output.
        print("Building top-hit indexes ...")
        build_ragged_top_hit_indexes(staged.path, encoding=encoding)
        print("Writing analyses.tsv ...")
        write_analysis_records(staged.path / "analyses.tsv", add_hit_counts(staged.path, analyses))
        _write_manifest(
            staged,
            store_id,
            release_id,
            n_variants=len(variants),
            n_analyses=len(probes),
            n_associations=csr.n_associations,
            besd_prefix=str(prefix),
            source_build=source_build,
            encoding=encoding,
        )
        result = RaggedBuildResult(out, len(variants), len(probes), csr.n_associations)
        print(
            f"Build complete: {result.n_variants:,} variants, "
            f"{result.n_analyses:,} analyses, "
            f"{result.n_associations:,} associations"
        )
    return result


def _read_sources(prefix: Path) -> tuple[list[SnpRecord], list[ProbeRecord]]:
    """Read the ESI (SNP) and EPI (probe) records of one BESD source set."""
    print(f"Reading ESI: {prefix}.esi")
    snps = read_esi(f"{prefix}.esi")
    print(f"Reading EPI: {prefix}.epi")
    probes = read_epi(f"{prefix}.epi")
    print(f"Loaded {len(snps)} SNPs and {len(probes)} probes")
    return snps, probes


def _lifted_coordinates(
    snps: list[SnpRecord], source_build: str
) -> dict[int, tuple[str, int]] | None:
    """Map ESI row indices to hg38 ``(chromosome, bp)`` when ``source_build``
    is not an hg38 alias.

    Returns ``{esi row_idx: (hg38 chrom, hg38 bp)}`` for the rows the lift
    landed on, or ``None`` when no lift is needed. A row the lift failed is
    omitted and therefore excluded from the store — never stored at a wrong
    coordinate (its absence is loud: the canonical-variant count and the
    per-probe association counts both fall, and the printed liftover report
    says how many failed).
    """
    if source_build.lower().strip() in _BUILDS_ALREADY_HG38:
        return None
    print(f"Lifting over {source_build} → hg38 ...")
    rows = []
    for s in snps:
        if s.a1 is not None and s.a2 is not None:
            rows.append((s.row_idx, s.chromosome, s.bp, s.a1, s.a2))
    lo_lookup = build_liftover_lookup(
        [(chrom, bp, a1, a2) for _row_idx, chrom, bp, a1, a2 in rows],
        from_build=source_build,
        to_build="hg38",
    )
    lifted = {}
    for row_idx, chrom, bp, a1, a2 in rows:
        alid = lo_lookup.get((chrom, bp, a1, a2))
        if alid is None:
            continue
        parts = alid.split(":")
        lifted[row_idx] = (parts[0], int(parts[1]))
    n_failed = len(rows) - len(lifted)
    print(f"Liftover {source_build}→hg38: {n_failed}/{len(snps)} variants failed")
    return lifted


def _usable_esi_position(
    snp: SnpRecord, lifted: dict[int, tuple[str, int]] | None
) -> tuple[tuple[str, int], str, str] | None:
    """The ``((chrom, bp), a1, a2)`` an ESI row is held at — its lifted hg38
    position when a lift was performed, else its own coordinate. ``None`` when
    the row must not enter the axis: no alleles, or a lift that did not land."""
    if snp.a1 is None or snp.a2 is None:
        return None
    position = (snp.chromosome, snp.bp) if lifted is None else lifted.get(snp.row_idx)
    if position is None:
        return None
    return position, snp.a1, snp.a2


def _canonical_variants(
    snps: list[SnpRecord],
    lifted: dict[int, tuple[str, int]] | None,
) -> tuple[list[CanonicalVariant], dict[str, str], dict[int, tuple[int, bool]]]:
    """Deduplicate ESI rows onto a position-ordered canonical variant axis.

    Returns ``(variants, rsid_by_alid, esi_to_variant)`` where
    ``esi_to_variant`` maps each surviving ESI row to ``(variant_index,
    flipped)``. Rows without alleles, unlifted hg19 rows, or rows whose
    alleles fail canonical orientation (``VariantNormalisationError``) are
    excluded from the axis — and therefore from every Analysis, because BESD
    association rows reach the store only through ``esi_to_variant``.
    """
    esi_to_variant: dict[int, tuple[int, bool]] = {}
    variants: list[CanonicalVariant] = []
    rsid_by_alid: dict[str, str] = {}
    alid_to_idx: dict[str, int] = {}
    # (chromosome_sort_key, esi row_idx, variant, flipped, rsid)
    candidate: list[
        tuple[tuple[tuple[int, str], int], int, CanonicalVariant, bool, str | None]
    ] = []

    for snp in snps:
        usable = _usable_esi_position(snp, lifted)
        if usable is None:
            continue
        (chrom, bp), a1, a2 = usable
        try:
            ori = orient_to_canonical(chrom, bp, a1, a2)
        except VariantNormalisationError:
            continue
        sort_key = (chromosome_sort_key(ori.variant.chromosome), ori.variant.position)
        rsid = snp.snp_id if snp.snp_id.startswith("rs") else None
        candidate.append((sort_key, snp.row_idx, ori.variant, ori.flipped, rsid))

    # Sort by genomic position so variant_index is position-ordered.
    candidate.sort(key=lambda x: x[0])

    for _sort_key, esi_row_idx, variant, flipped, rsid in candidate:
        alid = variant.alid
        existing_idx = alid_to_idx.get(alid)
        if existing_idx is not None:
            # Duplicate ALID — map to the same variant_index.
            esi_to_variant[esi_row_idx] = (existing_idx, flipped)
            continue
        variant_index = len(variants)
        alid_to_idx[alid] = variant_index
        variants.append(variant)
        esi_to_variant[esi_row_idx] = (variant_index, flipped)
        if rsid:
            rsid_by_alid[alid] = rsid
    return variants, rsid_by_alid, esi_to_variant


def _analysis_records(
    probes: list[ProbeRecord], tissue: str | None
) -> list[Analysis]:
    """One Analysis per EPI probe (issue #69's shared schema).

    assigned_ancestry (ADR 0028): BESD/ESI/EPI bulk-QTL sources carry no
    per-analysis ancestry assignment, unlike build_ssf's manifest-driven
    sources, so it stays at molecular_analysis's blank default here.
    """
    analyses: list[Analysis] = []
    for probe in probes:
        try:
            probe_chr = normalise_chromosome(probe.chromosome)
        except VariantNormalisationError:
            probe_chr = None

        analysis_id = probe.probe_id
        if tissue:
            analysis_id = f"{probe.probe_id}::{tissue}"

        is_ensembl = probe.probe_id.startswith("ENSG")
        analyses.append(
            molecular_analysis(
                analysis_id,
                analysis_label=probe.gene,
                trait_ontology_id=f"ENSEMBL:{probe.probe_id}" if is_ensembl else None,
                trait_ontology_label="Ensembl" if is_ensembl else None,
                tissue=tissue,
                context=None,
                trait_chr=probe_chr,
                trait_bp=probe.probe_bp if probe.probe_bp > 0 else None,
                n=None,
            )
        )
    return analyses


def _add_empty_analysis(csr: RaggedCSRWriter) -> None:
    """Record an Analysis with no associations at all (a source with no rows,
    or whose every row was filtered out)."""
    csr.add_analysis(
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.float16),
    )


def _map_probe_rows(
    raw_snp_idx: np.ndarray,
    betas: np.ndarray,
    ses: np.ndarray,
    esi_to_variant: dict[int, tuple[int, bool]],
) -> tuple[list[int], list[float], list[float]]:
    """Map one probe's raw BESD rows onto ``(variant_index, z, se)`` lists.

    A row whose ESI index never made the variant axis, or whose statistics
    are unusable (``se <= 0``, a non-finite beta or se), is dropped for that
    Analysis: the store must not carry a z-score the source did not vouch
    for. Rows that survive are oriented — ``z`` flips sign when the stored
    canonical allele is the flipped one.
    """
    vi_list: list[int] = []
    z_list: list[float] = []
    se_list: list[float] = []
    for esi_idx, beta, se in zip(
        raw_snp_idx.tolist(), betas.tolist(), ses.tolist(), strict=True
    ):
        mapping = esi_to_variant.get(int(esi_idx))
        if mapping is None:
            continue
        variant_index, flipped = mapping
        if se <= 0 or not np.isfinite(beta) or not np.isfinite(se):
            continue
        z = beta / se
        if flipped:
            z = -z
        vi_list.append(variant_index)
        z_list.append(z)
        se_list.append(se)
    return vi_list, z_list, se_list


def _decide_encoding(csr: RaggedCSRWriter, n_analyses: int) -> StoreEncoding:
    """One encoding plan per build, decided from what the build holds
    (ADR 0037, issue #119). BESD carries no frequencies, so this settles on
    ``eaf: absent`` — stated by the plan rather than left to be inferred from
    a missing array.
    """
    eaf_measurements = csr.eaf_measurements()
    preliminary = StoreEncoding.decide(
        EncodingMeasurements(n_analyses=n_analyses, eaf=eaf_measurements)
    )
    return StoreEncoding.decide(
        EncodingMeasurements(
            n_analyses=n_analyses,
            eaf=eaf_measurements,
            se=csr.se_measurements(preliminary),
        )
    )


def _ingest_besd(
    prefix: Path,
    probes: list[ProbeRecord],
    esi_to_variant: dict[int, tuple[int, bool]],
    n_variants: int,
    staged_path: Path,
) -> tuple[RaggedCSRWriter, StoreEncoding]:
    """Stream the BESD file into a CSR store under ``staged_path``.

    Returns the writer (for association counts) and the encoding plan the
    flush was performed under. Associations are sorted by variant_index per
    Analysis (every builder sorts before writing), so readers can resolve
    ``(variant, analysis)`` pairs with one ``searchsorted`` per Analysis.
    """
    print(f"Reading BESD: {prefix}.besd")
    besd = BESDReader(f"{prefix}.besd", len(probes))
    print(f"BESD format: SPARSE_FILE_TYPE_{besd.format_type}")

    csr = RaggedCSRWriter(n_variants)
    skipped_probes = 0

    for probe in probes:
        raw_snp_idx, betas, ses = besd.get_probe_associations(probe.row_idx)

        if len(raw_snp_idx) == 0:
            _add_empty_analysis(csr)
            continue

        vi_list, z_list, se_list = _map_probe_rows(
            raw_snp_idx, betas, ses, esi_to_variant
        )

        if not vi_list:
            skipped_probes += 1
            _add_empty_analysis(csr)
            continue

        # Sort by variant_index for consistent ordering within each analysis.
        order = np.argsort(vi_list)
        csr.add_analysis(
            np.array(vi_list, dtype=np.int32)[order],
            np.array(z_list, dtype=np.float32)[order],
            np.array(se_list, dtype=np.float32)[order],
        )

        if (probe.row_idx + 1) % 1000 == 0:
            print(f"  Processed {probe.row_idx + 1} / {len(probes)} probes")

    encoding = _decide_encoding(csr, len(probes))
    print(f"Flushing zarr CSR ({csr.n_associations:,} associations) ...")
    csr.flush(staged_path, encoding)

    if skipped_probes:
        print(f"  {skipped_probes} probes had no valid associations after filtering")
    return csr, encoding


def _write_manifest(
    staged: StagedRelease,
    store_id: str,
    release_id: str,
    *,
    n_variants: int,
    n_analyses: int,
    n_associations: int,
    besd_prefix: str,
    encoding: StoreEncoding,
    source_build: str = "hg38",
) -> None:
    manifest = StoreManifest(
        encoding=encoding,
        store_id=store_id,
        release_id=release_id,
        format_version=CURRENT_FORMAT_VERSION,
        primary_layout=PrimaryStorageLayout.RAGGED,
        association_coverage=AssociationCoverage.CIS_AND_SIGNALS,
        completion_state=CompletionState.OBSERVED_ONLY,
        reference_assembly=_REFERENCE_ASSEMBLY,
        created_at=datetime.now(UTC).isoformat(),
        provenance={
            "builder": "opengwasdb.v0.1_ragged_observed_besd",
            "source_besd_prefix": besd_prefix,
            "source_build": source_build,
            "n_variants": n_variants,
            "n_analyses": n_analyses,
            "n_associations": n_associations,
            "ragged": {
                "statistic_arrays": ["z", "se"],
                "se_dtype": encoding.se.dtype,
                "variant_axis": {
                    "format": VARIANT_AXIS_FORMAT,
                    "table": VARIANT_TABLE_FILENAME,
                    "tabix_index": VARIANT_TABIX_FILENAME,
                },
            },
        },
    )
    staged.write_manifest(manifest)
