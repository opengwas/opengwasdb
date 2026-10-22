"""Build a Ragged Observed-Only Store from BESD files.

``build_ragged_from_besd`` is a thin orchestrator over five private phases,
each a cohesive stage of the build: source reading (ESI/EPI), the canonical
variant axis (with inline hg19→hg38 liftover), Analysis metadata, CSR
ingestion and encoding, and the indexes + manifest output. The split exists so
no single function carries the whole build's branching (issue #130); the
fail-loud rules each phase applies live beside the code that applies them.
"""

from __future__ import annotations

import csv
import logging
import math
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from opengwasdb.build.liftover import (
    LiftedRow,
    lift_hg19_to_hg38,
    normalise_build,
    require_lifted_alleles_match_reference,
)
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
from opengwasdb.layouts.ragged.by_variant import build_variant_index
from opengwasdb.layouts.ragged.provenance import ragged_provenance
from opengwasdb.layouts.ragged.top_hits import build_ragged_top_hit_indexes
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRWriter
from opengwasdb.model.analyses import Analysis, PassthroughMetadata, write_analysis_records
from opengwasdb.model.enums import (
    AssociationCoverage,
    CompletionState,
    PrimaryStorageLayout,
)
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.model.manifest_columns import manifest_column
from opengwasdb.store.open import CURRENT_FORMAT_VERSION, OpenGWASDBStore, StagedRelease
from opengwasdb.variants import (
    CanonicalVariant,
    Orientation,
    VariantNormalisationError,
    chromosome_sort_key,
    normalise_chromosome,
    orient_to_canonical,
    write_variant_axis,
)

log = logging.getLogger(__name__)

_REFERENCE_ASSEMBLY = "GRCh38"
# Build names whose coordinates already are (or claim) hg38 — no liftover.
_BUILDS_ALREADY_HG38 = ("hg38", "grch38", "b38", "38")
#: The fraction of a lift's rows that may fail before the build stops: the
#: default `build_liftover_lookup` applies, now applied to the richer per-row
#: lift this builder needs for its strand.
_LIFTOVER_FAILURE_THRESHOLD = 0.01


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
    analyses_path: str | Path | None = None,
    tissue: str | None = None,
    source_build: str = "hg38",
    overwrite: bool = False,
    chain_file: str | Path | None = None,
    variant_reference: str | Path | None = None,
) -> RaggedBuildResult:
    """Build a Ragged Observed-Only Store from BESD files.

    besd_prefix: path without extension (.esi, .epi, .besd are appended).
    source_build: input assembly, "hg38" or "hg19" (hg19 is lifted inline).
    chain_file: the hg19→hg38 chain to lift by; the downloaded UCSC chain for
        the build pair when omitted. The resolver requires one explicitly, so a
        pilot can name the same chain its Resolution Records were made with.
    variant_reference: a Variant Reference on the target assembly -- a plain
        ALID list, a ``*.variant-ref.tsv.gz`` artifact, or a Store Variant
        Table -- that every lifted variant's alleles are checked against. A
        disagreement fails the build naming the variant (issue #284); with none
        supplied the lift is recorded as unchecked, and the build warns.
    When analyses_path is supplied, Analytical and Attribution Metadata are overlaid
    onto the EPI-derived analyses while keeping BESD coordinates authoritative (issue #173).
    """
    prefix = Path(besd_prefix)
    out = Path(output_path)
    with OpenGWASDBStore.staging(out, overwrite=overwrite) as staged:
        # Phases 1-2 — source reading and the canonical variant axis.
        axis = _esi_axis(
            prefix, source_build, chain_file=chain_file, variant_reference=variant_reference
        )
        write_variant_axis(staged.path, axis.variants, axis.rsid_by_alid)
        # Phase 3 — Analysis metadata from the EPI records (with optional manifest overlay).
        analyses = _analysis_records(axis.probes, tissue, analyses_path)
        staged.index_connection().close()
        # Phase 4 — CSR ingestion and the encoding plan it decides.
        csr, encoding = _ingest_besd(
            prefix, axis.probes, axis.esi_to_variant, len(axis.variants), staged.path,
        )
        # Phase 5 — indexes and manifest output.
        _write_besd_outputs(staged, analyses, encoding)
        _write_manifest(
            staged,
            store_id,
            release_id,
            n_variants=len(axis.variants),
            n_analyses=len(axis.probes),
            n_associations=csr.n_associations,
            besd_prefix=str(prefix),
            source_build=source_build,
            encoding=encoding,
            lift=axis.lift,
            collisions=axis.collisions,
        )
        result = _summarise(out, axis, csr.n_associations)
    return result


def _summarise(out: Path, axis: _EsiAxis, n_associations: int) -> RaggedBuildResult:
    """The build's totals, printed once and returned to the caller."""
    result = RaggedBuildResult(out, len(axis.variants), len(axis.probes), n_associations)
    print(
        f"Build complete: {result.n_variants:,} variants, "
        f"{result.n_analyses:,} analyses, "
        f"{result.n_associations:,} associations"
    )
    return result


@dataclass(frozen=True)
class _VariantCollisions:
    """ESI entries dropped because another entry claims their canonical variant.

    Two entries that lift onto one hg38 variant are ambiguous, and every entry
    in the group is dropped rather than one of them kept, so the count is of
    *entries*: a group of three drops three. `n_groups` is the number of
    canonical variants that were contested, so `n_entries - n_groups` is how
    many more entries the source has than variants the axis can carry -- the
    difference a build log's "canonical variants (from N ESI entries)" line
    shows.
    """

    n_entries: int
    n_groups: int
    example: str | None

    def __bool__(self) -> bool:
        return self.n_entries > 0

    def report(self) -> None:
        """Say on stdout what was dropped, and that it was dropped on purpose."""
        if not self:
            return
        print(
            f"Variant collisions: {self.n_entries} ESI entries in {self.n_groups} "
            f"group(s) claim a canonical variant another entry also claims "
            f"(e.g. {self.example}); every entry in each group is dropped as "
            f"ambiguous, with its associations"
        )

    def record(self, lift: dict[str, object] | None) -> dict[str, object] | None:
        """The `provenance.lift` block, with this build's collision counts in it.

        Beside `n_lifted`, because on a cross-assembly build the collision *is*
        a lift outcome. A same-assembly build has no lift block to describe, and
        inventing one would say a lift happened; `_write_manifest` records the
        same two fields at the top level there instead.
        """
        if lift is None or not self:
            return lift
        lift["n_collisions"] = self.n_entries
        lift["n_collision_groups"] = self.n_groups
        return lift


@dataclass(frozen=True)
class _CanonicalAxis:
    """Phase 2 of a BESD build: the ESI rows that make the variant axis.

    ``esi_to_variant`` maps each surviving ESI row to ``(variant_index,
    flipped)``; a row absent from it reaches no Analysis, because BESD
    association rows enter the store only through this map.
    """

    variants: list[CanonicalVariant]
    rsid_by_alid: dict[str, str]
    esi_to_variant: dict[int, tuple[int, bool]]
    collisions: _VariantCollisions


@dataclass(frozen=True)
class _EsiAxis:
    """Phases 1-2 of a BESD build: the source rows and the canonical axis.

    The lift runs before the axis exists, so a row that fails it never reaches
    the axis and no association can sit at a wrong coordinate.
    """

    probes: list[ProbeRecord]
    variants: list[CanonicalVariant]
    rsid_by_alid: dict[str, str]
    esi_to_variant: dict[int, tuple[int, bool]]
    collisions: _VariantCollisions
    lift: dict[str, object] | None


def _esi_axis(
    prefix: Path,
    source_build: str,
    *,
    chain_file: str | Path | None,
    variant_reference: str | Path | None,
) -> _EsiAxis:
    """Read one file set's `.esi`/`.epi` and canonicalise the ESI rows."""
    snps, probes = _read_sources(prefix)
    lifted, lift_provenance = _lifted_coordinates(
        snps, source_build, chain_file=chain_file, variant_reference=variant_reference
    )
    canonical = _canonical_variants(snps, lifted)
    canonical.collisions.report()
    print(f"Canonical variants: {len(canonical.variants)} (from {len(snps)} ESI entries)")
    lift_provenance = canonical.collisions.record(lift_provenance)
    return _EsiAxis(
        probes,
        canonical.variants,
        canonical.rsid_by_alid,
        canonical.esi_to_variant,
        canonical.collisions,
        lift_provenance,
    )


def _write_besd_outputs(
    staged: StagedRelease, analyses: list[Analysis], encoding: StoreEncoding
) -> None:
    """The top-hit indexes and `analyses.tsv` a BESD build finishes with."""
    print("Building top-hit indexes ...")
    build_ragged_top_hit_indexes(staged.path, encoding=encoding)
    print("Writing analyses.tsv ...")
    write_analysis_records(staged.path / "analyses.tsv", add_hit_counts(staged.path, analyses))


def _read_sources(prefix: Path) -> tuple[list[SnpRecord], list[ProbeRecord]]:
    """Read the ESI (SNP) and EPI (probe) records of one BESD source set."""
    print(f"Reading ESI: {prefix}.esi")
    snps = read_esi(f"{prefix}.esi")
    print(f"Reading EPI: {prefix}.epi")
    probes = read_epi(f"{prefix}.epi")
    print(f"Loaded {len(snps)} SNPs and {len(probes)} probes")
    return snps, probes


def _lifted_coordinates(
    snps: list[SnpRecord],
    source_build: str,
    *,
    chain_file: str | Path | None = None,
    variant_reference: str | Path | None = None,
) -> tuple[dict[int, Orientation] | None, dict[str, object] | None]:
    """Lift every ESI row and return its orientation plus the lift's record.

    Returns ``({esi row_idx: Orientation}, provenance)`` for the rows the lift
    landed on, or ``(None, None)`` when no lift is needed. The orientation is
    the resolver's for the same row: the alleles are projected onto the
    chain block's strand, so a reverse-strand SNP's ALID is the complemented
    variant rather than an hg38 position carrying hg19-strand alleles
    (issue #284), and the flip a probe's z-score is signed by is measured
    against the same pair.

    A row the lift failed is omitted and therefore excluded from the store --
    never stored at a wrong coordinate (its absence is loud: the
    canonical-variant count and the per-probe association counts both fall,
    and the printed liftover report says how many failed).

    ``variant_reference`` makes the lift checkable: every lifted variant the
    reference holds at its hg38 position must carry the same alleles, and a
    disagreement raises `LiftedAlleleMismatchError` naming the variant. With
    no reference the build says so loudly -- and records `reference: null` in
    provenance -- rather than letting an unchecked lift look like a checked one.
    """
    if source_build.lower().strip() in _BUILDS_ALREADY_HG38:
        return None, None
    # Refused before any lift runs: this builder knows one cross-build direction
    # (hg19→hg38), and an unrecognised build must not be silently lifted as hg19.
    normalise_build(source_build)
    print(f"Lifting over {source_build} → hg38 ...")
    rows = [
        (s.row_idx, s.chromosome, s.bp, s.a1, s.a2)
        for s in snps
        if s.a1 is not None and s.a2 is not None
    ]
    lifted_by_source = lift_hg19_to_hg38(
        [(chrom, bp, a1, a2) for _row_idx, chrom, bp, a1, a2 in rows],
        chain_file=chain_file,
        failure_threshold=_LIFTOVER_FAILURE_THRESHOLD,
    )
    lifted = _orient_lifted_rows(rows, lifted_by_source)
    print(
        f"Liftover {source_build}→hg38: {len(rows) - len(lifted)}/{len(snps)} "
        f"variants failed"
    )
    return lifted, _lift_provenance(
        source_build, chain_file, variant_reference, lifted_by_source
    )


def _orient_lifted_rows(
    rows: list[tuple[int, str, int, str, str]],
    lifted_by_source: dict[tuple[str, int, str, str], LiftedRow],
) -> dict[int, Orientation]:
    """One canonical orientation per ESI row the lift produced.

    A row whose projected alleles will not canonicalise (identical or
    unsupported alleles) is left out, as it was before #284: it can carry no
    variant, so it can carry no association either.
    """
    lifted: dict[int, Orientation] = {}
    for row_idx, chrom, bp, a1, a2 in rows:
        row = lifted_by_source.get((chrom, bp, a1, a2))
        if row is None:
            continue
        try:
            lifted[row_idx] = _orientation_of(row)
        except VariantNormalisationError:
            continue
    return lifted


def _orientation_of(row: LiftedRow) -> Orientation:
    """The canonical orientation of one lifted row, its alleles in source order.

    `LiftedRow` keeps the source's `(ref, alt)` order rather than the canonical
    one precisely so `orient_to_canonical` can report the flip: a probe's beta
    is for the source's a1, and after a reverse-strand complement that allele
    may no longer be the canonical A1.
    """
    return orient_to_canonical(
        row.chromosome, row.position, row.effect_allele, row.other_allele
    )


def _lift_provenance(
    source_build: str,
    chain_file: str | Path | None,
    variant_reference: str | Path | None,
    lifted_by_source: dict[tuple[str, int, str, str], LiftedRow],
) -> dict[str, object]:
    """What the lift was performed with, and what its allele check found.

    `reference: null` is the record of an unchecked lift; `n_reference_absent`
    counts the lifted variants the reference says nothing about, so a reference
    that overlaps the build by nothing is visible instead of reassuring.
    """
    provenance: dict[str, object] = {
        "source_build": normalise_build(source_build),
        "chain_file": str(chain_file) if chain_file is not None else None,
        "n_lifted": len(lifted_by_source),
        "reference": str(variant_reference) if variant_reference is not None else None,
        "n_alleles_verified": None,
        "n_reference_other_variant": None,
        "n_reference_absent": None,
    }
    if variant_reference is None:
        log.warning(
            "Lift %s→hg38: no --variant-reference supplied, so the lifted alleles were "
            "not checked against a reference. Pass one -- a Variant Reference on hg38 -- "
            "to make the lift verifiable (issue #284).",
            source_build,
        )
        return provenance
    check = require_lifted_alleles_match_reference(lifted_by_source, variant_reference)
    provenance["n_alleles_verified"] = check.n_verified
    provenance["n_reference_other_variant"] = check.n_other_variant
    provenance["n_reference_absent"] = check.n_absent
    print(
        f"Lift alleles checked against {variant_reference}: {check.n_verified}/"
        f"{check.n_lifted} variants confirmed at their hg38 position; "
        f"{check.n_other_variant} positions carry a different reference variant and "
        f"{check.n_absent} are absent from the reference"
    )
    return provenance


def _esi_orientation(
    snp: SnpRecord, lifted: dict[int, Orientation] | None
) -> Orientation | None:
    """The canonical orientation one ESI row is held at.

    A lifted row's orientation was computed from its *lifted* alleles, so its
    variant and its flip are the resolver's for the same row; an unlifted row's
    comes from its own coordinate. ``None`` when the row must not enter the
    axis: no alleles, or a lift that did not land.
    """
    if snp.a1 is None or snp.a2 is None:
        return None
    if lifted is not None:
        return lifted.get(snp.row_idx)
    try:
        return orient_to_canonical(snp.chromosome, snp.bp, snp.a1, snp.a2)
    except VariantNormalisationError:
        return None


#: One ESI entry that survived orientation: its position sort key, its row, its
#: canonical variant, the flip its statistics need, and its rsid when it has one.
_Candidate = tuple[tuple[tuple[int, str], int], int, CanonicalVariant, bool, str | None]


def _axis_candidates(
    snps: list[SnpRecord], lifted: dict[int, Orientation] | None
) -> list[_Candidate]:
    """The entries that can carry a variant, in position order.

    Rows without alleles, unlifted hg19 rows, and rows whose alleles fail
    canonical orientation (``VariantNormalisationError``) are left out: they can
    carry no variant, so they can carry no association either.
    """
    candidate: list[_Candidate] = []
    for snp in snps:
        ori = _esi_orientation(snp, lifted)
        if ori is None:
            continue
        sort_key = (chromosome_sort_key(ori.variant.chromosome), ori.variant.position)
        rsid = snp.snp_id if snp.snp_id.startswith("rs") else None
        candidate.append((sort_key, snp.row_idx, ori.variant, ori.flipped, rsid))
    candidate.sort(key=lambda x: x[0])
    return candidate


def _contested_variants(candidate: list[_Candidate]) -> tuple[set[str], _VariantCollisions]:
    """The ALIDs more than one entry claims, and the collision counts."""
    claims = Counter(entry[2].alid for entry in candidate)
    contested = {alid for alid, n_claims in claims.items() if n_claims > 1}
    return contested, _VariantCollisions(
        n_entries=sum(claims[alid] for alid in contested),
        n_groups=len(contested),
        example=min(contested) if contested else None,
    )


def _canonical_variants(
    snps: list[SnpRecord], lifted: dict[int, Orientation] | None
) -> _CanonicalAxis:
    """Canonicalise ESI rows onto a position-ordered variant axis, dropping collisions.

    Returns the axis plus the collisions it refused. Entries that cannot carry a
    variant are excluded (`_axis_candidates`), and so is every entry whose
    canonical variant another entry also claims.

    Two ESI entries that lift onto one hg38 variant are one coordinate with two
    sources, and nothing in the file set says which of them a probe's
    association belongs to; a shared ``variant_index`` would instead store both
    associations at one variant of one Analysis, which no Ragged reader can
    answer correctly. All entries in such a group go, exactly as a liftover
    failure does -- never a winner picked by position, by allele order or by
    which row came first. A source that lists one variant under two entries
    without a lift is refused the same way, because it is the store's invariant
    at stake rather than the chain's.
    """
    candidate = _axis_candidates(snps, lifted)
    contested, collisions = _contested_variants(candidate)
    survivors = [entry for entry in candidate if entry[2].alid not in contested]
    return _CanonicalAxis(
        variants=[entry[2] for entry in survivors],
        rsid_by_alid={entry[2].alid: entry[4] for entry in survivors if entry[4]},
        esi_to_variant={entry[1]: (idx, entry[3]) for idx, entry in enumerate(survivors)},
        collisions=collisions,
    )


def _opt(value: str | None) -> str | None:
    if value is None or value.strip() in ("", "NA", "NaN", "null", "None", "."):
        return None
    return value.strip()


def _validate_id_overlap(
    manifest_ids: set[str],
    epi_ids: set[str],
    analyses_path: str | Path,
) -> None:
    """Validate exact ID match between manifest and BESD EPI probes."""
    missing_in_epi = manifest_ids - epi_ids
    missing_in_manifest = epi_ids - manifest_ids
    if missing_in_epi and missing_in_manifest:
        raise ValueError(
            f"analyses manifest {analyses_path} does not match BESD EPI analyses: "
            f"manifest has IDs not in BESD ({', '.join(sorted(missing_in_epi))}); "
            f"BESD has IDs not in manifest ({', '.join(sorted(missing_in_manifest))})"
        )
    if missing_in_epi:
        raise ValueError(
            f"analyses manifest {analyses_path} has IDs not present in BESD EPI: "
            f"{', '.join(sorted(missing_in_epi))}"
        )
    if missing_in_manifest:
        raise ValueError(
            f"analyses manifest {analyses_path} is missing rows for BESD EPI analyses: "
            f"{', '.join(sorted(missing_in_manifest))}"
        )


def _load_analyses_overlay(
    analyses_path: str | Path,
    expected_ids: list[str],
) -> dict[str, dict[str, str]]:
    """Read analyses.tsv manifest and validate exact ID match against BESD EPI probes."""
    rows_by_id: dict[str, dict[str, str]] = {}
    with open(analyses_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        id_col = manifest_column(fieldnames, "analysis_id")
        if id_col is None:
            raise ValueError(
                f"analyses manifest {analyses_path} is missing required column: 'analysis_id'"
            )
        for r in reader:
            aid = r[id_col].strip()
            if aid in rows_by_id:
                raise ValueError(
                    f"analyses manifest {analyses_path} contains duplicate analysis_id: {aid!r}"
                )
            rows_by_id[aid] = r

    _validate_id_overlap(set(rows_by_id.keys()), set(expected_ids), analyses_path)
    return rows_by_id


def _overlay_label(r: dict[str, str], probe_gene: str | None) -> str:
    """Resolve analysis_label from manifest row, falling back to probe gene."""
    for col in ("analysis_label", "trait_name"):
        val = _opt(r.get(col))
        if val is not None:
            return val
    return probe_gene or ""


def _overlay_ontology(
    r: dict[str, str], default_id: str | None, default_label: str | None
) -> tuple[str | None, str | None]:
    """Resolve ontology CURIE and label from manifest row."""
    ont_id = _opt(r.get("trait_ontology_id")) or default_id
    ont_label = (
        _opt(r.get("trait_ontology_label"))
        or _opt(r.get("trait_ontology_name"))
        or default_label
    )
    return ont_id, ont_label


def _overlay_n(
    r: dict[str, str], analysis_id: str, manifest_path: str | Path | None = None
) -> int | None:
    """Resolve sample size from manifest row, failing loudly on malformed numbers."""
    for col in ("sample_size", "n"):
        val = _opt(r.get(col))
        if val is not None:
            try:
                return int(val)
            except ValueError as exc:
                loc = f"analyses manifest {manifest_path}: " if manifest_path else ""
                raise ValueError(
                    f"{loc}analysis {analysis_id!r} has invalid sample_size {val!r}"
                ) from exc
    return None


def _overlay_sd(
    r: dict[str, str], analysis_id: str, manifest_path: str | Path | None = None
) -> tuple[str, str]:
    """Validate and resolve original_sd_method and original_sd from manifest row."""
    sd_method = _opt(r.get("original_sd_method"))
    sd_val = _opt(r.get("original_sd"))
    if not sd_val:
        return sd_method or "", ""
    try:
        val_float = float(sd_val)
        if val_float <= 0 or not math.isfinite(val_float):
            raise ValueError
    except ValueError:
        loc = f"analyses manifest {manifest_path}: " if manifest_path else ""
        raise ValueError(
            f"{loc}analysis {analysis_id!r} has invalid original_sd {sd_val!r}"
        ) from None
    return sd_method or "", sd_val


def _overlay_probe_analysis(
    probe: ProbeRecord,
    tissue: str | None,
    probe_chr: str | None,
    manifest_row: dict[str, str],
    analyses_path: str | Path | None = None,
) -> Analysis:
    """Overlay manifest metadata onto a probe record while keeping coordinates authoritative."""
    analysis_id = f"{probe.probe_id}::{tissue}" if tissue else probe.probe_id
    is_ensembl = probe.probe_id.startswith("ENSG")
    def_id = f"ENSEMBL:{probe.probe_id}" if is_ensembl else None
    def_label = "Ensembl" if is_ensembl else None

    r = manifest_row
    label = _overlay_label(r, probe.gene)
    ont_id, ont_label = _overlay_ontology(r, def_id, def_label)
    m_tissue = _opt(r.get("tissue"))
    tissue_val = m_tissue if m_tissue is not None else tissue
    n = _overlay_n(r, analysis_id, analyses_path)
    _sd_method, original_sd = _overlay_sd(r, analysis_id, analyses_path)

    return molecular_analysis(
        analysis_id,
        analysis_label=label,
        trait_ontology_id=ont_id,
        trait_ontology_label=ont_label,
        tissue=tissue_val,
        context=_opt(r.get("context")),
        trait_chr=probe_chr,
        trait_bp=probe.probe_bp if probe.probe_bp > 0 else None,
        n=n,
        stored_effect_scale=_opt(r.get("stored_effect_scale")) or "",
        assigned_ancestry=_opt(r.get("assigned_ancestry")) or "",
        original_sd=original_sd,
        metadata=PassthroughMetadata.from_manifest_row(r),
    )


def _analysis_record_for_probe(
    probe: ProbeRecord,
    tissue: str | None,
    manifest_row: dict[str, str] | None,
    analyses_path: str | Path | None = None,
) -> Analysis:
    """Build one Analysis record for a probe, optionally overlaying manifest metadata."""
    try:
        probe_chr = normalise_chromosome(probe.chromosome)
    except VariantNormalisationError:
        probe_chr = None

    if manifest_row is not None:
        return _overlay_probe_analysis(probe, tissue, probe_chr, manifest_row, analyses_path)

    analysis_id = f"{probe.probe_id}::{tissue}" if tissue else probe.probe_id
    is_ensembl = probe.probe_id.startswith("ENSG")
    return molecular_analysis(
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


def _analysis_records(
    probes: list[ProbeRecord],
    tissue: str | None,
    analyses_path: str | Path | None = None,
) -> list[Analysis]:
    """One Analysis per EPI probe, optionally overlaid with manifest metadata (issue #173)."""
    if analyses_path is None:
        return [_analysis_record_for_probe(probe, tissue, None) for probe in probes]

    expected_ids = [
        f"{probe.probe_id}::{tissue}" if tissue else probe.probe_id for probe in probes
    ]
    overlay_by_id = _load_analyses_overlay(analyses_path, expected_ids)
    return [
        _analysis_record_for_probe(
            probe,
            tissue,
            overlay_by_id[f"{probe.probe_id}::{tissue}" if tissue else probe.probe_id],
            analyses_path,
        )
        for probe in probes
    ]


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
    (ADR 0037, issue #119). This builder stores no frequencies, so this
    settles on ``eaf: absent`` — stated by the plan rather than left to be
    inferred from a missing array.

    That is the builder's choice, not the format's: a `.esi` may carry the
    effect allele's frequency on every row (eQTLGen's does), `read_esi`
    parses it, and `resolve-analyses` measures ancestry and orientation from
    it (issue #282). Storing it as per-association EAF is not implemented.
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
    build_variant_index(staged_path, n_axis=n_variants)

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
    lift: dict[str, object] | None = None,
    collisions: _VariantCollisions | None = None,
) -> None:
    provenance: dict[str, object] = {
        "builder": "opengwasdb.v0.1_ragged_observed_besd",
        "source_besd_prefix": besd_prefix,
        "source_build": source_build,
        "n_variants": n_variants,
        "n_analyses": n_analyses,
        "n_associations": n_associations,
        "ragged": ragged_provenance(staged.path, encoding),
    }
    if lift is not None:
        # Only a build that lifted carries it: a same-assembly release has no
        # lift to describe, and inventing an empty record would say it had one.
        # The collision counts travel inside it (`_VariantCollisions.record`).
        provenance["lift"] = lift
    elif collisions:
        # Nothing lifted, so there is no lift block to hold them, but a source
        # that lists one variant twice still loses entries to the collision rule.
        provenance["n_collisions"] = collisions.n_entries
        provenance["n_collision_groups"] = collisions.n_groups
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
        provenance=provenance,
    )
    staged.write_manifest(manifest)
