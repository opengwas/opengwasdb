"""Build a Ragged Observed-Only Store from filtered GWAS-SSF files.

Adapter mirroring :func:`build_ragged_from_besd`, but the source is one filtered
GWAS-SSF ``.tsv.gz`` per analysis (as produced by the opengwasdb-stores
download+filter step) plus an analyses manifest carrying per-analysis metadata
(trait position, N, tissue/context, MHC flag). ``trait_id`` and other
gene-target columns are family-specific to gene-target Store Families (for
example SomaScan proteomics) and are not required here -- a manifest without
one (for example small-molecule metabolomics, no single encoding gene) builds
identically.

The ragged storage, variant axis, top-hit indexes and manifest are all reused
unchanged; the only new code is the read + group + variant-index-mapping glue.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import DTypeLike

from opengwasdb import admission
from opengwasdb.admission.declared import declared_rules
from opengwasdb.build.eaf_orientation import (
    EafOrientationReport,
    apply_orientation_evidence,
    check_eaf_orientation,
    enforce_eaf_orientation,
    load_eaf_reference,
    select_sites,
)
from opengwasdb.build.liftover import normalise_build
from opengwasdb.encoding import EncodingMeasurements, StoreEncoding
from opengwasdb.layouts.ragged import RAGGED_REFERENCE_ASSEMBLY, RaggedBuildResult
from opengwasdb.layouts.ragged.analyses import molecular_analysis
from opengwasdb.layouts.ragged.by_variant import build_variant_index
from opengwasdb.layouts.ragged.provenance import ragged_provenance
from opengwasdb.layouts.ragged.top_hits import build_ragged_top_hit_indexes
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRWriter
from opengwasdb.layouts.release_manifest import observed_only_release_manifest
from opengwasdb.model import coverage_for_layout
from opengwasdb.model.analyses import PassthroughMetadata, write_analysis_records
from opengwasdb.model.enums import (
    AssociationCoverage,
    EafScope,
    PrimaryStorageLayout,
    StoredEffectScale,
)
from opengwasdb.model.manifest_columns import (
    ManifestColumns,
    require_columns,
    resolve_manifest_columns,
)
from opengwasdb.readers.assembly import declared_build
from opengwasdb.readers.gwas_ssf import GWAS_SSF_CAPABILITY
from opengwasdb.readers.interface import (
    StreamTally,
)
from opengwasdb.readers.registry import SourceSpec, open_source
from opengwasdb.readers.rows import ReaderBatch
from opengwasdb.store.open import OpenGWASDBStore, StagedRelease
from opengwasdb.top_hits import add_hit_counts
from opengwasdb.variants.axis import (
    write_variant_axis,
)
from opengwasdb.variants.normalise import (
    CanonicalVariant,
    chromosome_sort_key,
)
from opengwasdb.variants.routing import SourceKey, collapse_duplicates, union_names
from opengwasdb.variants.rsid import RsidConflictPolicy

_MISSING = {"", ".", "NA", "NaN", "nan", "None"}


@dataclass
class AnalyteInput:
    analysis_index: int
    analysis_id: str
    analysis_label: str | None
    trait_ontology_id: str | None
    trait_ontology_label: str | None
    trait_chr: str | None
    trait_bp: int | None
    n: int | None
    tissue: str | None
    context: str | None
    mhc: bool
    filtered_path: Path
    #: The build the manifest row declares (`source_genome_build`), normalised:
    #: GRCh38's, since a row declaring another build, or none, is refused.
    source_build: str
    assigned_ancestry: str = ""  # optional manifest column (ADR 0028); "" when omitted
    # Shared-core analyses.tsv columns the manifest supplies and this builder only
    # copies (issue #83). Blank when the manifest omits them; see PassthroughMetadata.
    metadata: PassthroughMetadata = field(default_factory=PassthroughMetadata)


def _opt(value: str | None) -> str | None:
    if value is None or value.strip() in _MISSING:
        return None
    return value.strip()


def _analyte_from_row(
    r: dict[str, str],
    cols: ManifestColumns,
    filtered_dir: str | Path,
) -> AnalyteInput:
    """Parse one AnalyteInput from a manifest row."""
    bp = _opt(r.get("trait_bp"))
    n_raw = _opt(r.get(cols.sample_size)) if cols.sample_size else None
    n = int(n_raw) if n_raw else None
    raw_path = Path(r[cols.source_file])
    filtered_path = raw_path if raw_path.is_absolute() else Path(filtered_dir) / raw_path
    label = _opt(r.get(cols.analysis_label)) if cols.analysis_label else None
    analysis_id = r[cols.analysis_id]
    return AnalyteInput(
        analysis_index=int(r["analysis_index"]),
        analysis_id=analysis_id,
        analysis_label=label,
        trait_ontology_id=_opt(r.get("trait_ontology_id")),
        trait_ontology_label=_opt(r.get("trait_ontology_label")),
        trait_chr=_opt(r.get("trait_chr")),
        trait_bp=int(bp) if bp else None,
        n=n,
        tissue=_opt(r.get("tissue")),
        context=_opt(r.get("context")),
        mhc=str(r.get("mhc", "")).strip().upper() in {"TRUE", "1", "YES"},
        filtered_path=filtered_path,
        source_build=_grch38_source(r, analysis_id),
        assigned_ancestry=_opt(r.get("assigned_ancestry")) or "",
        metadata=PassthroughMetadata.from_manifest_row(r),
    )


def _grch38_source(row: Mapping[str, str], analysis_id: str) -> str:
    """The row's declared build, refused unless it is GRCh38.

    A Ragged SSF Store is one build, and every source declares its own. The
    build lifts nothing, so a source on another build would be stored at
    coordinates its variants do not have; one that declares none is not
    assumed to be on GRCh38.
    """
    build = declared_build(row, context=f"analysis {analysis_id!r}")
    if build is None:
        raise ValueError(
            f"analysis {analysis_id!r} declares no source_genome_build; "
            f"build-ragged-ssf reads {RAGGED_REFERENCE_ASSEMBLY} sources, and every "
            "manifest row must declare its source's build"
        )
    if build != normalise_build(RAGGED_REFERENCE_ASSEMBLY):
        raise ValueError(
            f"analysis {analysis_id!r} declares source_genome_build {build}; "
            f"build-ragged-ssf reads {RAGGED_REFERENCE_ASSEMBLY} sources and lifts none"
        )
    return build


def _read_manifest(manifest_path: str | Path, filtered_dir: str | Path) -> list[AnalyteInput]:
    """Read the build manifest into one `AnalyteInput` per analysis.

    Beyond the molecular/context columns this builder interprets (trait
    position, N, tissue/context, MHC flag), every shared-core `analyses.tsv`
    column the manifest carries is read through `PassthroughMetadata` and
    copied into the built store verbatim -- Analytical Metadata
    (``sample_size_kind``/``sample_size_scope``/``n_cases``/``n_controls``/
    ``original_effect_scale``/``original_sd_method``/
    ``ancestry_assignment_method``/``ancestry_prop_<population>``) and
    Attribution Metadata (``license``/``publication_doi``/
    ``publication_pmid``/``consortium``/``first_author``). Ragged dropped all
    of these before issue #83, silently: the manifest had the values, the
    built store did not. They stay blank when the manifest omits them, never
    inferred -- only the manifest producer knows them.

    Accepts canonical ``analyses.tsv`` column names (``sample_size``,
    ``source_file``) as well as legacy spellings (``n``, ``filtered_file``,
    ``file_path``) with canonical names winning (issue #172). When
    ``source_file``/``filtered_file`` is an absolute path, it is used as-is;
    relative paths are joined to ``filtered_dir``.
    """
    rows: list[AnalyteInput] = []
    with open(manifest_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        require_columns(fieldnames, manifest_path, "analysis_index")
        cols = resolve_manifest_columns(fieldnames, manifest_path)
        for r in reader:
            rows.append(_analyte_from_row(r, cols, filtered_dir))
    rows.sort(key=lambda a: a.analysis_index)
    # analysis_index must be a dense 0..n-1 sequence for CSR offset alignment.
    for expected, a in enumerate(rows):
        if a.analysis_index != expected:
            raise ValueError(
                f"analysis_index must be 0..n-1 in order; got {a.analysis_index} "
                f"at position {expected}"
            )
    return rows


class _Assoc(NamedTuple):
    """One usable source row, canonicalized: `z`/`eaf` are oriented to the
    stored effect allele, so both follow the same flip (ADR 0036)."""

    alid: str
    z: float
    se: float
    eaf: float | None


def _admitted_batches(
    a: AnalyteInput, gate: admission.Gate, scale: StoredEffectScale
) -> list[ReaderBatch]:
    """One filtered file's rows that its Analysis's Admission gate keeps, in file order.

    Read by the GWAS-SSF Source Reader, oriented to A1. The gate judges them
    by the Analysis's declared INFO and MAF filters too (D2), from the
    declared score column and the file's own frequency, after the rows the
    reader dropped for a malformed identity are named (#365).
    """
    declaration = gate.rules.info.imputation_score_declaration
    reader = open_source(
        SourceSpec(a.analysis_id, str(a.filtered_path), GWAS_SSF_CAPABILITY, a.source_build, scale),
        imputation_score_declaration=declaration,
    )
    tally = StreamTally()
    batches = list(reader.stream_batches(tally))
    gate.streamed(tally)
    declared = declaration is not None
    return [
        batch.take(
            gate.source_rows(
                admission.SourceRows(
                    eaf=batch.eaf,
                    imputation_score=batch.imputation_score if declared else None,
                    imputation_status=batch.imputation_status if declared else None,
                    se=batch.se,
                    z=batch.z,
                    keys_at=batch.keys_at,
                )
            )
        )
        for batch in batches
    ]


def _dedupe_per_analysis(
    batches: list[ReaderBatch], gate: admission.Gate
) -> tuple[list[_Assoc], int, int]:
    """Resolve multiple source rows canonicalizing to the same variant within
    one analysis (issue #101): rows with identical (z, se) are a harmless
    duplicate submission and collapse to one; rows that disagree are dropped
    entirely for that (analysis, variant) cell rather than silently keeping
    an arbitrary one -- GWAS-Catalog-SSF harmonised files can carry more than
    one row per source variant (multi-allelic splits, duplicate submissions),
    and when their canonicalized z-scores conflict there is no principled way
    to pick a winner from the data alone.

    The rule is Variant Routing's (`collapse_duplicates`), so every builder
    resolves a duplicate alike, with the reader's reference declaration.
    Agreement is judged on `(z, se)` alone. A duplicate pair that agrees on
    the statistics but not on `eaf` is still one association -- the frequency
    is annotation, not the finding -- so the first row's `eaf` is kept rather
    than the cell being dropped over it (ADR 0036). The gate counts the
    collapsed rows and counts and logs the dropped ones.

    Returns ``(deduped_recs, n_consistent_collapsed, n_conflicting_dropped)``,
    the last two counting variants.
    """
    alid, z, se, eaf, reference = _columns(batches)
    number: dict[str, int] = {}
    codes = np.fromiter(
        (number.setdefault(key, len(number)) for key in alid.tolist()),
        dtype=np.int64, count=len(alid),
    )
    found = collapse_duplicates(codes, z, se, reference)
    gate.duplicates(
        [str(alid[i]) for i in found.dropped.tolist()], found.collapsed,
        [str(alid[i]) for i in found.reversed_indels.tolist()],
    )
    repeated = int(np.count_nonzero(np.bincount(codes, minlength=len(number)) > 1))
    n_conflicting = len(found.conflicting)
    kept = found.keep
    recs = [
        _Assoc(key, value, error, None if math.isnan(frequency) else frequency)
        for key, value, error, frequency in zip(
            alid[kept].tolist(), z[kept].tolist(), se[kept].tolist(), eaf[kept].tolist(),
            strict=True,
        )
    ]
    return recs, repeated - n_conflicting, n_conflicting


def _columns(
    batches: list[ReaderBatch],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One Analysis's batches as ``(alid, z, se, eaf, reference codes)`` columns."""

    def joined(parts: list[np.ndarray], dtype: DTypeLike) -> np.ndarray:
        return np.concatenate(parts) if parts else np.empty(0, dtype=dtype)

    return (
        joined([b.alid for b in batches], object),
        joined([b.z for b in batches], np.float64),
        joined([b.se for b in batches], np.float64),
        joined([b.eaf for b in batches], np.float64),
        joined([b.reference_codes() for b in batches], np.int8),
    )


def build_ragged_from_ssf(
    manifest_path: str | Path,
    filtered_dir: str | Path,
    output_path: str | Path,
    *,
    store_id: str,
    release_id: str,
    association_coverage: AssociationCoverage | str,
    stored_effect_scale: str = StoredEffectScale.SD.value,
    overwrite: bool = False,
    eaf_reference: str | Path | None = None,
    eaf_reference_ancestry: str | None = None,
    allow_unverified_eaf: bool = False,
    admission_record: str | Path | None = None,
) -> RaggedBuildResult:
    """Build a Ragged Observed-Only Store from filtered GWAS-SSF inputs.

    A thin orchestrator: the Staged Release's lifetime here, the six phases in
    `_build_ssf_release`. ``association_coverage`` has no default (issue #298)
    -- the same command builds a cis-eQTL release, a signals-only metabolite
    release and a genome-wide one, so the layout cannot choose it (ADR 0064).
    ``admission_record`` is Association Admission's record (#357), beside the Store.
    Every manifest row must declare its `source_genome_build`, and declare
    GRCh38: a Ragged SSF Store is one build, and this build lifts nothing.
    """
    coverage = coverage_for_layout(association_coverage, PrimaryStorageLayout.RAGGED)
    _validate_stored_effect_scale(stored_effect_scale)
    out = Path(output_path)
    with OpenGWASDBStore.staging(out, overwrite=overwrite) as staged:
        return _build_ssf_release(
            staged,
            out,
            (manifest_path, filtered_dir),
            _SsfBuildRequest(
                store_id=store_id,
                release_id=release_id,
                stored_effect_scale=stored_effect_scale,
                coverage=coverage,
                eaf_reference=eaf_reference,
                eaf_reference_ancestry=eaf_reference_ancestry,
                allow_unverified_eaf=allow_unverified_eaf,
                admission_record=admission.admission_record_for(out, admission_record),
            ),
        )


@dataclass(frozen=True)
class _SsfBuildRequest:
    """The scalar configuration one SSF build's phases run under."""

    store_id: str
    release_id: str
    stored_effect_scale: str
    coverage: AssociationCoverage
    eaf_reference: str | Path | None
    eaf_reference_ancestry: str | None
    allow_unverified_eaf: bool
    admission_record: Path


def _build_ssf_release(
    staged: StagedRelease,
    out: Path,
    sources: tuple[str | Path, str | Path],
    request: _SsfBuildRequest,
) -> RaggedBuildResult:
    """Phases 1-6 of an SSF build, inside one Staged Release's lifetime.

    The EAF check (issue #115) runs before anything is written, and
    `unverified` is a normal Ragged outcome, recorded per Analysis, never
    passed over.
    """
    manifest_path, filtered_dir = sources
    analytes = _read_manifest(manifest_path, filtered_dir)
    print(f"Manifest: {len(analytes)} analyses")
    record = admission.AdmissionRecord.open(
        request.admission_record,
        [a.analysis_id for a in analytes],
        command="build-ragged-ssf",
    )
    rules = declared_rules(manifest_path, {a.analysis_id: GWAS_SSF_CAPABILITY for a in analytes})
    per_analysis, alid_variant, rsid_by_alid = _ingest_ssf_sources(analytes, record, rules, request)
    eaf_report = _verify_eaf_orientation(
        analytes, per_analysis, alid_variant,
        eaf_reference=request.eaf_reference,
        eaf_reference_ancestry=request.eaf_reference_ancestry,
        allow_unverified_eaf=request.allow_unverified_eaf,
    )
    variants, alid_to_idx = _build_variant_axis(staged, alid_variant, rsid_by_alid)
    csr, encoding, eaf_scopes = _encode_ssf_csr(
        per_analysis, alid_to_idx, len(analytes),
        len(variants), staged.path,
    )
    _write_indexes_and_metadata(
        staged,
        analytes,
        eaf_scopes,
        encoding=encoding,
        eaf_report=eaf_report,
        stored_effect_scale=request.stored_effect_scale,
    )
    _write_manifest(
        staged,
        request.store_id,
        request.release_id,
        n_variants=len(variants),
        n_analyses=len(analytes),
        n_associations=csr.n_associations,
        manifest_path=str(manifest_path),
        stored_effect_scale=request.stored_effect_scale,
        association_coverage=request.coverage,
        mhc_analyses=[a.analysis_id for a in analytes if a.mhc],
        encoding=encoding,
        eaf_orientation=eaf_report.provenance(
            allow_unverified=request.allow_unverified_eaf
        ),
    )
    return _finalise_result(
        out, len(variants), len(analytes), csr.n_associations, request.coverage
    )


def _finalise_result(
    out: Path,
    n_variants: int,
    n_analyses: int,
    n_associations: int,
    association_coverage: AssociationCoverage,
) -> RaggedBuildResult:
    """Build the result record and print the completion summary it carries."""
    result = RaggedBuildResult(out, n_variants, n_analyses, n_associations, association_coverage)
    print(
        f"Build complete: {result.n_variants:,} variants, "
        f"{result.n_analyses:,} analyses, {result.n_associations:,} associations"
    )
    return result


def _validate_stored_effect_scale(stored_effect_scale: str) -> None:
    """Reject an unsupported stored_effect_scale before any I/O happens."""
    try:
        StoredEffectScale(stored_effect_scale)
    except ValueError as exc:
        allowed = [member.value for member in StoredEffectScale]
        raise ValueError(
            f"invalid stored_effect_scale {stored_effect_scale!r}; expected one of {allowed}"
        ) from exc


def _ingest_ssf_sources(
    analytes: list[AnalyteInput],
    record: admission.AdmissionRecord,
    rules: Mapping[str, admission.AnalysisRules],
    request: _SsfBuildRequest,
) -> tuple[list[list[_Assoc]], dict[str, CanonicalVariant], dict[str, str]]:
    """Read every filtered source and resolve each analysis's associations.

    One pass per analysis: canonicalise each source row, admit it through the
    Analysis's gate, deduplicate rows that canonicalise to the same variant
    (issue #101), and collect the global canonical-variant set plus each
    variant's rsid, after comparing its spellings' names (`_Harvest`). The
    printed per-analysis count is the fail-loud record of what survived: a row
    Admission drops never reaches this point and is in ``record``, and a
    conflicting duplicate pair is dropped for that cell entirely rather than
    one of them silently kept.
    """
    scale = StoredEffectScale(request.stored_effect_scale)
    per_analysis: list[list[_Assoc]] = []
    harvest = _Harvest()
    for a in analytes:
        with record.gate(rules[a.analysis_id]) as gate:
            batches = _admitted_batches(a, gate, scale)
            for batch in batches:
                harvest.add(batch)
            recs, n_consistent, n_conflicting = _dedupe_per_analysis(batches, gate)
        per_analysis.append(recs)
        _print_ingest_summary(a, recs, n_consistent, n_conflicting)
    record.finish()
    return per_analysis, harvest.alid_variant, harvest.names()


#: What the harvest's refusal calls this build (ADR 0066).
HARVEST_LABEL = "Ragged SSF build"


@dataclass
class _Harvest:
    """The variants a build's admitted rows reach, and the names they carry.

    Rows are added in Manifest Row order and file order. Each row is read
    with the GWAS-SSF reader's rsid rule: `rsid`, else `variant_id`,
    whichever is an rs identifier.

    - ``alid_variant``: each variant, registered at its first admitted row.
    - ``named``: each raw site's first non-empty rsid, in the order sites are
      first named. A site is its ALID and its alleles as the source spells
      them, as Pass 1's union keeps a site, so `names` can compare a site's
      spellings.
    - ``unnamed``: the sites whose admitted rows have named nothing so far.

    The cost is kept to what the first-seen check before #379 paid
    (OGS-00006 admits 58M rows for 12M sites):
    - A row of a site already named costs one lookup.
    - The ALID holds the site's chromosome and position, so it is split once
      per variant here and once per site at naming, not once per row.
    - A site is keyed as one string, ``alid<TAB>ref<TAB>alt``. A map of
      strings holds nothing the cyclic garbage collector tracks, so it adds
      nothing to each of the build's collections. Tuple keys added 12M
      entries to every collection.
    """

    alid_variant: dict[str, CanonicalVariant] = field(default_factory=dict)
    named: dict[str, str] = field(default_factory=dict)
    unnamed: set[str] = field(default_factory=set)

    def add(self, batch: ReaderBatch) -> None:
        """Register each of ``batch``'s rows, a batch of admitted rows."""
        alid_variant, named, unnamed = self.alid_variant, self.named, self.unnamed
        for alid, ref, alt, rsid in zip(
            batch.alid.tolist(), batch.ref.tolist(), batch.alt.tolist(), batch.rsid.tolist(),
            strict=True,
        ):
            site = f"{alid}\t{ref}\t{alt}"
            if site in named:
                continue
            if site in unnamed:
                if rsid:
                    unnamed.discard(site)
                    named[site] = rsid
                continue
            if alid not in alid_variant:
                chromosome, position, a1, a2 = alid.split(":")
                alid_variant[alid] = CanonicalVariant(chromosome, int(position), a1, a2)
            if rsid:
                named[site] = rsid
            else:
                unnamed.add(site)

    def names(self) -> dict[str, str]:
        """Each variant's rsid, after comparing the names a site's spellings carry.

        The harvest goes through `union_names`, the test every harvest applies
        (ADR 0066, #379). `build-ragged-ssf` has no `--on-rsid-conflict`, so a
        Routing Key its sources name two ways is refused, naming both names.
        Otherwise a variant takes the first non-empty rsid among its admitted
        rows, as spec §4 says, whichever spelling carries it (#400).
        """
        rsid_by_site: dict[SourceKey, str] = {}
        site_alid: dict[SourceKey, str] = {}
        for key, rsid in self.named.items():
            alid, ref, alt = key.split("\t")
            chromosome, position, _a1, _a2 = alid.split(":")
            site = (chromosome, int(position), ref, alt)
            rsid_by_site[site] = rsid
            site_alid[site] = alid
        return union_names(
            rsid_by_site, site_alid, policy=RsidConflictPolicy.REFUSE, label=HARVEST_LABEL
        ).rsid_by_alid


def _print_ingest_summary(
    a: AnalyteInput, recs: list[_Assoc], n_consistent: int, n_conflicting: int
) -> None:
    """Print one analysis's per-file ingest count, collapsed or dropped rows loud."""
    msg = f"  {a.analysis_label or a.analysis_id}: {len(recs):,} associations"
    if n_consistent or n_conflicting:
        msg += (
            f" ({n_consistent} duplicate variant(s) collapsed, "
            f"{n_conflicting} conflicting duplicate(s) dropped)"
        )
    print(msg)


def _verify_eaf_orientation(
    analytes: list[AnalyteInput],
    per_analysis: list[list[_Assoc]],
    alid_variant: dict[str, CanonicalVariant],
    *,
    eaf_reference: str | Path | None,
    eaf_reference_ancestry: str | None,
    allow_unverified_eaf: bool,
) -> EafOrientationReport:
    """Check every analysis's A1-oriented EAF before anything is written.

    Each Analysis's stored-frequency observations are correlated against the
    reference, or -- with no reference and three or more Analyses -- against
    the consensus of the others, and a build whose frequencies are
    anti-correlated is refused (issue #115). `select_sites` samples each
    Analysis over its own variants and the observations only ever carry the
    sampled `wanted_sites`, so the check compares what the store will read
    back, over the sites the report records.
    """
    eaf_sites = select_sites(alid_variant.keys())
    wanted_sites = set(eaf_sites)
    reference = (
        None
        if eaf_reference is None
        else load_eaf_reference(eaf_reference, eaf_sites, ancestry=eaf_reference_ancestry)
    )
    report = check_eaf_orientation(
        {
            a.analysis_id: {
                r.alid: r.eaf for r in recs if r.eaf is not None and r.alid in wanted_sites
            }
            for a, recs in zip(analytes, per_analysis, strict=True)
        },
        reference=reference,
        n_sites=len(eaf_sites),
    )
    enforce_eaf_orientation(report, allow_unverified=allow_unverified_eaf)
    return report


def _build_variant_axis(
    staged: StagedRelease,
    alid_variant: dict[str, CanonicalVariant],
    rsid_by_alid: dict[str, str],
) -> tuple[list[CanonicalVariant], dict[str, int]]:
    """Sort the unique canonical variants into the axis and index the store.

    No `analyses` table (ADR 0034, issue #69): analyses.tsv is the sole
    source of truth for Analytical Metadata. The empty index file is still
    created here, so Reference Completion has somewhere to add
    completion_quality, and its connection is closed before the CSR phase.
    """
    variants = sorted(
        alid_variant.values(),
        key=lambda v: (chromosome_sort_key(v.chromosome), v.position),
    )
    alid_to_idx = {v.alid: i for i, v in enumerate(variants)}
    print(f"Canonical variants: {len(variants):,}")
    write_variant_axis(staged.path, variants, rsid_by_alid)
    staged.index_connection().close()
    return variants, alid_to_idx


def _encode_ssf_csr(
    per_analysis: list[list[_Assoc]],
    alid_to_idx: dict[str, int],
    n_analyses: int,
    n_variants: int,
    output: Path,
) -> tuple[RaggedCSRWriter, StoreEncoding, list[str]]:
    """Stream every analysis's associations into the CSR and flush.

    An analysis that ends up empty -- a source whose every row was unusable,
    or whose rows all resolved to conflicting duplicates (issue #101) --
    still occupies its CSR slot and declares `eaf_scope=absent`; skipping it
    would shift every later analysis's associations onto the wrong analysis.
    Returns the writer (for the association count), the one encoding plan the
    build flushed under, and each analysis's EAF scope for analyses.tsv.
    """
    csr = RaggedCSRWriter(n_variants)
    eaf_scopes: list[str] = []
    for index, recs in enumerate(per_analysis):
        if not recs:
            _add_empty_analysis(csr)
            eaf_scopes.append(EafScope.ABSENT.value)
            continue
        eaf_scopes.append(_add_ssf_associations(csr, recs, alid_to_idx, index))
    eaf_measurements = csr.eaf_measurements()
    preliminary = StoreEncoding.decide(
        EncodingMeasurements(n_analyses=n_analyses, eaf=eaf_measurements)
    )
    encoding = StoreEncoding.decide(
        EncodingMeasurements(
            n_analyses=n_analyses,
            eaf=eaf_measurements,
            se=csr.se_measurements(preliminary),
        )
    )
    csr.flush(output, encoding)
    build_variant_index(output, n_axis=n_variants)
    return csr, encoding, eaf_scopes


def _add_empty_analysis(csr: RaggedCSRWriter) -> None:
    """An analysis with no associations still occupies its CSR slot."""
    csr.add_analysis(
        np.empty(0, np.int32), np.empty(0, np.float32), np.empty(0, np.float32)
    )


def _add_ssf_associations(
    csr: RaggedCSRWriter, recs: list[_Assoc], alid_to_idx: dict[str, int], index: int
) -> str:
    """Append one non-empty analysis, sorted by variant_index.

    Returns the EAF scope the analysis's own rows earned: association when
    any row carries a frequency (rows without one contribute NaN, ADR 0036),
    absent when none do.
    """
    vi = np.fromiter((alid_to_idx[r.alid] for r in recs), dtype=np.int32, count=len(recs))
    z_arr = np.fromiter((r.z for r in recs), dtype=np.float32, count=len(recs))
    se_arr = np.fromiter((r.se for r in recs), dtype=np.float32, count=len(recs))
    eaf_arr = np.fromiter(
        (np.nan if r.eaf is None else r.eaf for r in recs), dtype=np.float32, count=len(recs)
    )
    where = f"build-ragged-ssf, analysis_index {index}"
    order = np.argsort(vi, kind="stable")
    has_eaf = bool(np.isfinite(eaf_arr).any())
    if has_eaf:
        csr.add_analysis(vi[order], z_arr[order], se_arr[order], eaf=eaf_arr[order], where=where)
        return EafScope.ASSOCIATION.value
    csr.add_analysis(vi[order], z_arr[order], se_arr[order], where=where)
    return EafScope.ABSENT.value


def _write_indexes_and_metadata(
    staged: StagedRelease,
    analytes: list[AnalyteInput],
    eaf_scopes: list[str],
    *,
    encoding: StoreEncoding,
    eaf_report: EafOrientationReport,
    stored_effect_scale: str,
) -> None:
    """Build the top-hit index and write analyses.tsv from the manifest.

    The shared-core metadata columns arrive as one PassthroughMetadata per
    analysis and are copied verbatim (issue #83); the orientation evidence
    each analysis earned in Phase 2 is stamped onto its row, and the hit
    counts are added from the just-built index.
    """
    build_ragged_top_hit_indexes(staged.path, encoding=encoding)

    print("Writing analyses.tsv...")
    analyses = [
        molecular_analysis(
            a.analysis_id,
            analysis_label=a.analysis_label,
            trait_ontology_id=a.trait_ontology_id,
            trait_ontology_label=a.trait_ontology_label,
            tissue=a.tissue,
            context=a.context,
            trait_chr=a.trait_chr,
            trait_bp=a.trait_bp,
            n=a.n,
            stored_effect_scale=stored_effect_scale,
            assigned_ancestry=a.assigned_ancestry,
            metadata=a.metadata,
            eaf_scope=eaf_scope,
        )
        for a, eaf_scope in zip(analytes, eaf_scopes, strict=True)
    ]
    write_analysis_records(
        staged.path / "analyses.tsv",
        add_hit_counts(staged.path, apply_orientation_evidence(analyses, eaf_report)),
    )



def _write_manifest(
    staged: StagedRelease,
    store_id: str,
    release_id: str,
    *,
    n_variants: int,
    n_analyses: int,
    n_associations: int,
    manifest_path: str,
    stored_effect_scale: str,
    association_coverage: AssociationCoverage,
    mhc_analyses: list[str],
    encoding: StoreEncoding,
    eaf_orientation: dict[str, Any] | None = None,
) -> None:
    staged.write_manifest(
        observed_only_release_manifest(
            PrimaryStorageLayout.RAGGED,
            reference_assembly=RAGGED_REFERENCE_ASSEMBLY,
            store_id=store_id,
            release_id=release_id,
            encoding=encoding,
            association_coverage=association_coverage,
            provenance={
                "builder": "opengwasdb.v0.1_ragged_observed_ssf",
                "source_manifest": manifest_path,
                "stored_effect_scale": stored_effect_scale,
                "mhc_analyses": mhc_analyses,
                "n_variants": n_variants,
                "n_analyses": n_analyses,
                "n_associations": n_associations,
                **({"eaf_orientation": eaf_orientation} if eaf_orientation is not None else {}),
                "ragged": ragged_provenance(staged.path, encoding),
            },
        )
    )
