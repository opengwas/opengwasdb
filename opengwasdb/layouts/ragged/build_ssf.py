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
import gzip
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from opengwasdb.build.eaf_orientation import (
    apply_orientation_evidence,
    check_eaf_orientation,
    enforce_eaf_orientation,
    load_eaf_reference,
    select_sites,
)
from opengwasdb.layouts.dense.build import add_hit_counts
from opengwasdb.layouts.ragged.analyses import molecular_analysis
from opengwasdb.layouts.ragged.top_hits import build_ragged_top_hit_indexes
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRWriter
from opengwasdb.model.analyses import PassthroughMetadata, write_analysis_records
from opengwasdb.model.enums import (
    AssociationCoverage,
    CompletionState,
    EafScope,
    PrimaryStorageLayout,
    StoredEffectScale,
)
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.stats import parse_af
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
    orient_to_canonical,
)

_REFERENCE_ASSEMBLY = "GRCh38"
_MISSING = {"", ".", "NA", "NaN", "nan", "None"}


@dataclass(frozen=True)
class RaggedBuildResult:
    output_path: Path
    n_variants: int
    n_analyses: int
    n_associations: int


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
    assigned_ancestry: str = ""  # optional manifest column (ADR 0028); "" when omitted
    # Shared-core analyses.tsv columns the manifest supplies and this builder only
    # copies (issue #83). Blank when the manifest omits them; see PassthroughMetadata.
    metadata: PassthroughMetadata = field(default_factory=PassthroughMetadata)


def _opt(value: str | None) -> str | None:
    if value is None or value.strip() in _MISSING:
        return None
    return value.strip()


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
    """
    rows: list[AnalyteInput] = []
    with open(manifest_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            bp = _opt(r.get("trait_bp"))
            n = _opt(r.get("n"))
            rows.append(AnalyteInput(
                analysis_index=int(r["analysis_index"]),
                analysis_id=r["analysis_id"],
                analysis_label=_opt(r.get("analysis_label")),
                trait_ontology_id=_opt(r.get("trait_ontology_id")),
                trait_ontology_label=_opt(r.get("trait_ontology_label")),
                trait_chr=_opt(r.get("trait_chr")),
                trait_bp=int(bp) if bp else None,
                n=int(n) if n else None,
                tissue=_opt(r.get("tissue")),
                context=_opt(r.get("context")),
                mhc=str(r.get("mhc", "")).strip().upper() in {"TRUE", "1", "YES"},
                filtered_path=Path(filtered_dir) / r["filtered_file"],
                assigned_ancestry=_opt(r.get("assigned_ancestry")) or "",
                metadata=PassthroughMetadata.from_manifest_row(r),
            ))
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


def _read_filtered(
    path: Path,
) -> Iterator[tuple[CanonicalVariant, float, float, str | None, float | None]]:
    """Yield (CanonicalVariant, z, se, rsid, eaf) for each usable row of a filtered file."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            try:
                ori = orient_to_canonical(
                    row["chromosome"], row["base_pair_location"],
                    row["effect_allele"], row["other_allele"],
                )
            except (VariantNormalisationError, KeyError):
                continue
            try:
                se = float(row.get("standard_error", ""))
                beta = float(row.get("beta", ""))
            except (TypeError, ValueError):
                continue
            if not (se > 0) or not np.isfinite(se) or not np.isfinite(beta):
                continue
            z = beta / se
            if ori.flipped:
                z = -z
            # GWAS-SSF harmonised files carry a dedicated `rsid` column; `variant_id`
            # is the harmonised (non-rs) identifier and only usable as a fallback
            # when `rsid` is absent (opengwasdb-stores' download-filter step selects
            # both columns when present).
            rsid = _opt(row.get("rsid")) or _opt(row.get("variant_id"))
            eaf = parse_af(row.get("effect_allele_frequency"))
            if eaf is not None and ori.flipped:
                # The column is the frequency of the source's effect allele;
                # the flip made the *other* allele the stored one (ADR 0036).
                eaf = 1.0 - eaf
            yield ori.variant, z, se, rsid, eaf


def _dedupe_per_analysis(
    recs: list[_Assoc],
) -> tuple[list[_Assoc], int, int]:
    """Resolve multiple source rows canonicalizing to the same variant within
    one analysis (issue #101): rows with identical (z, se) are a harmless
    duplicate submission and collapse to one; rows that disagree are dropped
    entirely for that (analysis, variant) cell rather than silently keeping
    an arbitrary one -- GWAS-Catalog-SSF harmonised files can carry more than
    one row per source variant (multi-allelic splits, duplicate submissions),
    and when their canonicalized z-scores conflict there is no principled way
    to pick a winner from the data alone.

    Agreement is judged on `(z, se)` alone. A duplicate pair that agrees on
    the statistics but not on `eaf` is still one association -- the frequency
    is annotation, not the finding -- so the first row's `eaf` is kept rather
    than the cell being dropped over it (ADR 0036).

    Returns ``(deduped_recs, n_consistent_collapsed, n_conflicting_dropped)``.
    """
    by_alid: dict[str, list[_Assoc]] = {}
    order: list[str] = []
    for rec in recs:
        if rec.alid not in by_alid:
            order.append(rec.alid)
        by_alid.setdefault(rec.alid, []).append(rec)

    out: list[_Assoc] = []
    n_consistent = 0
    n_conflicting = 0
    for alid in order:
        values = by_alid[alid]
        if len(values) == 1:
            out.append(values[0])
            continue
        first = values[0]
        if all((v.z, v.se) == (first.z, first.se) for v in values[1:]):
            out.append(first)
            n_consistent += 1
        else:
            n_conflicting += 1
    return out, n_consistent, n_conflicting


def build_ragged_from_ssf(
    manifest_path: str | Path,
    filtered_dir: str | Path,
    output_path: str | Path,
    *,
    store_id: str,
    release_id: str,
    stored_effect_scale: str = StoredEffectScale.SD.value,
    overwrite: bool = False,
    eaf_reference: str | Path | None = None,
    eaf_reference_ancestry: str | None = None,
    allow_unverified_eaf: bool = False,
) -> RaggedBuildResult:
    """Build a Ragged Observed-Only Store from filtered GWAS-SSF inputs.

    ``eaf_reference``/``eaf_reference_ancestry``/``allow_unverified_eaf`` drive
    the EAF orientation check (issue #115): each Analysis's A1-oriented EAF is
    correlated against the reference, or -- with no reference and three or more
    Analyses -- against the consensus of the others, before any array is
    written. A Ragged store is where the metabolome and eQTL pilots live, and
    those manifests routinely carry a single Analysis, so `unverified` is a
    normal outcome here; it is recorded per Analysis rather than passed over.
    """
    try:
        StoredEffectScale(stored_effect_scale)
    except ValueError as exc:
        allowed = [member.value for member in StoredEffectScale]
        raise ValueError(
            f"invalid stored_effect_scale {stored_effect_scale!r}; expected one of {allowed}"
        ) from exc

    out = Path(output_path)
    with OpenGWASDBStore.staging(out, overwrite=overwrite) as staged:
        analytes = _read_manifest(manifest_path, filtered_dir)
        print(f"Manifest: {len(analytes)} analyses")

        # ── Pass 1: read every filtered file; collect per-analysis associations
        #            and the global set of canonical variants ────────────────────
        per_analysis: list[list[_Assoc]] = []
        alid_variant: dict[str, CanonicalVariant] = {}
        rsid_by_alid: dict[str, str] = {}
        for a in analytes:
            recs: list[_Assoc] = []
            for variant, z, se, rsid, eaf in _read_filtered(a.filtered_path):
                alid = variant.alid
                recs.append(_Assoc(alid, z, se, eaf))
                if alid not in alid_variant:
                    alid_variant[alid] = variant
                    if rsid and rsid.startswith("rs"):
                        rsid_by_alid[alid] = rsid
            recs, n_consistent, n_conflicting = _dedupe_per_analysis(recs)
            per_analysis.append(recs)
            msg = f"  {a.analysis_label or a.analysis_id}: {len(recs):,} associations"
            if n_consistent or n_conflicting:
                msg += (
                    f" ({n_consistent} duplicate variant(s) collapsed, "
                    f"{n_conflicting} conflicting duplicate(s) dropped)"
                )
            print(msg)

        # ── EAF orientation (issue #115, ADR 0037 §6) ───────────────────────────
        # Before the variant axis, the CSR, or anything else is written: a
        # source reporting `effect_allele_frequency` against the other allele
        # must not produce a store at all.
        eaf_sites = select_sites(alid_variant.keys())
        wanted_sites = set(eaf_sites)
        eaf_reference_loaded = (
            None
            if eaf_reference is None
            else load_eaf_reference(eaf_reference, eaf_sites, ancestry=eaf_reference_ancestry)
        )
        eaf_report = check_eaf_orientation(
            {
                a.analysis_id: {
                    r.alid: r.eaf
                    for r in recs
                    if r.eaf is not None and r.alid in wanted_sites
                }
                for a, recs in zip(analytes, per_analysis, strict=True)
            },
            reference=eaf_reference_loaded,
            n_sites=len(eaf_sites),
        )
        enforce_eaf_orientation(eaf_report, allow_unverified=allow_unverified_eaf)

        # ── Variant axis: sort unique variants by (chr,pos), assign variant_index ─
        variants = sorted(
            alid_variant.values(),
            key=lambda v: (chromosome_sort_key(v.chromosome), v.position),
        )
        alid_to_idx = {v.alid: i for i, v in enumerate(variants)}
        print(f"Canonical variants: {len(variants):,}")
        write_variant_axis(staged.path, variants, rsid_by_alid)

        # No `analyses` table (ADR 0034, issue #69): analyses.tsv below is the sole
        # source of truth for Analytical Metadata. The file is still created here,
        # empty, so Reference Completion has somewhere to add completion_quality.
        staged.index_connection().close()

        # ── Stream per-analysis associations into the CSR store ──────────────────
        csr = RaggedCSRWriter()
        eaf_scopes: list[str] = []
        for recs in per_analysis:
            if not recs:
                csr.add_analysis(
                    np.empty(0, np.int32), np.empty(0, np.float16), np.empty(0, np.float16)
                )
                eaf_scopes.append(EafScope.ABSENT.value)
                continue
            vi = np.fromiter(
                (alid_to_idx[r.alid] for r in recs), dtype=np.int32, count=len(recs)
            )
            z_arr = np.fromiter((r.z for r in recs), dtype=np.float16, count=len(recs))
            se_arr = np.fromiter((r.se for r in recs), dtype=np.float16, count=len(recs))
            eaf_arr = np.fromiter(
                (np.nan if r.eaf is None else r.eaf for r in recs),
                dtype=np.float32,
                count=len(recs),
            )
            order = np.argsort(vi, kind="stable")
            has_eaf = bool(np.isfinite(eaf_arr).any())
            eaf_scopes.append(
                EafScope.ASSOCIATION.value if has_eaf else EafScope.ABSENT.value
            )
            csr.add_analysis(
                vi[order],
                z_arr[order],
                se_arr[order],
                eaf=eaf_arr[order] if has_eaf else None,
            )
        csr.flush(staged.path)

        build_ragged_top_hit_indexes(staged.path)

        print("Writing analyses.tsv...")
        analyses = [
            molecular_analysis(
                a.analysis_id,
                analysis_label=a.analysis_label,
                trait_ontology_id=a.trait_ontology_id,
                trait_ontology_label=a.trait_ontology_label,
                tissue=a.tissue, context=a.context,
                trait_chr=a.trait_chr, trait_bp=a.trait_bp, n=a.n,
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

        _write_manifest(
            staged, store_id, release_id,
            n_variants=len(variants), n_analyses=len(analytes),
            n_associations=csr.n_associations, manifest_path=str(manifest_path),
            stored_effect_scale=stored_effect_scale,
            mhc_analyses=[a.analysis_id for a in analytes if a.mhc],
            eaf_orientation=eaf_report.provenance(allow_unverified=allow_unverified_eaf),
        )

        result = RaggedBuildResult(out, len(variants), len(analytes), csr.n_associations)
        print(
            f"Build complete: {result.n_variants:,} variants, "
            f"{result.n_analyses:,} analyses, {result.n_associations:,} associations"
        )
    return result


def _write_manifest(
    staged: StagedRelease, store_id: str, release_id: str, *,
    n_variants: int, n_analyses: int, n_associations: int,
    manifest_path: str, stored_effect_scale: str, mhc_analyses: list[str],
    eaf_orientation: dict[str, Any] | None = None,
) -> None:
    manifest = StoreManifest(
        store_id=store_id,
        release_id=release_id,
        format_version=CURRENT_FORMAT_VERSION,
        primary_layout=PrimaryStorageLayout.RAGGED,
        association_coverage=AssociationCoverage.CIS_AND_SIGNALS,
        completion_state=CompletionState.OBSERVED_ONLY,
        reference_assembly=_REFERENCE_ASSEMBLY,
        created_at=datetime.now(UTC).isoformat(),
        provenance={
            "builder": "opengwasdb.v0.1_ragged_observed_ssf",
            "source_manifest": manifest_path,
            "stored_effect_scale": stored_effect_scale,
            "mhc_analyses": mhc_analyses,
            "n_variants": n_variants,
            "n_analyses": n_analyses,
            "n_associations": n_associations,
            **({"eaf_orientation": eaf_orientation} if eaf_orientation is not None else {}),
            "ragged": {
                "statistic_arrays": ["z", "se"],
                "dtype": "float16",
                "variant_axis": {
                    "format": VARIANT_AXIS_FORMAT,
                    "table": VARIANT_TABLE_FILENAME,
                    "tabix_index": VARIANT_TABIX_FILENAME,
                },
            },
        },
    )
    staged.write_manifest(manifest)
