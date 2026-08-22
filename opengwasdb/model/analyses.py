"""Analytical Metadata contract for `analyses.tsv` (store-format spec §7a).

`analyses.tsv` spans three column classes (opengwasdb-stores
docs/release-metadata-schema.md, "Column classes"): a *shared core* owned by
`opengwasdb` that carries interpretation-bearing Analysis metadata and appears
in both a registry release manifest and a built store; *registry-only*
build-input columns (source file location, checksum, licence, inclusion
reason, ...) that a manifest carries but a built store does not; and
*store-only* columns (Reference Completion quality rollups, ...) that cannot
exist before a build has run. A manifest and a built store's `analyses.tsv`
are therefore superset-related, not identical: this module's reader accepts
any column beyond the ones it knows about and carries it through unchanged in
`AnalysesTable.rows`, so a caller reading a manifest never has to strip
registry-only columns first, and a caller reading a built store never has to
strip store-only ones.

This is the schema `opengwasdb-stores#51` needs to validate its emitted
manifests against, replacing that repository's hand-rolled, per-generator
column-name checks (issue #16). Controlled-vocabulary literals here match the
registry's already-shipped vocabulary (`opengwasdb-stores`
docs/release-metadata-schema.md) rather than this package's earlier, unshipped
draft spelling -- see `opengwasdb.model.enums`.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from opengwasdb.model.enums import (
    AncestryAssignmentMethod,
    EafOrientationOutcome,
    EafScope,
    OriginalSdMethod,
    SampleSizeKind,
    SampleSizeScope,
    StoredEffectScale,
)

ANCESTRY_PROP_PREFIX = "ancestry_prop_"

# Columns interpretation-bearing enough that opengwasdb owns their meaning and
# vocabulary, present in both a registry manifest and a built store's
# analyses.tsv (store-format spec §7a). `ancestry_prop_<population>` columns
# are dynamic (one per reference population) and matched via
# ANCESTRY_PROP_PREFIX rather than listed here.
SHARED_CORE_COLUMNS: tuple[str, ...] = (
    "analysis_index",
    "analysis_id",
    "phenotype_id",
    "phenotype_label",
    "analysis_label",
    "stored_effect_scale",
    "assigned_ancestry",
    "ancestry_assignment_method",
    "sample_size_kind",
    "sample_size_scope",
    "sample_size",
    "n_cases",
    "n_controls",
    "original_effect_scale",
    "original_sd",
    "original_sd_method",
    "original_sd_dispersion",
)

# Top-Hit Count columns (ADR 0032, store-format spec §7a): one persisted
# per-Analysis count per threshold tier in
# opengwasdb.layouts.dense.constants.TOP_HIT_THRESHOLDS (5e-8/5e-6/5e-4, in
# that order). Not imported from there to keep this module layout-independent
# -- it is also the schema opengwasdb-stores validates manifests against.
TOP_HIT_COUNT_COLUMNS: tuple[str, ...] = ("n_hits_5e8", "n_hits_5e6", "n_hits_5e4")

# EAF orientation evidence (issue #115, ADR 0037 §6): the outcome of
# correlating this Analysis's A1-oriented EAF against a reference, the observed
# r, and how many variants it was computed over. Persisted per Analysis because
# standalone validation cannot depend on the reference panel still being
# available -- it checks the recorded evidence, and a separate audit re-runs the
# correlation when a panel is supplied.
EAF_ORIENTATION_COLUMNS: tuple[str, ...] = (
    "eaf_orientation",
    "eaf_orientation_r",
    "eaf_orientation_n",
)

# Produced during or after the build; must not be required of a release
# manifest, which is generated before a build runs.
STORE_ONLY_COLUMNS: tuple[str, ...] = (
    "completed_against",
    "completion_median_pearson_r",
    "completion_n_imputed_total",
    "completion_n_missing_total",
    *EAF_ORIENTATION_COLUMNS,
    *TOP_HIT_COUNT_COLUMNS,
)

# A reference listing of known registry build-input columns (issue #9's
# worked examples), for callers that want to distinguish "a column the
# registry is known to use" from "a column nobody has documented yet."
# classify_column() itself does NOT consult this tuple: every column outside
# SHARED_CORE_COLUMNS and STORE_ONLY_COLUMNS classifies as registry-only
# regardless of whether it appears here, so an undocumented registry
# build-input column still passes through correctly (the superset property).
REGISTRY_ONLY_COLUMNS: tuple[str, ...] = (
    "source_analysis_id",
    "source_label",
    "trait_ontology_name",
    "trait_ontology_id",
    "source_file",
    "source_bundle_id",
    "checksum",
    "checksum_algorithm",
    "size_bytes",
    "source_genome_build",
    "license",
    "publication_doi",
    "publication_pmid",
    "consortium",
    "source_ancestry_label",
    "analysis_group_id",
    "inclusion_reason",
    "exclude_from_build",
)

# The subset of SHARED_CORE_COLUMNS that must be present -- as columns, not
# necessarily non-blank per-row values -- in any analyses.tsv, manifest or
# store. Excludes columns opengwasdb only knows how to populate once a build
# has assigned ancestry/phenotype identity (phenotype_id, phenotype_label,
# analysis_label, assigned_ancestry, ancestry_prop_*, analysis_index) and
# columns that are legitimately blank pending resolution (original_sd,
# original_sd_dispersion, n_cases, n_controls -- the latter two required only
# conditionally, see _validate_case_control_counts below).
REQUIRED_COLUMNS: tuple[str, ...] = (
    "analysis_id",
    "stored_effect_scale",
    "sample_size_kind",
    "sample_size_scope",
    "sample_size",
    "original_effect_scale",
    "original_sd_method",
    "ancestry_assignment_method",
)

_VOCABULARIES: dict[str, type[StrEnum]] = {
    "stored_effect_scale": StoredEffectScale,
    "sample_size_kind": SampleSizeKind,
    "sample_size_scope": SampleSizeScope,
    "original_sd_method": OriginalSdMethod,
    "ancestry_assignment_method": AncestryAssignmentMethod,
    "eaf_scope": EafScope,
    "eaf_orientation": EafOrientationOutcome,
}

_CASE_CONTROL_SCALE_VALUES = {StoredEffectScale.LOG_OR.value, StoredEffectScale.LOG_HAZARD.value}


class ColumnClass(StrEnum):
    """Which of the three column classes a column belongs to."""

    SHARED_CORE = "shared_core"
    REGISTRY_ONLY = "registry_only"
    STORE_ONLY = "store_only"


def classify_column(name: str) -> ColumnClass:
    """Classify `name` as shared-core, registry-only, or store-only.

    A column absent from every list above still classifies as registry-only:
    the superset property means a manifest may legitimately carry a
    build-input column this module has never been told about.
    """
    if name in SHARED_CORE_COLUMNS or name.startswith(ANCESTRY_PROP_PREFIX):
        return ColumnClass.SHARED_CORE
    if name in STORE_ONLY_COLUMNS:
        return ColumnClass.STORE_ONLY
    return ColumnClass.REGISTRY_ONLY


@dataclass(frozen=True)
class AnalysesTable:
    """A parsed `analyses.tsv`: column order plus raw string rows.

    Rows are kept as raw strings, like `opengwasdb.ancestry.catalogue`'s
    Catalogue reader -- `analyses.tsv` is inherently a text format, and typed
    access belongs to callers that know which columns they need, not to this
    module.
    """

    fieldnames: tuple[str, ...]
    rows: tuple[dict[str, str], ...]


def read_analyses(path: str | Path) -> AnalysesTable:
    """Read an `analyses.tsv` file, preserving column order and every field.

    Columns beyond the known contract (registry-only build inputs, or
    anything else) are read without error and carried through in each row's
    dict unchanged -- the superset property a manifest and a built store's
    analyses.tsv are related by.
    """
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = tuple(reader.fieldnames or ())
        rows = tuple(dict(row) for row in reader)
    return AnalysesTable(fieldnames=fieldnames, rows=rows)


def write_analyses(path: str | Path, table: AnalysesTable) -> Path:
    """Write `table` back to a TSV file, in its own column order."""
    out_path = Path(path)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table.fieldnames), delimiter="\t")
        writer.writeheader()
        for row in table.rows:
            writer.writerow(row)
    return out_path


def validate_analyses(table: AnalysesTable) -> list[str]:
    """Validate `table` against the analyses.tsv contract.

    Returns a list of error strings (empty means valid), following this
    package's existing convention (`opengwasdb.validation.validate`) of
    accumulating actionable errors rather than raising on the first problem.
    """
    errors: list[str] = []
    fieldnames = set(table.fieldnames)

    missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing:
        errors.append(f"analyses.tsv is missing required column(s): {', '.join(missing)}")

    for row in table.rows:
        analysis_id = row.get("analysis_id") or "<unknown analysis_id>"
        # A required column present in the header but blank on this row is
        # equally a missing-required-column failure from the row's point of
        # view -- it just cannot be caught by the header-level check above.
        for column in REQUIRED_COLUMNS:
            if column in fieldnames and not row.get(column, ""):
                errors.append(
                    f"analysis {analysis_id!r} has no value for required column {column!r}"
                )
        for column, vocabulary in _VOCABULARIES.items():
            if column not in fieldnames:
                continue
            value = row.get(column, "")
            if not value:
                continue
            try:
                vocabulary(value)
            except ValueError:
                allowed = [member.value for member in vocabulary]
                errors.append(
                    f"analysis {analysis_id!r} has invalid {column} {value!r}; "
                    f"expected one of {allowed}"
                )
        _validate_case_control_counts(row, analysis_id, fieldnames, errors)

    return errors


def _validate_case_control_counts(
    row: dict[str, str],
    analysis_id: str,
    fieldnames: set[str],
    errors: list[str],
) -> None:
    if row.get("stored_effect_scale", "") not in _CASE_CONTROL_SCALE_VALUES:
        return
    scale = row["stored_effect_scale"]
    for column in ("n_cases", "n_controls"):
        if column in fieldnames and row.get(column, ""):
            continue
        errors.append(f"analysis {analysis_id!r} has stored_effect_scale={scale!r} but no {column}")


def to_json_schema() -> dict[str, Any]:
    """A JSON Schema description of the analyses.tsv contract.

    The machine-readable artifact issue #16 asks for, so a consumer outside
    this package (`opengwasdb-stores`) can validate emitted manifests without
    vendoring or hand-copying the vocabulary. `additionalProperties: true`
    encodes the superset property directly: registry-only, store-only, and
    genuinely unrecognised columns are all schema-valid.
    """
    properties: dict[str, Any] = {}
    for column in (*SHARED_CORE_COLUMNS, *STORE_ONLY_COLUMNS):
        prop: dict[str, Any] = {"type": "string"}
        vocabulary = _VOCABULARIES.get(column)
        if vocabulary is not None:
            prop["enum"] = [member.value for member in vocabulary]
        properties[column] = prop
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "OpenGWASDB analyses.tsv",
        "description": "Analytical Metadata contract, store-format spec §7a.",
        "type": "object",
        "properties": properties,
        "required": list(REQUIRED_COLUMNS),
        "additionalProperties": True,
    }


# --- Shared Analysis Metadata model (ADR 0034/0035, issue #67) ------------
#
# `Analysis` is the single in-repo representation of one analyses.tsv row
# under the unified schema every Primary Storage Layout shares (ADR 0034,
# store-format spec §7a): no `phenotype_id`/`phenotype_label`/`trait_id`,
# Trait identity carried instead by `analysis_label`/`trait_ontology_id`/
# `trait_ontology_label` -- polymorphic-by-Trait-kind enough to also carry
# gene identity for gene-centric Analyses, so there is no separate
# `gene_id`/`gene_name` pair either (ADR 0035). Dense and Hybrid build/
# completion (issue #68) write
# against this model directly. SHARED_CORE_COLUMNS/STORE_ONLY_COLUMNS/
# REGISTRY_ONLY_COLUMNS above still describe the pre-ADR-0034 schema and are
# retained for `opengwasdb-stores` manifest validation
# (`classify_column`/`to_json_schema`) until that registry-side migration
# (out of scope here) lands -- Ragged's SQLite `analyses` table migration onto
# this model is issue #69 onward.

ANALYSIS_COLUMNS: tuple[str, ...] = (
    "analysis_index",
    "analysis_id",
    "analysis_label",
    "trait_ontology_id",
    "trait_ontology_label",
    "tissue",
    "context",
    "trait_chr",
    "trait_bp",
    "stored_effect_scale",
    "assigned_ancestry",
    "ancestry_assignment_method",
    "sample_size_kind",
    "sample_size_scope",
    "sample_size",
    "n_cases",
    "n_controls",
    "eaf_scope",
    "original_effect_scale",
    "original_sd",
    "original_sd_method",
    "original_sd_dispersion",
    "license",
    "publication_doi",
    "publication_pmid",
    "consortium",
    "first_author",
    "completed_against",
    "completion_median_pearson_r",
    "completion_n_imputed_total",
    "completion_n_missing_total",
    *EAF_ORIENTATION_COLUMNS,
    *TOP_HIT_COUNT_COLUMNS,
)

# Columns a prior `analyses.tsv` schema carried that this one deliberately
# drops -- `phenotype_id`/`phenotype_label`/`trait_id` (ADR 0034) and
# `gene_id`/`gene_name` (ADR 0035, both superseded by `analysis_label`/
# `trait_ontology_id`/`trait_ontology_label`). A built store's analyses.tsv
# carrying any of these is a validation failure (store-format spec §7a: "a
# Store Release built against a prior column list is not a valid Store
# Release against this one"), checked by
# `opengwasdb.validation.validate._validate_analyses_tsv`.
RETIRED_ANALYSIS_COLUMNS: tuple[str, ...] = (
    "phenotype_id",
    "phenotype_label",
    "trait_id",
    "gene_id",
    "gene_name",
)

_ANALYSIS_SAMPLE_SIZE_KIND_INDEX = ANALYSIS_COLUMNS.index("sample_size_kind")


@dataclass(frozen=True)
class Analysis:
    """One Analysis's Analytical + Attribution Metadata (store-format spec
    §7a, ADR 0034/0035) -- the schema every layout's `analyses.tsv` shares.

    Every field beyond `analysis_id` is optional and blank by default: most
    build paths genuinely have no source for ontology mapping, sample size,
    ancestry, or attribution -- writing "" for an unresolved column is
    honest, not a fabrication, matching this module's existing tolerance for
    blank shared-core columns.

    `analysis_index` is captured on read so a caller inspecting an existing
    table can see it, but `analyses_table_from_records` always reassigns it
    from each record's position when writing -- never from this field. An
    Analysis's position in the list *is* its `analysis_index`, which is also
    the column index a Dense/Hybrid store's z/se arrays are keyed by
    (`opengwasdb.layouts.dense.build`); honouring a caller-supplied value
    that had drifted from list order would silently desynchronise
    `analyses.tsv` from those arrays, which is worse than the writer being
    authoritative here the way it already is for `analysis_index` on every
    fresh build.
    """

    analysis_id: str
    analysis_index: str = ""
    analysis_label: str = ""
    trait_ontology_id: str = ""
    trait_ontology_label: str = ""
    tissue: str = ""
    context: str = ""
    trait_chr: str = ""
    trait_bp: str = ""
    stored_effect_scale: str = ""
    assigned_ancestry: str = ""
    ancestry_assignment_method: str = ""
    ancestry_prop: dict[str, str] = field(default_factory=dict)  # {population: proportion}
    sample_size_kind: str = ""
    sample_size_scope: str = ""
    sample_size: str = ""
    n_cases: str = ""
    n_controls: str = ""
    # How to read this Analysis's `eaf` cells (ADR 0036, spec §9): "absent" or
    # "" when the build stored none, "association" when it stored a per-cell
    # value. Derived by the builder from what it actually wrote -- never copied
    # from a manifest, which cannot know what the source file turned out to
    # contain.
    eaf_scope: str = ""
    original_effect_scale: str = ""
    original_sd: str = ""
    original_sd_method: str = ""
    original_sd_dispersion: str = ""
    license: str = ""
    publication_doi: str = ""
    publication_pmid: str = ""
    consortium: str = ""
    first_author: str = ""
    completed_against: str = ""
    completion_median_pearson_r: str = ""
    completion_n_imputed_total: str = ""
    completion_n_missing_total: str = ""
    # EAF orientation evidence (issue #115). Blank on an Analysis no build-time
    # check ever ran over -- which validation treats as a store that predates
    # the check, not as one that passed it.
    eaf_orientation: str = ""
    eaf_orientation_r: str = ""
    eaf_orientation_n: str = ""
    n_hits_5e8: str = ""
    n_hits_5e6: str = ""
    n_hits_5e4: str = ""

    def __post_init__(self) -> None:
        if not self.analysis_id:
            raise ValueError("Analysis.analysis_id must not be blank")

# Shared-core `analyses.tsv` columns every layout's build manifest may supply
# and no builder may interpret: the manifest producer owns the fact, the
# builder copies it through verbatim (issue #86 for Dense/Hybrid, #83 for
# Ragged). Kept as one list so a column added here reaches every layout at
# once, rather than each builder growing its own drifting copy -- the
# divergence issue #83 exists to close.
PASSTHROUGH_MANIFEST_COLUMNS: tuple[str, ...] = (
    "ancestry_assignment_method",
    "sample_size_kind",
    "sample_size_scope",
    "n_cases",
    "n_controls",
    "original_effect_scale",
    "original_sd_method",
    "license",
    "publication_doi",
    "publication_pmid",
    "consortium",
    "first_author",
)


@dataclass(frozen=True)
class PassthroughMetadata:
    """The `PASSTHROUGH_MANIFEST_COLUMNS` values one manifest row carries.

    One type rather than a dozen loose parameters threaded through each
    builder: these columns always travel together, are always strings, and
    are always copied rather than computed. `ancestry_prop` rides along
    because it is shared-core too (`ANCESTRY_PROP_PREFIX`) and equally
    uninterpreted -- it is just dict-shaped rather than scalar, one column
    per reference population.

    Every field defaults to blank. A manifest that omits a column produces a
    blank value in the built store's `analyses.tsv`, which is honest: only
    the manifest producer knows how ancestry was assigned or what licence
    the source carries, so a builder that guessed would be fabricating.
    """

    ancestry_assignment_method: str = ""
    sample_size_kind: str = ""
    sample_size_scope: str = ""
    n_cases: str = ""
    n_controls: str = ""
    original_effect_scale: str = ""
    original_sd_method: str = ""
    license: str = ""
    publication_doi: str = ""
    publication_pmid: str = ""
    consortium: str = ""
    first_author: str = ""
    ancestry_prop: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_manifest_row(cls, row: Mapping[str, str]) -> PassthroughMetadata:
        """Read the passthrough columns out of one raw manifest row."""
        return cls(
            **{column: row.get(column) or "" for column in PASSTHROUGH_MANIFEST_COLUMNS},
            ancestry_prop={
                key[len(ANCESTRY_PROP_PREFIX):]: value
                for key, value in row.items()
                if key.startswith(ANCESTRY_PROP_PREFIX) and value
            },
        )

    def applied_to(self, analysis: Analysis) -> Analysis:
        """`analysis` with these columns filled in."""
        return replace(analysis, **asdict(self))


def _analysis_fieldnames(analyses: list[Analysis]) -> tuple[str, ...]:
    prop_populations = sorted({pop for a in analyses for pop in a.ancestry_prop})
    return (
        *ANALYSIS_COLUMNS[:_ANALYSIS_SAMPLE_SIZE_KIND_INDEX],
        *(f"{ANCESTRY_PROP_PREFIX}{pop}" for pop in prop_populations),
        *ANALYSIS_COLUMNS[_ANALYSIS_SAMPLE_SIZE_KIND_INDEX:],
    )


def _analysis_row(index: int, a: Analysis, fieldnames: tuple[str, ...]) -> dict[str, str]:
    # `index`, not `a.analysis_index`: the writer is authoritative on
    # position (see Analysis's docstring for why honouring a caller-supplied
    # value would be unsafe).
    row = {
        "analysis_index": str(index),
        "analysis_id": a.analysis_id,
        "analysis_label": a.analysis_label,
        "trait_ontology_id": a.trait_ontology_id,
        "trait_ontology_label": a.trait_ontology_label,
        "tissue": a.tissue,
        "context": a.context,
        "trait_chr": a.trait_chr,
        "trait_bp": a.trait_bp,
        "stored_effect_scale": a.stored_effect_scale,
        "assigned_ancestry": a.assigned_ancestry,
        "ancestry_assignment_method": a.ancestry_assignment_method,
        "sample_size_kind": a.sample_size_kind,
        "sample_size_scope": a.sample_size_scope,
        "sample_size": a.sample_size,
        "n_cases": a.n_cases,
        "n_controls": a.n_controls,
        "eaf_scope": a.eaf_scope,
        "original_effect_scale": a.original_effect_scale,
        "original_sd": a.original_sd,
        "original_sd_method": a.original_sd_method,
        "original_sd_dispersion": a.original_sd_dispersion,
        "license": a.license,
        "publication_doi": a.publication_doi,
        "publication_pmid": a.publication_pmid,
        "consortium": a.consortium,
        "first_author": a.first_author,
        "completed_against": a.completed_against,
        "completion_median_pearson_r": a.completion_median_pearson_r,
        "completion_n_imputed_total": a.completion_n_imputed_total,
        "completion_n_missing_total": a.completion_n_missing_total,
        "eaf_orientation": a.eaf_orientation,
        "eaf_orientation_r": a.eaf_orientation_r,
        "eaf_orientation_n": a.eaf_orientation_n,
        "n_hits_5e8": a.n_hits_5e8,
        "n_hits_5e6": a.n_hits_5e6,
        "n_hits_5e4": a.n_hits_5e4,
    }
    for column in fieldnames:
        if column.startswith(ANCESTRY_PROP_PREFIX):
            row[column] = a.ancestry_prop.get(column[len(ANCESTRY_PROP_PREFIX):], "")
    return row


def _analysis_from_row(row: dict[str, str], fieldnames: tuple[str, ...]) -> Analysis:
    return Analysis(
        analysis_id=row["analysis_id"],
        analysis_index=row.get("analysis_index", ""),
        analysis_label=row.get("analysis_label", ""),
        trait_ontology_id=row.get("trait_ontology_id", ""),
        trait_ontology_label=row.get("trait_ontology_label", ""),
        tissue=row.get("tissue", ""),
        context=row.get("context", ""),
        trait_chr=row.get("trait_chr", ""),
        trait_bp=row.get("trait_bp", ""),
        stored_effect_scale=row.get("stored_effect_scale", ""),
        assigned_ancestry=row.get("assigned_ancestry", ""),
        ancestry_assignment_method=row.get("ancestry_assignment_method", ""),
        ancestry_prop={
            key[len(ANCESTRY_PROP_PREFIX):]: row.get(key, "")
            for key in fieldnames
            if key.startswith(ANCESTRY_PROP_PREFIX)
        },
        sample_size_kind=row.get("sample_size_kind", ""),
        sample_size_scope=row.get("sample_size_scope", ""),
        sample_size=row.get("sample_size", ""),
        n_cases=row.get("n_cases", ""),
        n_controls=row.get("n_controls", ""),
        eaf_scope=row.get("eaf_scope", ""),
        original_effect_scale=row.get("original_effect_scale", ""),
        original_sd=row.get("original_sd", ""),
        original_sd_method=row.get("original_sd_method", ""),
        original_sd_dispersion=row.get("original_sd_dispersion", ""),
        license=row.get("license", ""),
        publication_doi=row.get("publication_doi", ""),
        publication_pmid=row.get("publication_pmid", ""),
        consortium=row.get("consortium", ""),
        first_author=row.get("first_author", ""),
        completed_against=row.get("completed_against", ""),
        completion_median_pearson_r=row.get("completion_median_pearson_r", ""),
        completion_n_imputed_total=row.get("completion_n_imputed_total", ""),
        completion_n_missing_total=row.get("completion_n_missing_total", ""),
        eaf_orientation=row.get("eaf_orientation", ""),
        eaf_orientation_r=row.get("eaf_orientation_r", ""),
        eaf_orientation_n=row.get("eaf_orientation_n", ""),
        n_hits_5e8=row.get("n_hits_5e8", ""),
        n_hits_5e6=row.get("n_hits_5e6", ""),
        n_hits_5e4=row.get("n_hits_5e4", ""),
    )


def analyses_table_from_records(analyses: list[Analysis]) -> AnalysesTable:
    """Build an `AnalysesTable` from `Analysis` records, assigning
    `analysis_index` by each record's position in `analyses`.

    Rejects a duplicate `analysis_id` here rather than leaving every caller
    to enforce store-format spec §7a's "`analysis_id` MUST be unique within
    a Store Release" independently.
    """
    seen: set[str] = set()
    for a in analyses:
        if a.analysis_id in seen:
            raise ValueError(f"duplicate analysis_id: {a.analysis_id!r}")
        seen.add(a.analysis_id)

    fieldnames = _analysis_fieldnames(analyses)
    return AnalysesTable(
        fieldnames=fieldnames,
        rows=tuple(_analysis_row(i, a, fieldnames) for i, a in enumerate(analyses)),
    )


def write_analysis_records(path: str | Path, analyses: list[Analysis]) -> Path:
    """Write `analyses` to `path` as an `analyses.tsv` file."""
    return write_analyses(path, analyses_table_from_records(analyses))


def read_analysis_records(path: str | Path) -> list[Analysis]:
    """Read `analyses.tsv` at `path` back into `Analysis` records, in file order."""
    table = read_analyses(path)
    return [_analysis_from_row(row, table.fieldnames) for row in table.rows]
