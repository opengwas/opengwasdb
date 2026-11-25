"""The provenance an LD Store carries, and the names of its validation rules (ld-store §7, §12).

The types mirror the JSON object in the root group's `ld_store` attribute key for key, so
mypy holds the builder (ogld#5), the writer (ogld#4) and validation to one shape.
`REQUIRED_KEYS` and `VALIDATION_RULES` are the spec's two tables as data; a test holds
each to its table, so neither can drift from the spec silently.
"""

from __future__ import annotations
from typing import Final, Literal, TypedDict

# The read side of this module is ogquery's (opengwas/ogquery#2): re-exported by name.
from ogquery.ld.format.provenance import (
    Adjustment as Adjustment,
    Build as Build,
    claims_partner_rows as claims_partner_rows,
    Counts as Counts,
    Encoding as Encoding,
    FORMAT_VERSION as FORMAT_VERSION,
    Panel as Panel,
    PARTNER_ROWS_SINCE as PARTNER_ROWS_SINCE,
    PcCheck as PcCheck,
    PcDiagnostic as PcDiagnostic,
    PcDiagnosticPairs as PcDiagnosticPairs,
    PcFile as PcFile,
    PcNull as PcNull,
    PcScan as PcScan,
    Provenance as Provenance,
    READABLE_FORMATS as READABLE_FORMATS,
    release_of as release_of,
    Source as Source,
    SourceKind as SourceKind,
    VariantFilters as VariantFilters,
)


class Sourced(TypedDict):
    """The provenance only a store's producer knows (ld-store §7; owner, 10 November 2026): what r
    was computed from and how, by what software, and at what cost.

    A producer hands this to `StoreWriter.finish`, which adds what the store derives from itself:
    `format_version`, `encoding`, `n_variants`, `near_bp`, `far_bp`, `t_far`, `t_far_q`,
    `far_code_floor`, `t_partner`, `partner_code_floor` and `counts`. Together they are
    `Provenance`, key for key. `software` is the producer's list of versions, to which the store
    adds `zstd`, the library that compressed its Partner Rows' frames.
    """

    source: Source
    panel: Panel
    reference_assembly: str
    variant_filters: VariantFilters
    variants_dropped: dict[str, int]
    adjustment: Adjustment
    pc_diagnostic: PcDiagnostic | None
    software: dict[str, str]
    build: Build


#: Every key ld-store §7 requires, dotted where it is nested. `provenance.complete` checks them.
REQUIRED_KEYS: Final = (
    "format_version",
    "encoding.r_dtype",
    "encoding.r_scale",
    "encoding.r_reserved_code",
    "encoding.rounding",
    "source.kind",
    "source.description",
    "source.sha256",
    "panel.id",
    "panel.ancestry",
    "panel.n",
    "panel.samples_sha256",
    "reference_assembly",
    "variant_filters",
    "variants_dropped",
    "n_variants",
    "near_bp",
    "far_bp",
    "adjustment.covariates",
    "adjustment.k",
    "adjustment.k_target",
    "adjustment.k_min",
    "adjustment.pc_file",
    "adjustment.pc_check",
    "t_far",
    "t_far_q",
    "far_code_floor",
    "t_partner",
    "partner_code_floor",
    "pc_diagnostic",
    "counts.near_pairs",
    "counts.far_candidates",
    "counts.far_pairs",
    "counts.partner_entries",
    "counts.alids_left_out",
    "software",
    "software.zstd",
    "build.created_at",
    "build.wall_seconds",
    "build.peak_memory_bytes",
    "build.threads",
)


#: The validation rules, by name, in the order ld-store §12 lists them. Each runs on write
#: and on demand; ogld#4 implements them, each with a store that breaks it.
VALIDATION_RULES: Final = (
    "envelope.arrays",
    "envelope.fill",
    "provenance.complete",
    "provenance.encoding",
    "provenance.distances",
    "provenance.threshold",
    "variants.chromosomes",
    "variants.order",
    "variants.identity",
    "variants.frequency",
    "alid_index.rows",
    "alid_index.order",
    "alid_index.keys",
    "alid_index.count",
    "near.rows",
    "near.codes",
    "far.rows",
    "far.codes",
    "far.distance",
    "far.threshold",
    "partners.files",
    "partners.frames",
    "partners.rows",
    "partners.codes",
    "partners.distance",
    "partners.threshold",
    "partners.symmetric",
    "partners.near",
    "partners.far",
    "provenance.partner_threshold",
)
