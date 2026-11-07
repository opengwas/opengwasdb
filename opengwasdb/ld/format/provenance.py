"""The provenance an LD Store carries, and the names of its validation rules (spec §7, §12).

The types mirror the JSON object in the root group's `ld_store` attribute key for key, so
mypy holds the builder (ogld#5), the writer (ogld#4) and validation to one shape.
`REQUIRED_KEYS` and `VALIDATION_RULES` are the spec's two tables as data; a test holds
each to its table, so neither can drift from the spec silently.
"""

from __future__ import annotations

from typing import Final, Literal, TypedDict

from ogld.format.encoding import R_SCALE, RESERVED_CODE

FORMAT_VERSION: Final = "0.1.0"


class Encoding(TypedDict):
    r_dtype: Literal["int16"]
    r_scale: int
    r_reserved_code: int
    rounding: Literal["half_to_even"]


#: The only encoding this package reads or writes (spec §3). A reader refuses any other.
ENCODING: Final[Encoding] = {
    "r_dtype": "int16",
    "r_scale": R_SCALE,
    "r_reserved_code": RESERVED_CODE,
    "rounding": "half_to_even",
}

SourceKind = Literal["genotypes", "ld_matrix"]


class Source(TypedDict):
    """What the LD was computed from: a Source Panel's genotypes, or an imported LD matrix."""

    kind: SourceKind
    description: str
    sha256: dict[str, str]


class Panel(TypedDict):
    id: str
    ancestry: str
    n: int
    samples_sha256: str | None


class PcFile(TypedDict):
    sha256: str
    source: str


class PcScan(TypedDict):
    """One PC of the PC File, as the PC Check scanned it.

    `window` is the 4 Mb window that explains most of the PC and `window_share` how much
    of it; `rest_share` is how much the rest of the genome explains, and `rest_null` how
    much it explains by chance. `local` is window_share ≥ min_share and rest_share ≤
    rest_null: the PC is dropped. `docs/spec/build-inputs.md` defines the statistics.
    """

    pc: str
    window: str
    window_share: float
    rest_share: float
    rest_null: float
    local: bool


class PcCheck(TypedDict):
    """The PC Check's result (owner, 7 November 2026; ADR 0002).

    A PC is local, and dropped from the covariates, when one `window_bp` window explains
    at least `min_share` of it and the rest of the genome explains it no better than
    chance (`PcScan.rest_null`); `null` records how chance was measured. `scan` holds every
    PC of the PC File, in order. A build with fewer than k PCs kept is refused.
    """

    window_bp: int
    min_share: float
    scan: list[PcScan]
    dropped: list[str]
    null: dict[str, float | str]


class Adjustment(TypedDict):
    """What was regressed out of the genotypes before r was computed (PC Adjustment).

    `covariates` are the PCs the PC Check kept, by name, the first up to `k_target`;
    `k` is their number, and it is t_far's k. A build with fewer than `k_min` is refused.
    """

    covariates: list[str]
    k: int
    k_target: int
    k_min: int
    pc_file: PcFile | None
    pc_check: PcCheck | None


class PcDiagnosticPairs(TypedDict):
    source: str
    seed: int
    n_pairs: int


class PcDiagnostic(TypedDict):
    pairs: PcDiagnosticPairs
    rate_r2_ge_0_01_unadjusted: float
    rate_r2_ge_0_01_adjusted: float
    rate_r2_ge_0_01_sampling: float
    near_100kb_change: dict[str, float]


class Counts(TypedDict):
    near_pairs: int
    far_candidates: int
    far_pairs: int
    index_entries: int


class Build(TypedDict):
    created_at: str
    wall_seconds: float
    peak_memory_bytes: int
    threads: int


class Provenance(TypedDict):
    """The root group's `ld_store` attribute (spec §7). Every key is required."""

    format_version: str
    encoding: Encoding
    source: Source
    panel: Panel
    reference_assembly: str
    variant_filters: dict[str, int | float | None]
    variants_dropped: dict[str, int]
    n_variants: int
    near_bp: int
    far_bp: int
    adjustment: Adjustment
    t_far: float
    t_far_q: float
    far_code_floor: int
    pc_diagnostic: PcDiagnostic | None
    counts: Counts
    software: dict[str, str]
    build: Build


#: Every key spec §7 requires, dotted where it is nested. `provenance.complete` checks them.
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
    "pc_diagnostic",
    "counts.near_pairs",
    "counts.far_candidates",
    "counts.far_pairs",
    "counts.index_entries",
    "software",
    "build.created_at",
    "build.wall_seconds",
    "build.peak_memory_bytes",
    "build.threads",
)

#: The validation rules, by name, in the order spec §12 lists them. Each runs on write
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
    "near.rows",
    "near.codes",
    "index.rows",
    "index.codes",
    "index.distance",
    "index.threshold",
    "index.symmetric",
    "index.near",
)
