"""The provenance an LD Store carries, and the names of its validation rules (spec §7, §12).

The types mirror the JSON object in the root group's `ld_store` attribute key for key, so
mypy holds the builder (ogld#5), the writer (ogld#4) and validation to one shape.
`REQUIRED_KEYS` and `VALIDATION_RULES` are the spec's two tables as data; a test holds
each to its table, so neither can drift from the spec silently.
"""

from __future__ import annotations

from typing import Final, Literal, TypedDict

FORMAT_VERSION: Final = "0.2.0"

#: The format versions this package reads: `0.1.0`, the stores built before the Partner Rows (no
#: `partners/`, and `clump` and `proxies` read the Proxy Index), and `0.2.0`, which adds them
#: (spec §13). A store records one, and a reader refuses any other.
READABLE_FORMATS: Final = ("0.1.0", FORMAT_VERSION)


class Encoding(TypedDict):
    """How the store's `r` arrays hold r (spec §3, §7). A store may record any values here;
    `ogld.format.REncoding.of` says whether this package implements them, and a reader refuses
    a store whose encoding it does not (spec §13)."""

    r_dtype: str
    r_scale: int
    r_reserved_code: int
    rounding: str


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
    """One PC of the PC File, as the PC Check scanned it (`docs/spec/build-inputs.md` §8).

    `window` is the 4 Mb window that explains most of the PC, as `chrom:start-end`, and
    `window_share` how much of it; `rest_share` is the mean share over every window that
    does not overlap it, and `ratio` is rest_share / window_share. `window_null_q99` is the
    99th percentile of the largest window share of the PC permuted across samples, the
    threshold the window share is judged against; `rest_null_q99` is reported, and decides
    nothing. `local` is the verdict: the PC is dropped.
    """

    pc: str
    window: str
    window_share: float
    rest_share: float
    ratio: float
    local: bool
    window_null_q99: float
    rest_null_q99: float


class PcNull(TypedDict):
    """How the PC Check's chance was made (`docs/spec/build-inputs.md` §8): `kind` says what was
    permuted, and `permutations`, `seed` and `quantile` how (200, 405 and 0.99 by the rule)."""

    kind: str
    permutations: int
    seed: int
    quantile: float


class PcCheck(TypedDict):
    """The PC Check's result: the local-PC rule (owner, 7 November 2026; ADR 0002).

    A PC is local, and dropped from the covariates, when all three hold: one `window_bp`
    window explains at least `min_share` of it; that share beats `PcScan.window_null_q99`;
    and `PcScan.ratio` ≤ `max_ratio`. `scan` holds every PC of the PC File, in order,
    `dropped` the local ones, and `null` how chance was made (kind, permutations, seed,
    quantile). `step_bp` and `windows` describe the scan: windows start every `step_bp`, and
    `windows` were scanned.
    """

    window_bp: int
    step_bp: int
    windows: int
    min_share: float
    max_ratio: float
    scan: list[PcScan]
    dropped: list[str]
    null: PcNull


class Adjustment(TypedDict):
    """What was regressed out of the genotypes before r was computed (spec §7).

    `covariates` names every covariate, in order, and `k` is their number: t_far's k. For a
    build from a PC File they are the first up to `k_target` PCs the PC Check kept, and
    `k_min` ≤ k ≤ `k_target`, since a build with fewer than `k_min` is refused. With no PC
    File read, `k_min` does not apply and is None: an unadjusted build records k =
    `k_target` = 0. An imported LD matrix records its source's covariates and their count
    as k, and `k_target` and `k_min` are None.
    """

    covariates: list[str]
    k: int
    k_target: int | None
    k_min: int | None
    pc_file: PcFile | None
    pc_check: PcCheck | None


class PcDiagnosticPairs(TypedDict):
    source: str
    seed: int
    n_pairs: int


class PcDiagnostic(TypedDict):
    """What PC Adjustment did to chance LD and to real LD (spec §7). The two measured rates are
    None when the Variant Set has no cross-chromosome pair (one chromosome), and `pairs.n_pairs`
    is then 0: a rate nothing measured is not a number."""

    pairs: PcDiagnosticPairs
    rate_r2_ge_0_01_unadjusted: float | None
    rate_r2_ge_0_01_adjusted: float | None
    rate_r2_ge_0_01_sampling: float
    near_100kb_change: dict[str, float]


class VariantFilters(TypedDict):
    """The build's variant filters and their values (`docs/spec/build-inputs.md` §6).

    `filter` lists the FILTER values that pass (`PASS` and `.`); `mac_min` is the minor
    allele count floor; `info_min` the INFO floor, None when none was applied (a sequenced
    panel); `autosomes_only` that a record off chromosomes 1-22 was refused (spec §1).
    """

    filter: list[str]
    mac_min: int
    info_min: float | None
    autosomes_only: bool


class Counts(TypedDict):
    near_pairs: int
    far_candidates: int
    far_pairs: int
    partner_entries: int
    index_entries: int
    alids_left_out: int


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
    variant_filters: VariantFilters
    variants_dropped: dict[str, int]
    n_variants: int
    near_bp: int
    far_bp: int
    adjustment: Adjustment
    t_far: float
    t_far_q: float
    far_code_floor: int
    t_partner: float
    partner_code_floor: int
    pc_diagnostic: PcDiagnostic | None
    counts: Counts
    software: dict[str, str]
    build: Build


class Sourced(TypedDict):
    """The provenance only a store's producer knows (spec §7; owner, 10 November 2026): what r
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
    "t_partner",
    "partner_code_floor",
    "pc_diagnostic",
    "counts.near_pairs",
    "counts.far_candidates",
    "counts.far_pairs",
    "counts.partner_entries",
    "counts.index_entries",
    "counts.alids_left_out",
    "software",
    "software.zstd",
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
    "alid_index.rows",
    "alid_index.order",
    "alid_index.keys",
    "alid_index.count",
    "near.rows",
    "near.codes",
    "index.rows",
    "index.codes",
    "index.distance",
    "index.threshold",
    "index.symmetric",
    "index.near",
    "far.rows",
    "far.codes",
    "far.distance",
    "far.threshold",
    "far.index",
)
