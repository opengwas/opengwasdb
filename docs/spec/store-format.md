# OpenGWASDB Store Format Specification

Status: draft  
Format version described: `0.1`

This document defines the contract for valid OpenGWASDB Store Releases. The v0.1 implementation target is **Dense Observed-Only**, but the specification also records accepted design semantics for Ragged layout and Reference-Completed releases so the first implementation does not block later extensions.

Normative language:

- **MUST** indicates a requirement for a valid store.
- **SHOULD** indicates a recommended default that implementations may override with documented reason.
- **MAY** indicates an optional feature.

## 1. Store release envelope

A Store Release is a self-contained directory.

```text
manifest.json
index.sqlite
analyses.tsv
overview.html
data.zarr/
variants.tsv.gz
variants.tsv.gz.tbi
variant_offsets.npy
variant_alid_bytes.npy
variant_alid_rows.npy
variant_rsid_bytes.npy
variant_rsid_rows.npy
```

`manifest.json` identifies the release and declares how to interpret it.
`index.sqlite` stores compact relational metadata such as key/value metadata,
small alias maps, and large fine-grained tables that are not practical as flat
text (such as Reference Completion Quality — see §12). `analyses.tsv` is the
sole source of truth for Analytical Metadata (§7a) — one row per Analysis;
`index.sqlite` MUST NOT also contain an `analyses` table. `overview.html` is a
generated, store-wide human-browsable rendering of `analyses.tsv`; it MAY be
regenerated from `analyses.tsv` and MUST NOT be treated as a second source of
truth — Ragged releases do not carry one (§11). `data.zarr/` stores compressed
numerical association arrays and layout-specific numerical indexes. Every
layout stores its high-cardinality variant axis in `variants.tsv.gz`, indexed
by `variants.tsv.gz.tbi`, with `variant_offsets.npy` mapping Store-local
Variant Indices to BGZF row offsets, `variant_alid_bytes.npy`/
`variant_alid_rows.npy` supporting ALID lookup, and
`variant_rsid_bytes.npy`/`variant_rsid_rows.npy` supporting rsid lookup
(issue #109). The rsid pair is written by every layout and MAY be empty — a
source that names no variants indexes none — but a release whose
`variants.tsv.gz` carries rsids and whose index does not is invalid (§20):
every rsid lookup against such a release returns an empty result
indistinguishable from a real absence.

This is the complete envelope for a standalone Dense or Ragged Store Release.
A Hybrid release additionally nests a `dense/` Dense Component directory
holding a self-contained Dense Store Release of its own (same envelope as
above, plus `dense_to_shared.npy` mapping its rows into the Hybrid release's
shared variant table) — see §16. §20 requires every Store Release's
directory to be closed: no file or directory beyond what its `primary_layout`
legitimately produces per this section and §10/§11/§16/§17.

Build and query commands operate on an explicit Store Release path. Directory naming, multi-store catalogues, default release selection, and remote API deployment are outside the store-format contract.

## 2. Manifest contract

Every Store Release MUST contain `manifest.json`.

Required fields:

| Field | Meaning |
|---|---|
| `store_id` | Stable identity of the logical Store |
| `release_id` | Identity of this immutable release |
| `format_version` | Store format version |
| `primary_layout` | `dense`, `ragged`, or `hybrid` |
| `association_coverage` | `full` or `cis_and_signals` |
| `completion_state` | `observed_only` or `reference_completed` |
| `reference_assembly` | One genome assembly for all coordinates in the release |
| `created_at` | Release creation timestamp |
| `provenance` | Source and build provenance object |

Reference-Completed releases MUST additionally declare:

| Field | Meaning |
|---|---|
| `ld_reference_panel` | The panel defining the Reference Variant Set and LD resources |
| `reference_completion_method` | Algorithm, software, version, and parameters used for imputation |

Observed-Only and Reference-Completed releases for the same source collection SHOULD share `store_id` and use different `release_id` values.

Published releases are immutable. Enhancing an Observed-Only release to Reference-Completed produces a new release.

## 3. Identity and terminology

An **Analysis** is one statistical analysis of one Trait. An Analysis produces associations between that Trait and variants.

A **Trait** is a measured or derived outcome. `phenotype` is treated as a synonym in user-facing descriptions, but the canonical term is Trait.

A Store Release contains one or more Analyses. Analysis metadata MUST be sufficient to interpret stored effect scales, sample-size semantics, and source provenance.

## 4. Variant identity and variant table

Each Store Release MUST contain a Store Variant Table.

The canonical within-Store variant key is ALID:

```text
chr:pos:A1:A2
```

where:

- coordinates are on the release's declared Reference Assembly;
- alleles are trimmed and left-aligned before identity assignment;
- `A1` is alphabetically first;
- `A2` is the other allele.

Cross-Store identity is:

```text
reference_assembly + ALID
```

`rsid` is an alias, not primary identity.

One rsid MAY name several Store-local Variant Indices — a multi-allelic site,
or one position stored under both allele orders — so an rsid is not a key.
Resolution therefore has two contracts (issue #109): an API returning a *set*
of variants MUST return every row the rsid names, and an API whose contract is
a single variant MUST resolve to the lowest Store-local Variant Index among
them. Neither may silently pick an arbitrary one.

Every Store Release assigns compact Store-local Variant Indices. Variant Indices MUST NOT be assumed stable across releases or stores.

Dense Observed-Only releases use a tabix-backed Store Variant Table:

```text
variants.tsv.gz
variants.tsv.gz.tbi
variant_offsets.npy
variant_alid_bytes.npy
variant_alid_rows.npy
variant_rsid_bytes.npy
variant_rsid_rows.npy
```

The `variants.tsv.gz` table MUST contain one row per Store-local Variant Index,
sorted by chromosome, position, effect allele, other allele, and variant index.
The v0.1 dense column contract is:

```text
chromosome
position
variant_index
effect_allele
other_allele
alid
rsid
```

`variants.tsv.gz.tbi` MUST index chromosome and position for genomic range and
single-position lookup. `variant_offsets.npy` MUST contain one fixed-width
integer offset per variant row so row-index materialisation does not require a
large SQL variant table.

Long alleles MAY use deterministic hashed ALIDs as compact identifiers, but the complete normalised alleles MUST be retained once per variant so exact export and validation do not depend on an irreversible hash.

Reference-Completed releases MUST expose Reference Panel Membership as variant metadata, distinguishing Reference Variant Set variants from observed off-panel variants.

## 5. Allele orientation and signed statistics

Every association MUST be normalised to the Store's canonical allele orientation.

If the source effect allele is not canonical `A1`, the builder MUST swap orientation and negate signed statistics. Z is signed and carries effect direction. SE is non-negative.

Derived beta is:

```text
beta = z * se
```

## 6. Stored statistics

OpenGWASDB stores the canonical statistic pair:

```text
z
se
```

Requirements:

- `z` is signed.
- `se` is non-negative.
- `se` is on the same Stored Effect Scale as beta.
- beta is queryable but derived from `z * se`.
- p-value is queryable but derived from Z.
- EAF, INFO, and sample size MUST NOT be required to reconstruct beta, SE, Z, or p-value.

The v0.1 target dtype for dense statistics is `float16`, subject to validation benchmarks.

## 7. Effect scale

Each Analysis MUST declare Stored Effect Scale from the controlled vocabulary:

```text
sd
log_or
log_hazard
```

There is no `other`, `unknown`, or original-units Stored Effect Scale in v0.1. Unsupported stored scales MUST fail ingestion until the vocabulary is deliberately extended.

Original Effect Scale MAY be recorded as free-text provenance. For continuous traits, builders SHOULD store effects in SD Units when phenotype standard deviation is available or can be derived with acceptable provenance, by rescaling the source statistics with a per-study phenotype SD rather than reconstructing them from Z, N, and allele frequency (ADR 0029). The method used to obtain that SD (`original_sd_method`) and a dispersion diagnostic over its supporting evidence MUST be recorded as Analytical Metadata (§7a).

## 7a. Analytical Metadata, Attribution Metadata, and `analyses.tsv`

A Store Release MUST carry its own Analytical Metadata — metadata affecting the
interpretation of association statistics — and its own Attribution Metadata —
metadata establishing how an Analysis may be cited, licensed, and attributed —
so a downloaded or mirrored copy remains usable and interpretable without a
catalogue service (ADR 0030, ADR 0034). Both live entirely in `analyses.tsv`,
one row per Analysis, keyed by `analysis_index`, **identically across every
Primary Storage Layout** (ADR 0034) — Ragged is not exempt, and `index.sqlite`
MUST NOT hold a second, layout-specific per-Analysis metadata table. This
column list is a **breaking revision** of the one ADR 0030 originally
specified (ADR 0034, further narrowed by ADR 0035's retirement of `gene_id`/
`gene_name`): a Store Release built against a prior column list is not a
valid Store Release against this one, and no reader is required to accept
both.

```text
analysis_index
analysis_id
analysis_label
trait_ontology_id            (CURIE, e.g. EFO:0001073; blank when unmapped)
trait_ontology_label
tissue
context
trait_chr                    (blank for Analyses with no single genomic position)
trait_bp
stored_effect_scale
assigned_ancestry
ancestry_assignment_method
ancestry_prop_<population>   (repeated, one column per reference population)
sample_size_kind
sample_size_scope
sample_size
n_cases
n_controls
eaf_scope                    (see §9; blank or `absent` when the build stored no EAF)
original_effect_scale
original_sd
original_sd_method
original_sd_dispersion
license
publication_doi
publication_pmid
consortium
first_author
completed_against   (Reference-Completed releases only; null when unimputed)
completion_median_pearson_r   (Reference-Completed releases only)
completion_n_imputed_total
completion_n_missing_total
eaf_orientation      (see §9.1; `passed`, `failed` or `unverified`; blank only on a store built before the check existed)
eaf_orientation_r    (the observed correlation; blank when none was computed)
eaf_orientation_n    (variants the correlation was computed over)
n_hits_5e8   (count of associations at p <= 5e-8, genome-wide significant)
n_hits_5e6   (count of associations at p <= 5e-6, suggestive)
n_hits_5e4   (count of associations at p <= 5e-4, nominal)
```

Every column beyond `analysis_index`/`analysis_id` MAY be blank for an
individual row when genuinely unknown or not applicable — a phenotype-level
Analysis leaves `trait_chr`/`trait_bp` blank; a Trait with no clean ontology
mapping yet leaves `trait_ontology_id`/`trait_ontology_label` blank rather
than fabricating one. `trait_ontology_id` is not required to be unique — one
Trait MAY have several Analyses (differing by cohort, ancestry, model, sample
subset, or meta-analysis) that share it, and it MAY be blank on some or all of
them. There is deliberately no other Trait-identifying column: a raw,
source-native trait identifier alongside `trait_ontology_id` was considered
and rejected (ADR 0034) as recreating the same "which column is authoritative"
ambiguity `phenotype_id` had. This extends to gene identity for gene-centric
(pQTL) Analyses: `trait_ontology_id`'s CURIE contract is polymorphic by Trait
kind, so an Ensembl gene ID (e.g. `ENSEMBL:ENSG00000152256`) belongs there and
in `trait_ontology_label` (`"Ensembl"`) rather than in dedicated `gene_id`/
`gene_name` columns, with the gene symbol carried as free text in
`analysis_label` (ADR 0035). `trait_ontology_label`'s meaning is therefore
polymorphic along with `trait_ontology_id`'s: for an EFO/MONDO CURIE it MUST
be that specific term's human-readable name; for a gene-centric CURIE, whose
human-readable name is already `analysis_label`, it instead names the
identifying vocabulary (`"Ensembl"`) rather than repeating the gene name.
`analysis_id` MUST be unique within a Store Release.

`analyses.tsv` MUST be sufficient on its own to interpret every Analysis's stored
effect scale, sample-size semantics, ancestry, and licensing/citation terms —
this supersedes the general requirement in §3 that "Analysis metadata MUST be
sufficient to interpret stored effect scales, sample-size semantics, and source
provenance," which this section makes concrete. `index.sqlite` MUST NOT
duplicate any column of `analyses.tsv`; large, fine-grained, tooling-only
evidence (such as Reference Completion Quality at LD-block-by-Analysis
granularity, §12) stays SQLite-only, with only its per-Analysis rollup
appearing in `analyses.tsv`.

`n_hits_5e8`/`n_hits_5e6`/`n_hits_5e4` are Analytical Metadata too (ADR 0032): a
signal of study power and test-statistic-inflation risk, derived at build time
from the store's own top-hit index (§18) and never recomputed at query or render
time. A store whose associations are split across more than one on-disk top-hit
index for the same Analysis (a Hybrid store's Dense Component and Ragged
Overflow Component) records the sum, since the two partition an Analysis's
associations disjointly.

## 8. Sample size metadata

Sample size semantics are represented by kind and scope.

Sample Size Kind:

```text
total
case_control
effective
variant_level
```

Sample Size Scope:

```text
analysis_level
variant_level
```

OpenGWASDB MUST NOT present sample size inferred from effect statistics as an observed participant count.

Physical sample-size encoding is an implementation detail. Builders SHOULD choose the smallest lossless representation compatible with source data, including scalar counts, total N plus constant case fraction, sparse residuals, or full per-variant counts.

## 9. EAF and INFO metadata

EAF Scope and INFO Scope use the same values:

```text
absent
variant
association
```

EAF and INFO are optional. They are not required for statistical reconstruction.

Variant-scoped EAF or INFO is valid only when the builder can establish that one value is genuinely shared. Builders MUST NOT average differing association values into variant-scoped values.

For imputed associations, EAF comes from the LD Reference Panel when stored. In v0.1, EAF provenance is inferred from Association Status:

- observed association: source EAF;
- imputed association: reference-panel EAF.

Each Analysis declares its EAF Scope in `analyses.tsv`'s `eaf_scope` column
(ADR 0036), derived by the builder from what it actually stored rather than
copied from a build manifest. Builders emit `absent` or `association`;
`variant` remains specified but unimplemented, since no ingestion path can
establish that one value is genuinely shared. Reference Completion carries
observed EAF across unchanged; supplying reference-panel EAF for imputed
cells is not yet implemented, and those cells read as NaN.

### 9.1 EAF orientation is verified, and the verification is recorded

Stored EAF MUST describe the *stored* effect allele (§5). A source that
reports `effect_allele_frequency` against the other allele produces a store
whose every frequency is wrong by `1 - f`, with nothing in the row to show it:
`0.42` where the truth is `0.58` is a perfectly plausible frequency. This is
not hypothetical — `GCST003566` in the `gwas-catalog-eur-hybrid` pilot does
exactly this (ADR 0037 §6, issue #115).

Builders MUST therefore correlate each Analysis's A1-oriented EAF against a
reference over a deterministic sample of the overlapping variants, and MUST
fail the build when the correlation is negative. The reference is an external
panel where one is supplied; otherwise, for a build of three or more Analyses,
it is the consensus of the other Analyses — which can establish that Analyses
disagree but not which is right, so an inconsistent build fails rather than
dropping a study.

The check MUST NOT be interpreted where it cannot be. Below a minimum overlap,
or below a minimum frequency variance on either side, the Analysis records
`unverified` — never `passed`. An Analysis that stores no EAF likewise records
`unverified`: there is no wrong column, but neither is there a checked one, and
the two must not read alike.

The reference itself is checked before it is trusted: one whose frequencies
exceed 0.5 for only a negligible fraction of its variants is minor allele
frequency, which is symmetric about 0.5 and would correlate a flipped source at
r ≈ +1. Such a reference is **refused outright** — the build fails on the
reference, not on the Analyses, because nothing correlated against it would mean
anything.

The threshold is on the *sign* of `r`, not its strength. A genuine study can
correlate weakly with a reference panel — `GCST007320` in the
`gwas-catalog-eur-hybrid` pilot reads +0.878 against the UKB EUR panel where the
other nine read above +0.99 — so a floor on `|r|` would reject real data while
adding nothing: the defect this catches sits at −0.999.

The outcome is Analytical Metadata: `eaf_orientation` (`passed`/`failed`/
`unverified`), `eaf_orientation_r` and `eaf_orientation_n` in `analyses.tsv`,
with the reference's identity and checksum, the sample size, and each
Analysis's evidence in `manifest.json`'s `provenance.eaf_orientation`.
Validation checks the recorded evidence rather than the panel, since a Store
Release must stay interpretable after the panel it was built against has moved;
re-deriving the answer from the store's arrays is a separate audit that takes a
panel as input.

A correlation is deliberately used in preference to any threshold on the *size*
of the difference: a bottlenecked cohort's frequencies legitimately differ from
a reference panel's by three orders of magnitude (FinnGen, max log-residual
8.07) while still correlating at r = +0.995, because a bottleneck changes
magnitudes, not direction.

## 10. Dense Observed-Only layout

Dense Observed-Only is the first implementation target.

Dense layout stores arrays with shape:

```text
n_variants x n_analyses
```

Required statistic arrays:

```text
data.zarr/z
data.zarr/se
```

Optional (ADR 0036), same shape and chunking as the required arrays:

```text
data.zarr/eaf
```

`eaf` is `float32` where `z`/`se` are `float16`: the canonical A1 is the
lexicographically smaller allele rather than the minor one, so EAF near 1 is
ordinary, and `float16` spacing near 1.0 (0.00049) would round a MAF of 1e-4
to zero. The array is absent entirely when no Analysis in the release carries
a frequency; per-cell NaN carries per-Analysis and per-cell absence alike.

In Observed-Only Dense stores:

- every finite cell represents an observed association;
- unavailable associations are represented by canonical NaN in both `z` and `se`;
- no imputed mask is required;
- the dense variant axis is source-faithful and does not require an LD Reference Panel.

Recommended compression for initial implementation is Zarr with Zstandard and bitshuffle, using benchmarked chunking appropriate for mixed range, variant, PheWAS, and full-analysis extraction workloads.

## 11. Ragged layout

Ragged layout stores Analysis-specific association sequences referencing the Store Variant Table.

Each retained association row contains at least:

```text
analysis index or analysis offset
variant_idx
z
se
```

`eaf` (ADR 0036) is an optional fourth parallel array (`data.zarr/ragged/eaf`,
`float32`), aligned with the CSR `z`/`se` and absent when no Analysis in the
release carries a frequency.

Ragged layout is used when Analyses do not share one dense source variant axis or when Association Coverage is Cis-and-Signals.

For Observed-Only Ragged stores, absence from an Analysis sequence means the association is not retained by that Store Release.

A Ragged store's Analytical and Attribution Metadata lives in `analyses.tsv`, the same schema §7a defines for every layout (ADR 0034) — not a layout-specific `index.sqlite` table, and not a second, tabix-indexed positional side-file either: an earlier draft of this section allowed such a side-file over `analyses.tsv`'s `trait_chr`/`trait_bp` columns as a query-acceleration structure, but issue #69 retired it before it shipped in favour of scanning `analyses.tsv`'s own columns directly (`range_by_analysis()`), so a Ragged Store Release's envelope (§1, §20) has no such file.

## 12. Reference completion model

Reference Completion is an optional build phase that produces a new Reference-Completed Store Release from an Observed-Only source release.

Reference Completion exists to avoid query-time LD proxy lookup and reduce missingness by filling gaps once at build time.

Reference-Completed releases MUST:

- declare an LD Reference Panel;
- declare a Reference Completion Method;
- expose Association Status;
- produce both Z and SE for imputed associations;
- record Reference Completion Quality at LD-block-by-Analysis granularity;
- preserve observed associations not present in the Reference Variant Set.

There is no Z-only completion mode.

## 13. LD Reference Panel requirements

An LD Reference Panel used for Reference Completion MUST define:

| Requirement | Meaning |
|---|---|
| `reference_panel_id` | Stable panel identity |
| `reference_panel_version` | Panel version |
| `reference_assembly` | Genome assembly, matching the Store Release |
| `ancestry` | Population or ancestry label |
| variant list | Canonical ALIDs defining the Reference Variant Set |
| allele orientation | Orientation compatible with Store canonical ALID convention |
| LD blocks | Block definitions used by the completion method |
| LD representation | Form in which LD is stored — see below |
| source cohort | Cohort the panel's LD was estimated from |
| sample size | Number of individuals contributing to the LD estimate |
| variant inclusion | Threshold governing which variants enter the panel |
| checksums | Integrity checks for reference files |
| provenance | Source, build, and filtering provenance for the panel |

If EAF is emitted for imputed associations, the LD Reference Panel MUST provide EAF for panel variants or declare why EAF is absent.

The Store Release reference assembly MUST match the LD Reference Panel reference assembly.

### 13.1 LD representation

A panel SHOULD store, per LD block, the **eigendecomposition** of the block's LD
matrix — all eigenvalues plus a retained set of eigenvectors — rather than the LD
matrix itself (ADR 0031). Reference completion consumes eigenvectors only, so the
matrix is not required to interpret or apply a panel.

The number of eigenvectors retained MUST be sufficient for the truncation the
Reference Completion Method requests, and the panel MUST record, per block, both
the retained count and the proportion of total eigenvalue mass it represents. A
fixed retained count is NOT sufficient: the number of components needed to reach a
given variance depends on the panel's sample size, so a count calibrated on one
cohort will silently under-resolve a panel built from a smaller one. Where a
consumer's requested truncation cannot be satisfied from the retained components,
it MUST surface that rather than proceeding with reduced variance.

Panels MAY additionally retain full LD matrices, but consumers MUST NOT require
them when a conforming eigendecomposition is present.

### 13.2 Panel power is not uniform across ancestries

Panels for different ancestries may be estimated from cohorts differing by orders
of magnitude in sample size, and a panel's LD matrix has rank at most one less
than its sample size regardless of how many variants it spans. Reference
completion quality therefore varies by panel construction independently of the
quality of the Analyses being completed. Because `sample size`, `source cohort`,
and `variant inclusion` are recorded per panel, that difference is discoverable;
comparisons of completion quality across ancestries MUST NOT assume panels are
equivalently powered.

## 14. Reference completion method requirements

The Reference Completion Method MUST record:

| Field | Meaning |
|---|---|
| `method_name` | Algorithm name |
| `method_version` | Method version |
| `software` | Software/package and version |
| `parameters` | Parameters affecting imputed Z or SE |
| `required_reference_inputs` | LD reference files/data required |
| `deterministic` | Whether same inputs produce same output |
| `quality_metric` | Definition of recorded quality values |
| `quality_thresholds` | Pass/fail or reporting thresholds, if used |

The initial intended method family is LD-eigenvector based imputation of Z and SE. The store format is not locked to one algorithm: different methods are valid if they record provenance, satisfy the `z + se` output contract, and pass validation.

The initial Reference Completion implementation should use the existing `pleiodb` imputation work as prior art:

- method source: `https://github.com/explodecomputer/pleiodb/blob/main/src/pleiodb/impute.py`;
- benchmark and reference-file location notes: `https://github.com/explodecomputer/pleiodb/blob/main/scratch/imputation_benchmark.qmd`;
- benchmarked LD panel root recorded there: `/local-scratch/projects/genotype-phenotype-map/data/ld_reference_panel_hg38/EUR`.

That prior art uses LD block TSV files alongside `.unphased.vcor1.gz` LD matrices, treating the eigendecomposition as a derived cache. Under §13.1 that relationship is inverted: the eigendecomposition is the primary panel artifact and the LD matrix is optional. Panels produced before that decision may still carry matrices, and a conforming reader MAY use one to derive a missing decomposition, but MUST NOT require it when a decomposition is present. These file types remain implementation inputs to the Reference Completion Method and MUST be captured through LD Reference Panel and Reference Completion Method provenance when used.

## 15. Association Status encoding

Reference-Completed Stores encode Association Status using a per-plane **missing marker** plus an imputed mask.

Each statistic plane's missing marker is defined by that plane's declared encoding, not by its dtype: a floating-point plane marks a missing cell with NaN, and an integer plane with the reserved sentinel its codec declares. Where a store declares no encoding, its planes are `float16` and the marker is NaN — the case every release up to `format_version` 0.1 is in. Stating it in terms of the codec rather than of NaN is what lets an integer `z` plane, which cannot hold NaN, express the same contract (ADR 0037, ADR 0038 §6).

State derivation:

| Z | SE | imputed mask | Association Status |
|---|---|---|---|
| present | present | false | observed |
| present | present | true | imputed |
| missing | missing | false | missing |

Invalid states:

- only one of Z or SE is missing — the two markers MUST agree;
- imputed mask is true while Z or SE is missing;
- a marker payload is non-canonical (for a float plane, a non-canonical NaN).

Builders and validators MUST reject invalid states.

The imputed mask is:

- dense boolean or uint8 Zarr for Dense Reference-Completed grids;
- association-aligned boolean or uint8 Zarr for Ragged Reference-Completed sequences;
- chunk-aligned with Z and SE arrays.

The imputed mask is not a sparse offsets index like the top-hit significance index.

## 16. Dense Reference-Completed layout and overflow

A Hybrid Store Release physically nests its Dense Component as a
self-contained Dense Store Release at `<store>/dense` (§1) — same envelope,
opened the same way as a standalone Dense release, plus `dense_to_shared.npy`
mapping each Dense Component row to its index in the Hybrid release's shared
variant table. Its Ragged Overflow Component lives in the Hybrid release's
own `data.zarr/ragged`, alongside the shared `variants.tsv.gz`/`index.sqlite`
covering both components (§4).

For Dense Reference-Completed releases:

- the dense matrix axis MUST contain only Reference Variant Set variants;
- observed off-panel associations MUST be stored in Ragged Overflow rather than discarded;
- the dense axis SHOULD be identical across Stores completed with the same LD Reference Panel.

Observed-Only Dense releases remain source-faithful and do not require an LD Reference Panel or panel-defined axis. A later Reference-Completed release MAY use a different dense axis defined by the LD Reference Panel.

Queries against Dense Reference-Completed releases with Ragged Overflow include overflow results by default for exact and range queries. Returned rows MUST expose Query Component when multiple components are involved.

## 17. Ragged Reference-Completed regions

For Ragged Cis-and-Signals molecular Stores, Reference Completion is bounded to retained regions:

- complete cis regions within existing cis boundaries;
- complete significant trans regions within existing trans-region boundaries;
- do not expand singleton suggestive associations.

Each completed region contains:

- the full slice of Reference Variant Set variants within the region boundary;
- observed off-panel variants inside the same boundary;
- NaN statistic rows for reference-panel variants that were neither observed nor imputed.

Observed, imputed, and missing rows belong to the same ragged association sequence. Ragged Reference-Completed stores do not create a separate imputed-ragged component.

Observed off-panel variants inside a Ragged Reference-Completed region have ordinary observed Association Status and are not labelled as a separate Query Component.

## 18. Top-hit query contract

Top-Hit Queries return associations ranked by statistical significance.

Dense, Ragged, Ragged Overflow, observed, and imputed associations have equal priority. Ranking is by significance, not storage component or Association Status.

Both Dense and Ragged components SHOULD provide Top-Hit Indexes using the same thresholds and result contract. The physical index encoding MAY differ by layout.

Default thresholds:

```text
5e-8
5e-6
5e-4
```

Top-hit results from Reference-Completed releases include imputed associations by default. Observed-Only Query mode applies to Top-Hit Queries as well as exact and range queries.

## 19. Query result contract

Query results SHOULD expose:

| Field | Required when |
|---|---|
| variant identity | always |
| analysis identity | always |
| z | always when available |
| se | always when available |
| beta | when requested, derived as `z * se` |
| p-value | when requested, derived from Z |
| stored_effect_scale | always or through Analysis metadata |
| association_status | Reference-Completed releases |
| query_component | multi-component releases |
| eaf | when stored/requested |
| info | when stored/requested |
| sample size | when stored/requested |

Reference-Completed queries include imputed associations by default. Query APIs MUST provide an Observed-Only mode that excludes imputed associations.

Detailed Reference Completion Quality is not joined onto every imputed association by default. Query APIs MAY expose it when requested.

## 20. Validation rules

Validators MUST check at least:

- `manifest.json` contains required fields;
- `reference_assembly` is single-valued across the release;
- canonical variant identity is valid and unique within the Store Variant Table;
- signed statistics have been normalised to canonical allele orientation;
- `se >= 0` for all finite SE values;
- Z and SE missingness is consistent;
- Stored Effect Scale values are in the controlled vocabulary;
- Dense arrays match declared dimensions;
- Reference-Completed releases declare LD Reference Panel and Reference Completion Method;
- Reference-Completed Dense axes match the Reference Variant Set;
- imputed mask is consistent with Z and SE;
- Ragged Reference-Completed regions include all Reference Variant Set variants within completed boundaries;
- top-hit indexes, when present, are consistent with stored Z values;
- `analyses.tsv` contains exactly one row per Analysis, covering every `analysis_index` referenced by `index.sqlite` (this is the one place SQLite cannot enforce the relationship as a foreign key, since `analyses.tsv` is a separate file);
- `index.sqlite` does not contain an `analyses` table;
- `original_sd_method`, `ancestry_assignment_method` and `eaf_scope` values are in their controlled vocabularies (ADR 0029, ADR 0030, ADR 0036);
- every rsid in the Store Variant Table is resolvable through the rsid search index (§1) — a release that carries rsids it cannot resolve fails silently at query time, so the check is on coverage, not merely presence (issue #109);
- `eaf`, when present, has the same shape/length as `z`/`se` and holds no finite value outside `[0, 1]` (ADR 0036);
- every Analysis with `eaf_scope=association` carries EAF orientation evidence (§9.1, issue #115): a blank `eaf_orientation` fails, since a frequency column that has never been checked is indistinguishable from one reported against the other allele; a recorded `failed` fails; `unverified` warns; and `analyses.tsv` and `manifest.json` MUST agree on the outcome recorded for each Analysis;
- the Store Release directory contains no top-level file or directory beyond what its `primary_layout` (and, for Hybrid, its nested Dense Component directory) legitimately produces per §1/§10/§11/§16/§17 — the envelope is closed, not merely a set of required entries (issue #80).

## 21. Compatibility

`format_version` describes compatibility of the store representation. It does not describe biological data release version or source publication version. The **package version and `format_version` are independent**: a package release may change neither, one, or both. Which package reads and writes which format is recorded in the package's `CHANGELOG.md` compatibility table.

`format_version` is `MAJOR.MINOR`, both non-negative integers. A release whose `format_version` is not of that shape MUST be rejected (ADR 0038).

### 21.1 What bumps major, what bumps minor

The distinction is defined by **what a reader that does not know about the change would do**, not by the size of the change:

- **MAJOR** — a reader that does not know about the change would *misinterpret* the store: the encoding of an existing array changes; a required entry is removed, renamed, or restructured; the meaning of an existing field changes while its name and type do not.
- **MINOR** — a reader that does not know about the change still reads correctly everything it already knew, and the new thing is discoverable through the manifest: a new optional array, index, or sidecar; a new `analyses.tsv` column; a new `provenance` block.

| change | bump |
|---|---|
| a new optional array (e.g. `eaf`, ADR 0036) | minor |
| a new `analyses.tsv` column | minor |
| a required column removed or renamed (ADR 0034) | **major** |
| the encoding of `z`, `se` or `eaf` changes (ADR 0037) | **major** |

### 21.2 Reader obligations

For a release at `M.m`, a reader that fully understands major `M` up to minor `k`:

| condition | behaviour |
|---|---|
| `M` unknown | MUST reject |
| `M` known, `m <= k` | MUST accept |
| `M` known, `m > k` | MUST accept, and SHOULD warn |

Accepting a newer minor follows from the definition of minor: if an older reader could not read it correctly, the change was major and was classified wrong. The warning is what makes such a misclassification visible instead of silently returning partial data.

A reader meeting a feature it does not implement — an encoding kind, an index type — MUST reject the release rather than guess or fall back.

Future format versions may add fields, arrays, or indexes, but MUST preserve explicit manifest-based feature discovery.

### 21.3 Writing

A build writes exactly one `format_version` and reads every major it implements. There is no facility for writing an older format: a store that needs to be in an older format already exists in that format.

### 21.4 I have an old store — now what?

Store Releases are immutable. Reference Completion, re-indexing and migration all produce a **new release**, with one narrow exception: a **Provenance Amendment** may fold additional facts into an existing release's `provenance` dict in place, including a format migration recording what it did to that release. Anything that changes association data or Analytical Metadata is outside the exception.

1. **Rebuild** — the default. Sources are retained and builds are reproducible, and a rebuild also picks up every build-time fix since the store was made.
2. **Migrate** — where a mechanical transformation is sufficient and a rebuild is disproportionate. `scripts/migrate_store_to_analyses_tsv.py` is the only such tool at present, and it predates this policy: it rewrites `analyses.tsv` in place, which is outside the Provenance Amendment exception. Its targets are stores that should be rebuilt instead (ADR 0038 §5).
3. **Rejected** — a store whose major version this build does not implement cannot be read, and no amount of validation makes it readable.

There is no support window for older minors: a known major reads every minor within it.

**Reference Completion preserves its source's `format_version`**, because it writes into the source's arrays and therefore its encoding — a completed release is the same format as its source. A build that can read a source but cannot write that format MUST refuse to complete it, rather than stamp a version onto arrays it did not encode that way.
