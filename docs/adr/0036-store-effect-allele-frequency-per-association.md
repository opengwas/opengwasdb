# Store effect allele frequency as a per-association array, per-Analysis opt-in

> **Partly superseded by [ADR 0037](0037-statistic-array-encodings.md).** The
> semantics below stand — EAF is per (variant, Analysis), oriented to the
> stored effect allele, declared per Analysis by `eaf_scope`. Decision 3
> (`float32`) is replaced by a per-variant baseline plus an `int8` log-residual
> (#116), and Decision 5's deferral of imputed-cell EAF no longer applies:
> panel EAF is a per-variant constant, so it never needs to travel through the
> completion checkpoint (#113, #116).

Implements store-format spec §9 ("EAF and INFO metadata"), which has declared
an EAF contract since v0.1 without any layout ever storing one. Additive: a
Store Release that stores no EAF is unchanged by this decision.

## Context

Every source format this project ingests reports effect allele frequency, and
every reader already parses it. `opengwasdb.build.source.NormalisedAssociation`
has an `eaf` field; `opengwasdb.readers.tabular.TabularRow` has `af_alt`;
`GwasVcfReader.extract_at_sites` runs bcftools for `%AF` explicitly. None of it
reaches a built store. Confirmed on real releases: `zarr.open_group(<store>/
data.zarr).keys()` returns `['se', 'z', 'top_hits']`, or `[..., 'imputed']` on
a Reference-Completed one. There is no `eaf` array in any Dense, Ragged or
Hybrid release, and `variants.tsv.gz` has no EAF column.

The EAF that *is* read at build time is the LD Reference Panel's, used
transiently by `opengwasdb.completion.block` for imputation and by
`opengwasdb.build.phenotype_sd` for SD estimation. Neither writes it down. A
user with a downloaded Store Release cannot recover the frequency the source
GWAS reported for any variant.

This matters because EAF is not a nicety in this format's neighbourhood: it is
a required column of GWAS-SSF, the GWAS Catalog harmonised format this project
ingests from, and downstream tools (colocalisation, meta-analysis, MAF-based
QC, power calculation) expect it alongside beta/se. A Store Release currently
carries strictly less information than the file it was built from.

Two constraints shape the design and neither is negotiable:

1. **EAF is per (variant, Analysis), not per variant.** Two cohorts genuinely
   report different frequencies for the same variant. The Store Variant Table
   is variant-identity-only by design, so EAF cannot live there without
   asserting a shared value that does not exist. Spec §9 already says this:
   *"Builders MUST NOT average differing association values into
   variant-scoped values."*

2. **EAF availability is heterogeneous within one store.** A Hybrid or Dense
   release built from a manifest spanning several sources will routinely have
   some Analyses with per-variant EAF and others with none at all. This is the
   same shape of problem `sample_size_kind`/`sample_size_scope` were
   introduced to model for N, and it rules out a store-wide "has EAF" flag.

## Decision

### 1. A third statistic array, parallel to `z`/`se`

| Layout | Array | Shape |
|---|---|---|
| Dense | `data.zarr/eaf` | `n_variants x n_analyses`, same chunking as `z`/`se` |
| Ragged | `data.zarr/ragged/eaf` | flat, parallel to the CSR `z`/`se` |
| Hybrid | both of the above | Dense Component and Ragged Overflow Component |

Per-cell NaN means "no EAF for this association" — the same missing marker
`z`/`se` already use (ADR-0020), so nothing new has to be taught to readers,
validators, or the query contract.

The array is created only when at least one Analysis in the build has EAF.
A store built entirely from sources that report none looks exactly as it does
today, which keeps this decision additive rather than a format break.

### 2. `eaf` is oriented to the stored ALID's effect allele

The canonical ALID convention orders alleles lexicographically
(`A1 = min(ref, alt)`), so a stored effect allele is frequently the source's
*other* allele. Stored EAF is the frequency of `effect_allele` **as stored**:
where the reader flipped the association's sign, it stores `1 - af`, the same
rule `extract_at_sites` already applies to `SiteMetrics.af`. Storing the
source's own orientation instead would make EAF disagree with the `z` sign in
the same row, which is the kind of silent inconsistency that is worse than
absence.

### 3. `float32`, not `float16`

`z` and `se` are `float16` and it would be tidy for EAF to match. It cannot.

`float16` spacing near 1.0 is 0.00049. Because A1 is the lexicographically
smaller allele and not the minor allele, EAF near 1 is ordinary, not
exceptional — and a variant with EAF 0.9999 (MAF 1e-4, exactly the rare
variant a user is filtering on) rounds to exactly 1.0, i.e. to "MAF zero".
`float16` would therefore destroy rare-variant frequency information
*silently*, in the half of the ALID space where the minor allele happens to
sort second. That is the same class of defect this whole stage exists to
close, so the extra two bytes are the right trade.

Cost is real and stated plainly: a Dense store whose Analyses all carry EAF
grows from 4 to 8 bytes per cell before compression. Two things blunt it —
Analyses with no EAF store an all-NaN column, which zstd compresses to
approximately nothing, and the array is absent entirely when no Analysis has
EAF.

### 4. `eaf_scope` per Analysis in `analyses.tsv`

A new shared-core column carrying the `EafScope` vocabulary spec §9 already
defines (`absent` / `variant` / `association`), populated by the builder from
what it actually stored — not passed through from the manifest. This is the
`sample_size_kind`/`sample_size_scope` precedent: one column that tells a
consumer, per Analysis, how to read the cells.

v0.1 builders emit `absent` or `association` only. `variant` stays reserved:
spec §9 permits it, but only when a builder *"can establish that one value is
genuinely shared"*, and no ingestion path this project has can establish that.
Reserving it now means a future variant-scoped encoding does not have to
renegotiate the vocabulary.

`eaf_scope` is derived, so it is deliberately **not** part of
`PassthroughMetadata`: a manifest cannot know whether the build found usable
EAF in the source file, and a builder that copied a manifest's claim would be
able to declare EAF it never stored.

### 5. Reference Completion carries observed EAF across; panel EAF for
imputed cells is a second step

Completion rewrites a store's arrays, so the first requirement is that it not
lose what the observed store had. It now copies EAF across the row remap for
every layout, and creates the array in the completed store only when the
source had one.

Spec §9 also asks for the other half — *"imputed association: reference-panel
EAF"* — and that is **deliberately not in this change**. The panel EAF is in
hand at the completion call site (`opengwasdb.completion.block` already
indexes it per block position), but delivering it to a stored cell means
carrying it through `FillRow` and the on-disk completion checkpoint shards,
which are resumable artifacts with their own compatibility question. That is a
separable piece of work whose risk does not belong in the same change as the
storage format itself, and is tracked as #113.

Until it lands, an imputed cell's EAF is NaN. That is honest rather than
wrong: `association_status` already tells a consumer the cell is imputed, and
NaN says the store recorded no frequency for it — which is true.

An Analysis whose observed cells had no EAF but whose imputed cells eventually
get panel EAF will be `eaf_scope=association`, with NaN on the observed cells.
Per-cell NaN carries that; no additional column is needed.

### 6. Query contract

`eaf` joins `variant_index`/`analysis_index`/`z`/`se`/`association_status` as a
parallel array in every facade's result (ADR-0020, ADR-0033), NaN where the
store has none, and surfaces in the CLI's resolved output under
`--variant-info` — the flag already named for "columns that describe the
variant rather than the association" (issue #104).

## Consequences

- **Storage.** Dense stores that carry EAF everywhere roughly double their
  statistic bytes before compression. Ragged and Hybrid Overflow pay per
  stored association rather than per grid cell, so they pay much less.
- **Additive, not breaking.** Existing stores have no `eaf` array and no
  `eaf_scope` column; both are optional on read. Validation treats a missing
  array as "no EAF", not as an error. This is the first format change since
  the compatibility policy question was raised (#112) and is deliberately
  shaped to need no version bump.
- **`ReaderAssociation` grows an `eaf` field.** The `SourceReader` protocol
  changes for the second time in this stage (`SourceVariant` was the first,
  issue #109). Both changes exist because the interface was too narrow to
  carry data the sources already had.
- **`variant` scope is specified but unimplemented.** A consumer must handle
  `absent` and `association`; encountering `variant` from a future builder is
  a legitimate reason to fail loudly rather than guess.
- **Imputed cells carry no EAF yet** (see Decision 5, #113). A
  Reference-Completed release's EAF coverage is exactly its observed
  coverage.
- **No INFO.** Spec §9 pairs EAF with INFO and this decision covers only EAF.
  INFO has no reader support at all today and no demonstrated consumer; the
  `InfoScope` vocabulary stays reserved and unused.

## Amendment: exactly 0 and exactly 1 are missing (stores #176)

`opengwasdb.stats.parse_af` reads exactly `0.0` and exactly `1.0` as missing,
not as usable frequencies. Both describe a monomorphic site, which carries no
frequency information at all, and a file reporting one of them on every row is
reporting a placeholder rather than an observation: `GCST90428462` and
`GCST90428463` carry `effect_allele_frequency = 0.0` throughout, and their two
Analyses were among those with no build-eligible row.

This extends this ADR's rule rather than reversing it -- an out-of-range or
absent value is *missing*, never clamped and never substituted -- and the same
value is never replaced by 0.5 or by a panel's frequency. The consequences,
which are the point of writing it down:

- `opengwasdb.build.row_admission.admit_rows` retains such a row and counts it
  `maf_missing`, exactly as it counts a NaN.
- Ancestry accumulation ignores it, and both phenotype-SD tiers already did
  (`se_scale_samples` requires a frequency strictly inside `(0, 1)`).
- The blocked metrics projection and GWAS-VCF's `AF` cell apply the same rule.
  Copies of "what is a source frequency" that disagreed about `0` and `1` were
  different answers to it.
- A reference panel's own frequencies (the EAF-orientation table) are not
  source frequencies and keep `0` and `1`: a site fixed in the reference is
  real evidence for orientation, not a placeholder.
- A store built from such a source records NaN for that cell, which Decision 1
  already defines as "no EAF for this association".

**Cost, stated plainly.** A genuinely monomorphic site is no longer stored: a
consumer can no longer distinguish "the source reported 0" from "the source
reported nothing" for that cell, because both are NaN. Treating a reported `0`
as a frequency was worse -- MAF admission would have computed a MAF of 0 and
dropped the row for a filter it cannot be judged by -- but the loss is real and
is recorded here rather than left implicit.
