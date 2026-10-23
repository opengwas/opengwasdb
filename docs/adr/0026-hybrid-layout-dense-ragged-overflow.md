# Hybrid Layout: Dense Component + Ragged Overflow for heterogeneous genome-wide collections

Some genome-wide collections are neither cleanly dense nor cleanly ragged. The UK
Biobank batch (ukb-b) is dense: every Analysis was imputed to one shared panel, so
the source-observed union of variants is essentially the panel, and a single dense
matrix is optimal (ADR 0022). eqtlgen is ragged: cis-only, genuinely sparse. But a
collection of **independent consortia GWAS** — OpenGWAS `ieu-a` / `ieu-b` — sits
between: the studies use **different imputation panels**, yet each is genome-wide
and they **share a large common core** of variants, with a heterogeneous,
study-specific long tail. For that regime the source-observed union (plausibly
50–100M+ variants) is far larger than any single reference panel, and most of the
excess is un-imputable and sparse.

## Decision

Introduce a **Hybrid Layout** (a third `PrimaryStorageLayout`). A single integrated
Store holds two Query Components over one shared Store Variant Table:

- **Dense Component** — a dense matrix whose variant axis is the **completion
  reference panel** (the imputable set). Behaves like a Dense Layout store.
- **Ragged Overflow Component** — a CSR store of each Analysis's **off-panel
  observed** associations (the study-specific tail). Off-panel variants lack the
  reference LD structure, so overflow associations are **always observed, never
  imputed**.

A variant is on-panel (a Dense Component row shared by all Analyses) *xor* off-panel
(a Ragged Overflow entry for whichever Analyses observed it), so the two components
partition each Analysis's associations disjointly.

**Generation** extends the existing two-pass dense VCF builder, reading each study
**once**: Pass 1 lifts over each study's variants, checks panel membership, and
collects the off-panel union (the panel axis is fixed up front, so no union pass is
needed for it); Pass 2 band-streams the dense fill for on-panel variants **and**
emits each study's off-panel `(variant_index, z, se)` into the ragged CSR in the
same read.

**Reference Completion** targets only the Dense Component (panel variants have LD
structure), reusing the dense completion pipeline unchanged; the Ragged Overflow
Component is left observed-only.

## Scope relationship to ADR 0022 — not a reversal

ADR 0022 deleted an earlier design (ADR 0015) that stored off-panel source variants
in a Ragged Overflow component, and it is **not superseded here**. ADR 0022 remains
correct for **shared-panel** collections, where source-union ≈ panel: the few
off-panel source variants ride along in the single dense matrix for free, and a
second component would only add query/validation complexity.

The Hybrid Layout applies to the **opposite regime** — source-union ≫ panel — that
ADR 0022 did not consider. There the off-panel tail is large, sparse, and
**never imputable**, so ADR 0022's "keep everything dense" would inflate the dense
axis several-fold with mostly-NaN rows that gain nothing from being dense, and every
dense query would pay for the larger variant axis, tabix index, and ALID mmap. The
deciding boundary is the ratio of source-union to reference-panel size.

## Considered options

- **Dense-of-union for the heterogeneous case (ADR 0022's rule, applied here).**
  Rejected: although zstd makes empty cells nearly free, the *axis-size* cost
  (tabix, ALID mmap, per-query axis handling, build-time union of 50–100M+ variants)
  is paid by everything, and the off-panel rows are permanently un-imputable — they
  gain nothing from density.

- **Frequency-threshold partition** (dense = variants observed in ≥ X% of studies).
  Rejected: needs a genome-wide cross-study frequency pass, and decouples the dense
  axis from the completion reference panel — so "dense" would no longer mean
  "imputable", and the axis would mix imputable and non-imputable variants.
  Panel-anchoring keeps Dense Component ≡ imputable. A comprehensive modern panel
  already contains the common core, so the frequency benefit is marginal.

- **Two separate stores (a dense store + a ragged store) joined at query time.**
  Rejected: splits the Store Variant Table into two `variant_index` spaces, creates
  two identities, and forces every query to open and reconcile two stores.

- **Bolt-on two-phase build** (build the dense store, then a separate job re-reads
  every study to append the overflow). Rejected: re-reads every genome-wide study
  twice, doubling the dominant I/O cost. The integrated build routes on-panel vs
  off-panel in a single read.

## Consequences

- A new `PrimaryStorageLayout = hybrid`; the manifest describes two components over
  one Store Variant Table (union of panel ∪ off-panel-observed variants).
- **Association Coverage is `gwas`**: the overflow retains every off-panel source
  association, so nothing is dropped. (ADR 0064 renamed the value from `full` and
  made it a build input; a Hybrid release can still carry only `gwas`, because
  its missing cells already mean the source did not report the association.)
- Reference Completion touches only the Dense Component; the overflow is permanently
  observed-only, and its associations always carry Association Status = Observed.
- Query dispatches by panel membership (on-panel → dense row/column, off-panel →
  ragged) and concatenates results; because a variant is in exactly one component,
  there is no cross-component deduplication.
- Off-panel PheWAS inherits the Ragged Layout's O(n) scan. On-panel (common-variant)
  PheWAS — the usual instrument case — is served fast by the Dense Component, so a
  variant-centric overflow index is deferred until off-panel PheWAS is shown to be a
  bottleneck.
