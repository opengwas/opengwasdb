# Reference Completion: what identifies a region, and what happens when a panel matches nothing

Refines ADR 0014 (Ragged Reference-Completed regions) and ADR 0028
(ancestry-matched completion). Neither is superseded: this answers two
questions they left open, which both turned out to be answered in practice by
completing a store to nothing and saying so only in a log line nobody sees.

## Context

Two defects, found by running completion at production scale against the
registry's own stores, share one shape — **a completion run that succeeds,
validates, and contains nothing**.

### A panel that matches no Analysis (issue #98)

ADR 0028 filters Analyses by `assigned_ancestry` so a panel only imputes the
Analyses it describes. The filter compared that column to the `--ancestry`
flag by string equality, and the two carry different vocabularies: LD panel
directories are named for super-population codes (`EUR`), while an Analysis's
`assigned_ancestry` may hold the word (`European`). Both spellings occur in a
single `analyses.tsv` — the registry writes the code down its AF-assignment
path and the source's own word down its trusted-label path.

`opengwasdb complete-hybrid ... --ancestry EUR` against a store recording
`European` therefore matched **zero** Analyses. Completion ran to a successful
finish and produced a release stamped `reference_completed` that passed
`validate_store()` and reported `0 imputed` — a line that reads as "nothing was
imputable". The LD work had all succeeded: the `completion_quality` table,
written before the filter is applied, held real correlations up to 0.97 and
per-Analysis counts summing to 51M+ imputable cells across 1,357 blocks.

### A Store Family with no gene target (issue #102)

ADR 0014 bounds Ragged completion to *retained regions* and identifies them
from each Analysis's cis window — a window around `trait_chr`/`trait_bp`. A
Store Family with no single encoding gene per Analysis has no such position, by
design and by the registry's documented schema. Those Analyses were passed
through the same branch as ancestry-mismatched ones: no blocks enumerated, no
panel variants added, nothing imputed. All four `metabolome-plasma-2023` full
releases (4,443 Analyses, real per-ancestry panels) completed this way and
changed nothing.

## Decision

### 1. A region is identified by observations at a block's own panel variants

Where an Analysis has a Trait position, its regions are the cis window's blocks
— unchanged. Where it has none, its regions are the LD blocks it **already
holds enough observations in**, and "enough" is `impute.min_observed_points()`.

Two details are load-bearing, and both were got wrong in a first draft:

- **The count is over the Analysis's observations at the block's own panel
  variants**, not over its variants falling inside the block's base-pair
  extent. `completion.block.run_block` fits on the former; an off-panel variant
  inside the extent inflates the latter and enumerates a block that then
  imputes nothing.
- **The threshold is the number the imputation gate actually enforces.**
  `poly_rescale` fits a degree-3 polynomial and returns `pearson_r` of NaN
  below `max(npoly + 1, 2)` = 4 observed points, which `impute_z_block`
  rejects. A threshold of 2 — the minimum `elastic_net_impute` alone needs —
  admits blocks that are then rejected downstream, adding their panel variants
  to the axis as missing rows and filling none. The number is therefore
  exported from `impute` rather than restated, so the two cannot drift.

Spec §17's "do not expand singleton suggestive associations" falls out of the
same threshold rather than needing a separate rule.

**Consequence worth stating: this admits blocks made of nominal associations.**
Four pruned suggestive leads in one block is a region under this rule. That is
deliberate — the alternative is to keep doing nothing for these families — but
it means a completed metabolome store's imputed cells derive from weaker
regions than a cis-anchored store's, and #117's rebuild is where that gets
looked at against real data rather than argued about here.

### 2. A panel that matches no Analysis refuses, rather than completing

`derive_impute_analysis_ids` raises `AncestryFilterError` when
`assigned_ancestry` is populated and nothing matches. An empty match cannot be
expressed as a completed store: zero imputed cells is indistinguishable
downstream from a store where imputation was attempted and failed, and
returning "impute everything" instead would apply one panel's LD and EAF to
Analyses known not to match it.

The error names the panel, the `assigned_ancestry` values the store actually
holds, and whether the panel's own name was understood as a super-population —
because "you asked for a panel nothing matches" and "you asked for a panel this
build cannot name" are different problems with the same symptom.

It deliberately does **not** advise passing `impute_analysis_ids`: that
parameter exists on `complete_dense_store` and `complete_ragged_store` but not
on `complete_hybrid_store` or any CLI command, and #98 was reported against
`complete-hybrid`. Advice most callers cannot take is not advice.

### 3. Ancestry spellings are reconciled by an exact alias table

`European` and `EUR` name one ancestry and match. The mapping is an explicit
table keyed on a normalised label (case, spacing, underscores, hyphens), and
matching is exact.

It deliberately does **not** reuse `ancestry.routing.reported_to_superpop`,
which substring-matches an ordered list and answers `AFR` for `"North
African"` because `african` precedes `north africa` in it. Guessing at a
cohort's free-text description that way is tolerable. Deciding which Analyses
an LD panel may impute that way is not: it would impute a North African store
against the AFR panel and refuse it against its own.

A label the table does not name is unroutable and matches nothing — `Mixed`
must not become `EUR` by failing to normalise. A panel named outside the
vocabulary entirely (a cohort-specific panel) still matches on exact equality,
so normalisation is not a requirement to be normalisable.

## Considered options

- **Warn instead of raising (issue #98).** Rejected: the run already logged at
  `info` and the operator still shipped an empty store. A warning is what the
  no-ancestry-anywhere fallback does (issue #108), and it is right *there*
  because that path produces a store with data in it. This one does not.
- **Return "impute everything" when nothing matches.** Rejected outright: that
  is ADR 0028's failure mode in reverse, applying one panel's LD to every
  Analysis precisely when the metadata says none of them match it.
- **Enumerate every block overlapping any of the Analysis's variants.**
  Rejected: it expands singleton associations (spec §17), and most of the
  blocks it adds cannot be fitted anyway.
- **Take the cis window's radius as a default region around each variant.**
  Rejected: a window is a proxy for "the region this Analysis has evidence in",
  and for these families the evidence is directly available. Inventing a
  window around each variant would also merge unrelated sparse leads into one
  region.
- **Normalise ancestry with `reported_to_superpop`.** Rejected — see decision
  3. It is the right tool for its own job and the wrong one for this.

## Consequences

- **Completion gains a failure mode.** A multi-panel loop over a
  multi-ancestry store now raises on the panels that match nothing, where it
  previously wrote an empty release per panel. Callers that genuinely want
  per-panel runs handle the exception or select Analyses themselves.
- **Gene-target-less Store Families become completable at all**, which changes
  what `metabolome-plasma-2023` produces — a registry-visible change
  (opengwasdb-stores#78).
- **`impute.min_observed_points()` is now a published contract**, not an
  implementation detail of `poly_rescale`. Changing `npoly` changes which
  blocks are enumerated.
- **The panel is read once per completion, one chromosome at a time.** Block
  enumeration for these families is resolved for all Analyses in a single pass
  keyed by ALID; holding a whole panel's blocks resident would cost roughly a
  gigabyte on a genome-wide panel, which is the scale this path exists for.
- **`complete_hybrid_store` still has no `impute_analysis_ids`.** The
  asymmetry is now visible in an error message rather than hidden; giving it
  one is its own change.
