# Reference Completion: what identifies a region, and what happens when a panel matches nothing

Refines ADR 0014 (Ragged Reference-Completed regions) and ADR 0028
(ancestry-matched completion). Neither is superseded: this answers two
questions they left open, which both turned out to be answered in practice by
completing a store to nothing and saying so only in a log line nobody sees.

**Decision 1 is amended (2026-11-11, issue #430):** a region is one the
Store's build recorded in the Store, as `regions.tsv`, completed only within
its own boundary. See
[Amendment: regions are the ones the build recorded](#amendment-regions-are-the-ones-the-build-recorded-issue-430);
the text of decision 1 below is kept as the record of what it replaced.

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

*Withdrawn by the amendment below (issue #430).*

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

## Amendment: regions are the ones the build recorded (issue #430)

**What decision 1 missed.** It foresaw that "four pruned suggestive leads in
one block is a region". It did not foresee a genome-wide QC panel. A Chen 2023
Release Bundle (`sig-trans-1mb_suggestive-leads`) keeps about 8,054 QC-panel
variants per Analysis for the build's ancestry and frequency checks, which puts
about six observations in almost every LD block. Every block passed the
threshold, and was filled whole: OGS-00006 (1,400 Analyses) completed
**10,001,552,498** cells, 4.6M to 7.4M per Analysis, about 885 of the 1,348 UKB
EUR blocks for each whatever its signals, from about 41 observations per block.
In 269,616 Analysis-blocks the elastic net shrank every coefficient to zero and
added 2.15B missing rows. Validation and the Observed-View Check passed,
because they test whether observed values are reproduced, not where imputation
happened. The cis rule broke ADR 0014 too: it filled whole blocks touching the
cis window, past the cis boundary, and never an Analysis's significant trans
regions.

**The owner's decision (2026-11-11).** "It's a signals only store so
imputation should only be done at the signals themselves", and, the same day,
"Please make sure that a cis-and-signals store will also be imputed
appropriately". For every Ragged Analysis:

- **its regions are its cis region, where it has a Trait position, plus its
  significant trans regions;**
- **each is completed only within its own boundary**: the Reference Variant Set
  variants inside its inclusive `[start, end]`, not the LD blocks it touches;
- **suggestive leads and QC-panel rows never make a region**, however many fall
  in one block;
- **an Analysis with no region is carried Observed-Only.**

**Where the boundaries come from: the owner's third decision (2026-11-11).**
A Store records which rows it holds, not why it holds them, so the boundaries
are the build's: the region table the bundle-prep filter already writes as each
release's `sidecars/sparse_regions.tsv`. The owner decided the build keeps that
table **in the Store, with its other metadata**: `build-ragged-ssf --regions`
checks it and writes it beside `analyses.tsv` as `regions.tsv`, declaring it in
the manifest's `regions` entry by path and SHA-256; completion and the
Observed-View Check find it through that entry in the source Store, with no
completion-time flag, and the completed Store carries the file and the entry
forward unchanged. Three reasons:
the table already exists, so the Release Bundle machinery does not change; a
Store should describe itself, rather than need a second file found and passed
beside it at every completion; and a summary of the Store (`opengwasdb info`)
can then show what it holds. Only the table's `cis` and `significant_trans`
rows are regions. The cis boundary is the table's `cis`
row where there is one, and otherwise the Trait position plus or minus
`--cis-window-bp`, which is the cis-only eQTLGen Stores' case (OGS-00001,
-00002 and -00027 record no table). A Store whose Association Coverage is not
`cis` and that holds no `regions.tsv` is refused, and must be rebuilt with the
table: a `cis_and_signals` Store whose every Analysis has a Trait position would
otherwise complete on its cis windows and silently leave out its significant
trans regions (OGS-00007 would lose 12,442 of its 18,352 imputed cells). A table row of an unknown kind, in another assembly, or
naming an Analysis the Store lacks refuses the completion, and so does an
Analysis with no Trait position the table says nothing of: the "four
observations anywhere" rule is never a fallback. A region no LD block overlaps
is a panel gap, completed nowhere and not refused; the completion names it by
`region_id` (`regions_without_block`) rather than say so only in a log line.

**What stays from decision 1.** The fit: a block is completed for an Analysis
when its extent overlaps one of the Analysis's regions, and
`completion.block.run_block` fits on the Analysis's observations at all of that
block's own panel variants, as before. Only the fills and counts are bounded:
the block neither fills nor counts a panel row outside the Analysis's regions,
so the checkpoint format and `completion_quality` still describe exactly the
rows the Store holds. `run_block` leaves those rows out by position (its
`outside` mask), beside the seam by ALID that leaves a shared endpoint to the
next block (ADR 0072). Leaving them out by ALID through that seam, as first
written, missed a panel row no ALID names, which ADR 0079 made common.
`completion.regions` stays the one region definition, called by both
`complete-ragged` and the Observed-View Check, which now also fails a missing
row completion added outside every region (`missing_outside_retained`).

**Consequences.**

- `impute.min_observed_points()` no longer decides which blocks are completed.
  A significant region whose block holds fewer observations than the fit needs
  is completed as missing rows, which is what spec §17 says a completed region
  holds where nothing could be imputed.
- `complete-ragged` refuses a Store whose coverage is not `cis` and that holds
  no `regions.tsv`, which it used to complete: existing Stores gain the table
  only when rebuilt, and opengwasdb-stores passes each bundle's sidecar to the
  builder (opengwas/opengwasdb-stores#459).
- `regions.tsv` is a new optional file in a Ragged release (spec §1, §2, §17),
  found through the manifest, which spec §21.2 requires of every feature a format
  adds (the owner, 2026-11-11). `validate` fails a file the manifest does not
  declare, a declared file that is missing, and one whose bytes are not the
  declared SHA-256, and completion refuses each. The owner chose to move
  `format_version` to 0.2.1 for it (§21.1); how a build stamps that version is
  spec §21.3's, and is recorded with the move.
- A resume of a checkpoint written before the amendment is refused: its blocks
  were filled whole.
- Every published Reference-Completed Ragged Store was completed under the old
  rules and differs under these; re-completing one is the owner's call.

## Considered options

- **Pass the build's region table to completion (`--regions`, issue #430).**
  Released in 0.8.0, and withdrawn by the owner's third decision: a Ragged Store
  should describe its own regions, and a table passed beside it at completion
  is a second record every caller must find, and that can disagree with the
  Store.
- **Infer the regions from the Store's own significant observations (issue
  #430).** Measured on OGS-00006's archived Observed-Only Store: its
  associations at p <= 5e-8, each +/- 1 Mb, merged per Analysis, reproduce 3,621
  of the bundle's 3,627 recorded `significant_trans` regions exactly, with none
  extra. Set aside for the recorded table, which is exact: the six it misses are
  regions whose centre or leftmost source hit is not a significant hit in the
  Store, and inference would also have to restate the filter's rule in the
  engine, where it could drift from the one that chose the rows.

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
  blocks are enumerated. *No longer true since the amendment (issue #430):
  regions come from the build's region table.*
- **The panel is read once per completion, one chromosome at a time.** Block
  enumeration for these families is resolved for all Analyses in a single pass
  keyed by ALID; holding a whole panel's blocks resident would cost roughly a
  gigabyte on a genome-wide panel, which is the scale this path exists for.
- **`complete_hybrid_store` still has no `impute_analysis_ids`.** The
  asymmetry is now visible in an error message rather than hidden; giving it
  one is its own change.
