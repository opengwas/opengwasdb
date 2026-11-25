# `overview.html` is one tabbed page; Top-Hit Counts are persisted `analyses.tsv` columns, not rendered on the fly

> **Amended by [ADR 0106](0106-the-5e-8-tier-is-clumped-at-build-time.md)** (issue #417,
> epic #62; the owner's plan of 23 Nov 2026): `n_hits_*` stay the raw Top-Hit Counts, and
> two columns are persisted beside them, `n_loci_5e8` (the Clump Leads of the 5e-8 tier,
> clumped at build time against each Analysis's ancestry's LD Store) and `n_unclumped_5e8`
> (its observed rows that LD Store could not clump). They are Analytical Metadata for the
> reason the hit counts are, and are computed once and never at render time; `overview.html`
> shows them from `analyses.tsv` like any other column. Both are blank, never 0, for an
> Analysis that was not clumped (spec §7a).

ADR 0030 added `overview.html` as a single searchable/sortable table over
`analyses.tsv`. Issue #23 asked for that table to actually be designed, and in
scoping it we widened the ask: an Ancestry Composition view and a file-guide
explaining a Store Release's layout, alongside per-Analysis Top-Hit Counts (a
signal of study power and test-statistic-inflation risk) in the table itself.

## Decision

- **One file, tab-switched client-side**, not a multi-page site. AC1/AC2 (store
  format spec §7a validation, ADR 0030) both name `overview.html` specifically as
  the file that must let a user browse every Analysis offline; keeping everything
  in that one file avoids inventing sibling filenames the spec never sanctioned,
  and keeps the "no external assets, no network" self-containment property
  trivially true for the whole thing at once. Tabs: Analyses (the original
  table), Ancestry (composition, reusing the same sortable-table code with
  `ancestry_prop_*` cells rendered as inline width-bars instead of bare numbers,
  rather than a hand-drawn chart), and Guide (a directory scan matched against a
  small filename→description lookup table, so it can't drift out of sync with
  what a given layout/completion-state combination actually produces).
- **Top-Hit Counts are Analytical Metadata**, one persisted `analyses.tsv` column
  per `TOP_HIT_THRESHOLDS` tier (`5e-8`/`5e-6`/`5e-4`), computed at build time from
  the store's existing top-hit zarr index and never recomputed at render time.
  This does stretch the CONTEXT.md definition of Analytical Metadata slightly —
  a hit count doesn't change how you read one association's Z/SE the way Assigned
  Ancestry does — but it does bear on whether an Analysis's statistics should be
  trusted at all (power, inflation risk), which puts it inside the existing
  definition rather than requiring a new category.
  Consequence: every builder that calls `write_analyses_tsv` (`dense/build.py`,
  `dense/build_vcf.py`, `dense/complete.py`, `hybrid/build.py`,
  `hybrid/complete.py`) must compute its top-hit index *before* that call, not
  after as today. The already-shipped migration script (#24) also gains this
  step — old-layout stores already have a top-hit index (it predates
  `analyses.tsv`), so leaving these columns blank there would fabricate an
  "unavailable" the script's own docstring says to avoid.
- **The standalone `overview.html` regeneration command (issue #23 AC3) reads
  only the store's already-persisted data** — `analyses.tsv`, `manifest.json`
  for the header, and a directory scan for the Guide tab — **never the zarr
  top-hit index.** Because Top-Hit Counts are persisted `analyses.tsv` columns
  rather than computed at render time, the one thing that would otherwise force
  a heavier read (re-deriving counts from the zarr index) never applies here;
  the command still needs more than `analyses.tsv` alone for the header and
  Guide tab, but never a rebuild of anything.

## Considered options

- **Compute Top-Hit Counts at `overview.html` render time**, reading the zarr
  top-hit index directly. Rejected: would force the standalone regeneration
  command to read store internals substantially heavier than the already-cheap
  `analyses.tsv`/`manifest.json`/directory-listing reads it needs anyway, and
  splits Analytical Metadata across two files for no benefit — the count
  doesn't change, only its access path would.
- **Separate HTML files per tab** (`overview.html`, `ancestry.html`,
  `guide.html`), cross-linked by a nav bar. Rejected: `overview.html` is the only
  filename the spec/validator actually requires; adding sibling files means
  deciding names, links, and a "what if one is missing" story for no reason a
  single tabbed file doesn't already solve more simply.
- **A new glossary term for Top-Hit Counts, separate from Analytical Metadata.**
  Rejected: a hit count is study-power/inflation-risk signal, which is
  interpretation-relevant by the existing definition — no need for a second
  category alongside it.
