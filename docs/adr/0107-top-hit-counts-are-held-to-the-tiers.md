# Top-Hit Counts are held to the tiers, and recounted in place

## Status

Accepted. Issue [#509](https://github.com/opengwas/opengwasdb/issues/509), under
opengwas/opengwasdb-stores epic #520. Decisions 1 and 2 follow from the evidence and from
[ADR 0032](0032-overview-html-is-a-single-tabbed-page-with-persisted-hit-counts.md)'s definition
of the counts. Decision 3, the in-place repair, is a second exception to immutability, beside
[ADR 0106](0106-the-5e-8-tier-is-clumped-at-build-time.md) §10. It is **the owner's**: approved
on 26 Nov 2026 (AskUserQuestion in the orchestrating session), recorded as item 10 of
[opengwas/opengwasdb-stores#520](https://github.com/opengwas/opengwasdb-stores/issues/520#issuecomment-7202276131).

What this ADR changes in other documents:

- Spec §7a: the Top-Hit Counts are the rows of the release's own tiers, and the rule holds them
  to that.
- Spec §20.0: a new rule, `analyses.top_hit_counts`.
- Spec §21.4: a fourth in-place change, `opengwasdb recount-top-hits`.
- [ADR 0038](0038-store-format-versioning-and-migration-policy.md) §4–§5 are narrowed once more.
  The banner ADR 0106 put there stands; this ADR adds a second exception to it.

## Context

ADR 0032 persists `n_hits_5e8`, `n_hits_5e6` and `n_hits_5e4` in `analyses.tsv`: each counts an
Analysis's rows in the matching tier of the Top-Hit Index. A Hybrid's root counts both
components, which partition an Analysis's associations. No rule compared them with the tiers.

`complete-hybrid` broke them (`layouts/hybrid/complete.py::_remap_shared_axis`):

1. `complete_dense_store` counted the Dense Component's tiers.
2. `_fold_panel_crossovers` then wrote observed Overflow values into the Dense cells of
   variants that crossed onto the panel.
3. `build_dense_top_hit_indexes` rebuilt the Dense tiers, but nothing recounted them.
4. The root added the Overflow's post-fold counts to the stale Dense ones.

A comment accepted this as a "narrow imprecision". Clumps exposed it: `clumps.counts` fails an
Analysis with more loci and unclumped rows than `n_hits_5e8`. Adding Clumps to OGS-00011 failed
26 Analyses this way. GCST006685, for example, records 0 against 1 Dense tier row.

**Every published Store, read-only (26 Nov 2026).** Each `analyses.tsv`'s counts were compared
with its tiers, summed over a Hybrid's components:

- 21 Stores were checked under `/data/opengwasdb/stores`: Dense, Ragged and Hybrid, Observed-Only
  and Reference-Completed, at 0.1.0, 0.2.2 and 0.3.0.
- **Only OGS-00011**, a completed 0.3.0 Hybrid, is wrong: 1,028, 2,080 and 3,037 of its 3,245
  Analyses at 5e-8, 5e-6 and 5e-4.
- Every Dense and Ragged Store matches, and so do the observed-only Hybrids OGS-00032 and
  OGS-00033. So does each 0.2.2 Hybrid's own Dense table (OGS-00004, OGS-00005).
- `complete-ragged` resets and recounts from its final tiers, and has no fold.

## Decision

### 1. `complete-hybrid` recounts the Dense Component after the fold

After the tier rebuild, `_recount_dense_hits` resets the Dense Component's counts and sets them
from its final tiers. It uses the same owners `complete-ragged` uses: `reset_top_hit_counts`,
`top_hits.add_hit_counts` and `write_analyses_tsv`. The root then adds the Overflow's counts, as
before. The "narrow imprecision" comment is removed.

### 2. A rule: `analyses.top_hit_counts`

Each count equals its tier's rows for that Analysis. This applies in every `analyses.tsv` a
release carries:

- a Dense or Ragged release's, from its own tiers;
- a Hybrid root's, from both components (`top_hits.summed_hit_counts`);
- a Dense Component that carries its own table (before 0.3.0, #472), from its own tiers.

Rows are matched by `analysis_index`. The rule reports one message per column, with how many
Analyses disagree and the first of them. It is judged where any component counted carries a
Top-Hit Index. A component that lacks one fails the rule, as does a row whose `analysis_index`
names no Analysis.

**Severity by format** (`SeverityByFormat`, moved beside `severity_owed` so that both of its
rules share it): an **error** on a release this package's builders write (0.2.2 and later), and a
warning before. The reasons:

- The count is a defined derivation of what the release stores, and a wrong one is the silent
  failure CLAUDE.md names.
- The only release it fails is OGS-00011, which decision 3 repairs in place.
- An earlier release, which no builder writes, is reported without being failed (#367).

### 3. `opengwasdb recount-top-hits STORE` repairs a release in place

The command recounts every `analyses.tsv` the release carries from its stored tiers: a 0.2
Hybrid's Dense Component's table first, then the root's. It changes nothing else. The rule and
the command share one owner, in `top_hits.py`, for which components a table counts
(`counted_components`) and for counting them (`table_slots`, `counted_hits`). So each row is
matched to its tier slot by `analysis_index` in both, and a permuted table is repaired in its
own row order.

- **Before it writes**, it reads every table and tier and computes every count. It refuses, and
  writes nothing, if:
  - a counted component has no Top-Hit Index, or lacks a tier;
  - a tier's offsets are malformed, or it counts a different number of Analyses from its table;
  - the table's `analysis_index` is not 0 to n-1, each once;
  - a table lacks the three columns.
- **It writes a table only where a count changes**, through `write_analysis_columns`, as `clump`
  writes its counts: every other column and value as read, and the file replaced atomically.
- **Then it regenerates each table's `overview.html`**, where the release carries one, if the page
  differs from what the table renders (`overview.refresh_overview_html`). Every `overview.html`
  is now written atomically (`model.files.write_atomically`), for every caller. A kill between a
  table and its page leaves the old page whole. The next run finds the counts right, writes no
  table, and regenerates the stale page, which atomicity alone would not do. A release whose
  counts and pages are right is left untouched, so a second run writes nothing.
- **It prints one JSON object:** `store`, `seconds`, and per table its Analyses, how many
  changed and the changes per column. `written` names the tables it wrote, and
  `overviews_written` the pages.

**An exception to immutability (the owner's, 26 Nov 2026).** ADR 0038 §4–§5 and spec §21.4 derive a new
release for any change to Analytical Metadata, and the Top-Hit Counts are Analytical Metadata.
The exception is exactly this: **the Top-Hit Counts may be set in place to what the release's own
stored tiers count.** That is the value ADR 0032 defines them as, so no association, tier or
other column changes, and neither the `release_id` nor the `format_version` moves. Anything else
that touches Analytical Metadata still derives a new release.

**The owner's ruling** (26 Nov 2026, AskUserQuestion in the orchestrating session; item 10 of
[opengwas/opengwasdb-stores#520](https://github.com/opengwas/opengwasdb-stores/issues/520#issuecomment-7202276131)):

- `recount-top-hits` may correct a published release's `n_hits_5e8`, `n_hits_5e6` and
  `n_hits_5e4` in place, from its own tiers.
- It runs inside the registry's in-place release: a reflink copy is recounted and validated, then
  published, and the old Store is archived to `replaced/`.
- **`n_loci_5e8` and `n_unclumped_5e8` are kept alongside `n_hits_*`, unchanged.** The recount
  writes only the three Top-Hit Count columns. It leaves every other column of `analyses.tsv`
  byte-identical, the Clumps' two included
  (`test_the_recount_leaves_every_other_column_byte_identical_beside_clumps`). On a clumped
  release whose stale `n_hits_5e8` is below the Analysis's loci and unclumped rows, OGS-00011's
  case, `clumps.counts` then passes with the Clumps untouched.
- OGS-00011, with 3,047 of its 3,245 Analyses stale, is repaired this way.

**On a reflink copy of OGS-00011 (26 Nov 2026):** the command took 1.14 s. It changed **3,047 of
3,245 Analyses**: 1,028 at 5e-8, 2,080 at 5e-6 and 3,037 at 5e-4. The rule then passes, a second
run writes nothing, and no other column of `analyses.tsv` changed.

## Considered options

- **Rebuild OGS-00011 from source.** Its completion is the expensive step, and the fix changes no
  association. A rebuild would only reproduce the same tiers with right counts.
- **A warning everywhere.** This would leave the next drift silent until a Clumps step fails on
  it, which is how this one was found.
- **Recount in `complete-hybrid`'s root step alone.** The Dense Component's own table, which a
  0.2 Hybrid carries, would then still be wrong.

## Consequences

- A completed Hybrid built after this fix counts its final tiers. One built before it fails
  `analyses.top_hit_counts` until `recount-top-hits` runs on it.
- The opengwasdb-stores registry runs `recount-top-hits` on OGS-00011 inside its in-place
  release, as the owner ruled (decision 3).
- `build-dense-top-hits` and `build-ragged-top-hits` rebuild tiers in place and do not recount.
  Their help names `recount-top-hits` as the follow-up, and the rule fails a release whose
  counts were left behind.
- The Store Reader contract bundle is now version 3: `hybrid_completed`'s `trait_a` counts go from
  7 to 8. The Store Reader ports (ogquery, opengwaspy, opengwasr) need no code change, but each
  must re-vendor bundle 3 and bump any pinned `bundle_version`, or its contract test fails on the
  stale digest.
