# The 5e-8 tier is clumped at build time, into a Clumps component and `n_loci_5e8`

Issue [#417](https://github.com/opengwas/opengwasdb/issues/417), under epic
[#62](https://github.com/opengwas/opengwasdb/issues/62). The spec is store-format §18.1,
with §2, §7a, §20 and §21. Amends [ADR 0032](0032-overview-html-is-a-single-tabbed-page-with-persisted-hit-counts.md):
`n_hits_*` stays the raw count, and `n_loci_5e8` and `n_unclumped_5e8` are persisted
beside it. It reads LD only through `opengwasdb.ld` (ADRs 0087–0101), and clumps exactly
as `clump` does (ld-store §9, partner-rows §8.2). The terms are in `CONTEXT.md`.

Three sources of decision, kept apart below:

- **The owner's**, on #62 on 23 Nov 2026 (the comment "Plan, 23 Nov 2026: store clumps of
  the 5e-8 tier, built from the merged LD engine",
  [#62 comment 7110664252](https://github.com/opengwas/opengwasdb/issues/62#issuecomment-7110664252)),
  and on #418 the same day for indels.
- **#417's**, the decisions its body says this ADR must make explicitly: absent rows marked and
  counted, imputed rows not clumped, blank counts without an ancestry, the provenance, and
  |z| with ties by Store Variant Index.
- **The orchestrator's**, 23 Nov 2026, in the brief for #417, taken on the code's evidence:
  no default LD Store in the engine; several ancestries in one Store; r² read from the LD
  Store; indels translated at lookup; `r` as the LD Store's own code.
- **This ADR's own**, each with a recommendation the owner may overrule: which LD Store
  clumped an Analysis, how an LD Store is identified, ties, the sentinels, and the format
  version.

## Context

`n_hits_5e8` counts associations, not loci. It mixes up "how many signals" with "how dense
the variant axis is", so a Reference-Completed Store and a HapMap3-density Store are not
comparable on it. Epic #62 asked whether a pruning rule over the tier, the Locus Lead, could
stand in for LD clumping. It cannot: no rule tried kept both spurious splits and lost signals
under 10% (#403, #409), and the owner dropped Locus Leads on 7 Nov 2026. The full report is
opengwas/opengwasdb-stores#447; the prototype stays at tag `locus-leads-prototype`.

Exact clumping is affordable. The LD engine is now this package's `opengwasdb.ld` (#453), and
`LDStore.clump` is greedy and equal to `plink2 --clump`: a variant joins the strongest lead it
reaches at r̂² > r2 and distance < W. Measured on #62 (23 Nov 2026) against the HGDP+1kGP EUR
Genotype Store at r² 0.0319 (its t_far) and kb 10,000, observed rows only, ranked by |z|, in
one process (the scripts time_clumps.py and store_clumps.py and their raw records, in
`/data/opengwasdb/work/clump-62/` on the IEU node):

| Store | Tier | Rows clumped | Clump time | Per Analysis median / p99 / max | Leads per Analysis |
|---|---|---|---|---|---|
| OGS-00009 (ukb-b) | 5e-8 | 5.49 M | 126 s | 23 ms / 1.0 s / 5.5 s | 63 |
| OGS-00027 | 5e-8 | 7.83 M | 126 s | 5 ms / 40 ms / 0.6 s | 7 |
| OGS-00009 | 5e-4 | 41.9 M | 19.8 min | 242 ms / 7.7 s / 26 s | 815 |
| OGS-00027 | 5e-4 | 13.3 M | 3.6 min | 7 ms / 60 ms / 0.5 s | 14 |

- **Storage.** Two arrays parallel to the tier, `lead` (int32) and `r` (int16), took 11.4 MB
  for OGS-00009's 5e-8 tier: 6.44 M rows, 1.8 B a row, against a 112 MB tier and a 58 GB Store.
  The prototype stored `r` × 32767; this ADR stores the LD Store's own code (decision 7), so
  #418 re-measures the size.
- **Query.** A clumped Top-Hit read took 18 ms median against 16 ms unclumped at 5e-8.
- **Absent.** 13.3% of OGS-00009's observed 5e-8 rows (732,086 of 5,491,801) and 15.1% of
  OGS-00027's (1,181,968 of 7,831,679) are not in HGDP+1kGP EUR (8.6 M variants, n = 665).
  In 86 of the 1,219 OGS-00009 Analyses clumped, and 117 of OGS-00027's 15,785, every observed
  row is absent: those Analyses have hits and no locus the LD Store can see. How many loci are
  lost entirely inside the other Analyses, meaning every row of a locus is absent, is #418's to
  measure and record here.
- **No errors** on any Analysis. Both Stores are format 0.2.0, whose indels are spelt as the
  LD Stores spell them, so no translation was exercised (`reference_first` 0 in every record).

## Decision

### 1. Only the 5e-8 tier is clumped and stored, in one configuration (owner)

A **Clumping Configuration** is (LD Store, r², kb). The owner decided that only the 5e-8 tier
is clumped at build time, in one configuration, and that every other tier or configuration is
clumped on the fly by the Top-Hit Query (#419). The stored configuration is named `default`,
and it is the only name the format admits: a second stored configuration is a later decision
and a format change, not a convention.

### 2. Each Analysis's Assigned Ancestry chooses its LD Store; the mapping is an input (owner; orchestrator)

The owner decided that each ancestry has one default LD Store, chosen by the Analysis's
**Assigned Ancestry**, and that development uses the HGDP+1kGP EUR Genotype Store.

**No default is hidden in the engine** (orchestrator). The step takes an explicit mapping from
Assigned Ancestry to LD Store location; the registry, opengwas/opengwasdb-stores, holds each
ancestry's default and passes it (opengwas/opengwasdb-stores#454). The default will change
(Pan-UKB EUR is coming, ADR 0101) without an engine release, and the engine must never pick a
store silently. An Analysis whose `assigned_ancestry` is blank (Unassigned), or names an
ancestry the mapping does not hold, is not clumped: its rows are marked (decision 7) and its two
counts are blank, never 0 (decision 8). 4,668 of the 38,221 published Analyses have no Assigned
Ancestry (#417).

**r² is the LD Store's own t_far, and the window is 10,000 kb** (owner; orchestrator). The owner
set r² to the lowest the LD Store answers at the window, and the window to 10,000 kb. At 10 Mb
that is the store's t_far, read from its provenance, never a constant in the engine: HGDP+1kGP
EUR 0.0319, AMR 0.0632, UK Biobank EUR 0.001. ld-store §9 answers r2 = t_far exactly. A store
that refuses the configuration, such as one whose recorded `far_bp` is under 10 Mb, fails the
step, naming the store and its Refusal; the Analyses are not silently left unclumped. So does an
LD Store whose Reference Assembly is not the Store's.

### 3. Which LD Store clumped an Analysis: keyed by Assigned Ancestry in the group's attributes (orchestrator; this ADR)

A Store can hold several ancestries: OGS-00014 holds 850 AMR Analyses and 203 Unassigned
(orchestrator). The stored configuration is therefore "each Analysis's ancestry's LD Store, at
that store's r² and kb", and a reader must be able to learn which LD Store clumped a given
Analysis.

**Chosen: the group's attributes record one entry per ancestry the step clumped,
keyed by the `assigned_ancestry` value, holding that LD Store's identity and settings.** A
reader reads the Analysis's `assigned_ancestry` from `analyses.tsv` and looks it up. That
lookup is the rule the step applied, so a reader repeats the step's own choice rather than
trusting a second copy of it. Validation checks both directions: every Analysis whose ancestry
has an entry is clumped, and every other Analysis is marked not clumped (`clumps.ancestry`,
§20.0).

**Rejected: a per-Analysis column naming the LD Store.** It would hold what `assigned_ancestry`
and the mapping already determine. Two copies of one fact can disagree, and then a reader must
choose which to believe; the rule would have to check that they agree anyway.

### 4. An LD Store is identified by what a reader can recompute from the store (this ADR)

LD Stores have no release id. Each mapped ancestry's entry records:

- `name`: the LD Store's `panel.id`, such as `hgdp1kgp-v2-EUR`;
- `kind`: `ld_store` or `genotype_store`, the root attribute it was read from;
- `format_version`: the LD Store's own (0.1.0 for the EUR Genotype Store, 0.2.0 for an LD
  Store with Partner Rows);
- `samples_sha256`: its `panel.samples_sha256`, null only where the LD Store records null (an
  `ld_matrix` import that publishes no sample list, ld-store §7);
- `provenance_sha256`: the SHA-256 of its whole root provenance record, serialised as
  canonical JSON (keys sorted, no whitespace, UTF-8).

The digest is the identity; the other four say in words what it identifies. Any rebuild changes
the record, if only `build.created_at`, so two different builds never share a digest, and a
reader holding the LD Store recomputes it from the store alone: no registry is needed and the LD
Store format does not change. The location the step was given is recorded too, as information,
since a store can move without changing.

**Rejected: the location alone.** A path or URL names where a store was, not what it held; a
rebuild in place keeps the path. **Rejected: `panel.samples_sha256` alone.** It names the
samples, not the build: two LD Stores from one panel at different k share it, and an imported
store may record null.

### 5. Observed rows only, by |z|, ties by Store Variant Index (#417)

Every observed row of an Analysis's 5e-8 tier is clumped; in a Hybrid Store, its Dense
Component's and its Ragged Overflow's rows together, since one locus can span both. **Imputed
associations are not clumped** (#417): an imputed z comes from the observed ones through LD, so
it cannot add an independent locus.

The score is |z| (the tier's `abs_z`), never raw p, which underflows to 0 past |z| ≈ 38. Ties
go to the lower Store Variant Index, the order the Top-Hit Index keeps its rows in (#417).
`clump` breaks ties by the LD Store's row order instead, which is the Store's order for every
variant except a reference-first indel translated to its sorted spelling (decision 6); so the
step gives `clump` a score that already carries the Store's order and has no ties, such as each
row's rank by (|z| descending, Store Variant Index ascending). `clump` uses a score only to
order.

### 6. A reference-first indel is translated at lookup; a shared spelling is its own class (owner; orchestrator)

The owner decided on #418 (23 Nov 2026) that the step translates a Store's reference-first indel
ALIDs (format ≥ 0.2.2, ADR 0079) to the LD Store's sorted spelling at lookup, through
`opengwasdb.variants`, which owns both spellings (`normalise.sorted_alid_alleles`,
`sorted_spelling`). The sign of r flips where A1 swaps, once for each of the row and its lead.
A Store at 0.2.0 or 0.2.1 already spells indels sorted, and its ALIDs go to the LD Store as they
are.

An insertion and a deletion of one sequence at one position share a sorted spelling: `T:TA`
spells both `T>TA` and `TA>T`. A Store at ≥ 0.2.2 may hold both, and both would look up one LD
Store row. **Such a row is ambiguous identity**: it is marked (−3), counted in
`n_unclumped_5e8`, and never looked up or dropped.

**What remains, and is #462's.** A sorted LD Store does not record which allele is the
reference, and at a site it keeps one record for both (`resolve_sorted_reversed_indels`). So
a Store indel that is the only one at its site still looks up the LD Store's row for that
spelling, which may be the reverse indel of the panel. A 0.2.0 Store has the same exposure today,
on both sides. The step counts the indel rows it looked up by a shared sorted spelling, per LD
Store (`indel_rows_by_sorted_spelling`), so the exposure is a number; #462's LD Store format
removes it.

### 7. Two arrays parallel to the tier; `r` is the LD Store's own code (owner; orchestrator; this ADR)

Under `top_hits/p_5e_08/clumps/<configuration>/`, here `default/`, one entry per tier row
(owner, as measured on #62, and #417):

- `lead` (int32): the offset of the row's **Clump Lead** within its Analysis's clump slice, its
  own offset for a lead; or a negative marker. The markers are this ADR's:
  - −1 **absent**: an observed row whose variant the LD Store does not hold;
  - −2 **imputed**: not clumped, by design (decision 5);
  - −3 **ambiguous**: a reference-first indel sharing its sorted spelling (decision 6);
  - −4 **not clumped**: every row of an Analysis no LD Store was mapped for (decision 2).
- `r` (int16): the row's signed r with its lead, **as the LD Store's own code**, signed to the
  Store's A1 of both (orchestrator). The engine returns r = code / r_scale, which `REncoding`
  quantises back to the same code exactly, so the stored code is a copy, never a second rounding
  (ADR 0097: every code is a copy). A lead holds +r_scale, r = 1. A marked row holds −32768, the
  LD Store's reserved code, which is never a correlation (ADR 0090). The scale is the LD Store's,
  recorded in its ancestry's entry, and a reader decodes a row with the scale of the LD Store its
  Analysis was clumped by, never a Store-wide one.

**Rejected: a fixed quantisation of r, such as × 32767** (the prototype's). Re-quantising a code
at another scale is a second rounding, and every scale an LD Store admits fits int16, so the copy
costs nothing.

The clump slice is an Analysis's rows of the 5e-8 tier; in a Hybrid Store, its Dense Component's
rows followed by its Ragged Overflow's. Each component stores the arrays parallel to its own tier,
so a Dense Component row may name a lead in the Ragged Overflow by an offset past its own rows.

### 8. `n_loci_5e8` and `n_unclumped_5e8`, beside `n_hits_5e8` (owner; #417)

Per Analysis in `analyses.tsv`, from the stored configuration:

- `n_loci_5e8`: its Clump Leads;
- `n_unclumped_5e8`: its observed rows marked absent (−1) or ambiguous (−3). Imputed rows are
  not counted: leaving them unclumped is the design, not a failure to clump.

Both are blank, never 0, for an Analysis that was not clumped (−4), and both are present on every
row, 0 included, for one that was. A Hybrid root counts both components, and its Dense Component
counts its own rows, as for `n_hits_*`: a Dense Component's `n_loci_5e8` is the leads among its
rows of the whole Hybrid's clumping. `n_hits_5e8` is unchanged and stays the raw count (owner,
amending ADR 0032).

### 9. `format_version` does not move (this ADR, against #417's "a `format_version` bump")

The Clumps component and its two columns are **derived, rebuildable data**: the Store's own
tier, clumped by a named LD Store, which #419's on-the-fly path reproduces row for row. Deleting
them changes no existing query answer. That is the class ADR 0061 narrowed out of a version move
for an Indexed Variant Subset, and the class spec §18 keeps for a tier's `eaf`: added in place,
discovered by the manifest's `clumps` entry, no version move. ADR 0067's `n` plane is the same
case.

A version move cannot be honest here. A build stamps the lowest version its features use
(spec §21.3), and the 0.2 versions are cumulative: 0.2.3 is 0.2.2 and the SE count, 0.2.2 is
reference-first indels. A 0.2.4 that adds Clumps would declare reference-first indels and the
count, yet the owner's plan clumps existing Stores in place (decision 10), and they are 0.2.0,
0.2.1 and 0.2.2. Stamped 0.2.4, a 0.2.0 Store would tell every reader its indels are spelt
reference first, `indels_reference_first` would believe it, and this step's own lookup would
then translate ALIDs that are already sorted. That is a wrong answer that looks like a right one.
Not stamped, a clumped 0.2.0 Store and a clumped new build would carry one feature under two
regimes.

What an older package does with a clumped Store: it reads every association and every column it
knew, correctly, and ignores the two new columns (ADR 0034's superset rule). Its `validate` fails
the Store on the arrays it does not register, as it fails ADR 0067's `n` plane, which is loud.
Under ADR 0038's test that is not a breaking change: no reader misreads anything it knew.

The owner gates a format version (CONTRIBUTING.md, Merging). If the owner prefers a version,
the alternative is a 0.2.4 written only on a release already at 0.2.3, with every older Store
rebuilt before it is clumped; ukb-b alone takes 11 h 35 min to build.

### 10. Written last by every build, and in place on an existing Store (owner)

The owner's plan (#62, 23 Nov 2026) is one step, run last by every build and in place on every
published Store Release (#418, #420). The component and its columns add derived data and change
no existing value, so they may be added in place, as an Indexed Variant Subset is (ADR 0061); the
`release_id` does not change. The step writes the group, then `analyses.tsv` and `overview.html`,
then the manifest's `clumps` entry last: until that entry is written the Store does not declare
the component, and `validate` fails a group it does not declare. Running the step again replaces
all four.

Clumping needs a 0.2 Store: a 0.1.0 Store (Zarr v2, OGS-00008) is converted first. It needs a
5e-8 tier, and on a Reference-Completed Store a tier carrying `imputed`; the step refuses a Store
without either, naming it.

A Reference Completion of a clumped Store rewrites the tier, so its release carries no Clumps
component, no `clumps` entry and neither column; the build then runs the step last, as every build
does.

## Considered options

- **Clump at query time only.** Rejected by the owner, who decided the 5e-8 tier is stored
  because it is the one used; `n_loci_5e8` is then a persisted column, as `n_hits_5e8` is
  (ADR 0032), for `overview.html` and the registry.
- **Store every tier.** Rejected by the owner: the 5e-8 tier is the one used. On OGS-00009 the
  5e-4 tier would cost 90 MB and 19.8 min of clumping, against 11.4 MB and 126 s.
- **A default LD Store in the engine**, for an Analysis the mapping misses. Rejected (decision 2):
  a default changes without a release, and a missed ancestry would be clumped against the wrong
  panel, giving counts that look right.
- **Count absent rows as their own leads.** Rejected by #417: 13–15% of observed rows would
  each become a locus, inflating `n_loci_5e8` by exactly the variants the LD Store cannot see.
- **Drop absent rows.** Rejected by #417: a reader could not tell "no locus" from "no LD".
- **Clump imputed rows too.** Rejected by #417 (decision 5).
- **Ties by the LD Store's row order**, as `clump` breaks them. Rejected (decision 5): the
  Store's order differs at translated indels, so a reader could not reproduce the lead from
  the Store alone.

## Consequences

- **#418** writes the step (`opengwasdb clump`), its validation rules, the `clumps` manifest entry
  and the two columns; it registers the two arrays with the tier's role and chunking
  (`ArrayRole.TOP_HIT_INDEX`, `TOP_HIT_CHUNK_SIZE`), so a clumped read touches the chunks the
  tier's read does. It does not move `CURRENT_FORMAT_VERSION` (decision 9). It measures how many
  loci are lost entirely and records the number in this ADR.
- **#419** reads the component for the stored configuration and decodes `r` with the LD Store
  its Analysis's ancestry maps to.
- **#420** runs the step in place on every published 0.2 Store; OGS-00008 (0.1.0) is converted
  first.
- **opengwas/opengwasdb-stores#454** passes the mapping and carries the two columns. A change to
  `analyses.tsv` columns is a documentation change there too.
- **#462** removes the shared-spelling exposure (decision 6). Until then the step counts it.
- `validate` gains the `clumps.*` rules (spec §20.0). The re-clump rule needs the LD Stores, so
  it runs only when they are given and is recorded as not run otherwise; a skip is not a pass.
