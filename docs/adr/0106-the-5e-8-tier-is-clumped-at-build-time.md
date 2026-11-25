# The 5e-8 tier is clumped at build time, into a Clumps component and `n_loci_5e8`

Issue [#417](https://github.com/opengwas/opengwasdb/issues/417), under epic
[#62](https://github.com/opengwas/opengwasdb/issues/62). The spec is store-format §18.1,
with §1.1, §2, §7a, §20 and §21. Amends [ADR 0032](0032-overview-html-is-a-single-tabbed-page-with-persisted-hit-counts.md):
`n_hits_*` stays the raw count, and `n_loci_5e8` and `n_unclumped_5e8` are persisted
beside it. It reads LD only through `opengwasdb.ld` (ADRs 0087–0101), and clumps exactly
as `clump` does (ld-store §9, partner-rows §8.2). It applies
[ADR 0102](0102-what-a-store-release-ships.md)'s rule to the Clumps and adds them to that
ADR's classification (decision 11). The Clumps are a component of format 0.3.0
([ADR 0105](0105-format-0-3-0-variable-length-alid-sidecar.md)) and of no earlier format
(decision 10). The terms are in `CONTEXT.md`.

This ADR was first written against the 0.2 series. It was re-decided at 0.3.0 once epic #450
merged, and renumbered from 0102, which #450's ADR took.

Each decision below names its source:

- **The owner's**, on #62 on 23 Nov 2026 (the comment "Plan, 23 Nov 2026: store clumps of
  the 5e-8 tier, built from the merged LD engine",
  [#62 comment 7110664252](https://github.com/opengwas/opengwasdb/issues/62#issuecomment-7110664252)),
  and on #418 the same day for indels. On 25 Nov 2026 the owner decided to rebuild every
  registered Store at 0.3.0 first (opengwas/opengwasdb-stores#498) and then add the Clumps to
  those rebuilt Stores in place
  ([#420 comment 7148370637](https://github.com/opengwas/opengwasdb/issues/420#issuecomment-7148370637)).
- **#417's**, the decisions its body says this ADR must make explicitly: absent rows marked and
  counted, imputed rows not clumped, blank counts without an ancestry, the provenance, and
  |z| with ties by Store Variant Index.
- **The evidence**, where a decision follows from the code, the layering rules or a
  measurement, which the decision cites: no default LD Store in the engine; several ancestries
  in one Store; r² read from the LD Store; `r` as the LD Store's own code; where the step runs;
  the class under ADR 0102; |z| taken from `z` by the function #471's review named.
- **This ADR's own**, each with a recommendation the owner may overrule: which LD Store
  clumped an Analysis, how an LD Store is identified, ties, the markers, and the format
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

| Store | Tier | Rows clumped | Clump time | Per Analysis median / p99 / max | Leads per Analysis, mean / median |
|---|---|---|---|---|---|
| OGS-00009 (ukb-b) | 5e-8 | 5.49 M | 126 s | 23 ms / 1.0 s / 5.5 s | 63 / 3 |
| OGS-00027 | 5e-8 | 7.83 M | 126 s | 5 ms / 40 ms / 0.6 s | 7.1 / 4 |
| OGS-00009 | 5e-4 | 41.9 M | 19.8 min | 242 ms / 7.7 s / 26 s | 815 / 503.5 |
| OGS-00027 | 5e-4 | 13.3 M | 3.6 min | 7 ms / 60 ms / 0.5 s | 13.5 / 9 |

The means and medians are over the Analyses in each run's records (`all-*.jsonl`,
`all5e4-*.jsonl`): 1,219 and 15,785 at 5e-8, 2,514 and 18,452 at 5e-4.

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
  These are measurements of 0.2.0 Stores, taken before format 0.3.0 existed. The Clumps
  themselves are written only on 0.3.0 Stores (decision 10), where every indel is spelt
  reference first and translated (decision 6).

## Decision

### 1. Only the 5e-8 tier is clumped and stored, in one configuration (owner)

A **Clumping Configuration** is (LD Store, r², kb). The owner decided that only the 5e-8 tier
is clumped at build time, in one configuration, and that every other tier or configuration is
clumped on the fly by the Top-Hit Query (#419). The stored configuration is named `default`,
and it is the only name the format admits: a second stored configuration is a later decision
and a change to the spec, not a convention.

### 2. Each Analysis's Assigned Ancestry chooses its LD Store; the mapping is an input (owner; this ADR)

The owner decided that each ancestry has one default LD Store, chosen by the Analysis's
**Assigned Ancestry**, and that development uses the HGDP+1kGP EUR Genotype Store.

**No default is hidden in the engine** (this ADR). The step takes an explicit mapping from
Assigned Ancestry to LD Store location; the registry, opengwas/opengwasdb-stores, holds each
ancestry's default and passes it (opengwas/opengwasdb-stores#454). The default will change
(Pan-UKB EUR is coming, ADR 0101) without an engine release, and the engine must never pick a
store silently. An Analysis whose `assigned_ancestry` is blank (Unassigned), or names an
ancestry the mapping does not hold, is not clumped: its rows are marked (decision 7) and its two
counts are blank, never 0 (decision 8). 4,668 of the 38,221 published Analyses have no Assigned
Ancestry (#417).

**r² is the LD Store's own t_far, and the window is 10,000 kb** (owner). The owner set r² to
the lowest the LD Store answers at the window, and the window to 10,000 kb. At 10 Mb that is,
by the refusal table of ld-store §9 (`judge_partner_request`), the store's t_far, read from its provenance, never a constant in the engine: HGDP+1kGP
EUR 0.0319, AMR 0.0632, UK Biobank EUR 0.001. ld-store §9 answers r2 = t_far exactly. A store
that refuses the configuration, such as one whose recorded `far_bp` is under 10 Mb, fails the
step, naming the store and its Refusal; the Analyses are not silently left unclumped. So does an
LD Store whose Reference Assembly is not the Store's.

### 3. Which LD Store clumped an Analysis: keyed by Assigned Ancestry in the group's attributes (this ADR)

A Store can hold several ancestries: OGS-00014's `analyses.tsv` holds 850 AMR Analyses and 203
Unassigned. The stored configuration is therefore "each Analysis's ancestry's LD Store, at
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

The score is |z|, never raw p, which underflows to 0 past |z| ≈ 38. A 0.3.0 tier stores `z`
and neither `abs_z` nor `p_value` (ADR 0102, #471), so the step takes |z| from the tier's `z`
through `top_hits.recomputed_abs_z_and_p` (`opengwasdb/top_hits.py`, the owner of |z| that
#471's review named). That is the |z| the validator recomputes for `top_hits.threshold`, so the
step and its rules rank by one number. Ties
go to the lower Store Variant Index, the order the Top-Hit Index keeps its rows in (#417).
`clump` breaks ties by the LD Store's row order instead, which is the Store's order for every
variant except a reference-first indel translated to its sorted spelling (decision 6). `clump`
leads with the **largest** score (`np.lexsort((rows, -score))` in `_component`,
`opengwasdb/ld/query/clumping.py`), so the step passes a score that is larger for a stronger
row and has no ties: n − rank, where rank 0 is the largest |z| and ties go by Store Variant
Index ascending. The score enters only that ordering, so for distinct |z| the leads are those of
plink2's greedy on |z| (strongest first, r̂² > r2, d < W). A score of the rank itself would lead
with the weakest row.

In a Hybrid Store the two components' tiers index different axes: the Dense Component's
`variant_index` is a panel row of its grid, the Ragged Overflow's a row of the shared Variant
Axis. From 0.3.0 the Dense Component has no Variant Axis of its own (#472): the Store Reader
resolves its rows through the shared axis and `dense_to_shared.npy` (`PanelVariantAxis`,
`opengwasdb/store_reader/panel_axis.py`). Ties and identity use the shared index, so the step
maps a Dense Component row through `dense_to_shared.npy` the same way first.

### 6. A reference-first indel is translated at lookup; a shared spelling is its own class (owner)

The owner decided on #418 (23 Nov 2026) that the step translates a Store's reference-first indel
ALIDs (ADR 0079) to the LD Store's sorted spelling at lookup, through
`opengwasdb.variants`, which owns both spellings (`normalise.sorted_alid_alleles`,
`sorted_spelling`). The sign of r flips where A1 swaps, once for each of the row and its lead.
A 0.3.0 Store keeps 0.2.2's reference-first indels (ADR 0105), and the step clumps only 0.3.0
Stores (decision 10), so every indel ALID it reads is spelt reference first and is translated.

An insertion and a deletion of one sequence at one position share a sorted spelling: `T:TA`
spells both `T>TA` and `TA>T`. A 0.3.0 Store may hold both, and both would look up one LD
Store row. **Such a row is ambiguous identity**: it is marked (−3), counted in
`n_unclumped_5e8`, and never looked up or dropped.

**What remains, and is #462's.** A sorted LD Store does not record which allele is the
reference, and at a site it keeps one record for both (`resolve_sorted_reversed_indels`). So
a Store indel that is the only one at its site still looks up the LD Store's row for that
spelling, which may be the reverse indel of the panel. The step counts the indel rows it looked
up by a shared sorted spelling, per LD Store (`indel_rows_by_sorted_spelling`), so the exposure
is a number; #462's LD Store format removes it.

### 7. Two arrays parallel to the tier; `r` is the LD Store's own code (owner; this ADR)

Under `top_hits/p_5e_08/clumps/<configuration>/`, here `default/`, one entry per tier row
(owner, as measured on #62, and #417):

- `lead` (int32): the offset of the row's **Clump Lead** within its Analysis's clump slice, its
  own offset for a lead; or a negative marker. The markers are this ADR's:
  - −1 **absent**: an observed row whose variant the LD Store does not hold;
  - −2 **imputed**: not clumped, by design (decision 5);
  - −3 **ambiguous**: a reference-first indel sharing its sorted spelling (decision 6);
  - −4 **not clumped**: every row of an Analysis no LD Store was mapped for (decision 2).
- `r` (int16): the row's signed r with its lead, **as the LD Store's own code**, signed to the
  Store's A1 of both. The engine returns r = code / r_scale, which `REncoding`
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
row, 0 included, for one that was. From 0.3.0 a Hybrid has one `analyses.tsv`, at its root
(#472), and its two columns count both components' rows, as `n_hits_*` does. Each component's
own row counts are in its group's `clumps` attribute (`rows_absent`, `rows_ambiguous`), not in
a second table. `n_hits_5e8` is unchanged and stays the raw count (owner, amending ADR 0032).

### 9. `format_version` does not move: the Clumps are optional in 0.3.0 (this ADR, against #417's "a `format_version` bump")

The Clumps component and its two columns are **derived, rebuildable data**: the Store's own
tier, clumped by a named LD Store, which #419's on-the-fly path reproduces row for row. Deleting
them changes no existing query answer. That is the class ADR 0061 kept out of a version move for
an Indexed Variant Subset, and the class spec §18 keeps for a tier's `eaf`. Each is optional,
added in place, and discovered through the manifest, here its `clumps` entry. One difference:
a subset rebuilds from the Store alone, and Clumps from the Store and a named LD Store, which is
why the LD Store's identity is recorded (decision 4).

This was first decided at 0.2.x. There, a version move would have been dishonest: the 0.2
versions are cumulative, and the published Stores were 0.2.0 to 0.2.3. That argument is gone.
Every registered Store is rebuilt at 0.3.0 before it is clumped (decision 10). So a 0.3.1 that
meant "0.3.0 and the Clumps" would be true of every Store it stamped. The question is therefore
only ADR 0038's: does a package that knows 0.3.0 and nothing of the Clumps misread, mis-validate
or mis-complete a clumped 0.3.0 Store? The code and a probe say it does not:

- **It opens and reads it.** The closed-envelope rule judges only a release's top-level
  entries (`_validate_closed_envelope`, `opengwasdb/validation/rules/release.py`), and
  `clumps/` is inside `data.zarr/top_hits/p_5e_08/`. `StoreManifest.from_dict`
  (`opengwasdb/store_reader/manifest.py`) ignores a key it does not know, so the `clumps` entry
  is ignored too. The `top_hits.arrays` rule asks a tier only for the arrays its format must
  carry and the two it dropped (`missing_arrays`, `dropped_arrays`, `opengwasdb/top_hits.py`),
  not for any other member. `_top_hit_tier_role` (`opengwasdb/store/arrays.py`) gives
  `top_hits/<tier>/clumps/default/lead` no role. The only callers of `role_for_array_path` that
  would see such a path are the 0.1.0 converter and the SE re-encoder, and neither reaches it.
  No `carries_*` gate (`opengwasdb/store_reader/format.py`) reads anything the Clumps add.
- **Probe** (25 Nov 2026, on `orch/62/417-rebase` after merging `main`). Clumps were grafted onto
  two of main's 0.3.0 fixture releases: a `lead` and an `r` per tier row, the manifest entry
  and both columns. The Hybrid fixture got them in both components (101 Dense Component rows,
  20 Ragged Overflow rows); the Dense fixture got them in its one tier (4 rows). `validate`
  failed no rule before or after. `top_hits(threshold=5e-8)` and `analyses_table()` answered
  identically before and after. #471's spec review found the same on OGS-00002 rewritten to 0.3.0.
- **Its completion drops them whole.** Main's Dense completion of a clumped 0.3.0 source wrote a
  0.3.0 release with no `clumps/` group, no `clumps` entry and neither column. That release
  failed no rule. Every completion writes its manifest from a `StoreManifest`, which has no
  `clumps` field, and its `analyses.tsv` from the Analysis records it knows. So the Ragged and
  Hybrid paths drop them too, though only the Dense path was run. Nothing is half-carried, and the Store Release Build clumps the
  completed release again (decision 10).

So no 0.3.0 reader misreads anything it knew, and **0.3.0 stays**: the Clumps are an optional,
manifest-declared component of 0.3.0, and `CURRENT_FORMAT_VERSION` does not move. #430's 0.2.1
is the contrast. An older package's completion would have completed a Store outside its
regions and dropped `regions.tsv`, and its `validate` refused the file. The patch move was what
stopped that. Neither happens here. A package that predates the Clumps simply does not check
them. Its `validate` passes a clumped Store without judging the Clumps, which only the
`clumps.*` rules of the package that writes them do (spec §20.0).

**What 0.3.1 would cost**, if the owner prefers it:
- #420 would re-stamp each published release's `format_version` in place. That is a second
  in-place exception to ADR 0038, beyond decision 10's.
- Every completion would stamp 0.3.0 again, since it drops the Clumps.
- Every 0.3.0-only package would warn on reading a clumped Store and refuse to complete one.
- The code would gain a `carries_clumps` input to `format_version_for`, 0.3.1 in
  `WRITTEN_FORMAT_VERSIONS` and `SUPPORTED_FORMAT_VERSIONS`, and a stamp rule that changes when
  the step runs.

It would buy nothing the probe found missing.

This goes against #417's text, "A `format_version` bump". The owner gates a format version
(CONTRIBUTING.md, Merging) and signs this decision off when epic #62's pull request merges.

### 10. Written after every build and completion, and in place on an existing Store (owner; this ADR)

The owner's plan (#62, 23 Nov 2026) is one step, run last by every build and in place on every
published Store Release (#418, #420). On 25 Nov 2026 the owner set the order: every registered
Store is first rebuilt at 0.3.0 (opengwas/opengwasdb-stores#498), and #420 then clumps those
rebuilt Stores in place. The step is the component `clumps`, in layer 4 beside
`completion`, because it imports `opengwasdb.ld.query`; the builders are layer 3 and cannot call
it. Its one entry is the CLI command `opengwasdb clump`, and "last by every build" means the
Store Release Build in opengwas/opengwasdb-stores runs it after the build and after any
completion, passing the mapping it holds (opengwas/opengwasdb-stores#454).

**An exception to immutability, narrowed** (this ADR, on the owner's plan). ADR 0038 §4–§5 and
spec §21.4 derive a new release for any change to Analytical Metadata, and ADR 0061 allows an
in-place addition because it changes none. `n_loci_5e8` and `n_unclumped_5e8` are Analytical
Metadata. The exception is exactly this: **a derived, rebuildable component and the count
columns derived from it, recomputable from the Store and a named LD Store**, may be added to or
replaced in an existing release in place. They change no existing value, and the `release_id`
does not change, nor does the `format_version` (decision 9), so nothing in the manifest but
its `clumps` entry is rewritten. Anything else touching Analytical Metadata still derives a new
release. ADRs 0038 and 0061 carry a banner saying so. The step writes the group, then `analyses.tsv` and `overview.html`,
then the manifest's `clumps` entry last: until that entry is written the Store does not declare
the component, and `validate` fails a group it does not declare. Running the step again replaces
all four.

**Clumping needs a 0.3.0 Store** (#418). The step refuses a Store of any other format,
naming its `format_version` and the one it supports. A 0.2 Store's tier stores `abs_z`, and its
Hybrid Dense Component keeps its own Variant Axis. Clumping one would mean a second code path
for Stores that are all being rebuilt (opengwas/opengwasdb-stores#498). A 0.1.0 Store is
rebuilt like any other, not converted and then clumped. The step also needs a 5e-8 tier, and
on a Reference-Completed Store a tier carrying `imputed`. It refuses a Store without either,
naming it.

A Reference Completion of a clumped Store rewrites the tier, so its release carries no Clumps
component, no `clumps` entry and neither column; the Store Release Build then runs the step again.
Every completion rewrites its tiers through `top_hits.write_index`, which replaces each tier group
whole (`create_group(replace=True)`, `opengwasdb/store/arrays.py`) and so deletes `clumps/` with
it: Dense through `write_top_hit_indexes`, Ragged and a Hybrid's Overflow through
`build_ragged_top_hit_indexes`, and a Hybrid's Dense Component through the Dense path. A
`clumps` entry or columns a completion carried over without the group fail `clumps.declared`.

### 11. Under ADR 0102's rule the Clumps are *read*, and ship once the Store Reader reads them (evidence; amends ADR 0102 §3)

[ADR 0102](0102-what-a-store-release-ships.md) §2's rule is applied to a component at the
format being written, here 0.3.0. ADR 0102 §1 makes the two arrays, `lead` and `r` under
`data.zarr/top_hits/p_5e_08/clumps/default/`, its components. The `clumps` attribute and the
manifest's `clumps` entry are keys, which ADR 0102 §1 leaves to the spec. `n_loci_5e8` and
`n_unclumped_5e8` are columns of the root `analyses.tsv`, which is already *read* through
`analyses_table()`.

**Step 1: the Store Reader reads them.** #419's Top-Hit Query reads `lead` and `r` beside the
tier's rows for the stored configuration (decision 1), in `opengwasdb/store_reader/`, where
`TopHitTier.read` (`opengwasdb/store_reader/top_hits.py`) reads the tier. So the class is
**read**, and ADR 0102 §2 puts two limits on that class:

- **They repeat no shipped component.** No other component in the release says which row leads
  which.
- **The accelerator limit.** ADR 0102 §1 defines an accelerator as a read component that can be
  recomputed from the primary data and exists for speed. The Clumps exist for speed: #419 can
  clump the same Analysis on the fly. They cannot be recomputed from the release alone, though,
  because they also need an LD Store outside it, which decision 4 records by digest. So they
  are an accelerator only in part. This ADR applies the stricter reading and states the gain
  against the size. The measurements are #62's (23 Nov 2026,
  [#62 comment 7110664252](https://github.com/opengwas/opengwasdb/issues/62#issuecomment-7110664252);
  the scripts and raw records are in `/data/opengwasdb/work/clump-62/` on the IEU node):
  - **Size.** 11.4 MB for OGS-00009's 5e-8 tier: 1.8 B a row, against a 112 MB tier and a
    58 GB Store. `r` stays `int16` as the LD Store's own code (decision 7), and #418
    re-measures.
  - **Gain.** A clumped Top-Hit read took 18 ms median, against 16 ms unclumped. Clumping on
    the fly took 23 ms median per Analysis, so the median gains little. The gain is in the
    tail and the whole Store. The p99 is 1.0 s and the maximum 5.5 s, both on OGS-00009, and
    every one of its 1,219 Analyses took 126 s in all, which `n_loci_5e8` needs for each
    Analysis. A stored read also needs no LD Store at query time.

**Until #419 lands the Store Reader reads nothing of them.** Under step 2 they are then not
recomputable from the release, and they record no part of how the release was made. ADR 0102
rules the same way for the Rho Matrix: it "ships only once the Store Reader reads it". So #420
adds the Clumps to published Stores only once #419's reader is released. The count columns are
not held back, since `analyses_table()` already reads them, but they come from the same step.

**Amends ADR 0102 §3.** Its inventory gains one row. `data.zarr`
`top_hits/p_5e_08/clumps/default/` `lead` and `r` are **read**: an optional accelerator with
the LD Store recorded (this ADR, decision 11), shipped from #420. Spec §1.1 carries the row,
and ADR 0102 has a banner pointing here.

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

- **#418** writes the step (`opengwasdb clump`, component `clumps`), its validation rules, the
  `clumps` manifest entry and the two columns. The step supports only 0.3.0 Stores and refuses
  any other format by name (decision 10). It takes |z| from `z` (decision 5). It registers the
  two arrays with the tier's role and chunking (`ArrayRole.TOP_HIT_INDEX`,
  `TOP_HIT_CHUNK_SIZE`), so a clumped read touches the chunks the tier's read does.
  `_top_hit_tier_role` maps only `top_hits/<tier>/<leaf>` today, so #418 extends it to the two
  deeper paths. It does not move `CURRENT_FORMAT_VERSION` (decision 9). It measures how many loci
  are lost entirely and records the number in this ADR.
- **#419** reads the component for the stored configuration in the Store Reader,
  `opengwasdb/store_reader/`, beside `TopHitTier.read`. It decodes `r` with the LD Store its
  Analysis's ancestry maps to. #475 moves the Store Reader into the `ogquery` package, and the
  Clumps reader moves with it.
- **#420** runs the step in place on every registered Store once opengwas/opengwasdb-stores#498
  has rebuilt it at 0.3.0, and once #419's reader is released (decision 11).
- **opengwas/opengwasdb-stores#454** passes the mapping and carries the two columns. A change to
  `analyses.tsv` columns is a documentation change there too.
- **#462** removes the shared-spelling exposure (decision 6). Until then the step counts it.
- `validate` gains the `clumps.*` rules (spec §20.0). The re-clump rule needs the LD Stores, so
  it runs only when they are given and is recorded as not run otherwise; a skip is not a pass.
