# Splice: Analyses enter and leave a published Hybrid Store Release by building small and splicing

## Status

Proposed. Issue [#520](https://github.com/opengwas/opengwasdb/issues/520), the design ticket of
epic [#508](https://github.com/opengwas/opengwasdb/issues/508). The implementing tickets are
[#521](https://github.com/opengwas/opengwasdb/issues/521) (drop) and
[#522](https://github.com/opengwas/opengwasdb/issues/522) (add). The registry half,
[opengwas/opengwasdb-stores#546](https://github.com/opengwas/opengwasdb-stores/issues/546),
follows this ADR and amends stores ADR 0004.

Where each decision comes from:

- **The owner's rulings of 28 Nov 2026** (the last comment on #508), marked *(owner)*:
  1. build against format 0.3.0 now;
  2. keep the Store id and write a new Store Release with a lineage list;
  3. drop physically and renumber, with no tombstones;
  4. add by building small, then splicing, with removal as the same operation's other direction.
- **The owner's rulings on this ADR's review** (Gibran Hemani, 28 Nov 2026, in the
  orchestration of #508), marked *(owner, review)*:
  - **D1 = A.** An added Analysis's EAF orientation is verified against the release's stored
    EAF Baselines (§3).
  - **D2 = A.** `eaf` within the residual-step bound counts as equivalent to a union build,
    and kept Analyses' answers never change (§5).
  - **Removal on real data** is proven by a staged, never-published drop from OGS-00011
    (§8c). OGS-00011's first increment is add-only.
- **This ADR**, marked *(this ADR)*.
- **The orchestrator's ruling on F1** (28 Nov 2026, in the orchestration of #508), marked
  *(ruling, F1)*: a release carrying `provenance.lineage` stays at 0.3.0 (§9).

What this ADR changes in other documents:

- Spec §2b (new): `provenance.lineage`, the record a spliced release carries of the releases
  it derives from.
- **Spec §21.1 is amended** (and §21.0 reconciled with it): a `provenance` block that only
  records how a release was made is a compatible change that does not move
  `format_version`. `converted_from` and `lineage` are its two instances (§9). This narrows
  [ADR 0038](0038-store-format-versioning-and-migration-policy.md) §1's "a new `provenance`
  block", and ADR 0038 carries a note saying so.
- CONTEXT.md: two new terms, **Splice** and **Increment**.
- [ADR 0066](0066-withhold-an-rsid-its-sources-dispute.md): §7 supersedes, for a splice
  only, its rule that two allele orders which each agree internally keep the first name.
  ADR 0066 carries a note saying so.
- [ADR 0002](0002-store-local-variant-indices.md) is extended, not contradicted. It already
  says Variant Indices are per release. §2 says the same of `analysis_index`, which no ADR
  stated before.

## Context

A published Store Release's membership is fixed. The only way to add or remove an Analysis
is to rebuild the Store. OGS-00011 (3,245 Analyses, Hybrid, Reference-Completed, 0.3.0) took
**71,851 s to build and 74,651 s to complete**, 40.7 h together, measured
(`/data/opengwasdb/stores/OGS-00011/records/{build,complete}.json`, `elapsed_seconds`). Every
recorded step of that release sums to **168,666 s, 46.9 h** (measured, the same records,
excluding the 0.2.3 → 0.3.0 `rewrite-release` and the zero-length `register` records).

Its first increment adds 89 Analyses (ruling 6, stores#557). stores#557 also named one member
whose scale flips from quantitative to case-control, GCST008970, but it was never an OGS-00011
member, so the first increment drops nothing *(owner, review)*. Removal is needed for defects:
on 26 Nov 2026, 11 Analyses with mis-scaled z were found in the published Hybrids OGS-00032
and OGS-00033 (stores#520, #532).

What the code holds, which this design rests on:

- **The Dense axis does not depend on the data.** A Variant Reference build stores every
  reference ALID as a Dense row, observed or not (`layouts/hybrid/build.py:744`
  `_resolve_reference_panel`). Completion then appends every LD panel variant the source
  lacks (`layouts/dense/complete.py:522` `_union_variant_table`). OGS-00011 grew from
  13,718,605 to 14,564,257 rows this way, 845,652 new panel variants (measured, `complete.json`
  stdout). So two releases built from one Variant Reference, and completed against one LD
  panel, have the same Dense axis row for row.
- **Reference Completion is per Analysis.** It imputes one Analysis at a time within each LD
  block, from that Analysis's own observed z and SE and the panel's LD and EAF
  (`completion/block.py:105` `complete_block_for_analysis`, the loop at `:268`). No
  Analysis's imputed cells depend on another's. **Which Analyses it imputes can depend on
  all of them:** when no row carries `assigned_ancestry`, every Analysis is imputed
  (`completion/ancestry_filter.py:82` `derive_impute_analysis_ids`).
- **The shared axis is sorted, and an off-panel variant is on it only because it is
  observed.** The shared axis is `sorted_alids(panel ∪ off_panel)`, where off_panel = observed −
  panel (`layouts/hybrid/build.py:770` `_partition_variants`; `variants/axis.py:61`). A new
  off-panel variant therefore interleaves into the axis and renumbers every row after it.
  The build already extends and renumbers the axis this way once: `_build_shared_key_table`
  (`build.py:1608`) returns an `old_to_new` map.
- **A key's assembly is decided over the whole build.** A source key declared under both
  assemblies has no single locus and is dropped with its associations
  (`layouts/hybrid/key_table.py:448` `resolve_keys`).
- **Panel crossovers are folded per association.** A completed Hybrid moves each Overflow
  association whose variant the LD panel added onto the Dense Component, as an observed cell
  (`layouts/hybrid/complete.py:803` `_fold_panel_crossovers`). On OGS-00011 that was
  995,632,726 associations (measured, `complete.json` stderr).
- **The Rho Matrix's thinned rows do not depend on the Analyses.** `select_thinned_variants`
  (`layouts/dense/rho.py:63`) takes the first Dense row in each `window_bp` window by
  position. Pair *(i, j)* depends only on Analyses *i* and *j* at those rows (`rho.py:236`).
  The packed strict lower triangle is row-major (`rho.py:455`) and stored as `float16`
  (`rho.py:352`), so Analyses appended at the end append rows to it.
- **A Clumps row depends on the whole shared axis.** An Analysis is clumped against its
  Assigned Ancestry's LD Store (`clumps.py:553` `_clump_analysis`), but whether a row is
  ambiguous depends on whether the axis also holds its reversed indel (`clumps.py:416–444`
  `identities`, `_shares_its_spelling`; ADR 0106 §6). Adding or removing an off-panel
  variant can change a kept Analysis's rows.
- **Exception tables are keyed by flat position.** A Dense table is keyed by
  `variant × n_analyses + analysis` and a CSR table by ordinal (spec §6a). Changing
  `n_analyses` re-keys every entry. `indexed_subsets.py:560` `_remap_side_table` already
  re-keys a Dense table by `index % n_analyses`.
- **The EAF Baseline is the release's.** It is a per-variant median, taken over the release's
  Analyses (`encoding/codec.py:356`). Completion carries the source's baseline rather than
  recomputing it, "so a cell decoded from the source and re-encoded here lands on the same
  code" (`layouts/dense/complete.py:1579` `_carried_eaf_baseline`). The Overflow plane's
  baseline is per shared row, NaN where no Overflow association has an EAF, and completion
  moves it onto the completed axis (`layouts/hybrid/complete.py:1047`
  `_remapped_overflow_baseline`).
- **EAF orientation is checked against a consensus of the other Analyses.** Each Analysis's
  EAF is correlated against the leave-one-out median of every other Analysis
  (`build/eaf_orientation.py:734` `_consensus_baselines`). OGS-00011 records `method:
  consensus`, `reference_id: consensus:3245-analyses`.
- **rsID disputes are decided per routing key.** A routing key that the sources name with
  two different rsIDs is disputed. Under `drop`, every variant it reaches is left unnamed;
  under `refuse`, the build fails (`variants/routing/names.py:75`
  `_candidates_by_routing_key`, ADR 0066). Two allele orders that each agree internally are
  not a dispute, and the variant keeps the first name. The release keeps no routing keys and
  no per-source names. OGS-00011 was built under `drop` with 2,252 disputed keys reaching
  2,152 variants, and its build record lists only the first 20 keys (`build.json` stdout,
  `rsid_conflicts`).
- **The MAF and INFO floors are per Analysis.** OGS-00011 records a MAF threshold of 0.005
  for 3,241 Analyses and 0.04, 0.025, 0.075 and 0.1 for one Analysis each. It records an INFO
  threshold of 0.6 for 628 Analyses, and none (`unavailable`) for 2,617 (measured,
  `provenance.maf` and `provenance.info_score`). Each comes from the Analysis's own manifest
  row (`model/maf_policy.py`).
- **Three inputs are recorded only outside the release.** OGS-00011's manifests do not
  record its rsID policy, its source assembly, a digest of its Variant Reference
  (`provenance.variant_reference` is a path), or which LD panel completion read. On main,
  completion records `ld_panel_id`, a constant label (`eur-hg38-gpm`,
  `layouts/dense/complete.py:137`), whatever panel it was given. These facts are in the
  registry's step records: `build.json` argv (`--on-rsid-conflict drop`,
  `--source-assembly hg38`, `--variant-reference …/EUR-variants.txt.gz`) and `complete.json`
  argv (`--ld-panel /data/opengwasdb/reference/ld`). #327 makes completion record the panel it
  read as `provenance.completion.ld_panel_digest`. It is on `orch/367-silent-errors`, not on
  main.
- **rewrite-release already stages, stamps and proves.** `store/rewrite.py:782`: a copy by
  `_copy_release` (`:982`), which reflinks through `copy_path` (`store/convert.py:1154`,
  `cp -a --reflink=auto`); named steps; a fresh `release_id` on every manifest (`:1017`); an
  answer-digest proof (`:1089`); and a full `validate`, all inside `OpenGWASDBStore.staging`
  (`store/open.py:748`, ADR 0043). Without `--convert-legacy`, `_refuse_version` (`:937`)
  refuses every source but 0.2.3. Behind the flag it converts a 0.2.0–0.2.2 Dense release
  (ADR 0105, #492 amendment). Its proof requires every answer unchanged.
- **The answer digests cannot be remapped.** `ogquery.digests` hashes whole answer arrays,
  which hold `analysis_index` and `variant_index`. It picks its questions from the release's
  own Analyses (`ogquery/digests.py:532`, `_spread` at `:233`). After a splice, both the
  questions and the indices differ.

## Decision

### 1. One operation, two directions: `splice-release` *(owner: ruling 4; interface: this ADR)*

One engine owner, `opengwasdb/layouts/hybrid/splice.py`, does both directions in one staged
pass.

```text
opengwasdb splice-release SOURCE DESTINATION
    [--add INCREMENT]                 a Reference-Completed Hybrid built and completed for SOURCE (§3)
    [--drop ID[,ID…]]                 analysis_ids to remove
    [--source-record FILE]            facts SOURCE's manifests lack, from the registry's records (§3)
    [--ld-store ANCESTRY=PATH …]      one per ancestry SOURCE's Clumps name; required when it has Clumps (§4)
    [--indexed-subset NAME=PATH,POLICY …]  one per subset SOURCE carries: its variant list and overflow policy (§4)
    [--reason TEXT]                   recorded in the lineage entry, e.g. "erratum: stores#532"
    [--plan]                          check every refusal and print the maps and the work; write nothing
    [--n-workers N]
```

```python
def splice_release(
    source: str | Path,
    destination: str | Path,
    *,
    add: str | Path | None = None,
    drop: Sequence[str] = (),
    source_record: SourceRecord | None = None,
    ld_stores: Mapping[str, str | Path] | None = None,
    indexed_subsets: Mapping[str, SubsetInput] | None = None,
    reason: str | None = None,
    n_workers: int = 1,
) -> SpliceRun: ...

def plan_splice(source, *, add=None, drop=(), source_record=None) -> SplicePlan: ...  # refusals, maps, counts
```

- **A replace is a drop and an add in one invocation.** An `--add` Analysis whose
  `analysis_id` SOURCE already holds is refused unless `--drop` names it too. A replace is
  therefore one splice, one new release and one lineage entry.
- **What it reuses from `rewrite-release`:**
  - `refuse_destination` and `OpenGWASDBStore.staging(destination)`, so the source is never
    written and the commit takes the parent-directory lock (ADR 0043);
  - `_copy_release` (`store/rewrite.py:982`), for every file splice does not rewrite (§4);
  - the `_stamp` pattern: one fresh `release_id` and `created_at` on the root manifest and on
    the nested Dense Component's manifest;
  - `_refresh_overviews`;
  - `require_valid_staged_release`, a full `validate`.

  It does not reuse rewrite's proof, which requires every answer unchanged. It has its own
  (§8). It does not reuse `_refuse_version` either: splice accepts exactly 0.3.0, the
  version it writes (§9, §11).
- **Never in place.** The previous release stays where it is. Moving it to
  `replaced/<UTC>/` and putting the new one at the published path is the registry's job
  (ruling 2, stores#546).
- **Scope.** A Reference-Completed Hybrid release, which is what OGS-00011, OGS-00032 and
  OGS-00033 are, and a Reference-Completed Increment. An Observed-Only Hybrid, a Dense
  release and a Ragged release are refused by name. The same surgery would apply to them,
  but nothing asks for it.

### 2. The index maps *(this ADR, extending ADR 0002)*

Splice computes three maps before it writes anything, and every structure in §4 is rewritten
through them. These maps are the whole of what "under the index mapping" means.

- **`analysis_map`: old `analysis_index` → new.** Kept Analyses keep their relative order.
  Dropped Analyses have no image. Added Analyses follow, in the Increment's `analysis_index`
  order. The map is monotone on the kept Analyses.
- **`shared_map` and `increment_shared_map`.** The first maps the release's shared rows onto
  the new shared axis. The second maps the Increment's shared rows onto it. The new axis is
  `sorted_alids(dense_axis ∪ observed_off_panel)`, where observed_off_panel is computed after
  the drop and the add. Both maps are monotone, so each Analysis's Overflow `variant_index`
  stays strictly increasing. They are computed as integer arrays by `AlidIndex`
  (`layouts/hybrid/key_table.py:226`), which binary-searches the axis's sorted ALID hashes
  and confirms each hit against the ALID string, as `_build_shared_key_table` does
  (`build.py:1608`). They are not built through an ALID → row dict.
  `complete.py:360` `_remap_shared_axis` builds per-association Python lists, and its phase
  took 8 h 29 m on OGS-00011 (05:17:40 → 13:47:22, `complete.json` stderr, measured).
- **The Dense row map is the identity.** The Dense axis is fixed (Context, first bullet), and
  §3 refuses any Increment built from another Variant Reference or completed against another
  panel.

**`analysis_index` is release-scoped, exactly as ADR 0002 makes `variant_index`.** The stable
key of an Analysis is `(store_id, analysis_id)`. **A stale index fails no check today.**
`ogquery/analyses.py:108` `require_index` raises `UnknownAnalysisError` for an unknown
`analysis_id`, and the rule `analyses.analysis_index` requires indices to cover `0..n-1`
(spec §20.0). Neither notices an index cached from SOURCE: after a splice it is still in range
and names whichever Analysis now holds it. So **every ogquery entry point that takes an
`analysis_index` must be given the `release_id` it came from, and must refuse a mismatch**.
Examples are `ogquery/rho.py:128`, `top_hits.py:250` and `indexed.py:240`. The same applies to
a `variant_index`, which an add or a drop renumbers too. The guard rides in the ogquery
release that #521 already needs (§8).

### 3. The Increment, and what it is pinned to *(owner: ruling 4, D1; the table: this ADR)*

An **Increment** is a small Hybrid release that holds only the Analyses to add. It is built
with `build-hybrid … --increment-of SOURCE` (#522) and completed with the usual
`complete-hybrid` against the release's LD panel and ancestry. Splice checks the result
against SOURCE and refuses every mismatch by name before staging anything.

**A pin is a recorded identity, never a resemblance.** Matching ALIDs, panel EAF and `on_panel`
cannot establish that two completions read the same LD matrices. Where SOURCE's manifests do
not record an identity, the registry supplies it in `--source-record`, a JSON object read from
its own step records. Splice accepts a key there only when SOURCE records none, refuses it when
SOURCE records a different value, and copies every key it used into the lineage entry (§9).

| Pinned | SOURCE records it | The Increment records it | What refuses a mismatch |
|---|---|---|---|
| Variant Reference | a path only (`provenance.variant_reference`). For OGS-00011, the registry supplies the file's sha256 from `build.json` argv's `--variant-reference` | its sha256 (#522) | digest equality. The completed Dense axes are also compared ALID for ALID, through `dense_to_shared` and the shared ALID sidecar, as a consistency check |
| LD panel | `provenance.completion.ld_panel_digest`, from #327 (not on main; #522 depends on it). OGS-00011 predates it: the registry supplies the digest of `complete.json` argv's `--ld-panel`, computed by #327's function | `ld_panel_digest` (#327) | digest equality, plus equality of `ancestry`, `method`, `min_cor` and `pca_thresh`, and a bitwise-equal `eaf_reference` and `on_panel` as consistency checks |
| Store Encoding: `z` kind and scale, `se` kind | `manifest.encoding` | the same | equality. The Increment must carry `eaf` exactly (`float32`, or `absent` when no added Analysis has EAF), so splice quantises each new cell once, under the release's plan (§5) |
| Reference assembly, chain file, Association Coverage | `reference_assembly`, `provenance.chain_file`, `association_coverage` | the same | equality |
| Source assembly | not recorded. For OGS-00011, the registry supplies `build.json` argv's `--source-assembly hg38` | recorded (#522) | one assembly for both. A splice that would mix assemblies is refused, because a union build drops a key declared in both (`key_table.py:448`) and a splice cannot find such keys afterwards |
| `format_version` | `format_version` | the same | both 0.3.0, the version splice writes (§9) |
| rsID conflict policy (ADR 0066) | not recorded, so it cannot be authenticated from SOURCE. For OGS-00011, the registry supplies it from `build.json` argv (`--on-rsid-conflict drop`) and `validation.yaml` (`rsid_conflicts.policy: drop`) | recorded in provenance. From #522 on, every `build-hybrid` records it, not only an Increment's | equality |
| EAF orientation | `provenance.eaf_orientation` (`method`, `min_overlap`, `min_variance`, `allow_unverified`) | the same settings | equality of the settings. Each added Analysis is verified against SOURCE's EAF Baselines (D1, below) |
| Ancestry filter | each row's `assigned_ancestry` | the same | each added Analysis is imputed exactly when its `assigned_ancestry` matches the panel's, by the rule a union build applies to all rows. #522 passes `impute_analysis_ids` computed over SOURCE's and the Increment's rows together, so the no-ancestry fallback (`ancestry_filter.py:82`) never fires for an Increment whose own rows carry no ancestry. Splice refuses an added Analysis whose imputed cells disagree with that rule |
| Clumping LD Stores | the Clumps group's `clumps` attribute: `ld_stores`, each identified by what a reader can recompute (ADR 0106 §4) | none needed | each `--ld-store` must match the identity recorded for its ancestry |
| Rho settings | `rho` attributes (`window_bp`, `z_thresh`, `min_nulls`) | none needed | not inputs: splice reads them from SOURCE |
| MAF and INFO floors | per Analysis, `provenance.maf.analyses[]` and `provenance.info_score.analyses[]` | per Analysis | not pinned. They are per Analysis (Context). Each added Analysis's comes from its own manifest row, which is what a union build would apply. Splice carries the entries |

**EAF orientation: verified against the release's EAF Baselines** *(owner, review: D1 = A)*.
`--increment-of` verifies each added Analysis against SOURCE's stored `eaf_baseline`. That is
the per-variant median of every SOURCE Analysis's EAF, in logit space (`encoding/codec.py:356`).
It records `reference_id: release:<SOURCE release_id>`, and costs a read of the baselines.
A union build's leave-one-out consensus (`build/eaf_orientation.py:734`) differs from this in
three ways:

- it includes the other added Analyses;
- it excludes any Analysis the same splice drops (a replaced Analysis's earlier version is in
  SOURCE's baseline);
- it is a median of frequencies at sites where at least two other Analyses report one, while
  the baseline counts any site where one SOURCE Analysis does.

The verification can therefore give a different verdict from a union build's only for an
Analysis whose correlation sits near the threshold (§8, difference 5).

Splice also refuses:

- an `--add` id already held and not also dropped, or a duplicate id;
- a `--drop` id that SOURCE does not hold;
- a splice that would leave no Analysis;
- an `--add` into a release whose `se` is `int8_residual`. Its codes are residuals against each
  cell's EAF and the Analysis's `se_coefficients`, and an added Analysis without EAF cannot be
  coded into the plane (ADR 0037 §3). A drop into such a release is allowed: it codes no new
  cell. No published Hybrid has such a plane today. OGS-00004, OGS-00005, OGS-00011, OGS-00032
  and OGS-00033 are all `float16` (measured, their manifests, 28 Nov 2026);
- a release whose `eaf` is `absent` when the Increment carries EAF, which would otherwise be
  dropped silently.

### 4. What splice rewrites, structure by structure *(this ADR)*

`a` is old `analysis_index`, `A(a)` is `analysis_map`, `S(v)` is `shared_map`, and `n′` is the
new Analysis count. "Carried" means the kept bytes are copied without decoding.

| Structure | How splice writes it |
|---|---|
| Dense planes `z`, `se`, `eaf`, `imputed` (spec §10) | Column surgery: kept columns are carried, code for code, to column `A(a)`, and the Increment's columns are appended. Shards are [100000, 1024] (spec §10a), and a shard file is rewritten only when one of its columns changes. An add-only splice rewrites only the last shard column. A drop rewrites every shard column from the first dropped index onward, and at worst the whole plane. Shard files splice does not touch are reflinked |
| Dense exception tables `z_overflow_*`, `eaf_exception_*` (spec §6a) | A kept key `v·n + a` becomes `v·n′ + A(a)`. A dropped column's entries are removed. The new columns' entries are merged in. Since `A` is monotone, the kept order holds and the merge is one streaming pass |
| Dense `eaf_baseline` | Carried at every row where SOURCE's is defined (§5). A row where SOURCE's is NaN has no SOURCE cell coded against it, and takes the baseline the owner (`encoding/codec.py:356`) computes from the Increment's exact `eaf` there. That is what a union build computes at such a row |
| `eaf_reference`, `on_panel` | Carried unchanged. They are proven equal to the Increment's (§3) |
| `se_coefficients[n, 2]` | Rows filtered and appended. Absent under `float16` |
| Overflow CSR `offsets`, `variant_index`, `z`, `se`, `eaf` (spec §11) | Segment surgery: kept segments in `A` order, with `variant_index` → `S(v)`; then the Increment's segments with `increment_shared_map`. Exception tables re-keyed by the new ordinals |
| Overflow `eaf_baseline` | Moved onto the new axis through `S`, as `_remapped_overflow_baseline` does (`layouts/hybrid/complete.py:1047`). A row where SOURCE's is NaN, including every off-panel variant only the Increment observes, takes the baseline computed from the Increment's exact `eaf` there, as for the Dense plane. A row left with no Overflow association after a drop becomes NaN |
| `ragged/by_variant/` (spec §11.1) | Rebuilt from the new CSR by its owner, `layouts/ragged/by_variant.py:480` `build_variant_index`. It is derived and variant-ordered, and a rebuild is simpler than a merge (13 m 29 s on OGS-00011's completion, measured, stderr 16:03:57 → 16:17:26) |
| Shared axis: `variants.tsv.gz` and `.tbi`, ALID, position and `source_alid` sidecars, rsID index (ADR 0105) | Rewritten for the new axis by `variants/axis.py:118` `write_variant_axis`. Names merged by §7 |
| `dense_to_shared.npy` | Values remapped through `S`. Still strictly ascending, because `S` is monotone |
| Top-Hit Index tiers, both components (spec §18) | Rebuilt from the spliced planes by their owners: `layouts/dense/top_hits.py:307` `build_top_hit_indexes` and `layouts/ragged/top_hits.py:155` `build_ragged_top_hit_indexes`. The tier writer regenerates `analysis_offsets` from the rows it writes (`opengwasdb/top_hits.py:148`), so no offset is remapped. A tier is a function of each Analysis's own cells, so every kept Analysis's slice comes out identical under `A` and `S`. An added Analysis's tier `eaf` is the value the spliced plane decodes. The single-worker Dense rescan after the fold took 1 h 16 m in OGS-00011's completion, so splice runs it with workers |
| Clumps (spec §18.1) | **Every Analysis is clumped again**, kept and added alike, by `clumps.compute_clumps` (`clumps.py:522`) against the `--ld-store` mapping. A kept row cannot be carried, because its ambiguity depends on the whole shared axis (Context). Clumping all of OGS-00011 took 591 s (measured, `completion-clump.json`). The tier writer replaces the whole tier group, and the Clumps with it (spec §18.1), so this runs after the tiers. The `clumps` manifest entry is written last, as ADR 0106 §10 requires |
| Rho Matrix (`dense/data.zarr/rho`) | Kept pairs carried: the triangle is re-packed to the kept rows and columns. Pairs involving an added Analysis are estimated by `rho.py`'s estimator at the release's thinned rows and appended. That reads every kept Analysis's `z` at those rows. `overflow_only` is remapped through `A`, plus the added Analyses that have no Dense observation. It is not a full rebuild, so no kept pair is re-estimated (§8) |
| Indexed Variant Subsets (spec §10b) | Rebuilt by `build_indexed_subset` (`layouts/dense/indexed_subsets.py:1290`) from the spliced grid, with the release's variant list and overflow policy. The subset records only `input_sha256`, its counts and a 20-ALID sample of the overflow-only ALIDs (`:951`). It does not record the list's path or the `--overflow-only` policy. Those are in the registry's step record. For OGS-00011, `completion-indexed-subset-hm3.json` argv gives `--variant-list /data/opengwasdb/reference/hapmap3-hg38/hm3.grch38.alid.txt --overflow-only skip`. Each `--indexed-subset` supplies both. Splice refuses a list whose sha256 is not `input_sha256`, and a `refuse` policy where the subset recorded overflow-only ALIDs (`overflow_only_count` > 0; 10,820 for OGS-00011's `hm3`). A subset plane is a re-keyed copy of the grid's codes (`:560`), so kept rows come out identical |
| `completion_quality` (`dense/index.sqlite`) | Kept rows renumbered through `A`, the Increment's rows inserted with their new index. PK `(analysis_index, block_id)` (`completion/schema.py:27`) |
| `analyses.tsv` (spec §7a) | Kept rows renumbered, the Increment's rows appended. `n_hits_*` recounted from the spliced tiers by the recount owner (`store/top_hit_counts.py:90`), so ADR 0107's rule holds by construction. `n_loci_5e8` and `n_unclumped_5e8` come from the new Clumps |
| Manifests (root and `dense/`) | `n_analyses`, `n_variants`, the `hybrid` counts, `ragged.by_variant` and completion totals recomputed. Per-Analysis provenance lists (`maf.analyses`, `info_score.analyses`, `eaf_orientation.analyses`, `completion.se_out_of_range.imputed`) are filtered and appended by `A`. `lineage` is appended (§9) |
| `overview.html` | Regenerated (`overview.write_overview_html`) |

### 5. The encoding plan and baselines are the release's, and are never re-derived *(this ADR; owner, review: D2 = A)*

**Splice codes every new cell under the release's plan and against its EAF Baselines. It
never re-derives either for the union.** The one exception is a baseline that SOURCE leaves
undefined (§4).

- **Every plan splice accepts can hold any value.**
  - A `z` beyond `int16_fixed`'s range, and an `eaf` residual beyond the range, become exact
    exceptions.
  - An `se` above 65,504 is stored as missing and counted, under every plan (ADR 0068).

  The plans that cannot hold a new Analysis are refused in §3. No value is ever clipped.
- **Re-deriving would rewrite the kept Analyses.** A new baseline re-points every carried
  `eaf` code. The release keeps no source floats, so each kept cell would be decoded and
  quantised a second time, and the kept answers would change. It would also cost a full `eaf`
  re-encode of both components (`Encode eaf: done in 1h42m` for the Dense plane at OGS-00011's
  build, measured, `build.json` stderr).
- **Completion already made this decision** for the same reason: it carries the baseline and
  never refits `se_coefficients` (ADR 0037 §3, ADR 0078).
- **The bound, and why it counts as equivalent** *(owner, review: D2 = A)*. Kept Analyses'
  answers never change. A new `eaf` cell is within half a residual step of its exact value,
  as in any build. But a union build may code the same value against another baseline and
  another `residual_range`. So the two decoded values differ by at most **the sum of both
  builds' half-steps**: `r_splice/254 + r_union/254` in logit, where each `r` is that build's
  own recorded `residual_range` (step `r/127`). The same bound holds between a kept cell and
  the union build's code for it. At OGS-00011's ±1.0, splice's half-step is 0.0039. The union's
  term is measured, never assumed: the equivalence test reads `R_union`'s recorded range
  (§8a). Splice records each added Analysis's exception share and reports it against ADR
  0037's 2% budget without enforcing it, as completion does (ADR 0078).

### 6. Variants left with no association after a drop are removed *(this ADR)*

An off-panel variant observed only by dropped Analyses leaves the shared axis. The axis is
rebuilt as `dense_axis ∪ observed_off_panel` (§2), which is the build's own invariant
(`build.py:770`: "Nothing observed is dropped, so an off-panel variant is always on the shared
root axis"; the converse holds too, since nothing unobserved is added). Dense rows never
leave: the Dense axis is the panel's.

Consequences:

- A drop renumbers the shared axis, so every sidecar and `dense_to_shared` is rewritten.
  That machinery is the add's, run in reverse.
- The pruned axis is the axis a rebuild of the kept Analyses forms, provided that rebuild
  admits the same associations. The admissions that depend on other Analyses are named in
  §8a, and each one is either refused by §3 or listed there as a permitted difference.
- A defect removal leaves no association-less variant behind for a reader to find with an
  empty PheWAS. Keeping such variants would pass `validate`: no rule requires an off-panel
  row to have an association. It was rejected because the release would then claim variants
  that no Analysis in it observed.

### 7. Variant names at the merge *(this ADR; supersedes part of ADR 0066 for a splice)*

ADR 0066 decides disputes per routing key, from every source's names. A splice has only each
release's final name per variant. A blank (`.`) there means either that no source named the
variant, or that its sources disputed the name and the build withheld it (2,152 such variants
in OGS-00011). Nothing in the release says which. The build record lists only 20 disputed
keys. Recovering the names would mean re-reading every source: Pass 1 took 7 h 00 m on
OGS-00011, measured.

So splice decides per variant, by ALID, and **never names a variant that a source might
dispute**:

| SOURCE's name | The Increment's name | Splice writes |
|---|---|---|
| `rsA` | `rsA` or blank | `rsA` |
| `rsA` | `rsB` | blank under `drop`; refused under `refuse` |
| blank | `rsA` | blank. The blank may be a withheld dispute |
| (variant new to the release) | `rsA` or blank | the Increment's name. Only the Increment's sources name it, so a union build decides the same |

**Where this differs from ADR 0066, and why it is superseded here.** ADR 0066 keeps the first
name when two allele orders each agree internally but differ from each other. A splice cannot
see allele orders, so in the second row it withholds a name a union build would keep. In the
third row it leaves blank a name a union build would give, when the blank was never a dispute.
Both cases lose a name and never give a disputed one. A build that recorded its disputed
variants would let a splice tell the two blanks apart. No build records them today.

**`source_alid`.** If the two disagree, the row becomes "none: two source variants collided
onto one row", the sidecar's existing state (ADR 0105 §8).

**A drop cannot undo a withheld or collided state.** A name that a now-dropped Analysis's
source made ambiguous stays absent, for the reason above.

Splice counts and lists every variant whose name or `source_alid` it changes, and every variant
it left blank under the third row (§12).

### 8. Equivalence: what is proven, where, and the differences permitted *(this ADR; owner, review)*

Three proofs. None is a second full OGS-00011 build.

**(a) The equivalence test, on a small Trial Store, in the test suite and gating #522's
merge.** Build B, complete it and run the post steps: that is `R_B`. Build an Increment for I
and complete it. Splice it into `R_B`: that is `R_splice`. Separately, build and complete
B ∪ I from scratch: that is `R_union`. Compare every structure exhaustively, keyed by
`analysis_id` and ALID. The two must be **identical** except for the permitted differences
below. Each one has a fixture case that produces it, and the test asserts the difference is
exactly the one named.

| # | Permitted difference | Fixture case |
|---|---|---|
| 1 | `analysis_index` order: the union orders by manifest row, splice appends | I's manifest rows interleaved with B's |
| 2 | `eaf` codes, and therefore exception-table membership: decoded values within `r_splice/254 + r_union/254` (§5), with both ranges read from the releases. The same applies to `eaf` in tiers and Indexed Variant Subsets | a variant where I's EAFs move the union median, and one where the union chooses another `residual_range` |
| 3 | the encoding plan, where the union would choose another. Splice keeps `R_B`'s | an I whose `z` would widen the union's plan |
| 4 | Rho values of kept pairs, within one `float16` ulp; `n_null` exact. Float64 summation order differs between products of different widths (`rho.py:236`, BLAS), and #512 measured such differences at 8 float64 ulps in another rebuilt store, now held to 1e-12 of the largest value. Either is far below a `float16` ulp (about 1e-3 relative), so two such values round to the same or adjacent codes | B wide enough that the union's products block differently |
| 5 | `provenance.eaf_orientation`: every Analysis's `r` and the `reference_id` (D1, §3). A verdict may differ only near the threshold. The fixtures hold no Analysis within 0.05 of it, and the test asserts the verdicts agree | an I whose consensus `r` differs measurably between the two references |
| 6 | names: a variant both name differently is withheld, and a variant SOURCE left blank stays blank (§7) | one opposite-order pair named consistently on each side, one blank in B that I names, and one genuine dispute |
| 7 | `release_id`, `created_at`, `lineage` | every run |

**Cases that are not differences, because §3 refuses them or splice matches the union:**

- **Variant axis.** The spliced shared axis is the sorted union (§2), the same axis a union
  build forms, so every `variant_index` agrees. Fixtures: an off-panel variant that only I
  observes and that sorts between two of B's, a panel crossover in I, a column move past a
  shard boundary, a `z` beyond `int16_fixed`'s range, and an Overflow-only Analysis.
- **Source assembly** (`key_table.py:448`). Refused unless both builds declare one assembly.
  Fixture: an I declared hg19 into a B declared hg38 is refused.
- **Ancestry filter** (`ancestry_filter.py:82`). Matched: the Increment's completion imputes by
  the union's rule (§3). Fixture: an I whose rows carry no `assigned_ancestry`, into a B whose
  rows do. Completed by the union's rule it splices, and completed alone it is refused.
- **Overflow and Dense baselines.** Remapped through `S`, with SOURCE-undefined rows taking the
  Increment's (§4). These rows match the union exactly. Fixture: an off-panel variant with
  EAF that only I observes (exact) and one both observe (difference 2).
- **Clumps.** Re-clumped from identical tiers on an identical axis, so identical. Fixture: a
  reversed indel that only I observes, which makes one of B's rows ambiguous.

A drop is tested the same way, as `splice(R, drop D)` against the release built from the kept
Analyses. The permitted differences are 2, 4, 5, 6 (as names a dropped Analysis made
ambiguous) and 7. Fixtures add an off-panel variant observed only by D, absent from both
results, and a reversed indel observed only by D, which changes a kept row's Clumps in both.

**(b) The splice proof, run by every splice before it publishes,** in the place
`_require_same_answers` holds in `rewrite_release`. It has three parts.

1. **The cell proof, exhaustive.**
   - Every kept Analysis decodes to the same values in every plane, cell for cell, under
     `A` and the identity Dense map. The same holds for its Overflow segment under `S`.
   - Every added Analysis decodes to the Increment's values: `z`, `se` and `imputed`
     exactly, `eaf` within half the release's step of the Increment's exact `float32`.

   It is a banded read of both grids and both CSRs.
2. **The answer proof, by projection.** Questions are posed by `analysis_id` and ALID, never
   by index. Each answer is projected before anything is compared:
   - **Numbers are compared before hashing**, as decoded arrays: exactly for a kept Analysis,
     and for an added one against the Increment, with `eaf` within the step. A digest is
     taken only of what has already been compared.
   - **Names are compared separately from numbers.** A kept variant's rsID and `source_alid`
     must equal SOURCE's, except at the variants splice listed (§7). Each listed variant
     must be in the state §7 gives, and the list must be exactly the set of differences.
   - **Indices are projected through the maps.** `analysis_index` → `analysis_id` and
     `variant_index` → ALID in each release's own terms. A PheWAS or range answer is compared
     row by row on `(ALID, analysis_id)`. SOURCE's rows for kept Analyses must equal the
     spliced release's, and the spliced rows for added Analyses must equal the Increment's
     answer to the same question. A Rho answer for kept ids must equal SOURCE's for the same
     ids, bitwise.
   - **Explicit probes for every added and dropped id.** An added id answers as the Increment
     does. A dropped id is refused with `UnknownAnalysisError`, never answered empty. A
     pruned variant is reported absent from the axis, as any ALID the release lacks is, and
     never as a present variant with no associations.
   - **Clumps are not compared with SOURCE's.** They are re-clumped (§4), and a kept row can
     legitimately change. `validate`'s `clumps.*` rules judge them, as after any build's clump
     step.

   The question set is SOURCE's own spread from `ogquery.digests`, plus every added and
   dropped id and every off-panel variant splice added or removed.
3. **`validate`, in full.**

Any difference discards the staging directory and leaves SOURCE untouched.

**ogquery's digests cannot simply be made mapping-aware, but its question set can.** A digest
of an array that holds indices cannot be translated after the fact. The questions can be posed
by key and the answers projected before digesting. `digests.py` is ogquery's single owner of
the question set (its docstring), so this belongs there: `release_answers` gains the reference
release's identifiers and a projection. With the `release_id` guard (§2), that is one ogquery
minor release, tracked under #521.

**(c) Removal on real data** *(owner, review)*. OGS-00011's first increment drops nothing, so
it proves only the add. Removal is proven on real data by a **staged, never-published drop**
of a few of OGS-00011's Analyses: `splice-release --drop` into a scratch destination, then
the proof of (b), with `validate`, and then the destination is discarded. The Analyses chosen
should include index 0 (the worst case: every shard column rewritten), the last index, and an
Analysis with Overflow associations no other Analysis observes, so that the axis is pruned.
#521 carries it out and records its timings, which replace §10's drop-only estimate.

### 9. Lineage and records *(owner: ruling 2; the field: this ADR)*

Every splice appends one entry to `provenance.lineage` (spec §2b) in both manifests. The
first splice of a release starts the list. An entry records:

- `source_release_id`, and the `increment_release_id` (or `null` for a drop-only splice);
- `added` and `dropped`, as `analysis_id` lists;
- `n_analyses_before` and `n_analyses_after`;
- `source_record`: the keys of `--source-record` splice used (§3), or `null`;
- `reason`, `at`, `tool` (`opengwasdb splice-release`), `opengwasdb_version` and
  `opengwasdb_git_hash`.

The field lives in `provenance`, not at the top level. ogquery's `StoreManifest` drops
top-level keys it does not know when it writes a manifest back (`ogquery/manifest.py:116`
`from_dict`, `:143` `to_dict`). Completion writes manifests that way
(`layouts/hybrid/complete.py:485, 973`), so a top-level `lineage` could be lost silently,
whereas `provenance` is carried as an opaque dict by every writer. A new rule,
`manifest.lineage` (#521), holds the last entry to the release:

- `n_analyses_after` must equal `analyses.tsv`'s row count, and
  `n_analyses_before − |dropped| + |added|`;
- every added id must be in `analyses.tsv`, and no dropped id that was not re-added.

**Concurrency: the parent is checked at publication.** `lineage[-1].source_release_id` is the
release this splice read. The engine's half is that field, which `--plan` also prints. The
registry's half (stores#546) is that publication refuses when the live release at the
published path is no longer that parent, for example when another splice or a rebuild was
published in between. The operator then re-runs the splice against the new live release.
ADR 0043's lock serialises the commit of a staging directory, not the choice of parent, so it
cannot prevent a stale-parent publication alone.

**A release carrying `lineage` stays at 0.3.0, and spec §21.1 is amended to say why**
*(ruling, F1: the orchestrator, 28 Nov 2026, in the orchestration of #508)*. Before the
amendment, §21.1 classed "a new `provenance` block" as a compatible change that moves the
remainder, which would have made a spliced release 0.3.1. The ruling takes the other branch,
on this evidence:

- **No 0.3.0 package misreads `lineage`.** This is ADR 0106 §9's test. The field is opaque
  `provenance`, no rule of such a package judges it, and its completion refuses a completed
  source anyway.
- **Moving would fail releases for metadata no reader interprets.** Every package before
  #521 would fail a spliced OGS-00011 on `clumps.declared`, because Clumps are allowed only on
  0.3.0 (`opengwasdb/clumps.py:84` `CLUMPS_FORMAT_VERSIONS`,
  `validation/rules/clumps.py:204`). The same packages would refuse to clump it
  (`clumps.py:219`), to build its `by_variant` (`layouts/ragged/by_variant.py:698`) and to
  complete it (`store/open.py:247`). A query reader would only warn: ogquery 0.5.0 reads a
  0.3.1 release with a warning (`ogquery/format.py:337` `check_format_version`).
- **Precedent.** `provenance.converted_from` (#492, ADR 0105) is a provenance block added at
  0.3.0 with no move, and until now with no rule in §21.1 to say why.

So §21.1 now names a class, *a record of how a release was made*: a `provenance` block that no
reader needs to decode an array or answer a query, and that only says how the release came to
be. Such a block is compatible and moves nothing. `converted_from` and `lineage` are its two
instances. A block that a reader must interpret still moves the remainder. §21.0 says
explicitly that it is a different rule: §21.0 edits values already in published Stores, while
this class appears only in releases written from then on. `format_version_for`,
`WRITTEN_FORMAT_VERSIONS` and `docs/store-format-compatibility.md` are unchanged.

**The Validation Record and `build_elapsed_s` are the registry's** (stores ADR 0004, amended
by stores#546). The engine's position, for that ADR:

- **The published release's `build_elapsed_s`** is the sum of the steps that produced it: the
  Increment's build and completion, the Increment's Observed-View Check, and the splice. It
  is not cumulative. The lineage links to the previous release's records, so the cumulative
  cost can be derived and is never double-counted.
- **`validate` runs once.** The splice's staged `validate` (§1) is the release's. The
  registry records its result rather than running it a second time.
- **The Observed-View Check runs on the Increment's observed → completed pair only.** The
  splice proof's cell proof already covers the kept Analyses exhaustively, where the check
  samples.

### 10. Cost model: OGS-00011's first increment *(this ADR)*

89 added, none dropped. **M** is a measurement, with its record. **E** is a projection, with
its basis. Every splice row is **E**, since no splice has run.

| Step | Time | Basis |
|---|---|---|
| **Rebuild today**: build + completion | 40.7 h | **M** `build.json` 71,851 s + `complete.json` 74,651 s |
| Rebuild, every recorded step | 46.9 h | **M** sum of `records/*.json` (Context) |
| Increment build, 89 Analyses | 1–1.5 h | **E** from **M** small builds that run the same phases: OGS-00032, 369 Analyses, 4,696 s; OGS-00033, 193 Analyses, 6,342 s (`records/build.json`). Fixed costs dominate: Pass 1 merge 16 m 52 s and 17 m 14 s |
| Increment completion | 1–2 h | **E** from **M** OGS-00032, 5,846 s, and OGS-00033, 7,170 s, whose LD-block phase was flat in Analyses (51 m 12 s for 369, 52 m 52 s for 193). OGS-00011's own LD-block phase scales to 14 min for 89 (8 h 30 m × 89/3,245) |
| Increment Observed-View Check | 5–30 min | **E** from **M** 12,129 s for OGS-00011's 3,245 (`completion-observed-view-check.json`), × 89/3,245 = 5.5 min, with fixed costs allowed for |
| Splice: Dense planes | 5–10 min | **E** add-only: only the last shard column changes, 262 of 3,334 columns (7.9%) in every row shard. Basis: completion's whole-grid `Band write: done in 43m30s` (**M**, `complete.json`), which writes every plane from decoded values, where splice moves codes |
| Splice: exception re-key, Overflow segment surgery and axis remap, baselines, shared sidecars, `by_variant` | 1–1.5 h | **E** from **M** completion's shared-table and Overflow phase (56 m 36 s, stderr 15:07:21 → 16:03:57) plus `by_variant` (13 m 29 s), at the same 1.71e9 Overflow associations. Requires §2's integer maps, not `_remap_shared_axis` |
| Splice: tiers rebuilt | ~15 min | **E** from **M** OGS-00011's completion: Dense scan and write 5 m 42 s at 24 workers, Overflow scan 7 m 26 s |
| Splice: Clumps, every Analysis | ~10 min | **E** from **M** 591 s for 3,245 (`completion-clump.json`), × 3,334/3,245 |
| Splice: Rho, 292,721 new pairs of 5,556,111, and the kept-column read | 5–36 min | **E** from **M** 2,145 s for all 5,263,390 pairs (`completion-rho.json`). The new pairs are 5.6% of that work, about 2 min. Reading every kept column at the thinned rows is part of what the full run does, so the full run bounds the step from above |
| Splice: `completion_quality`, `analyses.tsv`, manifests | 2–5 min | **E** row edits and a recount (`completion-recount-top-hits.json`, 2.6 s, **M**) |
| Splice: Indexed Variant Subset `hm3`, rebuilt | 20–23 min | **E** from **M** 1,152 s and 1,372 s (`indexed-subset-hm3.json`, `completion-indexed-subset-hm3.json`) |
| Splice proof: answers | ~75 min | **E** from **M** rewrite-release on OGS-00011: source 2,104 s + staged 2,258 s (`rewrite-release.json` stdout), plus projecting |
| Splice proof: cells | ~1 h | **E** two banded reads of both grids, at the band-write rate above |
| `validate`, once | 38–41 min | **E** from **M** 2,308 s and 2,459 s (`completion-validate.json`, `validate.json`) |
| **Increment total** | **≈ 7–10 h** | **E**, the rows summed: 6.9–10.1 h. That is 4–6× cheaper than the rebuild's 40.7 h, and 4.6–6.8× cheaper than its 46.9 h of recorded steps |

The floor is whole-release passes, not the 89 Analyses. The Overflow remap, the tiers, the
subset, both proofs and `validate` take about 4.5–5 h of the splice's 5–6 h. A drop-only
splice (an erratum) at OGS-00011's size is about 5.4–6.4 h (**E**): no Increment, but up to
45–60 min of Dense planes when the dropped Analysis is early in the grid, and the same
whole-release passes. §8c's rehearsal measures it. OGS-00032 and OGS-00033 are an order of
magnitude smaller: 369 and 193 Analyses, with 278,775,501 and 327,434,261 Overflow
associations (**M**, their `records/complete.json`). Every **E** is to be replaced by a
measurement when #521 and #522 run. Benchmark numbers are re-run, never hand-edited.

### 11. Format 0.4.0 (#517) *(owner: ruling 1; the split: this ADR)*

**Survives, written format-agnostic:**

- the index maps (§2);
- column surgery on any plane, through its declared Store Encoding, never a hard-coded dtype;
- exception re-keying, CSR segment surgery, the tier and Clumps rebuilds, Rho and
  `completion_quality` surgery;
- the name merge, as a rule over ALIDs and the sidecars;
- `provenance.lineage` and the rule `manifest.lineage`;
- all three proofs.

Splice orders the axis only through `variants.axis.sorted_alids`, never a copy. #517 changes
that owner's tie order (A1, A2 → REF, ALT), and splice follows it without change.

**Reworked at 0.4.0:**

- everything `--increment-of` pins at build time (#522): ALID formation, orientation and
  `allele_not_reference` all go through #517's single owner in `opengwasdb.variants`;
- the pin table (§3), which gains the reference sequence digest ADR 0108 §7 puts in the
  manifest;
- splice's version gate.

A 0.3 release can never take a 0.4.0 Increment: the two sign their statistics for different
alleles, a silent sign error (#517). The gate refuses it by name. OGS-00011's increment is
superseded when the Store is rebuilt at 0.4.0 (ruling 1).

### 12. What the engine gives the registry *(this ADR, for stores#546)*

1. **`build-hybrid … --increment-of SOURCE`** (#522). Inputs: the added Analyses' manifest
   rows, SOURCE, and the `--source-record` SOURCE lacks. Output: an Observed-Only Increment
   and its Admission Record, beside it as for any build. Each of the 89 passes or is refused
   at resolution with its evidence in the registry's `ancestry.tsv`, which the candidate
   workflow writes (`opengwasdb-stores` `manifest_generation/candidate_workflow.py`), as for
   any build (ruling 6).
2. **`complete-hybrid INCREMENT DEST --ld-panel … --ancestry …`**, with SOURCE's panel and
   ancestry, and the impute set computed over SOURCE's and the Increment's rows (§3).
3. **`splice-release SOURCE DESTINATION [--add INCREMENT] [--drop …] [--source-record FILE]
   [--ld-store …] [--indexed-subset …] [--reason …]`**. Run it with `--plan` first. It prints
   one JSON object:
   - `source_release_id`, `release_id`, `increment_release_id`;
   - `added` and `dropped` ids, `n_analyses_before` and `n_analyses_after`;
   - `n_variants_before` and `n_variants_after`, and the off-panel variants added and removed;
   - every rsID and `source_alid` change, by variant, and the blanks §7 kept;
   - each added Analysis's `eaf` exception share;
   - the proof's question counts;
   - `elapsed_s` and the phase timings.
4. **`--source-record`**, which the registry writes from its step records for a SOURCE that
   predates recording them: the rsID policy, the source assembly, and the digests of the
   Variant Reference and of the LD panel (§3), with the indexed-subset lists and policies
   (§4). For OGS-00011 they come from `records/build.json`, `records/complete.json`,
   `records/completion-indexed-subset-hm3.json` and `validation.yaml`.
5. **Manifest fields** the registry can read: `release_id`, `created_at`,
   `provenance.lineage[]` (§9), and `analyses.tsv`. The registry checks ruling 6's
   GCST90132314 by name against `added` and `analyses.tsv`, and refuses publication when the
   live release is not `lineage[-1].source_release_id` (§9).

## Considered options

- **Tombstones**: mark a dropped Analysis absent and leave its cells. Rejected by ruling 3: a
  reader that ignores a tombstone answers silently, which is this project's worst failure.
- **Extending the builders to accept an existing release as input**: append inside
  `build-hybrid` and `complete-hybrid`. Rejected. Completion refuses a completed source
  (`layouts/hybrid/complete.py:166`), and its one subset hook, `impute_analysis_ids`, still
  reads and rewrites the whole grid (`dense/complete.py:296`). Each builder would also gain
  a second mode with its own proof. Ruling 4 asks for one owner and one proof.
- **A new Store id per increment.** Rejected by ruling 2: every `OGS-00011:<analysis_id>`
  must keep resolving.
- **Re-deriving the encoding plan and baselines for the union, and re-encoding when they
  change.** Rejected in §5. Every kept `eaf` cell would be quantised a second time, so kept
  answers would change, and each increment would cost at least the 1 h 42 m Dense `eaf`
  encode (**M**) plus the Overflow plane. The owner's D2 ruling accepts the bound instead.
- **Verifying an added Analysis's EAF orientation against the Increment's own consensus, or an
  external panel.** Rejected by D1. The Increment's own consensus is a weaker check, and a
  1-Analysis Increment cannot pass it at all. An external panel (`method: reference`) is a
  different method from the one OGS-00011 was built with.
- **Allowing the spliced variant axis to differ from a union build's.** Rejected: the spliced
  axis is the sorted union (§2), so allowing a difference would only hide a defect.
- **Carrying kept Clumps rows.** Rejected: a row's ambiguity depends on the whole axis
  (Context), and re-clumping every Analysis costs 591 s (**M**).
- **Rebuilding Rho whole on every splice.** Rejected. A Rho rebuild re-estimates every kept
  pair, which BLAS order can move (§8, difference 4), so kept answers would change.
- **Filling a blank name from the Increment, or merging names per routing key as ADR 0066
  does.** Rejected in §7. The release cannot tell an unnamed variant from a disputed one, and
  recovering routing keys means re-reading every source.
- **Pinning the LD panel and Variant Reference by matching ALIDs and EAF.** Rejected in §3:
  resemblance is not identity.
- **Keeping a replaced Analysis's `analysis_index`** (replace in place). It saves at most one
  whole-grid rewrite, under 1 h (**E**), but makes replace a third operation with its own
  proof. Rejected.
- **Keeping association-less off-panel variants after a drop.** Rejected in §6.
- **Observed-Only support.** Not built: no Store that needs a splice is Observed-Only.

## Consequences

- An increment or erratum costs hours, not two days. The floor is the whole-release passes
  (§10), so batching Analyses into fewer splices is cheaper than many small ones.
- **`analysis_index` and `variant_index` are release-scoped** (§2). An index cached across a
  splice does not fail today: it names another Analysis or variant. The ogquery release #521
  needs makes every index-taking entry point check the `release_id` the index came from, so
  that a stale one fails loudly. The sibling walkthrough in `opengwasdb-stores` should say
  this when it documents `splice-release`.
- The rsID conflict policy, the source assembly and the Variant Reference's digest become
  recorded build outputs (#522), and #327 records the LD panel's digest. A release built
  before them is spliced only with a `--source-record` from the registry.
- ogquery gains a projected form of its question set and the `release_id` guard (§2, §8), a
  cross-repository change that #521 needs before it can publish.
- A spliced release is 0.3.0 (§9), read and validated by every 0.3.0 package, which ignores
  `provenance.lineage`. `docs/store-format-compatibility.md` gains no row. Spec §21.1 now
  states the class of provenance block that moves nothing, so a later one is judged by that
  rule and is not a silent exception.
- The Increment's `eaf` is `float32`, which `--increment-of` must be able to force. Today
  the plan is measured (`build.py:2066` `_plan_joint_encoding`). A resumed build already
  loads a frozen plan instead (`build.py:2686` `_recorded_plan`), and that is the seam #522
  uses.
- **#522 depends on #327's `ld_panel_digest` reaching main.** #327 is on
  `orch/367-silent-errors`. Until it lands, the Increment's completion panel can be pinned
  only by what main records, and main records no panel identity: `ld_panel_id` is a constant
  label (`layouts/dense/complete.py:137`). So #522 refuses, by name, an Increment whose
  completion records no panel identity. It never falls back to `ld_panel_id`, or to
  matching ALIDs or panel EAF. SOURCE's panel identity can come from `--source-record`
  (§3). The Increment's cannot, because the Increment is the release being checked.
