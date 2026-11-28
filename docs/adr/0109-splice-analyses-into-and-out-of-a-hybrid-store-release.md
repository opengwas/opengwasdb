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
- **This ADR**, marked *(this ADR)*. Some of these challenge the orchestrator's defaults, which
  were not owner rulings; §5 and §8 say where and why.
- **Two decisions still need the owner** (*Decisions needed*, D1 and D2). Each would cost a
  full recompute to avoid, and the stop rule for this ticket asks for them to be reported.
  This ADR is written on the recommended option of each.

What this ADR changes in other documents:

- Spec §2b (new): `provenance.lineage`, the record a spliced release carries of the releases
  it derives from. It moves no `format_version`: it is a provenance block (spec §21.0,
  ADR 0038 §1).
- CONTEXT.md: two new terms, **Splice** and **Increment**.
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

Its first increment adds 89 Analyses, and replaces one more whose scale flips from
quantitative to case-control (ruling 6, stores#557). That is 90 added and 1 dropped. Removal
is also needed for defects: on 26 Nov 2026, 11 Analyses with mis-scaled z were found in the
published Hybrids OGS-00032 and OGS-00033 (stores#520, #532).

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
  Analysis's imputed cells depend on another's.
- **The shared axis is sorted, and an off-panel variant is on it only because it is
  observed.** The shared axis is `sorted_alids(panel ∪ off_panel)`, where off_panel = observed −
  panel (`layouts/hybrid/build.py:770` `_partition_variants`; `variants/axis.py:61`). A new
  off-panel variant therefore interleaves into the axis and renumbers every row after it.
  The build already extends and renumbers the axis this way once: `_build_shared_key_table`
  (`build.py:1608`) returns an `old_to_new` map from `AlidIndex.lookup`.
- **Panel crossovers are folded per association.** A completed Hybrid moves each Overflow
  association whose variant the LD panel added onto the Dense Component, as an observed cell
  (`layouts/hybrid/complete.py:803` `_fold_panel_crossovers`). On OGS-00011 that was
  995,632,726 associations (measured, `complete.json` stderr).
- **The Rho Matrix's thinned rows do not depend on the Analyses.** `select_thinned_variants`
  (`layouts/dense/rho.py:63`) takes the first Dense row in each `window_bp` window by
  position. Pair *(i, j)* depends only on Analyses *i* and *j* at those rows (`rho.py:236`).
  The packed strict lower triangle is row-major (`rho.py:455`), so Analyses appended at the
  end append rows to it.
- **Clumps are per Analysis.** An Analysis is clumped against its Assigned Ancestry's LD
  Store (`clumps.py:553` `_clump_analysis`). `lead` is an offset into that Analysis's own
  clump slice, and ties break by Store Variant Index (spec §18.1). A monotone renumbering
  changes neither.
- **Exception tables are keyed by flat position.** A Dense table is keyed by
  `variant × n_analyses + analysis` and a CSR table by ordinal (spec §6a). Changing
  `n_analyses` re-keys every entry. `indexed_subsets.py:560` `_remap_side_table` already
  re-keys a Dense table by `index % n_analyses`.
- **The EAF Baseline is the release's.** It is a per-variant median, taken over the release's
  Analyses (`encoding/codec.py:356`). Completion carries the source's baseline rather than
  recomputing it, "so a cell decoded from the source and re-encoded here lands on the same
  code" (`layouts/dense/complete.py:1579` `_carried_eaf_baseline`; the Overflow equivalent is
  `layouts/hybrid/complete.py:1047`).
- **EAF orientation is checked against a consensus of the other Analyses.** Each Analysis's
  EAF is correlated against the leave-one-out median of every other Analysis
  (`build/eaf_orientation.py:734` `_consensus_baselines`). OGS-00011 records `method:
  consensus`, `reference_id: consensus:3245-analyses`. This is the one admission-time check
  that couples Analyses (D1).
- **The MAF and INFO floors are per Analysis.** OGS-00011 records a MAF threshold of 0.005
  for 3,241 Analyses and 0.04, 0.025, 0.075 and 0.1 for one Analysis each. It records an INFO
  threshold of 0.6 for 628 Analyses, and none (`unavailable`) for 2,617 (measured,
  `provenance.maf` and `provenance.info_score`). Each comes from the Analysis's own manifest
  row (`model/maf_policy.py`). They are not one axis-wide setting.
- **rewrite-release already stages, stamps and proves.** `store/rewrite.py:782`: a reflinked
  copy (`:982`), named steps, a fresh `release_id` on every manifest (`:1017`), an answer-digest
  proof (`:1089`) and a full `validate`, all inside `OpenGWASDBStore.staging`
  (`store/open.py:748`, ADR 0043). It refuses any source that is not 0.2.3 (`:937`), and its
  proof requires every answer unchanged.
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
    [--on-rsid-conflict refuse|drop]  required with --add; the policy SOURCE was built under (§7)
    [--ld-store ANCESTRY=PATH …]      required with --add when SOURCE declares Clumps (§4)
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
    on_rsid_conflict: RsidConflictPolicy | None = None,
    ld_stores: Mapping[str, str | Path] | None = None,
    reason: str | None = None,
    n_workers: int = 1,
) -> SpliceRun: ...

def plan_splice(source, *, add=None, drop=()) -> SplicePlan: ...  # refusals, maps, counts
```

- **A replace is a drop and an add in one invocation.** An `--add` Analysis whose
  `analysis_id` SOURCE already holds is refused unless `--drop` names it too. Ruling 6's
  replace is therefore one splice, one new release and one lineage entry.
- **What it reuses from `rewrite-release`:**
  - `refuse_destination` and `OpenGWASDBStore.staging(destination)`, so the source is never
    written and the commit takes the parent-directory lock (ADR 0043);
  - `_copy_release` with `cp -a --reflink=auto` (`store/convert.py:1154`), for every file
    splice does not rewrite (§4);
  - the `_stamp` pattern: one fresh `release_id` and `created_at` on the root manifest and on
    the nested Dense Component's manifest;
  - `_refresh_overviews`;
  - `require_valid_staged_release`, a full `validate`.

  It does not reuse rewrite's proof, which requires every answer unchanged. It has its own
  (§8). It does not reuse `_refuse_version` either: splice accepts exactly the
  `format_version` it writes, today 0.3.0 (§11).
- **Never in place.** The previous release stays where it is. Moving it to
  `replaced/<UTC>/` and putting the new one at the published path is the registry's job
  (ruling 2, stores#546).
- **Scope.** A Hybrid release, Observed-Only or Reference-Completed, but both releases in one
  splice must have the same Completion State. The first implementation is the
  Reference-Completed case, which is what OGS-00011, OGS-00032 and OGS-00033 are. A Dense or
  Ragged release is refused by name: the same surgery applies to them, but nothing asks for
  it yet.

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
  stays strictly increasing. They are computed as integer arrays from the sorted byte keys
  (`AlidIndex.lookup`, as in `build.py:1608`). They are not built through an ALID → row dict.
  `complete.py:360` `_remap_shared_axis` builds per-association Python lists, and its phase
  took 8 h 29 m on OGS-00011 (05:17:40 → 13:47:22, `complete.json` stderr, measured).
- **The Dense row map is the identity.** The Dense axis is fixed (Context, first bullet), and
  §3 refuses any Increment whose Dense axis differs.

**`analysis_index` is per release, exactly as ADR 0002 makes `variant_index`.** The stable key
of an Analysis is `(store_id, analysis_id)`. ogquery raises `UnknownAnalysisError` on an
`analysis_id` (`ogquery/analyses.py:20`), and the rule `analyses.analysis_index` requires
indices to cover `0..n-1` (spec §20.0). A consumer that kept an `analysis_index` across two
releases was already outside the contract. This ADR says so.

### 3. The Increment, and what it is pinned to *(owner: ruling 4; the table: this ADR)*

An **Increment** is a small Hybrid release that holds only the Analyses to add. It is built
with `build-hybrid … --increment-of SOURCE` (#522) and completed with the usual
`complete-hybrid` against the release's LD panel and ancestry. Splice checks the result
against SOURCE and refuses every mismatch by name before staging anything.

| Pinned | Where the release records it today | What refuses a mismatch |
|---|---|---|
| Variant Reference: the Dense axis before completion | `provenance.variant_reference`, a path with no digest | the completed Dense axes are compared ALID for ALID, through `dense_to_shared` and the shared ALID sidecar; the path is advisory |
| LD panel and completion method | Dense manifest `provenance.completion`: `ld_panel_id` (`eur-hg38-gpm`), `ancestry`, `method`, `min_cor`, `pca_thresh` | equality of each, plus the same Dense axis (which includes the 845,652 panel-added rows) and a bitwise-equal `eaf_reference` and `on_panel` |
| Store Encoding: `z` kind and scale, `se` kind | `manifest.encoding` | equality. The Increment must carry `eaf` exactly (`float32`, or `absent` when no added Analysis has EAF), so splice quantises each new cell once, against the release's baseline (§5) |
| Reference assembly, chain file, Association Coverage | `reference_assembly`, `provenance.chain_file`, `association_coverage` | equality |
| `format_version` | `format_version` | both equal to the version splice writes |
| rsID conflict policy (ADR 0066) | **not recorded in the release**: only in the build record's argv (`--on-rsid-conflict`) and the registry's `validation.yaml` (`rsid_conflicts.policy: drop` for OGS-00011) | an explicit `--on-rsid-conflict` with no default. The Increment's build records its own policy in provenance (#522), and the two must agree |
| EAF orientation reference | `provenance.eaf_orientation` (`method`, `reference_id`, `min_overlap`, `min_variance`, `allow_unverified`) | equality of the settings; the reference itself is D1 |
| Clumping LD Stores | the Clumps group's `clumps` attribute: `ld_stores`, each identified by what a reader can recompute (ADR 0106 §4) | each `--ld-store` must match the identity recorded for its ancestry |
| Rho settings, Indexed Variant Subset lists | `rho` attributes (`window_bp`, `z_thresh`, `min_nulls`); each subset's `input_sha256` | not inputs: splice reads them from SOURCE and reuses them |
| MAF and INFO floors | per Analysis, `provenance.maf.analyses[]` and `provenance.info_score.analyses[]` | not pinned. They are per Analysis (Context), each added Analysis's comes from its own manifest row, and a union build would apply the same one. Splice carries the entries |

Splice also refuses:

- an `--add` id already held and not also dropped, or a duplicate id;
- a `--drop` id that SOURCE does not hold;
- a splice that would leave no Analysis;
- a release whose `se` is `int8_residual`. Its codes are residuals against each cell's EAF and
  the Analysis's `se_coefficients`, and an added Analysis without EAF cannot be coded into the
  plane (ADR 0037 §3). No published Hybrid has such a plane: OGS-00004, OGS-00005, OGS-00011,
  OGS-00032 and OGS-00033 are all `float16` (measured, their manifests, 28 Nov 2026);
- a release whose `eaf` is `absent` when the Increment carries EAF, which would otherwise be
  dropped silently.

### 4. What splice rewrites, structure by structure *(this ADR)*

`a` is old `analysis_index`, `A(a)` is `analysis_map`, `S(v)` is `shared_map`, and `n′` is the
new Analysis count. "Carried" means the kept bytes are copied without decoding.

| Structure | How splice writes it |
|---|---|
| Dense planes `z`, `se`, `eaf`, `imputed` (spec §10) | Column surgery: kept columns are carried, code for code, to column `A(a)`, and the Increment's columns are appended. Shards [100000, 1024] (spec §10a): a shard file is rewritten only when one of its columns changes. An add-only splice rewrites only the last shard column. A drop rewrites every shard column from the first dropped index onward, and the whole plane at worst. Shard files splice does not touch are reflinked |
| Dense exception tables `z_overflow_*`, `eaf_exception_*` (spec §6a) | A kept key `v·n + a` becomes `v·n′ + A(a)`. A dropped column's entries are removed. The new columns' entries are merged in. Since `A` is monotone, the kept order holds and the merge is one streaming pass |
| `eaf_baseline`, `eaf_reference`, `on_panel` | Carried unchanged (§5). The two panel arrays are already proven equal (§3) |
| `se_coefficients[n, 2]` | Rows filtered and appended. Absent under `float16` |
| Overflow CSR `offsets`, `variant_index`, `z`, `se`, `eaf` (spec §11) | Segment surgery: kept segments in `A` order, with `variant_index` → `S(v)`; then the Increment's segments with `increment_shared_map`. Exception tables re-keyed by the new ordinals |
| `ragged/by_variant/` (spec §11.1) | Rebuilt from the new CSR by its owner, `layouts/ragged/by_variant.py:480` `build_variant_index`. It is derived and variant-ordered, and a rebuild is simpler than a merge (13 m 29 s on OGS-00011's completion, measured, stderr 16:03:57 → 16:17:26) |
| Shared axis: `variants.tsv.gz` and `.tbi`, ALID, position and `source_alid` sidecars, rsID index (ADR 0105) | Rewritten for the new axis by `variants/axis.py:118` `write_variant_axis`. Names merged by §7 |
| `dense_to_shared.npy` | Values remapped through `S`. Still strictly ascending, because `S` is monotone |
| Top-Hit Index tiers, both components (spec §18) | Rebuilt from the spliced planes by their owners: `layouts/dense/top_hits.py:307` `build_top_hit_indexes` and `layouts/ragged/top_hits.py:155` `build_ragged_top_hit_indexes`. A tier is a function of each Analysis's own cells, so every kept Analysis's slice comes out identical under `A` and `S`, and an added Analysis's tier `eaf` is the value the spliced plane decodes. Cost: a 3 m 46 s scan and a 1 m 56 s write at 24 workers (Dense), and a 7 m 26 s scan (Overflow), measured in OGS-00011's completion stderr. The single-worker Dense rescan after the fold took 1 h 16 m, so splice runs it with workers |
| Clumps (spec §18.1) | The tier writer replaces the whole tier group, and the Clumps with it (spec §18.1), so splice writes them again afterwards. `lead` and `r` are parallel to tier rows and the kept slices are identical, so each kept Analysis's rows are copied from SOURCE. Added Analyses are clumped by `clumps.compute_clumps` (`clumps.py:522`) against the `--ld-store` mapping. The `clumps` manifest entry is written last, as ADR 0106 §10 requires |
| Rho Matrix (`dense/data.zarr/rho`) | Kept pairs carried: the triangle is re-packed to the kept rows and columns. Pairs involving an added Analysis are estimated by `rho.py`'s estimator at the release's thinned rows and appended. `overflow_only` is remapped through `A`, plus the added Analyses that have no Dense observation. Not a full rebuild, so no kept pair is re-estimated (§8) |
| Indexed Variant Subsets (spec §10b) | Rebuilt by `build_indexed_subset` (`indexed_subsets.py:1290`) from the spliced grid, with the release's variant list. A subset plane is a re-keyed copy of the grid's codes (`:560`), so kept rows come out identical |
| `completion_quality` (`dense/index.sqlite`) | Kept rows renumbered through `A`, the Increment's rows inserted with their new index. PK `(analysis_index, block_id)` (`completion/schema.py:27`) |
| `analyses.tsv` (spec §7a) | Kept rows renumbered, the Increment's rows appended. `n_hits_*` recounted from the spliced tiers by the recount owner (`store/top_hit_counts.py:90`), so ADR 0107's rule holds by construction. `n_loci_5e8` and `n_unclumped_5e8` come from the Clumps |
| Manifests (root and `dense/`) | `n_analyses`, `n_variants`, the `hybrid` counts, `ragged.by_variant` and completion totals recomputed. Per-Analysis provenance lists (`maf.analyses`, `info_score.analyses`, `eaf_orientation.analyses`, `completion.se_out_of_range.imputed`) are filtered and appended by `A`. `lineage` is appended (§9) |
| `overview.html` | Regenerated (`overview.write_overview_html`) |

### 5. The encoding plan is the release's, and is never re-derived by a splice *(this ADR; challenges the orchestrator's default)*

The orchestrator's default was to re-derive the plan (ADR 0037) over the union, and to
re-encode the whole plane if it changed. This ADR does the opposite: **splice codes every new
cell under the release's plan and against its EAF Baselines, and never re-derives either.**

- **Every plan splice accepts can hold any value.**
  - A `z` beyond `int16_fixed`'s range, and an `eaf` residual beyond the range, become exact
    exceptions.
  - A new cell at a variant with no baseline is an exact exception. This is how completion
    already treats a panel-only row (`dense/complete.py:1579`).
  - An `se` above 65,504 is stored as missing and counted, under every plan (ADR 0068).

  The plans that cannot hold a new Analysis are refused in §3. No value is ever clipped.
- **Re-deriving would rewrite the kept Analyses.** A new baseline re-points every carried
  `eaf` code. The release keeps no source floats, so each kept cell would be decoded and
  quantised a second time, and the kept answers would change: the drop proof (§8) could not
  hold. It would also cost a full `eaf` re-encode of both components (`Encode eaf: done in
  1h42m` for the Dense plane at OGS-00011's build, measured, `build.json` stderr).
- **Completion already made this decision** for the same reason: it carries the baseline and
  never refits `se_coefficients` (ADR 0037 §3, ADR 0078).
- **What it costs.** A new `eaf` cell sits within half a residual step of its exact value, as
  in any build. But it is coded against a baseline its own Analysis did not help form, so it
  can differ from a union build's code by up to one step: `range/127` in logit, 0.0079 at
  OGS-00011's ±1.0. Splice records each added Analysis's exception share, and reports it
  against ADR 0037's 2% budget without enforcing it, as completion does (ADR 0078). Whether
  this counts as equivalent is D2.

### 6. Variants left with no association after a drop are removed *(this ADR)*

An off-panel variant observed only by dropped Analyses leaves the shared axis. The axis is
rebuilt as `dense_axis ∪ observed_off_panel` (§2), which is the build's own invariant
(`build.py:770`: "Nothing observed is dropped, so an off-panel variant is always on the shared
root axis"; the converse holds too, since nothing unobserved is added). Dense rows never
leave: the Dense axis is the panel's.

Consequences:

- A drop renumbers the shared axis, so every sidecar and `dense_to_shared` is rewritten.
  That machinery is the add's, run in reverse.
- After a drop, the axis equals that of a rebuild of the kept Analyses.
- A defect removal leaves no association-less variant behind for a reader to find with an
  empty PheWAS. Keeping such variants would pass `validate`: no rule requires an off-panel
  row to have an association. It was rejected because the release would then claim variants
  that no Analysis in it observed.

### 7. Variant names at the merge *(this ADR)*

When a variant is in both the release and the Increment, splice merges their names by the
union rules. It never uses the Increment's names in place of the release's.

- **rsID.** If the two agree, the rsID is kept. If they disagree, `--on-rsid-conflict`
  decides: under `drop`, the variant keeps no rsID; under `refuse`, splice refuses
  (ADR 0066).
- **`source_alid`.** If the two disagree, the row becomes "none: two source variants collided
  onto one row", the sidecar's existing state (ADR 0105 §8).

So an add can change a kept variant's name answers, exactly where a union build would. Splice
counts and lists every such variant in its output (§12).

A drop cannot undo a collision state. The release does not keep per-source names, so a name
that a now-dropped Analysis's source made ambiguous stays absent. A rebuild of the kept
Analyses would restore it. Restoring it would mean re-reading every kept source (Pass 1 took
7 h 00 m on OGS-00011, measured). The difference loses a name. It never makes one wrong.

### 8. Equivalence: what is proven, where, and the differences permitted *(this ADR; refines the orchestrator's default)*

Two proofs. Neither is a second full OGS-00011 build.

**(a) The equivalence test, on a small Trial Store, in the test suite and gating #522's
merge.** Build B, complete it and run the post steps: that is `R_B`. Build an Increment for I
and complete it. Splice it into `R_B`: that is `R_splice`. Separately, build and complete
B ∪ I from scratch: that is `R_union`. Compare every structure exhaustively, keyed by
`analysis_id` and ALID. The two must be **identical** except for these permitted differences:

1. **`analysis_index` order.** The union orders by manifest row, splice appends.
2. **`eaf` codes, and therefore exception-table membership.** Decoded values must agree
   within one residual step (§5, D2). The same applies to `eaf` in tiers and Indexed Variant
   Subsets.
3. **The encoding plan,** when the union would have chosen another. The fixtures must
   include one such case, and splice must keep `R_B`'s plan.
4. **Rho values of kept pairs, within one `float16` ulp.** `n_null` must be exact. The
   difference comes from float64 summation order in the pair sums (`rho.py:236`, BLAS), which
   differs between products of different widths. BLAS order has already moved another
   rebuilt store's floats by 8 ulps on CI's runners (#512).
5. **`provenance.eaf_orientation` r values of B's Analyses.** In a union build their
   consensus includes I. This is provenance only: no stored value depends on it unless
   verification fails, and then the build fails.
6. **Name merges, if the fixtures contain any (§7).** The test lists each one and checks that
   the union agrees.
7. **`release_id`, `created_at` and `lineage`.**

**There is no variant-axis difference.** The orchestrator's default allowed one, but the
spliced shared axis is the sorted union (§2), the same axis a union build forms, so every
`variant_index` agrees. The fixtures must include an off-panel variant that only I observes
and that sorts between two of B's, a panel crossover in I, a column move past a shard
boundary, a `z` beyond `int16_fixed`'s range, and an Overflow-only Analysis.

A drop is tested the same way: `splice(R, drop D)` against the release built from the kept
Analyses. The permitted differences are 2, 4, 5 and the collision states of §7.

**(b) The splice proof, run by every splice before it publishes,** in the place
`_require_same_answers` holds in `rewrite_release`. It has three parts:

1. **The cell proof, exhaustive.**
   - Every kept Analysis decodes to the same values in every plane, cell for cell, under
     `A` and the identity Dense map. The same holds for its Overflow segment under `S`.
   - Every added Analysis decodes to the Increment's values: `z`, `se` and `imputed`
     exactly, `eaf` within half the release's step of the Increment's exact `float32`.

   It is a banded read of both grids and both CSRs.
2. **The answer proof.** The question set of `ogquery.digests` is posed by `analysis_id` and
   ALID, taken from SOURCE's own spread. Each answer is normalised before it is digested:
   `analysis_index` → `analysis_id`, `variant_index` → ALID. Every kept answer must equal
   SOURCE's. Every question about a dropped id must raise `UnknownAnalysisError`. Every
   question about an added id must equal the Increment's, with `eaf` within the step.
3. **`validate`, in full.**

Any difference discards the staging directory and leaves SOURCE untouched.

**ogquery's digests cannot simply be made mapping-aware, but its question set can.** A digest
of an array that holds indices cannot be translated after the fact. The questions can be posed
by key and the answers normalised before digesting. `digests.py` is ogquery's single owner of
the question set (its docstring), so this belongs there: `release_answers` gains the reference
release's identifiers and a normaliser. That is an ogquery minor release, tracked under #521.

### 9. Lineage and records *(owner: ruling 2; the field: this ADR)*

Every splice appends one entry to `provenance.lineage` (spec §2b) in both manifests. The
first splice of a release starts the list. An entry records:

- `source_release_id`, and the `increment_release_id` (or `null` for a drop-only splice);
- `added` and `dropped`, as `analysis_id` lists;
- `n_analyses_before` and `n_analyses_after`;
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

**The Validation Record and `build_elapsed_s` are the registry's** (stores ADR 0004, amended
by stores#546). The engine's position, for that ADR:

- An increment release's `build_elapsed_s` should be the sum of its own steps: the Increment's
  build and completion, the splice, and `validate`. It should not be cumulative. The lineage
  links to the previous release's records, so the cumulative cost can be derived and is never
  double-counted.
- `validate` runs on the spliced release, as on any release.
- The observed-view check runs on the Increment's observed → completed pair only. The splice
  proof's cell proof already covers the kept Analyses exhaustively, where the check samples.

### 10. Cost model: OGS-00011's first increment *(this ADR)*

90 added, 1 dropped. **M** is measured, with its record. **E** is estimated, with its basis.

| Step | Time | Basis |
|---|---|---|
| **Rebuild today**: build + completion | 40.7 h | **M** `build.json` 71,851 s + `complete.json` 74,651 s |
| Rebuild, every recorded step | 46.9 h | **M** sum of `records/*.json` (Context) |
| Increment build, 90 Analyses | 1–1.5 h | **E** from **M** small builds that run the same phases: OGS-00032, 369 Analyses, 4,696 s; OGS-00033, 193 Analyses, 6,342 s (`records/build.json`). Fixed costs dominate: Pass 1 merge 16 m 52 s and 17 m 14 s |
| Increment completion | 1–2 h | **E** from **M** OGS-00032, 5,846 s, and OGS-00033, 7,170 s, whose LD-block phase was flat in Analyses (51 m 12 s for 369, 52 m 52 s for 193). OGS-00011's own LD-block phase scales to 14 min for 90 (8 h 30 m × 90/3,245), and its 8 h 29 m Overflow fold to about 15 min |
| Splice: Dense planes | ≤ 0.75–1 h | **E** worst case, the dropped Analysis early in the grid, every shard column rewritten. Basis: completion's whole-grid `Band write: done in 43m30s` (**M**, `complete.json`), which writes every plane from decoded values; splice moves codes |
| Splice: exception re-key, Overflow segment surgery and axis remap, shared sidecars, `by_variant` | 1–1.5 h | **E** from **M** completion's shared-table and Overflow phase (56 m 36 s, stderr 15:07:21 → 16:03:57) plus `by_variant` (13 m 29 s), at the same 1.71e9 Overflow associations. Requires §2's integer maps, not `_remap_shared_axis` |
| Splice: tiers rebuilt | ~15 min | **M** OGS-00011's completion: Dense scan and write 5 m 42 s at 24 workers, Overflow scan 7 m 26 s |
| Splice: Clumps for 90, Rho for 295,965 new pairs of 5,556,111, `completion_quality`, `analyses.tsv` | ≤ 15 min | **E** Rho 2,145 s for all 5,263,390 pairs (**M**, `completion-rho.json`) × 5.6%; Clumps 591 s for 3,245 (**M**) |
| Splice: Indexed Variant Subset `hm3`, rebuilt | 20–23 min | **M** 1,152 s and 1,372 s (`indexed-subset-hm3.json`, `completion-indexed-subset-hm3.json`) |
| Splice proof: answers | ~75 min | **M** rewrite-release on OGS-00011: source 2,104 s + staged 2,258 s (`rewrite-release.json` stdout), plus normalising |
| Splice proof: cells | ~1 h | **E** two banded reads of both grids, at the band-write rate above |
| `validate` | 38 min | **M** `completion-validate.json` 2,308 s |
| **Increment total** | **≈ 7–10 h** | **E**: about 4–6× cheaper than the rebuild's 40.7 h |

The floor is whole-release passes, not the 90 Analyses: the two proofs, `validate` and the
Overflow axis remap take about 4 h of a splice's 5–6 h. A drop-only splice (an erratum) at
OGS-00011's size is about 5 h (**E**): no Increment, but the same whole-release passes.
OGS-00032 and OGS-00033 are an order of magnitude smaller: 369 and 193 Analyses, with
278,775,501 and 327,434,261 Overflow associations (**M**, their `records/complete.json`). Every **E** is to be
replaced by a measurement when #521 and #522 run, and benchmark numbers are re-run, never
hand-edited.

### 11. Format 0.4.0 (#517) *(owner: ruling 1; the split: this ADR)*

**Survives, written format-agnostic:**

- the index maps (§2);
- column surgery on any plane, through its declared Store Encoding, never a hard-coded dtype;
- exception re-keying, CSR segment surgery, tier, Clumps, Rho and `completion_quality`
  surgery;
- the name merge, as a rule over ALIDs and the sidecars;
- `provenance.lineage` and the rule `manifest.lineage`;
- both proofs.

Splice orders the axis only through `variants.axis.sorted_alids`, never a copy. #517 changes
that owner's tie order (A1, A2 → REF, ALT), and splice follows it without change.

**Reworked at 0.4.0:**

- everything `--increment-of` pins at build time (#522): ALID formation, orientation and
  `allele_not_reference` all go through #517's single owner in `opengwasdb.variants`;
- the pin table (§3), which gains the reference sequence digest ADR 0108 §7 puts in the
  manifest;
- splice's version gate.

A 0.3.0 release can never take a 0.4.0 Increment: the two sign their statistics for different
alleles, a silent sign error (#517). The gate refuses it by name. OGS-00011's increment is
superseded when the Store is rebuilt at 0.4.0 (ruling 1).

### 12. What the engine gives the registry *(this ADR, for stores#546)*

1. **`build-hybrid … --increment-of SOURCE`** (#522). Inputs: the added Analyses' manifest
   rows and SOURCE. Output: an Observed-Only Increment and its Admission Record, beside it as
   for any build.
2. **`complete-hybrid INCREMENT DEST --ld-panel … --ancestry …`**, unchanged, with SOURCE's
   panel and ancestry.
3. **`splice-release SOURCE DESTINATION [--add INCREMENT] [--drop …] --on-rsid-conflict …
   [--ld-store …] [--reason …]`**. Run it with `--plan` first. It prints one JSON object:
   - `source_release_id`, `release_id`, `increment_release_id`;
   - `added` and `dropped` ids, `n_analyses_before` and `n_analyses_after`;
   - `n_variants_before` and `n_variants_after`, and the off-panel variants added and removed;
   - the rsID and `source_alid` merges, by variant;
   - each added Analysis's `eaf` exception share;
   - the proof's question counts;
   - `elapsed_s` and the phase timings.
4. **Manifest fields** the registry can read: `release_id`, `created_at`,
   `provenance.lineage[]` (§9), and `analyses.tsv`. The registry checks ruling 6's
   GCST90132314 by name against `added` and `analyses.tsv`.

## Decisions needed

**D1. Which reference verifies an added Analysis's EAF orientation.** A union build checks
each Analysis against the leave-one-out median of all the others
(`build/eaf_orientation.py:734`). An Increment built alone has only the other added Analyses.

- **A (recommended): the release's consensus.** Verify against SOURCE's EAF Baselines, the
  per-variant median of every SOURCE Analysis (`encoding/codec.py:356`). The added Analysis
  is not in that median, so it is the same leave-it-out comparison a union build makes. It
  differs only in taking the median in logit space, and in its sites: wherever any SOURCE
  Analysis reports a frequency, not only where two others do. Recorded as
  `reference_id: release:<release_id>`. Cost: reading the baselines, minutes.
- **B: the Increment's own consensus.** Only the added Analyses, so it is a weaker check. A
  1-Analysis Increment cannot pass at all, and the check is a different gate from the union's.
- **C: an external panel reference** (`method: reference`). Allowed by the code, but a
  different method from the one OGS-00011 was built with.

**D2. Whether `eaf` within one residual step counts as equivalent to a union build (§5, §8
difference 2).** Bitwise agreement is impossible without a full recompute. The baseline is a
median over every Analysis, so adding one moves it at every variant the Analysis reports.

- **A (recommended): accept within one step,** 0.0079 logit at ±1.0. Cost: none. Every kept
  answer stays exact, and this is the rule completion already follows.
- **B: re-derive the baselines over the union and re-encode both `eaf` planes on every
  splice.** Cost: at least the 1 h 42 m Dense encode (**M**) plus the Overflow plane, every
  increment. Every kept Analysis's `eaf` is quantised a second time and its answers change,
  so the drop proof (§8 b1) cannot hold for an add-and-drop splice.
- **C: rebuild.**

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
- **Re-deriving the encoding plan for the union** (the orchestrator's default). Rejected in §5.
- **Rebuilding Rho and the Clumps whole on every splice.** Rejected. A Rho rebuild
  re-estimates every kept pair, which BLAS order can move (§8, difference 4), so the drop
  proof could not require kept answers unchanged. A whole re-clump (591 s, **M**) would
  reproduce kept rows, but it needs every ancestry's LD Store, not only the added Analyses'.
  The tiers and the Indexed Variant Subset are rebuilt, because each is a deterministic
  function of its Analysis's cells.
- **Keeping a replaced Analysis's `analysis_index`** (replace in place). It saves at most one
  whole-grid rewrite, under 1 h (**E**), but makes replace a third operation with its own
  proof. Rejected.
- **Keeping association-less off-panel variants after a drop.** Rejected in §6.

## Consequences

- An increment or erratum costs hours, not two days. The floor is the whole-release passes
  (§10), so batching Analyses into fewer splices is cheaper than many small ones.
- **`analysis_index` is per release** (§2). A downstream that cached indices across releases
  breaks loudly, on `UnknownAnalysisError` or on the rule `analyses.analysis_index`. The
  sibling walkthrough in `opengwasdb-stores` should say this when it documents `splice-release`.
- The rsID conflict policy becomes a splice input, because the release never recorded it.
  #522 records it in the Increment's provenance, and a later build change should record it in
  every release's.
- ogquery gains a mapping-aware form of its question set (§8), a cross-repository change
  that #521 needs before it can publish.
- No `format_version` moves and `docs/store-format-compatibility.md` is unchanged. A spliced
  0.3.0 release is read by every 0.3.0 reader, and `provenance.lineage` is ignored by those
  that predate it.
- The Increment's `eaf` is `float32`, which `--increment-of` must be able to force. Today
  the plan is measured (`build.py:2066` `_plan_joint_encoding`). A resumed build already
  loads a frozen plan instead (`build.py:2686` `_recorded_plan`), and that is the seam #522
  uses.
