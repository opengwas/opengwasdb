# An SE beyond float16's range is not stored

Issue [#338](https://github.com/opengwas/opengwasdb/issues/338), under epic #347
("Builds never store a statistic that contradicts its source"). It rests on
[ADR 0013](0013-association-status-nan-plus-imputed-mask.md) (missing is a marker in
both planes), [ADR 0037](0037-statistic-array-encodings.md) (the encodings),
[ADR 0038](0038-store-format-versioning-and-migration-policy.md) (completion
writes into its source's encoding), [ADR 0077](0077-one-module-owns-how-an-association-is-encoded.md)
(one gate, `storable`, decides what a writer holds) and [ADR 0078](0078-completion-keeps-its-sources-se-model.md)
(completion keeps its source's SE model). Format `0.2.3` (spec §21). Amends ADR 0057's
statement of the current version and ADR 0077's "Where issue #338's rule goes".

The range is judged on the SE as stored, after SD standardisation
([ADR 0084](0084-storable-standardises-every-builders-se.md)): a source SE of 40,000
under an SD of 0.5 is stored as 80,000 and left out, and one of 100,000 under an SD
of 2.0 is stored as 50,000 and kept.

## Context

A source can report an SE above 65,504, `float16`'s largest finite value. The
builders narrowed it to `float16` and wrote `+inf` beside a finite z (an SE below
65,520 became 65,504, a finite wrong value). `beta = z × se` then reads as
infinity although the source value was finite. The published OGS-00011 holds 8
such cells: GCST012879 1, GCST90295971 5, GCST90827693 2. #337 made `validate`
say so and made a reader and the Observed-View Check treat such a cell as a
value (ADR 0070).

The owner's decision (1 Nov 2026): an SE that `float16` cannot represent is almost
certainly an error, so a build treats that association as **missing**, counted per
Analysis, at every point where it narrows an SE. A first implementation
(`orch/347/338`, 52edbf1) masked at each narrowing site; the owner then had #350
put the judgement in one place first (ADR 0077), and this rebuild changes that one
place. The review of 52edbf1 found seven defects, each fixed here (below).

## Decisions

1. **The bound is `np.finfo(np.float16).max`, 65,504; an SE strictly above it --
   `+inf` included -- is not stored, whatever the plan.** An `int8_residual` plane
   could hold it as an exact exception, but whether an association is kept would
   then depend on the plan the build measured. 65,504 itself is kept.
2. **One gate.** `storable` (ADR 0077) leaves such an association out: its z, SE
   and frequency become NaN, its imputed flag 0, and `stored` is False there; the
   batch it was handed is not edited. It counts them per Analysis
   (`se_out_of_range`) and, apart, those flagged imputed (`imputed_out_of_range`).
   Every writer already holds what `storable` returns (#350), so a grid holds a
   missing cell and a writer of rows (a CSR, a spill) holds no row. The Dense
   build judges its grid once, before the plan is measured, so the plan is fitted
   to the cells it holds; every other writer judges at the spill or the add, which
   precede measurement already.
3. **The encoders refuse.** `StoreCodec.encode_se` (both plans) and the streamed
   `float16` narrowing of a Dense scratch plane, the one SE write that does not go
   through `encode_se`, raise `SeOutOfRange`, naming the count and the first. A
   path that forgot `storable` fails loudly instead of writing `+inf`. Under
   `int8_residual` a kept cell whose code would decode above 65,504 is an exact
   exception. A code is the nearest step, so 65,504 itself decoded to 65,520 under
   a 0.5 range with a model predicting 65,520, and a BESD build of two 65,504 SEs
   stored 65,547 and 65,560. A reader also predicts from the `float32` frequency,
   and under a steep model that moves a prediction by whole units, so an SE of
   10,000 read as 11,806,119 under a slope of 1e9. The encoder therefore decodes
   every kept cell as a reader will, not only those near the bound. This is part of
   the one exception decision, `se_residual_codes`, which the candidate measurement
   and the Dense rewrite's count make too, so the plan is chosen with these
   exceptions charged against §6a's 2% cap, and the count that sizes the side table
   agrees with the rewrite. Every decoded SE is at most 65,504 under either plan.
4. **The count is a store-only `analyses.tsv` column, `n_missing_se_out_of_range`**:
   a non-negative integer, required on every row from 0.2.3 and optional below it,
   where absent or blank means **unknown, never 0**. Each writer counts what it
   leaves out. A Hybrid root counts both components; its Dense Component counts
   its own, as the Top-Hit Counts do (#107). In a Hybrid build an off-reference row
   is counted with the Overflow it would reach. The Pass 2 counts of a Dense VCF or
   Hybrid build come back from each worker with its column, and a Hybrid checkpoint
   records them (checkpoint format 5).
5. **"Missing" in each layout's own terms**, as ADR 0077 settled for every
   writer: a Dense cell is a missing cell; a Ragged sequence holds no row, so a
   query returns nothing there rather than a `missing` row.
6. **Query: no behaviour change.** A query reads what the Store holds; a 0.2.3
   Store holds no such SE. A release before 0.2.3 that holds `+inf` is still read
   as a value (ADR 0070), and the presence rule cites #338 for why a 0.2.3 one
   holds none.
7. **Validation.** `associations.se_infinite` keeps #337's warning for a release
   before 0.2.3 (OGS-00011's 8 cells until opengwasdb-stores#403). From 0.2.3 a
   release holding any SE above 65,504 fails `associations.se_out_of_range`, and
   `analyses.se_out_of_range_count` and `analyses.se_out_of_range_partition` judge
   the column, the Hybrid root's included.

### 0.2.3

A new `analyses.tsv` column is a patch (spec §21.1). Main's #430 took 0.2.1 and
#381 0.2.2, so this is **0.2.3**: 0.2.2 and the count. `format_version_for` stamps
it on every release whose indels are spelt reference first and which counts;
`CURRENT_FORMAT_VERSION` is 0.2.3. A build writes 0.2.3. A completion carries its
source's features and records the count, so a 0.2.2 source completes to 0.2.3
(`completed_format_version`); a source before 0.2.2 is refused, as since #381. The
converter keeps writing 0.2.0 (its copy of a 0.1.0 source sorts its indels) and
carries no count. A 0.2.2 release opens without a warning; a 0.2.4 one gets the
upgrade warning and is not judged by this build's stamp rule.

### Completion

- **A source cell above the bound is not carried.** Only a source before 0.2.3
  holds one. Completion reads its source as stored, and the gate leaves the cell
  out at the write: missing and unflagged in a grid, no row in a sequence. It is
  not re-imputed, and it is still an input to its block's imputation of other cells
  -- its z, and its SE in the median that scales imputed SEs -- as the source holds
  it. Rebuilt at 0.2.3, the source holds the cell missing, and its completion
  imputes it and reads nothing from it.
- **A Hybrid crossover above the bound is not folded.** It replaces nothing: the
  Dense cell keeps what Dense completion put there, imputed or missing, as had the
  source never held it, and its block's `completion_quality` row is left alone. It
  is counted in the Dense Component's own count as well as the root's -- the cell
  it would have been is that component's (defect 1).
- **A fill above the bound is left missing where candidate fills become the
  release's** (`fill_shards`, shared by Dense and Ragged), so the block's
  `completion_quality` row counts it missing, not imputed, and the arrays, the
  table and the count cannot disagree. A write that still leaves an imputed cell
  out is refused.
- **The count** is the source cells not carried (`observed`) plus the fills left
  missing (`imputed`), plus the source's own count where that is known. The two
  parts are recorded per Analysis in `provenance.completion.se_out_of_range`. A
  source at 0.2.3 or later records a count, carried forward; one with a blank is
  refused. A source before 0.2.3 records none, so its count is unknown and is
  never read as 0 (`carried_counts` returns none): its completion counts each
  source association above the bound as it meets it, exactly, since a completion
  reads every observed cell of its source, and records that. A path that could
  not count exactly would record unknown, never 0; none does.
- **The refit gap is closed by ADR 0078, not by a refusal.** ADR 0077 recorded
  that Ragged completion's refit could fail on an Analysis this rule left with
  fewer than two SEs under a residual plan, and the owner ruled on 18 Nov that
  completion then refuses, naming the Analysis. ADR 0078 (#301) removed the refit
  on 4 Nov: completion codes every cell against its source's coefficients, so such
  an Analysis completes, its model the source's bit for bit and its kept SE
  decoding as the source's does
  (`test_a_residual_ragged_analysis_left_with_one_se_completes_against_its_sources_model`).
  No refusal was added, as there is no failure or fallback left to refuse. A build
  cannot reach it either: a build measures after the gate, and an Analysis with
  fewer than two SEs sends the whole plane to `float16`, as it always has.

  *Note, 18 Nov 2026:* accepted. No
  completion path refits an SE model from stored cells under ADR 0078 (Ragged
  completion takes its source's coefficients; the builders keep the `float16`
  eligibility fallback), so the owner's 18 Nov refusal has nothing to guard and
  none is added.

### The Observed-View Check: exact, not a bound (defect 2)

The first implementation accepted any completed count at or above the cells its
sample found, clipped a falling count to 0 and read a missing target count as 0.
Here the check holds each Analysis's completed count to **exactly** its source's
plus the two recorded parts (`se_out_of_range_miscounted`), never reads a blank as
0, and fails when the parts are not recorded. The cells a sample finds left out
bound the `observed` part from below, and equal it where the sample compared every
source cell -- a whole Ragged Analysis, a Hybrid's every crossover and Overflow
row, or a Dense grid sampled in full (`se_out_of_range_unaccounted`). A cell is
set aside as left out only when a source before 0.2.3 was completed into a 0.2.3
or later target; a 0.2.3 source holding such an SE is invalid, and its cell is
compared like any other. Recording the parts is a provenance key, which moves no
format version (ADR 0078's precedent for `se_exceptions`), not a further column.

### The converter (defect 4)

The converter copies stored codes and cannot leave an association out without
rewriting z, the exception tables, the Top-Hit Index and a Ragged release's
offsets. It refuses a 0.1.0 source holding an SE above 65,504, before anything is
written, naming each Analysis and each array that holds one. Seven 0.1.0 Store
Releases remain (OGS-00001 to 00005, 00007, 00008), and none holds one, so the
refusal blocks no planned conversion.

## The seven review defects of 52edbf1

1. Hybrid completion now adds a crossover it leaves out to the Dense Component's
   own count, and the test asserts 1, not 0.
2. The Observed-View Check requires equality, from recorded parts (above).
3. The JSON schema types the column as a string matching a non-negative integer
   or blank (`^(0|[1-9][0-9]*)?$`); blank is allowed only below 0.2.3, which
   `validate` holds a release to, as a row cannot see its format.
4. The converter's refusal names the Analysis (above).
5. Spec §7a and the model say a counted cell may hold an imputed value after
   completion.
6. ADR 0057 carries a banner naming this ADR and the current version.
7. Every new test was observed to fail against the unfixed behaviour.

## Consequences

- No release this package builds or completes holds an SE above 65,504; `validate`
  fails one at 0.2.3 that does.
- A source reporting such an SE loses that association, and `analyses.tsv` says how
  many, per Analysis.
- A package before 0.2.3 reads a 0.2.3 release with a warning and ignores the
  column; its completion refuses it.
- OGS-00011's three Analyses, rebuilt as a Hybrid, count 1 / 5 / 2 at the root and
  0 / 2 / 2 in the Dense Component. Every other z and SE is unchanged; EAF can move
  by up to ~5e-5 at a variant where another Analysis's association was left out,
  because the residual EAF baseline is a per-variant median (5 cells there, at most
  4.6e-5).

## Rejected

- **Store such an SE as a residual exception.** Whether an association is kept
  would then depend on the plan.
- **Fall back to `float16` for an Analysis left with fewer than two SEs.** There is
  no fit left to fail (ADR 0078).
- **A sampled lower bound in the Observed-View Check.** It passed a count of
  999,999 where 1 was right.
- **Mask a source's over-range cells before imputing**, as the first implementation
  did. It needs a second mask at each layout's source read, outside the gate. Only
  a source before 0.2.3 is affected (OGS-00011: 8 cells, whose z is near 0 and whose
  SE a median outweighs), and the remedy for one is a rebuild.
- **Filter in the readers.** Not every builder reads through one reader, and the
  gate already covers every writer.
