# An SE beyond float16's range is stored as missing

Issue [#338](https://github.com/opengwas/opengwasdb/issues/338), under epic #347
("Builds never store a statistic that contradicts its source"). It rests on
[ADR 0013](0013-association-status-and-missing-markers.md) (missing is a marker in
both planes), [ADR 0037](0037-statistic-array-encodings.md) (the encodings) and
[ADR 0038](0038-reference-completion-preserves-the-source-encoding.md) (completion
writes into its source's encoding); the format-version rule is spec §21.

## Context

A source can report an SE above 65,504, `float16`'s largest finite value. The
builders narrowed it to `float16` and wrote `+inf` beside a finite `z`; numpy's
`RuntimeWarning: overflow encountered in cast` was the only signal. `beta = z × se`
then reads as infinity although the source value was finite, and the published
OGS-00011 holds 8 such cells (GCST012879, GCST90295971 and GCST90827693; counts
1, 5 and 2). #337 made the validator say so and made the Observed-View Check compare
such a cell rather than call it missing. `StoreQuery` filters with `np.isfinite`, so
a query never returned one -- which is not how spec §15 defines missingness.

The owner's decision (1 Nov 2026): an SE `float16` cannot represent is almost
certainly an error, so a build stores that association as **missing**, with a count
per Analysis, at every point where a build narrows SE: the Dense writers, the Ragged
writers, the Hybrid Overflow and completion.

## Decisions

1. **The threshold is `np.finfo(np.float16).max`, 65,504; an SE strictly above it is
   missing, whatever the plane's encoding.** An `int8_residual` plane could hold the
   value in its `float32` exception table, but then whether an association survived
   would depend on the encoding the build picked after measuring. The rule is applied
   to the data *before* the encoding is measured, so the plan sees the masked data.
   An SE of exactly 65,504 is kept.
2. **One rule, one function.** `opengwasdb.encoding.se_range.mask_se_out_of_range(z,
   se)` returns both with the Missing Marker (`NaN` in memory) where `se` exceeds the
   bound, and the count; `mask_se_grid` counts per Analysis and `mask_se` is the same
   rule where `z` is not in hand. It is called where an Analysis's `z` and `se` are
   both in hand -- the Dense in-memory scatter, the VCF spill (`_spill_column`), the
   Hybrid's Dense spill, the completion band passes -- so the two markers cannot
   disagree. No caller restates the comparison.
3. **The encoders refuse rather than write `+inf`.** `StoreCodec.encode_se` (both
   branches), `_write_se_plane`, `DenseSePlane.patch` and `_write_se_streamed` raise
   `SeOutOfRangeError`, naming the count and the first value. A path that forgot the
   rule fails loudly instead of producing a release that looks right.
4. **The count is a new store-only `analyses.tsv` column,
   `n_missing_se_out_of_range`**, following `completion_n_missing_total`'s precedent
   through `model/analyses.py`. In a Hybrid release the root counts both components and
   the Dense Component's own table counts only its cells, as `n_hits_*` do (#107).
5. **"Missing" in each layout's own terms.** Dense and the Dense Component: a Missing
   Marker cell. A Ragged Observed-Only sequence: the association is **not retained**, as
   an unusable source row is not, and it is counted (`RaggedCSRWriter.add_analysis` is
   the one seam every Ragged builder and the Hybrid Overflow reach). A Ragged
   Reference-Completed sequence stores explicit missing rows, so it stores one.
6. **Query: no behaviour change.** `StoreQuery` already treats a non-finite SE as
   missing; that is now the build's rule too, and the mask carries a comment citing
   #338.
7. **Validation keeps #337's infinite-SE warnings.** OGS-00011 still holds 8 such cells
   until opengwasdb-stores#403 repairs it; promoting the warning to an error is a later
   decision.
8. **`format_version` moves to 0.2.1** (below).

### 0.2.1 and the column's contract by version

The owner ruled (1 Nov 2026) that the format version moves: spec §21.1 already
classifies "a new `analyses.tsv` column" as a patch. `CURRENT_FORMAT_VERSION` is
`0.2.1`; `SUPPORTED_FORMAT_VERSIONS[(0, 2)]` is `(1,)`, so `0.2.1` reads without a
warning and `0.2.0` still reads. Completion writes into its source's arrays, so it
keeps the source's version and works on any release of the writable series (`0.2.x`).
The column is **required** from 0.2.1 -- validation errors when the header lacks it or a
value is not a non-negative integer -- and **optional** below it, where an absent or
empty value reads as unknown, never as 0. Builders and the converter always write an
integer.

### Completion

Completion re-encodes the whole `se` plane, so it applies the rule to the source's
observed cells as well as to imputed candidates, or a `0.2.0` source holding `+inf`
(OGS-00011 has 8) would meet the encoder guard.

- A source cell above the bound is masked in `z` and `se` and is then a missing cell, so
  the imputation neither conditions on it nor leaves it out of its targets.
- An imputed candidate with SE above the bound is left missing. It is dropped once, where
  a worker's candidate fills become the release's (`_shard_checkpoint_fills_by_band`,
  `_merge_checkpoint_rows`), so the arrays, the `completion_quality` rows
  (`n_imputed` falls and `n_missing` rises by the same count) and the count cannot disagree.
- The completed count is the source's recorded count (blank reads as 0: a release without
  the column stored every such SE in its plane, so the cells completion finds there are all
  of them) plus the observed cells it masked plus the imputed cells it left missing.
- A Hybrid's Overflow is rebuilt through the writer, which does not retain such rows; a
  crossover folded into the Dense Component is masked there. The root adds the source's
  Overflow share (root minus Dense Component, recorded) to its own.

### The Observed-View Check

The check accepts exactly this transition: a source cell above the bound is set aside
(counted as `source_se_out_of_range`, not compared), and the completed counts must
account for each one it samples -- per Analysis, `sampled <= completed count - source
recorded count`. An inequality, not an equality: the difference also covers an imputed cell
left missing, which is not a source cell, and a sampled check cannot count the source's
over-range cells exhaustively. Any other change to an observed cell still fails.

### The converter

The converter copies stored codes and does not re-encode. A `0.1.0` source that holds an SE
above the bound cannot be converted faithfully without rewriting `z`, the exception tables,
the top-hit tiers and, for Ragged, the offsets. It therefore **refuses** such a source,
naming the count, and tells the operator to rebuild; otherwise it writes `0.2.1` with the
count `0` (after scanning every `se` plane) -- a measured zero, not a default.

## Consequences

- A build no longer writes `+inf`; a release at 0.2.1 holds none. The validator's warning
  stays for 0.2.0 releases.
- A source reporting an SE above 65,504 loses that association. The count says how many.
- A package built before this reads a 0.2.1 release with a warning and ignores the column.

## Rejected

- **Keep `0.2.0` by amending spec §21.1 (or extending §21.0) so a new column applies in
  place.** §21.0 covers vocabulary changes and §21.1 says patch; amending a current row to
  avoid a bump would leave a release that differs in a required column carrying the same
  version as one that does not.
- **A `provenance` block instead of a column.** The count is per Analysis and every other
  per-Analysis count lives in `analyses.tsv`, where `n_hits_*` and the completion counts
  are already summed across a Hybrid's components.
- **Store such an SE in an `int8_residual` plane's exception table.** Whether an
  association is kept would then depend on the encoding the build measured.
- **Filter in the readers.** Not every builder reads through one reader (the VCF source,
  BESD and tabular readers differ), so a reader-level filter would not cover every path.
