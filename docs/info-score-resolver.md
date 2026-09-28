# CORE resolver INFO and MAF diagnostics (stores #175, #176)

`resolve-analyses` reads only the manifest-declared GWAS-SSF imputation-score
column, kind and independent provider provenance. A familiar source header is not
a declaration. An exact column mismatch is a controlled failure naming the
Analysis. It also reads an optional per-Analysis `maf_threshold`
(`opengwasdb.model.maf_policy`): a finite number in [0, 0.5], or the literal
`NaN`/an absent column meaning no MAF filter.

## Whole-stream evidence (stores #176)

The ancestry bound (`--max-ancestry-sites`) stops *ancestry accumulation only*.
Whenever whole-stream evidence is requested -- a variant reference, a declared
INFO policy, or a numeric MAF threshold -- the physical scan continues to EOF (or
an explicit `max_rows`), and every whole-stream count below covers **all rows
read**, with `stop_reason = eof`. The ancestry fields keep their bounded meaning.
Only a case-control Analysis with none of those requested keeps the pre-#176
early physical stop at the ancestry bound.

## INFO semantics (stores #176)

- A **usable score is any finite number**: a score above 1 passes any threshold
  <= 1, a negative score falls below any positive one. `info_rows_out_of_range`
  is informational -- usable scores outside [0, 1], retained or filtered by value
  like any other usable score; the `out_of_range` status is not a drop.
- A row is dropped by INFO **only** when the policy is `filtered`, its score is
  usable, and `score < info_score_threshold` (equality passes).
- Rows whose score is missing, malformed or non-finite are **retained** and
  counted by reason.
- A declared Analysis with **zero usable scores** is not a controlled failure:
  every row is retained and `info_score_state = "no_usable_scores"`. Stores then
  emits `NaN` INFO cells (its emission rule requires state `disabled`/`filtered`
  and `info_rows_usable > 0`) and the Analysis stays included.
- Explicit zero (`disabled`), literal `NaN` (`unavailable`) and absent legacy
  (`legacy_absent`) retain all canonical rows; their `info_score_state` and
  fingerprints differ.

## MAF semantics (stores #176)

- MAF is `min(af, 1 - af)` from the reader's `effect_allele_frequency`. A row
  whose `af` is missing, non-finite or outside [0, 1] is **retained** and counted
  `maf_rows_missing`.
- A row is dropped by MAF only when the policy is `filtered` (> 0), the MAF is
  available, and `MAF < maf_threshold` (equality passes). `0` disables.
- When both filters would drop a row it is counted once, under
  `info_rows_below_threshold` (INFO first), never also under
  `maf_rows_below_threshold`.

## One admission rule

`opengwasdb.build.row_admission.admit_rows` decides per batch which rows are
admitted (`keep = ~info_drop & ~maf_drop`). The resolver `_scan` and the Hybrid
builder's Pass 2 both call it, so `canonical_rows_retained` (resolver) and
`associations_retained` (builder) are the same rule on their own populations.
Ancestry/SD evidence and `build_eligible_rows*` see admitted rows only.

## Per-Analysis JSON `diagnostics` (all row counts are pre-deduplication)

- `canonical_rows_observed` (`rows_read` alias) counts canonical-identity rows
  yielded by the reader over the physical scan, not raw source lines.
  Unnormalisable identities dropped in the reader are not observable here.
  `stop_reason` is `eof`, `row_limit`, or `ancestry_site_limit`; under whole-stream
  evidence it is `eof`.
- `canonical_rows_retained` counts **admitted** rows. `info_rows_below_threshold`
  counts usable scores below a positive threshold. `info_rows_missing`,
  `info_rows_malformed`, `info_rows_nonfinite`, `info_rows_out_of_range`,
  `info_rows_usable` are dispositions among observed rows, including at threshold
  zero.
- `maf_state` is `unavailable` (`NaN`/absent), `disabled` (0) or `filtered` (> 0).
  `maf_rows_below_threshold` counts rows dropped by MAF that INFO did not already
  drop; `maf_rows_missing` counts rows with no usable frequency, all retained.
  Both are zero when no MAF filter is declared.
- `build_eligible_rows` counts admitted rows with a finite beta and strictly
  positive finite SE yielding a finite z. With `--variant-reference`,
  `build_eligible_rows_on_variant_reference` and
  `build_eligible_rows_off_variant_reference` partition this eligible count;
  without it both are `null`. These counts are **rows**, not stored Dense cells.
- `ancestry_reference_rows_matched` counts retained rows on the ancestry
  reference in the ancestry accumulation prefix. Legacy
  `variant_reference_rows_matched` remains **pre-INFO/pre-MAF** canonical rows on
  the variant reference over the physical scan, for backward compatibility. It
  must **not** be used as a post-admission build denominator.

## Fingerprint

`resolution_config` binds `info_score_threshold` and `maf_threshold` (each a
float or `None` for `NaN`/absent), so a change to either invalidates every
resumed record, exactly as the scan bound does.

## Stores integration

Replace the #174 publication projection's legacy `rows_scanned` /
`variant_reference_rows_matched` division with `build_eligible_rows_off_variant_reference
/ build_eligible_rows` from the new diagnostics, requiring a variant-reference
input and a full scan (`stop_reason=eof`) if publishing a whole-source projection.
Treat a zero eligible denominator as unavailable, not zero off-axis share. This is
a projected eligible *row* share, not a deduplicated Dense-cell share.
