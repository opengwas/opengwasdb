# Whole-stream evidence and one row-admission rule

ADR 0048 decoupled the ancestry-site bound from quantitative phenotype-SD
evidence: the bound stops ancestry accumulation, and a quantitative Analysis
continues its physical scan to EOF. It left the case-control path terminating
the physical scan at the ancestry bound. That was harmless while the only
counts it affected were ancestry's own; it stopped being harmless once the #174
overlap counts and the #175 INFO counts accrued in the same loop, because their
subject is then a ~1% prefix rather than the file. This ADR makes whole-stream
evidence an explicit property of a resolution, and introduces one shared
row-admission rule for the resolver and the Hybrid builder. It amends ADR 0048
and #175's INFO semantics; the ancestry-site bound itself is unchanged.

## Context

`resolve.py::_scan` accumulated ancestry evidence, the #174 overlap counts
(`build_eligible_rows*`, `variant_reference_rows_matched`) and the #175 INFO
dispositions in one pass. A case-control Analysis (`needs_sd=False`) ended that
pass at `max_ancestry_sites`, so for 880 of 4,783 OGS-00011 Analyses (~25% of
all rows) those counts described only a prefix. A store release projected from
them would under- or over-state overlap and build eligibility without raising
anything.

At the same time #175's INFO semantics proved wrong on real input: it treated a
score outside [0, 1] as an unusable disposition and dropped it, and it treated
an Analysis whose declared score had no usable value as a controlled failure.
Neither is right -- a finite score is a number a threshold can compare, and an
Analysis with none can still be built honestly by retaining every row.

## Decision

1. **Whole-stream evidence is requested, not inferred.** Whenever
   `variant_reference is not None`, the INFO policy declares a score, or a MAF
   threshold is numeric, the physical scan continues to EOF (or an explicit
   `max_rows`) and every whole-stream count -- `rows_read`,
   `canonical_rows_observed`, `canonical_rows_retained`, `build_eligible_rows*`,
   `variant_reference_rows_matched`, `info_rows_*`, `maf_rows_*` -- covers all
   rows read, with `stop_reason = eof`. Ancestry fields (`ancestry_rows_read`,
   `ancestry_reference_rows_matched`, `ancestry_stop_reason`) keep their bounded
   meaning. With none requested, ADR 0048's early stop is unchanged.
2. **A usable INFO score is any finite number.** Values above 1 pass any
   threshold <= 1, negative values fall below any positive threshold, and the
   `out_of_range` status is informational only. A row is dropped by INFO only
   when the policy is `filtered`, its score is usable, and it is strictly below
   the threshold; rows whose score is missing, malformed or non-finite are
   retained. A declared Analysis with zero usable scores is built, reports
   `info_score_state = "no_usable_scores"`, and retains every row.
3. **A new optional `maf_threshold` column** carries a finite value in [0, 0.5]
   or the literal `NaN`; absent and `NaN` both mean no filter and `0` disables.
   MAF is `min(af, 1 - af)` from the reader's `effect_allele_frequency`; a row
   with no usable frequency is retained and counted `maf_missing`. A row is
   dropped by MAF only when the threshold is > 0, the MAF is available, and it
   is strictly below the threshold.
4. **One admission rule.** `opengwasdb.build.row_admission.admit_rows` decides
   per batch `keep = ~info_drop & ~maf_drop`. The resolver `_scan` and the
   Hybrid builder's Pass 2 both call it; neither restates it. A row both filters
   would drop is counted once, under `info_rows_below_threshold` (INFO first).
   `canonical_rows_retained` and `build_eligible_rows*` count admitted rows, and
   ancestry/SD evidence sees admitted rows.
5. **Provenance.** The resolver's fingerprint `resolution_config` binds
   `maf_threshold` (float or `None`) beside `info_score_threshold`; the Hybrid
   build records a `provenance.maf` block beside `provenance.info_score`.

## Consequences

- **The #174 overlap counts and the INFO/MAF counts are whole-file where they
  claim to be.** The cost is that a case-control Analysis with a declared filter
  now reads its whole source instead of a prefix -- the same cost a quantitative
  Analysis already pays, and the reason the bound exists is to bound ancestry
  *memory*, which it still does.
- **A record produced under the old prefix semantics is not resumable as one
  produced under these**, because `maf_threshold` enters `resolution_config` and
  the counts differ. `--resume` recomputes it.
- **Stores must not read `variant_reference_rows_matched` as a post-admission
  denominator**; `build_eligible_rows_off_variant_reference /
  build_eligible_rows` remains the projection, now over the whole file.
- **The stores sibling must emit `maf_threshold` on resolver evidence and
  `NaN` otherwise**, and must stop excluding a declared-INFO Analysis with no
  usable scores (opengwasdb-stores #176).
