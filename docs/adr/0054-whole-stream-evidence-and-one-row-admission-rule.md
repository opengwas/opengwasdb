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

## Amendment: a MAF floor admits only rows whose MAF it can read (opengwasdb#331)

Decision 3 said a row with no usable frequency is **retained** and counted
`maf_missing`. That is wrong for a `filtered` policy, and the review that
prompted it found real input: the forensic audit of OGS-00011 (opengwasdb-stores
#371) found 192,435,653 associations and 101,098,177 off-panel variants admitted
as `maf_missing`, 83% of the off-panel variants from five publications whose
`effect_allele_frequency` is `0.0` or blank across tens of millions of
monomorphic and ultra-rare rows. A declared floor is a statement that the Store
holds common variation; a row whose frequency cannot be read cannot demonstrate
that, and retaining it defeated the filter exactly where it mattered.

`admit_rows` therefore reads a source frequency as evidence when it is finite
and inside `[0, 1]`, and drops a row under `filtered` when it has none. Exactly
`0.0` and exactly `1.0` are frequencies inside that domain: their MAF is `0`,
which is below any positive threshold, so a `filtered` policy drops them for the
filter's own reason. `0` (`disabled`) and an undeclared column (`unavailable`)
still drop nothing, and a row they keep keeps its frequency.

This amends decision 3's retention clause only. The domain of a *stored*
association's effect allele frequency is still ADR 0036's: `stats.parse_af`
reports no EAF for exactly `0` and exactly `1`, so in the pipeline those cells
reach `admit_rows` as "no usable frequency" and are dropped as such, counted
`maf_missing` rather than below-threshold. A caller that hands `admit_rows` a
raw `0.0` or `1.0` gets the below-threshold disposition; no reader does that
today and this amendment does not change `parse_af`.

Consequences, which are the point of writing it down:

- **`maf_rows_missing` now counts rows a declared floor dropped.** It keeps its
  meaning as a disposition among observed rows -- no usable frequency -- and
  gains a consequence. `maf_rows_below_threshold` keeps counting only rows a
  threshold was actually compared against, so the two are never both true of one
  row and neither is a synonym for "retained".
- **A common-variant Store built from a source that reports no frequency at all
  now admits nothing.** That is the declared contract rather than a regression:
  an Analysis that declares a floor over a file with no frequency column cannot
  satisfy it, and the record says so through its counts and its empty store
  rather than through a full one.
- **opengwasdb-stores must read `maf_rows_missing` as rows the floor dropped**,
  and must not present it as a retained-but-unjudged tally.
- **`parse_af` is deliberately unchanged.** Reversing ADR 0036's amendment would
  store a frequency of `0` for every monomorphic association -- a Store-format
  change (spec §EAF: no EAF means no frequency) with consequences in ancestry,
  phenotype-SD and EAF encoding, none of which this admission fix is about. The
  two rules answer different questions, and the cost of keeping them apart is
  only that the pipeline reports these rows as `maf_missing` rather than
  below-threshold.

### Scope: which builds apply it

The rule is applied where a build streams per-Association rows into a Store that
carries off-panel variation and consults an Analysis's declared policy: the
Hybrid builder's Pass 2, and the resolver that writes each Analysis's record.
Two builds are outside it, deliberately:

- **Standalone Dense (`build-dense-vcf`)** writes a fixed reference-panel axis
  (ADR 0009; the query facade's contract, ADR 0033): every stored cell is a panel
  variant, and an off-panel row is not admitted at all, so there is no admission
  decision for a MAF floor to make. Its axis is the panel's, not the source's.
- **The Ragged builders** (`build-ragged-ssf`, `build-ragged-besd`) do not call
  `admit_rows` and ignore a declared `maf_threshold` entirely. That is unchanged
  by this amendment and is said here so no reader assumes a Ragged release was
  filtered by a column it never read.

A single-pass Hybrid build with `--variant-reference` (OGS-00011's shape) has no
Pass 1 harvest, so a rejected row's off-panel variant cannot reach the shared
axis at all: only an admitted association's key does. A two-pass build keeps the
variant on the axis and drops its association (issue #255) -- the axis is Pass
1's harvest, not the admitted rows.

### Checkpoint records

`CHECKPOINT_FORMAT_VERSION` moves to 2. A format-1 checkpoint's Pass 2 spills
were written under the old retention rule and can hold the rows this amendment
drops, and nothing in the spills records which rule wrote them. A resume
therefore refuses a format-1 record outright rather than publish an unfiltered
Store from it: a resumed build may not reinterpret records its writer did not
mean (ADR 0023).
