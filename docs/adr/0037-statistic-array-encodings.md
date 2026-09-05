# Statistic array encodings: fixed-point z, residual-coded EAF, reference EAF for imputed cells

Supersedes ADR 0036's Decision 3 (`float32` for `eaf`) and refines its Decision 5
(Reference Completion). Leaves ADR 0036's semantics intact: EAF is still per
(variant, Analysis), still oriented to the stored effect allele, still declared
per Analysis by `eaf_scope`. Only the physical encoding changes — and, unlike
ADR 0036, this **is** a breaking format change.

## Context

ADR 0036 stored EAF as a `float32` plane parallel to `z`/`se`. That roughly
doubled a store's statistic bytes, which is a large price for a column that is
annotation rather than the finding. Reviewing that cost surfaced three
questions, each answered by measurement on this project's own pilot data
rather than from theory. All figures below are **compressed** bytes per stored
cell under the store's own codec (Blosc zstd level 3 + bitshuffle), because
raw size is not what a Store Release occupies.

### `float16` is the wrong encoding for `z`, and always was

`z` is finite but not usefully bounded by the pilot values first sampled for
this ADR: the later #117 rebuild found |z| up to **163.66** in pilot stores,
and the `ukb-b` top-hit index reaches **137.5**. It needs *uniform* precision
because p-value error scales with `z · Δz`. `float16` gives the opposite:
precision degrades exactly where p-values are steepest.

Three different quantities get confused here, so they are named separately
throughout (a first draft of this ADR conflated them — see the correction note
at the end):

- **worst-case error** — the largest p error any value can suffer under
  round-to-nearest, i.e. from half a quantisation step;
- **actual error** — what a specific value suffers;
- **bin width** — the p ratio between the two ends of one quantisation
  interval, which is *not* an error bound.

For `float16`, whose step doubles with magnitude:

| z | float16 step | worst-case p error | actual for this z |
|---|---|---|---|
| 5.0 | 0.0039 | 1.010× | 1.000× |
| 30.0 | 0.0156 | 1.264× | 1.000× |
| 47.8 | 0.0312 | 2.110× | 1.818× |
| 137.5 | 0.1250 | 5,400× | 1.000× |

The tail is not theoretical. A survey of the `ukb-b` store's top-hit index
(9.85M variants × 2,514 analyses; the index holds every |z| ≥ 3.48, so these
counts are exact for the whole store) gives:

| | |
|---|---|
| max \|z\| | **137.5** |
| p99 / p99.9 / p99.99 | 10.8 / 20.1 / 37.3 |
| \|z\| > 32 / 48 / 64 / 100 | 6,346 / 1,760 / 568 / 136 |
| non-finite | 0 |

The extremes are genuine, not malformed: the largest are HERC2/OCA2
(`15:28120472`, ukb-b-19560) and the MC1R region (`16:89.7–89.9Mb`,
ukb-b-533) — the classic pigmentation loci, with plausible SEs around 0.0015
and betas of 0.2–0.43.

So `z` is **not** bounded at 64, and a build limit there would reject 568 real
associations in a store this project already holds.

| encoding | B/cell | worst-case p error at z=30 |
|---|---|---|
| float16 | 1.85 | 26.4% |
| **int16 fixed 1/1024 + overflow table** | **~1.47** | **1.5%** |
| int8 | 0.47 | unusable |

Fixed-point int16 is smaller *and* substantially more accurate — smaller
because bitshuffle groups the near-constant high bytes of a bounded quantity,
where float16's exponent bits churn for the small values that dominate.
Accuracy no longer degrades with magnitude, and values outside the
representable range are held exactly in a sparse table rather than clipped or
rejected.

### `float16` is the *right* encoding for `se`

The mirror image, and worth stating so it is not "fixed" later by analogy:
`se` spans 3.2 decades and needs relative precision, which is what a
floating-point exponent already provides.

| encoding | B/cell | max relative error |
|---|---|---|
| **float16** | **1.75** | 4.9e-4 |
| int16 log-se | 1.84 | 1.1e-4 |
| int16 uniform | 1.36 | 4.1e-1 |

### EAF is highly redundant, but only against the right baseline

Cross-analysis EAF agreement, as the log-residual against a candidate
baseline:

| store | baseline | residual sd | p99 | max |
|---|---|---|---|---|
| FinnGen, 8 endpoints (one cohort) | within-store median | **0.0121** | 0.059 | 0.63 |
| FinnGen | EUR reference panel | 0.436 | 1.82 | **8.07** |
| GWAS Catalog, 7 EUR studies | within-store median | **0.0876** | 0.154 | 4.22 |

A within-store baseline leaves residuals small enough to code in 8 bits. The
reference panel does not: a max log-residual of 8.07 is a **3000× discrepancy
in MAF**, the Finnish bottleneck showing up exactly as it should. This is the
central negative result — reference-panel EAF is not a proxy for a cohort's
own allele frequency, and must never be substituted for a missing one.

Cross-*study* residuals (GWAS Catalog, one ancestry) are 7× wider than
cross-*endpoint* residuals within one cohort, which is what separates the
Hybrid case from Dense/Ragged — not the layout itself, but the cohort
heterogeneity the layout happens to correlate with.

### The measurement found a live data defect

While computing the above, GCST003566 — one of the ten studies in the
`gwas-catalog-eur-hybrid` pilot — turned out to report
`effect_allele_frequency` against the *other* allele: r = −0.999 against every
other study and against the reference panel. Nothing in the pipeline could
have caught it, because until ADR 0036 no store retained EAF at all. See #115.

## Decision

### 1. `z` is `int16` fixed-point plus a sparse exact overflow table

Default scale 1/1024 — recorded in the plan, not hard-coded, so it can be
chosen per store from measured data like every other parameter here.

Two codes are reserved, because **an integer array cannot hold NaN and the
missing-cell contract currently depends on it** (store-format §15; ADR 0013
derives Association Status from paired Z/SE NaNs):

| code | meaning |
|---|---|
| `-32768` | missing — decodes to NaN, and the paired `se` must also be NaN |
| `-32767` | out of range — exact `float32` in the overflow table |
| `-32766 … 32767` | z × 1024, i.e. −31.998 … +31.999 |

The representable interval is deliberately stated as exact endpoints rather
than "±32": signed fixed point is asymmetric, and reserving codes makes it
more so.

Measured overflow cost on `ukb-b`, the largest store available:

| scale | range | overflow cells (of 24.8e9) | table | worst-case p error at range edge |
|---|---|---|---|---|
| 1/512 | ±64 | 572 | 7 KB | 6.4% |
| **1/1024** | **±32** | **6,346** | **74 KB** | **1.6%** |
| 1/2048 | ±16 | 95,558 | 1.1 MB | 0.8% |

1/1024 is the default: a 74 KB side table against a multi-gigabyte store, with
worst-case p error of 1.6% at the range edge and 0.24% at z = 5, against
`float16`'s 26.4% at z = 30.

**A build fails only on non-finite or malformed statistics, never on a
legitimate strong association.** The earlier draft of this ADR proposed failing
at |z| > 64; the `ukb-b` survey shows that would reject real pigmentation hits.

`se` stays `float16`. Both choices follow from the shape of the quantity, not
from a general preference for integers.

**As implemented** (issue #114, store-format spec §6a, §15, §20). Four details
were settled during implementation and are recorded here rather than left to
the code:

- **The overflow table is keyed by flat position**, `variant_index x
  n_analyses + analysis_index` for a Dense grid and the association's ordinal
  for a CSR, in two sorted arrays (`z_overflow_index`, `z_overflow_value`)
  beside the plane they describe. Keying it by position rather than by
  (variant, analysis) is what lets one implementation serve both layout shapes,
  and it is why every read of a fixed-point plane says *where* its values came
  from: the codec cannot resolve an out-of-range cell it cannot locate.
- **The arrays are written even when empty**, so "this plane is fixed point"
  and "this plane has a table" are the same statement, and validation needs no
  special case for the store that overflows nothing.
- **`decode_se(raw_se)` is deliberately absent** while `decode_z()` is public.
  `z` is independently meaningful — Rho and the top-hit harvest need nothing
  else to interpret it — but once #118 codes `se` as a residual it cannot be
  decoded without EAF, and a function that looked decodable without it would
  invite exactly the half-done read that returns silent nonsense.
- **The plan carries `z` and `se` only.** `eaf`'s physical encoding is
  unchanged by this issue — it is still ADR 0036's `float32` plane, present or
  absent per source — so it joins `StoreEncoding` with #116, the change that
  gives it something to declare. The scaffolding is proven on the two planes
  whose encoding this issue actually decides.

### 2. EAF is a per-variant baseline plus a per-cell `int8` log-residual

- `eaf_baseline` — one `float32` per variant, the within-store representative
  observed EAF. It amortises only when many Analyses share a variant axis: a
  per-variant array costs roughly `4 / cells_per_variant` raw B/cell, which is
  negligible on a 2,514-Analysis Dense grid and material on sparse Ragged
  stores (#117).
- `eaf` — one `int8` per cell, the quantised log-ratio to that baseline.

Two codes are reserved: one for "this Analysis has no EAF here", one for "out
of range — exact value in a sparse side table". 253 levels remain for the
residual.

The builder **measures** the residual spread and selects the range, rather
than inferring it from the layout. Nothing stops a Dense manifest spanning
several cohorts, and a store that did would silently clip against a range
chosen on the assumption it did not.

| range | step (logit) | **worst-case** error | rebuilt-pilot plane B/cell (#117) |
|---|---|---|---|
| ±0.5 | 0.00395 | 0.20% | 0.276 FinnGen; 0.492 metabolome; 0.558 GWAS Catalog quant |
| ±1.0 | 0.00791 | 0.40% | 0.790 GWAS Catalog case-control |
| ±2.0 | 0.01581 | **0.79%** | not selected by the rebuilt pilots |

254 levels across a width of 2·range, so accuracy is a function of the range
chosen — a fixed "within 0.5%" acceptance criterion is unsatisfiable at ±2.0
and must be stated per range. (The table's error column was computed for 254;
an earlier draft said "two reserved, 253 remain", which does not add up for a
type with 256 codes.)

**As implemented** (issue #116, store-format spec §6a, §9, §20). This section
was under-specified at the mathematical, side-table and Hybrid-component
levels, so implementation had to settle six things. They are recorded here
rather than left to the code, and the spec states them normatively.

- **The transform is the logit**, `log(f / (1 − f))`, not `log(f)` and not
  `log(MAF)`. The draft moved between all three. The logit is the only one
  whose error is bounded on *both* sides of the frequency: a residual error of
  `d` moves `f` by a relative `(1 − f)·d` and `1 − f` by a relative `f·d`, so
  neither the effect allele frequency nor the minor allele frequency users
  filter on can be wrong by more than `d`. `log(f)` is blind to the minor side,
  which for a frequency near 1 is the only side anyone reads; a signed log-MAF
  would need a sign bit there is no room for and a rule for a cohort crossing
  0.5 relative to its baseline. Measured on FinnGen chr1 (8 endpoints, 429,961
  variants, 3.44M EAF-bearing cells) the logit and log-EAF residuals are the
  same width — sd 0.0170 against 0.0169, p99 0.081 against 0.081, max 0.632
  for both — so the choice costs nothing in bytes and buys the bound.

- **The code map**: `-128` absent, `-127` exception, `-126 … 127` residual.
  254 levels, `step = range / 127`, representable interval
  `[−126·step, +127·step]`. Half a step is the round-trip bound, measured at
  0.219% / 0.419% / 0.812% worst case at ±0.5 / ±1.0 / ±2.0 against the
  0.197% / 0.394% / 0.787% half-step, the difference being the `float32`
  result type's own rounding.

- **The baseline** is the median, in logit space, of the frequencies the
  release's Analyses report at that variant *strictly inside* (0, 1); the
  median of an even count is the mean of the two central logits. A variant
  with none gets a NaN baseline and all its cells become exceptions. A variant
  seen in one Analysis gets that Analysis's frequency and a residual of zero —
  correct, and the reason the byte rule below exists.

- **Frequency 0 and 1 have no logit, so they are exceptions**, stored exactly.
  No clamping and no epsilon: a monomorphic cohort is a fact about the data,
  not a number to nudge. Crossing 0.5 relative to the baseline needs no rule at
  all, which is the second reason the logit wins over a signed log-MAF.

- **The exception table is `z`'s overflow table with a different name.** Same
  structure (`eaf_exception_index` / `eaf_exception_value`, `int64` positions
  sorted and unique, `float32` values), same flat-position keying, same rule
  that it is written even when empty, same validation that every exception cell
  has an entry and the table describes no other cell. One implementation serves
  both, so they cannot drift. Measured lookup cost, which the review asked for
  alongside the bytes: **109 ns/cell** against a 1,000-entry table, 219 ns
  against 100,000, 684 ns against 1,000,000 — a binary search per cell, paid
  only for the cells that are exceptions.

- **The range is chosen by measured bytes, not by a clipping threshold.** The
  draft's 1e-3 rule was self-contradicting: the measured Hybrid clipping of
  0.11% exceeds it, so the rule would have picked ±2.0 and invalidated the
  scenario it was quoted for. Replaced by: the smallest candidate whose
  exception fraction is at most **2%** — an exception costs 12 raw bytes
  against the plane's 1 byte per cell, so 2% caps the side table at 24% of the
  plane — and within that budget the smaller range always wins, because it is
  2–4× more accurate on every cell that is not an exception and an exception is
  exact either way. Then the resulting raw byte total is compared against
  ADR 0036's `float32` plane and **`float32` is written when the residual
  coding would not be smaller**. That is where the baseline's economics are
  decided rather than assumed: the saving is per *cell*, the cost is per
  *variant*, so a sparse Ragged store with roughly one EAF-bearing cell per
  variant stays in `float32`, exactly as the review required.

  On this project's own data the rule reproduces the scenarios above. Measured
  by running the shipped encoder over chr1 of two pilots' source files, as a
  Dense grid, under the store's own codec:

  | store | cells/variant | chosen | `float32` B/cell | chosen B/cell | round-trip p99 / max |
  |---|---|---|---|---|---|
  | FinnGen R13, 8 endpoints | 8.00 | **±0.5** | 1.373 | **0.718** = 0.294 plane + 0.424 baseline + 0.000 table, **−48%** | 0.184% / 0.262% |
  | GWAS Catalog EUR, 9 studies | 4.86 | **±1.0** | 5.072 | **2.023** = 1.245 + 0.696 + 0.082, **−60%** | 0.374% / 0.394% |

  Exception fractions at ±0.5 / ±1.0 / ±2.0: FinnGen 0.0003% / 0% / 0%, GWAS
  Catalog 3.75% / 1.71% / 0.87%. Every exception cell round-trips **exactly**.
  The round-trip figures are the worse of the relative error on EAF and on
  1 − EAF, against half-step bounds of 0.197% and 0.394%.

  The baseline figure of 0.424 B/cell across 8 Analyses is this ADR's own
  prediction (0.42), measured.

  **The rebuilt pilots in #117 settled which table was wrong.** The earlier
  §2 table predicted 0.14 B/cell for one cohort and 0.52 for mixed studies;
  measured Store Releases miss those figures by 1.5–4.0×. The chr1 dry-run
  reproduced for FinnGen (0.276 full rebuild against 0.294 dry-run), so the
  error was the original prediction, not the implementation note. The decision
  still holds: the range-selection rule chose the expected ranges unprompted,
  and residual coding remains smaller than the `float32` plane it replaces.

  Absolute `float32` figures also differ from the 3.39 B/cell quoted earlier
  because these are chr1 subsets measured as Dense grids rather than whole
  stores; for the GWAS Catalog pilot, which is Hybrid, a Dense grid overstates
  the `float32` plane by counting cells no Analysis reports.

- **`int16_log_maf` is not implemented.** The decision tree in #119 named it as
  the escape valve for data no range fits. `float32` is a better one: it is
  exact where `int16_log_maf` is lossy, it already exists, and the byte
  comparison above reaches it by measurement rather than by a special case. The
  saving forgone is real (1.94 against 3.39 B/cell on a store with no
  cross-Analysis redundancy at all), and it can be added later as a fourth
  `kind` without changing anything else; it is left out because a third lossy
  encoding is a third thing to get wrong, for a case no pilot exhibits.

### 3. `se` may be coded as an `int8` residual, but only where every Analysis has EAF

`log(se) ≈ a + b·log(2f(1−f))` fits real data at R² = 0.992 with b = −0.52
against a theoretical −0.5 — `opengwasdb.completion.block` already relies on
this relationship to derive imputed SE. Coding the residual costs 0.86 B/cell
instead of 1.75, at 0.12% median / 0.24% p99 round-trip error, far below the
sampling error of the SE estimate itself.

It requires EAF **in the same cell**, and a zarr array has one dtype, so a
single EAF-less Analysis forces the whole array back to `float16`. This is why
the Hybrid case differs from Dense/Ragged.

### 4. Reference-panel EAF is stored once per variant, for imputed cells only

An imputed cell's EAF *is* the panel's, identical for every Analysis imputed
at that variant, so it is a per-variant constant: `eaf_reference`, one
`float32` per variant. It is ~0 B/cell only on Dense grids with many Analyses
per variant; #117 measured **3.206 B/cell** on completed `eqtlgen`, where the
Ragged store has only 2.4 cells per variant. Which cells it describes is
already recorded, by `association_status` (ADR 0013), exactly as spec §9
specifies.

Observed cells whose source reported no EAF stay NaN. They do **not** fall
back to the panel value — see the 3000× result above.

This supersedes ADR 0036's reason for deferring imputed EAF to #113. That
deferral assumed the value had to travel through the completion checkpoint
shards; as a per-variant constant read straight from the panel, it does not.

Constraint: one panel per completed store. True of
`complete_{dense,ragged,hybrid}_store` today, but it becomes load-bearing
here, so it is recorded in `manifest.json` rather than left implicit.

**As implemented** (issue #116, store-format spec §6a, §9). Three details were
settled during implementation:

- **`eaf_reference` is per component, not per release.** A single
  release-level flag cannot say *which* component holds the array, and a
  Hybrid release's Dense Component has imputed cells where its Ragged Overflow
  does not. Each component already carries its own manifest — spec §16 nests
  the Dense Component as a self-contained Dense release — so the flag lives in
  the component's own `encoding` block, and it is the one field on which the
  two components of a Hybrid release are allowed to differ. Validation compares
  their plans with it normalised away, so a real disagreement about the
  *encoding* still fails the store.

- **Decoding is one operation, not three.** `decode_eaf` requires the imputed
  mask and the reference array whenever the plan declares `eaf_reference`,
  rather than defaulting them: a decode that quietly skipped the substitution
  would return NaN for every imputed cell in a store that holds the value, and
  one that applied it without the mask would return the panel's frequency for
  observed cells — the 3000× error above. The failure mode of getting this
  half-right is silent, so it is made unrepresentable instead of documented.

- **Reference EAF is added independently of the observed plane** — including
  to a release whose source stored no EAF at all, which is the case issue #113
  was raised about. Such a release carries `eaf_reference` and no `eaf` array:
  NaN on every observed cell, the panel's frequency on every imputed one. The
  first implementation refused that pairing, on the argument that a release
  whose *only* frequencies were the panel's would be read as a frequency
  column contradicting its Analyses' `eaf_scope=absent`. That gets the
  direction wrong — `eaf_scope` is derived from what the release holds, so an
  Analysis that gains imputed cells is stamped `association` by completion and
  the declaration and the arrays agree. Refusing the pairing made the release
  with no frequencies of its own the one release that could not hold the
  panel's, which is precisely what #113 asked for.

  The panel may equally hold none: an LD Reference Panel is supplied for
  imputation and its `EAF` column is optional, so a completion against one
  without it carries no `eaf_reference` and leaves imputed cells NaN, rather
  than failing the run.

The baselines and the exception values travel with the observed cells across
Reference Completion's variant remap rather than being recomputed from the
decoded frequencies. Recomputing would move every baseline by up to half a step
and re-quantise every cell against the moved baseline, so a completed release
would be *less* accurate than the source it was completed from, for no reason.

### 5. Scenarios

| | z | se | eaf | total | vs today |
|---|---|---|---|---|---|
| **A** Dense/Ragged, no EAF | int16 1.47 | f16 1.75 | — | **3.22** | −11% |
| **B** Dense/Ragged, EAF, after #118 | int16 1.47 | int8 0.86 | 0.14 | **2.47** | −31% |
| **B′** Dense/Ragged, EAF, shipped before #118 | int16 1.47 | f16 1.75 | 0.14 | **3.36** | −7% |
| **C** Hybrid, mixed | int16 1.47 | f16 1.75 | 0.52 | **3.74** | +4% |
| **D** + Reference-Completed | | | +ref, `4 / cells_per_variant` raw B/cell | | |

Today's baseline is 3.60 B/cell carrying no EAF at all. The B row is
conditional on #118: shipped `format_version` 2.0 stores still declare
`"se": {"kind": "float16"}`. In #117, EAF-bearing pilot stores with few
cells per variant were often larger than these scenario totals because the
per-variant `eaf_baseline` cost was not negligible; the 2,514-Analysis `ukb-b`
case is the Dense grid where the baseline is expected to amortise away.

### 6. Allele-flipped EAF is rejected at build time

Each Analysis's A1-oriented EAF is correlated against the reference panel;
`r < 0` fails the build. Measured separation is unambiguous, and population
bottlenecking does not confuse it — Finnish frequencies differ from EUR by up
to 3000× in magnitude but not in direction.

| source | r vs EUR panel |
|---|---|
| GCST003566 | −0.9992 |
| GCST005076 | +0.9996 |
| FinnGen (bottlenecked isolate) | +0.9954 |
| Metabolome | +0.9996 |

**As implemented** (issue #115, store-format spec §9.1). This part of the ADR
landed ahead of the encoding work, since it needs no codec and no
`format_version` bump. Three details were settled during implementation and are
recorded here rather than left to the code:

- The outcome is three-valued, not two. `unverified` covers the cases where a
  correlation must not be interpreted — too few overlapping variants, too
  little frequency variance on either side, or an Analysis that stores no EAF —
  and is recorded per Analysis rather than passing quietly as `passed`.
- **The reference is checked before it is trusted.** A panel column that is
  minor allele frequency rather than allele-oriented EAF is symmetric about 0.5
  and would correlate a flipped source at r ≈ +1, certifying the exact defect
  this check exists to find. A reference reporting a frequency above 0.5 for
  only a negligible fraction of its variants is refused outright — a single
  rounded row must not be enough to get a MAF column accepted.
- **A build with no panel falls back to the consensus of the other Analyses**
  when there are at least three — any layout, not only Dense/Ragged. A
  consensus can establish that Analyses disagree but not which is right, so an
  inconsistent build fails rather than dropping an Analysis.
- **The correlation is computed per Analysis, over that Analysis's own
  variants.** Sampling the store's variant axis instead looks equivalent and is
  not: `GCST90199621` in the metabolome pilot covers 0.8% of the axis and drew
  89 of 20,000 axis-sampled sites against the EUR panel — too few to say
  anything — where 20,000 of its own variants are ample.

The evidence — reference identity and checksum, overlap `n`, observed `r`,
outcome — is persisted in `analyses.tsv` and `manifest.json`, because
standalone validation cannot depend on a panel that may no longer be available.

## Consequences

- **This is a breaking format change**, unlike ADR 0036. It needs a
  `format_version` bump and therefore depends on #112 settling the
  compatibility and migration policy first. Landed as `format_version` 1.0
  (#114, fixed-point `z`) and 2.0 (#116, residual-coded `eaf`). `0.1` and `1.0`
  stay readable and are never written again; neither can be *completed*, since
  completion writes into its source's encoding (ADR 0038 §4).
- **Existing stores must be rebuilt** to gain any of it. All four pilots need
  rebuilding regardless, since #83/#106/#109 were all build-time defects.
- **Arrays stop being independently interpretable.** An `int8` `eaf` plane is
  meaningless without `eaf_baseline`; a residual-coded `se` is meaningless
  without `eaf` and the per-Analysis fit coefficients. That is a real cost
  against a format whose stated goal is self-containment, and it is accepted
  deliberately, not chosen on byte count. It also argues for landing the three
  changes in order of increasing coupling — `z` first, `se` last.
- **Decode is no longer free.** Reads gain a vectorised transform and, for
  `se`, a second array plus per-Analysis coefficients. Negligible per query,
  but it removes the option of handing a caller a raw mmap'd slice.
- **`EafScope.VARIANT` stays reserved and unimplemented.** The measurements
  rule out collapsing per-Analysis EAF to one value per variant: no variant in
  either single-cohort pilot has identical EAF across analyses.
- **The missing-cell contract changes.** Store-format §15 currently *requires*
  paired Z/SE NaNs, which an integer Z array cannot express. §15 needs revising
  to define missingness in terms of each plane's declared codec, with
  validation checking that the Z sentinel and the SE NaN agree — the same
  invariant, expressed through the plan rather than through a dtype.
- **Derived indexes stay decoded.** Top-hit indexes and the Rho matrix hold
  their own small copies of Z/SE. They remain `float32` artifacts, explicitly
  rebuildable from the encoded planes, rather than being re-encoded too — the
  duplication is small and the alternative multiplies the number of decode
  sites for no saving.

## Correction note

A first draft of this ADR, and issues #114/#116 as originally filed, carried
three errors caught in review of #119:

1. **The `z` precision claim was wrong.** "p accurate to 1.02%" conflated an
   empirical figure on sampled data with a worst-case bound. At scale 1/512 the
   worst case is 4.8% at z = 47.8 and 6.4% at z = 64. The "4.45×" attributed to
   `float16` was the width of a quantisation bin, not a rounding error; the
   actual error for 47.8 is 1.82× and the worst case is 2.11×. The conclusion —
   fixed point is substantially better — survives; the numbers did not.
2. **The |z| ≤ 64 bound was unjustified**, and the survey above shows it is
   wrong: 568 genuine associations in `ukb-b` exceed it. Replaced by an
   overflow table.
3. **The decision tree was not executable**, and integer planes had no
   missing-value contract at all. Both are addressed above.
4. **The EAF byte table was a prediction, not a rebuilt-store measurement.**
   #117 rebuilt the pilots and found the original 0.14 / 0.52 B/cell plane
   figures were too optimistic by 1.5–4.0×; the ADR now quotes the rebuilt
   Store Release measurements and qualifies per-variant baseline/reference
   costs by cells per variant.
