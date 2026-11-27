# Refuse an effect that contradicts its source's own p-value

The resolver reads a source's effect and standard error and counts the rows a
build would store. Nothing in that reading can tell a beta from an odds ratio:
both are finite numbers, and a column called `beta` that holds `exp(beta)` yields
a z-score (`beta / se`) that is wrong by a scale on every row and plausible on
none that is looked at alone. `GCST002047` writes odds ratios into `beta`;
`GCST004461` writes log-odds into `odds_ratio`. Both resolved, both would have
been built, and both stored statistics that contradict the file they came from
(opengwasdb-stores#403, epic #347).

The file already holds the witness. A GWAS-SSF row reports a p-value beside its
effect and standard error, computed by the study from the *right* effect. If the
z-score the build would store, `beta / se`, implies a two-sided p that is
orders of magnitude away from the p-value the file reports — on the typical row,
not on a few — then the effect, the standard error or the p-value is on a scale
the reader did not account for.

## Decision

**The resolver compares the stored z-score with the source p-value, on a bounded
sample, and refuses an Analysis whose implied p-values contradict the source's
systematically.** Owner decision, 1 Nov 2026: refuse, not flag.

**What is compared.** Rows the build would admit (after the INFO and MAF
admission rule of ADR 0054), with a usable effect in the form the build stores
it — after `log(odds_ratio)` or any other transform, so `z = stored beta / se`,
or the signed z itself for a z-score source (ADR 0051, where `beta / se` is the
source z by construction) — and a source p-value that is finite and in `(0, 1]`.
**Rows whose standard error was derived from that p-value are excluded** (ADR
0055): their z is the p-value's own quantile, so the comparison agrees by
construction and proves nothing. A standard error derived from a confidence
interval is independent of the p-value and is compared.

**The statistic.** Per row, `Δ = log10 p_implied − log10 p_source`, with
`log10 p_implied = (ln 2 + norm.logsf(|z|)) / ln 10`, computed in log space. The
naive `2 * sf(|z|)` underflows to `0` from |z| ≈ 38 and this project sees |z| of
137. The refusal statistic is the **median |Δ|** over the sample; the median
signed Δ is recorded beside it. A median, because a minority of rows whose
p-value comes from a different test than the Wald statistic (a saddlepoint, Firth
or likelihood-ratio p beside a Wald beta) must not decide a file's fate, while a
scale error moves every row; the clean sources' own medians were at most 0.35. No slope is fitted: the median separated every
case below without one.

**The sample.** The same deterministic bottom-`k`-by-`site_hash` sample as the
phenotype-SD evidence (ADR 0044), now one mechanism, `build.hash_sample`, with
`k = evidence_sample` (default 20,000). Memory is bounded by configuration, and
the record says when the bound dropped rows (`sampled`). A case-control
Analysis whose scan stops at the ancestry-site bound (ADR 0048, 0054) is
measured on the rows that scan read.

**Refusal is a controlled failure.** The Analysis's `AnalysisResolution.error`
is set, the Resolution Record's status is `controlled_failure`, and the error
names the measured median |Δ|, the median signed Δ, the row count and the
threshold. The measurement is recorded either way.

**Skips are recorded, never silent.** The check does not refuse, and records its
reason, when:

- `no_source_p_value` — no admitted row carries a usable source p-value: no
  p-value column, or none in `(0, 1]`. GWAS-VCF and BESD are in this class for
  now; reading VCF's `LP` (−log10 p) is deferred, and until it lands a VCF source
  is not checked. A column that is entirely blank is the same fact as no column.
- `all_se_from_p_value` — every row with a source p-value had its standard error
  derived from it.
- `too_few_rows` — fewer than the minimum (50) rows *compared*, that is, in the
  bounded sample the verdict is drawn from, not merely seen.

**The policy is a documented default a caller may override.**
`EffectConsistencyPolicy(max_median_abs_delta=1.0, min_compared_rows=50)` on
`AnalysisRequest`, and the keyword `effect_consistency_policy` on
`resolve_analyses_manifest`. Both numbers are bound into every record's
`resolution_config.effect_consistency` fingerprint, as `maf_threshold` is: a
record resolved under one policy is never resumed under another. A sample bound below `min_compared_rows` could never reach a verdict, so
`resolve_analysis` and `resolve_analyses_manifest` raise a `ValueError` naming
both numbers rather than skip every Analysis as `too_few_rows`. There is no CLI
flag; the resolver CLI does not pass resolver tolerances through.

## Why this amends ADR 0044

ADR 0044 said "the module computes; the caller decides", and that no tolerance,
verdict or `exclude_from_build` is produced here. That is right for a *scientific
tolerance*: whether an ancestry fit passes its gates, whether a phenotype-SD
dispersion is acceptable, whether an overlap is enough. Those depend on what the
registry wants from a Study, and the resolver reports and leaves them to the
caller.

This is not that kind of decision. A file whose effect column contradicts its own
p-value column is a **fact about the file**, of the same kind as a duplicated
effect column (ADR 0049, already a controlled failure) or a z-score for a
case-control Analysis (ADR 0051, already refused). No caller has a use for the
statistics such a file would yield, and none can tell them from sound ones: a
flag in a record is read by nobody until a store has been built and queried, and
the failure this project worries about most is a wrong answer that looks right.
The refusal is therefore the resolver's, the threshold is a documented
default rather than a tuning knob, and ADR 0044's other decisions stand.

## Calibration

Measured on OGS-00011 sources (the 4,783 Analyses of
`/data/opengwasdb/work/ogs00011-regen-20260929/OGS-00011/resolver/analyses.tsv`),
reading each source through the resolver's own reader, admitting rows with its
declared MAF threshold, and measuring the first 1,000,000 canonical rows
(random sample) or 200,000 (whole manifest) of each, so the row cap is the only
way the numbers differ from a whole-file resolve. The three named sources were also resolved whole through
`opengwasdb resolve-analyses`, from their real files, with the numbers shown.

| | median \|Δ\| |
|---|---|
| `GCST006867` (clean, named) | 0.00145 (full file, 5,052,917 comparable rows, 20,000 sampled) |
| random 200 sources: 151 comparable, 44 no p-value, 5 all-SE-from-p | median 1.4e-4, p90 3.6e-3, **p99 0.126, max 0.177** |
| all 4,783: 3,767 comparable, 963 no p-value, 53 all-SE-from-p, 1 other failure | 3,763 below 1: p99 0.181, **max 0.346** |
| 40 random FinnGen R13 endpoints (all comparable; first 1M rows each; SAIGE p) | p99 5.3e-7, max **5.5e-7**, none skipped or refused |
| `GCST004461` (log-odds in `odds_ratio`) | **247** (full file, 5,864,147 rows); 249 (1M rows), 231 (200k rows) |
| `GCST002047` (odds ratios in `beta`) | **1,732** (full file, 2,314,363 rows); 1,724 (1M rows), 1,585 (200k rows) |
| `GCST90503316` (odds ratios in `beta`; found by this check) | 813 |
| `GCST90165257` (log-scale effect in `odds_ratio`; found by this check) | 12.9 |

**Threshold 1.0.** It sits 2.9× above the largest clean source in the whole
manifest (0.346, `GCST90275159`), 5.6× above the largest in the random sample, and 12.9×
below the smallest source this check refuses (`GCST90165257`, 12.9; the two
named sources are 247 and 1,732 on the full files). The clean distribution is a continuum that
tails off at 0.35; the refused sources are separated from it by a factor of 37
with nothing between. 0 of the 151 comparable random sources, and 4 of 3,767 comparable
sources overall (0.1%), are refused, and the four are the two named
sources and two the check found: `GCST90503316` writes odds ratios into `beta`
(first rows `1.07643`, `1.03504`, `0.996064`, against p-values of 0.49, 0.87,
0.85), and `GCST90165257` writes a REGENIE log-scale effect (`0.0159663`,
`-0.0481493`) into `odds_ratio`, so only its positive rows are usable.

**Minimum 50 rows.** The smallest comparable source in the random sample has
5,160 rows, and in the whole manifest every comparable source has at least 100.
To choose a floor, the five clean sources with the largest median |Δ| and two
typical ones were resampled 2,000 times at 10 to 200 rows: the largest median
|Δ| any 50-row subsample produced was 0.298, 3.4× under the threshold, and at 10
rows 0.458. 50 keeps the worst noise well under the threshold while leaving a
verdict for any study with a few hundred variants.

The two sources found by this check were not known to be wrong before it ran; an
owner reading the stores should see them (opengwasdb-stores#403's class).

**FinnGen.** R13's p-values are SAIGE's, so clean FinnGen sources might have sat
further from a Wald z than GWAS Catalog ones. They do not: on 40 random endpoint
files (`/data/opengwasdb/raw/finngen-r13/releases/r13-full/source/`, 1,000,000
rows each) the largest median |Δ| is 5.5e-7, because FinnGen's published `pval`
agrees with `beta / sebeta` to the digits it prints. The threshold stands.

## Consequences

- **The resolver refuses Analyses it used to accept.** Four of OGS-00011's
  4,783 in the calibration above, and any future source of that kind. A refused
  Analysis has no ancestry assignment or phenotype SD in its record, and
  `resolve-analyses` reports it as a controlled failure like any other.
- **Every Resolution Record gains `diagnostics.effect_consistency` and
  `resolution_config.effect_consistency`**, and old records stop being
  resumable (their fingerprint digest changes), exactly as ADR 0054's additions
  did: `--resume` recomputes. The sibling-repo work (documenting the
  fields and moving the pin) is opengwas/opengwasdb-stores#413. `record_schema_version` stays `1` under that
  module's additive-field rule.
- **The built-in `finngen-r13` Column Map now maps `pval`** (`p_value_form = "p"`),
  which it had not: without it every FinnGen Analysis silently skipped the check.
  Mapping it changes the map's checksum, which is part of every Resolution Record
  read through it (ADR 0063), so FinnGen records are recomputed by `--resume`.
  It also gives the map ADR 0055's recovery, which amends that ADR's decision 5:
  a Column Map that declares `p_value` (now including `finngen-r13`) derives a
  standard error from it for a row whose `sebeta` is unusable, where the dedicated
  `FinnGenR13Reader` does not, so the row has no usable SE and Association
  Admission drops it (ADR 0072; such a row is excluded from the check). Measured
  on real data the difference is inert: 0 such rows in 40 random R13 endpoints at
  1,000,000 rows each (40,000,000 rows), and 0 in 5 of them read whole
  (106,632,171 rows).
- **ADR 0063's parity claim has one carve-out.** The built-in `finngen-r13` map
  and the dedicated `FinnGenR13Reader` no longer read every row identically: a
  row with an unusable `sebeta` and a usable `pval` is read by the map, and
  read by the reader with no usable SE, so Admission drops it. 0 of 146.6M real
  rows are affected; ADR 0063 carries the amendment, and
  `tests/test_column_map_reader.py` names the carve-out.
- **GWAS-VCF and BESD are not checked yet.** Their records say
  `skip_reason = no_source_p_value`. Reading VCF `LP` is deferred, as is any
  check for BESD, whose per-probe statistics carry no p-value in the metrics
  projection.
- **A source whose reported standard error is not the effect's** (a standard
  error from a different model than the effect) is refused too when it moves the
  implied p by an order of magnitude on the typical row. That is intended.
- **Rows with an SE derived from the p-value, and p-values of exactly `0`**
  (underflow in the source) are not compared. A source that reports only
  underflowed p-values for its strong rows is measured on the rest.
- **The p-value travels in the metrics projection.** `MetricsChunk.p_value` and
  `TabularMetricsRow.p_value` carry the finite source value or `NaN`/`None`, with
  no range rule, and the row-wise and blocked projections agree field for field.

## Addendum, 18 Nov 2026: a second refusal on the median signed Δ (#370)

**The gap.** A stored z that carries no signal gives p_implied ≈ 1, so Δ ≈
−log10 p_source, whose median over a null-dominated Analysis is log10 2 = 0.30.
The median |Δ| rule at 1.0 can never refuse it, and the measurement behind #370
(#357 D9) found eleven such sources in OGS-00011: `GCST90275159`–`66` (SPACox,
`beta` 9.0 and `se` 1000 on every row, stored z 0.009), `GCST90572740`/`41`
(`standard_error` holds exp(SE); both are in the built Store) and
`GCST90446466`. The signed median separates them because every clean source's
sits near 0.

**Decision.** `EffectConsistencyPolicy` gains `max_median_delta` (default
**0.1**). An Analysis is refused when the median |Δ| exceeds
`max_median_abs_delta` *or* the median signed Δ exceeds `max_median_delta`. The
|Δ| rule is tested first. Only the positive side is refused: a negative signed
median (an inflated z) is what the |Δ| rule already catches when it is large,
and the clean sources' negative tail reaches −0.34. The refusal message names
the statistic that fired and its value; the record's
`diagnostics.effect_consistency.refused_by` is `median_abs_delta` or
`median_delta`. Both numbers are in the fingerprint, so every record is
recomputed by `--resume`, as when the first rule landed.

**Calibration.** The real resolver (`opengwasdb resolve-analyses`,
`--evidence-sample 20000 --max-rows 1000000`, `--source-build GRCh38`) over all
4,783 OGS-00011 sources: each source's scan stops at 1,000,000 rows and the check
is drawn from the bottom-20,000 `site_hash` sample of those rows: the
resolver's bounded sample, but of each source's first 1,000,000 rows, not of the
whole file. (A whole-file run of all 4,783 sources would take about 80 hours; it
managed 34 sources in 35 minutes.) The prefix is confirmed on whole files below.
3,767 sources were compared; 962 had
`no_source_p_value`, 53 `all_se_from_p_value`, one failed on a duplicate `beta`
header. Records and table: `scratch/370/records/`, `scratch/370/calibration.json`.

| | n | median signed Δ |
|---|---|---|
| compared, not refused by either rule | 3,752 | max **+0.0146** (`GCST90128518`), p99 +0.0057, median ≈ 0 |
| refused by the signed rule: GCST90275159–66, GCST90572740/41 (the ten data-broken) | 10 | min **+0.2819** (`GCST90572741`), max +0.3324 |
| refused by the signed rule: `GCST90446466` | 1 | +0.2394 (see below) |
| refused by the \|Δ\| rule (unchanged) | 4 | \|Δ\| 12.0, 239.6, 793.7, 1,724 (signed −12.0 … −1,724) |

Threshold **0.1** is 6.8× above the largest clean median (0.0146), 2.4× below the
smallest flagged (0.2394) and 2.8× below the smallest of the ten data-broken
sources. No source that passed the |Δ| rule is newly refused except these eleven.
The whole-row figures in #357 (clean maximum +0.0144, flagged +0.243 to +0.336)
agree to within sampling noise.

**Whole-file confirmation (18 Nov 2026).** The 11 flagged sources and the 50 clean
sources with the highest prefix median were resolved over their whole files
(`--evidence-sample 20000`, no `--max-rows`; 0.8 to 10.4 million comparable rows
each, sampled to 20,000). Whole-file signed medians: the 50 clean sources' maximum
is **+0.0148** (`GCST90025960`; prefix +0.0092; the prefix maximum was +0.0146),
the 11 flagged sources' minimum is **+0.2522** (`GCST90446466`; prefix +0.2394)
and the ten data-broken sources lie in +0.2750 to +0.3413. No clean source is
refused and every flagged one is. The prefix and whole-file medians differ by at
most 0.0055 (clean) and 0.0128 (flagged). Evidence:
`scratch/370/whole-file-check/whole_vs_prefix.json`. The other 3,700 clean
sources were not read whole: their prefix medians are at most +0.0146 against the
threshold 0.1, a margin the observed prefix-to-whole change (0.0055) does not
approach.
The clean sources' median |Δ| is unchanged: the |Δ| rule's four refusals are the
same four.

**`GCST90446466` is refused by the signed rule and is not exempted.** Its stored
statistics are right; its `p_value` column holds 1 − p/2 (owner decision, 2 Nov
2026). The fault is the p column, so the remedy is to read the source through a
Column Map that does not declare `p_value`: the check then records
`no_source_p_value` and skips. Resolved so on the real file, it succeeds with
`skip_reason = no_source_p_value`. The change belongs to the Source Collection's
manifest and Column Map in `opengwasdb-stores`; the resolver names no Analysis.

## Rejected

- **Flag, do not refuse.** The first design, and the pattern ADR 0044 describes.
  A flag is a number in a record that nothing reads before a build; the case
  that prompted this was a store already built from two such files.
- **A mean or a tail statistic.** A mean of |Δ| is dominated by the few rows
  with the largest |z|, where a small relative difference between two test
  statistics is many orders of magnitude of p. The median is set by the typical
  row, which a scale error moves and a different test statistic on a minority of
  rows does not.
- **`2 * sf(|z|)`.** Underflows to `0` at the |z| this project sees.
- **Comparing p-derived standard errors.** Agrees by construction.
- **A CLI flag for the threshold.** The resolver CLI passes no tolerances
  through; adding the first one for a refusal would invite loosening it per run.

## Addendum, 27 Nov 2026: every build runs the check, and a stored-z audit (#510)

**The gap.** OGS-00032 and OGS-00033 were published with eleven Analyses whose z was
wrong by a scale: ten of PMID 40222985 (Omae) whose harmonised `beta` holds the odds
ratio beside a log-scale SE, and GCST90278111 (Kamiza), whose `beta` is about −5.6 × 10⁵.
They surfaced only because clumping them took hours. The check above would have refused
every one: run with the current resolver on each source's first 500,000 rows, their
median |Δ| is 24.5 to 34.0 (Omae) and 1.2 × 10¹² (Kamiza), against a threshold of 1.0.
It never ran on them. Their Resolution Records were written by opengwasdb 0.3.3
(`1f1e289a`, Manifest Generation on 5 Nov; `release.yaml` records
`resolver_opengwasdb_version: 0.3.3`), and the check first shipped in 0.4.0. The Stores were
built later by 0.20.0 and 0.26.0 from that frozen bundle, and no builder measured anything
of the kind. Of the published Stores, only OGS-00011 was resolved by an engine with the
check. OGS-00005 still holds `GCST002047`, the example this ADR names.

**Decision 1: the build runs the check.** Each builder hands every Analysis to
`verify_effect_consistency` beside its EAF orientation check, and the shape is that
check's (ADR 0037 §6, `build/eaf_orientation.py`). `check_effect_consistency` measures,
`enforce_effect_consistency` raises one `EffectConsistencyError` naming every Analysis
that contradicts its p-value and the rule that fired, and `apply_consistency_evidence`
stamps each Analysis's `analyses.tsv`. A refused Analysis is handled as a flipped EAF is:
the build stops before writing, and the registry excludes it by that reason.

- *What is read.* Each source's first `DEFAULT_BUILD_SCAN_ROWS` = 1,000,000 rows,
  through the resolver's own projection (`stream_metric_chunks`) and Association
  Admission (`admit_block`, which the resolver now shares), sampled bottom-20,000 by ALID
  hash. That prefix is what this ADR's calibration measured, and its whole-file
  confirmation found the medians within 0.0055 (clean) and 0.0128 (refused) of the whole
  file's. A read costs about 5 s per GWAS-SSF source (4.7 s and 6.3 s measured at load
  60), run `n_workers` at a time: about 11 minutes for OGS-00011's 3,245 Analyses at 24
  workers, beside a build of hours. Routing carries no p-value, so the check is its own
  bounded read rather than a passenger on Pass 2.
- *Where.* `build-hybrid` (with its EAF phase, recorded in the checkpoint as
  `effect_consistency.json`), `build-dense-vcf` (before Pass 2) and `build-ragged-ssf`
  (before anything is written). A Hybrid checkpoint written before this change holds no
  report, and its resume measures one. `build-ragged-besd` and `build-dense` stamp every
  Analysis `skipped: no_source_p_value` without reading, because BESD's metrics and the
  association TSV format carry no p-value. GWAS-VCF sources are read and come out the
  same way, since the readers read no `LP`.
- *The declaration.* A build manifest may carry `p_value_test` (ADR 0085). `joint_2df`
  skips the check (`joint_test_p_value`) and `mr_mega` is refused unread, as in the
  resolver. Unlike the resolver's, the column is optional: an Analysis that declares
  nothing is checked. Silence therefore cannot store a wrong z. A joint test left
  undeclared is refused loudly, and the refusal names the declaration.
- *What is recorded.* `effect_consistency` (`passed` or `skipped`),
  `effect_consistency_reason`, both medians and `effect_consistency_n`, in `analyses.tsv`
  (spec §7a, §9.2). No build writes `refused`. `format_version` does not move: the
  columns are additive, and a blank outcome is a Store built before this change.
  `effect_consistency.evidence` fails a recorded `refused` and evidence that contradicts
  itself. `effect_consistency.unchecked` warns, once per release, on the Analyses with no
  outcome.

**Measured on real sources.** Through the build's own path (`check_effect_consistency`,
first 1,000,000 rows, 12 workers, load 140), all twelve mis-scaled sources (the eleven
above and `GCST002047`) are refused, with median |Δ| ≥ 24.4. Twenty-six sound ones all
pass, with median |Δ| ≤ 0.123: 20 random OGS-00011 sources and the six height scans with
the most 5e-8 hits in the published Stores. The 38 sources took 27.4 s
(`scratch/510/build_check_real.jsonl`). Three more OGS-00011 sources, published only
below p = 0.007 or p = 0.00067, pass in a second run with median |Δ| ≤ 0.0045.

**Decision 2: a stored-z audit of Stores already built.** A published Store keeps no
source p-value, so this check cannot be repeated on one. It keeps every z, and in a
genome-wide scan few variants are at 5e-8. `associations.z_plausible` is a fold over the
association scan that counts, per Analysis, the observed associations and those at
5e-8, summed over a Hybrid's Dense Component and Overflow. It flags an Analysis with
more than 30% at 5e-8 among at least 1,000 observed. It judges only genome-wide releases, those whose manifest declares
`association_coverage` `gwas`, and it is an error from `format_version` 0.2.2 and a
warning before that. Ruling of 27 Nov 2026, option A of the measurement:

- *The gap.* The twelve mis-scaled Analyses have 81% or more of their sources' first
  500,000 rows at 5e-8. Sound ones have at most 8.2% in the 206-source sample
  (`scratch/510/signals_table.md`), and at most 9.7% in every published Store
  (GCST90245992, height, OGS-00011; `scratch/510/survey.jsonl`), with each Hybrid
  Analysis's counts summed over its components (`option_a_summed_verdicts.txt`). Over
  every published Store the rule flags GCST002047 in OGS-00005 alone. In the replaced
  OGS-00032 and OGS-00033 releases it flags exactly the eleven.
- *A share above |z| 3 was dropped, not kept as a warning.* It flags sound genome-wide
  sources published only below a p threshold: GCST90444395 (p ≤ 0.007, 55% above
  |z| 3, 0.7% at 5e-8) and GCST90271757 (p ≤ 0.00067, every row above |z| 3, 2.2% at
  5e-8). Both pass the build's check with median |Δ| ≤ 0.0045.
- *Not cis.* With the coverage gate removed, a strong eGene's window is mostly at 5e-8:
  the bound would flag 560 of OGS-00027's genes and 2 of OGS-00001's.
- *Per Analysis, not per component.* A Hybrid's Overflow holds its off-panel, mostly
  rare variants. Their larger SEs keep an odds ratio's z lower there, so the ten Omae
  Analyses' Overflows sit at 14% to 25% at 5e-8, under the bound, while their Dense
  Components sit at 86% to 91%. Judged per component, the 1,000-observation floor left
  a part too small to judge alone, so an Analysis with 999 Dense associations all at
  5e-8 beside a 1,500-row Overflow at 20% passed: the Dense part went unjudged and the
  Overflow sat under the bound. Summed, the same Analysis is at 52% and fails. The rule
  therefore sums each Analysis's counts over both components before it judges. The first
  component's pass leaves its counts on the release, and the second sums and judges.
  This closes the gap. (A share under 0.3 in both components is under 0.3 summed, since
  the sum is a weighted mean; the floor was the hole.)

**Consequences.**

- A build refuses sources it used to build. The error says `effect_consistency=refused`
  and names the rule, so the registry can exclude by reason.
- Every release built before this change warns `effect_consistency.unchecked`.
  `validate` fails OGS-00005 on `associations.z_plausible` until it is rebuilt without
  GCST002047, which the owner rules is excluded by reason through the registry.
- Test fixtures that wrote a p-value unrelated to their beta/se were refused once a build
  checked. The batch-parity generators now write the p their rows imply.
- The build manifest's optional `p_value_test` and the five `analyses.tsv` columns are
  sibling-repo documentation: opengwasdb-stores should pass its reviewed
  `p_value_tests` table through to the build manifest, and exclude by
  `effect_consistency`.
