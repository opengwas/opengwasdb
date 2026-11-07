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
