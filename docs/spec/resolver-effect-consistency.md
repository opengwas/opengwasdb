# Resolver effect-consistency check (ADR 0069, issue #345)

`resolve-analyses` compares the z-score a build would store with the source's
own p-value, and **fails the Analysis** when the two contradict each other
systematically. The decision and its calibration are ADR 0069; this is the
contract.

Which disagreements this check refuses, and which it passes and the Store then
keeps, are listed in [statistic-disagreement.md](../statistic-disagreement.md),
each with the test that pins it.

## The declared p-value test (#371, ADR 0085)

Every row of a resolver manifest declares `p_value_test`: which test the
source's `p_value` comes from. There is no default. A manifest without the
column is refused when it is read, and so is a row whose value is blank or not
one of these three, naming its Analysis.

| `p_value_test` | The resolver |
|---|---|
| `wald_1df` | runs this check |
| `joint_2df` | reads the source and skips the check: `skip_reason = joint_test_p_value`. The p tests the stored coefficient together with a main effect, so it cannot witness beta/se |
| `mr_mega` | refuses the Analysis before reading it: `status = controlled_failure`, `error` begins `p_value_test is mr_mega`, `diagnostics.effect_consistency` is `null` and `rows_read` is 0. MR-MEGA's beta/se is a t on (cohorts − 2) df, not a z |

The declaration is trusted and is not checked against the file.
`fingerprints.resolution_config.p_value_test` records it, so a changed
declaration is never resumed.

## What is measured

On the rows the build would admit (the INFO and MAF rule of ADR 0054) that have
a usable effect and standard error, a source p-value that is finite and in
`(0, 1]`, and a standard error **not** derived from that p-value (ADR 0055),
per row:

```
z      = stored beta / se          (the source's signed z, for a z-score source)
Δ      = log10 p_implied − log10 p_source
log10 p_implied = (ln 2 + norm.logsf(|z|)) / ln 10      # log space, never 2*sf(|z|)
```

`stored beta` is the effect in the form the build stores it, after
`log(odds_ratio)` or the z-score derivation. The rows are a deterministic
bottom-`k`-by-`site_hash` sample, `k` = the resolver's `evidence_sample`
(default 20,000). Two statistics are taken over the sample: the **median |Δ|** and the
**median signed Δ** (#370, ADR 0069 addendum). The second exists because a stored z
that carries no signal (SPACox's `beta` 9.0 beside `se` 1000, or a squared-up
standard error) gives Δ ≈ -log10 p_source, whose null median is log10 2 = 0.30:
the |Δ| rule cannot refuse it, and every clean source's signed median is near 0.

## Verdicts

| | |
|---|---|
| refused | either rule fires: median \|Δ\| `>` `max_median_abs_delta` (default **1.0**), or median signed Δ `>` `max_median_delta` (default **0.1**, #370). The \|Δ\| rule is tested first. The resolution fails, the record's `status` is `controlled_failure`, `refused_by` names the rule, and `error` names the statistic that fired, its value, the other median, the row count and the threshold. |
| compared | otherwise; the measurement is recorded. |
| skipped | `skip_reason` says why, and nothing is refused: `joint_test_p_value` (the Analysis declares `p_value_test = joint_2df`; checked first), `no_source_p_value` (no admitted row has a usable p-value: no column, none in `(0, 1]`, or a reader that carries none: GWAS-VCF and BESD), `all_se_from_p_value`, or `too_few_rows` (fewer than `min_compared_rows`, default **50**, rows *compared*: in the bounded sample). |

A caller overrides all three numbers with `EffectConsistencyPolicy` (`AnalysisRequest
.effect_consistency_policy`; `resolve_analyses_manifest(effect_consistency_policy=)`).
An `evidence_sample` below `min_compared_rows` raises `ValueError` (it could never reach a verdict). There is no CLI flag.

## Calibration (summary; detail in ADR 0069)

Median |Δ| of clean sources, OGS-00011 (first 1M rows each; 200k for the whole manifest):

| | max | p99 |
|---|---|---|
| random 200 sources (151 comparable) | 0.177 | 0.126 |
| whole manifest (3,767 comparable; 3,763 below 1) | 0.346 | 0.181 |
| 40 FinnGen R13 endpoints | 5.5e-7 | 5.3e-7 |

Refused sources: `GCST002047` 1,732, `GCST004461` 247, `GCST90503316` 813,
`GCST90165257` 12.9. The threshold **1.0** is 2.9× above the largest clean source
in the manifest (0.346) and 12.9× below the smallest refused (12.9). The 50-row
minimum: the worst 50-row resample of a clean source gave 0.298.

## Resolution Record fields

Additive under `record_schema_version` 1. Records resolved before this check have
a different fingerprint digest and are recomputed by `--resume`. Documenting the
fields and moving the pin in `opengwasdb-stores` is opengwas/opengwasdb-stores#413.

`diagnostics.effect_consistency` is `null` only for an Analysis whose scan did not
complete (an opening or read failure, such as a missing path: `error` says which),
or that its `p_value_test = mr_mega` declaration refused before any scan.
That is *no measurement*, and is not the same as a skip: a `skip_reason` is a
recorded fact about a source that was read (no p-value column, every standard
error derived from the p-value, too few rows). A source never read is never
`no_source_p_value`. Otherwise an object:

| field | meaning |
|---|---|
| `n_rows_considered` | comparable rows the scan saw (before the sample bound) |
| `n_rows_compared` | rows in the bounded sample the medians are over; `0` for a skip |
| `sampled` | whether the bound dropped comparable rows |
| `median_abs_delta` | the refusal statistic; `null` for a skip |
| `median_delta` | median signed Δ (negative: implied p smaller than reported, i.e. the stored z is too large); `null` for a skip |
| `skip_reason` | `null` when compared, else one of the four reasons above |
| `refused` | whether either statistic exceeded its threshold |
| `refused_by` | `"median_abs_delta"` or `"median_delta"` when refused, else `null` |

`fingerprints.resolution_config.effect_consistency` is `{"max_median_abs_delta",
"max_median_delta", "min_compared_rows"}`, the policy the verdict was drawn under.

The p-value reaches the check through the metrics projection:
`MetricsChunk.p_value` / `TabularMetricsRow.p_value`, the finite source value or
`NaN`/`None`, with no range rule, identical in the row-wise and blocked
projections. Column-map collections supply it through their `p_value` column (`gwas-ssf`: `p_value`; `finngen-r13`: `pval`).
The FinnGen reader (`opengwasdb.finngen-r13`) reads `pval` too, for the check
only: it never derives a standard error from it.
