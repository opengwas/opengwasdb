# Resolver effect-consistency check (ADR 0069, issue #345)

`resolve-analyses` compares the z-score a build would store with the source's
own p-value, and **fails the Analysis** when the two contradict each other
systematically. The decision and its calibration are ADR 0069; this is the
contract.

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
(default 20,000). The statistic is the **median |Δ|** over the sample.

## Verdicts

| | |
|---|---|
| refused | median \|Δ\| `>` `max_median_abs_delta` (default **1.0**): the resolution fails, the record's `status` is `controlled_failure`, and `error` names the median \|Δ\|, the median signed Δ, the row count and the threshold. |
| compared | otherwise; the measurement is recorded. |
| skipped | `skip_reason` says why, and nothing is refused: `no_source_p_value` (no admitted row has a usable p-value: no column, none in `(0, 1]`, or a reader that carries none: GWAS-VCF and BESD), `all_se_from_p_value`, or `too_few_rows` (fewer than `min_compared_rows`, default **50**, rows *compared*: in the bounded sample). |

A caller overrides both with `EffectConsistencyPolicy` (`AnalysisRequest
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
complete (an opening or read failure, such as a missing path: `error` says which).
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
| `skip_reason` | `null` when compared, else one of the three reasons above |
| `refused` | whether the statistic exceeded the threshold |

`fingerprints.resolution_config.effect_consistency` is `{"max_median_abs_delta",
"min_compared_rows"}`, the policy the verdict was drawn under.

The p-value reaches the check through the metrics projection:
`MetricsChunk.p_value` / `TabularMetricsRow.p_value`, the finite source value or
`NaN`/`None`, with no range rule, identical in the row-wise and blocked
projections. Column-map collections supply it through their `p_value` column (`gwas-ssf`: `p_value`; `finngen-r13`: `pval`).
