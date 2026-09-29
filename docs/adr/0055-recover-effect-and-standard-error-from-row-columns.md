# Recover a row's effect and standard error from its own columns

A full OGS-00011 resolve found 476 Analyses with no build-eligible row — no row
carrying a finite beta and a positive standard error. 212 of those report both
quantities in columns the reader never opened: a `beta` that is `NA` on every
row while `odds_ratio` is populated, an `odds_ratio` with a 95% interval and no
`standard_error`, or an effect with only a `p_value`. This ADR records the
per-row rule that reads them, where it is applied, and what it refuses to do.

## Context

ADR 0049–0051 resolved *which column* an Analysis's effect comes from. That is a
file-level fact. Whether a given row carries a usable value in that column is
not: a harmonised file can pair `beta` and `odds_ratio` as two spellings of the
same quantity and populate only one of them per row, and a precision can arrive
as a reported standard error, a 95% interval, or a p-value.

| shape | example | why no row was build-eligible |
|---|---|---|
| `beta` present but `NA` on every row; `odds_ratio` and `standard_error` populated | `GCST004030` | `resolve_effect_source` picks `beta` first (ADR 0049), so no row has an effect |
| `odds_ratio` with `ci_lower`/`ci_upper`; `standard_error` empty | `GCST90162552`, `GCST002318` | the effect is read, but there is no precision to build with |
| effect (`beta` or `odds_ratio`) with `p_value`; no SE, no CI | `GCST002598`, `GCST90013554`, `GCST90000016` | as above |

The quantities are all present in the file. On 400 randomly chosen real Analyses
that *do* report a standard error (the OGS-00011 survey), the interval-derived
standard error divided by the reported one has a median of 1.0000, and the
p-derived one is within 2% for 388 of 397 Analyses. The interval is on the log
scale for odds-ratio files and on the linear scale for beta files — six beta
files carried linear intervals, where the log formula is 12x to 76x off.

## Decision

1. **The effect is `beta` when it is finite, else `log(odds_ratio)` when the
   file carries that second column and the value is positive and finite.**
   Otherwise the row has no effect. A file naming one effect column behaves
   exactly as it did before: an unusable cell yields no effect, never the other
   column's value. `EffectSource.fallback_column_name` reports the second column
   so a caller can see the fallback happening; `column_name` and `kind` keep
   meaning the primary column.

2. **The standard error is the first usable of three, in order.** A reported
   `standard_error` that is positive and finite; else a 95% interval whose
   bounds are finite, ordered, and around the row's own effect; else
   ``|beta| / -Φ⁻¹(p / 2)`` for a two-sided p in `(0, 1)` and a non-zero beta.
   A derived standard error must itself be positive and finite.

   The interval is read on the scale of the column that supplied *that row's*
   effect, and the row's effect must lie inside it:

   * effect from `odds_ratio`: both bounds positive,
     ``se = (ln hi - ln lo) / (2 * Φ⁻¹(0.975))``, and `lo <= odds_ratio <= hi`;
   * effect from `beta`: ``se = (hi - lo) / (2 * Φ⁻¹(0.975))``, and
     `lo <= beta <= hi`.

   The guard is what refuses an interval presented on the other scale, and it
   refuses rather than converts: nothing in the file says which scale an
   interval is on, and the two formulas differ by a factor of 12 to 76 on the
   real files that carry one. `-Φ⁻¹(p / 2)` and not `Φ⁻¹(1 - p / 2)`, which
   rounds to infinity for a p small enough to be interesting.

3. **One rule, three callers.** `opengwasdb.readers.effect_source.row_statistics`
   is the rule; `gwas_ssf`'s dict-row parser, `tabular`'s row-wise projection and
   `tabular`'s blocked projection all call it or mirror it, and
   `tests/test_effect_recovery.py` asserts all three agree row for row and that
   the rows the association stream retains are exactly the rows the resolver
   counts `build_eligible_rows`. The blocked projection is what the resolver
   reads, so a divergence would be a divergence between what a build stores and
   what its record says it stored — the one comparison this stage exists to
   keep exact.

4. **The row's provenance is reported.** `RowStatistics` and `MetricsChunk`
   carry, per row, whether the effect came from the fallback column and whether
   the standard error came from the interval or the p-value, and the resolver's
   record counts the build-eligible rows that did:
   `build_eligible_rows_effect_from_odds_ratio_fallback`,
   `build_eligible_rows_se_from_ci`, `build_eligible_rows_se_from_p_value`.
   Without them a record could say an Analysis has 900,000 eligible rows and
   nothing about whether the reader read the file's own numbers or derived them.

5. **GWAS-SSF only.** The recovery columns are declared in the GWAS-SSF reader's
   `_METRICS_COLUMNS` and nowhere else, so FinnGen's and GWAS-VCF's effect
   handling is unchanged. A projection that declares no recovery column behaves
   exactly as before, whatever its header carries.

6. **The z-score path is untouched.** A signed z derives both statistics from
   the row's own EAF and sample size (ADR 0051); the interval and p-value
   fallbacks do not apply to it.

## Consequences

- **A derived standard error carries the file's own rounding.** The interval and
  p-value are rounded in the source, so the derived SE is an estimate of the
  reported one, not it. It is exact to the precision the file published, which
  is why the survey was run against Analyses whose SE *is* reported rather than
  only against the files being recovered.
- **A p-value can be a weak precision.** For a large effect and a coarse p-value
  the derived SE is imprecise, and a `p_value` column rounded to two significant
  figures propagates that rounding. It is still far better than no precision at
  all, and the record says where the number came from.
- **A wrong-scale interval costs a row its precision.** It is refused rather than
  converted; on a file whose intervals are on the other scale, rows with no
  reported `standard_error` stay ineligible, which is the honest outcome.
- **`neg_log_10_p_value` is deliberately not read.** A file reporting it instead
  of `p_value` is a further shape, and reading a `-log10 p` column is a separate
  decision (it needs its own rounding and underflow handling).
- **Three counts were added to the resolver record** and none to its schema
  version: they are ordinary integers, zero when not applicable, and a reader of
  an older record sees them absent rather than wrong.
- **The frequency rule of ADR 0036 is unrelated but adjacent**, and this work also
  makes exactly `0.0` and exactly `1.0` missing there; see that ADR's amendment.
