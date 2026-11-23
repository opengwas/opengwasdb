- **The query TSV gains a `beta` column after `se`, so `p`, `eaf`, `n` (where
  present) and `association_status` each move one column right (issue #466,
  ADR 0104).** Default columns are now `analysis_id, analysis_label, chromosome,
  position, alid, effect_allele, other_allele, z, se, beta, p, eaf,
  association_status`: `p` moves from column 10 to 11, `eaf` from 11 to 12 and
  `association_status` from 12 to 13, one further right with `--variant-info`.
  A reader that selects columns by position must move with them. Every query
  result now carries `beta` and `pval`, derived once by the Store Reader:
  `association_statistics` turns a result's z and SE into `beta` (`z * se`),
  `pval` and `log10_p`, all `float64`, on Dense, Ragged and Hybrid alike, and
  resolved rows carry all three. A missing association is NaN in all three,
  and so is an infinite SE (ADR 0068), never `inf`; where p is below float64's
  range `pval` is 0.0 and `log10_p` stays exact. The TSV `p` column prints the
  reader's `pval`, and `NA` for an infinite SE where it used to print the p of
  z. `--format json` gains `beta`, `pval` and `log10_p`. The benchmarks read
  beta and -log10 p from the Store Reader instead of deriving them again.
