- **Every query result carries `beta` and `pval`, derived once by the Store
  Reader (issue #466, ADR 0104).** One function, `association_statistics`,
  turns a result's z and SE into `beta` (`z * se`), `pval` and `log10_p`, all
  `float64`. The `Associations` record that every query method returns exposes
  `beta` and `pval`, on Dense, Ragged and Hybrid releases alike. Resolved rows
  carry all three. A missing association is NaN in all three, and so is an
  infinite SE (ADR 0068), never `inf`. Where p is below float64's range,
  `pval` is 0.0 and `log10_p` stays exact. The CLI's TSV gains a `beta`
  column after `se`, so `p`, `eaf` and `association_status` each move one
  column right. Its `p` column prints `pval`, and prints `NA` for an infinite
  SE where it used to print the p of z. `--format json` gains `beta`, `pval`
  and `log10_p`. The benchmarks read beta and -log10 p from the Store Reader
  instead of deriving them again.
