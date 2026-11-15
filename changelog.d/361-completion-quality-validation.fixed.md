- `validate` now checks that each Analysis's `completion_quality` counts agree
  with the cells its completed Store actually holds (`completion.quality_counts`,
  issue #361). Lost or collapsed fills can no longer pass merely because some
  imputed cells survived. The rule counts imputed and missing panel cells on the
  existing association scan, without an additional plane read. Block ids alone
  do not record cell ownership, so the comparison is per Analysis, not per block.

- Hybrid completion now removes an Overflow observation that crosses onto the
  completed Dense axis from each owning block's imputed/missing quality counts,
  refreshing the imputed rollup and completion counts too. Previously the real
  observation replaced the cell but left a quality entry for the old attempt;
  the new validation rule exposed that stale count (issues #99, #307).
