- `validate` now compares `completion_quality` counts with stored cells for each
  Analysis having zero or one quality row (`completion.quality_counts`, issues
  #361, #307). Lost fills in these Analyses are detected even when other fills
  survive. With zero rows, stored ancestry-exclusion evidence exempts only the
  missing-cell comparison; the imputed count must still be zero. Analyses with
  multiple rows are not reconciled: persisted metadata cannot distinguish
  distinct attempts from supported overlapping fills collapsed or dropped under
  ADR 0072. Their lost fills and per-block count transfers remain undetectable by
  this rule. Counts use the existing association scan without another plane read.

- Hybrid completion now removes an Overflow observation that crosses onto the
  completed Dense axis from each owning block's imputed/missing quality counts,
  refreshing the imputed rollup and completion counts too. Previously the real
  observation replaced the cell but left a quality entry for the old attempt;
  the new validation rule exposed that stale count (issues #99, #307).
