- **A Dense or Hybrid release answers `top_hits` at a stored tier only, and an
  unknown `analysis_id` raises (#483).** A Dense or Hybrid release returned no
  rows, without an error, for a `threshold` it stores no tier for, and every layout
  (Ragged too) returned nothing for an `analysis_id` it does not hold; a tier without per-Analysis offsets also
  returned nothing for one Analysis. Now an unstored threshold raises
  `opengwasdb.store_reader.top_hits.UnstoredThresholdError` (a `ValueError`)
  naming the threshold and the stored tiers, and `opengwasdb query-top-hits`
  exits 1 with that message; a Dense release never scans its grid, since a scan
  is hours on a release the size of OGS-00016. An unknown `analysis_id` raises
  `opengwasdb.store_reader.analyses.UnknownAnalysisError` (a `ValueError`) on every
  layout, naming it. A stored tier without offsets answers one Analysis by
  filtering its rows. A Ragged release still scans an unstored threshold. A
  Hybrid's `limit` is applied to each component. `analysis`, `phewas` and
  `range_by_analysis` are unchanged.
