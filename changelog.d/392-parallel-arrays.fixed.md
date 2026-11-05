- **Every writer entry that takes parallel arrays refuses arrays of unequal
  length (issue #392, ADR 0077).** `Placement.sequence`, `write_encoded_planes`,
  `write_dense_cells`, the Top-Hit Index's `write_index`, and the Dense build's
  inline index writer and column spill took arrays they assumed describe the
  same cells. `Placement.sequence` now checks its variant index against its
  Analysis index, and both against the batch's statistics; the rest check a
  batch's z, SE, EAF and imputed mask, a cell list's rows and columns, and every
  candidate column. An array of another length was otherwise broadcast onto the
  cells — one cell's statistic stored under another's variant — or the planes
  were written at unequal lengths, or numpy refused it from inside a write,
  after the planes or the index group existed and naming no array. Each entry
  now refuses such a batch through `require_parallel`, before it writes
  anything, naming every array and its length. A scalar `analysis_index` in
  `Placement.sequence` still broadcasts by design. No builder or test passes
  unequal arrays today, so every Store this package builds is unchanged.
