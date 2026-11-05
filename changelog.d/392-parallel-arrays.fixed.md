- **Every writer entry that takes parallel arrays refuses arrays of unequal
  length (issue #392, ADR 0077).** `Placement.sequence` and `Placement.pairs`,
  `write_encoded_planes`, `write_dense_cells`, the exact-value side tables, the
  SE rewrite, the Dense VCF and Hybrid column spills, the Top-Hit Index's
  `write_index` and the inline writer that feeds it, the Ragged by-variant
  tables, the Rho Matrix and a Hybrid key run all took arrays they assumed
  describe the same cells. An array of another length was otherwise broadcast
  onto the cells — one cell's statistic stored under another's variant — or the
  planes were written at unequal lengths, or numpy refused it from inside a
  write, after the planes or a tier's group existed and naming no array. Each
  entry now refuses such a batch through `require_parallel`, before it writes
  anything, naming every array and its length. A scalar `analysis_index` in
  `Placement.sequence` still broadcasts by design, as do `Placement.band`'s
  grid and a table's entries against the component's cell count. No builder or
  test passes unequal arrays today, so every Store this package builds is
  unchanged.
