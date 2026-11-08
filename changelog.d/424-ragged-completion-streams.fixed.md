- **A genome-wide `complete-ragged` now fits in memory (issue #424).** Phase 3
  (merge, assembly and write) held the whole completed Store as Python objects,
  so OGS-00006's 10.0B fills would have needed about 1.7 TB; its first run was
  stopped at 178 GB with 11% of the fills read. Phase 3 now streams through code
  Dense completion and the Ragged builders already use: Dense's on-disk fill
  shards (moved to `opengwasdb.completion.fill_shards`, keyed by Analysis batch
  for Ragged), one implementation of the duplicate-fill rule (D5), and
  `RaggedCSRWriter`, which now also writes the `imputed` plane, `eaf_reference`
  and observed SEs carried exactly under the source's model (#301). What a
  completion writes is unchanged, array for array, and the checkpoint format
  does not change, with one fix: an Analysis completion carries through
  unchanged is now written in the completed axis's variant order. Where its
  source orders two variants at one position the other way round, its rows were
  written unsorted -- 265 descending pairs in 176 of OGS-00006's Analyses, a
  release `validate` refuses (spec §11) and whose lookups binary-search to a
  wrong row (#252).
