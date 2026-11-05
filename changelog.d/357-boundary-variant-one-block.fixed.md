- **Reference Completion fills a variant two LD blocks share with one block,
  whichever order the blocks run in (issue #357, ADR 0072).** Every LD panel
  completion reads puts each block-boundary variant in both adjacent blocks,
  so both imputed it. `complete-dense`, and so `complete-hybrid`'s Dense
  Component, stored whichever fill it wrote last and counted both. Blocks are
  now half-open: the block that starts at the variant owns it, fills it, and
  counts it once in `completion_quality` and `completion_n_imputed_total`.
  `complete-ragged` does the same for an Analysis completed against both
  blocks. A fill that still reaches a filled cell is resolved by the duplicate
  rule and logged. The published completed Stores are not changed here (#396).
