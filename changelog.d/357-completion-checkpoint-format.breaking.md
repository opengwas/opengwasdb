- **A resumed completion refuses block checkpoints an older package wrote
  (issue #357, ADR 0072).** Each LD block's checkpoint now records its format,
  2. An unmarked one may count and hold a fill of the variant the next block
  now owns, so resuming with it would put two fills on one cell. An interrupted
  `complete-dense` or `complete-ragged` from an older package must be
  restarted: delete its checkpoint directory, or rerun it with `--overwrite`.
  `complete-hybrid` keeps no checkpoint between runs.
