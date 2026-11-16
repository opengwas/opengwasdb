- **Five names this epic left without a caller are removed (issues #308, #315,
  #316, #318, #320).** `validation.scans.dense_bands` and `sequence_windows`
  went when a Dense pass started reading its bands through `GridSplit` (#315);
  `ReferenceAllele.declared` when an indel's ALID began to be formed from the
  declaration as a row is read (#316); `layouts.hybrid.unknown_keys.encode_keys`
  when Pass 2 started encoding a batch's fields directly with `encode_fields`
  (#318), and the tests keep it as the oracle `encode_fields` is held to; and
  `build.ordered_pool.ordered_map`'s `max_in_flight`, since every ordered pool
  now holds `RESULTS_HELD_PER_WORKER` (two) results a worker, the share its
  memory budget counts (`encoding.memory.pool_workers`). No caller exists in
  this repository's package code or in `opengwasdb-stores`.
