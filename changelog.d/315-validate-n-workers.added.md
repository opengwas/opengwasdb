- **`validate --n-workers N` spreads each Dense grid's band pass across forked
  workers (issue #315, ADR 0074).** The parent reads the first band, so what the
  folds load on their first block -- a Top-Hit Index's tiers -- is loaded once and
  inherited; the shared ordered fork pool reads the rest, one band a task, and the
  parent adds each band's fold outcomes and riding checks' parts in band order, so
  the report is the one a single process makes at any worker count. The EAF
  exception table is still read once a pass, by the parent, which hands each band
  only its span; the run's subset digest key is drawn there too and inherited by
  every worker. A worker opens everything it reads after the fork -- the planes and
  the side tables behind its codecs -- under the array seam's fused pipeline at one
  thread (ADR 0056), which is what keeps it from hanging: with that pipeline's thread
  pool on, a worker whose first read spans as many shards as the parent's did never
  returns. A failure of the pool itself -- a worker that dies, a fork the host
  refuses -- ends the run with an error naming the band whose result it was waiting
  for, not a Store finding. A worker holds one band at a time, which the decoded-band
  budget bounds (#254), as measured by its private memory, and the pool is no larger
  than a quarter of the host's available memory holds beside the parent
  (`encoding.memory.pool_workers`, #321). An association sequence's scan, and every
  check over the Store Variant Table, `analyses.tsv` and the envelope, still run in
  one process. `--n-workers 1`, the default, reads every band in process as before.
  On OGS-00009, at one worker and at 32 side by side, a whole validation took 3,857 s
  and 942 s and its band pass 2,904.6 s and 140.9 s, and the JSON report is
  byte-identical; a worker held at most 0.66 GiB of its own. `subsets.indexed`, which
  reads the index in one process, is now 757.8 s of the 942. OGS-00011's Dense
  Component's band pass took 274.9 s at 32 workers, against 14,497 s at one in #314's
  run, and the report is byte-identical to #314's.
