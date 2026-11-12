- **Reference Completion writes its planes across `--n-workers`, one pass per
  unit, within a quarter of the host's memory (issue #321, ADR 0080).** After
  the LD blocks, completion wrote in one process. Dense read its source grid
  three times: a statistic pass, an EAF pass that re-read the frequencies cell
  by cell, and a rewrite that read a `float32` SE scratch plane twice more.
  Ragged completed and flushed its Analyses one at a time.

  Now a Dense row band is read once, its fills land, and its z, EAF and SE are
  coded and written in one pass by a worker that owns the band's shards. No
  `float32` SE plane is written to disk. Ragged completes its Analyses on the
  same pool and writes its sequence one region of whole shards at a time, each
  region by its own worker. The checkpoint merge resolves its checkpoints on
  the pool too.

  The parent takes every unit's counts and side-table entries in unit order,
  so the release is the same at any worker count. Its decoded values, array
  metadata and on-disk sizes, `analyses.tsv`, `variants.tsv.gz`,
  `completion_quality` rows and manifest are all unchanged. The pool runs
  `--n-workers` workers, fewer when a quarter of `MemAvailable` holds fewer
  units beside the parent, and logs how many. A Dense band is budgeted at
  4 GiB + 24 bytes a cell. On OGS-00009 the budget is 16.4 GB, and a band
  worker held at most 10.2 GB private. Resume still keeps the LD blocks and
  redoes the write. `--n-workers`' help now names every pool it sizes.

  On OGS-00009, #309's profiler ran 2 of 56 bands with real fills, base and
  head side by side, twice. On one worker the one pass alone brings the band
  write, scaled to the full run, from 19,807 s and 19,422 s down to 10,722 s
  and 9,404 s. Over 8 bands, head's write on 4 workers scales to 3,358 s
  (56 min), against base's 20,485 s and #309's 21,179 s (5.9 h).

  On the OGS-00003 pilot the band write falls from 372 s and 359 s to 78–82 s
  on one worker and 12.8 s on 16. On the OGS-00006 100-Analysis subset,
  Ragged's assembly falls from 103–104 s to 12.5 s on 16 workers, and its
  flush from 30 s to 14–15 s on one worker (18–19 s on 16, where its two
  regions compress single-threaded). Peak scratch disk on OGS-00003 falls
  from 3.53 GB to 2.81–2.83 GB, against a 2.61 GB release; the checkpoint and
  fill-shard tiers #200 targets do not change. Base and head, at 1 and 16
  workers, are the same release by the release comparison tool. The
  full-size measurement is #322's end-to-end rebuild.
