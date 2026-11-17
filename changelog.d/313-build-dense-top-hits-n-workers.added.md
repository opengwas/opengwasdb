- **`build-dense-top-hits --n-workers` sizes the scan's process pool, and a band
  is bounded by a decoded-band budget rather than a row count (issue #313).**
  `build_top_hit_indexes` has always taken `n_workers`, and the index it writes
  does not depend on it, but the command passed none, so a rebuild from the
  command line scanned the `z` plane in one process; the Store Release Build's
  planner still passes no worker count, which is opengwas/opengwasdb-stores#358's
  to change. OGS-00003's index (21,230,615 variants × 10 Analyses) took 46.5 s
  and 48.0 s at `--n-workers 1` against 7.3 s and 6.5 s at `--n-workers 32`,
  with the same 408,822 / 805,360 / 2,383,859 hits each time. A band was
  `max(z.chunks[0], 250_000)` rows whatever the release's width, which on
  OGS-00009 (2,514 Analyses) decodes about 32 GiB of plane: enough for a
  32-worker pool to exhaust the host's memory and be killed. The scan and
  validation's streamed reads now take their band from
  `opengwasdb.encoding.banding` and its one-GiB decoded-band budget, so
  OGS-00009 reads 4,000 rows per band while a narrow release keeps its
  250,000-row band. The band leaves the scan's speed where it was — 1,060 s at
  four workers against 1,083 s — and the 3× a wider pool gives (353 s at
  sixteen) is affordable only because a worker now holds one band of decoded
  cells, 0.75 GiB, instead of 31.6 GiB of plane. The scan's pool is also sized
  as every memory-bounded pool is (`encoding.memory.pool_workers`): a worker
  counts its band's decoded budget, the 512 MiB floor every forked worker holds,
  and the two harvests the parent may hold for it, at most every cell of the
  band; no more workers run than a quarter of `MemAvailable` holds beside the
  parent, or than there are bands. On OGS-00009 a worker counts 2.7 GB, so 32
  workers ask for 85 GB. The default stays 1.
