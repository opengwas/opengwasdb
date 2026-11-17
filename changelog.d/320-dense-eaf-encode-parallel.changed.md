- **The Dense build's residual `eaf` encode runs its row bands across
  `--n-workers`, within a share of the host's memory (issue #320).** The
  transpose that turns the staged `float32` grid into the EAF Baseline, the
  `int8` residual plane and the exception table walked every row band in one
  process: 1 h 29 m on OGS-00009. Each band now encodes in a worker that reads
  the staged grid through the pool's `shared=` handle rather than a pickled
  copy, and the parent writes the bands back in row order. The three outputs
  are byte-identical at any worker count — every band owns its rows and its own
  run of flat positions, so the exception table is assembled in flat-position
  order and is neither re-sorted nor reordered by which worker finished first.
  A band's worker failure still leaves the staging plane in place, so a resume
  can tell the encode never finished.

  A band is at least one Dense shard, so on a wide release a worker holds a
  large fraction of a gigabyte, and the pool is therefore sized as every
  memory-bounded pool is (`encoding.memory.pool_workers`, #321): within a
  quarter of the host's `MemAvailable`, read when the phase starts, each worker
  counts one band and the 512 MiB every forked worker holds before its first
  (`FORK_FLOOR_BYTES`), the parent counts its own resident set and the two
  encoded bands it may hold for each worker, no more workers run than there are
  bands, and one worker runs, with a logged reason, when the host's memory
  cannot be read. The per-band figure is empirical —
  `benchmarks/measure_eaf_band_memory.py` fits a fixed cost plus a per-cell
  cost over two real OGS-00009 bands, from each forked worker's `VmHWM`
  (135 MiB + 66.9 B a cell, rounded up to 72; the artifact is
  `docs/benchmark-output/opengwasdb_320_eaf_band/eaf_band_memory.json`). A
  worker on OGS-00009's 251.4 M-cell band counts 19.7 GB, so a host with
  960 GB available runs 12 beside a parent of up to 4 GB, where the build's own
  32 would have asked for about 630 GB. On OGS-00003, whose five
  5,000,000-row bands keep at most five workers busy, the phase fell from 37–39 s
  to 11–14 s (2.4–3.5×). The encode's time at full size is measured in #322's
  end-to-end rebuild of OGS-00009, as #310's and #321's are.
