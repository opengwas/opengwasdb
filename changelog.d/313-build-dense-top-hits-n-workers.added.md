- **`build-dense-top-hits --n-workers` sizes the scan's process pool, as the
  other `--n-workers` commands do (issue #313).** `build_top_hit_indexes` has
  always taken `n_workers`, and the index it writes does not depend on it, but
  the command passed none, so a rebuild from the command line scanned the `z`
  plane in one process. Rebuilding OGS-00003's index (21,230,615 variants × 10
  Analyses) took 46.5 s and 48.0 s at `--n-workers 1` against 7.3 s and 6.5 s at
  `--n-workers 32`, with the same 408,822 / 805,360 / 2,383,859 hits each time.
  The default stays 1.
