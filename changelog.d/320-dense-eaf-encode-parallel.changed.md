- **The Dense build's residual `eaf` encode runs its row bands across
  `--n-workers` (issue #320).** The transpose that turns the staged `float32`
  grid into the EAF Baseline, the `int8` residual plane and the exception table
  walked every row band in one process: 1 h 29 m on OGS-00009. Each band now
  encodes in a worker that reads the staged grid through the pool's `shared=`
  handle rather than a pickled copy, and the parent writes the bands back in
  row order. The three outputs are byte-identical at any worker count — every
  band owns its rows and its own run of flat positions, so the exception table
  is assembled in flat-position order and is neither re-sorted nor reordered by
  which worker finished first. A band's worker failure still leaves the staging
  plane in place, so a resume can tell the encode never finished.
