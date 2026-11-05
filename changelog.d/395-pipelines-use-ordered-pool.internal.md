- **The ancestry and phenotype-SD pipelines run their workers through
  `opengwasdb.build.ordered_pool.fork_each` (issue #395).** Each hands the pool
  one shared value — the Ancestry Reference Panel with its gates and regions
  file, or the frequency source an ADR-0029 tier reads — which a forked worker
  inherits, so neither keeps module-level worker state or a private copy of the
  fork-pool setup any more. `fork_each` yields in completion order, so both put
  their results back in input order before anything is written: the Analysis
  Catalogue and the phenotype-SD TSV are byte-identical to the previous
  implementation's, on the fixture at one and four workers and on real inputs.
