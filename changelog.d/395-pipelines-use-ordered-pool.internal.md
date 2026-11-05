- **The ancestry and phenotype-SD pipelines run their workers through
  `opengwasdb.build.ordered_pool.ordered_map` (issue #395).** It takes one
  `shared` value, which a forked worker inherits rather than receiving a copy
  with each item, so each pipeline hands it its read-only state — the Ancestry
  Reference Panel with its gates, regions file and liftovers, or the frequency
  source an ADR-0029 tier reads — and neither keeps module-level worker state
  or a private copy of the fork-pool setup any more. Ordering stays in the
  module that owns it: `ordered_map` yields in input order, so both pipelines
  write the Analysis Catalogue and the phenotype-SD TSV in the order they
  always have, without either of them re-implementing that order. Both outputs
  are byte-identical to the previous implementation's, on the fixture at one
  and four workers and on real inputs.
