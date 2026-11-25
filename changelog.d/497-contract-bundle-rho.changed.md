- **The Store Reader contract bundle carries a Rho Matrix on its Dense and Hybrid
  releases, and asks the Rho readers (issue #497).** `dense_observed`,
  `dense_completed`, `hybrid_observed` and `hybrid_completed` carry one, built by
  `build-dense-rho`, and two new releases, `dense_rho` (an interior Rho on a Dense release) and `hybrid_overflow_only`, hold a Hybrid's Rho
  Matrix with an Overflow-only Analysis, so `OverflowOnlyAnalysisError` is exercised.
  `opengwasdb.store.query_digests` gains a `rho` section: `rho_matrix`, `rho_row` and
  `rho` of the spread Analyses, each array digested and spelt out in `readable`, and a
  refusal answered as its error. `differences` now compares the `rho` section too. Bundle version 2; no answer of the bundle's version 1
  changed.
