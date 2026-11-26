- **`build-dense-rho` builds a Hybrid release's Rho Matrix, in its Dense Component
  (issue #496).** Give it the Hybrid release: it estimates the matrix over the Dense
  Component's grid (the shared Variant Axis from 0.3.0, the component's own table on 0.2)
  from the associations the Association Reader reads as observed, and writes
  `dense/data.zarr/rho`. The Dense Component is refused with the release's name, however
  its path is spelt (a symlink included), and so is a Ragged release. The group records
  exactly the Analyses with no observed association in the component, its Overflow-only
  Analyses, as `overflow_only`.
  On a Hybrid, `rho()`, `rho_row()` and `rho_matrix()` answer for the others and raise
  `OverflowOnlyAnalysisError` for a pair naming one, never NaN. A new rule,
  `rho.overflow_only`, requires the record to equal what the association scan finds, and
  the existing Rho rules judge the component's matrix.
