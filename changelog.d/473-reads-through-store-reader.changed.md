- **The validator, Reference Completion, the top-hit harvest, `overview.html`
  and the Rho builder read Store Releases only through the Store Reader (issue
  #473, epic #450).** None opens a release's Zarr group, its Store Variant
  Table or its `index.sqlite` itself any more: each asks the Store Reader by
  what it wants (`opengwasdb.store_reader.stored`: a release's arrays, its
  Association Sequence group, the `eaf_baseline` and `se_coefficients` a
  source was coded against, a group's recorded attributes; the Top-Hit
  Index's per-Analysis count, `analysis_hit_counts`; a release's Variant Axis,
  which now also streams its rows, `VariantAxis.records`, and resolves source
  ALIDs, `VariantAxis.indices_by_source_alid`). The `index.sqlite` reads the
  validator makes are named reads in `opengwasdb.index.release_index`, beside
  the 0.2.x Variant Table's, since the Store Reader may not import sqlite3;
  every read of a release's `index.sqlite` now opens it read-only
  (`open_index_read_only`), so no read can create an index
  that is not there. `tests/test_engine_reads_through_store_reader.py` refuses
  any other raw read of a release, naming each exempt writer and its reason.
  Every validation finding, observed-view verdict, EAF audit and completed
  array on the fixture releases is what it was (`tests/test_engine_read_pins.py`,
  pinned by the code before the change, run with `OGDB_MIGRATION_PINS=1`).
  Hybrid Reference Completion now refuses a residual-SE Dense Component that
  holds no `se_coefficients` with a `HybridCompletionError` naming it, where it
  raised a bare `KeyError`.
