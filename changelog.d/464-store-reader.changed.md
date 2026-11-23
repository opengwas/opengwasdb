- **The Store Reader: every read the query facade makes now lives in one
  subpackage, `opengwasdb.store_reader`, which imports only numpy, zarr,
  numcodecs and the standard library (issue #464, epic #450).** The release
  envelope and format-version check, the encoding plan and decode, the decoded
  planes, the association readers, the Ragged CSR, By-Variant Index, Indexed
  Variant Subset, Rho and Top-Hit readers, the Analyses index, the Variant Axis,
  and the facade and result resolution moved there; builders, completion and
  validation import them from it. `import opengwasdb.store_reader` loads none of
  pandas, scipy, pysam, sqlite3 or threadpoolctl, and `import opengwasdb` no
  longer loads the builders (its names, and the `encoding`, `variants`, `index`,
  `store` and `layouts.dense` packages' re-exports, are imported on first use).
  `opengwasdb.query` keeps its public names and answers every query as before
  (`tests/test_store_reader_digests.py` pins the answers of the code before the
  move). `log10_p` and the Top-Hit `z_critical` are computed without scipy: within
  1e-15 of the exact value, and cutting every stored |z| where the scipy form did.
  The 0.2.x Store Variant Table's pysam and SQLite reads stay in
  `opengwasdb.variants.axis`, behind a `LegacyVariantTable` seam.
