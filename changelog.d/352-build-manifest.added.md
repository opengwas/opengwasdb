- **`opengwasdb.layouts.build_manifest` publishes the build manifest read
  (issue #352).** `read_manifest(path)` returns one `ManifestRow` per Analysis,
  each validated as it is read; `manifest_row_to_analysis` gives a row's
  Analytical and Attribution Metadata, and `column_map_provenance` the Column
  Map fingerprint a build records. The Dense and Hybrid builders and Variant
  Reference extraction read their manifests through it; the same functions were
  private to `layouts/dense/build_vcf.py` before.
