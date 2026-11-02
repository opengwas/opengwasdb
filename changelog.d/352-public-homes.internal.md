- **What the builders share that isn't Variant Routing gets a public name
  (issue #352).** The build manifest read moves to
  `opengwasdb.layouts.build_manifest` (`ManifestRow`, `read_manifest`,
  `manifest_row_to_analysis`, `column_map_provenance`), Variant Index ordering
  to `opengwasdb.variants.axis` (`sorted_alids`, `alid_sort_key`), and the
  fork-start pool to `opengwasdb.build.ordered_pool.fork_pool`. The Dense
  Component writers and Pass 2 helpers the Hybrid builder calls become public in
  `layouts/dense/build_vcf.py` without moving. No build output changes:
  `benchmarks/routing_fixture_trees.py`, new here, builds one fixture per
  routing path, and all 23 match the base byte for byte (1,704 files), with the
  base's checkpoints resuming to the same Stores.
