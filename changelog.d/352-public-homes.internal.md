- **What the builders share that isn't Variant Routing gets a public name
  (issue #352).** Variant Index ordering moves to `opengwasdb.variants.axis`
  (`sorted_alids`, `alid_sort_key`), and the fork-start pool to
  `opengwasdb.build.ordered_pool.fork_pool`. The Dense Component writers and the
  Pass 2 helpers the Hybrid builder calls become public in
  `layouts/dense/build_vcf.py` without moving. No build output changes.
  `benchmarks/routing_fixture_trees.py`, new here, builds one fixture per
  routing path and compares two builds file by file. The 26 cases the base
  6415be4 can build match it in every byte, apart from each manifest's
  `created_at` and the Variant Reference artifacts, which match once
  decompressed. Extraction's two reverse-strand refusals match 98a6e7a, where
  #354 added them. The base's checkpoints resume to the same Stores; a resume
  after Pass 2 rewrites the Dense Component's SQLite index, which matches by
  its dump.
