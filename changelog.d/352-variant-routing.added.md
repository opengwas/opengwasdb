- **Variant Routing is one module, `opengwasdb.variants.routing` (issue #352).**
  `route_variants(sources, authority, key=...)` reads every source once (Pass
  1), places each coordinate on GRCh38 — through the hg19→hg38 lift
  (`LiftedUnion`) or a Variant Reference's source keys (`FromReference`) — and
  names each routed ALID (`RoutedNames`). `VariantRouting.table(axes)` is the
  fork-safe `RouteTable` a Pass 2 worker routes one Analysis's associations
  through: each axis's rows come back with the last occurrence of each index
  kept, the indices and unmatched keys whose repeats disagree are named, and an
  optional `admit` callable judges each batch, returning a boolean keep mask,
  before it is matched. `RoutingKey`, `UnionOptions`, `Pass1Stats`,
  `AssociationBatch`, `RoutedColumn`, `Placed`, `Unmatched` and `last_wins`
  are published with it, and `ManifestRow.source` gives a manifest row's source
  as routing reads it.
- **`opengwasdb.readers.registry.SourceSpec` and `open_source(spec)` (issue
  #352).** A `SourceSpec` is one Analysis's source — its path, Source Reader
  capability, assembly, Stored Effect Scale and Column Map — and `open_source`
  resolves its `SourceReader` exactly as `resolve_reader` does.
- **`opengwasdb.build.ordered_pool.fork_each(fn, shared, items, n_workers)`
  (issue #352)** yields `fn(shared, item)` for each item in completion order.
  Forked workers inherit `shared`, so a large read-only value such as a route
  table is never pickled; with one worker the calls run in this process, in
  input order.
