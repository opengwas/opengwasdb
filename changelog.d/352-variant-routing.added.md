- **Variant Routing is one module, `opengwasdb.variants.routing` (issue #352).**
  `route_variants(sources, authority, key=...)` reads every source once (Pass
  1), places each coordinate on GRCh38 — through the hg19→hg38 lift
  (`LiftedUnion`) or a Variant Reference's source keys (`FromReference`) — and
  names each routed ALID (`RoutedNames`). `VariantRouting.table(axes)` is the
  fork-safe `RouteTable` a Pass 2 worker routes one Analysis's associations
  through: each axis's rows come back with the last occurrence of each index
  kept, the indices whose duplicates disagree named, and an optional `admit`
  callable judging each batch before it is matched. `RoutingKey`,
  `UnionOptions`, `Pass1Stats`, `AssociationBatch`, `RoutedColumn`, `Placed`,
  `Unmatched` and `last_wins` are published with it, and `ManifestRow.source`
  gives a manifest row's source as routing reads it.
