- **Variant Reference extraction moves into Variant Routing (issue #352, ADR
  0071).** Three names are now `opengwasdb.variants.routing`'s.
  `opengwasdb.variants.reference` no longer exports `extract_variant_reference`,
  `VariantReferenceExtraction` or `ReverseStrandLiftError`, and
  `opengwasdb.variants` no longer exports the first two; it never exported
  `ReverseStrandLiftError`. `extract_variant_reference` takes the sources a
  manifest's rows name (`[row.source for row in read_manifest(path)]`), not the
  manifest, since reading a manifest is the build manifest's job;
  `liftover_failure_threshold` is now `failure_threshold`, and the window, batch
  and spill settings come together as `union=UnionOptions(...)`. The artifact it
  writes is unchanged, as is the `extract-variant-reference` command, which
  still names the manifest when it has no rows or its sources hold no variant.
  Called directly, extraction raises `EmptyReferenceError` for sources holding
  no variant, before writing anything. `opengwasdb.variants.reference` keeps the
  artifact format and the rsid oracles, and no longer imports a manifest reader
  or routing, so the import cycle with the Source Readers is gone.
