- **`opengwasdb.variants.routing.place_sites` is the one lift (issue #352, ADR
  0071).** `place_sites({"hg38": sites, "hg19": sites}, chain_file=...,
  failure_threshold=...)` returns a `SitePlacement`: each placed site's GRCh38
  ALID, and the sites a reverse-strand chain block re-oriented, whose z and EAF
  must follow the projected alleles. An hg38 site is its own ALID; the failure
  threshold applies to the hg19 group alone, and a site given under both
  assemblies is refused, naming it, rather than placed once. Pass 1's union and
  the Hybrid fold's off-reference keys are now both placed through it, so they
  share one orientation rule; the fold still refuses a re-oriented key, naming
  the first. The routing package also publishes the extraction names that moved
  into it, `extract_variant_reference`, `VariantReferenceExtraction` and
  `ReverseStrandLiftError`, and a new one, `EmptyReferenceError`, which
  extraction raises for sources holding no variant.
