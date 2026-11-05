- **`build-ragged-ssf` builds only sources declared GRCh38 (issue #56, ADR
  0076).** It used to ignore `source_genome_build`. A manifest row that
  declares no build, or one other than GRCh38, is now refused, naming the
  Analysis: a Ragged SSF Store is one build, and the build lifts nothing. No
  published manifest omits the build, but `opengwasdb-stores`' own test
  fixtures do (#429).
