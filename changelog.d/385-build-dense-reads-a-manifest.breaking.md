- **`build-dense` takes a build manifest and one association TSV per Analysis (issue #385, ADR 0076).**
  `opengwasdb build-dense OUTPUT --manifest MANIFEST.tsv` replaces the positional
  source TSV, whose every row repeated its Analysis's id, label, Trait Ontology,
  attribution, effect scale and MAF filter. Those are now the manifest's
  columns, read as `build-dense-vcf` reads them; each row's `source_file` is that
  Analysis's association TSV, which holds associations only and is read by the
  registered Source Reader `opengwasdb.association-tsv`. A TSV header that still
  names one of those columns is refused, not ignored. An unreadable row still
  refuses the run, and a malformed identity is still dropped and logged. The
  Analyses are the manifest's rows in order, not sorted by id, and
  `analyses.tsv` now records the manifest's `original_sd_method`. `build-dense`
  does not lift: a row's `source_assembly` must be `--reference-assembly`. The
  `validate_store(source=...)`, the Python API's source-fidelity check, takes a
  build manifest, not a TSV: it compares each sampled source association with
  the Store after the manifest row's declared SD standardisation, reads a source
  as the build does (a row the build dropped as malformed is not a failure), and
  refuses a bare association TSV or a manifest with no rows (`fidelity.source`).
  A caller that passed a combined TSV now passes the manifest. No Store Release
  or `opengwasdb-stores` step calls `build-dense`.
