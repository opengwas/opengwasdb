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
  `validate --source` fidelity check names an association TSV through the same
  manifest. No Store Release or `opengwasdb-stores` step calls `build-dense`.
