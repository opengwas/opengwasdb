- **The Source Reader seam replaces its stand-ins (issue #56, ADR 0076).**
  - `opengwasdb.build.vcf_source` is removed: `stream_vcf_associations` and
    `stream_vcf_variants` are `GwasVcfReader.stream_associations` and
    `.stream_variants`.
  - `opengwasdb.build.resolve.MetricsReader` is removed. `resolve_analysis`
    takes any `SourceReader`, and no longer raises `TypeError` for a reader
    without metric blocks, since every reader has them.
  - `opengwasdb.readers.tabular.TabularRow` is now
    `opengwasdb.readers.rows.SourceRow`, re-exported from `readers.tabular` under
    its new name. `readers.tabular.extract_at_sites`, `stream_associations`,
    `stream_variants`, `metrics_chunks_from_rows` and `MetricsChunk` still
    import from there.
  - `is_palindromic` is defined in `opengwasdb.readers.orientation`, and now
    compares the alleles as `identity_allele` spells them.
    `opengwasdb.readers` and `opengwasdb.readers.gwas_vcf` still provide it.
  - `ReaderAssociation` takes `reference` as a required keyword, with no
    default: an association carries its reader's declaration, never an
    assumed one.
  - `GwasSsfReader` no longer takes `reference`. A GWAS-SSF file declares no
    reference allele, and a `TabularSource` now declares its own rows' through
    a `reference` property, which every implementation must provide.
  - `SourceSpec.stored_effect_scale` may be `None`, for a BESD probe only.
  - A caller of `assign_from_source` that passes `liftover` for a capability
    that cannot lift is refused, where the lift used to be silently ignored.
