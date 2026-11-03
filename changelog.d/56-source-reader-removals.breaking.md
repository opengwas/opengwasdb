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
  - `is_palindromic` moves to `opengwasdb.readers.orientation`; `opengwasdb.readers`
    still exports it, and `opengwasdb.readers.gwas_vcf` no longer does.
  - `SourceSpec.stored_effect_scale` may be `None`, for a BESD probe only.
  - A caller of `assign_from_source` that passes `liftover` for a capability
    that cannot lift is refused, where the lift used to be silently ignored.
