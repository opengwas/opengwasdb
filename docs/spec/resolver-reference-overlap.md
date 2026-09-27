# Resolver reference-overlap diagnostics

`resolve-analyses --variant-reference PATH` accepts a plain-text or gzip-compressed
Hybrid axis of canonical ALIDs (one per line, or a table with `alid`,
`variant_id`, or `id` header). The parent loads it once and fork-shares the set.
A SHA-256 fingerprint of its bytes invalidates resumed records when the axis
changes. Missing/empty axes fail setup rather than silently skipping overlap.

Each successful per-Analysis record's `diagnostics` includes
`ancestry_reference_rows_matched` / `ancestry_rows_read` and
`variant_reference_rows_matched` / `rows_read`. The former counts **rows** on
the ancestry extraction panel up to ancestry accumulation's stopping point,
before AF/SE/palindromic filters; `ancestry_sites` still counts *distinct usable*
sites for the ancestry fit. The latter counts **rows** on the supplied Hybrid
axis over the physical scan, regardless of AF/SE eligibility. Repeated rows are
counted repeatedly. The stream is never reread for either measurement. Without
`--variant-reference`, `variant_reference_rows_matched` is null. Failed scans
retain partial diagnostics; downstream release policy must not treat those as
complete overlap estimates. An ancestry-site bound can stop a non-quantitative
physical scan early; quantitative phenotype-SD scans normally continue to EOF.
