# Resolver reference-overlap diagnostics

`resolve-analyses --variant-reference PATH` accepts a plain-text or gzip-compressed
Hybrid axis of canonical ALIDs (one per line, or a table with `alid`,
`variant_id`, or `id` header). The parent loads it once and fork-shares the set.
A SHA-256 fingerprint of its bytes invalidates resumed records when the axis
changes. Missing/empty axes fail setup rather than silently skipping overlap.

Each successful per-Analysis record's `diagnostics` includes
`ancestry_reference_rows_matched` / `ancestry_rows_read` and
`variant_reference_rows_matched` / `rows_read`. The former counts **rows** on
the full ancestry Reference Resource up to ancestry accumulation's stopping point,
even when an extraction panel narrows the fit, before AF/SE/palindromic filters; `ancestry_sites` still counts *distinct usable*
sites for the ancestry fit. The latter counts **rows** on the supplied Hybrid
axis over the physical scan, regardless of AF/SE eligibility. Repeated rows are
counted repeatedly. The stream is never reread for either measurement. Without
`--variant-reference`, `variant_reference_rows_matched` is null. Failed scans
retain partial diagnostics; downstream release policy must not treat those as
complete overlap estimates. An ancestry-site bound can stop a non-quantitative
physical scan early; quantitative phenotype-SD scans normally continue to EOF.

Overlap is a canonical-ALID measurement, not a guarantee of Dense routing. For
GRCh38 sources, the Hybrid builder matches reference source keys without regard
to allele-letter case (while preserving the source effect-allele order, Z sign,
and EAF orientation); an on-axis source row must fill its Dense row, not become
an off-reference Overflow association. The resolver-versus-built-store parity
test covers this separately from ancestry/SD eligibility. A routing defect must
not be classified as a low-overlap source exclusion. Cross-assembly comparisons
still require explicit source-key/liftover evidence; canonical overlap alone
does not establish that an hg19 source matches a GRCh38 axis.

With a provider-declared INFO policy (stores #175) or a numeric MAF threshold
(stores #176), a positive threshold removes only below-threshold usable rows
before ancestry, SD and routing evidence, so `ancestry_reference_rows_matched`
counts only admitted rows while the legacy `variant_reference_rows_matched` and
`rows_read` stay pre-admission canonical-row counts. Naming a variant reference
also makes the Analysis whole-stream (stores #176, section 0): the physical scan
continues to EOF, so `rows_read` and `variant_reference_rows_matched` are no
longer a prefix even for a case-control Analysis the ancestry bound would once
have stopped early. A publication projection over a filtered source must use
that record's `build_eligible_rows_off_variant_reference` / `build_eligible_rows`
diagnostics rather than these legacy counts; see
[`info-score-resolver.md`](../info-score-resolver.md).
