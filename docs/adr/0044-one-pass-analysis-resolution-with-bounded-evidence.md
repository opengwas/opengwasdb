# One-pass pre-build Analysis resolution, with evidence bounded by configuration

Phase B writes an Analysis's Assigned Ancestry and its
`original_sd`/`original_sd_method` into `analyses.tsv` before a Release Bundle is
accepted (ADR 0029; `opengwasdb-stores`' Phase B/Phase A split). Two stages
produced those columns, and each opened the source separately: AF-based ancestry
assignment (`assign-ancestry` → `ancestry.pipeline`) and phenotype-SD estimation
(`estimate-phenotype-sd` → `build.phenotype_sd_pipeline`).

On the store family that made this urgent — 4,570 genome-wide GWAS-SSF sources,
1.7 TB compressed — that is the same file decompressed and parsed twice for one
number each. Both stages read the same row for it: ancestry wants the frequency,
the estimator wants the standard error, and the two live in the same seven
columns. Measured on one 258 MB, 8.26M-row source: 86.6 s for the ancestry pass
and 74.2 s for the SD pass, against 55.4 s for one pass doing both.

Memory was the second problem, and it was the one that could not be fixed by
scheduling. The ancestry fit's input was every site the source shares with the
reference — millions, against a 21-column reference — and the estimator's
evidence was every qualifying row held as Python objects before `np.asarray`,
which on an 85M-row file is several gigabytes per worker. Neither scaled with
anything a caller had decided.

## Decision

**One module resolves one Analysis from one scan.** `opengwasdb.build.resolve`
reads a source once and accumulates both stages' evidence in that pass, then
fits ancestry and applies the caller's requested SD tier. The logical order is
preserved — ancestry is resolved *before* an ancestry-specific reference is
chosen, which is why the reference-MAF tier looks up the Assigned Ancestry's
declared reference rather than re-reading the source.

**The module computes; the caller decides.** The method tier is the caller's
(`AnalysisRequest.original_sd_method`, the same vocabulary as the manifest
column), the extraction panel is the caller's, and no tolerance, dispersion
threshold, verdict or `exclude_from_build` is produced here. ADR 0029's split
already said this; the decision here is that a resolver which chains fallbacks
(source AF → reference AF → beta spread) would record a number under a method
that did not produce it, so it does not chain.

**Evidence is bounded by configuration, not by the source.** The ancestry fit
holds one frequency per *panel* site, and the SD evidence is a deterministic
bottom-`k`-by-hash sample (`evidence_sample`, default 20,000) sharing
`build.eaf_orientation.site_hash`'s selection rule. The resolution reports
`n_evidence_considered`, `n_estimate_inputs` and `evidence_sampled`, so a
sampled median is never presented as a whole-file one.

**A projection-aware metrics seam carries the rows.** `SourceReader.stream_metrics`
(`TabularMetricsRow`) yields identity plus `beta`/`standard_error`/`effect_allele_frequency`
without a dict per row. It is a third projection beside `stream_variants` and
`extract_at_sites`, not a replacement: `stream_associations` cannot serve this,
because it carries a `z` rather than the beta behind it and drops rows with an
unusable beta that ancestry assignment can still read a frequency from.

## Consequences

- **Below the bound, the answers are the existing answers.** With fewer
  qualifying rows than `evidence_sample`, nothing is sampled and the arrays are
  the ones `phenotype_sd_pipeline` would have built in file order, so the
  estimate is identical rather than close; with `extraction_panel=None` the
  ancestry fit is `assign_from_source`'s. Both are asserted against those
  implementations, not against constants.
- **Above the bound, the estimate is a sample of the file, and says so.** On
  `GCST90446781` (8,262,639 rows) the bounded estimate was 1.0026 against the
  unbounded 1.002753738 — 0.02%, on a median over 20,000 per-variant values.
  A caller that wants the whole-file number passes a bound at or above the row
  count; the default is a memory ceiling, not a claim about statistics.
- **A panel smaller than the reference is a scientific choice this module cannot
  make.** Passing the fixed 10,000-site QC panel bounds ancestry memory to the
  panel, but whether it is interchangeable with the full reference for routing is
  a question for a concordance study, not for the resolver.
- **Only tabular formats have this path.** A GWAS-VCF's cheap read is a targeted
  `bcftools -R` at reference sites, not a full-file row scan, so
  `resolve_analysis` refuses a reader without `stream_metrics` loudly instead of
  reading a 100 GB VCF row by row.
- **The manifest-level CLI is a separate change.** This decision fixes the
  per-Analysis seam and its evidence contract; per-Analysis checkpointing,
  resume and a batch entry point are the follow-on interface, and they belong on
  top of this rather than inside it.
- **A source that cannot be read is a per-Analysis outcome.** An unreadable file
  returns a resolution with `error` set rather than raising, because one bad file
  in a batch of thousands is data, while a caller defect (a non-positive
  `evidence_sample`, a reader with no metrics path) still raises.

## Rejected

- **Two scans with better scheduling.** Halves nothing: the cost is per row, and
  the second pass re-reads every byte.
- **Unbounded evidence with tighter dtypes.** `float64` pairs are 16 bytes a row
  against ~100 for boxed Python floats — a real 10x, and still O(rows), which is
  the property that broke.
- **A resolver that selects the tier itself.** Convenient, and it is exactly how
  a source-AF number ends up recorded as `estimated_from_reference_maf`.
- **A dict-based row stream (`TabularRow`) instead of a projection.** Measured at
  ~10.5 µs/row against ~6.7 µs/row for the projected metrics scan on the same
  real file; the difference is the per-row dict, the identifier columns and
  decoding every cell, none of which either stage reads.
