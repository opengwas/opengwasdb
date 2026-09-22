# Decouple the ancestry-site bound from quantitative phenotype-SD evidence

ADR 0047 adopted a 50,000-site ancestry bound (`--max-ancestry-sites 50000`) for
the full GWAS Catalog release based on v1 whole-scan truncation semantics. This
ADR amends that policy to ensure that quantitative phenotype-SD estimation is
not truncated by the ancestry bound, while preserving early physical termination
for case-control and non-quantitative studies.

## Context

The #209 study evaluated early-stop rules on a 106-Analysis frame (54 quantitative,
52 case-control). Under v1 semantics, `max_ancestry_sites` physically stopped the
entire source scan for both study types. At 50,000 usable ancestry sites,
phenotype-SD estimates on that small sample frame differed by at most 8.6%,
leading ADR 0047 to accept the bound for the release.

However, subsequent evaluation on the 400-Analysis rehearsal panel (opengwasdb-stores #154,
spanning 190 quantitative studies with numeric SD estimates) revealed that
truncating the physical source scan at 50,000 ancestry sites caused material
distortions in phenotype SD on larger genome-wide files:

- 9 of 190 quantitative estimates differed by more than 5% relative to the full scan,
  with a maximum difference of **23.9%** (e.g. `GCST90002310`: full SD 1.293 vs prefix SD 0.984).
- Several large files showed 11–24% SD shifts while dispersion misleadingly decreased,
  because chromosome-sorted GWAS files concentrated the prefix sample on chromosome 1.
- Two Analyses changed status from `passed` to `warning` under the release's
  `dispersion > 0.5` rule.

The root cause was that `_scan` treated `max_ancestry_sites` as a physical stop
for the entire one-pass resolver, truncating the bottom-`k` reservoir sample for
phenotype SD as well as ancestry.

## Decision

**`max_ancestry_sites` bounds ancestry evidence only.**

1. **Decoupled quantitative streaming:** When an Analysis requires phenotype-SD
   estimation (`_skip_reason(request) is None`), ancestry accumulation stops
   once `max_ancestry_sites` distinct reference sites are collected, but the
   **same physical source stream continues to EOF** (or until an explicit
   `max_rows` bound). The deterministic `_EvidenceSample` reservoir continues
   admitting qualifying rows from the entire file. The resulting phenotype-SD
   estimate and dispersion match the full-source (`--max-ancestry-sites 0`)
   result exactly.
2. **Early physical termination for non-quantitative traits:** When an Analysis
   does not require phenotype-SD estimation (case-control on `log_or`/`log_hazard`,
   `binary_trait`, or already declared scales), the physical scan terminates
   immediately at the ancestry bound.
3. **Hard row bounds:** An explicit `max_rows` limit remains a hard physical row
   bound on both ancestry and phenotype-SD evidence.
4. **Truthful diagnostics:** `ScanDiagnostics` separately records:
   - Physical scan completion: `rows_read` and `stop_reason` (`eof`, `row_limit`, or `ancestry_site_limit`);
   - Ancestry-prefix completion: `ancestry_rows_read` and `ancestry_stop_reason` (`eof`, `row_limit`, or `ancestry_site_limit`).
5. **Fingerprint version bump:** `SCAN_LIMIT_VERSION` is bumped from 1 to 2.
   Existing records produced under the old truncated-SD semantics are invalidated
   on `--resume` and recomputed.

## Consequences

- **Exact quantitative analytical metadata:** All quantitative Analyses carry
  phenotype-SD estimates and dispersions drawn from the whole source, eliminating
  prefix sampling bias and false dispersion warnings.
- **Fail-loud corruption checks on quantitative sources:** Because quantitative
  sources are streamed to EOF, any byte corruption or file truncation occurring
  *after* the 50,000th ancestry site is now encountered and fails loudly as a
  controlled per-Analysis error (`AnalysisResolution.error`), rather than being
  silently masked by an early physical stop.
- **Estimated throughput on the 106-Analysis evaluation frame:** Under the
  vectorized reader, the full 106-Analysis scan took 6,818.65 s (54 quantitative:
  3,387.36 s; 52 case-control: 3,431.29 s). Under v2 decoupled semantics,
  quantitative studies run full (3,387.36 s) while case-control studies run the
  50k bound (220.14 s), giving an estimated **3,607.50 s** total (**~1.89x
  speedup** over full vectorized scan on this frame). This is an estimate
  pending full downstream recalibration.
- **Release-wide impact:** In the frozen EBI GWAS Catalog European snapshot
  (`gwas-catalog-ssf-eur-hybrid-2026-09-10.tsv`), **1,057 of 4,570 ready Analyses
  are case-control (23.1%)** and benefit from early physical termination, while
  the 3,513 quantitative Analyses (76.9%) stream to EOF for exact SD estimation.
- **Single physical pass:** The source is never opened or decompress-streamed twice.
- **Memory safety:** Peak memory during the quantitative full-file pass remains
  strictly bounded by `evidence_sample` (default 20,000 items in a min-heap),
  independent of source file size.
