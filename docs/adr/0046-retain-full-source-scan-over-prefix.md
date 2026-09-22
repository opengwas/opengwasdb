# Retain the full source scan: a prefix is not a substitute for one-pass Analysis resolution

> **Superseded in part by [ADR 0047](./0047-adopt-50k-ancestry-site-bound.md).**
> The evaluation and its numbers stand unchanged; the *policy* decision to keep
> the full scan for the full-catalog release is superseded by ADR 0047, which
> adopts a 50,000-site bound on the maintainer's judgement that the compute
> saving across ~6,000 GWAS outweighs the measured false-positive EUR.

The Phase B resolver (`opengwasdb.build.resolve.resolve_analysis`, ADR 0044)
reads each compressed GWAS-SSF source once and accumulates the ancestry fit's
frequencies and the phenotype-SD evidence in that pass. It still decompresses
the whole source and projects every row in Python. The full-reference policy
study ([opengwasdb-stores#152](https://github.com/opengwas/opengwasdb-stores/issues/152))
rejected the fixed 10,000-site extraction panel on concordance grounds, but that
did not establish that every row of a sorted GWAS file must be read. Issue #209
asked whether a deterministic prefix -- a fixed number of source rows, or a
fixed number of usable ancestry-reference sites -- preserves the full scan's
resolution, and separately whether a compiled parser materially beats the
current Python projection.

Both questions were answered against a frozen 106-Analysis evaluation manifest,
the same stratified frame as #152, with the full-source #152 records as the
comparator. Every tested rule is reported in
[`docs/spec/bounded-evidence-scan-preregistration.md`](../spec/bounded-evidence-scan-preregistration.md)
and [`docs/spec/bounded-evidence-scan-report.md`](../spec/bounded-evidence-scan-report.md);
no threshold was moved after results were seen.

## Decision

**A prefix resolution is not interchangeable with a full scan, and the resolver
keeps the full scan.** `ScanLimit` is added to `resolve_analysis` only as an
explicit, non-default experimental bound whose `ScanDiagnostics.stop_reason`
records whether EOF or a bound ended the scan. It is not a release mode, and no
release may resolve an Analysis under a scan bound.

The evidence that forces this:

- **Every preregistered rule produced exactly one false-positive EUR
  assignment, which the locked criteria forbid.** `GCST90859377` (case-control,
  848 MB, 16.5M rows) has 5,360,202 overlapping reference sites and a
  full-genome NNLS residual of 0.110, above `residual_max = 0.06`, so the full
  scan leaves it **Unassigned**. Every tested prefix -- down to 5,000 usable
  sites and 14,597 rows -- fits it as **EUR** at a dominant proportion of
  0.995-0.998 with a residual of 0.006-0.008. The prefix samples a region the
  reference panel represents well; the cohort as a whole is not that. A rule
  that accepts a prefix therefore converts a known Unassigned Analysis into a
  confidently wrong European label, which is the exact failure the project's
  founding principle names.
- **The usable-site rules reach the 98% concordance bar and still fail.**
  `sites_5000` through `sites_50000` score 99.06% assignment/gate concordance
  and 100% orientation-failure sensitivity, so a concordance-only test would have
  accepted them. The false-positive criterion is what rejects them, and it is
  why the acceptance criteria were locked before the run.
- **Every rule moved the phenotype-SD estimate or dispersion beyond the locked
  2%.** At 1,000,000 rows the estimate differs by at most 1.6% but dispersion by
  up to 29.6%; at 50,000 usable sites dispersion differs by up to 90%. The
  evidence sample is a deterministic bottom-`k`-by-hash sample of the rows the
  scan saw, so a prefix necessarily draws a different sample from the whole
  source. No early stop can preserve it exactly.
- **Early stopping is not even uniformly fast.** The usable-site rules consume
  1.17M-1.30M rows *on average*, because sources with sparse reference coverage
  -- and the 12 with no usable source frequency at all -- run to EOF. Only the
  evidence-rich genome-wide files stop early. A stopping rule whose cost depends
  on coverage density is a poor scheduling primitive.
- **Fixed raw-row prefixes fail on concordance before anything else.**
  `rows_25000` scores 86.8% and `rows_100000` 95.3%; only from 250,000 rows does
  the frame reach ~98%, and even then the false-positive EUR remains.

**No parser swap is adopted.** The projected read is bound by the Python
per-row projection, not by decompression or CSV parsing. On the twelve-source
parser sample, external `gzip -dc` and `pigz -dc` feeding the *same* projection
bought 1.0-1.3x; pandas' C engine (parse only) bought 1.5-2.8x and R
`data.table::fread` (parse only, reference upper bound) 2.6-5.4x, but those
measure bytes-to-rows and not the projection that dominates the projected read.
A compiled parser can therefore not deliver a material speedup without
vectorizing the projection, which is a separate, higher-risk change with its own
parity burden. The external-decompressor prototype is retained only as the
parity-checked benchmark fixture in
`tests/test_resolver_evidence_scan_parsers.py`.

## Consequences

- **`ScanDiagnostics.stop_reason` travels with every resolution.** `eof` is a
  whole source; `row_limit` and `ancestry_site_limit` are a prefix. The
  manifest resolver writes it into each per-Analysis record, so a record
  produced under a bound can never be read back as a full-source resolution.
- **The frozen evaluation frame is committed.** The 106-Analysis manifest and
  the preregistered criteria live in this repository, so a later proposal must
  clear the same bar on the same frame rather than a fresh, friendlier one.
- **A future early-stop or parser proposal has a measured baseline.** The
  harness (`benchmarks/benchmark_resolver_evidence_scan.py`) re-runs both the
  rule comparison and the parser benchmark and writes the JSON artifacts under
  `docs/benchmark-output/`; the report is generated from those, never
  hand-edited.
- **The next material lever is the projection, not the parser.** The benchmark
  localises the cost: `GCST90628007` (491 MB, 12.4M rows) spends 50s in the
  current projected read, of which R `fread` would remove at most ~11s. A
  vectorized projection is the change that could reach a multiple, and it must
  preserve the projection semantics asserted by the #179 and #207 parity tests.
- **The full scan stays the cost of a correct answer.** For the genome-wide
  frame this is roughly the #207/#152 number (mean full-scan wall time on the
  106-Analysis frame is recorded in the artifact), and that cost is accepted
  rather than traded for a wrong label.

## Rejected

- **A fixed 10k extraction panel.** Already rejected by #152 at 95.10%
  concordance; this study does not reopen it.
- **Judging a prefix by concordance alone.** The usable-site rules score 99.06%
  and still mislabel `GCST90859377`; concordance without a false-positive count
  is exactly how the 10k panel looked acceptable until it was not.
- **A larger usable-site target as a fix.** The false positive is locality, not
  sample size: the prefix region is clean for this cohort at every target tested,
  so raising the target cannot be shown to find the anomalous region short of
  reading the source.
- **An external `gzip -dc`/`pigz -dc` production path.** Measured at 1.0-1.3x on
  the projected read, it does not justify subprocess lifetime, broken-pipe and
  portability surface for a path that is not the bottleneck.
- **Relaxing the SD tolerance to make a rule pass.** The criteria were locked at
  2% before the run; moving them afterwards would make the study an exercise in
  choosing the answer.
