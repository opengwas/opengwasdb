# Changelog

All notable changes to the `opengwasdb` package.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning is described in [`CONTRIBUTING.md`](CONTRIBUTING.md#versioning) —
in particular, **the package version and a Store Release's `format_version`
are different things** and move independently. See the compatibility table at
the end of this file.

## [Unreleased]

### Added

- **`opengwasdb.build.ordered_pool.ordered_map`**: a forked worker-pool map that
  yields results in input order with a bounded number in flight, and runs
  serially at `n_workers <= 1`. Shared by the post-Pass-2 consolidation phases
  parallelised under #217.
- **`opengwasdb.model.manifest_columns`**: extracted shared manifest column alias
  resolution supporting multiple legacy aliases per canonical name (`analysis_id`,
  `source_file`, `analysis_label`, `sample_size`), used across Dense, Ancestry,
  and Ragged builders (#172, #177).
- **`opengwasdb.readers.known_capabilities()`**: shared accessor returning all
  registered source reader capabilities in sorted order (#177).
- **`--source-reader-capability` and `--source-assembly` on `build-dense-vcf` and
  `build-hybrid`**: allow supplying per-release defaults for manifest rows that
  omit these columns (#174, #177, ADR 0042). Precedence is per-row manifest
  column > CLI option > hardcoded default. Invalid CLI values fail at argument
  parse time.
- **`--analyses <analyses.tsv>` on `build-ragged-besd`**: allows overlaying
  registry Analytical and Attribution Metadata (including `PassthroughMetadata`
  fields, `assigned_ancestry`, `sample_size`, and attribution columns) onto
  BESD-derived analyses joined by `analysis_id`, while keeping BESD `.epi`
  coordinates authoritative and failing loudly on ID mismatch in either direction
  (#173, #177, ADR 0042). When omitted, output is byte-identical to prior builds.
- **`--format json` on `validate` and `info`**: provides machine-readable output
  for validation evidence (`{"ok": bool, "errors": [...], "warnings": [...]}`)
  and manifest inspection (with decomposed structured `encoding`), while default
  human text output remains byte-for-byte unchanged (#175, #177, ADR 0042). On
  invalid stores, `validate --format json` emits the JSON object on stdout and
  exits non-zero.
- **`estimate-phenotype-sd`**: estimates a per-Analysis phenotype SD directly
  from a canonical `analyses.tsv`, resolving a `SourceReader` per row rather than
  requiring caller-pre-extracted `se`/`af`/`beta` arrays (#176, #177, ADR 0029,
  ADR 0042). `--af-source source|reference` selects the estimator's frequency
  source; the output TSV uses the shared-core `analyses.tsv` spellings
  (`analysis_id`, `original_sd`, `original_sd_method`, `original_sd_dispersion`,
  `notes`), reports `unavailable` rather than fabricating a value for a missing
  or unusable sample size, and is order-preserving and independent of
  `--n-workers`.
- **`--variant-reference <path>` on `build-dense-vcf`**: build the Dense axis
  from a precomputed variant reference (`*.variant-ref.tsv.gz`, a plain ALID
  list, or a store `variants.tsv.gz`), bypassing the Pass 1 variant union and
  liftover entirely and proceeding straight to Pass 2. The axis, index, and
  fork-safe Pass 2 lookup arrays are composed from the reference before workers
  fork; source variants absent from the reference are dropped and reference
  variants no study observes are stored as `NaN`. Omitting the option preserves
  the existing inline two-pass build (#185).
- **`--variant-reference <path>` on `build-hybrid`**: build the Dense Component
  axis from a precomputed variant reference and route associations by its
  source-coordinate map, bypassing Pass 1 variant discovery. On-reference
  variants fill the Dense Component; off-reference variants -- including ones
  the reference never named, which are resolved from the source's declared
  assembly during Pass 2 -- go to the Ragged Overflow. `--reference-panel`
  remains supported, alone or as a subset of the reference; an inconsistent
  panel is ignored in favour of the reference, with a warning (#186).
- **`extract-variant-reference`**: a standalone command (and the
  `opengwasdb.variants.extract_variant_reference` API) that reads every source in
  a manifest once through its registered reader, lifts hg19 rows to GRCh38,
  canonicalises alleles, and writes the `*.variant-ref.tsv.gz` artifact
  (`alid`, `chromosome`, `position`, `a1`, `a2`, `rsid`, `source_keys`) that
  `build-dense-vcf --variant-reference` and `build-hybrid --variant-reference`
  consume. First-named rsids and the variant union are identical to the builders'
  inline Pass 1, so the two stages reproduce the one-command store bit for bit (#187).
- **`--window-size-mb` and `--reduction-batch-size` on `extract-variant-reference`**:
  the variant union is now computed with a parallel map + genomic-window tree
  reduce instead of one parent-process k-way merge, so 1,000+ source manifests
  reduce in bounded-memory batches across the worker pool. Variants are
  partitioned into non-overlapping `(chromosome, floor(position / window))`
  windows; final window shards concatenate in genomic order without a global
  re-sort, and the artifact is bit-for-bit invariant across window size and
  batch size (#188).
- **Phase timings and window-shard counts on `VariantReferenceExtraction`**:
  `extract_variant_reference` records `map_seconds`, `reduce_seconds` and
  `write_seconds`, plus the window, total-shard and reduced-window counts, so an
  operator or benchmark can see where an extraction spent its time instead of
  one total (#191).
- **`--map-spill-records` on `extract-variant-reference`** (and the
  `extract_variant_reference` API): a map worker now spills every window buffer
  to disk once it has buffered that many distinct sites (default 5,000,000),
  instead of accumulating its whole manifest slice in memory. A shard's rank is
  `(chunk_idx, spill_idx)`, so a later spill of the same chunk still merges
  after an earlier one and first-named-rsid selection is unchanged. Windows
  holding more shards than `reduction_batch_size` descend more than one tree
  level, reported as `reduce_levels`; a non-positive threshold fails loudly. The
  artifact is bit-for-bit unchanged across spill thresholds (#194).
- **`opengwasdb resolve-analyses` CLI and `opengwasdb.build.resolve_manifest`**:
  manifest-level CLI and pipeline that processes a canonical `analyses.tsv` manifest
  using the bounded one-pass resolver, checkpointing each Analysis into an atomic
  versioned JSON record (`{records_dir}/{analysis_id}.json`) and a deterministic
  manifest-ordered `index.json`. Features content-aware `--resume` that invalidates
  records on changed source size/mtime/checksum, tool version, references, extraction
  panel, admission gates, or method tiers; fork-shares the ~1 GB ancestry reference once
  per invocation without per-Analysis reloads; mitigates straggler tails via largest-first
  scheduling; and isolates ordinary source/parser errors into `controlled_failure`
  records while failing systemic setup errors loudly (#208, ADR 0045).
- **`opengwasdb.build.resolve`**: resolves one Analysis's AF-based Ancestry
  Assignment and its phenotype-SD estimate from a *single* scan of its source
  (`resolve_analysis`), reusing the existing ancestry mixture, EAF-orientation
  and ADR-0029 estimator rather than restating any of them. The method tier, the
  extraction panel and every acceptance threshold stay the caller's: nothing
  here falls back from source AF to reference AF, and nothing here emits a
  verdict. Per-Analysis memory is bounded by the panel and by a deterministic
  bottom-`k`-by-hash sample of the evidence (`evidence_sample`), not by the
  source's row count, and the resolution says when that sample was drawn. For
  the same file it reproduces `assign-ancestry`'s fit and
  `estimate-phenotype-sd`'s estimate exactly; on a real 258 MB, 8.26M-row
  GWAS-Catalog source one pass took 55.4s where the two existing passes took
  160.8s and agreed (#207, ADR 0044).
- **`opengwasdb.readers.tabular.stream_projected_metrics`** and
  **`GwasSsfReader.stream_metrics`**: a column-projected scan yielding each
  row's variant identity, `beta`, standard error and effect-allele frequency in
  one pass, without a dict per row and without the identifier columns the
  one-pass resolver never reads. `TabularMetricsRow` is its row shape, and
  `stream_full_row_metrics` keeps the full-row parser's semantics so the two are
  asserted equal field for field (#207).
- **`opengwasdb.build.resolve.ScanLimit`**: an explicit, deterministic bound on
  one source scan -- a fixed number of source rows, a fixed number of distinct
  usable ancestry-reference sites, or neither (the default full scan). The bound
  is checked after the row has been fed to both stages, so a bounded resolution
  is exactly the full resolution of the rows read; the stream is closed when the
  bound stops it; and `ScanDiagnostics.stop_reason` records whether EOF, the row
  bound or the site bound ended the scan. It exists so the issue #209 evaluation
  can compare a prefix with the full source under a recorded rule; a full scan
  stays the default and the only mode a release relies on. The study itself
  rejects every bounded rule -- each one turns the Unassigned `GCST90859377`
  into a confident EUR label -- so no bound is exposed on the manifest CLI
  (#209, ADR 0046).
- **Issue #209 evaluation harness and parser parity fixtures**:
  `benchmarks/benchmark_resolver_evidence_scan.py` compares the full scan with
  every preregistered fixed-row and usable-site prefix across the frozen
  106-Analysis evaluation manifest (`docs/benchmark-output/opengwasdb_resolver_evidence_scan_manifest.tsv`),
  and benchmarks the current projection against external `gzip -dc`/`pigz -dc`,
  pandas' C engine and R `data.table::fread` with decompression measured
  separately. `tests/test_resolver_evidence_scan_parsers.py` asserts the
  external-decompressor prototype is field-for-field identical to
  `stream_projected_metrics` on projection, reordered/extra columns, the legacy
  `hm_*` layout, ragged and quoted rows, invalid alleles, missing values,
  orientation and duplicate rows (#209, ADR 0046).
- **`opengwasdb.build.phenotype_sd.has_usable_sample_size`**: ADR-0029's
  sample-size rule, split out of `estimate_phenotype_sd` so a caller reporting
  *why* it has no estimate asks the same question the estimator answers (#207).
- **`opengwasdb.readers.effect_source` and `GwasSsfReader.effect_source`**: a
  GWAS-SSF Analysis's effect is now read from whichever permitted column its
  file carries. A file reporting `odds_ratio` yields `beta = log(odds_ratio)`
  through the association stream instead of being dropped, and the recorded
  `standard_error` is carried through unchanged because GWAS-SSF reports it on
  the log scale already. A non-positive or unparseable `odds_ratio` drops the
  row exactly as an unusable `beta` does, and a header naming a candidate effect
  column twice (as `GCST006329` does with `beta ` and `beta`) raises
  `ValueError` rather than resolving last-wins — previously that file's `NA`
  `beta` was read and every row silently dropped. Candidate names are matched
  ignoring surrounding whitespace, so a padded spelling is the same column.
  `beta` wins over `odds_ratio` when a file carries both, under an explicit
  tested rule; files that already carry `beta` are unchanged.
  `EffectSource`/`EffectSourceKind` are exported from `opengwasdb.readers`, and
  both the row-wise and blocked metrics projections resolve the same way, so
  their parity holds for `odds_ratio`. Verified on real GWAS-Catalog sources:
  `GCST006980`/`GCST008225` resolve to `odds_ratio` with
  `beta == log(odds_ratio)` for 200,000/200,000 rows each, `standard_error`
  unchanged; `GCST006329` now raises `Duplicate effect column 'beta' in header`
  where it previously yielded an empty association stream (#213, ADR 0049).
- **`BETA` accepted as an enumerated spelling of the `beta` effect column**:
  the effect source resolves `BETA` to the same `EffectSourceKind.BETA` as
  `beta`, reporting the matched spelling in `effect_source.column_name`
  (`"BETA"` or `"beta"`). The accepted set is explicit — `("beta", "BETA")` —
  not a blanket case-insensitive match, so `Beta`/`bEtA` do not resolve and the
  other GWAS-SSF column names keep their specified spelling. A header carrying
  both `beta` and `BETA` (padding included) raises
  `ValueError("Ambiguous effect column: header carries both 'beta' and
  'BETA'")` rather than silently preferring one; two of either spelling is
  still a duplicate. Verified on the real `GCST90044776` (26,825,889 rows):
  resolves to `BETA`, `beta` equals the raw column bit-for-bit for
  300,000/300,000 rows, and row-wise/blocked projection parity holds. That
  file's `standard_error` is `NA` in all 26.8M rows, so it still yields no
  associations — a source-data gap, not the spelling (#214, ADR 0050).
- **`opengwasdb.readers.effect_source` derives an effect from a signed z-score**:
  a GWAS-SSF Analysis whose effect column is a signed `z_score` (spellings
  `("z_score", "Zscore", "ZScore", "z")`) now yields
  `se = 1 / sqrt(2 f (1 - f) (N + z^2))` and `beta = z * se` from the row's own
  effect-allele frequency and **per-row** sample size (`n`/`N`, resolved as an
  enumerated set), where previously every row was dropped. The formula assumes a
  standardised phenotype, so the resolved `EffectSource` reports
  `is_derived=True` and `assumes_standardised=True`; a case-control Analysis
  (`log_or`/`log_hazard`) is refused with `CaseControlZScoreError` rather than
  being handed a standardised beta, and a z column carrying no negative value is
  refused with `UnsignedZScoreError`. An EAF outside `(0, 1)`, a non-positive or
  absent N, or a missing `n`/`N` column drops the row, never a substituted
  frequency or study-level N. Precedence is
  `("beta", "BETA") > "odds_ratio" > z-score`; both the row-wise and blocked
  projections derive identically. Verified on the real GWAS-Catalog pool: of the
  77 z-only sources, exactly 36 are usable under this rule (matching the issue's
  count), and on `GCST90129599` (`Zscore`, uppercase `N`) and `GCST90559206`
  (`z_score`) the derived beta and se equal the formula bit-for-bit for 20/20
  rows each, with the case-control refusal reproduced on real data (#215,
  ADR 0051).

### Changed

- **Pass 2 off-reference keys are now fixed-width `uint64`, not pickled strings**:
  a Hybrid build with `--variant-reference` encodes each off-reference source
  coordinate in the Pass 2 worker. SNVs with one-base A/C/G/T alleles pack
  losslessly (chromosome 5 bits, position 28 bits, ref 2 bits, alt 2 bits) and
  decode back to the exact raw key; every other key is hashed into a tagged
  part of the `uint64` space with its raw string in a small per-column side
  file, since liftover and canonicalisation still need it. A hash collision
  between two distinct keys fails the build loudly, naming both, and a hash
  outside the 63-bit region is refused rather than truncated. The change
  removes `allow_pickle` from the Hybrid build and shrinks the off-reference
  spill from about 30 B/row to about 20 B/row (#218).

  source labels `23`/`X`, `24`/`Y`, and `25`/`26`/`M`/`MT` normalise to the
  explicit canonical labels `X`, `Y`, and `MT` respectively in every reader
  path (#216, ADR 0052). This is a breaking change to variant identity:
  affected Stores built with numeric or `M` non-autosomal ALIDs must be rebuilt
  before reference completion or joins against post-change Stores.
- **`resolve_analysis` decouples the ancestry site bound from quantitative phenotype-SD evidence**:
  `max_ancestry_sites` bounds ancestry accumulation only (#212, ADR 0048). When
  phenotype-SD estimation is required (quantitative traits), ancestry accumulation
  stops at the site bound while the same physical source stream continues to EOF
  (or an explicit `max_rows` bound) to collect whole-file SD evidence, matching
  the full unbounded SD estimate exactly. When phenotype-SD estimation is skipped
  (case-control/log-OR/log-hazard or declared scale), the physical scan terminates
  immediately at the ancestry bound. Diagnostics separately report physical scan
  completion (`stop_reason`, `rows_read`) and ancestry completion
  (`ancestry_stop_reason`, `ancestry_rows_read`). `SCAN_LIMIT_VERSION` is bumped
  from 1 to 2, invalidating any resume records produced under the old truncated-SD
  semantics.
- **The one-pass resolver reads its source in blocks, ~1.6x faster end to end**:
  `stream_projected_metric_chunks` projects a genome-wide source a block at a
  time into `MetricsChunk` columns, and `resolve_analysis` accumulates from
  those rather than from a dataclass per row (#209). Allele and chromosome
  normalisation now runs once per distinct string instead of once per row, and
  the statistics are parsed a column at a time; the projection itself is ~2.1x
  quicker and a whole resolution ~1.6x. Measured on four real GWAS-Catalog-SSF
  sources (28.4M rows): 201.2s to 125.4s, with every resolved value --
  `assigned_ancestry`, `gate_reason`, `residual`, `original_sd`, dispersion and
  the evidence counts -- bit-for-bit identical. The row-wise
  `stream_projected_metrics` is retained as the parity reference.
- **`GwasSsfReader(chunk_rows=...)`** bounds what one resolver worker holds:
  50,000 rows by default, about 23 MB of a block's ALIDs against a genome-wide
  source, where the whole file would be hundreds of megabytes per worker (#209).
  Peak memory is now bounded by this rather than being independent of the
  source; it is fractionally faster than larger blocks as well.
- **`MetricsChunk` carries `palindromic` rather than the source's `ref`/`alt`
  labels**, because deciding strand ambiguity is the only thing those labels
  were read for. It is computed from the verbatim labels, matching the row-wise
  answer for a whitespace-padded allele rather than the tidier normalised one.
- **Two projection edges now fail loudly rather than quietly**: a header that
  repeats a projected column name raises instead of letting pandas hand back the
  wrong column, and a byte that is not valid UTF-8 fails the block rather than
  dropping the one row it lands in. `resolve_analysis` turns the latter into a
  per-Analysis error, not a silently shorter file (#209).
- **`opengwasdb.build.resolve_manifest` records a scan's stop reason**: every
  per-Analysis record's `diagnostics` now carries `stop_reason` (`eof`,
  `row_limit` or `ancestry_site_limit`), so a record written under a scan bound
  can never be read back as a full-source resolution (#209).
- **`resolve-analyses` bounds each source scan at 50,000 usable ancestry-reference sites by default (v1 semantics; amended by #212 / ADR 0048)** (`--max-ancestry-sites`, `0` restores
  the full scan; `--max-rows` bounds by rows instead). The #209 evaluation
  measured a 13.4x aggregate speedup on the 106-Analysis frame under whole-scan
  truncation, 105/106 assignment-and-gate agreement, and one false-positive EUR
  (`GCST90859377`), which ADR 0047 records as accepted. The bound is part of
  every record's fingerprint (`resolution_config.scan_limit`, with a rule
  version) and a changed bound invalidates resume; `resolve_analysis` itself
  still defaults to a full scan (#209, ADR 0047).
- **`extract-variant-reference` streams every manifest, retiring the in-memory
  assembly**: hg19 and mixed manifests now lift each pre-lift window in a worker,
  re-bucket every survivor by post-lift window, merge those buckets per post-lift
  window and concatenate the compressed members in genomic order. Cross-assembly
  ambiguous raw tuples are dropped by the window-local intersection, which is
  exactly the global hg38 ∩ successfully-lifted-hg19 rule; liftover attempt and
  failure counts are aggregated across workers and `liftover_failure_threshold`
  is enforced before any artifact byte is written, so a breaching manifest leaves
  nothing on disk. The `_extract_materialised_reference` path and the in-memory
  `write_variant_reference` extraction path are removed: every extraction path
  keeps the parent free of a global site set or lookup, and the artifact is
  bit-for-bit identical to the old writer's for all-hg19 and mixed manifests
  across worker, window, batch and spill configurations (#197).
- **All-hg38 `extract-variant-reference` streams the artifact in parallel**:
  manifests whose sources all declare hg38 (no liftover) now compress each
  final window shard to a standalone gzip member in a worker pool and append
  those members' raw bytes in genomic order, writing the header as the first
  member. This removes both serial O(union) stages (`_concatenate_window_shards`
  and `write_variant_reference`) from this path, so the parent never reads a
  shard row or materialises a global site set or lookup. The artifact's
  decompressed content is bit-for-bit identical to the materialising writer's
  across worker counts, window sizes and batch sizes, and remains readable as
  ordinary (multi-member) gzip; all-hg38 is the fast case where post-lift
  windows equal pre-lift windows (#196).
- **Dense and Hybrid map phases split the manifest into size-balanced chunks**:
  `_split_manifest_rows` now targets `min(sources, 4 * n_workers)` contiguous
  chunks balanced by cumulative on-disk source size instead of exactly
  `n_workers` chunks of equal row count. This stops one oversized source from
  setting the makespan of the worker that happened to own it: a dominating
  source lands in its own chunk while the remaining sources spread across the
  rest. Chunk rank is its manifest-order index, fixed before submission, so
  task completion order cannot affect shard sorting or first-named-rsid
  selection, and the artifact stays bit-for-bit identical to the pre-change
  output and between serial and parallel modes (#195).
- **`benchmark_extract_variant_reference.py` now exercises the tree reduce**:
  every synthetic source carries the same genome-wide panel, so every worker
  contributes a shard to every window rather than almost every window being a
  single-shard no-op. The run reports map, reduce and write separately alongside
  the total, speedup and window/shard counts, and still asserts every artifact is
  byte-identical across configurations before any timing (#191).
- **FinnGen R13 and GWAS-SSF `stream_variants()` now project only variant
  identity and alias columns** instead of parsing association statistics and
  materializing a full tabular row during Dense/Hybrid Pass 1. Header names,
  source order, duplicates, rsid fallback, chromosome/allele normalization,
  and variants with unusable statistics retain their existing behavior. A
  reproducible benchmark records decompression, projected variants, the
  pre-change full-row path, associations, and per-path peak RSS (#179).
- **`build-ragged-ssf` accepts canonical `analyses.tsv` column names**
  (`sample_size`, `source_file`) alongside legacy names (`n`, `filtered_file`,
  `file_path`), with canonical spellings winning when both are present (#172, #177).
  When a resolved `source_file` path is absolute, it is used directly; relative
  paths are joined against `--filtered-dir`.
- **Pass 1 variant-union construction now honours `--n-workers`** in the Dense
  and Hybrid VCF builders. `n_workers <= 1` keeps the serial read; `n_workers > 1`
  splits manifest rows across a fork pool, each worker spills sorted,
  deduplicated shards to a temporary directory, and a k-way merge combines them
  into the global union. Workers return shard paths rather than variant sets, so
  no large object crosses the process pipe; the first-named-rsid rule (#109) is
  preserved deterministically by manifest order. Build logs now report input
  files, per-worker shard sizes, final unique variants, and extraction/merge/
  liftover timings (#5).
- **Dense and Hybrid Pass 2 lookup construction no longer pads every variant
  key to the longest allele.** `_build_variant_key_index` / `_build_routing_index`
  now sort variable-length Python bytes instead of an `S`-dtyped numpy array, so
  one rare 546-byte indel no longer forces a ~11.6 GB padded array and its
  multi-minute `np.argsort` at genome scale. `_axis_metadata` and the hybrid
  partition/completion ALID sorts use a vectorised byte-key sort instead of one
  `_alid_sort_key` call per ALID (~21M calls). Output order is unchanged (#182).
- **Variant-reference extraction verified at production scale (#198)**: an
  82-source real manifest (64 UKB GWAS-VCF hg19 + 16 EBI GWAS-SSF hg38 + 2
  FinnGen R13 hg38, 15.6 GB) extracts 28,488,575 variants from 38,302,643
  source keys in 204.3 s at 16 workers against 2232.3 s at one -- a 10.9x
  end-to-end speedup (map 9.6x, reduce 13.6x, write 14.6x) with the reduction
  observed descending three tree levels. The serial and 16-worker artifacts are
  byte-identical, and the parent's peak RSS stays flat (~1.9 GB) as an all-hg38
  union grows 2.3 M -> 21.3 M variants. On a real mixed manifest the current
  artifact matches the pre-#190 writer's `alid`, `chromosome`, `position`, `a1`,
  `a2` and `source_keys` columns for all 21,559,975 rows; the 1,229 `rsid`-only
  differences are the deterministic `(rank, site)` tie-break accumulated after
  the pre-#190 commit, plus three where the streaming path declines an rsid from
  a failed-liftover tuple that shares a valid hg38 tuple's raw string. A store built
  via `build-dense-vcf --variant-reference` is byte-identical to the unified
  two-pass build except `manifest.json` provenance, confirming #185. Numbers and
  tables: `benchmarks/README.md`.

### Fixed

- **`VariantAxis.identity_by_indices()` no longer assumes the ALID index is
  complete.** The method inverted `_alid_rows` as a permutation of every axis
  row, but long-allele ALIDs are deliberately left out of the fixed-width
  index (#127), so any store carrying one raises
  `ValueError: shape mismatch` the moment a request spans an unindexed row —
  on OGS-00010 (115,043 long-allele rows) this broke `query --format table`
  and any `resolve_rows()` path. Indexed rows still resolve with zero table
  I/O; unindexed rows now fall back to one `by_index()` seek each, which is
  correct and bounded by how many long-allele rows a query actually returns
  (found by the OGS-00010 completion benchmark).
- **An all-dropped lifted extraction no longer leaves a header-only artifact.**
  `_finish_members` writes nothing when no window produced a member, so an
  hg19/mixed manifest whose every variant is an ambiguous cross-assembly
  collision (or fails liftover under a permissive threshold) fails loudly with
  `yielded no hg38 variants` and leaves no partial artifact on disk. A window
  shard carrying an unknown `source_assembly` is now rejected instead of being
  treated as hg19 (#197 review).
- **The rsid an ALID carries is now deterministic, not set-iteration order.**
  When two source keys differing only in allele order (or two source positions
  lifting onto one hg38 ALID) carried different rsids, the winner was decided by
  the union set's iteration order, which varies with the per-process
  `PYTHONHASHSEED` -- one manifest produced `rsAG` or `rsGA` on different runs.
  The rule is now explicit and enforced: the rsid for an ALID is the first
  non-empty rsid in `(rank, site)` order, where `rank` is the shard's
  manifest-order rank and ties break by site. `rsid_by_site` is produced in that
  order and consumed directly, so nothing downstream depends on hash order
  (#192).
- **Concurrent staging runs for one destination no longer delete each other's
  work, and an interrupted run no longer leaks its staging directory.**
  `OpenGWASDBStore.staging()` used a fixed `.{name}.tmp` sibling, so a second
  invocation for the same destination deleted the directory the first was still
  writing — on a real build the first then failed mid-run with
  `variant_offsets.npy` ENOENT while appearing to be running. Each invocation
  now writes to its own unique `.{name}.tmp.{pid}.{random}` directory and
  removes only that directory; cleanup catches `BaseException`, so
  `KeyboardInterrupt` and `SystemExit` discard it too (ADR 0043). Publication
  of a finished release is serialised by an advisory lock on the destination's
  parent directory and re-checks the destination at commit time: a no-overwrite
  build that loses a race with a concurrent publisher now fails loudly with
  `FileExistsError` instead of silently replacing it. The lock is host-local
  (see ADR 0043 for the shared-filesystem caveat). The two-rename atomic
  replacement and its rollback are unchanged.

## [0.3.0] — 2026-09-12

Work lands on `dev` and appears here under *Unreleased* until `dev` merges to
`main`, at which point it is cut into a version.

### Changed

- **`format_version` is reset to `0.1.0`, and the pre-release formats are no
  longer readable** (#143, ADR 0041). The format reached `3.0` before the
  project published anything: `0.1`, then `1.0` (fixed-point `z`, #114), `2.0`
  (residual-coded `eaf`, #116) and `3.0` (residual-coded `se`, #118), three
  majors inside one pre-release cycle. To a new reader `3.0` claimed a third
  stable generation with two supported predecessors; what it recorded was that
  we changed our minds three times before the first release.

  **The three-component shape is the safety mechanism, not decoration.**
  Resetting to `0.1` was the one thing that could not happen — pre-reset stores
  carry that exact string, and two formats under one name is a store that reads
  as plausible and is wrong. ADR 0038 already required a reader to reject a
  `format_version` that is not `MAJOR.MINOR`, so every reader that exists —
  including any already in the wild — rejects `0.1.0` loudly rather than
  parsing it as `0.1` and decoding `int8` planes as `float16`.

  `format_version` is now semantic versioning, `MAJOR.MINOR.PATCH`, with the
  **leftmost non-zero component** carrying an incompatible change: below
  `1.0.0` that is `MINOR`, from `1.0.0` it is `MAJOR`. A release meeting `0.1`,
  `1.0`, `2.0` or `3.0` is refused by name, with a message that says *rebuild*
  rather than a shape complaint or a decode attempt.

  **The pre-release compatibility paths are deleted, not deprecated.** Gone:
  `StoreEncoding.legacy()` and `is_legacy`; the `float32_optional` `eaf` kind
  and ADR 0036's plane-presence contract it named; the "no `encoding` block"
  and "no `encoding.version`" inference paths; and #157's per-kind version
  gate, which is vacuous once one format admits every kind its parser
  implements. Every readable release now declares an `encoding` block, stating
  version 3 and a plan for each of `z`, `se` and `eaf` — one format, one
  decoder, one contract to test, in the surface where a defect is a plausible
  number rather than an error. `StoreManifest.encoding` has no default: a plan
  nobody stated is a plan nobody can be held to.

  `ENCODING_VERSION` is deliberately **not** reset with the format. Blocks
  stamped 1 and 2 were real shapes this project wrote, and reusing one of those
  numbers for a third would recreate, one level down, the collision the reset
  exists to avoid.

  **Every store on disk is unreadable until rebuilt or restamped**, including
  the published `eur-hybrid-quant-pilot-10`. The cost is paid once and paid
  loudly. `scripts/restamp_store_to_0_1_0.py` replaces
  `migrate_store_to_format_3.py` and derives a `0.1.0` release from a `3.0` one
  without reading a single array: the reset renumbered the format and deleted
  decoders, and did not change the bytes a build writes, so a `3.0` release
  already holds exactly what `0.1.0` describes. That is `ukb-b`'s path — 13h30m
  to rebuild (#148), minutes to restamp — and the pilots are **rebuilt**
  instead, because a rebuild is the only thing that proves the builders still
  produce what the format says. `0.1`, `1.0` and `2.0` are refused by the tool:
  their planes are genuinely different encodings, and no stamp makes their
  bytes mean what `0.1.0` says. Like every derived release it mints a fresh
  `release_id` and `created_at`, regenerates `overview.html`, stages the copy
  and publishes by rename only when it validates with **no** errors — as a
  `0.1.0` release, by the build that reads only `0.1.0`, which is what makes
  the restamp sound rather than asserted.

  ADR 0038 is superseded rather than contradicted: what makes a change
  incompatible, the accept/reject/warn structure, one-version-written, and the
  completion and Staged Release rules all stand. `opengwasdb-stores` names
  format versions in its release manifests, store catalogue and query
  walkthrough, and changes in the same cut.

- **The default human-readable TSV query output now includes `eaf`** (#136).
  Every facade query already decoded and returned effect allele frequency, but
  `query-phewas`, `query-range-phewas`, `query-analysis`, `query-lookup` and
  `query-top-hits` hid it from TSV output unless `--variant-info` was passed —
  a flag whose real cost is `rsid`'s `variants.tsv.gz` lookup. `eaf` is now a
  default column before `association_status`; `--variant-info` still adds
  `rsid` and no longer changes whether EAF is returned. A store with no EAF, or
  a cell with none, prints `.` rather than a substituted default.

- **Dense and Hybrid VCF-manifest builders accept canonical `analyses.tsv`
  column names directly** (#170): `analysis_id`, `source_file`,
  `analysis_label` and `sample_size`. The pre-ADR-0034 names (`trait_id`,
  `file_path`, `trait_name`, `n`) stay readable for a deprecation window, so
  Catalogue `BUILD_COLUMNS` manifests remain valid; the canonical spelling wins
  when a manifest carries both. The ancestry source-manifest reader follows the
  same rule, so a Store Release bundle can be handed to a builder with no
  registry-side rename.

- **`assign-ancestry` accepts `--n-workers`** as its process-pool flag, matching
  every other build command. `--workers` remains accepted as an alias so
  existing callers keep working.

### Fixed

- **The complexity gate's changed-file mode widened its configured scope**
  (#166). `gate.py --changed` now diffs against the shared `base_ref`, rather
  than silently falling back to the last release, and explicit `--only` files
  are filtered by configured sources, language extensions and exclusions
  before lizard runs. Generated benchmark HTML and cleat's own excluded source
  therefore no longer appear as production complexity findings. The development
  environment now declares lizard explicitly, so the same complexity gate runs
  on a clean CI worker rather than depending on an untracked system install.

- **The format-3 migration republished the source release's identity, and
  could publish a release that did not validate** (#164). `migrate_store_to_format_3.py`
  copied the manifest wholesale and re-stamped only version, encoding and
  provenance, so the "new" release carried the source's `release_id` and
  `created_at` — two releases of the same store that cannot be told apart. It
  also published a staged copy whose validation errors were merely a subset of
  the source's: an error string identical to one the source already carried
  was subtracted, and an identical string is exactly how a defect the
  migration itself introduced would hide. The migration now mints a fresh
  UUID4 `release_id` and a current-UTC `created_at` (recording the source
  `release_id` in its provenance), and publication is gated on the staged
  copy validating with **no** errors — inherited or introduced, since the two
  are no longer told apart. A refused validation is an `Exception`, so the
  Staged Release cleanup contract discards the staging directory rather than
  leaving a failed copy behind, and the destination is never created. The new
  release's own `overview.html` is regenerated from the staged manifest, so
  the page humans browse advertises the fresh `release_id` rather than the
  copied source page's. And because the gate is absolute, a 2.0 source that
  does not itself validate — e.g. one carrying #127's truncated-ALID or #135's
  unchunked-`eaf_baseline` defects — can no longer be migrated at all: the
  phases run, the gate refuses, and the staging directory is discarded. Such
  a store must be rebuilt before a format-3.0 release can be derived from it.

- **Hybrid Reference Completion rebuilt the Dense Top-Hit Index under the
  source encoding after folding in a panel crossover** (#163). The rebuild
  now uses the completed Dense Component's own encoding, which carries the
  Reference EAF completion added, so a residual-`se` Hybrid store whose LD
  panel extends the axis no longer aborts with a missing-EAF decode failure
  when the index is rebuilt; the crossed-over association is queried back
  with its real, correctly decoded standard error.

- **The SE size measurement undercharged zarr's padded edge chunks** (#158).
  `_packed_1d` and `_packed_2d` compressed each measured slice at the size it
  happened to be, but zarr stores every chunk at its declared shape: a plane's
  final edge chunks — shorter than the chunk whenever the extent does not
  divide it — are padded out with the array's fill value *before* they are
  compressed. The compressed-bytes gate therefore compared each residual
  candidate against costs measured small, most on exactly the small or
  awkwardly-shaped planes where the margin is narrowest, biasing the decision
  toward the coding.

  The same undercharge survived the first fix in the streamed Dense, Hybrid
  and format-migration optimiser (`encoding/se.py`), which measured its planes
  with a second cost implementation; it is now closed there too. Every
  measured array — the codes plane (charged at the `SE_MISSING` fill the
  Dense rewrite declares), its `float16` alternative (NaN after a scratch
  narrow, or a migration source's own declared fill), the Overflow's flat
  planes (whole-written, numeric default fill), the coefficients and both
  side tables — is charged at its padded size with the fill its own writer
  declares, and both the whole-grid fits and the streamed optimiser share one
  chunk-accounting function (`measure.packed_chunk_bytes`), so one path
  cannot drift from the other again. A tiny, well-fitted Overflow no longer
  vetoes the coding the way the undercounted measurement made it seem to.

  On the rebuilt `eqtlgen-cis` Ragged plane the final 58,034-row chunk of a
  200,000 chunk stores 70,799 bytes against the 50,812 the old measurement
  charged — 39% more than measured on that chunk alone. A plane whose extent
  divides its chunk evenly is measured exactly as before, and tests now pin
  each measurement — 1-D and 2-D, partial in both dimensions, coefficients,
  Overflow and both side tables, on the streamed path as well as the
  whole-grid one — against the bytes a real zarr array of the same extent,
  chunk and fill occupies.

- **`ukb-b` at format 3.0, measured** (#148). A full Observed-Only Dense build
  of `ukb-b` (9,847,701 × 2,511 = 24,727,577,211 cells) under format 3.0 takes
  **13h30m** against 11h35m at format 2.0 (+16.5%), and its `se` plane falls
  from 21,183,939,687 to 3,961,274,232 bytes — **−81.3%**, well beyond ADR 0037
  §3's −58.1% estimate and the FinnGen pilot's −59.0%, because only 0.0068% of
  its cells fall outside ±0.5. The complete Store Release drops 28.5%, to
  42.56 GB, and compresses 9.98× against its 424.84 GB of source GWAS-VCF.

  Query latency moves both ways, and the direction depends on what dominates.
  Decoding residual `se` costs 0.0700 µs/cell against `float16`'s 0.0050 —
  **14×**, since each cell needs `eaf` decoded and an `exp`. A cached
  all-Analysis regional scan of 5,224,822 cells therefore takes 2.25× as long,
  while restricting the same region to one Analysis changes only 4.8%
  (22.75→23.85 ms). An IO-bound whole-Analysis scan of 8,419,893 cells is
  **12.1% faster**, because the plane it reads is 5.3× smaller. Random lookups
  of 10 variants × 100 Analyses and 100 variants × 10 Analyses move by +7.8%
  and −3.1% respectively. See
  `docs/benchmark-output/opengwasdb_ukbb_dense_issue148_benchmark.md`.

- **A residual `se` cell could be written with no EAF, and only this package
  could read it** (#159). `encode_se` turned a finite standard error whose cell
  had no frequency into an exact exception, and `decode_se` was then relaxed to
  exempt exceptions from the finite-EAF check — so the codec accepted a plane
  outside the contract #118 and #138–#140 describe, and a conforming reader
  handed one had no way to reconstruct the cell. Encoding now refuses it, where
  the caller still holds the source value and can choose `float16`, and decoding
  requires a finite EAF for every cell carrying a standard error. Nothing
  upstream produced such a cell in the first place: an imputed standard error is
  derived from the panel frequency, so a cell without one gets no standard error
  either, and a test now pins that.

- **A pre-3.0 manifest could declare residual `se` and no reader would object**
  (#157). `StoreManifest` parsed a declared `encoding` block without checking
  it against `format_version`, so a release stamped 1.0 or 2.0 could declare
  the format-3 `int8_residual` representation and this package would decode it
  — while a conforming reader of that version, entitled to read `se` as
  `float16`, would read the `int8` codes as `float16` and return plausible,
  wrong standard errors. The version is now parsed first, and a declared kind
  must have existed by the release's own major version: residual `se` below
  3.0 and residual `eaf` below 2.0 are refused with a message naming the
  version and the kind, and any `encoding` block below 1.0 — a release that
  never declared one — is refused too. The gate is a per-kind table (spec
  §6a), so the next version-gated encoding adds a row rather than a branch.

- **The format-3 migration rewrote the release it was given** (#156).
  `scripts/migrate_store_to_format_3.py` treated its `--into` as optional and
  re-encoded the `se` plane in place by default — a Store Release is immutable
  (spec §21.4), and a failed validation would leave the source damaged, exactly
  the interrupted-in-place-migration failure ADR 0038 §5 records. `--into` is
  now required and names a new release; the migration opens the source
  read-only, builds the destination in a staging directory and publishes it by
  rename only when the migrated copy validates, so neither the source nor the
  destination is ever a half-migrated store.

- **The `ukb-b` benchmark published a compression ratio of 0.0** (#148). Its
  source-size figure came from `data/ukb-b/manifest.tsv`, which points at
  `/local-scratch` paths that no longer exist, and it skipped a source it could
  not stat. All 2,514 entries were being skipped, so the raw total was zero and
  the ratio that divides into it was zero — a published number that looked like
  a measurement. It now reads either that manifest or a release config's
  `analyses.tsv`, restricts to the analyses actually in the store, and refuses
  to report a ratio at all if a source is missing. The artifact also records the
  store's `format_version` and `encoding`, so a format-2.0 timing cannot be
  mistaken for a format-3.0 one.

- **The general Dense builder discarded every effect allele frequency it was
  given** (#118). `build_dense_observed_store` read `z` and `se` off each
  `NormalisedAssociation` and dropped `eaf`, writing no plane and stamping no
  `eaf_scope` — a store built this way reported "this Analysis has no
  frequencies" for a source that supplied them for every cell. It now writes
  the plane, its baseline and its exception table, and marks the Analyses that
  carry frequencies `eaf_scope=association` with orientation `unverified` (this
  builder has no reference panel to check against). Residual-coded `se` needs
  that plane, which is how the omission surfaced.

- **A long indel could answer another variant's lookup** (#127). The ALID
  search index is a fixed-width array — that is what makes `np.searchsorted`
  work over it as an mmap — but it was built with
  `np.array(..., dtype="|S64")`, which truncates silently. Two indels at one
  position whose alleles agreed over 64 bytes collapsed to one key, and a
  lookup by either full ALID returned whichever row sorted first. On the
  published `finngen-r13/r13-pilot-20` release: 6,216 truncated ALIDs, 248
  keys shared by more than one variant, **342 variants answering to another's
  name**, through both `by_alid` and the vectorised `indices_by_identifiers`
  that dense `lookup()` uses. An over-wide ALID is now left out of the index
  and counted, the rule `_write_rsid_index` has followed since #109; it stays
  reachable by exact scan over its position. `validate_store` rejects an index
  holding a key shared by two variants, so an affected store says so.

### Changed

- **The complexity ratchet's scope and baseline were reconciled** (baseline
  reconciliation). Test code and benchmark drivers were being ratcheted as if
  they shipped in the package; `quality.json` now excludes `*/tests/*` and
  `*/benchmarks/*` from the lizard scan, with production sources unchanged.
  Against that honest scope the over-ceiling production functions found by
  review were split along reviewed, behaviour-preserving seams —
  `OverflowCells` field normalisation, the Dense envelope and
  completion-metadata/quality-table seams and Ragged imputed seams in
  `validate.py`, the Overflow structure/value seams, a dedicated z-plan
  validator, Dense completion's source-row matcher, per-record Analysis
  construction, a shared Dense/Ragged top-hit finalizer, Ragged `lookup`, and
  the z/se/EAF band passes of the VCF builder. The regenerated
  `quality/complexity-baseline.json` holds the 60 production functions still
  over the gate at their exact current measurements (stale entries dropped,
  improved entries recorded at their current values). Among the entries this
  reconciliation touched, only the three reviewed current-debt functions
  (`complete_dense_store`, `RaggedCSRWriter.flush`, `HybridStoreQuery.lookup`)
  were retained at newly accepted values rather than split.

- **Ragged Reference-Completed validation now enforces the
  `completion_quality.analysis_index` range rule** (validator tightening).
  Dense validation has rejected `completion_quality` rows whose
  `analysis_index` lies outside `analyses.tsv`'s range [0, n_analyses); Ragged
  validation previously checked only the table's presence and columns, so a
  Ragged store whose quality rows described a nonexistent Analysis validated
  cleanly. Ragged Reference-Completed stores now run the same shared
  `_validate_completion_quality_table` check against the CSR Analysis count,
  so such a malformed release fails loudly instead of reporting quality for an
  Analysis the store does not have.

- **The escapes and duplication gates no longer scan the worktree copies under
  `.claude/` or the pixi environment under `.pixi/`**, and the duplication
  baseline drops 92.51% -> 7.10% (#130). Both gates walked the tree from its
  root, and the tree contains `.claude/worktrees/` — a complete managed
  checkout of the repository per agent worktree — and `.pixi/`, tooling
  environment with vendored conda code included. Every real file was read
  beside copies of itself: the duplication gate failed clone pairs that
  paired a project file with its twin in a managed worktree — a file cannot
  be refactored to differ from a copy of itself — and measured a 92.51%
  share (8,271,343 duplicated of 8,941,213 "significant" lines, against
  37,906 in the project alone), while the escapes baseline had accepted
  19,121 sites, 14,573 of them under `.pixi/` and 4,508 under `.claude/`.
  Both gates now skip `.claude` and `.pixi`, and the duplication gate's
  changed-lines judgment is made against `origin/dev` — the branch feature
  work actually merges to — rather than the default-branch guess. With the
  scope honest, the escapes baseline falls to 39 sites and the duplication
  share to 7.10% (2,692 of 37,906 lines): a new escape or copied block on a
  branch now fails loudly instead of vanishing into the environment's and
  the worktrees' own totals.

  What the honest measurement then found in the project's own code is real
  duplication, now extracted into shared helpers. Dense and Ragged Reference
  Completion drove their block pools with two copies of the same checkpointed
  loop and now share one `run_block_tasks` (`completion/parallel.py`); Ragged
  Completion rebuilt each Analysis's observed `{alid: z/se/eaf}` maps three
  times and now folds them through one `_observed_alid_maps`;
  `assign_ancestry` returned the identical overlap-zero refusal at two sites
  and now returns `_overlap_rejected()`; and `index.sqlite`'s private
  `_parse_canonical_alid` — a second implementation of the canonical-ALID
  parse the Variant Index already owns, unread since the relational variant
  table was dropped (#128) — is deleted.

- **Hybrid residual-SE Overflow cells get a named, validated contract** (#162).
  The Overflow Component's `(se, eaf, analysis_index)` inputs were a
  positional tuple whose three arrays were shape-valid if swapped or mis-shaped,
  so a wrong order could survive all the way into the shared fit. `OverflowCells`
  now carries `se_values`, `eaf_values` and `analysis_indices` by name and
  rejects mismatched lengths, non-1-D arrays, non-integer Analysis indices and
  out-of-range indices at construction, before any fit or measurement reads
  them. `RaggedCSRWriter.se_fit_inputs` returns the named type, and the
  optimiser's module-only `ComponentCost` is now `_ComponentCost`.

- **Ruff no longer lints the vendored `quality/` tree**, and the ruff baseline
  drops 66 -> 61. `quality/` is the cleat gate tooling, imported whole and
  written to its own rules; scanning it contributed 1,735 findings against a
  project total of 61, which made `scripts/check_baselines.py` — the check
  that a change adds no new findings — unable to see the project at all. mypy
  was never affected: it only ever checked `opengwasdb/`.

- **`index.sqlite` no longer records `se_dtype`** (#118). It duplicated the
  manifest's `encoding` block, which spec §6a makes authoritative, and two of
  the three builders wrote it *before* the SE encoding was measured — so from
  format 3.0 it would have claimed `float16` for an `int8` plane. The same
  reasoning as #128's `variants` table: a duplicate that cannot be right is
  worse than no duplicate. Nothing read it.

- **The Store Variant Axis no longer keeps a relational copy of itself**
  (#128). `index.sqlite`'s `variants` table duplicated every column of
  `variants.tsv.gz` — which carries `source_alid` besides — was written by one
  builder, read by one self-described "legacy" function that nothing called,
  and had a `rsid` column left `NULL` in all 21,230,615 FinnGen rows while the
  TSV had rsids for 96.9% of them. Dropped, along with its range index and
  `variant_by_identifier()`. Its duplicate-ALID validation moves onto the ALID
  search index — the structure queries actually read, and the one that also
  catches #127's collisions. Measured on the #117 rebuilds, the variant axis is
  69–88% of a pilot store against the statistics' 11–31%: **803 MB of FinnGen's
  4.86 GB** was this table.
- **The ALID index slot narrows from 64 to 32 bytes**, now that an over-wide
  ALID is excluded rather than truncated and the width is a size/latency trade
  rather than a correctness one. Measured across the FinnGen, GWAS Catalog and
  metabolome pilots, ALID length is mean 14.9 and p99 21; at 32 bytes the index
  halves (**1,359 MB → 679 MB on FinnGen**) and 0.21% of variants resolve by
  exact scan instead of binary search.

### Added

- **The format-3.0 SE passes report their own wall time** (#144). Migrating a
  Dense release to format 3.0 takes four full passes over the `se` plane — the
  per-Analysis fit, the candidate measurement, the rewrite and the top-hit
  index rebuild — and the only previously recorded number was their sum
  (3,863 s for the FinnGen R13 pilot, extrapolating to 62.5 hours for `ukb-b`).
  `migrate_store_to_format_3.py` now threads a `PhaseTimer` through all four
  and prints each phase's seconds and share of the accounted total, so the
  optimisation that follows targets the pass that actually dominates rather
  than a guess. `build_top_hit_indexes` charges its own scan and write, so the
  rebuild is measured through the function that performs it rather than by a
  benchmark reimplementing its body — measured at 63.5 s on the migrated
  FinnGen R13 pilot, 1.6% of that 3,863 s, against an inference that had put it
  at 75-89%. The same instrumentation is available to any caller of
  `optimise_dense_se_joint` or `build_top_hit_indexes`; a caller that passes no
  timer is unchanged, and a migration records its own breakdown under
  `provenance.format_migration.phase_seconds`.

  What the breakdown showed, on the FinnGen R13 pilot (424,612,300 cells):
  **the migration's cost was almost entirely issue #135's unchunked
  `eaf_baseline`, not the format-3 encoding.** The same source data migrates in
  2,913 s from a release carrying that defect and in **244.5 s** from a
  repaired one. Reading `se` plus decoded `eaf` costs 2.35 us/cell on the
  defective store against 0.036 us/cell on the sound one -- every 1,000-row
  band decompresses the whole 21.2M-element baseline, 21,231 times per pass --
  so the three full plane passes fall from 2,680 s to 61 s. The 62.5-hour
  `ukb-b` extrapolation that motivated this work was taken from the defective
  store.

  The 2,913 s figure is **pre-#164** historical evidence (issue #164): the run
  that measured it published the format-3.0 release still carrying the
  source's three validation errors, a publication the migration no longer
  permits (see that issue's entry above). A 2.0 source holding #127- or
  #135-class defects cannot be migrated today — the phase work above would
  still run and then be refused at the publication gate and discarded, which
  is exactly why such a store must be rebuilt rather than migrated. The
  figure is preserved as measured and has not been re-measured against the
  current tool; the phase-cost conclusion it supports (the defect, not the
  format-3 encoding, dominates) is unchanged.

- **The #117 pilot-rebuild measurements are reproducible from the repository**
  (ADR 0037 evidence). `benchmarks/measure_pilot_releases.py` records, per
  Store Release named on the command line, the per-store and per-component
  cell counts, the declared encoding plans, compressed bytes for the whole
  release, each component and each statistic plane, the standalone validation
  outcome, the source identity and checksums the release records, and the
  measured commit and timestamp, into
  `docs/benchmark-output/opengwasdb_pilot_rebuild_measurements.json`. The
  ADR's B/cell figures are derivable from the artifact's per-plane
  `bytes / n_cells` without re-reading the stores; a missing store, manifest
  or component aborts the run rather than publishing a partial artifact. The
  format-2.0 #117 rebuilds predate later validator rules, so their recorded
  validation errors (#127 truncated-ALID index, #135 unchunked per-variant
  planes) are evidence recorded in the artifact, not run failures. The SE and
  UKB benchmark scripts (`benchmark_se_residual_queries.py`,
  `benchmark_ukbb_dense.py`) now write through the shared
  `benchmarks/_artifact.py` plumbing and record `commit` and `measured_at`
  (and each measured store's `format_version`/`encoding`), and the stale `uv`
  invocation in the UKB script's usage is replaced with the Pixi command;
  `benchmarks/README.md` now documents both scripts and the pilot driver with
  their exact regeneration commands.

- **Getting-started documentation now covers the first local Store Release**
  (#110). A fresh checkout can follow `docs/getting-started.md` to install with
  Pixi, build and validate the in-repository tiny Dense fixture, run PheWAS,
  exact-lookup and top-hit queries, and distinguish absent EAF from a fabricated
  value.

- **The SE residual-coding expectation is measured on EAF-bearing stores**
  (#118). `benchmarks/estimate_se_residual.py` samples rebuilt FinnGen, UKB and
  EBI Hybrid Store Releases, projects residual-coded SE bytes, and renders the
  query-speed/precision/storage trade-off in
  `docs/benchmark-output/opengwasdb_se_residual_expectation.qmd`.

- **Top-hit indexes carry decoded effect allele frequency** (#131–#134,
  ADR 0040). Dense, Ragged, and both Hybrid components now answer `top_hits()`
  from one compact derived structure without reopening or gathering from the
  source EAF plane. Rebuilding an existing index adds `float32 eaf`; readers
  retain a plane-backed fallback for older indexes. Dense/Hybrid VCF builds
  populate it inline with a variant-row-ordered pass. On the rebuilt ukb-b
  index, per-Analysis top hits recovered from the 86.6 ms regression to
  1.301 ms (pre-EAF: 1.17 ms), while global top hits recovered from 7,129 ms
  to 473.945 ms (pre-EAF: 488 ms).

- **Store format 3.0 adds conditional residual-coded Standard Errors** (#118,
  #137–#142). Every SE consumer now reads physical `float32` values through a
  decoded Dense/CSR plane. Eligible builds fit against decoded EAF, select a
  measured ±0.5/±1/±2 residual range, and persist coefficients plus exact
  exceptions; ineligible or non-saving planes remain `float16`.
  - **The exception gate is per Analysis, not pooled.** A share taken over
    every cell in the plane lets one badly fitting Analysis hide behind its
    well-fitting neighbours — the GCST007320 case the issue was raised about.
  - **Dense builders keep their scratch SE plane in `float32`**, as Dense
    completion already did, so an exact exception is the source's own value
    and not one already rounded to the dtype the plane started in.
  - The coding is **blind to allele flips** and provides no incidental check on
    EAF orientation (#115): `f(1−f)` is symmetric about 0.5.

- **`eaf` is stored as a per-variant baseline plus a per-cell `int8` logit
  residual, and `format_version` moves to `2.0`** (#116, ADR 0037 §2/§4).
  ADR 0036 shipped EAF as a `float32` plane parallel to `z`/`se`, which nearly
  doubled a store's statistic bytes for a column that is annotation rather than
  the finding. The semantics are unchanged — EAF is still per (variant,
  Analysis), still oriented to the stored effect allele, still declared per
  Analysis by `eaf_scope`. Only the physical encoding changes.
  - **The transform is the logit**, `log(f / (1 − f))`, not `log(f)`: a residual
    error of `d` moves `f` by a relative `(1 − f)·d` and `1 − f` by a relative
    `f·d`, so neither the effect allele frequency nor the minor allele frequency
    users filter on can be wrong by more than `d`. `log(f)` is blind to the
    minor side, which for a frequency near 1 is the only side anyone reads.
    Measured on FinnGen chr1 (8 endpoints, 429,961 variants, 3.44M EAF-bearing
    cells) the two transforms leave residuals of the same width — sd **0.0170
    against 0.0169**, max 0.632 for both — so the bound costs nothing.
  - **Two of the 256 codes are reserved**, leaving 254 levels: `-128` is "this
    Analysis reports no EAF here" (decodes to NaN, and is *not* a residual of
    zero), `-127` is "exception — exact `float32` in the plane's table". A cell
    that is neither round-trips to within half a step. On real pilot
    frequencies: **p99 0.184% / max 0.262%** at ±0.5 (FinnGen) and **p99
    0.374% / max 0.394%** at ±1.0 (GWAS Catalog), against half-step bounds of
    0.197% and 0.394%; on a synthetic sweep spanning all three ranges, max
    0.219% / 0.419% / 0.812%. All figures are the worse of the relative error
    on EAF and on 1 − EAF, because the minor allele frequency is what users
    filter on. Every exception cell round-trips exactly.
  - **Frequencies of 0 and 1, residuals outside the range, and cells at a
    variant with no usable baseline are held exactly** in an
    `eaf_exception_index` / `eaf_exception_value` table beside the plane — the
    same structure, keying and validation as `z`'s overflow table, sharing one
    implementation so the two cannot drift. Resolving one costs **109 ns/cell**
    against a 1,000-entry table, 219 ns against 100,000 and 684 ns against
    1,000,000, paid only for the cells that are exceptions.
  - **The range is chosen from measured data and measured bytes**, never
    inferred from the layout: the smallest of ±0.5 / ±1.0 / ±2.0 whose exception
    fraction is within 2%, then compared against ADR 0036's `float32` plane, and
    `float32` is written when the residual coding would not be smaller. A sparse
    store with roughly one EAF-bearing cell per variant pays more for the
    per-variant baseline than the `int8` cell saves, and stays in `float32`.
    Measured by running the shipped encoder over chr1 of two pilots' sources as
    a Dense grid, under the store's own zstd-3 + bitshuffle codec: FinnGen
    (429,961 variants x 8 endpoints, 8.00 EAF cells/variant) picks ±0.5 and
    costs **0.718 B/cell against `float32`'s 1.373, −48%**; GWAS Catalog EUR
    (419,192 x 9 studies, 4.86 cells/variant) picks ±1.0 and costs **2.023
    against 5.072, −60%**. The per-variant baseline measures 0.424 B/cell
    across 8 Analyses, which is ADR 0037's own prediction of 0.42. The *plane*
    figures do not reproduce ADR 0037 §2's 0.14 / 0.52 B/cell — the same
    encoding measures 0.294 / 1.245 here, 2.1x and 2.4x — so the issue's
    "within 20% of the table" criterion is **not met by these numbers**. It
    cannot be settled from source files anyway: it is a measurement on rebuilt
    Store Releases, which is #117, and whichever way that lands one of the two
    tables is wrong. The decision is unaffected — the residual coding is
    48–60% smaller than the plane it replaces on both pilots.
  - **Reference-panel EAF for imputed cells** (#113, superseded in approach). An
    imputed cell's EAF *is* the panel's and is identical for every Analysis
    imputed at that variant, so it is a per-variant `eaf_reference` array rather
    than per-cell data, applied on read through the imputed mask that Association
    Status already records. It amortises to ~0 B/cell on wide Dense grids, but
    #117 measured **3.206 B/cell** on completed `eqtlgen`, where a sparse Ragged
    store has only 2.4 cells per variant. Read straight from the panel, it never
    travels through the resumable completion checkpoint — which was #113's
    stated blocker.
  - **An observed cell whose source reported no EAF stays NaN.** It does not
    fall back to the panel: FinnGen's frequencies differ from the EUR panel by
    up to **3000×**, so substituting one would hand a user a plausible number
    that is wrong by three orders of magnitude, precisely for the rare variants
    they filter on. `decode_eaf` requires the imputed mask and the reference
    array together whenever a release declares reference EAF, so the
    substitution cannot be half-applied.
  - **`eaf_reference` is declared per component.** A Hybrid release's Dense
    Component has imputed cells where its Ragged Overflow does not, and each
    carries its own manifest; it is the one field on which the two components'
    plans may differ, and validation compares them with it normalised away.
  - **A release whose Analyses reported no frequency at all still carries panel
    EAF on the cells it imputes** — the case #113 was actually raised about. It
    carries `eaf_reference` and no `eaf` array: NaN on every observed cell, the
    panel's frequency on every imputed one, and `eaf_scope=association` on the
    Analyses that gained them. Both completion pipelines and both read planes
    do this; declaring `reference` beside an `absent` plane is legal, and
    validation no longer reads that pairing as a contradiction.
  - **An LD Reference Panel that declares no EAF completes rather than
    failing.** The panel is supplied for imputation and its frequencies are
    optional (`completion.ld_panel` has always read a missing one as NaN), so
    asking it for reference EAF must not turn a supported panel into a failed
    completion that takes the store's own frequencies with it. The same holds
    for a panel with no ancestry directory or no block tables: it is asked, not
    depended on.
  - **A Hybrid release's `eaf_scope` cross-check reads both components.**
    `eaf_reference` is per component, and a Hybrid release's Ragged Overflow is
    observed-only, so its top-level plan declares none while its Dense
    Component does. Judging the release from the top-level plan alone called a
    Hybrid store whose only frequencies were the panel's a contradiction of its
    own `analyses.tsv`.
  - **A pair-resolving CSR read reports the panel's frequencies too.**
    `eaf_pairs` short-circuited on "this component stores no `eaf` array",
    which is true of a release carrying only reference EAF — and is not the
    same question as "this component has a frequency to report". Top-hit reads
    go through it.
  - **A completed Hybrid release records the panel at the top level**, not only
    inside its Dense Component: `provenance.completion` now names the
    `ld_panel_id`, `ancestry` and `method`, read back from the component that
    did the imputation so the two manifests cannot name different panels.
    Manifest surface, so `opengwasdb-stores` needs it too.
  - **An `encoding` block of this version must declare its `eaf` plan.** A
    missing `eaf` key names ADR 0036's optional plane only below plan-schema
    version 2; at or above it the release is malformed and is rejected rather
    than read as an older one. A block with no `version` at all is read as
    version 1, since `version` was added with the plan itself.
  - **Validation gains the plan-versus-arrays rules for `eaf`**: a plane whose
    dtype contradicts the manifest, a residual plane missing its baseline or its
    exception table, an exception cell with no entry, a table describing a cell
    that is not an exception, an `eaf_reference` array the plan does not declare,
    and — the disagreement that got through review on #106 — an `eaf_scope` that
    contradicts the release's declared plan.
  - **`ogdb info` reports the `eaf` encoding** alongside `z` and `se`. Manifest
    and CLI surface, so `opengwasdb-stores` needs the same change.
  - **An ancestry-excluded Analysis no longer declares completion it did not
    get.** Dense completion imputes every LD block for every Analysis and
    applies the ancestry-match filter afterwards (ADR 0028); the filter was
    applied to the fills at the write but not to the `completion_quality` rows
    they came with, so a nonmatching Analysis carried through observed-only
    declared a nonzero `completion_n_imputed_total` — and, on a source with no
    frequencies of its own completed against a panel with some, an `eaf_scope`
    of `association` derived from that count, on an Analysis whose every cell
    reads NaN. The filter is now applied once, where checkpoint output becomes
    the release's, so the table and the arrays cannot disagree; the writer
    checks the shards honour it rather than silently re-filtering. Ragged was
    never affected — it excludes nonmatching Analyses at block assignment.
  - **Validation compares each Analysis's completion metadata with its own
    cells**, not only the store's plan with the store's arrays: an Analysis
    that declares imputed cells must hold at least one, and one that holds them
    must account for them. Categorical rather than a count comparison, since
    the rollup and the arrays count different things. Alongside it, a
    layout-independent rule read from `analyses.tsv` alone — a blank
    `completed_against` and a nonzero `completion_n_imputed_total` are a
    contradiction.
  - **A completed Hybrid release fails rather than name no panel.** Copying the
    panel identity up from the Dense Component caught every exception and wrote
    nulls, so a manifest could reach `reference_completed` with no
    `ld_panel_id` at all — the field #116 made load-bearing — and only a log
    line to say so.
  - **A release whose only frequencies are the panel's needs no orientation
    evidence.** Found by rebuilding `eqtlgen-cis-pilot` on the merged code, not
    by a test: it is built from BESD, which carries no EAF at all, so its build
    runs no orientation check and leaves `eaf_orientation` blank. Completion
    then stamps `eaf_scope=association` for the panel's reference EAF, and
    §9.1's evidence rule rejected the release — demanding a check on a cohort
    frequency column that does not exist and that no build could supply. Every
    Reference-Completed release built from an EAF-less source failed
    `validate_store`. The rule now applies only where some component declares
    an `eaf` plane; reference EAF is oriented by construction, through the same
    reader the check itself uses. Both no-frequency fixtures missed it because
    their builders record `unverified`, which warns rather than fails.
  - **A panel with no `EAF` imputes nothing, and the spec now says so.** An
    imputed `se` is scaled by the panel's heterozygosity, so such a panel
    produces no imputed cells rather than an `se` derived from a substituted
    frequency. The completion succeeds and the release's own frequencies
    survive it, which is what the fix above is for; "imputed cells read NaN" is
    vacuous here, and §6a no longer implies otherwise.

- **`z` is stored as `int16` fixed point, and `format_version` moves to `1.0`**
  (#114, ADR 0037 §1). `float16`'s step doubles with magnitude, so it was least
  precise exactly where p-values are steepest: worst-case p error was 26.4% at
  |z| = 30 and the actual error at the FADS1/FADS2 hit (|z| = 47.8) was a factor
  of 1.82. Stored `z` is now `round(z x 1024)` with uniform precision — 1.6%
  worst-case p error at the range edge, 0.24% at z = 5 — and smaller: **1.400
  against 1.591 B/cell, −12%**, measured by re-encoding a 50.3M-cell band
  (28.5M present) of the `ukb-b` pilot under the store's own zstd-3 + bitshuffle
  codec. Bitshuffle groups the near-constant high bytes of a bounded quantity,
  where `float16`'s exponent bits churn for the small values that dominate.
  (ADR 0037 reports −20% from a different sample; the `int16` side agrees at
  ~1.4–1.47 B/cell and the `float16` baseline is what varies with the data.)
  - **Two codes are reserved**, because an integer plane cannot hold NaN and the
    missing-cell contract depended on it: `-32768` is missing (decodes to NaN,
    paired `se` must also be missing), `-32767` is out of range. Spec §15 now
    defines missingness per plane's declared codec rather than as NaN.
  - **Out-of-range values are held exactly, not clipped and not rejected.** The
    `ukb-b` survey found max |z| = 137.5 (HERC2/OCA2) with 6,346 cells above 32
    (6,312 in the copy re-surveyed here, at `/data/opengwasdb/wip/ukb-b`),
    so the representable range is backed by a sparse `z_overflow_index` /
    `z_overflow_value` table beside each plane — 74 KB against a multi-gigabyte
    store. The 500 strongest cells of that pilot, |z| up to 137.5, round-trip
    through the table exactly (max |Δz| = 0.000000, p ratio 1.000000; the
    largest is log10 p = −4107.68). A build now fails only on a non-finite or
    malformed statistic.
  - **`se` stays `float16`,** deliberately: it spans 3.2 decades and needs
    relative precision, which a float exponent already provides. The right
    encoding follows from the shape of the quantity; this is not a preference
    for integers.
  - **`StoreEncoding` and `StoreCodec`** (`opengwasdb.encoding`, issue #119) —
    the plan is decided once per build in `StoreEncoding.decide()`, recorded in
    `manifest.json`, and read back by every builder, completion pass, query
    adapter and validation rule. No read path re-derives it, and a reader
    meeting an encoding kind it does not implement rejects the release.
    `StoreEncoding.legacy()` covers pre-#114 stores, whose planes are `float16`.
  - **Validation cross-checks the plan against the arrays**: a plane whose dtype
    contradicts the manifest, an out-of-range cell with no overflow entry, a
    table describing a cell that is not out of range, or a Hybrid release whose
    two components declare different plans all fail the store.
  - **A release at `format_version` 1.0 or above must declare its encoding.** A
    missing block is refused rather than falling back to the legacy plan, which
    would decode an `int16` plane as `float16` and return z-scores a thousand
    times too large — a plausible number, and so the worst possible outcome.
    The codec refuses the same disagreement reached directly.
  - **Manifest change**: the per-layout `provenance.*.dtype` field is now
    `se_dtype`. It only ever described the float planes, and leaving it named
    `dtype` next to an `int16` `z` would have made the manifest quietly wrong.
    A reader of the old key gets nothing rather than a wrong answer.
  - `format_version` becomes `1.0`. `0.1` releases stay readable — they decode
    under the legacy plan — and are never written again; **completing a `0.1`
    store is now refused** (ADR 0038 §4) rather than stamping its version onto
    newly encoded arrays. Rebuild instead.
- **Allele-flipped EAF is rejected at build time** (#115, ADR 0037 §6). Each
  Analysis's A1-oriented `eaf` is correlated against a reference over a
  deterministic sample of variants before any statistic array is written;
  `r < 0` fails the build, naming the Analysis and the observed `r`. This
  catches `GCST003566` in the `gwas-catalog-eur-hybrid` pilot, which reports
  `effect_allele_frequency` against the *other* allele (r = -0.9992 against the
  EUR panel, where every other study in the release reads +0.999). A
  correlation rather than a difference threshold, so a bottlenecked cohort
  whose frequencies differ from the panel by 3000x still passes.
  - New `--eaf-reference` / `--eaf-reference-ancestry` /
    `--allow-unverified-eaf` options on `build-dense-vcf`, `build-hybrid` and
    `build-ragged-ssf`, reading either an LD panel directory or a table with an
    `eaf` column.
  - With no reference and three or more Analyses, the consensus of the other
    Analyses is used instead; a build whose Analyses contradict each other
    fails rather than guessing which is right.
  - Outcomes are persisted: `eaf_orientation`, `eaf_orientation_r` and
    `eaf_orientation_n` in `analyses.tsv`, plus the reference's identity and
    checksum in `manifest.json`'s `provenance.eaf_orientation`.
  - **The same check at ancestry assignment**, where the frequencies were
    already being compared to a reference: a mis-oriented Analysis is left
    Unassigned with `gate_reason=eaf_orientation`, and the evidence is written
    into the Analysis Catalogue (`eaf_orientation`, `eaf_orientation_r`). The
    NNLS residual gate already rejected such an Analysis (0.5788 against a
    threshold of 0.06 for `GCST003566`) but could not say why, and an inverted
    Analysis does not merely fail to fit — it fits as another super-population,
    AFR at 0.696, above the τ = 0.50 gate. Recalibrating τ/δ cannot re-admit it.
  - `audit-eaf-orientation` re-runs the correlation against a supplied panel
    over a built store's own arrays, for stores built before the check existed.
- **A store-format versioning and migration policy** (#112, ADR 0038).
  `format_version` is `MAJOR.MINOR`, with the split defined by what a reader
  that does not know about a change would do: major if it would misinterpret
  the store, minor if it would still read correctly everything it knew about.
  Spec §21 now answers what bumps which, what a reader owes a store it did not
  write, and "I have an old store — now what?".
  - `SUPPORTED_FORMAT_VERSIONS` becomes a major → highest-known-minor mapping.
    An unknown major is rejected; a newer minor within a known major is read
    with a warning. It was previously `frozenset({"0.1"})`, exact-set
    membership over opaque strings, against which §21's "reject unsupported
    *major* versions, MAY support older *minor* versions" was not implementable
    — and which had no test coverage at all.
  - A `format_version` that is not `MAJOR.MINOR` is rejected
    (`MalformedFormatVersion`, a subclass of `UnsupportedFormatVersion`: to a
    caller deciding whether it can read a release, unparseable and
    from-the-future are the same answer).
  - Migration expectations are documented against the **Provenance Amendment**
    exception CONTEXT.md already defines. Noted rather than smoothed over:
    `scripts/migrate_store_to_analyses_tsv.py` rewrites `analyses.tsv` in place,
    which is outside that exception. It predates the policy and its targets are
    stores that should be rebuilt instead; bringing it into line is a change of
    its own.
- CI (`.github/workflows/ci.yml`): tests, tooling baselines, and a changelog
  gate on every pull request and push to `dev`/`main`. The repository had no CI
  at all before this.
- A pull-request template carrying the correctness, test and documentation
  checklists from `CONTRIBUTING.md`.
- `CLAUDE.md` — a short orientation file for AI coding sessions, pointing at the
  same standards rather than restating them.

### Changed

- **Ancestry assignment reads any Source Format, not only GWAS-VCF** (#115).
  `assign_from_source` resolves the reader through
  `opengwasdb.readers.registry`; a source manifest may now carry an optional
  `source_reader_capability` column, defaulting to `opengwasdb.gwas-vcf`. The
  old restriction is why `GCST003566` was never examined: the
  `gwas-catalog-eur-hybrid` family is harmonised GWAS-SSF, so every Analysis in
  it carries `ancestry_assignment_method=source_trusted_no_af`. The Catalogue
  carries the capability through, so it can now drive a build of those sources.
- **`validate` now rejects a store that carries EAF it never checked** (#115).
  Every Analysis with `eaf_scope=association` must record EAF orientation
  evidence; `unverified` is reported as a warning. Deliberate: a frequency
  column nobody has checked is indistinguishable from one reported against the
  other allele. It invalidates any Store Release built between ADR 0036 (which
  began retaining EAF) and this change, until it is rebuilt or audited with
  `audit-eaf-orientation` — but no such release exists today: every pilot on
  disk predates ADR 0036 and carries no `eaf_scope` column at all, so the rule
  does not apply to them.
- The Analysis Catalogue gains `eaf_orientation`, `eaf_orientation_r`,
  `source_reader_capability` and `gate_orientation_flip_r` columns (#115), and
  `assign-ancestry` gains `--orientation-flip-r`. All are annotation columns:
  the build-manifest superset invariant (`BUILD_COLUMNS` first, in order) is
  unchanged. Recalibrating an older Catalogue appends the new gate column
  rather than failing on it.
- `analyses.tsv` gains three columns (`eaf_orientation`, `eaf_orientation_r`,
  `eaf_orientation_n`, #115). Store-only: a release manifest neither carries
  nor needs them, and `opengwasdb-stores` accepts them through the schema's
  existing superset property (nothing there enumerates the column list). Its
  build generators, however, invoke the CLI: they should start passing
  `--eaf-reference` so pilot rebuilds are verified rather than `unverified`.
  `GCST003566` itself is already excluded from the EUR hybrid pilot there, with
  the evidence recorded in `inclusion_reason` (`opengwasdb-stores` eefcc81).
- `CONTRIBUTING.md` gains a Documentation section: what goes stale, what
  regenerates it, that benchmark numbers are re-run rather than edited, that
  `opengwasdb-stores` holds docs depending on this package's CLI surface, and
  a pre-merge checklist.

### Fixed

- **Reference Completion refuses a panel no Analysis matches, instead of
  completing to nothing** (#98, ADR 0039 §2). `derive_impute_analysis_ids`
  compared the `--ancestry` flag to each Analysis's `assigned_ancestry` by
  string equality, and the two carry different vocabularies — panel
  directories are named `EUR`, the registry may record `European`. A store
  recording the word therefore matched **zero** Analyses, and completion ran to
  a successful finish, produced a release stamped `reference_completed`, passed
  `validate_store()`, and reported `0 imputed` in a line that reads as "nothing
  was imputable". The LD work had all succeeded: `completion_quality`, written
  before the filter applies, held correlations up to 0.97 and 51M+ imputable
  cells across 1,357 blocks. Two changes:
  - spellings of one ancestry are reconciled through an explicit alias table,
    matched exactly on a normalised label, in either direction. Not through
    `ancestry.routing.reported_to_superpop`, whose ordered substring matching
    answers `AFR` for `"North African"` — fine for guessing at a cohort's
    free-text description, not for deciding which Analyses a panel may impute.
    A panel named outside the vocabulary still matches on exact equality;
  - a genuinely empty match raises `AncestryFilterError`, naming the panel, the
    values the store holds, and whether the panel's own name was understood.
- **Ragged Reference Completion no longer does nothing for a Store Family with
  no gene target** (#102, ADR 0039 §1, spec §17). Regions were identified only
  from a cis window around `trait_chr`/`trait_bp`, so an Analysis without one —
  small-molecule metabolomics, by design — was silently passed through: 0
  blocks enumerated, 0 imputed, a "completed" store byte-identical to its
  source. All four `metabolome-plasma-2023` full releases (4,443 Analyses)
  completed this way. Such an Analysis's regions are now the LD blocks it
  already holds enough observations in.
  - "Enough" is `impute.min_observed_points()` (4), the number the imputation
    gate actually enforces: `poly_rescale` returns NaN below it and
    `impute_z_block` then rejects the block, so a lower threshold enumerates
    blocks that add panel variants as missing rows and impute none of them.
    The constant is exported from `impute` rather than restated.
  - The count is over the Analysis's observations **at the block's own panel
    variants**, which is what `run_block` fits on — not over its variants
    falling inside the block's base-pair extent, which an off-panel variant
    inflates.
  - The panel is read once per completion, one chromosome at a time: holding a
    genome-wide panel's blocks resident costs roughly a gigabyte, and these are
    the families that span the genome.
  - `RaggedCSRReader.variant_indices()` exposes an Analysis's variant footprint
    without decoding its statistics.
- **Reference Completion could stamp a `format_version` onto arrays it had not
  encoded that way** (#112). All three completion paths copied the source's
  `format_version` into the completed release, which is the right rule —
  completion writes into the source's arrays and therefore its encoding — but
  did it incidentally. The moment `CURRENT_FORMAT_VERSION` moves ahead of a
  store on disk (which #114 does), that produces a release that lies about its
  own encoding. Completion now refuses a source it can read but cannot write,
  and does so *before* the imputation rather than at manifest-write time.
  Nothing can reach it today; that is why it was worth adding now.
- `benchmarks/README.md` documented seven commands as `uv run`, which stopped
  working when the project moved to Pixi.
- `README.md` described a two-month-old codebase as "newly scaffolded".

## [0.2.0] — 2026-08-22

The first tagged version. `0.1.0` was the placeholder the project carried from
its first commit and was never released, so this entry backfills the changes a
user would notice across the whole history to date.

### Added

- **Effect allele frequency is stored** (ADR 0036, #106). Every source format
  reported it and every reader parsed it; no layout kept it. Now a per
  (variant, Analysis) plane in Dense, Ragged and Hybrid, oriented to the stored
  effect allele, declared per Analysis by a new `eaf_scope` column, and exposed
  through the query facade and `--variant-info`.
- **rsid lookups work** (#109). An rsid search index is written by
  `write_variant_axis`, so every layout gets one. Collision policy is explicit:
  an rsid names every row it appears on.
- **Human-readable query output by default** (#104), with `--variant-info` for
  rsid and eaf, and `--format json` for the raw index-keyed result.
- **`overview.html`** — a generated, store-wide browsable rendering of
  `analyses.tsv` with persisted Top-Hit Counts (ADR 0032).
- **Hybrid layout** — a Dense Component over a reference panel plus a Ragged
  Overflow Component for off-panel observations (ADR 0026).
- **Reference Completion** for Dense, Ragged and Hybrid, with ancestry-matched
  imputation (ADR 0028) and per-block quality records.
- **Ancestry assignment from allele frequencies** (ADR 0029) and the Analysis
  Catalogue ingestion hub (ADR 0027).
- **Rho matrix** for Dense stores (ADR 0025).
- Validation: a closed-envelope rule (#80), rsid index coverage, and EAF range
  and shape checks.

### Changed

- **`analyses.tsv` is the single Analytical Metadata contract for every layout**
  (ADR 0030, ADR 0034). `phenotype_id`/`phenotype_label`/`trait_id` retired in
  favour of `analysis_label`/`trait_ontology_id`/`trait_ontology_label`;
  `gene_id`/`gene_name` retired too (ADR 0035). Ragged's divergent SQLite
  `analyses` table is gone — a leftover one is now a validation failure.
- **Manifest metadata reaches the store.** Dense and Hybrid in #86, Ragged in
  #83 — fifteen shared-core columns that manifests supplied and builders
  silently dropped, now carried through one shared `PassthroughMetadata`.
- The `SourceReader` interface carries variant identifiers (`SourceVariant`)
  and per-association EAF, both of which sources provided and the interface
  previously had nowhere to put.
- Query results are a documented adapter contract across layouts (ADR 0033).

### Fixed

- **Hybrid `analyses_table()` undercounted Top-Hit Counts** by delegating to the
  Dense Component (#107). Hid 4,476 hits on the gwas-catalog pilot; one Analysis
  reported 10,727 instead of 14,706.
- **Ragged and Hybrid Reference Completion silently dropped data** on rebuild —
  rsids (#109) and observed EAF (found in review of #106). The eqtlgen pilot had
  49,967 rsids observed and none in its completed sibling.
- **Dense and Hybrid never captured rsids at all** (#109) — 0 in 50,000 rows of
  both the finngen and gwas-catalog pilots.
- Disjoint-partition violation on Hybrid LD-panel extension (#99).
- Duplicate canonical-variant rows within a Ragged Analysis (#101); the
  `trait_id` requirement dropped from the Ragged SSF builder (#100).
- Reference completion now warns when no Analysis carries `assigned_ancestry`
  and it would otherwise impute everything against one panel (#108).

### Known limitations

- Compressed sizes and the overflow table's real cost are measured in ADR 0037
  on pilot data, not yet on a rebuilt genome-scale store (#117).
- `eaf` is still `float32` and `se` is still `float16` per stored cell; the
  residual encodings that shrink them are #116 and #118, and the plan carries
  only `z` and `se` until then.
- Reference-Completed releases carry no EAF on imputed cells (#113, #116).
- Existing Store Releases predate every build-time fix above and must be
  rebuilt to gain them (#117).

## Store format compatibility

The package version and the store `format_version` are independent. A store
records the `format_version` it was written against; the package records which
it can read.

| package | writes format_version | reads |
|---|---|---|
| 0.2.0 | 0.1 | 0.1 |
| 0.3.0 | 0.1.0 | 0.1.0 only |

The two `format_version` values in that table are different formats despite
reading alike: `0.1` is the pre-release format 0.2.0 wrote, and `0.1.0` is the
reset (#143, ADR 0041). Nothing on `dev` reads `0.1`, and the shapes cannot be
confused by a reader — only by a person reading this table, which is why it
says so here.

[Unreleased]: https://github.com/opengwas/opengwasdb/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/opengwas/opengwasdb/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/opengwas/opengwasdb/releases/tag/v0.2.0
