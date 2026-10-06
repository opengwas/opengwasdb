# Dense chunk and shard shapes for format 0.2.0

Epic #240, decided from #246. This ADR records the physical shapes #246 chose
for format 0.2.0's array roles, the measurements that decided them, the
candidates rejected, and the constraints the decision hands to #247 (the
builders) and #252 (the variant-side index). It supersedes ADR 0021's
`DEFAULT_CHUNK_SHAPE` as the statement of what a Dense grid's chunk is.

## Context

Two tickets met in #240. #237 wanted the Dense Analysis-axis chunk narrower than
1,000, because a whole-Analysis read then decodes 1/16 as much per chunk and
runs ~16× fewer bytes. #239 wanted Zarr v3's sharding codec, because narrowing
the chunk multiplies the file count: `[1000, 64]` is roughly 16× the files of
`[1000, 1000]` on a release that already holds 119,118. #245 built the
converter and proposed defaults; #246 had to measure the candidates and decide.

The decision also had to satisfy three tests, all required (the ticket's
"decision"):

1. every one of seven query shapes meets its **set-L** budget (typical time, p95
   time, peak memory) on OGS-00009;
2. one whole Analysis takes no more than **1.25×** the zarr 2.18 time, in the
   same back-to-back pair;
3. the file count meets #239's goal of "hundreds or low thousands".

A fourth, unmeasured axis was on the table: the **shard** Analysis width
`A_s`. It does not change what a query reads (the inner chunk is the read unit),
but it sets the file count and the Dense VCF band writer's scratch, which is
`n_variants × A_s × itemsize` of float32.

## Decision

**Adopt format 0.2.0**, with these shapes:

| array role | inner chunk | shard |
|---|---|---|
| Dense statistic planes (`z`, `se`, `eaf`) and the imputed mask | `[1000, 64]` | `[100000, 1024]` |
| top-hit index columns (`top_hits/<tier>/*`, `z`/`se`/`eaf`/…) | 16,384 (as 0.1.0), clipped to the array | **sharded**: 64 inner chunks |
| top-hit per-Analysis offsets | whole array | one shard holding the array |
| per-variant side arrays (`eaf_baseline`, `eaf_reference`) | the serving plane's variant-axis inner chunk (1,000), capped at 200,000 | about 1,000,000 elements |
| flat CSR association sequences | 200,000, or an explicit `chunks=(...)` | about 1,000,000 elements |
| CSR per-Analysis offsets | 10,000 | one shard holding the array |
| exception / overflow tables (Z, EAF, SE) | the role policy's 200,000, clipped to the array length | one shard holding the array |
| SE coefficients | `(min(n_analyses, 1024), 2)` | one shard holding the array |
| Rho Matrix arrays | 1,000,000, clipped to the array | about 1,000,000 elements |

The Dense shapes are the converter's defaults: `--dense-analysis-chunk 64`,
`--dense-shard 100000x1024`, `--top-hit-shard-chunks 64` (the seam's
`DENSE_SHARD_SHAPE` and `TOP_HIT_SHARD_CHUNKS`). A test pins both the seam
constants and the converter's signature defaults to them, so a silent drift
fails loudly.

**`A_s = 1024`, not 256**, is decided on read behaviour. The cost is accepted:
the Dense VCF band writer needs about **40 GB** of float32 scratch on ukb-b and
**87 GB** on FinnGen at this width (see the constraint to #247).

## What was measured

All of it on OGS-00009 (`ukb-b-full-observed`, Dense Observed-Only, 9,847,701
variants × 2,024 Analyses), under the heavy-job lock, with the reader
configuration pinned below, and recorded in
`docs/benchmark-output/opengwasdb_246_shapes/` — the harness artifacts, the
generated set-L tables, the conversion logs, the screening outputs, the top-hit
A/B, and the rendered report `opengwasdb_ogs00009_shapes.html`. Every number
below is in one of those files.

### Conversions

| release | inner chunk | Dense shard | top-hit shard | wall clock | peak RSS |
|---|---|---|---|---|---|
| `v3-c1000` | `[1000, 1000]` | `[100000, 2000]` | 64 | 1:38:49 | 36.29 GB |
| `v3-c128` | `[1000, 128]` | `[100000, 1024]` | 64 | 1:50:58 | 36.04 GB |
| `v3-c64` (#245) | `[1000, 64]` | `[100000, 1024]` | 64 | 1:38 (#245) | ~35 GB |
| `v3-c64-s256` | `[1000, 64]` | `[100000, 256]` | 64 | 2:00:20 | 35.85 GB |
| `v3-c64-topshard1` | `[1000, 64]` | `[100000, 1024]` | **1** | 2:01:12 | 36.22 GB |

Each was verified bit-exact against its source by the converter and validated
with no errors before publishing.

### Set L (two back-to-back pairs; zarr 3 vs the 2.18 run in the same window)

All four candidates pass every typical-time, p95-time and peak-memory limit in
both pairs. Medians, pair 1 (ms unless noted):

| query | set L typical | `v2-c1000` (2.18) | `v3-c1000` | `v3-c128` | `v3-c64` |
|---|---:|---:|---:|---:|---:|
| Top hits for one Analysis | 50 | 1.14 | 10.5 | 10.1 | 11.2 |
| One variant, all Analyses | 100 | 5.1 | 24.4 | 39.5 | 45.8 |
| One window, one Analysis | 250 | 24.0 | 45.0 | 40.7 | 35.5 |
| 10 variants × 100 Analyses | 500 | 70.1 | 114 | 256 | 233 |
| 100 variants × 10 Analyses | 3000 | 495 | 526 | 1143 | 831 |
| One window, all Analyses | 5000 | 1455 | 1380 | 1250 | 1194 |
| One whole Analysis | 60000 | 27693 | 23092 | 9491 | 6478 |

Whole-Analysis guard, pair 1 / pair 2: `v2-c1000` 0.67× / 0.66×, `v3-c1000`
0.83× / 1.07×, `v3-c128` 0.34× / 0.39×, `v3-c64` 0.23× / 0.26× (limit 1.25×).
Peak whole-Analysis memory: 1.02–1.18 GB for the 0.2.0 shapes, against
**1.14 GB on zarr 3** and **12.04 GB on zarr 2.18** (#244's measurements, the
same release and Analysis).

### Files and shard sizes

| | 0.1.0 (`v2-c1000`) | `v3-c1000` | `v3-c128` | `v3-c64` | `v3-c64-s256` |
|---|---:|---:|---:|---:|---:|
| files | 119,118 | 994 | 994 | 994 | 2776 |
| `z` / `se` / `eaf` shards each | ~29,545 | 198 | 198 | 198 | 792 |
| largest `z` shard | 1.6 MB | 267.5 MB | 133.9 MB | 134.5 MB | 35.5 MB |

`v3-c1000`'s shard is wider on both axes, so it has fewer, larger files; the
narrow shapes are all inside #239's goal.

### Shard width 1024 against 256

Two back-to-back pairs, s256 minus s1024 in each (positive is s256 slower):

| query | pair 1 | pair 2 |
|---|---:|---:|
| Top hits for one Analysis | +2.40 ms | −0.37 ms |
| One variant, all Analyses | +6.90 ms | +5.78 ms |
| One window, one Analysis | +0.63 ms | +1.93 ms |
| 10 variants × 100 Analyses | +107.2 ms | +86.7 ms |
| 100 variants × 10 Analyses | +312.7 ms | +372.1 ms |
| One window, all Analyses | +16.5 ms | −115.7 ms |
| One whole Analysis | +290 ms | −203 ms |

The row-shaped random reads are consistently slower (a read touching N inner
chunks touches more *shards* when each shard holds fewer), bulk and the
2,000-row band are within run-to-run noise, and peak memory is unchanged. The
100 × 10 query is 27–39% slower at s256 — still inside its 3 s typical budget,
but a real cost for no bulk gain. s256's whole-Analysis guard is 0.22× / 0.21×,
so it is not disqualified; it is simply not better where the priority is.

### The top-hit index, sharded against effectively unsharded

An interleaved A/B (`top_hit_shard_ab.py`; both stores opened in one process,
sides alternating sample by sample, 5 rounds × 100 pairs) because a conversion
ran in the same window:

| top-hit shard | files in the release | median Top-Hit Query |
|---|---:|---:|
| 64 inner chunks (`v3-c64`) | 994 | 12.06 ms |
| 1 inner chunk (`v3-c64-topshard1`) | 21,224 | 11.08 ms |

Ratio 1.09×, **0.16 ms per read** (six reads per Top-Hit Query, ADR 0056).
Unsharding the top-hit index is ~0.9 ms faster per query and ~20,230 files
larger. Against a 50 ms budget the time is not worth the file count.

### Screening beyond the four converted shapes

A 100,000-variant slice of OGS-00009 `z` in each shape, read in one process
(`shape_slice.py`), with the cost model (`shape_screen.py`) ranking the same
way. Ratios against `[1000, 1000]`:

| read (slice analogue) | `[1000,64]` | `[1000,128]` | `[1000,256]` | `[500,256]` |
|---|---:|---:|---:|---:|
| one whole Analysis (`fullcol_100k`) | 0.21 | 0.35 | 0.82 | 1.10 |
| one variant, all Analyses (`row`) | 1.91 | 1.82 | 2.15 | 1.32 |
| one window, one Analysis (`colseg_3000`) | 0.45 | 0.62 | 1.18 | 1.01 |
| 10 variants × 100 Analyses | 2.41 | 2.37 | 2.25 | 1.47 |
| 100 variants × 10 Analyses | 0.91 | 1.43 | 2.53 | 1.76 |

`[500, 256]` is a little better than 64/128 on the row-shaped reads and 5× worse
on the whole-Analysis read; `[1000, 256]` is worse on both. Neither is converted.

## Rejected candidates

- **`v3-c1000`** (inner `[1000, 1000]`, shard `[100000, 2000]`). It passes every
  budget and is the control that isolates sharding's own cost, but its bulk read
  is 23.1 s against 6.5 s at 64 — the whole-Analysis read is exactly what the
  epic exists to slash — and its bulk p95 varies most between pairs. Kept as the
  measurement's control, not chosen.
- **`v3-c128`** (inner `[1000, 128]`). It is the runner-up: 6 ms faster on
  one-variant reads and ~3 s slower on bulk. Rejected on the bulk priority; the
  one-variant difference is 39.5 vs 45.8 ms against a 100 ms budget.
- **Screened `[1000, 256]` and `[500, 256]`.** Neither beats `[1000, 64]` on the
  deciding shape (see the screen above), and neither 64 nor 128 failed memory or
  builder memory, which is the ticket's condition for converting a screened
  shape at all.
- **Dense shard width 256** (`v3-c64-s256`). Measured directly: consistently
  slower on row-shaped random reads, no bulk gain, 2776 files instead of 994.
  Its draw was 4× less builder scratch (10.1 GB ukb-b / 21.7 GB FinnGen against
  40.3 / 87.0 GB at 1024); the human accepted the larger scratch instead. The
  converter keeps `--dense-shard` as a parameter, so a later builder ticket can
  narrow it without a format change.
- **An unsharded top-hit index** (one inner chunk per shard). Measured directly:
  ~0.9 ms faster per Top-Hit Query and ~20,230 more files. Under set L the time
  is not a problem to solve, and the file count is.
- **A shard spanning every Analysis.** Rejected in ADR 0057 before measurement:
  the Dense VCF builder writes `[all variants × band]` column bands, so such a
  shard would never be written whole.

## Reader configuration this was measured under

Every zarr 3 run recorded the configuration it actually ran under, read back
from the process rather than assumed:

- `numcodecs.blosc.use_threads = True`;
- `zarr.config` `codec_pipeline.path = "zarr.core.codec_pipeline.FusedCodecPipeline"`
  with **`codec_pipeline.max_workers = 1`**;
- each top-hit array opened once per query facade (#244);
- each query's EAF read once, shared between SE decoding and the result's `eaf`
  column (#253).

The artifacts also record the measuring commit and a sha256 fingerprint of the
imported `opengwasdb` package (#253's provenance), so a number cannot be
attributed to a revision that did not produce it. A benchmark under any other
reader configuration is not comparable and does not count.

## Consequences

### For the converter and the seam

`DENSE_SHARD_SHAPE = (100_000, 1024)`, `TOP_HIT_SHARD_CHUNKS = 64` and the
Dense Analysis-axis default of 64 are now the **decided** shapes, not proposals.
The spec's §10a table changes from "the proposed defaults, which #246
benchmarks" to the decided ones. Nothing about the mechanism changes: inner
chunk and shard still come from `chunk_layout`/`shard_layout`, the one seam
authority #247 also reads.

### Constraints this hands to #247 (the builders)

- **The Dense VCF band writer's scratch is `n_variants × 1024 × itemsize`.**
  For float32 that is ~40 GB on ukb-b (9,847,701 variants) and ~87 GB on FinnGen
  (21,230,615). This is accepted, not a target. A builder that cannot hold it
  must either narrow `A_s` (the shard is a per-role policy) or stage the band
  differently — and the ADR that changes the width must supersede this one.
- **A shard is written whole.** The band width equals `A_s`; completion and the
  SE rewrite write row blocks. A writer whose write does not cover a whole shard
  turns it into a read-modify-write of the rest.
- **ADR 0056's fork constraint still applies.** Do not raise
  `codec_pipeline.max_workers`. A forked worker whose read spans more than one
  chunk, and no more chunks than the idle threads the parent's pool left behind,
  waits forever (zarr-developers/zarr-python#4478). Shard writers are where a
  larger worker count looks attractive; it is not available. Shard-aligned
  writes are a throughput concern, not a correctness one — nothing writes Zarr
  from more than one process.
- A writer must read the layout from the seam, never restate it: the manifest,
  the `index.sqlite` `dense` blob and the root attrs are validated against the
  arrays (ADR 0057 §5).

### Constraint this hands to #252 (the variant-side index)

The variant-side index's on-disk shape must be compatible with this shard
policy: it sits beside a Dense plane whose variant-axis inner chunk is 1,000
(any per-variant array is no coarser than that, capped at 200,000, and 1-D
arrays shard at about 1,000,000 elements), and it must survive the same
`shard_layout` role table rather than choosing its own. #252 records the shape
it picks against this ADR.

### Other

- 0.2.0 is adopted; #247 (builders), #248–#250 proceed. The file-count problem
  is solved by the format: OGS-00009 goes from 119,118 files to 994.
- The `[1000, 64]` inner chunk makes the whole-Analysis read ~3.5× faster than
  `[1000, 1000]` (6.5 s against 23.1 s) and costs a little on the one-variant
  read (45.8 ms against 24.4 ms, both inside set L's 100 ms budget). No shape
  misses a limit; the difference is where the priority lies.

## Alternatives rejected

The candidates in "Rejected candidates" above, and additionally: keeping the
Dense chunk at `[1000, 1000]` unsharded (`v3-c1000` is the control, not the
choice — it fails the epic's purpose), and a single shape for every role (the
roles' access patterns differ: a top-hit column is read at points, a CSR
sequence is appended and read whole, a Dense plane is read in bands).

## References

- The ticket: opengwas/opengwasdb#246; the epic: #240.
- The measurements: `docs/benchmark-output/opengwasdb_246_shapes/` and
  `PROVENANCE.md` there; the harness `benchmarks/benchmark_store_comparison.py`;
  the tables `benchmarks/zarr3_lever_tables.py shapes`; the top-hit A/B
  `benchmarks/top_hit_shard_ab.py`; the screening `benchmarks/shape_{slice,screen}.py`.
- ADR 0057: what format 0.2.0 is and how it is produced.
- ADR 0056: the zarr runtime configuration and the fork constraint.
- ADR 0021: superseded by this ADR as the statement of a Dense grid's chunk.
