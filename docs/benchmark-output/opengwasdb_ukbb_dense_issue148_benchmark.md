# ukb-b at Store format 3.0: residual-coded `se` (issue #148)

Query timings re-measured 2026-09-08 against two Observed-Only Dense `ukb-b`
releases built from the same 2,511 GWAS-VCF sources by the same generator
(`opengwasdb-stores/families/ukb-b/releases/dense-observed-vcf-c128-rebuild117`),
9,847,701 variants × 2,511 Analyses = 24,727,577,211 cells. Each timing is the
median of five warmed repetitions on an otherwise idle machine.

The two encoding plans differ only in `se`:

| | format 2.0 | format 3.0 |
|---|---|---|
| `z` | `int16_fixed` scale 1024 | `int16_fixed` scale 1024 |
| `se` | **`float16`** | **`int8_residual` ±0.5** |
| `eaf` | `int8_residual` ±0.5 | `int8_residual` ±0.5 |
| store | `ukb-b__dense-observed-vcf-c128` | `…-c128-issue148-se3` |

## Storage

| array | 2.0 | 3.0 | change |
|---|---:|---:|---:|
| `se` codes | 21,183,939,687 | 3,957,471,004 | |
| `se` side tables | — | 3,786,980 | |
| `se_coefficients` | — | 16,248 | |
| **`se` total** | **21,183,939,687** | **3,961,274,232** | **−81.3%** |
| `z` (unchanged) | 31,530,596,349 | 31,530,596,349 | 0% |
| `eaf` + baseline (unchanged) | 5,165,550,483 | 5,165,550,483 | 0% |
| **`data.zarr` payload** | **58,602,429,684** | **41,659,124,813** | **−28.9%** |
| **complete Store Release** | **59,502,552,404** | **42,559,247,618** | **−28.5%** |

0.857 B/cell down to 0.160 B/cell. **The saving is far larger than ADR 0037 §3
predicts (−58.1%) or the FinnGen R13 pilot measured (−59.0%)**, because
`ukb-b`'s Analyses share a design: only 1,677,442 cells of 24,727,577,211
(0.0068%) fall outside ±0.5, against a 2% budget, so almost every residual is
near zero and compresses accordingly.

Against 424.84 GB of source GWAS-VCF, whole-store compression goes from 7.14×
to 9.98×. `z` is now 75.7% of the `data.zarr` payload.

## Build

11h35m at format 2.0, **13h30m** at format 3.0 (+16.5%). The SE fit,
measurement and rewrite run inside every Dense build from format 3.0 onwards.

## Query latency

| Query | Cells | 2.0 (ms) | 3.0 (ms) | Change |
|---|---:|---:|---:|---:|
| bulk Analysis | 8,419,893 | 29,063.17 | 25,535.45 | **−12.1%** |
| regional, all Analyses | 5,224,822 | 786.83 | 1,773.15 | **+125.4%** |
| regional, one Analysis | 3,197 | 22.75 | 23.85 | +4.8% |
| phewas | 1,341 | 6.56 | 5.92 | −9.7% |
| random lookup, 10 variants × 100 Analyses | 637 | 79.42 | 85.65 | +7.8% |
| random lookup, 100 variants × 10 Analyses | 517 | 675.59 | 654.35 | −3.1% |
| per-Analysis top hits | 7,390 | 78.97 | 1.36 | *not comparable — see below* |

All result counts matched, and the MR IVW validation agrees to three decimal
places (`beta` 0.1151 vs 0.1150, 66 instruments both).

**The two large changes have different causes, and only one is the encoding.**

A wide scan pays for the decode. Reading one band directly through
`DenseSePlane`, 52,253,910 cells: **0.0050 µs/cell at `float16`, 0.0700 µs/cell
at `int8_residual` — 14×**, because reconstructing `se` needs `eaf` decoded and
an `exp` per cell. That is the whole of the all-Analysis `regional` regression.
Restricting the same region to one Analysis through `lookup` changes only 4.8%
(22.75→23.85 ms), because the decode is no longer multiplied across all 2,511
Analyses. The two random-lookup shapes likewise move by less than 8% in opposite
directions; scattered row-chunk access matters more than requested cell count.

A whole-Analysis scan is IO-bound, so the same encoding *wins*: the plane is
5.3× smaller and `bulk` gets 12.1% faster despite the more expensive decode.

**Per-Analysis top hits is not an `se` result.** The format-2.0 release predates
issues #131–#134 and its `top_hits/p_5e_08` group has no `eaf` array, so it
takes the pre-fix path; the script's own constants record that regression at
86.6 ms and the pre-EAF baseline at 1.17 ms. The 3.0 store's 1.36 ms is the
indexed path, not a benefit of residual `se`.

## Follow-up

ADR 0037 §3's query-latency figures (173→179 ms top hits; 7,631→11,312 ms
Analysis scan) were measured against a FinnGen store whose top-hit index was
stale, and its −58.1% storage estimate is contradicted here. Both want
re-running before the ADR is trusted.

Raw output is in `opengwasdb_ukbb_dense_issue148_se2_benchmark.json` and
`opengwasdb_ukbb_dense_issue148_se3_benchmark.json`.
