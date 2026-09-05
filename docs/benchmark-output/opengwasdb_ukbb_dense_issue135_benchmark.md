# ukb-b EAF baseline rechunk benchmark (issue #135)

Measured 2026-09-05 against a copy-on-write clone of the format-2.0
Observed-Only `ukb-b` store (9,847,701 variants × 2,511 Analyses). Each result
is the median of five warmed repetitions using the same code and store data.
The only store change between runs was `data.zarr/eaf_baseline` chunking:
`(9847701,)` before and `(1000,)` after.

| Query | Cells | Before (ms) | After (ms) | Change |
|---|---:|---:|---:|---:|
| bulk Analysis | 8,419,893 | 24,864.50 | 24,998.38 | +0.5% |
| phewas | 1,341 | 20.61 | 8.14 | -60.5% |
| regional | 5,224,822 | 1,146.52 | 1,060.08 | -7.5% |
| per-Analysis top hits | 7,390 | 90.76 | 78.81 | -13.2% |
| random lookup | 447 | 638.36 | 610.38 | -4.4% |

All result counts matched. The MR validation was unchanged (`beta=0.1151`,
66 instruments). The decoded pre/post baseline arrays were byte-identical with
SHA-256 `08f7b0d5435fc27689a133074a4fb90e8159008e1060f542a66ddaab1d73a249`.

The repaired clone is
`/data/opengwasdb/wip/rebuild-117/ukb-b__dense-observed-vcf-c128-issue135-benchmark`.
The source store was left unchanged because reference completion may be reading
it concurrently.

Raw post-repair output is in
`opengwasdb_ukbb_dense_issue135_benchmark.json`.
