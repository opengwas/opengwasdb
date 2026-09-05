# Reference-completed ukb-b EAF rechunk benchmark (issue #135)

Measured 2026-09-05 against a copy-on-write clone of the format-2.0,
Reference-Completed `ukb-b` store (11,192,757 variants × 2,511 Analyses).
Each result is the median of five warmed repetitions using the same code and
store data. The only store changes were the chunks of `eaf_baseline` and
`eaf_reference`: both changed from `(11192757,)` to `(1000,)`.

| Query | Cells | Before (ms) | After (ms) | Change |
|---|---:|---:|---:|---:|
| bulk Analysis | 10,003,315 | 33,327.52 | 30,833.51 | -7.5% |
| phewas | 2,511 | 38.23 | 7.18 | -81.2% |
| regional | 9,659,830 | 1,203.80 | 1,145.90 | -4.8% |
| per-Analysis top hits | 8,595 | 168.04 | 148.65 | -11.5% |
| random lookup | 890 | 739.80 | 797.52 | +7.8% |

All result counts matched. The MR validation was unchanged (`beta=0.1088`,
49 instruments). The decoded arrays were identical before and after repair:

- `eaf_baseline` SHA-256:
  `8081a020f439498d186b9e83cc50b93744408d04185239c67d9b110d10f2ff0e`
- `eaf_reference` SHA-256:
  `69681f6fb044708877d9491899d3fdbb267cfde0159d946b40e1e4ae543d52de`

The repaired clone is
`/data/opengwasdb/wip/rebuild-117/ukb-b__dense-observed-vcf-c128-completed-issue135-benchmark`.
The completed source store was left unchanged.

Raw post-repair output is in
`opengwasdb_ukbb_dense_completed_issue135_benchmark.json`.
