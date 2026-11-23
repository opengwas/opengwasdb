# #477 Rust spike: scattered Dense lookups on OGS-00009

Raw data: `rust_spike_ogs00009.json` (every rep, load at each column's start, hardware, the re-chunk report).
Driver: `benchmarks/rust_spike/run_spike.py`; Rust: `benchmarks/rust_spike/` (`zarrs` 0.23.14); re-chunk: `benchmarks/rust_spike/rechunk.py`.

**Setup.** OGS-00009 (Dense, 11,191,070 variants x 2,514 Analyses, format 0.2.0, inner chunk 1000x64, shard
100000x1024), the harness's seed-0 selection, shapes `random_lookup_10_variants_100_analyses` and
`random_lookup_100_variants_10_analyses`. 224-thread Xeon 8480+, 1 TB RAM, XFS on `/dev/mapper/datavg-data`.
Shared node: 1-minute load was 10-19 during every column, so absolute numbers are noisy and the default
`--max-start-load 3.0` never cleared; I used 20. Warm page cache; "cold" is the first call of a fresh process
(arrays open, nothing decoded), not first-touch disk. Each column is a fresh process: 1 cold call + 7 warm calls.
Every candidate's result digests equal the Python reader's (the driver aborts otherwise; none did).

**Candidates.** `python-fused` = the reader as shipped (fused pipeline, one worker, `store/arrays.py:175`).
`python-zarrs` = same reader under `zarrs.ZarrsCodecPipeline` (zarrs-python 0.2.3). `rust-Nthread` = minimal Rust
Dense lookup, chunk groups decoded on N threads. `-1000`/`-100` = inner variant chunk (the shipped 1000, or a
full-axis copy of z, se, eaf, imputed, eaf_baseline, eaf_reference at 100; 52 GB, 57 minutes, every block read back
bitwise equal). `python-fused-threads16` = zarr's own pool at `max_workers=16`, in one process (measured only).

## Attribution

ALID-to-row is **not** the cost. `indices_by_identifiers` (what `lookup` calls; not `by_index`) takes 0.3-0.9 ms warm.
Decode is 99.9 % of the ~1.3 s on the 100x10 shape. The 5x bar applies to the end-to-end lookup, which the decode
step is.

## End-to-end warm medians (ms)

| candidate | 10 var x 100 an | 100 var x 10 an | vs python-fused-1000 |
|---|---|---|---|
| python-fused-1000 (baseline) | 412 | 1276 | 1x |
| python-fused-100 (smaller chunk, no port) | 207 | 838 | 2.0x / 1.5x |
| python-zarrs-1000 | 1047 | 3780 | 0.4x / 0.3x |
| python-fused-threads16-1000 | 926 | 2919 | 0.4x / 0.4x |
| rust-1thread-1000 | 333 | 770 | 1.2x / 1.7x |
| rust-16thread-1000 | 73 | 109 | 5.6x / 11.8x |
| rust-16thread-100 | 66 | 103 | 6.2x / 12.4x |

Full table, cold figures and spreads: below and in the JSON.

## Caveats

* The Rust gain is parallel chunk decode, not the language: one thread is 1.2-1.7x. Python cannot take it today:
  zarr's worker pool is slower here (0.4x) and ADR 0056 pins one worker because a forked worker hangs on it.
* 16 threads on a shared node; a loaded node gives less. The Rust times also include reading three whole exception
  tables per call (~30 ms), left unoptimised.
* The Rust spike is Dense only, reads the planes the Python lookup reads, and resolves ALIDs wider than the 32-byte
  index slot through the caller (one of the 100 selected ALIDs; Python scans the variant table for it). It is not a
  reader: no Ragged, Hybrid, observed-only, rsid, range or top-hits paths.
* A first full run aborted on that over-wide ALID (the spike refuses it rather than guess); the log is kept in scratch.

## All columns

| candidate | shape | stage | cold ms | warm median ms | warm min-max ms |
|---|---|---|---|---|---|
| python-fused-1000 | 10_variants_100_analyses | axis_resolution | 0.4 | 0.26 | 0.25-0.31 |
| python-fused-1000 | 10_variants_100_analyses | decode | 437.8 | 406.87 | 404.36-411.77 |
| python-fused-1000 | 10_variants_100_analyses | end_to_end | 448.5 | 411.94 | 409.58-419.61 |
| python-fused-1000 | 100_variants_10_analyses | axis_resolution | 1.1 | 0.84 | 0.81-0.88 |
| python-fused-1000 | 100_variants_10_analyses | decode | 1339.7 | 1268.43 | 1257.26-1272.79 |
| python-fused-1000 | 100_variants_10_analyses | end_to_end | 1314.4 | 1275.74 | 1266.69-1293.72 |
| python-zarrs-1000 | 10_variants_100_analyses | axis_resolution | 0.4 | 0.24 | 0.16-0.31 |
| python-zarrs-1000 | 10_variants_100_analyses | decode | 1134.8 | 1074.95 | 961.62-1167.00 |
| python-zarrs-1000 | 10_variants_100_analyses | end_to_end | 1194.5 | 1047.25 | 972.53-1176.59 |
| python-zarrs-1000 | 100_variants_10_analyses | axis_resolution | 0.8 | 0.59 | 0.58-0.66 |
| python-zarrs-1000 | 100_variants_10_analyses | decode | 3456.2 | 3251.97 | 2752.51-3633.97 |
| python-zarrs-1000 | 100_variants_10_analyses | end_to_end | 3851.0 | 3779.94 | 3338.60-3945.73 |
| rust-1thread-1000 | 10_variants_100_analyses | axis_resolution | 0.0 | 0.00 | 0.00-0.01 |
| rust-1thread-1000 | 10_variants_100_analyses | decode | 349.1 | 289.99 | 256.00-334.72 |
| rust-1thread-1000 | 10_variants_100_analyses | end_to_end | 352.1 | 332.64 | 325.76-341.64 |
| rust-1thread-1000 | 100_variants_10_analyses | axis_resolution | 0.1 | 0.02 | 0.02-0.04 |
| rust-1thread-1000 | 100_variants_10_analyses | decode | 771.1 | 764.05 | 756.39-773.49 |
| rust-1thread-1000 | 100_variants_10_analyses | end_to_end | 779.3 | 769.60 | 758.27-782.58 |
| rust-16thread-1000 | 10_variants_100_analyses | axis_resolution | 0.0 | 0.00 | 0.00-0.01 |
| rust-16thread-1000 | 10_variants_100_analyses | decode | 78.3 | 71.37 | 68.17-81.93 |
| rust-16thread-1000 | 10_variants_100_analyses | end_to_end | 82.2 | 73.18 | 70.43-83.91 |
| rust-16thread-1000 | 100_variants_10_analyses | axis_resolution | 0.1 | 0.03 | 0.03-0.06 |
| rust-16thread-1000 | 100_variants_10_analyses | decode | 137.8 | 112.04 | 107.79-133.75 |
| rust-16thread-1000 | 100_variants_10_analyses | end_to_end | 128.0 | 108.51 | 102.35-114.78 |
| python-fused-100 | 10_variants_100_analyses | axis_resolution | 0.4 | 0.27 | 0.25-0.33 |
| python-fused-100 | 10_variants_100_analyses | decode | 244.2 | 209.45 | 207.25-211.28 |
| python-fused-100 | 10_variants_100_analyses | end_to_end | 243.7 | 207.24 | 206.00-215.44 |
| python-fused-100 | 100_variants_10_analyses | axis_resolution | 1.1 | 0.84 | 0.82-0.89 |
| python-fused-100 | 100_variants_10_analyses | decode | 1412.3 | 863.69 | 843.66-884.33 |
| python-fused-100 | 100_variants_10_analyses | end_to_end | 939.9 | 838.26 | 823.00-872.76 |
| python-zarrs-100 | 10_variants_100_analyses | axis_resolution | 0.4 | 0.27 | 0.23-0.36 |
| python-zarrs-100 | 10_variants_100_analyses | decode | 879.3 | 841.24 | 829.18-901.60 |
| python-zarrs-100 | 10_variants_100_analyses | end_to_end | 831.7 | 906.06 | 852.60-977.33 |
| python-zarrs-100 | 100_variants_10_analyses | axis_resolution | 0.9 | 0.59 | 0.58-0.66 |
| python-zarrs-100 | 100_variants_10_analyses | decode | 2963.9 | 2823.19 | 2650.27-3022.11 |
| python-zarrs-100 | 100_variants_10_analyses | end_to_end | 3490.4 | 2957.48 | 2605.99-3117.42 |
| rust-1thread-100 | 10_variants_100_analyses | axis_resolution | 0.0 | 0.00 | 0.00-0.01 |
| rust-1thread-100 | 10_variants_100_analyses | decode | 248.4 | 233.30 | 227.90-236.40 |
| rust-1thread-100 | 10_variants_100_analyses | end_to_end | 249.1 | 234.12 | 229.63-237.32 |
| rust-1thread-100 | 100_variants_10_analyses | axis_resolution | 0.1 | 0.02 | 0.02-0.04 |
| rust-1thread-100 | 100_variants_10_analyses | decode | 564.3 | 548.53 | 536.27-564.76 |
| rust-1thread-100 | 100_variants_10_analyses | end_to_end | 569.7 | 561.42 | 508.06-604.90 |
| rust-16thread-100 | 10_variants_100_analyses | axis_resolution | 0.0 | 0.00 | 0.00-0.01 |
| rust-16thread-100 | 10_variants_100_analyses | decode | 86.0 | 65.97 | 59.01-74.29 |
| rust-16thread-100 | 10_variants_100_analyses | end_to_end | 87.6 | 65.92 | 62.01-75.57 |
| rust-16thread-100 | 100_variants_10_analyses | axis_resolution | 0.1 | 0.02 | 0.01-0.04 |
| rust-16thread-100 | 100_variants_10_analyses | decode | 117.8 | 108.97 | 104.73-122.56 |
| rust-16thread-100 | 100_variants_10_analyses | end_to_end | 129.3 | 103.24 | 99.09-113.99 |
| python-fused-threads16-1000 | 10_variants_100_analyses | axis_resolution | 0.4 | 0.26 | 0.16-0.32 |
| python-fused-threads16-1000 | 10_variants_100_analyses | decode | 1096.3 | 962.16 | 918.46-1051.97 |
| python-fused-threads16-1000 | 10_variants_100_analyses | end_to_end | 1011.6 | 925.74 | 907.13-978.81 |
| python-fused-threads16-1000 | 100_variants_10_analyses | axis_resolution | 1.6 | 0.86 | 0.83-0.90 |
| python-fused-threads16-1000 | 100_variants_10_analyses | decode | 3192.8 | 2861.89 | 2797.20-3027.40 |
| python-fused-threads16-1000 | 100_variants_10_analyses | end_to_end | 2905.7 | 2919.28 | 2674.94-2967.53 |
| python-fused-threads16-100 | 10_variants_100_analyses | axis_resolution | 0.4 | 0.26 | 0.23-0.31 |
| python-fused-threads16-100 | 10_variants_100_analyses | decode | 618.7 | 588.62 | 462.72-640.82 |
| python-fused-threads16-100 | 10_variants_100_analyses | end_to_end | 532.6 | 500.55 | 494.97-551.96 |
| python-fused-threads16-100 | 100_variants_10_analyses | axis_resolution | 1.0 | 0.44 | 0.43-0.66 |
| python-fused-threads16-100 | 100_variants_10_analyses | decode | 1966.4 | 2001.79 | 1914.11-2167.24 |
| python-fused-threads16-100 | 100_variants_10_analyses | end_to_end | 2074.0 | 2082.74 | 2020.41-2124.20 |