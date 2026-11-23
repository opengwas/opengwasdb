# The fast reader's tables (ogld#32)

_Rendered by `scripts/ld/bench/deltas.py` from `results/32/` and ogld#8's step-1 records in
`results/`; see [`fast-reader.md`](fast-reader.md). Seconds before → after, and how many
times faster._

## ogld#8's step-1 reads, before and after ogld#32

| read | requests | local | +10 ms | +50 ms |
|---|---|---|---|---|
| clump, r² 0.01, cold, sum of 4 | 18,408 → 7,537 | 91.02 s → 52.22 s (1.7×) | 349 s → 102 s (3.4×) | 1,110 s → 224 s (5.0×) |
| clump, r² 0.001, cold, sum of 4 | 12,716 → 6,887 | 71.94 s → 46.39 s (1.6×) | 246 s → 90.88 s (2.7×) | 766 s → 205 s (3.7×) |
| region 3 Mb, cold, median of 3 | 58 → 45 | 1.47 s → 1.46 s (1.0×) | 2.35 s → 2.05 s (1.1×) | 4.00 s → 3.79 s (1.1×) |
| region 3 Mb, warm, median of 3 | 0 → 0 | 895 ms → 896 ms (1.0×) | 901 ms → 920 ms (1.0×) | 678 ms → 1.10 s (0.6×) |
| region 10 Mb, cold, median of 3 | 224 → 149 | 8.37 s → 10.35 s (0.8×) | 11.09 s → 13.11 s (0.8×) | 18.42 s → 17.90 s (1.0×) |
| region 10 Mb, warm, median of 3 | 205 → 0 | 8.19 s → 8.01 s (1.0×) | 10.63 s → 7.51 s (1.4×) | 17.46 s → 6.78 s (2.6×) |
| 500 scattered, chr22, cold, median of 1 | 2,245 → 1,241 | 6.56 s → 1.80 s (3.6×) | 40.02 s → 5.79 s (6.9×) | 129 s → 14.77 s (8.7×) |
| 500 scattered, chr22, open, median of 1 | 2,226 → 1,222 | 5.99 s → 1.93 s (3.1×) | 39.52 s → 5.59 s (7.1×) | 128 s → 13.73 s (9.3×) |
| 500 scattered, chr22, warm, median of 1 | 1,512 → 0 | 2.92 s → 116 ms (25.1×) | 26.21 s → 116 ms (226.1×) | 86.27 s → 115 ms (749.1×) |
| 500 scattered, genome, cold, median of 1 | 3,726 → 3,059 | 17.02 s → 9.03 s (1.9×) | 75.12 s → 18.24 s (4.1×) | 222 s → 41.78 s (5.3×) |
| 500 scattered, genome, open, median of 1 | 3,502 → 2,856 | 20.03 s → 7.85 s (2.6×) | 69.64 s → 14.38 s (4.8×) | 209 s → 29.96 s (7.0×) |
| 500 scattered, genome, warm, median of 1 | 3,312 → 0 | 13.23 s → 787 ms (16.8×) | 62.58 s → 726 ms (86.1×) | 196 s → 768 ms (255.3×) |
| proxies, 1 variant, cold, median of 10 | 26 → 26 | 106 ms → 92 ms (1.2×) | 403 ms → 408 ms (1.0×) | 1.30 s → 1.32 s (1.0×) |
| proxies, 1 variant, open, median of 10 | 6 → 6 | 31 ms → 29 ms (1.0×) | 107 ms → 108 ms (1.0×) | 352 ms → 356 ms (1.0×) |
| proxies, 1 variant, warm, median of 10 | 0 → 0 | 1.6 ms → 1.5 ms (1.0×) | 1.5 ms → 1.6 ms (0.9×) | 1.6 ms → 1.7 ms (0.9×) |
| proxies, 10 variants, cold, median of 1 | 150 → 142 | 846 ms → 578 ms (1.5×) | 2.68 s → 2.25 s (1.2×) | 8.18 s → 6.89 s (1.2×) |
| proxies, 10 variants, open, median of 1 | 58 → 55 | 285 ms → 145 ms (2.0×) | 1.02 s → 630 ms (1.6×) | 3.40 s → 2.06 s (1.7×) |
| proxies, 10 variants, warm, median of 1 | 0 → 0 | 12 ms → 12 ms (1.0×) | 11 ms → 12 ms (0.9×) | 12 ms → 13 ms (0.9×) |
| proxies, 100 variants, cold, median of 1 | 753 → 669 | 3.75 s → 2.16 s (1.7×) | 14.26 s → 6.50 s (2.2×) | 44.30 s → 18.54 s (2.4×) |
| proxies, 100 variants, open, median of 1 | 540 → 475 | 1.83 s → 1.06 s (1.7×) | 10.42 s → 2.73 s (3.8×) | 32.09 s → 7.10 s (4.5×) |
| proxies, 100 variants, warm, median of 1 | 0 → 0 | 86 ms → 88 ms (1.0×) | 89 ms → 89 ms (1.0×) | 86 ms → 89 ms (1.0×) |

## Each lever alone, at +10 ms

| read | levers on | requests | MB read | seconds | load |
|---|---|---|---|---|---|
| 500 scattered, chr22, cold | none (ogld#8's reader) | 2,245 | 125.3 | 37.02 s | 15–17 |
| 500 scattered, chr22, cold | cache | 2,243 | 124.6 | 37.42 s | 18–21 |
| 500 scattered, chr22, cold | concurrency | 2,245 | 125.3 | 7.07 s | 11–12 |
| 500 scattered, chr22, cold | count | 2,245 | 125.3 | 35.30 s | 5–5 |
| 500 scattered, chr22, cold | indexes | 1,243 | 123.0 | 22.25 s | 6–7 |
| 500 scattered, chr22, cold | all four | 1,241 | 122.3 | 7.07 s | 12–13 |
| 500 scattered, chr22, open | none (ogld#8's reader) | 2,226 | 124.3 | 37.44 s | 15–17 |
| 500 scattered, chr22, open | cache | 2,224 | 123.6 | 37.81 s | 21–22 |
| 500 scattered, chr22, open | concurrency | 2,226 | 124.3 | 7.14 s | 10–11 |
| 500 scattered, chr22, open | count | 2,226 | 124.3 | 35.66 s | 4–5 |
| 500 scattered, chr22, open | indexes | 1,224 | 122.0 | 22.52 s | 6–6 |
| 500 scattered, chr22, open | all four | 1,222 | 121.3 | 6.53 s | 12–12 |
| 500 scattered, chr22, warm | none (ogld#8's reader) | 1,512 | 28.9 | 27.53 s | 17–17 |
| 500 scattered, chr22, warm | cache | 0 | 0.000 | 240 ms | 22–22 |
| 500 scattered, chr22, warm | concurrency | 1,512 | 28.9 | 4.56 s | 10–10 |
| 500 scattered, chr22, warm | count | 1,512 | 28.9 | 25.27 s | 4–4 |
| 500 scattered, chr22, warm | indexes | 851 | 28.1 | 15.30 s | 6–6 |
| 500 scattered, chr22, warm | all four | 0 | 0.000 | 235 ms | 12–12 |
| 500 scattered, genome, cold | none (ogld#8's reader) | 3,726 | 186.3 | 69.06 s | 17–30 |
| 500 scattered, genome, cold | cache | 3,642 | 176.8 | 74.05 s | 23–24 |
| 500 scattered, genome, cold | concurrency | 3,726 | 186.3 | 22.87 s | 9–10 |
| 500 scattered, genome, cold | count | 3,726 | 186.3 | 72.06 s | 4–5 |
| 500 scattered, genome, cold | indexes | 3,106 | 185.9 | 61.68 s | 5–6 |
| 500 scattered, genome, cold | all four | 3,059 | 176.0 | 20.81 s | 10–12 |
| 500 scattered, genome, open | none (ogld#8's reader) | 3,502 | 119.7 | 67.07 s | 30–35 |
| 500 scattered, genome, open | cache | 3,418 | 110.2 | 62.77 s | 15–23 |
| 500 scattered, genome, open | concurrency | 3,502 | 119.7 | 18.04 s | 8–9 |
| 500 scattered, genome, open | count | 3,502 | 119.7 | 68.86 s | 5–5 |
| 500 scattered, genome, open | indexes | 2,903 | 119.4 | 56.68 s | 5–6 |
| 500 scattered, genome, open | all four | 2,856 | 109.5 | 16.06 s | 10–11 |
| 500 scattered, genome, warm | none (ogld#8's reader) | 3,312 | 98.0 | 60.10 s | 28–35 |
| 500 scattered, genome, warm | cache | 0 | 0.000 | 875 ms | 15–15 |
| 500 scattered, genome, warm | concurrency | 3,312 | 98.0 | 15.99 s | 8–8 |
| 500 scattered, genome, warm | count | 3,312 | 98.0 | 59.92 s | 5–24 |
| 500 scattered, genome, warm | indexes | 2,800 | 97.8 | 51.07 s | 5–6 |
| 500 scattered, genome, warm | all four | 0 | 0.000 | 830 ms | 11–11 |
| clump GCST90319613, r² 0.01, cold | none (ogld#8's reader) | 1,832 | 375.9 | 39.33 s | 17–28 |
| clump GCST90319613, r² 0.01, cold | cache | 1,796 | 368.1 | 38.70 s | 13–15 |
| clump GCST90319613, r² 0.01, cold | concurrency | 1,832 | 375.9 | 17.45 s | 7–8 |
| clump GCST90319613, r² 0.01, cold | count | 1,832 | 375.9 | 38.56 s | 21–24 |
| clump GCST90319613, r² 0.01, cold | indexes | 1,212 | 374.7 | 28.53 s | 5–5 |
| clump GCST90319613, r² 0.01, cold | all four | 1,184 | 366.9 | 17.33 s | 10–11 |
| clump GCST90319613, r² 0.01, warm | none (ogld#8's reader) | 1,606 | 308.0 | 33.43 s | 12–17 |
| clump GCST90319613, r² 0.01, warm | cache | 0 | 0.000 | 781 ms | 13–13 |
| clump GCST90319613, r² 0.01, warm | concurrency | 1,606 | 308.0 | 13.17 s | 7–7 |
| clump GCST90319613, r² 0.01, warm | count | 1,606 | 308.0 | 33.73 s | 18–21 |
| clump GCST90319613, r² 0.01, warm | indexes | 1,005 | 306.8 | 24.10 s | 5–5 |
| clump GCST90319613, r² 0.01, warm | all four | 0 | 0.000 | 613 ms | 10–10 |

