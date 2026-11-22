# The Far Pairs beside the Proxy Index: reads before and after (ogld#31)

_Measured 12–13 November 2026, with ogld#8's benchmark (`scripts/bench/latency.py`) on the
published HGDP+1kGP EUR store. The records are in [`results/far-pairs/`](results/far-pairs/);
ogld#8's own are in [`results/`](results/), and its method, targets and states are in
[`latency.md`](latency.md)._

## In brief

- **A matrix of 500 variants drawn uniformly on one chromosome** (one draw, seed 8) reads its
  pairs beyond 3 Mb from the Far Pairs, never from the index. On chr22 that is 10 chunks where it
  was 894, and the read takes 53 s at
  50 ms a request where it took 129 s. On chr11, where the Far Pairs are densest (161.5 a
  variant against chr22's 0.85), it is 102 chunks where it was 1,058, and 63 s at 50 ms where
  it was 161 s. Its requests fall by 61% on chr22 (2,245 to 883) and by 62% on chr11 (2,827
  to 1,083), and its far part from 77 s to 0.5 s and from 105 s to 6.7 s at 50 ms. Across
  ogld#8's uniform draws the Far Pairs touch 10–91 times fewer chunks than the index; a list
  clustered where they are densest touches about 2–3 times fewer (1.7 for 500 rows at chr11's
  45–55 Mb; "How the Far Pairs are chunked", below).
- **A 10 Mb region** reads its rows' Far Pairs where it read their whole index rows. Where Far
  Pairs are sparse, as on chr22 and at chr11's 25 and 40 Mb, that part fell from 1.7–5.1 s to
  12–491 ms; the rest of such a region is its Near Band read, which did not change, so its total
  moved within the run-to-run noise ogld#8 measured, and was faster in most cells. Where they
  are densest, chr11's 50 Mb, its rows hold 68.75 M Far Pairs against 196 M index entries: it
  read 203 MB of Far Pairs rows against 709 MB of index rows, and took 12 s locally and 24 s at
  50 ms where it took 30 s and 47 s.
- **The answers are the same,** byte for byte, NaN included (ogld#31's comparison on chr22 and
  the Test Store), and so is every other array of the store.

## What was measured, and on what

The store is `hgdp1kgp-v2-EUR-k10.store`, whose Far Pairs were derived in place from its own
index (ADR 0007): every array the index-only store held is unchanged, so **before** and
**after** read the same bytes and differ only in the reader.

- **Before** is the reader at `orch/1-ld-stores@1cb5c6e`, which reads a matrix's and a
  region's pairs beyond 3 Mb from Proxy Index rows and ignores `far/`. For chr22 it is
  ogld#8's own records (12 November, `latency.md`, step 1). For chr11 it was run here, from this
  branch's benchmark with that reader first on `PYTHONPATH`; each record's environment names
  the source it imported (`ogld_source`).
- **After** is this branch's reader.

The reads are ogld#8's step 1: a matrix of 500 variants drawn on the chromosome with seed 8,
cold, open and warm; and regions 10 Mb wide, cold and warm. chr22's regions are centred at 25,
32 and 40 Mb, as ogld#8's were. chr11's are centred at 25, 40 and 50 Mb: chr11 holds 69.0 M
Far Pairs, 68.75 M of them in rows at 45–55 Mb, so 25 and 40 Mb are typical of the chromosome
and 50 Mb its heaviest (`results/far_pairs.json`, ogld#8's per-chromosome counts). Targets are
local disk, and HTTP with 0, 10 and 50 ms added per request.

**When.** ogld#8's lane read about 6 GB of this store's `index/<c>/partner` three times between
23:50 and 00:20 UTC on 12–13 November. No timing here was taken in that window: chr22's after
ran 23:26–23:43, and chr11's before and after were run from 00:22. A chr11 run made inside it
was discarded.

**Commands.**

```bash
pixi run -e dev bench-latency step1 --target local http0 http10 http50 --case scattered region \
    --seeds 1 --out docs/benchmarks/results/far-pairs/step1-chr22-after.jsonl
PYTHONPATH=<a checkout of orch/1-ld-stores@1cb5c6e> pixi run -e dev bench-latency step1 \
    --target local http0 http10 http50 --case scattered region --seeds 1 --chrom 11 \
    --centres 25000000 40000000 50000000 --widths 10 \
    --out docs/benchmarks/results/far-pairs/step1-chr11-before.jsonl
pixi run -e dev bench-latency step1 ... --out docs/benchmarks/results/far-pairs/step1-chr11-after.jsonl
pixi run -e dev python scripts/bench/report.py docs/benchmarks/results/far-pairs/<file>
```

Each file is kept gzipped, as ogld#8 keeps its records, and `report.py` reads it so. The tables
below put each read's before and after side by side; their numbers are the records', with
`report.py`'s helpers.

## chr22: 0.85 Far Pairs a variant

**500 variants on chr22** (seed 8; one draw):

| target | state | time, before → after | requests | index chunks → Far Pairs chunks | far part | load, before / after |
|---|---|---|---|---|---|---|
| local | cold | 6.56 s → 3.47 s | 2,245 → 883 | 894 → 10 | 2.60 s → 29 ms | 3–3 / 8–19 |
| local | open | 5.99 s → 2.72 s | 2,226 → 860 | 894 → 10 | 2.17 s → 13 ms | 3–4 / 18–19 |
| local | warm | 2.92 s → 80 ms | 1,512 → 0 | 894 → 0 | 2.19 s → 0.6 ms | 4–4 / 18–18 |
| http0 | cold | 9.95 s → 5.08 s | 2,245 → 883 | 894 → 10 | 4.60 s → 41 ms | 15–16 / 9–9 |
| http0 | open | 10.23 s → 4.88 s | 2,226 → 860 | 894 → 10 | 4.56 s → 24 ms | 16–18 / 9–9 |
| http0 | warm | 6.65 s → 80 ms | 1,512 → 0 | 894 → 0 | 5.44 s → 0.6 ms | 18–18 / 9–9 |
| http10 | cold | 40.02 s → 14.46 s | 2,245 → 883 | 894 → 10 | 23.30 s → 136 ms | 3–4 / 8–8 |
| http10 | open | 39.52 s → 14.29 s | 2,226 → 860 | 894 → 10 | 23.26 s → 69 ms | 3–3 / 7–8 |
| http10 | warm | 26.21 s → 82 ms | 1,512 → 0 | 894 → 0 | 23.31 s → 0.6 ms | 3–3 / 7–7 |
| http50 | cold | 129 s → 53.27 s | 2,245 → 883 | 894 → 10 | 77.34 s → 498 ms | 2–5 / 13–24 |
| http50 | open | 128 s → 52.06 s | 2,226 → 860 | 894 → 10 | 77.29 s → 246 ms | 2–5 / 10–13 |
| http50 | warm | 86.27 s → 130 ms | 1,512 → 0 | 894 → 0 | 77.58 s → 0.7 ms | 2–2 / 10–10 |

**Regions 10 Mb wide on chr22**, each centre:

| centre | target | state | variants | time, before → after | MB read | index rows MB → Far Pairs rows MB | far part | load, before / after |
|---|---|---|---|---|---|---|---|---|
| 25 Mb | local | cold | 33,475 | 10.63 s → 7.07 s | 313 → 295 | 18.2 → 0.06 | 2.41 s → 27 ms | 4–4 / 11–12 |
| 25 Mb | local | warm | 33,475 | 10.10 s → 8.04 s | 312 → 294 | 18.2 → 0.06 | 2.24 s → 15 ms | 4–4 / 11–11 |
| 25 Mb | http0 | cold | 33,475 | 16.13 s → 9.19 s | 313 → 295 | 18.2 → 0.06 | 2.59 s → 52 ms | 13–14 / 6–7 |
| 25 Mb | http0 | warm | 33,475 | 11.16 s → 7.97 s | 312 → 294 | 18.2 → 0.06 | 2.32 s → 30 ms | 12–13 / 6–6 |
| 25 Mb | http10 | cold | 33,475 | 11.09 s → 17.17 s | 313 → 295 | 18.2 → 0.06 | 1.67 s → 177 ms | 3–3 / 8–8 |
| 25 Mb | http10 | warm | 33,475 | 10.63 s → 15.59 s | 312 → 294 | 18.2 → 0.06 | 1.69 s → 87 ms | 3–3 / 8–18 |
| 25 Mb | http50 | cold | 33,475 | 20.80 s → 20.82 s | 313 → 295 | 18.2 → 0.06 | 2.95 s → 479 ms | 2–2 / 7–8 |
| 25 Mb | http50 | warm | 33,475 | 17.51 s → 18.16 s | 312 → 294 | 18.2 → 0.06 | 1.98 s → 247 ms | 2–2 / 6–7 |
| 32 Mb | local | cold | 33,788 | 8.37 s → 6.72 s | 314 → 287 | 26.7 → 0.07 | 2.25 s → 25 ms | 4–4 / 11–11 |
| 32 Mb | local | warm | 33,788 | 8.19 s → 6.48 s | 313 → 286 | 26.7 → 0.07 | 2.22 s → 13 ms | 4–4 / 10–11 |
| 32 Mb | http0 | cold | 33,788 | 9.36 s → 7.45 s | 314 → 287 | 26.7 → 0.07 | 2.35 s → 53 ms | 11–12 / 6–7 |
| 32 Mb | http0 | warm | 33,788 | 10.08 s → 7.27 s | 313 → 286 | 26.7 → 0.07 | 2.59 s → 29 ms | 11–11 / 7–7 |
| 32 Mb | http10 | cold | 33,788 | 11.24 s → 13.76 s | 314 → 287 | 26.7 → 0.07 | 2.42 s → 142 ms | 3–3 / 18–27 |
| 32 Mb | http10 | warm | 33,788 | 13.61 s → 11.65 s | 313 → 286 | 26.7 → 0.07 | 3.83 s → 88 ms | 3–3 / 27–27 |
| 32 Mb | http50 | cold | 33,788 | 18.42 s → 20.25 s | 314 → 287 | 26.7 → 0.07 | 2.84 s → 491 ms | 2–3 / 6–6 |
| 32 Mb | http50 | warm | 33,788 | 17.46 s → 19.00 s | 313 → 286 | 26.7 → 0.07 | 2.80 s → 247 ms | 3–3 / 6–6 |
| 40 Mb | local | cold | 29,544 | 7.56 s → 4.90 s | 260 → 243 | 17.3 → 0.04 | 2.17 s → 24 ms | 4–4 / 10–10 |
| 40 Mb | local | warm | 29,544 | 8.06 s → 4.80 s | 259 → 242 | 17.3 → 0.04 | 2.20 s → 12 ms | 4–4 / 9–10 |
| 40 Mb | http0 | cold | 29,544 | 10.33 s → 5.48 s | 260 → 243 | 17.3 → 0.04 | 2.87 s → 48 ms | 11–14 / 7–8 |
| 40 Mb | http0 | warm | 29,544 | 12.68 s → 5.44 s | 259 → 242 | 17.3 → 0.04 | 3.16 s → 28 ms | 14–15 / 8–8 |
| 40 Mb | http10 | cold | 29,544 | 10.34 s → 8.95 s | 260 → 243 | 17.3 → 0.04 | 2.53 s → 153 ms | 3–3 / 25–27 |
| 40 Mb | http10 | warm | 29,544 | 10.22 s → 8.63 s | 259 → 242 | 17.3 → 0.04 | 2.58 s → 249 ms | 3–4 / 24–25 |
| 40 Mb | http50 | cold | 29,544 | 13.61 s → 11.98 s | 260 → 243 | 17.3 → 0.04 | 1.97 s → 467 ms | 3–3 / 6–6 |
| 40 Mb | http50 | warm | 29,544 | 13.97 s → 10.84 s | 259 → 242 | 17.3 → 0.04 | 2.62 s → 233 ms | 3–3 / 5–6 |

ogld#8's before ran at a load of 2–18, the after at 5–27. A region's total is its Near Band
read, 242–314 MB in 162–224 requests, and its far part was 1.5–3.7 s shorter in every cell. The
after was slower in 7 of 24 region cells: at 10 ms at 25 and 32 Mb (by 2–6 s, at a load of 8–27
against 3) and at 50 ms at 25 and 32 Mb (by up to 10%). ogld#8 found local medians moving by up
to 45% between runs of the same code.

## chr11: 161.5 Far Pairs a variant

**500 variants on chr11** (seed 8; one draw):

| target | state | time, before → after | requests | index chunks → Far Pairs chunks | far part | load, before / after |
|---|---|---|---|---|---|---|
| local | cold | 8.86 s → 5.46 s | 2,827 → 1,083 | 1,058 → 102 | 4.23 s → 355 ms | 9–10 / 6–6 |
| local | open | 10.88 s → 4.72 s | 2,808 → 1,060 | 1,058 → 102 | 5.40 s → 244 ms | 8–9 / 6–6 |
| local | warm | 6.65 s → 1.33 s | 2,132 → 216 | 1,058 → 102 | 4.74 s → 262 ms | 7–8 / 5–6 |
| http0 | cold | 17.79 s → 6.24 s | 2,827 → 1,083 | 1,058 → 102 | 11.29 s → 389 ms | 4–4 / 5–6 |
| http0 | open | 19.83 s → 6.13 s | 2,808 → 1,060 | 1,058 → 102 | 11.40 s → 394 ms | 4–4 / 6–6 |
| http0 | warm | 14.57 s → 1.53 s | 2,132 → 216 | 1,058 → 102 | 11.43 s → 346 ms | 4–4 / 6–6 |
| http10 | cold | 45.39 s → 18.08 s | 2,827 → 1,083 | 1,058 → 102 | 29.07 s → 1.82 s | 11–17 / 5–5 |
| http10 | open | 47.24 s → 17.82 s | 2,808 → 1,060 | 1,058 → 102 | 29.16 s → 1.69 s | 8–11 / 5–5 |
| http10 | warm | 36.94 s → 4.37 s | 2,132 → 216 | 1,058 → 102 | 31.61 s → 1.92 s | 5–8 / 5–5 |
| http50 | cold | 161 s → 63.43 s | 2,827 → 1,083 | 1,058 → 102 | 105 s → 6.73 s | 7–17 / 4–5 |
| http50 | open | 163 s → 62.51 s | 2,808 → 1,060 | 1,058 → 102 | 106 s → 6.46 s | 17–35 / 4–5 |
| http50 | warm | 123 s → 13.18 s | 2,132 → 216 | 1,058 → 102 | 106 s → 6.45 s | 14–35 / 5–5 |

**Regions 10 Mb wide on chr11**, each centre:

| centre | target | state | variants | time, before → after | MB read | index rows MB → Far Pairs rows MB | far part | load, before / after |
|---|---|---|---|---|---|---|---|---|
| 25 Mb | local | cold | 36,471 | 19.56 s → 10.50 s | 328 → 302 | 25.7 → 0.07 | 5.12 s → 53 ms | 6–7 / 5–5 |
| 25 Mb | local | warm | 36,471 | 11.04 s → 9.10 s | 325 → 299 | 25.7 → 0.07 | 2.60 s → 12 ms | 6–6 / 5–5 |
| 25 Mb | http0 | cold | 36,471 | 14.85 s → 10.68 s | 328 → 302 | 25.7 → 0.07 | 3.84 s → 50 ms | 4–5 / 6–6 |
| 25 Mb | http0 | warm | 36,471 | 11.41 s → 10.97 s | 325 → 299 | 25.7 → 0.07 | 2.62 s → 36 ms | 4–5 / 6–6 |
| 25 Mb | http10 | cold | 36,471 | 17.59 s → 11.75 s | 328 → 302 | 25.7 → 0.07 | 4.00 s → 150 ms | 5–5 / 5–5 |
| 25 Mb | http10 | warm | 36,471 | 16.93 s → 10.95 s | 325 → 299 | 25.7 → 0.07 | 3.83 s → 70 ms | 5–13 / 5–5 |
| 25 Mb | http50 | cold | 36,471 | 25.83 s → 21.90 s | 328 → 302 | 25.7 → 0.07 | 4.51 s → 489 ms | 11–14 / 5–5 |
| 25 Mb | http50 | warm | 36,471 | 26.29 s → 20.37 s | 325 → 299 | 25.7 → 0.07 | 4.90 s → 237 ms | 10–11 / 5–5 |
| 40 Mb | local | cold | 31,266 | 7.86 s → 5.99 s | 247 → 226 | 20.9 → 0.06 | 2.16 s → 62 ms | 5–6 / 5–5 |
| 40 Mb | local | warm | 31,266 | 7.74 s → 5.76 s | 244 → 223 | 20.9 → 0.06 | 2.14 s → 13 ms | 5–5 / 5–5 |
| 40 Mb | http0 | cold | 31,266 | 10.35 s → 6.40 s | 247 → 226 | 20.9 → 0.06 | 3.11 s → 57 ms | 4–4 / 6–6 |
| 40 Mb | http0 | warm | 31,266 | 10.17 s → 6.26 s | 244 → 223 | 20.9 → 0.06 | 3.11 s → 27 ms | 4–5 / 6–6 |
| 40 Mb | http10 | cold | 31,266 | 12.09 s → 8.05 s | 247 → 226 | 20.9 → 0.06 | 3.24 s → 150 ms | 13–14 / 5–5 |
| 40 Mb | http10 | warm | 31,266 | 10.05 s → 7.62 s | 244 → 223 | 20.9 → 0.06 | 2.33 s → 72 ms | 13–14 / 5–5 |
| 40 Mb | http50 | cold | 31,266 | 19.48 s → 15.65 s | 247 → 226 | 20.9 → 0.06 | 4.03 s → 490 ms | 9–9 / 5–5 |
| 40 Mb | http50 | warm | 31,266 | 19.93 s → 14.49 s | 244 → 223 | 20.9 → 0.06 | 4.49 s → 239 ms | 8–9 / 5–5 |
| 50 Mb | local | cold | 21,425 | 29.50 s → 12.11 s | 861 → 354 | 709.2 → 202.86 | 26.28 s → 8.83 s | 4–5 / 5–5 |
| 50 Mb | local | warm | 21,425 | 27.87 s → 12.02 s | 857 → 351 | 709.2 → 202.86 | 24.92 s → 8.82 s | 4–4 / 5–5 |
| 50 Mb | http0 | cold | 21,425 | 30.33 s → 12.74 s | 861 → 354 | 709.2 → 202.86 | 26.88 s → 9.17 s | 5–6 / 6–6 |
| 50 Mb | http0 | warm | 21,425 | 43.77 s → 12.58 s | 857 → 351 | 709.2 → 202.86 | 40.12 s → 9.08 s | 6–17 / 5–6 |
| 50 Mb | http10 | cold | 21,425 | 31.60 s → 14.17 s | 861 → 354 | 709.2 → 202.86 | 27.11 s → 9.59 s | 9–13 / 5–5 |
| 50 Mb | http10 | warm | 21,425 | 43.54 s → 13.65 s | 857 → 351 | 709.2 → 202.86 | 38.93 s → 9.42 s | 7–9 / 5–5 |
| 50 Mb | http50 | cold | 21,425 | 47.00 s → 24.34 s | 861 → 354 | 709.2 → 202.86 | 38.15 s → 15.45 s | 7–8 / 5–5 |
| 50 Mb | http50 | warm | 21,425 | 54.14 s → 22.52 s | 857 → 351 | 709.2 → 202.86 | 45.62 s → 14.73 s | 6–7 / 5–22 |

At 50 Mb the far part is the scatter itself: 68.75 M pairs read and matched against the
region's rows, 8.8 s locally. The index rows held those pairs too, and 127 M entries more:
the same Far Pairs again in their later rows, and the Near Band's pairs at r̂² ≥ t_far.

## How the Far Pairs are chunked

They keep the index's shape, 16,384 entries an inner chunk and 64 to a shard (spec §6;
ADR 0007, decision 6). The counts below are from each chromosome's `indptr`, for the same 500
rows (seed 8): chunks of each array, `partner` and `r` alike, the runs of neighbouring chunks
among them, each one Zarr call, and what the chunks decode to.

| chromosome | Far Pairs | the index, at 16,384 | Far Pairs at 4,096 | at 16,384 | at 65,536 | at 262,144 |
|---|---|---|---|---|---|---|
| 22 | 101,886 | 447 chunks in 339 runs | 15 in 1, 0.4 MiB | **5 in 1**, 0.5 MiB | 2 in 1, 0.8 MiB | 1 in 1, 1.5 MiB |
| 11 | 69,017,129 | 529 in 463 | 98 in 37, 2.3 MiB | **51 in 28**, 4.8 MiB | 33 in 26, 12.4 MiB | 26 in 24, 39.0 MiB |
| 6 | 12,150,924 | 526 in 476 | 74 in 46, 1.7 MiB | **45 in 21**, 4.2 MiB | 25 in 13, 9.4 MiB | 16 in 10, 24.0 MiB |
| 11, clustered at 45–55 Mb | 68,750,767 in those rows | 748 in 427 | 725 in 348, 17 MiB | **440 in 313**, 41 MiB | 328 in 198, 123 MiB | 196 in 48, 294 MiB |

The first three rows are one uniform draw each (seed 8); the last is 500 rows drawn at chr11's
45–55 Mb, the other end. On the uniform draws, at 16,384 a list's Far Pairs make 10–17 times
fewer runs than its index rows; a smaller chunk adds runs where rows are long, and a larger one
removes few and decodes 2.6–8 times the bytes. On the clustered draw the Far Pairs touch only
1.7 times fewer chunks than the index, and a larger chunk does remove runs (198 and 48 against
313), for 3 and 7 times the bytes. Across ogld#8's uniform draws the ratio of index to Far Pairs
chunks is 10–91 (10–29 on the eight chromosomes with more than 10 Far Pairs a variant), and the
spec review's list, 300 of its 500 rows at 44–56 Mb, gave 2.8 (438 against 1,248). The shape is
kept for the uniform lists: ogld#32 fetches a read's runs together, and a chunk shape is not a
format version (spec §13). Shards of 1,048,576 to 16,777,216 entries leave chr11's 28 runs
at 28 within shards, and a writer holds one shard of each array, so the shard stays 64 chunks.
Draws with seeds 9 and 10 touch 32 and 45 Far Pairs chunks on chr11, and 6 on chr22
(`results/far_pairs.json`).
