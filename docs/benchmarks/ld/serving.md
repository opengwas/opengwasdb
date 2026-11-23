# What serving LD costs: a job's clump, a remote client's reads, a worker's memory (ogld#9)

_Measured 13 November 2026 on the published HGDP+1kGP stores: the clump and read runs at ogld
`6bae783` (ogld#31's Far Pairs and ogld#32's reader, both merged), the memory and lag runs at
`1dcd548`, whose `ogld/` is the same. The records are in [`results/9/`](results/9/), and
[`serving-tables.md`](serving-tables.md) is every one of them rendered by `bench-serving tables`.
ogld#8's method, targets and states are in [`latency.md`](latency.md). ADR 0094 uses these
numbers._

## In brief

- **A clump job on the server.** #403's 37 Analyses clump in 442 s (7.4 min) at r² 0.01 and
  373 s at 0.001 from local disk, cold, against ogld#8's 17.4 and 11.9 minutes: 6.3 s an
  Analysis at the median, 69 s at most. Four of the 37 (160,706–260,821 hits) run past
  ogdb-api's 30 s store timeout at r² 0.01, one at 0.001. They read 48.6 GB, where ogld#8's
  reader read 75 GB.
- **A remote client reading static Zarr**, with ogld#31's Far Pairs and ogld#32's reader
  together: a matrix of 500 chr22 variants takes 1.2 s locally, 2.9 s at 10 ms a request, 6.8 s
  at 50 ms and 11.8 s at 100 ms, cold; drawn genome-wide, 8.3, 15.4, 36.7 and 61.2 s, a quarter
  to a third of which over HTTP is loading the touched chromosomes' positions and `indptr`
  arrays. `proxies` for one variant on an opened store takes 29 ms, 108 ms, 346 ms and 655 ms.
- **One process holding the five stores** is 1.52 GiB resident once it has met every chromosome,
  and 21.46 GiB with every chunk cache full at the default 4 GiB (peak 22.07 GiB). The caches
  counted 19.80 GiB of it: the bound holds in resident memory.
- **Sharing a process.** While ogld reads on a worker thread, an asyncio loop on the main thread
  wakes 2–16 ms late at the 99th percentile, and once in each setting more than 100 ms late (at
  most 206 ms). Under the zarr settings the engine makes process-wide when imported, every query
  took longer than under zarr's defaults, round against round: a clump 17% and 49%, a matrix 19%
  and 45%, a 3 Mb region 12% and 14%, at a node load of 55–94. The answers were the same.

## What was measured, and why

ogld#9 decides how LD is served (ADR 0094). ogld#8 and ogld#32 timed the reads; four things the
decision turns on were not measured:

1. **How long a clump job runs on the server**, against ogdb-api's 30 s store timeout. ogld#8
   clumped #403's 37 Analyses with the reader before ogld#32; ogld#32 re-timed four of them.
   Here all 37 are clumped again, cold, from local disk, with the reader as it now is.
2. **What a client reading static Zarr waits**, with ogld#31's Far Pairs and ogld#32's reader
   together, which no earlier run timed: ogld#31's after was its reader before ogld#32 merged.
   ogld#8's step 1, its lists, regions and proxies, at 10, 50 and, new here, 100 ms a request.
3. **The memory of one process holding the five stores**, as an LD worker would: after
   opening them, after loading every chromosome's arrays, and with every chunk cache full.
4. **What ogld costs a process it shares**: how late an asyncio event loop's ticks wake while a
   query runs on a worker thread, as ogdb-api runs its store queries, and how fast ogld reads
   under the zarr settings the engine makes process-wide when imported (opengwasdb ADR 0056).

## How

- **Clumps:** `pixi run -e dev bench-latency step1 --target local --case clump --cold-only`,
  every Analysis cold: its pages evicted, the store opened inside the timing.
- **Static Zarr:** `pixi run -e dev bench-latency step1 --target local http10 http50 http100 --case
  proxies scattered region --seeds 1`. ogld#8's static host and proxy on loopback, with the wait
  added to every request. **This is loopback with injected latency, not a real WAN.** 10, 50 and
  100 ms stand for a client within the UK, across Europe, and across the Atlantic; none was
  measured on a real path.
- **Memory:** `pixi run -e dev bench-serving memory --out <file>`. One process opens the five
  stores with the default 4 GiB cache each (§10), loads every chromosome's positions and its Near
  Band, Proxy Index and Far Pairs `indptr` through the reader's own loaders, then reads 3 Mb
  regions on each store until its cache holds 95% of its budget; a store that ran out of regions
  first would stop the run. `VmRSS` and `VmHWM` are read from `/proc/self/status` after each step,
  beside each store's cache count.
- **Sharing a process:** `pixi run -e dev bench-serving lag --out <file>`, then the same with
  `--engine-zarr`. The main thread runs an asyncio loop whose task sleeps 5 ms and records how
  late it woke; the query runs on one worker thread (`run_in_executor`), cold: the EUR store's
  pages evicted and the store opened on that thread. The queries are ogld#8's own
  (`latency.clump_query`, `matrix_query`, `region_query`): clumping GCST90319613 (one of ogld#8's
  four) at r² 0.01 within 3 Mb, a matrix of 500 variants drawn genome-wide (seed 8), and a 3 Mb
  region on chr22 at 32 Mb. Each records what it answered: leads, or the pairs stored.
  `--engine-zarr` sets what `opengwasdb/store/arrays.py` sets on import: `FusedCodecPipeline` with
  one worker, and `write_empty_chunks`.

Threads: zarr's pool 6, BLAS 1 (both tasks set it); the static host and proxy at most two each;
`lag` one worker thread more. Every record carries the node's load.

## A clump job on the server

#403's 37 Analyses, each clumped cold from local disk, against ogld#8's run of the same 37 with
the reader before ogld#32 (`results/step1-local.jsonl.gz`); this run is
`results/9/step1-clump.jsonl.gz`. The node's load was 5–56.

| r² | time, sum of 37 | median | largest | over 10 s | over 30 s | MB read | requests | leads |
|---|---|---|---|---|---|---|---|---|
| 0.01, now | 442 s (7.4 min) | 6.3 s | 69.2 s | 16 | 4 | 48,649 | 86,286 | 23,445 |
| 0.01, ogld#8 | 1,043 s (17.4 min) | 14.5 s | 190.8 s | — | 12 | 75,081 | 234,528 | 23,445 |
| 0.001, now | 373 s (6.2 min) | 6.8 s | 48.6 s | 15 | 1 | 38,938 | 70,736 | 13,866 |
| 0.001, ogld#8 | 716 s (11.9 min) | 13.5 s | 101.2 s | — | 10 | 44,995 | 134,231 | 13,866 |

By the number of 5e-8 hits an Analysis has, at r² 0.01:

| hits | Analyses | time | over 30 s |
|---|---|---|---|
| under 1,000 | 6 | 0.2–1.2 s | 0 |
| 1,000–10,000 | 13 | 0.3–13.1 s | 0 |
| 10,000–50,000 | 6 | 5.8–16.7 s | 0 |
| 50,000–120,000 | 8 | 15.3–24.9 s | 0 |
| 120,000–261,000 | 4 | 31.8–69.2 s | 4 |

- **The leads are the same**, 23,445 and 13,866, as ogld#8's.
- **The reader is 2.4× and 1.9× faster than ogld#8 measured, and at r² 0.01 reads 35% fewer
  bytes.** The
  4 GiB cache keeps a chunk an Analysis needs again, and the kept shard indexes save a request a
  shard: 86,286 requests where ogld#8 made 234,528.
- **Against ogdb-api's 30 s store timeout**, four Analyses fail at r² 0.01: GCST90475446
  (163,521 hits, 31.8 s), GCST90475442 (160,706, 33.9 s), GCST90000025 (167,972, 34.2 s) and
  GCST90476460 (260,821, 69.2 s); at r² 0.001 only GCST90476460 (48.6 s). Every Analysis with
  up to 107,358 hits clumped within 25 s.
- ogld#8 measured local timings moving by up to 45% between runs on this node, so a single
  Analysis's time here is good to about half; the sums and the ordering by hits are firmer.

## One process holding the five stores

`bench-serving memory`, one process, the default `cache_bytes` (4 GiB a store), one row a
record, as `bench-serving tables` renders them. The node's load was 54–91.

| run | step | store | resident, GiB | peak, GiB | caches counted, GiB | chromosomes or regions | seconds | load |
|---|---|---|---|---|---|---|---|---|
| 05:33 | imported | — | 0.06 | 0.06 | 0.00 | — | — | 54.8 |
| 05:33 | opened | AFR | 0.06 | 0.06 | 0.00 | — | 0.0 | 54.8 |
| 05:33 | opened | EUR | 0.06 | 0.06 | 0.00 | — | 0.0 | 54.8 |
| 05:33 | opened | EAS | 0.06 | 0.06 | 0.00 | — | 0.0 | 54.8 |
| 05:33 | opened | SAS | 0.06 | 0.06 | 0.00 | — | 0.0 | 54.8 |
| 05:33 | opened | AMR | 0.06 | 0.06 | 0.00 | — | 0.0 | 54.8 |
| 05:33 | chromosomes | AFR | 0.53 | 0.53 | 0.00 | 22 | 2.0 | 54.8 |
| 05:33 | chromosomes | EUR | 0.79 | 0.79 | 0.00 | 22 | 1.4 | 53.9 |
| 05:33 | chromosomes | EAS | 1.02 | 1.02 | 0.00 | 22 | 1.2 | 53.9 |
| 05:33 | chromosomes | SAS | 1.29 | 1.29 | 0.00 | 22 | 1.3 | 53.9 |
| 05:33 | chromosomes | AMR | 1.52 | 1.52 | 0.00 | 22 | 1.3 | 53.9 |
| 05:33 | filled | AFR | 5.71 | 7.35 | 4.00 | 11 | 34.2 | 82.1 |
| 05:33 | filled | EUR | 9.69 | 11.25 | 8.00 | 26 | 33.0 | 91.0 |
| 05:33 | filled | EAS | 13.64 | 14.52 | 11.98 | 29 | 29.3 | 78.7 |
| 05:33 | filled | SAS | 17.60 | 18.27 | 15.94 | 25 | 29.3 | 67.9 |
| 05:33 | filled | AMR | 21.46 | 22.07 | 19.80 | 31 | 30.2 | 60.4 |

- **Opening is free; meeting every chromosome costs 1.45 GiB.** The 47.2 M variants' positions
  and three `indptr` arrays at 8 bytes each are 1.51 GB, which is what the step added. A
  long-lived worker reaches it once it has served a request on every chromosome of every store.
- **The cache's bound holds in resident memory.** With every cache full, resident memory was
  19.94 GiB above the loaded chromosomes, against the 19.80 GiB the caches counted. The peaks
  above it, up to 1.64 GiB, are a 3 Mb region's float64 matrix while it is built (AFR's hold
  about 15,000 variants) and the batch in flight (§10).
- **So a worker's memory is 1.5 GiB + 5 × `cache_bytes` + its reads in progress.** At 4 GiB that
  is about 22 GiB. At 1 GiB a store it would be about 6.5 GiB, and a repeated read would then be
  warm only if its chunks fit in 1 GiB (ogld#8: 500 chr22 variants need 239 MiB of chunks, and a
  10 Mb region's did not fit in 256 MiB).

## Sharing a process: the event loop and the engine's zarr settings

`bench-serving lag`, twice with zarr's defaults and twice with `--engine-zarr`, alternating, one
row a record. Each query cold. Lag is how late a 5 ms sleep on the main thread's event loop woke
while the query ran on a worker thread ("idle": with no query, for 5 s). The node's load was
55–94, near the highest it reached through any run here.

| run | zarr settings | query | query, s | lag p50, ms | lag p99, ms | lag max, ms | ticks over 100 ms | ticks | answer | load |
|---|---|---|---|---|---|---|---|---|---|---|
| 05:31 | default | idle | — | 0.17 | 0.3 | 0.7 | 0 | 967 | — | 94.4 |
| 05:31 | default | clump | 6.18 | 0.20 | 4.3 | 31.0 | 0 | 1,112 | hits 13,513, leads 226 | 91.3 |
| 05:31 | default | matrix_500_genome | 7.74 | 0.27 | 4.7 | 107.6 | 1 | 1,338 | variants 500, stored 1,144 | 86.6 |
| 05:31 | default | region_3mb_chr22 | 2.15 | 0.11 | 4.4 | 47.1 | 0 | 399 | variants 10,651, stored 113,443,801 | 86.6 |
| 05:31 | engine's | idle | — | 0.09 | 0.2 | 0.5 | 0 | 981 | — | 83.4 |
| 05:31 | engine's | clump | 9.19 | 0.18 | 11.8 | 206.1 | 1 | 1,474 | hits 13,513, leads 226 | 76.3 |
| 05:31 | engine's | matrix_500_genome | 11.26 | 0.38 | 15.1 | 86.0 | 0 | 1,628 | variants 500, stored 1,144 | 72.2 |
| 05:31 | engine's | region_3mb_chr22 | 2.40 | 0.10 | 2.3 | 3.9 | 0 | 463 | variants 10,651, stored 113,443,801 | 70.5 |
| 05:32 | default | idle | — | 0.09 | 0.2 | 0.2 | 0 | 982 | — | 68.3 |
| 05:32 | default | clump | 7.93 | 0.21 | 4.7 | 77.5 | 0 | 1,401 | hits 13,513, leads 226 | 64.5 |
| 05:32 | default | matrix_500_genome | 9.23 | 0.26 | 4.3 | 76.5 | 0 | 1,586 | variants 500, stored 1,144 | 63.4 |
| 05:32 | default | region_3mb_chr22 | 2.14 | 0.13 | 2.9 | 4.5 | 0 | 404 | variants 10,651, stored 113,443,801 | 62.4 |
| 05:32 | engine's | idle | — | 0.09 | 0.2 | 0.4 | 0 | 981 | — | 60.7 |
| 05:32 | engine's | clump | 9.31 | 0.18 | 10.4 | 70.4 | 0 | 1,524 | hits 13,513, leads 226 | 58.9 |
| 05:32 | engine's | matrix_500_genome | 11.03 | 0.47 | 16.2 | 77.2 | 0 | 1,600 | variants 500, stored 1,144 | 56.0 |
| 05:32 | engine's | region_3mb_chr22 | 2.43 | 0.11 | 2.5 | 5.2 | 0 | 466 | variants 10,651, stored 113,443,801 | 54.8 |

- **ogld holds the interpreter's lock in short stretches.** A service's own requests, on the
  event loop, would wait 2–16 ms at the 99th percentile while a query runs beside them, against
  0.2–0.3 ms idle. One tick in each setting was more than 100 ms late: 108 ms under the
  defaults and 206 ms under the engine's settings, both on a node at load 76–87. That is a cost of
  sharing a process, not a reason to avoid it.
- **The engine's zarr settings slow ogld.** Under `FusedCodecPipeline` with one worker, every
  query took longer than under zarr's defaults, round against round: the clump 9.19 s against
  6.18 s and 9.31 s against 7.93 s (+49%, +17%), the matrix +45% and +19%, the region +12% and
  +14%. The answers were the same, as each record's answer column shows: 226 leads, 1,144 and
  113,443,801 stored pairs. ogld's arrays are zstd, so the engine's Blosc threads do not touch
  them. An earlier run at `6bae783` and a load of 16–44, whose records this one replaces because
  `serving.py` was then uncommitted (they are kept outside the repository, under
  `/data/opengwasdb/work/ogld/scratch/9/`), found the engine's settings slower in five of six
  pairs, from −7% to +17%: the direction held, and the size moved with the load. It matters wherever ogld runs
  inside an engine process: the engine's on-the-fly top-hit clumps (opengwasdb#419).

## A remote client reading static Zarr

ogld#8's step 1 on the published EUR store, cold unless named: the store's pages evicted and the
store opened inside the timing (`open`: an opened reader, its chunk cache emptied). One draw a
list (seed 8), the median of 10 draws for one variant's proxies and of the three centres for a
region. [`serving-tables.md`](serving-tables.md) has every state, the bytes, the requests and the
load of each row, rendered by `report.py`; the node's load was 8–51.

| read | local | 10 ms | 50 ms | 100 ms | requests | MB read |
|---|---|---|---|---|---|---|
| matrix, 500 chr22 variants | 1.18 s | 2.93 s | 6.82 s | 11.75 s | 480 | 111.5 |
| matrix, 500 variants genome-wide | 8.31 s | 15.41 s | 36.74 s | 61.24 s | 1,437 | 167.6 |
| the same, `open` | 6.98 s | 9.20 s | 19.12 s | 30.53 s | 1,146 | 98.6 |
| `proxies`, 1 variant, `open` | 29 ms | 108 ms | 346 ms | 655 ms | 6 | 0.1 |
| `proxies`, 100 variants | 2.27 s | 6.49 s | 18.34 s | 33.03 s | 669 | 75.8 |
| the same, `open` | 1.08 s | 2.82 s | 6.95 s | 12.24 s | 475 | 10.2 |
| region 3 Mb on chr22 | 1.75 s | 2.28 s | 3.67 s | 9.48 s | 48 | 97.4 |
| region 10 Mb on chr22 | 8.14 s | 8.49 s | 13.76 s | 19.50 s | 119 | 287.0 |

Against ogld#32's records of the same draws (`results/32/step1-reads.jsonl.gz`), the Far Pairs
halve the chr22 list over a network: 14.77 s at 50 ms there, 6.82 s here, and 5.79 s against
2.93 s at 10 ms. The genome-wide list gains less, 41.78 s against 36.74 s at 50 ms, because
its far pairs were a smaller share of it, and `proxies` reads no Far Pairs and is unchanged.

- **Opening is a remote client's first cost.** A cold read loads each chromosome it touches:
  positions and both `indptr` arrays, then the Far Pairs' `indptr` if it needs pairs beyond
  3 Mb (§10). For the genome-wide list that was 0.9 s locally and 11.3 s at 50 ms, 20.5 s at
  100 ms; for 100 variants' proxies, 1.1 s locally and 19.9 s at 100 ms. A client that keeps
  its reader open pays it once (the `open` rows).
- **A clump over static Zarr** was not re-timed here: ogld#31 changes no clump read, and ogld#32
  timed ogld#8's four Analyses at 102 s at 10 ms and 224 s at 50 ms, against 52 s locally. The
  37 Analyses read 48.6 GB (above), which on a real link costs its transfer time as well: 6.5
  minutes at 1 Gbit/s, 65 at 100 Mbit/s.
- **What the server routes add to the local column** is the job's round trips: a submission, the
  polls at the interval ogdb-api recommends (0.25 s at the least), and the result. A 500 × 500
  matrix is 2 MB of float64.

## The runs

Each file in [`results/9/`](results/9/) holds one or more runs, each starting with a record of
the code that ran. `step1-clump` and `memory` and `lag` ran on a clean checkout. `step1-reads` ran
`bench-latency` at `6bae783` with an earlier `serving.py` untracked beside it, which its record
names; that process did not import it.

| file | started (UTC) | load | command |
|---|---|---|---|
| `step1-clump` | 03:19 | 17.8 | `pixi run -e dev bench-latency step1 --target local --case clump --cold-only` |
| `step1-reads` | 03:37 | 11.9 | `pixi run -e dev bench-latency step1 --target local http10 http50 http100 --case proxies scattered region --seeds 1` |
| `lag` | 05:31, 05:31, 05:32, 05:32 | 93.5, 86.6, 70.5, 62.4 | `pixi run -e dev bench-serving lag`, then `--engine-zarr`, twice |
| `memory` | 05:33 | 54.8 | `pixi run -e dev bench-serving memory` |

Each took `--out <file>.jsonl`; the files are kept gzipped. Every table in
[`serving-tables.md`](serving-tables.md) was rendered from them by:

```bash
pixi run -e dev bench-serving tables docs/benchmarks/results/9/*.jsonl.gz \
    > docs/benchmarks/serving-tables.md
```

The memory and lag tables above are copied from it, one row a record. The clump, read and
summary tables above gather those records, against ogld#8's and ogld#32's where they say so.
`tests/test_bench_serving.py` holds `serving-tables.md` to the records, and this page's memory
and lag tables to it.
