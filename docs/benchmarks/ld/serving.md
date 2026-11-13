# What serving LD costs: a job's clump, a remote client's reads, a worker's memory (ogld#9)

_Measured 13 November 2026 on the published HGDP+1kGP stores, at ogld `6bae783` (ogld#31's Far
Pairs and ogld#32's reader, both merged). The records are in [`results/9/`](results/9/); ogld#8's
method, targets and states are in [`latency.md`](latency.md). ADR 0008 uses these numbers._

## In brief

- **A clump job on the server.** #403's 37 Analyses clump in 442 s (7.4 min) at r² 0.01 and
  373 s at 0.001 from local disk, cold, against ogld#8's 17.4 and 11.9 minutes: 6.3 s an
  Analysis at the median, 69 s at most. Four of the 37 (160,706–260,821 hits) run past
  ogdb-api's 30 s store timeout at r² 0.01, one at 0.001. They read 48.6 GB, where ogld#8's
  reader read 75 GB.
- **A remote client reading static Zarr**, with ogld#31's Far Pairs and ogld#32's reader
  together: a matrix of 500 chr22 variants takes 1.2 s locally, 2.9 s at 10 ms a request, 6.8 s
  at 50 ms and 11.8 s at 100 ms, cold; drawn genome-wide, 8.3, 15.4, 36.7 and 61.2 s, a quarter
  to a third of which over HTTP is loading the touched chromosomes' positions and `indptr` arrays. `proxies`
  for one variant on an opened store takes 29 ms, 108 ms, 346 ms and 655 ms.
- **One process holding the five stores** is 1.52 GiB resident once it has met every chromosome,
  and 21.46 GiB with every chunk cache full at the default 4 GiB (peak 22.05 GiB). The caches
  counted 19.80 GiB of it: the bound holds in resident memory.
- **Sharing a process costs little.** While ogld reads on a worker thread, an asyncio loop on
  the main thread wakes 1–16 ms late at the 99th percentile, at most 116 ms. Under the zarr
  settings the engine makes process-wide, a clump and a matrix take within the run-to-run
  spread, and a 3 Mb region 15–17% longer.

## What was measured, and why

ogld#9 decides how LD is served (ADR 0008). ogld#8 and ogld#32 timed the reads; four things the
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
- **Memory:** `pixi run -e dev python scripts/bench/serving.py memory`. One process opens the five
  stores with the default 4 GiB cache each (§10), loads every chromosome's positions and its Near
  Band, Proxy Index and Far Pairs `indptr` through the reader's own loaders, then reads 3 Mb
  regions on each store until its cache holds 95% of its budget. `VmRSS` and `VmHWM` are read
  from `/proc/self/status` after each step, beside each store's cache count.
- **Sharing a process:** `pixi run -e dev python scripts/bench/serving.py lag`, then the same with
  `--engine-zarr`. The main thread runs an asyncio loop whose task sleeps 5 ms and records how
  late it woke; the query runs on one worker thread (`run_in_executor`), cold: the EUR store's
  pages evicted and the store opened on that thread. The queries: clumping GCST90319613 (one of
  ogld#8's four) at r² 0.01 within 3 Mb, a matrix of 500 variants drawn genome-wide (seed 8), and
  a 3 Mb region on chr22 at 32 Mb. `--engine-zarr` sets what `opengwasdb/store/arrays.py` sets on
  import: `FusedCodecPipeline` with one worker, and `write_empty_chunks`.

Threads: zarr's pool 6, BLAS 1; the static host and proxy at most two each; `lag` one worker
thread more. Every record carries the node's load before and after.

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

`serving.py memory`, one process, the default `cache_bytes` (4 GiB a store). Resident memory
after each step:

| step | resident | peak | chunk caches, counted |
|---|---|---|---|
| imported | 0.06 GiB | 0.06 GiB | 0 |
| the five stores opened | 0.07 GiB | 0.07 GiB | 0 |
| every chromosome loaded: AFR | 0.53 GiB | 0.53 GiB | 0 |
| … EUR, EAS, SAS | 0.79, 1.02, 1.29 GiB | the same | 0 |
| … AMR: all five | **1.52 GiB** | 1.52 GiB | 0 |
| AFR's cache filled | 5.70 GiB | 7.35 GiB | 4.00 GiB |
| EUR's too | 9.69 GiB | 11.21 GiB | 8.00 GiB |
| EAS's too | 13.64 GiB | 14.52 GiB | 11.98 GiB |
| SAS's too | 17.60 GiB | 18.27 GiB | 15.94 GiB |
| AMR's too: all five full | **21.46 GiB** | 22.05 GiB | 19.80 GiB |

- **Opening is free; meeting every chromosome costs 1.45 GiB.** The 47.2 M variants' positions
  and three `indptr` arrays at 8 bytes each are 1.51 GB, which is what the step added. A
  long-lived worker reaches it once it has served a request on every chromosome of every store.
- **The cache's bound holds in resident memory.** With every cache full, resident memory was
  19.94 GiB above the loaded chromosomes, against the 19.80 GiB the caches counted. The peaks
  above it, up to 1.65 GiB, are a 3 Mb region's float64 matrix while it is built (AFR's hold
  about 15,000 variants) and the batch in flight (§10).
- **So a worker's memory is 1.5 GiB + 5 × `cache_bytes` + its reads in progress.** At 4 GiB that
  is about 22 GiB. At 1 GiB a store it would be about 6.5 GiB, and a repeated read would then be
  warm only if its chunks fit in 1 GiB (ogld#8: 500 chr22 variants need 239 MiB of chunks, and a
  10 Mb region's did not fit in 256 MiB).

## Sharing a process: the event loop and the engine's zarr settings

`serving.py lag`, twice with zarr's defaults and twice with `--engine-zarr`, alternating. Each
query cold. Lag is how late a 5 ms sleep on the main thread's event loop woke while the query
ran on a worker thread.

| query | zarr settings | query time, runs 1 and 2 | lag p50 | lag p99 | lag max | ticks over 100 ms |
|---|---|---|---|---|---|---|
| none (idle, 5 s) | default | — | 0.09 ms | 0.1–0.2 ms | 0.3–0.4 ms | 0 |
| none (idle, 5 s) | engine's | — | 0.09–0.13 ms | 0.1–0.2 ms | 0.3–0.4 ms | 0 |
| clump GCST90319613, r² 0.01 (13,513 hits, 226 leads) | default | 7.38 s, 7.51 s | 0.17–0.18 ms | 4.3–4.6 ms | 66–79 ms | 0 |
| clump GCST90319613, r² 0.01 | engine's | 8.26 s, 7.63 s | 0.13–0.17 ms | 11.4–12.0 ms | 35–65 ms | 0 |
| matrix of 500 variants genome-wide | default | 9.11 s, 9.60 s | 0.22–0.24 ms | 4.7–5.3 ms | 99–116 ms | 1 |
| matrix of 500 variants genome-wide | engine's | 10.51 s, 8.91 s | 0.43 ms | 14.0–15.8 ms | 78–94 ms | 0 |
| region 3 Mb, chr22 at 32 Mb (10,651 variants) | default | 1.50 s, 1.47 s | 0.10–0.13 ms | 2.1–2.6 ms | 2.9 ms | 0 |
| region 3 Mb, chr22 at 32 Mb | engine's | 1.73 s, 1.72 s | 0.09–0.13 ms | 1.1–2.7 ms | 3.6–4.8 ms | 0 |

The node's load was 16–44 through these runs.

- **ogld holds the interpreter's lock in short stretches.** A service's own requests, on the
  event loop, would wait 1–16 ms at the 99th percentile while a query runs beside them, and at
  most 116 ms once in four runs. That is a cost of sharing a process, not a reason to avoid it.
- **The engine's zarr settings cost ogld little.** Under `FusedCodecPipeline` with one worker
  the clump and the matrix took within the run-to-run spread of the defaults (−7% to +15%, round
  against round), and the region 15–17% longer in both rounds. ogld's arrays are zstd, so the engine's Blosc
  threads do not touch them.

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

Each file in [`results/9/`](results/9/) is one run, at `6bae783`. `step1-clump` ran on a clean
checkout. The others ran with `scripts/bench/serving.py` untracked, as each record's `code`
says; it was committed afterwards unchanged (SHA-256 `3f840ccf…5fdb0a`, before and after).

| file | started (UTC) | load | command |
|---|---|---|---|
| `step1-clump` | 03:19 | 17.8 | `pixi run -e dev bench-latency step1 --target local --case clump --cold-only` |
| `lag` | 03:33, 03:33, 03:34, 03:34 | 37.0, 28.7, 19.6, 28.4 | `pixi run -e dev python scripts/bench/serving.py lag`, then `--engine-zarr`, twice |
| `memory` | 03:35 | 44.4 | `pixi run -e dev python scripts/bench/serving.py memory` |
| `step1-reads` | 03:37 | 11.9 | `pixi run -e dev bench-latency step1 --target local http10 http50 http100 --case proxies scattered region --seeds 1` |

`serving.py` ran with `OPENBLAS_NUM_THREADS=1` and its siblings, as the `bench-latency` task
sets them. Each took `--out <file>.jsonl`; the files are kept gzipped. The tables were rendered
by:

```bash
pixi run -e dev python scripts/bench/report.py docs/benchmarks/results/9/step1-*.jsonl.gz \
    > docs/benchmarks/serving-tables.md
```

The memory and lag tables above are the records' own fields, one row a record.
