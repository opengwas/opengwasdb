# LD Store read latency, from local disk and over HTTP (ogld#8)

_Measured 12 November 2026. Every table below is rendered from the records in
[`results/`](results/), and [`latency-tables.md`](latency-tables.md) has all of them._

## In brief

- **Clumping** #403's 37 Analyses on the published EUR store took 17.4 minutes at r² 0.01 and
  11.9 at 0.001 from local disk, cold, and 29 and 18 minutes over HTTP at 0 ms. ogld's reader waits
  one round trip per request, so at 10 ms the r² 0.01 run would take about 71 minutes and at 50 ms
  about 3.8 hours. That is an estimate: the per-request cost measured in step 2 predicts the four
  Analyses timed at every latency to within 2%. It reads 75 GB of Near Band chunks.
- **Layout C** clumps below t_far fastest at every latency: EUR chr22 in 12.3 s locally, 44 s at
  10 ms and 143 s at 50 ms, against B's 17.1, 132 and 488 s and A's 37.8, 585 and 2,243 s. It
  reads regions fastest and is the smallest store. B reads 3–6× fewer bytes on small reads, which
  take the same time on every layout here. **Decided: layout C is final** (ADR 0006).
- **The Far Pairs, merged into the Proxy Index,** cost a 10 Mb region the 6–9% more bytes §11
  estimated, and 13–30% of its time. They cost a matrix of 500 variants on chr22 58–60% of its time
  over HTTP: about 890 index chunks, where a Far Pairs component reads about 11. chr22 holds few
  Far Pairs, but they cluster in few rows on every chromosome, so the component reads 10–16× fewer
  chunks even on chr11, which holds 69 M of them. **Decided: the Far Pairs come back (ogld#31).**
- **Reader levers**, measured, not implemented: fetching a read's runs of chunks together (5–9×
  faster lists and 2–3.3× faster clumping on C at 50 ms), the chunk cache's size (at 4 GiB a
  repeated read is warm), and `ChunkCache._counted` (6–22% of reads that touch strings).

The owner ruled on 12 November 2026, on ogld#8: layout C is final, the Far Pairs come back as a
component of their own (ogld#31), and the reader's speed-ups are ogld#32 ("Decisions", below).

## What was measured, and on what

**Step 1** times the four queries on the published HGDP+1kGP EUR LD Store,
`/data/opengwasdb/work/ogld/stores/hgdp1kgp-v2-EUR-k10.store`. It has 8,586,889 variants and is
at × 511 and k = 10, in layout C, then spec §6's provisional choice and now its final one
(ADR 0006). The queries:

- clumping each of opengwas/opengwasdb#403's 37 Analyses' 5e-8 tiers at r² 0.01 and 0.001
  within 3 Mb, the inputs of ogld#6's acceptance run (`hits.tsv`, |z| as the score);
- region matrices 3 Mb and 10 Mb wide on chr22, centred at 25, 32 and 40 Mb. The 10 Mb one reads
  the Proxy Index rows of its variants for its pairs beyond `near_bp` (spec §10, §11);
- `matrix` for 500 variants drawn at random, on chr22 and genome-wide (seeds 8, 9 and 10);
- `proxies` at r² 0.8 within 10 Mb for 1, 10 and 100 variants drawn genome-wide.

**Step 2** times the reads that differ between the Near Band layouts on HGDP+1kGP EUR chr22,
built three times by `ogld build` (× 511, k = 10, the published store's inputs), once in each
of spec §6's layouts:

| layout | `near/<c>/r` inner chunk | shard |
|---|---|---|
| A | 4,096 codes (8 KiB) | 4,194,304 codes (8 MiB) |
| B | 32,768 codes (64 KiB) | 16,777,216 codes (32 MiB) |
| C | 262,144 codes (512 KiB) | 67,108,864 codes (128 MiB) |

The three builds decode to the same `near/22/r`, `near/22/indptr` and `index/22/*`
(`build_layouts.py --check`), and C's shard files are byte-identical to the published store's
chr22. So step 2 compares layouts on identical data, and its C is step 1's chr22. The reads:

- clumping the 37 Analyses' chr22 hits at r² 0.01 and 0.001 within 3 Mb. Both are below t_far
  (0.0319), so each lead reads one Near Band element per open hit in its window: §11's column
  read, in `clump`'s form;
- a 3 Mb region at the same three centres;
- `matrix` for 500 chr22 variants;
- single pairs within 3 Mb, read by `StoreReader.r_of_pairs`: 20 drawn at random.

## How

**Targets.** `local` reads the store from `/data` (XFS on a 42 TB non-rotational LVM volume).
`http<ms>` reads it through `scripts/bench/http_static.py`. That is a static host (aiohttp's
`FileResponse`, which honours one byte range, a suffix range included, and sends with
sendfile) with a proxy in front that waits `<ms>` before forwarding each request and counts the
requests and body bytes per array. ogld opens the URL as it opens any store, through zarr's
fsspec store and aiohttp.

**This is loopback with injected latency, not a real WAN.** Every request waits the set time
once, as one round trip would, and requests in flight together wait together. Nothing limits
bandwidth, nothing is lost, and no connection pays the extra round trip (and TLS's) a real one
pays when it opens. The byte counts are exact, so a reader can add what those bytes cost on a
link of a known speed.

**States.**

- `cold` drops the store's pages from the OS cache (`posix_fadvise(DONTNEED)` on every file)
  and times `ogld.open` and the query together. That includes loading the positions and both
  `indptr` arrays of each chromosome the query touches, shown apart as "chromosome loads".
- `open` keeps the opened reader and its chromosomes, empties its chunk cache and drops the
  pages again: a long-lived service meeting a new query.
- `warm` repeats the query on the reader as the last run left it.

The other readers on the node can warm a page between the eviction and the read. Nothing else
read these stores while the runs were made, as far as the node's process list showed.

**Threads.** zarr's pool is 6 threads, BLAS 1, and the static host and the proxy run one
event loop and at most 2 worker threads each. Every record carries the node's load average
before and after it; the node has 224 cores and its 1-minute load was 1.7–44 over the runs.

**How much a local timing varies.** Step 1's regions, lists and proxies were timed locally twice,
seven hours apart: at a load of 25–42 and again at 3–5. Medians moved by up to 45% between the
runs, in both directions, and not with the load. The regions were 29–37% faster in the second run,
and the cold lists 29–45% slower, a genome-wide list even warm. Within one run the genome-wide
list's far part ranged 2.3–5.8 s. So a local difference under about half is noise here. Step 2
times each case on A, B and C back to back, so the layouts meet the same conditions, and its
differences are multiples. Over HTTP with added latency the round trips dominate, and repeat
closely: step 2's single pairs at 50 ms took 113–116 ms on every layout.

**Software.** `ogld/` at `883eb03` (`0.0.1.dev226`), zarr 3.4.2, numpy 2.4.6, fsspec 2026.9.0,
aiohttp 3.14.5, Python 3.12, on an Intel Xeon Platinum 8480+.

**The runs.** Each file in [`results/`](results/) is one run, in this order. The commit is the
branch's when the run started; `ogld/` is `883eb03`'s throughout. `step1-local` ran the scripts
of `74e7cb1` before that commit was made. A later run's record of the same read supersedes an
earlier one's in the tables.

| file | started (UTC) | load | `pixi run -e dev bench-latency …` |
|---|---|---|---|
| `step1-local` | 15:19 | 25.4 | `step1 --target local` |
| `step2-local-http0` | 16:20 | 7.7 | `step2 --target local http0` |
| `step2-http10` | 16:34 | 7.1 | `step2 --target http10 --seeds 1` |
| `step2-http50` | 17:08 | 6.7 | `step2 --target http50 --seeds 1 --cold-only` |
| `step2-concurrent` | 19:15 | 26.1 | `step2 --target local http10 http50 --seeds 1 --cold-only --reader concurrent` |
| `step1-http0` | 19:45 | 10.9 | `step1 --target http0 --cold-only` |
| `step1-http-reads` | 20:40 | 5.6 | `step1 --target http10 http50 --case proxies scattered region --seeds 1` |
| `step1-http-clump` | 21:08 | 2.6 | `step1 --target http10 http50 --case clump --cold-only --analyses` (the four) |
| `step1-concurrent` | 21:49 | 1.9 | `step1 --target local http50 --case clump proxies scattered --seeds 1 --cold-only --reader concurrent --analyses` (the four) |
| `step1-cache4g` | 22:05 | 5.2 | `step1 --target local --case clump scattered --seeds 1 --cache-mib 4096 --analyses` (the four) |
| `step1-local-reads` | 22:10 | 3.2 | `step1 --target local --case proxies scattered region` |

Each takes `--out <file>.jsonl`; the files are kept gzipped. Each file's first record says which
code ran and whether the checkout could be cited as that code (`code`, from
`test_panel.checkout.problem`); for these eleven it was added after the runs, from the branch's
history, and says so. None was clean: `step1-local` ran untracked scripts, and the installed ogld
named `883eb03` throughout, whose `ogld/` is every later commit's. `results/far_pairs.json` is
`pixi run -e dev python scripts/bench/far_pairs.py --out …`, run on 13 November, which times
nothing. The four Analyses are
`OGS-00009__ukb-b-4171 OGS-00011__GCST90319613 OGS-00011__GCST90025965 OGS-00011__GCST90029004`.
The layouts were built, and the tables rendered, by:

```bash
pixi run -e dev python scripts/bench/build_layouts.py /data/opengwasdb/work/ogld/scratch/8/layouts
pixi run -e dev python scripts/bench/report.py docs/benchmarks/results/*.jsonl.gz \
    --layouts docs/benchmarks/results > docs/benchmarks/latency-tables.md
```

`--reader concurrent` and `--cache-mib` measure two reader levers from outside `ogld/` (below).

## Step 1: the published EUR store, layout C

Seconds, ogld's reader, from the records in [`results/`](results/). "Sum of 37" is every Analysis;
"sum of 4" the four timed at every latency (ukb-b-4171, GCST90319613, GCST90025965 and
GCST90029004: 560–107,358 hits). "Median of 1" rows are the one draw timed at 10 and 50 ms, beside
the same draw locally and at 0 ms. MB read and requests are the first target's (they are the same
at every target, to within the metadata a URL open probes for). The local column of the
regions, lists and proxies is a second run at a load of 3–5 (`step1-local-reads`), which
supersedes the first (`step1-local`, at 25–42; see "How much a local timing varies").

| read | local | http0 | http10 | http50 | MB read | requests |
|---|---|---|---|---|---|---|
| clump, r² 0.01, cold, sum of 37 | 1,043 s | 1,729 s | — | — | 75,081.0 | 234,528 |
| clump, r² 0.01, cold, sum of 4 | 91.02 s | 144 s | 349 s | 1,110 s | 5,722.0 | 18,408 |
| clump, r² 0.01, warm, sum of 37 | 946 s | — | — | — | 72,797.8 | 226,602 |
| clump, r² 0.001, cold, sum of 37 | 716 s | 1,108 s | — | — | 44,994.8 | 134,231 |
| clump, r² 0.001, cold, sum of 4 | 71.94 s | 105 s | 246 s | 766 s | 3,751.3 | 12,716 |
| clump, r² 0.001, warm, sum of 37 | 622 s | — | — | — | 42,767.8 | 126,481 |
| region 3 Mb, cold, median of 3 | 1.47 s | 1.67 s | 2.35 s | 4.00 s | 97.5 | 58 |
| region 3 Mb, warm, median of 3 | 895 ms | 960 ms | 901 ms | 678 ms | 0.000 | 0 |
| region 10 Mb, cold, median of 3 | 8.37 s | 10.33 s | 11.09 s | 18.42 s | 313.0 | 224 |
| region 10 Mb, warm, median of 3 | 8.19 s | 11.16 s | 10.63 s | 17.46 s | 312.0 | 205 |
| 500 scattered, chr22, cold, median of 3 | 6.56 s | 9.95 s | — | — | 125.0 | 2,183 |
| 500 scattered, chr22, cold, median of 1 | 6.56 s | 9.95 s | 40.02 s | 129 s | 125.3 | 2,245 |
| 500 scattered, chr22, open, median of 3 | 4.68 s | 10.23 s | — | — | 124.0 | 2,164 |
| 500 scattered, chr22, open, median of 1 | 5.99 s | 10.23 s | 39.52 s | 128 s | 124.3 | 2,226 |
| 500 scattered, chr22, warm, median of 3 | 1.70 s | 5.51 s | — | — | 27.5 | 1,448 |
| 500 scattered, chr22, warm, median of 1 | 2.92 s | 6.65 s | 26.21 s | 86.27 s | 28.9 | 1,512 |
| 500 scattered, genome, cold, median of 3 | 21.83 s | 30.34 s | — | — | 183.0 | 3,780 |
| 500 scattered, genome, cold, median of 1 | 17.02 s | 29.20 s | 75.12 s | 222 s | 186.3 | 3,726 |
| 500 scattered, genome, open, median of 3 | 19.52 s | 27.40 s | — | — | 116.4 | 3,556 |
| 500 scattered, genome, open, median of 1 | 20.03 s | 27.40 s | 69.64 s | 209 s | 119.7 | 3,502 |
| 500 scattered, genome, warm, median of 3 | 13.23 s | 24.21 s | — | — | 94.7 | 3,324 |
| 500 scattered, genome, warm, median of 1 | 13.23 s | 24.21 s | 62.58 s | 196 s | 98.0 | 3,312 |
| proxies, 1 variant, cold, median of 10 | 106 ms | 230 ms | 403 ms | 1.30 s | 3.5 | 26 |
| proxies, 1 variant, open, median of 10 | 31 ms | 61 ms | 107 ms | 352 ms | 0.1 | 6 |
| proxies, 1 variant, warm, median of 10 | 1.6 ms | 1.9 ms | 1.5 ms | 1.6 ms | 0.000 | 0 |
| proxies, 10 variants, cold, median of 3 | 846 ms | 1.55 s | — | — | 27.8 | 150 |
| proxies, 10 variants, cold, median of 1 | 846 ms | 1.75 s | 2.68 s | 8.18 s | 26.8 | 150 |
| proxies, 10 variants, open, median of 3 | 285 ms | 363 ms | — | — | 1.1 | 58 |
| proxies, 10 variants, open, median of 1 | 285 ms | 545 ms | 1.02 s | 3.40 s | 1.1 | 58 |
| proxies, 10 variants, warm, median of 3 | 12 ms | 12 ms | — | — | 0.000 | 0 |
| proxies, 10 variants, warm, median of 1 | 12 ms | 13 ms | 11 ms | 12 ms | 0.000 | 0 |
| proxies, 100 variants, cold, median of 3 | 3.49 s | 4.62 s | — | — | 75.9 | 756 |
| proxies, 100 variants, cold, median of 1 | 3.75 s | 4.64 s | 14.26 s | 44.30 s | 75.9 | 753 |
| proxies, 100 variants, open, median of 3 | 1.86 s | 2.96 s | — | — | 10.3 | 550 |
| proxies, 100 variants, open, median of 1 | 1.83 s | 2.29 s | 10.42 s | 32.09 s | 10.3 | 540 |
| proxies, 100 variants, warm, median of 3 | 86 ms | 89 ms | — | — | 0.000 | 0 |
| proxies, 100 variants, warm, median of 1 | 86 ms | 86 ms | 89 ms | 86 ms | 0.000 | 0 |

**Clumping the 37 Analyses** took 1,043 s cold at r² 0.01 and 716 s at 0.001 from local disk
(946 s and 622 s warm), and 1,729 s and 1,108 s over HTTP at 0 ms. That is 23,445 and 13,866
leads; the slowest Analysis, GCST90476460 (260,821 hits, 3,869 leads), took 191 s at r² 0.01.
ogld#8's comment measured 15.1 and 9.6 minutes on the k = 0 store; here it is 17.4 and 11.9.

- **Where the time goes.** At r² 0.01 the Near Band took 103,789 Zarr calls for 317,147 chunks:
  4.4 calls and 13.5 chunks a lead, 166 GB decoded and 75 GB read. The string reads, `rows_of`
  and `variants`, took 24% of the time (35% at 0.001); `_counted` is 6–9% of it, inside them.
  Loading each chromosome's positions and `indptr` arrays took 0.86 s an Analysis.
- **Over HTTP every request waits its round trip.** On four Analyses of the 37 (560–107,358 hits,
  2,131 leads at r² 0.01), clumping took 91 s locally, 144 s at 0 ms, 349 s at 10 ms and 1,110 s at
  50 ms. Step 2 measured what a request costs a clump: 10.8 ms at 10 ms and 51.6 ms at 50 ms on
  C. The 0 ms time plus that cost for each of the four Analyses' 18,408 requests predicts 343 s and
  1,094 s, within 2% of what was measured (and within 1.4% at r² 0.001). By the same model, all 37
  at r² 0.01 (234,528 requests) would take about 71 minutes at 10 ms and 3.8 hours at 50 ms, and at
  0.001 (134,231 requests) about 43 minutes and 2.2 hours. These are estimates, not runs: at 10
  and 50 ms only the four Analyses were clumped.
- **And it reads 75 GB.** At r² 0.01 the clump fetches 75 GB of Near Band chunks: 10 minutes at
  1 Gbit/s and 100 minutes at 100 Mbit/s, before any round trip.
- **Warm is not much faster**, 9–13%, because the reads do not fit the chunk cache (below).

**Region matrices.** A 3 Mb region (6,316–10,651 variants) took 0.6–2.4 s cold and 0.4–1.6 s
warm over the two local runs, reading 38–100 MB. A 10 Mb region (29,544–33,788 variants, a 7–9 GB
float64 matrix) took 7.6–19.7 s cold and 7.9–12.4 s warm, reading its 241–293 MB of Near Band
chunks again each time, because they do not fit the cache.
Over HTTP the region reads are few requests (58 and 224): at 50 ms the 3 Mb region took 4.0 s and
the 10 Mb one 18.4 s. The 10 Mb region's Proxy Index rows are the Far Pairs question, below.

**A matrix of 500 variants** is bound by requests. On chr22 it made about 2,200 requests and took
4.0–6.8 s locally, 10 s at 0 ms, 40 s at 10 ms and 129 s at 50 ms; drawn genome-wide it made
about 3,700 and took 15–23 s locally and 222 s at 50 ms. Genome-wide, the string columns are
most of it: `variants` and `rows_of` took 77% of the first run's local time, `_counted` 20%.
Warm, neither fits the cache.

**Proxies** for one variant took about 0.1 s cold locally, 60 ms of it loading its chromosome.
With the chromosome loaded and the chunk cache empty they took 30–31 ms locally, 107 ms at 10 ms
and 352 ms at 50 ms: six requests, three reads of two requests each. For 100 variants that is
1.8–2.0 s locally and 32 s at 50 ms, and 70% of the local time is finding the ALIDs' rows
(`rows_of`).

## Step 2: the Near Band's layout

### Size and build, EUR chr22

| layout | `near/22/r` on disk | against C | shard files | mean compressed chunk | shard index | build, wall | build, CPU | peak RSS |
|---|---|---|---|---|---|---|---|---|
| A | 1,327,224,461 B | +32.4% | 279 | 4.7 KB | 16 KiB | 259 s | 556 s | 2.20 GiB |
| B | 1,191,999,135 B | +18.9% | 71 | 33.5 KB | 8 KiB | 60.2 s | 269 s | 2.35 GiB |
| C | 1,002,545,575 B | — | 19 | 225 KB | 4 KiB | 37.6 s | 214 s | 2.87 GiB |

Each build ran on 8 threads at a load of 39–41. At × 511 a smaller chunk compresses much worse
than spec §6's codec table, measured at × 32767, suggests: 1.76× (A), 1.96× (B) and 2.33× (C)
here, against 1.15–1.21× there. The five published stores' Near Bands are 486 GB, at C. At EUR
chr22's ratios, B would add about 92 GB to them and A about 157 GB.

### Reads, A / B / C

Seconds, with ogld's reader as it is. Clumping is summed over the 27 Analyses with chr22 hits
(25,514 hits; 454 leads at r² 0.01, 263 at 0.001), each Analysis cold. A region is the median of
the three centres, a single pair the median of 20, and 500 scattered one draw (seed 8; locally and
at 0 ms the median of three draws is within 2.2% of it).

| read | local | +0 ms | +10 ms | +50 ms | MB read | requests |
|---|---|---|---|---|---|---|
| clump, r² 0.01, cold | 37.8 / 17.1 / **12.3** | 110 / 34.5 / **18.1** | 585 / 132 / **44.3** | 2,243 / 488 / **143** | 464 / 415 / 683 | 39,780 / 8,508 / 2,419 |
| clump, r² 0.001, cold | 30.8 / 15.2 / **11.3** | 87.3 / 30.2 / **16.6** | 466 / 116 / **40.6** | 1,786 / 426 / **131** | 376 / 373 / 616 | 31,618 / 7,442 / 2,225 |
| 3 Mb region, cold | 11.1 / 1.9 / **1.4** | 33.7 / 2.3 / **1.5** | 166 / 2.9 / **2.1** | 620 / 5.0 / **4.2** | 192 / 116 / 97 | 10,959 / 65 / 57 |
| 500 scattered, open | 3.0 / **2.9** / 4.4 | 7.7 / **7.6** / 8.7 | 35.8 / 34.9 / **34.2** | 133 / 132 / **126** | 29 / 39 / 125 | 2,366 / 2,356 / 2,221 |
| 500 scattered, warm | **0.056** / 0.063 / 1.8 | **0.055** / 0.057 / 4.8 | 0.050 / **0.041** / 22.5 | **0.071** / 0.085 / 84.1 | 0 / 0 / 28 | 0 / 0 / 1,506 |
| single pair, open | 0.005 / 0.005 / 0.007 | 0.009 / 0.009 / 0.012 | 0.032 / 0.032 / 0.035 | 0.113 / 0.113 / 0.116 | 0.021 / 0.043 / 0.2 | 2 / 2 / 2 |

The same reads with every run of missing chunks fetched at once (`--reader concurrent`, below):

| read | local | +10 ms | +50 ms |
|---|---|---|---|
| clump, r² 0.01, cold | 27.7 / 12.0 / **9.4** | 79.0 / 34.1 / **24.8** | 289 / 111 / **71.3** |
| clump, r² 0.001, cold | 22.3 / 10.6 / **8.5** | 63.5 / 30.4 / **23.2** | 232 / 97.9 / **66.2** |
| 3 Mb region, cold | 11.7 / 2.0 / **1.5** | 20.3 / 3.2 / **2.4** | 68.2 / 4.9 / **3.9** |
| 500 scattered, open | 2.2 / **2.0** / 2.1 | 3.8 / **3.7** / 3.9 | 14.8 / 14.6 / **14.0** |
| 500 scattered, warm | **0.057** / 0.065 / 1.4 | **0.055** / 0.065 / 2.6 | **0.054** / 0.077 / 9.5 |
| single pair, open | 0.006 / **0.005** / 0.008 | **0.031** / 0.032 / 0.034 | 0.113 / 0.113 / 0.117 |

The node's 1-minute load was 4–8 through ogld's reader's runs, rising to 26 in their last minutes,
and 3–31 through the concurrent runs (median 22).
[`latency-tables.md`](latency-tables.md) has every state, and the load of each row.

**Clumping below t_far is fastest on C, at every latency and with either reader.** It needs one
Near Band element for each lead and each open hit within 3 Mb, the column read in `clump`'s form.
GWAS hits are dense, so a C chunk, about 29 consecutive rows, serves many of a lead's hits at once.
For the 454 leads at r² 0.01, C reads 2,752 chunks in 784 calls, 1.7 calls a lead; B reads 9,913
in 3,827 (8.4 a lead) and A 20,771 in 19,473 (42.9). C decodes the most, 1,443 MB against B's
650 MB, and still wins locally, because each Zarr call pays a fixed cost, and over HTTP each call
is two sequential round trips (the shard index, then the chunks). With ogld's reader the time is
the 0 ms time plus one round trip per request. At 50 ms that is 51.6 ms a request on C and 53 ms
on B, so the gap grows with latency: 3.4× B's time at 50 ms. Concurrent reads narrow it to 1.6×,
because a lead's chunks are then fetched together, but leads still go one after another.

**C reads the most bytes for a clump.** It fetches 683 MB against B's 415 MB. Loopback does not
price those bytes. On a 100 Mbit/s link the 268 MB difference costs about 21 s. That is less
than B's extra round trips at 10 ms (88 s) or 50 ms (345 s) with ogld's reader, and about what
they cost with concurrent reads at 10 ms (9 s) and 50 ms (40 s). So on a slow link with a
concurrent reader, B and C clump about equally fast.

**A region is fastest on C, and A is ruled out again.** A region reads each row's codes inside
the region as one slice (spec §10). On B and C the slices of a block of rows fall in contiguous
chunks, read in one call per block: 65 and 57 requests. On A the gap between one row's slice and
the next spans whole 8 KiB chunks, so the chunks are not contiguous, and the read makes 10,959
requests. That is 166 s at 10 ms against 2.1–2.9 s. A's 16 KiB shard index is also larger than its
chunks and is read on every call: 319 MB of the 420 MB A's chr22 clump fetches.

**B wins the small reads on bytes, not on time.** For 500 scattered variants B reads 39 MB
against C's 125 MB, and for a single pair 43 KB against 242 KB. Both reads are bound by requests,
about the same number on every layout, so over HTTP the layouts take the same time. Locally B
reads the 500 variants 1.5× faster (2.9 s against 4.4 s). On a 100 Mbit/s link, B's 86 MB saving
is about 7 s on the 500 variants, and 199 KB is 16 ms on a single pair.

**B and A keep the 500 variants warm, and C does not.** Their working set fits the reader's
256 MiB chunk cache: 638 B chunks (40 MiB) against 478 C chunks (239 MiB), beside the Proxy Index
chunks. C's overflows it, and LRU then evicts the index chunks the next run needs. So a repeated
read is warm on A and B (41–85 ms) and is not on C: 1.8 s locally and 84 s at 50 ms. The chunk
cache's size is the lever there, not the layout (below).

## The Far Pairs: what merging them into the Proxy Index costs

Spec §11 merged the Far Pairs into the Proxy Index on 8 November. It said a region wider than
`near_bp` would read about 3–9% more bytes. It also said a matrix for a list of ALIDs would be
unchanged: "each reads one row per variant, and a row of either component sits in one inner
chunk, so the requests are the same". The first holds; the second does not.

**A region 10 Mb wide.** Its variants' Proxy Index rows were 17.3–26.7 MB beside a Near Band read
of 241–293 MB: **6.2–9.3% more bytes**, as §11 estimated. The time is larger. The far-pair
scatter took 2.2–3.5 s of 7.6–19.7 s over the two local runs, **13–30%**, and 1.7–3.0 s at 0, 10
and 50 ms (14–28%; at 50 ms the Near Band's round trips dominate the read). It reads every entry
of every row, 11.4–16.6 M of them, to place the 6,228–11,651 Far Pairs inside the region; 99.6% of
those entries are copies of Near Band pairs within 3 Mb, which it drops. A separate Far Pairs component
holds 16,541–24,511 entries for the same rows, 0.13–0.21% as many.

**A matrix for a list of variants on one chromosome.** Once the list spans more than 3 Mb,
`pair_r` reads the Proxy Index row of every listed variant. An EUR chr22 row holds about 380
entries, so a 16,384-entry chunk holds about 43 rows. 500 rows drawn from 119,632 rarely share
one: they touch 433–448 chunks of each of `index/22/partner` and `index/22/r`. A Far Pairs
component on chr22 holds 101,886 pairs, under one a row, so its whole array is 7 chunks, and the
same 500 rows touch 5–6 of them. **So the requests are not the same: about 890 chunks against
about 11.** The far part of the 500-variant read, on every layout:

| reader | local | +0 ms | +10 ms | +50 ms |
|---|---|---|---|---|
| ogld's | 1.4 of 4.4 s (32%) | 3.9 of 8.7 s (44%) | 20.0 of 34.3 s (58%) | 76.2 of 126 s (60%) |
| concurrent | 1.0 of 2.1 s (48%) | — | 2.0 of 3.9 s (52%) | 8.1 of 14.0 s (58%) |

(layout C, seed 8, chunk cache emptied.) With a Far Pairs component that part would be about 11
chunk reads: about 0.2 s at 10 ms and 1 s at 50 ms with ogld's reader, estimated from the time
per request above. So the merge more than doubles a regional list matrix over a network,
whichever reader reads it.

**chr22 holds few Far Pairs; the saving holds on every chromosome.** chr22 has 0.85 Far Pairs a
row, where the store averages 15.5 and chr11 162 (`scripts/bench/far_pairs.py`, which counts
every chromosome's from its Proxy Index; the counts sum to the 132,924,736 the store's
provenance records). The review of ogld#8 estimated from those averages that a Far Pairs read
would touch only 3–5× fewer chunks than the index on the dense chromosomes. Counted, it touches
10–16× fewer even on chr11, because Far Pairs cluster in rows: on every chromosome only 14–19% of
rows hold any (chr11's 69 M lie in 76,296 of its 427,295 rows), so most listed rows read nothing,
and those that do share chunks. For three draws of 500 variants on each chromosome (seeds 8–10, as
the benchmark draws them), the distinct chunks of one array each would read:

| chromosome | Far Pairs | a row | rows holding any | index chunks | Far Pairs chunks | fewer |
|---|---|---|---|---|---|---|
| 1 | 816,962 | 1.23 | 16% | 501–507 | 28–29 | 17–18× |
| 2 | 1,066,935 | 1.51 | 16% | 502–507 | 26–30 | 17–20× |
| 3 | 2,321,033 | 3.81 | 17% | 504–512 | 33–37 | 14–16× |
| 4 | 858,715 | 1.37 | 16% | 503–510 | 24–28 | 18–21× |
| 5 | 6,560,407 | 11.88 | 17% | 508–512 | 31–32 | 16× |
| 6 | 12,150,924 | 20.79 | 18% | 515–528 | 36–45 | 12–14× |
| 7 | 5,626,236 | 11.03 | 17% | 505–513 | 23–29 | 17–22× |
| 8 | 2,778,549 | 5.96 | 18% | 498–508 | 25–34 | 15–20× |
| 9 | 335,049 | 0.90 | 16% | 491–501 | 16–17 | 29–31× |
| 10 | 13,381,159 | 30.26 | 18% | 507–511 | 28–34 | 15–18× |
| 11 | 69,017,129 | 161.52 | 18% | 522–529 | 32–51 | 10–16× |
| 12 | 9,210,890 | 22.21 | 17% | 503–512 | 25–30 | 17–20× |
| 13 | 271,424 | 0.84 | 16% | 498–503 | 14–16 | 31–36× |
| 14 | 212,673 | 0.76 | 16% | 486–497 | 11–12 | 41–45× |
| 15 | 138,945 | 0.58 | 16% | 470–482 | 9 | 52–54× |
| 16 | 287,786 | 1.11 | 17% | 478–480 | 11–15 | 32–44× |
| 17 | 1,512,699 | 6.55 | 16% | 475–483 | 14–15 | 32–34× |
| 18 | 365,278 | 1.47 | 16% | 478–488 | 12–13 | 37–41× |
| 19 | 3,520,100 | 17.48 | 19% | 470–484 | 16–27 | 18–29× |
| 20 | 2,320,861 | 12.19 | 17% | 459–469 | 16–22 | 21–29× |
| 21 | 69,096 | 0.58 | 14% | 439–457 | 5 | 88–91× |
| 22 | 101,886 | 0.85 | 16% | 433–448 | 5–6 | 72–89× |

A genome-wide list spreads its 500 rows over 22 chromosomes, about 23 each. Step 1's three draws
touch 514–521 index chunks of each array and would touch 71–87 Far Pairs chunks, 5.9–7.3× fewer.
Its far part took 2.3–5.8 s of 15–23 s locally (15–30%), and the string columns cost more
(below).

**Proxies and clumping are unaffected.** `proxies` reads one index row a variant, and so does
`clump` at r² ≥ t_far: the index is what they need. Below t_far they read the Near Band.

## Reader levers, measured but not implemented

None of these changes an answer or the format. Each is measured from outside `ogld/`, and the
read path is unchanged until the owner has decided the layout.

**`ChunkCache._counted`'s per-string loop.** A string chunk is counted one string at a time
(`len(s.encode("utf-8"))` over `chunk.tolist()`) to bound its arena. It took 62 s of the 1,043 s
of step 1's cold clumping at r² 0.01 (5.9%) and of the 716 s at r² 0.001 (8.6%). Of a genome-wide
500-variant matrix it took 20%, and of `proxies` for 100 variants 22%. It runs inside the string
reads, `rows_of` and `variants`, which are 24–35% of clumping and 70% of those proxies. The
lever is counting with numpy's vectorised string functions instead; it was not tried here.

**The chunk cache's size.** 256 MiB holds 512 of C's Near Band chunks. A read whose chunks do not
fit gains nothing from being repeated:

- clumping: warm took 91% of cold at r² 0.01 (946 s against 1,043 s) and 87% at 0.001;
- a 10 Mb region: the same 286 MB again, warm or cold;
- 500 variants on chr22: 1.8 s warm on C, against 56–63 ms on A and B, whose chunks fit.

With a 4 GiB budget (`--cache-mib 4096`, load 3.5–3.9), a repeated read is warm. Clumping the four
Analyses again took 38.0 s at r² 0.01 against 79.9 s cold, and 39.7 s against 71.6 s at 0.001. The
500 variants took 0.13 s warm on chr22 and 0.52 s genome-wide, against 1.7–2.9 s and 10–15 s with
256 MiB. A single cold pass reads distinct chunks, which no budget saves; #8's profile found a
4 GiB cache saved 11% of one.

**Sequential runs of chunks.** `ChunkCache._chunks` reads each run of neighbouring missing chunks
in its own Zarr call, one after another. Over HTTP each call is two round trips in sequence: the
shard's index (zarr re-reads it on every call), then the chunks. So with ogld's reader every
request waits its own round trip: step 2's clumps cost 51.6–53.6 ms a request at 50 ms. Fetching a
request's runs together (`--reader concurrent`, a prototype in `scripts/bench/latency.py`; same
chunks, same answers, checked on a clump and a matrix) reads 500 chr22 variants 8.8–9× faster
at 10 and 50 ms (34.2 s → 3.9 s and 126 s → 14.0 s on C). It clumps 1.8–2× faster on C, and
3.9–4.4× on B, whose leads need more runs. On the published store at 50 ms it clumped the four Analyses in
339 s against 1,110 s at r² 0.01 (253 s against 766 s at 0.001), read the 500 variants in 15.6 s
against 129 s on chr22 and 44 s against 222 s genome-wide, and `proxies` for 100 variants in 6.9 s
against 32 s, though its runs met a load of 3.7–34 against ogld's 1.7–9.1. Caching each shard's
index would remove the first of the two round trips a call makes; that was not measured.

## Decisions

The owner ruled on 12 November 2026, on ogld#8, from this benchmark's recommendations, which follow
as they were made, with their numbers corrected where the review of ogld#8 found them wrong:

1. **Layout C is final** (ADR 0006; spec §6).
2. **The Far Pairs come back** as a component of their own, before the first release (ogld#31,
   which rewrites spec §11).
3. **The reader's speed-ups are ogld#32**: fetching a read's runs of chunks together, the chunk
   cache's budget and `ChunkCache._counted`.

**Keep layout C.** It is the fastest layout for the read that costs most, clumping below t_far,
at every latency and with either reader: 1.3–3.4× faster than B with ogld's reader, and 1.25–1.6×
with concurrent reads. It is also the fastest region read, the smallest store (B +18.9%, A +32.4%)
and the fastest build (B 1.6×, A 6.9× slower on chr22). B's advantage is bytes on small reads:
3.2× fewer for 500 scattered variants (39 MB against 125 MB) and 5.6× fewer for a single pair
(43 KB against 242 KB). Those reads are bound by requests, not bytes, and take the same time on
B and C at every latency here. B's other
advantage, a 500-variant working set that stays warm, belongs to the chunk cache's size rather
than to the layout. Two things would change this: clients on links slower than about 100 Mbit/s,
which a concurrent reader also serves (there B and C clump about equally fast, and B wins the
small reads, by these byte counts), or a cache too small for C's chunks. A is ruled out: its
clumps take 2.6–15.7× C's time and its regions 7.9–147×.

**Bring the Far Pairs back before the first release.** The merge costs a 10 Mb region the 6–9%
more bytes §11 estimated, and 13–30% of its time, because the scatter walks every index entry
to find 0.2% of them. It costs a matrix for a list of variants on one chromosome much more: on
chr22 about 890 index chunks where a Far Pairs component needs about 11, and on the densest
chromosome, chr11, still 10–16× as many. That is 58–60% of the chr22 read at 10 and 50 ms with
ogld's reader, and 52–58% with concurrent reads, so such a matrix takes more than twice as long
over a network. Bringing them back costs about 1% more bytes at k = 10 (EUR's 132.9 M Far Pairs
are 0.80 GB raw beside a 78.9 GB store; §11's 0.03–0.76% was measured at k = 0), a second output
from the same scan, and the four `far.*` rules plus one holding the Far Pairs equal to the index's
far entries. `proxies` and `clump` would not change. If regional list
matrices over HTTP are not something the first release serves, the merge can stay; this
benchmark cannot weigh that use.
