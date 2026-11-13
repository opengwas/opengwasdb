# The fast reader: ogld#8's reads after ogld#32

_Measured 13 November 2026 with ogld#8's benchmark (`scripts/bench/latency.py`) on the
published HGDP+1kGP EUR store, at ogld `c5a2eb8`. Every number is rendered from the records in
[`results/32/`](results/32/), against ogld#8's in [`results/`](results/), by
`scripts/bench/deltas.py`, and [`fast-reader-tables.md`](fast-reader-tables.md) has all of them._

## In brief

With all four of ogld#32's changes, against ogld#8's tables, on the draws both timed:

- **A matrix of 500 chr22 variants** makes 1,241 requests where it made 2,245. Cold, it takes
  14.8 s at 50 ms against 129 s (8.7×), 5.8 s at 10 ms against 40 s (6.9×), and 1.8 s locally
  against 6.6 s (3.6×). Drawn genome-wide: 41.8 s against 222 s at 50 ms (5.3×).
- **Clumping ogld#8's four Analyses** makes 7,537 requests where it made 18,408 at r² 0.01. It
  takes 224 s against 1,110 s at 50 ms (5.0×), 102 s against 349 s at 10 ms (3.4×) and 52 s
  against 91 s locally (1.7×). At r² 0.001: 205 s against 766 s (3.7×) at 50 ms.
- **A repeated read is warm.** The 500 chr22 variants read again in 0.12 s with no request,
  against 86 s at 50 ms; a 10 Mb region in 6.8 s against 17.5 s, every chunk held.
- **`proxies` for 100 variants**, on an opened store: 7.1 s against 32 s at 50 ms (4.5×).
- **Unchanged:** `proxies` for one variant, three reads that each need the last (6 requests),
  and a cold region, which is bound by decoding, not requests: 3 Mb in 1.5 s and 10 Mb in
  7–11 s locally, as before, though it makes 45 and 149 requests where it made 58 and 224.
- **Every answer is the same.** The 202 records of 28 reads timed by both tickets name the same
  leads, stored pairs and proxies.

## What changed

ogld#32 made four changes to how the reader fetches, and none to an answer or a stored byte
(spec §10):

- **Concurrent runs.** A read's missing runs of chunks are cut where a shard ends and fetched
  in batches, every piece of a batch at once on zarr's event loop: at most zarr's
  `async.concurrency` pieces (10) and 256 MiB a batch, as the cache counts memory, one batch at a
  time. Before, each run was its own Zarr call, one after another.
- **Kept shard indexes.** The chunk cache keeps each shard index zarr reads, in its budget, so a
  read of a shard read before is one request, not two.
- **A vectorised string count.** A string chunk's memory is counted without visiting its
  strings in Python.
- **A 4 GiB cache.** `ogld.open(location, cache_bytes=...)`, 4 GiB by default, was 256 MiB.

## Each lever alone

`latency.py --without` turns a lever off from outside `ogld/`, as it was before ogld#32. With all
four off it reproduces ogld#8's request counts exactly on these reads: 2,245, 2,226 and 1,512 for
the chr22 list cold, opened and warm. Each lever on alone, the others off, at 10 ms
(`fast-reader-tables.md` has every row):

| read | none (ogld#8's reader) | concurrency | indexes | count | cache | all four |
|---|---|---|---|---|---|---|
| 500 chr22, cold: requests | 2,245 | 2,245 | 1,243 | 2,245 | 2,243 | 1,241 |
| 500 chr22, cold: seconds | 37.0 | 7.07 | 22.3 | 35.3 | 37.4 | 7.07 |
| 500 chr22, warm: seconds | 27.5 | 4.56 | 15.3 | 25.3 | 0.24 | 0.24 |
| 500 genome, cold: seconds | 69.1 | 22.9 | 61.7 | 72.1 | 74.1 | 20.8 |
| clump GCST90319613, cold: requests | 1,832 | 1,832 | 1,212 | 1,832 | 1,796 | 1,184 |
| clump GCST90319613, cold: seconds | 39.3 | 17.5 | 28.5 | 38.6 | 38.7 | 17.3 |
| clump GCST90319613, warm: seconds | 33.4 | 13.2 | 24.1 | 33.7 | 0.78 | 0.61 |

- **Concurrency** changes no request and most of the time over a network: a 500-variant list
  5.2× faster at 10 ms, a clump 2.3×.
- **Kept indexes** take 45% of a list's requests and 34% of a clump's.
- **The cache** makes a repeat warm, and changes nothing on a first read.
- **The count** shows nothing over HTTP, where round trips dominate. Locally, the time inside it
  fell from 3.29 s of a cold genome-wide list's 17.0 s (19%) to 1.12 s of 9.03 s (12%), and from
  0.41 s to 0.18 s of `proxies` for 100 variants, opened.

A first version of the concurrent reader made 1,665 requests for the chr22 list with every lever
on, more than kept indexes alone. Two runs of one shard, fetched at once, both missed its index
and both fetched it. They now share one fetch (`_KeptIndexes._shared`), and all four make 1,241.

## How

ogld#8's method, unchanged: the store's directory served by `scripts/bench/http_static.py`, a
static host on loopback with a proxy that waits the set time before forwarding each request and
counts the requests and bytes. **This is loopback with injected latency, not a real WAN.** zarr's
thread pool is 6 threads and BLAS 1. See [`latency.md`](latency.md), "How", for the states
(`cold`, `open`, `warm`) and for how much a local timing varies on this shared node: up to about
45% between runs, so a local difference under about half is noise. Over HTTP with added latency
the round trips dominate, and repeat closely.

**Load.** The node's 1-minute load was 4–35 through these runs, median 9, against 1.7–44,
median 5.7, through ogld#8's step-1 runs. A cold 10 Mb region read 0.8× as fast locally and at 10 ms as in ogld#8's run. That is
within the noise above: run back to back with concurrency on and off, it took the same time
(8.0–11.5 s against 7.7–11.3 s over three centres), and per centre it was faster in some places
and slower in others. Runs timed between 23:50 and 00:20 UTC, while another lane read 6 GB of
this store and ran the test suite, were discarded and re-run.

**The runs.** Each file in [`results/32/`](results/32/) is one run, in this order, at
`c5a2eb8`, whose `code` record says the checkout was clean. Each took
`pixi run -e dev bench-latency step1` and these arguments:

| file | started (UTC) | load | arguments |
|---|---|---|---|
| `lever-off` | 00:33 | 16.9 | `--target http10 --case scattered clump --analyses OGS-00011__GCST90319613 --r2 0.01 --seeds 1 --without concurrency indexes count cache` |
| `lever-concurrency` | 00:40 | 12.3 | the same, `--without indexes count cache` |
| `lever-indexes` | 00:41 | 6.8 | the same, `--without concurrency count cache` |
| `lever-count` | 00:46 | 5.4 | the same, `--without concurrency indexes cache` |
| `lever-cache` | 00:53 | 17.9 | the same, `--without concurrency indexes count` |
| `lever-on` | 00:57 | 12.7 | the same, with no `--without` |
| `step1-reads` | 00:58 | 10.3 | `--target local http10 http50 --case proxies scattered region --seeds 1` |
| `step1-clump` | 01:06 | 6.9 | `--target local http10 http50 --case clump --cold-only --analyses` (ogld#8's four) |

The four Analyses are `OGS-00009__ukb-b-4171 OGS-00011__GCST90319613 OGS-00011__GCST90025965
OGS-00011__GCST90029004`. The tables were rendered by:

```bash
pixi run -e dev python scripts/bench/deltas.py --before docs/benchmarks/results/step1-*.jsonl.gz \
    --after docs/benchmarks/results/32/step1-*.jsonl.gz \
    --levers docs/benchmarks/results/32/lever-*.jsonl.gz > docs/benchmarks/fast-reader-tables.md
```
