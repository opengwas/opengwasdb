# An ALID search index: what a lookup costs, and what it saves

_Ticket [#42](https://github.com/opengwas/ogld/issues/42), ADR [0010](../adr/0096-alid-search-index.md).
Measured 15 November 2026 on the IEU node, against opengwas/ogld#40's baseline
[`fast-decode-round2.md`](./fast-decode-round2.md) and opengwas/ogld#8's
[`latency.md`](./latency.md)._

## In brief

- **Finding the variants of #403's 37 Analyses took 94.0 s of the 272.5 s they clumped at
  r² 0.01, cold, 8 threads** (opengwas/ogld#40's phase table). Reading ALID strings was 59.1 s of it and
  loading chromosomes to search their positions 31.1 s. With the ALID index it is **50.4 s**, and
  the same 37 clumps take **229.4 s**. All 37 result fingerprints are identical, lead for lead.
- **The index's own cost is the chunk count, not the bytes.** 8.55 M keys are 273 MB decoded.
  In 16 KiB inner chunks that is 16,693 chunks, and a whole read took **4.6 s**; in 262 KiB chunks
  it is 1,044 chunks and **0.2–0.7 s**. The index is written in 262 KiB inner chunks, 16 MiB
  shards, and the rows in 262,144-row chunks.
- **A short list must not be sent to the index.** One ALID's exact scan is 0.07 s and 500 of
  them 0.11 s, where reading the whole index is 0.9 s cold; at 4,000 ALIDs the index wins and
  keeps winning. So a list naming about a two-thousandth of the Variant Set or more reads the
  index, and a shorter one takes the exact scan. A binary search over the index, reading only the
  chunks one key's path touches, was measured and rejected: with chunks large enough to read the
  store's index in a second, a few hundred keys touch nearly all of them.
- **A lookup reads the index one shard at a time** — the unit the store itself reads and fetches,
  and the batch comes from the array's own shapes, so a store written with others is read one of
  its shards at a time too. On EUR a shard is 16 MiB of keys: 524,288 of them, 64 inner chunks,
  each shard let go before the next is fetched, so a lookup holds two shards' worth at most (50 MB
  without that release, 33.6 MB with it, at 1.1 M keys). Reading the array whole held 273 MB of
  keys twice (548 MB peaked, measured by opengwas/ogld#42's first review). For a 260,821-key list: one chunk a
  read 4.4 s, 16 chunks 1.0 s, a whole shard 0.79 s, one whole read 0.70 s.
- **opengwas/ogld#8's step-1 reads are unchanged**, since every list they time is short: the same chunks and
  the same requests, and the same times within the run-to-run spread of this node.

## The 37 Analyses, r² 0.01, cold

`scripts/bench/latency.py`'s step 1, cold only, a fresh reader and advisory page eviction per
Analysis, `taskset -c 0-7`, BLAS 1, seven shared decode workers, zarr's helper at one thread.
Uptime before and after each run is in the records; the load was 1.5–3.6 before and 2.6–5.1 after,
against opengwas/ogld#40's 3.9–6.8. The phase times of `rows_of`, `prefetch` and `partners` overlap, as they do
in opengwas/ogld#40's table: they are the seconds spent in each, not a partition.

| phase | opengwas/ogld#40, no index | with the index |
|---|---:|---:|
| open | 0.87 s | 0.87 s |
| `rows_of` | 94.05 s | **50.40 s** |
| ↳ chromosome loads, inside `rows_of` | 31.11 s | 28.69 s |
| ↳ ALID string reads (`_strings`) | 59.09 s | 9.09 s |
| `prefetch` | 161.17 s | 148.40 s |
| `partners` | 9.89 s | 9.97 s |
| greedy bookkeeping | 4.13 s | (in `greedy`, 162.44 s) |
| **37 cold clumps** | **272.47 s** | **229.39 s** |

23,445 leads and 1,608,735 hits in both runs, and all 37 fingerprints equal, byte for byte.
Chromosome loads are still inside `rows_of` because 40,080 of EUR's 8,586,889 ALIDs are wider than
the slot, so an Analysis's list always holds some, and each takes the exact scan at its position.
The remaining 9.09 s of string reads is theirs.

Per Analysis, the index wins where the list is long and is a wash where it is short — which is why
the reader keeps the scan for short lists:

| hits | `rows_of`, no index | with the index |
|---:|---:|---:|
| 520 | 0.54 s | 0.70 s |
| 2,801 | 2.60 s | 0.67 s |
| 13,513 | 1.96 s | 1.10 s |
| 65,072 | 4.07 s | 3.08 s |
| 107,358 | 4.67 s | 1.03 s |
| 260,821 | 5.48 s | 3.55 s |

## The index read, by chunk

`chunk-probe.py` read the whole key array of the published EUR store through the chunk cache
(`ChunkCache.runs`) with the pages warm, and 1, 10 and 100 random keys through `ChunkCache.take`:

| inner chunk | shard | whole read | 1 key | 10 keys | 100 keys |
|---:|---:|---:|---:|---:|---:|
| 16 KiB | 1 MiB | 4.62 s | 13 ms | 25 ms | 169 ms |
| 64 KiB | 1 MiB | 2.48 s | 6 ms | 19 ms | 155 ms |
| 256 KiB | 1 MiB | 0.44 s | 4 ms | 13 ms | 118 ms |
| 256 KiB | 4 MiB | 0.66 s | 6 ms | 20 ms | 149 ms |
| 256 KiB | 16 MiB | 0.23 s | 7 ms | 15 ms | 116 ms |
| 512 KiB | 4 MiB | 0.25 s | 5 ms | 16 ms | 94 ms |
| 1 MiB | 4 MiB | 0.29 s | 7 ms | 19 ms | 107 ms |

The store writes 262 KiB inner chunks in 16 MiB shards: one key's path reads at most a few
shards' indexes and one chunk a probe, and the whole array is 1,044 chunks.

## A list, cold, by length

`list-probe.py`: random ALIDs from the published EUR store, pages evicted and a fresh reader for
each length, on one core set, the head and the tip alternately.

| ALIDs | exact scan | whole index |
|---:|---:|---:|
| 1 | 0.12 s | 0.10 s |
| 10 | 0.07 s | 0.10 s |
| 100 | 0.07 s | 0.28 s |
| 500 | 0.11 s | 0.33 s |
| 1,000 | 0.13 s | 1.01 s |
| 2,000 | 0.17 s | 0.73 s |
| 5,000 | 0.29 s | 0.80 s |
| 20,000 | 0.88 s | 1.23 s |
| 100,000 | 4.33 s | 4.49 s |

A whole list of one real Analysis is where they cross: 91,618 ALIDs, 3.93 s against 3.06 s, and
260,821 ALIDs, 5.64 s against 2.74 s, both cold.

## opengwas/ogld#8's step-1 reads, head against tip

`scripts/bench/latency.py`'s step 1, `--case proxies scattered --seeds 1`, locally and over HTTP
with 50 ms added per request, the tip and the head in turn on disjoint cores. Medians of the
records; `proxies` was run three times and `scattered` once.

| read | state | local, tip | local, head | 50 ms, tip | 50 ms, head |
|---|---|---:|---:|---:|---:|
| proxies, 1 variant | cold | 0.126 s | 0.135 s | 1.426 s | 1.435 s |
| proxies, 1 variant | open | 0.036 s | 0.033 s | 0.360 s | 0.355 s |
| proxies, 100 variants | cold | 2.149 s | 3.059 s → 2.17–2.32 s | 20.89 s | 21.46 s |
| proxies, 100 variants | open | 1.038 s | 1.321 s → 1.03–1.04 s | 7.41 s | 7.32 s |
| 500 scattered, chr22 | cold | 0.653 s | 0.960 s | 7.018 s | 7.021 s |
| 500 scattered, genome | cold | 7.619 s | 7.545 s | 38.16 s | 38.35 s |

The first head run of `proxies, 100` was 1.4× the tip's, and the reads were identical: 81 gets of
`variants/alid`, 197 chunks of `index/partner` and 197 of `index/r` in both, and 537 chunks for
chr22's 500 scattered variants in both. Two more runs of `proxies, 100` put the head at 2.17 and
2.32 s against the tip's 2.13 and 2.21. The node's local times
vary by that much (see `latency.md`, "How much a local timing varies"); the request and chunk
counts, which do not vary, are the same, because a list this short takes the exact scan in both.

## What was changed to make this true

- `variants/alid_index/alid` holds one 32-byte key an entry and `variants/alid_index/row` the
  Variant Set row of each, in 262 KiB inner chunks and 16 MiB shards, `uint8` and `int32`.
- `rows_of` reads the index whole for a long list and searches it with `numpy.searchsorted`; every
  ALID the index does not hold, and every ALID of a short list, takes the exact scan.
- A one-byte `bytes` codec declares no byte order, which `ogld/store/decoding.py` now accepts for
  a one-byte element only, so the keys decode through opengwas/ogld#40's direct decoder as every numeric array
  does (ADR 0095).

## What was rejected

- **16 KiB chunks.** 4.6 s a whole read: 16,693 chunk decodes, where the Near Band's 512 KiB
  chunks make it cheap to read many bytes and expensive to read many chunks.
- **A binary search for a few keys.** Measured above: at 500 keys, 0.33 s against the scan's
  0.11 s, because with chunks this large the probes touch most of the index anyway.
- **Sending short lists to the index.** 0.9 s of index for an ALID whose scan is 0.07 s.
- **A `(n, 32)` `uint8` array, or a fixed-width `|S32` Zarr array.** The first is not
  one-dimensional, so the direct decoder does not read it; the second is an unstable Zarr
  extension (ADR 0096).

## The records

`clump-final.jsonl` (the 37), `reads-final.jsonl` and `reads-tip.jsonl` (step 1, head and tip),
`proxies-head-*.jsonl` and `proxies-tip-*.jsonl` (the repeats), `chunk-probe.py`, `list-probe.py`
and `analysis-probe.py` under `/data/opengwasdb/work/ogld/scratch/42/`. Each record names its
uptime before and after, its load, the chunks and the bytes it read, and its result fingerprint.
