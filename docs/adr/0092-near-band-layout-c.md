# The Near Band is chunked by layout C: 262,144 codes an inner chunk, 256 to a shard

Issue [#8](https://github.com/opengwas/ogld/issues/8), part of epic [#1](https://github.com/opengwas/ogld/issues/1).
The format is in `docs/spec/ld-store.md` (§6 for the chunk and shard shapes, §10 for the reads);
the measurements are in `docs/benchmarks/latency.md`, rendered from the records in
`docs/benchmarks/results/`; the terms are in `CONTEXT.md`.

**Who decided.** The owner, on 12 November 2026, on ogld#8, from the benchmark of
`orch/1/8-latency@2f354ae`: layout C is final. The same ruling brings the Far Pairs back as a
component of their own (ogld#31) and moves the reader's speed-ups to ogld#32. Each of those is
its own decision; this record is the layout's.

## Context

Spec §6 named three shapes for the Near Band's `near/<c>/r`, as (inner chunk, shard) in codes:
A (4,096, 4,194,304), B (32,768, 16,777,216) and C (262,144, 67,108,864). The prototype chose C
provisionally on a 12 Mb slice, where C's lead was the variant column read that the Proxy Index
then took away (§11). §6 left the choice to ogld#8, because B read 2.5–7× fewer bytes on small
reads, which could matter to a client limited by bandwidth rather than by requests.

ogld#8 built HGDP+1kGP EUR chr22 three times by `ogld build` (× 511, k = 10, the published
store's inputs; `scripts/bench/build_layouts.py`), once in each layout. The three decode to the
same arrays, and C's shards are byte-identical to the published store's chr22. It timed each read
that differs, locally and over HTTP from a static host on loopback with 0, 10 and 50 ms added to
every request (`scripts/bench/latency.py`), with ogld's reader and with a prototype that fetches a
read's runs of chunks together.

## Decision

1. **`near/<c>/r` is chunked 262,144 codes an inner chunk and 67,108,864 to a shard** (256 inner
   chunks, 128 MiB raw), zstd level 3 inside the shard, as §6's table says. `ogld.store.layout`
   holds the shape (`NEAR_R`), and the writer takes it from there.
2. **Every other array keeps its one shape**, 16,384 to an inner chunk and 1,048,576 to a shard.
   ogld#8 measured no read the Proxy Index's or the variant table's shape held back.
3. **A shape is not a format version** (§13). A store re-sharded for serving (ogld#9) reads the
   same, and a reader depends on no shape: each array's metadata records its own.

## Considered options

What ogld#8 measured on EUR chr22, A / B / C (`docs/benchmarks/latency.md`, "Step 2"):

| | A | B | C |
|---|---|---|---|
| `near/22/r` on disk | 1,327,224,461 B (+32.4%) | 1,191,999,135 B (+18.9%) | 1,002,545,575 B |
| build, wall (8 threads) | 259 s | 60.2 s | 37.6 s |
| clump below t_far, r² 0.01, 27 Analyses: local / +10 ms / +50 ms | 37.8 / 585 / 2,243 s | 17.1 / 132 / 488 s | **12.3 / 44.3 / 143 s** |
| the same, fetching a read's runs together: local / +10 ms / +50 ms | 27.7 / 79.0 / 289 s | 12.0 / 34.1 / 111 s | **9.4 / 24.8 / 71.3 s** |
| 3 Mb region, cold: local / +50 ms | 11.1 / 620 s | 1.9 / 5.0 s | **1.4 / 4.2 s** |
| 500 variants, chunk cache empty: local / +50 ms | 3.0 / 133 s | **2.9 / 132 s** | 4.4 / **126 s** |
| single pair, chunk cache empty: bytes, +50 ms | 21 KB, 113 ms | 43 KB, 113 ms | 242 KB, 116 ms |

- **B.** It reads 3.2× fewer bytes than C for 500 variants (39 MB against 125 MB) and 5.6× fewer
  for a single pair. Those reads are bound by requests, about the same number on every layout, so
  they took the same time on B and C at every latency. Its 500-variant working set also fits the
  256 MiB chunk cache, where C's does not; that is the cache's size, which ogld#32 takes up, not
  the layout. Against that, B clumps below t_far 1.3–3.4× slower than C with ogld's reader and
  1.25–1.6× slower with concurrent reads. GWAS hits are dense, so a C chunk, about 29 consecutive
  rows, serves many of a lead's open hits: C makes 1.7 Zarr calls a lead, B 8.4. B is also 18.9%
  larger, about 92 GB more over the five published Near Bands, and builds 1.6× slower. Rejected.
- **A.** Its 16 KiB shard index is larger than its chunks and is read on every call, and its region
  rows leave whole chunks between them, so a 3 Mb region makes 10,959 requests where C makes 57.
  Its clumps took 2.6–15.7× C's time and its regions 7.9–147×. Rejected.

## Consequences

- **Nothing is rebuilt.** The five published stores (ogld#7) are already layout C.
- **A small read fetches whole C chunks.** A single pair is 242 KB where B's would be 43 KB, and
  500 scattered variants 125 MB where B's would be 39 MB. On loopback that cost nothing. A client
  on a link slower than about 100 Mbit/s pays it, and there, with concurrent reads, B and C would
  clump about equally fast by the byte counts; no such client was measured.
- **Clumping reads many bytes.** #403's 37 Analyses at r² 0.01 fetch 75 GB of Near Band chunks,
  10 minutes at 1 Gbit/s. That is a property of the column read below t_far on any layout; C reads
  1.6× B's bytes for it and makes 3.5× fewer requests.
- **The chunk cache must hold C's chunks for a repeated read to be warm.** At 256 MiB it holds
  512; a 500-variant list or a 10 Mb region needs more. Its budget is ogld#32's, with the reader's
  concurrent fetching, which ogld#8 measured at 5–9× faster for lists over HTTP.
