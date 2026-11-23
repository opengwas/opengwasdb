# Scattered Dense lookups: what a Rust spike and a smaller inner chunk gain, and the options they leave

Issue [#477](https://github.com/opengwas/opengwasdb/issues/477), part of epic #450. ADR 0081 refused a compiled
extension after epic #307 and said it would be reconsidered only if a phase on the milestone's critical path showed
a per-row interpreted kernel that vectorised code cannot express. #477 asked, as a time-boxed spike, whether the
`zarrs` crate or a smaller inner chunk would make scattered lookups fast enough to reopen that.

**Status:** proposed. This ADR records measurements and sets out options A-D; it decides none of them. The owner
decides. ADR 0081 stands until then.

## Context and method

Scattered lookups on the current OGS-00009 (100 variants x 10 Analyses) take about 1.3 s through the Python reader.
The measurements are in [`rust_spike_ogs00009_tables.md`](../benchmark-output/rust_spike_ogs00009_tables.md),
generated from [`rust_spike_ogs00009.json`](../benchmark-output/rust_spike_ogs00009.json); the setup, caveats,
provenance and exact commands are in [`rust_spike_ogs00009.md`](../benchmark-output/rust_spike_ogs00009.md). Every
number below is read from those tables.

Seed-0 selection. The two lookup shapes had 7 interleaved rounds, each candidate a fresh process (1 cold call, then
3 warm calls); the harness's other shapes (whole Analysis, phewas, regional, regional one Analysis) had 5 rounds of
1 cold and 1 warm call, with the Python reader only. Ratios are computed within a round and quoted as median (range
over rounds). The effective codec pipeline, zarr `max_workers` and Blosc threads were read back in every Python
process, and the run aborts if they differ from what the candidate names. Rust's thread count and its zarrs codec
concurrency (capped at one) are recorded as the binary set them, not read back. Every cold and warm output of every
candidate has the Python reader's digests (for Rust, the cold call and the untimed warm-up call). The host was shared
and busy (1-minute load 11.1-25.0, median 21.4), so ratios are firmer than absolute times.

The 5x bar is judged **end to end** (the whole `lookup`), on **both** shapes. The ALID-to-row step is 0.06 % and 0.07 %
of the lookup, so the time is in the cell read and decode.

## Findings

End-to-end warm medians, 10 variants x 100 Analyses / 100 x 10; ratio against the shipped reader (above 1 is faster):

| candidate | 10 x 100 | 100 x 10 |
|---|---|---|
| Python reader as shipped (fused, 1 worker) | 430.1 ms | 1296.7 ms |
| zarrs-python pipeline (verified `ZarrsCodecPipeline`) | 1318.5 ms, 0.32x (0.23-0.37) | 3601.7 ms, 0.36x (0.24-0.45) |
| zarr's own pool, 16 workers, Blosc threads on (a set-up `store_reader/arrays.py:107-110` records as queueing on numcodecs' lock) | 1107.4 ms, 0.39x (0.30-0.44) | 2967.5 ms, 0.44x (0.32-0.46) |
| Python thread pool, 16 threads, Blosc threads off | 605.7 ms, 0.69x (0.45-0.73) | 3062.9 ms, 0.43x (0.34-0.45) |
| Rust, 1 thread | 375.2 ms, 1.13x (1.00-1.18) | 785.9 ms, 1.65x (1.42-1.67) |
| Rust, 16 threads | 80.4 ms, 5.37x (4.23-7.34) | 111.0 ms, 11.73x (7.68-12.39) |
| smaller inner chunk 100x64, Python reader | 216.9 ms, 1.96x (1.54-2.10) | 875.6 ms, 1.50x (1.26-1.55) |

1. **Threaded Rust clears the bar on the 100 x 10 shape and does not clearly clear it on 10 x 100.** The warm median
   is 5.37x, but rounds run from 4.23x to 7.34x; the cold ratio is 4.99x (3.60-5.60), below the bar at its median;
   the decode stage alone is 5.28x (4.47-6.10).
2. **Matched thread counts.** One decode worker against one (Blosc's internal threads as the shipped reader leaves
   them; moot for a [1000, 64] chunk, which is a single Blosc block): Rust is 1.13x (1.00-1.18), parity at worst, and
   1.65x (1.42-1.67). Sixteen against sixteen: Rust is 8.65x (6.05-10.16) and 26.27x (19.06-29.01) the Python thread
   pool. The pool of zarr's own workers is not compared, because it ran in a set-up already known to queue.
3. **The Python thread pool was slower than the serial reader, and the cause is unknown.** It issues one `grid.read`
   per row partition, so per-call overhead that holds the GIL, a serialisation point in zarr's event loop, and the
   decode itself are all candidates; none was profiled in a committed artefact. This spike therefore does not show
   whether another Python design could parallelise the decode.
4. **The smaller inner chunk (100 x 64) helps scattered lookups modestly:** 1.96x (1.54-2.10) and 1.50x (1.26-1.55)
   warm with the Python reader; cold 1.99x (1.49-2.17) and 1.43x (1.31-1.62).
5. **It does not compound with Rust.** Rust at 16 threads on the 100-row copy is 5.40x (4.85-6.72) and 12.29x
   (10.76-12.99), against 5.37x and 11.73x on the shipped chunk: no further gain from the smaller chunk.
6. **The smaller chunk costs elsewhere.** The six lookup arrays grow by 7.2 % (48,924 to 52,428 MiB): z +2.5 %,
   se +19.3 %, eaf +18.0 %, imputed +94.0 %. With the Python reader, original chunking over smaller chunk (above 1
   favours the smaller chunk), warm and cold:

   | shape | warm | cold |
   |---|---|---|
   | whole Analysis | 0.35x (0.28-0.49), 12.4 s to 35.3 s | 0.40x (0.40-0.42) |
   | phewas | 1.68x (1.45-2.20) | 1.31x (0.84-1.94), straddling 1 |
   | regional window | 0.86x (0.56-0.94) | 0.85x (0.79-0.92) |
   | regional one Analysis | 0.75x (0.63-0.96) | 0.85x (0.74-0.91) |

7. **Baseline.** The shipped reader's lookup times here are above `store_reader/arrays.py:103-105`'s 103 / 575 ms,
   and the committed facts account for most of the gap. Those figures come from the earlier OGS-00009, format 0.1.0
   (its artifact ran at load 2.9). ADR 0058 (lines 340-341) measured the 0.1.0 to 0.2.0 move on OGS-00009 as 2.80x and
   1.82x slower on these shapes, which takes 103 / 575 ms to about 288 / 1047 ms. The rest is not accounted for.

## Options for the owner

On 23 November the owner decided that hosted releases are not rewritten: once #467-#472 are in, every store is
rebuilt from source at 0.3.0 (opengwas/opengwasdb-stores#498).

* **A. Do nothing.** Keep ADR 0081 and the 1000 x 64 chunk.
* **B. A compiled Dense reader (supersede ADR 0081 for the Store Reader only).** The gain on scattered lookups needs
  parallel decode (items 1 and 2) and is borderline on 10 x 100. ADR 0081's criterion is not shown: no per-row
  interpreted kernel was found, and the measured gain is decode parallelism. The cost is a second implementation of
  the encodings (int16 z, residual SE and EAF, exception tables, imputed mask), where this project's silent failures
  live, kept equal to Python by differential tests. The spike is Dense only; Ragged and Hybrid are unmeasured.
* **C. Adopt a smaller inner chunk in the 0.3.0 rebuild.** It ships no compiled code, so it is not judged by B's 5x
  bar. It is a format change (`docs/spec/store-format.md:1498` fixes `[1000, A_c]`, and ADR 0058 chose 1000 x 64).
  Because every store is rebuilt from source at 0.3.0 anyway, C adds no rebuild, but it must land in every builder
  before that rebuild, as part of #467-#472. Build time at the 100-row chunk was not measured, and it is paid on
  every rebuild (ukb-b already takes about 11.5 h). It gains 1.5-2.0x on scattered lookups and costs +7.2 % size and
  slower whole-Analysis and regional reads (item 6). Intermediate sizes were not measured.
* **D. Python-side parallel decode.** Not demonstrated, and not closed: the one design tried was slower for a cause
  that is unknown (item 3). ADR 0056 pins one zarr worker because a forked worker hangs on zarr's pool.

| option | gain | cost | risk |
|---|---|---|---|
| A. nothing | none | none | scattered lookups stay at about 0.43 s and 1.3 s on OGS-00009 |
| B. Rust Dense reader | 5.37x (4.23-7.34) and 11.73x (7.68-12.39) at 16 threads; 1.13x and 1.65x at 1 | second implementation of the encodings; a compiled build | borderline on 10 x 100; Dense only; ADR 0081's criterion not shown; silent-failure surface doubles |
| C. 100 x 64 inner chunk | 1.96x and 1.50x scattered; phewas 1.68x warm, 1.31x cold (straddles 1) | +7.2 % size; whole Analysis 0.35x; regional 0.86x; regional one Analysis 0.75x; build time unmeasured | a format change landing in every builder (#467-#472) before the rebuild; the Python gain does not add to B |
| D. Python parallel decode | unknown | unknown | the one design tried was slower; ADR 0056's fork hazard |

## Consequences

* Nothing in `opengwasdb/` changes with this ADR; the spike lives in `benchmarks/rust_spike/` and ships nothing.
* Not measured: Ragged and Hybrid layouts, OGS-00016, a quiet host, first-touch disk reads (no root to drop the page
  cache), intermediate inner-chunk sizes, and write or build time at the smaller chunk.
