# Scattered Dense lookups: a threaded Rust reader clears the 5x bar only on one shape; no port, and a smaller inner chunk is an owner option

Issue [#477](https://github.com/opengwas/opengwasdb/issues/477), part of epic #450. ADR 0081 refused a compiled
extension after epic #307 and said it would be reconsidered only if a phase on the milestone's critical path showed
a per-row interpreted kernel that vectorised code cannot express. #477 asked, as a time-boxed spike, whether the
`zarrs` crate or a smaller inner chunk would make scattered lookups fast enough to reopen that.

**Status:** proposed. It records measurements and sets out options; the owner decides. ADR 0081 stands until then.

## Context and method

Scattered lookups on the current OGS-00009 (100 variants x 10 Analyses) take about 1.3 s through the Python reader.
The measurements are in [`rust_spike_ogs00009_tables.md`](../benchmark-output/rust_spike_ogs00009_tables.md), generated from
[`rust_spike_ogs00009.json`](../benchmark-output/rust_spike_ogs00009.json); the setup, caveats and exact commands are in
[`rust_spike_ogs00009.md`](../benchmark-output/rust_spike_ogs00009.md). Every number below is read from those tables.

Seed-0 selection; at least 7 rounds, interleaved across candidates, each candidate a fresh process (cold call, then
warm calls); ratios are computed within a round and quoted as median (range over rounds). The effective codec
pipeline, zarr `max_workers` and Blosc threads were read back in every Python process and the run aborts if they differ
from what the candidate names. Rust's zarrs codec concurrency was capped at one, with the thread count set explicitly.
Every cold and warm output of every candidate has the digests of the Python reader's. The host was shared and busy
(1-minute load 11.1-25.0, median 21.4), so ratios are firmer than absolute times.

The 5x bar is judged **end to end** (the whole `lookup`), on **both** shapes. The ALID-to-row step is 0.06 % and 0.07 %
of the lookup, so the time is in the cell read and decode.

## Findings

End-to-end warm medians, 10 variants x 100 Analyses / 100 x 10; ratio against the shipped reader:

| candidate | 10 x 100 | 100 x 10 |
|---|---|---|
| Python reader as shipped (fused, 1 worker) | 430.1 ms | 1296.7 ms |
| zarrs-python pipeline (verified `ZarrsCodecPipeline`) | 1318.5 ms, 0.32x (0.23-0.37) | 3601.7 ms, 0.36x (0.24-0.45) |
| zarr's own pool, 16 workers | 1107.4 ms, 0.39x (0.30-0.44) | 2967.5 ms, 0.44x (0.32-0.46) |
| Python thread pool, 16 threads, Blosc threads off | 605.7 ms, 0.69x (0.45-0.73) | 3062.9 ms, 0.43x (0.34-0.45) |
| Rust, 1 thread | 375.2 ms, 1.13x (1.00-1.18) | 785.9 ms, 1.65x (1.42-1.67) |
| Rust, 16 threads | 80.4 ms, **5.37x (4.23-7.34)** | 111.0 ms, **11.73x (7.68-12.39)** |
| smaller inner chunk 100x64, Python reader | 216.9 ms, 1.96x (1.54-2.10) | 875.6 ms, 1.50x (1.26-1.55) |

1. **Threaded Rust clears the bar on the 100 x 10 shape and straddles it on 10 x 100**: the median is 5.37x but
   individual rounds run from 4.23x to 7.34x.
2. **Matched thread counts.** One thread against one: Rust is 1.13x (1.00-1.18) and 1.65x (1.42-1.67) the shipped
   reader. Sixteen against sixteen: Rust is 8.65x (6.05-10.16) and 26.27x (19.06-29.01) the Python thread pool, and
   13.77x and 27.21x zarr's own 16-worker pool. In this implementation the Python parallel designs were slower than
   the serial reader; this spike does not show whether another Python design could parallelise the decode.
3. **The smaller inner chunk (100 x 64) helps scattered lookups only modestly** (1.5-2.0x with the Python reader),
   and Rust at one thread on it is 1.61x (1.30-1.77) and 2.24x (2.08-2.25) the shipped reader.
4. **The smaller chunk costs elsewhere.** The six lookup arrays grow by 7.2 % (48,924 to 52,428 MiB): z +2.5 %,
   se +19.3 %, eaf +18.0 %, imputed +94.0 %. With the Python reader, the harness's other shapes move as follows
   (original chunking over smaller chunk; above 1 favours the smaller chunk): whole Analysis 0.35x (0.28-0.49), i.e.
   12.4 s to 35.3 s; phewas 1.68x (1.45-2.20); regional window 0.86x (0.56-0.94); regional one Analysis 0.75x
   (0.63-0.96).
5. **Baseline.** The shipped reader's lookup times here are well above `store_reader/arrays.py:103-105`'s 103 / 575 ms.
   The selection does not explain it (the earlier artifact's own selection on the current store is slower, 474.8 and
   1688.2 ms), and the baseline moved little between the earlier 10-19 load and this run's 11-25. That figure was
   measured on the earlier OGS-00009 (9.85 M variants x 2,024 Analyses, observed-only), which can no longer be
   re-timed, so the gap is not reconciled; the store and the node's page-cache state are untested causes.

## Options for the owner

* **A. Do nothing.** Keep ADR 0081 and the 1000 x 64 chunk. No cost, no gain.
* **B. A compiled Dense reader (supersede ADR 0081 for the Store Reader only).** The gain on scattered lookups is
  large only with parallel decode (item 2) and is borderline on the 10 x 100 shape (item 1). ADR 0081's criterion is
  not shown: no per-row interpreted kernel was found, and the measured gain is decode parallelism. The cost is a second
  implementation of the encodings (int16 z, residual SE and EAF, exception tables, imputed mask), where this project's
  silent failures live, kept equal to Python by differential tests. The spike is Dense only; Ragged and Hybrid are
  unmeasured. Nothing on the milestone's critical path currently needs it.
* **C. Adopt a smaller inner chunk at the 0.3.0 rewrite.** #467 and opengwasdb-stores#498 rewrite every release
  anyway, so the rewrite cost is not extra. It ships no compiled code, so it is not judged by B's 5x bar. It is a
  format change (`docs/spec/store-format.md:1498` fixes `[1000, A_c]`, and ADR 0058 chose 1000 x 64), and it trades
  a 1.5-2.0x scattered-lookup gain and a 1.7x phewas gain (Python reader) against +7.2 % size and slower whole-Analysis
  (0.35x), regional (0.86x) and regional one-Analysis (0.75x) reads. Intermediate sizes were not measured.
* **D. Python-side parallel decode.** Not demonstrated. ADR 0056 pins one zarr worker because a forked worker hangs on
  zarr's pool, and the two Python parallel designs tried here were slower.

## Consequences

* Nothing in `opengwasdb/` changes with this ADR; the spike lives in `benchmarks/rust_spike/` and ships nothing.
* Not measured: Ragged and Hybrid layouts, OGS-00016, a quiet host, first-touch disk reads (no root to drop the page
  cache), intermediate inner-chunk sizes, and write or build time at the smaller chunk.
