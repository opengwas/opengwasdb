# A Rust Dense lookup meets the 5x bar by threading, not by language: no port, no re-chunk yet

Issue [#477](https://github.com/opengwas/opengwasdb/issues/477), part of epic #450. ADR 0081 refused a compiled
extension after epic #307 and said it would be reconsidered only if a phase on the milestone's critical path showed
a per-row interpreted kernel that vectorised code cannot express. #477 asked, as a time-boxed spike, whether the
`zarrs` crate or a smaller inner chunk would make scattered lookups fast enough to reopen that.

**Status:** proposed. It records numbers and a recommendation; the owner decides whether to supersede ADR 0081
for the Store Reader. Until then ADR 0081 stands.

## Context

Scattered lookups on OGS-00009 (100 variants x 10 Analyses) take about 1.3 s. The ticket suspected variant-table
cost (`VariantAxis.by_index` opens a BGZF per call) rather than codec cost. Measurements are in
[`docs/benchmark-output/rust_spike_ogs00009.md`](../benchmark-output/rust_spike_ogs00009.md) and the JSON beside it
(seed-0 selection, 1 cold + 7 warm fresh-process calls per column, results digest-equal to the Python reader, host
load 10-19 on the shared node).

## Findings

1. **Attribution.** `lookup` resolves ALIDs with `indices_by_identifiers` (a vectorised search), not `by_index`.
   ALID-to-row is 0.3-0.9 ms. The 1.3 s is decode: one decompressed 1000x64 chunk per touched (row-block, column-block)
   pair, four planes, serially. #465's BGZF change does not touch this path.
2. **zarrs codec pipeline under the Python reader: 2.5-3x slower** (1047 / 3780 ms against 412 / 1276 ms).
3. **Smaller inner chunk (100 x 64), no port: 2.0x and 1.5x** (207 / 838 ms). Below the bar, and a format
   change for every Store Release (ADR 0057/0058).
4. **Rust (`zarrs` 0.23.14), one thread: 1.2x and 1.7x** (333 / 770 ms). The language is not the gain.
5. **Rust, chunk groups on 16 threads: 5.6x and 11.8x** (73 / 109 ms), 6.2x and 12.4x on the 100-chunk copy.
   Both shapes clear the ~5x bar, the 10x100 one narrowly, on a loaded host.
6. **Python cannot reach the same parallelism today.** zarr's own pool at 16 workers is 0.4x (slower), and ADR 0056
   pins one worker because a forked worker hangs on the pool.

## Decision

**Neither port nor re-chunk now.**

* ADR 0081's criterion is not met. The cost is not a per-row interpreted kernel; it is serial chunk decompression, and
  the lookup is not on a milestone critical path. A port would also duplicate decoding (allele-free but
  encoding-dependent: int16 z, residual SE and EAF, exception tables, imputed mask) in the place this project's
  silent failures live, for a ~1 s query.
* The re-chunk is below the bar (1.5-2x) and changes the format.
* The 5x bar *is* met by the Rust spike, so this is recorded for the owner to weigh against 0081, not dismissed.

## Consequences

* If a lookup latency target appears on a critical path, the first things to try are a **Python-side** way to decode
  chunks in parallel in one process (threads, with the fork hazard of ADR 0056 resolved, or a process pool of
  readers), not a second reader; this spike shows the headroom is about 5-12x.
* If the owner supersedes ADR 0081 for the Store Reader only, scope is the Dense decode path behind
  `store/arrays.py`, Python kept as the reference, differential-tested by the digests the driver uses.
* Not measured: Ragged and Hybrid, OGS-00016, cold disk (no root to drop the cache), a quiet host. The numbers are
  warm-cache on a node at load 10-19; treat ratios as indicative, absolute times as upper bounds.
* The spike code stays in `benchmarks/rust_spike/`; it is not shipped and `opengwasdb/` is untouched.
