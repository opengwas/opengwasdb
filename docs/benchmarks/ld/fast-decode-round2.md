# #40 round 2: fill the pool before the greedy pass

## Measure before choosing

GCST90476460, published EUR, r² 0.01, 3 Mb, cold advisory eviction. Seven numeric workers,
one fallback helper, BLAS 1, CPU affinity 0–7. The prototype whole-result SHA-256 remains
`d9c56db72fe6ba9890203382f1a657dc40619e634eaa5ea7458b762146e23b44` (3,869 leads).

| prototype | seconds | prefetch | chunks | note |
|---|---:|---:|---:|---|
| chromosome prefetch, old per-range dispatch | 76.203 | 66.811 | 43,453 | the pool is still starved |
| preallocated output, same dispatch | 79.089 | 69.686 | 43,453 | destination buffers alone do not fix scheduling |
| batch decode after fetching, one I/O helper | 55.672 | 47.790 | 43,453 | I/O serialises |
| shared range-I/O/decode pool | 31.186 | 22.103 | 43,453 | choose (a) |
| pipelined one decode job per range | 31.670 | 22.778 | 43,453 | no further gain |
| bounded per-chromosome sparse graph | 68.419 | — | 43,466 | 1 GiB graph/work cap, one fallback |

The graph prototype streams blocks and declines before its conservative 128 bytes per retained
edge plus 256 MiB block workspace exceeds 1 GiB. It falls back to prefetch, not to an unbounded
allocation. Its maximum planned graph/work allowance was 1,050,960,128 bytes. The cache and
bounded fetch buffers are additional. Prefer prefetch: it is faster and needs no retained graph.

On the orchestrator's same real shard (`near/22/r/c/3`), raw allocating decode measured 1.270 ms
per chunk on one worker and 0.191 ms wall on seven; preallocated output measured 1.202 and
0.178 ms respectively. Every timing's full uptime is kept with the records. These small raw
measurements do not include store I/O or query lookups.

## Implemented bounds and seams

Candidate chunk intervals are merged without enumerating hit pairs. A whole-chromosome warm
is attempted only if its buffers plus conservative index storage fit the existing cache and
at most 65,536 chunk keys are planned. Oversized chromosomes use positional windows, halved
until they fit. Cache zero skips warming. A normal point read streams bounded chunk groups,
so a lead cannot pin an unbounded chromosome outside the cache.

Decoded prefetch buffers use the 256 MiB flight allowance, reserving conversion/trim scratch;
compressed ranges have a separate 256 MiB credit limit held until decode completes. Single
oversized chunks remain the documented exception. Ranges coalesce at most 512 KiB gaps into
16 MiB reads; only requested chunks decode. Actual requests are semaphore-bounded by zarr's
async concurrency. Positions, indptr, request/results, codec workspace and cache bookkeeping
are additional, as in spec §10; this is not an RSS limit.

The wrapper bridges exact LocalStore's public get_sync through its async interface on the
same seven-worker pool as decompression. HTTP and subclasses keep their own async interface.
No global zarr executor/configuration changes. Fallback codecs use the application's zarr
helper allocation (one in these runs). Known zstd content sizes are checked against RFC 8878
before out= decoding; unknown-size frames retain allocating decode and exact length validation.

The prefetch seam failed before implementation. Disabling streamed point reads failed with
8,781,360 traced bytes held to return 128 bytes; the bounded implementation passes. Sliding
windows and cache zero return the same clumps. The initial full slice passes 1,259 tests in
29.42 s, lint, strict types and changed-scope gates. Final measurements and certification follow.

The first production warm run took 48.307 s with identical result bytes. It exposed a long
coalesced range being decoded by one job. Ranges now distribute their requested frames over
the pool, while retaining both credit and cancellation bounds. Lookup also routes positions
and Near Band indptr through the numeric decoder, and defers unused Proxy Index indptr data.
The lazy-index test failed before that change. ALIDs already verified by rows_of name clump
results directly, preserving present-only string width. This next slice passes 1,260 tests
in 14.87 s at load 4.90→6.81 with an eight-CPU cap and one fallback helper.

That slice's instrumented cold query took 22.423 s: rows_of 5.542 s, prefetch 14.203 s,
partners 1.607 s. Near Band decode consumed 44.069 thread-CPU seconds for 43,453 chunks:
**1.014 ms per chunk**, versus the orchestrator's 1.17 ms raw reference. That individual run
no longer shows the extra 1.4 ms per chunk; the repeated final measurements below are slower,
so this is not a universal CPU cost. Prefetch I/O/dispatch and lookup still limit the query.
Adaptive small-range jobs were slower (33.956 s); sparse uncached string selection reduced
lookup to 3.711 s but not the whole query (22.738 s), and would weaken the shared cache/bounds
contract. One-owned-string-chunk dispatch was also slower (35.763 s). None is shipped.

Unknown-content-size zstd frames, missing/corrupt prefetched shards and prefetch-key limits
have explicit tests. Completed worker closures release their captured range before publishing
the Future result: an executor retaining completed work items must not retain already released
byte credits. Completed Futures return None and hand their actual result through a popped
box, so the same guarantee holds for returned I/O buffers. The weak-reference regression was
observed red. HTTP flight tests now measure awaited client requests: server finalisers can
briefly overlap a subsequent request after the preceding body was already delivered. The
client measurement retains the exact concurrency bound rather than adding timing slack.
Full default and coverage slices pass 1,266 tests, with all postflight and changed gates green.


## Final cold runs at `0e23281`

**Neither performance target is met.** BLAS 1, seven shared numeric/I/O workers, one zarr
fallback helper, inherited CPU affinity 0–7, advisory file-page eviction and a fresh reader
per cold query. Every record includes both uptimes. The resumed clean isolated query takes
**38.107 s**, with the same SHA-256 and 3,869 leads. The orchestrator measured **34.0 s,
6.0 GB peak RSS** on the preceding working tree against **141.1 s** at the integration tip.
These repeated times vary; do not present the earlier 22.423 s diagnostic as the final timing.

The 37 cold queries take **272.473 s at 0.01** and **281.736 s at 0.001** (round 1:
454.787/398.388 s; #38: 488.595/404.409 s). All 74 complete result fingerprints equal round 1,
not just the 23,445/13,866 lead totals. Each threshold warms **265,270 distinct Near Band
chunks**, with no re-fetches: warming candidate pairs also decodes chunks the eventual greedy
pass would not visit (GCST90476460: 43,453 versus round 1's 33,394). This is a scheduling versus
extra-work tradeoff, not a reduction to #38's approximately 203,000-chunk decode floor.

### Where the 37's time goes

Seconds at r² 0.01. The five slowest in this run are GCST90476460, GCST90000025, ukb-b-9685,
GCST90475446 and GCST90475442. Rows sum across independent fresh-reader queries; the
chromosome and string subrows are **inside** rows_of, not additional time.

| phase | all 37 | five slowest | other 32 |
|---|---:|---:|---:|
| open | 0.869 | 0.115 | 0.755 |
| rows_of | 94.047 | 25.993 | 68.053 |
| ↳ chromosome loads | 31.113 | 5.814 | 25.299 |
| ↳ ALID string reads | 59.094 | 18.238 | 40.856 |
| prefetch | 161.172 | 77.264 | 83.908 |
| per-lead partners | 9.889 | 4.888 | 5.001 |
| remaining greedy bookkeeping | 4.133 | 1.996 | 2.137 |
| other result/setup work | 2.362 | 1.346 | 1.016 |
| **total** | **272.473** | **111.603** | **160.871** |

Compared with round 1, chromosome loads fall from 45.575 to 31.113 s and rows_of from
108.055 to 94.047 s after routing numeric lookup columns directly and deferring unused Proxy
Index indptr data. Opens are only 0.869 s; per-lead partners plus greedy bookkeeping average
0.598 ms over 23,445 leads. They are not the dominant residual. ALID strings still go through
zarr, including its bounded string copies and counting. Rejected sparse string-selection and
owned-string prototypes above did not improve the full query.

### What bounds the targets now

The instrumented GCST90476460 query within the 37 takes **30.671 s**: rows_of 5.478 s,
prefetch 22.351 s, partners 1.717 s, other greedy work 0.659 s. Complete Near Band worker jobs
consume **63.158 CPU-seconds / 43,453 chunks = 1.453 ms/chunk**, including frame inspection,
allocation, numpy wrapping and job-loop work. This exceeds the owner's 1.17 ms single-shard
reference by 0.283 ms, but different chunks and working sets mean that difference is not a
measurement of Python overhead. Across all 37, worker jobs consume **427.675 CPU-seconds /
265,270 = 1.612 ms/chunk**. The phase driver times whole jobs, without per-frame locks; the
previous frame-instrumented driver itself perturbs scheduling.

A controlled repeat on the owner's exact chr22 shard, 1,024 frames per case, isolates the
wrapper: one-worker CPU is **1.306 ms/chunk allocating**, **1.169 ms with out=**, and **1.162 ms
through the production _frame**, against the 1.17 ms reference. Production wrapping adds no
measurable CPU over out= in this sample. On seven workers its wall cost is **0.207 ms/chunk**.
The remaining live-query cost cannot simply be called another 1.4 ms of decoder overhead.

The seven workers therefore require at least **9.022 s** for this query's measured Near Band
CPU work alone, or **61.096 s** across the 37, even with no waiting. Near Band CPU utilisation
against prefetch wall time is only about **40%** and **38%** respectively. Replacing decoded
outputs with empty diagnostic tokens (no LD answers computed) leaves the identical 6,895
Near Band range/index gets and 10,664,181,675 compressed bytes, taking **8.442 s**. This isolates
range I/O/dispatch, not an irreducible physical-disk floor. CPU and I/O overlap: do not add these
two times. Lookup precedes both: even perfect overlap leaves about **14.5 s** (5.478 + 9.022)
for GCST90476460 and **155.1 s** (94.047 + 61.096) for the 37 before lead processing. Merely
adding an eighth decode worker would not reach either target. Larger asynchronous warm
batches or a different sparse streaming plan still need measurement, not a claim of success.

The uncommitted native unknown-size-frame fix was retained: validate the exact decoded length
then reuse its raw buffer rather than make a second full copy. Its allocation test was seen
red before the fix, and the sliding-window fixture now asserts joined members. All 44,515
numeric frames in a separate production probe declare their sizes, so no real-query speedup
is attributed to that fallback fix.

The cache/flight limits above are per reader/read, not RSS limits. The legacy 37-query runner
retains its preceding reader until its next cold query returns; its process high-water mark
(11.375 GB) therefore includes **two** independently bounded caches and must not be cited as
a single clump's peak. The orchestrator's isolated 6.0 GB measurement does not have that
harness retention. Request/result arrays, positions/indptr and allocator retention are also
outside the cache count. No graph, whole-chromosome pin or unbounded prefetch is retained.

A final disposable native-flight prototype used all but one chunk of the decoded allowance
instead of half, reserving the tail copy without raising the 256 MiB bound. Alternating cold
runs took **35.137 s (prototype), 29.100 s (production), 24.129 s (prototype)** with identical
fingerprints. Gets fall only from 6,895 to 6,825, while bytes slightly increase. This does not
establish a repeatable >15% win; retain the simpler conservative reservation, rather than
shipping the last and fastest observation. Raw uptimes and both prototype runs are retained.

## Final correctness and constrained test timing

All 47 real-chr22 answer arrays, including every dtype, shape and byte, equal the integration
tip. All 74 whole-clump fingerprints equal round 1. The 10 realdata tests pass in **311.99 s**,
including first/middle/last chunks of every array in all five stores against zarr; this exceeds
the documented five-minute tier budget by 11.99 s. (ogld#40 later moved those five store checks
to `acceptance-stores`, so the tier loses the 11.99 s overrun; the timings here are the run's own
record and are not restated.) Acceptance-clump passes **3 in 431.32 s**,
asserting exactly the same eight explained differences.

To avoid multiplying seven-worker pools across pytest processes, final certification uses
**-n 0**, BLAS 1, one fallback helper, affinity 0–7. Head: **1,267 passed in 159.24 s**, load
1.85→2.29; coverage: **1,267 in 143.07 s**, 99%, longest reported call 1.50 s. The integration
tip, tested from a detached worktree without changing cwd, passes **1,227 in 136.74 s**, load
1.52→2.26. The initial archive-only tip run failed three git-provenance checks because an
archive has no git metadata; the real worktree rerun resolves those harness failures.
These serial times do **not** certify the normal parallel default tier's 30 s budget; the
previous round-2 eight-process run was 1,266 tests in 15.68 s. Lint, strict types, all 12
postflight and all 10 changed-scope gates pass. Integration merge remains already up to date
at f5e4845. All final timings have their full uptimes in the compressed records below.

## Final step-1 reads against #32

Seconds, before → head. Same seeded draws, centres and summaries as the round-1 table.
Local runs use seven numeric/I/O workers and one fallback helper. **Final HTTP runs reserve
four helpers for the two host processes**, leaving three numeric workers and one fallback
helper: at most eight configured workers including the hosts, with affinity 0–7 inherited
by every child. This is a benchmark-only private allocation, not a new public knob.
Initial seven-worker-client HTTP records are retained for provenance but are **not** used in
this final table. #32's historical helper allocation differs, so this is not a controlled
worker-count comparison. Small changes, especially on local disk, should not be over-read.

| read | local | +10 ms | +50 ms |
|---|---:|---:|---:|
| chr22 matrix, cold | 1.800 → 0.576 | 5.788 → 2.404 | 14.766 → 6.844 |
| genome matrix, cold | 9.029 → 7.497 | 18.243 → 14.963 | 41.776 → 38.037 |
| 3 Mb region, cold median | 1.461 → 1.778 | 2.051 → 2.528 | 3.786 → 4.436 |
| 10 Mb region, cold median | 10.354 → 7.696 | 13.112 → 8.483 | 17.900 → 14.673 |
| 100 proxies, open | 1.062 → 1.056 | 2.732 → 2.578 | 7.101 → 7.251 |
| four clumps, 0.01, cold sum | 52.219 → 24.505 | 101.900 → 53.715 | 224.212 → 124.225 |
| four clumps, 0.001, cold sum | 46.387 → 27.442 | 90.884 → 55.604 | 204.645 → 121.637 |

All 162 read summaries and all 24 clump lead counts match #32. Each four-clump run now makes
5,174 requests at either threshold, versus #32's 7,537/6,887. The small-region and 50 ms proxy
regressions are retained rather than hidden behind aggregate clump wins.

### Retained evidence

`docs/benchmarks/results/40/round2-*.gz` holds the complete JSONL records, both certification
logs, same-shard microbenchmark, I/O-only diagnostic, phase driver, constrained HTTP driver,
allocation red run, and exploration logs/scripts. For the final table use local records from
`round2-initial-reads.gz` / `round2-initial-clump-http.gz`, and HTTP records from
`round2-http-bounded-reads.gz` / `round2-http-bounded-clump.gz`. `round2-37.gz` is the final
74-query cold run; `round2-isolated.gz` is the clean isolated query, not a prototype.
`round2-post-certification.gz` contains the successful detached-worktree tip timing, controlled
microbenchmark and final bounded-host HTTP run. Source scripts expect the repository root as
cwd; the stores are opened read-only. Long uncompressed records remain under
`/data/opengwasdb/work/ogld/scratch/40/`.
