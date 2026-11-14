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
