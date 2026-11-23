# Decode recognised numeric chunks directly on a bounded pool

Formerly ogld ADR 0009.

> **The Proxy Index is gone** ([ADR 0097](./0097-partner-rows-replace-the-proxy-index.md), opengwas/ogld#72).
> Where this record speaks of it in the present tense — as a component a store holds, a stream a
> build writes or a read a query makes — read it in the past tense: the Partner Rows hold every
> entry it held, and `clump` and `proxies` read them. **Round 2's chromosome prefetch went with
> it** (opengwas/ogld#72): `ChunkCache.prefetch`, its plan and its compressed-range credit served `clump`'s
> warm windows, and nothing calls them now. The round's measurements and its direct-decoding,
> shared-pool and single-flight decisions stand, and those are what serve `matrix` and `region`.

Issue [#40](https://github.com/opengwas/ogld/issues/40), part of epic
[#1](https://github.com/opengwas/ogld/issues/1). ld-store §6 defines the bytes; §10 defines reads.

**Who decided.** The owner, 13 November 2026, on
[epic opengwas/ogld#1's diagnosis](https://github.com/opengwas/ogld/issues/1#issuecomment-6866202125):
decode numeric chunks directly and in parallel, clump chromosome by chromosome, and read
only the variant columns clump returns. No format or public signature changes.

## Context

The diagnosis counted 51,641 fetches for GCST90476460, 33,394 distinct Near Band chunks.
About 90% of clumping time was zarr's per-chunk machinery, serialised on its event loop;
a direct zstd decode took 0.25–0.43 ms against about 2.4 ms through zarr. opengwas/ogld#32 already
owns chunk planning, the cache budget, concurrent runs and kept shard indexes.

## Decision

`opengwasdb/ld/store/location.py` calls the numeric backend in `opengwasdb/ld/store/decoding.py`, which
recognises a one-dimensional numeric array only when its declared
codec chain is exactly sharding_indexed, with bytes then zstd internally and bytes then
CRC32C in an end-located index. Both bytes codecs' declared endianness is honoured. A one-byte
element has no byte order and zarr declares none for it, so a `bytes` codec without an `endian`
is read as it lies — that is how the ALID index's `uint8` keys decode directly (ADR 0096). Other
chains, strings included, still use zarr. Shapes and keys come from the opened array's
metadata, never the layout defaults or an array path.

The decoder implements the published sharding_indexed index: a C-order array of uint64
(offset, length) pairs, absolute offsets, with both fields all ones denoting an empty inner
chunk. It verifies CRC32C with numcodecs before interpreting the index. Missing shards and
empty inner chunks produce the array's fill value; the reader's existing refusal is unchanged.
Truncated indexes and corrupt zstd frames raise, not fall back.

Indexes and ranges use the store's async interface, through opengwas/ogld#32's wrapper, on local disk
and HTTP alike. Only physically adjacent requested ranges are merged. The existing batch
planner bounds each read's decoded buffers. Each native decoded chunk owns its buffer and
transfers it to the cache without a second copy. The planner reserves one chunk for trimming
a padded final chunk, or half the allowance for non-native byte-order conversion scratch.
A frame's decoded length is checked before adoption. As refined in round 2 below,
decoding directly into a destination without validating its declared size is rejected,
because numcodecs accepts a short frame without raising. At least one chunk
can be read even when it exceeds the allowance. The process shares one seven-worker executor,
reset after fork. A batch submits
at most seven jobs per range, not a pool per reader or chromosome. Cancellation drains
already-running decode jobs before the read releases its buffers. The executor's standard
Python shutdown waits for its workers. Applications still budget zarr's I/O helper threads
separately; this is not a process-wide thread limit.

The implementation was checked against zarr 3.4.2's `codecs/sharding.py`, especially
`_ShardIndex`, `_get_index_chunk_spec`, `_shard_index_size` and `_shard_index_byte_range`.
It calls none of those private methods. It uses public codec descriptions, metadata key
encoding, StorePath.get and byte requests. Zarr remains the format, metadata parser,
fallback reader and sole writer. numcodecs is already a direct runtime dependency.

## Alternatives and costs

- Keep zarr's per-chunk pipeline: rejected by the owner's diagnosis; it dominates the read.
- Decode on zarr's event loop: rejected; zstd releases the GIL but would still run serially.
- A pool per chromosome or reader: rejected; it silently multiplies threads and working sets.
- A symmetric Near Band or lower Proxy Index floor: held back by the owner; these change the
  format's cost, whereas this change removes implementation overhead without rebuilding.
- Copy decoded numeric buffers into a run and then into the cache: rejected after a stage
  measurement spent 14.54 s in `_keep_run` alone. Owned single-chunk buffers avoid both copies
  without retaining uncounted neighbours; zarr's fallback runs still need to be copied.
- Copy zarr's private sharding machinery: rejected; implementing this deliberately narrow
  published chain makes fallback explicit and avoids a private-API dependency.

## Round 2: bounded chromosome prefetch and shared I/O

The orchestrator's round-2 instruction on opengwas/ogld#40 requires warming a chromosome's candidate
chunks before the greedy pass, with positional windows when they do not fit. A bounded
sparse-graph prototype (1 GiB graph/work allowance plus cache and fetch buffers) took
68.419 s on GCST90476460, including one graph-cap fallback; prefetch with a shared I/O/decode
pool took 31.186 s with identical whole-result bytes. Choose prefetch, not the retained graph.
These are prototypes, not a claim that the 10-second target is met.

The cache plans the union of chunk intervals without enumerating hit pairs. It warms only
when the candidate chunk buffers plus conservative shard-index storage fit its budget, and
when at most 65,536 chunk keys are needed. Otherwise the reader warms a sliding range of W
around a lead, halving that range until it fits. The ordinary read still resolves every
requested pair, including partners outside a smaller warm range. Zero cache skips warming.
Sparse point reads stream bounded chunk groups instead of pinning a whole chromosome.

Prefetch has at most 256 MiB of decoded buffers (with conversion/trim allowance), and a
separate 256 MiB credit limit on compressed ranges held until their decode finishes.
Individual oversized chunks retain the documented one-chunk exception. Prefetch coalesces
gaps up to 512 KiB into ranges up to 16 MiB, decoding only requested chunks. At most zarr's
async.concurrency store requests run together. Candidate planning adds O(hits + 65,536)
metadata, not O(hit pairs). The cache budget is unchanged; positions/indptr, request/result
arrays, codec workspace and cache bookkeeping remain additional, as in §10.

The store wrapper's async interface now bridges the **public LocalStore.get_sync** method
onto the same seven-worker pool as decoding. Exact LocalStore only: subclass overrides and
HTTP stores continue through their own async get. This avoids serialising all local ranges
on the loop's one default helper without making another I/O pool or changing global zarr
settings. Applications budget fallback codec helpers separately; benchmarks use one helper,
BLAS 1 and an eight-CPU cap. Cancellation drains both I/O and decode jobs. A cancelled
shard-index waiter shields and drains the shared fetch, even through repeated cancellation;
it never cancels another reader's fetch. The waiter retains its cancellation if shared I/O
fails, while an uncancelled reader still receives that failure. Consequently cancellation
may wait for slow index I/O, rather than abandon a worker outside the read's lifetime.
Worker closures
clear captured buffers before Future completion and hand returned ranges through a popped
result box (the Future itself returns None). Retained executor work items or Futures therefore
cannot pin ranges after their compressed-byte credits are released.

Known-size zstd frames decode into preallocated numpy buffers. The reader checks the frame
content size from [RFC 8878 §3.1.1](https://www.rfc-editor.org/rfc/rfc8878.html#section-3.1.1)
first, including the two-byte +256 rule and declared dictionary/window header widths.
Unknown-size or malformed headers retain allocating decode plus an exact output-length
check. For native byte order the resulting numpy view retains that raw buffer without a
second copy; non-native data still converts. A traced-allocation regression fails with
the unnecessary copy. This preserves the short-frame refusal: merely passing out= is not sufficient.
No new dependency or private zarr codec/index API is introduced.

A chromosome's positions and Near Band indptr now also use the numeric decoder. The Proxy
Index's handles and fill values are still checked on chromosome opening, but its indptr data
is loaded only when used. Below t_far this removes an entire unused column load. Clump names
its present results from the complete ALIDs rows_of already verified, without reading them
again. A regression test observes no index-indptr data read until an index query asks for it.

The cache budget is not an RSS limit. A read also holds requested chunks, its result, one
batch of decoded buffers (and cache copies for fallback runs), compressed ranges, codec
workspace, and small index/job bookkeeping. Concurrent reads each have their own batch. Compressed bytes are not counted
as decoded bytes; a codec's encoded size is not in general bounded by its decoded size.

## Review fixes and acceptance ruling, 14 November 2026

The owner accepts opengwas/ogld#40 with both performance targets **recorded as unmet**: cold
GCST90476460 improved from 141 to 38 s, independently 116.6 to 25.2 s in review, and the
37 Analyses from 489 to 272 s. The measured-work bounds, approximately 14.5 s and 155.1 s
against targets of ≤10 s and ≤90 s, constrain this implementation and workload, not the
format. Lookup and speculative warming can change. Further speed work belongs to opengwas/ogld#42's
ALID search index and opengwas/ogld#41's symmetric Near Band prototype, under the owner's opengwas/ogld#40 review-fix
ruling of 14 November 2026; acceptance does not redefine those targets as met.

The accepted review extraction puts codec recognition, shard-index size and parsing, zstd
frame validation, owned chunk construction and executor lifecycle in `decoding.py`.
`location.py` remains the read interface and owns cache accounting, transport wrapping,
shared index I/O, prefetch planning and encoded credits. Its prefetch requests validated
physical ranges and decode/ownership operations from the backend; it no longer knows the
index's byte layout or frame headers. The backend imports no location policy. Reject
compatibility aliases for the old private decoder/pool names: they would obscure the one
owner, and no public signature changes. Existing equality, corruption, ownership and
cancellation tests exercise the extracted backend and location's unchanged interface.

The compressed-credit regression also drives real sparse ranges from three actual Zarr
shards through the wrapped store and zstd. Weak references count the compressed numpy
buffers held during decoding and verify their release before returning credits. Removing
admission control fails that test, unlike merely asserting synthetic credit counters.
