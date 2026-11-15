# Direct decode and chromosome-local clumping (ogld#40)

**Partial result: the 10-second target is not met.** Answers and bounds pass. The final
cold GCST90476460 runs took 40.573 s alone and 60.323 s within the 37-Analysis run, not ≤10 s.
Do not treat this implementation as satisfying #40's performance acceptance.

## Measure first

At `f5e4845`, GCST90476460, 260,821 hits, r² 0.01, 3 Mb, the published EUR store:

| cold local run | seconds | all chunk fetches / distinct | Near Band fetches / distinct | leads |
|---|---:|---:|---:|---:|
| tip, zarr helper pool 1 | 137.220 | 51,641 / 35,474 | 49,561 / 33,394 | 3,869 |
| disposable prototype | 72.719 | 33,914 / 33,914 | 33,394 / 33,394 | 3,869 |

The prototype reads byte ranges through StorePath, verifies the index CRC with numcodecs,
decodes on five workers, visits chromosomes serially and omits unused variant columns.
It was measured before the production implementation. Both fingerprints (dtype and bytes
of all six result columns) were
`085516239b15dca279de9dd6869e7b275fddf62a3e1661d6fa615d315f4f2c36`.
The target is not met by this prototype; its remaining costs require measurement, not a
format change assumed to fix them.

Pages were advised away with `latency.evict`; opening is inside the timing. Shared-file
pages can be warmed by other processes. BLAS 1, zarr helper 1, five decode workers: eight
client threads including the caller and event loop. Full `uptime` beside each timing:

```
Tip before: 22:15:28 up 9 days, 3:38, 1 user, load average: 1.31, 1.33, 1.22
Tip after:  22:17:45 up 9 days, 3:40, 1 user, load average: 2.65, 1.86, 1.43
Probe before: 22:18:26 up 9 days, 3:41, 1 user, load average: 2.38, 1.88, 1.46
Probe after:  22:19:39 up 9 days, 3:42, 1 user, load average: 2.58, 2.06, 1.55
```

## Decoder evidence

Every Test Store array equals zarr, and a re-sharded numeric array with a padded last
chunk equals it in both endian orders. Tests cover an empty inner chunk, corrupt frame,
short decoded frame, corrupt CRC and truncated index. Four unsupported chains fall back
to zarr. Every Test Store array also equals local zarr through the contract HTTP server.
Wrong endianness failed the big-endian comparison; shifting the byte offset by one failed
both endian comparisons with invalid zstd input. Decoding a short frame directly into a
larger destination was observed to pass silently in numcodecs; the regression test failed
before the decoder was changed to check the decoded length.

#32's HTTP flight and held-buffer tests run on the direct path. The planner now reserves
space inside the decoded-byte allowance for trimming or byte-order conversion; the cache
budget is unchanged. Native decoded single-chunk buffers transfer to the cache without copying.
The complete default tier passed 1,240 tests in 26.76 s with eight pytest workers at this
slice. Tip: 1,227 tests in 16.77 s, load 1.47→4.30. Both are below 30 s.

## Sparse gathers and chromosome scheduling

The production decoder plus chromosome-local scheduling still took 75.023 s under profiling,
with the same fingerprint and 33,394 Near Band chunks. This was a diagnostic run, not a
clean timing (two small red/green tests also ran during it). Its profile exposed a further
cost: every warm sparse `take` concatenated whole decoded chunks before selecting a handful
of values. The regression test held 8,391,715 traced bytes to return 128 bytes; it failed
before the gather was changed to select values inside each chunk first. The same test now
passes without a full-chunk join. Interleaved chromosomes, tied scores, duplicate requests,
absent ALIDs and thresholds 0 through 1 keep the same answers; the column-read test failed
before removing the unused columns. The default tier then passed 1,247 tests in 20.24 s.

The clean gathered run took 55.617 s (load 4.09→3.42), with identical bytes and chunks.
A following stage-instrumented run took 54.611 s: 49.966 s in `take`, including 33.958 s
waiting for `_fetched` and 14.544 s in `_keep_run`; `rows_of` took 4.544 s. Decode reached
five workers. Worker elapsed seconds sum to 65.464 (overlapping), so these are not additive
wall times. This motivated transferring each decoded chunk's own buffer directly to the
cache rather than copying it into a run and then copying it again. The transfer test failed
before that change. Removing cancellation draining also failed its test; a cancelled read
must await workers still holding its buffers.

A reproducible wrapper, `scripts/bench/fast_decode.py`, now runs #8's existing cases and
records whole-clump SHA-256, distinct chunks counted at `_keep_run`, and full uptime beside
every timing. The acceptance run of the stores samples first/middle/last chunks of every array in
all five published stores against zarr (`pixi run -e dev acceptance-stores`, ogld#40); it needs
no input variable, since `scripts/build_hgdp1kgp.py`'s `store_path` says where they are. That
sample reads every array of five whole stores, which took the realdata tier past its 5-minute
budget, so it moved to the run that already reads them all.

## Final runs at `2811c06`

Records and diagnostic sources are in [`results/40/`](results/40/), compressed with gzip.
The production source was clean; the wrapper records its installed version and commit.
The long runs inherited CPU affinity 0–7, BLAS 1 and zarr helper 1; the decode pool has five
workers. Local clients had eight threads including the caller and event loop. HTTP uses
#8's separate static/proxy processes (the proxy is asynchronous, the static host has two
file helpers); all descendants inherit the same eight-CPU cap. Every timing carries full
`uptime`. The cold method remains advisory page eviction, not a guarantee against another
reader warming shared files. The original tip run used one zarr helper, whereas #32/#38's
comparison records used six: ratios across these records are not controlled thread-allocation
experiments. Load was about 2–5 here, versus 4–35 for #32 and 39–60 for #38.

### GCST90476460 and the 37

GCST90476460 still gives 3,869 leads and identical whole-result bytes. Fetches fall from
51,641 overall to 33,914; Near Band fetches from 49,561 to 33,394, each distinct chunk once.
It reads 7.413 GB compressed and decodes 17.508 GB of Near Band chunks. The isolated timing
was 40.573 s, load 4.65→4.94; within the 37 it was 60.323 s, load 4.20→4.50. The same whole-result
fingerprint is recorded in both. At r² 0.001 it took 51.411 s and fetched 28,085 Near Band chunks.

Each Analysis cold, r² 0.01 / 0.001, compared with #38's reader records:

| r² | before seconds | head seconds | before / head Near Band chunks | head leads | head over 30 s |
|---|---:|---:|---:|---:|---:|
| 0.01 | 488.595 | 454.787 | 203,420 / 182,519 | 23,445 | 4 |
| 0.001 | 404.409 | 398.388 | 160,430 / 155,645 | 13,866 | 3 |

All 74 per-Analysis lead counts equal the baseline. No Analysis re-fetches a Near Band chunk.
Near Band bytes fall from 45.681 to 41.008 GB and 35.970 to 34.893 GB. Cold `rows_of` totals
108.055/108.113 s, against 86.475/87.337 s with the earlier helper allocation. Removing unused
columns eliminates the earlier 81.535/80.604 s in `variants`, but numeric fetching still costs.
The first four Analyses finished before a diagnostic pause; the remaining 33 resumed with
identical settings, hence two environment headers in `clump-37.jsonl.gz`.

### What still bounds the target

A final instrumented GCST90476460 took 54.584 s: 47.400 s in `_fetched`, 1.349 s in
`_keep_run`, and 3.027 s in `rows_of` (nested timers are not additive). The pool reached five
workers. Its 20,491 decode jobs consumed **86.002 thread-CPU seconds**, including decompression,
allocation and numpy wrapping, not just the raw zstd kernel. Five workers therefore need at
least 17.2 s for that measured work, before lookup or I/O; this implementation cannot reach
10 s merely by removing evictions. The eight threads consumed about 120.5 CPU seconds in all.
The remaining bottleneck is the numeric fetch/decode implementation, not a demonstrated format
lower bound. Worker elapsed time sums overlap and must not be mistaken for CPU time.

One decode job per range did not help (55.391 s), nor a disposable process-wide Python switch
interval of 100 µs (52.548 s, same bytes). Neither is shipped. In particular, no interpreter
global is changed by the library. More profiling of decode/allocation costs is needed before
choosing another implementation or asking the owner to revisit the held-back format change.

### #8 step-1 reads against #32

Seconds, before → head. Matrices are the seeded 500-variant draw, cold. Regions are medians
of the same three centres, cold; proxies are 100 variants on an opened store. All 162 read
summaries match #32. These are single runs; local differences below about 45% can be noise.

| read | local | +10 ms | +50 ms |
|---|---:|---:|---:|
| chr22 matrix | 1.800 → 0.876 | 5.788 → 2.996 | 14.766 → 6.733 |
| genome-wide matrix | 9.029 → 8.372 | 18.243 → 18.582 | 41.776 → 37.793 |
| 3 Mb region | 1.461 → 1.903 | 2.051 → 2.148 | 3.786 → 4.303 |
| 10 Mb region | 10.354 → 6.860 | 13.112 → 8.432 | 17.900 → 14.829 |
| 100 proxies, open | 1.062 → 1.042 | 2.732 → 3.143 | 7.101 → 7.326 |
| four clumps, r² 0.01, sum | 52.219 → 32.222 | 101.900 → 87.158 | 224.212 → 201.251 |
| four clumps, r² 0.001, sum | 46.387 → 30.996 | 90.884 → 80.184 | 204.645 → 186.710 |

The four clumps make 6,279 requests instead of 7,537 at 0.01, and 5,729 instead of 6,887 at
0.001. The small-region and proxy regressions above are retained, not hidden by aggregate wins.

## Final correctness, bounds and gates

- The adapted real-chr22 `answers.py` uses the published EUR store and its actual chromosome
  range. All **47 arrays** have identical dtype, shape and every byte against the untouched tip,
  including the 21,850 × 21,850 region. The adapted source is archived with the comparison log.
- The realdata tier passed **10 tests in 262.35 s**: first/middle/last chunks of every array
  in each of all five published stores equal zarr, and real build/query checks pass. (Those five
  store checks now live in `acceptance-stores`; ogld#40 moved them out of the tier's budget.)
- `acceptance-clump` passed **3 tests in 585.32 s**, asserting exactly the same eight explained
  differences, not merely the same aggregate lead count.
- Unit/contract tests cover every Test Store array locally and over HTTP, both data byte orders,
  big-endian index bytes, physical offsets out of logical order, padded tails, missing shards,
  elided inner chunks and their refusal, corruption and unsupported-chain fallback. Endianness,
  offset, short-frame, sparse-gather, ownership-transfer and cancellation mutations went red;
  the red outputs are archived.
- Cache-held buffers and indexes stay within the configured budget under overlapping threaded
  reads. The HTTP flight tests and held-buffer accounting exercise the direct path, including
  reserved conversion/trim space and transfer of owned buffers. Cancelled decode drains workers.
  Compressed bytes, codec workspace, results and requested chunks pinned by a read are not an
  RSS promise. A single oversized chunk remains the documented exception to the flight allowance.
- `orch/1-ld-stores` was merged before certification: already up to date at `f5e4845`.
  Final default: **1,255 passed in 15.05 s**, load 2.04→3.86, eight pytest workers; tip:
  **1,227 in 16.77 s**, load 1.47→4.30. Final coverage: **1,255 in 13.08 s**, 99% overall,
  slowest 1.99 s. One earlier default run took 34.08 s and was outside the budget; it is not
  used as a passing timing. The final `lint`, strict types, postflight (12 gates) and changed
  scope (10 gates) all pass. Postflight initially exposed uncovered sync-wrapper and unsupported
  grid paths; tests now cover them, without changing a quality setting or baseline.

Reproduce the final read records with the wrapper, BLAS 1 and zarr helper 1:

```bash
pixi install --locked -e dev
pixi reinstall --locked -e dev ogld
# Prefix commands with taskset -c 0-7 on this node; descendants inherit it.
pixi run -e dev python scripts/bench/fast_decode.py step1 --target local \
  --case clump --cold-only --zarr-threads 1 --out clump-37.jsonl
pixi run -e dev python scripts/bench/fast_decode.py step1 --target local http10 http50 \
  --case proxies scattered region --seeds 1 --zarr-threads 1 --out reads.jsonl
```

For the four-clump comparison, pass `--case clump --cold-only --analyses
OGS-00009__ukb-b-4171 OGS-00011__GCST90319613 OGS-00011__GCST90025965
OGS-00011__GCST90029004`. The archived disposable prototype is run on `f5e4845`, not the
finished decoder; diagnostic stage scripts expect it unpacked as `/tmp/40-probe.py`.
