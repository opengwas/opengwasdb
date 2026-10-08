# #244 step 1: zarr 3 read levers. Results

Worktree `/home/gh13047/repo/opengwasdb-wt/244-zarr-python-3`, branch
`feature/244-zarr-python-3`. **Local commits only: nothing pushed, nothing
posted to GitHub.** Every number below comes from a file named where it is used.

**Those files are committed** under
`docs/benchmark-output/opengwasdb_zarr3_read_levers/`, at the relative paths they
had in `/tmp/epic240/244/levers/` when this report was written. `PROVENANCE.md`
there gives the commit and time each one measured. The tables are generated, not
typed by hand. The scripts named below were ported into `benchmarks/` in step 6,
with paths taken as arguments; `benchmarks/README.md` documents them:

| named in this report | committed as |
|---|---|
| `scripts/shapes_cfg.py`, `run_attribution.sh` | `benchmarks/zarr3_attribution.py` |
| `scripts/summarise_step1.py` | `benchmarks/zarr3_lever_tables.py` |
| `scripts/fork_probe.py` | `benchmarks/zarr3_fork_probe.py` |
| `scripts/fork_paths.py`, #243's `build_stores.py` | `benchmarks/zarr3_fork_paths.py`, `benchmarks/zarr3_fixture_trees.py` |
| `scripts/encode_scope.py` | `benchmarks/zarr3_encode_scope.py` |
| `scripts/se_plan_check.py` | `benchmarks/zarr3_se_plan.py` |
| `scripts/compare_split.py`, `chunk_diff_decode.py` | `benchmarks/zarr3_compare_trees.py` |
| `/tmp/epic240/244/spot_queries.py`, `scripts/compare_spot.py` | `benchmarks/zarr3_spot_queries.py` |

These are not committed:

- `shapes_one_config.py`, which `shapes_cfg.py` superseded;
- `compare_trees.py` and its raw count;
- `pytest-commit2.log`;
- the `trees-*` directories. `zarr3_fixture_trees.py` regenerates those, and it
  also builds Hybrid Reference Completion.

## Headline

- **All three levers are in, and answers are unchanged.** Results are identical
  to zarr 2.18 on every shape and every array. That holds for the seven harness
  shapes on OGS-00009, and for spot queries on OGS-00009 (Dense), OGS-00001
  (Ragged) and OGS-00004 (Hybrid).
- **On OGS-00009, zarr 3 is now faster than 2.18 on bulk**: 0.77× and 0.85× in
  two back-to-back harness pairs, and 0.87× in the attribution. It is within
  1.0–1.6× on regional and random lookups, and still slower on the
  latency-bound shapes: phewas 2.4–3.1×, regional one Analysis 1.8–2.3×,
  tophits 5.1–6.8×. A floor of about 1 ms remains, as the brief predicted.
  *Corrected in step 5:* the floor is per read call, not per chunk as the
  brief had it (ADR 0056).
  (Stage A at 708d179: bulk 2.85×, phewas 8.3×, tophits 19×.)
- **New finding: the brief's lever 2 as specified would hang builds.**
  `FusedCodecPipeline` with its default pool deadlocks forked workers that read
  more than one chunk, because zarr 3.4's at-fork reset misses the pipeline's
  module-level thread pool. I reproduced it on a real build function with real
  data. The shipped setting, `codec_pipeline.max_workers = 1`, never creates the
  pool. It is also *faster*: with Blosc threads on, decodes serialise on
  numcodecs' lock, so the pool only adds contention. Upstream zarr should hear
  about this. I have not reported it, per the hard limits.
- **numcodecs floor raised from 0.14 to 0.17.** Threaded Blosc decompress is
  only locked from 0.17. The lockfile was already on 0.17.0, so the only lock
  change is this package's `requires_dist` line.
- **The levers only pay off together, and the brief's lever-1 figure did not
  reproduce.** Under zarr's default pipeline, Blosc threads alone made bulk
  *slower* (76 → 88 s, medians of 3; ranges 72–80 vs 62–95 s) and slowed
  random lookups by 21–41%, because concurrent decodes from zarr's pool queue
  on numcodecs' lock. Adding fused with one worker took bulk to 24 s, under
  2.18's 26 s. Opening arrays once moved only tophits, from 20.6 to 6.8 ms.
  I had quoted the brief's single-run "52 → 37 s" in commit 1, so `d1c6c41`
  corrects the seam comment and the CHANGELOG. Commit 1's *message* was reworded
  when the branch was rebuilt before its first push (step 5; pushed as `bd0b52d`).

## Commits

| commit (measured at) | pushed as | summary |
|---|---|---|
| `1c27e64` | `bd0b52d` | Decode with Blosc's threads again under zarr 3: `numcodecs.blosc.use_threads = True` set once at the seam (`opengwasdb/store/arrays.py`), global scope; `numcodecs>=0.17`. |
| `8cc6ba1` | `af5e20a` | Read and write through zarr's `FusedCodecPipeline` with `codec_pipeline.max_workers = 1`, at the seam; this setting is what keeps fork pools from hanging. |
| `79e4858` | `3339776` | Open each top-hit array once per query facade (`DenseTopHitReader` caches its arrays; new `TopHitTiers` per facade); `DenseEafPlane` stops reopening `z` for its width. |
| `d1c6c41` | `0c54ab3` | Correct the bulk timings quoted for the read levers: seam comment and CHANGELOG now cite the attribution medians (comments and CHANGELOG only; no code change). |

Base: `708d179`. The four commits change 9 files (+536 / −34). Each commit
message carries its own evidence, except where `d1c6c41` corrects
`1c27e64`'s timing. Stage A, attribution, spot checks and the full suite ran
at `79e4858`. `d1c6c41` changes no code.

**SHAs after the pre-push rebuild (step 5).** Before the first push, the branch was rebuilt
non-interactively: cherry-picked onto `708d179`, with messages amended. The rebuild:

- reworded `1c27e64`'s message so it no longer quotes the 52.5 → 37.5 s figure (it gives
  the attribution result instead);
- corrected `8cc6ba1`'s fork-hang sentence (the hang depends on read size);
- mapped the sibling SHAs cited in `79e4858`'s and `d1c6c41`'s messages.

The final tree is byte-identical to `d1c6c41`'s (`git diff --quiet d1c6c41 0c54ab3`). This
file and the measurement JSONs keep the old SHAs, because those are what the runs recorded.
The pushed commits are in the "pushed as" column. Step 5 then added `9b0be2c` (ADR 0056
and wording fixes in comments, a docstring and the CHANGELOG, with no code change) and
pushed the branch at that head.

## 1a. Blosc threads: thread safety, fork safety, scope

**Source read.** numcodecs 0.17.0 `src/numcodecs/blosc.pyx` (fetched from the
`v0.17.0` tag; the env ships only the `.so`, and its runtime attributes match),
its c-blosc submodule at `04c06fb` (1.21.7.dev), and zarr 3.4.0
`zarr/codecs/blosc.py`.

- **What `use_threads=True` does off the main thread.** Every compress or
  decompress in the importing process uses Blosc's single *global* context.
  numcodecs takes a module `threading.Lock` (`_MUTEX`) around it and releases
  the GIL inside. c-blosc also takes its own `global_comp_mutex` inside
  `blosc_compress` / `blosc_decompress`. Concurrent calls from zarr's pool
  threads, or from the fused pipeline's worker, therefore **queue**: no data
  race, and no parallelism across chunks, but 8 threads inside each chunk
  (`set_nthreads(min(8, ncores))` at import).
- **Why 0.17 matters.** In numcodecs before 0.17 the threaded *decompress*
  path takes no numcodecs lock. I read 0.14.0, 0.15.0, 0.16.0 and 0.16.5, the
  last 0.16 release; 0.16.5 also lacks the at-fork mutex reset. When
  `BLOSC_NTHREADS` is set and differs from the current count, c-blosc's
  `blosc_decompress` destroys and re-creates the global context *before*
  taking its own mutex. So two first concurrent decompresses could race. 0.17
  locks both directions, hence the new floor.
- **zarr's reason for turning it off.** The docstring says "to avoid threading
  issues in asyncio contexts". The line dates from the v3 rewrite (2024) and
  links to zarr 2's "configuring Blosc" guidance, which is about oversubscription
  and contention in multi-threaded programs, not memory safety. Measured
  contention is real: see 1b, where it decides the worker count.
- **Fork safety, from the source.** `_get_use_threads()` returns False whenever
  `multiprocessing.current_process().pid != _importer_pid`. A forked child
  therefore always uses the context functions, single-threaded, whatever
  `use_threads` says. numcodecs re-creates `_MUTEX` in the child
  (`os.register_at_fork`), and c-blosc's own `pthread_atfork` child handler
  discards the inherited global context (`blosc_atfork_child`).
- **Fork safety, measured.**
  - `scripts/fork_probe.py` → `fork_probe.out`. The parent threaded-encodes and
    decodes 16 multi-block chunks (2 MB each), then 8 forked workers read
    multi-chunk selections and Blosc-encode. With Blosc threads alone (`bt`):
    ok, `child_use_threads_effective: false`.
  - `scripts/fork_paths.py`. The parent warms up the same way (asserting
    `_get_use_threads()` is True in the parent), then runs every fork-pool path
    with `n_workers=2`: Dense VCF build, Hybrid build (which ran the top-hit
    gather with 2 workers), Dense Reference Completion, Ragged Reference
    Completion, `_gather_in_row_chunks` directly, and the gather on real
    OGS-00009 bands. All complete:
    - Blosc threads only, at the commit-1 state: `fork_paths-bt.log`, the five
      fixture paths. The real-data gather step was added afterwards, so it ran
      only at the shipped config.
    - Shipped config: `fork_paths-fused-mw1.log`, all six steps; real-data
      gather 1.7 s and equal to serial.
- **Scope chosen: global, set at the seam on import.** The flag is
  process-global in numcodecs, with no per-call or per-thread form. A
  "query-only" switch could only mean "on from the first query in this process
  onwards": a mode that depends on history, that would leave `validate` and the
  builders' own reads single-threaded, and that a build after a query in the
  same process would inherit anyway. The seam imports `zarr.codecs.blosc` by
  name first, so the ordering holds. It raises `ImportError` if numcodecs ever
  drops `use_threads`, since an assignment nothing reads would fail silently.
- **The cost of global (stated in the commit and CHANGELOG).** A chunk of two
  or more Blosc blocks compressed in the parent (≥256 KiB uncompressed at zstd
  clevel 3, so every `[1000,1000]` Dense chunk) has its blocks written in
  completion order, so **its bytes are not reproducible run to run**. That is
  exactly as under zarr 2.18; #243 already compared under `BLOSC_NTHREADS=1`
  for this reason. Measured: 20 encodes of one real OGS-00009 chunk gave 16
  distinct byte strings, one size, all decoding identically. Single-threaded,
  they were identical.
- **Test.** `tests/test_zarr_runtime_config.py::test_blosc_threads_stay_on_after_zarr_reads_and_writes`
  runs in a fresh interpreter. zarr writes and reads a Blosc array, and
  `zarr.codecs.blosc` is re-imported. Then `use_threads` must be True, and so
  must `_get_use_threads()` evaluated on a worker thread, which is where zarr
  decodes. Observed failing against `708d179`, `{use_threads: False,
  on_worker: False}` (`observed-failing-before.log`).

## 1b. FusedCodecPipeline: worker count, fork hazard, write path

**Measurement that chose one worker.** `decide_mw.out`: two fresh processes
each, Blosc threads on, OGS-00009, ms:

| shape | fused, default pool (224 workers) | fused, `max_workers=1` |
|---|---:|---:|
| bulk | 45,496 / 41,116 | **26,508 / 25,665** |
| random 10×100 | 181 / 175 | **134 / 117** |
| random 100×10 | 957 / 1,082 | **781 / 881** |
| regional | **1,645 / 1,625** | 1,759 / 1,756 |
| phewas, tophits | bimodal, 24–56 | bimodal, 28–58 |

**The fork hazard (not in the brief).** With `max_workers > 1` the pipeline
keeps `zarr.core.codec_pipeline._pool`, a module-level `ThreadPoolExecutor`.
zarr 3.4's `reset_resources_after_fork` (`zarr/core/sync.py`) resets the loop,
the I/O thread and `_executor`, but not `_pool`. A forked worker whose read
spans more than one chunk submits to a pool whose threads exist only in the
parent, and waits forever. Reproduced three ways:

- `fork_probe.out`: `fused` and `fused+bt` gave `HANG (no result within 60s)`.
  `fused+bt+reset` (resetting `_pool` at fork) was ok, and so was
  `fused_mw1+bt`.
- The suite test `test_fork_pools_finish_after_the_parent_read_through_the_pipeline`
  runs through the package's `ordered_map` and `run_block_tasks`. With
  `max_workers` unset it failed: "probe did not finish within 120s: a fork-pool
  worker hung" (`observed-failing-fork-guard.log`). The probe runs in its own
  session and the whole group is killed on timeout, so no orphans are left (0
  verified).
- **A real build function on real data**: `_collect_top_hit_eaf` and
  `_collect_top_hit_se` over 200 OGS-00009 cells in four row chunks,
  `n_workers=2`. Each worker decodes a `[1000, 2024]` band, three chunks. With
  the pooled pipeline it was killed at 240 s (`fork_paths-fused-pool-control.log`;
  I removed its orphaned workers by PID afterwards). With one worker it takes
  1.7 s and equals serial.
- **The fixture build paths cannot detect it.** Their arrays are one chunk
  each, and the pool is only used for multi-chunk batches. All five completed
  under the pooled pipeline too. I record this so nobody reads them as
  evidence of safety.

`max_workers=1` is the fix by construction: every `_get_pool` call in zarr
3.4.0 sits behind `max_workers > 1`, and the only other zarr thread pool
(`sync._executor`) *is* reset at fork. I rejected the alternative, an at-fork
hook that resets zarr's private `_pool`. It works (`fused+bt+reset` in the
probe), but it keeps the pool, which measured slower (the table above), and it
depends on a private name. I did not time the reset variant separately: in the
parent it is the pooled configuration.

**Write path: re-proved, not scoped away.** Scoping the pipeline to reads
would mean choosing it per array at open time, through a config context around
every open the query facade makes, including lazy ones. And builders read in
forked workers too, so it would not have removed the fork hazard. Instead:

- byte identity and SE plan, below;
- the full suite on the fused pipeline: 1734 passed, 1 skipped, the only 3
  failures being the then-uncommitted 1c test (`pytest-commit2.log`);
- band-write cost (`encode_scope.out`): a 60-chunk band of real z took 217–227 ms
  on zarr 3's default, 248 ms with Blosc threads, and 300–329 ms with fused
  + Blosc threads. Fused with one worker but *without* Blosc threads took
  1,479 ms, so the two levers belong together.

**Tests.** `test_store_arrays_read_through_the_fused_pipeline` failed against
`708d179` (`BatchedCodecPipeline`). The fork test is the one above. It asserts
first that the parent read went through the fused pipeline across 16 chunks,
so it cannot pass vacuously.

## Byte identity and the SE plan

`scripts/compare_split.py` separates chunk files from metadata. The raw
`compare_trees.py` count (546 "differ" against 2.18) mixes in the known
metadata serialisation difference.

| trees (14 `data.zarr`, #243's `build_stores.py`) | vs `708d179` trees | vs zarr 2.18 trees |
|---|---|---|
| commit 1 (Blosc threads), `BLOSC_NTHREADS=1` | 332/332 chunk files and 950/950 metadata files byte-identical | 332/332 chunk files identical; 398 `.zarray` differ only by the explicit default `"dimension_separator": "."` (classified, no other key) |
| commit 2 (+ fused), `BLOSC_NTHREADS=1` | 332/332 chunk + 950/950 metadata identical | 332/332 chunk identical; same 398 metadata |
| commit 2, default Blosc threads | 20 chunk files differ | same 20 |

The 20 (`chunk-diff-decode-fused-default.json`) are all Ragged
association-sequence chunks of ≥400,000 bytes uncompressed. All 20 have
identical sizes and decode to identical bytes: block order, as predicted.
Files: `compare-split-bt-n1.txt`, `compare-split-fused.txt`, trees in
`trees-bt-n1/`, `trees-fused-n1/`, `trees-fused-default/`.

**SE encoding plan, on real data.** The fixtures cannot answer this: every
fixture plan is `se float16`, because there are too few cells for the residual
path. Instead:

- **Sizes.** All 180 `[1000,1000]` tiles of z (int16), se (int8) and eaf
  (int8) over a 20,000-row OGS-00009 slice compress to identical sizes,
  threaded and not, under every pipeline (`encode_scope.out`). That is the
  input `_packed()` sizes candidates with.
- **The production chooser.** `optimise_dense_se` ran on a staged
  50,000-row × 2,024 OGS-00009 slice (56,580,175 finite SE cells; float32 se
  and eaf, as a Dense build stages them). It chose `int8_residual @0.5`, with
  stored-code hash `8ce5426a67079d3e` and 16,334,536 se bytes, in every case:
  threads off and on × serial and 4 workers (`se_plan_check.out`), and with the
  fused pipeline, serial and 4 workers (`se_plan_check_fused.out`). That is the
  plan the OGS-00008 pilot chose under both 2.18 and `708d179`. I did **not**
  re-run that 10-Analysis pilot rebuild.
- **Build-side timing.** The serial SE optimisation of that slice took 48.4 s
  at `708d179`, 24.9 s with Blosc threads, and 39.0 s with fused + threads.
  With 4 workers, 11.4–11.8 s throughout.

## 1c. Open arrays once

- **Re-opens found.** All three facades' `top_hits` re-resolved the tier group
  (`path in root` + `root[path]`). The Ragged and Hybrid facades reopened the
  store root per call. `DenseTopHitReader` did `group[name]` per field plus
  `name in group` for each optional field. One more was in a plane:
  `DenseEafPlane.n_analyses`, on a release with no `eaf` plane, took the width
  from `z` (`"z" in group` + `group["z"]`) on every `range_phewas`.
- **None anywhere else.** I counted rather than read: every facade shape
  (`analysis`, `phewas`, `range_phewas`, `lookup`, tier `top_hits`,
  per-Analysis `top_hits`) on Dense, Ragged and Hybrid fixtures. After the fix,
  every repeated call reads 0 metadata keys.
- **Test.** `tests/test_query_metadata_reads.py` counts metadata-key reads
  through zarr's `LocalStore` (`get`, `exists`). Before the fix, a repeated
  round read 69 keys on Dense, 134 on Hybrid and 65 on Ragged; with only the
  plane fix reverted, Dense read 6 (`observed-failing-1c.log`). Fixtures are
  asserted meaningful first: every shape returns rows, the tier has hits, and
  Hybrid has overflow hits.
- **Answers unchanged.** The top-hit and result-pinning tests pass (86 + 94).
  The spot identity check is below. The full suite at `79e4858` gives **1737
  passed, 1 skipped, 0 failed** (21 min 47 s; `pytest-final.log`). CLAUDE.md's
  "~2 min" is not this node's figure: Stage A's run took 18 min.

## 1d. Stage A re-run: the #242 harness, OGS-00009, `--reps 3 --skip-rss`

`scripts/run_stage_a_step1.sh` follows `run_stage_a.sh`, with
`PYTHONNOUSERSITE=1` and `uptime` around each run (`stage-a-step1.out`,
`stage-a-step1-pair2.out`). I ran two back-to-back pairs because the head run
of pair 1 started in a busy window. Its harness-recorded 1-minute load was
14.67, although `uptime` showed 4.54 just before it started. In pair 2 I waited
for the load to drop below 3 first. The base run's own bulk reads lift the load
to ~3.9, so a head run that follows one cannot start much below that. Treat
small-shape ratios as ±30%; they moved that much between the pairs.

### Pair 1 (`stage-a-step1-{base,head}.json`)

| | zarr 2.18 | head |
|---|---|---|
| commit | 745796c | 79e4858 |
| python | 3.11.15 | 3.12.14 |
| numpy | 2.4.6 | 2.4.6 |
| zarr | 2.18.7 | 3.4.0 |
| numcodecs | 0.12.1 | 0.17.0 |
| measured_at | 2026-10-04T10:27:52.045626+00:00 | 2026-10-04T10:29:43.125203+00:00 |
| 1-min load before -> after | 2.32 -> 4.54 | 14.67 -> 5.29 |

| shape | 2.18 median | head median | ratio | 2.18 p95 | head p95 | p95 ratio | rows | 708d179 median (Stage A) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bulk | 27,495 | 21,123 | **0.77x** | 30,547 | 21,376 | 0.70x | 7,578,651 | 70,657 |
| phewas | 6.88 | 18.6 | **2.71x** | 7.22 | 20.0 | 2.77x | 1,088 | 67.4 |
| regional | 1,481 | 1,638 | **1.11x** | 2,034 | 1,647 | 0.81x | 4,241,966 | 1,668 |
| regional_one_analysis | 23.5 | 41.2 | **1.75x** | 58.9 | 71.5 | 1.21x | 2,954 | 76.7 |
| tophits | 1.14 | 5.83 | **5.12x** | 1.22 | 6.16 | 5.06x | 2,564 | 22.0 |
| random 10 variants x 100 Analyses | 71.8 | 75.7 | **1.05x** | 75.5 | 81.0 | 1.07x | 654 | 145 |
| random 100 variants x 10 Analyses | 498 | 559 | **1.12x** | 507 | 694 | 1.37x | 628 | 729 |

Result digests, 2.18 run against head run, per shape (every returned array):
bulk: identical; phewas: identical; regional: identical; regional_one_analysis: identical; tophits: identical; random 10 variants x 100 Analyses: identical; random 100 variants x 10 Analyses: identical

### Pair 2 (`stage-a-step1-pair2-{base,head}.json`)

| | zarr 2.18 | head |
|---|---|---|
| commit | 745796c | 79e4858 |
| python | 3.11.15 | 3.12.14 |
| numpy | 2.4.6 | 2.4.6 |
| zarr | 2.18.7 | 3.4.0 |
| numcodecs | 0.12.1 | 0.17.0 |
| measured_at | 2026-10-04T10:32:41.833619+00:00 | 2026-10-04T10:34:53.750122+00:00 |
| 1-min load before -> after | 2.83 -> 3.87 | 3.72 -> 2.81 |

| shape | 2.18 median | head median | ratio | 2.18 p95 | head p95 | p95 ratio | rows | 708d179 median (Stage A) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bulk | 26,865 | 22,941 | **0.85x** | 28,824 | 24,841 | 0.86x | 7,578,651 | 70,657 |
| phewas | 8.52 | 26.1 | **3.06x** | 10.7 | 27.8 | 2.59x | 1,088 | 67.4 |
| regional | 1,427 | 1,720 | **1.20x** | 1,436 | 1,722 | 1.20x | 4,241,966 | 1,668 |
| regional_one_analysis | 23.7 | 50.5 | **2.13x** | 59.4 | 87.1 | 1.47x | 2,954 | 76.7 |
| tophits | 1.16 | 7.79 | **6.75x** | 1.21 | 8.37 | 6.90x | 2,564 | 22.0 |
| random 10 variants x 100 Analyses | 65.0 | 101 | **1.56x** | 69.2 | 102 | 1.48x | 654 | 145 |
| random 100 variants x 10 Analyses | 495 | 576 | **1.16x** | 507 | 598 | 1.18x | 628 | 729 |

Result digests, 2.18 run against head run, per shape (every returned array):
bulk: identical; phewas: identical; regional: identical; regional_one_analysis: identical; tophits: identical; random 10 variants x 100 Analyses: identical; random 100 variants x 10 Analyses: identical

## Attribution (`scripts/shapes_cfg.py` with bulk, fresh process per run)

`scripts/run_attribution.sh` runs three rounds with the five configs
interleaved inside each round, so a noisy window lands on all of them. The
cells show the median of the per-process medians, with the min–max across
processes below it. Each process times every shape 7 times, and bulk twice
after one warm-up. (i)–(iii) run 708d179's code (a `git archive` export) with
the lever set by the script. (iv) is `79e4858` as committed. Each config's
*effective* Blosc and pipeline settings are read back from the process, not
assumed. Raw output: `attribution/attribution.jsonl`.

| config | processes | effective settings (from the run) | 1-min load range |
|---|---:|---|---|
| zarr 2.18 (745796c) | 3 | `{"use_threads": null, "use_threads_after": null}` | 2.4-6.5 |
| (i) zarr 3, no fix (708d179) | 3 | `{"use_threads": false, "pipeline": "BatchedCodecPipeline", "codec_pipeline.max_workers": null, "use_threads_after": false}` | 3.2-6.5 |
| (ii) + Blosc threads | 3 | `{"use_threads": true, "pipeline": "BatchedCodecPipeline", "codec_pipeline.max_workers": null, "use_threads_after": true}` | 3.5-6.0 |
| (iii) + fused, 1 worker | 3 | `{"use_threads": true, "pipeline": "FusedCodecPipeline", "codec_pipeline.max_workers": 1, "use_threads_after": true}` | 2.9-4.6 |
| (iv) + arrays opened once (79e4858) | 3 | `{"use_threads": true, "pipeline": "FusedCodecPipeline", "codec_pipeline.max_workers": 1, "use_threads_after": true}` | 2.7-3.5 |

| shape | zarr 2.18 (745796c) | (i) zarr 3, no fix (708d179) | (ii) + Blosc threads | (iii) + fused, 1 worker | (iv) + arrays opened once (79e4858) |
|---|---:|---:|---:|---:|---:|
| bulk | 26,395<br><sub>25,551-29,996</sub> | 75,805 (2.87x)<br><sub>72,136-79,866</sub> | 88,270 (3.34x)<br><sub>61,569-95,162</sub> | 23,905 (0.91x)<br><sub>21,636-24,858</sub> | 22,894 (0.87x)<br><sub>21,404-25,000</sub> |
| phewas | 9.92<br><sub>7.92-11.8</sub> | 49.9 (5.03x)<br><sub>48.5-59.4</sub> | 34.1 (3.44x)<br><sub>32.7-41.3</sub> | 22.7 (2.29x)<br><sub>22.4-50.1</sub> | 23.6 (2.38x)<br><sub>22.6-24.5</sub> |
| regional | 1,654<br><sub>1,642-1,663</sub> | 1,712 (1.03x)<br><sub>1,671-1,733</sub> | 1,789 (1.08x)<br><sub>1,764-2,349</sub> | 1,748 (1.06x)<br><sub>1,724-1,784</sub> | 1,711 (1.03x)<br><sub>1,702-2,350</sub> |
| regional_one_analysis | 25.8<br><sub>25.2-31.1</sub> | 79.2 (3.07x)<br><sub>78.5-101</sub> | 74.3 (2.88x)<br><sub>72.0-86.1</sub> | 61.6 (2.39x)<br><sub>55.6-84.0</sub> | 58.8 (2.28x)<br><sub>56.0-59.9</sub> |
| tophits | 1.12<br><sub>1.12-1.15</sub> | 18.8 (16.76x)<br><sub>17.5-42.7</sub> | 20.7 (18.45x)<br><sub>19.8-38.9</sub> | 20.6 (18.39x)<br><sub>18.3-57.4</sub> | 6.84 (6.11x)<br><sub>6.65-7.34</sub> |
| random 10 variants x 100 Analyses | 73.5<br><sub>51.8-101</sub> | 154 (2.10x)<br><sub>143-179</sub> | 186 (2.53x)<br><sub>170-193</sub> | 103 (1.40x)<br><sub>94.5-141</sub> | 111 (1.51x)<br><sub>100-115</sub> |
| random 100 variants x 10 Analyses | 707<br><sub>526-730</sub> | 744 (1.05x)<br><sub>742-1,442</sub> | 1,051 (1.49x)<br><sub>968-1,133</sub> | 575 (0.81x)<br><sub>569-824</sub> | 689 (0.98x)<br><sub>572-751</sub> |

Reading it, step by step:

- **(i) → (ii), Blosc threads alone.** The latency-bound shapes gain (phewas
  50 → 34 ms, regional one-Analysis 79 → 74 ms). Bulk and both random lookups
  get *worse* (76 → 88 s; 154 → 186 ms; 744 → 1,051 ms). Under the batched
  pipeline zarr decodes many chunks concurrently on its pool, and each
  threaded decode queues on numcodecs' lock. The brief's single-run
  52.5 → 37.5 s does not reproduce; even (ii)'s fastest process took 61.6 s.
- **(ii) → (iii), fused with one worker.** This is the large step: bulk
  88 → 24 s (now under 2.18), random 186 → 103 and 1,051 → 575 ms, phewas
  34 → 23 ms. One worker hands each chunk to Blosc's threads in turn.
- **(iii) → (iv), arrays opened once.** Only tophits moves, 20.6 → 6.8 ms,
  which is what 1c targeted. The other shapes are within noise. Random
  100×10 at 689 ms is (iv)'s median, but one of its three processes ran at
  572 ms and (iii)'s spread reaches 824 ms. That is noise, not a regression:
  1c touches only the top-hit path and one width lookup.
- **What remains** against 2.18: tophits ~6×, phewas ~2.4×, regional
  one-Analysis ~2.3×, random 10×100 ~1.5×. These are the latency-bound shapes
  that read few chunks, so zarr's per-read cost dominates them. Bulk is 0.87×,
  and regional and random 100×10 are at parity.

## Spot identity (`benchmarks/zarr3_spot_queries.py`, all three levers in place)

`spot-step1-OGS-*.json`, compared with `scripts/compare_spot.py` against
Stage A's `spot-base-*.json` (zarr 2.18): `spot-step1-compare.txt`.

| store | layout | identical to 2.18 | rows (analysis / phewas / regional / lookup / top_hits) |
|---|---|---|---|
| OGS-00009 | Dense | **yes**, every array of all 5 shapes | 9,847,701 / 2,024 / 7,271 / 4 / 149,098 |
| OGS-00001 | Ragged | **yes** | 7,449 / 1 / 139 / 1 / 3,358 |
| OGS-00004 | Hybrid | **yes** | 10,767,485 / 1 / 215 / 1 / 3,552 |

The harness's own result digests also match between the 2.18 and head runs,
for all seven shapes in both pairs.

## Standards

- `CHANGELOG.md` `Unreleased` has three entries, one per lever. The
  compatibility table is unchanged (still format 0.1.0).
- ruff 60 and mypy 39. Both are **identical to 708d179 line for line** (diffed,
  not counted). `scripts/check_baselines.py` reports both at baseline.
- `python3 quality/bin/gate.py --changed` passes at each commit. One escape it
  flagged (a `# noqa` inside a test's probe string) was removed rather than
  accepted.
- Every new test was observed failing against the unfixed code first (logs
  named above).
- No spec, ADR or doc mentions these runtime settings, and the Store format,
  CLI and manifest columns are unchanged, so nothing in `opengwasdb-stores`
  moves.

## What failed, or could not be established

- **The brief's lever 2, as written, would hang builds** (above). Shipped as
  `max_workers=1` instead.
- **The brief's lever-1 bulk figure (52.5 → 37.5 s) did not reproduce.**
  Blosc threads alone, under zarr's default pipeline, made bulk 76 → 88 s
  (attribution). The seam comment and CHANGELOG are corrected (`d1c6c41`).
  `1c27e64`'s commit message quoted the old figure until step 5's pre-push
  rebuild reworded it (pushed as `bd0b52d`).
- **Scripts** (committed in step 6; see the table at the top).
  `levers/scripts/shapes_one_config.py` is untouched. The adapted
  copy is `shapes_cfg.py`: it adds `fused_mw1`, `asis`, p95, row counts, and
  the effective settings read back from the process. New scripts:
  `fork_probe.py`, `fork_paths.py`, `encode_scope.py`, `se_plan_check.py`,
  `compare_split.py`, `chunk_diff_decode.py`, `compare_spot.py`,
  `summarise_step1.py`, and the `run_*.sh` drivers.
- **Small-shape timings are bimodal between processes** (e.g. tophits 19 ms
  vs 56 ms for the same config in two processes, `decide_mw.out`), beyond the
  load effect. NUMA placement on this 224-core node is a guess I did not test.
  It is why the attribution uses three processes per config and Stage A was
  run twice.
- **The tophits floor remains.** About 5–7× slower than 2.18, as the brief
  predicted. Measured, not hidden. *Corrected in step 5:* the cost is zarr 3's
  per read call, not per chunk (ADR 0056). Top hits makes six one-chunk reads.
- **Not re-run:** the OGS-00008 pilot rebuild (the real-slice SE chooser
  stands in for it); `opengwasdb validate` on the three stores (78 min on
  OGS-00009; correctness is covered by spot identity and the harness digests);
  Stage B, the committed `opengwasdb_store_comparison_ogs00009_zarr3.json`.
  That belongs to the later brief, along with the stop rule and #246.
- **Not measured:** the effect of one-worker fused reads on a real *build's*
  forked workers. They now decode one chunk at a time, single-threaded, where
  zarr 3's default decoded a multi-chunk selection in parallel. Builds get
  their parallelism from processes, so I expect this to be small, but it is
  unmeasured.
- **For the reviewer:** whether "fused pipeline at one worker, or a fork reset
  of zarr's pool" deserves an ADR. #247 moves builders to v3 shards, and anyone
  who raises `codec_pipeline.max_workers` reintroduces the hang. Today that
  constraint is documented in the seam's comment and pinned by the fork test.
  Upstream zarr has not been told about the unreset `_pool`.
