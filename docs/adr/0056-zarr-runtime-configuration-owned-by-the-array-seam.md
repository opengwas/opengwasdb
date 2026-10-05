# zarr's process-wide runtime configuration is owned by the array seam

zarr-python 3 keeps several behaviours that decide what a built Store Release contains, how
fast it reads, and whether a forked build worker can finish, in **process-wide runtime
configuration** rather than in array metadata. `opengwasdb/store/arrays.py`, the seam every
Store array is created and opened through (#243), sets four of them once, on import:

| setting | value |
|---|---|
| `array.write_empty_chunks` | `True` |
| `numcodecs.blosc.use_threads` | `True` |
| `codec_pipeline.path` | `zarr.core.codec_pipeline.FusedCodecPipeline` |
| `codec_pipeline.max_workers` | `1` |

This ADR records why, what was rejected, what it costs, and the constraint it places on later
work. **`codec_pipeline.max_workers` stays at 1** until zarr resets the fused pipeline's thread
pool after `fork` (zarr-developers/zarr-python#4478).

## Context

#244 moved the package from zarr 2.18 to zarr-python 3.4 without changing the Store format.
Its first read benchmark, Stage A at `708d179` on OGS-00009 (Dense, 9,847,701 variants ×
2,024 Analyses), found zarr 3 slower on every query. One Analysis genome-wide was 2.85×
slower and top hits 19×.
Three causes were behind most of it:

1. **Blosc ran single-threaded.** `import zarr` runs `numcodecs.blosc.use_threads = False`
   for the whole process (`zarr/codecs/blosc.py`). zarr 3 also decodes on worker threads,
   where numcodecs' adaptive default would say no anyway. A `[1000, 1000]` int16 chunk
   decodes in about 4.5 ms single-threaded and about 0.6 ms with Blosc's 8 threads.
   That is from two runs on 4 Oct 2026, 4,440–4,490 µs against 596–602 µs. #244 first
   quoted about 1.1 ms threaded from a run whose output was not kept, and threaded
   decodes varied 2× between processes there.
2. **The default `BatchedCodecPipeline`** schedules each chunk's fetch and decode as
   separate event-loop tasks. The opt-in `FusedCodecPipeline` (zarr ≥ 3.3) fetches, decodes
   and scatters a selection in one hop to a worker thread.
3. **Repeated metadata opens.** zarr 3 reads metadata from the store on every
   `group[name]` and `name in group`. That one was fixed in the query facade (`3339776`),
   not in configuration, so it is not part of this ADR.

The builders fork worker processes (`build/ordered_pool.ordered_map`,
`completion/parallel.run_block_tasks`, and the Dense VCF and SE phases), and those workers
read Store arrays. Any process-wide setting has to be safe across `fork`.

The settings were attributed separately. OGS-00009, three fresh processes per configuration,
configurations interleaved, medians in ms:

| configuration | one whole Analysis | one variant, all Analyses | top hits | 10 × 100 lookup | 100 × 10 lookup |
|---|---:|---:|---:|---:|---:|
| zarr 2.18 | 26,395 | 9.92 | 1.12 | 73.5 | 707 |
| zarr 3, defaults | 75,805 | 49.9 | 18.8 | 154 | 744 |
| + Blosc threads | 88,270 | 34.1 | 20.7 | 186 | 1,051 |
| + fused pipeline, one worker | 23,905 | 22.7 | 20.6 | 103 | 575 |
| + top-hit arrays opened once | 22,894 | 23.6 | 6.84 | 111 | 689 |

Peak memory for one whole Analysis is **12.04 GB on zarr 2.18 and 1.14 GB on zarr 3 with
these settings**, in each of two runs, with identical answers.

## Decision

### 1. `array.write_empty_chunks = True`

zarr 2 wrote a chunk even when it was entirely the fill value; zarr 3 defaults to dropping
it. Dropping it would silently change the file set a release holds. The flag is runtime
configuration, not stored metadata, so a per-array value is lost the moment an array is
reopened. It is set process-wide (`708d179`). #247 revisits it when builders write Zarr v3
shards.

**Enforced by** the files builds write, not by the setting:

- `tests/test_array_conformance.py` requires every chunk of every array in every
  fixture build to exist as a file. The fixtures write 358 chunk files, some of them
  entirely fill, and the test asserts that some are.
- `tests/test_zarr_runtime_config.py` writes all-fill arrays whole and band by band, in
  the parent and in fork-pool workers, and counts the files.
- Both fail with the setting removed. The same conformance module also requires every
  build to write Zarr v2 metadata under a manifest declaring 0.1.0.

### 2. Blosc's internal threads on, process-wide

`numcodecs.blosc.use_threads = True` is set after importing `zarr.codecs.blosc` by name. That
makes sure zarr's `False` has already run. The seam raises `ImportError` if numcodecs ever
drops the attribute, because an assignment nothing reads would fail silently.

**It is safe from numcodecs 0.17, which becomes the floor** (`numcodecs>=0.17`):

- **Threads.** A threaded call uses Blosc's one global context. numcodecs 0.17 serialises
  every such call, compress and decompress, under a module `threading.Lock`, and releases
  the GIL inside it. Concurrent decodes queue rather than race.
- **Before 0.17.** 0.14.0, 0.15.0, 0.16.0 and 0.16.5 take no numcodecs lock on decompress.
  With `BLOSC_NTHREADS` set, c-blosc then rebuilds its global context before taking its own
  mutex.
- **Forks.** A forked process never uses the global context. numcodecs compares the pid
  with the importing process's and uses single-threaded context functions in a child,
  whatever `use_threads` says. It re-creates its lock after `fork`, and c-blosc's
  `pthread_atfork` child handler discards the inherited context.

### 3. `FusedCodecPipeline` for every array, with `codec_pipeline.max_workers = 1`

One worker is the measured choice:

- **Faster.** With Blosc threads on, decodes queue on numcodecs' lock anyway, so a pool adds
  contention, not throughput. With the default pool (224 workers on this node), one
  Analysis genome-wide took 41.1–45.5 s against 25.7–26.5 s with one worker, in two fresh
  processes each. The random lookups were 23–42% slower on the mean of the two. Only the
  one-window read (4,241,966 associations) gained, by about 7%.
- **Forks.** With more than one worker, the pipeline keeps a module-level
  `ThreadPoolExecutor` (`zarr.core.codec_pipeline._pool`). zarr 3.4's
  `reset_resources_after_fork` (`zarr/core/sync.py`) clears the event loop, the I/O thread
  and the executor in a forked child, but **not that pool**. The child inherits the
  parent's executor, without its threads. `ThreadPoolExecutor.submit` consumes an idle
  permit the parent left instead of starting a thread. So a child read hangs when it spans
  **more than one chunk, but no more chunks than the idle permits the parent's pool left
  behind**: the queued work has no thread to run it.

**The hang depends on read size.**

- A single-chunk read never uses the pool.
- A read of more chunks than the inherited permits starts a fresh thread and completes.

That is why fixture builds, with one chunk per array, never showed it.

It was reproduced:

- standalone, with no opengwasdb code (zarr-developers/zarr-python#4478);
- through the package's own `ordered_map` and `run_block_tasks`
  (`tests/test_zarr_runtime_config.py`);
- on real build functions on real data: `_collect_top_hit_eaf` and `_collect_top_hit_se`
  over OGS-00009 row bands of three chunks each, with `n_workers=2`. They were killed after
  240 s. With one worker they take 1.7 s and equal the serial result.

With one worker the pool is never created, because every `_get_pool` call in zarr 3.4.0 is
behind `max_workers > 1`.

**Enforced by** `tests/test_zarr_runtime_config.py`, each case in a fresh interpreter:

- the Blosc flag survives zarr's own imports and is in effect on a worker thread;
- Store arrays read through `FusedCodecPipeline`;
- after a multi-chunk read in the parent, fork pools complete within 120 s.

**A timeout in that last test means this constraint was broken, not that the test is
slow.** Do not raise its timeout to make it pass.

### 4. Writes refuse a group that consolidated metadata describes

zarr 3's `open_group` reads consolidated metadata in place of the live array metadata
whenever a record exists. A record is a v2 `.zmetadata`, or a v3 `consolidated_metadata`
block in `zarr.json`. zarr 2.18's `open_group` ignored it.

Nothing the package writes updates such a record, whether it creates, deletes or moves an
array. A write beneath one therefore leaves a release whose next open reads stale shapes
and chunks. The #244 review reproduced this: an EAF repair moved a rechunked
`eaf_baseline` into place, and the release then reopened with its old `(8,)` chunks and
failed to reshape.

The package never consolidates, and no registered Store Release carries a record (checked
5 Oct 2026). So the seam refuses, before changing anything, when a record describes the
group:

- every open in a mode other than `r`, checking the group's own directory and every
  enclosing group's (`mode="w"` deletes the group's own record, so only enclosing ones
  count);
- every metadata write and every delete, at the store, through any handle the seam
  opened;
- `move_in_group`, which renames directories outside zarr.

The second point closes a gap round 2 of the review found. A handle opened before a record
appeared passed the open-time check. The review replaced an 8-element array with a
3-element one through such a handle; the record kept shape 8, and a fresh open silently
returned `[0, 1, 2, 0, 0, 0, 0, 0]`.

Writes take routes no single seam function sees: `create_array`, `create_group` and
`require_group` go through the seam, but attribute writes and `del group[name]` go
through zarr's own API. So the seam opens every group on its own `LocalStore` subclass,
through the same `LocalStore.open` call zarr makes for a path:

- **Metadata writes.** Any write of `.zarray`, `.zgroup`, `.zattrs` or `zarr.json` is
  checked against the records that describe its directory.
- **Deletes.** A deleted directory is checked against the records enclosing it. Records
  inside it go with it.
- **Chunk writes** are not checked. A record holds no chunk data, so band writes cost
  nothing extra.

Read opens are unaffected. **Enforced by** `tests/test_consolidated_metadata.py`, for Zarr
v2 and v3 records alike: ten kinds of write through a handle opened before consolidation.

### What these settings do not remove

The remaining gap to zarr 2.18 is a fixed cost per **read call**, not per chunk:

- **Top hits** makes six array reads of one chunk each. It takes 6.84 ms against 1.12 ms
  on 2.18, so about 1.1 ms a read against about 0.2 ms.
- **One variant across all Analyses** makes seven reads over 16 chunks. It takes 23.6 ms
  against 9.92 ms.
- **On a 100,000-variant slice**, a read cost model fitted to the sharded zarr 3 shapes puts
  each extra inner chunk at 0.08–0.22 ms; the two fits bracket the value.

No configuration setting tried removes this cost. #246 measures it under sharding, and
sharding adds about 0.8 ms more to each read.

## Consequences

- **Every array the process touches gets these settings**, including user code that imports
  the package. That is the point: no Store array can be opened without them. It is also a
  side effect on other zarr users in the same process.
- **Build output is not byte-reproducible run to run,** as under zarr 2.18. A chunk of at
  least two Blosc blocks (256 KiB uncompressed at zstd clevel 3, so every `[1000, 1000]`
  Dense chunk) compressed in the parent writes its blocks in completion order. Decoded
  values and compressed sizes are reproducible, so the SE encoding plan, which is chosen
  from compressed sizes, is unchanged. Compare built stores byte for byte under
  `BLOSC_NTHREADS=1`, as #243 did.
- **Writes take the fused path too.** A band write encodes its chunks one at a time:
  300–329 ms for a 60-chunk band of real `z`, against 217–227 ms under zarr 3's defaults.
- **Forked workers decode single-threaded and one chunk at a time.** Builds get their
  parallelism from processes.
- **#245 and #247 must not raise `codec_pipeline.max_workers`.** The converter and the
  v3-shard writers are write-heavy, and zarr presents that setting as its parallelism
  lever. Raising it reintroduces the hang.
- **A successor ADR may relax the constraint** once zarr resets `_pool` after `fork`
  (zarr-developers/zarr-python#4478). It should re-measure first, because the speed
  argument for one worker holds independently of the fork bug.
- **Every benchmark in epic #240 runs under this configuration**, and records the effective
  values. #246 pins it; a benchmark under any other configuration is not comparable.
- **zarr renaming or dropping something fails loudly at import:** the seam raises if
  `use_threads` or `FusedCodecPipeline` disappears.

## Alternatives rejected

- **Leave zarr 3's defaults.** One Analysis genome-wide was 2.85× slower than 2.18 and top
  hits 19×.
- **Blosc threads alone, under the default pipeline.** One Analysis genome-wide went from
  76 s to 88 s: decodes issued concurrently from zarr's pool queue on numcodecs' lock.
- **The fused pipeline with its default pool.** It is slower here, and it hangs forked
  workers.
- **An at-fork hook that resets zarr's private `_pool`.** It works in a probe. But it keeps
  the slower pool, and it depends on a private name that zarr can change without notice.
- **The fused pipeline for reads only.** The pipeline is chosen per array at open time, so
  this means a configuration context around every open the query facade makes, lazy ones
  included. It would not remove the hazard either: builders read Store arrays in forked
  workers.
- **Blosc threads for queries only.** numcodecs' flag has no per-call or per-thread form.
  A "query-only" switch means "on from the first query onwards", a mode that depends on what
  the process did earlier. `opengwasdb validate` and the builders' own reads would stay
  single-threaded.
- **Keeping `numcodecs>=0.14`.** Releases before 0.17 do not lock threaded decompress.
- **Opening with `use_consolidated=False`, as zarr 2.18 effectively did.** The package
  would then read live metadata, but any other reader of the release would still get the
  stale record a package write left. It would also decide, ahead of #245 and #246, that
  Zarr v3 releases never use consolidated metadata, which is a read-latency lever.
- **Re-consolidating after each write.** The record must be rewritten after the whole
  write, not after each step, and at every enclosing consolidated root. A repair that dies
  midway would leave it stale anyway. Refusing is the answer that cannot return stale
  arrays.

## Evidence

The reasoning for each setting is in the messages of commits `bd0b52d` (Blosc threads),
`af5e20a` (the fused pipeline at one worker) and `0c54ab3` (a timing correction). The
measurements are in #244's Stage A re-run comment. The fork hang has a standalone
reproducer in zarr-developers/zarr-python#4478.

The measurements, and the scripts that made them, are in this repository. The outputs are in
`docs/benchmark-output/opengwasdb_zarr3_read_levers/`, committed as produced; its
`PROVENANCE.md` gives each output's script, commit and time. The scripts are in
`benchmarks/`, documented in `benchmarks/README.md`:

- the attribution table: `attribution/attribution.jsonl`, from `zarr3_attribution.py`, and
  tabulated by `zarr3_lever_tables.py attribution`;
- the decode times: `blosc_decode_{1,2}.json`, from `zarr3_blosc_decode.py`;
- the worker count: `decide_mw.out`, from `zarr3_attribution.py`'s child;
- the fork hang: `fork_probe.out`, `fork_paths-*.log`, `repro_pool_fork_min.out` and
  `observed-failing-fork-guard.log`, from `zarr3_fork_probe.py`, `zarr3_fork_paths.py`,
  `zarr3_pool_fork_repro.py` and the fork test;
- write cost and compressed sizes: `encode_scope.out`, from `zarr3_encode_scope.py`;
- peak memory: `rss-pair-*.json` and `rss-step1-head.json`, from the #242 harness;
- the per-read and per-chunk costs: `slice/harness_geometry.json` and
  `slice/cost_model.json`, from `shape_harness_geometry.py` and `shape_screen.py`.

Apart from the decode times, these were measured once, at #244, on zarr-python 3.4.0 and
numcodecs 0.17.0, and not re-run. Treat them that way until #246's harness re-measures under
this configuration.
