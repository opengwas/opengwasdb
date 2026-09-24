# Parallel Dense VCF build: serial Pass 1, disk-spill Pass 2, no large IPC

Builds on ADR-0019 (two-pass VCF build with inline liftover). The two-pass Dense
builder streams every VCF twice: Pass 1 collects the union variant set (then runs
one liftover), Pass 2 fills the `(n_variants × n_analyses)` `z`/`se` matrices one
analysis-column at a time. For the full ukb-b store (2514 genome-wide VCFs, ~9.85M
variants) the serial Pass 2 is a ~30–40h job, so it needs parallelising.

## Decision

Parallelise across analyses with a single-level fork-based `ProcessPoolExecutor`
(`n_workers`), but structure it so **no large object ever crosses the process
boundary**:

- **Pass 1 stays serial.** It streams each VCF's variants into one growing union
  set. Same-cohort VCFs share nearly identical variant lists, so the union
  converges after the first file (observed directly: the unique-variant count
  freezes after the first log checkpoint). Parallelising it buys almost no speedup
  and forces each worker to ship its whole ~9.85M-element variant set back to be
  merged in the parent — see rejected options.
- **Pass 2 uses disk-spill.** Each worker streams one VCF, resolves associations
  against the fork-inherited read-only lookups, and writes its compact
  `(rows, z, se)` result to `{spill_dir}/{col_idx}.npz` (atomic temp-then-rename).
  It returns only the integer `col_idx`. The parent loads each `.npz` as its
  future completes and scatters it into `z_mat`/`se_mat`, then deletes it. The
  spill dir is created next to the output store and removed in a `finally`.

Workers inherit `hg19_lookup` and `variant_index` via fork (module globals set
before the pool is created), so those dicts are never pickled per task.

## Considered options

- **Parallel Pass 1 returning per-file variant sets (the original attempt).**
  Rejected — it deadlocked at genome-wide scale. Every worker returned a
  ~9.85M-element `set`, and because the union has already converged the parent
  had to unpickle and merge 2514 near-identical giant sets serially. That pipe
  traffic, combined with fork plus the pool's internal feeder threads, hung the
  parent on a futex. The chr1 test set never exposed it (files ~13× smaller, only
  100 of them). This is the bug [[0023-dense-completion-block-parallel-build]]'s
  "return results to the main process" pattern is safe from only because its
  per-block results are small; per-file results here are not.
- **Pass 2 workers returning `(rows, z, se)` arrays over IPC.** Rejected for the
  same large-result hazard — up to ~9.85M rows per file across 2514 files. numpy
  arrays pickle more efficiently than sets, but the failure mode is the same
  class; disk-spill removes it entirely by returning only a path-sized token.
- **Workers writing directly into a shared-memory `z`/`se` array.** Rejected for
  now. `/dev/shm` can hold the 99 GB, and disjoint columns mean no write
  conflicts, but it adds fork + `mmap` + concurrent-write semantics — exactly the
  class of subtle-at-scale bug that just cost a 9h run. Disk-spill keeps the
  parent's memory path byte-identical to the proven serial builder, which is the
  lower-risk change immediately after a scale failure. Shared memory is the
  natural next step if the disk I/O ever dominates.

## Consequences

- `n_workers` is a pure runtime knob; results are independent of it. Verified by
  `test_two_workers_matches_serial` (serial and 2-worker builds are bit-identical)
  and by a full chr1×1000 build (6m23s end to end, Pass 2 in 29s, no deadlock).
- Pass 1 is now the dominant wall-clock cost (~2.5h serial for ukb-b) since Pass 2
  drops from tens of hours to tens of minutes. Pass 1 is safely parallelisable
  later with the same disk-spill trick if that time matters, but the union-
  converges-immediately property means the payoff is small.
- **Memory is high and fork-driven, not fundamental.** Measured mid Pass 2 on the
  ukb-b build: `AnonPages ~492 GB` against a 99 GB logical `z`+`se` matrix. Two
  effects inflate it — the matrices COW-double because they are allocated before
  the pool forks and written after (children pin the originals while the parent
  copies pages away), and the large lookup dicts get progressively private-copied
  in each worker by CPython refcounting on read. Both are avoidable; tracked in
  `issues/043-dense-vcf-build-memory-footprint.md`. It fits the 1 TB target box
  with headroom today, but it is why a larger analysis count would hit a memory
  wall before a CPU one.
- Transient disk: the spill dir holds at most ~`n_workers` `.npz` files at once
  (the parent drains them as fast as workers produce), so it stayed at ~289 MB
  during the ukb-b run despite ~400 GB of total spill written over the build.

## Scope note (issue #220)

The "no large object ever crosses the process boundary" rule above governs
**Pass 2**, where the parent had to hold every worker's result to merge it and
the volume was the whole build's. It is not a rule against every pool in the
build. The band write that consumes Pass 2's spills (ADR 0036) loads a *band's*
columns in parallel and returns each already-encoded column to the parent,
bounded by `ordered_map`'s in-flight window (`2 * n_workers`, issue #220). There
the result is a fraction of one band, the parent consumes results in Analysis
order and discards each as it is scattered, and peak memory is the band buffer
plus that fixed window. Writing the encoded column to disk and reading it back
would instead pay two extra passes over the same bytes to avoid a bounded pipe.
