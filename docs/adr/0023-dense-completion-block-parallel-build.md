# Dense Reference-Completion build: process pool over LD blocks, no thread layer, checkpointed resume

Builds on ADR 0022. For a `gwas`-coverage Dense store (e.g. UKB-B), reference completion runs elastic-net imputation once per (LD block, Analysis) pair across the genome-wide reference panel — roughly 1,700 blocks × n_analyses fits, a long-running batch job with no incremental write path in the existing `_write_zarr` (builds the whole array in memory, one final write).

## Decision

Parallelise with a single-level `ProcessPoolExecutor` over LD blocks (`n_workers`); no thread pool. Each worker independently opens the read-only Observed-Only zarr and LD panel files and reads its own block's row slice — the main process submits only lightweight block descriptors, not pre-sliced payloads. Within a block, a worker loops serially over analyses (`ElasticNetCV` per analysis is CPU-bound; there is no I/O step to overlap the way pleiodb's inner thread pool overlaps `bcftools` subprocess reads). Workers return filled `(block_rows × n_analyses)` Z/SE arrays to the main process, which applies all fills in one serial pass after the pool closes and performs the single existing `_write_zarr` write.

Checkpoint/resume: each completed block's result is persisted atomically (temp-file-then-rename, matching the existing `.ldeig.npz` cache pattern) as its own file in a checkpoint directory. `complete-dense --resume <checkpoint_dir>` accepts only that path — no other build-parameter flags. All other parameters (min_cor, pca thresh, ld_panel_id, ancestry, source/dest paths) are loaded from a `build_params.json` written once on the first run, so a resumed run can never silently apply a different parameter set than the one its existing checkpoints were computed under.

Consequence for the completion-quality schema: `n_missing_off_panel` is computed once per Analysis during the sequential axis-construction phase — it's knowable from Observed-Only ∩ union-axis membership alone, with no LD block involved. `completion_quality` stays strictly grained at LD-block × Analysis, and its `n_missing` records only `n_missing_imputation_failed`.

## Considered options

- **Two-level worker/thread nesting, mirroring pleiodb exactly.** Rejected: pleiodb's thread layer earns its keep by overlapping GIL-releasing `bcftools` subprocess I/O. Dense completion here reads from a local zarr store with no analogous I/O-bound inner step, so an added thread layer would only contend for the GIL around CPU-bound sklearn fits.
- **Workers writing directly to disjoint row-ranges of the output zarr.** Rejected for now: LD-block boundaries are variable-width genomic intervals with no reason to align to zarr's declared chunk shape (ADR 0021), so concurrent workers could land writes in the same physical chunk file. Returning results to the main process for one serial write avoids that correctness surface entirely.
- **Checkpoint validation by comparing resume-time flags against stored params.** Rejected in favour of `--resume` accepting no build-parameter flags at all, loading everything from `build_params.json` — this makes a mismatched resume unrepresentable rather than merely detected.

## Consequences

- `n_workers` is a pure runtime knob with no effect on computed results, so it can be freely changed between runs, including on resume.
- The main process enumerates the full genome-wide LD block list up front (glob-based); unlike the Ragged pipeline, there is no per-analysis cis-window scan, since a `gwas`-coverage Dense store's block set is simply the reference panel's entire block list.
- Off-panel missingness bookkeeping lives on the analyses table as a per-Analysis scalar rather than in `completion_quality`, which stays a clean LD-block × Analysis grain (see CONTEXT.md `Reference Completion Quality`).
