# Resumable manifest Analysis resolution with atomic per-Analysis records

Production Store Release manifests contain thousands of independent Analyses and
may execute across hours of compute time. The existing manifest-level ancestry
and phenotype-SD commands aggregated all results in parent memory and wrote their
tables only after the entire batch completed. An interruption (timeout, node
preemption, kill signal) lost all completed work, and resolving a single corrected
or failed source required re-executing the entire collection.

The one-pass resolver seam (`opengwasdb.build.resolve.resolve_analysis`, ADR 0044)
reduced per-Analysis work to a single source pass with bounded memory, but left
batch orchestration and durable checkpointing unaddressed.

## Decision

**Expose a manifest-level CLI with atomic per-Analysis records and resume.**
`opengwasdb resolve-analyses` (`opengwasdb.build.resolve_manifest`) processes a
canonical `analyses.tsv` manifest and writes one versioned JSON record per
Analysis (`{records_dir}/{analysis_id}.json`) plus an aggregate `index.json`.

**Resume is content- and configuration-aware.** With `--resume`, an existing
record is skipped if and only if its status is `success` and every fingerprint
input matches the current run:
- source file modification time, size, and manifest-recorded checksum/size;
- OpenGWASDB package version and git revision;
- ancestry reference path, identifier, and SHA-256;
- ancestry groups mapping SHA-256;
- extraction panel path, variant count, and SHA-256;
- declared reference-AF resources and SHA-256s;
- resolution configuration (method tiers, stored effect scale, sample size,
  reader capability, admission gates, MAF floor, evidence sample limit).

Any change to data, references, tool version, or configuration invalidates the
fingerprint digest and triggers re-execution. Records with `controlled_failure`
status or corrupt JSON are always rerun.

**Writes are atomic via temporary-file-plus-rename.** Every record and the aggregate
`index.json` are written to unique sibling temporary files, flushed, fsynced, and
committed via `os.replace`. An interrupted run cannot leave a corrupt or partial
record that a subsequent resume would treat as valid.

**References are loaded once and fork-shared.** The genome-scale Ancestry
Reference Panel (~1 GB) and declared AF reference panels are loaded once in the
parent process before starting the worker pool. Workers inherit read-only memory
pages via POSIX fork copy-on-write and never re-read reference panels from disk.

**Task scheduling mitigates stragglers.** By default (`largest_first=True`),
tasks are ordered by source file size descending, ensuring multi-gigabyte sources
start immediately. Deterministic manifest order is preserved in `index.json`
regardless of completion order.

**Ordinary errors are isolated; systemic errors fail loudly.** An unreadable or
malformed source records a `controlled_failure` JSON record and the batch
continues. Systemic configuration errors (missing manifest, duplicate analysis IDs,
missing references, unwritable output directories) fail the command immediately
with a non-zero exit code.

## Consequences

- **Durable checkpointing across long runs.** Interrupted runs resume from the
  exact set of remaining, failed, or invalidated Analyses without re-running
  successful ones.
- **Byte-equivalent outputs across concurrency levels.** Runs with 1 worker or
  multiple workers, executed largest-first or manifest-order, produce
  byte-equivalent per-Analysis records and deterministic index ordering.
- **Clean separation between engine computation and registry policy.** OpenGWASDB
  owns the statistical resolution, fingerprinting, worker pooling, and durable
  records; `opengwasdb-stores` owns candidate generation, release membership,
  acceptance thresholds, and Store construction.
