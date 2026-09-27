# Phase-granularity checkpoint and resume for the Hybrid build tail

OGS-00011's Hybrid build failed in `_fit_joint_se` (issue #226) after 8 h 15 m,
with the Dense Component fully written and verified, and all of it was
discarded. Nothing survived *by design*: `_build_components` ended with
`finally: shutil.rmtree(spill_dir)`, whose docstring names the invariant — "the
store's files are only touched while the spills exist" — and the Staged Release
context (ADR 0043) removes its own work directory on any `BaseException`. Two
phases cost the hours: Pass 2 routing at 4 h 04 m and the fold-through-band-write
stretch at ~3 h 31 m; the tail that crashed was 36 m of that run.

The invariant is worth keeping as the default, so what this ADR adds is a way to
opt out of it, and a design for what "resume" then means.

## Decision

**A checkpoint is a hidden sibling of the destination, holding what each phase
produces.** `.{name}.checkpoint` beside the output path, the same convention
Reference Completion uses (ADR 0023), reached through the same
`checkpoint_dir_for`/`require_fresh_destination` pair. It holds:

| | |
|---|---|
| `build_params.json` | every build parameter, the record's own `format_version`, and each external input's path, size, mtime and SHA-256 |
| `<phase>.done` | a completion marker per recorded phase: `pass2`, `fold`, `orientation`, `plan`, `dense_bands` |
| `spill/` | the Pass 2 spill directory itself, retained rather than removed |
| `plates.json` | the spill plates the phases still to run read, with their sizes |
| `info_counts.json` | each Analysis's declared-score dispositions (stores #175) |
| `fold/` | a completion file per folded column |
| `axis.npz`, `axis_*.txt`, `provenance_*.tsv` | the post-Pass-2 axis, frozen: key table, `old_to_new`, the ALID lists, the Dense-row map and the provenance maps |
| `orientation.json`, `plan.json`, `dense_hits.npz` | the report, the encoding plan and the Dense band write's top-hit harvest |
| `staged/` | the partially built release, renamed out of the Staged Release work directory |

**The build tail carries no marker and re-runs wholesale.** After
`dense_bands` come the Overflow CSR assembly, the frequency plane, the joint SE
fit, the Dense Component finish, the CSR flush and the shared metadata. Each is
cheap beside the phases above once the spills and the plan are in hand, and each
is idempotent over what a failed attempt left there: `write_eaf_plane` recreates
the ragged group with `mode="w"`, zarr chunk writes and top-hit tiers are
replaced rather than appended, and the manifest, `analyses.tsv`, variant table
and index are rewritten. So the Overflow `.ovf` plates are not consumed under
`--checkpoint` (`_assemble_overflow_csr(consume_spills=False)`), because the CSR
itself lives only in memory and the plates are its only durable input.

**The measured states are frozen, and only the plan's own phase may measure
them.** The encoding plan and the post-Pass-2 axis are measured from the data,
and the Dense bands already on disk were written under the plan as measured and
keyed on that axis. Re-measuring either on resume could quantise the same cells
differently — silent corruption of a store that still validates — so
`plan.json` is written *before* the first band write and reloaded verbatim, and
the fold's key table is written *before* its first column: the resolved key set
is a function of *every* column's spill, so a fold resumed without it could
neither re-derive it from the columns that are left nor be trusted to resolve
the same keys again. A crash *inside* the plan phase re-measures, which is safe
by construction: the marker is set after the plan and the Dense zarr skeleton
are on disk, and no band exists yet to contradict.

**Resume takes the checkpoint directory and nothing else.** `resume_hybrid_build`
reloads every parameter from `build_params.json`, so a mismatched resume is
unrepresentable rather than merely detected (ADR 0023). The CLI's `--resume` is
the convenience form: it requests the build again from the operator's own
arguments and *compares* them against the record, refusing with the differing
keys named. `n_workers` is excluded from that comparison — a pure runtime knob
no computed value depends on — as is the record's own `format_version`, which
`read_build_params` checks separately.

**The partially built release is retained, and adopted on resume.** The Dense
bands written before the crash are the reason a retry is cheap, and they live in
the Staged Release work directory that ADR 0043's context deletes. Rather than
race that cleanup from outside, `OpenGWASDBStore.staging` grows two opt-in
parameters: `retain_on_failure_to=<checkpoint>/staged`, which moves the work
directory there instead of removing it on failure, and `adopt=<checkpoint>/staged`,
which replaces this invocation's fresh work directory with the retained one.
Publication is unchanged — the resumed build commits through the same
two-rename swap under the same parent-directory lock — so a resumed build
publishes as atomically as an uninterrupted one. Both parameters are renames and
both default to `None`, so every existing caller is unaffected. The retained
release is never publishable on its own: it sits inside the hidden checkpoint
directory, and the run that left it wrote no manifest for the outer release.

**The budget is bounded up front.** The retained spills are the cost — 734 GB for
OGS-00011 — and a phase marker is a promise that everything the next phase reads
is on disk in full, so a resume validates the plate inventory (name and size)
before reading anything, and refuses a checkpoint whose staged release is gone, a
torn marker set, an absent or mismatched format version, an absent
`build_params.json`, or a parameter or input identity that changed. The
`--overwrite` that already discards a stale destination discards a stale
checkpoint too, and a build finding one without `--resume` refuses naming
`resume_hybrid_build`.

## Considered options

- **Re-running the phases from the retained spills instead of recording their
  products.** Rejected for the band write: `_write_dense_bands` unlinks each
  column's dense spill once both its passes have read it, so the phase is
  re-runnable only *within* itself and a crash after it leaves nothing to write
  from — the marker and the recorded harvest are what make a resume at a later
  phase possible at all.
- **Persisting the assembled Overflow CSR instead of keeping the `.ovf` plates.**
  Rejected: it is 16 bytes per association cell (about 240 GB at OGS-00011's
  15,078,327,210 Overflow cells), it needs a round trip through the writer's
  private per-Analysis arrays, and it duplicates what the plates already hold.
  Re-assembling costs a read of the retained plates, which is minutes beside the
  hours the checkpoint saves.
- **Letting the Staged Release context's exception handler delete its work
  directory, and re-raising into a wrapper that had moved it first.** Rejected:
  racing another component's cleanup is exactly the kind of implicit ordering
  that fails silently when either side changes. The retention is a parameter of
  the context that owns the directory.
- **A `.{name}.hybridspill.*` directory left where it was, with its path recorded
  in the record.** Rejected: a discarded checkpoint would leave the multi-terabyte
  spill behind, and "discard" has to mean one removal.
- **Re-deriving the pre-fold state on every resume, including after the fold.**
  Rejected on cost and on risk: for the legacy `--reference-panel` path it means
  re-reading every source, and it would re-measure an axis the Dense rows and
  Overflow plates are already keyed on. A resume after the fold rebuilds the
  prepared build from the record alone, touching neither Pass 1 nor any
  measurement.
- **A separate `build-hybrid-resume CHECKPOINT_DIR` command only.** Kept as the
  Python entry point (`resume_hybrid_build(checkpoint_dir)`) and reachable as
  `--resume` on `build-hybrid`, where the operator's own parameters supply the
  comparison. A command that took the directory alone would have no parameters to
  compare, which is the point of the API and the loss of the check.
- **Sub-phase resume (row-chunk ranges, completed Overflow columns).** Deferred:
  Phase 1's phase granularity captures most of the value because failures land in
  the tail, and per-chunk records are what make a checkpoint's format expensive
  to change (issue #227's Phase 2).

## Consequences

- Without `--checkpoint`, behaviour and on-disk footprint are unchanged: the
  spills are a private `mkdtemp` removed in every case, the Staged Release work
  directory is removed on failure, and no checkpoint directory is created. A
  test pins that alongside the resumed-build comparisons.
- The checkpoint's cost is dominated by the spills, so `--checkpoint` is for a
  release-scale build that expects to be interrupted, not a default. A build that
  neither fails nor is abandoned pays for writing the small records and for
  keeping the plates until the tail has read them.
- The retained plates make the pipeline's spill-consumption policy conditional:
  `_assemble_overflow_csr` takes `consume_spills`, and `_write_dense_bands`
  (dense, unchanged) still unlinks. A future reader of the spill directory has to
  know which of those applies.
- `checkpoint_dir_for` and `require_fresh_destination` are imported from
  `opengwasdb.completion.checkpoint` rather than re-implemented, so the
  `.{name}.checkpoint` convention and the refusal wording have one home across the
  two checkpointed builds.
- A checkpoint is a durable record of a build's *inputs* as well as its
  intermediates: an edited manifest, a rewritten reference or a touched file
  refuses a resume, and the honest response is a fresh `--checkpoint` build.
- What a resumed build produces is a store that decodes exactly as an
  uninterrupted one does — every zarr array (dtype, shape, values, NaN-aware),
  the same encodings, and the same `manifest.json` and `analyses.tsv` apart from
  `created_at`. Compressed chunk bytes are not the contract: Blosc produces
  different-but-equivalent streams for equal input in different processes
  (#230/#231).
