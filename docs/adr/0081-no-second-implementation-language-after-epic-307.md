# No compiled extension after epic #307: what remains is compiled numerics, layout and one reader

Issue [#322](https://github.com/opengwas/opengwasdb/issues/322), the last engine
ticket of epic #307, whose step 7 reserved this question: does any per-row
kernel still dominate a Store Release once the epic's other changes have
landed, and if one does, should it become a compiled extension (for example in
Rust), with the Python path kept as the reference it must match? This records
the answer and the profile behind it. It follows ADR 0046, which turned down a
compiled parser for the resolver because the per-row projection, not the parse,
was the cost.

**Status:** accepted by the owner (`explodecomputer`) on 16 November 2026, in
[#322's decision](https://github.com/opengwas/opengwasdb/issues/322#issuecomment-6904046834).
The acceptance does not waive milestone #388's nine-hour miss.

## Context

The epic's case against a port (#307, "Problem Statement") was that most of a
Store Release's hours were not interpreter time, and that a second
implementation would duplicate the code where this project's silent failures
live -- allele orientation, effect resolution, row admission -- without removing
most of the hours. Steps 1-6 then removed repeated work and single-process
work. #322 rebuilt OGS-00009 (2,514 Analyses, GWAS-VCF, hg19 lifted to GRCh38)
end to end at `8acd22b` with every one of them merged, as the Store Release
Build plans it after opengwasdb-stores#357 and #358, outside any backup window,
on the shared 224-core host (`docs/benchmark-output/opengwasdb_epic307_after.json`):

| step | October (#311's baseline) | #322 |
|---|---|---|
| build (32 workers) | 19h27m (much of it inside the backup window) | 2h53m51s |
| standalone Top-Hit Index | 1h20m52s | not planned |
| hm3 subset, overview | 13m12s, 4 s | 7m42s, 3 s |
| validate (32 workers) | 4h15m57s (one process) | 14m28s |
| complete-dense (32 workers) | 14h04m42s | 6h16m45.599s |
| the completed release's subset, overview, validation, Observed-View Check | 8m41s, 2 s, 4h40m33s, 15m13s | 7m49.687s, 2.327 s, 16m29.389s, 15m45.868s |
| **Store Release, first start to last end** | not a comparable contiguous span: records include later repairs | **10h12m57.574s** |

Milestone #388 asks for that whole Store Release in an evening, about 9 h.
It is **missed by 1h12m57.574s**. The epic's narrower sum (build, standalone
index, subset, both validations and completion) is 9h49m16s, below all three
projections (19.4, 18.2 and 16.3 h); that is not the complete release span.
The baseline's same six steps total 44h02m15s, including backup contention,
so their ratio is not an isolated measurement of the code's speed-up.

**How the profile was taken.** Every step was recorded whole by py-spy
(`--subprocesses --nonblocking`, 50 Hz, the step's whole process tree) for 40 s
every 20 min, each recording filed with the phase line the step had last
logged (`$RD/scratch/322/run/e2e/prof/`, summarised in `phases.json`). Without
`--native`, time inside a compiled extension is charged to the Python frame
that called it, so for the LD-block phase three native recordings of single
workers (`--native`, 20 Hz, 10 s each) split compiled from interpreted time. The three
recordings pool 790 active samples: a bounded sample of three workers, not a
wall-time decomposition. The reproducible summary, including every whole-tree
snapshot and input hashes, is `docs/benchmark-output/opengwasdb_epic307_profiles.json`;
`benchmarks/summarise_epic307_profiles.py` reads the existing recordings only.

## Where the time goes now

| phase (wall, #322) | busy | where the samples are | kind of cost |
|---|---|---|---|
| Reference Completion's LD-block phase (16,443.2 s, 32 workers) | about 87% of 32 workers in the early snapshot | `ElasticNetCV`: OpenBLAS 20.0%, numpy C 26.6%, scikit-learn Cython 7.3%; interpreted frames 28.6%, other native 17.5% (three native recordings pooled) | compiled numerics and Python orchestration per (LD block, Analysis) |
| the Dense build's band writes z / se / eaf (22m23s, 29m12s, 28m14s) | `_ordered_bands` is 82-93% of parent self samples when filling a band; workers mostly idle | `_ordered_bands`: one column at a time into a row-major band of 9.8 M rows × 1,024 Analyses, then the band's zarr encode and write | memory layout and one writer, not the interpreter |
| completion's Top-Hit Index (4,060.9 s) | one parent, no scan pool | vectorised codec calls (`logit`, `expit`, SE prediction); completion does not forward its worker count to `build_top_hit_indexes` | unpropagated parallelism, not a per-row interpreter kernel |
| Pass 2 (34m55s, 32 workers) | 91% | the GWAS-VCF reader one association at a time: `stream_associations`, `swap_one`, `_rows_of`, `_validated_cells`, `batched`, `_row`, `_listed`, `_oriented` together about 79% (78.57% pooled); the key join (#316) 3% | per-row Python |
| Pass 1 extraction (25m14s, 32 workers) | 77% | `_variants_of` 39% (the reader's per-record loop over `bcftools query`), `_validated_cells` 11%, the routing's per-variant bookkeeping about 30% | per-row Python |
| everything else (validation 14m28s, EAF survey 2m02s, residual EAF encode 6m22s, SE fit and rewrite with the inline index 14m21s, hm3 subset 7m42s) | -- | vectorised numpy, Blosc and zarr | already off the critical path |

The remaining costs need different remedies:

- **The LD-block solver is already compiled.** The pooled native sample puts
  53.9% in OpenBLAS, numpy C and scikit-learn Cython, versus 28.6% in Python
  frames. This corrects the draft's 70%, which used only the first recording.
  Rewriting surrounding Python cannot remove the compiled solver's cost, and
  the Python share spans orchestration and library checks rather than one
  demonstrated dominant per-row kernel. What would make the phase cheaper
  is fewer or cheaper fits -- fewer cross-validation folds, a shared or fixed
  regularisation path, or warm starts across Analyses. Changes to fits that
  alter imputed values are a new Reference Completion Method (#423), not a
  port. Sharing provably identical work across Analyses can instead be a
  bit-identical optimisation, but this run does not measure one.
- **The band writes are a layout cost.** Writing one Analysis's column into a
  C-order `(n_variants, 1024)` band touches one cache line per variant. A
  column-major working band would make those fills contiguous, followed by
  one C-contiguous copy for the write. This is a hypothesis to benchmark, not
  a measured full-build speed-up. The writer's encode can also use more than
  one process, as #321 did for completion.
- **The two GWAS-VCF passes are per-row Python, and they have a Python fix.**
  #318 and #319 gave FinnGen and GWAS-SSF sources association batches that match
  the per-row reader bit for bit; Pass 2 took 81% less time per Analysis on the
  FinnGen pilot and about 5.9× less on the GWAS-SSF one. The GWAS-VCF reader has
  no batch path yet. Its floor -- `bcftools query` alone, already compiled --
  reads ukb-b-10001 about 2.5× faster than the reader does (#311), so a columnar
  parse of that query's output in numpy bounds Pass 2's gain the same way, and
  the same parse serves Pass 1.

## Decision

**No compiled extension is built after epic #307; retain the existing
Python/Numba implementation.** No per-row kernel left after the epic justifies
one: the largest remaining phase is already compiled numerics, the next is a memory
layout, and the per-row Python that remains -- the GWAS-VCF reader in Pass 1's
extraction and in Pass 2, an hour of a 2h54m build at 32 workers, most of it in
the reader's per-record frames -- has the columnar approach #318 and #319
proved on other Source Readers, with the per-row reader kept as the reference
its batches must match.

A compiled extension is reconsidered only if, after a columnar GWAS-VCF reader
and a column-major band write, a phase on the critical path of milestone #388
still spends most of its samples in interpreted per-row frames that no
vectorised form can express. It would then be a contained kernel behind an
existing seam (a Source Reader's batch method, for example), with the Python
path as the reference and the equality tests #318 and #319 use, and this ADR
would be superseded with that profile.

## Considered options

- **A Rust (or C) GWAS-VCF reader feeding Pass 1 and Pass 2.** Rejected for now.
  It is the one place a port could help, but a numpy parse of `bcftools query`
  output reaches most of the same floor without a second implementation of
  orientation and row admission, the code ADR 0076's one reader seam exists to
  keep in one place.
- **A compiled elastic-net or fill kernel for completion.** Rejected. The
  solver is scikit-learn's Cython coordinate descent over OpenBLAS already; a
  second solver would have to reproduce its arithmetic bit for bit or be a new
  Reference Completion Method, and the interpreted share it could remove is
  small beside the fits themselves.
- **numba for the per-row loops.** Not needed for the same reason: the per-row
  loops left are the reader's, and their replacement is a vectorised parse,
  not a jitted loop.

## Consequences

- Milestone #388's evening is missed by **4,377.574 s**, and the LD-block phase
  is **44.7%** of the 36,777.574 s run. No merged ticket changed the fits.
  Closing the gap needs either more of the host for that phase (it is one fit per block and Analysis, with no
  ordering between blocks) or a cheaper fit, which is a new Reference
  Completion Method (#423).
- Four Python follow-ups are proposed, not filed here: a columnar GWAS-VCF
  reader (Pass 1 and Pass 2), a column-major Dense working band, forwarding
  completion's worker count to its Top-Hit Index scan, and an SE rewrite memory
  budget. SE rewriting peaked at **323.5 GiB private** across the build's
  processes (largest worker 12.1 GiB), exceeding the plan's approximately
  233 GiB allowance; available host memory stayed above 625 GiB. #320's and
  #321's pool budgets do not cover that rewrite.
- The profile behind this decision is committed beside the after-artifact, so a
  later proposal argues from the same measurements.
