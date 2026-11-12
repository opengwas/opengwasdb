# Completion writes its planes a unit at a time, across workers

Issue [#321](https://github.com/opengwas/opengwasdb/issues/321), under epic
#307. This ADR supersedes the part of ADR 0023 that kept Reference
Completion's write in one process: ADR 0023's LD-block pool, its checkpoints
and its resume are unchanged.

## Context

ADR 0023 parallelised the LD blocks and left everything after them to the
parent: it applied every block's fills and made the one write, because block
boundaries do not align with zarr's chunks and two workers could have written
one chunk file. Since then issue 044 made that write stream row bands from
per-band fill shards, #247 made every completion band whole Dense shards, and
#424 gave Ragged completion the same fill shards and one CSR writer.

The write stayed serial, and on the largest Stores it is hours. #309's profile
of OGS-00009 (`docs/benchmark-output/reference_completion_bands_ogs00009.json`)
put the Dense band write at 5.9 h of the 14.1 h completion:

- 63% of it read source planes. Three passes each read the grid: the
  statistic pass read z and SE, the EAF pass read the source's frequencies
  again cell by cell (`DenseEafPlane.points`, 26.8%), and the SE rewrite read
  the `float32` scratch plane twice more, once to count its exceptions (11.8%)
  and once to code it (7.9%).
- 25% computed, and 11% wrote.

OGS-00012's write ran for over 1.5 h at about 26% CPU. The owner wants a Store
of OGS-00009's size completed in an evening, about 9 h.

## Decision

**The write after the LD blocks runs a unit at a time on a pool of forked
workers: a Dense row band, a Ragged Analysis. Each unit is read once and
completed whole; the parent takes the units' results in unit order.**

- **A Dense band is one pass** (`_complete_band`). One read of the band's
  source rows yields z, SE and the EAF the reader decoded the SE against. The
  band's fills land, its missingness is counted, and `AssociationCodec` (ADR
  0077) codes z, then EAF, then SE against the EAF a reader decodes. Each
  plane's rows are written once, in the destination's dtype. There is no
  `float32` SE scratch plane and no rewrite. The band's reads, counts and
  encodes step through it 2^24 cells at a time, so their `float64`
  intermediates stay bounded at any width.
- **A Dense worker writes its own band.** A band is whole Dense shards
  (`completion_band_rows`, #247), so no two workers write one shard file: the
  alignment ADR 0023 lacked. The worker returns its counts and its side-table
  entries. The parent adds the counts, lays the z overflow, EAF exception and
  SE exception entries end to end in band order, and writes each table once.
  The entries are not sorted into place, so a band taken out of order makes a
  table its own rule refuses. The parent also logs, by variant, the fills
  each band dropped (D8).
- **A Ragged worker completes one Analysis** (`_complete_analysis`): it reads
  the Analysis's source rows and lands its fills. The parent reads each batch's
  fill shard once, hands each Analysis its fills, and adds the completed rows
  to the one `RaggedCSRWriter` in Analysis order, so the sequence and its
  offsets are the ones one process writes.
- **The Ragged sequence is then written a region at a time on the same pool**
  (`RaggedCSRWriter.flush_regions`). A region is whole sequence shards (#249),
  so its worker writes every plane of it; it codes its cells once, through
  `AssociationCodec`, SE against the EAF a reader decodes rather than against
  the frequency plane read back from disk. The parent lays the regions' side-
  table entries end to end in region order (`codec.table_end_to_end`), and adds
  their SE exception counts. A writer
  that fits its own SE model, a Ragged build's, needs its whole frequency plane
  before it can code SE, so builds still write through `flush`.
- **The fill-shard pass resolves its checkpoints on the same pool.** The
  parent stays the only writer of the shard files and of `completion_quality`,
  in checkpoint-name order.
- **The pool is `--n-workers`, capped by memory** (`encoding.memory.pool_workers`,
  one helper for every pool). The workers and the parent together stay within
  a quarter of `MemAvailable` at the phase's start, as for the Dense build's
  EAF encode (#320). The parent counts at its resident set when the pool
  starts, and a Ragged assembly's parent also counts the 17 bytes a row its
  writer will hold by the end. Each unit's working set was measured with
  `--write-workers` above 1 (the profiler) or by the pilot harness, each worker's
  private memory sampled:

  | unit | budget | measured |
  |---|---|---|
  | Dense band | 4 GiB + 24 B a cell: 16.4 GB for a 200,000-row band of OGS-00009's 2,514 Analyses | 10.2 GB |
  | checkpoint | 256 MiB + 4 x the largest checkpoint file | up to 3.9 GB on OGS-00009 |
  | Ragged Analysis | 256 MiB + 16 B a merged-axis variant + 32 B a row of the largest Analysis | 264 MB on the OGS-00006 subset (budget 666 MB) |
  | Ragged region | 256 MiB + 120 B a cell | 5.15 GB for a 50 M-cell region (budget 6.3 GB) |

  On OGS-00009 a ~15 GB parent leaves room for 13 band workers of a 235 GB
  budget. Each pool logs the count it ran with, the unit's figure, the
  parent's, and the budget.
- **Resume is unchanged: it redoes the write.** Only LD blocks are
  checkpointed. A failure in the write discards the staged release (ADR 0043),
  and a resume runs the merge and the whole write again from the block
  checkpoints.

**"Identical" means decoded values and metadata, not stored bytes.** The
release comparison tool (`benchmarks/compare_store_releases.py`) compares every
array's metadata, its on-disk size included, and its decoded values. It also
compares `analyses.tsv` and `variants.tsv.gz`. The tests add the
`completion_quality` rows in order, and the manifest but for `created_at`. The
stored bytes are not compared. The parent compresses a chunk with Blosc
threads, which write its blocks in the order they finish, while a forked worker
compresses single-threaded (ADR 0056). Two runs of the serial code already
differ there.

## Considered options

- **Keep the three passes and run each on the pool** (the SE rewrite already
  takes `n_workers`). Rejected: every pass would still read the grid, so the
  EAF pass and the SE count and read (46.5% of the write between them) would
  stay. The `float32` scratch plane, 4 bytes a cell beside the release, would
  stay too.
- **Workers return codes and the parent writes**, as the SE rewrite and the
  Dense build's EAF encode do. Rejected: the parent's compression and write
  would be serial again (11% of the write on one process). Every band's codes
  would also cross the pool's pipe, 5 bytes a cell: about 140 GB on OGS-00009.
- **Continue the write on resume.** Rejected. The staged release would have to
  survive a failure, which ADR 0043's staging does not allow, and every
  band's counts and side-table entries would need a checkpoint: a new
  checkpoint format. The LD blocks are the hours of a run, and a resume
  already keeps them.
- **Units of one shard on both axes**, a row band by 1,024 Analyses, so a
  worker's memory would not grow with the Store's width. Rejected for now. Fill
  shards are cut by row band, so each unit would read its band's whole shard.
  The memory cap already turns width into fewer workers.

## Consequences

- The write can run on fewer workers than `--n-workers`. The log line names
  how many, the unit's estimate, and the budget.
- Peak scratch disk falls by the `float32` SE plane and its `int8` copy. At
  every array write the staged planes hold no more than the release they
  become (`tests/test_completion_band_write.py`). The tiers #200 targets, the
  block checkpoints and the fill shards, do not change: on OGS-00003 they are
  77.3 and 16.0 bytes per imputed cell, 93.3 together at base and at head,
  where #200 measured 92 on OGS-00010 and sets a target of 16 or fewer. That
  target is #200's work (one ALID per block, no second copy of the fills),
  not this decision's.
- A Dense band worker holds its band whole: 18 bytes a cell of values and
  codes, and its reads' and encodes' steps beside them, measured at 10.2 GB
  private for a 200,000-row band of OGS-00009's 2,514 Analyses.
- `benchmarks/profile_reference_completion.py` attributes the one pass to
  reading, computing and writing. With `--write-workers` above 1 it times the
  pool's band write as one stage, records each worker's private memory, and
  refuses a slice with fewer bands than workers, whose time would not scale
  by bands.
- A Ragged Analysis's own warnings (an observed duplicate, a fill D5 drops)
  are logged by the worker that completes it, through the handlers it
  inherited. A Dense band's dropped fills are logged by the parent.
