# Reference Completion's write across workers (#321)

Measurements behind ADR 0080, taken on 12 Nov 2026 on the 224-core shared
host. Base is the integration tip `9dab184`. Head is #321's branch before #361
merged into it: `3fe26e9` for the profiles and `37db027` for the pilots, as the
artifacts record them. The branch was then rebuilt so that each commit passes
its tests, and those commits are now `5ef19c2` and `9ce2776`, with the same
`opengwasdb/` trees (`3360f41`, `fc8bf0b`) and `benchmarks/` tree (`8a87126`).
The 4-worker run's `0ca759f` is now `7cdc9bf` (`opengwasdb/` `117644f`).
Each timing pair ran in fresh processes. The pilots ran interleaved under the
run's timing lock. The OGS-00009 profiles ran base beside head, at the same
moment. Each artifact records the host load it ran under.

## OGS-00009: #309's profiler, the band slice with real fills

`bands_ogs00009_{base,head}-r{1,2}.json` come from
`benchmarks/profile_reference_completion.py bands`. Each covers the first 2 of
56 bands (400,000 rows of 2,514 Analyses) and the 50 LD blocks they need, on 16
workers. The source is OGS-00009's Observed-Only release, format 0.2.0. The
profiler calls the completion phases directly, below the format check
`complete_dense_store` makes, so the release is profiled as it is, and the
slice is discarded unpublished. Head ran its band write on one worker
(`--write-workers 1`), so every stage is read in one process.

| | base r1 | base r2 | head r1 | head r2 |
|---|---|---|---|---|
| source reads, slice (s) | 447 | 432 | 117 | 98 |
| band write scaled to 56 bands (s) | 19,807 | 19,422 | 10,722 | 9,404 |

Base's source reads are its statistic pass, its EAF pass (`points`), and the SE
rewrite's count and read. Head's are its one `inspect` read per band step.
#309 recorded 21,179 s at `703fb07`.

`bands_ogs00009_head-w4.json` is head after #361 merged (`0ca759f`, now
`7cdc9bf`) on 4
write workers. The merge's checkpoints and both bands run on the pool, and
each worker's private memory is sampled every 0.2 s. The two band workers held
10.2 and 10.1 GB private, against the 16.4 GB the pool budgets a band. The four
merge workers held 2.9–3.9 GB. The two bands took 203 s on the pool. One worker
took 383 s (head r1) and 336 s (head r2) for the same stages. The fill
sharding took 41 s on the pool, against 101–119 s on one worker. The host load
is in each artifact.

## Pilots: one set of LD blocks, resumed by each revision

`pilot_ogs00003/` is the FinnGen R13 pilot: 21,618,934 variants, 10 Analyses,
format 0.2.2. `pilot_ogs00006/` is the first 100 of OGS-00006's 1,400
Analyses, chromosomes 15–22, with its regions: 55,327,252 associations. Each
`<side>-<workers>-r<round>.json` comes from `benchmarks/completion_write_pilot.py
resume`. It records each post-block phase's seconds, the peak scratch disk
(checkpoint plus staged release, sampled every second), and the release's
size. `compare_summaries.json` is the release comparison tool's verdict on
each pair: every pair is identical in array metadata (on-disk size included),
decoded values, `analyses.tsv`, `variants.tsv.gz`, `completion_quality` rows,
and the manifest but for `created_at`.

| pilot, phase (s) | base, 1 | head, 1 | head, 16 |
|---|---|---|---|
| OGS-00003 band write, r1 / r2 | 528 / 502 | 95 / 94 | 11.6 / 11.9 |
| OGS-00003 checkpoint merge, r1 / r2 | 111 / 112 | 112 / 113 | 114 / 116 |
| OGS-00003 peak scratch (GB), r1 | 3.53 | 2.81 | 2.80 |
| OGS-00006 assembly, r1 / r2 | 189 / 165 | 163 / 191 | 12.1 / 12.7 |
| OGS-00006 CSR flush, r1 / r2 | 32.2 / 29.8 | 16.9 / 16.9 | 14.0 / 14.3 |
| OGS-00006 peak scratch (GB), r1 | 1.42 | 1.42 | 1.42 |

OGS-00003's release is 2.61 GB. Base's peak held a `float32` SE plane beside
it, while head's staged planes never exceed the release. OGS-00003's merge is
no shorter at 16 workers: it has 2.9 M fills to shard, and the rest of the
merge (the panel's reference EAF, the union map, the empty planes) is not on
the pool. On OGS-00006 the flush is two 50 M-cell regions, so two workers is
all it can use. At head the variant index (64–80 s) is most of the Ragged
write that remains.
