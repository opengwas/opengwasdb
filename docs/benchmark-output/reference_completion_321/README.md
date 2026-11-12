# Reference Completion's write across workers (#321)

Measurements behind ADR 0080, taken on 12 Nov 2026 on the 224-core shared
host. Base is the integration tip `9dab184`. Head is #321's branch before #361
merged into it: `3fe26e9` for the profiles and `37db027` for the pilots, as the
artifacts record them. The branch was then rebuilt so that each commit passes
its tests, and those commits are now `5ef19c2` and `9ce2776`, with the same
`opengwasdb/` trees (`3360f41`, `fc8bf0b`) and `benchmarks/` tree (`8a87126`).
The 8-band pool run and the head pilots ran at `25530c9`, after #361 merged in.
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

### The pool's write: 8 bands on 4 workers

`bands_ogs00009_{base-b8,head-b8-w4}.json` ran 8 bands (1.6 M rows), base
beside head at the same moment, from `9dab184` and `25530c9`. Head ran its
merge and its band write on 4 write workers, sampling each worker's private
memory every 0.2 s. `bands_ogs00009_b8_memory.log` is the host's available
memory through both runs, never below 738 GB.

| | base, serial | head, 4 workers |
|---|---|---|
| band write, slice (s) | 2,926 (stage sum) | 480 (`band_write_pool`) |
| band write scaled to 56 bands (s) | 20,485 | 3,358 |
| fill sharding, slice (s) | 205 | 143 |
| LD blocks completed | 89 | 194 |
| imputed cells in the slice | 737,774,600 | 1,606,680,019 |

Base's profiler took only the blocks of the slice's last chromosome
(chromosome 2), so its chromosome-1 bands carried no fills: 2.2× fewer cells
to fill and shard than head's, which favours base. #309 recorded 21,179 s
(5.9 h). Head's four band workers held 10.1–10.2 GB private against the
16.4 GB the pool budgets a band, and its four merge workers 3.2–4.6 GB.

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
