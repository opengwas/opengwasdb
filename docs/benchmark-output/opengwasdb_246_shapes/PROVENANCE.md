# Provenance: #246's OGS-00009 shape comparison

These are the artifacts behind ticket #246 (`epic #240`'s go/no-go), produced on
the IEU compute node `app-dc3-ogws-p0` on 5–6 October 2026. Every number in the
rendered report (`opengwasdb_ogs00009_shapes.html`) is read from a file here at
render time; nothing is typed by hand.

Read this before re-using a number.

## Environments

| | python | zarr | numcodecs |
|---|---|---|---|
| zarr 2.18 base, checkout `745796c` (`/tmp/epic240/244/base-src`) | 3.11 | 2.18.7 | 0.12.1 |
| zarr 3 head, this worktree | 3.12 | 3.4.0 | 0.17.0 |

Every zarr 3 run used the pinned reader configuration, read back from the
process and recorded in each artifact: `numcodecs.blosc.use_threads = True`,
`zarr.core.codec_pipeline.FusedCodecPipeline`, `codec_pipeline.max_workers = 1`,
and #253's single EAF read. The imported package is named by
`opengwasdb_fingerprint` as well as the commit.

## Files

| file | made by | code | when (UTC) |
|---|---|---|---|
| `opengwasdb_store_comparison_ogs00009_zarr2.json` | `benchmark_store_comparison.py --reps 5` in the zarr 2.18 environment, store `v2-c1000` | `745796c` | 2026-10-06T01:27 |
| `opengwasdb_store_comparison_ogs00009_shapes.json` | the same harness in one zarr 3 process: `v2-c1000`, `v3-c1000`, `v3-c128`, `v3-c64` | `34b4ed1` | 2026-10-06T01:37 |
| `opengwasdb_store_comparison_ogs00009_zarr2_pair2.json` | the zarr 2.18 half of the **second** back-to-back pair | `745796c` | 2026-10-06T01:47 |
| `opengwasdb_store_comparison_ogs00009_shapes_pair2.json` | the zarr 3 half of the second pair, same four stores | `34b4ed1` | 2026-10-06T01:58 |
| `opengwasdb_store_comparison_ogs00009_shapes_set_l.md` | `zarr3_lever_tables.py shapes --base …_zarr2.json --head …_shapes.json` | analysis of the above | — |
| `opengwasdb_store_comparison_ogs00009_shapes_set_l_pair2.md` | the same generator on the second pair | analysis of the above | — |
| `opengwasdb_top_hit_shard_ab.json` | `top_hit_shard_ab.py` (interleaved A/B, 5 rounds × 100 alternating pairs, 500 samples per side, every sample kept) | `e2eaf4c` | 2026-10-06T05:26 |
| `conversions/*.log` | `/usr/bin/time -v` around `convert_store_to_0_2_0.py` | see each log | 2026-10-05/06 |
| `conversions/OGS-00009-v3-c64-245.log` | #245's conversion of `v3-c64`, reused here unmodified; copied from `/tmp/epic240/245/convert-ogs00009-run2.log` | `5cf7f78` era | 2026-10-05T05:02 |
| `screening/slice_read_step1_246.jsonl` | `shape_slice.py read` on a fresh 100,000-variant slice | `f54eb9d` | 2026-10-06T02:22 |
| `screening/screen_rank_check.json` | `shape_screen.py rank-check` over that read plus #244's committed outputs | `f54eb9d` | 2026-10-06T03:0x |

The second pair exists because the first pair put four peak-memory results
within 30% of the 300 MB cap (212–217 MB), and the ticket requires such a result
to be confirmed by a second back-to-back pair. Both pairs pass every limit.

The four conversion logs are the converter's own phase timings plus
`/usr/bin/time -v`'s wall clock and peak RSS. `v3-c1000` was converted before
the `top_hit_shard_chunks` provenance field existed (its write path is
unchanged, so its bytes are what today's converter writes); its top-hit shard is
still recorded in `provenance.zarr_v3_conversion.layouts`.

## The top-hit A/B is a paired measurement, not an absolute run

A conversion (w248's `OGS-00010`) was running in the same window, so the
top-hit comparison is an interleaved A/B rather than a quiet absolute harness
run: `v3-c64` and `v3-c64-topshard1` alternate **within one process, sample by
sample** (an earlier cross-process design was too noisy to settle the
difference), 100 pairs in each of 5 fresh-process rounds, and the per-array
result digests are compared between the sides in every round. The committed JSON
records 500 samples per side, medians `12.064 ms` (sharded, `v3-c64`) against
`11.083 ms` (unsharded), ratio `1.0885`, `0.1635 ms` per read, identity holding.
Those absolute medians are contended-machine numbers; the **ratio** is what
answers whether the top-hit index should be sharded. `python
benchmarks/zarr3_lever_tables.py decision` regenerates the tables from this file
and reproduces the numbers above.

## Regeneration

```bash
export PATH="$HOME/.pixi/bin:$PATH"
D=docs/benchmark-output/opengwasdb_246_shapes
OGS9=/data/opengwasdb/stores/OGS-00009/store.opengwasdb

# the harness pair (needs the heavy-job lock; start each at a 1-minute load below 3)
(cd /tmp/epic240/244/base-src && pixi run -e dev python benchmarks/benchmark_store_comparison.py \
   --store v2-c1000=$OGS9 --reps 5 --output $D/opengwasdb_store_comparison_ogs00009_zarr2.json)
pixi run -e dev python benchmarks/benchmark_store_comparison.py \
   --store v2-c1000=$OGS9 \
   --store v3-c1000=/data/opengwasdb/work/epic240/246/OGS-00009-v3-c1000 \
   --store v3-c128=/data/opengwasdb/work/epic240/246/OGS-00009-v3-c128 \
   --store v3-c64=/data/opengwasdb/work/epic240/245/OGS-00009-v3-c64 \
   --reps 5 --output $D/opengwasdb_store_comparison_ogs00009_shapes.json
pixi run -e dev python benchmarks/zarr3_lever_tables.py shapes \
   --base $D/opengwasdb_store_comparison_ogs00009_zarr2.json \
   --head $D/opengwasdb_store_comparison_ogs00009_shapes.json \
   > $D/opengwasdb_store_comparison_ogs00009_shapes_set_l.md

# the top-hit A/B
pixi run -e dev python benchmarks/top_hit_shard_ab.py \
   --config sharded=/data/opengwasdb/work/epic240/245/OGS-00009-v3-c64 \
   --config unsharded=/data/opengwasdb/work/epic240/246/OGS-00009-v3-c64-topshard1 \
   --rounds 5 --reps 100 --output $D/opengwasdb_top_hit_shard_ab.json

# the report
cd $D && pixi run -e report quarto render opengwasdb_ogs00009_shapes.qmd
```

## Addendum: the 256-wide Dense shard (`v3-c64-s256`)

Added after the human asked to try a 256-wide Dense shard before deciding.

| file | made by | code | when (UTC) |
|---|---|---|---|
| `opengwasdb_store_comparison_ogs00009_zarr2_s256.json` | `benchmark_store_comparison.py --reps 5` in the zarr 2.18 environment, store `v2-c1000` | `745796c` | 2026-10-06T09:04 |
| `opengwasdb_store_comparison_ogs00009_shapes_s256.json` | one zarr 3 process: `v2-c1000`, `v3-c64` (shard 1024), `v3-c64-s256` (shard 256) | `7774928` | 2026-10-06T09:10 |
| `opengwasdb_store_comparison_ogs00009_zarr2_s256_pair2.json` | the zarr 2.18 half of the confirming second pair | `745796c` | 2026-10-06T09:17 |
| `opengwasdb_store_comparison_ogs00009_shapes_s256_pair2.json` | the zarr 3 half of the second pair | `7774928` | 2026-10-06T09:23 |
| `opengwasdb_store_comparison_ogs00009_shapes_s256_set_l.md` / `..._set_l_pair2.md` | `zarr3_lever_tables.py shapes` | analysis of the pairs | — |
| `conversions/OGS-00009-v3-c64-s256.log` | `/usr/bin/time -v` around the conversion | see the log | 2026-10-06T09:56 |

`v3-c64-s256` is `--dense-analysis-chunk 64 --dense-shard 100000x256
--top-hit-shard-chunks 64`, wall clock 2:00:20, peak RSS 35.85 GB, writing
1407.6 s, verifying 457.4 s, validating 5352.0 s; bit-exact and validated clean,
published at `/data/opengwasdb/work/epic240/246/OGS-00009-v3-c64-s256`. The
first pair put two results within 30% of a limit (10 × 100's typical time and
its memory), so the second pair was run; both pass set L and the
whole-Analysis guard.

Regenerate the s256 tables with:

```bash
D=docs/benchmark-output/opengwasdb_246_shapes
pixi run -e dev python benchmarks/zarr3_lever_tables.py shapes \
  --base $D/opengwasdb_store_comparison_ogs00009_zarr2_s256.json \
  --head $D/opengwasdb_store_comparison_ogs00009_shapes_s256.json \
  > $D/opengwasdb_store_comparison_ogs00009_shapes_s256_set_l.md
```

Both s256 runs started at a 1-minute load below 3 (2.96 / 2.75 in the first
pair, 2.15 / 2.86 in the second). In the second pair the machine spiked inside
the load window between the check and the head run's first store, which recorded
12.88; the load had fallen to 3.5 and 2.72 by the two stores under comparison,
and the 2.18 half of that pair started at 2.15. The first pair's head run is
therefore the quiet measurement and the second confirms it; the two agree on
every direction reported.
