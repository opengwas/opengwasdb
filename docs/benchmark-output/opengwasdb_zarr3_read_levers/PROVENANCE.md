# Provenance: #244's zarr-python 3 read-lever outputs

These are the outputs behind ADR 0056 and the numbers quoted in #244, #240, #246, #252 and
#253. They are committed **as produced**, byte for byte, from the runs made during #244
(`/tmp/epic240/244/levers/` and `/tmp/epic240/244/` on the IEU compute node). The scripts
that produce them are in `benchmarks/`, documented in `benchmarks/README.md`.

**Read this before re-using a number.**

- Every output here except `blosc_decode_{1,2}.json` was produced by an **ad hoc script that
  was then ported** into `benchmarks/` for this commit. The port changes argument handling, not
  the measurement. "Port checked" below says how that was checked.
- None of them was re-run for this commit, except `blosc_decode_{1,2}.json`. They were measured
  once, at #244, on 2 and 4 October 2026.
- Only the #242 harness outputs (`stage-a*.json`, `rss-*.json`) and `blosc_decode_*.json`
  record their own commit and time. For everything else, **the commit and time below are
  reconstructed** from the files' modification times, the run logs and the commit times.
  Work measured before a commit landed was measured on that commit's changes, uncommitted
  in the worktree.

## Environments

All runs were on node `app-dc3-ogws-p0` (224 cores) with a warm page cache.

| | python | zarr | numcodecs | numpy |
|---|---|---|---|---|
| zarr 2.18 base, checkout `745796c` | 3.11.15 | 2.18.7 | 0.12.1 | 2.4.6 |
| zarr 3 head | 3.12.14 | 3.4.0 | 0.17.0 | 2.4.6 |

Paths recorded in the outputs are as produced:

- `/tmp/epic240/244/base-src` is the zarr 2.18 base checkout at `745796c`.
- `.../scratchpad/base708`, the `cwd` of attribution configurations (i)–(iii), is a
  `git archive` extract of `708d179`, run with the zarr 3 environment.
- The other `/tmp/claude-...` paths in the logs are fixture scratch directories.

## Commit names

The branch was rebuilt before its first push, to reword two commit messages; every tree is
unchanged. Outputs and logs record the SHAs from before the rebuild:

| measured as | pushed as | commit |
|---|---|---|
| `1c27e64` | `bd0b52d` | Blosc threads on |
| `8cc6ba1` | `af5e20a` | FusedCodecPipeline, one worker |
| `79e4858` | `3339776` | top-hit arrays opened once |
| `d1c6c41` | `0c54ab3` | timing corrections; no code change |

## Outputs

Times are UTC on 4 October 2026 unless stated.

| output | made by | code measured | when | runs | port checked |
|---|---|---|---|---|---|
| `stage-a3-{base,head}-nousersite.json`, `.log` | `benchmark_store_comparison.py` (#242), `--reps 3 --skip-rss` | `745796c` / `708d179` (recorded) | 2 Oct 02:08–02:10 (recorded) | one pair | committed harness, unchanged |
| `stage-a-step1{,-pair2}-{base,head}.json`, `.log`, `stage-a-step1{,-pair2}.out` | the same harness, back to back | `745796c` / `79e4858` (recorded) | 10:25–10:35 (recorded) | two pairs | committed harness, unchanged |
| `rss-pair-{base,head}.json`, `rss-step1-head.json`, `.log` | the same harness, `--reps 1` with peak-memory probes | `745796c` / `d1c6c41` (recorded) | 12:39–12:44 (recorded) | one pair, plus one zarr 3 run | committed harness, unchanged |
| `attribution/attribution.jsonl`, `attribution.out` | `zarr3_attribution.py` (was `scripts/shapes_cfg.py` + `run_attribution.sh`) | `745796c`, `708d179` with labels, `79e4858` as committed | 10:35–11:17 | 3 rounds × 5 configurations | `zarr3_lever_tables.py attribution` output byte-identical to the original summariser's; driver smoke-run against both environments |
| `decide_mw.out` | `zarr3_attribution.py`'s child, labels `fused+bt` and `fused_mw1+bt`, run directly | the `1c27e64` changes | 08:53–09:02 | 2 processes per label | as above |
| `fork_probe.out` | `zarr3_fork_probe.py` (was `scripts/fork_probe.py`) | zarr and numcodecs only | 06:54 | one per case | smoke-run: `fused_mw1+bt` ok, `fused` HANG |
| `repro_pool_fork_min.out` | `zarr3_pool_fork_repro.py` (was `drafts/repro_pool_fork_min.py`) | zarr only | 12:17 | one per mode | smoke-run: hangs by default, finishes with `--max-workers 1` |
| `fork_paths-bt.log` | `zarr3_fork_paths.py` (was `scripts/fork_paths.py` + #243's `build_stores.py`) | the `1c27e64` changes | 09:08 | one | smoke-run at `0c54ab3`: every path completes, gathers as logged |
| `fork_paths-fused-mw1.log`, `fork_paths-fused-pool-control.log` | as above; `--mode asis` and `--mode pool` | the `8cc6ba1` changes | 09:21 | one each; the pool control was killed at 240 s | as above |
| `observed-failing-{before,1c,fork-guard}.log`, `pytest-final.log` | pytest, on the committed tests | `708d179`; `8cc6ba1`; code without `max_workers`; `79e4858` | 09:04, 09:52, 09:20, 10:25 | one each | n/a |
| `encode_scope.out` | `zarr3_encode_scope.py` (was `scripts/encode_scope.py`) | zarr and numcodecs only | 09:02 | one per label | smoke-run on 2,000 rows: sizes equal threaded and not |
| `se_plan_check.out`, `se_plan_check_fused.out` | `zarr3_se_plan.py` (was `scripts/se_plan_check.py`) | the `1c27e64` / `8cc6ba1` changes | 09:13 / 09:28 | one per label and worker count | smoke-run on 2,000 rows |
| `compare-split-bt-n1.txt`, `compare-split-fused.txt`, `chunk-diff-decode-fused-default.json` | `zarr3_compare_trees.py` (was `scripts/compare_split.py`, `scripts/chunk_diff_decode.py`), on trees from #243's `build_stores.py` | trees at `708d179`, the `1c27e64` and `8cc6ba1` changes, and 2.18 | 09:09–09:27 | one | smoke-run at `0c54ab3` against 2.18: 0 of 466 chunk files differ. The trees now come from `zarr3_fixture_trees.py`, which adds Hybrid Reference Completion, so the counts are higher than these outputs' |
| `spot-base-OGS-*.json` | `zarr3_spot_queries.py record` (was `/tmp/epic240/244/spot_queries.py`) | `745796c` | 1 Oct 23:01 | one | n/a |
| `spot-step1-OGS-*.json`, `spot-step1-compare.txt` | `zarr3_spot_queries.py record`, `compare` (was `spot_queries.py`, `scripts/compare_spot.py`) | the `79e4858` changes | 10:01 | one | smoke-run at `0c54ab3`: the OGS-00001 record is byte-identical to `spot-step1-OGS-00001.json` |
| `slice_build.out` | `shape_slice.py build` (was `scripts/slice_build.py`) | `d1c6c41` | 11:56 | one | smoke-run on 10,000 rows |
| `slice/round1/*`, `slice/slice_read_*.jsonl`, `slice/run_slice_read.out` | `shape_slice.py read` (was `scripts/slice_read.py` + `run_slice_read.sh`) | `745796c` (base) / `d1c6c41` (step1) | 11:59 and 12:11 | 3 + 3 processes per environment | smoke-run under both environments; selections and chunk counts identical to the original's |
| `slice/decode_by_plane.jsonl` | `shape_slice.py decode` (was `scripts/decode_by_chunk.py`) | numcodecs only | 12:08 | 2 runs | smoke-run |
| `slice/harness_geometry.json` | `shape_harness_geometry.py` (was `scripts/harness_geometry.py`) | `d1c6c41` | 12:04 | one | smoke-run without bulk: all 47 non-bulk records identical |
| `slice/{cost_model,screen,expected_020,screen_rank_check}.{json,out}` | `shape_screen.py` (was `scripts/{cost_model,screen_shapes,expected_020,screen_rank_check}.py`) | analysis of the files above | 12:11–12:12 | n/a | re-run on these files: `screen`, `expected` and `rank-check` byte-identical. `cost-model` differs in the last digits of its floats, for example `threaded_ms_fixed` 0.35348597604342147 against 0.35348597604342163 here, and the original script re-run today gives the port's digits. The least-squares fit is not bit-reproducible |
| `eaf/eaf_split_{1,2}.json` | `eaf_read_split.py` (was `eaf/eaf_split.py`) | `d1c6c41` | 12:58, 13:00 | 2 fresh processes | smoke-run on two shapes: shares agree |
| `eaf/eaf_semantics_check.json` | `eaf_semantics_check.py` (was `eaf/eaf_semantics_check.py`) | `d1c6c41` | 17:24 | one | re-run at `0c54ab3`: byte-identical |
| `blosc_decode_1.json`, `blosc_decode_2.json` | `zarr3_blosc_decode.py`, **re-run for this commit** (the original `decode.py`'s output was not kept) | `9b0be2c` (recorded) | 22:18 and 22:19 (recorded) | 2 processes | n/a: these are its own outputs |
| `RESULTS-step1.md` | the #244 worker's report of these runs | | | | its citations point here |

## Not committed

- The fixture trees and the 100,000-variant slice: both are regenerated by the scripts above.
- #244's earlier Stage A tooling in `/tmp/epic240/244/`: `profile_shapes.py`,
  `tune_shapes.py`, `measure_reader_cache.py`, `compare_pilot.py`, `compare_trees.py` and
  their outputs. Its findings were superseded by these runs (#244's Stage A re-run comment,
  "What earlier reports on this issue got wrong").
