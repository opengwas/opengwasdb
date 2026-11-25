# #477 Rust spike: scattered Dense lookups on OGS-00009

Results: [`rust_spike_ogs00009_tables.md`](rust_spike_ogs00009_tables.md), generated from
[`rust_spike_ogs00009.json`](rust_spike_ogs00009.json) (the driver's own output: every round, every digest, the
effective configuration read back in each process, load at each measurement's start, hardware, array sizes) by
`benchmarks/rust_spike/summarise_spike.py`. The selection is committed beside it
([`rust_spike_selection.json`](rust_spike_selection.json)). Every number in the ADR is read from the tables.

## What was measured

OGS-00009 as released (Dense, Reference-Completed, 11,191,070 variants x 2,514 Analyses, format 0.2.0, inner chunk
1000x64, shard 100000x1024), the seed-0 selection, shapes `random_lookup_10_variants_100_analyses` and
`random_lookup_100_variants_10_analyses`. The bar the ticket sets (about 5x) applies to the **end-to-end
lookup** on **both** shapes; ALID-to-row resolution is timed apart and is under 1 ms.

Candidates: `python-fused` (the reader as shipped), `python-zarrs` (zarrs-python's pipeline), `python-zarr-pool16`
(zarr's own pool at 16 workers), `python-threadpool16` (the reader's `grid.read` on a 16-thread pool over row
partitions, Blosc threads off), `rust-1thread` and `rust-16thread` (the `zarrs` crate; zarrs' own codec concurrency
capped at 1; `RAYON_NUM_THREADS` set to the thread count). `-1000` / `-100` is the inner variant chunk: the shipped
1000, or a full-axis copy of z, se, eaf, imputed, eaf_baseline and eaf_reference at 100.

Each measurement is a fresh process (1 cold call, then 3 warm calls); rounds are interleaved, so each round runs every
candidate once. At least 7 rounds, hence 7 cold calls per column. "Cold" is the first call of a fresh process, not a
first-touch disk read: the page cache cannot be dropped without root. Agreement is checked on every process: the
first `python-fused` run is the reference and the run aborts on any difference, Rust cold output included.

## Caveats

* Shared 224-thread node. The 1-minute load at measurement start is in the tables; `--max-start-load 25` because 3.0
  never cleared. Ratios are within a round; absolute times are not comparable across sessions.
* The Rust spike is Dense only, reads the planes the Python lookup reads, and takes the row of an ALID wider than the
  32-byte index slot from the caller (the Python reader scans the variant table for it). It is not a reader. It
  also reads whole exception tables per call, which is unoptimised.
* The Python thread pool is one design; it does not show what Python can or cannot do, or why it was slower.
* The baseline here is slower than the figures in `store_reader/arrays.py:103-105` (103 / 575 ms), and the
  committed facts account for most of the gap. Those figures come from the earlier OGS-00009 (9,847,701 variants x
  2,024 Analyses, observed-only, **format 0.1.0**; `opengwasdb_store_comparison_ogs00009_zarr3.json`, which ran at
  load 2.9). ADR 0058 (lines 340-341) measured the 0.1.0 to 0.2.0 move on OGS-00009 as 2.80x and 1.82x slower on
  these two shapes, which takes 103 / 575 ms to about 288 / 1047 ms. That earlier store no longer exists to re-time.
  `rust_spike_ogs00009_old_selection.json` ([tables](rust_spike_ogs00009_old_selection_tables.md)) times the Python
  reader on that artifact's own selection against the current store; it is a separate run, and its absolute times are
  not compared with the main run's.
* Rounds: the two lookup shapes had 7 rounds, each a fresh process with 1 cold call and 3 warm calls; the harness's
  other shapes (whole Analysis, phewas, regional, regional one Analysis) had 5 rounds of 1 cold call and 1 warm call;
  the old-selection run had 5 rounds of 1 cold and 3 warm.
* "1 vs 1" means one decode worker with numcodecs' Blosc threads as the shipped reader leaves them on. That is
  moot for a [1000, 64] chunk: `store_reader/arrays.py` notes a chunk under 256 KiB uncompressed is a single Blosc
  block, so internal threads have nothing to split (z is 125 KiB, se, eaf and imputed 62.5 KiB). It was not measured
  separately.
* Agreement digests: a Python process digests its first and last (timed) call; a Rust process digests its cold call
  and its untimed warm-up call, not a timed warm call. The Rust thread count and zarrs codec concurrency are
  recorded as the binary set them, not read back from zarrs. The `python-zarrs` rows record zarr settings that zarrs
  ignores; zarrs-python's own concurrency is not recorded.
* `python-zarr-pool16` ran with Blosc threads on, a set-up `store_reader/arrays.py:107-110` records as queueing on
  numcodecs' lock. It is kept as measured and is not compared with Rust.

## Provenance

The main run started at commit `07a46faf` (17:15 UTC) and ran for about two hours. Commit `d348d949` (17:20 UTC)
changed `run_spike.py` while it was running, to satisfy the escapes gate: four `lambda` assignments became nested
functions with the same bodies. It is behaviour-identical, so every child spawned after 17:20 ran the function
version; the JSON records the commit the run started from (`07a46faf`, `tree_clean: true`). The old-selection JSON
records `d348d949` with `tree_clean: false` because `summarise_spike.py` (the matched-comparison section) was edited
and not yet committed when it ran (it was committed in `34d220ed`); `run_spike.py` was the committed `d348d949`
version. The scripts were later reformatted (`0561acda`) and this round changed the docstrings and the summariser;
the measurements were not repeated.

## Commands

```bash
export PATH="$HOME/.pixi/bin:$PATH"
S=/data/opengwasdb/work/orchestrate/opengwasdb-450/scratch/477     # scratch pixi project: zarr 3.4.2, zarrs-python 0.2.3, rust
B=/data/opengwasdb/stores/OGS-00009/store.opengwasdb

# the Rust binary
( cd benchmarks/rust_spike && CARGO_TARGET_DIR=$S/target pixi run --manifest-path $S/pixi.toml cargo build --release )

# the re-chunked copy (52 GB, about an hour; every block is read back and compared bytewise)
pixi run -e dev python benchmarks/rust_spike/rechunk.py --source $B \
    --destination $S/ogs00009_inner100/store.opengwasdb --variant-chunk 100
R=$S/ogs00009_inner100/store.opengwasdb

# the benchmark (one process at a time on the shared host)
pixi run -e dev python benchmarks/rust_spike/run_spike.py \
    --selection docs/benchmark-output/rust_spike_selection.json --output docs/benchmark-output/rust_spike_ogs00009.json \
    --base-store $B --rust-binary $S/target/release/opengwasdb-rust-spike --zarrs-python $S --scratch $S \
    --rounds 7 --warm 3 --extra-rounds 5 --extra-warm 1 --max-start-load 25 \
    --extra-shape bulk --extra-shape phewas --extra-shape regional --extra-shape regional_one_analysis \
    --run python-fused-1000=python-fused@$B --run python-zarrs-1000=python-zarrs@$B \
    --run python-zarr-pool16-1000=python-zarr-pool16@$B --run python-threadpool16-1000=python-threadpool16@$B \
    --run rust-1thread-1000=rust:1@$B --run rust-16thread-1000=rust:16@$B \
    --run python-fused-100=python-fused@$R --run python-threadpool16-100=python-threadpool16@$R \
    --run rust-1thread-100=rust:1@$R --run rust-16thread-100=rust:16@$R

# the tables
pixi run -e dev python benchmarks/rust_spike/summarise_spike.py docs/benchmark-output/rust_spike_ogs00009.json \
    > docs/benchmark-output/rust_spike_ogs00009_tables.md

# the baseline on the earlier artifact's selection
pixi run -e dev python benchmarks/rust_spike/run_spike.py \
    --selection docs/benchmark-output/rust_spike_selection_2026-10-05_artifact.json \
    --output docs/benchmark-output/rust_spike_ogs00009_old_selection.json --base-store $B \
    --rust-binary $S/target/release/opengwasdb-rust-spike --zarrs-python $S --scratch $S \
    --rounds 5 --warm 3 --max-start-load 25 --run python-fused-1000=python-fused@$B
```
