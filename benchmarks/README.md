# Benchmarks

Each script builds (or reuses) a store, runs timed queries, and writes its results
to `docs/benchmark-output/` as a JSON file.  The comparison Quarto document
(`docs/benchmark-output/opengwasdb_vs_besdq_comparison.qmd`) reads those JSON
files at render time — re-running a script and re-rendering the QMD is all that
is needed to reproduce or update the report.

---

## Scripts

### `benchmark_reader_projection.py`

Measures the projection-aware tabular reader paths added in issue #179 against
the retained full-row parser that previously powered `stream_variants()`. For
one FinnGen R13 source and one GWAS-SSF source it records, under the same warm
operating-system cache condition:

1. decompression-only throughput;
2. projected `stream_variants()` throughput;
3. legacy full-row variant throughput;
4. `stream_associations()` throughput; and
5. peak RSS for each isolated workload.

The harness refuses to write an artifact if projected and legacy variant row
counts differ. The committed production measurement can be regenerated with:

```bash
pixi run -e dev python benchmarks/benchmark_reader_projection.py \
  --finngen /data/opengwasdb/raw/finngen-r13-10/finngen_R13_BMI_IRN.gz \
  --gwas-ssf /data/opengwasdb/raw/ebi-sun-pqtl-10/filtered/GCST90240120.filtered.tsv.gz \
  --repetitions 1 \
  --output docs/benchmark-output/opengwasdb_reader_projection_benchmark.json
```

Use more repetitions when reporting stable timing beyond the issue-179
acceptance measurement; each repetition performs both full-row scans of the
21.3-million-row FinnGen file and therefore takes several minutes.

---

### `benchmark_extract_variant_reference.py`

Synthetic scaling benchmark for the genomic-window map + tree-reduce
(issues #188 and #191). Every generated GWAS-SSF source carries the same
genome-wide panel -- thousands of positions spread across all chromosomes at a
near-uniform spacing -- so cross-file overlap is total and every worker
contributes a shard to every genomic window. That is the production shape: the
tree reduce cannot be skipped, and the `reduced` column below stays equal to the
window count. The run asserts every artifact is byte-identical to the first
before reporting any timing, so a fast wrong answer fails.

Map (per-source extraction), reduce (windowed tree-merge) and write (artifact
assembly) are timed and reported separately, alongside the end-to-end total and
speedup and the window/shard counts, so a regression in one phase cannot be
averaged away into the total.

```bash
pixi run -e dev python benchmarks/benchmark_extract_variant_reference.py \
  --n-files 128 --variants-per-file 4000 \
  --worker-counts 1 2 4 8 --window-sizes-mb 5 20 --reduction-batch-sizes 4 16 \
  --repetitions 3 --output /tmp/opengwasdb_extract_variant_reference_benchmark.json
```

**Current-implementation baseline** (128 sources × 4,002 genome-wide panel
positions = 4,002 union variants, `nproc=224` Intel Xeon Platinum 8480+, medians
of 3 repetitions, `speedup` against the median serial total). Re-run, not
hand-edited:

```
128 sources x 4002 genome-wide panel positions, 4002 unique variants
workers  window batch    total      map   reduce    write    other speedup  windows  shards  reduced
      1       5     4    0.966    0.909    0.000    0.045    0.012   1.00x        0       0        0
      1       5    16    0.970    0.922    0.000    0.045    0.003   0.99x        0       0        0
      1      20     4    0.964    0.916    0.000    0.044    0.004   1.00x        0       0        0
      1      20    16    0.960    0.911    0.000    0.045    0.004   1.01x        0       0        0
      2       5     4    1.238    0.956    0.085    0.071    0.125   0.78x      614    1228      614
      2       5    16    0.910    0.674    0.082    0.071    0.083   1.06x      614    1228      614
      2      20     4    0.788    0.658    0.029    0.070    0.031   1.23x      162     324      162
      2      20    16    0.793    0.660    0.031    0.070    0.031   1.22x      162     324      162
      4       5     4    0.937    0.664    0.087    0.071    0.116   1.03x      614    2456      614
      4       5    16    0.959    0.672    0.090    0.072    0.126   1.01x      614    2456      614
      4      20     4    0.776    0.629    0.031    0.071    0.046   1.24x      162     648      162
      4      20    16    0.775    0.635    0.029    0.071    0.039   1.25x      162     648      162
      8       5     4    1.000    0.462    0.236    0.077    0.225   0.97x      614    4912      614
      8       5    16    0.878    0.466    0.103    0.073    0.237   1.10x      614    4912      614
      8      20     4    0.569    0.352    0.075    0.072    0.070   1.70x      162    1296      162
      8      20    16    0.492    0.353    0.035    0.038    0.066   1.96x      162    1296      162
byte-identical artifact across all 16 configurations
```

The map phase dominates and parallelises to ~2.6x on 8 workers at 20 Mb windows;
the reduce stays small at this union size, while the 5 Mb grid shows the cost of
merging 614 windows. Later #190 work is judged against these numbers.

**Real-data verification** (2026-09-19, IEU compute node, `nproc=224`, real
sources under `/data/opengwasdb/raw/`). The synthetic grid above isolates the
reduce; this run checks the rewrite against real input. The manifest is 82
sources -- 64 UKB GWAS-VCF (hg19), 16 EBI GWAS-SSF (hg38) and 2 FinnGen R13
(hg38), 15.6 GB -- and every run calls `extract_variant_reference(manifest, out,
window_size_mb=20, reduction_batch_size=4, liftover_failure_threshold=1.0)`
with the worker count shown:

```
82 real sources (64 UKB hg19 + 16 EBI hg38 + 2 FinnGen hg38), 15.6 GB
workers    total      map   reduce    write   variants  source_keys  windows  shards  reduced  levels    RSS
      1   2232.3   1326.2    394.8    509.4   28,488,575  38,302,643      322    8779      321       3  1909 MB
     16    204.3    138.1     29.0     34.9   28,488,575  38,302,643      322   10514      321       3  1893 MB
speedup    10.9x     9.6x     13.6x     14.6x
```

The serial and 16-worker artifacts are byte-identical (decompressed sha256
`aa55d7003241a2d7c51e169c41f2cfe0a536eba5e042ba15a40ce491e1e192b8`), and the
reduction descends three levels at this union size, so it is not the
single-level no-op the small synthetic panels can hide.

Parent peak RSS is flat with respect to union size. All-hg38 EBI manifests at 16
workers, union grown ~9x, `ru_maxrss` of the parent process only (a fork-pool
worker's memory is not the parent's):

| EBI sources | union variants | parent peak RSS | reduce levels |
|---:|---:|---:|---:|
| 1 | 2,314,363 | 1893 MB | 0 |
| 4 | 8,786,033 | 1901 MB | 1 |
| 16 | 21,236,235 | 1911 MB | 2 |
| 32 | 21,328,262 | 1889 MB | 2 |

Artifact identity: on a 9-source real mixed manifest (6 UKB hg19 + 1 FinnGen
hg38 + 2 EBI hg38) the current streaming artifact and the pre-#190 materialising
writer's both carry 21,559,975 variants; the `alid`, `chromosome`, `position`,
`a1`, `a2` and `source_keys` columns are byte-identical. 1,229 of the 21,559,975
rows differ in the `rsid` column only: 1,226 are the deterministic `(rank,
site)` tie-break accumulated between the pre-#190 commit and the current writer
(#192, #194, #195), and 3 are the streaming path declining to take an rsid from
a pre-lift tuple that failed liftover but whose raw string coincides with a
valid hg38 tuple. No variant, coordinate, allele or source-key value differs.

Store identity (#185): a real 4-source manifest (2 UKB hg19 + 2 EBI hg38,
4,616,591 variants x 4 analyses) built once with `build-dense-vcf` and once with
`build-dense-vcf --variant-reference` from that manifest's extraction artifact
produces stores whose every file is byte-identical except `manifest.json`, and
that differs only in `created_at` and the provenance `builder` /
`variant_reference` fields.

---

### `benchmark_vcf_ukb_chr1_dense.py`

Builds and benchmarks a dense observed-only store from the 100 UKB chr1 GWAS-VCF
dataset.  Requires the VCFs at `/home/gh13047/repo/besdq/data/vcf-ukb/` and the
besdq repo at `/home/gh13047/repo/besdq/`.

**Output files written to `docs/benchmark-output/`:**

| File | Description |
|---|---|
| `opengwasdb_vcf_ukb_chr1_benchmark.json` | Post-optimisation query timings |
| `besdq_ukb_chr1_benchmark.json` | besdq baseline (copied from besdq repo) |
| `opengwasdb_vcf_ukb_chr1_benchmark.qmd` | Per-run standalone QMD |

**Usage** (run from the repo root — uses this repo's Pixi-managed `dev`
environment, no separate conda env required):

```bash
# First run — build the store and benchmark (takes ~10 min)
pixi run -e dev python benchmarks/benchmark_vcf_ukb_chr1_dense.py --rebuild --reps 10

# Subsequent runs — reuse existing store, re-benchmark only
pixi run -e dev python benchmarks/benchmark_vcf_ukb_chr1_dense.py --reps 10
```

The `--row-baseline` flag accepts a path to an earlier JSON to show speedup ratios:

```bash
python benchmarks/benchmark_vcf_ukb_chr1_dense.py \
    --row-baseline docs/benchmark-output/opengwasdb_vcf_ukb_chr1_array_benchmark.json \
    --reps 10
```

---

### `benchmark_vcf_ukb_chr1_1000_dense.py`

Builds and benchmarks dense observed-only stores from the larger UKB chr1
GWAS-VCF manifest. The `--analysis-count` flag selects the first N analyses from
the source manifest, so the same script can produce the 128-analysis and
1000-analysis comparison JSONs.

**Output files written to `docs/benchmark-output/`:**

| File | Description |
|---|---|
| `opengwasdb_vcf_ukb_chr1_128_benchmark.json` | 128-analysis scaling benchmark |
| `opengwasdb_vcf_ukb_chr1_1000_benchmark.json` | 1000-analysis scaling benchmark |
| `opengwasdb_vcf_ukb_chr1_1000_benchmark.qmd` | Standalone 128 vs 1000 report |

**Usage**:

```bash
pixi run -e dev python benchmarks/benchmark_vcf_ukb_chr1_1000_dense.py \
  --analysis-count 128 --rebuild --reps 10

pixi run -e dev python benchmarks/benchmark_vcf_ukb_chr1_1000_dense.py \
  --analysis-count 1000 --rebuild --reps 10
```

---

### `benchmark_ragged_besd.py`

Benchmarks a ragged observed-only store built from BESD files.  Six query
patterns are timed and a storage comparison against the source BESD is included.
Defaults to the pre-built eqtlgen-cis store.

**Output files written to `docs/benchmark-output/`:**

| File | Description |
|---|---|
| `opengwasdb_eqtlgen_ragged_benchmark.json` | Query timings + storage comparison |
| `opengwasdb_eqtlgen_ragged_benchmark.qmd` | Self-contained Quarto report (rendered to HTML) |

**Usage** (run from the repo root):

```bash
# Benchmark existing store (no rebuild)
pixi run -e dev python benchmarks/benchmark_ragged_besd.py --reps 5

# Force a full rebuild then benchmark
pixi run -e dev python benchmarks/benchmark_ragged_besd.py --rebuild --reps 5

# Use a different BESD source (e.g. hg19 with liftover)
pixi run -e dev python benchmarks/benchmark_ragged_besd.py \
    --besd /path/to/prefix \
    --store /path/to/out.opengwasdb \
    --source-build hg19 \
    --tissue Whole_Blood
```

**Render the QMD** (uses this repo's `report` pixi environment, which provides
`quarto` plus the Jupyter/matplotlib stack the reports need — see
`pyproject.toml`'s `[tool.pixi.feature.report]`):

```bash
cd docs/benchmark-output
pixi run -e report quarto render opengwasdb_eqtlgen_ragged_benchmark.qmd
```

**Query patterns:**

| Pattern | Description |
|---|---|
| `analysis` | All cis associations for one probe (O(1) CSR slice) |
| `range_by_probe` | All analyses whose TSS falls in a 2 Mb window |
| `range` | All associations where the variant falls in a 2 Mb window (O(n) scan) |
| `phewas` | One variant across all analyses (O(n) scan) |
| `tophits` | Top-10 hits by \|z\| from precomputed index |
| `random_lookup` | 100 random variants × 10 random analyses |

---

### `benchmark_ukbb_dense.py`

Benchmarks the genome-wide `ukb-b` Dense Store Release: query timings for the
bulk/phewas/regional/top-hits/random-lookup shapes, per-shape peak memory,
storage vs its source VCF, build time, and an MR IVW validation. Requires the
store tree, a build manifest and the source VCFs. The artifact records the
store's `format_version` and `encoding`, so a format-2.0 timing cannot be
mistaken for a format-3.0 one (issue #148).

The current release under measurement is **OGS-00009** (`ukb-b-full-observed`):
2,024 Analyses, 9,847,701 variants. The script's built-in `--store` default
points at an older 2,514-Analysis store that no longer exists, so pass the
paths explicitly as below.

The MR pair is `ukb-b-17805` (cholesterol lowering medication) ->
`ukb-b-1668` (ICD10 I25.1 atherosclerotic heart disease). The exposure is a
treatment proxy, so that estimate is confounded by indication and is a
pipeline correctness check, not a causal result; the `ukb-b` collection carries
no LDL or lipid biomarker to do better with.

`--skip-rss` drops the memory probes when only timings are wanted. Each probe
re-opens the store in a fresh interpreter, so they roughly double the run.

**Output files written to `docs/benchmark-output/`:**

| File | Description |
|---|---|
| `opengwasdb_ukbb_dense_benchmark.json` | Query timings + memory + storage + MR result |
| `opengwasdb_ogs00009_dense_benchmark.qmd` / `.html` | Rendered report (linked from `docs/index.html`) |

Note that `opengwasdb_ukbb_dense_benchmark.qmd` is a *different* report — the
format-2.0 vs 3.0 comparison for issue #148 — and reads the
`opengwasdb_ukbb_dense_issue148_*.json` artifacts, not this one.

**Usage** (run from the repo root, in the Pixi `dev` environment):

```bash
pixi run -e dev python benchmarks/benchmark_ukbb_dense.py \
    --reps 5 \
    --store /data/opengwasdb/stores/OGS-00009/store.opengwasdb \
    --manifest /data/opengwasdb/stores/OGS-00009/work/analyses.tsv \
    --build-seconds 21298 \
    --output docs/benchmark-output/opengwasdb_ukbb_dense_benchmark.json
```

`--manifest` takes the release's own `work/analyses.tsv`, whose
`analysis_id`/`source_file` columns the harness reads directly. `--build-seconds`
supplies the build wall clock for a release whose build time lives in its
`records/build.json` step record rather than in a parsable build log.

Render the report with the `report` environment:

```bash
cd docs/benchmark-output && pixi run -e report quarto render \
    opengwasdb_ogs00009_dense_benchmark.qmd --to html
```

The `--top-hits-experiment` mode re-measures top-hit index chunk sizes
against the same store and updates the existing JSON:

```bash
pixi run -e dev python benchmarks/benchmark_ukbb_dense.py --top-hits-experiment
```

Render the report with the `report` environment:

```bash
pixi run -e report quarto render docs/benchmark-output/opengwasdb_ukbb_dense_benchmark.qmd
```

---

### `benchmark_ogs00010_completed.py`

Benchmarks the Reference-Completed full-scale release **OGS-00010** (the
OGS-00009 `ukb-b` store completed against the EUR LD panel) and writes the
artifact the Reference-Completed showcase report renders. Covers:

1. the same query shapes as `benchmark_ukbb_dense.py`, driven identically
   (same Analysis, same PheWAS variant, same region, and the same random
   variant/analysis selections re-derived from the source release's recorded
   axis size and seed), plus the `observed_only=True` variants of the
   top-hits and bulk shapes;
2. per-shape baseline/peak RSS, same fresh-interpreter probe method;
3. the cell budget (observed / imputed / rejected / off-panel) from the
   release manifest's completion provenance and the indexed top-hit tiers;
4. imputation performance and quality: the `complete` step record
   (`records/complete.json` on the release tree) for wall clock, blocks and
   throughput, and the `completion_quality` table for the per-(Analysis,
   block) Pearson-r distribution;
5. three MR pairs (BMI -> CHD, the statin-use LDL proxy -> CHD, past tobacco
   smoking -> the C34.1 cancer-site trait), each on all variants, on
   `observed_only`, and on imputed-in-both sides, with per-instrument
   scatter data and a 2 Mb regional window around each exposure's strongest
   imputed hit carrying every axis variant's status;
6. a cross-release fidelity check: the statin-pair cells OGS-00009 reports
   must decode identically (z and se) and keep status `observed` inside
   OGS-00010, or the run fails loudly.

The per-shape RSS probe method (fresh interpreter, background peak sampler,
`/proc/self/statm`) lives in `benchmarks/_rss.py` and is shared with
`benchmark_ukbb_dense.py`, so the two reports measure memory the same way and
their numbers stay comparable.

**Output files written to `docs/benchmark-output/`:**

| File | Description |
|---|---|
| `opengwasdb_ogs00010_completed_benchmark.json` | The full measurement artifact |
| `opengwasdb_ogs00010_completed_benchmark.qmd` / `.html` | Rendered showcase report (linked from `docs/index.html`) |

**Usage** (run from the repo root, on the machine holding the stores):

```bash
pixi run -e dev python benchmarks/benchmark_ogs00010_completed.py \
    --reps 5 \
    --store /data/opengwasdb/stores/OGS-00010/store.opengwasdb \
    --source-store /data/opengwasdb/stores/OGS-00009/store.opengwasdb \
    --manifest /data/opengwasdb/stores/OGS-00009/work/analyses.tsv \
    --output docs/benchmark-output/opengwasdb_ogs00010_completed_benchmark.json

cd docs/benchmark-output && pixi run -e report quarto render \
    opengwasdb_ogs00010_completed_benchmark.qmd
```

All paths above are the script's built-in defaults. The report's prose is
computed from the artifact at render time (no hand-edited numbers); the
gate-rejected tail of the imputation-quality histogram is exported as
explicit bins plus `n_attempts_below_gate` so no attempt row can silently
drop out of the plot.

The script refuses to run against a stale reference: it re-measures the
source release's PheWAS count and aborts if it disagrees with the published
OGS-00009 artifact, and it aborts on any MR condition that resolves to zero
instruments. `--skip-rss` drops the memory probes for a timings-only run.

---

### `benchmark_store_comparison.py`

Compares several Store Releases holding **the same data in different physical
shapes** — the instrument epic #240 measures with. Later tickets convert one
source release into Zarr v3 sharded copies (`[1000, 1000]`, `[1000, 128]`,
`[1000, 64]` inner chunks) and compare them against the pre-upgrade baseline
this script records for **OGS-00009 on zarr 2.18**. For every labelled store it
records:

1. **footprint**: total file count, `du -sb` apparent bytes and
   `du -s --block-size=1` allocated bytes, plus a per-array breakdown of all
   three and of the array's **largest file** — the shard a 0.2.0 copy, host or
   HTTP range request moves (arrays are found by `.zarray` **or** `zarr.json`,
   so the same walker covers the v2 baseline and the v3 sharded copies); group
   metadata and the non-Zarr envelope (variant table, SQLite index, `.npy`
   indexes, manifest) are totalled separately. The walked totals are checked
   against `du`, so a footprint that disagrees with itself fails rather than
   being published;
2. **the seven query shapes** of the OGS-00009/OGS-00016 reports, through the
   shared `benchmarks/_query_shapes.py`: median and p95 time over `--reps`
   after one warm-up, result count, and peak RSS from the fresh-interpreter
   probe (`benchmarks/_rss.py`);
3. **the identity check**: every store must return IDENTICAL results for every
   shape — the same six arrays, in the same order, with the same dtype and
   bit-equal values (NaN compared by position, not payload). Each array is
   hashed (`sha256` of dtype, shape, order and values) rather than held. A
   mismatch exits non-zero, naming the shape, the stores and the differing
   arrays, and writes no artifact; results are never sorted or normalised to
   make them agree.

The **selection** — the statin-use exposure `ukb-b-17805`, the chr19
APOE/APOC region, the PheWAS variant taken from the exposure's strongest
genome-wide hit, and the seeded random variant/Analysis draws — is resolved
ONCE against the first store and applied unchanged to every store, and is
recorded in the artifact. `--skip-rss` drops the per-shape RSS probes (roughly
halving the run); each probe re-invokes the script in a fresh interpreter with
the same selection passed on the command line.

The environment block records the zarr/numcodecs/numpy/python versions, the
installed `opengwasdb` commit, host, `nproc`, the 1-minute load average before
and after each store's run, the cache condition, and the **effective reader
configuration read back from the process** (`use_threads`, the real pipeline
class on an opened plane, `codec_pipeline.max_workers`). The top level also
carries #253's `opengwasdb_path` and `opengwasdb_fingerprint`, so an artifact
cannot claim a revision it did not import. Each store's `dataset.layout`
records its planes' inner chunk and shard, read back from the arrays. The node
holds about 1 TB of page cache and a cold cache cannot be forced without root,
so the artifact states plainly that the numbers are warm-cache numbers
(`environment.cache`).

**Output files written to `docs/benchmark-output/`:**

| File | Description |
|---|---|
| `opengwasdb_store_comparison_ogs00009_zarr2.json` | The committed OGS-00009 zarr-2.18 baseline and #240's before/after reference |
| `opengwasdb_store_comparison_ogs00009_zarr3.json` | OGS-00009 on zarr-python 3 at `83b8b23`, #244's Stage B run (`--reps 5`, peak memory) |
| `opengwasdb_store_comparison_ogs00009_zarr2_stage_b_pair.json` | The zarr 2.18 run at `745796c` made back to back with it, which the whole-Analysis guard compares against |
| `opengwasdb_store_comparison_ogs00009_zarr3_set_l.md` | The pair judged against set L, generated by `zarr3_lever_tables.py stage-b` |
| `opengwasdb_246_shapes/opengwasdb_store_comparison_ogs00009_zarr2.json` | #246's fresh zarr 2.18 run of `v2-c1000`, the pair the whole-Analysis guard uses |
| `opengwasdb_246_shapes/opengwasdb_store_comparison_ogs00009_shapes.json` | #246's zarr 3 run: `v2-c1000`, `v3-c1000`, `v3-c128`, `v3-c64` and `v3-c64-topshard1` in one process, so the identity check spans every configuration |
| `opengwasdb_246_shapes/opengwasdb_store_comparison_ogs00009_shapes_set_l.md` | Every configuration judged against set L and the whole-Analysis guard, generated by `zarr3_lever_tables.py shapes` |
| `opengwasdb_246_shapes/opengwasdb_ogs00009_shapes.qmd` / `.html` | The rendered #246 report: footprint, conversion cost, set L, whole-Analysis memory, shard sizes, builder memory, screening and the top-hit comparison |

**Usage** (run from the repo root, on the machine holding the stores):

```bash
# The committed baseline: OGS-00009 as it is today, read with zarr 2.18.
pixi run -e dev python benchmarks/benchmark_store_comparison.py \
    --store v2-c1000=/data/opengwasdb/stores/OGS-00009/store.opengwasdb \
    --reps 5 \
    --output docs/benchmark-output/opengwasdb_store_comparison_ogs00009_zarr2.json

# #246: the 2.18 pair first, from its own (zarr 2.18) environment, then every
# zarr 3 configuration in one process so the identity check spans them all.
(cd /tmp/epic240/244/base-src && pixi run -e dev python \
    benchmarks/benchmark_store_comparison.py \
    --store v2-c1000=/data/opengwasdb/stores/OGS-00009/store.opengwasdb \
    --reps 5 --output /tmp/epic240/246/zarr2.json)
pixi run -e dev python benchmarks/benchmark_store_comparison.py \
    --store v2-c1000=/data/opengwasdb/stores/OGS-00009/store.opengwasdb \
    --store v3-c1000=/data/opengwasdb/work/epic240/246/OGS-00009-v3-c1000 \
    --store v3-c128=/data/opengwasdb/work/epic240/246/OGS-00009-v3-c128 \
    --store v3-c64=/data/opengwasdb/work/epic240/245/OGS-00009-v3-c64 \
    --reps 5 --output /tmp/epic240/246/shapes.json
pixi run -e dev python benchmarks/zarr3_lever_tables.py shapes \
    --base /tmp/epic240/246/zarr2.json --head /tmp/epic240/246/shapes.json
```

Every store but the first is checked against the first, so the first store is
the reference for both the selection and the identity check. The script
refuses a duplicate label, a missing store directory, an exposure Analysis
with no genome-wide hits, or a partial random selection.

---

### `benchmark_ogs00011_hybrid.py` (#252)

Benchmarks the GWAS-Catalog-EUR Hybrid Store Release **OGS-00011**: the seven
shared query shapes plus an off-axis PheWAS and the largest-Overflow bulk read,
five known-locus checks, two pleiotropic PheWAS and three IVW Mendelian
randomisation pairs. It runs the same fresh-interpreter RSS probe as
`benchmark_ukbb_dense.py` and `benchmark_finngen_dense.py`.

**This script is not committed here.** It lives untracked on the
`bench/ogs00011-hybrid` worktree (it was never committed there, and no PR
carries it), and it shares several report helpers with
`benchmark_finngen_dense.py` -- the duplication gate fails if it is added
as-is. #252's before/after therefore ran it by absolute path from that
worktree, with `PYTHONPATH` set to the tree under measurement so `opengwasdb`
and `benchmarks` came from that tree, and `commit()`/`opengwasdb_fingerprint`
recorded which code ran. `benchmarks/measure_hybrid_component_split.py` is the
same story.

**Artifacts written for #252:**

| File | Description |
|---|---|
| `opengwasdb_ogs00011_hybrid_252_ab.json` | Interleaved before/after: each shape run on the branch base `f168ef1` and then on the fix's HEAD, back to back, with elapsed, peak RSS, the 1/5/15-minute loads at each run's start and end, a sha256 of the answer, and per-shape time limits. Generated by the harness plus a thin interleaving probe; every shape's before and after hash is equal. **Not a quiet-node measurement.** |
| `opengwasdb_252_spot_identity.json` | The same public query surface run against the two trees on OGS-00001, OGS-00006 and OGS-00004, with per-query sha256 and row counts, and each tree's commit and `opengwasdb` fingerprint. |

---

### `benchmark_finngen_dense.py`

Benchmarks the full-scale FinnGen R13 Dense Store Release **OGS-00016**. It
measures the same seven query shapes on the same fresh-interpreter RSS probe as
`benchmark_ukbb_dense.py` (`benchmarks/_query_shapes.py` holds the shared shape
construction and probe contract, issue #241), so the two full-scale reports are
comparable, and adds the checks a FinnGen release can be held to that a UK
Biobank one cannot:

1. storage against the source `.gz` files, by store component, plus the build's
   own step records (build / top-hits / overview / validate);
2. the bulk shape against the shape the source is laid out for, a single
   `pd.read_csv` of the same Analysis's file;
3. known-locus checks: the lead variants of well-established associations must
   be recovered with the published risk allele and genome-wide significance;
4. a PheWAS of two pleiotropic variants across all Analyses; and
5. three IVW Mendelian randomisation pairs run end to end through the store.

Cell-by-cell agreement with the source files is a separate, heavier harness:
`validate_finngen_source_fidelity.py` below. The benchmark refuses to report a
compression ratio when any source file is missing, rather than dividing by a
partial total.

**Output files written to `docs/benchmark-output/`:**

| File | Description |
|---|---|
| `opengwasdb_ogs00016_finngen_benchmark.json` | Query timings + memory + storage + known loci + MR |
| `opengwasdb_ogs00016_finngen_benchmark.qmd` / `.html` | Rendered report. **Not** linked from `docs/index.html`: #237 holds the `[1000, 1000]` report back until the rechunked store is benchmarked. |

**Usage** (run from the repo root, on the machine holding the store):

```bash
pixi run -e dev python benchmarks/benchmark_finngen_dense.py \
    --reps 5 \
    --store /data/opengwasdb/stores/OGS-00016/store.opengwasdb \
    --source-dir /data/opengwasdb/raw/finngen-r13/releases/r13-full/source \
    --records /data/opengwasdb/stores/OGS-00016/records \
    --output docs/benchmark-output/opengwasdb_ogs00016_finngen_benchmark.json
```

`--skip-rss` drops the memory probes for a timings-only run. One Analysis's
bulk shape on OGS-00016 takes roughly a minute and peaks around 23 GB RSS, so a
full run is long and is usually started in the background with its log in
`/tmp`.

---

### `validate_finngen_source_fidelity.py`

Compares a FinnGen R13 Dense Store Release cell by cell against the source files
it was built from. `opengwasdb validate` checks a store's internal consistency;
it never opens a source file, so this harness closes that loop: for each sampled
Analysis it parses the whole source `.gz`, matches every row to the store's
variant axis with its own allele-pair matcher (deliberately not the builder's),
decodes the store's cells through the public query API, and compares `z` against
`beta / sebeta`, `se` against `sebeta`, and `eaf` against `af_alt` (or
`1 - af_alt` when the orientation is flipped). It reports source rows with no
cell, store cells with no source row, and disagreements about missingness.

The gates come from the encoding the store declares in its manifest (ADR 0037):
`z` within half a step of its int16 fixed-point scale, `se` within the ADR's 1%
ordinary-cell bound, and `eaf` within the half-step of its int8 residual code. A
store that fails any gate exits non-zero after writing the record, so a partial
pass cannot be mistaken for a clean one.

The sample is fixed by the harness: six named anchors (the two
inverse-rank-normalised quantitative traits, the flagship endpoints, an endpoint
and its `_WIDE` twin, and the smallest case count), `--n-random` further random
Analyses drawn with `--seed`, or an explicit `--analyses` list. Each source's
SHA-256 is checked against the build manifest unless `--no-checksum` is passed.

**Output file written to `docs/benchmark-output/`:**

| File | Description |
|---|---|
| `opengwasdb_ogs00016_source_fidelity.json` | Per-Analysis cell counts, gate errors and verdicts |

**Usage** (run from the repo root, on the machine holding the store and sources):

```bash
pixi run -e dev python benchmarks/validate_finngen_source_fidelity.py \
    --store /data/opengwasdb/stores/OGS-00016/store.opengwasdb \
    --source-dir /data/opengwasdb/raw/finngen-r13/releases/r13-full/source \
    --manifest /data/opengwasdb/stores/OGS-00016/work/analyses.tsv \
    --output docs/benchmark-output/opengwasdb_ogs00016_source_fidelity.json
```

A shorter run is possible by narrowing the sample, e.g. `--analyses
finngen-r13-BMI_IRN` (or `--n-random 0`), which is the form used to check the
harness still runs after a change. The full sampled run reads one whole source
file per Analysis plus the 21-million-row variant axis, so it is slow and
memory-hungry.

---

### `benchmark_se_residual_queries.py`

Compares physical-SE query latency between the format-2.0 `float16` release
and its format-3.0 residual-`se` twin (ADR 0037 section 3, #118). Takes the
**before** store path and the **after** store path as positional arguments;
both must be real Store Releases with top-hit indexes, because the script
picks its Analysis, variant and region from the first store's own top hits.
The regeneration command for the current FinnGen R13 pilot evidence is
recorded in `docs/benchmark-output/opengwasdb_se_residual_implementation.md`
and reproduces:

```bash
pixi run -e dev python benchmarks/benchmark_se_residual_queries.py \
  /data/opengwasdb/wip/rebuild-117/finngen-r13__r13-pilot-20 \
  /data/opengwasdb/wip/rebuild-117/finngen-r13__r13-pilot-20-se3-benchmark \
  --repetitions 5 \
  --output docs/benchmark-output/opengwasdb_se_residual_implementation.json
```

The float16/after pair must be **copies of the same release** at the two
formats: the comparison is meaningless across different stores. Each store
record carries that store's `format_version` and `encoding`, so the JSON
cannot be mistaken about which format a latency column describes.

---

### `measure_pilot_releases.py`

Records, for each Store Release named on the command line, what the #117
pilot-rebuild evidence in ADR 0037 and the CHANGELOG was measured against:
per-store and per-component cell counts, the encoding plan each release
declares, compressed bytes for the whole release, each component and each
statistic plane, the standalone validation outcome, the source identity and
checksums the release itself records, and the measured commit and timestamp.
It is the repository-side harness the ADR's B/cell tables needed: a `du` of
the `eaf` plane divided by that plane's recorded `n_cells` reproduces the
"rebuilt-pilot plane B/cell" column, and a per-variant array such as
`eaf_reference` divided by its `n_cells` reproduces ADR 0037 section 4's
figures, without re-reading the stores.

**Output file written to `docs/benchmark-output/`:**

| File | Description |
|---|---|
| `opengwasdb_pilot_rebuild_measurements.json` | One record per measured store |

**Usage** — regenerate the committed artifact from the rebuilt pilot stores
(must be run where the stores live, e.g. the IEU compute node's
`/data/opengwasdb/wip/rebuild-117/`):

```bash
pixi run -e dev python benchmarks/measure_pilot_releases.py \
  /data/opengwasdb/wip/rebuild-117/finngen-r13__r13-pilot-20 \
  /data/opengwasdb/wip/rebuild-117/eqtlgen-cis-pilot__pilot-10 \
  /data/opengwasdb/wip/rebuild-117/eqtlgen-cis-pilot__pilot-10-completed \
  /data/opengwasdb/wip/rebuild-117/gwas-catalog-eur-hybrid__eur-hybrid-pilot-10 \
  /data/opengwasdb/wip/rebuild-117/gwas-catalog-eur-hybrid__eur-hybrid-quant-pilot-10 \
  /data/opengwasdb/wip/rebuild-117/metabolome-plasma-2023__2023-chen-full-european \
  /data/opengwasdb/wip/rebuild-117/pqtl-interval-2018__2018-sun-pilot-10 \
  --output docs/benchmark-output/opengwasdb_pilot_rebuild_measurements.json
```

A missing store, unreadable manifest or component without a plane to count
stops the run with a message naming the store, and nothing is written — the
driver refuses to publish a partial artifact. Validation failures are
recorded, not fatal: the format-2.0 #117 rebuilds predate later validator
rules (#127 truncated ALIDs, #135 unchunked per-variant planes), so an
artifact that silently dropped a store failing those rules would
misrepresent what was measured.

---

### #244: zarr-python 3's read levers, and the 0.2.0 shape screen

#244 moved the package to zarr-python 3.4. These scripts measured what that
cost, which runtime settings recovered it (ADR 0056), and whether the move is
safe for builds. They also screened chunk shapes for format 0.2.0 (#246), and
measured the duplicate EAF read (#253). Their outputs are committed under
`docs/benchmark-output/opengwasdb_zarr3_read_levers/`.
`PROVENANCE.md` in that directory says, for each output, which script made it,
which commit it measured and when. Most were measured once at #244 and not
re-run.

`_zarr3_levers.py` holds what the scripts share: the pipeline labels, the order
the shapes were timed in, and the readers for the attribution and harness
outputs. The shapes themselves are `_query_shapes.common_query_patterns`.

| Script | Measures | Output (in the directory above) |
|---|---|---|
| `zarr3_attribution.py` | the query shapes per zarr configuration, each in a fresh process under the checkout and environment it measures, configurations interleaved per round | `attribution/attribution.jsonl`, `decide_mw.out` |
| `zarr3_lever_tables.py` | the tables #244, #240, #246 and #253 quote, from those outputs, plus `decision` -- #246's configurations, conversion costs, set-L medians, whole-Analysis guard and memory, builder memory, files and shard sizes, the s256 difference, the top-hit A/B and the slice screen, all from the committed `docs/benchmark-output/opengwasdb_246_shapes/` artifacts -- for the ADR and the issue comments to paste verbatim | stdout |
| `top_hit_shard_ab.py` | #246's top-hit index, sharded against effectively unsharded: an interleaved A/B (both stores in one process, sides alternating sample by sample, every sample kept, result digests compared) | `docs/benchmark-output/opengwasdb_246_shapes/opengwasdb_top_hit_shard_ab.json` |
| `zarr3_blosc_decode.py` | one real `[1000, 1000]` chunk's decode time, Blosc threads off and on | `blosc_decode_{1,2}.json` |
| `zarr3_fork_probe.py` | whether forked workers finish their read under each lever | `fork_probe.out` |
| `zarr3_pool_fork_repro.py` | the standalone reproducer for zarr-developers/zarr-python#4478 | `repro_pool_fork_min.out` |
| `zarr3_fork_paths.py` | every fork-pool build path at `n_workers=2`, after the parent used the levers | `fork_paths-*.log` |
| `zarr3_fixture_trees.py` | one fixture store per builder path, with a given checkout's code | (trees, not committed) |
| `zarr3_compare_trees.py` | two such trees: chunk bytes apart from metadata, and whether differing chunks decode equal | `compare-split-*.txt`, `chunk-diff-decode-fused-default.json` |
| `zarr3_encode_scope.py` | compressed tile sizes and band-write time per lever, on real OGS-00009 | `encode_scope.out` |
| `zarr3_se_plan.py` | the SE encoding plan chosen on a real OGS-00009 slice, per lever and worker count | `se_plan_check*.out` |
| `zarr3_spot_queries.py` | spot queries with every returned array hashed, and the comparison of two records | `spot-*.json`, `spot-step1-compare.txt` |
| `shape_slice.py` | a 100,000-variant slice of OGS-00009 `z` in candidate shapes; its raw reads; per-plane decode times | `slice/slice_read_*.jsonl`, `slice/round1/`, `slice/decode_by_plane.jsonl`, `slice_build.out` |
| `shape_harness_geometry.py` | the arrays and chunks each harness shape reads, re-expressed per candidate chunk | `slice/harness_geometry.json` |
| `shape_screen.py` | the read cost model, the shape screen, the expected 0.2.0 times, the model's rank check, the slice table | `slice/{cost_model,screen,expected_020,screen_rank_check}.{json,out}` |
| `eaf_read_split.py` | each query's time inside the EAF reads, split by caller (#253) | `eaf/eaf_split_{1,2}.json` |
| `eaf_semantics_check.py` | which EAF residual SE decodes against on a Reference-Completed release (#253) | `eaf/eaf_semantics_check.json` |

The #242 harness made the paired 2.18 / zarr 3 runs (`stage-a*`) and the
peak-memory runs (`rss-*`), each environment running its own checkout's copy:

```bash
(cd /path/to/base && pixi run -e dev python benchmarks/benchmark_store_comparison.py \
    --store zarr2=/data/opengwasdb/stores/OGS-00009/store.opengwasdb \
    --reps 3 --skip-rss --output /tmp/stage-a-step1-base.json)
pixi run -e dev python benchmarks/benchmark_store_comparison.py \
    --store zarr3=/data/opengwasdb/stores/OGS-00009/store.opengwasdb \
    --reps 3 --skip-rss --output /tmp/stage-a-step1-head.json
# Peak memory: the same with --reps 1 and without --skip-rss.
```

**Stage B**, #244's acceptance run, is the same harness as a back-to-back pair:
zarr 2.18 first, then zarr 3. Each run uses `--reps 5` with the peak-memory
probes, and starts only when the 1-minute load is below 3. A committed script,
not a hand, judges the pair against set L and the whole-Analysis guard:

```bash
(cd /path/to/base && PYTHONNOUSERSITE=1 pixi run -e dev python \
    benchmarks/benchmark_store_comparison.py --store zarr2=$OGS9 --reps 5 \
    --output /tmp/stage-b-base.json)
PYTHONNOUSERSITE=1 pixi run -e dev python benchmarks/benchmark_store_comparison.py \
    --store zarr3=$OGS9 --reps 5 --output /tmp/stage-b-head.json
pixi run -e dev python benchmarks/zarr3_lever_tables.py stage-b \
    --pair /tmp/stage-b-base.json /tmp/stage-b-head.json
```

A result within 30% of a limit, either side, makes the verdict "NOT YET" until a
second pair is given with another `--pair`. The whole-Analysis guard counts as a
limit for this. The committed pair is in the table above.

**Usage.** OGS-00009 is `/data/opengwasdb/stores/OGS-00009/store.opengwasdb` on
the IEU compute node. The query selection is the one the committed #242
baseline recorded. Start a timing run only when the 1-minute load is below 3.

```bash
# The attribution: five configurations, three rounds. The checkouts are the
# zarr 2.18 base (745796c, with its own environment), 708d179 from before the
# levers, and the head.
pixi run -e dev python benchmarks/zarr3_attribution.py \
    --store $OGS9 \
    --selection docs/benchmark-output/opengwasdb_store_comparison_ogs00009_zarr2.json \
    --config base_218=default:$BASE:$BASE/.pixi/envs/dev/bin/python \
    --config i_none=default:$C708:$HEAD_PY \
    --config ii_bt=default+bt:$C708:$HEAD_PY \
    --config iii_bt_fused=fused_mw1+bt:$C708:$HEAD_PY \
    --config iv_head=asis:.:$HEAD_PY \
    --rounds 3 --reps 7 --with-bulk --output-dir /tmp/attribution
pixi run -e dev python benchmarks/zarr3_lever_tables.py attribution   # or pairs, memory, ...

# Fork safety. A hang is the failure looked for: the probe prints HANG and exits 3.
pixi run -e dev python benchmarks/zarr3_fork_probe.py fused_mw1+bt --scratch /tmp/probe
pixi run -e dev python benchmarks/zarr3_pool_fork_repro.py --max-workers 1
timeout 600 pixi run -e dev python benchmarks/zarr3_fork_paths.py --out /tmp/fork-paths \
    --store $OGS9

# Build output: build the trees under each checkout with BLOSC_NTHREADS=1, then compare.
BLOSC_NTHREADS=1 pixi run -e dev python benchmarks/zarr3_fixture_trees.py --repo . \
    --out /tmp/trees-head
pixi run -e dev python benchmarks/zarr3_compare_trees.py split /tmp/trees-base /tmp/trees-head
pixi run -e dev python benchmarks/zarr3_encode_scope.py fused_mw1+bt --store $OGS9 \
    --scratch /tmp/enc
pixi run -e dev python benchmarks/zarr3_se_plan.py asis 4 --store $OGS9 --scratch /tmp/se

# The 0.2.0 screen: build the slice, read it (shape_slice.py's docstring has the
# three-round loop), then run the analysis steps in order.
pixi run -e dev python benchmarks/shape_slice.py build --source $OGS9 --out /tmp/slice
pixi run -e dev python benchmarks/shape_slice.py decode --source $OGS9 >> decode_by_plane.jsonl
pixi run -e dev python benchmarks/shape_harness_geometry.py --store $OGS9 \
    --output harness_geometry.json
S="pixi run -e dev python benchmarks/shape_screen.py"   # each step reads the one before
$S cost-model --outputs $OUT --json $OUT/slice/cost_model.json
$S screen --outputs $OUT --json $OUT/slice/screen.json
$S expected --outputs $OUT --json $OUT/slice/expected_020.json
$S rank-check --outputs $OUT --json $OUT/slice/screen_rank_check.json
```

The analysis scripts (`zarr3_lever_tables.py`, `shape_screen.py`) read the
committed directory by default, so they reproduce the committed tables and
JSON without a store. Every table #244, #246 and #253 quotes is byte-identical
to their output. `shape_screen.py cost-model` matches the committed
`cost_model.json` except in the last digits of its floats, and so does the
original script re-run: the least-squares fit is not bit-reproducible run to
run.

**#246's screening beyond the four converted shapes.** The cost model already
enumerates `[1000, 256]` and `[500, 256]` (`shape_harness_geometry.py`'s
`CANDIDATES`), so `screen.json` carries them; `shape_slice.py` now builds and
reads them too (`v3_r1000c256_s`, `v3_r500c256_s`). To keep #244's committed
round files intact, that read is a separate file each `shape_screen.py` step
picks up if present:

```bash
OUT=/tmp/epic240/246/levers           # a copy of the committed levers directory
pixi run -e dev python benchmarks/shape_slice.py build --source $OGS9 --out /tmp/epic240/246/slice
pixi run -e dev python benchmarks/shape_slice.py read /tmp/epic240/246/slice step1 \
  v2_c1000,v3_c1000_s,v3_c128_s,v3_c64_s,v3_r2000c128_s,v3_r4000c256_s,v3_r250c512_s,\
  v3_r1000c256_s,v3_r500c256_s > $OUT/slice/slice_read_step1_246.jsonl
S="pixi run -e dev python benchmarks/shape_screen.py"
$S rank-check --outputs $OUT --json $OUT/slice/screen_rank_check.json
```

The #242 harness needs one more piece of care: `v3-c64-topshard1` is the same
conversion as `v3-c64` with `--top-hit-shard-chunks 1`, so the artifact's
`dataset.layout` says which top-hit shard each store holds.

---

## Comparison document

After all JSONs are present in `docs/benchmark-output/`, render the comparison report:

```bash
pixi run -e report quarto render docs/benchmark-output/opengwasdb_vs_besdq_comparison.qmd
```

The rendered HTML is written to the same directory.

---

## JSON schema

Every artifact written through `benchmarks/_artifact.py` —
`measure_pilot_releases.py`, `measure_top_hit_rebuild.py`,
`benchmark_se_residual_queries.py` and `benchmark_ukbb_dense.py` — records
its own provenance at the top level: `commit` (the short SHA the measurement
was taken at) and `measured_at` (UTC ISO-8601). An artifact without them, or
whose `commit` is older than the numbers beside it, is stale evidence.

All opengwasdb benchmark JSONs share a common top-level structure:

```json
{
  "dataset":  { "n_variants": int, "n_analyses": int },
  "build":    { "store_path": str, "build_seconds": float|null, "liftover_failure_count": int },
  "storage":  { "store_bytes": int, "store_mb": float },
  "selection": { ... query parameters used ... },
  "timings": [
    {
      "query": str,
      "median_ms": float,
      "p95_ms": float,
      "result_count": int,
      "besdq_zstd_median_ms": float,   // present if besdq baseline loaded
      "ratio_vs_besdq": float,         // present if besdq baseline loaded
      "row_api_median_ms": float,      // present if --row-baseline supplied
      "speedup_vs_row_api": float,     // present if --row-baseline supplied
      "notes": str                     // present if ratio > 2×
    }
  ]
}
```

The besdq baseline JSON (`besdq_ukb_chr1_benchmark.json`) uses a different
structure produced by `besdq/scripts/dense_05_query_benchmark.py`:

```json
{
  "zstd_bitshuffle": {
    "regional":      { "median_ms": float, ... },
    "phewas":        { "median_ms": float, ... },
    ...
  },
  "raw_float16": { ... }
}
```
