# Benchmarks

Each script builds (or reuses) a store, runs timed queries, and writes its results
to `docs/benchmark-output/` as a JSON file.  The comparison Quarto document
(`docs/benchmark-output/opengwasdb_vs_besdq_comparison.qmd`) reads those JSON
files at render time — re-running a script and re-rendering the QMD is all that
is needed to reproduce or update the report.

---

## Scripts

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
bulk/phewas/regional/top-hits/random-lookup shapes, storage vs its source
VCF, build time from `data/ukb-b/build.log`, and an MR IVW validation
(self-reported high cholesterol -> heart attack). Requires the `ukb-b` store
tree, its build manifest and its source VCFs — on the IEU compute node at
the defaults below, or pass `--store`, `--manifest` and `--build-log` to
point at a copy. The artifact records the store's `format_version` and
`encoding`, so a format-2.0 timing cannot be mistaken for a format-3.0 one
(issue #148).

**Output files written to `docs/benchmark-output/`:**

| File | Description |
|---|---|
| `opengwasdb_ukbb_dense_benchmark.json` | Query timings + storage + MR result |
| `opengwasdb_ukbb_dense_benchmark.qmd` / `.html` | Rendered report |

**Usage** (run from the repo root, in the Pixi `dev` environment):

```bash
pixi run -e dev python benchmarks/benchmark_ukbb_dense.py --reps 5
```

Optional arguments mirror the defaults in the script header:

```bash
pixi run -e dev python benchmarks/benchmark_ukbb_dense.py \
    --reps 5 \
    --store /local-scratch/data/opengwas/opengwasdb/ukb-b.opengwasdb \
    --output docs/benchmark-output/opengwasdb_ukbb_dense_benchmark.json
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
