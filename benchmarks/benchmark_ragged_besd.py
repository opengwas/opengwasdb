#!/usr/bin/env python3
"""Build and benchmark a Ragged Observed-Only Store from BESD files.

Benchmarks six query patterns against a ragged store and writes results to
docs/benchmark-output/ as JSON + QMD.

Usage:
  # Benchmark existing store (default: eqtlgen-cis)
  conda run -n snakemake python benchmarks/benchmark_ragged_besd.py --reps 5

  # Force a full rebuild then benchmark
  conda run -n snakemake python benchmarks/benchmark_ragged_besd.py --rebuild --reps 5

  # Use a different store / source
  conda run -n snakemake python benchmarks/benchmark_ragged_besd.py \\
      --besd /path/to/prefix \\
      --store /path/to/out.opengwasdb \\
      --source-build hg38
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import zarr

from opengwasdb.layouts.ragged.build_besd import build_ragged_from_besd
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRReader
from opengwasdb.query import query_store
from opengwasdb.validation import validate_store
from opengwasdb.variants import VariantAxis

DEFAULT_BESD = Path("/local-scratch/data/hg38/eqtlgen/cis")
DEFAULT_STORE = Path("/local-scratch/data/opengwas/opengwasdb/eqtlgen-cis.opengwasdb")
DEFAULT_OUTPUT = Path("docs/benchmark-output/opengwasdb_eqtlgen_ragged_benchmark.json")
DEFAULT_QMD = Path("docs/benchmark-output/opengwasdb_eqtlgen_ragged_benchmark.qmd")

RNG = np.random.default_rng(42)


def main() -> None:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    build_seconds: float | None = None
    liftover_failure_count = 0
    source_bytes: int | None = None

    # Measure source BESD size if the files exist
    source_bytes = _measure_besd_bytes(args.besd)

    if args.rebuild and args.store.exists():
        shutil.rmtree(args.store)

    if args.rebuild or not args.store.exists():
        print(f"Building store from {args.besd}.{{besd,esi,epi}} ...")
        t0 = time.perf_counter()
        result = build_ragged_from_besd(
            args.besd,
            args.store,
            store_id=args.store.stem,
            release_id="ragged-observed-v1",
            tissue=args.tissue or None,
            source_build=args.source_build,
        )
        build_seconds = time.perf_counter() - t0
        print(
            f"Build complete in {build_seconds:.1f}s: "
            f"{result.n_variants:,} variants × {result.n_analyses:,} analyses "
            f"× {result.n_associations:,} associations"
        )

    print("Validating store ...")
    validation = validate_store(args.store)
    if not validation.ok:
        raise SystemExit(f"Store is invalid: {validation.errors}")
    print("Validation passed.")

    print(f"Running benchmark ({args.reps} reps per pattern) ...")
    results = _run_benchmark(
        args.store, args.reps, build_seconds, liftover_failure_count, source_bytes,
        max_start_load=args.max_start_load,
    )

    args.output.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Results written to {args.output}")

    _write_qmd(results, args.qmd, args.output, args.reps)
    print(f"QMD written to {args.qmd}")
    _print_summary(results)


def _measure_besd_bytes(prefix: Path) -> int | None:
    total = 0
    found = False
    for ext in (".besd", ".esi", ".epi"):
        p = Path(str(prefix) + ext)
        if p.exists():
            total += p.stat().st_size
            found = True
    return total if found else None


def _run_benchmark(
    store_path: Path,
    n_reps: int,
    build_seconds: float | None,
    liftover_failure_count: int,
    source_bytes: int | None,
    *,
    max_start_load: float = 3.0,
) -> dict:
    q = query_store(store_path)
    selection = _choose_queries(store_path, q)

    query_specs: dict[str, object] = {
        "analysis": lambda: q.analysis(selection["analysis_probe_id"]),
        "range_by_analysis": lambda: q.range_by_analysis(
            selection["region_chrom"], selection["region_start"], selection["region_end"]
        ),
        "range_phewas": lambda: q.range_phewas(
            selection["region_chrom"], selection["region_start"], selection["region_end"]
        ),
        "phewas": lambda: q.phewas(selection["phewas_alid"]),
        "tophits": lambda: q.top_hits(
            analysis_id=selection["top_hit_analysis_id"],
            threshold=selection["top_hit_threshold"],
            limit=10,
        ),
        "random_lookup": lambda: q.lookup(
            selection["random_alids"], selection["random_analysis_ids"]
        ),
    }

    timings = []
    for name, fn in query_specs.items():
        median_ms, p95_ms, result_count = _bench(fn, n_reps, max_start_load=max_start_load)
        timings.append(
            {
                "query": name,
                "median_ms": round(median_ms, 3),
                "p95_ms": round(p95_ms, 3),
                "result_count": result_count,
            }
        )
        print(f"  {name}: median={median_ms:.2f}ms  p95={p95_ms:.2f}ms  rows={result_count:,}")

    csr = RaggedCSRReader(store_path)
    n_associations = csr.n_associations
    n_analyses = len(q.analyses_table())

    va = VariantAxis(store_path)
    n_variants = va.n_variants

    store_bytes = sum(f.stat().st_size for f in store_path.rglob("*") if f.is_file())

    storage: dict = {
        "store_bytes": store_bytes,
        "store_mb": round(store_bytes / 1_000_000, 2),
    }
    if source_bytes is not None:
        storage["source_bytes"] = source_bytes
        storage["source_mb"] = round(source_bytes / 1_000_000, 2)
        storage["compression_ratio"] = round(source_bytes / store_bytes, 3)

    return {
        "provenance": _provenance(),
        "dataset": {
            "n_variants": n_variants,
            "n_analyses": n_analyses,
            "n_associations": n_associations,
        },
        "build": {
            "store_path": str(store_path),
            "build_seconds": round(build_seconds, 2) if build_seconds is not None else None,
            "liftover_failure_count": liftover_failure_count,
        },
        "storage": storage,
        "selection": selection,
        "timings": timings,
    }


def _provenance() -> dict:
    """The commit, time, harness hash and load this artifact was measured under."""
    import hashlib
    import os

    from benchmarks._artifact import provenance

    harness = Path(__file__).resolve()
    return {
        **provenance(),
        "harness": str(harness.name),
        "harness_sha256": hashlib.sha256(harness.read_bytes()).hexdigest(),
        "loadavg_at_write": list(os.getloadavg()),
    }


def _choose_queries(store_path: Path, q) -> dict:
    """Pick representative query parameters from the store content.

    Analysis metadata comes from `analyses.tsv` through the query facade -- the
    retired `index.sqlite` `analyses` table (ADR 0034) is not consulted.
    """
    # Probe with most associations — good for analysis query
    csr = RaggedCSRReader(store_path)
    offsets = csr._offsets[:]
    counts = np.diff(offsets.astype(np.int64))
    best_analysis_idx = int(np.argmax(counts))

    analyses = q.analyses_table()
    probe_row = analyses[best_analysis_idx]
    all_analyses = sorted(analyses.items())
    root = zarr.open_group(str(store_path / "data.zarr"), mode="r")
    hit_offsets = root["top_hits/p_5e_08/analysis_offsets"][:]
    top_hit_analysis_idx = int(np.argmax(np.diff(hit_offsets)))
    top_hit_row = analyses[top_hit_analysis_idx]

    analysis_probe_id = str(probe_row["analysis_id"])
    # Use probe chr/bp to define a 2 Mb region around it for range queries
    region_chrom = str(probe_row["trait_chr"])
    region_centre = int(probe_row["trait_bp"])
    region_start = max(1, region_centre - 1_000_000)
    region_end = region_centre + 1_000_000

    # Phewas: variant that appears in the most analyses
    vi_all = csr._variant_index[:]
    vi_counts = np.bincount(vi_all.astype(np.int64))
    phewas_vi = int(np.argmax(vi_counts))
    va = VariantAxis(store_path)
    all_variants = va.all()
    phewas_alid = str(all_variants[phewas_vi].alid)

    # Top-hit threshold — use 5e-8 (index always present)
    top_hit_threshold = 5e-8

    # Random lookup: 100 random variants × 10 random analyses
    n_random_variants = min(100, len(all_variants))
    n_random_analyses = min(10, len(all_analyses))
    random_vi = sorted(RNG.choice(len(all_variants), n_random_variants, replace=False).tolist())
    random_ai = sorted(RNG.choice(len(all_analyses), n_random_analyses, replace=False).tolist())

    return {
        "analysis_probe_id": analysis_probe_id,
        "region_chrom": region_chrom,
        "region_start": region_start,
        "region_end": region_end,
        "phewas_alid": phewas_alid,
        "top_hit_threshold": top_hit_threshold,
        "top_hit_analysis_id": str(top_hit_row["analysis_id"]),
        "random_alids": [str(all_variants[i].alid) for i in random_vi],
        "random_analysis_ids": [str(all_analyses[i][1]["analysis_id"]) for i in random_ai],
    }


def _bench(fn, n_reps: int, *, max_start_load: float) -> tuple[float, float, int]:
    """Median and p95 ms over `n_reps`, each repetition gated at a quiet load.

    A warm-up runs first.  Every timed repetition waits for the 1-minute load to
    fall below `max_start_load` (0 disables), so a contended node is recorded
    rather than silently timed.
    """
    from benchmarks._quiet import wait_for_quiet

    result = fn()  # warm-up
    times = []
    for _ in range(n_reps):
        wait_for_quiet(max_start_load)
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return (
        float(np.median(times) * 1000),
        float(np.percentile(times, 95) * 1000),
        int(len(result["z"])),
    )


def _print_summary(results: dict) -> None:
    print("\n--- Summary ---")
    d = results["dataset"]
    s = results["storage"]
    b = results["build"]
    print(f"  Variants:      {d['n_variants']:>15,}")
    print(f"  Analyses:      {d['n_analyses']:>15,}")
    print(f"  Associations:  {d['n_associations']:>15,}")
    if b["build_seconds"] is not None:
        print(f"  Build time:    {b['build_seconds']:>14.1f}s")
    print(f"  Store size:    {s['store_mb']:>13.1f} MB")
    if "source_mb" in s:
        print(f"  Source (BESD): {s['source_mb']:>13.1f} MB")
        print(f"  Ratio:         {s['compression_ratio']:>14.3f}×")
    print()
    print(f"  {'Query':<20} {'Median ms':>10} {'p95 ms':>10} {'Results':>10}")
    for row in results["timings"]:
        print(
            f"  {row['query']:<20} {row['median_ms']:>10.2f}"
            f" {row['p95_ms']:>10.2f} {row['result_count']:>10,}"
        )


def _write_qmd(
    results: dict,
    qmd_path: Path,
    json_path: Path,
    n_reps: int,
) -> None:
    json_rel = json_path.name
    text = f"""\
---
title: "eQTLgen cis Ragged Store — Query Benchmark"
subtitle: "Ragged CSR zarr layout · {results['dataset']['n_analyses']:,} analyses · {results['dataset']['n_associations']:,} associations"
date: today
format:
  html:
    toc: true
    embed-resources: true
execute:
  echo: false
---

```{{python}}
import json, pathlib
data = json.loads(pathlib.Path("{json_rel}").read_text())
d, b, s = data["dataset"], data["build"], data["storage"]
timings = data["timings"]
sel = data["selection"]
```

## Dataset

```{{python}}
from IPython.display import Markdown
rows = [
    ("Variants", f"{{d['n_variants']:,}}"),
    ("Analyses", f"{{d['n_analyses']:,}}"),
    ("Associations", f"{{d['n_associations']:,}}"),
    ("Store size", f"{{s['store_mb']:.1f}} MB"),
]
if "source_mb" in s:
    rows += [
        ("Source BESD size", f"{{s['source_mb']:.1f}} MB"),
        ("Size ratio (BESD / store)", f"{{s['compression_ratio']:.2f}}×"),
    ]
if b["build_seconds"]:
    rate = d["n_associations"] / b["build_seconds"]
    rows += [
        ("Build time", f"{{b['build_seconds']:.1f}} s"),
        ("Build rate", f"{{rate:,.0f}} assoc/s"),
    ]
rows.append(("Store path", f"`{{b['store_path']}}`"))
table = "| Field | Value |\\n|---|---:|\\n"
table += "\\n".join(f"| {{k}} | {{v}} |" for k, v in rows)
Markdown(table)
```

## Storage comparison

```{{python}}
if "source_mb" in s:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 2.5))
    labels = ["BESD (.besd+.esi+.epi)", "opengwasdb (ragged)"]
    values = [s["source_mb"], s["store_mb"]]
    bars = ax.barh(labels, values, color=["#aec6cf", "#4a90d9"])
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
                f"{{v:,.0f}} MB", va="center", fontsize=9)
    ax.set_xlabel("Size (MB)")
    ax.set_xlim(0, max(values) * 1.25)
    ax.set_title("File size comparison")
    plt.tight_layout()
    plt.show()
```

## Query timings

Each pattern ran {n_reps} repetitions after one warm-up call.

```{{python}}
header = "| Query | Median ms | p95 ms | Result count | Notes |"
sep    = "|---|---:|---:|---:|---|"
rows_md = []
for t in timings:
    rows_md.append(
        f"| {{t['query']}} | {{t['median_ms']:.1f}} | {{t['p95_ms']:.1f}}"
        f" | {{t['result_count']:,}} | {{t.get('notes', '')}} |"
    )
Markdown("\\n".join([header, sep] + rows_md))
```

```{{python}}
import matplotlib.pyplot as plt
import numpy as np

names = [t["query"] for t in timings]
medians = [t["median_ms"] for t in timings]
p95s = [t["p95_ms"] for t in timings]
errs = [p - m for m, p in zip(medians, p95s)]

fig, ax = plt.subplots(figsize=(7, 4))
y = np.arange(len(names))
bars = ax.barh(y, medians, xerr=errs, align="center",
               color="#4a90d9", ecolor="#333", capsize=4)
ax.set_yticks(y)
ax.set_yticklabels(names)
ax.set_xlabel("Time (ms) — median + p95 error bar")
ax.set_title("Query latency — ragged CSR store")
ax.set_xscale("log")
for bar, v in zip(bars, medians):
    ax.text(v * 1.05, bar.get_y() + bar.get_height()/2,
            f"{{v:.1f}}", va="center", fontsize=8)
plt.tight_layout()
plt.show()
```

## Query parameters

```{{python}}
param_rows = [
    ("Analysis probe", sel["analysis_probe_id"]),
    ("Region", f"{{sel['region_chrom']}}:{{sel['region_start']:,}}–{{sel['region_end']:,}}"),
    ("PheWAS variant", sel["phewas_alid"]),
    ("Top-hit threshold", str(sel["top_hit_threshold"])),
    ("Random lookup variants", str(len(sel["random_alids"]))),
    ("Random lookup analyses", str(len(sel["random_analysis_ids"]))),
]
table = "| Parameter | Value |\\n|---|---|\\n"
table += "\\n".join(f"| {{k}} | {{v}} |" for k, v in param_rows)
Markdown(table)
```
"""
    qmd_path.parent.mkdir(parents=True, exist_ok=True)
    qmd_path.write_text(text, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--besd", type=Path, default=DEFAULT_BESD,
                        help="BESD prefix (without extension)")
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE,
                        help="Output / existing store path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="JSON output path")
    parser.add_argument("--qmd", type=Path, default=DEFAULT_QMD,
                        help="QMD output path")
    parser.add_argument("--source-build", default="hg38",
                        help="Assembly of BESD input (hg38 or hg19)")
    parser.add_argument("--tissue", default=None,
                        help="Tissue label for analysis IDs")
    parser.add_argument("--rebuild", action="store_true", default=False,
                        help="Tear down and rebuild the store before benchmarking")
    parser.add_argument("--reps", type=int, default=5,
                        help="Query repetitions per pattern (default 5)")
    parser.add_argument("--max-start-load", type=float, default=3.0,
                        help="Wait for the 1-minute load to fall below this before "
                             "each timed repetition (0 disables; default 3)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
