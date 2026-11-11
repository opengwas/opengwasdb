"""Benchmark for ragged reference completion.

Measures:
  - Completion build time for the eqtlgen-cis store
  - File sizes (observed-only vs reference-completed)
  - Query timings for all access patterns on the completed store
  - Imputation yield (n_imputed / total_reference_panel_variants)

Usage:
    python benchmarks/benchmark_ragged_completion.py \\
        --observed /local-scratch/data/opengwas/opengwasdb/eqtlgen-cis.opengwasdb \\
        --completed /local-scratch/data/opengwas/opengwasdb/eqtlgen-cis-completed.opengwasdb \\
        --ld-panel /local-scratch/projects/genotype-phenotype-map/data/ld_reference_panel_hg38 \\
        --out benchmarks/results/completion_benchmark.json \\
        [--build]  # set to run completion (slow); omit to benchmark pre-built store
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import time
from pathlib import Path

import numpy as np

from opengwasdb.layouts.ragged.complete import complete_ragged_store
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRReader
from opengwasdb.model.analyses import read_analysis_records
from opengwasdb.query import query_store


def _dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def _measure_n(fn, n_reps: int = 5) -> dict:
    times = []
    result_size = None
    for _ in range(n_reps):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
        if isinstance(result, dict) and "z" in result:
            result_size = int(len(result["z"]))
    return {
        "mean_s": round(statistics.mean(times), 4),
        "median_s": round(statistics.median(times), 4),
        "p95_s": round(sorted(times)[int(0.95 * len(times))], 4),
        "n_reps": n_reps,
        "result_size": result_size,
    }


def _choose_queries(completed_path: Path) -> dict:
    """Select representative query parameters from the completed store."""
    csr = RaggedCSRReader(completed_path)
    analyses = sorted(
        read_analysis_records(completed_path / "analyses.tsv"),
        key=lambda a: int(a.analysis_index),
    )

    # The Analysis with the most associations, from the offsets alone: its
    # length needs no decode.
    best_ai = int(np.argmax(np.diff(csr.offsets)))

    rec = analyses[best_ai]

    trait_bp = int(rec.trait_bp) if rec.trait_bp else 1_000_000
    return {
        "analysis_id": rec.analysis_id,
        "cis_chrom": rec.trait_chr or None,
        "cis_start": max(1, trait_bp - 1_000_000),
        "cis_end": trait_bp + 1_000_000,
        "phewas_variant": None,  # will be set from the analysis result
    }


def _write_qmd(results: dict, out_qmd: Path) -> None:
    out_qmd.write_text(f"""\
---
title: "OpenGWASDB Ragged Reference Completion Benchmark"
format:
  html:
    embed-resources: true
    code-fold: true
execute:
  echo: false
jupyter: python3
---

```{{python}}
import json, pathlib
results = json.loads(pathlib.Path("{out_qmd.with_suffix('.json').name}").read_text())
```

## Dataset

```{{python}}
import pandas as pd
info = results["dataset"]
pd.DataFrame([
    ("Observed-only store", info["observed_path"]),
    ("Completed store",    info["completed_path"]),
    ("Analyses",           f"{{info['n_analyses']:,}}"),
    ("Variants (observed)", f"{{info['n_variants_observed']:,}}"),
    ("Variants (completed)", f"{{info['n_variants_completed']:,}}"),
    ("Associations (completed)", f"{{info['n_associations_completed']:,}}"),
    ("Imputed associations", f"{{info['n_imputed']:,}}"),
    ("Missing (ref panel not imputed)", f"{{info['n_missing']:,}}"),
], columns=["Property", "Value"]).set_index("Property")
```

## Storage

```{{python}}
import matplotlib.pyplot as plt

storage = results["storage"]
labels  = ["Observed-only", "Reference-completed"]
sizes   = [storage["observed_mb"], storage["completed_mb"]]

fig, ax = plt.subplots(figsize=(6, 3))
bars = ax.barh(labels, sizes, color=["#4c72b0", "#dd8452"])
ax.set_xlabel("Size (MB)")
ax.set_title("Store size comparison")
for bar, v in zip(bars, sizes):
    ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
            f"{{v:,.0f}} MB", va="center")
plt.tight_layout()
plt.show()
```

Expansion factor: **{{results["storage"]["expansion_factor"]:.2f}}×**

## Build Time

Completion build: **{{results["build"]["elapsed_s"]:.1f}} s**
({results["build"].get("note", "")})

## Query Timings (completed store)

```{{python}}
timings = results["query_timings"]
rows = []
for name, t in timings.items():
    rows.append({{
        "Query": name,
        "Mean (s)": t["mean_s"],
        "Median (s)": t["median_s"],
        "p95 (s)": t["p95_s"],
        "Result size": t["result_size"],
        "Reps": t["n_reps"],
    }})
df = pd.DataFrame(rows).set_index("Query")
df.style.format({{
    "Mean (s)": "{{:.4f}}",
    "Median (s)": "{{:.4f}}",
    "p95 (s)": "{{:.4f}}",
}})
```

```{{python}}
fig, ax = plt.subplots(figsize=(8, 4))
names = list(timings.keys())
means = [timings[n]["mean_s"] for n in names]
p95s  = [timings[n]["p95_s"]  for n in names]
yerr  = [p - m for m, p in zip(means, p95s)]
ax.bar(names, means, yerr=yerr, capsize=4, color="#4c72b0")
ax.set_yscale("log")
ax.set_ylabel("Time (s, log scale)")
ax.set_title("Query latency on reference-completed store")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.show()
```

## Query Parameters

```{{python}}
params = results["query_params"]
pd.DataFrame(params.items(), columns=["Parameter", "Value"]).set_index("Parameter")
```
""")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observed", required=True, help="Observed-only ragged store path")
    parser.add_argument("--completed", required=True, help="Completed store path (output or pre-built)")
    parser.add_argument("--ld-panel", required=True, help="LD reference panel root directory")
    parser.add_argument("--out", default="benchmarks/results/completion_benchmark.json")
    parser.add_argument("--build", action="store_true", help="Run completion build (slow)")
    parser.add_argument("--ancestry", default="EUR")
    parser.add_argument("--min-cor", type=float, default=0.7)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--regions", default=None,
        help="The source build's region table; required unless its coverage is cis (#430)",
    )
    args = parser.parse_args()

    observed_path = Path(args.observed)
    completed_path = Path(args.completed)
    out_json = Path(args.out)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    results: dict = {}

    # ── Build ─────────────────────────────────────────────────────────────────
    if args.build:
        print("Running reference completion...")
        t0 = time.time()
        cr = complete_ragged_store(
            observed_path, completed_path, args.ld_panel,
            ancestry=args.ancestry, min_cor=args.min_cor, overwrite=args.overwrite,
            regions=args.regions,
        )
        elapsed = time.time() - t0
        results["build"] = {
            "elapsed_s": round(elapsed, 1),
            "n_variants": cr.n_variants,
            "n_analyses": cr.n_analyses,
            "n_associations": cr.n_associations,
            "n_imputed": cr.n_imputed,
            "n_missing": cr.n_missing,
        }
        print(f"Build done in {elapsed:.1f} s")
    else:
        if not completed_path.exists():
            raise SystemExit(f"Completed store not found: {completed_path}. Run with --build first.")
        # Read stats from manifest
        import json as _json
        m = _json.loads((completed_path / "manifest.json").read_text())
        c = m.get("provenance", {}).get("completion", {})
        results["build"] = {
            "elapsed_s": None,
            "note": "Pre-built store; elapsed not measured",
            "n_imputed": c.get("n_imputed"),
            "n_missing": c.get("n_missing"),
        }

    # ── Dataset info ──────────────────────────────────────────────────────────
    csr_obs = RaggedCSRReader(observed_path)
    csr_comp = RaggedCSRReader(completed_path)
    from opengwasdb.variants.axis import VariantAxis
    va_obs = VariantAxis(observed_path)
    va_comp = VariantAxis(completed_path)
    import json as _json
    m_comp = _json.loads((completed_path / "manifest.json").read_text())
    c_info = m_comp.get("provenance", {}).get("completion", {})

    results["dataset"] = {
        "observed_path": str(observed_path),
        "completed_path": str(completed_path),
        "n_analyses": csr_obs.n_analyses,
        "n_variants_observed": va_obs.n_variants,
        "n_variants_completed": va_comp.n_variants,
        "n_associations_completed": csr_comp.n_associations,
        "n_imputed": c_info.get("n_imputed"),
        "n_missing": c_info.get("n_missing"),
    }
    va_obs.close()
    va_comp.close()

    # ── Storage ───────────────────────────────────────────────────────────────
    obs_mb = _dir_size_mb(observed_path)
    comp_mb = _dir_size_mb(completed_path)
    results["storage"] = {
        "observed_mb": round(obs_mb, 1),
        "completed_mb": round(comp_mb, 1),
        "expansion_factor": round(comp_mb / obs_mb, 2) if obs_mb > 0 else None,
    }

    # ── Query params ──────────────────────────────────────────────────────────
    print("Choosing query parameters...")
    params = _choose_queries(completed_path)

    # Get a variant for phewas from the best analysis
    q = query_store(completed_path)
    analysis_result = q.analysis(params["analysis_id"])
    if len(analysis_result["variant_index"]) > 0:
        obs_mask = analysis_result["association_status"] == "observed"
        obs_vi = analysis_result["variant_index"][obs_mask]
        if len(obs_vi) > 0:
            mid_vi = int(obs_vi[len(obs_vi) // 2])
            params["phewas_variant_index"] = mid_vi
    q.close()
    results["query_params"] = {k: str(v) for k, v in params.items() if v is not None}

    # ── Query timings ─────────────────────────────────────────────────────────
    print("Benchmarking queries...")
    timings: dict = {}

    q = query_store(completed_path)

    timings["analysis (all)"] = _measure_n(
        lambda: q.analysis(params["analysis_id"]), n_reps=10
    )
    timings["analysis (observed_only)"] = _measure_n(
        lambda: q.analysis(params["analysis_id"], observed_only=True), n_reps=10
    )
    if params.get("cis_chrom"):
        timings["range_by_analysis (2Mb)"] = _measure_n(
            lambda: q.range_by_analysis(params["cis_chrom"], params["cis_start"], params["cis_end"]),
            n_reps=5,
        )
        timings["range_phewas (2Mb)"] = _measure_n(
            lambda: q.range_phewas(params["cis_chrom"], params["cis_start"], params["cis_end"]),
            n_reps=5,
        )
    if params.get("phewas_variant_index") is not None:
        vi = params["phewas_variant_index"]
        from opengwasdb.variants.axis import VariantAxis
        va = VariantAxis(completed_path)
        rec = va.by_index(vi)
        va.close()
        if rec:
            alid = rec.alid
            timings["phewas (single variant)"] = _measure_n(
                lambda: q.phewas(alid), n_reps=10
            )
    timings["top_hits (5e-8)"] = _measure_n(
        lambda: q.top_hits(threshold=5e-8), n_reps=10
    )
    q.close()

    results["query_timings"] = timings

    # ── Write JSON ────────────────────────────────────────────────────────────
    out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"Results written to {out_json}")

    # ── Write QMD ────────────────────────────────────────────────────────────
    out_qmd = out_json.with_suffix(".qmd")
    _write_qmd(results, out_qmd)
    print(f"QMD written to {out_qmd}")


if __name__ == "__main__":
    main()
