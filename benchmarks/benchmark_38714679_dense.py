#!/usr/bin/env python3
"""Build an archived sparse-fixture smoke-test store from local besdq data.

This runner uses `besdq/data/38714679/38714679`, a collection of 60 gzipped
per-analysis GWAS summary TSVs with overlapping variant sets. The fixture is
sparse and is not part of Dense-mode vertical-slice benchmarking or comparison.

Outputs are written under `docs/benchmark-output/` and a static Quarto report is
written to `docs/opengwasdb-dense-38714679-benchmark.qmd` as an archived record.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from opengwasdb.index import connect
from opengwasdb.query import query_store
from opengwasdb.validation import validate_store

DEFAULT_SOURCE_DIR = Path("/Users/gh13047/repo/besdq/data/38714679/38714679")
DEFAULT_OUTPUT_DIR = Path("/Users/gh13047/repo/opengwasdb/docs/benchmark-output")
DEFAULT_DOC = Path("/Users/gh13047/repo/opengwasdb/docs/opengwasdb-dense-38714679-benchmark.qmd")
N_REPS = 20
RNG = np.random.default_rng(42)


@dataclass(frozen=True)
class BenchmarkSelection:
    region_chromosome: str
    region_start: int
    region_end: int
    phewas_alid: str
    bulk_analysis_id: str
    top_hit_threshold: float
    random_alids: list[str]
    random_analysis_ids: list[str]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_source = output_dir / "38714679_combined_source.tsv.gz"
    store_path = output_dir / "38714679_dense.opengwasdb"
    results_path = output_dir / "opengwasdb_38714679_benchmark.json"

    source_files = sorted(args.source_dir.glob("*.tsv.gz"))
    if not source_files:
        raise SystemExit(f"No source .tsv.gz files found in {args.source_dir}")

    if args.rebuild or not combined_source.exists():
        write_combined_source(source_files, combined_source)

    build_seconds = None
    if args.rebuild and store_path.exists():
        shutil.rmtree(store_path)
    if args.rebuild or not store_path.exists():
        # The fixture builder is the tests': it splits the combined source into the
        # manifest and per-Analysis files `build-dense` reads.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
        from dense_sources import build_dense_fixture

        start = time.perf_counter()
        build_dense_fixture(
            [combined_source],
            store_path,
            store_id="38714679-stage1",
            release_id="dense-observed-benchmark",
            reference_assembly="GRCh37",
            overwrite=True,
        )
        build_seconds = time.perf_counter() - start

    validation = validate_store(store_path)
    if not validation.ok:
        raise SystemExit(f"Built store is invalid: {validation.errors}")

    results = run_benchmark(
        source_dir=args.source_dir,
        source_files=source_files,
        combined_source=combined_source,
        store_path=store_path,
        build_seconds=build_seconds,
    )
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_qmd(results, args.doc_path, results_path)
    print(f"Wrote {results_path}")
    print(f"Wrote {args.doc_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def write_combined_source(source_files: list[Path], output_path: Path) -> None:
    header = [
        "analysis_id",
        "phenotype_id",
        "phenotype_label",
        "analysis_label",
        "chromosome",
        "position",
        "effect_allele",
        "other_allele",
        "beta",
        "se",
        "rsid",
        "eaf",
        "stored_effect_scale",
    ]
    with gzip.open(output_path, "wt", encoding="utf-8", newline="") as out_handle:
        writer = csv.DictWriter(out_handle, delimiter="\t", fieldnames=header)
        writer.writeheader()
        for source_file in source_files:
            analysis_id = source_file.name.removesuffix(".tsv.gz")
            phenotype_label = read_trait_name(source_file.with_suffix("").with_suffix(".yaml"))
            with gzip.open(source_file, "rt", encoding="utf-8", newline="") as in_handle:
                reader = csv.DictReader(in_handle, delimiter="\t")
                for row in reader:
                    writer.writerow(
                        {
                            "analysis_id": analysis_id,
                            "phenotype_id": analysis_id,
                            "phenotype_label": phenotype_label or analysis_id,
                            "analysis_label": analysis_id,
                            "chromosome": row["chr"],
                            "position": row["bp"],
                            "effect_allele": row["a1"],
                            "other_allele": row["a2"],
                            "beta": row["beta"],
                            "se": row["se"],
                            "rsid": row["rsid"],
                            "eaf": row.get("eaf", ""),
                            "stored_effect_scale": "sd_units",
                        }
                    )


def read_trait_name(path: Path) -> str | None:
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("trait_name:"):
            continue
        value = line.split(":", 1)[1].strip()
        if value:
            return value.strip("'\"")
        continuation = []
        for next_line in lines[i + 1 :]:
            if next_line.startswith(" ") or next_line.startswith("\t"):
                continuation.append(next_line.strip())
            else:
                break
        return " ".join(continuation) if continuation else None
    return None


def run_benchmark(
    *,
    source_dir: Path,
    source_files: list[Path],
    combined_source: Path,
    store_path: Path,
    build_seconds: float | None,
) -> dict[str, object]:
    selection = choose_queries(store_path)
    query = query_store(store_path)
    timings = []

    query_specs = {
        "regional": lambda: query.range_phewas(
            selection.region_chromosome,
            selection.region_start,
            selection.region_end,
        ),
        "phewas": lambda: query.phewas(selection.phewas_alid),
        "tophits": lambda: query.top_hits(threshold=selection.top_hit_threshold),
        "bulk": lambda: query.analysis(selection.bulk_analysis_id),
        "random_lookup": lambda: query.lookup(
            selection.random_alids,
            selection.random_analysis_ids,
        ),
    }
    for name, fn in query_specs.items():
        median_ms, p95_ms, result_count = bench(fn)
        timings.append(
            {
                "query": name,
                "median_ms": round(median_ms, 3),
                "p95_ms": round(p95_ms, 3),
                "result_count": result_count,
            }
        )

    shape = store_shape(store_path)
    raw_tsv_gz_bytes = sum(path.stat().st_size for path in source_files)
    raw_with_yaml_bytes = raw_tsv_gz_bytes + sum(
        yaml.stat().st_size for yaml in source_dir.glob("*.yaml")
    )
    store_bytes = dir_bytes(store_path)
    combined_source_bytes = combined_source.stat().st_size
    return {
        "dataset": {
            "source_dir": str(source_dir),
            "n_source_files": len(source_files),
            "n_variants": shape[0],
            "n_analyses": shape[1],
            "n_cells": shape[0] * shape[1],
            "finite_cells": finite_cell_count(store_path),
        },
        "build": {
            "store_path": str(store_path),
            "combined_source": str(combined_source),
            "build_seconds": round(build_seconds, 3) if build_seconds is not None else None,
        },
        "storage": {
            "raw_tsv_gz_bytes": raw_tsv_gz_bytes,
            "raw_tsv_gz_plus_yaml_bytes": raw_with_yaml_bytes,
            "combined_source_gz_bytes": combined_source_bytes,
            "opengwasdb_store_bytes": store_bytes,
            "raw_tsv_gz_to_store_ratio": raw_tsv_gz_bytes / store_bytes,
        },
        "selection": selection.__dict__,
        "timings": timings,
    }


def choose_queries(store_path: Path) -> BenchmarkSelection:
    with connect(store_path / "index.sqlite") as connection:
        variants = connection.execute(
            "SELECT variant_index, alid, chromosome, position FROM variants ORDER BY variant_index"
        ).fetchall()
        analyses = connection.execute(
            "SELECT analysis_index, analysis_id FROM analyses ORDER BY analysis_index"
        ).fetchall()

    import zarr

    root = zarr.open_group(str(store_path / "data.zarr"), mode="r")
    finite = np.isfinite(root["z"][:].astype("float32"))
    row_counts = finite.sum(axis=1)
    col_counts = finite.sum(axis=0)

    phewas_row = int(np.argmax(row_counts))
    bulk_col = int(np.argmax(col_counts))

    positions_by_window: dict[tuple[str, int], list[int]] = {}
    for variant in variants:
        key = (str(variant["chromosome"]), int(variant["position"]) // 1_000_000)
        positions_by_window.setdefault(key, []).append(int(variant["position"]))
    (region_chr, window), positions = max(
        positions_by_window.items(),
        key=lambda item: len(item[1]),
    )
    region_start = window * 1_000_000
    region_end = region_start + 999_999

    random_rows = sorted(RNG.choice(len(variants), min(100, len(variants)), replace=False))
    random_cols = sorted(RNG.choice(len(analyses), min(10, len(analyses)), replace=False))
    return BenchmarkSelection(
        region_chromosome=region_chr,
        region_start=region_start,
        region_end=region_end,
        phewas_alid=str(variants[phewas_row]["alid"]),
        bulk_analysis_id=str(analyses[bulk_col]["analysis_id"]),
        top_hit_threshold=choose_top_hit_threshold(root),
        random_alids=[str(variants[row]["alid"]) for row in random_rows],
        random_analysis_ids=[str(analyses[col]["analysis_id"]) for col in random_cols],
    )


def choose_top_hit_threshold(root) -> float:
    for threshold in (5e-8, 5e-6, 5e-4):
        key = f"p_{threshold:.0e}".replace("-", "_").replace("+", "")
        if "top_hits" in root and key in root["top_hits"] and len(root["top_hits"][key]["z"]) > 0:
            return threshold
    return 5e-8


def bench(fn) -> tuple[float, float, int]:
    times = []
    result = fn()
    for _ in range(N_REPS):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return float(np.median(times) * 1000), float(np.percentile(times, 95) * 1000), len(result)


def store_shape(store_path: Path) -> tuple[int, int]:
    import zarr

    root = zarr.open_group(str(store_path / "data.zarr"), mode="r")
    return tuple(root["z"].shape)


def finite_cell_count(store_path: Path) -> int:
    import zarr

    root = zarr.open_group(str(store_path / "data.zarr"), mode="r")
    return int(np.isfinite(root["z"][:]).sum())


def dir_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def write_qmd(results: dict[str, object], doc_path: Path, results_path: Path) -> None:
    dataset = results["dataset"]
    storage = results["storage"]
    build = results["build"]
    selection = results["selection"]
    timings = results["timings"]

    def mb(value: int | float) -> str:
        return f"{value / 1_000_000:.2f} MB"

    storage_rows = [
        ("Raw source TSV.gz", storage["raw_tsv_gz_bytes"], "1.00×"),
        (
            "Raw source TSV.gz + YAML",
            storage["raw_tsv_gz_plus_yaml_bytes"],
            (
                f"{storage['raw_tsv_gz_plus_yaml_bytes'] / storage['raw_tsv_gz_bytes']:.2f}"
                "× raw TSV.gz"
            ),
        ),
        (
            "Combined benchmark source TSV.gz",
            storage["combined_source_gz_bytes"],
            f"{storage['combined_source_gz_bytes'] / storage['raw_tsv_gz_bytes']:.2f}× raw TSV.gz",
        ),
        (
            "OpenGWASDB dense store",
            storage["opengwasdb_store_bytes"],
            f"{storage['opengwasdb_store_bytes'] / storage['raw_tsv_gz_bytes']:.2f}× raw TSV.gz",
        ),
    ]
    storage_table = "\n".join(
        f"| {label} | {mb(size)} | {comparison} |" for label, size, comparison in storage_rows
    )
    timing_table = "\n".join(
        "| {query} | {median_ms:.3f} | {p95_ms:.3f} | {result_count:,} |".format(**row)
        for row in timings
    )
    build_line = (
        f"{build['build_seconds']} seconds"
        if build["build_seconds"] is not None
        else "not rebuilt; existing store reused"
    )
    region_label = (
        f"{selection['region_chromosome']}:"
        f"{selection['region_start']}-{selection['region_end']}"
    )
    random_lookup_label = (
        f"{len(selection['random_alids'])} variants × "
        f"{len(selection['random_analysis_ids'])} analyses"
    )
    text = f"""---
title: "Archived Sparse Fixture Smoke Test"
subtitle: "Local 38714679 fixture · not a Dense vertical-slice benchmark"
date: today
format:
  html:
    toc: true
    toc-depth: 3
    code-fold: true
    embed-resources: true
---

## Summary

This document is retained only as an archived smoke-test record for the local
`besdq/data/38714679/38714679` fixture. The dataset is sparse and is not used or
compared for Dense-mode vertical-slice evaluation.

The historical workload was:

1. regional query;
2. PheWAS-style single variant across analyses;
3. top-hit query;
4. full analysis extraction;
5. random variant × analysis lookup.

## Dataset

| Field | Value |
|---|---:|
| Source files | {dataset['n_source_files']:,} |
| Store variants | {dataset['n_variants']:,} |
| Analyses | {dataset['n_analyses']:,} |
| Dense cells | {dataset['n_cells']:,} |
| Finite observed cells | {dataset['finite_cells']:,} |
| Observed cell fraction | {dataset['finite_cells'] / dataset['n_cells']:.2%} |
| Build time | {build_line} |

## Storage footprint

| Format | Size | Comparison |
|---|---:|---:|
{storage_table}

For this sparse molecular-style fixture, the dense store was expected to be
larger than the raw gzipped inputs because most dense cells are missing. This is
not the target shape for Dense mode. The active Dense-mode benchmark now uses
the full GWAS-SSF files in `besdq/data/ebi_input`.

## Query selection

| Query input | Value |
|---|---|
| Regional query | `{region_label}` |
| PheWAS variant | `{selection['phewas_alid']}` |
| Bulk analysis | `{selection['bulk_analysis_id']}` |
| Top-hit threshold | `{selection['top_hit_threshold']}` |
| Random lookup | {random_lookup_label} |

## Query timings

Timings are from the Python query engine with one store opened once and each
query repeated {N_REPS} times.

| Query | Median ms | p95 ms | Result rows |
|---|---:|---:|---:|
{timing_table}

## Interpretation

- This fixture should not be used as evidence for Dense-mode storage footprint.
- Batched query paths are fast on this small local fixture: PheWAS is sub-ms,
  random lookup and top hits are tens of milliseconds, and the largest regional
  query is dominated by materialising tens of thousands of Python result rows.
- Dense storage is a bad footprint match for this particular molecular-style
  fixture because the observed cell fraction is low. That reinforces the current
  design split: Dense for many analyses over a shared variant axis; Ragged/Sparse
  for cis-and-signals molecular datasets.
- The active Dense-mode benchmark and comparison target is
  `docs/opengwasdb-dense-ebi-input-benchmark.qmd`.

## Artifacts

- JSON results: `{results_path.relative_to(doc_path.parent)}`
- Store path: `{build['store_path']}`
- Combined source path: `{build['combined_source']}`
"""
    doc_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
