#!/usr/bin/env python3
"""Build and benchmark Dense Observed-Only storage from full GWAS-SSF files.

This uses the three full harmonised files in `besdq/data/ebi_input`, which are
the active local Dense-mode vertical-slice benchmark inputs. The sparse
38714679 stage-1 fixture is intentionally excluded from Dense-mode comparisons.

The script intentionally uses a streaming two-pass builder: first collect the
union variant axis, then fill dense Z/SE arrays. The fixture-oriented package
builder keeps normalised records in memory and is not the right implementation
shape for full GWAS-SSF inputs.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import shutil
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc

from opengwasdb.index import create_lookup_indexes, initialise_schema, set_metadata
from opengwasdb.layouts.dense.constants import DEFAULT_CHUNK_SHAPE, DEFAULT_COMPRESSOR
from opengwasdb.layouts.dense.top_hits import build_top_hit_indexes
from opengwasdb.model.enums import AssociationCoverage, CompletionState, PrimaryStorageLayout
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.query import query_store
from opengwasdb.validation import validate_store
from opengwasdb.variants import (
    VARIANT_AXIS_FORMAT,
    VARIANT_OFFSETS_FILENAME,
    VARIANT_TABIX_FILENAME,
    VARIANT_TABLE_FILENAME,
    CanonicalVariant,
    VariantAxis,
    chromosome_sort_key,
    orient_to_canonical,
    write_variant_axis,
)

DEFAULT_SOURCE_DIR = Path("/Users/gh13047/repo/besdq/data/ebi_input")
DEFAULT_OUTPUT_DIR = Path("/Users/gh13047/repo/opengwasdb/docs/benchmark-output")
DEFAULT_DOC = Path("/Users/gh13047/repo/opengwasdb/docs/opengwasdb-dense-ebi-input-benchmark.qmd")
N_REPS = 10
RNG = np.random.default_rng(42)


@dataclass(frozen=True)
class VariantRow:
    alid: str
    chromosome: str
    position: int
    effect_allele: str
    other_allele: str
    rsid: str | None


@dataclass(frozen=True)
class AnalysisRow:
    analysis_id: str
    phenotype_label: str


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_files = sorted(args.source_dir.glob("*.h.tsv.gz"))
    if not source_files:
        raise SystemExit(f"No *.h.tsv.gz files found in {args.source_dir}")

    analyses = read_analyses(args.source_dir, source_files)
    store_path = args.output_dir / "ebi_input_dense.opengwasdb"
    results_path = args.output_dir / "opengwasdb_ebi_input_benchmark.json"

    build_seconds = None
    if args.rebuild and store_path.exists():
        shutil.rmtree(store_path)
    if args.rebuild or not store_path.exists():
        start = time.perf_counter()
        build_streaming_dense_store(source_files, analyses, store_path)
        build_seconds = time.perf_counter() - start

    validation = validate_store(store_path)
    if not validation.ok:
        raise SystemExit(f"Built store is invalid: {validation.errors[:10]}")

    results = run_benchmark(
        source_dir=args.source_dir,
        source_files=source_files,
        store_path=store_path,
        build_seconds=build_seconds,
    )
    if results["build"]["build_seconds"] is None and results_path.exists():
        previous = json.loads(results_path.read_text(encoding="utf-8"))
        previous_build_seconds = previous.get("build", {}).get("build_seconds")
        if previous_build_seconds is not None:
            results["build"]["build_seconds"] = previous_build_seconds
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


def read_analyses(source_dir: Path, source_files: list[Path]) -> list[AnalysisRow]:
    labels = {}
    traits_path = source_dir / "traits.tsv"
    if traits_path.exists():
        with traits_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                labels[row["trait_id"]] = row["trait_name"]
    return [
        AnalysisRow(
            analysis_id=source.name.split(".h.tsv.gz")[0],
            phenotype_label=labels.get(source.name.split(".h.tsv.gz")[0], source.name),
        )
        for source in source_files
    ]


def build_streaming_dense_store(
    source_files: list[Path],
    analyses: list[AnalysisRow],
    store_path: Path,
) -> None:
    store_path.mkdir(parents=True, exist_ok=True)
    variants, alid_to_row = collect_variant_axis(source_files)
    z = np.full((len(variants), len(source_files)), np.nan, dtype=np.float16)
    se = np.full((len(variants), len(source_files)), np.nan, dtype=np.float16)

    for col, source in enumerate(source_files):
        fill_column(source, col, alid_to_row, z, se)

    write_manifest(store_path, source_files, variants, analyses)
    write_variant_axis(
        store_path,
        [
            CanonicalVariant(
                chromosome=variant.chromosome,
                position=variant.position,
                effect_allele=variant.effect_allele,
                other_allele=variant.other_allele,
            )
            for variant in variants
        ],
        {
            variant.alid: variant.rsid
            for variant in variants
            if variant.rsid is not None
        },
    )
    write_index(store_path, variants, analyses)
    write_zarr(store_path, z, se)
    build_top_hit_indexes(store_path)


def collect_variant_axis(source_files: list[Path]) -> tuple[list[VariantRow], dict[str, int]]:
    variants: list[VariantRow] = []
    alid_to_row: dict[str, int] = {}
    for source in source_files:
        for row in iter_source_rows(source):
            oriented = orient_to_canonical(
                row["chromosome"],
                row["base_pair_location"],
                row["effect_allele"],
                row["other_allele"],
            )
            alid = oriented.variant.alid
            if alid in alid_to_row:
                continue
            alid_to_row[alid] = len(variants)
            rsid = row.get("rsid") or row.get("rs_id") or None
            variants.append(
                VariantRow(
                    alid=alid,
                    chromosome=oriented.variant.chromosome,
                    position=oriented.variant.position,
                    effect_allele=oriented.variant.effect_allele,
                    other_allele=oriented.variant.other_allele,
                    rsid=rsid if rsid and rsid != "." else None,
                )
            )
    order = sorted(
        range(len(variants)),
        key=lambda i: (
            chromosome_sort_key(variants[i].chromosome),
            variants[i].position,
            variants[i].effect_allele,
            variants[i].other_allele,
        ),
    )
    sorted_variants = [variants[i] for i in order]
    sorted_lookup = {variant.alid: i for i, variant in enumerate(sorted_variants)}
    return sorted_variants, sorted_lookup


def fill_column(
    source: Path,
    col: int,
    alid_to_row: dict[str, int],
    z: np.ndarray,
    se: np.ndarray,
) -> None:
    for row in iter_source_rows(source):
        oriented = orient_to_canonical(
            row["chromosome"],
            row["base_pair_location"],
            row["effect_allele"],
            row["other_allele"],
        )
        row_index = alid_to_row[oriented.variant.alid]
        se_value = float(row["standard_error"])
        if not math.isfinite(se_value) or se_value < 0:
            continue
        beta = float(row["beta"])
        z_value = beta / se_value if se_value != 0 else np.nan
        if oriented.flipped:
            z_value = -z_value
        z[row_index, col] = z_value
        se[row_index, col] = se_value


def iter_source_rows(source: Path) -> Iterable[dict[str, str]]:
    with gzip.open(source, "rt", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def write_manifest(
    store_path: Path,
    source_files: list[Path],
    variants: list[VariantRow],
    analyses: list[AnalysisRow],
) -> None:
    manifest = StoreManifest(
        store_id="ebi-input-full-gwas",
        release_id="dense-observed-benchmark",
        format_version="0.1",
        primary_layout=PrimaryStorageLayout.DENSE,
        association_coverage=AssociationCoverage.FULL,
        completion_state=CompletionState.OBSERVED_ONLY,
        reference_assembly="GRCh38",
        created_at=datetime.now(UTC).isoformat(),
        provenance={
            "builder": "benchmarks/benchmark_ebi_input_dense.py",
            "source_files": [str(path) for path in source_files],
            "source_file_count": len(source_files),
            "n_variants": len(variants),
            "n_analyses": len(analyses),
            "dense": {
                "statistic_arrays": ["z", "se"],
                "dtype": "float16",
                "chunk_shape": list(DEFAULT_CHUNK_SHAPE),
                "compressor": DEFAULT_COMPRESSOR,
                "top_hit_thresholds": [5e-8, 5e-6, 5e-4],
                "variant_axis": {
                    "format": VARIANT_AXIS_FORMAT,
                    "table": VARIANT_TABLE_FILENAME,
                    "tabix_index": VARIANT_TABIX_FILENAME,
                    "row_offsets": VARIANT_OFFSETS_FILENAME,
                },
            },
        },
    )
    (store_path / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_index(store_path: Path, variants: list[VariantRow], analyses: list[AnalysisRow]) -> None:
    connection = sqlite3.connect(store_path / "index.sqlite")
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = OFF")
        connection.execute("PRAGMA synchronous = OFF")
        connection.execute("PRAGMA temp_store = MEMORY")
        initialise_schema(connection)
        set_metadata(connection, "schema_version", 2)
        set_metadata(connection, "n_variants", len(variants))
        set_metadata(connection, "n_analyses", len(analyses))
        set_metadata(
            connection,
            "dense",
            {
                "dtype": "float16",
                "chunk_shape": list(DEFAULT_CHUNK_SHAPE),
                "compressor": DEFAULT_COMPRESSOR,
                "variant_axis": {
                    "format": VARIANT_AXIS_FORMAT,
                    "table": VARIANT_TABLE_FILENAME,
                    "tabix_index": VARIANT_TABIX_FILENAME,
                    "row_offsets": VARIANT_OFFSETS_FILENAME,
                },
            },
        )
        connection.executemany(
            """
            INSERT OR IGNORE INTO variant_aliases(alias, variant_index)
            VALUES (?, ?)
            """,
            (
                (
                    variant.rsid,
                    i,
                )
                for i, variant in enumerate(variants)
                if variant.rsid is not None
            ),
        )
        create_lookup_indexes(connection)
        connection.executemany(
            """
            INSERT INTO analyses(
                analysis_index, analysis_id, phenotype_id, phenotype_label,
                analysis_label, stored_effect_scale
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    i,
                    analysis.analysis_id,
                    analysis.analysis_id,
                    analysis.phenotype_label,
                    analysis.analysis_id,
                    "sd_units",
                )
                for i, analysis in enumerate(analyses)
            ),
        )
        connection.commit()
    finally:
        connection.close()


def write_zarr(store_path: Path, z: np.ndarray, se: np.ndarray) -> None:
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    root = zarr.open_group(str(store_path / "data.zarr"), mode="w", zarr_format=2)
    root.create_array(
        "z",
        data=np.asarray(z, dtype="float16"),
        chunks=DEFAULT_CHUNK_SHAPE,
        compressors=compressor,
    )
    root.create_array(
        "se",
        data=np.asarray(se, dtype="float16"),
        chunks=DEFAULT_CHUNK_SHAPE,
        compressors=compressor,
    )
    root.attrs["layout"] = "dense"
    root.attrs["completion_state"] = "observed_only"
    root.attrs["compressor"] = DEFAULT_COMPRESSOR
    root.attrs["chunk_shape"] = list(DEFAULT_CHUNK_SHAPE)


def run_benchmark(
    *,
    source_dir: Path,
    source_files: list[Path],
    store_path: Path,
    build_seconds: float | None,
) -> dict[str, object]:
    query = query_store(store_path)
    selection = choose_queries(store_path)
    query_specs = {
        "regional": lambda: query.range_phewas(
            selection["region_chromosome"],
            selection["region_start"],
            selection["region_end"],
        ),
        "phewas": lambda: query.phewas(selection["phewas_alid"]),
        "tophits": lambda: query.top_hits(threshold=selection["top_hit_threshold"]),
        "bulk_arrays": lambda: query.analysis_arrays(selection["bulk_analysis_id"]),
        "random_lookup": lambda: query.lookup(
            selection["random_alids"],
            selection["random_analysis_ids"],
        ),
    }
    timings = []
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
    raw_gz_bytes = sum(path.stat().st_size for path in source_files)
    raw_sidecar_bytes = raw_gz_bytes + sum(
        path.stat().st_size for path in source_dir.glob("*meta.yaml")
    )
    raw_sidecar_bytes += sum(path.stat().st_size for path in source_dir.glob("*.tbi"))
    component_sizes = store_component_sizes(store_path)
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
            "build_seconds": round(build_seconds, 3) if build_seconds is not None else None,
        },
        "storage": {
            "raw_tsv_gz_bytes": raw_gz_bytes,
            "raw_tsv_gz_plus_sidecars_bytes": raw_sidecar_bytes,
            "opengwasdb_store_bytes": dir_bytes(store_path),
            "components": component_sizes,
            "raw_tsv_gz_to_store_ratio": raw_gz_bytes / dir_bytes(store_path),
        },
        "selection": selection,
        "timings": timings,
    }


def choose_queries(store_path: Path) -> dict[str, object]:
    connection = sqlite3.connect(store_path / "index.sqlite")
    connection.row_factory = sqlite3.Row
    try:
        analyses = connection.execute(
            "SELECT analysis_index, analysis_id FROM analyses ORDER BY analysis_index"
        ).fetchall()
        variant_axis = VariantAxis(store_path, connection)
        try:
            variants = variant_axis.all()
        finally:
            variant_axis.close()
    finally:
        connection.close()

    windows: dict[tuple[str, int], int] = {}
    for variant in variants:
        key = (variant.chromosome, variant.position // 1_000_000)
        windows[key] = windows.get(key, 0) + 1
    region_chromosome, region_window = max(windows.items(), key=lambda item: item[1])[0]
    root = zarr.open_group(str(store_path / "data.zarr"), mode="r")
    finite = np.isfinite(root["z"][:].astype("float32"))
    row_counts = finite.sum(axis=1)
    col_counts = finite.sum(axis=0)
    phewas_row = int(np.argmax(row_counts))
    bulk_col = int(np.argmax(col_counts))
    random_rows = sorted(RNG.choice(len(variants), min(100, len(variants)), replace=False))
    random_cols = sorted(RNG.choice(len(analyses), min(3, len(analyses)), replace=False))
    start = region_window * 1_000_000
    return {
        "region_chromosome": region_chromosome,
        "region_start": start,
        "region_end": start + 999_999,
        "phewas_alid": variants[phewas_row].alid,
        "bulk_analysis_id": str(analyses[bulk_col]["analysis_id"]),
        "top_hit_threshold": choose_top_hit_threshold(root),
        "random_alids": [variants[row].alid for row in random_rows],
        "random_analysis_ids": [str(analyses[col]["analysis_id"]) for col in random_cols],
    }


def choose_top_hit_threshold(root) -> float:
    for threshold in (5e-8, 5e-6, 5e-4):
        key = f"p_{threshold:.0e}".replace("-", "_").replace("+", "")
        if "top_hits" in root and key in root["top_hits"] and len(root["top_hits"][key]["z"]) > 0:
            return threshold
    return 5e-8


def bench(fn) -> tuple[float, float, int]:
    result = fn()
    times = []
    for _ in range(N_REPS):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return (
        float(np.median(times) * 1000),
        float(np.percentile(times, 95) * 1000),
        result_size(result),
    )


def result_size(result) -> int:
    if result is None:
        return 0
    if isinstance(result, dict) and "z" in result:
        return int(np.asarray(result["z"]).size)
    return len(result)


def store_shape(store_path: Path) -> tuple[int, int]:
    root = zarr.open_group(str(store_path / "data.zarr"), mode="r")
    return tuple(root["z"].shape)


def finite_cell_count(store_path: Path) -> int:
    root = zarr.open_group(str(store_path / "data.zarr"), mode="r")
    return int(np.isfinite(root["z"][:]).sum())


def dir_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def file_bytes(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def store_component_sizes(store_path: Path) -> dict[str, int]:
    return {
        "manifest_json": file_bytes(store_path / "manifest.json"),
        "index_sqlite": file_bytes(store_path / "index.sqlite"),
        "variants_tsv_gz": file_bytes(store_path / VARIANT_TABLE_FILENAME),
        "variants_tbi": file_bytes(store_path / VARIANT_TABIX_FILENAME),
        "variant_offsets_npy": file_bytes(store_path / VARIANT_OFFSETS_FILENAME),
        "data_zarr": dir_bytes(store_path / "data.zarr"),
        "top_hits": dir_bytes(store_path / "data.zarr" / "top_hits"),
    }


def write_qmd(results: dict[str, object], doc_path: Path, results_path: Path) -> None:
    dataset = results["dataset"]
    storage = results["storage"]
    build = results["build"]
    selection = results["selection"]
    timings = results["timings"]

    def mb(value: int | float) -> str:
        return f"{value / 1_000_000:.2f} MB"

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
    random_label = (
        f"{len(selection['random_alids'])} variants × "
        f"{len(selection['random_analysis_ids'])} analyses"
    )
    raw_sidecar_ratio = storage["raw_tsv_gz_plus_sidecars_bytes"] / storage["raw_tsv_gz_bytes"]
    store_ratio = storage["opengwasdb_store_bytes"] / storage["raw_tsv_gz_bytes"]
    components = storage["components"]
    storage_table = "\n".join(
        [
            f"| Raw source TSV.gz | {mb(storage['raw_tsv_gz_bytes'])} | 1.00× |",
            (
                "| Raw source TSV.gz + metadata/TBI | "
                f"{mb(storage['raw_tsv_gz_plus_sidecars_bytes'])} | "
                f"{raw_sidecar_ratio:.2f}× raw |"
            ),
            (
                "| OpenGWASDB dense store | "
                f"{mb(storage['opengwasdb_store_bytes'])} | {store_ratio:.2f}× raw |"
            ),
        ]
    )
    component_table = "\n".join(
        [
            f"| manifest.json | {mb(components['manifest_json'])} |",
            f"| index.sqlite | {mb(components['index_sqlite'])} |",
            f"| variants.tsv.gz | {mb(components['variants_tsv_gz'])} |",
            f"| variants.tsv.gz.tbi | {mb(components['variants_tbi'])} |",
            f"| variant_offsets.npy | {mb(components['variant_offsets_npy'])} |",
            f"| data.zarr | {mb(components['data_zarr'])} |",
            f"| data.zarr/top_hits | {mb(components['top_hits'])} |",
        ]
    )
    text = f"""---
title: "OpenGWASDB Dense Benchmark — EBI Full GWAS-SSF Inputs"
subtitle: "Three full harmonised GWAS files from besdq/data/ebi_input"
date: today
format:
  html:
    toc: true
    toc-depth: 3
    code-fold: true
    embed-resources: true
---

## Summary

This benchmark uses `besdq/data/ebi_input`, which contains three full harmonised
GWAS-SSF files. This is the active local Dense-mode vertical-slice benchmark.
The older `38714679` fixture is sparse and is no longer used or compared for
Dense-mode evaluation.

The query workload mirrors the `besdq` dense benchmark: regional, PheWAS,
top-hits, full-study bulk extraction, and random variant × analysis lookup.

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

## Store component footprint

| Component | Size |
|---|---:|
{component_table}

## Query selection

| Query input | Value |
|---|---|
| Regional query | `{region_label}` |
| PheWAS variant | `{selection['phewas_alid']}` |
| Bulk analysis | `{selection['bulk_analysis_id']}` |
| Top-hit threshold | `{selection['top_hit_threshold']}` |
| Random lookup | {random_label} |

## Query timings

Timings are from the Python query engine with one store opened once and each
query repeated {N_REPS} times. `bulk_arrays` is the dense full-study extraction
path that returns Z and SE arrays without materialising millions of Python row
objects.

| Query | Median ms | p95 ms | Result rows/values |
|---|---:|---:|---:|
{timing_table}

## Interpretation

- The full GWAS-SSF inputs are almost perfectly dense across the three analyses,
  so this is the right local shape for Dense mode.
- The sparse `38714679` fixture is intentionally excluded from Dense-mode
  benchmark comparisons because it represents a different storage shape.
- Component sizes identify whether footprint is dominated by dense arrays,
  compact metadata, the tabix variant table, row offsets, or top-hit indexes.
- Full-analysis extraction should use array/streaming output. Returning millions
  of Python dataclass rows is not the right API shape for that path.

## Artifacts

- JSON results: `{results_path.relative_to(doc_path.parent)}`
- Store path: `{build['store_path']}`
"""
    doc_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
