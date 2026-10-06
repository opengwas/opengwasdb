#!/usr/bin/env python3
"""Benchmark and sanity-check the GWAS Catalog EUR Hybrid Store (OGS-00011).

Measures the seven query shapes every full-scale benchmark shares, on the same
fresh-interpreter RSS probe (`benchmarks/_query_shapes.py`), so this report is
comparable with OGS-00009 and OGS-00016. A Hybrid store answers a query from
two components, the Dense Component for variants on the reference axis and the
Ragged Overflow for each Analysis's off-axis variants, so it adds:

* two Hybrid shapes: a PheWAS of an off-axis variant, which can only be
  answered from the Overflow, and the bulk read of the Analysis with the
  largest Overflow;
* storage by component, with file counts, against the source GWAS-SSF files;
* the build's own step records;
* the bulk shape against a single `pd.read_csv` of the same Analysis's source;
* known-locus checks, each naming the component that answered;
* PheWAS of two pleiotropic variants across all Analyses;
* three IVW Mendelian randomisation pairs run end to end through the store,
  each instrument tagged with the component that holds it.

How each Analysis splits between the components is measured separately by
`measure_hybrid_component_split.py`, whose TSV the report plots.

Usage:
  pixi run -e dev python benchmarks/benchmark_ogs00011_hybrid.py \
      --store /data/opengwasdb/stores/OGS-00011/store.opengwasdb \
      --manifest /data/opengwasdb/stores/OGS-00011/work/analyses.tsv \
      --records /data/opengwasdb/stores/OGS-00011/records \
      --output docs/benchmark-output/opengwasdb_ogs00011_hybrid_benchmark.json
"""

from __future__ import annotations

# This harness measures the Hybrid Store Release OGS-00011 and shares its report
# helpers with the FinnGen Dense harness through `benchmarks._hybrid_report`.
import argparse
import csv
import gzip
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmarks import _hybrid_report as _report
from benchmarks import _query_shapes
from benchmarks._artifact import provenance, write_artifact
from benchmarks._rss import run_probe
from benchmarks.benchmark_finngen_dense import run_mr
from benchmarks.benchmark_ukbb_dense import _dir_bytes
from opengwasdb.query import query_store

# Type 2 diabetes (Xue 2018) and TCF7L2, the FinnGen benchmark's exposure and
# region, so the shared shapes ask the same biological question of both stores.
EXPOSURE = "GCST006867"
PHEWAS_ALID = "10:112998590:C:T"  # rs7903146, TCF7L2; on the reference axis
REGION = ("10", 112_500_000, 113_500_000)
# The Analysis with the largest Ragged Overflow (75.7 M off-axis associations,
# a whole-genome-sequencing BMI study), per the component-split measurement.
OVERFLOW_HEAVY = "GCST90502911"
# The first off-axis variant `default_rng(1)` draws from the shared axis: an
# arbitrary off-axis PheWAS rather than one chosen for being fast or slow.
OFF_AXIS_SEED = 1

# (analysis, alid, rsid, risk allele, locus, why it is expected)
KNOWN_LOCI = [
    ("GCST006867", "10:112998590:C:T", "rs7903146", "T", "TCF7L2", "strongest common T2D locus"),
    ("GCST006867", "3:12351626:C:G", "rs1801282", "C", "PPARG", "Pro12Ala, protective Ala allele"),
    ("GCST90029007", "16:53786615:A:T", "rs9939609", "A", "FTO", "strongest common BMI locus"),
    ("GCST90029007", "18:60183864:C:T", "rs17782313", "C", "MC4R", "BMI-raising allele"),
    # EADB rather than Bellenguez 2022 (GCST90027158), which has no rs429358 row.
    ("GCST90704648", "19:44908684:C:T", "rs429358", "C", "APOE", "epsilon-4 defining allele"),
    ("GCST003116", "9:22125504:C:G", "rs1333049", "C", "CDKN2B-AS1", "9p21 coronary risk locus"),
    ("GCST003116", "9:22098575:A:G", "rs4977574", "G", "CDKN2B-AS1", "9p21 coronary risk locus"),
    ("GCST90239658", "1:55039974:G:T", "rs11591147", "G", "PCSK9", "R46L; T lowers LDL"),
]
# rs1801282 is a modest-effect variant; the others must clear genome-wide significance.
MODEST = {"rs1801282"}

MR_PAIRS = [
    ("GCST90239658", "GCST003116", "LDL cholesterol (GLGC) -> coronary artery disease "
     "(CARDIoGRAMplusC4D): the canonical positive control"),
    ("GCST90029007", "GCST006867", "BMI (UK Biobank) -> type 2 diabetes (DIAGRAM/UKB): "
     "a large, well-replicated causal effect; samples overlap"),
    ("GCST90310294", "GCST003116", "systolic blood pressure (MVP) -> coronary artery disease: "
     "a well-replicated causal effect"),
]

PHEWAS_VARIANTS = [
    ("10:112998590:C:T", "rs7903146", "TCF7L2"),
    ("19:44908684:C:T", "rs429358", "APOE"),
]

# A shape whose warm-up takes longer than this is timed once, not `--reps`
# times. The off-axis Overflow scan makes some shapes take minutes per call on
# this store; repeating them would add hours and measure nothing new. Each
# timing records how many repetitions its median is over.
SLOW_SHAPE_SECONDS = 60.0

# GWAS-SSF columns a z/se/eaf answer needs, harmonised spelling first.
SOURCE_COLUMNS = {
    "chrom": ["hm_chrom", "chromosome"],
    "pos": ["hm_pos", "base_pair_location"],
    "effect_allele": ["hm_effect_allele", "effect_allele"],
    "other_allele": ["hm_other_allele", "other_allele"],
    "beta": ["hm_beta", "beta"],
    "se": ["standard_error"],
    "eaf": ["hm_effect_allele_frequency", "effect_allele_frequency"],
}


def _n_shared_variants(store: Path) -> int:
    return int(json.loads((store / "manifest.json").read_text())["provenance"]["n_variants"])


def _off_axis_alid(q: Any, n_variants: int) -> str:
    rng = np.random.default_rng(OFF_AXIS_SEED)
    while True:
        index = int(rng.integers(0, n_variants))
        if not q._shared_is_on_panel(index):
            return q._variant_axis.by_index(index).alid


def _patterns(q: Any, store: Path) -> dict[str, Any]:
    analyses = q.analyses_table()
    n_variants = _n_shared_variants(store)
    patterns = _query_shapes.build_query_patterns(
        q, analyses, n_variants, len(analyses),
        exposure=EXPOSURE, phewas_alid=PHEWAS_ALID, region=REGION,
    )
    off_axis = _off_axis_alid(q, n_variants)
    patterns["phewas_off_axis"] = lambda: q.phewas(off_axis)
    patterns["bulk_overflow_heavy"] = lambda: q.analysis(OVERFLOW_HEAVY)
    return patterns


def _time_shape(fn: Any, reps: int) -> dict[str, Any]:
    """`_median_ms` (one warm-up, then `reps` timed calls), except that a shape
    slower than `SLOW_SHAPE_SECONDS` is timed once after its warm-up."""
    started = time.perf_counter()
    fn()
    warmup_s = time.perf_counter() - started
    n = 1 if warmup_s > SLOW_SHAPE_SECONDS else reps
    times = []
    count = 0
    for _ in range(n):
        started = time.perf_counter()
        count = len(fn()["z"])
        times.append((time.perf_counter() - started) * 1000.0)
    times.sort()
    return {
        "median_ms": round(times[len(times) // 2], 3),
        "p95_ms": round(times[min(len(times) - 1, int(0.95 * len(times)))], 3),
        "warmup_ms": round(warmup_s * 1000.0, 3),
        "repetitions": n,
        "result_count": count,
    }


def _measure_shape_rss(args: argparse.Namespace, shape: str) -> dict[str, float]:
    q = query_store(args.store)
    # A factory, so measure_shape_rss owns the only reference (issue #241).
    return _query_shapes.measure_shape_rss(lambda: _patterns(q, args.store), shape)


def _source_rows(manifest: Path) -> dict[str, dict[str, str]]:
    with open(manifest, newline="") as fh:
        return {row["analysis_id"]: row for row in csv.DictReader(fh, delimiter="\t")}


def _source_bulk_seconds(source: Path, reps: int) -> dict[str, Any]:
    """Parse one Analysis's source the way a per-file consumer would: only the
    columns a z/se/eaf answer needs, into numpy arrays."""
    with gzip.open(source, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
    columns = {}
    for role, names in SOURCE_COLUMNS.items():
        found = next((n for n in names if n in header), None)
        if found is None:
            raise SystemExit(f"{source}: no {role} column among {names}")
        columns[role] = found
    times = []
    rows = 0
    for _ in range(reps + 1):
        started = time.perf_counter()
        frame = pd.read_csv(
            source, sep="\t", usecols=list(columns.values()),
            dtype={columns["chrom"]: str, columns["effect_allele"]: "category",
                   columns["other_allele"]: "category"},
            na_values=["NA"],
        )
        z = frame[columns["beta"]].to_numpy() / frame[columns["se"]].to_numpy()
        rows = int(np.isfinite(z).sum())
        times.append(time.perf_counter() - started)
    times = sorted(times[1:])
    return {"file": source.name, "bytes": source.stat().st_size, "rows": rows,
            "median_s": round(times[len(times) // 2], 2), "repetitions": reps}


def _file_count(path: Path) -> int:
    if path.is_file():
        return 1
    return sum(len(files) for _, _, files in os.walk(path))


def _components(store: Path) -> list[dict[str, Any]]:
    """Bytes and files by component: each Dense and Overflow plane, the top-hit
    tiers of each component, the shared variant axis files."""
    parts: list[tuple[str, str, Path]] = []
    dense_zarr = store / "dense" / "data.zarr"
    parts += [("dense", f"dense/data.zarr/{p.name}", p)
              for p in sorted(dense_zarr.iterdir()) if p.is_dir()]
    parts += [("dense", f"dense/{p.name}", p)
              for p in sorted((store / "dense").iterdir()) if p.is_file()]
    ragged = store / "data.zarr" / "ragged"
    parts += [("overflow", f"data.zarr/ragged/{p.name}", p)
              for p in sorted(ragged.iterdir()) if p.is_dir()]
    parts += [("overflow", f"data.zarr/{p.name}", p)
              for p in sorted((store / "data.zarr").iterdir()) if p.is_dir() and p != ragged]
    parts += [("shared", p.name, p) for p in sorted(store.iterdir()) if p.is_file()]
    return [{"part": part, "component": name, "bytes": _dir_bytes(path),
             "files": _file_count(path)} for part, name, path in parts]


def _build_records(records: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for record_path in sorted(records.glob("*.json")):
        record = json.loads(record_path.read_text())
        step = record.get("step", record_path.stem)
        out[step] = {
            "elapsed_seconds": float(record["elapsed_seconds"]),
            "start_time": record["start_time"],
            "end_time": record["end_time"],
            "opengwasdb_rev": record.get("opengwasdb_rev"),
        }
        argv = record.get("argv") or []
        if isinstance(argv, list) and "--n-workers" in argv:
            out[step]["n_workers"] = int(argv[argv.index("--n-workers") + 1])
    return out


def _analysis_summary(store: Path) -> dict[str, Any]:
    rows = _report.analysis_rows(store)
    hits = _report.numeric_column(rows, "n_hits_5e8")
    sizes = _report.numeric_column(rows, "sample_size")
    return {
        "n_analyses": len(rows),
        "stored_effect_scale": _report.tally(rows, "stored_effect_scale"),
        "assigned_ancestry": _report.tally(rows, "assigned_ancestry"),
        "eaf_orientation": _report.tally(rows, "eaf_orientation"),
        "n_publications": len({r["publication_pmid"] for r in rows}),
        "sample_size": {"min": float(sizes.min()), "median": float(np.median(sizes)),
                        "max": float(sizes.max())},
        "n_hits_5e8": _report.hits_summary(hits),
    }


def _component_of(q: Any, variant_index: int) -> str:
    return "dense" if q._shared_is_on_panel(int(variant_index)) else "overflow"


def known_loci(q: Any, table: dict[int, dict[str, Any]], by_id: dict[str, int]) -> list[dict]:
    """Look each locus up and orient the effect to its published risk allele."""
    out = []
    for analysis_id, alid, rsid, risk, gene, note in KNOWN_LOCI:
        common = {
            "analysis_id": analysis_id, "label": table[by_id[analysis_id]]["analysis_label"],
            "alid": alid, "expected_rsid": rsid, "gene": gene, "note": note,
            "expected_genome_wide": rsid not in MODEST,
        }
        oriented = _report.oriented_locus(q, alid, analysis_id, risk)
        if oriented is None:
            out.append({**common, "found": False})
            continue
        out.append({
            **common,
            "found": True,
            "component": _component_of(q, oriented["variant_index"]),
            "risk_allele": risk,
            **oriented,
        })
    return out


def phewas_top(q: Any, table: dict[int, dict[str, Any]], n: int = 8) -> list[dict]:
    return _report.phewas_top(q, table, PHEWAS_VARIANTS, n)


def mr_with_components(q: Any, by_id: dict[str, int], exposure: str, outcome: str) -> dict:
    started = time.perf_counter()
    result = run_mr(q, by_id, exposure, outcome)
    result["elapsed_s"] = round(time.perf_counter() - started, 2)
    for instrument in result["instruments"]:
        record = q._variant_axis.by_identifier(instrument["alid"])
        instrument["component"] = _component_of(q, record.variant_index)
    result["n_instruments_overflow"] = sum(
        i["component"] == "overflow" for i in result["instruments"]
    )
    return result


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    _query_shapes.add_common_args(ap)
    return ap


def main() -> None:
    started_open = time.perf_counter()
    args, q, plan = _query_shapes.start_benchmark(_parser(), _measure_shape_rss)
    open_seconds = time.perf_counter() - started_open
    table = q.analyses_table()
    by_id = {row["analysis_id"]: index for index, row in table.items()}
    n_analyses, n_variants = len(table), _n_shared_variants(args.store)
    manifest = json.loads((args.store / "manifest.json").read_text())
    hybrid = manifest["provenance"]["hybrid"]

    patterns = _patterns(q, args.store)
    timings = []
    for name, fn in patterns.items():
        timing = _time_shape(fn, args.reps)
        timings.append({"query": name, **timing})
        print(f"{name:42s} median={timing['median_ms']:12.2f} ms  "
              f"reps={timing['repetitions']}  count={timing['result_count']:,}", flush=True)

    memory = []
    if not args.skip_rss:
        for name in patterns:
            record = run_probe(name, ["--store", str(args.store), "--manifest",
                                      str(args.manifest), "--records", str(args.records),
                                      "--output", str(args.output)])
            memory.append(record)
            print(f"{name:42s} peak={record['peak_mb']:9.1f} MB  "
                  f"delta={record['delta_mb']:9.1f} MB", flush=True)

    sources = _source_rows(args.manifest)
    absent = [r["source_file"] for r in sources.values() if not Path(r["source_file"]).exists()]
    if absent:
        raise SystemExit(f"{len(absent)} source file(s) missing, e.g. {absent[0]}; no honest ratio")
    raw_bytes = sum(Path(r["source_file"]).stat().st_size for r in sources.values())
    store_bytes = _dir_bytes(args.store)

    source_bulk = _source_bulk_seconds(Path(sources[EXPOSURE]["source_file"]),
                                       reps=min(args.reps, 3))
    print(f"source bulk parse: {source_bulk['median_s']} s", flush=True)

    loci = known_loci(q, table, by_id)
    for row in loci:
        if row["found"]:
            print(f"{row['gene']:10s} {row['expected_rsid']:11s} {row['analysis_id']:14s} "
                  f"{row['component']:8s} z(risk)={row['z_risk_allele']:8.2f} "
                  f"-log10p={row['neglog10_p']:.0f}", flush=True)
        else:
            print(f"{row['gene']:10s} {row['expected_rsid']:11s} NOT FOUND", flush=True)
    mr = [mr_with_components(q, by_id, e, o) | {"note": note} for e, o, note in MR_PAIRS]
    for r in mr:
        print(f"MR {r['exposure_id']} -> {r['outcome_id']}: beta={r['ivw_beta']:.3f} "
              f"se={r['ivw_se']:.3f} p={r['ivw_pval']:.1e} n={r['n_instruments']} "
              f"(overflow {r['n_instruments_overflow']}) in {r['elapsed_s']} s", flush=True)

    result = {
        "dataset": {
            "store": str(args.store), "n_variants": n_variants, "n_analyses": n_analyses,
            "n_panel_variants": int(hybrid["n_panel"]),
            "n_off_panel_variants": int(hybrid["n_off_panel"]),
            "n_overflow_associations": int(hybrid["n_overflow_associations"]),
            "reference_assembly": plan.reference_assembly,
            "format_version": plan.format_version, "encoding": plan.encoding.to_manifest(),
            "completion_state": getattr(plan, "completion_state", None),
            "chunk_shape": hybrid["chunk_shape"],
            "variant_reference": manifest["provenance"].get("variant_reference"),
            "open_seconds": round(open_seconds, 2),
        },
        "analyses": _analysis_summary(args.store),
        "storage": _report.storage_summary(
            store_bytes=store_bytes, source_bytes=raw_bytes, n_source_files=len(sources),
            components=_components(args.store),
            extra={"store_files": _file_count(args.store)},
        ),
        "build": _build_records(args.records),
        "selection": _report.selection_summary(
            exposure=EXPOSURE, phewas_alid=PHEWAS_ALID, region=REGION,
            random_lookup_shapes=_query_shapes.RANDOM_LOOKUP_SHAPES,
            extra={"overflow_heavy": OVERFLOW_HEAVY, "off_axis_seed": OFF_AXIS_SEED},
        ),
        "timings": timings,
        "memory": memory,
        "reps": args.reps,
        "slow_shape_seconds": SLOW_SHAPE_SECONDS,
        "source_bulk": source_bulk,
        "known_loci": loci,
        "phewas_top": phewas_top(q, table),
        "mr": mr,
        "labels": {
            a: table[by_id[a]]["analysis_label"]
            for a in {x for pair in MR_PAIRS for x in pair[:2]} | {EXPOSURE, OVERFLOW_HEAVY}
        },
        **provenance(),
    }
    write_artifact(args.output, result)


if __name__ == "__main__":
    main()
