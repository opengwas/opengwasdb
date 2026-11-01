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
* the four Dense-only twins of the region and random-lookup shapes, so the
  report can say what a reader who queries only the reference axis pays rather
  than attributing every Hybrid overhead to the Dense Component;
* storage by component, with file counts, against the source GWAS-SSF files;
* the build's own step records;
* the bulk shape against a single `pd.read_csv` of the same Analysis's source;
* known-locus checks, each naming the component that answered;
* PheWAS of two pleiotropic variants across all Analyses;
* three IVW Mendelian randomisation pairs run end to end through the store,
  each instrument tagged with the component that holds it, and each estimate
  reported for every instrument and separately for the observed/imputed
  strata;
* a ±500 kb regional profile around three lead variants, every row carrying
  the Association Status and the component that answered it.

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
from benchmarks.benchmark_finngen_dense import CLUMP_KB
from benchmarks.benchmark_ukbb_dense import _clump, _dir_bytes
from opengwasdb.query import query_store

# Type 2 diabetes (Xue 2018) and TCF7L2, the FinnGen benchmark's exposure and
# region, so the shared shapes ask the same biological question of both stores.
EXPOSURE = "GCST006867"
PHEWAS_ALID = "10:112998590:C:T"  # rs7903146, TCF7L2; on the reference axis
REGION = ("10", 112_500_000, 113_500_000)
# The Analysis with the largest Ragged Overflow, read from the Overflow's CSR
# offsets: 15.0 M off-axis associations, prostate cancer. The earlier choice,
# GCST90502911 (a whole-genome-sequencing BMI study, 75.7 M), left the Store
# when Analyses with no MAF floor were excluded (opengwas/opengwasdb-stores#203).
OVERFLOW_HEAVY = "GCST90274713"
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

# One regional profile per lead variant, each a window of this half-width either
# side of it: the three loci the MR pairs and known-locus checks already name,
# so a profile can be read against an estimate the harness reports anyway.
REGIONAL_WINDOW_KB = 500
REGIONAL_LOCI = [
    {"analysis_id": "GCST006867", "gene": "TCF7L2", "lead_alid": "10:112998590:C:T"},
    {"analysis_id": "GCST90029007", "gene": "FTO", "lead_alid": "16:53786615:A:T"},
    {"analysis_id": "GCST90239658", "gene": "PCSK9", "lead_alid": "1:55039974:G:T"},
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
    random_alids, random_analyses = _query_shapes.resolve_axis_selections(
        q._variant_axis, analyses, n_variants, len(analyses)
    )
    patterns = _query_shapes.common_query_patterns(
        q,
        exposure=EXPOSURE, phewas_alid=PHEWAS_ALID, region=REGION,
        random_alids=random_alids, random_analyses=random_analyses,
    )
    # The same region, exposure and Analyses asked of the Dense Component alone:
    # what a reader who only queries the reference axis pays, beside the Hybrid
    # answer the common shapes time.
    patterns.update(_query_shapes.dense_component_query_patterns(
        q, exposure=EXPOSURE, region=REGION, random_analyses=random_analyses,
    ))
    off_axis = _off_axis_alid(q, n_variants)
    patterns["phewas_off_axis"] = lambda: q.phewas(off_axis)
    patterns["bulk_overflow_heavy"] = lambda: q.analysis(OVERFLOW_HEAVY)
    # The Dense Component of the same Analysis, which is the one shape whose
    # column carries many Dense EAF exception cells: the named bulk control for
    # the windowed-table regression (review round 3, finding 1).
    patterns["bulk_dense_exceptions"] = lambda: q._dense.analysis(OVERFLOW_HEAVY)
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


def mr_instruments(
    q: Any, by_id: dict[str, int], exposure: str, outcome: str
) -> tuple[list[dict], int]:
    """Distance-clumped instruments, each keeping both sides' Association Status.

    Instrument selection is the one the FinnGen harness's `run_mr` makes: the
    exposure's genome-wide significant top hits, `_clump` at `CLUMP_KB`, then
    one lookup for both Analyses. What is added here is the status of the
    exposure cell and of the outcome cell beside each instrument, which is what
    lets one run be split into observed and imputed strata at all -- the status
    is a property of the cell, and a batch estimate throws it away.

    Returns the instruments and how many raw hits the clump started from.
    """
    hits = q.top_hits(analysis_id=exposure, threshold=_query_shapes.GENOME_WIDE)
    raw = []
    for vi, z, se in zip(hits["variant_index"], hits["z"], hits["se"], strict=True):
        record = q._variant_axis.by_index(int(vi))
        if record is not None:
            raw.append({"alid": record.alid, "chrom": record.chromosome,
                        "pos": int(record.position), "z_exp": float(z),
                        "se_exp": float(se)})
    clumped = _clump(raw, CLUMP_KB)
    look = q.lookup([c["alid"] for c in clumped], [exposure, outcome])
    per: dict[int, dict[str, Any]] = {}
    for vi, ai, z, se, status in zip(
        look["variant_index"], look["analysis_index"], look["z"], look["se"],
        look["association_status"], strict=True,
    ):
        cell = per.setdefault(int(vi), {})
        side = "exp" if int(ai) == by_id[exposure] else "out"
        cell[f"z_{side}"], cell[f"se_{side}"], cell[f"status_{side}"] = (
            float(z), float(se), str(status)
        )
    instruments = []
    for vi, cell in per.items():
        if {"z_exp", "z_out"} <= cell.keys():
            record = q._variant_axis.by_index(vi)
            instruments.append({
                "alid": record.alid, "chrom": record.chromosome,
                "pos": int(record.position),
                "beta_exp": cell["z_exp"] * cell["se_exp"], "se_exp": cell["se_exp"],
                "beta_out": cell["z_out"] * cell["se_out"], "se_out": cell["se_out"],
                "status_exp": cell["status_exp"], "status_out": cell["status_out"],
            })
    return instruments, len(raw)


def mr_with_components(q: Any, by_id: dict[str, int], exposure: str, outcome: str) -> dict:
    """One IVW pair end to end, each instrument tagged with its component.

    The whole-set estimate is `_hybrid_report.ivw` -- fixed-effect, the same
    function every stratum uses -- so `ivw_beta`, `ivw_se` and `ivw_pval`
    describe all instruments under one convention. The random-effects SE the
    earlier artifact reported as `ivw_se` is kept beside it as `ivw_se_random`
    (and `cochran_q`/`q_df` are unchanged), so the heterogeneity it inflates for
    is still visible rather than dropped. `strata` is the same estimate per
    observed/imputed stratum, `None` where a stratum has fewer than
    `_report.MIN_STRATUM_INSTRUMENTS`.
    """
    started = time.perf_counter()
    instruments, n_raw = mr_instruments(q, by_id, exposure, outcome)
    if not instruments:
        raise SystemExit(f"{exposure} -> {outcome}: no instruments; refusing an empty MR")
    for instrument in instruments:
        record = q._variant_axis.by_identifier(instrument["alid"])
        instrument["component"] = _component_of(q, record.variant_index)
    elapsed_s = round(time.perf_counter() - started, 2)

    overall = _report.ivw(instruments)
    be = np.array([i["beta_exp"] for i in instruments])
    bo = np.array([i["beta_out"] for i in instruments])
    so = np.array([i["se_out"] for i in instruments])
    se_exp = np.array([i["se_exp"] for i in instruments])
    weight = be**2 / so**2
    ratio = bo / be
    q_stat = float(np.sum(weight * (ratio - overall["beta"]) ** 2))
    df = len(instruments) - 1
    se_random = overall["se"] * max(1.0, float(np.sqrt(q_stat / df))) if df > 0 else overall["se"]
    return {
        "exposure_id": exposure, "outcome_id": outcome, "clump_kb": CLUMP_KB,
        "n_instruments_raw": n_raw, "n_instruments": len(instruments),
        "n_instruments_overflow": sum(
            i["component"] == "overflow" for i in instruments
        ),
        "ivw_beta": overall["beta"], "ivw_se": overall["se"],
        "ivw_pval": overall["p"], "ivw_se_fixed": overall["se"],
        "ivw_se_random": se_random,
        "cochran_q": q_stat, "q_df": df,
        "sign_concordant_fraction": float(np.mean(np.sign(be) == np.sign(bo))),
        "median_instrument_F": float(np.median((be / se_exp) ** 2)),
        "elapsed_s": elapsed_s,
        "instruments": instruments,
        "strata": _report.mr_strata(instruments),
    }


def _window(lead_alid: str) -> tuple[str, int, int]:
    """The `REGIONAL_WINDOW_KB` window either side of a lead ALID's position."""
    chromosome, position, *_alleles = lead_alid.split(":")
    half = REGIONAL_WINDOW_KB * 1000
    return (chromosome, int(position) - half, int(position) + half)


def regional_windows(q: Any) -> list[dict[str, Any]]:
    """One regional profile per `REGIONAL_LOCI` entry, timed end to end.

    Every shared-axis ALID in the window is looked up for the one Analysis, the
    Hybrid question the report's regional shapes ask of a whole store. Each row
    is `[position, neglog10_p, status, component]`, so the plot says which cells
    the Dense Component answered and which came from the Overflow; the counts
    are the observed/imputed split the owner asked to see beside the estimate.
    """
    out = []
    for locus in REGIONAL_LOCI:
        chromosome, start, end = _window(locus["lead_alid"])
        started = time.perf_counter()
        alids = _query_shapes.regional_alids(q._variant_axis, (chromosome, start, end))
        result = q.lookup(alids, [locus["analysis_id"]])
        rows = []
        n_observed = n_imputed = 0
        for vi, z, status in zip(
            result["variant_index"], result["z"], result["association_status"], strict=True
        ):
            record = q._variant_axis.by_index(int(vi))
            status = str(status)
            n_observed += int(status == "observed")
            n_imputed += int(status == "imputed")
            rows.append([
                int(record.position),
                round(_report.neglog10_p(float(z)), 3),
                status,
                _component_of(q, int(vi)),
            ])
        out.append({
            "analysis_id": locus["analysis_id"], "gene": locus["gene"],
            "lead_alid": locus["lead_alid"], "chromosome": chromosome,
            "start": start, "end": end,
            "n_observed": n_observed, "n_imputed": n_imputed,
            "rows": rows,
            "elapsed_s": round(time.perf_counter() - started, 2),
        })
    return out


def _require_selected_analyses(by_id: dict[str, int]) -> None:
    """Refuse to run when an Analysis the selection names is not in the store.

    A whole-Analysis read of an absent Analysis returns no rows rather than
    raising, so a stale selection would time an empty read and report it.
    """
    selected = {EXPOSURE, OVERFLOW_HEAVY}
    selected |= {analysis for analysis, *_ in KNOWN_LOCI}
    selected |= {analysis for pair in MR_PAIRS for analysis in pair[:2]}
    selected |= {locus["analysis_id"] for locus in REGIONAL_LOCI}
    absent = sorted(selected - set(by_id))
    if absent:
        raise SystemExit(f"the store holds no Analysis {', '.join(absent)}; update the selection")


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
    _require_selected_analyses(by_id)
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
        for stratum, estimate in r["strata"].items():
            shown = ("too few instruments" if estimate is None
                     else f"beta={estimate['beta']:.3f} se={estimate['se']:.3f} "
                          f"p={estimate['p']:.1e} n={estimate['n']}")
            print(f"    {stratum:14s} {shown}", flush=True)

    windows = regional_windows(q)
    for window in windows:
        print(f"regional {window['gene']:8s} {window['analysis_id']:14s} "
              f"{window['chromosome']}:{window['start']}-{window['end']} "
              f"rows={len(window['rows'])} observed={window['n_observed']} "
              f"imputed={window['n_imputed']} in {window['elapsed_s']} s", flush=True)

    # A Hybrid root's HM3 index lives in its nested Dense Component (#271).
    hm3_extract = _query_shapes.report_hm3_extract(
        q, EXPOSURE, store_path=args.store, reps=args.reps
    )

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
        "hm3_extract": hm3_extract,
        "reps": args.reps,
        "slow_shape_seconds": SLOW_SHAPE_SECONDS,
        "source_bulk": source_bulk,
        "known_loci": loci,
        "phewas_top": phewas_top(q, table),
        "mr": mr,
        "regional_windows": windows,
        "labels": {
            a: table[by_id[a]]["analysis_label"]
            for a in {x for pair in MR_PAIRS for x in pair[:2]} | {EXPOSURE, OVERFLOW_HEAVY}
        },
        **provenance(),
    }
    write_artifact(args.output, result)


if __name__ == "__main__":
    main()
