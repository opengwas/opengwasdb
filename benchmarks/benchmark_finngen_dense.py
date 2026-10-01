#!/usr/bin/env python3
"""Benchmark and sanity-check the FinnGen R13 full Dense Store (OGS-00016).

Measures the same query shapes, on the same fresh-interpreter RSS probe, as
`benchmark_ukbb_dense.py`, so the two full-scale reports are comparable, and
adds the checks a FinnGen release can be held to that a UK Biobank one cannot:

* storage against the source `.gz` files, by store component;
* the build's own step records (build / top-hits / overview / validate);
* the bulk shape against the shape the source is laid out for, a single
  `pd.read_csv` of the same Analysis's file;
* known-locus checks: the lead variants of well-established associations must
  be recovered with the published risk allele and genome-wide significance;
* PheWAS of two pleiotropic variants across all Analyses;
* three IVW Mendelian randomisation pairs run end to end through the store.

Cell-by-cell agreement with the source files is a separate, heavier harness:
`validate_finngen_source_fidelity.py`.

Usage:
  pixi run -e dev python benchmarks/benchmark_finngen_dense.py \
      --store /data/opengwasdb/stores/OGS-00016/store.opengwasdb \
      --source-dir /data/opengwasdb/raw/finngen-r13/releases/r13-full/source \
      --records /data/opengwasdb/stores/OGS-00016/records \
      --output docs/benchmark-output/opengwasdb_ogs00016_finngen_benchmark.json
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import erfc, log_ndtr

from benchmarks import _query_shapes
from benchmarks._artifact import provenance, write_artifact
from benchmarks._rss import run_probe
from benchmarks.benchmark_ukbb_dense import _clump, _dir_bytes, _median_ms
from opengwasdb.query import query_store

PREFIX = "finngen-r13-"
EXPOSURE = PREFIX + "T2D"
PHEWAS_ALID = "10:112998590:C:T"  # rs7903146, TCF7L2
# TCF7L2 (chr10:112.95-113.17 Mb, GRCh38) and 0.3 Mb either side.
REGION = ("10", 112_500_000, 113_500_000)
CLUMP_KB = 1000

# (analysis, alid, rsid, risk allele, locus, why it is expected)
KNOWN_LOCI = [
    ("T2D", "10:112998590:C:T", "rs7903146", "T", "TCF7L2", "strongest common T2D locus"),
    ("T2D", "3:12351626:C:G", "rs1801282", "C", "PPARG", "Pro12Ala, protective Ala allele"),
    ("BMI_IRN", "16:53786615:A:T", "rs9939609", "A", "FTO", "strongest common BMI locus"),
    ("BMI_IRN", "18:60183864:C:T", "rs17782313", "C", "MC4R", "BMI-raising allele"),
    ("G6_ALZHEIMER", "19:44908684:C:T", "rs429358", "C", "APOE", "epsilon-4 defining allele"),
    ("G6_ALZHEIMER", "19:44892362:A:G", "rs2075650", "G", "TOMM40", "APOE-linked risk allele"),
    ("I9_CHD", "9:22125504:C:G", "rs1333049", "C", "CDKN2B-AS1", "9p21 coronary risk locus"),
    ("I9_CHD", "9:22098575:A:G", "rs4977574", "G", "CDKN2B-AS1", "9p21 coronary risk locus"),
]
# rs1801282 is a modest-effect variant; the others must clear genome-wide significance.
MODEST = {"rs1801282"}

MR_PAIRS = [
    ("BMI_IRN", "T2D", "adiposity -> type 2 diabetes: a large, well-replicated causal effect"),
    ("BMI_IRN", "I9_CHD", "adiposity -> coronary disease: a smaller, mediated effect"),
    ("T2D", "I9_CHD", "type 2 diabetes -> coronary disease: a modest causal effect"),
]

PHEWAS_VARIANTS = [
    ("10:112998590:C:T", "rs7903146", "TCF7L2"),
    ("19:44908684:C:T", "rs429358", "APOE"),
]


def _analysis_index(q: Any) -> tuple[dict[int, dict[str, Any]], dict[str, int]]:
    table = q.analyses_table()
    return table, {row["analysis_id"]: index for index, row in table.items()}


def _measure_shape_rss(args: argparse.Namespace, shape: str) -> dict[str, float]:
    q = query_store(args.store)
    analyses = q.analyses_table()
    # Pass a factory, not a mapping: measure_shape_rss must own the mapping so
    # its drop-and-collect actually releases the shapes this probe is not
    # measuring (see measure_shape_rss, issue #241).
    return _query_shapes.measure_shape_rss(
        lambda: _query_shapes.build_query_patterns(
            q, analyses, int(q._root["z"].shape[0]), len(analyses),
            exposure=EXPOSURE, phewas_alid=PHEWAS_ALID, region=REGION,
        ),
        shape,
    )


def _source_bulk_seconds(source: Path, reps: int) -> dict[str, Any]:
    """Parse one Analysis's source the way a per-file consumer would: only the
    columns a z/se/eaf answer needs, into numpy arrays."""
    times = []
    rows = 0
    for _ in range(reps + 1):
        started = time.perf_counter()
        frame = pd.read_csv(
            source, sep="\t", usecols=["#chrom", "pos", "ref", "alt", "beta", "sebeta", "af_alt"],
            dtype={"#chrom": str, "ref": "category", "alt": "category"},
        )
        z = frame.beta.to_numpy() / frame.sebeta.to_numpy()
        rows = int(np.isfinite(z).sum())
        times.append(time.perf_counter() - started)
    times = sorted(times[1:])
    return {"file": source.name, "bytes": source.stat().st_size, "rows": rows,
            "median_s": round(times[len(times) // 2], 2), "repetitions": reps}


def _components(store: Path) -> list[dict[str, Any]]:
    """Bytes by store component: each statistic plane, the top-hit tiers, the axis."""
    zarr_root = store / "data.zarr"
    parts = [(f"data.zarr/{p.name}", p) for p in sorted(zarr_root.iterdir()) if p.is_dir()]
    parts += [(p.name, p) for p in sorted(store.iterdir()) if p.is_file()]
    return [{"component": name, "bytes": _dir_bytes(path)} for name, path in parts]


def _build_records(records: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for step in ("build", "top-hits", "overview", "validate"):
        record = json.loads((records / f"{step}.json").read_text())
        out[step] = {
            "elapsed_seconds": record["elapsed_seconds"],
            "start_time": record["start_time"],
            "end_time": record["end_time"],
            "opengwasdb_rev": record.get("opengwasdb_rev"),
        }
        if step == "build":
            argv = record["argv"]
            out[step]["n_workers"] = int(argv[argv.index("--n-workers") + 1])
    return out


def _analysis_summary(store: Path) -> dict[str, Any]:
    with open(store / "analyses.tsv", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))

    def numbers(column: str) -> np.ndarray:
        return np.array([float(r[column]) for r in rows if r[column] != ""])

    scales: dict[str, int] = {}
    for row in rows:
        scales[row["stored_effect_scale"]] = scales.get(row["stored_effect_scale"], 0) + 1
    hits = numbers("n_hits_5e8")
    cases = numbers("n_cases")
    return {
        "n_analyses": len(rows),
        "stored_effect_scale": scales,
        "n_wide_definition": sum(r["analysis_id"].endswith("_WIDE") for r in rows),
        "n_cases": {"min": float(cases.min()), "median": float(np.median(cases)),
                    "max": float(cases.max())},
        "n_hits_5e8": {"median": float(np.median(hits)), "max": float(hits.max()),
                       "zero": int((hits == 0).sum()), "total": float(hits.sum())},
        "sample_size": {"min": float(numbers("sample_size").min()),
                        "max": float(numbers("sample_size").max())},
    }


def known_loci(q: Any, by_id: dict[str, int], table: dict[int, dict[str, Any]]) -> list[dict]:
    """Look each locus up and orient the effect to its published risk allele."""
    out = []
    for analysis, alid, rsid, risk, gene, note in KNOWN_LOCI:
        analysis_id = PREFIX + analysis
        look = q.lookup([alid], [analysis_id])
        record = q._variant_axis.by_index(int(look["variant_index"][0]))
        z, se, eaf = float(look["z"][0]), float(look["se"][0]), float(look["eaf"][0])
        sign = 1.0 if record.effect_allele == risk else -1.0
        risk_freq = eaf if record.effect_allele == risk else 1.0 - eaf
        neglog10_p = float(-(log_ndtr(-abs(z)) + np.log(2.0)) / np.log(10.0))
        out.append({
            "analysis_id": analysis_id, "label": table[by_id[analysis_id]]["analysis_label"],
            "alid": alid, "rsid": record.rsid, "expected_rsid": rsid, "gene": gene, "note": note,
            "risk_allele": risk, "effect_allele": record.effect_allele,
            "z": z, "se": se, "beta_risk_allele": sign * z * se, "z_risk_allele": sign * z,
            "risk_allele_frequency": risk_freq, "neglog10_p": neglog10_p,
            "expected_genome_wide": rsid not in MODEST,
        })
    return out


def phewas_top(q: Any, table: dict[int, dict[str, Any]], n: int = 8) -> list[dict]:
    out = []
    for alid, rsid, gene in PHEWAS_VARIANTS:
        res = q.phewas(alid)
        order = np.argsort(-np.abs(res["z"]))[:n]
        out.append({
            "alid": alid, "rsid": rsid, "gene": gene, "n_analyses": int(len(res["z"])),
            "n_genome_wide": int((np.abs(res["z"]) > 5.4520).sum()),
            "top": [
                {"analysis_id": table[int(res["analysis_index"][i])]["analysis_id"],
                 "label": table[int(res["analysis_index"][i])]["analysis_label"],
                 "z": float(res["z"][i])}
                for i in order
            ],
        })
    return out


def run_mr(q: Any, by_id: dict[str, int], exposure: str, outcome: str) -> dict[str, Any]:
    """Distance-clumped IVW. Every Analysis is oriented to the store's canonical
    effect allele, so exposure and outcome betas need no harmonisation."""
    hits = q.top_hits(analysis_id=exposure, threshold=_query_shapes.GENOME_WIDE)
    raw = []
    for vi, z, se in zip(hits["variant_index"], hits["z"], hits["se"], strict=True):
        rec = q._variant_axis.by_index(int(vi))
        if rec is not None:
            raw.append({"alid": rec.alid, "chrom": rec.chromosome, "pos": int(rec.position),
                        "z_exp": float(z), "se_exp": float(se)})
    clumped = _clump(raw, CLUMP_KB)
    look = q.lookup([c["alid"] for c in clumped], [exposure, outcome])
    per: dict[int, dict[str, float]] = {}
    for vi, ai, z, se in zip(
        look["variant_index"], look["analysis_index"], look["z"], look["se"], strict=True
    ):
        d = per.setdefault(int(vi), {})
        side = "exp" if int(ai) == by_id[exposure] else "out"
        d[f"z_{side}"], d[f"se_{side}"] = float(z), float(se)
    instruments = []
    for vi, d in per.items():
        if {"z_exp", "z_out"} <= d.keys():
            rec = q._variant_axis.by_index(vi)
            instruments.append({
                "alid": rec.alid, "chrom": rec.chromosome, "pos": int(rec.position),
                "beta_exp": d["z_exp"] * d["se_exp"], "se_exp": d["se_exp"],
                "beta_out": d["z_out"] * d["se_out"], "se_out": d["se_out"],
            })
    if not instruments:
        raise SystemExit(f"{exposure} -> {outcome}: no instruments; refusing an empty MR")

    be = np.array([i["beta_exp"] for i in instruments])
    bo = np.array([i["beta_out"] for i in instruments])
    so = np.array([i["se_out"] for i in instruments])
    se_exp = np.array([i["se_exp"] for i in instruments])
    weight = be**2 / so**2
    beta = float(np.sum(be * bo / so**2) / np.sum(weight))
    se_fixed = float(np.sqrt(1.0 / np.sum(weight)))
    ratio = bo / be
    q_stat = float(np.sum(weight * (ratio - beta) ** 2))
    df = len(instruments) - 1
    se_random = se_fixed * max(1.0, float(np.sqrt(q_stat / df))) if df > 0 else se_fixed
    z_ivw = beta / se_random
    return {
        "exposure_id": exposure, "outcome_id": outcome, "clump_kb": CLUMP_KB,
        "n_instruments_raw": len(raw), "n_instruments": len(instruments),
        "ivw_beta": beta, "ivw_se_fixed": se_fixed, "ivw_se": se_random,
        "ivw_pval": float(erfc(abs(z_ivw) / np.sqrt(2.0))),
        "cochran_q": q_stat, "q_df": df,
        "sign_concordant_fraction": float(np.mean(np.sign(be) == np.sign(bo))),
        "median_instrument_F": float(np.median((be / se_exp) ** 2)),
        "instruments": instruments,
    }


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--source-dir", type=Path, required=True)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    _query_shapes.add_common_args(ap)
    return ap


def main() -> None:
    args, q, plan = _query_shapes.start_benchmark(_parser(), _measure_shape_rss)
    table, by_id = _analysis_index(q)
    n_analyses, n_variants = len(table), int(q._root["z"].shape[0])

    patterns = _query_shapes.build_query_patterns(
        q, table, n_variants, n_analyses,
        exposure=EXPOSURE, phewas_alid=PHEWAS_ALID, region=REGION,
    )
    timings = []
    for name, fn in patterns.items():
        med, p95, count = _median_ms(fn, args.reps)
        timings.append({"query": name, "median_ms": round(med, 3), "p95_ms": round(p95, 3),
                        "result_count": count})
        print(f"{name:42s} median={med:10.2f} ms  count={count:,}", flush=True)

    memory = []
    if not args.skip_rss:
        for name in patterns:
            record = run_probe(name, ["--store", str(args.store), "--source-dir",
                                      str(args.source_dir), "--records", str(args.records),
                                      "--output", str(args.output)])
            memory.append(record)
            print(f"{name:42s} peak={record['peak_mb']:9.1f} MB  "
                  f"delta={record['delta_mb']:9.1f} MB", flush=True)

    sources = [args.source_dir / f"finngen_R13_{a.removeprefix(PREFIX)}.gz" for a in by_id]
    absent = [s for s in sources if not s.exists()]
    if absent:
        raise SystemExit(f"{len(absent)} source file(s) missing, e.g. {absent[0]}; no honest ratio")
    raw_bytes = sum(s.stat().st_size for s in sources)
    store_bytes = _dir_bytes(args.store)
    n_cells = n_variants * n_analyses

    exposure_source = args.source_dir / f"finngen_R13_{EXPOSURE.removeprefix(PREFIX)}.gz"
    source_bulk = _source_bulk_seconds(exposure_source, reps=min(args.reps, 3))
    print(f"source bulk parse: {source_bulk['median_s']} s", flush=True)

    loci = known_loci(q, by_id, table)
    for row in loci:
        print(f"{row['gene']:10s} {row['rsid']:11s} {row['analysis_id']:28s} "
              f"z(risk)={row['z_risk_allele']:8.2f} -log10p={row['neglog10_p']:.0f}", flush=True)
    mr = [run_mr(q, by_id, PREFIX + e, PREFIX + o) | {"note": note} for e, o, note in MR_PAIRS]
    for r in mr:
        print(f"MR {r['exposure_id']} -> {r['outcome_id']}: beta={r['ivw_beta']:.3f} "
              f"se={r['ivw_se']:.3f} p={r['ivw_pval']:.1e} n={r['n_instruments']}", flush=True)

    result = {
        "dataset": {
            "store": str(args.store), "n_variants": n_variants, "n_analyses": n_analyses,
            "n_associations": n_cells, "reference_assembly": plan.reference_assembly,
            "format_version": plan.format_version, "encoding": plan.encoding.to_manifest(),
            "completion_state": getattr(plan, "completion_state", None),
            "chunk_shape": json.loads((args.store / "manifest.json").read_text())["provenance"][
                "dense"]["chunk_shape"],
        },
        "analyses": _analysis_summary(args.store),
        "storage": {
            "store_bytes": store_bytes, "store_gb": round(store_bytes / 1e9, 2),
            "source_bytes": raw_bytes, "source_gb": round(raw_bytes / 1e9, 2),
            "n_source_files": len(sources),
            "compression_ratio": round(raw_bytes / store_bytes, 2),
            "bytes_per_association": store_bytes / n_cells,
            "components": _components(args.store),
        },
        "build": _build_records(args.records),
        "selection": {
            "exposure": EXPOSURE, "phewas_alid": PHEWAS_ALID,
            "region": {"chrom": REGION[0], "start": REGION[1], "end": REGION[2]},
            "random_lookup_shapes": _query_shapes.RANDOM_LOOKUP_SHAPES,
        },
        "timings": timings,
        "memory": memory,
        "source_bulk": source_bulk,
        "known_loci": loci,
        "phewas_top": phewas_top(q, table),
        "mr": mr,
        "labels": {a: table[i]["analysis_label"] for a, i in by_id.items()
                   if a in {PREFIX + x for pair in MR_PAIRS for x in pair[:2]}},
        **provenance(),
    }
    write_artifact(args.output, result)


if __name__ == "__main__":
    main()
