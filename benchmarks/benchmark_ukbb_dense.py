#!/usr/bin/env python3
"""Benchmark the genome-wide ukb-b Dense store: query timings, storage vs VCF,
build time, and an MR IVW validation (self-reported high cholesterol -> heart
attack) with per-instrument scatter data.

Writes docs/benchmark-output/opengwasdb_ukbb_dense_benchmark.json, which the
companion QMD renders.

Usage:
  uv run python benchmarks/benchmark_ukbb_dense.py [--reps N] [--store PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import zarr

from opengwasdb.layouts.dense.top_hits import threshold_key, write_top_hit_indexes
from opengwasdb.query import query_store

STORE = Path("/local-scratch/data/opengwas/opengwasdb/ukb-b.opengwasdb")
MANIFEST = Path("/home/gh13047/repo/opengwasdb/data/ukb-b/manifest.tsv")
BUILD_LOG = Path("/home/gh13047/repo/opengwasdb/data/ukb-b/build.log")
OUTPUT = Path(
    "/home/gh13047/repo/opengwasdb/docs/benchmark-output/"
    "opengwasdb_ukbb_dense_benchmark.json"
)

# MR: exposure = self-reported high cholesterol (LDL-raising proxy),
# outcome = doctor-diagnosed heart attack (CHD). Both binary (log-OR scale).
EXPOSURE = "ukb-b-10912"
OUTCOME = "ukb-b-11590"
CLUMP_KB = 1000  # greedy distance-based pruning window (approx. independence)

# Regional query window: chr19 44.5-45.5 Mb spans the APOE/APOC cluster.
REGION = ("19", 44_500_000, 45_500_000)

PRE_EAF_TOP_HIT_MS = 1.17
EAF_REGRESSION_TOP_HIT_MS = 86.6
PRE_EAF_GLOBAL_TOP_HIT_MS = 488.0
EAF_REGRESSION_GLOBAL_TOP_HIT_MS = 7_129.0


def _median_ms(fn, reps: int) -> tuple[float, float, int]:
    fn()  # warm-up
    times = []
    count = 0
    for _ in range(reps):
        t0 = time.perf_counter()
        res = fn()
        times.append((time.perf_counter() - t0) * 1000.0)
        count = len(res["z"])
    times.sort()
    med = times[len(times) // 2]
    p95 = times[min(len(times) - 1, int(0.95 * len(times)))]
    return med, p95, count


def _legacy_analysis_top_hits(q, analysis_index: int) -> dict[str, np.ndarray]:
    """Previous behavior: materialise the global tier, then filter it."""
    result = q.top_hits(threshold=5e-8)
    keep = result["analysis_index"] == analysis_index
    return {name: values[keep] for name, values in result.items()}


def _top_hit_index_bytes(store: Path) -> int:
    return _dir_bytes(store / "data.zarr" / "top_hits")


def _repack_top_hits(store: Path, chunk_size: int) -> float:
    """Repack from the loosest existing tier, avoiding a dense-matrix rescan."""
    root = zarr.open_group(str(store / "data.zarr"), mode="r")
    group = root[f"top_hits/{threshold_key(5e-4)}"]
    rows = group["variant_index"][:]
    cols = group["analysis_index"][:]
    z_values = group["z"][:]
    se_values = group["se"][:]
    imputed = group["imputed"][:] if "imputed" in group else None
    eaf = group["eaf"][:] if "eaf" in group else None
    del group, root
    started = time.perf_counter()
    write_top_hit_indexes(
        store, rows, cols, z_values, se_values, imputed=imputed, eaf=eaf,
        chunk_size=chunk_size
    )
    return time.perf_counter() - started


def _top_hit_trial(store: Path, chunk_size: int, reps: int) -> dict:
    """Repack the index at `chunk_size`, then time and verify one read shape."""
    rebuild_seconds = _repack_top_hits(store, chunk_size)
    q = query_store(store)
    analysis_index = next(
        index for index, row in q.analyses_table().items() if row["analysis_id"] == EXPOSURE
    )
    started = time.perf_counter()
    first = q.top_hits(analysis_id=EXPOSURE, threshold=5e-8)
    first_ms = (time.perf_counter() - started) * 1_000
    median_ms, p95_ms, count = _median_ms(
        lambda query=q: query.top_hits(analysis_id=EXPOSURE, threshold=5e-8), reps
    )
    global_ms, _, _ = _median_ms(lambda query=q: query.top_hits(threshold=5e-8), reps)
    global_result = q.top_hits(threshold=5e-8)
    keep = global_result["analysis_index"] == analysis_index
    correct = all(np.array_equal(first[name], global_result[name][keep]) for name in first)
    q.close()
    print(f"top-hits chunk={chunk_size:,}: median={median_ms:.3f} ms global={global_ms:.1f} ms")
    return {
        "chunk_size": chunk_size,
        "first_read_ms": round(first_ms, 3),
        "median_ms": round(median_ms, 3),
        "p95_ms": round(p95_ms, 3),
        "global_median_ms": round(global_ms, 3),
        "result_count": count,
        "repack_seconds": round(rebuild_seconds, 3),
        "index_bytes": _top_hit_index_bytes(store),
        "matches_global_subset": correct,
    }


def run_top_hit_experiment(store: Path, output: Path, reps: int) -> None:
    """Evaluate narrow-slice chunks and update an existing benchmark result."""
    previous = json.loads(output.read_text())
    benchmark_timing = next(row for row in previous["timings"] if row["query"] == "tophits")
    trials = [_top_hit_trial(store, chunk_size, reps) for chunk_size in (1_024, 4_096, 16_384)]

    fastest = min(trial["median_ms"] for trial in trials)
    # Narrow reads are the priority; among configurations within 10% of the
    # fastest narrow read, retain the one that least penalises global reads.
    eligible = [trial for trial in trials if trial["median_ms"] <= fastest * 1.10]
    selected_trial = min(eligible, key=lambda trial: trial["global_median_ms"])
    selected_chunk = selected_trial["chunk_size"]
    # Leave the physical store in the measured winning configuration.
    if trials[-1]["chunk_size"] != selected_chunk:
        _repack_top_hits(store, selected_chunk)
    selected = selected_trial
    medians = [trial["median_ms"] for trial in trials]
    previous["top_hit_experiment"] = {
        "analysis_id": EXPOSURE,
        "threshold": 5e-8,
        "repetitions": reps,
        "pre_eaf_baseline_ms": PRE_EAF_TOP_HIT_MS,
        "eaf_regression_ms": EAF_REGRESSION_TOP_HIT_MS,
        "pre_eaf_global_filter_ms": PRE_EAF_GLOBAL_TOP_HIT_MS,
        "eaf_regression_global_filter_ms": EAF_REGRESSION_GLOBAL_TOP_HIT_MS,
        "selected_chunk_size": selected_chunk,
        "selected": selected,
        "speedup_over_global_filter": round(
            selected["global_median_ms"] / selected["median_ms"], 2
        ),
        "chunk_configuration_spread_percent": round(
            (max(medians) - min(medians)) / min(medians) * 100.0, 1
        ),
        "variance_note": (
            "Identical historical runs varied by 13% for bulk and 27% for top hits; "
            "changes below about 1.5x are not distinguished from noise at this repetition count."
        ),
        "unchanged_by_design": ["phewas", "regional"],
        "target_ms": 10.0,
        "meets_target": selected["median_ms"] < 10.0,
        "trials": trials,
    }
    benchmark_timing.update({
        "median_ms": selected["median_ms"], "p95_ms": selected["p95_ms"],
        "result_count": selected["result_count"],
    })
    output.write_text(json.dumps(previous, indent=2) + "\n")


def _dir_bytes(path: Path) -> int:
    out = subprocess.run(["du", "-sb", str(path)], capture_output=True, text=True, check=True)
    return int(out.stdout.split()[0])


def _raw_vcf_bytes(manifest: Path) -> tuple[int, int]:
    paths = []
    with open(manifest) as fh:
        next(fh)  # header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                paths.append(parts[1])
    total = 0
    for p in paths:
        try:
            total += Path(p).stat().st_size
        except OSError:
            pass
    return total, len(paths)


def _build_seconds(log: Path) -> float | None:
    if not log.exists():
        return None
    text = log.read_text()
    timestamp = r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:[.,]\d+)?)"
    starts = re.findall(rf"^{timestamp} INFO.*Pass 1:", text, re.M)
    ends = re.findall(rf"^{timestamp} INFO.*Build complete", text, re.M)
    if not starts or not ends:
        return None
    start = datetime.fromisoformat(starts[0].replace(",", "."))
    end = datetime.fromisoformat(ends[-1].replace(",", "."))
    return (end - start).total_seconds()


def _clump(instr: list[dict], kb: int) -> list[dict]:
    """Greedy distance pruning: keep the strongest |z| instrument, drop others
    within +/- kb on the same chromosome, repeat."""
    window = kb * 1000
    remaining = sorted(instr, key=lambda d: -abs(d["z_exp"]))
    kept: list[dict] = []
    for cand in remaining:
        if all(
            not (k["chrom"] == cand["chrom"] and abs(k["pos"] - cand["pos"]) < window)
            for k in kept
        ):
            kept.append(cand)
    return kept


def run_mr(q, analyses_by_id: dict[str, int], *, imputed_only: bool = False) -> dict:
    exp_idx = analyses_by_id[EXPOSURE]
    # exposure genome-wide significant hits -> raw instruments
    th = q.top_hits(threshold=5e-8)
    m = th["analysis_index"] == exp_idx
    n_raw_all = int(np.sum(m))
    if imputed_only:
        m = m & (th["association_status"] == "imputed")
    vidx = th["variant_index"][m]
    z_exp = th["z"][m]
    se_exp = th["se"][m]
    status_exp = th["association_status"][m]

    raw = []
    for vi, ze, see, status in zip(vidx, z_exp, se_exp, status_exp, strict=True):
        rec = q._variant_axis.by_index(int(vi))
        if rec is None:
            continue
        raw.append(
            {"alid": rec.alid, "chrom": rec.chromosome, "pos": int(rec.position),
             "z_exp": float(ze), "se_exp": float(see), "status_exp": str(status)}
        )
    clumped = _clump(raw, CLUMP_KB)

    # Exposure + outcome effects at the clumped instruments in one lookup. The
    # store orients every analysis to the canonical ALID allele, so exposure and
    # outcome betas are already on the same effect allele (no harmonisation).
    alids = [c["alid"] for c in clumped]
    look = q.lookup(alids, [EXPOSURE, OUTCOME])
    exp_i = analyses_by_id[EXPOSURE]
    out_i = analyses_by_id[OUTCOME]
    per_vi: dict[int, dict] = {}
    for vi, ai, z, se in zip(
        look["variant_index"], look["analysis_index"], look["z"], look["se"], strict=True
    ):
        d = per_vi.setdefault(int(vi), {})
        if int(ai) == exp_i:
            d["z_exp"], d["se_exp"] = float(z), float(se)
        elif int(ai) == out_i:
            d["z_out"], d["se_out"] = float(z), float(se)
    for vi, ai, status in zip(
        look["variant_index"], look["analysis_index"], look["association_status"], strict=True
    ):
        d = per_vi.setdefault(int(vi), {})
        if int(ai) == exp_i:
            d["status_exp"] = str(status)
        elif int(ai) == out_i:
            d["status_out"] = str(status)

    axis = q._variant_axis
    instruments = []
    for vi, d in per_vi.items():
        if not {"z_exp", "z_out"} <= d.keys():
            continue
        if imputed_only and (
            d.get("status_exp") != "imputed" or d.get("status_out") != "imputed"
        ):
            continue
        rec = axis.by_index(vi)
        instruments.append(
            {
                "alid": rec.alid if rec else str(vi),
                "chrom": rec.chromosome if rec else "",
                "pos": int(rec.position) if rec else 0,
                "beta_exp": d["z_exp"] * d["se_exp"], "se_exp": d["se_exp"],
                "beta_out": d["z_out"] * d["se_out"], "se_out": d["se_out"],
                "status_exp": d.get("status_exp", "observed"),
                "status_out": d.get("status_out", "observed"),
            }
        )

    be = np.array([i["beta_exp"] for i in instruments])
    bo = np.array([i["beta_out"] for i in instruments])
    so = np.array([i["se_out"] for i in instruments])
    w = be**2 / so**2
    ivw_beta = float(np.sum(be * bo / so**2) / np.sum(w))
    ivw_se = float(np.sqrt(1.0 / np.sum(w)))
    ivw_z = ivw_beta / ivw_se
    from scipy.special import erfc

    ivw_p = float(erfc(abs(ivw_z) / np.sqrt(2.0)))
    return {
        "exposure_id": EXPOSURE,
        "outcome_id": OUTCOME,
        "clump_kb": CLUMP_KB,
        "instrument_filter": "imputed_only" if imputed_only else "all",
        "n_instruments_raw_all": n_raw_all,
        "n_instruments_raw": len(raw),
        "n_instruments": len(instruments),
        "ivw_beta": ivw_beta,
        "ivw_se": ivw_se,
        "ivw_z": ivw_z,
        "ivw_pval": ivw_p,
        "instruments": instruments,
    }


def regional_imputation_check(q, analyses_by_id: dict[str, int]) -> dict:
    """Return one 1 Mb exposure region around the strongest imputed top hit."""
    exp_idx = analyses_by_id[EXPOSURE]
    th = q.top_hits(threshold=5e-8)
    m = (th["analysis_index"] == exp_idx) & (th["association_status"] == "imputed")
    if not np.any(m):
        m = th["analysis_index"] == exp_idx
    strongest = np.argmax(np.abs(th["z"][m]))
    center_vi = int(th["variant_index"][m][strongest])
    center_z = float(th["z"][m][strongest])
    center = q._variant_axis.by_index(center_vi)
    start = max(1, int(center.position) - 500_000)
    end = int(center.position) + 500_000

    region = q.range_phewas(center.chromosome, start, end)
    keep = region["analysis_index"] == exp_idx
    points = []
    for vi, z, se, status in zip(
        region["variant_index"][keep],
        region["z"][keep],
        region["se"][keep],
        region["association_status"][keep],
        strict=True,
    ):
        rec = q._variant_axis.by_index(int(vi))
        points.append(
            {
                "alid": rec.alid if rec else str(int(vi)),
                "pos": int(rec.position) if rec else int(vi),
                "z": float(z),
                "se": float(se),
                "association_status": str(status),
                "is_center": int(vi) == center_vi,
            }
        )

    return {
        "analysis_id": EXPOSURE,
        "chrom": center.chromosome,
        "start": start,
        "end": end,
        "center_alid": center.alid,
        "center_pos": int(center.position),
        "center_z": center_z,
        "points": points,
    }


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--store", type=Path, default=STORE)
    ap.add_argument("--output", type=Path, default=OUTPUT)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--build-log", type=Path, default=BUILD_LOG)
    ap.add_argument("--top-hits-experiment", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    if args.top_hits_experiment:
        run_top_hit_experiment(args.store, args.output, args.reps)
        return

    q = query_store(args.store)
    an = q.analyses_table()
    analyses_by_id = {v["analysis_id"]: k for k, v in an.items()}
    n_analyses = len(an)
    n_variants = int(q._root["z"].shape[0])

    # pick a strong instrument variant for phewas + random selections
    th = q.top_hits(threshold=5e-8)
    m = th["analysis_index"] == analyses_by_id[EXPOSURE]
    strong_vi = int(th["variant_index"][m][np.argmax(np.abs(th["z"][m]))])
    phewas_alid = q._variant_axis.by_index(strong_vi).alid

    rng = np.random.default_rng(0)
    rand_vi = rng.choice(n_variants, size=100, replace=False)
    rand_alids = [
        r.alid for r in (q._variant_axis.by_index(int(v)) for v in rand_vi) if r is not None
    ]
    rand_a = rng.choice(n_analyses, size=10, replace=False)
    rand_analyses = [an[int(a)]["analysis_id"] for a in rand_a]

    patterns = {
        "bulk": lambda: q.analysis(EXPOSURE),
        "phewas": lambda: q.phewas(phewas_alid),
        "regional": lambda: q.range_phewas(*REGION),
        "tophits": lambda: q.top_hits(analysis_id=EXPOSURE, threshold=5e-8),
        "random_lookup": lambda: q.lookup(rand_alids, rand_analyses),
    }
    timings = []
    for name, fn in patterns.items():
        med, p95, cnt = _median_ms(fn, args.reps)
        timings.append({"query": name, "median_ms": round(med, 3),
                        "p95_ms": round(p95, 3), "result_count": cnt})
        print(f"{name:15s} median={med:9.2f} ms  count={cnt:,}")

    store_bytes = _dir_bytes(args.store)
    raw_bytes, n_files = _raw_vcf_bytes(args.manifest)
    build_seconds = _build_seconds(args.build_log)

    imputed_only_mr = bool(getattr(q, "_is_completed", False))
    mr = run_mr(q, analyses_by_id, imputed_only=imputed_only_mr)
    print(f"MR IVW: beta={mr['ivw_beta']:.4f} se={mr['ivw_se']:.4f} "
          f"p={mr['ivw_pval']:.2e}  n_instruments={mr['n_instruments']}")

    result = {
        "dataset": {"n_variants": n_variants, "n_analyses": n_analyses,
                    "reference_assembly": "GRCh38", "store": str(args.store)},
        "storage": {
            "store_bytes": store_bytes, "store_gb": round(store_bytes / 1e9, 2),
            "raw_vcf_bytes": raw_bytes, "raw_vcf_gb": round(raw_bytes / 1e9, 2),
            "n_source_files": n_files,
            "compression_ratio": round(raw_bytes / store_bytes, 2) if store_bytes else None,
        },
        "build": {"build_seconds": build_seconds,
                  "build_hours": round(build_seconds / 3600, 2) if build_seconds else None},
        "selection": {
            "bulk_analysis_id": EXPOSURE, "phewas_alid": phewas_alid,
            "region": {"chrom": REGION[0], "start": REGION[1], "end": REGION[2]},
            "n_random_variants": len(rand_alids), "n_random_analyses": len(rand_analyses),
        },
        "timings": timings,
        "mr": mr,
        "regional_imputation_check": regional_imputation_check(q, analyses_by_id),
        "labels": {
            EXPOSURE: an[analyses_by_id[EXPOSURE]]["analysis_label"],
            OUTCOME: an[analyses_by_id[OUTCOME]]["analysis_label"],
        },
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
