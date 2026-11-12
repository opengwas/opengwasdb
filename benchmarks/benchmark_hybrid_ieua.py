#!/usr/bin/env python3
"""Capstone benchmark (issue 061): Hybrid vs Dense-of-union on a small, real,
heterogeneous consortium collection (ieu-a-2 BMI, ieu-a-7 CAD, ieu-a-300 LDL).

Builds **both** a Hybrid store (Dense Component + Ragged Overflow) and a
Dense-of-union store from the *same* three GWAS-VCFs against the *same* reference
panel, reference-completes both, validates both, then measures:

  * storage (with the hybrid dense/overflow split + on-/off-panel counts),
  * per-query latency (analysis / phewas / range_phewas / lookup / top_hits),
  * imputed-only positive-control MR (LDL→CAD and BMI→CAD) vs the observed-only
    estimate,
  * a regional imputation window (imputed highlighted against observed).

Writes docs/benchmark-output/opengwasdb_hybrid_ieua_benchmark.json for the QMD.

The build is genome-wide (each study ~9 M variants, hg19→hg38 liftover) and the
completion runs the full EUR LD panel, so this is a long job — run it detached and
monitor memory. Stores are reused if they already exist (``--skip-build``).

Usage:
  uv run python benchmarks/benchmark_hybrid_ieua.py \
      --work /local-scratch/data/opengwas/opengwasdb/hybrid-ieua \
      --ld-panel /local-scratch/projects/genotype-phenotype-map/data/ld_reference_panel_hg38 \
      [--reps 5] [--skip-build] [--n-workers 32]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.special import erfc

from opengwasdb.completion.ld_panel import list_all_blocks, list_chromosomes
from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
from opengwasdb.layouts.dense.complete import complete_dense_store
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.layouts.hybrid.complete import complete_hybrid_store
from opengwasdb.query import query_store
from opengwasdb.validation import validate_store

IGD = Path("/local-scratch/data/opengwas/igd")
STUDIES = {
    "ieu-a-2": "Body mass index (GIANT)",
    "ieu-a-7": "Coronary heart disease (CARDIoGRAMplusC4D)",
    "ieu-a-300": "LDL cholesterol",
}
# Positive-control MR pairs (exposure → outcome).
MR_PAIRS = [
    ("ieu-a-300", "ieu-a-7", "LDL cholesterol → CAD"),
    ("ieu-a-2", "ieu-a-7", "BMI → CAD"),
]
CLUMP_KB = 1000
OUTPUT = Path(
    "/home/gh13047/repo/opengwasdb/docs/benchmark-output/"
    "opengwasdb_hybrid_ieua_benchmark.json"
)


# ── build helpers ─────────────────────────────────────────────────────────────


def _write_manifest(work: Path) -> Path:
    manifest = work / "manifest.tsv"
    lines = ["trait_id\tfile_path\ttrait_name\tn"]
    for sid, label in STUDIES.items():
        vcf = IGD / sid / f"{sid}.vcf.gz"
        if not vcf.exists():
            raise FileNotFoundError(vcf)
        lines.append(f"{sid}\t{vcf}\t{label}\t100000")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _extract_panel(ld_dir: Path, ancestry: str, out_path: Path) -> int:
    """Write the reference-panel ALID list (the Dense Component axis) from the LD
    reference panel, so 'dense' == 'imputable' exactly: each block's ALIDs as
    completion names them (`LDBlock.alids`, an indel spelt reference first)."""
    if out_path.exists():
        return sum(1 for _ in out_path.open())
    alids: set[str] = set()
    for chrom in list_chromosomes(ld_dir, ancestry):
        for block in list_all_blocks(ld_dir, ancestry, chrom):
            alids.update(alid for alid in block.alids if alid is not None)
    with out_path.open("w") as fh:
        for a in sorted(alids):
            fh.write(a + "\n")
    return len(alids)


def _dir_bytes(path: Path) -> int:
    out = subprocess.run(["du", "-sb", str(path)], capture_output=True, text=True, check=True)
    return int(out.stdout.split()[0])


def _raw_vcf_bytes() -> int:
    return sum((IGD / s / f"{s}.vcf.gz").stat().st_size for s in STUDIES)


# ── query latency ─────────────────────────────────────────────────────────────


def _median_ms(fn, reps: int) -> tuple[float, float, int]:
    fn()
    times, count = [], 0
    for _ in range(reps):
        t0 = time.perf_counter()
        res = fn()
        times.append((time.perf_counter() - t0) * 1000.0)
        count = len(res["z"])
    times.sort()
    return times[len(times) // 2], times[min(len(times) - 1, int(0.95 * len(times)))], count


def _latencies(
    store: Path, reps: int, region, phewas_alid, rand_alids, rand_analyses
) -> list[dict]:
    q = query_store(store)
    exp = "ieu-a-300"
    patterns = {
        "bulk": lambda: q.analysis(exp),
        "phewas": lambda: q.phewas(phewas_alid),
        "regional": lambda: q.range_phewas(*region),
        "tophits": lambda: q.top_hits(analysis_id=exp, threshold=5e-8),
        "random_lookup": lambda: q.lookup(rand_alids, rand_analyses),
    }
    out = []
    for name, fn in patterns.items():
        med, p95, cnt = _median_ms(fn, reps)
        out.append({"query": name, "median_ms": round(med, 3),
                    "p95_ms": round(p95, 3), "result_count": cnt})
    return out


# ── MR (generalised, imputed-only capable) ────────────────────────────────────


def _clump(instr: list[dict], kb: int) -> list[dict]:
    window = kb * 1000
    kept: list[dict] = []
    for cand in sorted(instr, key=lambda d: -abs(d["z_exp"])):
        if all(
            not (k["chrom"] == cand["chrom"] and abs(k["pos"] - cand["pos"]) < window)
            for k in kept
        ):
            kept.append(cand)
    return kept


def run_mr(q, by_id: dict[str, int], exposure: str, outcome: str, *, imputed_only: bool) -> dict:
    exp_idx = by_id[exposure]
    th = q.top_hits(threshold=5e-8)
    m = th["analysis_index"] == exp_idx
    n_raw_all = int(np.sum(m))
    if imputed_only:
        m = m & (th["association_status"] == "imputed")
    raw = []
    for vi, ze in zip(th["variant_index"][m], th["z"][m], strict=True):
        rec = q._variant_axis.by_index(int(vi))
        if rec is not None:
            raw.append({"alid": rec.alid, "chrom": rec.chromosome, "pos": int(rec.position),
                        "z_exp": float(ze)})
    clumped = _clump(raw, CLUMP_KB)
    alids = [c["alid"] for c in clumped]
    look = q.lookup(alids, [exposure, outcome])
    per: dict[int, dict] = {}
    for vi, ai, z, se, st in zip(
        look["variant_index"], look["analysis_index"], look["z"], look["se"],
        look["association_status"], strict=True,
    ):
        d = per.setdefault(int(vi), {})
        if int(ai) == exp_idx:
            d["z_exp"], d["se_exp"], d["st_exp"] = float(z), float(se), str(st)
        elif int(ai) == by_id[outcome]:
            d["z_out"], d["se_out"], d["st_out"] = float(z), float(se), str(st)
    instruments = []
    for vi, d in per.items():
        if not {"z_exp", "z_out"} <= d.keys():
            continue
        if imputed_only and (d.get("st_exp") != "imputed" or d.get("st_out") != "imputed"):
            continue
        rec = q._variant_axis.by_index(vi)
        instruments.append({
            "alid": rec.alid if rec else str(vi),
            "chrom": rec.chromosome if rec else "", "pos": int(rec.position) if rec else 0,
            "beta_exp": d["z_exp"] * d["se_exp"], "se_exp": d["se_exp"],
            "beta_out": d["z_out"] * d["se_out"], "se_out": d["se_out"],
        })
    return _ivw(instruments, exposure, outcome, imputed_only, n_raw_all, len(raw))


def _ivw(instruments, exposure, outcome, imputed_only, n_raw_all, n_raw) -> dict:
    if not instruments:
        return {"exposure_id": exposure, "outcome_id": outcome,
                "instrument_filter": "imputed_only" if imputed_only else "all",
                "n_instruments_raw_all": n_raw_all, "n_instruments_raw": n_raw,
                "n_instruments": 0, "ivw_beta": None, "ivw_se": None, "ivw_z": None,
                "ivw_pval": None, "instruments": []}
    be = np.array([i["beta_exp"] for i in instruments])
    bo = np.array([i["beta_out"] for i in instruments])
    so = np.array([i["se_out"] for i in instruments])
    w = be**2 / so**2
    ivw_beta = float(np.sum(be * bo / so**2) / np.sum(w))
    ivw_se = float(np.sqrt(1.0 / np.sum(w)))
    ivw_z = ivw_beta / ivw_se
    return {
        "exposure_id": exposure, "outcome_id": outcome, "clump_kb": CLUMP_KB,
        "instrument_filter": "imputed_only" if imputed_only else "all",
        "n_instruments_raw_all": n_raw_all, "n_instruments_raw": n_raw,
        "n_instruments": len(instruments), "ivw_beta": ivw_beta, "ivw_se": ivw_se,
        "ivw_z": ivw_z, "ivw_pval": float(erfc(abs(ivw_z) / np.sqrt(2.0))),
        "instruments": instruments,
    }


def regional_check(q, exposure: str, by_id: dict[str, int]) -> dict:
    exp_idx = by_id[exposure]
    th = q.top_hits(threshold=5e-8)
    m = (th["analysis_index"] == exp_idx) & (th["association_status"] == "imputed")
    if not np.any(m):
        m = th["analysis_index"] == exp_idx
    if not np.any(m):
        return {}
    strongest = np.argmax(np.abs(th["z"][m]))
    center_vi = int(th["variant_index"][m][strongest])
    center = q._variant_axis.by_index(center_vi)
    start = max(1, int(center.position) - 500_000)
    end = int(center.position) + 500_000
    region = q.range_phewas(center.chromosome, start, end)
    keep = region["analysis_index"] == exp_idx
    points = []
    for vi, z, st in zip(region["variant_index"][keep], region["z"][keep],
                         region["association_status"][keep], strict=True):
        rec = q._variant_axis.by_index(int(vi))
        points.append({"pos": int(rec.position) if rec else int(vi), "z": float(z),
                       "association_status": str(st), "is_center": int(vi) == center_vi})
    return {"analysis_id": exposure, "chrom": center.chromosome, "start": start, "end": end,
            "center_pos": int(center.position), "center_z": float(th["z"][m][strongest]),
            "points": points}


# ── orchestration ─────────────────────────────────────────────────────────────


def _build_all(
    work: Path, manifest: Path, panel: Path, ld_dir: Path, n_workers: int
) -> dict[str, Path]:
    stores = {
        "hybrid_obs": work / "hybrid.opengwasdb",
        "hybrid_completed": work / "hybrid-completed.opengwasdb",
        "dense_obs": work / "dense-union.opengwasdb",
        "dense_completed": work / "dense-union-completed.opengwasdb",
    }
    if not stores["dense_obs"].exists():
        print("Building Dense-of-union store...", flush=True)
        build_dense_from_vcf_manifest(
            manifest, stores["dense_obs"], store_id="ieua-dense", release_id="obs-v1",
            n_workers=n_workers, overwrite=True,
        )
    if not stores["hybrid_obs"].exists():
        print("Building Hybrid store...", flush=True)
        build_hybrid_from_vcf_manifest(
            manifest, stores["hybrid_obs"], reference_panel=panel,
            store_id="ieua-hybrid", release_id="obs-v1", n_workers=n_workers, overwrite=True,
        )
    if not stores["dense_completed"].exists():
        print("Completing Dense-of-union store...", flush=True)
        complete_dense_store(stores["dense_obs"], stores["dense_completed"], ld_dir,
                             n_workers=n_workers, overwrite=True)
    if not stores["hybrid_completed"].exists():
        print("Completing Hybrid store...", flush=True)
        complete_hybrid_store(stores["hybrid_obs"], stores["hybrid_completed"], ld_dir,
                              n_workers=n_workers, overwrite=True)
    return stores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--ld-panel", type=Path, required=True)
    ap.add_argument("--ancestry", default="EUR")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--n-workers", type=int, default=16)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--output", type=Path, default=OUTPUT)
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)

    manifest = _write_manifest(args.work)
    panel_path = args.work / "panel_alids.txt"
    n_panel = _extract_panel(args.ld_panel, args.ancestry, panel_path)
    print(f"Reference panel: {n_panel:,} ALIDs", flush=True)

    if not args.skip_build:
        stores = _build_all(args.work, manifest, panel_path, args.ld_panel, args.n_workers)
    else:
        stores = {
            "hybrid_obs": args.work / "hybrid.opengwasdb",
            "hybrid_completed": args.work / "hybrid-completed.opengwasdb",
            "dense_obs": args.work / "dense-union.opengwasdb",
            "dense_completed": args.work / "dense-union-completed.opengwasdb",
        }

    print("Validating stores...", flush=True)
    validation = {k: validate_store(p).errors for k, p in stores.items()}
    for k, errs in validation.items():
        print(f"  {k}: {'OK' if not errs else errs}", flush=True)

    # Selections for latency (shared across both stores).
    qh = query_store(stores["hybrid_completed"])
    by_id = {v["analysis_id"]: k for k, v in qh.analyses_table().items()}
    th = qh.top_hits(threshold=5e-8)
    m = th["analysis_index"] == by_id["ieu-a-300"]
    strong_vi = int(th["variant_index"][m][np.argmax(np.abs(th["z"][m]))])
    phewas_alid = qh._variant_axis.by_index(strong_vi).alid
    center = qh._variant_axis.by_index(strong_vi)
    region = (center.chromosome, max(1, center.position - 500_000), center.position + 500_000)
    n_shared = qh._variant_axis.n_variants
    rng = np.random.default_rng(0)
    rand_vi = rng.choice(n_shared, size=100, replace=False)
    rand_alids = [r.alid for r in (qh._variant_axis.by_index(int(v)) for v in rand_vi) if r]
    rand_analyses = list(STUDIES)

    print("Measuring latency...", flush=True)
    timings = {
        "hybrid": _latencies(stores["hybrid_completed"], args.reps, region, phewas_alid,
                             rand_alids, rand_analyses),
        "dense_union": _latencies(stores["dense_completed"], args.reps, region, phewas_alid,
                                  rand_alids, rand_analyses),
    }

    # Storage split.
    from opengwasdb.model.manifest import StoreManifest
    hyb_manifest = StoreManifest.load(stores["hybrid_completed"])
    hyb = hyb_manifest.provenance.get("hybrid", {})
    storage = {
        "raw_vcf_bytes": _raw_vcf_bytes(),
        "dense_union_completed_bytes": _dir_bytes(stores["dense_completed"]),
        "hybrid_completed_bytes": _dir_bytes(stores["hybrid_completed"]),
        "hybrid_dense_component_bytes": _dir_bytes(stores["hybrid_completed"] / "dense"),
        "hybrid_overflow_bytes": _dir_bytes(stores["hybrid_completed"] / "data.zarr" / "ragged"),
        "n_panel": hyb.get("n_panel"),
        "n_off_panel": hyb.get("n_off_panel"),
        "n_shared": n_shared,
    }

    # MR: imputed-only on completed stores + observed-only on observed stores.
    print("Running MR...", flush=True)
    q_hyb_obs = query_store(stores["hybrid_obs"])
    by_id_obs = {v["analysis_id"]: k for k, v in q_hyb_obs.analyses_table().items()}
    mr = []
    for exposure, outcome, label in MR_PAIRS:
        mr.append({
            "label": label, "exposure": exposure, "outcome": outcome,
            "observed": run_mr(q_hyb_obs, by_id_obs, exposure, outcome, imputed_only=False),
            "imputed_only": run_mr(qh, by_id, exposure, outcome, imputed_only=True),
        })

    result = {
        "dataset": {"studies": STUDIES, "n_analyses": len(STUDIES),
                    "n_shared_variants": n_shared, "reference_assembly": "GRCh38"},
        "validation": validation,
        "storage": storage,
        "timings": timings,
        "mr": mr,
        "regional_imputation_check": regional_check(qh, "ieu-a-300", by_id),
        "labels": STUDIES,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
