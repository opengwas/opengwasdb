#!/usr/bin/env python3
"""Benchmark the Reference-Completed full-scale release OGS-00010.

Measures, against the OGS-00009 observed-only twin and the same source VCF
collection:

* storage, build and completion cost of imputing 10.7 billion cells;
* the same query shapes as ``benchmark_ukbb_dense.py`` (timings + per-shape
  peak RSS) on the completed axis, plus ``observed_only`` filter variants;
* imputation quality from the release's ``completion_quality`` table;
* three end-to-end MR runs (BMI -> CHD, an LDL proxy -> CHD, smoking ->
  lung cancer), each on all variants, observed-only variants and
  imputed-only variants, with per-instrument scatter data;
* regional windows around each exposure's strongest imputed genome-wide
  hit, with every variant's association status (observed vs imputed);
* a cross-release fidelity check: the observed cells OGS-00009 holds must
  decode identically inside OGS-00010.

Writes docs/benchmark-output/opengwasdb_ogs00010_completed_benchmark.json,
which the companion QMD renders.

Usage:
  pixi run -e dev python benchmarks/benchmark_ogs00010_completed.py \
      [--reps N] [--store PATH] [--output PATH] [--skip-rss]
"""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import time
from bisect import bisect_left
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import erfc

from benchmarks._artifact import provenance, write_artifact
from benchmarks._rss import run_probe, sample_query
from benchmarks.benchmark_ukbb_dense import (
    _dir_bytes,
    _median_ms,
    _raw_vcf_bytes,
)
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.query import query_store

STORE = Path("/data/opengwasdb/stores/OGS-00010/store.opengwasdb")
SOURCE_STORE = Path("/data/opengwasdb/stores/OGS-00009/store.opengwasdb")
RECORDS = Path("/data/opengwasdb/stores/OGS-00010/records")
MANIFEST = Path("/data/opengwasdb/stores/OGS-00009/work/analyses.tsv")
SOURCE_BENCHMARK = Path(
    "/home/gh13047/repo/opengwasdb/docs/benchmark-output/"
    "opengwasdb_ukbb_dense_benchmark.json"
)
OUTPUT = Path(
    "/home/gh13047/repo/opengwasdb/docs/benchmark-output/"
    "opengwasdb_ogs00010_completed_benchmark.json"
)

# Query-shape driver, matching the OGS-00009 benchmark exactly: same analysis
# (EXPOSURE), same PheWAS variant (taken from the OGS-00009 artifact), same
# regional window — so every timing comparison in the report is the same query
# against the two releases, not two similar queries.
EXPOSURE = "ukb-b-17805"  # cholesterol-lowering medication (statin use)

# MR pairs. Every trait here is one the ukb-b collection actually carries:
# there is no LDL or lipid biomarker measurement in ukb-b, so the LDL pair
# uses statin (cholesterol-lowering medication) as the liability proxy, and
# the lung-cancer outcome is the ICD10 C34.1 cancer-site field, which is
# coded only among cancer cases (a site-shift, not incidence, endpoint).
# Both caveats are carried into the artifact and the report.
MR_PAIRS: list[dict[str, str]] = [
    {
        "key": "bmi_chd",
        "exposure_id": "ukb-b-19953",
        "outcome_id": "ukb-b-1668",
        "label": "BMI -> CHD (I25.1)",
        "proxy_note": "",
    },
    {
        "key": "ldl_chd",
        "exposure_id": "ukb-b-17805",
        "outcome_id": "ukb-b-1668",
        "label": "LDL proxy (cholesterol-lowering medication) -> CHD (I25.1)",
        "proxy_note": (
            "ukb-b carries no LDL biomarker; the exposure is statin use, a "
            "treatment proxy confounded by indication. A pipeline check, not "
            "a causal LDL estimate."
        ),
    },
    {
        "key": "smoking_lung",
        "exposure_id": "ukb-b-2134",
        "outcome_id": "ukb-b-18798",
        "label": "Past tobacco smoking -> lung cancer (C34.1)",
        "proxy_note": (
            "C34.1 is coded only among cancer cases (cancer-site field, ~1k "
            "cases), so conditioning on any diagnosis makes it a collider: "
            "the IVW sign here is a site-shift ratio, not the causal effect "
            "of smoking on lung-cancer incidence."
        ),
    },
]

MR_CONDITIONS = ("all", "observed", "imputed")
CLUMP_KB = 1000
# Per-condition scatter points kept in the artifact for the report. The IVW
# estimate always uses every instrument; only the plotted tail is thinned.
MAX_SCATTER_POINTS = 2500

# Regional query window shared with the OGS-00009 benchmark (APOE/APOC, chr19).
REGION = ("19", 44_500_000, 45_500_000)
RANDOM_AXIS_SIZE = 100
LOOKUP_NARROW_AXIS_SIZE = 10
REGION_HALF_WINDOW = 500_000


def _g4(x: float) -> float:
    """Round to 4 significant digits so the committed artifact stays small."""
    return float(f"{x:.4g}")


def clump_indices(
    chrom: np.ndarray, pos: np.ndarray, z: np.ndarray, kb: int
) -> np.ndarray:
    """Greedy 1 Mb distance pruning by descending |z| (same rule as
    ``benchmark_ukbb_dense._clump``), over numpy columns.

    Returns the indices of kept candidates. A candidate is dropped when a
    stronger kept instrument sits within +/- window on its chromosome;
    per-chromosome kept positions stay sorted, so the check is two bisects.
    BMI's ~93k hits clump in milliseconds -- the cost that made the old
    per-record clump slow was never the window check, it was resolving every
    hit through ``by_index()`` first (see ``exposure_columns``).
    """
    window = kb * 1000
    kept_pos: dict[str, list[int]] = {}
    kept: list[int] = []
    for i in np.argsort(-np.abs(z), kind="stable"):
        c = str(chrom[i])
        p = int(pos[i])
        lst = kept_pos.get(c)
        if lst is None:
            kept_pos[c] = [p]
            kept.append(int(i))
            continue
        j = bisect_left(lst, p)
        if (j < len(lst) and lst[j] - p < window) or (j > 0 and p - lst[j - 1] < window):
            continue
        lst.insert(j, p)
        kept.append(int(i))
    return np.asarray(kept, dtype="int64")


def exposure_columns(q: Any, analysis_id: str) -> dict[str, Any]:
    """The 5e-8 tier of one analysis as numpy columns.

    Identity for every hit row is resolved by ``identity_by_indices`` -- the
    mmap'd ALID index inverted, all numpy, zero per-row ``variants.tsv.gz``
    I/O. The old path (``by_index`` per hit, one BGZF seek each) put ~30s and
    92,646 objects between asking for BMI's hits and being able to clump
    them; MR top-hit -> instrument -> outcome-lookup should take seconds.
    """
    th = q.top_hits(analysis_id=analysis_id, threshold=5e-8)
    ident = q._variant_axis.identity_by_indices(th["variant_index"])
    if ident is None:
        raise SystemExit(
            "store carries no mmap'd ALID index; refusing to resolve "
            f"{len(th['variant_index']):,} hit rows with per-row table seeks"
        )
    return {
        "alid": ident["alid"],
        "chrom": ident["chromosome"],
        "pos": ident["position"],
        "z": th["z"],
        "se": th["se"],
        "status": th["association_status"],
    }


def _instrument_rows(
    cols: dict[str, Any], keep_idx: np.ndarray, look: dict[str, np.ndarray],
    axis: Any, exp_i: int, out_i: int,
) -> list[dict[str, Any]]:
    """Join clumped exposure hits to their lookup rows by ALID."""
    lid = axis.identity_by_indices(look["variant_index"])
    per_alid: dict[str, dict[str, Any]] = {}
    for alid, ai, z, se, st in zip(
        lid["alid"],
        look["analysis_index"],
        look["z"],
        look["se"],
        look["association_status"],
        strict=True,
    ):
        d = per_alid.setdefault(str(alid), {})
        tag = "exp" if int(ai) == exp_i else "out"
        d[f"z_{tag}"], d[f"se_{tag}"], d[f"st_{tag}"] = float(z), float(se), str(st)
    rows: list[dict[str, Any]] = []
    for i in keep_idx:
        d = per_alid.get(str(cols["alid"][i]))
        if d is None or "z_out" not in d:
            continue
        rows.append(
            {
                "chrom": str(cols["chrom"][i]),
                "pos": int(cols["pos"][i]),
                "beta_exp": d["z_exp"] * d["se_exp"],
                "se_exp": d["se_exp"],
                "beta_out": d["z_out"] * d["se_out"],
                "se_out": d["se_out"],
                "status_exp": d["st_exp"],
                "status_out": d["st_out"],
            }
        )
    return rows


def mr_condition(
    q: Any,
    analyses_by_id: dict[str, int],
    pair: dict[str, str],
    condition: str,
    cols: dict[str, Any],
) -> dict[str, Any]:
    """One IVW MR under one variant-availability condition.

    The exposure's hit tier is already cached as columns; the conditions are
    masks on its status vector -- ``all`` keeps every instrument and every
    finite outcome cell, ``observed`` restricts both sides to
    source-reported cells (what OGS-00009 could do), ``imputed`` keeps only
    instruments imputed in the exposure *and* imputed in the outcome, the
    estimates that exist only because completion ran. Then the whole MR is
    one lookup of the surviving ALIDs on exposure + outcome.
    """
    exp, out_id = pair["exposure_id"], pair["outcome_id"]
    status = cols["status"]
    if condition == "observed":
        mask = status == "observed"
    elif condition == "imputed":
        mask = status == "imputed"
    else:
        mask = np.ones(status.shape, dtype=bool)
    idx = np.flatnonzero(mask)
    keep_idx = idx[clump_indices(cols["chrom"][idx], cols["pos"][idx], cols["z"][idx], CLUMP_KB)]

    alids = [str(cols["alid"][i]) for i in keep_idx]
    look = q.lookup(alids, [exp, out_id], observed_only=(condition == "observed"))
    instruments = _instrument_rows(
        cols, keep_idx, look, q._variant_axis,
        analyses_by_id[exp], analyses_by_id[out_id],
    )
    if condition == "imputed":
        instruments = [
            i for i in instruments
            if i["status_exp"] == "imputed" and i["status_out"] == "imputed"
        ]

    if not instruments:
        raise SystemExit(
            f"MR {pair['label']} / {condition}: zero instruments survived the "
            "lookup — refusing to publish a silent null."
        )
    be = np.array([i["beta_exp"] for i in instruments])
    bo = np.array([i["beta_out"] for i in instruments])
    so = np.array([i["se_out"] for i in instruments])
    ivw_beta = float(np.sum(be * bo / so**2) / np.sum(be**2 / so**2))
    ivw_se = float(np.sqrt(1.0 / np.sum(be**2 / so**2)))
    ivw_p = float(erfc(abs(ivw_beta / ivw_se) / np.sqrt(2.0)))

    n = len(instruments)
    stride = max(1, n // MAX_SCATTER_POINTS)
    thin = instruments[::stride]
    return {
        "condition": condition,
        "n_hits_gwsig": int(mask.sum()),
        "n_clumped": len(keep_idx),
        "n_instruments": n,
        "ivw_beta": ivw_beta,
        "ivw_se": ivw_se,
        "ivw_pval": ivw_p,
        "scatter": {
            "stride": stride,
            "chrom": [i["chrom"] for i in thin],
            "pos": [i["pos"] for i in thin],
            "beta_exp": [float(f"{i['beta_exp']:.4g}") for i in thin],
            "se_exp": [float(f"{i['se_exp']:.4g}") for i in thin],
            "beta_out": [float(f"{i['beta_out']:.4g}") for i in thin],
            "se_out": [float(f"{i['se_out']:.4g}") for i in thin],
            "status_exp": [i["status_exp"] for i in thin],
            "status_out": [i["status_out"] for i in thin],
        },
    }


def observed_cell_fidelity(
    q_completed: Any, q_source: Any, exp_cols: dict[str, Any]
) -> dict[str, Any]:
    """The completion MUST preserve OGS-00009's observed cells (ADR 0011).

    Check it on every observed cell of the statin pair's exposure tier in
    both analyses: each must be present in the completed store, carry status
    ``observed``, and decode to identical z and se.
    """
    pair = MR_PAIRS[1]
    exp, out_id = pair["exposure_id"], pair["outcome_id"]
    obs = exp_cols["status"] == "observed"
    alids = [str(a) for a in exp_cols["alid"][obs]]
    look_c = q_completed.lookup(alids, [exp, out_id])
    look_s = q_source.lookup(alids, [exp, out_id])
    idc = q_completed._variant_axis.identity_by_indices(look_c["variant_index"])
    ids = q_source._variant_axis.identity_by_indices(look_s["variant_index"])
    key_c = {
        (str(alid), int(ai)): (float(z), float(se), str(st))
        for alid, ai, z, se, st in zip(
            idc["alid"],
            look_c["analysis_index"],
            look_c["z"],
            look_c["se"],
            look_c["association_status"],
            strict=True,
        )
    }
    n_compared = 0
    max_dz = 0.0
    max_dse = 0.0
    wrong_status = 0
    for alid, ai, z, se in zip(
        ids["alid"],
        look_s["analysis_index"],
        look_s["z"],
        look_s["se"],
        strict=True,
    ):
        got = key_c.get((str(alid), int(ai)))
        if got is None:
            raise SystemExit(
                f"observed cell {alid} x {ai} present in OGS-00009 but missing "
                "from OGS-00010 — completion dropped an observed cell"
            )
        n_compared += 1
        if got[2] != "observed":
            wrong_status += 1
        max_dz = max(max_dz, abs(got[0] - float(z)))
        max_dse = max(max_dse, abs(got[1] - float(se)))
    return {
        "pair": pair["label"],
        "analyses": [exp, out_id],
        "n_compared": n_compared,
        "max_abs_delta_z": max_dz,
        "max_abs_delta_se": max_dse,
        "wrong_status": wrong_status,
        "identical": max_dz == 0.0 and max_dse == 0.0 and wrong_status == 0,
    }


def region_imputation_profile(
    q: Any, analysis_id: str, cols: dict[str, Any]
) -> dict[str, Any]:
    """2 Mb window centred on the analysis's strongest imputed genome-wide hit.

    Covers EVERY variant of the completed axis in the window: axis rows are a
    contiguous slice of the genomic-sorted table, so the plot data is one
    ``identity_by_indices`` read plus a numpy scatter of that analysis's
    region column -- real cells keep their z/se and observed/imputed status,
    rows whose cell is still NaN (off-panel or imputation-failed) are
    reported with status ``missing`` and zero statistics. The plot cannot
    silently drop the third class.
    """
    imp = np.flatnonzero(cols["status"] == "imputed")
    if imp.size == 0:
        raise SystemExit(f"{analysis_id}: no imputed genome-wide hits to centre a region on")
    ci = imp[int(np.argmax(np.abs(cols["z"][imp])))]
    chrom = str(cols["chrom"][ci])
    start = max(1, int(cols["pos"][ci]) - REGION_HALF_WINDOW)
    end = int(cols["pos"][ci]) + REGION_HALF_WINDOW

    axis = q._variant_axis
    rows = axis.range_indices(chrom, start, end).astype("int64")
    if rows.size and not np.array_equal(rows, np.arange(rows.min(), rows.min() + rows.size)):
        raise SystemExit(
            "the axis region slice is not contiguous — the completed store's "
            "variant table is not genomic-sorted; refusing a mis-aligned plot"
        )
    ident = axis.identity_by_indices(rows)
    if ident is None:
        raise SystemExit("no mmap'd ALID index for the region rows — refusing per-row seeks")
    z = np.zeros(rows.size, dtype="float64")
    se = np.zeros(rows.size, dtype="float64")
    status = np.full(rows.size, "missing", dtype=object)

    # One analysis's column over the window, not the whole rectangle: the
    # range query across all 2,024 Analyses would decode ~18M cells to keep
    # ~5k of them.
    region = q.lookup([str(a) for a in ident["alid"]], [analysis_id])
    rrows = region["variant_index"].astype("int64")
    if rrows.size and not (
        rows.size and rrows.min() >= rows.min() and rrows.max() < rows.min() + rows.size
    ):
        raise SystemExit(
            "region query returned rows outside the axis slice — refusing to "
            "plot a silently mis-aligned region"
        )
    offset = rrows - int(rows.min())
    z[offset] = region["z"]
    se[offset] = region["se"]
    status[offset] = region["association_status"]

    return {
        "analysis_id": analysis_id,
        "chrom": chrom,
        "start": start,
        "end": end,
        "center_alid": str(cols["alid"][ci]),
        "center_pos": int(cols["pos"][ci]),
        "center_z": float(cols["z"][ci]),
        "n_points": int(rows.size),
        "n_observed": int((status == "observed").sum()),
        "n_imputed": int((status == "imputed").sum()),
        "n_missing": int((status == "missing").sum()),
        "points": {
            "pos": [int(p) for p in ident["position"]],
            "z": [_g4(float(v)) for v in z],
            "se": [_g4(float(v)) for v in se],
            "status": [str(s) for s in status],
        },
    }


def _completion_section(store: Path, records: Path, n_variants: int, n_analyses: int) -> dict:
    manifest = json.loads((store / "manifest.json").read_text())
    comp = manifest["provenance"]["completion"]
    record = json.loads((records / "complete.json").read_text())
    stdout = record["stdout"]
    n_blocks = int(stdout.split("completion across ")[1].split(" LD blocks")[0].replace(",", ""))
    panel_variants = int(
        stdout.split("LD panel: ")[1].split(", ")[1].split(" panel variants")[0].replace(",", "")
    )
    finished = datetime.fromisoformat(record["end_time"].replace("Z", "+00:00"))
    elapsed = float(record["elapsed_seconds"])
    argv = record["argv"]
    n_workers = int(argv[argv.index("--n-workers") + 1])
    total_cells = n_variants * n_analyses
    imputed = int(comp["n_imputed"])
    con = sqlite3.connect(f"file:{store / 'index.sqlite'}?mode=ro", uri=True)
    n_rows_total = con.execute("select count(*) from completion_quality").fetchone()[0]
    r = np.array(
        [x[0] for x in con.execute(
            "select pearson_r from completion_quality where pearson_r is not null"
        )],
        dtype="float64",
    )
    per_analysis = {}
    for ai, pr in con.execute(
        "select analysis_index, pearson_r from completion_quality "
        "where pearson_r is not null"
    ):
        per_analysis.setdefault(int(ai), []).append(pr)
    analysis_meds = np.array([np.median(v) for v in per_analysis.values()])
    con.close()
    edges = np.round(np.arange(0.10, 1.0001, 0.01), 2)
    hist, _ = np.histogram(r, bins=edges)
    assert (r >= edges[0]).all() and (r <= edges[-1]).all(), (
        "histogram edges must not silently drop any attempt row; the "
        "gate-rejected tail below min_cor must be inside the plotted range"
    )
    ahist, _ = np.histogram(analysis_meds, bins=np.round(np.arange(0.94, 1.0001, 0.005), 3))
    return {
        "completed_at": finished.isoformat(),
        "step_record_elapsed_seconds": elapsed,
        "step_record_elapsed_hours": round(elapsed / 3600, 2),
        "n_workers": n_workers,
        "n_ld_blocks": n_blocks,
        "ld_panel": {
            "id": comp["ld_panel_id"],
            "ancestry": comp["ancestry"],
            "path": record["argv"][record["argv"].index("--ld-panel") + 1],
            "n_blocks": n_blocks,
            "n_variants": panel_variants,
        },
        "method": comp["method"],
        "min_cor": comp["min_cor"],
        "pca_thresh": comp["pca_thresh"],
        "n_imputed": imputed,
        "n_missing_imputation_failed": int(comp["n_missing_imputation_failed"]),
        "n_missing_off_panel": int(comp["n_missing_off_panel"]),
        "n_variants_source": n_variants - int(comp["n_variants_new"]),
        "n_variants_new": int(comp["n_variants_new"]),
        "total_cells": total_cells,
        "n_observed_cells": total_cells - imputed
        - int(comp["n_missing_imputation_failed"]) - int(comp["n_missing_off_panel"]),
        "imputed_cells_per_second": round(imputed / elapsed),
        "imputed_cells_per_worker_second": round(imputed / elapsed / n_workers),
        "seconds_per_block": round(elapsed / n_blocks, 1),
        "quality": {
            "n_block_rows": int(r.size),
            "n_block_rows_null": int(n_rows_total) - int(r.size),
            "r_percentiles_1_5_50_95_99": [float(x) for x in np.percentile(r, [1, 5, 50, 95, 99])],
            "r_mean": round(float(r.mean()), 4),
            "r_min": round(float(r.min()), 4),
            "r_hist_edges": [float(e) for e in edges],
            "r_hist_counts": [int(c) for c in hist],
            "n_attempts_below_gate": int((r < float(comp["min_cor"])).sum()),
            "frac_r_below_0_9": round(float((r < 0.9).mean()), 4),
            "n_analyses_with_quality": int(analysis_meds.size),
            "analysis_median_r_min": round(float(analysis_meds.min()), 4),
            "analysis_median_r_median": round(float(np.median(analysis_meds)), 4),
            "analysis_median_r_max": round(float(analysis_meds.max()), 4),
            "analysis_r_hist_edges": [
                float(e) for e in np.round(np.arange(0.94, 1.0001, 0.005), 3)
            ],
            "analysis_r_hist_counts": [int(c) for c in ahist],
        },
    }


def _resolve_source_lookup_selections(q_src: Any, src_bench: dict) -> tuple[list[str], list[str]]:
    """The exact random variant/analysis sets the OGS-00009 artifact used.

    Re-derived, not approximated: same seed, same source-axis row indices
    (its recorded ``n_variants``), resolved to ALIDs against the source
    store, so each random-lookup shape times the SAME identifiers on both
    releases. The old artifact recorded only counts, so the identifiers are
    reproducible only while the source store exists — it is an immutable
    release, so that is acceptable.
    """
    rng = np.random.default_rng(0)
    random_variants = rng.choice(
        int(src_bench["dataset"]["n_variants"]), size=RANDOM_AXIS_SIZE, replace=False
    )
    random_alids = [
        record.alid
        for record in (q_src._variant_axis.by_index(int(v)) for v in random_variants)
        if record is not None
    ]
    an_src = q_src.analyses_table()
    random_analyses = [
        an_src[int(a)]["analysis_id"]
        for a in rng.choice(int(src_bench["dataset"]["n_analyses"]),
                            size=RANDOM_AXIS_SIZE, replace=False)
    ]
    if len(random_alids) != RANDOM_AXIS_SIZE:
        raise SystemExit(
            f"resolved {len(random_alids)} of {RANDOM_AXIS_SIZE} source-lookup "
            "variants from the source store — refusing a partial selection"
        )
    return random_alids, random_analyses


def _query_patterns(
    q: Any,
    analyses: dict[int, dict[str, Any]],
    phewas_alid: str,
    random_alids: list[str],
    random_analyses: list[str],
) -> dict[str, Any]:
    """The OGS-00009 shapes on the completed axis, plus the observed_only variants."""
    regional_rows = q._variant_axis.range_indices(*REGION)
    regional_alids = [
        record.alid
        for record in (q._variant_axis.by_index(int(row)) for row in regional_rows)
        if record is not None
    ]
    return {
        "tophits": lambda: q.top_hits(analysis_id=EXPOSURE, threshold=5e-8),
        "tophits_observed_only": lambda: q.top_hits(
            analysis_id=EXPOSURE, threshold=5e-8, observed_only=True
        ),
        "phewas": lambda: q.phewas(phewas_alid),
        "regional_one_analysis": lambda: q.lookup(regional_alids, [EXPOSURE]),
        "random_lookup_10_variants_100_analyses": lambda: q.lookup(
            random_alids[:LOOKUP_NARROW_AXIS_SIZE], random_analyses
        ),
        "random_lookup_100_variants_10_analyses": lambda: q.lookup(
            random_alids, random_analyses[:LOOKUP_NARROW_AXIS_SIZE]
        ),
        "regional": lambda: q.range_phewas(*REGION),
        "bulk": lambda: q.analysis(EXPOSURE),
        "bulk_observed_only": lambda: q.analysis(EXPOSURE, observed_only=True),
    }


def _measure_shape_rss(args: argparse.Namespace, shape: str) -> dict[str, float]:
    q = query_store(args.store)
    an = q.analyses_table()
    src_bench = json.loads(args.source_benchmark.read_text())
    q_src = query_store(args.source_store)
    random_alids, random_analyses = _resolve_source_lookup_selections(q_src, src_bench)
    q_src.close()
    patterns = _query_patterns(
        q, an, args.phewas_alid, random_alids, random_analyses
    )
    record = sample_query(patterns[shape])
    record["query"] = shape
    return record


def _shape_rss_subprocess(args: argparse.Namespace, shape: str) -> dict[str, float]:
    extra = [
        "--store", str(args.store),
        "--source-store", str(args.source_store),
        "--source-benchmark", str(args.source_benchmark),
        "--phewas-alid", args.phewas_alid,
    ]
    return run_probe(shape, extra)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--store", type=Path, default=STORE)
    ap.add_argument("--source-store", type=Path, default=SOURCE_STORE)
    ap.add_argument("--records", type=Path, default=RECORDS)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--source-benchmark", type=Path, default=SOURCE_BENCHMARK,
                    help="OGS-00009 artifact: its selection drives the shared shapes")
    ap.add_argument("--source-build-seconds", type=float, default=21298.0,
                    help="OGS-00009 build wall clock (records/build.json)")
    ap.add_argument("--output", type=Path, default=OUTPUT)
    ap.add_argument("--skip-rss", action="store_true")
    ap.add_argument("--rss-shape", default=None, help="internal: one-shape RSS probe")
    ap.add_argument("--phewas-alid", default=None, help="internal: parent-supplied variant")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    if args.rss_shape:
        print(json.dumps(_measure_shape_rss(args, args.rss_shape)))
        return

    q = query_store(args.store)
    plan = StoreManifest.load(args.store)
    an = q.analyses_table()
    analyses_by_id = {v["analysis_id"]: k for k, v in an.items()}
    n_analyses = len(an)
    n_variants = int(q._root["z"].shape[0])

    # --- cell budget from the indexed tiers: what imputation adds at the
    # genome-wide-significant end, where MR and inspection actually live.
    t0 = time.perf_counter()
    th = q.top_hits(threshold=5e-8)
    global_tophits_seconds = time.perf_counter() - t0
    status_counts = collections.Counter(str(s) for s in th["association_status"])

    exp_idx = analyses_by_id[EXPOSURE]
    m = th["analysis_index"] == exp_idx
    # The PheWAS shape runs on the SAME variant the OGS-00009 artifact timed,
    # so its result-count change (observed analyses -> all analyses) is a
    # like-for-like completeness measurement, not a different query.
    src_bench = json.loads(args.source_benchmark.read_text())
    phewas_alid = src_bench["selection"]["phewas_alid"]
    published_src_count = next(
        t["result_count"] for t in src_bench["timings"] if t["query"] == "phewas"
    )

    q_source = query_store(args.source_store)
    n_src = len(q_source.phewas(phewas_alid)["z"])
    if n_src != published_src_count:
        raise SystemExit(
            f"{args.source_benchmark} says phewas returned {published_src_count} "
            f"rows at {phewas_alid}; the store returned {n_src}. The artifact "
            "and the store disagree — not measuring against a stale reference."
        )
    n_done = len(q.phewas(phewas_alid)["z"])
    published_th_count = next(
        t["result_count"] for t in src_bench["timings"] if t["query"] == "tophits"
    )
    completeness = {
        "alid": phewas_alid,
        "n_analyses": n_analyses,
        "phewas_source_results": n_src,
        "phewas_completed_results": n_done,
        "tophits_exposure": EXPOSURE,
        "tophits_source_results": published_th_count,
        "tophits_completed_results": int(m.sum()),
    }

    timings = []
    random_alids, random_analyses = _resolve_source_lookup_selections(q_source, src_bench)
    patterns = _query_patterns(q, an, phewas_alid, random_alids, random_analyses)
    for name, fn in patterns.items():
        med, p95, cnt = _median_ms(fn, args.reps)
        timings.append(
            {"query": name, "median_ms": round(med, 3), "p95_ms": round(p95, 3),
             "result_count": cnt}
        )
        print(f"{name:38s} median={med:9.2f} ms  count={cnt:,}", flush=True)

    memory = []
    if not args.skip_rss:
        args.phewas_alid = phewas_alid
        probed = {name: _shape_rss_subprocess(args, name) for name in patterns}
        for name, rec in probed.items():
            print(f"{name:38s} baseline={rec['baseline_mb']:9.1f} MB  "
                  f"peak={rec['peak_mb']:9.1f} MB  delta={rec['delta_mb']:9.1f} MB",
                  flush=True)
        memory = list(probed.values())

    completion = _completion_section(args.store, args.records, n_variants, n_analyses)
    source_bytes = _dir_bytes(args.source_store)
    store_bytes = _dir_bytes(args.store)
    raw_bytes, n_files = _raw_vcf_bytes(args.manifest, set(analyses_by_id))

    mr = []
    hits_cache: dict[str, dict[str, Any]] = {}

    def hits(analysis_id: str) -> dict[str, Any]:
        """Top-hit columns per analysis: fetch/resolve once, clump per condition."""
        if analysis_id not in hits_cache:
            t_hit = time.perf_counter()
            hits_cache[analysis_id] = exposure_columns(q, analysis_id)
            print(
                f"top-hit columns for {analysis_id}: "
                f"{len(hits_cache[analysis_id]['z']):,} hits in "
                f"{time.perf_counter() - t_hit:.2f} s",
                flush=True,
            )
        return hits_cache[analysis_id]

    for pair in MR_PAIRS:
        cols = hits(pair["exposure_id"])
        conditions = {}
        for cond in MR_CONDITIONS:
            conditions[cond] = mr_condition(q, analyses_by_id, pair, cond, cols)
            c = conditions[cond]
            print(
                f"MR {pair['label']:55s} {cond:9s} n={c['n_instruments']:6d} "
                f"beta={c['ivw_beta']:+.4f} se={c['ivw_se']:.4f} p={c['ivw_pval']:.2g}",
                flush=True,
            )
        mr.append(
            {
                **pair,
                "exposure_label": an[analyses_by_id[pair["exposure_id"]]]["analysis_label"],
                "outcome_label": an[analyses_by_id[pair["outcome_id"]]]["analysis_label"],
                "conditions": conditions,
                "region": region_imputation_profile(q, pair["exposure_id"], cols),
            }
        )

    fidelity = observed_cell_fidelity(q, q_source, hits(MR_PAIRS[1]["exposure_id"]))
    q_source.close()
    print(
        f"observed-cell fidelity: {fidelity['n_compared']} cells, "
        f"identical={fidelity['identical']}"
    )

    result = {
        "dataset": {
            "release_id": plan.release_id,
            "source_release_id": plan.store_id,
            "n_variants": n_variants,
            "n_analyses": n_analyses,
            "completion_state": str(plan.completion_state),
            "reference_assembly": getattr(plan, "reference_assembly", None),
            "store": str(args.store),
            "format_version": plan.format_version,
            "encoding": plan.encoding.to_manifest(),
        },
        "cell_budget": {
            "total_cells": n_variants * n_analyses,
            "n_observed": completion["n_observed_cells"],
            "n_imputed": completion["n_imputed"],
            "n_missing_imputation_failed": completion["n_missing_imputation_failed"],
            "n_missing_off_panel": completion["n_missing_off_panel"],
            "tophits_5e8": {
                "total": int(sum(status_counts.values())),
                "observed": int(status_counts.get("observed", 0)),
                "imputed": int(status_counts.get("imputed", 0)),
                "global_read_seconds": round(global_tophits_seconds, 2),
            },
        },
        "storage": {
            "store_bytes": store_bytes,
            "store_gb": round(store_bytes / 1e9, 2),
            "source_store_bytes": source_bytes,
            "source_store_gb": round(source_bytes / 1e9, 2),
            "raw_vcf_bytes": raw_bytes,
            "raw_vcf_gb": round(raw_bytes / 1e9, 2),
            "n_source_files": n_files,
            "compression_ratio": round(raw_bytes / store_bytes, 2),
            "source_compression_ratio": round(raw_bytes / source_bytes, 2),
            "bytes_per_cell": round(store_bytes / (n_variants * n_analyses), 3),
            "imputation_added_bytes_per_cell": round(
                (store_bytes - source_bytes) / completion["n_imputed"], 3
            ),
        },
        "build": {
            "source_build_seconds": args.source_build_seconds,
            "completion_seconds": completion["step_record_elapsed_seconds"],
            "end_to_end_seconds": args.source_build_seconds
            + completion["step_record_elapsed_seconds"],
            "end_to_end_hours": round(
                (args.source_build_seconds + completion["step_record_elapsed_seconds"]) / 3600, 2
            ),
        },
        "completion": completion,
        "phewas_completeness": completeness,
        "selection": {
            "bulk_analysis_id": EXPOSURE,
            "phewas_alid": phewas_alid,
            "phewas_alid_source": str(args.source_benchmark),
            "region": {"chrom": REGION[0], "start": REGION[1], "end": REGION[2]},
            "regional_analysis_id": EXPOSURE,
        },
        "timings": timings,
        "memory": memory,
        "mr": mr,
        "observed_cell_fidelity": fidelity,
        "labels": {
            aid: an[analyses_by_id[aid]]["analysis_label"]
            for pair in MR_PAIRS
            for aid in (pair["exposure_id"], pair["outcome_id"])
        },
        **provenance(),
    }
    write_artifact(args.output, result)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
