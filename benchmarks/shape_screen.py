"""Screen candidate Dense chunk shapes for format 0.2.0 from a 100,000-variant slice.

Converting the whole of OGS-00009 into one candidate shape takes hours. This
screens shapes first, from raw reads measured on a 100,000-variant slice of its
`z` plane (`benchmarks/shape_slice.py`), the per-plane decode times
(`benchmarks/shape_decode_by_chunk.py`), what each harness shape reads
(`benchmarks/shape_harness_geometry.py`) and the #244 attribution timings
(`benchmarks/zarr3_attribution.py`). #246's body quotes its results. Steps, in
order, each reading the previous step's output from `<outputs>/slice/`:

  cost-model   fit t = k + inner_chunks x (o + decode) + m x MB per read, with the
               decode term measured rather than fitted, on the sharded
               `[1000, 1000]` and `[1000, 64]` slice arrays; every other array is
               held out and predicted -> cost_model.json
  screen       per harness shape and candidate shape, the modelled end-to-end time:
               measured at today's layout, minus the modelled read there, plus the
               modelled read at the candidate -> screen.json
  expected     model-free: each shape's measured zarr 3 time, scaled by the slice's
               within-process read ratio candidate / today's layout -> expected_020.json
  rank-check   does the model rank candidates the way the slice measures them?
               -> screen_rank_check.json
  summary      the slice's raw read latencies as a markdown table

The model ranks shapes; it does not set numbers. On the full store it
over-counts absolute read time by 2-4x (the negative "non-read remainder"
`screen` prints), so only ratios between shapes are used.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks import _zarr3_levers as levers

#: The arrays the model is fitted on; every other slice array is held out.
FIT_V3 = ["v3_c1000_s", "v3_c64_s"]
#: c-blosc decodes a buffer on one thread unless it spans two 128 KiB blocks.
THREADED_BYTES = 2 * 131_072
#: Each harness shape's raw-read analogue on the slice.
ANALOGUE = {
    "bulk": "fullcol_100k",
    "phewas": "row",
    "regional_one_analysis": "colseg_3000",
    "regional": "band_2000",
    "rand_10x100": "oindex_10x100",
    "rand_100x10": "oindex_100x10",
}
READ_BOUND = {"bulk", "phewas", "rand_10x100", "rand_100x10"}
SLICE_V3 = [
    "v3_c1000_s",
    "v3_c128_s",
    "v3_c64_s",
    "v3_r2000c128_s",
    "v3_r4000c256_s",
    "v3_r250c512_s",
]
#: Candidate shapes built on the slice, by the candidate name the screen uses.
BUILT = {
    "1000x128": "v3_c128_s",
    "1000x64": "v3_c64_s",
    "2000x128": "v3_r2000c128_s",
    "4000x256": "v3_r4000c256_s",
    "250x512": "v3_r250c512_s",
}


# ── the decode term ──────────────────────────────────────────────────────────


def decode_table(slice_dir: Path) -> dict[tuple[str, int, int], float]:
    """Measured decode ms per (plane, R, C): the median over the decode runs."""
    acc: dict[tuple[str, int, int], list[float]] = {}
    for run in levers.read_jsonl(slice_dir / "decode_by_plane.jsonl"):
        for d in run["decode"]:
            acc.setdefault((d["plane"], d["chunk"][0], d["chunk"][1]), []).append(d["decode_ms"])
    return {k: statistics.median(v) for k, v in acc.items()}


def decode_model(slice_dir: Path) -> dict[str, float]:
    """Serial ms/MB, and threaded intercept + ms/MB, from the `z` decode measurements.

    Used for the slice fit (the slice is `z`) and for arrays the per-plane table
    does not cover (1-D per-variant and top-hit arrays).
    """
    runs = levers.read_jsonl(slice_dir / "decode_by_plane.jsonl")
    pts: dict[int, list[float]] = {}
    for run in runs:
        for d in run["decode"]:
            if d["plane"] == "z":
                pts.setdefault(d["bytes"], []).append(d["decode_ms"])
    ser = [(b, statistics.median(v)) for b, v in pts.items() if b < THREADED_BYTES]
    thr = [(b, statistics.median(v)) for b, v in pts.items() if b >= THREADED_BYTES]
    ser_rate = statistics.median(ms / (b / 1e6) for b, ms in ser)
    x = np.array([[1.0, b / 1e6] for b, _ in thr])
    y = np.array([ms for _, ms in thr])
    (thr_k, thr_rate), *_ = np.linalg.lstsq(x, y, rcond=None)
    return {
        "serial_ms_per_MB": ser_rate,
        "threaded_ms_fixed": float(thr_k),
        "threaded_ms_per_MB": float(thr_rate),
        "n_sizes": len(pts),
        "runs": len(runs),
    }


def decode_ms(dm: dict[str, float], chunk_bytes: float) -> float:
    if chunk_bytes < THREADED_BYTES:
        return dm["serial_ms_per_MB"] * chunk_bytes / 1e6
    return dm["threaded_ms_fixed"] + dm["threaded_ms_per_MB"] * chunk_bytes / 1e6


# ── cost-model ───────────────────────────────────────────────────────────────


def _slice_reads(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    """Per (array, read): the median over processes, with min and max."""
    acc: dict[tuple[str, str], dict[str, list[float]]] = {}
    for run in levers.read_jsonl(path):
        for name, arr in run["arrays"].items():
            r_, c_ = arr["chunk"]
            for shape, r in arr["reads"].items():
                d = acc.setdefault((name, shape), {"ms": [], "chunks": [], "shards": [], "cb": []})
                d["ms"].append(r["ms"])
                d["chunks"].append(r["chunks"])
                d["shards"].append(r["shards"])
                d["cb"].append(r_ * c_ * 2)
    return {
        key: {
            "ms": statistics.median(v["ms"]),
            "ms_min": min(v["ms"]),
            "ms_max": max(v["ms"]),
            "chunks": v["chunks"][0],
            "shards": v["shards"][0],
            "chunk_bytes": v["cb"][0],
            "rounds": len(v["ms"]),
        }
        for key, v in acc.items()
    }


def fit(
    rows: list[dict[str, float]], dm: dict[str, float], with_m: bool = True
) -> tuple[float, float, float]:
    """k, o and m by weighted least squares, the measured decode term subtracted first."""
    x = np.array(
        [
            [1.0, r["chunks"]] + ([r["chunks"] * r["chunk_bytes"] / 1e6] if with_m else [])
            for r in rows
        ]
    )
    y = np.array([r["ms"] - r["chunks"] * decode_ms(dm, r["chunk_bytes"]) for r in rows])
    w = 1.0 / np.array([r["ms"] for r in rows])
    coef, *_ = np.linalg.lstsq(x * w[:, None], y * w, rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2]) if with_m else 0.0


def predict(
    k: float, o: float, m: float, dm: dict[str, float], chunks: float, chunk_bytes: float
) -> float:
    return k + chunks * (o + decode_ms(dm, chunk_bytes)) + m * chunks * chunk_bytes / 1e6


def _error(rows: list[dict[str, Any]]) -> dict[str, float]:
    e = [abs(float(t["pred_ms"]) - float(t["ms"])) / float(t["ms"]) for t in rows]
    ratio = [float(t["pred_ms"]) / float(t["ms"]) for t in rows]
    return {
        "n": len(e),
        "median_abs_pct": 100 * statistics.median(e),
        "max_abs_pct": 100 * max(e),
        "within_25pct": sum(x <= 0.25 for x in e),
        "within_50pct": sum(x <= 0.5 for x in e),
        "pred_over_meas_min": min(ratio),
        "pred_over_meas_max": max(ratio),
    }


def _fits(
    base: dict[tuple[str, str], dict[str, float]],
    head: dict[tuple[str, str], dict[str, float]],
    dm: dict[str, float],
) -> dict[str, Any]:
    # One chunk size on 2.18 and on zarr 3 unsharded: handling and decode are not
    # separable there, so those fits carry no m.
    bk, bo, _ = fit(list(base.values()), dm, with_m=False)
    uk, uo, _ = fit([v for (n, _), v in head.items() if n == "v2_c1000"], dm, with_m=False)
    fitted = [v for (n, _), v in head.items() if n in FIT_V3]
    ak, ao, _ = fit(fitted, dm, with_m=False)
    hk, ho, hm = fit(fitted, dm)
    return {
        "fit_218": {
            "k_ms": bk,
            "per_chunk_overhead_ms": bo,
            "fitted_on": ["v2_c1000"],
            "note": "one chunk size: overhead and handling not separable",
        },
        "fit_z3_step1_sharded": {
            "k_ms": hk,
            "per_chunk_overhead_ms": ho,
            "per_MB_handling_ms": hm,
            "fitted_on": FIT_V3,
            "model": "B (with handling term)",
        },
        "fit_z3_step1_sharded_A": {
            "k_ms": ak,
            "per_chunk_overhead_ms": ao,
            "per_MB_handling_ms": 0.0,
            "fitted_on": FIT_V3,
            "model": "A (decode measured, no handling term)",
        },
        "fit_z3_step1_unsharded": {
            "k_ms": uk,
            "per_chunk_overhead_ms": uo,
            "fitted_on": ["v2_c1000"],
        },
    }


def _fit_params(f: dict[str, Any]) -> tuple[float, float, float]:
    return f["k_ms"], f["per_chunk_overhead_ms"], f.get("per_MB_handling_ms", 0.0)


def _prediction_table(
    base: dict[tuple[str, str], dict[str, float]],
    head: dict[tuple[str, str], dict[str, float]],
    dm: dict[str, float],
    fits: dict[str, Any],
) -> list[dict[str, Any]]:
    hk, ho, hm = _fit_params(fits["fit_z3_step1_sharded"])
    bk, bo, _ = _fit_params(fits["fit_218"])
    table: list[dict[str, Any]] = []
    for (name, shape), v in sorted(head.items()):
        p = None if name == "v2_c1000" else predict(hk, ho, hm, dm, v["chunks"], v["chunk_bytes"])
        table.append(
            {
                "env": "zarr3-step1",
                "array": name,
                "shape": shape,
                **v,
                "pred_ms": p,
                "held_out": name not in FIT_V3 and name != "v2_c1000",
            }
        )
    for (name, shape), v in sorted(base.items()):
        pred = predict(bk, bo, 0.0, dm, v["chunks"], v["chunk_bytes"])
        table.append(
            {
                "env": "zarr2.18",
                "array": name,
                "shape": shape,
                **v,
                "pred_ms": pred,
                "held_out": False,
            }
        )
    return table


def _held_out_errors(
    table: list[dict[str, Any]], dm: dict[str, float], fits: dict[str, Any]
) -> dict[str, Any]:
    ak, ao, _ = _fit_params(fits["fit_z3_step1_sharded_A"])
    held = [t for t in table if t["held_out"]]
    held_a = [
        {**t, "pred_ms": predict(ak, ao, 0.0, dm, t["chunks"], t["chunk_bytes"])} for t in held
    ]
    lo_hi = [
        (min(a["pred_ms"], b["pred_ms"]), max(a["pred_ms"], b["pred_ms"]), b["ms"])
        for a, b in zip(held_a, held, strict=True)
    ]
    return {
        "held_out_error": _error(held),
        "held_out_error_A": _error(held_a),
        "held_out_envelope": {
            "n": len(lo_hi),
            "measured_inside_A_B_range": sum(lo <= m <= hi for lo, hi, m in lo_hi),
            "measured_within_25pct_of_range": sum(
                lo * 0.75 <= m <= hi * 1.25 for lo, hi, m in lo_hi
            ),
        },
        "held_out_error_reads_over_5ms": _error([t for t in held if float(t["ms"]) > 5]),
        "in_sample_error": _error(
            [t for t in table if t["env"] == "zarr3-step1" and t["array"] in FIT_V3]
        ),
        "in_sample_error_218": _error([t for t in table if t["env"] == "zarr2.18"]),
    }


def cost_model(slice_dir: Path) -> dict[str, Any]:
    dm = decode_model(slice_dir)
    base = _slice_reads(slice_dir / "slice_read_base.jsonl")
    head = _slice_reads(slice_dir / "slice_read_step1.jsonl")
    fits = _fits(base, head, dm)
    table = _prediction_table(base, head, dm, fits)
    out: dict[str, Any] = {"decode": dm, **fits, "table": table}
    out.update(_held_out_errors(table, dm, fits))
    print(json.dumps({k: v for k, v in out.items() if k != "table"}, indent=1))
    print(
        f"{'env':12s} {'array':16s} {'shape':14s} {'chunks':>7s} {'chunkKB':>8s} {'meas ms':>9s} "
        f"{'range':>17s} {'pred ms':>9s} held"
    )
    for t in table:
        p = "" if t["pred_ms"] is None else f"{t['pred_ms']:9.2f}"
        print(
            f"{t['env']:12s} {t['array']:16s} {t['shape']:14s} {t['chunks']:7.0f} "
            f"{t['chunk_bytes'] / 1e3:8.0f} {t['ms']:9.2f} {t['ms_min']:8.2f}-{t['ms_max']:<8.2f} "
            f"{p:>9s} {'*' if t['held_out'] else ''}"
        )
    return out


# ── screen ───────────────────────────────────────────────────────────────────


def _read_cost(
    fitp: dict[str, float],
    dm: dict[str, float],
    decode: dict[tuple[str, int, int], float],
    read: tuple[int, int, str, tuple[int, ...]],
) -> float:
    """The modelled time of one array read: (inner chunks, chunk bytes, array, chunk)."""
    n, chunk_bytes, array, chunk = read
    m = fitp.get("per_MB_handling_ms", 0.0)
    if len(chunk) == 2 and (array, chunk[0], chunk[1]) in decode:
        d = decode[(array, chunk[0], chunk[1])]
    else:
        d = decode_ms(dm, chunk_bytes)
    return fitp["k_ms"] + n * (fitp["per_chunk_overhead_ms"] + d) + m * n * chunk_bytes / 1e6


def _stored_read(r: dict[str, Any]) -> tuple[int, int, str, tuple[int, ...]]:
    stored = r["stored_chunks"]
    chunk_bytes = int(r["itemsize"]) * (stored[0] * (stored[1] if len(stored) > 1 else 1))
    return int(r["stored_n"]), chunk_bytes, str(r["array"]), tuple(stored)


def _candidate_read(r: dict[str, Any], cand: str) -> tuple[int, int, str, tuple[int, ...]]:
    rows, cols = (int(x) for x in cand.split("x"))
    if r["role"] == "plane":
        chunk: tuple[int, ...] = (rows, cols)
    elif r["role"] == "other":
        chunk = tuple(r["stored_chunks"])
    else:
        chunk = (min(rows, 200_000),)
    c = r["candidates"][cand]
    return int(c["n"]), int(c["chunk_bytes"]), str(r["array"]), chunk


def _screen_shape(
    recs: list[dict[str, Any]],
    cands: list[str],
    cm: dict[str, Any],
    decode: dict[tuple[str, int, int], float],
    measured: tuple[float, float],
) -> dict[str, Any]:
    dm = cm["decode"]
    fa, fb = cm["fit_z3_step1_sharded_A"], cm["fit_z3_step1_sharded"]
    meas_head, meas_218 = measured
    r_u = sum(_read_cost(cm["fit_z3_step1_unsharded"], dm, decode, _stored_read(r)) for r in recs)
    remainder = meas_head - r_u
    rows = {}
    for cand in cands:
        ra = sum(_read_cost(fa, dm, decode, _candidate_read(r, cand)) for r in recs)
        rb = sum(_read_cost(fb, dm, decode, _candidate_read(r, cand)) for r in recs)
        lo, hi = min(ra, rb) * 0.75, max(ra, rb) * 1.25
        rows[cand] = {
            "chunks": sum(int(r["candidates"][cand]["n"]) for r in recs),
            "read_A_ms": ra,
            "read_B_ms": rb,
            "e2e_lo_ms": remainder + lo,
            "e2e_hi_ms": remainder + hi,
            "e2e_mid_ms": remainder + (ra + rb) / 2,
        }
    return {
        "reads": len(recs),
        "stored_chunks": sum(int(r["stored_n"]) for r in recs),
        "measured_head_ms": meas_head,
        "measured_218_ms": meas_218,
        "read_unsharded_c1000_ms": r_u,
        "non_read_remainder_ms": remainder,
        "candidates": rows,
    }


def screen(outputs: Path) -> dict[str, Any]:
    slice_dir = outputs / "slice"
    cm = json.loads((slice_dir / "cost_model.json").read_text())
    geo = json.loads((slice_dir / "harness_geometry.json").read_text())
    attribution = outputs / "attribution" / "attribution.jsonl"
    head = levers.attribution_medians(attribution, "iv_head")
    b218 = levers.attribution_medians(attribution, "base_218")
    decode = decode_table(slice_dir)
    reads: dict[str, list[dict[str, Any]]] = {}
    for rec in geo["log"]:
        if not rec.get("summary"):
            reads.setdefault(str(rec["shape_name"]), []).append(rec)
    cands = [f"{r}x{c}" for r, c in geo["candidates"]]
    shapes = {
        shape: _screen_shape(recs, cands, cm, decode, (head[shape], b218[shape]))
        for shape, recs in reads.items()
    }
    for shape, d in shapes.items():
        print(
            f"\n{shape}: reads {d['reads']}, stored chunks {d['stored_chunks']}, "
            f"measured head {d['measured_head_ms']:.1f} ms (2.18 {d['measured_218_ms']:.1f}), "
            f"modelled read at unsharded c1000 {d['read_unsharded_c1000_ms']:.1f} ms, "
            f"non-read remainder {d['non_read_remainder_ms']:.1f} ms"
        )
        for cand, r in d["candidates"].items():
            print(
                f"   {cand:10s} chunks {r['chunks']:7d}  e2e {r['e2e_lo_ms']:10.1f} - "
                f"{r['e2e_hi_ms']:10.1f} ms  (mid {r['e2e_mid_ms']:.1f})"
            )
    return {"shapes": shapes}


# ── expected ─────────────────────────────────────────────────────────────────


def slice_runs(slice_dir: Path, label: str) -> list[dict[str, Any]]:
    """Every slice-read process of one label: round 1, then the later rounds."""
    return levers.read_jsonl(
        slice_dir / "round1" / f"slice_read_{label}.jsonl"
    ) + levers.read_jsonl(slice_dir / f"slice_read_{label}.jsonl")


def _expected_row(
    shape: str,
    measured: tuple[float, float],
    share: float,
    ratios: dict[str, dict[str, list[float]]],
) -> dict[str, Any]:
    head, b218 = measured
    row: dict[str, Any] = {"measured_218_ms": b218, "measured_head_ms": head}
    row["read_share"] = share
    row["analogue"] = ANALOGUE[shape]
    for name in SLICE_V3:
        vals = ratios[name].get(ANALOGUE[shape], [])
        if not vals:
            continue
        med = statistics.median(vals)
        row[name] = {
            "ratio_median": med,
            "ratio_min": min(vals),
            "ratio_max": max(vals),
            "n": len(vals),
            "expected_ms": head * ((1 - share) + share * med),
            "expected_lo_ms": head * ((1 - share) + share * min(vals)),
            "expected_hi_ms": head * ((1 - share) + share * max(vals)),
        }
    return row


def expected(outputs: Path) -> dict[str, Any]:
    slice_dir = outputs / "slice"
    runs = slice_runs(slice_dir, "step1")
    ratios: dict[str, dict[str, list[float]]] = {}
    shard_cost: list[float] = []
    for run in runs:
        base = run["arrays"]["v2_c1000"]["reads"]
        for name in SLICE_V3:
            for read, r in run["arrays"][name]["reads"].items():
                ratios.setdefault(name, {}).setdefault(read, []).append(r["ms"] / base[read]["ms"])
        shard_cost.append(run["arrays"]["v3_c1000_s"]["reads"]["cell"]["ms"] - base["cell"]["ms"])
    attribution = outputs / "attribution" / "attribution.jsonl"
    head = levers.attribution_medians(attribution, "iv_head")
    b218 = levers.attribution_medians(attribution, "base_218")
    screened = json.loads((slice_dir / "screen.json").read_text())["shapes"]
    per_read = statistics.median(shard_cost)
    shapes: dict[str, Any] = {}
    for shape in levers.HARNESS_ORDER:
        if shape == "tophits":
            # Only top-hit arrays, which the Dense inner chunk does not touch. If
            # 0.2.0 shards them, each of the 6 reads pays the slice's per-read cost.
            shapes[shape] = {
                "measured_218_ms": b218[shape],
                "measured_head_ms": head[shape],
                "expected_if_tophit_arrays_unsharded_ms": head[shape],
                "expected_if_tophit_arrays_sharded_ms": head[shape] + 6 * per_read,
            }
            continue
        share = (
            1.0
            if shape in READ_BOUND
            else min(1.0, screened[shape]["read_unsharded_c1000_ms"] / head[shape])
        )
        shapes[shape] = _expected_row(shape, (head[shape], b218[shape]), share, ratios)
    cost = {"median": per_read, "min": min(shard_cost), "max": max(shard_cost)}
    print(
        f"processes {len(runs)}; per-read sharding cost (cell) median {cost['median']:.2f} ms "
        f"[{cost['min']:.2f}, {cost['max']:.2f}]"
    )
    for shape, row in shapes.items():
        if shape == "tophits":
            print(
                f"{shape:22s} 2.18 {row['measured_218_ms']:9.2f} "
                f"head {row['measured_head_ms']:9.2f} | "
                f"unsharded top-hit arrays {row['expected_if_tophit_arrays_unsharded_ms']:.2f}, "
                f"sharded {row['expected_if_tophit_arrays_sharded_ms']:.2f}"
            )
            continue
        cells = " ".join(
            f"{n.replace('v3_', '').replace('_s', '')}={row[n]['expected_ms']:.1f} "
            f"(x{row[n]['ratio_median']:.2f})"
            for n in SLICE_V3
            if n in row
        )
        print(
            f"{shape:22s} 2.18 {row['measured_218_ms']:9.1f} head {row['measured_head_ms']:9.1f} "
            f"s={row['read_share']:.2f} | {cells}"
        )
    return {"processes": len(runs), "per_read_shard_cost_ms": cost, "shapes": shapes}


# ── rank-check ───────────────────────────────────────────────────────────────


def rank_check(outputs: Path) -> dict[str, Any]:
    slice_dir = outputs / "slice"
    screened = json.loads((slice_dir / "screen.json").read_text())["shapes"]
    runs = slice_runs(slice_dir, "step1")
    out: dict[str, Any] = {}
    errs = []
    print(f"{'shape':22s} {'cand':10s} {'model A':>8s} {'model B':>8s} {'slice':>8s}")
    for shape, analogue in ANALOGUE.items():
        cands = screened[shape]["candidates"]
        ref = cands["1000x1000"]
        rows = {}
        for cand, r in cands.items():
            ma, mb = r["read_A_ms"] / ref["read_A_ms"], r["read_B_ms"] / ref["read_B_ms"]
            meas = None
            if cand in BUILT:
                vals = [
                    run["arrays"][BUILT[cand]]["reads"][analogue]["ms"]
                    / run["arrays"]["v3_c1000_s"]["reads"][analogue]["ms"]
                    for run in runs
                    if analogue in run["arrays"][BUILT[cand]]["reads"]
                ]
                meas = statistics.median(vals) if vals else None
                if meas:
                    # Zero when the measured ratio lies between A and B, else the
                    # distance from the nearer of the two.
                    outside = (ma - meas) * (mb - meas) > 0
                    errs.append(max(abs(ma / meas - 1), abs(mb / meas - 1)) if outside else 0.0)
            rows[cand] = {"model_A": ma, "model_B": mb, "slice_measured": meas}
            m = "" if meas is None else f"{meas:8.2f}"
            print(f"{shape:22s} {cand:10s} {ma:8.2f} {mb:8.2f} {m:>8s}")
        out[shape] = rows
    out["ratio_error_summary"] = {
        "n": len(errs),
        "inside_A_B": sum(e == 0.0 for e in errs),
        "median_pct": 100 * statistics.median(errs),
        "max_pct": 100 * max(errs),
    }
    print(json.dumps(out["ratio_error_summary"]))
    return out


# ── summary ──────────────────────────────────────────────────────────────────

SUMMARY_READS = [
    ("cell", "one cell"),
    ("row", "one row, all 2,024 Analyses (phewas-shaped)"),
    ("oindex_10x100", "10 variants × 100 Analyses"),
    ("oindex_100x10", "100 variants × 10 Analyses"),
    ("colseg_3000", "3,000 variants × 1 Analysis (regional one-Analysis)"),
    ("band_2000", "2,000 variants × all Analyses (regional)"),
    ("fullcol_100k", "100,000 variants × 1 Analysis (bulk, per 100k rows)"),
]
SUMMARY_COLUMNS = [
    ("base", "v2_c1000", "2.18, v2 `[1000,1000]`"),
    ("step1", "v2_c1000", "zarr 3, v2 `[1000,1000]` (today)"),
    ("step1", "v3_c1000_s", "zarr 3, v3 sharded `[1000,1000]`"),
    ("step1", "v3_c128_s", "zarr 3, v3 sharded `[1000,128]`"),
    ("step1", "v3_c64_s", "zarr 3, v3 sharded `[1000,64]`"),
]


def summary(outputs: Path) -> None:
    slice_dir = outputs / "slice"
    data = {"base": slice_runs(slice_dir, "base"), "step1": slice_runs(slice_dir, "step1")}
    print("| read | " + " | ".join(c[2] for c in SUMMARY_COLUMNS) + " |")
    print("|---|" + "---:|" * len(SUMMARY_COLUMNS))
    for key, label in SUMMARY_READS:
        cells = []
        for kind, arr, _ in SUMMARY_COLUMNS:
            v = [
                r["arrays"][arr]["reads"][key]["ms"]
                for r in data[kind]
                if key in r["arrays"][arr]["reads"]
            ]
            cells.append(f"{statistics.median(v):.1f} ({min(v):.1f}–{max(v):.1f})")
        print(f"| {label} | " + " | ".join(cells) + " |")
    n = {k: len(v) for k, v in data.items()}
    print(
        f"\nms; median over processes (2.18: {n['base']}, zarr 3: {n['step1']}; the 100 × 10 "
        "read only in the last 3 of each), min–max across processes."
    )


STEPS: dict[str, Callable[[Path], dict[str, Any]]] = {
    "cost-model": lambda outputs: cost_model(outputs / "slice"),
    "screen": screen,
    "expected": expected,
    "rank-check": rank_check,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=[*STEPS, "summary"])
    ap.add_argument("--outputs", type=Path, default=levers.OUTPUTS, help="#244's output directory")
    ap.add_argument("--json", type=Path, default=None, help="write the step's result here")
    args = ap.parse_args()
    if args.step == "summary":
        summary(args.outputs)
        return
    result = STEPS[args.step](args.outputs)
    if args.json is not None:
        args.json.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
