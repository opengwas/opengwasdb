#!/usr/bin/env python3
"""Estimate #118's SE residual coding on rebuilt pilot Store Releases.

This is deliberately an estimate, not a store writer. It samples real `se` and
decoded `eaf` cells from existing Store Releases, fits the proposed per-Analysis
model

    log(se) = a + b * log(2 * eaf * (1 - eaf)) + residual

and projects how a residual `int8` plane would compress under the same zarr
compressor as the current `se` plane. Exception cells are charged at their raw
`int64 index + float32 value` size, so the byte estimate is conservative when
exceptions cluster.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from opengwasdb.encoding.plan import StoreEncoding
from opengwasdb.encoding.planes import DenseEafPlane, RaggedEafPlane
from opengwasdb.model import StoreManifest

MISSING = np.int8(-128)
EXCEPTION = np.int8(-127)
CANDIDATE_RANGES = (0.25, 0.5, 1.0, 2.0)
EXCEPTION_BUDGET = 0.02

DEFAULT_STORES = {
    "finngen-r13 pilot-20": "/data/opengwasdb/wip/rebuild-117/finngen-r13__r13-pilot-20",
    "ukb-b dense observed": "/data/opengwasdb/wip/rebuild-117/ukb-b__dense-observed-vcf-c128",
    "EBI GWAS Catalog hybrid": (
        "/data/opengwasdb/wip/rebuild-117/"
        "gwas-catalog-eur-hybrid__eur-hybrid-pilot-10"
    ),
}


@dataclass
class Sums:
    n: np.ndarray
    sx: np.ndarray
    sy: np.ndarray
    sxx: np.ndarray
    sxy: np.ndarray
    syy: np.ndarray
    finite_se: np.ndarray
    finite_pair: np.ndarray

    @classmethod
    def zeros(cls, n_analyses: int) -> Sums:
        return cls(*(np.zeros(n_analyses, dtype=np.float64) for _ in range(8)))

    def add_dense(self, se: np.ndarray, eaf: np.ndarray) -> None:
        se64 = np.asarray(se, dtype=np.float64)
        eaf64 = np.asarray(eaf, dtype=np.float64)
        finite_se = np.isfinite(se64) & (se64 > 0)
        usable = finite_se & np.isfinite(eaf64) & (eaf64 > 0.0) & (eaf64 < 1.0)
        self.finite_se += finite_se.sum(axis=0)
        self.finite_pair += usable.sum(axis=0)
        if not np.any(usable):
            return
        x = np.zeros_like(se64, dtype=np.float64)
        y = np.zeros_like(se64, dtype=np.float64)
        x[usable] = np.log(2.0 * eaf64[usable] * (1.0 - eaf64[usable]))
        y[usable] = np.log(se64[usable])
        self.n += usable.sum(axis=0)
        self.sx += x.sum(axis=0)
        self.sy += y.sum(axis=0)
        self.sxx += (x * x).sum(axis=0)
        self.sxy += (x * y).sum(axis=0)
        self.syy += (y * y).sum(axis=0)

    def add_ragged(self, analysis_index: np.ndarray, se: np.ndarray, eaf: np.ndarray) -> None:
        ai = np.asarray(analysis_index, dtype=np.int64)
        se64 = np.asarray(se, dtype=np.float64)
        eaf64 = np.asarray(eaf, dtype=np.float64)
        finite_se = np.isfinite(se64) & (se64 > 0)
        usable = finite_se & np.isfinite(eaf64) & (eaf64 > 0.0) & (eaf64 < 1.0)
        self.finite_se += np.bincount(ai[finite_se], minlength=len(self.n))
        self.finite_pair += np.bincount(ai[usable], minlength=len(self.n))
        if not np.any(usable):
            return
        x = np.log(2.0 * eaf64[usable] * (1.0 - eaf64[usable]))
        y = np.log(se64[usable])
        uai = ai[usable]
        self.n += np.bincount(uai, minlength=len(self.n))
        self.sx += np.bincount(uai, weights=x, minlength=len(self.n))
        self.sy += np.bincount(uai, weights=y, minlength=len(self.n))
        self.sxx += np.bincount(uai, weights=x * x, minlength=len(self.n))
        self.sxy += np.bincount(uai, weights=x * y, minlength=len(self.n))
        self.syy += np.bincount(uai, weights=y * y, minlength=len(self.n))


def coefficients(s: Sums) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    den = s.n * s.sxx - s.sx * s.sx
    slope = np.full_like(s.n, np.nan, dtype=np.float64)
    intercept = np.full_like(s.n, np.nan, dtype=np.float64)
    ok = (s.n >= 2) & (den > 0)
    slope[ok] = (s.n[ok] * s.sxy[ok] - s.sx[ok] * s.sy[ok]) / den[ok]
    intercept[ok] = (s.sy[ok] - slope[ok] * s.sx[ok]) / s.n[ok]
    sst = s.syy - (s.sy * s.sy / np.maximum(s.n, 1))
    sse = s.syy + slope * slope * s.sxx + s.n * intercept * intercept
    sse -= 2 * intercept * s.sy + 2 * slope * s.sxy - 2 * slope * intercept * s.sx
    r2 = np.full_like(s.n, np.nan, dtype=np.float64)
    ok_r2 = ok & (sst > 0)
    r2[ok_r2] = 1.0 - sse[ok_r2] / sst[ok_r2]
    return intercept, slope, r2


def dir_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except FileNotFoundError:
                pass
    return total


def evenly_spaced_starts(length: int, chunk: int, max_chunks: int) -> list[int]:
    if length <= 0:
        return []
    starts = list(range(0, length, chunk))
    if len(starts) <= max_chunks:
        return starts
    idx = np.linspace(0, len(starts) - 1, max_chunks, dtype=np.int64)
    return [starts[int(i)] for i in idx]


def residual_dense(
    se: np.ndarray, eaf: np.ndarray, intercept: np.ndarray, slope: np.ndarray
) -> np.ndarray:
    se64 = np.asarray(se, dtype=np.float64)
    eaf64 = np.asarray(eaf, dtype=np.float64)
    residual = np.full(se64.shape, np.nan, dtype=np.float64)
    usable = np.isfinite(se64) & (se64 > 0) & np.isfinite(eaf64) & (eaf64 > 0.0) & (eaf64 < 1.0)
    if np.any(usable):
        x = np.log(2.0 * eaf64[usable] * (1.0 - eaf64[usable]))
        cols = np.broadcast_to(np.arange(se64.shape[1]), se64.shape)[usable]
        residual[usable] = np.log(se64[usable]) - (intercept[cols] + slope[cols] * x)
    return residual


def residual_ragged(
    analysis_index: np.ndarray,
    se: np.ndarray,
    eaf: np.ndarray,
    intercept: np.ndarray,
    slope: np.ndarray,
) -> np.ndarray:
    ai = np.asarray(analysis_index, dtype=np.int64)
    se64 = np.asarray(se, dtype=np.float64)
    eaf64 = np.asarray(eaf, dtype=np.float64)
    residual = np.full(se64.shape, np.nan, dtype=np.float64)
    usable = np.isfinite(se64) & (se64 > 0) & np.isfinite(eaf64) & (eaf64 > 0.0) & (eaf64 < 1.0)
    if np.any(usable):
        x = np.log(2.0 * eaf64[usable] * (1.0 - eaf64[usable]))
        residual[usable] = np.log(se64[usable]) - (intercept[ai[usable]] + slope[ai[usable]] * x)
    return residual


def code_residual(
    residual: np.ndarray, residual_range: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    step = residual_range / 127.0
    codes = np.full(residual.shape, MISSING, dtype=np.int8)
    finite = np.isfinite(residual)
    exception = finite & ((residual < -126.0 * step) | (residual > 127.0 * step))
    encodable = finite & ~exception
    codes[exception] = EXCEPTION
    codes[encodable] = (
        np.rint(residual[encodable] / step).astype(np.int16).clip(-126, 127).astype(np.int8)
    )
    quantised = np.full(residual.shape, np.nan, dtype=np.float64)
    quantised[encodable] = codes[encodable].astype(np.float64) * step
    # Exceptions would be exact in the side table, so their residual error is zero.
    rel_error = np.zeros(residual.shape, dtype=np.float64)
    rel_error[encodable] = np.abs(np.exp(quantised[encodable] - residual[encodable]) - 1.0)
    rel_error[~finite] = np.nan
    return codes, exception, rel_error


def _new_accumulators(ranges: Iterable[float]) -> dict[float, dict[str, Any]]:
    return {
        r: {
            "compressed": 0,
            "exceptions": 0,
            "finite": 0,
            "errors": [],
            "roundtrip_sse": 0.0,
            "roundtrip_sum": 0.0,
            "roundtrip_sumsq": 0.0,
            "roundtrip_n": 0,
            "scatter_true": [],
            "scatter_decoded": [],
        }
        for r in ranges
    }


def record_roundtrip(
    accumulator: dict[str, Any],
    residual: np.ndarray,
    se: np.ndarray,
    codes: np.ndarray,
    exception: np.ndarray,
    residual_range: float,
) -> None:
    """Record true-versus-decoded SE on this real encoded sample.

    Exceptions stand for exact side-table values. This answers the representation
    question, rather than the different question the MAF-only regression R²
    answers: after adding the stored residual back to the predictor, how close
    is decoded SE to the SE now held in the Store Release?
    """
    finite = np.isfinite(residual)
    if not np.any(finite):
        return
    step = residual_range / 127.0
    decoded_residual = np.asarray(residual, dtype=np.float64).copy()
    encoded = finite & ~exception
    decoded_residual[encoded] = codes[encoded].astype(np.float64) * step
    true_log_se = np.log(np.asarray(se, dtype=np.float64)[finite])
    decoded_log_se = true_log_se + (decoded_residual[finite] - residual[finite])
    delta = decoded_log_se - true_log_se
    accumulator["roundtrip_sse"] += float(np.dot(delta, delta))
    accumulator["roundtrip_sum"] += float(true_log_se.sum())
    accumulator["roundtrip_sumsq"] += float(np.dot(true_log_se, true_log_se))
    accumulator["roundtrip_n"] += int(len(true_log_se))

    # The ±2.0 range is the widest candidate and the one used for the visual
    # decision; retain a bounded deterministic scatter sample in physical SE.
    if residual_range != 2.0:
        return
    remaining = 3_000 - len(accumulator["scatter_true"])
    if remaining <= 0:
        return
    stride = max(1, math.ceil(len(true_log_se) / remaining))
    accumulator["scatter_true"].extend(np.exp(true_log_se[::stride]).tolist())
    accumulator["scatter_decoded"].extend(np.exp(decoded_log_se[::stride]).tolist())


def compressed_len(compressor: Any, data: np.ndarray) -> int:
    return len(compressor.encode(np.ascontiguousarray(data)))


class Component:
    def __init__(self, name: str, path: Path, root: Any, encoding: StoreEncoding, kind: str):
        self.name = name
        self.path = path
        self.root = root
        self.encoding = encoding
        self.kind = kind
        self.se = (
            root["se"] if kind == "dense" else root["ragged/se"] if "ragged" in root else root["se"]
        )

    @property
    def n_cells(self) -> int:
        return int(np.prod(self.se.shape))

    @property
    def n_analyses(self) -> int:
        if self.kind == "dense":
            return int(self.se.shape[1])
        group = self.root["ragged"] if "ragged" in self.root else self.root
        return int(len(group["offsets"][:]) - 1)


def components_for_store(name: str, store_path: Path) -> list[Component]:
    manifest = StoreManifest.load(store_path)
    out: list[Component] = []
    if manifest.primary_layout == "hybrid":
        dense_manifest = StoreManifest.load(store_path / "dense")
        out.append(
            Component(
                f"{name} dense",
                store_path / "dense",
                zarr.open(store_path / "dense/data.zarr", mode="r"),
                dense_manifest.encoding,
                "dense",
            )
        )
        out.append(
            Component(
                f"{name} overflow",
                store_path,
                zarr.open(store_path / "data.zarr", mode="r"),
                manifest.encoding,
                "ragged",
            )
        )
    elif manifest.primary_layout == "ragged":
        out.append(
            Component(
                name,
                store_path,
                zarr.open(store_path / "data.zarr", mode="r"),
                manifest.encoding,
                "ragged",
            )
        )
    else:
        out.append(
            Component(
                name,
                store_path,
                zarr.open(store_path / "data.zarr", mode="r"),
                manifest.encoding,
                "dense",
            )
        )
    return out


def add_fit_sums(component: Component, sums: Sums, max_chunks: int) -> None:
    if component.kind == "dense":
        eaf_plane = DenseEafPlane.open(component.root, component.encoding)
        row_chunk = int(component.se.chunks[0])
        for start in evenly_spaced_starts(int(component.se.shape[0]), row_chunk, max_chunks):
            end = min(start + row_chunk, int(component.se.shape[0]))
            sums.add_dense(component.se[start:end, :], eaf_plane.band(start, end))
    else:
        group = component.root["ragged"] if "ragged" in component.root else component.root
        eaf_plane = RaggedEafPlane.open(
            group, component.encoding, imputed=group["imputed"] if "imputed" in group else None
        )
        offsets = np.asarray(group["offsets"][:], dtype=np.int64)
        chunk = int(group["se"].chunks[0])
        for start in evenly_spaced_starts(int(group["se"].shape[0]), chunk, max_chunks):
            end = min(start + chunk, int(group["se"].shape[0]))
            pos = np.arange(start, end, dtype=np.int64)
            ai = np.searchsorted(offsets[1:], pos, side="right")
            sums.add_ragged(ai, group["se"][start:end], eaf_plane.slice(start, end))


def estimate_component(
    component: Component,
    intercept: np.ndarray,
    slope: np.ndarray,
    ranges: Iterable[float],
    max_chunks: int,
) -> dict[str, Any]:
    candidates: dict[str, dict[str, Any]] = {}
    se_array_path = (
        component.path / "data.zarr" / ("se" if component.kind == "dense" else "ragged/se")
    )
    current_bytes = dir_size(se_array_path)
    if component.kind == "dense":
        eaf_plane = DenseEafPlane.open(component.root, component.encoding)
        row_chunk, col_chunk = (int(x) for x in component.se.chunks)
        starts = evenly_spaced_starts(int(component.se.shape[0]), row_chunk, max_chunks)
        total_sample_cells = 0
        accum = _new_accumulators(ranges)
        read_seconds = 0.0
        se_read_seconds = 0.0
        eaf_read_seconds = 0.0
        decode_seconds = 0.0
        for start in starts:
            end = min(start + row_chunk, int(component.se.shape[0]))
            t0 = time.perf_counter()
            se = np.asarray(component.se[start:end, :])
            t1 = time.perf_counter()
            eaf = eaf_plane.band(start, end)
            t2 = time.perf_counter()
            read_seconds += t2 - t0
            se_read_seconds += t1 - t0
            eaf_read_seconds += t2 - t1
            t1 = time.perf_counter()
            residual = residual_dense(se, eaf, intercept, slope)
            decode_seconds += time.perf_counter() - t1
            total_sample_cells += residual.size
            for r in ranges:
                codes, exception, rel_error = code_residual(residual, r)
                for c0 in range(0, codes.shape[1], col_chunk):
                    accum[r]["compressed"] += compressed_len(
                        component.se.compressor, codes[:, c0 : c0 + col_chunk]
                    )
                accum[r]["exceptions"] += int(exception.sum())
                accum[r]["finite"] += int(np.isfinite(residual).sum())
                finite_err = rel_error[np.isfinite(rel_error)]
                if len(finite_err):
                    # keep a bounded deterministic sample of error values
                    accum[r]["errors"].append(finite_err[:: max(1, len(finite_err) // 20_000)])
                record_roundtrip(accum[r], residual, se, codes, exception, r)
    else:
        group = component.root["ragged"] if "ragged" in component.root else component.root
        eaf_plane = RaggedEafPlane.open(
            group, component.encoding, imputed=group["imputed"] if "imputed" in group else None
        )
        offsets = np.asarray(group["offsets"][:], dtype=np.int64)
        chunk = int(group["se"].chunks[0])
        starts = evenly_spaced_starts(int(group["se"].shape[0]), chunk, max_chunks)
        total_sample_cells = 0
        accum = _new_accumulators(ranges)
        read_seconds = 0.0
        se_read_seconds = 0.0
        eaf_read_seconds = 0.0
        decode_seconds = 0.0
        for start in starts:
            end = min(start + chunk, int(group["se"].shape[0]))
            pos = np.arange(start, end, dtype=np.int64)
            ai = np.searchsorted(offsets[1:], pos, side="right")
            t0 = time.perf_counter()
            se = np.asarray(group["se"][start:end])
            t1 = time.perf_counter()
            eaf = eaf_plane.slice(start, end)
            t2 = time.perf_counter()
            read_seconds += t2 - t0
            se_read_seconds += t1 - t0
            eaf_read_seconds += t2 - t1
            t1 = time.perf_counter()
            residual = residual_ragged(ai, se, eaf, intercept, slope)
            decode_seconds += time.perf_counter() - t1
            total_sample_cells += residual.size
            for r in ranges:
                codes, exception, rel_error = code_residual(residual, r)
                accum[r]["compressed"] += compressed_len(group["se"].compressor, codes)
                accum[r]["exceptions"] += int(exception.sum())
                accum[r]["finite"] += int(np.isfinite(residual).sum())
                finite_err = rel_error[np.isfinite(rel_error)]
                if len(finite_err):
                    accum[r]["errors"].append(finite_err[:: max(1, len(finite_err) // 20_000)])
                record_roundtrip(accum[r], residual, se, codes, exception, r)

    scale = component.n_cells / max(total_sample_cells, 1)
    for r, a in accum.items():
        errors = np.concatenate(a["errors"]) if a["errors"] else np.empty(0)
        projected = (a["compressed"] + a["exceptions"] * 12) * scale
        finite = max(a["finite"], 1)
        roundtrip_n = int(a["roundtrip_n"])
        roundtrip_sst = a["roundtrip_sumsq"] - (a["roundtrip_sum"] ** 2 / max(roundtrip_n, 1))
        candidates[str(r)] = {
            "projected_bytes": projected,
            "projected_bpcell": projected / component.n_cells,
            "exception_fraction": a["exceptions"] / finite,
            "sample_finite_cells": int(a["finite"]),
            "relative_error_median": float(np.nanmedian(errors)) if len(errors) else None,
            "relative_error_p99": float(np.nanquantile(errors, 0.99)) if len(errors) else None,
            "relative_error_max_sample": float(np.nanmax(errors)) if len(errors) else None,
            "roundtrip_log_se_r2": (
                float(1.0 - a["roundtrip_sse"] / roundtrip_sst) if roundtrip_sst > 0 else None
            ),
            "roundtrip_sample_cells": roundtrip_n,
            "scatter_true_se": a["scatter_true"] if r == 2.0 else [],
            "scatter_decoded_se": a["scatter_decoded"] if r == 2.0 else [],
        }
    return {
        "name": component.name,
        "kind": component.kind,
        "shape": list(component.se.shape),
        "chunks": list(component.se.chunks),
        "n_cells": component.n_cells,
        "current_se_bytes": current_bytes,
        "current_se_bpcell": current_bytes / component.n_cells,
        "sample_cells": total_sample_cells,
        "read_seconds": read_seconds,
        "se_read_seconds": se_read_seconds,
        "eaf_read_seconds": eaf_read_seconds,
        "residual_compute_seconds": decode_seconds,
        "candidates": candidates,
    }


def analyse_store(name: str, path: Path, max_chunks: int) -> dict[str, Any]:
    components = components_for_store(name, path)
    n_analyses = max(c.n_analyses for c in components)
    sums = Sums.zeros(n_analyses)
    for c in components:
        add_fit_sums(c, sums, max_chunks=max_chunks)
    intercept, slope, r2 = coefficients(sums)
    component_results = [
        estimate_component(c, intercept, slope, CANDIDATE_RANGES, max_chunks) for c in components
    ]
    totals: dict[str, Any] = {}
    n_cells = sum(c["n_cells"] for c in component_results)
    current = sum(c["current_se_bytes"] for c in component_results)
    for r in CANDIDATE_RANGES:
        key = str(r)
        projected = sum(c["candidates"][key]["projected_bytes"] for c in component_results)
        finite = sum(c["candidates"][key]["sample_finite_cells"] for c in component_results)
        exceptions = sum(
            c["candidates"][key]["exception_fraction"] * c["candidates"][key]["sample_finite_cells"]
            for c in component_results
        )
        totals[key] = {
            "projected_bytes": projected,
            "projected_bpcell": projected / n_cells,
            "saving_bytes": current - projected,
            "saving_fraction": (current - projected) / current,
            "exception_fraction": exceptions / max(finite, 1),
        }
    feasible = bool(
        np.all(np.isfinite(r2))
        and np.nanmin(r2) >= 0.99
        and np.all(sums.finite_pair == sums.finite_se)
    )
    chosen_range = None
    if feasible:
        for r in CANDIDATE_RANGES:
            if totals[str(r)]["exception_fraction"] <= EXCEPTION_BUDGET:
                chosen_range = r
                break
    return {
        "name": name,
        "path": str(path),
        "n_analyses": n_analyses,
        "components": component_results,
        "current_se_bytes": current,
        "current_se_bpcell": current / n_cells,
        "n_cells": n_cells,
        "fit": {
            "min_r2": float(np.nanmin(r2)),
            "median_r2": float(np.nanmedian(r2)),
            "max_r2": float(np.nanmax(r2)),
            "min_slope": float(np.nanmin(slope)),
            "median_slope": float(np.nanmedian(slope)),
            "max_slope": float(np.nanmax(slope)),
            "min_eaf_coverage": float(np.nanmin(sums.finite_pair / np.maximum(sums.finite_se, 1))),
            "analyses_below_0_99_r2": int(np.sum(r2 < 0.99)),
        },
        "feasible_all_or_none_r2_0_99": feasible,
        "chosen_range_at_2pct_exceptions": chosen_range,
        "totals_by_range": totals,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/benchmark-output/opengwasdb_se_residual_expectation.json"),
    )
    parser.add_argument("--max-chunks", type=int, default=40)
    args = parser.parse_args()
    results = []
    for name, path_text in DEFAULT_STORES.items():
        path = Path(path_text)
        if path.exists():
            results.append(analyse_store(name, path, max_chunks=args.max_chunks))
    payload = {
        "method": "sampled log(se) ~ a + b*log(2*eaf*(1-eaf)); int8 residual projection",
        "exception_budget": EXCEPTION_BUDGET,
        "ranges": list(CANDIDATE_RANGES),
        "max_chunks_per_component": args.max_chunks,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "stores": len(results)}, indent=2))


if __name__ == "__main__":
    main()
