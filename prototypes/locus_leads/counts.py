"""Judging `n_loci_*` by counts: rank, density and cM against bp-only (#403 fix round 1).

The per-lead rates say where Locus Leads and plink2's r^2 0.01 index variants
disagree one by one; the owner's decision is that a per-Analysis count is judged
by counts. The criteria below were fixed before they were computed, and are
written into the output with this wording.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

BP_ONLY = "bp_only"
PRIMARY_R2 = "0.01"

COUNT_CRITERIA: dict[str, Any] = {
    "C1": "rank. Spearman rho(Locus Leads, r2 0.01 index variants) over the swept Analyses is "
          ">= 0.95.",
    "C2": "density. For each swept Analysis, restrict the shared set to HapMap3 variants and "
          "recompute the leads at every (D, W) and bp-only; no re-clumping. Density ratio = "
          "leads(HM3) / leads(full). C2 holds when the median ratio over Analyses is in "
          "[0.8, 1.25] and its median |log2| is at most half that of the same ratio for raw "
          "row counts (rows(HM3) / rows(full)).",
    "C3": "cM against bp-only, at the same W. cM adds value when its best D improves on bp-only "
          "by at least 0.01 in rho (C1) or by at least 0.1 in median |log2 density ratio| (C2), "
          "without being worse on the other by more than those margins.",
    "rho_min": 0.95,
    "ratio_range": [0.8, 1.25],
    "log2_at_most_this_share_of_rows": 0.5,
    "c3_margin_rho": 0.01,
    "c3_margin_abs_log2": 0.1,
    "c3_reading": "at a W, C3 holds when some D improves on that W's bp-only by at least one "
                  "margin and is worse on neither by more than its margin; the D that do are "
                  "listed.",
    "named": "the (D, W), bp-only included, that meets C1 and C2 and has the smallest pooled "
             "share of leads lost to density, sum over Analyses of max(0, leads(full) - "
             "leads(HM3)) / sum of leads(full); the best cM point is named the same way.",
    "index_variants": f"plink2 at r2 {PRIMARY_R2}, on the same shared set as the leads",
    "hm3": "the canonical ALIDs of /data/opengwasdb/reference/hapmap3-hg38/hm3.grch38.alid.txt",
}


def _ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks, ties sharing the mean of the ranks they span."""
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(values.size)
    start = 0
    while start < values.size:
        end = start
        while end + 1 < values.size and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        ranks[order[start:end + 1]] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def spearman(x: Any, y: Any) -> float:
    """Spearman's rho: the Pearson correlation of average ranks."""
    a, b = _ranks(np.asarray(x, dtype="float64")), _ranks(np.asarray(y, dtype="float64"))
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        raise ValueError("Spearman's rho is undefined for a constant series")
    return float(np.corrcoef(a, b)[0, 1])


def _abs_log2(ratio: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.asarray(np.abs(np.log2(ratio)))


def _finite_or_none(value: float) -> float | None:
    return None if math.isinf(value) else float(value)


def _median_or_inf(values: np.ndarray) -> float:
    return float(np.median(values))


def _density_holds(median_ratio: float, median_log: float, rows_median_log: float) -> bool:
    """C2: the median ratio is in range and its |log2| at most half the rows' own."""
    low, high = COUNT_CRITERIA["ratio_range"]
    share = COUNT_CRITERIA["log2_at_most_this_share_of_rows"]
    return bool(low <= median_ratio <= high and median_log <= share * rows_median_log)


def _config_counts(analyses: list[dict[str, Any]], key: tuple[Any, int],
                   rows_median_abs_log2: float) -> dict[str, Any]:
    entries = [next(c for c in a["configs"] if (c["d_cm"], c["w_bp"]) == key) for a in analyses]
    return {"d_cm": key[0], "w_bp": key[1]} | count_row(entries, rows_median_abs_log2, str(key))


def count_row(entries: list[dict[str, Any]], rows_median_abs_log2: float,
              label: str) -> dict[str, Any]:
    """C1 and C2 for one rule, from each Analysis's `density` and r^2 0.01 `index_variants`.

    Any rule's leads can be scored so, (D, W) or not (#409).
    """
    full = np.array([e["density"]["leads_full"] for e in entries], dtype="float64")
    hm3 = np.array([e["density"]["leads_hm3"] for e in entries], dtype="float64")
    index = [e["by_r2"][PRIMARY_R2]["index_variants"] for e in entries]
    if (full <= 0).any():
        raise ValueError(f"{label}: an Analysis has no Locus Lead, so its density ratio is "
                         "undefined")
    ratio = hm3 / full
    median_log = _median_or_inf(_abs_log2(ratio))
    rho = spearman(full, index)
    return {
        "rho": rho,
        "c1": bool(rho >= COUNT_CRITERIA["rho_min"]),
        "median_ratio": float(np.median(ratio)),
        "median_abs_log2": _finite_or_none(median_log),
        "rows_median_abs_log2": rows_median_abs_log2,
        "analyses_without_hm3_leads": int((hm3 == 0).sum()),
        "c2": _density_holds(float(np.median(ratio)), median_log, rows_median_abs_log2),
        "leads_full": int(full.sum()), "leads_hm3": int(hm3.sum()),
        "share_of_leads_lost_to_density": float(np.maximum(full - hm3, 0).sum() / full.sum()),
    }


def _abs_log2_of(row: dict[str, Any]) -> float:
    value = row["median_abs_log2"]
    return math.inf if value is None else float(value)


def adds_value(cm: dict[str, Any], bp: dict[str, Any]) -> bool:
    """C3 for one D, or any rule's count row (#409): better than bp-only at the same W by a
    margin on one count, worse on neither."""
    d_rho = cm["rho"] - bp["rho"]
    # Positive is better; two infinite distances are no change.
    d_log = 0.0 if _abs_log2_of(cm) == _abs_log2_of(bp) else _abs_log2_of(bp) - _abs_log2_of(cm)
    m_rho, m_log = COUNT_CRITERIA["c3_margin_rho"], COUNT_CRITERIA["c3_margin_abs_log2"]
    eps = 1e-9
    return bool((d_rho >= m_rho - eps or d_log >= m_log - eps)
                and d_rho >= -m_rho - eps and d_log >= -m_log - eps)


def _c3_at(at_w: list[dict[str, Any]]) -> dict[str, Any]:
    bp = next(r for r in at_w if r["d_cm"] == BP_ONLY)
    holding = [r["d_cm"] for r in at_w if r["d_cm"] != BP_ONLY and adds_value(r, bp)]
    return {"w_bp": bp["w_bp"], "holds": bool(holding), "d_cm_that_holds": holding,
            "bp_only_rho": bp["rho"], "bp_only_median_abs_log2": bp["median_abs_log2"]}


def _c3(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_c3_at([r for r in rows if r["w_bp"] == w])
            for w in dict.fromkeys(r["w_bp"] for r in rows)]


def _named(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    meeting = [r for r in rows if r["c1"] and r["c2"]]
    if not meeting:
        return None
    best = min(meeting, key=lambda r: r["share_of_leads_lost_to_density"])
    return {"d_cm": best["d_cm"], "w_bp": best["w_bp"],
            "share_of_leads_lost_to_density": best["share_of_leads_lost_to_density"]}


def leads_per_index_variant(analyses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Locus Leads over r2 0.01 index variants, pooled by the Analyses' max |z| and row bands."""
    table = []
    for key in [(c["d_cm"], c["w_bp"]) for c in analyses[0]["configs"]]:
        row: dict[str, Any] = {"d_cm": key[0], "w_bp": key[1]}
        for name, band in (("z", "z_band"), ("rows", "rows_band")):
            groups: dict[str, dict[str, int]] = {}
            for analysis in analyses:
                entry = next(c for c in analysis["configs"] if (c["d_cm"], c["w_bp"]) == key)
                total = groups.setdefault(analysis["selection"][band],
                                          {"analyses": 0, "locus_leads": 0, "index_variants": 0})
                total["analyses"] += 1
                total["locus_leads"] += entry["by_r2"][PRIMARY_R2]["locus_leads"]
                total["index_variants"] += entry["by_r2"][PRIMARY_R2]["index_variants"]
            row[name] = {
                band_name: total | {"ratio": total["locus_leads"] / total["index_variants"]}
                for band_name, total in groups.items()}
        table.append(row)
    return table


def row_density(analyses: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                         float]:
    """Shared-set rows in full and on HapMap3, their ratio, and its median |log2|."""
    rows_full = np.array([a["density"]["rows_full"] for a in analyses], dtype="float64")
    rows_hm3 = np.array([a["density"]["rows_hm3"] for a in analyses], dtype="float64")
    if (rows_full <= 0).any():
        raise ValueError("an Analysis has no rows in its shared set")
    ratio = rows_hm3 / rows_full
    median_log = _median_or_inf(_abs_log2(ratio))
    if math.isinf(median_log):
        raise ValueError("more than half of the Analyses have no HapMap3 row")
    return rows_full, rows_hm3, ratio, median_log


def evaluate_counts(analyses: list[dict[str, Any]]) -> dict[str, Any]:
    """C1-C3 at every (D, W), the point they name, and the lead / index-variant ratios."""
    rows_full, rows_hm3, rows_ratio, rows_log = row_density(analyses)
    keys = [(c["d_cm"], c["w_bp"]) for c in analyses[0]["configs"]]
    rows = [_config_counts(analyses, key, rows_log) for key in keys]
    cm_only = [r for r in rows if r["d_cm"] != BP_ONLY]
    return {
        "criteria": COUNT_CRITERIA,
        "analyses": len(analyses),
        "rows": {"median_ratio": float(np.median(rows_ratio)), "median_abs_log2": rows_log,
                 "rows_full": int(rows_full.sum()), "rows_hm3": int(rows_hm3.sum())},
        "configs": rows,
        "c3": _c3(rows),
        "named": _named(rows),
        "named_cm": _named(cm_only),
        "leads_per_index_variant": leads_per_index_variant(analyses),
    }
