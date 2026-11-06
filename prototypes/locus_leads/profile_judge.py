"""The prominence and LD-block rules judged by #403's count criteria (#409).

C1 and C2 are #403's, with its wording. A rule here has no W, so C3 compares
it with the bp-only window at each W of #403's grid; W 10 Mb is the point #403
named. Per-lead rates by the lead's |z| band are diagnostics, set beside #403's
for its bp-only 10 Mb window and its D 0.5 cM / W 1 Mb point. The readings of
"beats distance pruning" in `CRITERIA` were written before the sweep was run.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .counts import COUNT_CRITERIA, PRIMARY_R2, adds_value, count_row, row_density
from .sweep import W_GRID, pool

STRONG_BAND = "[20,inf)"
STRONG_CUT = 0.90
NAMED_ANALYSIS = "GCST90162563"
# The distance rules the new rules are set beside, as #403's records label them.
SET_BESIDE = (("bp_only", 10_000_000), (0.5, 1_000_000))

CRITERIA: dict[str, Any] = {
    "C1": COUNT_CRITERIA["C1"],
    "C2": COUNT_CRITERIA["C2"],
    "C3": COUNT_CRITERIA["C3"],
    "c2_profile": "for C2 the HapMap3 restriction applies to the 5e-4 profile as well as to the "
                  "5e-8 candidates: both are cut to HapMap3 ALIDs before a rule runs; the raw-row "
                  "ratio C2 compares with is #403's, from the same shared set",
    "c3_reading": "a rule has no W, so at each W of #403's grid it is compared with that W's "
                  "bp-only window: C3 holds when it improves on bp-only by at least one margin "
                  "and is worse on neither by more than its margin (#403's test); W 10 Mb is "
                  "#403's named point",
    "rho_min": COUNT_CRITERIA["rho_min"],
    "ratio_range": COUNT_CRITERIA["ratio_range"],
    "c3_margin_rho": COUNT_CRITERIA["c3_margin_rho"],
    "c3_margin_abs_log2": COUNT_CRITERIA["c3_margin_abs_log2"],
    "index_variants": COUNT_CRITERIA["index_variants"],
    "beats": {
        "overall": "rank: rho at least c3_margin_rho above bp-only's at that W, and median "
                   "|log2| not worse by more than c3_margin_abs_log2",
        "hapmap3": "density: median |log2| at least c3_margin_abs_log2 below bp-only's at that "
                   "W, and rho not worse by more than c3_margin_rho",
        "strong_peaks": f"in the lead's |z| band {STRONG_BAND}, the pooled disagreement rate "
                        "(spurious splits + lost signals) / (Locus Leads + index variants) is at "
                        f"most {STRONG_CUT:.2f} of the distance rule's, for each rule set beside",
    },
    "set_beside": [{"d_cm": d, "w_bp": w} for d, w in SET_BESIDE],
}


def _log2_or_inf(row: dict[str, Any]) -> float:
    value = row["median_abs_log2"]
    return math.inf if value is None else float(value)


def margins(row: dict[str, Any], bp_only: dict[str, Any]) -> dict[str, Any]:
    """The rule against one W's bp-only window: both margins (positive is better), C3, and the
    two readings of "beats" it splits into."""
    d_rho = row["rho"] - bp_only["rho"]
    same = _log2_or_inf(row) == _log2_or_inf(bp_only)
    d_log = 0.0 if same else _log2_or_inf(bp_only) - _log2_or_inf(row)
    m_rho, m_log, eps = CRITERIA["c3_margin_rho"], CRITERIA["c3_margin_abs_log2"], 1e-9
    return {"w_bp": bp_only["w_bp"], "delta_rho": d_rho,
            "delta_abs_log2": None if math.isinf(d_log) else d_log,
            "c3": adds_value(row, bp_only),
            "beats_overall": bool(d_rho >= m_rho - eps and d_log >= -m_log - eps),
            "beats_hapmap3": bool(d_log >= m_log - eps and d_rho >= -m_rho - eps)}


def disagreement(band: dict[str, int]) -> float | None:
    denominator = band["locus_leads"] + band["index_variants"]
    if denominator == 0:
        return None
    return (band["spurious_splits"] + band["lost_signals"]) / denominator


def _summed_bands(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    first = records[0]["by_abs_z"]
    return {band: {k: sum(r["by_abs_z"][band][k] for r in records) for k in counts}
            for band, counts in first.items()}


def _per_z_band(analyses: list[dict[str, Any]], leads: list[int],
                index: list[int]) -> dict[str, dict[str, Any]]:
    """Locus Leads over index variants, pooled by the Analysis's largest |z|."""
    groups: dict[str, dict[str, Any]] = {}
    for analysis, n_leads, n_index in zip(analyses, leads, index, strict=True):
        total = groups.setdefault(analysis["selection"]["z_band"],
                                  {"analyses": 0, "locus_leads": 0, "index_variants": 0})
        total["analyses"] += 1
        total["locus_leads"] += n_leads
        total["index_variants"] += n_index
    return {b: t | {"ratio": t["locus_leads"] / t["index_variants"]} for b, t in groups.items()}


def _pooled_extra(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Valleys or rows outside blocks, summed over Analyses, at both densities."""
    out: dict[str, Any] = {}
    for key in ("valleys", "valleys_hm3", "outside", "outside_hm3"):
        if key in entries[0]:
            out[key] = {k: sum(e[key][k] for e in entries) for k in entries[0][key]}
    return out


def _find(rows: list[dict[str, Any]], d_cm: Any, w_bp: int) -> dict[str, Any]:
    """The record of one (D, W) among #403's."""
    return next(r for r in rows if (r["d_cm"], r["w_bp"]) == (d_cm, w_bp))


def _without(record: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {k: v for k, v in record.items() if k not in keys}


def baseline_rows(committed: dict[str, Any]) -> list[dict[str, Any]]:
    """#403's counts and pooled per-lead rates for the distance rules set beside, read from its
    committed sweep."""
    out = []
    for d_cm, w_bp in SET_BESIDE:
        counts = _find(committed["counts"]["configs"], d_cm, w_bp)
        pooled = _find(committed["evaluation"]["configs"], d_cm, w_bp)["by_r2"][PRIMARY_R2]
        ratios = _find(committed["counts"]["leads_per_index_variant"], d_cm, w_bp)["z"]
        out.append({"d_cm": d_cm, "w_bp": w_bp, **_without(counts, "d_cm", "w_bp"),
                    "per_lead": _without(pooled, "spurious_split_pairs", "lost_signal_pairs"),
                    "leads_per_index_variant": ratios})
    return out


def _beats_strong_peaks(per_lead: dict[str, Any],
                        beside: list[dict[str, Any]]) -> dict[str, bool]:
    """`CRITERIA["beats"]["strong_peaks"]`, against each distance rule set beside."""
    strong = disagreement(per_lead["by_abs_z"][STRONG_BAND])
    out = {}
    for b in beside:
        theirs = disagreement(b["per_lead"]["by_abs_z"][STRONG_BAND])
        out[f"{b['d_cm']}/{b['w_bp']}"] = bool(
            strong is not None and theirs is not None and strong <= STRONG_CUT * theirs)
    return out


def _entries(analyses: list[dict[str, Any]], position: int,
             label: dict[str, Any]) -> list[dict[str, Any]]:
    """Each Analysis's record of one rule, checked to be that rule's."""
    entries = [a["rules"][position] for a in analyses]
    if {e["label"] for e in entries} != {label["label"]}:
        raise ValueError(f"the Analyses' records do not hold {label['label']} at {position}")
    return entries


def _rule_row(position: int, label: dict[str, Any], analyses: list[dict[str, Any]],
              rows_log: float, bp_only: dict[int, dict[str, Any]],
              beside: list[dict[str, Any]]) -> dict[str, Any]:
    entries = _entries(analyses, position, label)
    counts = count_row(entries, rows_log, label["label"])
    records = [e["by_r2"][PRIMARY_R2] for e in entries]
    per_lead = pool(records) | {"by_abs_z": _summed_bands(records)}
    return label | counts | {
        "c3": [margins(counts, bp_only[w]) for w in W_GRID],
        "beats_strong_peaks": _beats_strong_peaks(per_lead, beside),
        "per_lead": per_lead,
        "leads_per_index_variant": _per_z_band(analyses, [r["locus_leads"] for r in records],
                                               [r["index_variants"] for r in records]),
        **_pooled_extra(entries),
    }


def _named(analyses: list[dict[str, Any]], committed: dict[str, Any],
           labels: list[dict[str, Any]]) -> dict[str, Any] | None:
    """One Analysis's leads under every rule, beside #403's distance rules and its clumps."""
    found = {a["analysis"]["analysis_id"]: a for a in analyses}.get(NAMED_ANALYSIS)
    if found is None:
        return None
    theirs = {a["analysis"]["analysis_id"]: a for a in committed["analyses"]}[NAMED_ANALYSIS]
    return {"analysis_id": NAMED_ANALYSIS, "max_abs_z": found["selection"]["max_abs_z"],
            "index_variants": found["index_variants"],
            "set_beside": [{"d_cm": d, "w_bp": w, "locus_leads": _find(
                theirs["configs"], d, w)["by_r2"][PRIMARY_R2]["locus_leads"]}
                for d, w in SET_BESIDE],
            "rules": [{"label": label["label"]} | {
                k: rule["by_r2"][PRIMARY_R2][k]
                for k in ("locus_leads", "spurious_splits", "lost_signals")}
                for label, rule in zip(labels, found["rules"], strict=True)]}


def judge(analyses: list[dict[str, Any]], committed: dict[str, Any],
          rules: list[Any]) -> dict[str, Any]:
    """Every rule's C1-C3 and per-lead rates, beside #403's distance rules."""
    rows_full, rows_hm3, _, rows_log = row_density(analyses)
    bp_only = {c["w_bp"]: c for c in committed["counts"]["configs"] if c["d_cm"] == "bp_only"}
    beside = baseline_rows(committed)
    labels = [r.label() for r in rules]
    return {
        "criteria": CRITERIA, "analyses": len(analyses),
        "rows": {"rows_full": int(np.sum(rows_full)), "rows_hm3": int(np.sum(rows_hm3)),
                 "median_abs_log2": rows_log},
        "rules": [_rule_row(i, label, analyses, rows_log, bp_only, beside)
                  for i, label in enumerate(labels)],
        "bp_only": [bp_only[w] for w in W_GRID],
        "set_beside": beside,
        "named_analysis": _named(analyses, committed, labels),
    }
