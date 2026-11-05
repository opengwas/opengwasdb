"""The (D, W) sweep over the selected Analyses, against a bp-window baseline (#403).

Per Analysis: read the tier, build the shared set and clump once per r^2 (all
independent of D and W), then compare Locus Leads at every grid point and at the
bp-only baseline (D = infinity: the same rule with the cM condition removed),
all on the same shared set. Each disagreeing pair gets its r^2 from dosages
exported once. The pooled rates, and whether any (D, W) meets the working
criteria, are computed from the per-Analysis records.

The map, the panel directory and the selection are inputs, so another map or
ancestry reruns this with configuration only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .genetic_map import LoadedMap
from .ld import export_dosages, pair_r2
from .pairs import (
    R2_THRESHOLDS,
    Pair,
    Spike,
    chance_ld,
    check_pairs_follow_threshold,
    summarise,
)
from .panel import Panel
from .run import (
    SharedSet,
    TierRows,
    build_shared_set,
    check_assemblies,
    clump_shared_set,
    compare_leads,
    read_tier,
)

D_GRID = (0.1, 0.25, 0.5, 1.0, 2.5)
W_GRID = (250_000, 1_000_000, 2_500_000, 10_000_000)
R2_VALUES = [0.01, 0.001]
PRIMARY_R2 = "0.01"
BP_ONLY = float("inf")
# The EUR mean map's two single-population spikes found in #402.
EUR_SPIKES = (Spike("7", 131_782_269, "TSI"), Spike("1", 235_056_161, "GBR"))
NULL_MAX_VARIANTS = 3_000
MIN_SHARED_SET_SHARE = 0.70

# Written down before the sweep was run, and unchanged since.
CRITERIA = {
    "spurious_split_rate_max": 0.10,
    "lost_signal_rate_max": 0.10,
    "at_r2": PRIMARY_R2,
    "clearly_better": {
        "than": "the bp-only rule at the same W, on the same Analyses",
        "measure": "disagreement rate = (spurious splits + lost signals) / "
                   "(Locus Leads + index variants), per Analysis and pooled",
        "pooled_rate_at_most": 0.90,  # of the bp-only rate: a cut of at least 10%
        "analyses_lower_at_least": 2 / 3,  # of those where the two rates differ
    },
    "qualifies": "both rates at most 0.10 and clearly better than bp-only, judged on the "
                 "pooled rates and, separately, on the per-Analysis medians",
    "best": "of the qualifying (D, W), the lowest pooled disagreement rate; if none, the one "
            "with the smallest larger-of-the-two pooled rates",
}


@dataclass(frozen=True)
class Config:
    d_cm: float
    w_bp: int

    def label(self) -> dict[str, Any]:
        return {"d_cm": "bp_only" if self.d_cm == BP_ONLY else self.d_cm, "w_bp": self.w_bp}


def grid_configs() -> list[Config]:
    return [Config(d, w) for w in W_GRID for d in (*D_GRID, BP_ONLY)]


def _rates(records: list[dict[str, int]]) -> tuple[np.ndarray, np.ndarray]:
    leads = np.array([r["locus_leads"] for r in records], dtype="float64")
    index = np.array([r["index_variants"] for r in records], dtype="float64")
    split = np.array([r["spurious_splits"] for r in records], dtype="float64")
    lost = np.array([r["lost_signals"] for r in records], dtype="float64")
    return np.c_[split, lost, leads, index], (split + lost) / (leads + index)


def pool(records: list[dict[str, int]]) -> dict[str, Any]:
    """Rates pooled over Analyses (sum of counts over sum of denominators) and their medians."""
    counts, each_disagreement = _rates(records)
    split, lost, leads, index = counts.T
    total = counts.sum(axis=0)
    return {
        "analyses": len(records),
        "locus_leads": int(total[2]), "index_variants": int(total[3]),
        "spurious_splits": int(total[0]), "lost_signals": int(total[1]),
        "spurious_split_rate": float(total[0] / total[2]),
        "lost_signal_rate": float(total[1] / total[3]),
        "disagreement_rate": float((total[0] + total[1]) / (total[2] + total[3])),
        "median_spurious_split_rate": float(np.median(split / leads)),
        "median_lost_signal_rate": float(np.median(lost / index)),
        "median_disagreement_rate": float(np.median(each_disagreement)),
    }


def pool_pairs(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum `summarise` records over Analyses."""
    def total(get: Any) -> Any:
        return sum(get(s) for s in summaries)
    first = summaries[0]
    return {
        "n": total(lambda s: s["n"]), "r2_undefined": total(lambda s: s["r2_undefined"]),
        "straddling_a_spike": total(lambda s: s["straddling_a_spike"]),
        "straddling": {k: total(lambda s, k=k: s["straddling"][k]) for k in first["straddling"]},
        "r2_histogram": {k: total(lambda s, k=k: s["r2_histogram"][k])
                         for k in first["r2_histogram"]},
        "by_distance_band": {band: {k: total(lambda s, b=band, k=k: s["by_distance_band"][b][k])
                                    for k in counts}
                             for band, counts in first["by_distance_band"].items()},
    }


def pool_chance(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Cross-chromosome pairs pooled over Analyses, with the share reaching each r^2."""
    pairs = sum(r["pairs"] for r in records)
    reached = {f"pairs_r2_ge_{t:g}": sum(r[f"pairs_r2_ge_{t:g}"] for r in records)
               for t in R2_THRESHOLDS}
    return {"analyses": len(records), "pairs": pairs, "r2_undefined":
            sum(r["r2_undefined"] for r in records), **reached,
            **{k.replace("pairs_", "share_"): v / pairs for k, v in reached.items()}}


def within_tolerance(pooled: dict[str, Any], prefix: str = "") -> bool:
    return bool(pooled[f"{prefix}spurious_split_rate"] <= CRITERIA["spurious_split_rate_max"]
                and pooled[f"{prefix}lost_signal_rate"] <= CRITERIA["lost_signal_rate_max"])


def clearly_better(cm: list[dict[str, int]], bp_only: list[dict[str, int]]) -> bool:
    """The cM rule's disagreement rate against the bp-only rule's, as `CRITERIA` defines."""
    rule = CRITERIA["clearly_better"]
    cm_rate, bp_rate = pool(cm)["disagreement_rate"], pool(bp_only)["disagreement_rate"]
    each_cm, each_bp = _rates(cm)[1], _rates(bp_only)[1]
    differ = each_cm != each_bp
    if not differ.any() or cm_rate > rule["pooled_rate_at_most"] * bp_rate:
        return False
    return bool(np.mean(each_cm[differ] < each_bp[differ]) >= rule["analyses_lower_at_least"])


def evaluate(analyses: list[dict[str, Any]]) -> dict[str, Any]:
    """Every config's pooled rates at each r^2, and the working criteria at the primary r^2."""
    by_config: dict[tuple[Any, int], dict[str, list[dict[str, Any]]]] = {}
    for analysis in analyses:
        for entry in analysis["configs"]:
            key = (entry["d_cm"], entry["w_bp"])
            for r2, record in entry["by_r2"].items():
                by_config.setdefault(key, {}).setdefault(r2, []).append(record)
    rows = []
    for (d_cm, w_bp), by_r2 in by_config.items():
        row: dict[str, Any] = {"d_cm": d_cm, "w_bp": w_bp,
                               "by_r2": {r2: pool(records) | _pooled_detail(records)
                                         for r2, records in by_r2.items()}}
        if d_cm != "bp_only":
            better = clearly_better(by_r2[PRIMARY_R2], by_config[("bp_only", w_bp)][PRIMARY_R2])
            primary = row["by_r2"][PRIMARY_R2]
            row["clearly_better_than_bp_only"] = better
            row["qualifies_pooled"] = better and within_tolerance(primary)
            row["qualifies_median"] = better and within_tolerance(primary, "median_")
        rows.append(row)
    return {"criteria": CRITERIA, "configs": rows, "best": _best(rows)}


def _pooled_detail(records: list[dict[str, Any]]) -> dict[str, Any]:
    zbands = records[0]["by_abs_z"]
    return {
        "by_abs_z": {band: {k: sum(r["by_abs_z"][band][k] for r in records) for k in counts}
                     for band, counts in zbands.items()},
        "spurious_split_pairs": pool_pairs([r["spurious_split_pairs"] for r in records]),
        "lost_signal_pairs": pool_pairs([r["lost_signal_pairs"] for r in records]),
    }


def _best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cm_rows = [r for r in rows if r["d_cm"] != "bp_only"]
    qualifying = [r for r in cm_rows if r["qualifies_pooled"]]
    primary = [r["by_r2"][PRIMARY_R2] for r in cm_rows]
    if qualifying:
        chosen = min(qualifying, key=lambda r: r["by_r2"][PRIMARY_R2]["disagreement_rate"])
        return {"qualifies": True, "d_cm": chosen["d_cm"], "w_bp": chosen["w_bp"]}
    nearest = min(range(len(cm_rows)), key=lambda i: max(primary[i]["spurious_split_rate"],
                                                         primary[i]["lost_signal_rate"]))
    return {"qualifies": False, "nearest_d_cm": cm_rows[nearest]["d_cm"],
            "nearest_w_bp": cm_rows[nearest]["w_bp"]}


class SharedSetTooSmall(ValueError):
    """Too many of an Analysis's rows fall outside the rows both methods can see."""

    def __init__(self, analysis: dict[str, Any], accounting: dict[str, Any]) -> None:
        share = accounting["shared_set_share"]
        super().__init__(f"{analysis['analysis_id']}: only {share:.1%} of rows are in the "
                         f"shared set (floor {MIN_SHARED_SET_SHARE:.0%})")
        self.record = {"analysis": analysis, "tier_rows": accounting["tier_rows"],
                       "shared_set_share": share, "partition": accounting["partition"]}


def analyse(tier: TierRows, identity: dict[str, Any], genetic_map: LoadedMap, panel: Panel,
            *, work_dir: Path, threads: int, spikes: tuple[Spike, ...],
            configs: list[Config]) -> dict[str, Any]:
    """One Analysis, every config: the compact record the sweep keeps."""
    check_assemblies(identity["store"]["reference_assembly"], genetic_map.identity["assembly"],
                     panel.assembly())
    if identity["query"]["observed"] != tier.chromosome.size:
        raise ValueError(f"{tier.chromosome.size:,} rows, but the query recorded "
                         f"{identity['query']['observed']:,} observed")
    work_dir.mkdir(parents=True, exist_ok=True)
    shared, accounting = build_shared_set(tier, panel, genetic_map.map)
    if accounting["shared_set_share"] < MIN_SHARED_SET_SHARE:
        raise SharedSetTooSmall(identity["analysis"], accounting)
    clumps = clump_shared_set(shared, panel, R2_VALUES, threshold=identity["query"]["threshold"],
                              work_dir=work_dir, threads=threads)
    compared = {c: compare_leads(shared, clumps, d_cm=c.d_cm, w_bp=c.w_bp) for c in configs}
    ld = _PairLD(shared, panel, work_dir, threads)
    ld.load(_needed_rows(shared, clumps.index_of[R2_VALUES[0]], compared))
    return {
        "analysis": identity["analysis"], "store": identity["store"]["store_id"],
        "query": identity["query"], "rows": accounting,
        "chance_ld": ld.chance(clumps.index_of[R2_VALUES[0]], NULL_MAX_VARIANTS),
        "configs": [c.label() | {"by_r2": {
            r2: _record(shared, ld, result, spikes, float(r2))
            for r2, result in compared[c]["by_r2"].items()}} for c in configs],
    }


def _record(shared: SharedSet, ld: _PairLD, result: dict[str, Any], spikes: tuple[Spike, ...],
            threshold: float) -> dict[str, Any]:
    split = ld.pairs(result["spurious_split_pairs"])
    lost = ld.pairs(result["lost_signal_pairs"])
    check_pairs_follow_threshold(split, threshold)
    return {key: result[key] for key in ("rows", "locus_leads", "index_variants", "agreed",
                                         "spurious_splits", "lost_signals",
                                         "spurious_split_rate", "lost_signal_rate",
                                         "by_abs_z")} | {
        "spurious_split_pairs": summarise(split, spikes),
        "lost_signal_pairs": summarise(lost, spikes)}


def _needed_rows(shared: SharedSet, index_of: np.ndarray,
                 compared: dict[Config, dict[str, Any]]) -> list[tuple[str, str]]:
    """The alids whose genotypes are needed: every index variant and every disagreeing pair's."""
    needed = set(shared.rows.alid[np.unique(index_of)].tolist())
    for result in compared.values():
        for summary in result["by_r2"].values():
            for pair in summary["spurious_split_pairs"] + summary["lost_signal_pairs"]:
                needed.update((pair["alid"], pair["stronger_alid"]))
    return [(shared.rows.alid[i], shared.rows.chromosome[i])
            for i in range(shared.rows.alid.size) if shared.rows.alid[i] in needed]


class _PairLD:
    """r^2 between rows of one Analysis's shared set, from dosages exported once."""

    def __init__(self, shared: SharedSet, panel: Panel, work_dir: Path, threads: int) -> None:
        self.shared, self.panel, self.work_dir, self.threads = shared, panel, work_dir, threads
        self.row_of = {a: i for i, a in enumerate(shared.rows.alid.tolist())}
        if len(self.row_of) != shared.rows.alid.size:
            raise ValueError("two shared-set rows carry one ALID")
        self.dosage: dict[int, np.ndarray] = {}

    def load(self, needed: list[tuple[str, str]]) -> None:
        for chromosome in dict.fromkeys(c for _, c in needed):
            rows = [self.row_of[a] for a, c in needed if c == chromosome]
            ids = [str(self.shared.panel_id[r]) for r in rows]
            matrix = export_dosages(self.panel, chromosome, ids,
                                    self.work_dir / "dosage", self.threads)
            self.dosage.update(zip(rows, matrix, strict=True))

    def pairs(self, found: list[dict[str, Any]]) -> list[Pair]:
        rows = self.shared.rows
        made = []
        for item in found:
            a, b = self.row_of[item["alid"]], self.row_of[item["stronger_alid"]]
            made.append(Pair(str(rows.chromosome[a]), (int(rows.position[a]),
                                                       int(rows.position[b])),
                             item["distance_cm"], pair_r2(self.dosage[a], self.dosage[b])))
        return made

    def chance(self, index_of: np.ndarray, max_variants: int) -> dict[str, Any]:
        index = np.unique(index_of)
        matrix = np.array([self.dosage[int(i)] for i in index])
        return chance_ld(matrix, self.shared.rows.chromosome[index], max_variants=max_variants,
                         seed=int(index.size))


def sweep(selection: dict[str, Any], stores_dir: Path, genetic_map: LoadedMap, panel_dir: Path,
          work_dir: Path, *, threads: int, spikes: tuple[Spike, ...] = EUR_SPIKES,
          only: int | None = None) -> dict[str, Any]:
    """Every chosen Analysis, each cached under `work_dir` so an interrupted sweep resumes."""
    panel = Panel(panel_dir)
    analyses, below_floor = [], []
    for chosen in selection["chosen"][:only]:
        cache = work_dir / "results" / f"{chosen['store']}_{chosen['analysis_id']}.json"
        if not cache.exists():
            store = stores_dir / chosen["store"] / "store.opengwasdb"
            tier, identity = read_tier(store, chosen["analysis_id"], 5e-8)
            try:
                record = analyse(tier, identity, genetic_map, panel, work_dir=work_dir / "work" /
                                 f"{chosen['store']}_{chosen['analysis_id']}", threads=threads,
                                 spikes=spikes, configs=grid_configs())
            except SharedSetTooSmall as small:
                below_floor.append(small.record | {"store": chosen["store"]})
                continue
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(record, allow_nan=False))
        analyses.append(json.loads(cache.read_text()))
    return {
        "selection": {"seed": selection["rule"]["seed"], "analyses": len(analyses),
                      "sha256": hashlib.sha256(json.dumps(selection, sort_keys=True).encode()
                                               ).hexdigest()},
        "genetic_map": genetic_map.identity,
        "panel": {"path": str(panel_dir), "assembly": panel.assembly()},
        "grid": {"d_cm": list(D_GRID), "w_bp": list(W_GRID), "bp_only": "D = infinity",
                 "r2": R2_VALUES, "tier": 5e-8, "observed_only": True},
        "spikes": [{"chromosome": s.chromosome, "position": s.position, "population": s.name}
                   for s in spikes],
        "left_out_below_shared_set_floor": {"floor": MIN_SHARED_SET_SHARE,
                                            "analyses": below_floor},
        "chance_ld": pool_chance([a["chance_ld"] for a in analyses]),
        "evaluation": evaluate(analyses),
        "analyses": analyses,
    }

