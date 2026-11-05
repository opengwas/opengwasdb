"""One comparison of Locus Leads with plink2 clumping, for one Analysis.

A run is a function of the Store, the Analysis, the map file, the panel
directory, D, W and the r^2 thresholds. It reads the Analysis's Observed
Associations at one Top-Hit tier (imputed Z is a function of observed Z
through an LD panel, so including it would measure the completion, not the
loci), keeps the rows both methods can see (on the panel, with finite cM), and
counts every other row under the reason it was left out.

Clumping does not depend on D or W, so `clump_shared_set` runs once per r^2
and `compare_leads` can be called for any number of (D, W) pairs against it.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from opengwasdb.query.facade import query_store
from opengwasdb.stats import p_value_from_z

from .clump import (
    CLUMP_KB,
    INDEX_P,
    TIE_STEP_ULPS,
    RankedScores,
    check_clumps_follow_rule_order,
    clump_rows,
    ranked_neg_log10_p,
)
from .compare import DEFINITIONS, Classification, classify
from .genetic_map import GeneticMap, LoadedMap
from .panel import STATUSES, Panel, match_to_panel, plink2
from .rule import locus_leads

ABS_Z_BINS = (0.0, 7.0, 10.0, 20.0, float("inf"))


@dataclass(frozen=True)
class TierRows:
    """One Analysis's rows at one tier, resolved to coordinates; arrays of equal length."""

    chromosome: np.ndarray
    position: np.ndarray
    alid: np.ndarray
    allele_1: np.ndarray
    allele_2: np.ndarray
    abs_z: np.ndarray
    variant_index: np.ndarray

    def take(self, keep: np.ndarray) -> TierRows:
        return TierRows(*(getattr(self, f.name)[keep] for f in fields(self)))


@dataclass(frozen=True)
class SharedSet:
    rows: TierRows
    panel_id: np.ndarray
    cm: np.ndarray


@dataclass(frozen=True)
class ClumpRuns:
    """plink2's clumps at each r^2, with the thresholds it was actually run at."""

    ranked: RankedScores
    index_of: dict[float, np.ndarray]
    p1: float
    p2: float
    kb: int

    def parameters(self) -> dict[str, Any]:
        return {"r2": list(self.index_of), "clump_kb": self.kb, "clump_p1": self.p1,
                "clump_p2": self.p2, "clump_r2": "unphased"}


def read_tier(store: Path, analysis_id: str, threshold: float) -> tuple[TierRows, dict[str, Any]]:
    """The Analysis's Observed Associations at `threshold`, with the Store's identity."""
    query = query_store(store)
    try:
        analysis = _analysis_record(query.analyses_table(), analysis_id)
        every = query.top_hits(analysis_id=analysis_id, threshold=threshold)
        observed = query.top_hits(analysis_id=analysis_id, threshold=threshold,
                                  observed_only=True)
        resolved = list(query.resolve(observed))
        manifest = query.store.manifest
    finally:
        query.close()
    imputed = int(np.sum(every["association_status"] == "imputed")) if every.count else 0
    if every.count == 0 or every.count != observed.count + imputed:
        raise ValueError(f"{analysis_id}: {every.count} rows at {threshold:g}, "
                         f"{observed.count} observed and {imputed} imputed")
    column = {key: np.array([row[key] for row in resolved]) for key in
              ("chromosome", "position", "alid", "effect_allele", "other_allele")}
    rows = TierRows(column["chromosome"], column["position"].astype("int64"), column["alid"],
                    column["effect_allele"], column["other_allele"],
                    np.abs(np.asarray(observed["z"], dtype="float64")),
                    np.asarray(observed["variant_index"], dtype="int64"))
    hits_column = "n_hits_" + f"{threshold:.0e}".replace("e-0", "e").replace("e-", "e")
    identity = {
        "store": {"path": str(store), "store_id": manifest.store_id,
                  "release_id": manifest.release_id, "format_version": manifest.format_version,
                  "primary_layout": str(manifest.primary_layout.value),
                  "completion_state": str(manifest.completion_state.value),
                  "reference_assembly": manifest.reference_assembly},
        "analysis": {key: analysis.get(key) for key in
                     ("analysis_id", "analysis_label", "assigned_ancestry", "sample_size")},
        "query": {"threshold": threshold, "observed_only": True, "rows": int(every.count),
                  "observed": int(observed.count), "imputed": imputed,
                  hits_column: analysis.get(hits_column)},
    }
    return rows, identity


def build_shared_set(rows: TierRows, panel: Panel,
                     genetic_map: GeneticMap) -> tuple[SharedSet, dict[str, Any]]:
    """The rows on the panel with finite cM, and a count of every other row by reason."""
    present = set(rows.chromosome.tolist())
    match = match_to_panel(rows.chromosome, rows.position, rows.allele_1, rows.allele_2,
                           {c: panel.variants(c) for c in panel.chromosomes if c in present})
    cm = genetic_map.cm_at(rows.chromosome, rows.position)
    check_cm(rows.chromosome, rows.position, cm, genetic_map)
    no_cm = np.isnan(cm)
    keep = (match.status == "matched") & ~no_cm
    accounting: dict[str, Any] = {
        "tier_rows": int(keep.size),
        "panel": match.counts(),
        "no_cm": int(no_cm.sum()),
        "no_cm_by_panel_status": {s: int(np.sum(no_cm & (match.status == s))) for s in STATUSES},
        "shared_set": int(keep.sum()),
        "shared_set_share": float(keep.mean()),
    }
    if sum(accounting["panel"].values()) != keep.size:
        raise ValueError("panel statuses do not account for every row")
    return SharedSet(rows.take(keep), match.panel_id[keep], cm[keep]), accounting


def check_cm(chromosome: np.ndarray, position: np.ndarray, cm: np.ndarray,
             genetic_map: GeneticMap) -> None:
    """Refuse cM that is NaN anywhere but off the map, or falls with position on a chromosome."""
    off_map = ~np.isin(chromosome, genetic_map.chromosomes)
    if not np.array_equal(np.isnan(cm), off_map):
        raise ValueError("cM is NaN on a mapped chromosome, or finite on an unmapped one")
    for label in np.unique(chromosome[~off_map]):
        on = np.flatnonzero(chromosome == label)
        by_position = cm[on][np.argsort(position[on], kind="stable")]
        if np.any(np.diff(by_position) < 0):
            raise ValueError(f"chromosome {label}: interpolated cM falls with position")


def clump_shared_set(shared: SharedSet, panel: Panel, r2_values: list[float], *,
                     threshold: float, work_dir: Path, threads: int) -> ClumpRuns:
    """plink2's clumps at each r^2, ranked as the rule ranks; independent of D and W.

    The tier's threshold is both p1 and p2: every row is in the tier, so every
    row may be an index variant or a member.
    """
    rows = shared.rows
    ranked = ranked_neg_log10_p(rows.abs_z, rows.variant_index)
    index_of = {}
    for r2 in r2_values:
        index_of[r2] = clump_rows(panel, rows.chromosome, shared.panel_id, ranked.neg_log10_p,
                                  r2=r2, work_dir=work_dir, kb=CLUMP_KB, p1=threshold,
                                  p2=threshold, threads=threads)
        check_clumps_follow_rule_order(index_of[r2], rows.abs_z, rows.variant_index)
    return ClumpRuns(ranked, index_of, p1=threshold, p2=threshold, kb=CLUMP_KB)


def compare_leads(shared: SharedSet, clumps: ClumpRuns, *, d_cm: float,
                  w_bp: int) -> dict[str, Any]:
    """Locus Leads at (D, W), against each r^2's clumps."""
    rows = shared.rows
    lead_of = locus_leads(rows.chromosome, rows.position, shared.cm, rows.abs_z,
                          rows.variant_index, d_cm=d_cm, w_bp=w_bp)
    return {
        "d_cm": d_cm, "w_bp": w_bp, "locus_leads": int(np.sum(lead_of == np.arange(lead_of.size))),
        "by_r2": {f"{r2:g}": _summary(shared, classify(lead_of, index_of), lead_of, index_of)
                  for r2, index_of in clumps.index_of.items()},
    }


def run(store: Path, analysis_id: str, genetic_map: LoadedMap, panel_dir: Path, *, d_cm: float,
        w_bp: int, r2_values: list[float], work_dir: Path, threads: int = 4,
        threshold: float = INDEX_P) -> dict[str, Any]:
    tier, identity = read_tier(store, analysis_id, threshold)
    panel = Panel(panel_dir)
    shared, accounting = build_shared_set(tier, panel, genetic_map.map)
    clumps = clump_shared_set(shared, panel, r2_values, threshold=threshold, work_dir=work_dir,
                              threads=threads)
    return identity | {
        "genetic_map": genetic_map.identity,
        "panel": _panel_identity(panel),
        "parameters": {"d_cm": d_cm, "w_bp": w_bp} | clumps.parameters(),
        "rows": accounting,
        "ranking": _ranking(shared, clumps.ranked),
        "definitions": DEFINITIONS,
        "comparison": compare_leads(shared, clumps, d_cm=d_cm, w_bp=w_bp),
    }


def _analysis_record(table: dict[int, dict[str, Any]], analysis_id: str) -> dict[str, Any]:
    found = [record for record in table.values() if record.get("analysis_id") == analysis_id]
    if len(found) != 1:
        raise ValueError(f"{analysis_id}: {len(found)} Analyses carry this id")
    return found[0]


def _panel_identity(panel: Panel) -> dict[str, Any]:
    psam = panel.pfile(panel.chromosomes[0]).with_suffix(".psam").read_text().splitlines()
    version = subprocess.run([plink2(), "--version"], check=True, capture_output=True, text=True)
    return {"path": str(panel.directory), "chromosomes": panel.chromosomes,
            "samples": len(psam) - 1, "plink2": version.stdout.strip()}


def _ranking(shared: SharedSet, ranked: RankedScores) -> dict[str, Any]:
    abs_z = shared.rows.abs_z
    raw_p = np.array([p_value_from_z(z) for z in abs_z])
    return {
        "order": "abs_z descending, then Store Variant Index ascending",
        "plink2_input": "-log10 p from z (log10_p_two_sided), --clump-log10 input-only",
        "plink2_tie_order": "panel order, whatever the input order",
        "tie_separation": f"within each run of equal -log10 p, each row {TIE_STEP_ULPS} ulps "
                          "(np.spacing) below the one before it in the rule's order",
        "tied_rows": ranked.tied_rows, "tied_groups": ranked.tied_groups,
        "largest_tied_group": ranked.largest_group, "largest_shift": ranked.largest_shift,
        "rows_abs_z_over_38": int(np.sum(abs_z > 38)),
        "rows_p_zero_in_float64": int(np.sum(raw_p == 0.0)),
        "rows_p_subnormal_in_float64": int(np.sum((raw_p > 0.0) & (raw_p < sys.float_info.min))),
    }


def _summary(shared: SharedSet, result: Classification, lead_of: np.ndarray,
             index_of: np.ndarray) -> dict[str, Any]:
    rows = np.arange(lead_of.size)
    split = np.isin(rows, [row for row, _ in result.spurious_splits])
    lost = np.isin(rows, [row for row, _ in result.lost_signals])
    return result.counts() | {
        "spurious_split_rate": result.spurious_split_rate,
        "lost_signal_rate": result.lost_signal_rate,
        "by_abs_z": _by_abs_z(shared.rows.abs_z, {
            "locus_leads": lead_of == rows, "spurious_splits": split,
            "index_variants": index_of == rows, "lost_signals": lost,
        }),
        "spurious_split_pairs": [_pair(shared, row, lead) for row, lead in result.spurious_splits],
        "lost_signal_pairs": [_pair(shared, row, lead) for row, lead in result.lost_signals],
    }


def _by_abs_z(abs_z: np.ndarray, flags: dict[str, np.ndarray]) -> dict[str, dict[str, int]]:
    """Each flag's count within each abs_z bin; a split or lost signal binned by its own abs_z."""
    bins = np.digitize(abs_z, ABS_Z_BINS[1:-1])
    edges = zip(ABS_Z_BINS[:-1], ABS_Z_BINS[1:], strict=True)
    return {f"[{lo:g},{hi:g})": {name: int(np.sum(flag & (bins == b)))
                                 for name, flag in flags.items()}
            for b, (lo, hi) in enumerate(edges)}


def _pair(shared: SharedSet, row: int, other: int) -> dict[str, Any]:
    rows = shared.rows
    return {"chromosome": str(rows.chromosome[row]), "alid": str(rows.alid[row]),
            "abs_z": float(rows.abs_z[row]), "stronger_alid": str(rows.alid[other]),
            "stronger_abs_z": float(rows.abs_z[other]),
            "distance_bp": int(abs(int(rows.position[row]) - int(rows.position[other]))),
            "distance_cm": float(abs(shared.cm[row] - shared.cm[other]))}
