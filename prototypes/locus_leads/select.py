"""Which EUR Analyses the sweep runs on (#403): a seeded, stratified draw.

The pool is every Analysis the Stores assign EUR, measured by its Observed
Associations at the 5e-8 tier: the row count and the largest |z|. Strata cross
the row count with the largest |z|, so weak and strong Analyses, and the ones
whose p underflows to 0 (|z| >= 38.5), are all represented. The draw is
reproducible from its seed and refuses, rather than drawing fewer, when a
quota cannot be met.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from opengwasdb.query.facade import query_store

ROW_BANDS = ((0, "<1k"), (1_000, "1k-10k"), (10_000, "10k-100k"), (100_000, ">=100k"))
Z_BANDS = ((0.0, "<38.5"), (38.5, "38.5-100"), (100.0, ">=100"))
P_ZERO_Z = 38.5


class QuotaUnmet(ValueError):
    """The pool cannot supply what the rule asks for."""


@dataclass(frozen=True)
class PoolEntry:
    store: str
    analysis_id: str
    observed_rows: int
    max_abs_z: float


@dataclass(frozen=True)
class SelectionRule:
    seed: int
    target: int
    min_z100: int
    min_p0: int
    max_rows: int


@dataclass(frozen=True)
class Selection:
    chosen: list[PoolEntry]
    excluded: list[PoolEntry]
    pool_by_stratum: dict[str, int]
    empty: list[PoolEntry]  # Analyses with no observed row at the tier: nothing to compare


def stratum(observed_rows: int, max_abs_z: float) -> tuple[str, str]:
    rows = [name for low, name in ROW_BANDS if observed_rows >= low][-1]
    z = [name for low, name in Z_BANDS if max_abs_z >= low][-1]
    return rows, z


def draw(pool: list[PoolEntry], rule: SelectionRule) -> Selection:
    """The quotas first (strongest |z| first), then round-robin over the strata."""
    ordered = sorted(pool, key=lambda e: (e.store, e.analysis_id))
    excluded = [e for e in ordered if e.observed_rows > rule.max_rows]
    eligible = [e for e in ordered if 0 < e.observed_rows <= rule.max_rows]
    counts: dict[str, int] = {}
    for e in eligible:
        key = " x ".join(stratum(e.observed_rows, e.max_abs_z))
        counts[key] = counts.get(key, 0) + 1
    rng = random.Random(rule.seed)
    chosen: list[PoolEntry] = []
    _take(chosen, eligible, rng, rule.min_z100, lambda e: e.max_abs_z >= 100.0, "|z| >= 100")
    _take(chosen, eligible, rng, rule.min_p0, lambda e: e.max_abs_z >= P_ZERO_Z,
          f"|z| >= {P_ZERO_Z:g} (p = 0)")
    if len(eligible) < rule.target:
        raise QuotaUnmet(f"the target of {rule.target} needs {rule.target} eligible Analyses; "
                         f"the pool has {len(eligible)}")
    _fill(chosen, eligible, rng, rule.target)
    return Selection(sorted(chosen, key=lambda e: (e.store, e.analysis_id)), excluded,
                     dict(sorted(counts.items())), [e for e in ordered if e.observed_rows == 0])


def _take(chosen: list[PoolEntry], eligible: list[PoolEntry], rng: random.Random, quota: int,
          qualifies: Any, label: str) -> None:
    """Draw until `quota` chosen Analyses qualify, from those that do and are not yet chosen."""
    have = sum(qualifies(e) for e in chosen)
    candidates = [e for e in eligible if qualifies(e) and e not in chosen]
    if have + len(candidates) < quota:
        raise QuotaUnmet(f"the quota of {quota} at {label} cannot be met: the pool has "
                         f"{have + len(candidates)}")
    chosen.extend(rng.sample(candidates, quota - have) if quota > have else [])


def _fill(chosen: list[PoolEntry], eligible: list[PoolEntry], rng: random.Random,
          target: int) -> None:
    queues: dict[tuple[str, str], list[PoolEntry]] = {}
    for e in eligible:
        if e not in chosen:
            queues.setdefault(stratum(e.observed_rows, e.max_abs_z), []).append(e)
    for queue in queues.values():
        rng.shuffle(queue)
    order = sorted(queues)
    while len(chosen) < target:
        for key in order:
            if queues[key] and len(chosen) < target:
                chosen.append(queues[key].pop())


def survey_store(store: Path, threshold: float = 5e-8) -> list[PoolEntry]:
    """Each EUR Analysis of a Store: its observed row count and largest |z| at the tier."""
    query = query_store(store)
    try:
        analyses = [(index, row["analysis_id"]) for index, row in query.analyses_table().items()
                    if row.get("assigned_ancestry") == "EUR"]
        entries = []
        for _, analysis_id in analyses:
            hits = query.top_hits(analysis_id=analysis_id, threshold=threshold,
                                  observed_only=True)
            z = float(np.max(np.abs(np.asarray(hits["z"], dtype="float64")))) if hits.count else 0.0
            entries.append(PoolEntry(store.parent.name, analysis_id, int(hits.count), z))
    finally:
        query.close()
    return entries


GENOME_WIDE = "gwas"


def build_selection(stores_dir: Path, rule: SelectionRule) -> dict[str, Any]:
    """The selection record: pool, strata, seed, rule, chosen Analyses and every exclusion.

    A Store enters the pool when its association coverage is genome-wide. The
    rest are listed with why they were left out, and so is a genome-wide Store
    that will not open.
    """
    pool: list[PoolEntry] = []
    left_out: list[dict[str, str]] = []
    for path in sorted(stores_dir.glob("OGS-*/store.opengwasdb")):
        store_id = path.parent.name
        manifest = json.loads((path / "manifest.json").read_text())
        coverage = manifest["association_coverage"]
        if coverage != GENOME_WIDE:
            left_out.append({"store": store_id, "reason": f"association coverage is {coverage}"})
            continue
        try:
            found = survey_store(path)
        except Exception as error:  # a Store that will not open is recorded, not skipped quietly
            left_out.append({"store": store_id, "reason": f"{type(error).__name__}: {error}"})
            continue
        pool += found
        if not found:
            left_out.append({"store": store_id, "reason": "no Analysis assigned EUR"})
    return selection_record(rule, left_out, pool, draw(pool, rule))


def selection_record(rule: SelectionRule, left_out: list[dict[str, str]], pool: list[PoolEntry],
                     result: Selection) -> dict[str, Any]:
    """Everything the draw saw and did, each exclusion by Store and Analysis with its reason."""
    return {
        "rule": {"seed": rule.seed, "target": rule.target, "min_abs_z_100": rule.min_z100,
                 "min_p_zero": rule.min_p0, "max_observed_rows": rule.max_rows,
                 "strata": "observed 5e-8 rows [<1k, 1k-10k, 10k-100k, >=100k] x max |z| "
                           "[<38.5, 38.5-100, >=100]",
                 "draw": "quota Analyses first (|z| >= 100, then |z| >= 38.5), then "
                         "round-robin over the strata in sorted order; seeded random.Random",
                 "pool": "assigned_ancestry == EUR in genome-wide Stores; observed rows at 5e-8"},
        "stores_left_out": left_out,
        "pool_size": len(pool),
        "pool_by_stratum": result.pool_by_stratum,
        "n_analyses_without_observed_rows": len(result.empty),
        "analyses_without_observed_rows": [
            {"store": e.store, "analysis_id": e.analysis_id, "observed_rows": e.observed_rows,
             "reason": "no Observed Association at 5e-8, so nothing to compare"}
            for e in result.empty],
        "excluded_over_max_rows": [_entry(e) for e in result.excluded],
        "chosen": [_entry(e) for e in result.chosen],
    }


def _entry(entry: PoolEntry) -> dict[str, Any]:
    return {"store": entry.store, "analysis_id": entry.analysis_id,
            "observed_rows": entry.observed_rows, "max_abs_z": entry.max_abs_z,
            "stratum": " x ".join(stratum(entry.observed_rows, entry.max_abs_z))}
