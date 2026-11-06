"""Which Analyses the sweep runs on (#403): a seeded, stratified draw.

The pool is every Analysis a Store assigns one ancestry, measured by its
Observed Associations at the 5e-8 tier: the row count and the largest |z|.
Strata cross the row count with the largest |z|, so weak and strong Analyses,
and the ones whose p underflows to 0 (|z| >= 38.5), are all represented. The
draw is reproducible from its seed. The |z| quotas and the target floor at what
the pool holds, and the record keeps what each stratum holds, so a small pool
is measured rather than refused (#405).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from opengwasdb.query.facade import query_store

ROW_BANDS = ((0, "<1k"), (1_000, "1k-10k"), (10_000, "10k-100k"), (100_000, ">=100k"))
Z_BANDS = ((0.0, "<38.5"), (38.5, "38.5-100"), (100.0, ">=100"))
P_ZERO_Z = 38.5
# The ancestry the sweep drew first; its record is #403's and keeps its shape.
DEFAULT_ANCESTRY = "EUR"


class QuotaUnmet(ValueError):
    """The pool cannot supply what the rule asks for."""


@dataclass(frozen=True)
class PoolEntry:
    store: str
    analysis_id: str
    observed_rows: int
    max_abs_z: float
    ancestry: str = DEFAULT_ANCESTRY


@dataclass(frozen=True)
class SelectionRule:
    seed: int
    target: int
    min_z100: int
    min_p0: int
    max_rows: int
    ancestry: str = DEFAULT_ANCESTRY


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
    ordered = sorted((e for e in pool if e.ancestry == rule.ancestry),
                     key=lambda e: (e.store, e.analysis_id))
    excluded = [e for e in ordered if e.observed_rows > rule.max_rows]
    eligible = [e for e in ordered if 0 < e.observed_rows <= rule.max_rows]
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
                     _stratum_counts(eligible), [e for e in ordered if e.observed_rows == 0])


def with_floors(pool: list[PoolEntry], rule: SelectionRule) -> tuple[SelectionRule, dict[str, Any]]:
    """The rule with the target and the |z| quotas no larger than the pool can supply.

    A quota is a gap to fill, not a fact about the pool: an ancestry with two
    Analyses at |z| >= 100 is swept, and its record says so, rather than the
    draw refusing. Both the rule asked for and the floor are returned.
    """
    eligible = [e for e in pool if e.ancestry == rule.ancestry
                and 0 < e.observed_rows <= rule.max_rows]
    in_pool = {"target": len(eligible),
               "min_abs_z_100": sum(e.max_abs_z >= 100.0 for e in eligible),
               "min_p_zero": sum(e.max_abs_z >= P_ZERO_Z for e in eligible)}
    asked = {"target": rule.target, "min_abs_z_100": rule.min_z100,
             "min_p_zero": rule.min_p0}
    floors = {key: {"asked": asked[key], "in_pool": in_pool[key],
                    "floor": min(asked[key], in_pool[key])} for key in asked}
    floored = replace(rule, target=floors["target"]["floor"],
                      min_z100=floors["min_abs_z_100"]["floor"],
                      min_p0=floors["min_p_zero"]["floor"])
    return floored, floors


def _stratum_counts(eligible: list[PoolEntry]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in eligible:
        key = " x ".join(stratum(e.observed_rows, e.max_abs_z))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


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


def survey_store(store: Path, ancestry: str = DEFAULT_ANCESTRY,
                 threshold: float = 5e-8) -> list[PoolEntry]:
    """Each Analysis of a Store assigned `ancestry`: its observed rows and largest |z|."""
    query = query_store(store)
    try:
        analyses = [(index, row["analysis_id"]) for index, row in query.analyses_table().items()
                    if row.get("assigned_ancestry") == ancestry]
        entries = []
        for _, analysis_id in analyses:
            hits = query.top_hits(analysis_id=analysis_id, threshold=threshold,
                                  observed_only=True)
            z = float(np.max(np.abs(np.asarray(hits["z"], dtype="float64")))) if hits.count else 0.0
            entries.append(PoolEntry(store.parent.name, analysis_id, int(hits.count), z, ancestry))
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
            found = survey_store(path, rule.ancestry)
        except Exception as error:  # a Store that will not open is recorded, not skipped quietly
            left_out.append({"store": store_id, "reason": f"{type(error).__name__}: {error}"})
            continue
        pool += found
        if not found:
            left_out.append({"store": store_id,
                             "reason": f"no Analysis assigned {rule.ancestry}"})
    floored, floors = with_floors(pool, rule)
    return selection_record(floored, left_out, pool, draw(pool, floored), floors=floors)


def selection_record(rule: SelectionRule, left_out: list[dict[str, str]], pool: list[PoolEntry],
                     result: Selection, *, floors: dict[str, Any] | None = None) -> dict[str, Any]:
    """Everything the draw saw and did, each exclusion by Store and Analysis with its reason.

    A draw for an ancestry other than EUR also records that ancestry and the
    floors its pool set, because the EUR record is #403's committed output and
    keeps its shape.
    """
    record = {
        "rule": {"seed": rule.seed, "target": rule.target, "min_abs_z_100": rule.min_z100,
                 "min_p_zero": rule.min_p0, "max_observed_rows": rule.max_rows,
                 "strata": "observed 5e-8 rows [<1k, 1k-10k, 10k-100k, >=100k] x max |z| "
                           "[<38.5, 38.5-100, >=100]",
                 "draw": "quota Analyses first (|z| >= 100, then |z| >= 38.5), then "
                         "round-robin over the strata in sorted order; seeded random.Random",
                 "pool": f"assigned_ancestry == {rule.ancestry} in genome-wide Stores; "
                         "observed rows at 5e-8"},
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
    if rule.ancestry != DEFAULT_ANCESTRY:
        return record | {"ancestry": rule.ancestry, "quota_floors": floors}
    return record


def _entry(entry: PoolEntry) -> dict[str, Any]:
    return {"store": entry.store, "analysis_id": entry.analysis_id,
            "observed_rows": entry.observed_rows, "max_abs_z": entry.max_abs_z,
            "stratum": " x ".join(stratum(entry.observed_rows, entry.max_abs_z))}
