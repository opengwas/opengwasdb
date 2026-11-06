"""What chance LD the panel itself gives, from the sweeps' shared sets (#405).

The sweep's `chance_ld` is measured among each Analysis's clump index variants,
so it depends on how many hits an ancestry has: EAS's rests on 92 pairs, which
is too few to describe a panel. Here the variant pool is the union of the swept
Analyses' shared sets — the panel as the sweeps actually used it — and both the
variants and the cross-chromosome pairs are drawn with a fixed seed. The share
of unlinked pairs reaching an r^2 is then a property of the panel's sample size,
not of any Analysis's hit count.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .genetic_map import LoadedMap
from .ld import export_dosages, pair_r2
from .panel import Panel
from .run import build_shared_set, read_tier

PAIR_SAMPLE = 200_000
MAX_VARIANTS = 3_000
SEED = 405
THRESHOLDS = (0.001, 0.01, 0.05, 0.1)
TIER = 5e-8
WHAT = ("chance LD at each panel's own sample size: the share of unlinked "
        "(cross-chromosome) variant pairs reaching an r^2, over the panel variants the "
        "swept Analyses share, both drawn with a fixed seed")


@dataclass(frozen=True)
class Pooled:
    """The panel variants the swept Analyses share, one entry per variant."""

    chromosome: np.ndarray
    panel_id: np.ndarray
    analyses: int

    def __len__(self) -> int:
        return int(self.chromosome.size)


def merge_chance(existing: dict[str, Any] | None, ancestry: str, entry: dict[str, Any],
                 seed: int = SEED) -> dict[str, Any]:
    """The record with one ancestry's entry added or replaced, written sorted by its caller.

    One command per ancestry fills one file, so the file is a function of the
    four entries and not of the order they were written in.
    """
    record = existing or {"what": WHAT, "seed": seed, "ancestries": {}}
    if record["seed"] != seed:
        raise ValueError(f"the record was drawn with seed {record['seed']}, not {seed}")
    return record | {"ancestries": {**record["ancestries"], ancestry: entry}}


def pool_shared_sets(swept: list[dict[str, Any]], stores_dir: Path, panel: Panel,
                     genetic_map: LoadedMap, threshold: float = TIER) -> Pooled:
    """Every panel variant any swept Analysis has in its shared set, once."""
    seen: dict[str, tuple[str, str]] = {}
    for entry in swept:
        tier, _ = read_tier(stores_dir / entry["store"] / "store.opengwasdb",
                            entry["analysis"]["analysis_id"], threshold)
        shared, _ = build_shared_set(tier, panel, genetic_map.map)
        for chromosome, panel_id in zip(shared.rows.chromosome.tolist(),
                                        shared.panel_id.tolist(), strict=True):
            seen.setdefault(str(panel_id), (str(chromosome), str(panel_id)))
    if not seen:
        raise ValueError(f"{len(swept)} swept Analyses share no panel variant")
    found = list(seen.values())
    return Pooled(np.array([chromosome for chromosome, _ in found], dtype=object),
                  np.array([panel_id for _, panel_id in found], dtype=object), len(swept))


def sample_variants(pooled: Pooled, max_variants: int, seed: int) -> Pooled:
    """At most `max_variants` of the pool, drawn by a fixed seed."""
    if len(pooled) <= max_variants:
        return pooled
    keep = np.sort(np.random.default_rng(seed).choice(len(pooled), max_variants, replace=False))
    return Pooled(pooled.chromosome[keep], pooled.panel_id[keep], pooled.analyses)


def sample_pairs(chromosome: np.ndarray, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """`count` index pairs from different chromosomes, drawn by a fixed seed."""
    labels = np.asarray(chromosome)
    if np.unique(labels).size < 2:
        raise ValueError("one chromosome only, so no pair is cross-chromosome")
    rng = np.random.default_rng(seed)
    first: list[np.ndarray] = []
    second: list[np.ndarray] = []
    placed = 0
    while placed < count:
        want = count - placed
        draw = rng.integers(0, labels.size, size=(max(2 * want, 64), 2))
        keep = labels[draw[:, 0]] != labels[draw[:, 1]]
        first.append(draw[keep, 0][:want])
        second.append(draw[keep, 1][:want])
        placed += int(first[-1].size)
    return np.concatenate(first), np.concatenate(second)


def dosages(pooled: Pooled, panel: Panel, work_dir: Path, threads: int) -> dict[str, np.ndarray]:
    """Each pooled variant's allele counts, by panel ID, one export per chromosome."""
    counted: dict[str, np.ndarray] = {}
    for chromosome in dict.fromkeys(pooled.chromosome.tolist()):
        ids = pooled.panel_id[pooled.chromosome == chromosome].tolist()
        matrix = export_dosages(panel, str(chromosome), [str(i) for i in ids],
                                work_dir / "dosage", threads)
        counted.update(zip(ids, matrix, strict=True))
    return counted


def summarise(values: np.ndarray) -> dict[str, Any]:
    """The pairs drawn, the ones with a defined r^2, and the share reaching each threshold."""
    called = values[~np.isnan(values)]
    if called.size == 0:
        raise ValueError(f"no pair of the {values.size} drawn has a defined r^2, so the chance "
                         "share is not defined")
    return {"pairs_sampled": int(values.size), "r2_undefined": int(values.size - called.size),
            "pairs_evaluable": int(called.size),
            **{f"pairs_r2_ge_{t:g}": int(np.sum(called >= t)) for t in THRESHOLDS},
            **{f"share_r2_ge_{t:g}": float(np.mean(called >= t)) for t in THRESHOLDS}}


def sample_count(panel: Panel) -> int:
    """The panel's samples: the psam's header plus one line per sample."""
    path = panel.pfile(panel.chromosomes[0]).with_suffix(".psam")
    return max(sum(1 for _ in path.open()) - 1, 0)


def panel_chance(sweep_record: dict[str, Any], stores_dir: Path, genetic_map: LoadedMap,
                 panel_dir: Path, work_dir: Path, *, threads: int = 1, seed: int = SEED,
                 pair_sample: int = PAIR_SAMPLE,
                 max_variants: int = MAX_VARIANTS) -> dict[str, Any]:
    """The panel's chance LD, from the variants its swept Analyses share."""
    panel = Panel(panel_dir)
    pooled = pool_shared_sets(sweep_record["analyses"], stores_dir, panel, genetic_map)
    used = sample_variants(pooled, max_variants, seed)
    first, second = sample_pairs(used.chromosome, pair_sample, seed)
    counted = dosages(used, panel, work_dir, threads)
    values = np.array([pair_r2(counted[str(used.panel_id[a])], counted[str(used.panel_id[b])])
                       for a, b in zip(first, second, strict=True)])
    return {
        "panel": {"path": str(panel_dir), "assembly": panel.assembly(),
                  "samples": sample_count(panel), "chromosomes": len(panel.chromosomes)},
        "seed": seed, "pair_sample": pair_sample, "max_variants": max_variants,
        "tier": TIER,
        "pool": {"variants": len(pooled), "analyses": pooled.analyses,
                 "variants_on": {label: int(np.sum(pooled.chromosome == label))
                                 for label in panel.chromosomes
                                 if np.any(pooled.chromosome == label)}},
        "variants": {"used": len(used),
                     "on": {label: int(np.sum(used.chromosome == label))
                            for label in panel.chromosomes
                            if np.any(used.chromosome == label)}},
        **summarise(values),
    }
