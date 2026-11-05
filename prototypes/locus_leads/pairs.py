"""Each disagreeing pair's r^2, distance and place, summarised for the sweep (#403).

A spurious split pairs a Locus Lead with the plink2 index variant whose clump
holds it; a lost signal pairs an index variant with the Locus Lead it merged
into. Both carry the pair's unphased r^2, its distance in bp and cM, and
whether it straddles a spike in the genetic map.

Chance LD at the panel's sample size is estimated from index variants on
different chromosomes, which are unlinked: the share of such pairs with
r^2 >= a threshold is the share a pair at any distance reaches by chance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .ld import r2_matrix

# (upper bound in bp inclusive, name): the bands a disagreement's distance falls in.
DISTANCE_BANDS = ((1_000_000, "<=1Mb"), (2_500_000, "1-2.5Mb"), (10_000_000, "2.5-10Mb"),
                  (float("inf"), ">10Mb"))
R2_THRESHOLDS = (0.01, 0.001)
R2_EDGES = (0.0, 0.001, 0.01, 0.02, 0.05, 0.1, 0.5, float("inf"))
R2_TOLERANCE = 1e-4  # plink2 and numpy agree to this on a pair's r^2


@dataclass(frozen=True)
class Spike:
    """A position where one population's map jumps far beyond the mean's."""

    chromosome: str
    position: int
    name: str


@dataclass(frozen=True)
class Pair:
    chromosome: str
    position: tuple[int, int]
    distance_cm: float
    r2: float

    @property
    def distance_bp(self) -> int:
        return abs(self.position[0] - self.position[1])


def distance_band(distance_bp: int) -> str:
    return next(name for bound, name in DISTANCE_BANDS if distance_bp <= bound)


def straddles(pair: Pair, spike: Spike) -> bool:
    low, high = sorted(pair.position)
    return pair.chromosome == spike.chromosome and low <= spike.position <= high


def summarise(pairs: list[Pair], spikes: tuple[Spike, ...]) -> dict[str, Any]:
    """Counts of a kind of disagreement by distance band, r^2, and spike straddled."""
    defined = [p for p in pairs if not np.isnan(p.r2)]
    r2 = np.array([p.r2 for p in defined])
    histogram = np.histogram(r2, bins=[*R2_EDGES[:-1], np.inf])[0] if r2.size else \
        np.zeros(len(R2_EDGES) - 1, dtype=int)
    bands = {name: [p for p in pairs if distance_band(p.distance_bp) == name]
             for _, name in DISTANCE_BANDS}
    return {
        "n": len(pairs),
        "r2_undefined": len(pairs) - len(defined),
        "r2_histogram": {f"[{lo:g},{hi:g})": int(count) for (lo, hi), count in zip(
            zip(R2_EDGES[:-1], R2_EDGES[1:], strict=True), histogram, strict=True)},
        "by_distance_band": {name: {"n": len(members)} | {
            f"r2_ge_{t:g}": sum(p.r2 >= t for p in members if not np.isnan(p.r2))
            for t in R2_THRESHOLDS} for name, members in bands.items()},
        "straddling_a_spike": sum(any(straddles(p, s) for s in spikes) for p in pairs),
        "straddling": {f"{s.name} chr{s.chromosome}:{s.position:,}":
                       sum(straddles(p, s) for p in pairs) for s in spikes},
    }


def check_pairs_follow_threshold(pairs: list[Pair], threshold: float) -> None:
    """A clump member's r^2 with its index variant is at least the clump's threshold."""
    short = [p for p in pairs if p.r2 < threshold - R2_TOLERANCE]
    if short:
        raise ValueError(f"{len(short)} clump members have r^2 below {threshold:g} with their "
                         f"index variant, e.g. {short[0]}; plink2's LD and ours disagree")


def chance_ld(dosage: np.ndarray, chromosome: np.ndarray, *, max_variants: int,
              seed: int) -> dict[str, Any]:
    """Share of cross-chromosome pairs among the rows of `dosage` with r^2 at each threshold."""
    keep = np.arange(chromosome.size)
    if keep.size > max_variants:
        keep = np.sort(np.random.default_rng(seed).choice(keep, max_variants, replace=False))
    r2 = r2_matrix(dosage[keep])
    labels = chromosome[keep]
    different = np.triu(labels[:, None] != labels[None, :], k=1)
    values = r2[different]
    called = values[~np.isnan(values)]
    return {"variants": int(keep.size), "variants_available": int(chromosome.size),
            "pairs": int(values.size), "r2_undefined": int(values.size - called.size),
            **{f"pairs_r2_ge_{t:g}": int(np.sum(called >= t)) for t in R2_THRESHOLDS}}
