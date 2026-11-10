"""The PC Diagnostic: what PC Adjustment did to chance LD, and to real LD (spec §7).

**Chance LD.** opengwas/opengwasdb#405 measured chance LD on unlinked pairs drawn this way
(`prototypes/locus_leads/panel_chance.py` at `6b97e57`): at most 3,000 variants drawn without
replacement by seed 405, then 200,000 pairs of them on different chromosomes, drawn with
replacement by a second generator seeded 405. A build draws its pairs the same way from its
Variant Set, and records the share at r² ≥ 0.01 before PC Adjustment, after it, and as sampling
alone predicts, P(χ²₁ ≥ 0.01 × (n − k − 1)). A Variant Set on one chromosome has no
cross-chromosome pair, so its rates are null, never a number nothing measured.

**Real LD.** The quantiles 0, 0.5, 0.9, 0.99 and 1 of the change in r (adjusted minus
unadjusted, signed to A1) over every pair within 100 kb with both variants outside the
Long-Range LD Regions: ogld#5's "real LD survives". A whole panel has about 10⁹ such pairs, so
they are counted into bins 2⁻²⁰ (about 10⁻⁶) wide rather than held; the extremes are exact,
and each other quantile is the order statistic at rank ⌊q(N − 1)⌋, to within half a bin:
the centre of the bin that holds it.
"""

from __future__ import annotations

import functools
import importlib.resources
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib.resources.abc import Traversable
from typing import Final

import numpy as np
import numpy.typing as npt

from ogld.format import InputError, PcDiagnostic

SEED: Final = 405
MAX_VARIANTS: Final = 3_000
PAIRS: Final = 200_000
RATE_R2: Final = 0.01
CHANGE_BP: Final = 100_000
QUANTILES: Final = (0.0, 0.5, 0.9, 0.99, 1.0)
PAIR_SOURCE: Final = (
    "opengwas/opengwasdb#405's seeded cross-chromosome pairs "
    "(prototypes/locus_leads/panel_chance.py at 6b97e57): at most 3,000 Variant Set variants "
    "drawn without replacement with seed 405, then 200,000 pairs of them on different "
    "chromosomes drawn with seed 405"
)
NO_PAIR_SOURCE: Final = (
    "none: the Variant Set is on one chromosome, so it has no cross-chromosome pair and the "
    "rates are not measured"
)

#: The Long-Range LD Regions' table (build-inputs §9), data of this package, which owns it.
REGIONS_FILE: Final = "long-range-ld-regions-grch38.tsv"


def long_range_ld_regions_file() -> Traversable:
    """The table's file: tab-separated, a `#` comment, a header, one region a row."""
    return importlib.resources.files("ogld.build").joinpath(REGIONS_FILE)


@functools.cache
def long_range_ld_regions() -> tuple[tuple[str, str, int, int], ...]:
    """The Long-Range LD Regions, GRCh38, 1-based inclusive: (name, chromosome, start, end), in
    the table's order. The one reader of the table: the build's PC diagnostic, the PC Files'
    recipe (`scripts/make_pcs.py`) and the Test Store's script all call it."""
    lines = long_range_ld_regions_file().read_text().splitlines()
    rows = [line.split("\t") for line in lines if line and not line.startswith("#")][1:]
    return tuple((row[0], row[1], int(row[2]), int(row[3])) for row in rows)


def outside_regions(chrom: str, position: npt.NDArray[np.int64]) -> npt.NDArray[np.bool_]:
    """True for each variant of chromosome `chrom` outside every Long-Range LD Region."""
    inside = np.zeros(len(position), dtype=np.bool_)
    for _, on, start, end in long_range_ld_regions():
        if on == chrom:
            inside |= (position >= start) & (position <= end)
    return ~inside


def require_change_pairs(
    positions: Sequence[tuple[str, npt.NDArray[np.int64]]], panel: str
) -> None:
    """Refuse a Variant Set with no pair within 100 kb outside the Long-Range LD Regions, such as
    an MHC-only slice: its change in r cannot be measured (spec §7). Each chromosome's sorted
    positions say so before the build writes anything, so nothing is left half-written."""
    for chrom, position in positions:
        clear = position[outside_regions(chrom, position)]
        if len(clear) > 1 and int(np.diff(clear).min()) <= CHANGE_BP:
            return
    raise InputError(
        f"panel {panel}: no two variants within 100 kb lie outside the Long-Range LD Regions, "
        "so the PC diagnostic's change in r cannot be measured (spec §7)"
    )


def draw_variants(count: int) -> npt.NDArray[np.int64]:
    """#405's variant draw: all of them, or 3,000 without replacement by seed 405, ascending."""
    if count <= MAX_VARIANTS:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(SEED)
    return np.sort(rng.choice(count, MAX_VARIANTS, replace=False)).astype(np.int64)


def draw_pairs(
    chromosome: npt.NDArray[np.str_],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """#405's pair draw: 200,000 index pairs on different chromosomes, by a fresh seed 405."""
    rng = np.random.default_rng(SEED)
    first: list[npt.NDArray[np.int64]] = []
    second: list[npt.NDArray[np.int64]] = []
    placed = 0
    while placed < PAIRS:
        want = PAIRS - placed
        draw = rng.integers(0, chromosome.size, size=(max(2 * want, 64), 2))
        keep = chromosome[draw[:, 0]] != chromosome[draw[:, 1]]
        first.append(draw[keep, 0][:want])
        second.append(draw[keep, 1][:want])
        placed += int(first[-1].size)
    return np.concatenate(first), np.concatenate(second)


@dataclass(frozen=True)
class ChanceRates:
    """The shares of the drawn pairs at r² ≥ 0.01, before and after PC Adjustment."""

    pairs: int
    unadjusted: float | None
    adjusted: float | None


def chance_rates(
    chromosome: npt.NDArray[np.str_],
    units_of: Callable[[npt.NDArray[np.int64]], tuple[npt.NDArray[np.float64], ...]],
) -> ChanceRates:
    """The rates over #405's pairs. `units_of(rows)` gives the drawn Variant Set rows' unit
    vectors at k = 0 and adjusted, in that order."""
    if len(set(chromosome.tolist())) < 2:
        return ChanceRates(0, None, None)
    used = draw_variants(len(chromosome))
    first, second = draw_pairs(chromosome[used])
    unadjusted, adjusted = units_of(used)
    return ChanceRates(PAIRS, _rate(unadjusted, first, second), _rate(adjusted, first, second))


def _rate(
    units: npt.NDArray[np.float64], first: npt.NDArray[np.int64], second: npt.NDArray[np.int64]
) -> float:
    r = (units @ units.T)[first, second]
    return float((r * r >= RATE_R2).mean())


def sampling_rate(n: int, k: int) -> float:
    """P(χ²₁ ≥ 0.01 × (n − k − 1)): the share of unlinked pairs at r² ≥ 0.01 by sampling alone."""
    return math.erfc(math.sqrt(RATE_R2 * (n - k - 1) / 2))


@dataclass
class ChangeHistogram:
    """The change in r, counted into bins of 2⁻²⁰ over [−2, 2], with the exact extremes."""

    width: float = 2.0**-20
    low: float = -2.0
    counts: npt.NDArray[np.int64] = field(
        default_factory=lambda: np.zeros(4 * 2**20 + 1, dtype=np.int64)
    )
    n: int = 0
    least: float = math.inf
    most: float = -math.inf

    def add(self, change: npt.NDArray[np.float64]) -> None:
        if not len(change):
            return
        at = np.floor((change - self.low) / self.width).astype(np.int64)
        at = np.clip(at, 0, len(self.counts) - 1)
        base = int(at.min())
        tally = np.bincount(at - base)
        self.counts[base : base + len(tally)] += tally
        self.n += len(change)
        self.least = min(self.least, float(change.min()))
        self.most = max(self.most, float(change.max()))

    def quantiles(self) -> dict[str, float]:
        """`near_100kb_change`: quantiles 0, 0.5, 0.9, 0.99 and 1, keyed as spec §7 names them."""
        if not self.n:
            raise InputError(
                "no pair within 100 kb lies outside the Long-Range LD Regions, so the PC "
                "diagnostic's change in r cannot be measured (spec §7)"
            )
        cumulative = np.cumsum(self.counts)
        out: dict[str, float] = {}
        for q in QUANTILES:
            rank = math.floor(q * (self.n - 1))
            at = int(np.searchsorted(cumulative, rank, side="right"))
            centre = self.low + (at + 0.5) * self.width
            out[f"{q:g}"] = min(max(centre, self.least), self.most)
        out["0"], out["1"] = self.least, self.most
        return out


def pc_diagnostic(rates: ChanceRates, n: int, k: int, change: ChangeHistogram) -> PcDiagnostic:
    """The provenance's `pc_diagnostic` (spec §7)."""
    return {
        "pairs": {
            "source": PAIR_SOURCE if rates.pairs else NO_PAIR_SOURCE,
            "seed": SEED,
            "n_pairs": rates.pairs,
        },
        "rate_r2_ge_0_01_unadjusted": rates.unadjusted,
        "rate_r2_ge_0_01_adjusted": rates.adjusted,
        "rate_r2_ge_0_01_sampling": sampling_rate(n, k),
        "near_100kb_change": change.quantiles(),
    }
