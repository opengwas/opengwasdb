"""The rule that explains a lead that differs from plink2's (opengwas/ogld#6, spec §3).

`clump` joins two variants at r̂² > r2 within the window, on the Stored r of a pair, which is the
true r rounded to the store's encoding: |r̂ − r| ≤ `max_error`, half a code. plink2 judges the same
pair on the true r. So the two can disagree about a pair only when the true r² and r̂² lie on
opposite sides of r2, and then r̂² is within 2|r̂|·e + e² of r2 (e = `max_error`), because
|r̂² − r²| ≤ 2|r̂|·e + e². `explain` finds every pair the two judge on opposite sides and refuses
the case when one lies outside that band: a difference the encoding cannot explain is a defect.
Otherwise it replays the greedy twice, on the store's joins and with exactly those pairs flipped,
and returns both lead sets; the caller compares the first with `clump` and the second with plink2.

Pure numpy, so the tier, the acceptance run and the tests share one definition.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


class OutsideBand(ValueError):
    """A pair the store and plink2 judge differently, which the encoding cannot explain."""


@dataclass(frozen=True)
class Pair:
    """A pair the store and plink2 judge on opposite sides of the threshold."""

    i: int
    j: int
    store_r2: float
    true_r2: float


@dataclass(frozen=True)
class Explained:
    """`flipped` are the pairs judged differently; `leads_store` and `leads_true` the leads of
    the greedy on the store's joins and on the joins with exactly those pairs flipped, as indices
    into the variants given."""

    flipped: list[Pair]
    leads_store: list[int]
    leads_true: list[int]


def replay(order: Sequence[int], join: npt.NDArray[np.bool_]) -> list[int]:
    """The greedy clumping of `order` (strongest first) under `join`: each variant not yet
    clumped leads, and takes every unclumped variant it joins. The leads, in order."""
    clumped = np.zeros(len(join), dtype=bool)
    leads: list[int] = []
    for i in order:
        if clumped[i]:
            continue
        leads.append(int(i))
        clumped[i] = True
        clumped |= join[i]
    return leads


def explain(
    *,
    order: Sequence[int],
    position: npt.NDArray[np.int64],
    r_hat: npt.NDArray[np.float64],
    true_r2: npt.NDArray[np.float64],
    r2: float,
    window_bp: float,
    max_error: float,
) -> Explained:
    """Explain the pairs `r_hat` (the store's signed r) and `true_r2` (plink2's) judge apart.

    Within `window_bp` (distance < window) a pair joins at r̂² > r2 on the store's side and at
    r² > r2 on plink2's. Raises `OutsideBand` for an unstored pair inside the window, and for a
    flipped pair whose r̂² is further from r2 than the encoding's error allows.
    """
    near = np.abs(position[:, None] - position[None, :]) < window_bp
    np.fill_diagonal(near, False)
    if np.isnan(r_hat[near]).any():
        i, j = np.argwhere(near & np.isnan(r_hat))[0]
        raise OutsideBand(f"pair ({i}, {j}) is inside the window and not stored (spec §3)")
    stored = np.where(near, r_hat, 0.0) ** 2 > r2
    truth = near & (true_r2 > r2)
    differ = stored ^ truth
    flipped = []
    for i, j in np.argwhere(np.triu(differ)):
        slack = 2 * abs(r_hat[i, j]) * max_error + max_error**2
        gap = abs(r_hat[i, j] ** 2 - r2)
        if gap > slack:
            raise OutsideBand(
                f"pair ({i}, {j}): stored r̂² {r_hat[i, j] ** 2:.6g}, true r² {true_r2[i, j]:.6g} "
                f"fall on opposite sides of {r2}, but r̂² is {gap:.3g} from it and the encoding "
                f"allows {slack:.3g}"
            )
        flipped.append(Pair(int(i), int(j), float(r_hat[i, j] ** 2), float(true_r2[i, j])))
    return Explained(flipped, replay(order, stored), replay(order, stored ^ differ))
