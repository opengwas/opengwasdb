"""The rule that explains a lead that differs from plink2's (opengwas/ogld#6, ld-store §3).

`clump` joins two variants at r̂² > r2 within the window, on the Stored r of a pair, which is the
true r rounded to the store's encoding: |r̂ − r| ≤ `max_error`, half a code. plink2 judges the same
pair on the true r. So the two can disagree about a pair only when the true r² and r̂² lie on
opposite sides of r2, and then r̂² is within 2|r̂|·e + e² of r2 (e = `max_error`), because
|r̂² − r²| ≤ 2|r̂|·e + e². `explain` finds every pair the two judge on opposite sides and refuses
the case when one lies outside that band: a difference the encoding cannot explain is a defect.
It also checks plink2's r against the Stored r for every pair in the window, not only the flipped
ones: |r̂| must be within half a code (and the six digits plink2 prints) of √r²_plink2, or the
band proves nothing, since a wrong r̂ that lands on the threshold's code would pass it. Otherwise
it replays the greedy twice, on the store's joins and with exactly those pairs flipped, and
returns both lead sets; the caller compares the first with `clump` and the second with plink2.

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


#: plink2 prints r² to six significant digits; a printed r is that near the true one.
PRINTED = 1e-6


def within_band(r_hat: float, r2: float, max_error: float) -> bool:
    """Whether a pair stored as `r_hat` can be on either side of r2 once rounded: r̂² is within
    2|r̂|·e + e² of r2 (e = `max_error`). The one definition; the acceptance run uses it too."""
    return bool(abs(r_hat**2 - r2) <= 2 * abs(r_hat) * max_error + max_error**2)


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
        raise OutsideBand(f"pair ({i}, {j}) is inside the window and not stored (ld-store §3)")
    if np.isnan(true_r2[near]).any():
        i, j = np.argwhere(near & np.isnan(true_r2))[0]
        raise OutsideBand(f"pair ({i}, {j}) is inside the window and plink2 gave no r² for it")
    off = near & (
        np.abs(np.abs(np.where(near, r_hat, 0.0)) - np.sqrt(np.where(near, true_r2, 0.0)))
        > max_error + PRINTED
    )
    if off.any():
        i, j = np.argwhere(off)[0]
        raise OutsideBand(
            f"pair ({i}, {j}): stored r̂ {r_hat[i, j]:.6g} is further than half a code from the "
            f"r plink2 measured, √{true_r2[i, j]:.6g}: it is not that r rounded"
        )
    stored = np.where(near, r_hat, 0.0) ** 2 > r2
    truth = near & (true_r2 > r2)
    differ = stored ^ truth
    flipped = []
    for i, j in np.argwhere(np.triu(differ)):
        if not within_band(r_hat[i, j], r2, max_error):
            raise OutsideBand(
                f"pair ({i}, {j}): stored r̂² {r_hat[i, j] ** 2:.6g}, true r² {true_r2[i, j]:.6g} "
                f"fall on opposite sides of {r2}, and r̂² is further from it than the encoding "
                f"allows at half a code of {max_error:.3g}"
            )
        flipped.append(Pair(int(i), int(j), float(r_hat[i, j] ** 2), float(true_r2[i, j])))
    return Explained(flipped, replay(order, stored), replay(order, stored ^ differ))
