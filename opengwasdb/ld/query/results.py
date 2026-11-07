"""What a query returns: an LD matrix, a proxies table, a clumping (spec §8, §9).

Each is a frozen dataclass of numpy arrays plus the store facts that make it
self-describing: t_far and the distances the store measured. So a result read on its own
still says what every unstored pair means, and no second matrix is needed for it (owner,
7 November 2026).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ogld.format import answer_kinds
from ogld.store.tables import Variants, check_columns, columns_of


@dataclass(frozen=True)
class LDMatrix:
    """One signed-r matrix over `variants`, rows and columns in the same order (spec §8).

    `r[i, j]` is the Stored r of variants i and j, signed to each one's A1, with 1 on the
    diagonal. A pair the store does not hold is NaN, unless the caller passed `fill`, in
    which case it is `fill`. `absent` lists the requested ALIDs outside the Variant Set;
    they are not rows. `kinds()` says which answer each cell is.
    """

    r: npt.NDArray[np.float64]
    variants: Variants
    absent: tuple[str, ...]
    t_far: float
    near_bp: int
    far_bp: int
    fill: float | None

    def __post_init__(self) -> None:
        if self.r.shape != (len(self.variants), len(self.variants)):
            raise ValueError(f"r is {self.r.shape}, but there are {len(self.variants)} variants")

    def kinds(self) -> npt.NDArray[np.str_]:
        """The `AnswerKind` value of every cell: stored, bound or not_measured (spec §8).

        Recovered from the variants' chromosomes and positions and t_far. A `fill` whose
        square reaches t_far could be a stored r beyond near_bp, so it raises rather
        than guess.
        """
        if self.fill is not None and self.fill * self.fill >= self.t_far:
            detail = f"fill={self.fill} could be a stored r (r² ≥ t_far={self.t_far})"
            raise ValueError(f"{detail}, so the kinds are ambiguous")
        return answer_kinds(
            self.r,
            chrom=self.variants.chrom,
            position=self.variants.position,
            t_far=self.t_far,
            near_bp=self.near_bp,
            far_bp=self.far_bp,
        )


@dataclass(frozen=True)
class Proxies:
    """Every proxy of each query variant: one row per (query, proxy) pair (spec §9).

    A proxy is another Variant Set variant within the window with r̂² ≥ `r2_min`; a
    variant is never its own proxy. Rows are grouped by query in request order, then by
    r² descending, distance ascending and proxy ALID. `r` is signed to each variant's A1;
    `distance` is |position difference| in bp.
    """

    query: npt.NDArray[Any]
    proxy: npt.NDArray[Any]
    r: npt.NDArray[np.float64]
    r2: npt.NDArray[np.float64]
    distance: npt.NDArray[np.int64]
    absent: tuple[str, ...]
    r2_min: float
    window_bp: float
    t_far: float

    def __post_init__(self) -> None:
        check_columns(self)

    def __len__(self) -> int:
        return len(self.r)

    def columns(self) -> dict[str, npt.NDArray[Any]]:
        return columns_of(self)


@dataclass(frozen=True)
class Clumps:
    """A greedy clumping of the requested variants: one row per present variant (ogld#6).

    In request order. `lead` is the ALID of the clump the variant joined (its own for a
    lead) and `r` its signed r with that lead (1 for a lead). Clumping is strongest
    score first, ties by genomic order then ALID, and a variant joins the strongest lead
    it reaches at r̂² ≥ `r2` within `window_bp`.
    """

    alid: npt.NDArray[Any]
    chrom: npt.NDArray[np.str_]
    position: npt.NDArray[np.int64]
    score: npt.NDArray[np.float64]
    lead: npt.NDArray[Any]
    r: npt.NDArray[np.float64]
    absent: tuple[str, ...]
    r2: float
    window_bp: float

    def __post_init__(self) -> None:
        check_columns(self)

    def __len__(self) -> int:
        return len(self.alid)

    def columns(self) -> dict[str, npt.NDArray[Any]]:
        return columns_of(self)

    def leads(self) -> npt.NDArray[Any]:
        """The lead ALIDs, strongest first, ties by genomic order then ALID (ogld#6)."""
        raise NotImplementedError("ogld#6")
