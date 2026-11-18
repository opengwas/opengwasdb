"""Where an Analysis's standard errors reach SD Units (ADR 0029, ADR 0084).

An Analysis whose `original_sd_method` carries a Phenotype Standard Deviation
stores `se = source_se / original_sd`; `z` is scale-free and is stored as the
source gave it. This module holds the divide (`PhenotypeSd.standardised`). Two
callers use it: `storable`, which applies it to every cell a writer stores
(ADR 0077), and `validate`'s source-fidelity check, which uses it to compute
the SE a build stored and stores nothing.

A writer hands `storable` its cells with the `PhenotypeSd` they are in; the
argument has no default:

- a source builder hands its source's SEs with `PhenotypeSd.dividing_by`, one
  SD per Analysis of the batch -- a count that does not match is refused, and
  "no rescaling" is the 1.0 the manifest parser supplies, never an omission;
- a writer of cells a Store, or a Hybrid Pass 2 spill, already holds -- a
  completion, an encode, the Overflow assembled from its spills -- hands
  `PhenotypeSd.APPLIED`, which divides by nothing.

What keeps a source's SE from being stored undivided, or divided twice, is
enforced by test rather than by type: `tests/test_phenotype_sd_standardisation.py`
searches the whole package and fails if any module names `APPLIED` that is not
on its allowlist of stored-cell writers (`_MAY_SAY_APPLIED`, each entry with its
reason), or if anything but this module constructs a `PhenotypeSd`;
`tests/test_sd_standardisation_seam.py` builds one fixture through every source
path and expects SE / SD exactly once. A writer that called `storable` twice
with `dividing_by` on the same cells would divide twice; no type prevents it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

__all__ = ["PhenotypeSd"]


@dataclass(frozen=True)
class PhenotypeSd:
    """The SD each Analysis's SEs are divided by on their way into a Store.

    `divisors` holds one per Analysis, indexed as `storable`'s `analyses`
    are; 1.0 divides by nothing. `None` is `APPLIED`: the SEs are already in
    the Store's scale.
    """

    divisors: tuple[float, ...] | None

    #: SEs a Store already holds, or that a writer already took through
    #: `storable`: dividing them again would standardise them twice.
    APPLIED: ClassVar[PhenotypeSd]

    @classmethod
    def dividing_by(cls, *divisors: float) -> PhenotypeSd:
        """Source SEs, divided by each Analysis's SD: one value per Analysis, in
        the order `storable`'s `analyses` index them.

        At least one, and each finite and positive -- the manifest parser has
        already refused anything else, naming the Analysis -- so this refuses
        none at all, which would divide nothing, and a value that would store
        every SE as 0 or `inf`.
        """
        if not divisors:
            raise ValueError(
                "at least one phenotype SD is needed to divide a source's SEs by; an "
                "Analysis that is not rescaled is divided by 1.0, which the manifest "
                "parser supplies"
            )
        values = tuple(float(divisor) for divisor in divisors)
        bad = [value for value in values if not (math.isfinite(value) and value > 0)]
        if bad:
            raise ValueError(
                f"a phenotype SD must be finite and positive to divide an SE by; got {bad[0]!r}"
            )
        return cls(values)

    @classmethod
    def of_analyses(cls, divisors: Iterable[float]) -> PhenotypeSd:
        """`dividing_by` for an iterable of per-Analysis SDs."""
        return cls.dividing_by(*divisors)

    def standardised(
        self, se: np.ndarray, analyses: np.ndarray | int, n_analyses: int
    ) -> np.ndarray:
        """`se` in SD Units: each cell divided by its Analysis's SD.

        `analyses` broadcasts against `se` as `storable`'s does, over a batch of
        `n_analyses` Analyses; a `dividing_by` that does not hold one SD for each
        is refused, naming both counts. `se` itself is
        returned, not a copy, when nothing divides it -- `APPLIED`, or every SD
        1.0 -- so a Store that standardises nothing is written from the very
        array it was handed. The SD is a `float32`, so a `float32` SE is
        divided in `float32`: the arithmetic every builder used before the
        divide moved here, so a standardised Store holds the bytes it held then.
        """
        if self.divisors is None:
            return se
        if len(self.divisors) != n_analyses:
            raise ValueError(
                f"{len(self.divisors)} phenotype SD(s) for {n_analyses} Analyses: each "
                "Analysis of the batch needs its own"
            )
        if all(divisor == 1.0 for divisor in self.divisors):
            return se
        sd = np.asarray(self.divisors, dtype=np.float32)
        if len(self.divisors) == 1:
            one: np.ndarray = np.asarray(se) / sd[0]
            return one
        owner = np.broadcast_to(np.asarray(analyses, dtype=np.int64), np.shape(se))
        return np.asarray(se) / sd[owner]


PhenotypeSd.APPLIED = PhenotypeSd(None)
