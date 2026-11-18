"""Where an Analysis's standard errors reach SD Units (ADR 0029, ADR 0084).

An Analysis whose `original_sd_method` carries a Phenotype Standard Deviation
stores `se = source_se / original_sd`; `z` is scale-free and is stored as the
source gave it. This module owns that divide, and `storable` is the only
caller that applies it: every writer hands `storable` its cells with the
`PhenotypeSd` they are in, and writes what `storable` returns (ADR 0077). A
writer therefore cannot hold a source's SE without its Analysis's SD applied,
and cannot apply it twice:

- a builder hands source SEs with `PhenotypeSd.dividing_by`, one SD per
  Analysis, as its manifest declares it (`layouts.build_manifest`);
- a writer that re-judges cells a Store already holds -- a completion, a
  re-encode, a component written from its Pass 2 spill -- hands
  `PhenotypeSd.APPLIED`, which divides by nothing.

`storable` takes the argument with no default, so a writer that says neither
does not run. Issue #366 found the divide restated at five builder call sites
and missing from the two Ragged builders, which is how four published Ragged
SSF Store Releases came to hold their sources' SEs.
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

        Each must be finite and positive -- the manifest parser has already
        refused anything else, naming the Analysis -- so this refuses one that is
        not rather than store every SE as 0 or `inf`.
        """
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

    def standardised(self, se: np.ndarray, analyses: np.ndarray | int) -> np.ndarray:
        """`se` in SD Units: each cell divided by its Analysis's SD.

        `analyses` broadcasts against `se` as `storable`'s does. `se` itself is
        returned, not a copy, when nothing divides it -- `APPLIED`, or every SD
        1.0 -- so a Store that standardises nothing is written from the very
        array it was handed. The SD is a `float32`, so a `float32` SE is
        divided in `float32`: the arithmetic every builder used before the
        divide moved here, so a standardised Store holds the bytes it held then.
        """
        if self.divisors is None or all(divisor == 1.0 for divisor in self.divisors):
            return se
        sd = np.asarray(self.divisors, dtype=np.float32)
        if len(self.divisors) == 1:
            one: np.ndarray = np.asarray(se) / sd[0]
            return one
        owner = np.broadcast_to(np.asarray(analyses, dtype=np.int64), np.shape(se))
        return np.asarray(se) / sd[owner]


PhenotypeSd.APPLIED = PhenotypeSd(None)
