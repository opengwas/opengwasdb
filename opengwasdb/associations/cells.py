"""A region of one component's association cells, in that component's own indices."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np

Kind = Literal["analysis", "analyses", "variants", "cross", "pairs", "positions"]


def _indices(values: Iterable[int] | np.ndarray | int) -> np.ndarray:
    return np.atleast_1d(np.asarray(values, dtype=np.int64)).ravel()


@dataclass(frozen=True, eq=False)
class Cells:
    """What a reader is asked for. Each region names its rows' order.

    * `analysis(a)`: one Analysis, variants ascending.
    * `analyses(a…)`: those Analyses in the order given, each as `analysis`.
    * `variants(v…)`: every Analysis at those variants -- variant-major on a
      Dense grid, the by-variant index's order on a CSR (ADR 0060).
    * `cross(v…, a…)`: those variants × those Analyses, Analysis-major in the
      order given; within each, a grid keeps the variants' order given and a
      CSR its stored, ascending order.
    * `pairs(v…, a…)`: the cells `(v[i], a[i])`, in request order, one row each.
    * `positions(p…)`: flat sequence positions -- a CSR's own address space, for
      code that scans the sequence itself; no other reader accepts it.
    """

    kind: Kind
    #: The variant indices a region names (`variants`, `cross`, `pairs`).
    vi: np.ndarray | None = None
    #: The Analysis indices a region names (`analysis`, `analyses`, `cross`, `pairs`).
    ai: np.ndarray | None = None
    #: The flat positions of a `positions` region.
    pos: np.ndarray | None = None

    @classmethod
    def analysis(cls, analysis_index: int) -> Cells:
        return cls("analysis", ai=_indices(int(analysis_index)))

    @classmethod
    def analyses(cls, analysis_indices: Iterable[int] | np.ndarray) -> Cells:
        return cls("analyses", ai=_indices(analysis_indices))

    @classmethod
    def variants(cls, variant_indices: Iterable[int] | np.ndarray) -> Cells:
        return cls("variants", vi=_indices(variant_indices))

    @classmethod
    def cross(
        cls,
        variant_indices: Iterable[int] | np.ndarray,
        analysis_indices: Iterable[int] | np.ndarray,
    ) -> Cells:
        return cls(
            "cross",
            vi=_indices(variant_indices),
            ai=_indices(analysis_indices),
        )

    @classmethod
    def pairs(cls, variant_indices: np.ndarray, analysis_indices: np.ndarray) -> Cells:
        vi, ai = _indices(variant_indices), _indices(analysis_indices)
        if len(vi) != len(ai):
            raise ValueError(f"pairs need equal lengths, got {len(vi)} and {len(ai)}")
        return cls("pairs", vi=vi, ai=ai)

    @classmethod
    def positions(cls, positions: np.ndarray) -> Cells:
        return cls("positions", pos=_indices(positions))

    @property
    def analysis_index(self) -> int:
        """The one Analysis of an `analysis` region."""
        assert self.ai is not None and self.kind == "analysis"
        return int(self.ai[0])
