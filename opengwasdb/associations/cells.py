"""A region of one component's association cells, in that component's own indices."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, overload

import numpy as np

Kind = Literal["analysis", "analyses", "variants", "cross", "pairs", "positions"]


def _indices(values: Iterable[int] | np.ndarray | int) -> np.ndarray:
    return np.atleast_1d(np.asarray(values, dtype=np.int64)).ravel()


@dataclass(frozen=True, eq=False)
class Cells:
    """What a reader is asked for. Each region names its rows' order.

    * `analysis(a)`: one Analysis, variants ascending.
    * `analyses(first, stop)`: the span of Analyses `[first, stop)`, each as
      `analysis`; `analyses(a…)`: those Analyses in the order given, likewise.
    * `variants(v…)`: every Analysis at those variants -- variant-major on a
      Dense grid, the by-variant index's order on a CSR (ADR 0060).
    * `cross(v…, a…)`: those variants × those Analyses, Analysis-major in the
      order given; within each, a grid keeps the variants' order given and a
      CSR its stored, ascending order. `DenseGridReader.windows` cuts a scan
      into these, a run of one Analysis's variants each.
    * `pairs(v…, a…)`: the cells `(v[i], a[i])`, in request order, one row each.
    * `positions(p…)`: flat sequence positions -- a CSR's own address space, and
      what `CsrReader.windows` cuts a scan into; no other reader accepts it.
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

    @overload
    @classmethod
    def analyses(cls, first: int, stop: int) -> Cells: ...

    @overload
    @classmethod
    def analyses(cls, first: Iterable[int] | np.ndarray) -> Cells: ...

    @classmethod
    def analyses(cls, first: int | Iterable[int] | np.ndarray, stop: int | None = None) -> Cells:
        if stop is None:
            return cls("analyses", ai=_indices(first))
        if not isinstance(first, (int, np.integer)) or not 0 <= first <= stop:
            raise ValueError(f"a span of Analyses runs forward from 0, not [{first}, {stop})")
        return cls("analyses", ai=np.arange(first, stop, dtype=np.int64))

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
    def span(self) -> tuple[int, int] | None:
        """The `[first, stop)` an `analysis` or `analyses` region's Analyses run over,
        when they are one ascending run; None for any other region."""
        ai = self.ai
        if self.kind not in ("analysis", "analyses") or ai is None:
            return None
        if len(ai) == 0:
            return 0, 0
        if not bool(np.all(np.diff(ai) == 1)):
            return None
        return int(ai[0]), int(ai[-1]) + 1

    @property
    def analysis_index(self) -> int:
        """The one Analysis of an `analysis` region."""
        assert self.ai is not None and self.kind == "analysis"
        return int(self.ai[0])


def require_within(cells: Cells, n_analyses: int, source: str) -> None:
    """Refuse a region that names an Analysis outside `[0, n_analyses)`.

    Neither answer a reader would give is an error of its own. Analysis -1 reads the
    last Analysis's cells, numbered -1, as numpy counts from the end. An Analysis past
    the end can read as no associations, which a scan takes for no hits.
    """
    ai = cells.ai
    if ai is None or not len(ai):
        return
    outside = (ai < 0) | (ai >= n_analyses)
    if outside.any():
        raise IndexError(
            f"{source}: Analysis index {int(ai[outside][0])} is outside "
            f"[0, {n_analyses}), the Analyses this component holds"
        )


def window_region(
    cells: Cells | None,
    max_cells: int | None,
    *,
    default: int,
    n_analyses: int,
    source: str,
) -> tuple[Cells, int]:
    """The region a reader's `windows` cuts, and the cells each window holds, checked.

    Every reader checks the same things before it cuts a window: that a window
    holds at least one cell, that the region names Analyses (none means all of
    them), and that each of those Analyses is one the component holds.
    """
    step = default if max_cells is None else int(max_cells)
    if step < 1:
        raise ValueError(f"a window holds at least one cell, not {max_cells}")
    region = Cells.analyses(0, n_analyses) if cells is None else cells
    if region.kind not in ("analysis", "analyses"):
        raise TypeError(f"windows cut Analyses, not a {region.kind!r} region")
    require_within(region, n_analyses, source)
    return region, step
