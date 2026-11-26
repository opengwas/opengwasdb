"""Reading a Dense release's Rho Matrix (ADR 0025): the stored packed lower triangle, unpacked.

Estimating and writing the matrix is `opengwasdb.layouts.dense.rho`'s. A Hybrid
release's Rho Matrix is its Dense Component's (#496), which records the Analyses it
holds no Rho for: those with no observed association in the component, its
Overflow-only Analyses (`recorded_overflow_only`). A Rho query naming one is refused
(`OverflowOnlyAnalysisError`), never answered NaN.
"""

from __future__ import annotations

import numpy as np
import zarr

#: The attribute of a Hybrid Dense Component's `rho` group listing, by `analysis_index`
#: ascending, its Overflow-only Analyses (#496). A standalone Dense release's carries none.
OVERFLOW_ONLY_ATTR = "overflow_only"


class OverflowOnlyAnalysisError(LookupError):
    """A Rho query on a Hybrid release named an Analysis with no observed association in
    its Dense Component, over whose Analyses the Rho Matrix is estimated (#496).

    Its Rho with any Analysis is not estimable; a NaN would read as "too little shared
    null support" between two Analyses that both have some. `fields` is what a reader of
    the Store Reader contract compares; the message is for people.
    """

    def __init__(self, analysis_id: str) -> None:
        self.analysis_id = analysis_id
        super().__init__(
            f"analysis_id {analysis_id!r} has no Rho: it has no observed association in this "
            "Hybrid release's Dense Component, over whose Analyses the Rho Matrix is "
            "estimated (its associations are all in the Ragged Overflow)"
        )

    @property
    def fields(self) -> dict[str, object]:
        return {"analysis_id": self.analysis_id}


def recorded_overflow_only(group: zarr.Group, n_analyses: int, source: str) -> frozenset[int]:
    """The Overflow-only Analyses a Hybrid Dense Component's `rho` `group` records.

    The record is a list of distinct `analysis_index` values, ascending, each below
    `n_analyses`, and is read exactly so: a string, a float, a boolean, a repeat or an
    index past the grid is refused, not coerced. A matrix that records none -- written
    before #496 -- is refused too: a NaN in it could be a pair without support or an
    Analysis without Rho, and the two must not be told apart by guessing. `source`
    names the group in the error.
    """
    if OVERFLOW_ONLY_ATTR not in group.attrs:
        raise ValueError(
            f"{source}: this Hybrid Dense Component's Rho Matrix does not record its "
            f"Overflow-only Analyses (`{OVERFLOW_ONLY_ATTR}`), so a pair without Rho cannot "
            "be told from one without support; rebuild it with `opengwasdb build-dense-rho` "
            "on the Hybrid release"
        )
    listed = group.attrs[OVERFLOW_ONLY_ATTR]
    well_formed = isinstance(listed, list) and all(
        type(i) is int and 0 <= i < n_analyses for i in listed
    )
    if not well_formed or listed != sorted(set(listed)):
        raise ValueError(
            f"{source}: `{OVERFLOW_ONLY_ATTR}` {listed!r} is not a list of distinct "
            f"ascending analysis_index values below {n_analyses}"
        )
    return frozenset(listed)


def packed_rho(group: zarr.Group) -> tuple[np.ndarray, np.ndarray]:
    """The stored packed lower triangle: each pair's rho (float32) and its null
    support, `n_null` (int64), in pair order, as stored (#473)."""
    return (
        np.asarray(group["rho"][:], dtype=np.float32),
        np.asarray(group["n_null"][:], dtype=np.int64),
    )


class DenseRhoReader:
    """Unpack a stored packed strict-lower-triangle Rho Matrix into symmetric
    form. The packed arrays are tiny at any supported scale (PRD "Storage" /
    "Known Limitation"), so they -- and the reconstructed full matrix, once
    asked for -- are held in memory rather than re-read per query."""

    def __init__(self, group: zarr.Group, n_analyses: int):
        self.group = group
        self.n_analyses = n_analyses
        self._packed_rho, self._packed_n_null = packed_rho(group)
        self._overflow_only: frozenset[int] | None = None
        self._full_rho: np.ndarray | None = None
        self._full_n_null: np.ndarray | None = None

    def overflow_only(self, source: str) -> frozenset[int]:
        """The Overflow-only Analyses this matrix records (`recorded_overflow_only`): a
        Hybrid Dense Component's, read once. `source` names the matrix in a refusal."""
        if self._overflow_only is None:
            self._overflow_only = recorded_overflow_only(self.group, self.n_analyses, source)
        return self._overflow_only

    def _ensure_full(self) -> None:
        if self._full_rho is not None:
            return
        n = self.n_analyses
        rows, cols = np.tril_indices(n, k=-1)
        full_rho = np.eye(n, dtype=np.float32)
        full_n_null = np.zeros((n, n), dtype=np.int64)
        full_rho[rows, cols] = self._packed_rho
        full_rho[cols, rows] = self._packed_rho
        full_n_null[rows, cols] = self._packed_n_null
        full_n_null[cols, rows] = self._packed_n_null
        self._full_rho = full_rho
        self._full_n_null = full_n_null

    def pair(self, i: int, j: int) -> tuple[float, int]:
        """Rho and support between analysis indices ``i`` and ``j``."""
        self._ensure_full()
        assert self._full_rho is not None and self._full_n_null is not None
        return float(self._full_rho[i, j]), int(self._full_n_null[i, j])

    def row(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Rho and support of analysis index ``i`` against every analysis
        (including itself: rho=1.0, n_null=0 at the diagonal)."""
        self._ensure_full()
        assert self._full_rho is not None and self._full_n_null is not None
        return self._full_rho[i, :].copy(), self._full_n_null[i, :].copy()

    def matrix(self, indices: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Full symmetric matrix, or the dense submatrix for ``indices``."""
        self._ensure_full()
        assert self._full_rho is not None and self._full_n_null is not None
        if indices is None:
            return self._full_rho.copy(), self._full_n_null.copy()
        idx = np.asarray(indices, dtype=np.int64)
        if len(idx) == 0:
            return (
                np.empty((0, 0), dtype=np.float32),
                np.empty((0, 0), dtype=np.int64),
            )
        return self._full_rho[np.ix_(idx, idx)], self._full_n_null[np.ix_(idx, idx)]
