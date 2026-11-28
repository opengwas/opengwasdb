"""Per Analysis, the SEs a residual plane holds as exact exceptions (spec §6a, issue #301).

Counted as each region or band of codes is written, so no pass over the plane is
needed; what a Reference Completion records against the 2% exception budget
(`opengwasdb.completion.se_exceptions`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from opengwasdb.store_reader.encoding.plan import SE_EXCEPTION, SE_MISSING


@dataclass(frozen=True)
class SeExceptionCounts:
    """Per Analysis, the SEs a residual plane holds and those it holds as exact
    exceptions (spec §6a), each also counted among the imputed cells alone.

    What a Reference Completion records against the 2% exception budget, which
    it reports rather than enforces (issue #301): counted as each region's codes
    are written, so no pass over the plane is needed to know them.
    """

    n_se: np.ndarray
    n_exceptions: np.ndarray
    n_imputed_se: np.ndarray
    n_imputed_exceptions: np.ndarray

    @classmethod
    def zeros(cls, n_analyses: int) -> SeExceptionCounts:
        return cls(*(np.zeros(n_analyses, dtype=np.int64) for _ in range(4)))

    def merge(self, other: SeExceptionCounts, *, sign: int = 1) -> None:
        """Add another band's counts (or, with `sign=-1`, take them off), in place."""
        for total, part in zip(self._totals(), other._totals(), strict=True):
            total += sign * part

    def _totals(self) -> tuple[np.ndarray, ...]:
        return self.n_se, self.n_exceptions, self.n_imputed_se, self.n_imputed_exceptions

    def add(self, codes: np.ndarray, analysis_index: np.ndarray, imputed: np.ndarray) -> None:
        """Count one region's codes, in place."""
        held, exact, flagged = codes != SE_MISSING, codes == SE_EXCEPTION, imputed == 1
        n = len(self.n_se)
        for total, mask in (
            (self.n_se, held),
            (self.n_exceptions, exact),
            (self.n_imputed_se, held & flagged),
            (self.n_imputed_exceptions, exact & flagged),
        ):
            total += np.bincount(analysis_index[mask], minlength=n)
