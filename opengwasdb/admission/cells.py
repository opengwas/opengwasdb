"""What Admission guarantees of every cell a Store holds, and its checks.

A present cell has a finite z and a finite, positive SE; z is missing exactly
where SE is (spec §15, `opengwasdb.associations.record.states`); a frequency is
NaN or inside (0, 1); and a cell is flagged imputed only when it is present.
A builder's rows reach a Store through a `Gate`, which admits only such
cells; completion's candidates and the cells a writer is handed are checked
here.
"""

from __future__ import annotations

import numpy as np

__all__ = ["NotAdmitted", "admissible"]


class NotAdmitted(ValueError):
    """A cell Admission would not admit reached a Store's writer."""


def admissible(z: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Which candidate cells may be stored as present: z finite, SE finite and positive."""
    with np.errstate(invalid="ignore"):
        keep: np.ndarray = np.isfinite(z) & np.isfinite(se) & (se > 0)
    return keep
