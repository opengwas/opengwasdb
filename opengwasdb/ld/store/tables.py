"""The tables the store layer hands up: variants, and per-variant partners.

Tables are frozen dataclasses of equal-length one-dimensional numpy arrays, one per
column, so they need no dependency beyond numpy, type-check under mypy --strict, and
convert to a data frame in one call (`pandas.DataFrame(table.columns())`). A string
column is a numpy unicode array (`np.str_`), one string per row.
"""

from __future__ import annotations
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING
import numpy as np
import numpy.typing as npt

# The read side of this module is ogquery's (opengwas/ogquery#2): re-exported by name.
from ogquery.ld.store.tables import (
    check_columns as check_columns,
    columns_of as columns_of,
    DataclassInstance as DataclassInstance,
    Variants as Variants,
)


@dataclass(frozen=True)
class Partners:
    """Each of a list of variants' partners, their Stored r and distance, in CSR form.

    The partners of the i-th variant asked for are `partner[indptr[i]:indptr[i + 1]]`,
    as global Variant Set rows in ascending order, never the variant itself, with their
    Stored r in `r` and their distance from it in bp, |p_partner − p_variant|, in
    `distance` at the same offsets. The distance comes from the positions the reader
    keeps, so a query filters on it without reading a string column (ogld#14).
    """

    indptr: npt.NDArray[np.int64]
    partner: npt.NDArray[np.int64]
    r: npt.NDArray[np.float64]
    distance: npt.NDArray[np.int64]

    def __post_init__(self) -> None:
        entries = len(self.partner)
        if (
            len(self.r) != entries
            or len(self.distance) != entries
            or len(self.indptr) == 0
            or self.indptr[-1] != entries
        ):
            raise ValueError("Partners: indptr must end at len(partner) == len(r) == len(distance)")

    def __len__(self) -> int:
        return len(self.indptr) - 1
