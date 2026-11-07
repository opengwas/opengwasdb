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

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


def columns_of(table: DataclassInstance) -> dict[str, npt.NDArray[np.generic]]:
    """A table's array columns by name, in declaration order."""
    return {
        f.name: getattr(table, f.name)
        for f in fields(table)
        if isinstance(getattr(table, f.name), np.ndarray)
    }


def check_columns(table: DataclassInstance) -> None:
    """Raise ValueError unless every array column of a table is 1-D and all are one length."""
    shapes = {name: column.shape for name, column in columns_of(table).items()}
    if len(set(shapes.values())) > 1 or any(len(shape) != 1 for shape in shapes.values()):
        raise ValueError(f"{type(table).__name__} columns must be 1-D and equally long: {shapes}")


@dataclass(frozen=True)
class Variants:
    """Rows of the Variant Set (spec §2): one per variant, in the order the caller asked."""

    chrom: npt.NDArray[np.str_]
    position: npt.NDArray[np.int64]
    alid: npt.NDArray[np.str_]
    a1: npt.NDArray[np.str_]
    a2: npt.NDArray[np.str_]
    a1_frequency: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        check_columns(self)

    def __len__(self) -> int:
        return len(self.position)

    def columns(self) -> dict[str, npt.NDArray[np.generic]]:
        return columns_of(self)


@dataclass(frozen=True)
class Partners:
    """Each of a list of variants' partners and their Stored r, in CSR form.

    The partners of the i-th variant asked for are `partner[indptr[i]:indptr[i + 1]]`,
    as global Variant Set rows in ascending order, never the variant itself, with their
    Stored r in `r` at the same offsets.
    """

    indptr: npt.NDArray[np.int64]
    partner: npt.NDArray[np.int64]
    r: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        if (
            len(self.partner) != len(self.r)
            or len(self.indptr) == 0
            or self.indptr[-1] != len(self.partner)
        ):
            raise ValueError("Partners: indptr must end at len(partner) == len(r)")

    def __len__(self) -> int:
        return len(self.indptr) - 1
