"""The Indexed Variant Subset adapter (ADR 0061): a Dense release's Analysis-major copy."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from opengwasdb.associations.cells import Cells
from opengwasdb.associations.record import Associations, Inspection, Status
from opengwasdb.layouts.dense.indexed_subsets import IndexedSubset, open_indexed_subset


class SubsetReader:
    """Reads one Analysis at a time from a published Indexed Variant Subset.

    Opening refuses an unknown, stale, incomplete or corrupt subset with the
    `IndexedSubsetError` family, before any value is decoded. Like the grid it
    copies, a subset stores no association at a missing cell, so `read` returns
    the present cells, in Store Variant Index order.
    """

    carries_n = False

    def __init__(self, subset: IndexedSubset, *, source: str) -> None:
        self._subset = subset
        self._source = source

    @classmethod
    def open(cls, store_path: str | Path, name: str) -> SubsetReader:
        return cls(
            open_indexed_subset(store_path, name),
            source=f"Indexed Variant Subset {name!r} of {store_path}",
        )

    def read(self, cells: Cells, *, observed_only: bool = False) -> Associations:
        """The Analysis's present cells; raises `InvalidAssociations` on an invalid one."""
        found = self.inspect(cells).associations(self._source)
        keep = found.status != Status.MISSING
        if observed_only:
            keep &= found.status != Status.IMPUTED
        return found.where(keep)

    def inspect(self, cells: Cells) -> Inspection:
        if cells.kind != "analysis":
            raise TypeError(f"an Indexed Variant Subset reads one Analysis, not {cells.kind!r}")
        col = cells.analysis_index
        decoded = self._subset.decode_analysis(col)
        eaf = (
            np.full(len(decoded.z), np.nan, dtype=np.float32)
            if decoded.eaf is None
            else decoded.eaf
        )
        return Inspection.of(
            decoded.variant_index, col, decoded.z, decoded.se, eaf, decoded.imputed
        )
