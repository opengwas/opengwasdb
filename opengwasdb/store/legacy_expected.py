"""The answers a converted release must give: its source's, with the conversion's
deliberate changes applied (#492, `opengwasdb.store.legacy_conversion`).

The rewrite's proof asks the source and the staged release the same questions
(`query_digests`) and refuses on any difference. A legacy conversion changes three
things on purpose, so its source is asked through this view of its 0.2 reader, which
makes exactly those changes to what it reads and nothing else:

- **A masked cell reads as missing** where the grid is decoded (`_MaskedGrid`): its z,
  SE and frequency NaN and its imputed flag 0, so every question treats it as the
  0.2 reader treats a cell that was always missing -- left out of a region's
  present cells, answered as `missing` by a read of named cells.
- **It leaves every Top-Hit tier** (`_MaskedTiers`), and each Analysis's
  `n_hits_*` drops by the masked cells its tier held, while
  `n_missing_se_out_of_range` counts them (`_expected_analyses`).
- **An unlifted row names its expected `source_alid`** (`_ExpectedAxis`,
  `LegacyPlan.source_alid`).

An Indexed Variant Subset's answer is a region's present cells, so a masked cell
leaves it too. Each change is therefore compared cell for cell, never left out of the
comparison: a converted release that masked one cell more or less, or named another
`source_alid`, answers some question differently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from opengwasdb.model.analyses import N_MISSING_SE_OUT_OF_RANGE, TOP_HIT_COUNT_COLUMNS
from opengwasdb.store.legacy_conversion import LegacyPlan
from opengwasdb.store.legacy_masking import cell_positions
from opengwasdb.store.open import open_store
from opengwasdb.store_reader.analyses import AnalysesIndex
from opengwasdb.store_reader.associations import Associations, Cells, DenseGridReader
from opengwasdb.store_reader.facade import StoreQuery
from opengwasdb.store_reader.top_hits import (
    TOP_HIT_THRESHOLDS,
    TierRows,
    TopHitTier,
    TopHitTiers,
)
from opengwasdb.store_reader.variant_axis import VariantAxis


def _hit(plan: LegacyPlan, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Which of the cells `(rows, cols)` (broadcast together) the plan masks."""
    return np.isin(cell_positions(rows, cols, plan.n_analyses), plan.masked)


class _MaskedGrid(DenseGridReader):
    """The source's grid reader, reading each masked cell as missing."""

    def __init__(self, grid: DenseGridReader, plan: LegacyPlan) -> None:
        self.__dict__.update(grid.__dict__)
        self._plan = plan

    def _decode(self, cells: Cells) -> Any:
        grid = super()._decode(cells)
        if not self._plan.masked.size:
            return grid
        if grid.elementwise:
            hit = _hit(self._plan, grid.rows, grid.cols)
        else:
            hit = _hit(self._plan, grid.rows[:, None], grid.cols[None, :])
        if not hit.any():
            return grid
        z, se, eaf = (np.array(values, copy=True) for values in (grid.z, grid.se, grid.eaf))
        z[hit], se[hit], eaf[hit] = np.nan, np.nan, np.nan
        flags = grid.flags
        if flags is not None:
            flags = np.array(flags, copy=True)
            flags[hit] = 0
        return grid._replace(z=z, se=se, eaf=eaf, flags=flags)

    def z(self, cells: Cells) -> np.ndarray:
        values = super().z(cells)
        if cells.vi is None or cells.ai is None or not values.size or not self._plan.masked.size:
            return values
        rows = np.tile(cells.vi, len(cells.ai))
        cols = np.repeat(cells.ai, len(cells.vi))
        out = np.array(values, copy=True)
        out[_hit(self._plan, rows, cols)] = np.nan
        return out


class _MaskedTier(TopHitTier):
    """One tier of the source, without the masked cells' rows."""

    def __init__(self, tier: TopHitTier, plan: LegacyPlan) -> None:
        self.__dict__.update(tier.__dict__)
        self._plan = plan

    def read(self, analysis_index: int | None = None, *, require: Any = ()) -> TierRows:
        rows = super().read(analysis_index, require=require)
        keep = ~_hit(self._plan, rows.variant_index, rows.analysis_index)
        if keep.all():
            return rows
        def kept(column: np.ndarray | None) -> np.ndarray | None:
            return None if column is None else column[keep]

        return TierRows(
            rows.variant_index[keep], rows.analysis_index[keep], rows.z[keep],
            kept(rows.se), kept(rows.eaf), kept(rows.imputed),
        )


class _MaskedTiers(TopHitTiers):
    """The source's tiers, each without the masked cells' rows."""

    def __init__(self, tiers: TopHitTiers, plan: LegacyPlan) -> None:
        self.__dict__.update(tiers.__dict__)
        self._plan = plan

    def tier(self, threshold: float) -> TopHitTier | None:
        tier = super().tier(threshold)
        return None if tier is None else _MaskedTier(tier, self._plan)


class _ExpectedAxis(VariantAxis):
    """The source's Variant Axis, naming each row's `source_alid` as the plan does."""

    def __init__(self, axis: VariantAxis, plan: LegacyPlan) -> None:
        self.__dict__.update(axis.__dict__)
        self._plan = plan

    def source_alids_by_indices(self, variant_index: np.ndarray) -> np.ndarray:
        vi = np.asarray(variant_index, dtype=np.int64).reshape(-1)
        carried = super().source_alids_by_indices(vi)
        records = self.by_indices(vi.tolist())
        out = np.full(vi.size, None, dtype=object)
        for slot, row in enumerate(vi.tolist()):
            record = records.get(int(row))
            if record is not None:
                out[slot] = self._plan.source_alid(row, record.alid, carried[slot])
        return out

    def rows_by_indices(self, rows: np.ndarray) -> dict[int, dict[str, Any]]:
        table = super().rows_by_indices(rows)
        for row, values in table.items():
            values["source_alid"] = self._plan.source_alid(
                row, values["alid"], values["source_alid"]
            )
        return table


def _expected_analyses(path: Path, plan: LegacyPlan, tiers: TopHitTiers) -> AnalysesIndex:
    """`analyses.tsv` of the source, with each Analysis's count of masked cells and its
    Top-Hit Counts less the masked cells its tiers held."""
    analyses = AnalysesIndex(path)
    dropped = _dropped_hits(plan, tiers)
    for index, row in analyses.all().items():
        row[N_MISSING_SE_OUT_OF_RANGE] = str(int(plan.out_of_range[index]))
        for column, counts in dropped.items():
            if counts[index]:
                row[column] = str(int(row.get(column) or 0) - int(counts[index]))
    return analyses


def _dropped_hits(plan: LegacyPlan, tiers: TopHitTiers) -> dict[str, np.ndarray]:
    """Per `n_hits_*` column, how many masked cells each Analysis's tier held."""
    out = {}
    for threshold, column in zip(TOP_HIT_THRESHOLDS, TOP_HIT_COUNT_COLUMNS, strict=True):
        tier = tiers.tier(threshold)
        if tier is None:
            continue
        rows = tier.read()
        hit = _hit(plan, rows.variant_index, rows.analysis_index)
        out[column] = np.bincount(
            np.asarray(rows.analysis_index, dtype=np.int64)[hit], minlength=plan.n_analyses
        )
    return out


class _ExpectedStoreQuery(StoreQuery):
    """The source's Dense query facade, answering as the converted release must.

    It, and the readers it holds, override private methods (`_decode`,
    `_indexed_analysis`) and replace `_grid` and `_top_hits`: the deliberate changes are
    made where every question's cells are decoded, so no question can be answered around
    them. That cannot hide a read-path defect: a question that reached the source by a
    path the view does not cover would answer unchanged, and the proof would refuse the
    conversion for the difference (#492 fix round 2, the standards review's finding).
    """

    def __init__(self, path: Path, plan: LegacyPlan) -> None:
        store = open_store(path)
        tiers = TopHitTiers(store.arrays())
        axis = store.variant_axis(aliases=True)
        if not plan.lifted:
            assert isinstance(axis, VariantAxis), "a Dense release's own axis"
            axis = _ExpectedAxis(axis, plan)
        super().__init__(
            store, variant_axis=axis, analyses=_expected_analyses(path, plan, tiers)
        )
        self._owns_axis = True
        self._grid = _MaskedGrid(self._grid, plan)
        self._top_hits = _MaskedTiers(self._top_hits, plan)
        self._plan = plan

    def _indexed_analysis(
        self, analysis_id: str, subset_name: str, *, observed_only: bool
    ) -> Associations:
        found = super()._indexed_analysis(analysis_id, subset_name, observed_only=observed_only)
        keep = ~_hit(self._plan, found.variant_index, found.analysis_index)
        return found if keep.all() else found.take(keep)


def open_expected(path: Path, plan: LegacyPlan) -> StoreQuery:
    """The source at `path`, read through its 0.2 reader as the converted release must
    answer (module docstring)."""
    return _ExpectedStoreQuery(Path(path), plan)


__all__ = ["open_expected"]
