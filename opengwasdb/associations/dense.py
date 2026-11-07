"""The Dense grid adapter: a Dense Store Release, or a Hybrid's Dense Component."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, NamedTuple

import numpy as np

from opengwasdb.associations.cells import Cells, require_within, window_region
from opengwasdb.associations.record import (
    Associations,
    Inspection,
    InvalidAssociations,
    State,
    Status,
    classify,
    mask_required,
    non_canonical,
    require_mask,
)
from opengwasdb.encoding import DenseEafPlane, DenseSePlane, DenseZPlane, StoreEncoding


class _Grid(NamedTuple):
    """One region decoded as a grid: `rows` × `cols`, or elementwise for pairs."""

    rows: np.ndarray
    cols: np.ndarray
    z: np.ndarray
    se: np.ndarray
    eaf: np.ndarray
    flags: np.ndarray | None
    #: Whether rows are emitted Analysis-major (columns outer).
    analysis_major: bool
    #: Whether `rows[i]` pairs with `cols[i]` rather than spanning a grid.
    elementwise: bool


def _contiguous(rows: np.ndarray) -> bool:
    """Whether `rows` is one ascending run, which reads as a band, not a block."""
    return bool(len(rows)) and int(rows[-1]) - int(rows[0]) + 1 == len(rows) and bool(
        np.all(np.diff(rows) == 1)
    )


class DenseGridReader:
    """Reads associations from a `n_variants × n_analyses` grid.

    A grid stores no association at a missing cell, so `read` returns the
    region's present cells (ADR 0020) -- except for `pairs`, which answers every
    cell it names. Each region is one read of each of Z, the imputed mask, EAF
    and SE, with the EAF read handed to SE decoding (#253).

    Z is read before the frequency plane. The order is measured, not cosmetic:
    reading EAF first made a one-variant PheWAS 23% slower on OGS-00016
    (44.2 ms against 35.9 ms, the same chunk reads in another order).
    """

    carries_n = False

    def __init__(
        self,
        group: Any,
        encoding: StoreEncoding,
        *,
        source: str,
        requires_mask: bool | None = None,
    ) -> None:
        self._group = group
        self._encoding = encoding
        self._source = source
        # Checked before any plane opens, so the refusal names the mask.
        self.has_imputed = "imputed" in group
        required = mask_required(requires_mask, group.attrs.get("completion_state"), source)
        require_mask(required, self.has_imputed, source)
        self._z = DenseZPlane.open(group, encoding)
        self._se = DenseSePlane.open(group, encoding)
        # Opened on first use: a top-hit index carrying its decoded EAF must not
        # touch the much larger frequency plane.
        self._eaf_plane: DenseEafPlane | None = None

    @classmethod
    def open(
        cls,
        group: Any,
        encoding: StoreEncoding,
        *,
        source: str = "Dense grid",
        requires_mask: bool | None = None,
    ) -> DenseGridReader:
        """A Reference-Completed grid without its imputed mask is refused rather than
        read as all-observed (spec §15). Whether the grid is completed is the
        group's own `completion_state` record unless the caller says
        (`requires_mask`); a grid that records none is refused."""
        return cls(group, encoding, source=source, requires_mask=requires_mask)

    @property
    def n_variants(self) -> int:
        return int(self._z.shape[0])

    @property
    def n_analyses(self) -> int:
        return self._z.n_analyses

    @property
    def _eaf(self) -> DenseEafPlane:
        if self._eaf_plane is None:
            self._eaf_plane = DenseEafPlane.open(self._group, self._encoding)
        return self._eaf_plane

    @property
    def scan_window(self) -> int:
        """Cells one window holds unless a scan says: a shard's rows of the Z plane
        (a chunk's, unsharded), so each window reads one shard of each plane."""
        array = self._z.array
        return int((getattr(array, "shards", None) or array.chunks)[0])

    # ── the interface ───────────────────────────────────────────────────────

    def windows(
        self, cells: Cells | None = None, *, max_cells: int | None = None
    ) -> Iterator[Cells]:
        """An `analysis` or `analyses` region -- or, with none, the whole grid -- as
        consecutive `cross` regions of at most `max_cells` cells each.

        For a scan that must not hold a whole Analysis at once: each cell of the
        region comes once, in the region's order, so the windows' reads joined are
        the region's read. A window is a run of one Analysis's variants, since a
        grid region is variants × Analyses and cannot hold the end of one Analysis
        and the start of the next. `max_cells` defaults to `scan_window`. The
        region is checked when this is called, before any window is cut
        (`window_region`).
        """
        region, step = window_region(
            cells,
            max_cells,
            default=self.scan_window,
            n_analyses=self.n_analyses,
            source=self._source,
        )
        return self._windows(region, step)

    def _windows(self, region: Cells, step: int) -> Iterator[Cells]:
        assert region.ai is not None
        n = self.n_variants
        for analysis in region.ai:
            for lo in range(0, n, step):
                rows = np.arange(lo, min(lo + step, n), dtype=np.int64)
                yield Cells.cross(rows, [int(analysis)])

    def z(self, cells: Cells) -> np.ndarray:
        """Decoded Z of a `cross` region -- a window -- for a scan that thresholds Z
        alone: every cell, a missing one as NaN, in the region's order.

        A non-canonical marker is refused here, as every read refuses one (spec §15):
        a NaN clears no cutoff, so a scan that compared it would drop the cell.
        """
        if cells.kind != "cross" or cells.vi is None or cells.ai is None:
            raise TypeError(f"Z alone is read for a window, not a {cells.kind!r} region")
        require_within(
            cells, self._source, n_analyses=self.n_analyses, n_variants=self.n_variants
        )
        rows, cols = cells.vi, cells.ai
        if len(rows) == 0 or len(cols) == 0:
            return np.empty(0, dtype=np.float32)
        z: np.ndarray = np.asarray(self._z.block(rows.tolist(), cols.tolist())).T.ravel()
        odd = non_canonical(z)
        if odd.any():
            raise InvalidAssociations(
                self._source,
                np.tile(rows, len(cols))[odd],
                np.repeat(cols, len(rows))[odd],
                np.full(int(odd.sum()), State.NON_CANONICAL, dtype=np.int8),
            )
        return z

    def read(self, cells: Cells, *, observed_only: bool = False) -> Associations:
        """The region's associations; raises `InvalidAssociations` on a §15-invalid cell."""
        grid = self._decode(cells)
        state = classify(grid.z, grid.se, grid.flags)
        if np.any(state > State.IMPUTED):
            bad = state > State.IMPUTED
            rows, cols = self._coordinates(grid, bad)
            raise InvalidAssociations(self._source, rows, cols, self._values(grid, state, bad))
        keep = np.ones(state.shape, dtype=bool) if grid.elementwise else state != Status.MISSING
        if observed_only:
            keep &= state != Status.IMPUTED
        rows, cols = self._coordinates(grid, keep)
        return Associations(
            rows,
            cols,
            self._values(grid, grid.z, keep),
            self._values(grid, grid.se, keep),
            self._values(grid, grid.eaf, keep),
            self._values(grid, state, keep),
        )

    def inspect(self, cells: Cells) -> Inspection:
        """Every cell of the region, decoded and classified; never raises on one."""
        grid = self._decode(cells)
        every = np.ones(np.shape(grid.z), dtype=bool)
        rows, cols = self._coordinates(grid, every)
        return Inspection.of(
            rows,
            cols,
            self._values(grid, grid.z, every),
            self._values(grid, grid.se, every),
            self._values(grid, grid.eaf, every),
            None if grid.flags is None else self._values(grid, grid.flags, every),
        )

    # ── decoding ────────────────────────────────────────────────────────────

    def _decode(self, cells: Cells) -> _Grid:
        require_within(
            cells, self._source, n_analyses=self.n_analyses, n_variants=self.n_variants
        )
        decode = {
            "analysis": self._analysis,
            "analyses": self._analyses,
            "variants": self._variants,
            "cross": self._cross,
            "pairs": self._pairs,
        }.get(cells.kind)
        if decode is None:
            raise TypeError(f"a Dense grid cannot read a {cells.kind!r} region")
        return decode(cells)

    def _analysis(self, cells: Cells) -> _Grid:
        """One column, every variant, in variant order."""
        col = cells.analysis_index
        z = self._z.column(col)
        eaf = self._eaf.read_column(col, want_imputed=True)
        se = self._se.column(col, eaf=eaf.values)
        return _Grid(
            np.arange(len(z), dtype=np.int64),
            np.array([col], dtype=np.int64),
            z[:, None],
            se[:, None],
            np.asarray(eaf.values)[:, None],
            None if eaf.imputed is None else np.asarray(eaf.imputed)[:, None],
            analysis_major=True,
            elementwise=False,
        )

    def _analyses(self, cells: Cells) -> _Grid:
        """Every variant of each Analysis named, Analysis-major, in one read of each
        plane: an `analysis` region apiece would read every chunk the Analyses
        share once per Analysis."""
        assert cells.ai is not None
        return self._cross(Cells.cross(np.arange(self.n_variants), cells.ai))

    def _variants(self, cells: Cells) -> _Grid:
        """Whole rows, variant-major; one row reads as a row, several as a band or block."""
        rows = cells.vi
        assert rows is not None
        if len(rows) == 1:
            row = int(rows[0])
            z = self._z.row(row)[None, :]
            eaf = self._eaf.read_row(row, want_imputed=True)
            se = self._se.row(row, eaf=eaf.values)[None, :]
            values = np.asarray(eaf.values)[None, :]
            flags = None if eaf.imputed is None else np.asarray(eaf.imputed)[None, :]
        else:
            z = self._z.rows(rows)
            block = self._eaf.read_rows(rows, want_imputed=True)
            se = (
                self._se.band(int(rows[0]), int(rows[-1]) + 1, eaf=block.values)
                if _contiguous(rows)
                else self._se.rows(rows, eaf=block.values)
            )
            values, flags = np.asarray(block.values), block.imputed
        cols = np.arange(self.n_analyses, dtype=np.int64)
        return _Grid(rows, cols, z, se, values, flags, analysis_major=False, elementwise=False)

    def _cross(self, cells: Cells) -> _Grid:
        """`rows × cols` by orthogonal reads, emitted Analysis-major."""
        rows, cols = cells.vi, cells.ai
        assert rows is not None and cols is not None
        if len(rows) == 0 or len(cols) == 0:
            empty = np.empty((len(rows), len(cols)), dtype=np.float32)
            return _Grid(rows, cols, empty, empty, empty, None, True, False)
        z = self._z.block(rows.tolist(), cols.tolist())
        block = self._eaf.read_block(rows, cols, want_imputed=True)
        se = self._se.block(rows.tolist(), cols.tolist(), eaf=block.values)
        return _Grid(rows, cols, z, se, np.asarray(block.values), block.imputed, True, False)

    def _pairs(self, cells: Cells) -> _Grid:
        """The cells `(rows[i], cols[i])`, elementwise."""
        rows, cols = cells.vi, cells.ai
        assert rows is not None and cols is not None
        z = self._z.points(rows, cols)
        points = self._eaf.read_points(rows, cols, want_imputed=True)
        se = self._se.points(rows, cols, eaf=points.values)
        return _Grid(rows, cols, z, se, np.asarray(points.values), points.imputed, False, True)

    @staticmethod
    def _coordinates(grid: _Grid, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The variant and Analysis index of each selected cell, in emission order."""
        if grid.elementwise:
            return grid.rows[mask], grid.cols[mask]
        if grid.analysis_major:
            c, r = np.nonzero(mask.T)
        else:
            r, c = np.nonzero(mask)
        return grid.rows[r], grid.cols[c]

    @staticmethod
    def _values(grid: _Grid, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """`values` at the selected cells, in the order `_coordinates` gives."""
        values = np.asarray(values)
        if grid.elementwise or not grid.analysis_major:
            out: np.ndarray = values[mask]
            return out
        transposed: np.ndarray = values.T[mask.T]
        return transposed
