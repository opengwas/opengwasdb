"""The streamed reads value rules fold over: a component's associations, a block at a time.

`ASSOCIATIONS` reads a Dense grid as row bands. Every plane is read once per block,
as stored codes, and decoded from those codes: SE from the block's own EAF read, EAF with
the panel's frequency on imputed cells. Every rule that judges values folds over
the same blocks, so a component's planes are read once however many rules there
are (issue #254 bounds each block's decoded size).

A `Block` is the seam: `dense_bands` makes one from a grid band, and `Block.of`
from arrays a test holds.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import cached_property, partial

import numpy as np

from opengwasdb.associations import Tally, non_canonical
from opengwasdb.encoding.codec import (
    EafExceptionTable,
    SeExceptionTable,
    StoreCodec,
    ZOverflowTable,
)
from opengwasdb.encoding.planes import DenseEafPlane, DenseSePlane, DenseZPlane
from opengwasdb.model.enums import CompletionState
from opengwasdb.store.arrays import inner_chunk_of
from opengwasdb.validation.rule import Kind, Scan, Target

# Dense validation decodes several raw, float32/float64 and boolean views of a
# cell at once. OGS-00009/10 profiling measured at most 59 bytes per cell in
# the Dense pass and 47 in the Top-Hit pass; the 96-byte allowance leaves 63%
# headroom inside this one-GiB decoded working-set budget. The row cap preserves
# the old call size for a narrow store, while the byte budget makes a wide
# Store's band shrink instead of scaling its memory with the number of Analyses
# (issue #254).
_BAND_ROWS = 250_000
_WORKING_BYTES = 1024**3
BYTES_PER_CELL = 96


def band_rows(n_analyses: int, row_chunk: int) -> int:
    """Rows whose decoded working set fits the Dense validation byte budget.

    Whole inner chunks avoid decoding a physical chunk into two consecutive
    bands. A single row is the irreducible minimum when an exceptionally wide
    store exceeds the budget by itself.
    """
    if n_analyses <= 0:
        return _BAND_ROWS
    rows = min(_BAND_ROWS, max(1, _WORKING_BYTES // (BYTES_PER_CELL * n_analyses)))
    if row_chunk > 0 and rows >= row_chunk:
        rows = max(row_chunk, rows // row_chunk * row_chunk)
    return rows


def band_limits(n_variants: int, n_analyses: int, row_chunk: int) -> Iterator[tuple[int, int]]:
    """Every Dense row band, exhaustively, under the decoded-byte budget."""
    rows = band_rows(n_analyses, row_chunk)
    for r0 in range(0, n_variants, rows):
        yield r0, min(r0 + rows, n_variants)


def per_variant(target: Target, name: str) -> np.ndarray:
    """A per-variant array beside the planes, read once for every rule and band that needs it.

    `eaf_baseline` and `eaf_reference` hold one value per variant (tens of MB on the
    largest Store), so holding one whole costs less than reading it once to check
    it and again, a band at a time, to decode the plane.
    """
    return target.remember(name, lambda: np.asarray(target.group[name][:], dtype=np.float32))


class Undecodable(Exception):
    """A block whose planes the declared plan cannot decode; the message says which."""


@dataclass(frozen=True, eq=False)
class Block:
    """Cells of one component, every plane read once, as codes and decoded.

    A Dense band is `rows x n_analyses`; `offset` is the flat position of its
    first cell, the key the exact-value side tables use. `eaf` is None where the
    component holds no frequency; `eaf_undecodable` says the plane could not be
    decoded here, which the exception-table rule explains.
    """

    offset: int
    n_analyses: int
    z_codes: np.ndarray
    se_codes: np.ndarray
    z_missing: np.ndarray
    se: np.ndarray
    eaf_codes: np.ndarray | None = None
    eaf: np.ndarray | None = None
    eaf_undecodable: bool = False
    imputed: np.ndarray | None = None
    on_panel: np.ndarray | None = None
    #: Decodes `z_codes`; a block built from values holds them as their own codes.
    z_decoder: Callable[[np.ndarray], np.ndarray] | None = field(default=None, repr=False)
    rows: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", int(self.z_codes.shape[0]))

    @classmethod
    def of(
        cls,
        z: np.ndarray,
        se: np.ndarray,
        *,
        offset: int = 0,
        imputed: np.ndarray | None = None,
        eaf: np.ndarray | None = None,
        on_panel: np.ndarray | None = None,
    ) -> Block:
        """A band from decoded arrays, its codes the values themselves: for tests."""
        z = np.asarray(z, dtype=np.float32)
        se = np.asarray(se, dtype=np.float32)
        return cls(
            offset=offset,
            n_analyses=int(z.shape[1]),
            z_codes=z,
            se_codes=se,
            z_missing=np.isnan(z),
            se=se,
            eaf_codes=None if eaf is None else np.asarray(eaf, dtype=np.float32),
            eaf=None if eaf is None else np.asarray(eaf, dtype=np.float32),
            imputed=None if imputed is None else np.asarray(imputed),
            on_panel=None if on_panel is None else np.asarray(on_panel),
        )

    @cached_property
    def z(self) -> np.ndarray:
        """The block's decoded Z, from the codes it holds, decoded on first use."""
        if self.z_decoder is None:
            return np.asarray(self.z_codes, dtype=np.float32)
        return self.z_decoder(self.z_codes)

    @cached_property
    def tally(self) -> Tally:
        """The block's cells under the presence rule (spec §15, ADR 0070), counted once."""
        se_missing = np.isnan(self.se)
        odd = non_canonical(self.se, se_missing)
        if np.issubdtype(self.z_codes.dtype, np.floating):
            odd |= non_canonical(self.z_codes)
        return Tally.of(self.z_missing, se_missing, self.imputed, odd)

    def per_analysis(self, mask: np.ndarray) -> np.ndarray:
        """How many cells `mask` holds in each Analysis."""
        counts: np.ndarray = np.asarray(mask).sum(axis=0, dtype=np.int64)
        return counts

    def locate(self, cells: CellIndex) -> tuple[np.ndarray, np.ndarray]:
        """The cells of `cells` this block holds: their indices, and where they sit here.

        Returns the indices into `cells` of the cells this block holds and, for each,
        its index into the block's flattened arrays. Found from the cells' row order,
        so a band's share costs what the band holds, not the whole index.
        """
        r0 = self.offset // self.n_analyses
        order, sorted_rows = cells.by_row
        lo, hi = np.searchsorted(sorted_rows, (r0, r0 + self.rows))
        found = order[lo:hi]
        flat = (cells.rows[found].astype(np.int64) - r0) * self.n_analyses + cells.cols[found]
        return found, flat

    def positions(self, mask: np.ndarray) -> np.ndarray:
        """The flat positions of the cells `mask` holds."""
        found: np.ndarray = np.flatnonzero(np.asarray(mask)).astype(np.int64) + self.offset
        return found


class CellIndex:
    """Cells named by `(variant, Analysis)`, arranged once to be found block by block."""

    def __init__(self, rows: np.ndarray, cols: np.ndarray) -> None:
        self.rows = rows
        self.cols = cols

    @cached_property
    def by_row(self) -> tuple[np.ndarray, np.ndarray]:
        """The cells' order by variant row, and their rows in that order."""
        order = np.argsort(self.rows, kind="stable").astype(np.uint32)
        return order, self.rows[order]


class _RowWindow:
    """One band of a plane already in memory, read in the plane's own row numbers.

    The plane decoders read `array[r0:r1]` and the array's shape; this answers
    both from the band the scan holds, so a decode reads nothing a second time.
    """

    def __init__(self, band: np.ndarray, r0: int, shape: tuple[int, ...]) -> None:
        self._band = band
        self._r0 = r0
        self.shape = shape

    def __getitem__(self, rows: slice) -> np.ndarray:
        return self._band[rows.start - self._r0 : rows.stop - self._r0]


class _DenseGrid:
    """What every band of one Dense grid shares: arrays, codecs and per-variant vectors."""

    def __init__(
        self, target: Target, on_panel: np.ndarray | None, completed: bool
    ) -> None:
        group = target.group
        encoding = target.encoding
        self.group, self.encoding, self.on_panel = group, encoding, on_panel
        # The mask is read for the panel substitution whenever it is there, but only a
        # completed grid's mask flags imputed cells.
        self.completed = completed
        self.z, self.se_array = group["z"], group["se"]
        self.eaf_array = group["eaf"] if "eaf" in group else None
        self.imputed = group["imputed"] if "imputed" in group else None
        self.baseline = per_variant(target, "eaf_baseline") if "eaf_baseline" in group else None
        self.reference = (
            per_variant(target, "eaf_reference") if "eaf_reference" in group else None
        )
        self.decodes_eaf = self.eaf_array is not None or bool(encoding.eaf.reference)
        self.eaf_codec = StoreCodec(encoding, eaf_exceptions=EafExceptionTable.open(group))
        self.se_codec = StoreCodec(encoding, se_exceptions=SeExceptionTable.open(group))
        self.z_codec = StoreCodec(encoding, z_overflow=ZOverflowTable.open(group))
        self.coefficients = (
            np.asarray(group["se_coefficients"][:], dtype=np.float32)
            if encoding.se.is_residual
            else None
        )
        self.missing = StoreCodec(encoding).missing_mask
        self.shape = (int(self.z.shape[0]), int(self.z.shape[1]))

    def band(self, r0: int, r1: int) -> Block:
        z_codes = np.asarray(self.z[r0:r1])
        se_codes = np.asarray(self.se_array[r0:r1])
        eaf_codes = None if self.eaf_array is None else np.asarray(self.eaf_array[r0:r1])
        imputed = None if self.imputed is None else np.asarray(self.imputed[r0:r1])
        eaf, failure = self._eaf(eaf_codes, imputed, r0, r1)
        return Block(
            offset=r0 * self.shape[1],
            n_analyses=self.shape[1],
            z_codes=z_codes,
            se_codes=se_codes,
            z_missing=self.missing(z_codes),
            se=_dense_band_se(self, se_codes, eaf, failure, r0, r1),
            eaf_codes=eaf_codes,
            eaf=eaf,
            eaf_undecodable=failure is not None,
            imputed=imputed if self.completed else None,
            on_panel=None if self.on_panel is None else self.on_panel[r0:r1],
            z_decoder=partial(self._z, r0, r1),
        )

    def _z(self, r0: int, r1: int, codes: np.ndarray) -> np.ndarray:
        """One band's Z, decoded from the codes the band holds."""
        return DenseZPlane(_RowWindow(codes, r0, self.shape), self.z_codec).band(r0, r1)

    def _eaf(
        self, codes: np.ndarray | None, imputed: np.ndarray | None, r0: int, r1: int
    ) -> tuple[np.ndarray | None, ValueError | None]:
        """The band's decoded frequencies, from the codes it holds, or why they cannot be.

        A decode that cannot resolve a cell is not reported from here: the
        exception-table rule says exactly which cell and why.
        """
        if not self.decodes_eaf:
            return None, None
        plane = DenseEafPlane(
            None if codes is None else _RowWindow(codes, r0, self.shape),
            self.eaf_codec,
            baseline=self.baseline,
            reference=self.reference,
            imputed=None if imputed is None else _RowWindow(imputed, r0, self.shape),
            group=self.group,
        )
        try:
            return plane.band(r0, r1), None
        except ValueError as exc:
            return None, exc


def _dense_band_se(
    grid: _DenseGrid,
    codes: np.ndarray,
    eaf: np.ndarray | None,
    eaf_failure: ValueError | None,
    r0: int,
    r1: int,
) -> np.ndarray:
    """One band's SE, decoded from its codes against the band's own EAF read.

    A residual plane is decoded against the frequency, so when the band's
    frequency cannot be decoded neither can its SE, for the same reason, and the
    read stops there.
    """
    if grid.encoding.se.is_residual and eaf_failure is not None:
        raise Undecodable(f"cannot decode se: {eaf_failure}") from eaf_failure
    plane = DenseSePlane(
        _RowWindow(codes, r0, grid.shape), grid.se_codec, None, grid.coefficients, grid.group
    )
    try:
        return plane.band(r0, r1, eaf=eaf)
    except ValueError as exc:
        raise Undecodable(f"cannot decode se: {exc}") from exc


def dense_bands(target: Target) -> Iterator[Block]:
    """A Dense grid's rows in byte-bounded bands aligned to its physical read unit."""
    group = target.group
    on_panel = None
    if "on_panel" in group:
        on_panel = target.remember("on_panel", lambda: np.asarray(group["on_panel"][:]))
    completed = target.store.manifest.completion_state is CompletionState.REFERENCE_COMPLETED
    grid = _DenseGrid(target, on_panel, completed)
    n_variants, n_analyses = grid.shape
    for r0, r1 in band_limits(n_variants, n_analyses, inner_chunk_of(grid.z)[0]):
        yield grid.band(r0, r1)


def _blocks(target: Target) -> Iterator[Block]:
    if target.kind is Kind.GRID:
        yield from dense_bands(target)


def _read_failure(_target: Target, exc: Exception) -> str:
    if isinstance(exc, Undecodable):
        return str(exc)
    return f"validation failed: {exc}"


ASSOCIATIONS: Scan[Block] = Scan("associations", _blocks, _read_failure)
