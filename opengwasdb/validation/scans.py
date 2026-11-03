"""The streamed reads value rules fold over: a component's associations, a block at a time.

`ASSOCIATIONS` reads a Dense grid as row bands and an association sequence -- a
Ragged release's, or a Hybrid's Overflow -- as windows of consecutive cells. Every
plane is read once per block, as stored codes, and decoded from those codes: SE
from the block's own EAF read, EAF with the panel's frequency on imputed cells.
Every rule that judges values folds over the same blocks, so a component's planes
are read once however many rules there are (issue #254 bounds each block's
decoded size).

SE and Z are decoded on first use, so a plane that cannot be decoded reaches only
the rules that read it (`Unreadable`).

A `Block` is the seam: `dense_bands` makes one from a grid band, `sequence_windows`
from a run of a sequence, and `Block.of` from arrays a test holds.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import cached_property, partial
from typing import Any

import numpy as np

from opengwasdb.associations import Tally, non_canonical
from opengwasdb.encoding.codec import (
    EAF_BASELINE,
    EAF_REFERENCE,
    EafExceptionTable,
    SeExceptionTable,
    StoreCodec,
    ZOverflowTable,
    positions_flat,
)
from opengwasdb.encoding.plan import EafBaselineError
from opengwasdb.encoding.planes import (
    SE_COEFFICIENTS,
    DenseEafPlane,
    DenseSePlane,
    DenseZPlane,
    RaggedEafPlane,
)
from opengwasdb.model.enums import CompletionState, PrimaryStorageLayout
from opengwasdb.store.arrays import array_length, inner_chunk_of
from opengwasdb.validation.rule import Kind, Scan, Target, Unreadable

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


class Undecodable(Unreadable):
    """A block whose planes the declared plan cannot decode; the message says which."""


@dataclass(frozen=True, eq=False)
class Block:
    """Cells of one component, every plane read once, as codes and decoded.

    A Dense band is `rows x n_analyses`; a sequence window is a run of cells,
    with each cell's `analysis` and `variant`, and the cell before it in
    `previous`. `offset` is the flat position of the first cell, the key the
    exact-value side tables use. `eaf` is None where the component holds no
    frequency; `eaf_undecodable` says the plane could not be decoded here, which
    the exception-table rule explains. `se`, `z` and `n_codes` are read or decoded
    on first use.
    """

    offset: int
    n_analyses: int
    z_codes: np.ndarray
    se_codes: np.ndarray
    z_missing: np.ndarray
    eaf_codes: np.ndarray | None = None
    eaf: np.ndarray | None = None
    eaf_undecodable: bool = False
    imputed: np.ndarray | None = None
    on_panel: np.ndarray | None = None
    analysis: np.ndarray | None = None
    variant: np.ndarray | None = None
    #: The `(analysis, variant)` of the cell before a window, when there is one.
    previous: tuple[int, int] | None = None
    #: Decode `z_codes` and `se_codes`; a block built from values holds them as codes.
    z_decoder: Callable[[np.ndarray], np.ndarray] | None = field(default=None, repr=False)
    se_decoder: Callable[[np.ndarray], np.ndarray] | None = field(default=None, repr=False)
    #: Reads the block's `n` codes, when the component holds a sample-size plane.
    read_n: Callable[[], np.ndarray] | None = field(default=None, repr=False)
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
    def se(self) -> np.ndarray:
        """The block's decoded SE, decoded on first use; `Undecodable` when it cannot be."""
        if self.se_decoder is None:
            return np.asarray(self.se_codes, dtype=np.float32)
        return self.se_decoder(self.se_codes)

    @cached_property
    def n_codes(self) -> np.ndarray | None:
        """The block's stored sample sizes, read on first use; None without a plane."""
        return None if self.read_n is None else self.read_n()

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
        if self.analysis is not None:
            held: np.ndarray = np.bincount(
                self.analysis[np.asarray(mask, dtype=bool)], minlength=self.n_analyses
            ).astype(np.int64)
            return held
        counts: np.ndarray = np.asarray(mask).sum(axis=0, dtype=np.int64)
        return counts

    def locate(self, cells: CellIndex) -> tuple[np.ndarray, np.ndarray]:
        """The cells of `cells` this block holds: their indices, and where they sit here.

        Returns the indices into `cells` of the cells this block holds and, for each,
        its index into the block's flattened arrays. Found from the cells' row order,
        so a band's share costs what the band holds, not the whole index.
        """
        if self.analysis is not None:
            return self._locate_in_sequence(cells)
        r0 = self.offset // self.n_analyses
        order, sorted_rows = cells.by_row
        lo, hi = np.searchsorted(sorted_rows, (r0, r0 + self.rows))
        found = order[lo:hi]
        flat = (cells.rows[found].astype(np.int64) - r0) * self.n_analyses + cells.cols[found]
        return found, flat

    @cached_property
    def _keys(self) -> tuple[np.ndarray | None, np.ndarray]:
        """The window's `(analysis, variant)` keys in ascending order, and how to undo it.

        A sound sequence holds them ascending already; one whose segments are out of
        order (which `csr.segment_order` reports) is sorted here so a lookup is still
        exact.
        """
        assert self.analysis is not None and self.variant is not None
        keys = _cell_keys(self.analysis, self.variant)
        if len(keys) < 2 or bool(np.all(keys[1:] > keys[:-1])):
            return None, keys
        order = np.argsort(keys, kind="stable")
        return order, keys[order]

    def _locate_in_sequence(self, cells: CellIndex) -> tuple[np.ndarray, np.ndarray]:
        order, keys = self._keys
        if len(keys) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        wanted = cells.keys
        lo, hi = np.searchsorted(wanted, (keys[0], keys[-1]))
        hi = int(np.searchsorted(wanted, keys[-1], side="right"))
        candidates = np.arange(lo, hi, dtype=np.int64)
        slots = np.minimum(np.searchsorted(keys, wanted[candidates]), len(keys) - 1)
        matched = keys[slots] == wanted[candidates]
        local = slots[matched] if order is None else order[slots[matched]]
        return candidates[matched], local.astype(np.int64)

    def positions(self, mask: np.ndarray) -> np.ndarray:
        """The flat positions of the cells `mask` holds."""
        found: np.ndarray = np.flatnonzero(np.asarray(mask)).astype(np.int64) + self.offset
        return found


def _cell_keys(analysis: np.ndarray, variant: np.ndarray) -> np.ndarray:
    """One sortable key per cell: Analysis-major, then variant, as a sequence is held."""
    keys: np.ndarray = (analysis.astype(np.uint64) << np.uint64(32)) | variant.astype(np.uint64)
    return keys


class CellIndex:
    """Cells named by `(variant, Analysis)`, arranged once to be found block by block."""

    def __init__(self, rows: np.ndarray, cols: np.ndarray) -> None:
        self.rows = rows
        self.cols = cols

    @cached_property
    def keys(self) -> np.ndarray:
        """The cells' `(Analysis, variant)` keys, ascending for a sound tier."""
        return _cell_keys(self.cols, self.rows)

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
            eaf_codes=eaf_codes,
            eaf=eaf,
            eaf_undecodable=failure is not None,
            imputed=imputed if self.completed else None,
            on_panel=None if self.on_panel is None else self.on_panel[r0:r1],
            z_decoder=partial(self._z, r0, r1),
            se_decoder=partial(_dense_band_se, self, eaf=eaf, eaf_failure=failure, r0=r0, r1=r1),
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
    *,
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


def completed(target: Target) -> bool:
    """Whether the component's mask flags imputed cells: a Reference-Completed one's does."""
    return target.store.manifest.completion_state is CompletionState.REFERENCE_COMPLETED


def dense_bands(target: Target) -> Iterator[Block]:
    """A Dense grid's rows in byte-bounded bands aligned to its physical read unit."""
    group = target.group
    on_panel = None
    if "on_panel" in group:
        on_panel = target.remember("on_panel", lambda: np.asarray(group["on_panel"][:]))
    grid = _DenseGrid(target, on_panel, completed(target))
    n_variants, n_analyses = grid.shape
    for r0, r1 in band_limits(n_variants, n_analyses, inner_chunk_of(grid.z)[0]):
        yield grid.band(r0, r1)


# --- an association sequence, a window of consecutive cells at a time ---


def release_rows(target: Target) -> int:
    """The rows of the Store Variant Table the target's cells address, read once.

    A Ragged release's own table; for a Hybrid, the table both components share.
    """
    release = target
    while release.parent is not None:
        release = release.parent
    return release.remember("n_variants_shared", lambda: release.variant_axis.n_variants)


def sequence_offsets(target: Target) -> np.ndarray:
    """A sequence's `offsets`, which bound each Analysis's segment, read once per run."""
    return target.remember(
        "offsets", lambda: np.asarray(target.group["offsets"][:], dtype=np.int64)
    )


def window_cells(cell_chunk: int) -> int:
    """Cells one sequence window holds: the Dense band's byte budget, in whole chunks."""
    cells = max(1, _WORKING_BYTES // BYTES_PER_CELL)
    if cell_chunk > 0 and cells >= cell_chunk:
        cells = cells // cell_chunk * cell_chunk
    return cells


def analyses_of(offsets: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """The Analysis holding each cell of `[lo, hi)`, from the segments' bounds."""
    if hi <= lo:
        return np.empty(0, dtype=np.int32)
    first = int(np.searchsorted(offsets, lo, side="right")) - 1
    last = int(np.searchsorted(offsets, hi - 1, side="right")) - 1
    held = np.diff(np.clip(offsets[first : last + 2], lo, hi))
    found: np.ndarray = np.repeat(np.arange(first, last + 1, dtype=np.int32), held)
    return found


class _Held:
    """A per-variant array already in memory, read the way a plane reads a Zarr array."""

    def __init__(self, values: np.ndarray) -> None:
        self._values = values

    def __getitem__(self, key: slice) -> np.ndarray:
        return self._values[key]

    @property
    def oindex(self) -> np.ndarray:
        return self._values


def _parallel(group: Any, name: str, n_cells: int) -> Any:
    """An optional array beside the sequence, when it has one entry per cell, else None.

    A length that disagrees with `offsets` is a rule's finding; read anyway, it
    would hand each cell another association's value.
    """
    if name not in group:
        return None
    array = group[name]
    return array if array_length(array) == n_cells else None


class _Sequence:
    """What every window of one association sequence shares: arrays, codecs, side vectors."""

    def __init__(self, target: Target) -> None:
        group = target.group
        encoding = target.encoding
        self.group, self.encoding = group, encoding
        self.offsets = sequence_offsets(target)
        self.n_cells = int(self.offsets[-1]) if len(self.offsets) else 0
        self.n_analyses = max(len(self.offsets) - 1, 0)
        self.shape = (self.n_cells,)
        self.completed = completed(target)
        self.z, self.se_array, self.variant_index = group["z"], group["se"], group["variant_index"]
        self.imputed = _parallel(group, "imputed", self.n_cells)
        self.n_array = _parallel(group, "n", self.n_cells) if encoding.n is not None else None
        self.decodes_eaf = "eaf" in group or bool(encoding.eaf.reference)
        self.eaf_array = group["eaf"] if "eaf" in group else None
        self.baseline = _held(target, EAF_BASELINE)
        self.reference = _held(target, EAF_REFERENCE)
        self.eaf_codec = StoreCodec(encoding, eaf_exceptions=EafExceptionTable.open(group))
        self.se_codec = StoreCodec(encoding, se_exceptions=SeExceptionTable.open(group))
        self.z_codec = StoreCodec(encoding, z_overflow=ZOverflowTable.open(group))
        self.coefficients = _coefficients(target)
        self.missing = StoreCodec(encoding).missing_mask
        overflow = target.layout is PrimaryStorageLayout.HYBRID
        self.se_label = "Ragged Overflow se" if overflow else "data.zarr/ragged/se"

    def window(self, lo: int, hi: int, previous: tuple[int, int] | None) -> Block:
        z_codes = np.asarray(self.z[lo:hi])
        variant = np.asarray(self.variant_index[lo:hi])
        imputed = None if self.imputed is None else np.asarray(self.imputed[lo:hi])
        eaf_codes = None if self.eaf_array is None else np.asarray(self.eaf_array[lo:hi])
        eaf, failure = self._eaf(eaf_codes, variant, imputed, lo, hi)
        analysis = analyses_of(self.offsets, lo, hi)
        return Block(
            offset=lo,
            n_analyses=self.n_analyses,
            z_codes=z_codes,
            se_codes=np.asarray(self.se_array[lo:hi]),
            z_missing=self.missing(z_codes),
            eaf_codes=eaf_codes,
            eaf=eaf,
            eaf_undecodable=failure is not None,
            imputed=imputed if self.completed else None,
            analysis=analysis,
            variant=variant,
            previous=previous,
            z_decoder=partial(self._z, lo),
            se_decoder=partial(
                _sequence_window_se, self, eaf=eaf, eaf_failure=failure, analysis=analysis, lo=lo
            ),
            read_n=None if self.n_array is None else partial(self._n, lo, hi),
        )

    def _z(self, lo: int, codes: np.ndarray) -> np.ndarray:
        return self.z_codec.decode_z(codes, positions=positions_flat(lo))

    def _n(self, lo: int, hi: int) -> np.ndarray:
        assert self.n_array is not None
        return np.asarray(self.n_array[lo:hi])

    def _eaf(
        self,
        codes: np.ndarray | None,
        variant: np.ndarray,
        imputed: np.ndarray | None,
        lo: int,
        hi: int,
    ) -> tuple[np.ndarray | None, Exception | None]:
        """The window's decoded frequencies, from the codes it holds, or why they cannot be.

        A mask whose length disagrees with `offsets` is not read, so a plan that
        substitutes the panel's frequency cannot be decoded either; the mask's own
        rule says why.
        """
        if not self.decodes_eaf:
            return None, None
        try:
            plane = RaggedEafPlane(
                None if codes is None else _RowWindow(codes, lo, self.shape),
                self.eaf_codec,
                _RowWindow(variant, lo, self.shape),
                baseline=self.baseline,
                reference=self.reference,
                imputed=None if imputed is None else _RowWindow(imputed, lo, self.shape),
                group=self.group,
            )
            return plane.slice(lo, hi), None
        except (ValueError, EafBaselineError) as exc:
            return None, exc


def _coefficients(target: Target) -> np.ndarray:
    """A residual SE plane's per-Analysis fit, held for every window."""
    if not target.encoding.se.is_residual:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(target.group[SE_COEFFICIENTS][:], dtype=np.float32)


def _held(target: Target, name: str) -> _Held | None:
    return _Held(per_variant(target, name)) if name in target.group else None


def _sequence_window_se(
    sequence: _Sequence,
    codes: np.ndarray,
    *,
    eaf: np.ndarray | None,
    eaf_failure: Exception | None,
    analysis: np.ndarray,
    lo: int,
) -> np.ndarray:
    """One window's SE, decoded from its codes against the window's own EAF read.

    A residual plane is predicted from the frequency, so when the window's
    frequency cannot be decoded neither can its SE, for the same reason.
    """
    label = sequence.se_label
    if sequence.encoding.se.is_residual and eaf_failure is not None:
        raise Undecodable(
            f"{label} cannot be decoded under the declared plan: {eaf_failure}"
        ) from eaf_failure
    frequencies = np.full(len(codes), np.nan, dtype=np.float32) if eaf is None else eaf
    try:
        return sequence.se_codec.decode_se(
            codes,
            eaf=frequencies,
            analysis_index=analysis,
            coefficients=sequence.coefficients,
            positions=positions_flat(lo),
        )
    except ValueError as exc:
        raise Undecodable(f"{label} cannot be decoded under the declared plan: {exc}") from exc


def sequence_windows(target: Target) -> Iterator[Block]:
    """A sequence's cells in byte-bounded windows aligned to its physical read unit.

    Each window carries the Analysis and variant of the cell before it, so a rule
    about consecutive cells sees across a window's edge.
    """
    sequence = _Sequence(target)
    step = window_cells(inner_chunk_of(sequence.z)[0])
    previous: tuple[int, int] | None = None
    for lo in range(0, sequence.n_cells, step):
        block = sequence.window(lo, min(lo + step, sequence.n_cells), previous)
        assert block.analysis is not None and block.variant is not None
        previous = (int(block.analysis[-1]), int(block.variant[-1]))
        yield block


def _blocks(target: Target) -> Iterator[Block]:
    if target.kind is Kind.GRID:
        yield from dense_bands(target)
    elif target.kind is Kind.CSR:
        yield from sequence_windows(target)


def _read_failure(_target: Target, exc: Exception) -> str:
    if isinstance(exc, Undecodable):
        return str(exc)
    return f"validation failed: {exc}"


ASSOCIATIONS: Scan[Block] = Scan("associations", _blocks, _read_failure)
