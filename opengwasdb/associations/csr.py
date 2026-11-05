"""The CSR adapter: a Ragged Store Release, or a Hybrid's Ragged Overflow Component."""

from __future__ import annotations

from collections.abc import Iterator
from operator import itemgetter
from pathlib import Path

import numpy as np

from opengwasdb.associations import record
from opengwasdb.associations.cells import (
    Cells,
    require_run_within,
    require_within,
    window_region,
)
from opengwasdb.associations.record import Associations, Inspection, InvalidAssociations, State
from opengwasdb.encoding import StoreEncoding
from opengwasdb.layouts.ragged.by_variant import (
    ByVariantReader,
    VariantIndexError,
    has_variant_index,
)
from opengwasdb.layouts.ragged.zarr_csr import RAGGED_ZARR_PATH, RaggedCSRReader
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.store import arrays as store_arrays

#: A by-variant index decode's columns, in `Inspection.of`'s order.
_block_columns = itemgetter("variant_index", "analysis_index", "z", "se", "eaf", "imputed")


def _manifest(path: Path) -> StoreManifest | None:
    """The release manifest beside a sequence, or None for a bare sequence."""
    return StoreManifest.load(path) if (path / "manifest.json").exists() else None


def _run(positions: np.ndarray) -> tuple[int, int] | None:
    """`[start, stop)` when `positions` is one ascending run, which reads as a slice.

    Positions whose ends are not `len - 1` apart are refused at once, as a scan's
    hits are; positions that are, a window among them, are a run exactly when
    each exceeds the one before. Neither test builds a copy of the positions: a
    scan asks this of every window, and of every hit it then reads.
    """
    n = len(positions)
    if not n or int(positions[-1]) - int(positions[0]) != n - 1:
        return None
    if not bool(np.all(positions[1:] > positions[:-1])):
        return None
    return int(positions[0]), int(positions[-1]) + 1


def _run_analyses(offsets: np.ndarray, start: int, stop: int) -> np.ndarray:
    """The Analysis of each row of the flat run `[start, stop)`, repeated over the
    spans it crosses: the cost is the Analyses touched, not a search per row."""
    first = int(np.searchsorted(offsets, start, side="right")) - 1
    last = int(np.searchsorted(offsets, stop, side="left"))
    rows = np.diff(np.clip(offsets[first : last + 1], start, stop))
    return np.repeat(np.arange(first, first + len(rows), dtype=np.int32), rows)


class CsrReader:
    """Reads associations from an Analysis-sorted association sequence (§11).

    A CSR holds a row for every pair a build attempted, so `read` returns every
    row of the region -- a stored missing row too, as status `MISSING`. Variant
    regions read the by-variant index (ADR 0060); a component without one
    refuses them with `VariantIndexError`. `cross` binary-searches each
    Analysis's sorted segment (#252), so its cost is the request, not the
    component. Each region reads Z, the imputed mask, EAF, SE and `n` once, with
    the EAF read handed to SE decoding (#253).
    """

    def __init__(
        self,
        store_path: str | Path,
        encoding: StoreEncoding | None = None,
        *,
        source: str,
        requires_mask: bool | None = None,
    ) -> None:
        self._path = Path(store_path)
        manifest = _manifest(self._path)
        if encoding is None:
            if manifest is None:
                raise ValueError(f"{source}: no manifest to read the encoding from")
            encoding = manifest.encoding
        self._encoding = encoding
        self._source = source
        # Checked before any plane opens, so the refusal names the mask.
        group = store_arrays.open_group(self._path / RAGGED_ZARR_PATH)
        state = (
            manifest.completion_state
            if manifest is not None
            else group.attrs.get("completion_state")
        )
        record.require_mask(
            record.mask_required(requires_mask, state, source), "imputed" in group, source
        )
        self.structure = RaggedCSRReader(store_path, encoding)
        self._by_variant: ByVariantReader | None = None
        #: The variant axis `open_variant_index` was given; a CSR records none itself.
        self._n_variants: int | None = None
        self._offsets: np.ndarray | None = None

    @classmethod
    def open(
        cls,
        store_path: str | Path,
        encoding: StoreEncoding | None = None,
        *,
        source: str | None = None,
        requires_mask: bool | None = None,
    ) -> CsrReader:
        """A Reference-Completed component's imputed mask says which rows are
        imputed, so one without it is refused rather than read as all-observed
        (spec §15). Whether the component is completed is its release's manifest
        -- or, with none, its sequence's `completion_state` record -- unless the
        caller says (`requires_mask`); a component that records neither is refused."""
        return cls(
            store_path,
            encoding,
            source=source or f"Ragged component {store_path}",
            requires_mask=requires_mask,
        )

    @property
    def carries_n(self) -> bool:
        return self.structure.has_n

    @property
    def n_analyses(self) -> int:
        return self.structure.n_analyses

    @property
    def has_variant_index(self) -> bool:
        return has_variant_index(self._path)

    def open_variant_index(self, n_variants: int) -> None:
        """Open the by-variant index now, checking it spans `n_variants`.

        Absence is an error: a release without one is refused with an instruction
        to build it (ADR 0060).
        """
        if not self.has_variant_index:
            raise VariantIndexError(
                f"Store at {self._path} carries no variant index (ragged/by_variant); "
                f"build it with 'ogdb build-variant-index {self._path}'"
            )
        self._by_variant = ByVariantReader(self._path, self._encoding, n_axis=n_variants)
        self._n_variants = int(n_variants)

    # ── the interface ───────────────────────────────────────────────────────

    def windows(
        self, cells: Cells | None = None, *, max_cells: int | None = None
    ) -> Iterator[Cells]:
        """An `analysis` or `analyses` region -- or, with none, the whole component --
        as consecutive `positions` regions of at most `max_cells` rows each.

        For a scan that must not hold a whole Analysis, or a whole sequence, at
        once: each row of the region comes once, in the region's order, so the
        windows' reads joined are the region's read. A window may run on from one
        Analysis into the next. `max_cells` defaults to the scan window, eight of
        the sequence's inner chunks (#252). The region is checked when this is
        called, before any window is cut (`window_region`).
        """
        region, step = window_region(
            cells,
            max_cells,
            default=self.structure.scan_window,
            n_analyses=self.n_analyses,
            source=self._source,
        )
        return self._windows(region, step)

    def _windows(self, region: Cells, step: int) -> Iterator[Cells]:
        for start, stop in self._runs(region):
            for lo in range(start, stop, step):
                yield Cells.positions(np.arange(lo, min(lo + step, stop), dtype=np.int64))

    def z(self, cells: Cells) -> np.ndarray:
        """Decoded Z of a `positions` region -- a window -- for a scan that thresholds
        Z alone, in the region's order.

        A non-canonical marker is refused here, as every read refuses one (spec §15):
        a NaN clears no cutoff, so a scan that compared it would drop the row.
        """
        if cells.kind != "positions" or cells.pos is None:
            raise TypeError(f"Z alone is read for positions, not a {cells.kind!r} region")
        run = self._rows_of(cells)
        if run is not None:
            z = self.structure.z_slice(*run)
        else:
            unique, back = np.unique(cells.pos, return_inverse=True)
            z = self.structure.z_at(unique)[back]
        odd = record.non_canonical(z)
        if odd.any():
            rows = cells.pos[odd]
            raise InvalidAssociations(
                self._source,
                self.structure.variant_index_at(rows),
                self.structure.analysis_index_at(rows),
                np.full(len(rows), State.NON_CANONICAL, dtype=np.int8),
            )
        return z

    def read(self, cells: Cells, *, observed_only: bool = False) -> Associations:
        """The region's rows; raises `InvalidAssociations` on a §15-invalid one."""
        return self.inspect(cells).associations(self._source).observed_only(observed_only)

    def inspect(self, cells: Cells) -> Inspection:
        """Every row of the region, decoded and classified; never raises on one.

        A region naming an index outside its axis is refused (`require_within`).
        A CSR records no variant axis of its own, so a region naming variants is
        read only once `open_variant_index` has given it one.
        """
        if cells.vi is not None and self._n_variants is None:
            raise VariantIndexError(
                f"{self._source}: a region naming variants is checked against the "
                "variant axis the by-variant index is opened on; open the variant "
                "index first"
            )
        require_within(
            cells, self._source, n_analyses=self.n_analyses, n_variants=self._n_variants
        )
        decode = {
            "analysis": self._analyses,
            "analyses": self._analyses,
            "variants": self._variants,
            "cross": self._cross,
            "pairs": self._pairs,
            "positions": self._positions,
        }.get(cells.kind)
        if decode is None:
            raise TypeError(f"a CSR cannot read a {cells.kind!r} region")
        return decode(cells)

    # ── decoding ────────────────────────────────────────────────────────────

    def _empty(self) -> Inspection:
        return Inspection.concat([], with_n=self.carries_n)

    def _runs(self, cells: Cells) -> list[tuple[int, int]]:
        """The flat `[start, stop)` runs an `analysis` or `analyses` region's rows sit
        in, in its order: one for a span of Analyses, else one per Analysis."""
        assert cells.ai is not None
        offsets = self.structure.offsets
        span = cells.span
        bounds = [span] if span is not None else [(int(a), int(a) + 1) for a in cells.ai]
        return [(int(offsets[first]), int(offsets[stop])) for first, stop in bounds]

    def _analyses(self, cells: Cells) -> Inspection:
        assert cells.ai is not None
        span = cells.span
        if span is not None and span[1] - span[0] > 1:
            [(start, stop)] = self._runs(cells)
            return self._rows(start, stop)
        found = [self._analysis(int(a)) for a in cells.ai]
        return found[0] if len(found) == 1 else Inspection.concat(found, with_n=self.carries_n)

    def _rows(self, start: int, stop: int) -> Inspection:
        """The rows of the flat run `[start, stop)`, each plane read as one slice."""
        if start == stop:
            return self._empty()
        csr = self.structure
        eaf = csr.eaf_slice_read(start, stop, want_imputed=True)
        return Inspection.of(
            csr.variant_index(start, stop),
            _run_analyses(csr.offsets, start, stop),
            csr.z_slice(start, stop),
            csr.se_slice(start, stop, eaf=eaf.values),
            eaf.values,
            eaf.imputed,
            csr.n_slice(start, stop) if self.carries_n else None,
        )

    def _analysis(self, analysis_index: int) -> Inspection:
        csr = self.structure
        start, end = csr.span(analysis_index)
        if start == end:
            return self._empty()
        variants = csr.variant_index(start, end)
        z = csr.z_slice(start, end)
        eaf = csr.eaf_slice_read(start, end, want_imputed=True)
        return Inspection.of(
            variants,
            analysis_index,
            z,
            csr.se_slice(start, end, analysis_index=analysis_index, eaf=eaf.values),
            eaf.values,
            eaf.imputed,
            csr.n_slice(start, end) if self.carries_n else None,
        )

    def _variants(self, cells: Cells) -> Inspection:
        """The by-variant index's rows for the region's variants: the block spanning
        the least and greatest, less the rows of variants between them it did not name."""
        wanted = cells.vi
        assert wanted is not None and self._by_variant is not None
        if len(wanted) == 0:
            return self._empty()
        low, high = int(wanted.min()), int(wanted.max())
        # One variant reads its two offsets in one call; a range reads its ends.
        first, last = (
            self._by_variant.rows_for_variant(low)
            if low == high
            else self._by_variant.rows_for_variant_range(low, high)
        )
        if first == last:
            return self._empty()
        block = self._by_variant.decode(low, high, first, last)
        vi, ai, z, se, eaf, flags = _block_columns(block)
        found = Inspection.of(vi, ai, z, se, eaf, flags, block.get("n"))
        if len(np.unique(wanted)) == high - low + 1:
            return found
        return found.take(np.isin(found.variant_index, wanted))

    def _cross(self, cells: Cells) -> Inspection:
        assert cells.vi is not None and cells.ai is not None
        return self._at(self._cross_positions(cells.vi, cells.ai))

    def _pairs(self, cells: Cells) -> Inspection:
        assert cells.vi is not None and cells.ai is not None
        return self._at(self._pair_positions(cells.vi, cells.ai))

    def _positions(self, cells: Cells) -> Inspection:
        assert cells.pos is not None
        run = self._rows_of(cells)
        return self._at(cells.pos) if run is None else self._rows(*run)

    def _rows_of(self, cells: Cells) -> tuple[int, int] | None:
        """A `positions` region's run, when it is one, its positions checked against
        the sequence's rows: a run from its ends, any other from every position."""
        assert cells.pos is not None
        run = _run(cells.pos)
        n_rows = self.structure.sequence_length
        if run is None:
            require_within(cells, self._source, n_analyses=self.n_analyses, n_rows=n_rows)
        else:
            require_run_within(*run, n_rows, self._source)
        return run

    def _cross_positions(self, variants: np.ndarray, analyses: np.ndarray) -> np.ndarray:
        wanted = np.unique(variants).astype(np.int32)
        found = [
            self.structure.segment_positions(wanted, analysis_index=int(a)) for a in analyses
        ]
        return np.concatenate(found) if found else np.empty(0, dtype=np.int64)

    def _pair_positions(self, variants: np.ndarray, analyses: np.ndarray) -> np.ndarray:
        slots, positions = self.structure.pair_positions(variants, analyses)
        if len(slots) != len(variants):
            raise LookupError(
                f"{self._source}: {len(variants) - len(slots):,} requested (variant, Analysis) "
                "pair(s) are not rows of this component"
            )
        out = np.empty(len(variants), dtype=np.int64)
        out[slots] = positions
        return out

    def _at(self, positions: np.ndarray) -> Inspection:
        """The rows at flat positions, in the order given; a repeat reads once."""
        positions = np.asarray(positions, dtype=np.int64)
        if len(positions) == 0:
            return self._empty()
        unique, back = np.unique(positions, return_inverse=True)
        csr = self.structure
        variants = csr.variant_index_at(unique)
        eaf = csr.eaf_at_read(unique, want_imputed=True)
        found = Inspection.of(
            variants,
            csr.analysis_index_at(unique),
            csr.z_at(unique),
            csr.se_at(unique, eaf=eaf.values),
            eaf.values,
            eaf.imputed,
            csr.n_at(unique) if self.carries_n else None,
        )
        return found if np.array_equal(unique, positions) else found.take(back)

