"""The CSR adapter: a Ragged Store Release, or a Hybrid's Ragged Overflow Component."""

from __future__ import annotations

from operator import itemgetter
from pathlib import Path

import numpy as np

from opengwasdb.associations.cells import Cells
from opengwasdb.associations.record import Associations, Inspection, require_mask
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
        requires_mask: bool = False,
    ) -> None:
        self._path = Path(store_path)
        if encoding is None:
            encoding = StoreManifest.load(self._path).encoding
        self._encoding = encoding
        self._source = source
        if requires_mask:  # checked before any plane opens, so the refusal names the mask
            group = store_arrays.open_group(self._path / RAGGED_ZARR_PATH)
            require_mask(requires_mask, "imputed" in group, source)
        self.structure = RaggedCSRReader(store_path, encoding)
        self._by_variant: ByVariantReader | None = None
        self._offsets: np.ndarray | None = None

    @classmethod
    def open(
        cls,
        store_path: str | Path,
        encoding: StoreEncoding | None = None,
        *,
        source: str | None = None,
        requires_mask: bool = False,
    ) -> CsrReader:
        """`requires_mask` says the component is Reference-Completed: its imputed
        mask is what says which rows are imputed, so a component without one is
        refused here rather than read as all-observed (spec §15)."""
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

    # ── the interface ───────────────────────────────────────────────────────

    def read(self, cells: Cells, *, observed_only: bool = False) -> Associations:
        """The region's rows; raises `InvalidAssociations` on a §15-invalid one."""
        return self.inspect(cells).associations(self._source).observed_only(observed_only)

    def inspect(self, cells: Cells) -> Inspection:
        """Every row of the region, decoded and classified; never raises on one."""
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

    def _analyses(self, cells: Cells) -> Inspection:
        assert cells.ai is not None
        found = [self._analysis(int(a)) for a in cells.ai]
        return found[0] if len(found) == 1 else Inspection.concat(found, with_n=self.carries_n)

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
        assert wanted is not None
        if self._by_variant is None:
            raise VariantIndexError(
                f"{self._source}: a variant region needs the by-variant index; open it first"
            )
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
        return self._at(cells.pos)

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

