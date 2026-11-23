"""Reading a Ragged component's By-Variant Index (ADR 0060).

A Ragged component optionally carries a second copy of its associations, sorted
by variant, under `ragged/by_variant`: one variant's associations are then one
contiguous slice. `ByVariantReader` reads it; building and installing it is
`opengwasdb.layouts.ragged.by_variant`'s.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from opengwasdb.store_reader import arrays as store_arrays
from opengwasdb.store_reader.arrays import array_length
from opengwasdb.store_reader.csr import RAGGED_ZARR_PATH
from opengwasdb.store_reader.encoding.codec import (
    EAF_BASELINE,
    EAF_REFERENCE,
    SE_COEFFICIENTS,
    EafExceptionTable,
    SeExceptionTable,
    StoreDecoder,
    ZOverflowTable,
    positions_flat,
)
from opengwasdb.store_reader.encoding.plan import EafBaselineError, StoreEncoding
from opengwasdb.store_reader.manifest import StoreManifest
from opengwasdb.store_reader.stored import optional_array

#: The index group, under the Ragged component's own group (ADR 0060).
BY_VARIANT_GROUP = "by_variant"

#: The cell-keyed arrays a component may carry beside `z` and `se`, each copied
#: only when present.  `n` has no exception table to re-key: its codes are the
#: values (ADR 0067).
OPTIONAL_CELL_ARRAYS = ("eaf", "imputed", "n")


def has_variant_index(store_path: str | Path, *, group_name: str = BY_VARIANT_GROUP) -> bool:
    """Whether the component at `store_path` carries a by-variant index."""
    path = Path(store_path) / RAGGED_ZARR_PATH / group_name
    return (path / "zarr.json").is_file() or (path / ".zgroup").is_file()


class ByVariantReader:
    """Read a Ragged component's by-variant index (ADR 0060).

    A block is contiguous in the index's own order, so `rows_for_variant` and
    `rows_for_variant_range` turn a variant question into a row range, and
    `decode` reads and decodes exactly that range.  The per-row variant is
    recovered from the offsets rather than stored (the duplicate deliberately
    does not carry `variant_index`, ADR 0060), and the exception tables, the
    `eaf_baseline`/`eaf_reference` and the `se_coefficients` come from the
    index and the shared component respectively.
    """

    def __init__(
        self,
        store_path: str | Path,
        encoding: StoreEncoding | None = None,
        *,
        n_axis: int | None = None,
    ):
        self._ragged = store_arrays.open_group(Path(store_path) / RAGGED_ZARR_PATH)
        if BY_VARIANT_GROUP not in self._ragged:
            raise ValueError(
                f"{store_path}: {RAGGED_ZARR_PATH}/{BY_VARIANT_GROUP} is absent; a variant "
                "index must be built before it is read (ADR 0060)"
            )
        self._index = self._ragged[BY_VARIANT_GROUP]
        if encoding is None:
            encoding = StoreManifest.load(Path(store_path)).encoding
        self._encoding = encoding
        self._require_arrays()
        self._require_members()
        self._offsets = self._index["offsets"]
        self._require_span(store_path, n_axis)
        self._analysis_index = self._index["analysis_index"]
        self._z = self._index["z"]
        self._se = self._index["se"]
        self._eaf = optional_array(self._index, "eaf")
        self._imputed = optional_array(self._index, "imputed")
        self._n = optional_array(self._index, "n")
        self._baseline = optional_array(self._ragged, EAF_BASELINE)
        self._reference = optional_array(self._ragged, EAF_REFERENCE)
        self._coefficients = optional_array(self._ragged, SE_COEFFICIENTS)
        self._codec_cache: StoreDecoder | None = None

    def _require_arrays(self) -> None:
        """A missing required array is named, not a bare `KeyError` from a later read."""
        missing = [
            name
            for name in ("offsets", "analysis_index", "z", "se")
            if name not in self._index
        ]
        if missing:
            raise ValueError(
                f"{self._index.name}: the variant index is missing {missing}; a "
                "half-written group is invalid (ADR 0060)"
            )

    def _require_members(self) -> None:
        """The index must carry exactly the component's optional members.

        A missing `imputed` on a release that declares reference EAF is refused
        here for the same reason the scan plane refuses it: without the mask the
        panel substitution cannot be applied, and substituting zeros would
        silently read every imputed cell as observed.
        """
        if self._encoding.eaf.reference and "imputed" not in self._index:
            raise EafBaselineError(
                "this release declares reference EAF for its imputed cells but its "
                "variant index carries no imputed mask; without it an indexed read "
                "would read every imputed cell as observed (spec §9, §15)"
            )
        for name in OPTIONAL_CELL_ARRAYS:
            if (name in self._index) != (name in self._ragged):
                raise ValueError(
                    f"{self._index.name}/{name} is present on one side only; the "
                    "component and its variant index must carry the same members"
                )

    def _require_span(self, store_path: str | Path, n_axis: int | None) -> None:
        """The index must span the component it duplicates, and the declared axis.

        `offsets[-1]` is the row count and must equal the Analysis-sorted
        component's own; a mismatch is a stale index answering with another
        store's rows.  When the caller knows the variant axis length, the
        offsets array must be `n_axis + 1` long.  Only scalars are read -- the
        length, the first and the last offset -- so opening the reader does not
        materialise a 164 M-entry (1.28 GB) array (review round 2, finding 2).
        """
        entries = array_length(self._offsets)
        component_rows = int(self._ragged["offsets"][-1])
        if entries == 0 or int(self._offsets[-1]) != component_rows:
            last = int(self._offsets[-1]) if entries else None
            raise ValueError(
                f"{store_path}: {BY_VARIANT_GROUP}/offsets ends at {last} but the "
                f"component holds {component_rows} associations; the index is stale"
            )
        if int(self._offsets[0]) != 0:
            raise ValueError(
                f"{store_path}: {BY_VARIANT_GROUP}/offsets starts at "
                f"{int(self._offsets[0])}, not 0"
            )
        if n_axis is not None and entries != int(n_axis) + 1:
            raise ValueError(
                f"{store_path}: {BY_VARIANT_GROUP}/offsets has {entries} entries "
                f"but the variant axis is {n_axis} (expected {int(n_axis) + 1})"
            )

    @property
    def _codec(self) -> StoreDecoder:
        """The codec, built on first decode -- and not before.

        The index's re-keyed exception tables are read whole by `StoreDecoder`, and
        at OGS-00011 the EAF exception table is 180,396,687 entries (~2.1 GiB).
        Reading them at open made every query that merely *opened* the store pay
        that 2.1 GiB, including the analysis-side shapes the index does not
        serve; a first decode is the first time the tables are needed.
        """
        if self._codec_cache is None:
            self._codec_cache = StoreDecoder(
                self._encoding,
                z_overflow=ZOverflowTable.open(self._index),
                eaf_exceptions=EafExceptionTable.open(self._index),
                se_exceptions=SeExceptionTable.open(self._index),
            )
        return self._codec_cache

    def warm(self) -> None:
        """Read the codec's exception tables now, so a caller can time past them.

        The tables are a one-off per store open -- ~2.1 GiB and ~2.4 s at
        OGS-00011 -- and a benchmark that wants the per-query cost excludes them.
        """
        _ = self._codec

    @property
    def n_axis(self) -> int:
        return array_length(self._offsets) - 1

    @property
    def n_rows(self) -> int:
        return array_length(self._analysis_index)

    def rows_for_variant(self, variant_index: int) -> tuple[int, int]:
        """The half-open row range one variant's associations occupy.

        A variant the index does not cover (a Hybrid's on-panel variants) has an
        empty range, so the caller returns no rows rather than falling back to a
        scan -- the index *is* the answer for every variant it indexes.
        """
        variant = int(variant_index)
        if variant < 0 or variant >= self.n_axis:
            raise ValueError(
                f"variant index {variant} is outside the index's axis [0, {self.n_axis})"
            )
        pair = self._offsets[variant : variant + 2]
        return int(pair[0]), int(pair[1])

    def rows_for_variant_range(self, low: int, high: int) -> tuple[int, int]:
        """The half-open row range a contiguous variant range covers.

        `low` is the first and `high` the last variant in the range (inclusive);
        every row between their offsets belongs to a variant in `[low, high]`,
        because the index is variant-ordered.
        """
        first, _ = self.rows_for_variant(low)
        _, last = self.rows_for_variant(high)
        return first, last

    def decode(
        self, first_variant: int, last_variant: int, start: int, end: int
    ) -> dict[str, np.ndarray]:
        """Decode the index's rows `[start, end)`, whose variants span `[first, last]`.

        Returns the six result arrays' raw parts: `variant_index`,
        `analysis_index`, decoded `z`/`se`/`eaf` and the `imputed` mask (all
        zeros on an observed-only component) -- and decoded `n` when the
        component carries per-association sample sizes (ADR 0067).
        """
        start, end = int(start), int(end)
        empty = {
            "variant_index": np.empty(0, dtype="int32"),
            "analysis_index": np.empty(0, dtype="int32"),
            "z": np.empty(0, dtype="float32"),
            "se": np.empty(0, dtype="float32"),
            "eaf": np.empty(0, dtype="float32"),
            "imputed": np.empty(0, dtype="uint8"),
        }
        if self._n is not None:
            empty["n"] = np.empty(0, dtype="float64")
        if end <= start:
            return empty
        variants = self._row_variants(first_variant, last_variant, start, end)
        analysis_index = np.asarray(self._analysis_index[start:end], dtype=np.int32)
        imputed = self._imputed_slice(start, end)
        eaf = self._decode_eaf(start, end, variants, imputed)
        se = self._decode_se(start, end, analysis_index, eaf)
        z = self._codec.decode_z(
            np.asarray(self._z[start:end]), positions=positions_flat(start)
        )
        block = {
            "variant_index": variants.astype("int32"),
            "analysis_index": analysis_index,
            "z": z,
            "se": se,
            "eaf": eaf,
            "imputed": imputed,
        }
        if self._n is not None:
            block["n"] = self._codec.decode_n(np.asarray(self._n[start:end]))
        return block

    def _row_variants(self, first: int, last: int, start: int, end: int) -> np.ndarray:
        """The variant of every row in `[start, end)`, from the offsets.

        The duplicate stores no per-row `variant_index` (ADR 0060), so the
        variant is recovered from the offsets: the local offsets of `[first,
        last]` are enough, which is why a query never holds the whole array.
        """
        local = np.asarray(self._offsets[int(first) : int(last) + 2], dtype=np.int64)
        rows = np.arange(start, end, dtype=np.int64)
        return (np.searchsorted(local, rows, side="right") - 1 + int(first)).astype(np.int64)

    def _imputed_slice(self, start: int, end: int) -> np.ndarray:
        if self._imputed is None:
            return np.zeros(end - start, dtype=np.uint8)
        return np.asarray(self._imputed[start:end], dtype=np.uint8)

    def _decode_eaf(
        self, start: int, end: int, variants: np.ndarray, imputed: np.ndarray
    ) -> np.ndarray:
        """Decoded frequencies, with the panel's value on imputed cells (ADR 0037 §4)."""
        baseline = self._gather(self._baseline, variants)
        reference = self._gather(self._reference, variants)
        want_imputed = self._encoding.eaf.reference
        imputed_arg = imputed if want_imputed else None
        if self._eaf is None:
            if not self._encoding.eaf.reference:
                return np.full(end - start, np.nan, dtype=np.float32)
            blank = np.full(end - start, np.nan, dtype=np.float32)
            return self._codec.decode_eaf(blank, imputed=imputed_arg, reference=reference)
        return self._codec.decode_eaf(
            np.asarray(self._eaf[start:end]),
            baseline=baseline,
            positions=positions_flat(start),
            imputed=imputed_arg,
            reference=reference,
        )

    @staticmethod
    def _gather(array: Any, variants: np.ndarray) -> np.ndarray | None:
        if array is None:
            return None
        return np.asarray(array.oindex[variants], dtype=np.float32)

    def _decode_se(
        self, start: int, end: int, analysis_index: np.ndarray, eaf: np.ndarray
    ) -> np.ndarray:
        raw = np.asarray(self._se[start:end])
        if not self._encoding.se.is_residual:
            return self._codec.decode_se(
                raw,
                eaf=eaf,
                analysis_index=analysis_index.astype(np.int64),
                coefficients=np.empty((0, 2), dtype=np.float32),
                positions=positions_flat(start),
            )
        if self._coefficients is None:
            raise ValueError(
                "the release declares int8_residual se but the component carries no "
                "se_coefficients; the index cannot decode it"
            )
        return self._codec.decode_se(
            raw,
            eaf=eaf,
            analysis_index=analysis_index.astype(np.int64),
            coefficients=np.asarray(self._coefficients[:], dtype=np.float32),
            positions=positions_flat(start),
        )


class VariantIndexError(RuntimeError):
    """The augment path cannot install an index into this release."""
