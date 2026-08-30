"""Zarr-backed Compressed Sparse Row storage for Ragged Layout associations."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import zarr
from numcodecs import Blosc

from opengwasdb.encoding import (
    EafMeasurements,
    RaggedEafPlane,
    StoreCodec,
    StoreEncoding,
    ZOverflowBuilder,
    ZOverflowTable,
    eaf_baseline_from_pairs,
    measure_eaf,
    positions_at,
    positions_flat,
    write_eaf_csr,
)
from opengwasdb.model.manifest import StoreManifest

RAGGED_ZARR_PATH = "data.zarr/ragged"
_COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
# Chunk size for the flat association arrays (~400 KB per chunk at float16).
_ASSOC_CHUNK = 200_000
_OFFSET_CHUNK = 10_000


class AnalysisAssociations(NamedTuple):
    variant_index: np.ndarray  # int32
    z: np.ndarray              # float32, decoded from the plane's own encoding
    se: np.ndarray             # float16
    eaf: np.ndarray            # float32, all-NaN when the store carries no EAF


class RaggedCSRWriter:
    """Accumulate per-analysis associations and flush to zarr CSR arrays.

    The store's plan (ADR 0037) is passed to `flush`, not to the constructor,
    because deciding it needs the frequencies this writer is still being fed:
    a build calls `eaf_measurements()` once everything is in, runs
    `StoreEncoding.decide()` once, and flushes with the answer.
    """

    def __init__(self, n_variants: int) -> None:
        self._n_variants = int(n_variants)
        self._variant_indices: list[np.ndarray] = []
        self._zscores: list[np.ndarray] = []
        self._ses: list[np.ndarray] = []
        self._eafs: list[np.ndarray] = []
        self._offsets: list[int] = [0]

    def add_analysis(
        self,
        variant_index: np.ndarray,
        z: np.ndarray,
        se: np.ndarray,
        eaf: np.ndarray | None = None,
    ) -> None:
        """Append one analysis. Arrays must be parallel and the same length.

        `eaf` is optional (ADR 0036): an Analysis whose source reports no
        frequency passes None and contributes all-NaN, so the flat array stays
        aligned with `z`/`se` whatever mix of Analyses a build spans. If *no*
        Analysis supplies one, `flush` writes no `eaf` array at all and the
        store looks exactly as it did before EAF existed.
        """
        n = len(variant_index)
        self._variant_indices.append(np.asarray(variant_index, dtype=np.int32))
        # Held as float32 and quantised once, by the codec, at flush -- never
        # pre-rounded into a stored dtype here.
        self._zscores.append(np.asarray(z, dtype=np.float32))
        self._ses.append(np.asarray(se, dtype=np.float16))
        if eaf is None:
            self._eafs.append(np.full(n, np.nan, dtype=np.float32))
        else:
            self._eafs.append(np.asarray(eaf, dtype=np.float32))
        self._offsets.append(self._offsets[-1] + n)

    @property
    def n_analyses(self) -> int:
        return len(self._offsets) - 1

    @property
    def n_associations(self) -> int:
        return self._offsets[-1]

    def _flat(self) -> tuple[np.ndarray, np.ndarray]:
        """The concatenated `(variant_index, eaf)` this writer holds."""
        if self.n_associations == 0:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
        return (
            np.concatenate(self._variant_indices).astype(np.int32),
            np.concatenate(self._eafs).astype(np.float32),
        )

    def eaf_measurements(self) -> EafMeasurements:
        """What the encoding tree needs to know about this component's EAF.

        Measured, not inferred from the layout: a Ragged manifest can span one
        cohort or twenty, and a store built on the assumption that it spanned
        one would clip silently against a range chosen for the wrong data
        (ADR 0037 §2).
        """
        variant_index, eaf = self._flat()
        return measure_eaf(variant_index, eaf, n_variants=self._n_variants)

    def flush(
        self,
        store_path: str | Path,
        encoding: StoreEncoding,
        *,
        eaf_baseline: np.ndarray | None = None,
    ) -> None:
        """Write CSR arrays to data.zarr/ragged/ inside store_path.

        `eaf_baseline` lets Reference Completion carry its source's baselines
        across a variant remap instead of recomputing them from the decoded
        frequencies -- recomputation would move every baseline by up to half a
        step and re-quantise every cell against it (ADR 0037 §2).
        """
        out = Path(store_path) / RAGGED_ZARR_PATH
        root = zarr.open_group(str(out), mode="w")

        offsets_arr = np.asarray(self._offsets, dtype=np.int64)

        codec = StoreCodec(encoding)
        if self.n_associations > 0:
            vi_arr = np.concatenate(self._variant_indices).astype(np.int32)
            z_values = np.concatenate(self._zscores).astype(np.float32)
            se_arr = np.concatenate(self._ses).astype(np.float16)
            eaf_arr = np.concatenate(self._eafs).astype(np.float32)
        else:
            vi_arr = np.empty(0, dtype=np.int32)
            z_values = np.empty(0, dtype=np.float32)
            se_arr = np.empty(0, dtype=np.float16)
            eaf_arr = np.empty(0, dtype=np.float32)
        # A CSR cell's flat position is its ordinal in the concatenated array,
        # which is what its overflow entry is keyed on.
        overflow = ZOverflowBuilder()
        z_arr = codec.encode_z(z_values, positions=positions_flat(0), overflow=overflow)
        if not encoding.eaf.is_residual:
            baseline = None
        elif eaf_baseline is not None:
            baseline = np.asarray(eaf_baseline, dtype=np.float32)
        else:
            baseline = eaf_baseline_from_pairs(vi_arr, eaf_arr, self._n_variants)

        root.create_dataset(
            "offsets", data=offsets_arr,
            chunks=(_OFFSET_CHUNK,), compressor=_COMPRESSOR, dtype=np.int64,
        )
        root.create_dataset(
            "variant_index", data=vi_arr,
            chunks=(_ASSOC_CHUNK,), compressor=_COMPRESSOR, dtype=np.int32,
        )
        root.create_dataset(
            "z", data=z_arr,
            chunks=(_ASSOC_CHUNK,), compressor=_COMPRESSOR, dtype=codec.z_dtype,
        )
        overflow.table().write(root)
        root.create_dataset(
            "se", data=se_arr,
            chunks=(_ASSOC_CHUNK,), compressor=_COMPRESSOR, dtype=np.float16,
        )
        if not encoding.eaf.is_absent:
            write_eaf_csr(
                root, codec, vi_arr, eaf_arr,
                baseline=baseline, compressor=_COMPRESSOR, chunks=(_ASSOC_CHUNK,),
            )
        root.attrs["layout"] = "ragged"
        root.attrs["completion_state"] = "observed_only"
        root.attrs["n_analyses"] = self.n_analyses
        root.attrs["n_associations"] = self.n_associations


class RaggedCSRReader:
    """Read per-analysis associations from zarr CSR arrays."""

    def __init__(self, store_path: str | Path, encoding: StoreEncoding | None = None):
        path = Path(store_path) / RAGGED_ZARR_PATH
        self._root = zarr.open_group(str(path), mode="r")
        self._offsets: zarr.Array = self._root["offsets"]
        self._variant_index: zarr.Array = self._root["variant_index"]
        self._z: zarr.Array = self._root["z"]
        self._se: zarr.Array = self._root["se"]
        # The plan is read from the release's manifest, never inferred from the
        # array's dtype: a store that disagrees with its own manifest must fail
        # validation, not decode as whatever the bytes happen to look like.
        if encoding is None:
            encoding = StoreManifest.load(Path(store_path)).encoding
        self._codec = StoreCodec(encoding, z_overflow=ZOverflowTable.read(self._root))
        # Every EAF read goes through the plane, which gathers the per-variant
        # baseline (and, on a Reference-Completed release, the panel frequency
        # for imputed cells) onto the cells being read. Absent on stores whose
        # sources report no frequency at all -- those read back as all-NaN.
        self._eaf_plane = RaggedEafPlane.open(
            self._root,
            encoding,
            imputed=self._root["imputed"] if "imputed" in self._root else None,
        )

    @property
    def n_analyses(self) -> int:
        return int(self._root.attrs.get("n_analyses", len(self._offsets) - 1))

    @property
    def n_associations(self) -> int:
        return int(self._root.attrs.get("n_associations", len(self._variant_index)))

    def _span(self, analysis_index: int) -> tuple[int, int]:
        """The `[start, end)` slice of the flat arrays one Analysis occupies."""
        offsets = self._offsets[analysis_index: analysis_index + 2]
        return int(offsets[0]), int(offsets[1])

    def get_analysis(self, analysis_index: int) -> AnalysisAssociations:
        """Return (variant_index, z, se, eaf) arrays for one analysis. O(1) zarr reads."""
        start, end = self._span(analysis_index)
        if start == end:
            return AnalysisAssociations(
                variant_index=np.empty(0, dtype=np.int32),
                z=np.empty(0, dtype=np.float32),
                se=np.empty(0, dtype=np.float16),
                eaf=np.empty(0, dtype=np.float32),
            )
        return AnalysisAssociations(
            variant_index=self._variant_index[start:end],
            z=self.z_slice(start, end),
            se=self._se[start:end],
            eaf=self.eaf_slice(start, end),
        )

    def variant_indices(self, analysis_index: int) -> np.ndarray:
        """The variant indices one Analysis holds associations at.

        Separate from `get_analysis` because the callers that need only the
        Analysis's genomic footprint -- LD-block enumeration for a Store
        Family with no gene target (issue #102) -- should not decode its
        statistics to find it.
        """
        start, end = self._span(analysis_index)
        if start == end:
            return np.empty(0, dtype=np.int32)
        return np.asarray(self._variant_index[start:end], dtype=np.int32)

    def z_slice(self, start: int, end: int) -> np.ndarray:
        """Decoded `z[start:end]` -- the only way a caller gets z-scores."""
        return self._codec.decode_z(
            self._z[start:end], positions=positions_flat(int(start))
        )

    def z_at(self, positions: np.ndarray) -> np.ndarray:
        """Decoded z at arbitrary flat CSR positions."""
        positions = np.asarray(positions, dtype=np.int64)
        if len(positions) == 0:
            return np.empty(0, dtype=np.float32)
        return self._codec.decode_z(
            np.asarray(self._z[:])[positions], positions=positions_at(positions)
        )

    def z_all(self) -> np.ndarray:
        """Every decoded z, in flat CSR order."""
        return self.z_slice(0, int(len(self._z)))

    @property
    def has_eaf(self) -> bool:
        """Whether this component stores EAF at all (ADR 0036)."""
        return self._eaf_plane.has_values

    def eaf_slice(self, start: int, end: int) -> np.ndarray:
        """Decoded `eaf[start:end]`, or all-NaN when this store carries none."""
        return self._eaf_plane.slice(start, end)

    def eaf_at(self, positions: np.ndarray) -> np.ndarray:
        """EAF at arbitrary flat CSR positions; all-NaN when there is no array.

        For the scanning query paths, which already hold flat positions into
        the concatenated arrays and would otherwise pay `eaf_pairs`'
        per-Analysis searchsorted to recover what they already know.
        """
        return self._eaf_plane.at(positions)

    def eaf_pairs(self, variant_index: np.ndarray, analysis_index: np.ndarray) -> np.ndarray:
        """EAF for elementwise (variant, analysis) pairs (ADR 0036).

        All-NaN when this component stores no `eaf` array -- built before ADR
        0036, or from sources reporting no frequency. Each Analysis's CSR slice
        is sorted by variant_index (every builder sorts before writing), so one
        `searchsorted` per distinct Analysis resolves its pairs; a pair whose
        variant is absent from that Analysis stays NaN rather than silently
        taking a neighbour's frequency.
        """
        out = np.full(len(variant_index), np.nan, dtype=np.float32)
        if not self.has_eaf or len(variant_index) == 0:
            return out
        offsets = self._offsets[:]
        # Resolve every pair to a flat CSR position first, then decode once:
        # the plane needs the position to resolve an exception cell, and a
        # per-Analysis decode would gather the baseline slice by slice.
        slots: list[np.ndarray] = []
        found: list[np.ndarray] = []
        for ai in np.unique(analysis_index):
            ai_int = int(ai)
            if ai_int < 0 or ai_int + 1 >= len(offsets):
                continue
            start, end = int(offsets[ai_int]), int(offsets[ai_int + 1])
            if start == end:
                continue
            slot = np.where(analysis_index == ai)[0]
            slice_vi = np.asarray(self._variant_index[start:end])
            pos = np.searchsorted(slice_vi, variant_index[slot])
            in_bounds = pos < len(slice_vi)
            hit = np.zeros(len(slot), dtype=bool)
            hit[in_bounds] = slice_vi[pos[in_bounds]] == variant_index[slot][in_bounds]
            if hit.any():
                slots.append(slot[hit])
                found.append(pos[hit].astype(np.int64) + start)
        if slots:
            out[np.concatenate(slots)] = self.eaf_at(np.concatenate(found))
        return out

    def get_analyses(self, analysis_indices: list[int]) -> list[AnalysisAssociations]:
        """Return associations for multiple analyses."""
        return [self.get_analysis(i) for i in analysis_indices]
