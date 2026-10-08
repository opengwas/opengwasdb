"""Zarr-backed Compressed Sparse Row storage for Ragged Layout associations."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import zarr

from opengwasdb.encoding import (
    EafExceptionBuilder,
    EafMeasurements,
    EafRead,
    OverflowCellBatches,
    OverflowCells,
    RaggedEafPlane,
    RaggedSePlane,
    SeExceptionBuilder,
    SeMeasurements,
    StoreCodec,
    StoreEncoding,
    ZOverflowBuilder,
    ZOverflowTable,
    eaf_baseline_from_sorted_runs,
    fit_se,
    measure_eaf,
    positions_at,
    positions_flat,
    write_eaf_baseline,
)
from opengwasdb.encoding.planes import write_se_coefficients
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.store import arrays as store_arrays
from opengwasdb.store.arrays import ArrayRole, array_length

RAGGED_ZARR_PATH = "data.zarr/ragged"
_COMPRESSOR = store_arrays.compressor()
# Chunk size for the flat association arrays (~400 KB per chunk at float16).
# Read from the seam so the SE measurement charges the bytes the seam writes.
_ASSOC_CHUNK = store_arrays.ASSOCIATION_SEQUENCE_CHUNK
#: Cells one `se_fit_batches` batch aims to carry. The batch holds the variant
#: indices, the source and round-tripped frequencies, the gathered baseline and
#: the Analysis indices -- about 30 bytes a cell -- so 2**24 is roughly a
#: 500 MiB working set, whatever the plane's total cell count (issue #228).
DEFAULT_SE_FIT_CELL_BUDGET = 1 << 24
#: Association-array chunks one scan window spans. A window of a single chunk
#: makes a whole-store scan issue one zarr read per chunk, which measured ~5x
#: slower than the batched whole-array read it replaces on OGS-00011's
#: Overflow (132.9 s against 27.3 s for off-axis PheWAS, #252); a handful of
#: chunks amortises the per-read overhead while peak memory stays bounded by
#: the window (8 x 200,000 int32 = 6.4 MB) and not by the array.
SCAN_WINDOW_CHUNKS = 8
#: Cells one `flush` region writes at a time, as a *floor*.  The region holds
#: the four source planes, the codes it encodes them to and the frequencies it
#: decodes back -- about 30 bytes a cell -- so 2**22 is roughly a 120 MiB working
#: set whatever the component's cell count (issue #228).  The Ragged sequence
#: planes are written one **shard** at a time even when that is larger (issue
#: #249), because a write covering part of a shard is a read-modify-write of the
#: whole shard; see `RaggedCSRWriter._flat_regions`.
DEFAULT_FLUSH_REGION_CELLS = 1 << 22


def sequence_region_step(total: int, region_cells: int) -> int:
    """The write step for a Ragged sequence plane: whole shards, at least one (#249).

    A write covering part of a shard is a read-modify-write of the whole shard,
    so writing a 50,000,000-element shard once per 4,194,304-cell region would
    decode and re-encode it about twelve times.  The step is therefore a whole
    number of shards: `region_cells` is rounded **up** to the next whole shard
    (never lowered below one), so a caller asking for a larger working set keeps
    it and no region can end inside a shard.  A module-level function so
    `benchmarks/measure_write_amplification.py` can reproduce the pre-#249 step
    on the same code path; production never calls it with a different one.
    """
    shard = int(
        store_arrays.shard_layout(
            ArrayRole.ASSOCIATION_SEQUENCE,
            (total,),
            inner_chunk=(store_arrays.ASSOCIATION_SEQUENCE_CHUNK,),
        )[0]
    )
    shards_per_region = max(1, -(-max(1, int(region_cells)) // shard))
    return shards_per_region * shard


class AnalysisAssociations(NamedTuple):
    variant_index: np.ndarray  # int32
    z: np.ndarray  # float32, decoded from the plane's own encoding
    se: np.ndarray  # decoded float32
    eaf: np.ndarray  # float32, all-NaN when the store carries no EAF


class RaggedCSRWriter:
    """Accumulate per-analysis associations and flush to zarr CSR arrays.

    The store's plan (ADR 0037) is passed to `flush`, not to the constructor,
    because deciding it needs the frequencies this writer is still being fed:
    a build calls `eaf_measurements()` once everything is in, runs
    `StoreEncoding.decide()` once, and flushes with the answer.

    The writer also holds the one per-variant EAF baseline derived from the
    cells it has been fed (issue #230), because every phase that codes or fits
    a cell needs it: a Hybrid build asks for it first in the joint SE fit's
    Analysis-aligned batches, again in that fit's chunk-aligned measurement
    batches, and again in the CSR flush.
    """

    def __init__(self, n_variants: int) -> None:
        self._n_variants = int(n_variants)
        self._variant_indices: list[np.ndarray] = []
        self._zscores: list[np.ndarray] = []
        self._ses: list[np.ndarray] = []
        self._eafs: list[np.ndarray] = []
        self._offsets: list[int] = [0]
        #: The baseline this writer has already derived, or `None` before the
        #: first ask. `None` can only mean "not derived": a derivation under a
        #: residual plan always returns an array, so it is never the answer.
        self._derived_baseline: np.ndarray | None = None
        #: The group `write_eaf_plane` created and the plan it wrote it under,
        #: or `None` before it has. The joint SE fit and its byte measurement
        #: open a decoded view of the `eaf` plane from these, so neither
        #: re-encodes a cell the write already encoded (issue #232).
        self._eaf_root: Any = None
        self._eaf_encoding: StoreEncoding | None = None
        self._eaf_view: RaggedEafPlane | None = None

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
        vi = np.asarray(variant_index, dtype=np.int32)
        # Every builder sorts an Analysis's associations by variant_index before
        # adding them (build_besd, build_ssf, Hybrid's `_assemble_overflow_column`
        # and completion's `_remapped_analysis_arrays` all argsort). The variant-
        # side queries rely on it: `lookup` and the Hybrid overflow lookup
        # binary-search each Analysis's segment instead of scanning the store
        # (#252). An unsorted segment would make those searches return a
        # plausible, wrong row, so it is refused here rather than assumed.
        if vi.size > 1 and bool(np.any(vi[1:] < vi[:-1])):
            offset = int(np.argmax(vi[1:] < vi[:-1])) + 1
            raise ValueError(
                "an Analysis's associations must be sorted ascending by variant_index "
                f"(decrease at offset {offset}); the variant-side queries binary-search "
                "each Analysis's CSR segment and cannot be correct on unsorted rows (#252)"
            )
        self._variant_indices.append(vi)
        # Held as float32 and quantised once, by the codec, at flush -- never
        # pre-rounded into a stored dtype here.
        self._zscores.append(np.asarray(z, dtype=np.float32))
        self._ses.append(np.asarray(se, dtype=np.float32))
        if eaf is None:
            self._eafs.append(np.full(n, np.nan, dtype=np.float32))
        else:
            self._eafs.append(np.asarray(eaf, dtype=np.float32))
        self._offsets.append(self._offsets[-1] + n)
        # A baseline is a median over every cell at its variant, so a new cell
        # can move one: a held baseline must never outlive the cells it came
        # from. Only `add_analysis` changes those cells.
        self._derived_baseline = None

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

    def se_measurements(self, encoding: StoreEncoding) -> SeMeasurements:
        """Fit SE against the EAF values this plan will actually decode."""
        cells = self.se_fit_inputs(encoding)
        return fit_se(
            cells.se_values,
            cells.eaf_values,
            cells.analysis_indices,
            n_analyses=self.n_analyses,
            compressor=_COMPRESSOR,
            chunks=_ASSOC_CHUNK,
        )[1]

    def se_fit_batches(
        self, *, cell_budget: int = DEFAULT_SE_FIT_CELL_BUDGET
    ) -> Iterator[OverflowCells]:
        """The plane's cells, in Analysis-aligned batches.

        The frequencies are read back from the written `eaf` plane -- exactly
        what a query decodes -- rather than re-encoded from the source values,
        so this pass derives no round trip the byte measurement derives again
        (issue #232). Concatenated, the batches are the plane's cells in flat
        order.

        Batches never split an Analysis. The fit these feed accumulates
        `numpy.bincount` sufficient statistics per Analysis, so an Analysis
        confined to one batch has its sums added in the order the whole-array
        pass would have added them -- which is what keeps the coefficients
        bit-for-bit identical rather than merely close (issue #228). An
        Analysis larger than the budget is its own batch: the budget bounds the
        working set, and the plane's largest Analysis is the floor it cannot go
        below.
        """
        for first, last in self._analysis_batches(max(1, int(cell_budget))):
            lo, hi = self._offsets[first], self._offsets[last]
            if hi == lo:
                continue
            se = np.concatenate(self._ses[first:last]).astype(np.float32)
            yield OverflowCells(
                se_values=se,
                eaf_values=self._decoded_eaf(lo, hi),
                analysis_indices=np.repeat(
                    np.arange(first, last, dtype=np.int64),
                    np.diff(np.asarray(self._offsets[first : last + 1], dtype=np.int64)),
                ),
                n_analyses=self.n_analyses,
            )

    def se_fit_chunk_batches(
        self,
        multiple: int,
        *,
        cell_budget: int = DEFAULT_SE_FIT_CELL_BUDGET,
    ) -> Iterator[OverflowCells]:
        """The same cells, in batches whose lengths are multiples of `multiple`.

        What the byte measurement needs rather than what the fit needs: it
        charges each candidate chunk by chunk and pads only the plane's final
        edge chunk (#158), so a batch that ended anywhere but a chunk boundary
        would have its own edge padded and change the selected range. Batches
        here therefore cut on flat position, crossing Analyses freely -- the
        measurement groups by the Analysis index carried per cell, not by the
        batch it arrived in. The frequencies are read back from the `eaf`
        plane, as in `se_fit_batches` (issue #232).
        """
        total = self.n_associations
        if total == 0:
            return
        offsets = np.asarray(self._offsets, dtype=np.int64)
        step = max(1, cell_budget // max(1, multiple)) * max(1, multiple)
        for lo in range(0, total, step):
            hi = min(lo + step, total)
            batch = self._gather_flat(offsets, lo, hi)
            if batch is not None:
                yield batch

    def _gather_flat(self, offsets: np.ndarray, lo: int, hi: int) -> OverflowCells | None:
        """One flat cell range `[lo, hi)`, gathered across the Analyses it spans."""
        errors: list[np.ndarray] = []
        analyses: list[np.ndarray] = []
        for analysis in self._analyses_spanning(offsets, lo, hi):
            start = int(offsets[analysis])
            head, tail = max(lo, start) - start, min(hi, int(offsets[analysis + 1])) - start
            errors.append(self._ses[analysis][head:tail])
            analyses.append(np.full(tail - head, analysis, dtype=np.int64))
        if not errors:
            return None
        return OverflowCells(
            se_values=np.concatenate(errors).astype(np.float32),
            eaf_values=self._decoded_eaf(lo, hi),
            analysis_indices=np.concatenate(analyses),
            n_analyses=self.n_analyses,
        )

    def _decoded_eaf(self, lo: int, hi: int) -> np.ndarray:
        """The frequencies a reader decodes for flat cells `[lo, hi)`.

        Read from the plane `write_eaf_plane` wrote -- the same array, exception
        table and per-variant baseline a query reads -- so the joint SE fit and
        its byte measurement derive each cell's round trip once, at the write,
        rather than each re-encoding the source values (issue #232). Reading a
        region at a time keeps the footprint the batch's, not the plane's. The
        view is opened on the first ask, so a caller that only writes (a
        standalone Ragged `flush`) never pays to open one.
        """
        if self._eaf_view is None:
            if self._eaf_root is None or self._eaf_encoding is None:
                raise RuntimeError(
                    "the SE fit reads frequencies back from the written eaf plane; call "
                    "write_eaf_plane before se_fit_batches or se_fit_chunk_batches"
                )
            self._eaf_view = RaggedEafPlane.open(self._eaf_root, self._eaf_encoding)
        return self._eaf_view.slice(lo, hi)

    def se_fit_source(
        self, *, cell_budget: int = DEFAULT_SE_FIT_CELL_BUDGET
    ) -> OverflowCellBatches:
        """This component's cells as a source the joint SE optimiser can stream.

        Both batchings the optimiser needs, from one writer: Analysis-aligned
        for the fit's per-Analysis sums, chunk-aligned for the byte measurement
        (issue #228). Both read their frequencies back from the `eaf` plane
        `write_eaf_plane` wrote (issue #232).
        """
        return OverflowCellBatches(
            n_analyses=self.n_analyses,
            analysis_batches=lambda: self.se_fit_batches(cell_budget=cell_budget),
            chunk_batches=lambda multiple: self.se_fit_chunk_batches(
                multiple, cell_budget=cell_budget
            ),
        )

    def _analysis_batches(self, cell_budget: int) -> Iterator[tuple[int, int]]:
        """Half-open Analysis ranges whose cells stay within `cell_budget`."""
        first = 0
        for analysis in range(self.n_analyses):
            spans = self._offsets[analysis + 1] - self._offsets[first]
            if spans > cell_budget and analysis > first:
                yield first, analysis
                first = analysis
        if first < self.n_analyses:
            yield first, self.n_analyses

    def _derive_eaf_baseline(self) -> np.ndarray:
        """One pass over the per-Analysis runs, bounded in cells (issue #226)."""
        return eaf_baseline_from_sorted_runs(self._variant_indices, self._eafs, self._n_variants)

    def _eaf_baseline(self, encoding: StoreEncoding) -> np.ndarray | None:
        """The per-variant Effect Allele Frequency Baseline, or None.

        From the per-Analysis runs rather than the concatenation: every column
        was sorted by variant index before it was added, and the bounded path
        keeps a billion-cell Overflow inside memory (issue #226).

        Derived once and held for the writer's remaining life (issue #230).
        The baseline is a function of the cells alone, and a Hybrid build asks
        for it over the same cells three times -- the fit's Analysis-aligned
        batches, the measurement's chunk-aligned batches and the flush -- so
        deriving it per ask walks a 15-billion-cell Overflow twice more than it
        needs to. What is held is one `float32` per variant, not per cell, and
        each pass already held its own copy for that pass's duration, so the
        reuse removes the passes and not a byte of peak.
        """
        if not encoding.eaf.is_residual:
            return None
        if self._derived_baseline is None:
            self._derived_baseline = self._derive_eaf_baseline()
        return self._derived_baseline

    def _decode_round_trip(
        self,
        encoding: StoreEncoding,
        eaf: np.ndarray,
        baseline: np.ndarray | None,
        vi: np.ndarray,
    ) -> np.ndarray:
        """The frequencies a reader gets back for these cells.

        The SE model has to predict from what is stored, not from the source
        (ADR 0037), so the fit and the measurement both see the round trip.
        """
        if encoding.eaf.is_absent:
            return np.full(eaf.shape, np.nan, dtype=np.float32)
        gathered = None if baseline is None else baseline[vi]
        exceptions = EafExceptionBuilder()
        raw = StoreCodec(encoding).encode_eaf(
            eaf, baseline=gathered, positions=positions_flat(0), exceptions=exceptions
        )
        return StoreCodec(encoding, eaf_exceptions=exceptions.table()).decode_eaf(
            raw, baseline=gathered, positions=positions_flat(0)
        )

    def se_fit_inputs(self, encoding: StoreEncoding) -> OverflowCells:
        """Named `(se, decoded_eaf, analysis_index)` for a shared plan."""
        vi, eaf = self._flat()
        se = (
            np.concatenate(self._ses).astype(np.float32)
            if self.n_associations
            else np.empty(0, dtype=np.float32)
        )
        baseline = self._eaf_baseline(encoding)
        codec = StoreCodec(encoding)
        exceptions = EafExceptionBuilder()
        raw = codec.encode_eaf(
            eaf,
            baseline=None if baseline is None else baseline[vi],
            positions=positions_flat(0),
            exceptions=exceptions,
        )
        decoded = (
            StoreCodec(encoding, eaf_exceptions=exceptions.table()).decode_eaf(
                raw,
                baseline=None if baseline is None else baseline[vi],
                positions=positions_flat(0),
            )
            if not encoding.eaf.is_absent
            else np.full(eaf.shape, np.nan, dtype=np.float32)
        )
        offsets = np.asarray(self._offsets, dtype=np.int64)
        ai = np.searchsorted(offsets[1:], np.arange(len(se)), side="right")
        return OverflowCells(
            se_values=se,
            eaf_values=decoded,
            analysis_indices=ai,
            n_analyses=self.n_analyses,
        )

    def _plane(self, root: Any, name: str, total: int, dtype: Any) -> Any:
        """An empty plane at full length, to be filled region by region."""
        return store_arrays.create_array(
            root,
            name,
            ArrayRole.ASSOCIATION_SEQUENCE,
            shape=(total,),
            dtype=dtype,
            compressor=_COMPRESSOR,
            overwrite=True,
        )

    def _flat_regions(self, total: int, region_cells: int) -> Iterator[tuple[int, int]]:
        """Half-open flat cell ranges covering the component, in plane order.

        Ascending, because the z overflow and both exception tables are keyed on
        global flat position and appended as they are encountered: visiting the
        regions in order is what keeps their rows in the order a single pass
        over the whole plane would have produced.

        The step is a whole number of Ragged sequence **shards**, at least one.  A
        write that covers part of a shard is a read-modify-write of the whole
        shard, so writing a 50,000,000-element shard once per 4,194,304-cell region
        would decode and re-encode it about twelve times -- silent write
        amplification that the tiny pilots cannot show (#249).  `region_cells` is
        therefore rounded up to a whole shard, never lowered, so a caller asking
        for a larger working set keeps it and no region ends inside a shard; a
        sequence shorter than one shard is still written in one region, exactly as
        before.
        """
        step = sequence_region_step(total, region_cells)
        for lo in range(0, total, step):
            yield lo, min(lo + step, total)

    def _gather_region(
        self, offsets: np.ndarray, lo: int, hi: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The four source planes for flat cells `[lo, hi)`, across the
        Analyses that range spans."""
        indices: list[np.ndarray] = []
        scores: list[np.ndarray] = []
        errors: list[np.ndarray] = []
        frequencies: list[np.ndarray] = []
        for analysis in self._analyses_spanning(offsets, lo, hi):
            start = int(offsets[analysis])
            head, tail = max(lo, start) - start, min(hi, int(offsets[analysis + 1])) - start
            indices.append(self._variant_indices[analysis][head:tail])
            scores.append(self._zscores[analysis][head:tail])
            errors.append(self._ses[analysis][head:tail])
            frequencies.append(self._eafs[analysis][head:tail])
        if not indices:
            empty = np.empty(0, dtype=np.float32)
            return np.empty(0, dtype=np.int32), empty, empty, empty
        return (
            np.concatenate(indices).astype(np.int32),
            np.concatenate(scores).astype(np.float32),
            np.concatenate(errors).astype(np.float32),
            np.concatenate(frequencies).astype(np.float32),
        )

    def _analyses_spanning(self, offsets: np.ndarray, lo: int, hi: int) -> Iterator[int]:
        """Each Analysis contributing a cell to flat range `[lo, hi)`."""
        first = int(np.searchsorted(offsets, lo, side="right")) - 1
        for analysis in range(max(first, 0), self.n_analyses):
            start, stop = int(offsets[analysis]), int(offsets[analysis + 1])
            if start >= hi:
                return
            if min(hi, stop) > max(lo, start):
                yield analysis

    def _region_analysis_indices(self, offsets: np.ndarray, lo: int, hi: int) -> np.ndarray:
        """The Analysis each cell of flat range `[lo, hi)` belongs to."""
        parts = [
            np.full(
                min(hi, int(offsets[analysis + 1])) - max(lo, int(offsets[analysis])),
                analysis,
                dtype=np.int64,
            )
            for analysis in self._analyses_spanning(offsets, lo, hi)
        ]
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

    def _write_se_streamed(
        self,
        root: Any,
        codec: StoreCodec,
        encoding: StoreEncoding,
        offsets: np.ndarray,
        se_coefficients: np.ndarray | None,
        total: int,
        region_cells: int,
    ) -> None:
        """Encode `se` against the EAF plane just written, region by region.

        The residual has to predict from the frequencies a reader will get back,
        so this runs after the `eaf` plane is written and decodes what it wrote --
        a region at a time, read back from the plane rather than kept from the
        encode (issue #228). A Hybrid build supplies `se_coefficients` because
        both components share one fit; a Ragged build has only itself to fit
        against.
        """
        plane = self._plane(root, "se", total, encoding.se.dtype)
        if not encoding.se.is_residual:
            for lo, hi in self._flat_regions(total, region_cells):
                plane[lo:hi] = self._gather_region(offsets, lo, hi)[2].astype(np.float16)
            return
        coefficients = (
            se_coefficients
            if se_coefficients is not None
            else self._fit_own_coefficients(root, encoding, offsets, total)
        )
        decoded = RaggedEafPlane.open(root, encoding)
        exceptions = SeExceptionBuilder()
        for lo, hi in self._flat_regions(total, region_cells):
            plane[lo:hi] = codec.encode_se(
                self._gather_region(offsets, lo, hi)[2],
                eaf=decoded.slice(lo, hi),
                analysis_index=self._region_analysis_indices(offsets, lo, hi),
                coefficients=coefficients,
                positions=positions_flat(lo),
                exceptions=exceptions,
            )
        write_se_coefficients(root, coefficients, compressor=_COMPRESSOR)
        exceptions.table().write(
            root, compressor=_COMPRESSOR, role=ArrayRole.RAGGED_EXCEPTION_TABLE
        )

    def _fit_own_coefficients(
        self, root: Any, encoding: StoreEncoding, offsets: np.ndarray, total: int
    ) -> np.ndarray:
        """Fit this component alone, for a Ragged build with no shared plan.

        Still whole-plane: a standalone Ragged store is the size of one cohort,
        not of a Hybrid's Overflow, and the Hybrid builder never reaches here
        because its joint fit supplies the coefficients (issue #228).
        """
        errors = (
            np.concatenate(self._ses).astype(np.float32) if total else np.empty(0, dtype=np.float32)
        )
        return fit_se(
            errors,
            RaggedEafPlane.open(root, encoding).slice(0, total),
            np.searchsorted(offsets[1:], np.arange(total), side="right"),
            n_analyses=self.n_analyses,
            compressor=_COMPRESSOR,
            chunks=_ASSOC_CHUNK,
        )[0]

    def _flush_baseline(
        self, encoding: StoreEncoding, eaf_baseline: np.ndarray | None
    ) -> np.ndarray | None:
        """The per-variant baseline the written `eaf` plane is coded against.

        None when the plan has no residual to code. A supplied `eaf_baseline`
        wins: Reference Completion carries its source's baselines across a
        variant remap, because recomputing them from the decoded frequencies
        would move every baseline by up to half a step and re-quantise every
        cell against it (ADR 0037 §2). Otherwise this is the writer's one held
        derivation (issue #230).
        """
        if not encoding.eaf.is_residual:
            return None
        if eaf_baseline is not None:
            return np.asarray(eaf_baseline, dtype=np.float32)
        return self._eaf_baseline(encoding)

    def _write_frequency_regions(
        self,
        root: Any,
        codec: StoreCodec,
        encoding: StoreEncoding,
        offsets_arr: np.ndarray,
        baseline: np.ndarray | None,
        region_cells: int,
    ) -> None:
        """Write `variant_index`, `z` and `eaf` region by region, in flat order.

        A CSR cell's flat position is its ordinal in the concatenated arrays,
        which is what its overflow and exception entries are keyed on, so
        `positions_flat(lo)` supplies it per region and both tables come out in
        the order a single pass over the whole plane would have appended them.
        """
        total = self.n_associations
        variant_index = self._plane(root, "variant_index", total, np.int32)
        z_plane = self._plane(root, "z", total, codec.z_dtype)
        eaf_plane = (
            None if encoding.eaf.is_absent else self._plane(root, "eaf", total, codec.eaf_dtype)
        )
        z_overflow = ZOverflowBuilder()
        eaf_exceptions = EafExceptionBuilder()
        for lo, hi in self._flat_regions(total, region_cells):
            indices, scores, _errors, frequencies = self._gather_region(offsets_arr, lo, hi)
            variant_index[lo:hi] = indices
            z_plane[lo:hi] = codec.encode_z(
                scores, positions=positions_flat(lo), overflow=z_overflow
            )
            if eaf_plane is not None:
                eaf_plane[lo:hi] = codec.encode_eaf(
                    frequencies,
                    baseline=None if baseline is None else baseline[indices],
                    positions=positions_flat(lo),
                    exceptions=eaf_exceptions,
                )
        z_overflow.table().write(root, role=ArrayRole.RAGGED_EXCEPTION_TABLE)
        if encoding.eaf.is_residual:
            assert baseline is not None
            write_eaf_baseline(
                root, baseline, compressor=_COMPRESSOR, role=ArrayRole.RAGGED_PER_VARIANT
            )
            eaf_exceptions.table().write(root, role=ArrayRole.RAGGED_EXCEPTION_TABLE)

    def write_eaf_plane(
        self,
        store_path: str | Path,
        encoding: StoreEncoding,
        *,
        eaf_baseline: np.ndarray | None = None,
        region_cells: int = DEFAULT_FLUSH_REGION_CELLS,
    ) -> None:
        """Create the component's zarr group and write its frequency half.

        The `eaf` plane's encoding is decided before the joint SE fit runs, so
        the plane is written here, ahead of the fit, and both the fit and its
        byte measurement read the frequencies back from it instead of each
        re-encoding every cell (issue #232). This creates the group, exactly
        once; `flush_se` adds the SE half to the group this leaves rather than
        replacing it, and a group with only this half carries no
        `completion_state` for a later phase to mistake for a finished
        component.

        Written a region of cells at a time rather than from concatenated
        planes, so the footprint is bounded by the region and not the
        component's cell count: on OGS-00011's 15,078,327,210 Overflow cells the
        concatenating write cost a measured 72.9 bytes a cell, or 1.10 TB (issue
        #228).  Each region is a whole Ragged sequence shard (issue #249), so
        each shard is written exactly once; on the 50,000,000-element shard that
        is roughly a 1.5 GB (1.40 GiB) working set (about 30 bytes a cell),
        against the 120 MiB a 4,194,304-cell region used before.  What is stored is
        unchanged -- each plane's codes are a per-cell function of its value,
        keyed on global flat position (`positions_flat(lo)` per region).
        `eaf_baseline` lets Reference Completion carry its source's baselines
        across a variant remap; see `_flush_baseline`.
        """
        out = Path(store_path) / RAGGED_ZARR_PATH
        root = store_arrays.open_group_for_write(out, "w")
        offsets_arr = np.asarray(self._offsets, dtype=np.int64)
        codec = StoreCodec(encoding)
        baseline = self._flush_baseline(encoding, eaf_baseline)
        store_arrays.create_array(
            root,
            "offsets",
            ArrayRole.ASSOCIATION_OFFSETS,
            data=offsets_arr,
            dtype=np.int64,
            compressor=_COMPRESSOR,
        )
        self._write_frequency_regions(root, codec, encoding, offsets_arr, baseline, region_cells)
        # Held so the joint SE fit and its byte measurement read these cells
        # back rather than re-encoding them (issue #232). The decoded view is
        # opened lazily, on the first ask, so a write-only caller never opens
        # one.
        self._eaf_root = root
        self._eaf_encoding = encoding
        self._eaf_view = None

    def flush_se(
        self,
        store_path: str | Path,
        encoding: StoreEncoding,
        *,
        se_coefficients: np.ndarray | None = None,
        region_cells: int = DEFAULT_FLUSH_REGION_CELLS,
    ) -> None:
        """Add the SE half to the group `write_eaf_plane` created.

        The residual has to predict from the frequencies a reader will get
        back, so this encodes against the `eaf` plane already written rather
        than re-encoding the source values (ADR 0037, issue #228). The
        completion marker is written last, and `write_eaf_plane` deliberately
        does not write it: a group carrying only the frequency half is one a
        failed build left, not a finished component (issue #232).
        """
        out = Path(store_path) / RAGGED_ZARR_PATH
        root = store_arrays.open_group_for_write(out, "a")
        offsets_arr = np.asarray(self._offsets, dtype=np.int64)
        codec = StoreCodec(encoding)
        self._write_se_streamed(
            root, codec, encoding, offsets_arr, se_coefficients, self.n_associations, region_cells
        )
        root.attrs["layout"] = "ragged"
        root.attrs["completion_state"] = "observed_only"
        root.attrs["n_analyses"] = self.n_analyses
        root.attrs["n_associations"] = self.n_associations

    def flush(
        self,
        store_path: str | Path,
        encoding: StoreEncoding,
        *,
        eaf_baseline: np.ndarray | None = None,
        se_coefficients: np.ndarray | None = None,
        region_cells: int = DEFAULT_FLUSH_REGION_CELLS,
    ) -> None:
        """Write the whole CSR: the frequency half, then the SE half.

        One call for callers that hold every cell already -- the standalone
        Ragged builders and Reference Completion's overflow rebuild -- for whom
        splitting the write in two buys nothing. A Hybrid build calls the two
        halves itself, because its joint SE fit sits between them (issue #232).
        """
        self.write_eaf_plane(
            store_path, encoding, eaf_baseline=eaf_baseline, region_cells=region_cells
        )
        self.flush_se(
            store_path, encoding, se_coefficients=se_coefficients, region_cells=region_cells
        )


class RaggedCSRReader:
    """Read per-analysis associations from zarr CSR arrays."""

    def __init__(self, store_path: str | Path, encoding: StoreEncoding | None = None):
        path = Path(store_path) / RAGGED_ZARR_PATH
        self._root = store_arrays.open_group(path)
        self._offsets: zarr.Array = self._root["offsets"]
        self._variant_index: zarr.Array = self._root["variant_index"]
        self._z: zarr.Array = self._root["z"]
        self._se: zarr.Array = self._root["se"]  # shape metadata only; values use se_* below
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
        self._se_plane = RaggedSePlane.open(self._root, encoding, self._eaf_plane)

    @property
    def n_analyses(self) -> int:
        return int(self._root.attrs.get("n_analyses", array_length(self._offsets) - 1))

    @property
    def n_associations(self) -> int:
        return int(self._root.attrs.get("n_associations", array_length(self._variant_index)))

    def _span(self, analysis_index: int) -> tuple[int, int]:
        """The `[start, end)` slice of the flat arrays one Analysis occupies."""
        offsets = self._offsets[analysis_index : analysis_index + 2]
        return int(offsets[0]), int(offsets[1])

    def get_analysis(self, analysis_index: int) -> AnalysisAssociations:
        """Return (variant_index, z, se, eaf) arrays for one analysis. O(1) zarr reads."""
        start, end = self._span(analysis_index)
        if start == end:
            return AnalysisAssociations(
                variant_index=np.empty(0, dtype=np.int32),
                z=np.empty(0, dtype=np.float32),
                se=np.empty(0, dtype=np.float32),
                eaf=np.empty(0, dtype=np.float32),
            )
        # One read of the frequency region, shared by SE decoding and the
        # returned `eaf` column (#253), rather than `se_slice` reading the
        # plane and `eaf_slice` reading it again.
        eaf_read = self._eaf_plane.read_slice(start, end)
        return AnalysisAssociations(
            variant_index=self._variant_index[start:end],
            z=self.z_slice(start, end),
            se=self._se_plane.slice(
                start, end, analysis_index=analysis_index, eaf=eaf_read.values
            ),
            eaf=eaf_read.values,
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
        return self._codec.decode_z(self._z[start:end], positions=positions_flat(int(start)))

    def z_at(self, positions: np.ndarray) -> np.ndarray:
        """Decoded z at arbitrary flat CSR positions.

        Read through `oindex[positions]`, so the chunks a hit touches bound the
        work. Slicing the whole plane first (`self._z[:]`) decoded and held every
        z in the component -- about 6.2 GB on OGS-00011's Overflow -- for a
        handful of rows, on every off-axis PheWAS, lookup and region query
        (`HybridStoreQuery._overflow_by_variants`, #252).
        """
        positions = np.asarray(positions, dtype=np.int64)
        if len(positions) == 0:
            return np.empty(0, dtype=np.float32)
        return self._codec.decode_z(
            np.asarray(self._z.oindex[positions]), positions=positions_at(positions)
        )

    def z_all(self) -> np.ndarray:
        """Every decoded z, in flat CSR order."""
        return self.z_slice(0, array_length(self._z))

    def se_slice(
        self,
        start: int,
        end: int,
        *,
        eaf: np.ndarray | None = None,
        analysis_index: int | None = None,
    ) -> np.ndarray:
        """Decoded `se[start:end]`; callers may supply a pre-read frequency
        block and a known Analysis (#253)."""
        return self._se_plane.slice(start, end, analysis_index=analysis_index, eaf=eaf)

    def se_at(self, positions: np.ndarray, *, eaf: np.ndarray | None = None) -> np.ndarray:
        """Decoded SE at arbitrary CSR ordinals."""
        positions = np.asarray(positions, dtype=np.int64)
        offsets = np.asarray(self._offsets[:], dtype=np.int64)
        analyses = np.searchsorted(offsets[1:], positions, side="right").astype(np.int64)
        return self._se_plane.at(positions, analysis_index=analyses, eaf=eaf)

    def se_all(self) -> np.ndarray:
        return self.se_slice(0, array_length(self._se))

    @property
    def association_chunk(self) -> int:
        """Length of the association arrays' own inner chunk.

        The scan paths read in windows built from this, so a window never reads
        a chunk twice and peak memory is one window whatever the component's
        cell count (#252). Read from the array rather than restated, so a
        sharded 0.2.0 array's inner chunk is honoured too.
        """
        return max(1, int(self._variant_index.chunks[0]))

    @property
    def scan_window(self) -> int:
        """Elements one scan window spans: `SCAN_WINDOW_CHUNKS` inner chunks."""
        return self.association_chunk * SCAN_WINDOW_CHUNKS

    def variant_index_at(self, positions: np.ndarray) -> np.ndarray:
        """Variant indices at arbitrary flat CSR positions."""
        positions = np.asarray(positions, dtype=np.int64)
        if len(positions) == 0:
            return np.empty(0, dtype=np.int32)
        return np.asarray(self._variant_index.oindex[positions], dtype=np.int32)

    def variant_positions(self, wanted: np.ndarray) -> np.ndarray:
        """Flat CSR positions whose variant is in `wanted`, ascending.

        A windowed scan of every Analysis's segment, used where every Analysis
        may hold the variant and there is no index (#252). The window is a few
        inner chunks, so peak memory is a window and not the component. A
        lookup that knows its Analyses must use `segment_positions` instead,
        which searches the requested segment in O(log) chunk reads rather than
        scanning it.
        """
        wanted = np.unique(np.asarray(wanted, dtype=np.int32))
        if len(wanted) == 0:
            return np.empty(0, dtype=np.int64)
        lo, hi = 0, self.n_associations
        window = self.scan_window
        parts: list[np.ndarray] = []
        single = int(wanted[0]) if len(wanted) == 1 else None
        for start in range(lo, hi, window):
            stop = min(start + window, hi)
            vi = np.asarray(self._variant_index[start:stop], dtype=np.int32)
            if single is not None:
                # One wanted variant is the common off-axis PheWAS case, and a
                # direct compare is 16x fewer operations than a searchsorted.
                hit = vi == single
            else:
                pos = np.searchsorted(wanted, vi)
                in_bounds = pos < len(wanted)
                hit = np.zeros(len(vi), dtype=bool)
                hit[in_bounds] = wanted[pos[in_bounds]] == vi[in_bounds]
            if hit.any():
                parts.append(np.where(hit)[0].astype(np.int64) + start)
        if not parts:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(parts)

    def _scan_chunk(self, chunk: int, cache: dict[int, np.ndarray], total: int) -> np.ndarray:
        """One inner chunk of `variant_index`, read once and kept for the search.

        A chunk read decompresses the whole inner chunk whatever the element
        asked for, so caching it is what makes a binary search cost O(log)
        chunk reads rather than O(log) re-reads.
        """
        cached = cache.get(chunk)
        if cached is None:
            size = self.association_chunk
            start = chunk * size
            cached = np.asarray(
                self._variant_index[start : min(start + size, total)], dtype=np.int32
            )
            cache[chunk] = cached
        return cached

    def _segment_tail(
        self, chunk: int, lo: int, hi: int, cache: dict[int, np.ndarray], total: int
    ) -> int:
        """The last row of `chunk` that belongs to the segment `[lo, hi)`."""
        data = self._scan_chunk(chunk, cache, total)
        size = self.association_chunk
        offset = min(hi, (chunk + 1) * size) - 1 - chunk * size
        return int(data[offset])

    def _chunk_lower_bound(
        self, lo: int, hi: int, target: int, cache: dict[int, np.ndarray], total: int
    ) -> int:
        """First chunk in the segment whose last segment row is >= `target`.

        The halving touches O(log) chunks, which is the whole point of the
        search: the segment is never read whole.
        """
        size = self.association_chunk
        low, high = lo // size, (hi - 1) // size
        while low < high:
            mid = (low + high) // 2
            if self._segment_tail(mid, lo, hi, cache, total) < target:
                low = mid + 1
            else:
                high = mid
        return low

    def _run_end(
        self, start: int, hi: int, target: int, cache: dict[int, np.ndarray], total: int
    ) -> int:
        """End of the run of `target` starting at `start`, across chunk boundaries."""
        size = self.association_chunk
        end = start
        while end < hi:
            here = end // size
            here_data = self._scan_chunk(here, cache, total)
            here_off = end - here * size
            limit = min(hi - here * size, len(here_data))
            run_end = here_off + int(
                np.searchsorted(here_data[here_off:limit], target, side="right")
            )
            end = here * size + run_end
            if run_end < limit:
                break
        return end

    def segment_positions(self, wanted: np.ndarray, *, analysis_index: int) -> np.ndarray:
        """Flat CSR positions in one Analysis whose variant is in `wanted`.

        A genuine bounded binary search over the Analysis's sorted segment
        (#252). `_chunk_lower_bound` finds the chunk whose last *segment* row
        first reaches the target -- O(log) chunk reads -- and the target is
        located inside it with `searchsorted` on that chunk's segment part. A
        lookup is therefore proportional to the number of requested variants
        and not to the requested Analysis's size. The chunk's tail is taken
        from the segment, never from the chunk's full extent: the tail of a
        chunk may hold the next Analysis's rows, whose variant indices are
        unrelated.

        A duplicate variant (the writer accepts a non-decreasing sequence) is
        found by walking the equal run forward (`_run_end`); every chunk it
        spans is read at most once through the cache. The returned positions
        are ascending and in segment order, matching `analysis()` row for row.
        """
        wanted = np.unique(np.asarray(wanted, dtype=np.int32))
        if len(wanted) == 0:
            return np.empty(0, dtype=np.int64)
        lo, hi = self._span(analysis_index)
        if lo >= hi:
            return np.empty(0, dtype=np.int64)
        size = self.association_chunk
        total = array_length(self._variant_index)
        cache: dict[int, np.ndarray] = {}
        parts: list[np.ndarray] = []
        for value in wanted:
            target = int(value)
            base = self._chunk_lower_bound(lo, hi, target, cache, total) * size
            data = self._scan_chunk(base // size, cache, total)
            window_lo = max(lo - base, 0)
            window_hi = min(hi - base, len(data))
            offset = window_lo + int(
                np.searchsorted(data[window_lo:window_hi], target, side="left")
            )
            if offset >= window_hi or int(data[offset]) != target:
                continue
            start = base + offset
            parts.append(
                np.arange(start, self._run_end(start, hi, target, cache, total), dtype=np.int64)
            )
        if not parts:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(parts)

    @property
    def has_eaf(self) -> bool:
        """Whether this component stores EAF at all (ADR 0036)."""
        return self._eaf_plane.has_values

    def eaf_slice_read(self, start: int, end: int, *, want_imputed: bool = False) -> EafRead:
        """`eaf[start:end]` and the imputed mask, in one read (#253)."""
        return self._eaf_plane.read_slice(start, end, want_imputed=want_imputed)

    def eaf_slice(self, start: int, end: int) -> np.ndarray:
        """Decoded `eaf[start:end]`, or all-NaN when this store carries none."""
        return self.eaf_slice_read(start, end).values

    def eaf_at(self, positions: np.ndarray) -> np.ndarray:
        """EAF at arbitrary flat CSR positions; all-NaN when there is no array.

        For the scanning query paths, which already hold flat positions into
        the concatenated arrays and would otherwise pay `eaf_pairs`'
        per-Analysis searchsorted to recover what they already know.
        """
        return self._eaf_plane.at(positions)

    def eaf_at_read(self, positions: np.ndarray, *, want_imputed: bool = False) -> EafRead:
        """`eaf` and the imputed mask at flat CSR positions, in one read (#252).

        The variant-side scan paths read the frequency once and hand the same
        decoded array to SE decoding and to the result's `eaf` column, with
        #253's correctness rules: the decoded EAF carries the panel substitution
        on imputed cells, and the mask it was substituted under comes back with
        it so Association Status cannot be derived from a different alignment.
        """
        return self._eaf_plane.read_at(positions, want_imputed=want_imputed)

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
        # The plane's own answer, not `has_eaf`: a release carrying only the
        # panel's frequencies has no `eaf` array and still has frequencies to
        # report on its imputed cells (issue #113).
        if not self._eaf_plane.can_report_frequencies or len(variant_index) == 0:
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
