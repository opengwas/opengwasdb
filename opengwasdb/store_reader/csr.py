"""Reading a Ragged association sequence: Zarr-backed Compressed Sparse Row storage (§11).

The structure of a Ragged component's sequence -- its offsets, spans, variant
index, imputed mask and pair search -- and the plane decodes beneath it. Writing
one is `opengwasdb.layouts.ragged.zarr_csr`'s.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from opengwasdb.store_reader import arrays as store_arrays
from opengwasdb.store_reader.arrays import array_length
from opengwasdb.store_reader.encoding.codec import (
    StoreDecoder,
    ZOverflowTable,
    positions_at,
    positions_flat,
)
from opengwasdb.store_reader.encoding.plan import StoreEncoding
from opengwasdb.store_reader.encoding.planes import EafRead, RaggedEafPlane, RaggedSePlane
from opengwasdb.store_reader.manifest import StoreManifest

RAGGED_ZARR_PATH = "data.zarr/ragged"

#: Association-array chunks one scan window spans. A window of a single chunk
#: makes a whole-store scan issue one zarr read per chunk, which measured ~5x
#: slower than the batched whole-array read it replaces on OGS-00011's
#: Overflow (132.9 s against 27.3 s for off-axis PheWAS, #252); a handful of
#: chunks amortises the per-read overhead while peak memory stays bounded by
#: the window (8 x 200,000 int32 = 6.4 MB) and not by the array.
SCAN_WINDOW_CHUNKS = 8


class RaggedCSRReader:
    """An association sequence's structure (§11), and the plane decodes beneath it.

    What is public is structure, for code that walks the sequence itself: the
    offsets and spans, the variant index, the imputed mask and the pair search.
    Associations are read through `opengwasdb.store_reader.associations.CsrReader`,
    which applies the presence rule (ADR 0070). The decode members here serve it
    and one other caller, the Ragged Top-Hit harvest (`layouts/ragged/top_hits.py`).
    """

    def __init__(self, store_path: str | Path, encoding: StoreEncoding | None = None):
        path = Path(store_path) / RAGGED_ZARR_PATH
        self._root = store_arrays.open_group(path)
        self._offsets: zarr.Array = self._root["offsets"]
        self._offsets_cache: np.ndarray | None = None
        self._variant_index: zarr.Array = self._root["variant_index"]
        self._z: zarr.Array = self._root["z"]
        # The plan is read from the release's manifest, never inferred from the
        # array's dtype: a store that disagrees with its own manifest must fail
        # validation, not decode as whatever the bytes happen to look like.
        if encoding is None:
            encoding = StoreManifest.load(Path(store_path)).encoding
        self._codec = StoreDecoder(encoding, z_overflow=ZOverflowTable.open(self._root))
        # Every EAF read goes through the plane, which gathers the per-variant
        # baseline (and, on a Reference-Completed release, the panel frequency
        # for imputed cells) onto the cells being read. Absent on stores whose
        # sources report no frequency at all -- those read back as all-NaN.
        # Opened once: a membership test or `root[name]` reads the array's
        # metadata from the store every time (#244).
        self._imputed_array: zarr.Array | None = (
            self._root["imputed"] if "imputed" in self._root else None
        )
        self._eaf_plane = RaggedEafPlane.open(
            self._root, encoding, imputed=self._imputed_array
        )
        self._se_plane = RaggedSePlane.open(self._root, encoding, self._eaf_plane)
        # The plan, not the array's presence, says whether there is a plane: one
        # declared and missing is a damaged release, not a release without N.
        if encoding.n is not None and "n" not in self._root:
            raise ValueError(
                f"{path}: the manifest declares an n plane but the component carries no "
                "n array (ADR 0067)"
            )
        self._n: zarr.Array | None = self._root["n"] if encoding.n is not None else None

    @property
    def n_analyses(self) -> int:
        return int(self._root.attrs.get("n_analyses", array_length(self._offsets) - 1))

    @property
    def n_associations(self) -> int:
        return int(self._root.attrs.get("n_associations", array_length(self._variant_index)))

    # ── structure: where the associations sit, for code that walks the sequence ──

    @property
    def offsets(self) -> np.ndarray:
        """Every Analysis's start in the flat sequence, plus its end (int64).

        One entry per Analysis, not per association, so it is the one array a
        caller may hold whole; read once and kept.
        """
        if self._offsets_cache is None:
            self._offsets_cache = np.asarray(self._offsets[:], dtype=np.int64)
        return self._offsets_cache

    @property
    def sequence_length(self) -> int:
        """How many rows the flat association arrays hold."""
        return array_length(self._variant_index)

    def span(self, analysis_index: int) -> tuple[int, int]:
        """The `[start, end)` slice of the flat arrays one Analysis occupies."""
        offsets = self._offsets[analysis_index : analysis_index + 2]
        return int(offsets[0]), int(offsets[1])

    def variant_index(self, start: int, end: int) -> np.ndarray:
        """`variant_index[start:end]`, as int32."""
        return np.asarray(self._variant_index[int(start) : int(end)], dtype=np.int32)

    def analysis_index_at(self, positions: np.ndarray) -> np.ndarray:
        """The Analysis each flat position belongs to, from the offsets."""
        positions = np.asarray(positions, dtype=np.int64)
        return np.searchsorted(self.offsets[1:], positions, side="right").astype(np.int32)

    @property
    def has_imputed(self) -> bool:
        """Whether the sequence carries an imputed mask."""
        return self._imputed_array is not None

    def imputed_at(self, positions: np.ndarray) -> np.ndarray:
        """The imputed mask at flat positions, as stored (uint8)."""
        positions = np.asarray(positions, dtype=np.int64)
        return np.asarray(self._require_imputed().oindex[positions], dtype=np.uint8)

    def _require_imputed(self) -> zarr.Array:
        if self._imputed_array is None:
            raise ValueError("this sequence carries no imputed mask; ask has_imputed first")
        return self._imputed_array

    @property
    def can_report_frequencies(self) -> bool:
        """Whether an EAF read can return anything but NaN (issue #113)."""
        return self._eaf_plane.can_report_frequencies

    def z_slice(self, start: int, end: int) -> np.ndarray:
        """Decoded `z[start:end]`, as `CsrReader` reads it."""
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

        A duplicate variant is found by walking the equal run forward
        (`_run_end`); every chunk it spans is read at most once through the
        cache. The writer accepts a non-decreasing sequence, but a release that
        held a repeat inside one Analysis's segment is invalid (§11), so this
        scan is a defensive read of a store validation refuses, not a shape a
        builder produces. The returned positions are ascending and in segment
        order, matching `analysis()` row for row.
        """
        wanted = np.unique(np.asarray(wanted, dtype=np.int32))
        if len(wanted) == 0:
            return np.empty(0, dtype=np.int64)
        lo, hi = self.span(analysis_index)
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
        0036, or from sources reporting no frequency. A pair whose variant is
        absent from that Analysis stays NaN rather than silently taking a
        neighbour's frequency (`pair_positions`).
        """
        out = np.full(len(variant_index), np.nan, dtype=np.float32)
        # The plane's own answer, not `has_eaf`: a release carrying only the
        # panel's frequencies has no `eaf` array and still has frequencies to
        # report on its imputed cells (issue #113).
        if not self._eaf_plane.can_report_frequencies or len(variant_index) == 0:
            return out
        # Resolve every pair to a flat CSR position first, then decode once:
        # the plane needs the position to resolve an exception cell, and a
        # per-Analysis decode would gather the baseline slice by slice.
        slots, positions = self.pair_positions(variant_index, analysis_index)
        if len(slots):
            out[slots] = self.eaf_at(positions)
        return out

    @property
    def has_n(self) -> bool:
        """Whether this component stores per-association sample sizes (ADR 0067)."""
        return self._n is not None

    def _require_n(self) -> zarr.Array:
        if self._n is None:
            raise ValueError(
                "this component stores no n plane; a caller asks has_n before reading one"
            )
        return self._n

    def n_slice(self, start: int, end: int) -> np.ndarray:
        """Decoded `n[start:end]`, NaN where the build was given no N for the pair."""
        return self._codec.decode_n(np.asarray(self._require_n()[int(start) : int(end)]))

    def n_at(self, positions: np.ndarray) -> np.ndarray:
        """Decoded `n` at arbitrary flat CSR positions, read through `oindex`."""
        plane = self._require_n()
        positions = np.asarray(positions, dtype=np.int64)
        if len(positions) == 0:
            return np.empty(0, dtype=np.float64)
        return self._codec.decode_n(np.asarray(plane.oindex[positions]))

    def n_pairs(self, variant_index: np.ndarray, analysis_index: np.ndarray) -> np.ndarray:
        """`n` for elementwise (variant, analysis) pairs, gathered as `eaf_pairs` is.

        The top-hit tiers carry no `n` (ADR 0067), so an indexed top-hit query
        resolves each hit back to its CSR cell. A pair the component does not
        hold is NaN.
        """
        self._require_n()
        out = np.full(len(variant_index), np.nan, dtype=np.float64)
        slots, positions = self.pair_positions(variant_index, analysis_index)
        if len(slots):
            out[slots] = self.n_at(positions)
        return out

    def pair_positions(
        self, variant_index: np.ndarray, analysis_index: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """`(slots, flat positions)` for the pairs this component holds.

        Each Analysis's CSR slice is sorted by variant_index (every builder sorts
        before writing), so one `searchsorted` per distinct Analysis resolves its
        pairs. `slots` index the request; a pair whose variant is absent from its
        Analysis has no slot, so a caller leaves it missing.
        """
        variant_index = np.asarray(variant_index)
        analysis_index = np.asarray(analysis_index)
        offsets = self._offsets[:]
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
        if not slots:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        return np.concatenate(slots), np.concatenate(found)
