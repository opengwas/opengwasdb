"""Ragged top-hit harvest: scan the CSR for candidate cells and hand them to the
Top-Hit Index module (`opengwasdb.top_hits`), which owns the format."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from opengwasdb import top_hits
from opengwasdb.encoding import StoreEncoding
from opengwasdb.encoding.timing import log_phase
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRReader
from opengwasdb.store.arrays import (
    open_group_for_write,
)
from opengwasdb.top_hits import TOP_HIT_THRESHOLDS

log = logging.getLogger(__name__)

#: Cells one top-hit scan slice covers. A slice holds its decoded ``z``
#: (float32), a boolean keep mask and the Analysis indices the offsets assign
#: its cells (int32) -- about 11 bytes a cell -- so 2**21 is roughly a 25 MiB
#: working set whatever the component's cell count (issue #233).
DEFAULT_TOP_HIT_SCAN_CELLS = 1 << 21

#: The dtype each candidate array is gathered as. One table, so an optional
#: array (`imputed`, `eaf`) cannot acquire a different dtype from the required
#: ones by being gathered at a separate site.
_CANDIDATE_DTYPES = {
    "variant_index": "int32",
    "analysis_index": "int32",
    "z": "float32",
    "se": "float32",
    "imputed": "uint8",
    "eaf": "float32",
}


def _slice_analysis_indices(offsets: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """The Analysis each cell of flat CSR range ``[lo, hi)`` belongs to.

    Derived from the offsets' spans rather than by searching a materialised
    position range: a slice is bounded, so repeating the Analysis index over
    the spans it crosses costs O(analyses touched), not O(cells) (issue #233).
    """
    first = int(np.searchsorted(offsets, lo, side="right")) - 1
    parts: list[np.ndarray] = []
    for analysis in range(max(first, 0), len(offsets) - 1):
        start, stop = int(offsets[analysis]), int(offsets[analysis + 1])
        if start >= hi:
            break
        head, tail = max(lo, start), min(hi, stop)
        if tail > head:
            parts.append(np.full(tail - head, analysis, dtype=np.int32))
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int32)


def _gather_slice_candidates(
    csr: RaggedCSRReader,
    offsets: np.ndarray,
    lo: int,
    hi: int,
    loosest: float,
    has_imputed: bool,
    has_eaf: bool,
) -> dict[str, np.ndarray] | None:
    """The candidate cells of one flat CSR slice, or None when none pass.

    Only the slice's decoded ``z`` is held, and only the passing cells' companion
    arrays are gathered -- the bounded working set issue #233 asks for.
    """
    z = csr.z_slice(lo, hi)
    keep = np.abs(z) >= loosest
    if not keep.any():
        return None
    positions = np.flatnonzero(keep).astype(np.int64) + lo
    columns: dict[str, np.ndarray] = {
        "variant_index": csr.variant_index_at(positions),
        "analysis_index": _slice_analysis_indices(offsets, lo, hi)[keep],
        "z": z[keep],
        "se": csr.se_at(positions),
    }
    if has_imputed:
        columns["imputed"] = csr.imputed_at(positions)
    if has_eaf:
        columns["eaf"] = csr.eaf_at(positions)
    return columns


def _concat_candidate_parts(
    parts: dict[str, list[np.ndarray]],
) -> dict[str, np.ndarray]:
    """The gathered per-slice arrays, one candidate array per column."""
    columns: dict[str, np.ndarray] = {}
    for name, values in parts.items():
        columns[name] = (
            np.concatenate(values) if values else np.empty(0, dtype=_CANDIDATE_DTYPES[name])
        )
    return columns


def _collect_ragged_candidates(
    csr: RaggedCSRReader,
    thresholds: tuple[float, ...],
    slice_cells: int,
) -> tuple[dict[str, np.ndarray], int]:
    """Every cell clearing the loosest tier, gathered slice by slice.

    The component is never decoded whole. Each slice contributes only the cells
    whose decoded ``|z|`` clears ``top_hits.z_critical(max(thresholds))`` -- a tiny
    fraction of a real component -- and the downstream tier write sorts that
    candidate set, not a plane (issue #233).

    Analysis indices come from the CSR offsets (``_slice_analysis_indices``),
    and the optional ``imputed``/``eaf`` columns are gathered only for the cells
    that pass, so a no-frequency component and a Reference-Completed component
    with an ``imputed`` column keep exactly the columns the materialising loader
    produced.
    """
    total = csr.sequence_length
    offsets = csr.offsets
    n_analyses = len(offsets) - 1
    loosest = top_hits.z_critical(max(thresholds))
    has_imputed = csr.has_imputed
    has_eaf = csr.can_report_frequencies

    parts: dict[str, list[np.ndarray]] = {
        "variant_index": [],
        "analysis_index": [],
        "z": [],
        "se": [],
    }
    if has_imputed:
        parts["imputed"] = []
    if has_eaf:
        parts["eaf"] = []

    step = max(1, int(slice_cells))
    for lo in range(0, total, step):
        slice_columns = _gather_slice_candidates(
            csr, offsets, lo, min(lo + step, total), loosest, has_imputed, has_eaf
        )
        if slice_columns is None:
            continue
        for name, values in slice_columns.items():
            parts[name].append(values)

    columns = _concat_candidate_parts(parts)
    return columns, n_analyses


def build_ragged_top_hit_indexes(
    store_path: str | Path,
    thresholds: tuple[float, ...] = TOP_HIT_THRESHOLDS,
    encoding: StoreEncoding | None = None,
    n_workers: int = 1,
    slice_cells: int = DEFAULT_TOP_HIT_SCAN_CELLS,
) -> None:
    """Build ranked top-hit arrays for each configured p-value threshold.

    Writes to data.zarr/top_hits/<key>/ through `opengwasdb.top_hits`, the one
    owner of the format every layout shares.

    Thresholding is on the **stored** z, decoded through the store's own codec:
    an index built from unrounded values would name hits the store itself
    contradicts (issue 046). ``encoding`` is supplied by a builder that has not
    written its manifest yet; otherwise it is read from the release.

    The CSR is scanned in slices of ``slice_cells``, never decoded whole
    (issue #233): each slice contributes only the cells clearing the loosest
    tier, and the tier write below sorts that candidate set. Analysis indices
    are derived from the CSR offsets rather than by searching a materialised
    position range. Each step logs its start and elapsed time (issue #221).
    ``n_workers`` is accepted for a uniform build surface but the pass is left
    serial: the slices feed one candidate sort and are I/O-bound, not
    CPU-bound (see issue #221's profiling note).
    """
    store_path = Path(store_path)
    log.info(
        "Ragged top-hit index: scanning CSR in slices (n_workers=%d, serial, slice_cells=%d)",
        n_workers,
        slice_cells,
    )
    csr = RaggedCSRReader(store_path, encoding)
    with log_phase(log, "Ragged top-hit scan"):
        columns, n_analyses = _collect_ragged_candidates(csr, thresholds, slice_cells)

    root = open_group_for_write(store_path / "data.zarr", "a")
    top_hits.write_index(root, columns, thresholds, n_analyses)
