"""Dense top-hit harvest: scan the stored matrix for candidate cells and hand
them to the Top-Hit Index module (`opengwasdb.top_hits`), which owns the format."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from opengwasdb import top_hits
from opengwasdb.build.ordered_pool import ordered_map
from opengwasdb.encoding import DenseEafPlane, DenseSePlane, DenseZPlane, StoreEncoding
from opengwasdb.encoding.association_codec import require_parallel
from opengwasdb.encoding.timing import PhaseTimer, log_phase, log_progress
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.store import arrays as store_arrays
from opengwasdb.top_hits import TOP_HIT_CHUNK_SIZE, TOP_HIT_THRESHOLDS

log = logging.getLogger(__name__)


def write_top_hit_indexes(
    store_path: str | Path,
    rows: np.ndarray,
    cols: np.ndarray,
    z: np.ndarray,
    se: np.ndarray,
    thresholds: tuple[float, ...] = TOP_HIT_THRESHOLDS,
    imputed: np.ndarray | None = None,
    eaf: np.ndarray | None = None,
    chunk_size: int = TOP_HIT_CHUNK_SIZE,
) -> None:
    """Write ranked top-hit groups from pre-collected candidate cells.

    ``rows``/``cols``/``z``/``se`` describe candidate cells — every cell with
    ``|z| >= z_critical(max(thresholds))`` (the loosest tier). Extra cells below
    every threshold are harmless; each tier re-filters by its own ``z_critical``.
    Because candidates are a tiny fraction of a dense matrix, only the
    significant cells are ever held in memory — there is no full-matrix scan
    here. Cells are ordered by analysis index and canonical genomic position. When
    present, ``imputed`` is written in the same order so completed-store queries
    can label top hits without random reads back into ``data.zarr/imputed``.
    """

    columns: dict[str, np.ndarray] = {
        "variant_index": np.asarray(rows, dtype="uint32"),
        "analysis_index": np.asarray(cols, dtype="uint32"),
        "z": np.asarray(z, dtype="float32"),
        "se": np.asarray(se, dtype="float32"),
    }
    if imputed is not None:
        columns["imputed"] = np.asarray(imputed, dtype="uint8")
    if eaf is not None:
        columns["eaf"] = np.asarray(eaf, dtype="float32")

    root = store_arrays.open_group_for_write(Path(store_path) / "data.zarr", "a")
    top_hits.write_index(
        root, columns, thresholds, int(root["z"].shape[1]), chunk_size, store_arrays.compressor()
    )


def _concat_or_empty(parts: list[np.ndarray], dtype: type) -> np.ndarray:
    """One candidate array from its per-band pieces, empty when nothing passed."""
    return np.concatenate(parts) if parts else np.empty(0, dtype=dtype)


@dataclass
class _ScanContext:
    """Read-only state one scanned row band needs (see `se._MeasureContext`)."""

    z_plane: DenseZPlane
    se_plane: DenseSePlane
    imputed_arr: Any
    eaf_plane: DenseEafPlane
    n_variants: int
    band_rows: int
    loosest: float


_SCAN: _ScanContext | None = None


def _selected(values: Any, br: np.ndarray, bc: np.ndarray, dtype: Any) -> np.ndarray:
    """The candidate cells of a band's decoded plane, or an empty array."""
    return np.asarray(values[br, bc], dtype=dtype)


def _scan_one_band(r0: int) -> dict[str, np.ndarray]:
    """Every cell above the loosest tier in one row band, or empty arrays."""
    ctx = _SCAN
    if ctx is None:
        raise RuntimeError("top-hit scan worker ran without a scan context")
    r1 = min(r0 + ctx.band_rows, ctx.n_variants)
    z_band = ctx.z_plane.band(r0, r1)
    br, bc = np.where(np.abs(z_band) >= ctx.loosest)  # NaN compares False
    return {
        "rows": br.astype(np.int64) + r0,
        "cols": bc.astype(np.int64),
        "z": _selected(z_band, br, bc, np.float32),
        "se": _selected(ctx.se_plane.band(r0, r1), br, bc, np.float32),
        "imputed": (
            _selected(ctx.imputed_arr[r0:r1], br, bc, np.uint8)
            if ctx.imputed_arr is not None
            else np.empty(0, dtype=np.uint8)
        ),
        "eaf": (
            _selected(ctx.eaf_plane.band(r0, r1), br, bc, np.float32)
            if ctx.eaf_plane.can_report_frequencies
            else np.empty(0, dtype=np.float32)
        ),
    }


def _scan_candidates(
    z_plane: DenseZPlane,
    se_plane: DenseSePlane,
    imputed_arr: Any,
    eaf_plane: DenseEafPlane,
    n_variants: int,
    band_rows: int,
    loosest: float,
    n_workers: int,
) -> dict[str, list[np.ndarray]]:
    """Every cell clearing the loosest tier, gathered band by band.

    The matrix is never held whole: each band contributes only the cells that
    pass, which on a real store is a tiny fraction of it. Bands are independent
    and scan across ``n_workers``; the parent concatenates their harvests in
    band order, so the candidate order is the serial pass's.
    """
    global _SCAN
    parts: dict[str, list[np.ndarray]] = {
        name: [] for name in ("rows", "cols", "z", "se", "imputed", "eaf")
    }
    starts = range(0, n_variants, band_rows)
    n_bands = len(starts)
    _SCAN = _ScanContext(z_plane, se_plane, imputed_arr, eaf_plane, n_variants, band_rows, loosest)
    started = time.monotonic()
    try:
        with log_phase(log, "Top-hit scan"):
            bands = ordered_map(_scan_one_band, starts, n_workers)
            for index, band in enumerate(bands, start=1):
                for name, values in band.items():
                    if len(values):
                        parts[name].append(values)
                log_progress(
                    log, "Top-hit scan", index, n_bands, started,
                    every=max(1, n_bands // 20),
                )
    finally:
        _SCAN = None
    return parts


def _write_scan_harvest(
    store_path: Path,
    parts: dict[str, list[np.ndarray]],
    thresholds: tuple[float, ...],
    imputed_arr: Any,
    eaf_plane: DenseEafPlane,
    timer: PhaseTimer,
) -> None:
    """Concatenate a scan's per-band harvests and write every tier."""
    with log_phase(log, "Top-hit index write"):
        with timer.phase("top_hits.write"):
            write_top_hit_indexes(
                store_path,
                _concat_or_empty(parts["rows"], np.int64),
                _concat_or_empty(parts["cols"], np.int64),
                _concat_or_empty(parts["z"], np.float32),
                _concat_or_empty(parts["se"], np.float32),
                thresholds,
                imputed=(
                    _concat_or_empty(parts["imputed"], np.uint8)
                    if imputed_arr is not None
                    else None
                ),
                eaf=(
                    _concat_or_empty(parts["eaf"], np.float32)
                    if eaf_plane.can_report_frequencies
                    else None
                ),
            )


def build_top_hit_indexes(
    store_path: str | Path,
    thresholds: tuple[float, ...] = TOP_HIT_THRESHOLDS,
    encoding: StoreEncoding | None = None,
    timer: PhaseTimer | None = None,
    n_workers: int = 1,
) -> None:
    """(Re)build ranked top-hit arrays by scanning the stored dense matrix.

    Used by build paths that do not harvest hits inline, and to rebuild the index
    on an existing store. Scans ``z`` in row-bands (never the full matrix in RAM)
    and thresholds on the **stored** values -- decoded through the store's own
    codec, so the index matches exactly what a query reads back from ``z``
    (issue 046, ADR 0037). Collects only candidate cells
    (``|z| >= z_critical(loosest)``). The bands are independent and scan across
    ``n_workers``; the tier write itself is one sort and one zarr write and stays
    serial (issue #221).

    ``encoding`` is the store's declared plan; when omitted it is read from the
    release's manifest, never re-derived from the arrays.

    A caller that passes a ``PhaseTimer`` gets the scan and the write charged
    separately, which is how the re-encoding passes of a store-wide rewrite are
    told apart from this one (issue #144): 63.5 s on the migrated FinnGen R13
    pilot, against an inference that had put it at 75-89% of the whole.
    """
    timer = timer or PhaseTimer()
    store_path = Path(store_path)
    if encoding is None:
        encoding = StoreManifest.load(store_path).encoding
    root = store_arrays.open_group(store_path / "data.zarr")
    z_plane = DenseZPlane.open(root, encoding)
    imputed_arr = root["imputed"] if "imputed" in root else None
    eaf_plane = DenseEafPlane.open(root, encoding)
    log.info(
        "Top-hit index: scanning %d variants (n_workers=%d)",
        int(z_plane.array.shape[0]),
        n_workers,
    )
    with timer.phase("top_hits.scan"):
        parts = _scan_candidates(
            z_plane,
            DenseSePlane.open(root, encoding),
            imputed_arr,
            eaf_plane,
            int(z_plane.array.shape[0]),
            max(int(z_plane.array.chunks[0]), 250_000),
            top_hits.z_critical(max(thresholds)),
            n_workers,
        )
    _write_scan_harvest(store_path, parts, thresholds, imputed_arr, eaf_plane, timer)
    log.info("Top-hit phase timings:\n%s", timer.format_report())


def write_top_hit_indexes_for_store(
    store_path: str | Path,
    rows: np.ndarray,
    cols: np.ndarray,
    z: np.ndarray,
    se: np.ndarray,
    encoding: StoreEncoding,
    n_workers: int = 1,
) -> None:
    """Decode the candidates' frequencies, then write every tier.

    The pairing an inline build needs: a builder that wrote the frequency plane
    holds the candidate coordinates already, and the index must carry the same
    decoded values a query would read back (ADR 0040). Keeping the two calls
    together is what stops a build writing tiers with no `eaf` array. The two
    gathers and the tier write are each logged with their elapsed time, and the
    gathers spread their row chunks across ``n_workers`` (issue #221).

    ``rows``, ``cols``, ``z`` and ``se`` describe the same candidate cells, and
    one of another length is refused here, before anything is gathered: a
    residual plan discards the caller's ``se`` for the plane's, so a short one
    would otherwise be dropped without a word (issue #392).
    """
    require_parallel(
        (("rows", rows), ("cols", cols), ("z", z), ("se", se)),
        where="write_top_hit_indexes_for_store",
    )
    log.info("Collecting top-hit EAF in variant-row order for %d candidate cells", len(rows))
    with log_phase(log, "Top-hit EAF gather"):
        eaf = _collect_top_hit_eaf(store_path, rows, cols, encoding, n_workers)
    if encoding.se.is_residual:
        # The harvested `se` is what the source reported; the plane now holds a
        # residual of it. ADR 0040 asks the index to carry what a query reads
        # back, so re-read it through the plane rather than keep the un-encoded
        # value the band-writer happened to still be holding.
        log.info("Re-reading top-hit se through the residual plane")
        with log_phase(log, "Top-hit se re-read"):
            se = _collect_top_hit_se(store_path, rows, cols, encoding, n_workers)
    log.info("Writing top-hit index from %d harvested candidate cells", len(rows))
    with log_phase(log, "Top-hit index write"):
        write_top_hit_indexes(store_path, rows, cols, z, se, eaf=eaf)


def _collect_top_hit_eaf(
    store_path: str | Path,
    rows: np.ndarray,
    cols: np.ndarray,
    encoding: StoreEncoding,
    n_workers: int = 1,
) -> np.ndarray | None:
    """Collect candidate EAF values in row-chunk order for an inline build."""
    root = store_arrays.open_group(Path(store_path) / "data.zarr")
    plane = DenseEafPlane.open(root, encoding)
    if not plane.can_report_frequencies:
        return None
    return _gather_in_row_chunks(root, rows, cols, plane.band, n_workers)


def _collect_top_hit_se(
    store_path: str | Path,
    rows: np.ndarray,
    cols: np.ndarray,
    encoding: StoreEncoding,
    n_workers: int = 1,
) -> np.ndarray:
    """Collect candidate SE values in row-chunk order, decoded as a query sees them."""
    root = store_arrays.open_group(Path(store_path) / "data.zarr")
    return _gather_in_row_chunks(
        root, rows, cols, DenseSePlane.open(root, encoding).band, n_workers
    )


class _GatherTask(NamedTuple):
    """One row chunk's scattered candidate cells, in absolute output slots."""

    chunk_start: int
    chunk_stop: int
    slots: np.ndarray


_GATHER_BAND: Callable[[int, int], np.ndarray] | None = None
_GATHER_ROWS: np.ndarray | None = None
_GATHER_COLS: np.ndarray | None = None


def _gather_one_chunk(task: _GatherTask) -> tuple[np.ndarray, np.ndarray]:
    """Decode one row chunk and return the values at its candidate slots."""
    band, rows, cols = _GATHER_BAND, _GATHER_ROWS, _GATHER_COLS
    if band is None or rows is None or cols is None:
        raise RuntimeError("top-hit gather worker ran without a gather context")
    decoded = band(task.chunk_start, task.chunk_stop)
    offsets = rows[task.slots] - task.chunk_start
    return task.slots, np.asarray(decoded[offsets, cols[task.slots]], dtype=np.float32)


def _gather_in_row_chunks(
    root: Any,
    rows: np.ndarray,
    cols: np.ndarray,
    band: Callable[[int, int], np.ndarray],
    n_workers: int = 1,
) -> np.ndarray:
    """Gather `(rows[i], cols[i])` from a decoded plane, one row chunk at a time.

    Decoding a plane is a band operation; the candidate cells are scattered.
    Visiting each row chunk once keeps the read sequential and bounds peak
    memory at one band, which is what makes this affordable inline in a build.
    The chunks are independent, so they decode across ``n_workers``; each
    result lands at its own absolute slots, which makes the reduction a
    positional scatter rather than an order-sensitive fold.
    """
    global _GATHER_BAND, _GATHER_ROWS, _GATHER_COLS
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    out = np.empty(len(rows), dtype=np.float32)
    if len(rows) == 0:
        return out
    row_chunk = int(root["z"].chunks[0])
    n_variants = int(root["z"].shape[0])
    order = np.argsort(rows, kind="stable")
    sorted_rows = rows[order]
    tasks: list[_GatherTask] = []
    for chunk_start in np.unique((sorted_rows // row_chunk) * row_chunk):
        chunk_stop = min(int(chunk_start) + row_chunk, n_variants)
        lo = int(np.searchsorted(sorted_rows, chunk_start, side="left"))
        hi = int(np.searchsorted(sorted_rows, chunk_stop, side="left"))
        tasks.append(_GatherTask(int(chunk_start), chunk_stop, order[lo:hi]))
    _GATHER_BAND, _GATHER_ROWS, _GATHER_COLS = band, rows, cols
    started = time.monotonic()
    try:
        with log_phase(log, "Top-hit gather"):
            gathered = ordered_map(_gather_one_chunk, tasks, n_workers)
            for index, (slots, values) in enumerate(gathered, start=1):
                out[slots] = values
                log_progress(
                    log, "Top-hit gather", index, len(tasks), started,
                    every=max(1, len(tasks) // 20),
                )
    finally:
        _GATHER_BAND, _GATHER_ROWS, _GATHER_COLS = None, None, None
    return out
