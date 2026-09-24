"""Ragged top-hit index builder — mirrors opengwasdb/layouts/dense/top_hits.py."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc

from opengwasdb.encoding import StoreEncoding
from opengwasdb.encoding.timing import log_phase
from opengwasdb.layouts.dense.constants import TOP_HIT_THRESHOLDS
from opengwasdb.layouts.dense.top_hits import (
    TOP_HIT_CHUNK_SIZE,
    threshold_key,
    write_threshold_tier,
)
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRReader

log = logging.getLogger(__name__)


def _read_ragged_columns(
    store_path: Path, encoding: StoreEncoding | None
) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    """Decode every CSR association into the dense builder's parallel columns."""
    csr = RaggedCSRReader(store_path, encoding)
    offsets = csr._offsets[:]
    vi_all = csr._variant_index[:].astype(np.int32)
    z_all = csr.z_all()
    se_all = csr.se_all()
    n_analyses = len(offsets) - 1
    # Derive analysis_index for every association via searchsorted on CSR offsets.
    # offsets[i+1] is the exclusive end of analysis i -> searchsorted(offsets[1:], pos) gives i.
    positions = np.arange(len(vi_all), dtype=np.int64)
    analysis_indices = np.searchsorted(offsets[1:], positions, side="right").astype(np.int32)
    columns: dict[str, np.ndarray] = {
        "variant_index": vi_all,
        "analysis_index": analysis_indices,
        "z": z_all,
        "se": se_all,
    }
    if "imputed" in csr._root:
        columns["imputed"] = csr._root["imputed"][:].astype(np.uint8)
    if csr._eaf_plane.can_report_frequencies:
        columns["eaf"] = csr.eaf_at(np.arange(len(vi_all), dtype=np.int64))
    return columns, np.abs(z_all), n_analyses


def build_ragged_top_hit_indexes(
    store_path: str | Path,
    thresholds: tuple[float, ...] = TOP_HIT_THRESHOLDS,
    encoding: StoreEncoding | None = None,
    n_workers: int = 1,
) -> None:
    """Build ranked top-hit arrays for each configured p-value threshold.

    Writes to data.zarr/top_hits/<key>/ using the same schema as the dense
    builder so the query facade and validator can share one code path.

    Thresholding is on the **stored** z, decoded through the store's own codec:
    an index built from unrounded values would name hits the store itself
    contradicts (issue 046). ``encoding`` is supplied by a builder that has not
    written its manifest yet; otherwise it is read from the release.

    Each step logs its start and elapsed time (issue #221). ``n_workers`` is
    accepted for a uniform build surface but the pass is left serial: it
    decodes the flat CSR arrays whole and reduces them with one ``lexsort`` per
    tier, so there is no independent row-chunk work to spread (see issue #221's
    profiling note).
    """
    store_path = Path(store_path)
    log.info("Ragged top-hit index: reading CSR arrays (n_workers=%d, serial)", n_workers)
    with log_phase(log, "Ragged top-hit read"):
        columns, abs_z, n_analyses = _read_ragged_columns(store_path, encoding)

    root = zarr.open_group(str(store_path / "data.zarr"), mode="a")
    top = root.require_group("top_hits")
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    # The same parallel-array contract the dense builder writes, so both layouts
    # produce one schema and the facade and validator keep one code path.
    _write_ragged_tiers(top, thresholds, columns, abs_z, n_analyses, compressor)
    top.attrs["thresholds"] = list(thresholds)


def _write_ragged_tiers(
    top: zarr.Group,
    thresholds: tuple[float, ...],
    columns: dict[str, np.ndarray],
    abs_z: np.ndarray,
    n_analyses: int,
    compressor: Blosc,
) -> None:
    """Select, rank and write one tier per threshold, logging each."""
    for threshold in thresholds:
        with log_phase(log, f"Ragged top-hit tier {threshold_key(threshold)}"):
            n_hits = write_threshold_tier(
                top, threshold, columns, abs_z, n_analyses, TOP_HIT_CHUNK_SIZE, compressor
            )
        log.info("%s: %d hits", threshold_key(threshold), n_hits)
