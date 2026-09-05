"""Ragged top-hit index builder — mirrors opengwasdb/layouts/dense/top_hits.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc

from opengwasdb.encoding import StoreEncoding
from opengwasdb.layouts.dense.constants import TOP_HIT_THRESHOLDS
from opengwasdb.layouts.dense.top_hits import (
    TOP_HIT_CHUNK_SIZE,
    threshold_key,
    write_threshold_tier,
)
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRReader


def build_ragged_top_hit_indexes(
    store_path: str | Path,
    thresholds: tuple[float, ...] = TOP_HIT_THRESHOLDS,
    encoding: StoreEncoding | None = None,
) -> None:
    """Build ranked top-hit arrays for each configured p-value threshold.

    Writes to data.zarr/top_hits/<key>/ using the same schema as the dense
    builder so the query facade and validator can share one code path.

    Thresholding is on the **stored** z, decoded through the store's own codec:
    an index built from unrounded values would name hits the store itself
    contradicts (issue 046). ``encoding`` is supplied by a builder that has not
    written its manifest yet; otherwise it is read from the release.
    """
    store_path = Path(store_path)
    csr = RaggedCSRReader(store_path, encoding)

    offsets = csr._offsets[:]
    vi_all = csr._variant_index[:].astype(np.int32)
    z_all = csr.z_all()
    se_all = csr._se[:].astype(np.float32)
    n_analyses = len(offsets) - 1
    imputed_all = (
        csr._root["imputed"][:].astype(np.uint8) if "imputed" in csr._root else None
    )
    eaf_all = (
        csr.eaf_at(np.arange(len(vi_all), dtype=np.int64))
        if csr._eaf_plane.can_report_frequencies else None
    )

    # Derive analysis_index for every association via searchsorted on CSR offsets.
    # offsets[i+1] is the exclusive end of analysis i → searchsorted(offsets[1:], pos) gives i.
    positions = np.arange(len(vi_all), dtype=np.int64)
    analysis_indices = np.searchsorted(offsets[1:], positions, side="right").astype(np.int32)

    abs_z = np.abs(z_all)
    root = zarr.open_group(str(store_path / "data.zarr"), mode="a")
    top = root.require_group("top_hits")
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

    # The same parallel-array contract the dense builder writes, so both layouts
    # produce one schema and the facade and validator keep one code path.
    columns: dict[str, np.ndarray] = {
        "variant_index": vi_all,
        "analysis_index": analysis_indices,
        "z": z_all,
        "se": se_all,
    }
    if imputed_all is not None:
        columns["imputed"] = imputed_all
    if eaf_all is not None:
        columns["eaf"] = eaf_all

    for threshold in thresholds:
        n_hits = write_threshold_tier(
            top, threshold, columns, abs_z, n_analyses, TOP_HIT_CHUNK_SIZE, compressor
        )
        print(f"  {threshold_key(threshold)}: {n_hits:,} hits")

    top.attrs["thresholds"] = list(thresholds)
