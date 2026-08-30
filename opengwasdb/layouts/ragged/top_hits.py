"""Ragged top-hit index builder — mirrors opengwasdb/layouts/dense/top_hits.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc

from opengwasdb.encoding import StoreEncoding
from opengwasdb.layouts.dense.constants import TOP_HIT_THRESHOLDS
from opengwasdb.layouts.dense.top_hits import TOP_HIT_CHUNK_SIZE, threshold_key, z_critical
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

    # Derive analysis_index for every association via searchsorted on CSR offsets.
    # offsets[i+1] is the exclusive end of analysis i → searchsorted(offsets[1:], pos) gives i.
    positions = np.arange(len(vi_all), dtype=np.int64)
    analysis_indices = np.searchsorted(offsets[1:], positions, side="right").astype(np.int32)

    abs_z = np.abs(z_all)
    root = zarr.open_group(str(store_path / "data.zarr"), mode="a")
    top = root.require_group("top_hits")
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

    for threshold in thresholds:
        keep = abs_z >= z_critical(threshold)

        kept_vi = vi_all[keep]
        kept_ai = analysis_indices[keep]
        kept_abs = abs_z[keep]
        kept_z = z_all[keep]
        kept_se = se_all[keep]
        # Compute float64 p-values only for the survivors
        from scipy.special import erfc  # type: ignore[import-untyped]

        kept_p = erfc(kept_abs.astype("float64") / np.sqrt(2.0))
        kept_imputed = None if imputed_all is None else imputed_all[keep]

        # CSR is analysis-major; make the genomic ordering contract explicit.
        order = np.lexsort((kept_vi, kept_ai))
        kept_vi = kept_vi[order]
        kept_ai = kept_ai[order]
        kept_abs = kept_abs[order]
        kept_z = kept_z[order]
        kept_se = kept_se[order]
        kept_p = kept_p[order]
        kept_imputed = None if kept_imputed is None else kept_imputed[order]

        key = threshold_key(threshold)
        if key in top:
            del top[key]
        group = top.create_group(key)
        analysis_offsets = np.empty(n_analyses + 1, dtype="uint64")
        analysis_offsets[0] = 0
        np.cumsum(
            np.bincount(kept_ai, minlength=n_analyses),
            dtype=np.uint64,
            out=analysis_offsets[1:],
        )
        chunk = max(1, min(len(kept_vi), TOP_HIT_CHUNK_SIZE))
        group.create_dataset(
            "analysis_offsets", data=analysis_offsets, chunks=(len(analysis_offsets),),
            compressor=compressor, dtype="uint64",
        )

        for name, data, dtype in [
            ("variant_index", kept_vi, "uint32"),
            ("analysis_index", kept_ai, "uint32"),
            ("abs_z", kept_abs, "float32"),
            ("z", kept_z, "float32"),
            ("se", kept_se, "float32"),
            ("p_value", kept_p, "float64"),
        ]:
            group.create_dataset(
                name,
                data=data.astype(dtype),
                chunks=(chunk,),
                compressor=compressor,
                dtype=dtype,
            )
        if kept_imputed is not None:
            group.create_dataset(
                "imputed", data=kept_imputed, chunks=(chunk,),
                compressor=compressor, dtype="uint8",
            )
        group.attrs["threshold"] = threshold
        group.attrs["order"] = "analysis_index,variant_index"
        print(f"  {key}: {len(kept_vi):,} hits")

    top.attrs["thresholds"] = list(thresholds)
