"""Dense top-hit index builder."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc
from scipy.special import erfc, erfcinv  # type: ignore[import-untyped]

from opengwasdb.encoding import DenseZPlane, StoreEncoding
from opengwasdb.layouts.dense.constants import TOP_HIT_THRESHOLDS
from opengwasdb.model.analyses import TOP_HIT_COUNT_COLUMNS
from opengwasdb.model.manifest import StoreManifest

TOP_HIT_CHUNK_SIZE = 16_384

# Positional pairing of TOP_HIT_THRESHOLDS with the analyses.tsv column each
# tier persists to (model.analyses.TOP_HIT_COUNT_COLUMNS). A dict, not a zip
# over the caller's own `thresholds`, so read_top_hit_counts accepts any
# subset of TOP_HIT_THRESHOLDS rather than silently requiring exactly three.
_THRESHOLD_COLUMNS = dict(zip(TOP_HIT_THRESHOLDS, TOP_HIT_COUNT_COLUMNS, strict=True))


class DenseTopHitReader:
    """Address one threshold tier without exposing its physical arrays."""

    def __init__(self, group: zarr.Group):
        self.group = group

    def bounds(self, analysis_index: int | None) -> tuple[int, int]:
        if analysis_index is None:
            return 0, int(self.group["z"].shape[0])
        offsets = self.group["analysis_offsets"]
        if analysis_index < 0 or analysis_index + 1 >= int(offsets.shape[0]):
            return 0, 0
        pair = offsets[analysis_index : analysis_index + 2]
        return int(pair[0]), int(pair[1])

    def read(self, name: str, bounds: tuple[int, int], dtype: str) -> np.ndarray:
        start, stop = bounds
        return np.asarray(self.group[name][start:stop], dtype=dtype)


def read_top_hit_counts(
    store_path: str | Path,
    n_analyses: int,
    thresholds: tuple[float, ...] = TOP_HIT_THRESHOLDS,
) -> dict[str, list[int]]:
    """Per-Analysis hit counts for each threshold tier, from an already-built
    top-hit index (``build_top_hit_indexes``/``write_top_hit_indexes``/
    ``build_ragged_top_hit_indexes`` -- all three share this schema). Keyed by
    the ``analyses.tsv`` column each tier persists to (ADR 0032), in
    ``analysis_index`` order.

    ``n_analyses`` is needed for the fallback path below, not just for
    sizing a zero-filled result: real pre-issue-#22-era stores (e.g. ukb-b)
    have a top-hit index that predates the ``analysis_offsets`` array this
    function otherwise reads directly, so it falls back to counting the
    flat ``analysis_index`` array by hand for those.
    """
    root = zarr.open_group(str(Path(store_path) / "data.zarr"), mode="r")
    top = root["top_hits"]
    counts: dict[str, list[int]] = {}
    for threshold in thresholds:
        column = _THRESHOLD_COLUMNS[threshold]
        group = top[threshold_key(threshold)]
        if "analysis_offsets" in group:
            offsets = np.asarray(group["analysis_offsets"], dtype=np.int64)
            counts[column] = (offsets[1:] - offsets[:-1]).tolist()
        else:
            analysis_index = np.asarray(group["analysis_index"], dtype=np.int64)
            counts[column] = np.bincount(analysis_index, minlength=n_analyses)[:n_analyses].tolist()
    return counts


def threshold_key(threshold: float) -> str:
    """Stable Zarr group key for a p-value threshold."""

    return f"p_{threshold:.0e}".replace("-", "_").replace("+", "")


def z_critical(threshold: float) -> float:
    """|z| cutoff equivalent to two-sided p <= threshold.

    The stored p-value is ``p = erfc(|z| / sqrt(2))`` (see the query facade and
    the old scalar ``p_value_from_z``), so ``p <= threshold`` is exactly
    ``|z| >= sqrt(2) * erfcinv(threshold)``. Thresholding on ``|z|`` lets the
    hot path avoid computing a p-value for every cell.
    """

    return math.sqrt(2.0) * float(erfcinv(threshold))


def write_top_hit_indexes(
    store_path: str | Path,
    rows: np.ndarray,
    cols: np.ndarray,
    z: np.ndarray,
    se: np.ndarray,
    thresholds: tuple[float, ...] = TOP_HIT_THRESHOLDS,
    imputed: np.ndarray | None = None,
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

    rows = np.asarray(rows, dtype="uint32")
    cols = np.asarray(cols, dtype="uint32")
    z = np.asarray(z, dtype="float32")
    se = np.asarray(se, dtype="float32")
    imputed_values = None if imputed is None else np.asarray(imputed, dtype="uint8")
    abs_z = np.abs(z).astype("float32")

    root = zarr.open_group(str(Path(store_path) / "data.zarr"), mode="a")
    top = root.require_group("top_hits")
    n_analyses = int(root["z"].shape[1])
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

    for threshold in thresholds:
        key = threshold_key(threshold)
        if key in top:
            del top[key]
        group = top.create_group(key)
        keep = abs_z >= z_critical(threshold)
        kept_rows = rows[keep]
        kept_cols = cols[keep]
        kept_abs_z = abs_z[keep]
        kept_z = z[keep]
        kept_se = se[keep]
        kept_imputed = None if imputed_values is None else imputed_values[keep]
        kept_p = erfc(kept_abs_z.astype("float64") / math.sqrt(2.0))
        order = np.lexsort((kept_rows, kept_cols))
        kept_rows = kept_rows[order]
        kept_cols = kept_cols[order]
        kept_abs_z = kept_abs_z[order]
        kept_z = kept_z[order]
        kept_se = kept_se[order]
        kept_imputed = None if kept_imputed is None else kept_imputed[order]
        kept_p = kept_p[order]
        offsets = np.empty(n_analyses + 1, dtype="uint64")
        offsets[0] = 0
        np.cumsum(
            np.bincount(kept_cols, minlength=n_analyses),
            dtype=np.uint64,
            out=offsets[1:],
        )
        chunk = max(1, min(len(kept_rows), chunk_size))
        group.create_dataset(
            "analysis_offsets", data=offsets, chunks=(len(offsets),),
            compressor=compressor, dtype="uint64"
        )
        group.create_dataset(
            "variant_index", data=kept_rows, chunks=(chunk,), compressor=compressor, dtype="uint32"
        )
        group.create_dataset(
            "analysis_index", data=kept_cols, chunks=(chunk,), compressor=compressor, dtype="uint32"
        )
        group.create_dataset(
            "abs_z", data=kept_abs_z, chunks=(chunk,), compressor=compressor, dtype="float32"
        )
        group.create_dataset(
            "z", data=kept_z, chunks=(chunk,), compressor=compressor, dtype="float32"
        )
        group.create_dataset(
            "se", data=kept_se, chunks=(chunk,), compressor=compressor, dtype="float32"
        )
        group.create_dataset(
            "p_value", data=kept_p, chunks=(chunk,), compressor=compressor, dtype="float64"
        )
        if kept_imputed is not None:
            group.create_dataset(
                "imputed",
                data=kept_imputed,
                chunks=(chunk,),
                compressor=compressor,
                dtype="uint8",
            )
        group.attrs["threshold"] = threshold
        group.attrs["order"] = "analysis_index,variant_index"
    top.attrs["thresholds"] = list(thresholds)


def build_top_hit_indexes(
    store_path: str | Path,
    thresholds: tuple[float, ...] = TOP_HIT_THRESHOLDS,
    encoding: StoreEncoding | None = None,
) -> None:
    """(Re)build ranked top-hit arrays by scanning the stored dense matrix.

    Used by build paths that do not harvest hits inline, and to rebuild the index
    on an existing store. Scans ``z`` in row-bands (never the full matrix in RAM)
    and thresholds on the **stored** values -- decoded through the store's own
    codec, so the index matches exactly what a query reads back from ``z``
    (issue 046, ADR 0037). Collects only candidate cells
    (``|z| >= z_critical(loosest)``).

    ``encoding`` is the store's declared plan; when omitted it is read from the
    release's manifest, never re-derived from the arrays.
    """

    store_path = Path(store_path)
    if encoding is None:
        encoding = StoreManifest.load(store_path).encoding
    root = zarr.open_group(str(store_path / "data.zarr"), mode="r")
    z_plane = DenseZPlane.open(root, encoding)
    z_arr = z_plane.array
    se_arr = root["se"]
    imputed_arr = root["imputed"] if "imputed" in root else None
    n_variants = int(z_arr.shape[0])
    loosest = z_critical(max(thresholds))
    band_rows = max(int(z_arr.chunks[0]), 250_000)

    rows_parts: list[np.ndarray] = []
    cols_parts: list[np.ndarray] = []
    z_parts: list[np.ndarray] = []
    se_parts: list[np.ndarray] = []
    imputed_parts: list[np.ndarray] = []
    for r0 in range(0, n_variants, band_rows):
        r1 = min(r0 + band_rows, n_variants)
        z_band = z_plane.band(r0, r1)
        mask = np.abs(z_band) >= loosest  # NaN compares False
        br, bc = np.where(mask)
        if len(br):
            se_band = se_arr[r0:r1]
            rows_parts.append(br.astype(np.int64) + r0)
            cols_parts.append(bc.astype(np.int64))
            z_parts.append(z_band[br, bc])
            se_parts.append(se_band[br, bc].astype("float32"))
            if imputed_arr is not None:
                imputed_band = imputed_arr[r0:r1]
                imputed_parts.append(imputed_band[br, bc].astype("uint8"))

    if rows_parts:
        rows = np.concatenate(rows_parts)
        cols = np.concatenate(cols_parts)
        z = np.concatenate(z_parts)
        se = np.concatenate(se_parts)
        imputed = np.concatenate(imputed_parts) if imputed_parts else None
    else:
        rows = np.empty(0, dtype=np.int64)
        cols = np.empty(0, dtype=np.int64)
        z = np.empty(0, dtype=np.float32)
        se = np.empty(0, dtype=np.float32)
        imputed = np.empty(0, dtype=np.uint8) if imputed_arr is not None else None
    write_top_hit_indexes(store_path, rows, cols, z, se, thresholds, imputed=imputed)
