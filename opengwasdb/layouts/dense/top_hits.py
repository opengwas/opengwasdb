"""Dense top-hit index builder."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from numcodecs import Blosc
from scipy.special import erfc, erfcinv  # type: ignore[import-untyped]

from opengwasdb.encoding import DenseEafPlane, DenseZPlane, StoreEncoding
from opengwasdb.layouts.dense.constants import TOP_HIT_THRESHOLDS
from opengwasdb.model.analyses import TOP_HIT_COUNT_COLUMNS
from opengwasdb.model.manifest import StoreManifest

log = logging.getLogger(__name__)

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

    def read_or(
        self,
        name: str,
        bounds: tuple[int, int],
        dtype: str,
        fallback: Callable[[], np.ndarray],
    ) -> np.ndarray:
        """Read an indexed result field, or derive it for an older index."""
        if name in self.group:
            return self.read(name, bounds, dtype)
        return np.asarray(fallback(), dtype=dtype)


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


# The dtype each top-hit index array is written as. One table, so a tier's
# optional arrays (`imputed`, `eaf`) cannot acquire a different dtype from the
# required ones by being written at a separate call site.
_TIER_DTYPES = {
    "variant_index": "uint32",
    "analysis_index": "uint32",
    "abs_z": "float32",
    "z": "float32",
    "se": "float32",
    "p_value": "float64",
    "imputed": "uint8",
    "eaf": "float32",
}


def write_threshold_tier(
    top: zarr.Group,
    threshold: float,
    columns: dict[str, np.ndarray],
    abs_z: np.ndarray,
    n_analyses: int,
    chunk_size: int,
    compressor: Blosc,
) -> int:
    """Select, rank and write the one tier for `threshold`; return its hit count.

    `columns` holds every parallel array the caller has, required and optional
    alike; each is filtered, ordered and written by the same three lines, so an
    array cannot be dropped from one of those steps and not the others.
    """
    key = threshold_key(threshold)
    if key in top:
        del top[key]
    group = top.create_group(key)

    keep = abs_z >= z_critical(threshold)
    kept = {name: values[keep] for name, values in columns.items()}
    kept["abs_z"] = abs_z[keep]
    kept["p_value"] = erfc(kept["abs_z"].astype("float64") / math.sqrt(2.0))

    order = np.lexsort((kept["variant_index"], kept["analysis_index"]))
    kept = {name: values[order] for name, values in kept.items()}

    offsets = np.empty(n_analyses + 1, dtype="uint64")
    offsets[0] = 0
    np.cumsum(
        np.bincount(kept["analysis_index"], minlength=n_analyses),
        dtype=np.uint64,
        out=offsets[1:],
    )
    chunk = max(1, min(len(kept["variant_index"]), chunk_size))
    group.create_dataset(
        "analysis_offsets", data=offsets, chunks=(len(offsets),),
        compressor=compressor, dtype="uint64",
    )
    for name, values in kept.items():
        group.create_dataset(
            name, data=values, chunks=(chunk,),
            compressor=compressor, dtype=_TIER_DTYPES[name],
        )
    group.attrs["threshold"] = threshold
    group.attrs["order"] = "analysis_index,variant_index"
    return int(len(kept["variant_index"]))


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
    abs_z = np.abs(columns["z"]).astype("float32")

    root = zarr.open_group(str(Path(store_path) / "data.zarr"), mode="a")
    top = root.require_group("top_hits")
    n_analyses = int(root["z"].shape[1])
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

    for threshold in thresholds:
        write_threshold_tier(
            top, threshold, columns, abs_z, n_analyses, chunk_size, compressor
        )
    top.attrs["thresholds"] = list(thresholds)


def _concat_or_empty(parts: list[np.ndarray], dtype: type) -> np.ndarray:
    """One candidate array from its per-band pieces, empty when nothing passed."""
    return np.concatenate(parts) if parts else np.empty(0, dtype=dtype)


def _scan_candidates(
    z_plane: DenseZPlane,
    se_arr: Any,
    imputed_arr: Any,
    eaf_plane: DenseEafPlane,
    n_variants: int,
    band_rows: int,
    loosest: float,
) -> dict[str, list[np.ndarray]]:
    """Every cell clearing the loosest tier, gathered band by band.

    The matrix is never held whole: each band contributes only the cells that
    pass, which on a real store is a tiny fraction of it.
    """
    parts: dict[str, list[np.ndarray]] = {
        name: [] for name in ("rows", "cols", "z", "se", "imputed", "eaf")
    }
    for r0 in range(0, n_variants, band_rows):
        r1 = min(r0 + band_rows, n_variants)
        z_band = z_plane.band(r0, r1)
        br, bc = np.where(np.abs(z_band) >= loosest)  # NaN compares False
        if not len(br):
            continue
        parts["rows"].append(br.astype(np.int64) + r0)
        parts["cols"].append(bc.astype(np.int64))
        parts["z"].append(z_band[br, bc])
        parts["se"].append(se_arr[r0:r1][br, bc].astype("float32"))
        if imputed_arr is not None:
            parts["imputed"].append(imputed_arr[r0:r1][br, bc].astype("uint8"))
        if eaf_plane.can_report_frequencies:
            parts["eaf"].append(eaf_plane.band(r0, r1)[br, bc].astype("float32"))
    return parts


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
    imputed_arr = root["imputed"] if "imputed" in root else None
    eaf_plane = DenseEafPlane.open(root, encoding)

    parts = _scan_candidates(
        z_plane,
        root["se"],
        imputed_arr,
        eaf_plane,
        int(z_plane.array.shape[0]),
        max(int(z_plane.array.chunks[0]), 250_000),
        z_critical(max(thresholds)),
    )
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


def write_top_hit_indexes_for_store(
    store_path: str | Path,
    rows: np.ndarray,
    cols: np.ndarray,
    z: np.ndarray,
    se: np.ndarray,
    encoding: StoreEncoding,
) -> None:
    """Decode the candidates' frequencies, then write every tier.

    The pairing an inline build needs: a builder that wrote the frequency plane
    holds the candidate coordinates already, and the index must carry the same
    decoded values a query would read back (ADR 0040). Keeping the two calls
    together is what stops a build writing tiers with no `eaf` array.
    """
    log.info("Collecting top-hit EAF in variant-row order for %d candidate cells", len(rows))
    eaf = collect_top_hit_eaf(store_path, rows, cols, encoding)
    log.info("Writing top-hit index from %d harvested candidate cells", len(rows))
    write_top_hit_indexes(store_path, rows, cols, z, se, eaf=eaf)


def collect_top_hit_eaf(
    store_path: str | Path,
    rows: np.ndarray,
    cols: np.ndarray,
    encoding: StoreEncoding,
) -> np.ndarray | None:
    """Collect candidate EAF values in row-chunk order for an inline build."""
    root = zarr.open_group(str(Path(store_path) / "data.zarr"), mode="r")
    plane = DenseEafPlane.open(root, encoding)
    if not plane.can_report_frequencies:
        return None
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    out = np.empty(len(rows), dtype=np.float32)
    if len(rows) == 0:
        return out
    row_chunk = int(root["z"].chunks[0])
    order = np.argsort(rows, kind="stable")
    sorted_rows = rows[order]
    for chunk_start in np.unique((sorted_rows // row_chunk) * row_chunk):
        chunk_stop = min(int(chunk_start) + row_chunk, int(root["z"].shape[0]))
        lo = int(np.searchsorted(sorted_rows, chunk_start, side="left"))
        hi = int(np.searchsorted(sorted_rows, chunk_stop, side="left"))
        slots = order[lo:hi]
        band = plane.band(int(chunk_start), chunk_stop)
        out[slots] = band[rows[slots] - int(chunk_start), cols[slots]]
    return out
