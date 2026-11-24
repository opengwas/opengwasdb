"""On-disk layout of a Hybrid store.

A single integrated Store directory holds two components over one shared Store
Variant Table::

    <store>/
      manifest.json                 # primary_layout = hybrid
      variants.tsv.gz (+ sidecars)  # SHARED union table (panel ∪ off-panel-observed)
      analyses.tsv                  # the Analyses, one analysis_index space
      dense/                        # nested Dense Component
        manifest.json,
        data.zarr/ (z, se, top_hits[, imputed, on_panel]),
        dense_to_shared.npy         # int32[n_panel]: dense row -> shared variant_index
        index.sqlite                # Reference-Completed only: completion_quality
      data.zarr/
        ragged/                     # Overflow CSR (offsets, variant_index=SHARED, z, se)
        top_hits/                   # (optional) merged hybrid top-hit cache

The Dense Component's rows are panel-local (0..n_panel-1, == its zarr rows).
``dense_to_shared.npy`` maps those rows onto the shared variant index space, which
the Ragged Overflow references directly. The map is strictly ascending (panel
variants keep their genomic order within the union), so ``searchsorted`` gives
shared → dense translation (`panel_rows`).

**From format 0.3.0 the Dense Component carries no variant identity of its own**
(#472, ADR 0102 §4): no Variant Table, `.tbi`, ALID, position or rsID sidecar, no
`analyses.tsv` and no `overview.html`. Its Variant Axis is the shared one seen through
``dense_to_shared`` (`opengwasdb.store_reader.panel_axis`), and its Analyses are the
root's. Before 0.3.0 it carried a copy of each, which a completed release let fall out
of step with the shared table (#481).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DENSE_SUBDIR = "dense"
DENSE_TO_SHARED_FILENAME = "dense_to_shared.npy"


def dense_component_path(store_path: str | Path) -> Path:
    """Path of the nested Dense Component store."""
    return Path(store_path) / DENSE_SUBDIR


def dense_to_shared_path(store_path: str | Path) -> Path:
    """Path of the dense-row → shared-variant-index map (int32, length n_panel)."""
    return dense_component_path(store_path) / DENSE_TO_SHARED_FILENAME


class DenseComponentSharesAxis(ValueError):
    """A 0.3.0 Hybrid's Dense Component was opened as though it carried a Variant Axis of
    its own. It carries none (#472): `StoreRelease.variant_axis()` gives the shared one,
    in its panel rows."""


def hybrid_root_of(component_path: str | Path) -> Path | None:
    """The Hybrid release that nests the release at `component_path` as its Dense
    Component, or None when it is not one.

    A Dense Component is the `dense/` directory of a release whose manifest's
    `primary_layout` is `hybrid` (spec §16). Its own manifest says `dense`, so only its
    place says what it is.
    """
    path = Path(component_path)
    if path.name != DENSE_SUBDIR:
        return None
    manifest = path.parent / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        layout = json.loads(manifest.read_text(encoding="utf-8")).get("primary_layout")
    except (OSError, ValueError):
        return None
    return path.parent if layout == "hybrid" else None


def panel_rows(
    dense_to_shared: np.ndarray, shared_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """`(on_panel, rows)` for shared variant indices: which are rows of the Dense
    Component's panel, and the panel row of each that is (-1 for each that is not).

    The one place the panel map is searched: the Hybrid query facade and the Dense
    Component's Variant Axis (`panel_axis`) both ask it. It is one vectorised
    `searchsorted` with the needles cast to the map's own dtype. The per-variant
    Python form cast the whole 13.4 M-entry `int32` map on every call (22.6 ms against
    0.004 ms for an `int32` scalar on OGS-00011), and `range_phewas` made one call per
    shared variant -- 58,006 of them for a 1 Mb TCF7L2 window (#252). Searching that
    map with `int64` needles still promoted and copied it on every call (review round
    1, and again in #465 for one needle: 38 ms on OGS-00004), so the needles are
    narrowed instead; a needle outside the map's range cannot be a panel row and is
    masked out before the cast, so narrowing cannot wrap it into range, and a needle
    between two panel rows is off panel, never its neighbour's row.
    """
    shared_indices = np.asarray(shared_indices)
    on_panel = np.zeros(shared_indices.shape, dtype=bool)
    rows = np.full(shared_indices.shape, -1, dtype=np.int64)
    panel = dense_to_shared
    if shared_indices.size == 0 or panel.size == 0:
        return on_panel, rows
    low, high = int(panel[0]), int(panel[-1])
    in_range = (shared_indices >= low) & (shared_indices <= high)
    if in_range.any():
        needles = shared_indices[in_range].astype(panel.dtype, copy=False)
        pos = np.minimum(np.searchsorted(panel, needles), panel.size - 1)
        hit = panel[pos] == needles
        on_panel[in_range] = hit
        rows[np.flatnonzero(in_range)[hit]] = pos[hit]
    return on_panel, rows
