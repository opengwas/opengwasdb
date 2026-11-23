"""On-disk layout of a Hybrid store.

A single integrated Store directory holds two components over one shared Store
Variant Table::

    <store>/
      manifest.json                 # primary_layout = hybrid
      variants.tsv.gz (+ sidecars)  # SHARED union table (panel ∪ off-panel-observed)
      index.sqlite                  # analyses (shared analysis_index space)
      dense/                        # nested Dense Component — a *valid* dense store
        manifest.json, index.sqlite, variants.tsv.gz (+ sidecars),
        data.zarr/ (z, se, top_hits[, imputed, on_panel]),
        dense_to_shared.npy         # int32[n_panel]: dense row -> shared variant_index
      data.zarr/
        ragged/                     # Overflow CSR (offsets, variant_index=SHARED, z, se)
        top_hits/                   # (optional) merged hybrid top-hit cache

The Dense Component uses its own panel-local ``variant_index`` (0..n_panel-1, ==
its zarr rows) so the *unchanged* dense builder / completer / validator operate on
it directly. ``dense_to_shared.npy`` maps those rows back onto the shared variant
index space, which the Ragged Overflow references directly. The map is ascending
(panel variants keep their genomic order within the union), so ``searchsorted``
gives shared → dense translation.
"""

from __future__ import annotations

from pathlib import Path

DENSE_SUBDIR = "dense"
DENSE_TO_SHARED_FILENAME = "dense_to_shared.npy"


def dense_component_path(store_path: str | Path) -> Path:
    """Path of the nested Dense Component store."""
    return Path(store_path) / DENSE_SUBDIR


def dense_to_shared_path(store_path: str | Path) -> Path:
    """Path of the dense-row → shared-variant-index map (int32, length n_panel)."""
    return dense_component_path(store_path) / DENSE_TO_SHARED_FILENAME
