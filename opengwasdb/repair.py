"""Representation-only repairs for existing Store Releases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from opengwasdb.encoding import EAF_BASELINE, EAF_REFERENCE, per_variant_chunk_size
from opengwasdb.layouts.hybrid.layout import dense_component_path
from opengwasdb.model.enums import PrimaryStorageLayout
from opengwasdb.store import open_store


@dataclass(frozen=True)
class EafChunkRepair:
    array: str
    old_chunk: int
    new_chunk: int


def _replace_with_rechunked(group: Any, name: str, chunk: int) -> None:
    """Replace one Zarr array, keeping its bytes and metadata unchanged."""
    source = group[name]
    temporary = f".{name}.rechunking"
    backup = f".{name}.old"
    for stale in (temporary, backup):
        if stale in group:
            del group[stale]
    target = group.create_dataset(
        temporary,
        shape=source.shape,
        chunks=(chunk,),
        dtype=source.dtype,
        compressor=source.compressor,
        filters=source.filters,
        fill_value=source.fill_value,
        order=source.order,
    )
    for key, value in source.attrs.items():
        target.attrs[key] = value
    for start in range(0, len(source), chunk):
        stop = min(start + chunk, len(source))
        target[start:stop] = np.asarray(source[start:stop])
    group.move(name, backup)
    try:
        group.move(temporary, name)
    except Exception:
        group.move(backup, name)
        raise
    del group[backup]


def _repair_group(group: Any, label: str) -> list[EafChunkRepair]:
    repaired: list[EafChunkRepair] = []
    for name in (EAF_BASELINE, EAF_REFERENCE):
        if name not in group:
            continue
        array = group[name]
        wanted = per_variant_chunk_size(group, len(array))
        current = int(array.chunks[0])
        if current <= wanted:
            continue
        _replace_with_rechunked(group, name, wanted)
        repaired.append(EafChunkRepair(f"{label}/{name}", current, wanted))
    return repaired


def repair_eaf_chunks(store_path: str | Path) -> list[EafChunkRepair]:
    """Rechunk EAF per-variant arrays in place without changing stored values.

    This is a physical representation repair, not a format migration: manifests,
    association data, release identity, and format version are untouched.
    """
    store = open_store(store_path)
    layout = store.manifest.primary_layout
    repaired: list[EafChunkRepair] = []
    if layout is PrimaryStorageLayout.DENSE:
        repaired.extend(_repair_group(store.arrays(mode="r+"), "data.zarr"))
    elif layout is PrimaryStorageLayout.RAGGED:
        group = zarr.open_group(str(store.data_path / "ragged"), mode="r+")
        repaired.extend(_repair_group(group, "data.zarr/ragged"))
    else:
        dense = open_store(dense_component_path(store.path))
        repaired.extend(_repair_group(dense.arrays(mode="r+"), "dense/data.zarr"))
        group = zarr.open_group(str(store.data_path / "ragged"), mode="r+")
        repaired.extend(_repair_group(group, "data.zarr/ragged"))
    return repaired
