"""Representation-only repairs for existing Store Releases."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from opengwasdb.encoding import EAF_BASELINE, EAF_REFERENCE, per_variant_chunk_size
from opengwasdb.layouts.hybrid.layout import dense_component_path
from opengwasdb.model.enums import PrimaryStorageLayout
from opengwasdb.store import open_store
from opengwasdb.store.arrays import (
    ArrayRole,
    array_length,
    compressor_of,
    create_array,
    move_in_group,
    open_group,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EafChunkRepair:
    array: str
    old_chunk: int
    new_chunk: int


def _swap_names(name: str) -> tuple[str, str]:
    """The rechunked copy's name and the original's name while they are swapped."""
    return f".{name}.rechunking", f".{name}.old"


def _recover_interrupted_swap(group: Any, name: str, label: str) -> None:
    """Put `name` back in one piece after a repair that died mid-swap.

    Each rename in the swap is atomic but the pair is not, and an exception
    handler cannot roll back a process that died. Each death leaves one state,
    and each state has one safe reading:

    * the copy only: died while writing it; the original is untouched, so the
      copy is dropped;
    * the backup without `name`: died between the renames; the backup is the
      complete original, so it is restored, and the repair runs again;
    * the backup beside `name`: died after the swap; `name` is the complete
      rechunked copy, so the backup is dropped.

    All three together cannot come from one death, so nothing is touched.
    """
    temporary, backup = _swap_names(name)
    present = {entry for entry in (name, temporary, backup) if entry in group}
    if present == {name, temporary, backup}:
        raise RuntimeError(
            f"{label}/{name}: {temporary} and {backup} both exist beside it; the repair "
            "cannot tell which is the original. Inspect them and remove the stale one."
        )
    if backup in present:
        _settle_backup(group, name, label, present)
    elif temporary in present:
        del group[temporary]
        log.warning("%s/%s: removed the unfinished copy %s", label, name, temporary)


def _settle_backup(group: Any, name: str, label: str, present: set[str]) -> None:
    """A backup is left: restore it if `name` is gone, else drop it (see above)."""
    temporary, backup = _swap_names(name)
    if name in present:
        del group[backup]
        log.warning("%s/%s: removed %s left after a completed swap", label, name, backup)
        return
    if temporary in present:
        del group[temporary]
    move_in_group(group, backup, name)
    log.warning("%s/%s: restored the original from %s (a dead repair)", label, name, backup)


def _replace_with_rechunked(group: Any, name: str, chunk: int) -> None:
    """Replace one Zarr array, keeping its bytes and metadata unchanged.

    The swap is two renames; a death between them is recovered by the next
    run's `_recover_interrupted_swap`.
    """
    source = group[name]
    temporary, backup = _swap_names(name)
    target = create_array(
        group,
        temporary,
        ArrayRole.PER_VARIANT,
        shape=source.shape,
        dtype=source.dtype,
        compressor=compressor_of(source),
        filters=source.filters,
        fill_value=source.fill_value,
        order=source.order,
        hint=chunk,
    )
    for key, value in source.attrs.items():
        target.attrs[key] = value
    for start in range(0, array_length(source), chunk):
        stop = min(start + chunk, array_length(source))
        target[start:stop] = np.asarray(source[start:stop])
    move_in_group(group, name, backup)
    try:
        move_in_group(group, temporary, name)
    except Exception:
        move_in_group(group, backup, name)
        raise
    del group[backup]


def _repair_group(group: Any, label: str) -> list[EafChunkRepair]:
    repaired: list[EafChunkRepair] = []
    for name in (EAF_BASELINE, EAF_REFERENCE):
        _recover_interrupted_swap(group, name, label)
        if name not in group:
            continue
        array = group[name]
        wanted = per_variant_chunk_size(group, array_length(array))
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
        group = open_group(store.data_path / "ragged", "r+")
        repaired.extend(_repair_group(group, "data.zarr/ragged"))
    else:
        dense = open_store(dense_component_path(store.path))
        repaired.extend(_repair_group(dense.arrays(mode="r+"), "dense/data.zarr"))
        group = open_group(store.data_path / "ragged", "r+")
        repaired.extend(_repair_group(group, "data.zarr/ragged"))
    return repaired
