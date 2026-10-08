"""Representation-only repairs for existing Store Releases."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
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


def _drop_copy(group: Any, name: str, label: str) -> None:
    """Died while writing the copy: the original is untouched, so drop the copy."""
    temporary, _ = _swap_names(name)
    del group[temporary]
    log.warning("%s/%s: removed the unfinished copy %s", label, name, temporary)


def _restore_backup(group: Any, name: str, label: str) -> None:
    """Died between the renames: the backup is the whole original, so put it back.

    The move comes first, so a death between the two steps leaves the array and
    its copy, a state the next run drops the copy from.
    """
    temporary, backup = _swap_names(name)
    move_in_group(group, backup, name)
    del group[temporary]
    log.warning("%s/%s: restored the original from %s (a dead repair)", label, name, backup)


def _drop_backup(group: Any, name: str, label: str) -> None:
    """Died after the swap: `name` is the whole rechunked copy, so drop the backup."""
    _, backup = _swap_names(name)
    del group[backup]
    log.warning("%s/%s: removed %s left after a completed swap", label, name, backup)


def _nothing_left(group: Any, name: str, label: str) -> None:
    """No leftover: the array is absent, or present on its own."""


#: Every state a single interrupted repair can leave, keyed by which of the
#: array, its rechunked copy and its backup are present, mapped to the step that
#: settles it. The swap is: write the copy, rename the array to the backup, rename
#: the copy to the array, delete the backup. Each step is a state below. Of the
#: eight possible states, the other three (the copy alone, the backup alone, and
#: all three together) cannot come from one interruption, and are refused.
_RECOVERY: Mapping[frozenset[str], Callable[[Any, str, str], None]] = MappingProxyType(
    {
        frozenset(): _nothing_left,
        frozenset({"array"}): _nothing_left,
        frozenset({"array", "copy"}): _drop_copy,
        frozenset({"copy", "backup"}): _restore_backup,
        frozenset({"array", "backup"}): _drop_backup,
    }
)


def _recover_interrupted_swap(group: Any, name: str, label: str) -> None:
    """Put `name` back in one piece after a repair that was interrupted mid-swap.

    Each rename in the swap is atomic but the pair is not, and an exception
    handler cannot roll back a process that died. `_RECOVERY` names the one safe
    reading of each state a death leaves. Any other state is refused with
    nothing touched: guessing which entry is the original could delete the only
    copy of it.
    """
    temporary, backup = _swap_names(name)
    roles = {"array": name, "copy": temporary, "backup": backup}
    present = frozenset(role for role, entry in roles.items() if entry in group)
    settle = _RECOVERY.get(present)
    if settle is None:
        found = ", ".join(roles[role] for role in sorted(present))
        raise RuntimeError(
            f"{label}/{name}: found {found}, a state no single interrupted repair leaves; "
            "the repair cannot tell which is the original. Inspect them and remove the stale ones."
        )
    settle(group, name, label)


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
    # Write whole shards (issue #247): a block that ends inside a shard turns
    # every write into a read-modify-write of it.  A v2 array has no shard, so
    # its inner chunk is the whole stored unit and is used as before.
    shards = getattr(target, "shards", None)
    block = int(shards[0]) if shards is not None else chunk
    for start in range(0, array_length(source), block):
        stop = min(start + block, array_length(source))
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
