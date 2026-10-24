"""Retrospective re-encode of a Store's SE plane and exception index (issue #326).

In every published 0.2.0 Dense Store the `se` plane (`int8`) and its
`se_exception_index` (`int64`) carry a Blosc ``typesize`` of 4, inherited from
the float32 scratch plane the SE rewrite staged.  Bitshuffling `int8` codes as
4-byte elements defeats the filter, and the exception index is wrong the same
way.  The value stored in either array is unchanged; only the codec
configuration (``typesize`` -> the stored dtype's item size) and the chunk bytes
change, which is why the re-encode is verified by *decoded* equality, never
byte equality of the compressed frames.

This module is the core primitive only.  It does **not** decide policy an
`opengwasdb-stores` maintenance command owns: no release lock or publication
marker refusal, no ``records/se-reencode.json`` writer and no CLI wiring.  The
stores lane opens the release, refuses a running release through `ogstores`
(the authoritative lock/marker API), calls :func:`plan_se_reencode` for a
dry-run and :func:`reencode_se_codecs` to apply it, then writes the record from
the returned plans/results.

The write is staged, verified, then swapped:

1. a fresh array is created beside the live one (``.NAME.reencoding``) on the
   same filesystem, with the seam's tunable Blosc codec so ``typesize``
   resolves to the target dtype;
2. every shard is copied whole from the decoded source, and the staged array is
   read back shard by shard -- a zarr decode that validates each shard-index
   CRC -- and compared to the source inner chunk by inner chunk, as raw bytes;
3. the staged array is swapped in with the two renames `repair` uses
   (``name -> .NAME.old``, ``.NAME.reencoding -> name``), keeping the original
   until the verification passed; a death between the renames is recovered by
   the next run, and any other state is refused rather than guessed.

Only ``se`` and ``se_exception_index`` of a Dense release (or of a Hybrid
release's nested Dense Component) are ever touched.  Ragged ``se`` planes are
float16 with the correct ``typesize`` and are out of scope; archived copies
under ``<store>/replaced/`` are out of scope and never walked.
"""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numcodecs
import numpy as np
from zarr.codecs import BloscCodec

from opengwasdb.layouts.hybrid.layout import dense_component_path
from opengwasdb.model.enums import PrimaryStorageLayout
from opengwasdb.store import open_store
from opengwasdb.store.arrays import (
    create_array,
    inner_chunk_of,
    move_in_group,
    role_for_array_path,
    sharded_compressor,
)

log = logging.getLogger(__name__)

__all__ = [
    "SeReencodeError",
    "SeReencodePlan",
    "SeReencodeResult",
    "plan_se_reencode",
    "reencode_se_codecs",
]

#: The two arrays the issue's retrospective re-encode corrects.  Both are
#: Dense-grid arrays; a Ragged release's `se` is float16 at the correct
#: `typesize` and is deliberately not listed.
_TARGET_ARRAYS = ("se", "se_exception_index")

#: How many shards the dry-run recompresses to estimate the byte saving.  The
#: number mirrors the issue's "200 sampled chunks"; the estimate extrapolates
#: the sampled compressed-size ratio to the array's measured on-disk bytes.
DEFAULT_ESTIMATE_SHARDS = 200

#: numcodecs' shuffle codes, keyed by zarr's `BloscCodec` spelling.
_SHUFFLE_CODES: Mapping[str, int] = MappingProxyType(
    {
        "noshuffle": numcodecs.Blosc.NOSHUFFLE,
        "shuffle": numcodecs.Blosc.SHUFFLE,
        "bitshuffle": numcodecs.Blosc.BITSHUFFLE,
    }
)


class SeReencodeError(RuntimeError):
    """A re-encode cannot finish without risking a torn or unreadable array."""


@dataclass(frozen=True)
class SeReencodePlan:
    """What one array's re-encode would change, and the estimated byte saving."""

    group: str
    array: str
    dtype: str
    current_typesize: int
    expected_typesize: int
    before_bytes: int
    estimated_after_bytes: int

    @property
    def estimated_saving_bytes(self) -> int:
        """Estimated bytes the re-encode would free, never negative."""
        return max(0, self.before_bytes - self.estimated_after_bytes)


@dataclass(frozen=True)
class SeReencodeResult:
    """What one array's completed re-encode changed."""

    group: str
    array: str
    before_bytes: int
    after_bytes: int
    equality: bool
    chunks: int


def _dense_groups(store: Any, mode: str) -> list[tuple[str, Any]]:
    """The `(label, group)` pairs holding the Dense `se` arrays to re-encode.

    A Dense release's group is its `data.zarr`; a Hybrid release stores its
    Dense grid in the nested Dense Component's `data.zarr`.  A Ragged release
    has no Dense `se` plane, so it yields nothing.
    """
    layout = store.manifest.primary_layout
    if layout is PrimaryStorageLayout.DENSE:
        return [("data.zarr", store.arrays(mode))]
    if layout is PrimaryStorageLayout.HYBRID:
        dense = open_store(dense_component_path(store.path))
        return [("dense/data.zarr", dense.arrays(mode))]
    return []


def _array_directory(group: Any, name: str) -> Path:
    """The on-disk directory of `name` inside `group`, for byte accounting."""
    root = Path(group.store.root)
    prefix = (group.path or "").strip("/")
    base = root / prefix if prefix else root
    return base / name


def _array_bytes(group: Any, name: str) -> int:
    """The bytes an array occupies on disk: its metadata file and chunk files."""
    directory = _array_directory(group, name)
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _typesize_mismatch(array: Any) -> tuple[int, int] | None:
    """The array's Blosc `typesize` and the dtype item size, when they disagree.

    `None` means the array needs no re-encode: it is not a v3 sharded array, is
    stored uncompressed, does not use a single `BloscCodec`, or already has the
    correct ``typesize``.  A v3 array with more than one compressor is a format
    violation and is refused rather than silently skipped.
    """
    if getattr(array.metadata, "zarr_format", None) != 3:
        return None
    codecs = tuple(array.compressors)
    if not codecs:
        return None
    if len(codecs) != 1:
        raise SeReencodeError(
            f"array {array.name!r} is stored with {len(codecs)} codecs; "
            "a Store Release array has exactly one"
        )
    codec = codecs[0]
    if not isinstance(codec, BloscCodec):
        return None
    itemsize = np.dtype(array.dtype).itemsize
    if int(codec.typesize) == itemsize:
        return None
    return int(codec.typesize), itemsize


def _axes(shape: tuple[int, ...], steps: tuple[int, ...]) -> list[range]:
    """One `range` of start positions per axis; an empty axis yields no starts."""
    return [range(0, dim, step) for dim, step in zip(shape, steps, strict=True)]


def _slices(
    starts: tuple[int, ...], steps: tuple[int, ...], shape: tuple[int, ...]
) -> tuple[slice, ...]:
    """The `[start, start+step)` slices clipped to `shape`, per axis."""
    return tuple(
        slice(start, min(start + step, dim))
        for start, step, dim in zip(starts, steps, shape, strict=True)
    )


def _stage_reencoded(group: Any, source: Any, temporary: str) -> Any:
    """Create `temporary` beside `source` and copy every shard, decoded, into it.

    The new array is created through the seam with the tunable compressor, so
    its Blosc ``typesize`` resolves to the target dtype's item size rather than
    inheriting the source's resolved codec.
    """
    role = role_for_array_path(source.path)
    if role is None:
        raise SeReencodeError(
            f"array {source.path!r} has no Store role; refusing to re-encode it"
        )
    target = create_array(
        group,
        temporary,
        role,
        shape=source.shape,
        dtype=source.dtype,
        fill_value=source.fill_value,
        compressor=sharded_compressor(),
        inner_chunk=inner_chunk_of(source),
        shards=tuple(int(size) for size in source.shards),
        filters=source.filters,
        order=source.order,
    )
    for key, value in source.attrs.items():
        target.attrs[key] = value
    shape = tuple(int(size) for size in source.shape)
    shards = tuple(int(size) for size in source.shards)
    for starts in itertools.product(*_axes(shape, shards)):
        block = _slices(starts, shards, shape)
        target[block] = np.asarray(source[block])
    return target


def _raw_bytes(values: Any) -> np.ndarray:
    """`values` as a C-contiguous uint8 view, so NaN payloads count as bytes."""
    return np.ascontiguousarray(values).view(np.uint8)


def _verify_reencoded(source: Any, target: Any) -> int:
    """Decode `target` fully and compare every inner chunk to `source`, as bytes.

    Reading a staged shard through zarr runs the sharding codec's index chain
    (`BytesCodec` + `Crc32cCodec`), so a shard whose stored index checksum does
    not match its computed one raises here rather than comparing equal.  The
    comparison then walks each decoded shard's inner chunks and requires the
    raw decoded bytes to match the source exactly.  Returns the number of inner
    chunks compared.
    """
    shape = tuple(int(size) for size in source.shape)
    chunks = inner_chunk_of(source)
    shards = tuple(int(size) for size in source.shards)
    verified = 0
    for starts in itertools.product(*_axes(shape, shards)):
        shard_slices = _slices(starts, shards, shape)
        source_block = np.asarray(source[shard_slices])
        target_block = np.asarray(target[shard_slices])
        block_shape = tuple(int(size) for size in source_block.shape)
        for chunk_starts in itertools.product(*_axes(block_shape, chunks)):
            chunk_slices = _slices(chunk_starts, chunks, block_shape)
            source_bytes = _raw_bytes(source_block[chunk_slices])
            target_bytes = _raw_bytes(target_block[chunk_slices])
            if source_bytes.shape != target_bytes.shape or not np.array_equal(
                source_bytes, target_bytes
            ):
                raise SeReencodeError(
                    f"{source.path}: inner chunk {chunk_slices!r} is not bit-identical "
                    "after re-encoding"
                )
            verified += 1
    return verified


def _reencode_names(name: str) -> tuple[str, str]:
    """The staged copy's name and the original's name while they are swapped."""
    return f".{name}.reencoding", f".{name}.old"


def _drop_copy(group: Any, name: str, label: str) -> None:
    """Died while writing the copy: the original is untouched, so drop the copy."""
    temporary, _ = _reencode_names(name)
    del group[temporary]
    log.warning("%s/%s: removed the unfinished re-encode copy %s", label, name, temporary)


def _restore_backup(group: Any, name: str, label: str) -> None:
    """Died between the renames: the backup is the whole original, so restore it."""
    temporary, backup = _reencode_names(name)
    move_in_group(group, backup, name)
    del group[temporary]
    log.warning("%s/%s: restored the original from %s (a dead re-encode)", label, name, backup)


def _drop_backup(group: Any, name: str, label: str) -> None:
    """Died after the swap: `name` is the whole re-encoded copy, so drop the backup."""
    _, backup = _reencode_names(name)
    del group[backup]
    log.warning("%s/%s: removed %s left after a completed swap", label, name, backup)


def _nothing_left(group: Any, name: str, label: str) -> None:
    """No leftover: the array is absent, or present on its own."""


#: Every state a single interrupted re-encode can leave, keyed by which of the
#: array, its staged copy and its backup are present, mapped to the step that
#: settles it.  The swap is: write the copy, rename the array to the backup,
#: rename the copy to the array, delete the backup.  Of the eight possible
#: states, the other three cannot come from one interruption and are refused.
_RECOVERY: Mapping[frozenset[str], Callable[[Any, str, str], None]] = MappingProxyType(
    {
        frozenset(): _nothing_left,
        frozenset({"array"}): _nothing_left,
        frozenset({"array", "copy"}): _drop_copy,
        frozenset({"copy", "backup"}): _restore_backup,
        frozenset({"array", "backup"}): _drop_backup,
    }
)


def _recover_interrupted_reencode(group: Any, name: str, label: str) -> None:
    """Put `name` back in one piece after a re-encode interrupted mid-swap.

    Each rename is atomic but the pair is not, and no exception handler runs
    for a process that died.  `_RECOVERY` names the one safe reading of each
    state a death leaves; any other state is refused with nothing touched.
    """
    temporary, backup = _reencode_names(name)
    roles = {"array": name, "copy": temporary, "backup": backup}
    present = frozenset(role for role, entry in roles.items() if entry in group)
    settle = _RECOVERY.get(present)
    if settle is None:
        found = ", ".join(roles[role] for role in sorted(present))
        raise SeReencodeError(
            f"{label}/{name}: found {found}, a state no single interrupted re-encode "
            "leaves; the re-encode cannot tell which is the original. Inspect them "
            "and remove the stale ones."
        )
    settle(group, name, label)


def _swap_staged(group: Any, name: str, temporary: str, backup: str) -> None:
    """Swap `temporary` into `name`, keeping `name` under `backup` until it lands."""
    move_in_group(group, name, backup)
    try:
        move_in_group(group, temporary, name)
    except Exception:
        move_in_group(group, backup, name)
        raise
    del group[backup]


def _reencode_array(group: Any, name: str, label: str) -> SeReencodeResult | None:
    """Re-encode one array, or return `None` when it needs no re-encode."""
    _recover_interrupted_reencode(group, name, label)
    if name not in group:
        return None
    source = group[name]
    if _typesize_mismatch(source) is None:
        return None
    before = _array_bytes(group, name)
    temporary, backup = _reencode_names(name)
    target = _stage_reencoded(group, source, temporary)
    try:
        chunks = _verify_reencoded(source, target)
    except Exception:
        # The live array is untouched; drop the staged copy that failed to verify.
        if temporary in group:
            del group[temporary]
        raise
    _swap_staged(group, name, temporary, backup)
    after = _array_bytes(group, name)
    return SeReencodeResult(
        group=label,
        array=name,
        before_bytes=before,
        after_bytes=after,
        equality=True,
        chunks=chunks,
    )


def _blosc_for(codec: BloscCodec, typesize: int) -> numcodecs.Blosc:
    """A numcodecs `Blosc` in `codec`'s configuration, at the given `typesize`."""
    return numcodecs.Blosc(
        cname=str(codec.cname),
        clevel=int(codec.clevel),
        shuffle=_SHUFFLE_CODES[str(codec.shuffle)],
        typesize=typesize,
        blocksize=int(codec.blocksize),
    )


def _full_shard(array: Any, starts: tuple[int, ...], shards: tuple[int, ...]) -> np.ndarray:
    """The decoded shard at `starts`, padded to the full shard shape.

    zarr stores an edge shard padded to the declared shard shape with the
    array's fill value, so the estimate compresses the same shape the store
    writes rather than the short slice a decode returns.
    """
    shape = tuple(int(size) for size in array.shape)
    piece = np.asarray(array[_slices(starts, shards, shape)])
    if tuple(piece.shape) == shards:
        return piece
    pads = tuple((0, want - have) for have, want in zip(piece.shape, shards, strict=True))
    return np.pad(piece, pads, mode="constant", constant_values=array.fill_value)


def _estimated_after_bytes(array: Any, before_bytes: int, max_shards: int) -> int:
    """Estimated on-disk bytes after re-encoding `array`, from a shard sample.

    Each sampled shard is decoded, then compressed both at the stored (wrong)
    `typesize` and at the corrected one; the ratio of those compressed sizes is
    applied to the measured `before_bytes`.  Blosc's compressed *size* is stable
    run to run even though its frame bytes are not, so the estimate is
    reproducible.
    """
    shape = tuple(int(size) for size in array.shape)
    if any(dim == 0 for dim in shape):
        return before_bytes
    mismatch = _typesize_mismatch(array)
    if mismatch is None:
        return before_bytes
    current_typesize, expected_typesize = mismatch
    codecs = tuple(array.compressors)
    assert len(codecs) == 1
    codec = codecs[0]
    assert isinstance(codec, BloscCodec)
    current = _blosc_for(codec, current_typesize)
    corrected = _blosc_for(codec, expected_typesize)
    shards = tuple(int(size) for size in array.shards)
    positions = list(itertools.product(*_axes(shape, shards)))
    if not positions:
        return 0
    step = max(1, math.ceil(len(positions) / max(1, max_shards)))
    sampled = positions[::step]
    old_total = 0
    new_total = 0
    for starts in sampled:
        block = _full_shard(array, tuple(starts), shards)
        contiguous = np.ascontiguousarray(block)
        old_total += len(current.encode(contiguous))
        new_total += len(corrected.encode(contiguous))
    if old_total == 0:
        return before_bytes
    return max(0, int(math.ceil(before_bytes * new_total / old_total)))


def plan_se_reencode(
    store_path: str | Path, *, max_shards: int = DEFAULT_ESTIMATE_SHARDS
) -> list[SeReencodePlan]:
    """What re-encoding one Store's SE arrays would change, without writing.

    Returns one plan per array that needs re-encoding (a Blosc ``typesize``
    that differs from the stored dtype's item size), across the Dense grid of a
    Dense release or of a Hybrid release's nested Dense Component.  A release
    that needs nothing yields an empty list.  The estimated saving is a
    reproducible extrapolation from at most `max_shards` decoded shards.
    """
    store = open_store(store_path)
    plans: list[SeReencodePlan] = []
    for label, group in _dense_groups(store, "r"):
        for name in _TARGET_ARRAYS:
            if name not in group:
                continue
            array = group[name]
            mismatch = _typesize_mismatch(array)
            if mismatch is None:
                continue
            current_typesize, expected_typesize = mismatch
            before = _array_bytes(group, name)
            plans.append(
                SeReencodePlan(
                    group=label,
                    array=name,
                    dtype=str(array.dtype),
                    current_typesize=current_typesize,
                    expected_typesize=expected_typesize,
                    before_bytes=before,
                    estimated_after_bytes=_estimated_after_bytes(array, before, max_shards),
                )
            )
    return plans


def reencode_se_codecs(store_path: str | Path) -> list[SeReencodeResult]:
    """Re-encode a Store's `se` and `se_exception_index` in place, losslessly.

    Idempotent: an array already at the correct ``typesize`` is left alone, and
    a second run returns an empty list.  A crash during staging, verification
    or the swap is settled by the next run (the original is restored or the
    re-encoded array is kept); the original is never removed until the staged
    array has verified.  Returns one result per array changed, with measured
    before/after bytes, the decoded-equality verdict and the chunk count.
    """
    store = open_store(store_path)
    results: list[SeReencodeResult] = []
    for label, group in _dense_groups(store, "r+"):
        for name in _TARGET_ARRAYS:
            result = _reencode_array(group, name, label)
            if result is not None:
                results.append(result)
    return results
