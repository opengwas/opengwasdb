"""The one seam through which every Store Release array and group is created.

Every Zarr array this package writes is created here, via `create_array`, and
every Zarr group via `create_group`/`require_group`.  That is not tidiness for
its own sake: array creation is where a Store Release's physical layout is
decided, and it has to be decided in exactly one place so the pieces that will
change it -- the switch to Zarr v3 sharding (#247) and the converter that has
to reproduce the same layout (#245) -- cannot disagree about how an array is
laid out.

The seam owns five things:

* **the compressor**, `compressor()`, one Blosc zstd / clevel 3 / bitshuffle
  configuration described by `COMPRESSOR_RECORD`.  The record is the same dict
  the manifest and `index.sqlite` publish, so the bytes a plane is stored with
  and the bytes the manifest claims it was stored with cannot drift apart.
* **the chunk layout**, through the role -> policy table `_LAYOUTS`.  A caller
  names its array's `ArrayRole`; the role, the array's shape and -- for the one
  role that depends on a sibling, `PER_VARIANT` -- the component plane's
  variant-axis chunk fix the default chunks.  The component chunk is an
  explicit argument (`component_chunk`), never sniffed from the destination
  group, so the converter (#245) can reproduce any array's layout from facts it
  already has (role, shape, the plane chunk).  A build-wide hint (the Dense
  `chunk_shape`, the top-hit `chunk_size`, an explicit `chunks=(...)` on a CSR
  writer) is an override the policy honours; no call site computes a chunk
  tuple itself.
* **chunk clipping to the array dimensions** (ADR 0021): the dense grid policy
  clips the two-dimensional hint to the array's rows and columns, and the
  length-indexed policies clip to the array's length.
* **the fill value**, i.e. each plane's own missing marker (spec §15).  The fill
  is a property of the plane, not of the seam, so the caller passes it; the
  seam is the only code that hands it to Zarr.  Omitting `fill_value` means
  "the dtype's default", which is *not* the same as `fill_value=None` and is
  what several whole-array writes rely on.
* **zarr's process-wide runtime configuration**: chunks that are all fill
  value are still written, Blosc decodes with its internal threads, and every
  array goes through zarr's fused codec pipeline with one worker.  zarr-python
  3 holds these in runtime config rather than array metadata, so they are set
  once, on import, by the read side of this seam,
  `opengwasdb.store_reader.arrays`, which this module imports (#244, #464;
  ADR 0056).  That module also opens a group read-only; this one opens it in
  every other mode.

The role -> layout policy is one table in this module (`_LAYOUTS`) and nothing
else chooses chunks.  A role that does not fit an existing entry gets a new
one rather than a call site that passes `chunks=` by hand.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal

import numpy as np
import zarr
from numcodecs import Blosc
from zarr.codecs import BloscCodec
from zarr.core.buffer import Buffer
from zarr.core.sync import sync as zarr_sync
from zarr.storage import LocalStore

# The read side of the seam: importing it sets zarr's runtime configuration
# (ADR 0056), so no array is created here without it. `array_length` and
# `open_group`'s read mode are its; `array_length` is re-exported because
# opengwasdb-stores imports it from here.
from opengwasdb.store_reader import arrays as reader_arrays
from opengwasdb.store_reader.arrays import array_length as array_length
from opengwasdb.store_reader.arrays import inner_chunk_of

# The format's physical layout -- each role's inner chunk and shard -- is the Store
# Reader's (`opengwasdb.store_reader.layout`, #464); arrays are created on it here.
from opengwasdb.store_reader.layout import (
    ArrayRole,
    chunk_layout,
    component_chunk_size,
    require_shard_multiple,
    shard_layout,
)

__all__ = [
    "BY_VARIANT_OFFSETS_CHUNK",
    "COMPRESSOR_RECORD",
    "SE_COEFFICIENTS",
    "PartialShardWriteError",
    "SHARDED_COMPRESSOR_RECORD",
    "SHARDED_STORE_ZARR_FORMAT",
    "STORE_ZARR_FORMAT",
    "array_length",
    "component_variant_chunk",
    "compressor",
    "compressor_of",
    "consolidate_group_metadata",
    "count_shard_writes",
    "create_array",
    "create_group",
    "inner_chunk_hint_for_path",
    "inner_chunk_of",
    "is_recorded_group_path",
    "move_in_group",
    "open_group",
    "open_group_for_write",
    "per_variant_chunk_size",
    "require_group",
    "require_whole_shard_write",
    "require_whole_shard_writes",
    "role_for_array_path",
    "sharded_compressor",
    "write_shard_cells",
]

# ── compressor ───────────────────────────────────────────────────────────────

#: The one compressor configuration every statistic array in a Store Release is
#: stored with, in the shape the manifest's `compressor` and `index.sqlite`'s
#: `dense` blob publish it.  `DEFAULT_COMPRESSOR` in the Dense constants module
#: *is* this dict, so the published claim and the bytes written come from one
#: source.
COMPRESSOR_RECORD: dict[str, Any] = {
    "library": "numcodecs.Blosc",
    "cname": "zstd",
    "clevel": 3,
    "shuffle": "bitshuffle",
}

#: numcodecs' shuffle codes, keyed by the `COMPRESSOR_RECORD` spelling.
_SHUFFLE_CODES: Mapping[str, int] = MappingProxyType(
    {"noshuffle": Blosc.NOSHUFFLE, "shuffle": Blosc.SHUFFLE, "bitshuffle": Blosc.BITSHUFFLE}
)


def compressor() -> Blosc:
    """A Blosc compressor in the Store Release's one configuration.

    A fresh instance each call, not a cached one: the SE measurement encodes
    chunks with it on worker processes, and a process-shared codec is not worth
    the risk.  Configuration is fixed by `COMPRESSOR_RECORD`, so every instance
    is equivalent.
    """
    return Blosc(
        cname=COMPRESSOR_RECORD["cname"],
        clevel=COMPRESSOR_RECORD["clevel"],
        shuffle=_SHUFFLE_CODES[COMPRESSOR_RECORD["shuffle"]],
    )




def compressor_of(array: Any) -> Any:
    """The codec an existing Store array is stored with.

    zarr-python 3 moved this from ``Array.compressor`` (deprecated) to the
    ``Array.compressors`` tuple; the v2 format has exactly one.  Reading it
    through this helper keeps the deprecation out of the writers and refuses a
    layout the Store format does not define rather than taking the first codec
    of several.
    """
    codecs = tuple(array.compressors)
    if len(codecs) != 1:
        raise ValueError(
            f"array {array.name!r} is stored with {len(codecs)} codecs; "
            "a Store Release array has exactly one"
        )
    return codecs[0]


def blosc_like(codec: BloscCodec, *, typesize: int) -> Blosc:
    """A numcodecs `Blosc` matching a zarr `BloscCodec`, at an explicit `typesize`.

    The re-encode dry run re-compresses decoded shards at the stored and the
    corrected element sizes to estimate the byte saving; a zarr `BloscCodec`
    has no `encode` for a bare buffer, so the estimate needs the numcodecs
    spelling with the same cname/clevel/shuffle and a fixed `typesize`.
    """
    return Blosc(
        cname=str(codec.cname),
        clevel=int(codec.clevel),
        shuffle=_SHUFFLE_CODES[str(codec.shuffle)],
        typesize=typesize,
        blocksize=int(codec.blocksize),
    )


def sharded_compressor_like(codec: BloscCodec, *, typesize: int) -> BloscCodec:
    """A v3 `BloscCodec` matching `codec`, with only `typesize` changed.

    The retrospective re-encode writes a fresh array whose codec differs from
    the source's in exactly one field: `typesize` moves to the stored dtype's
    item size.  cname, clevel and shuffle are the source's own, so a Store that
    was written with a non-default compressor is preserved rather than silently
    normalised to the seam defaults.
    """
    return BloscCodec(
        cname=codec.cname,
        clevel=int(codec.clevel),
        shuffle=codec.shuffle,
        typesize=typesize,
        blocksize=int(codec.blocksize),
    )


#: `create_array`'s `compressor=` parameter would otherwise shadow the function
#: inside that function's body.
_new_compressor = compressor


# ── chunk-layout policy ──────────────────────────────────────────────────────

#: The inner chunk of a `ragged/by_variant/offsets` array (ADR 0060).  The
#: index's offset array is read at a single variant, so one 8 KiB inner chunk
#: per variant is the right read unit -- unlike `ASSOCIATION_OFFSETS_CHUNK`'s
#: 10,000, which is tuned for streaming one Analysis's segment.  The value is a
#: **path** hint, not a role one: `RAGGED_PER_VARIANT`'s default follows the
#: component plane, which is wrong for this array, so every creator (builders,
#: the converter, the legacy relayout and the augment command) must derive the
#: same hint from the path -- see `inner_chunk_hint_for_path`.
BY_VARIANT_OFFSETS_CHUNK = 1_000

def per_variant_chunk_size(group: Any, length: int) -> int:
    """The per-variant chunk for a group's component plane (legacy signature).

    Kept as the exported `opengwasdb.encoding.per_variant_chunk_size(group,
    length)` contract: an existing caller passing a Zarr group still works.  It
    reads the component plane's chunk and delegates to `component_chunk_size`;
    new code that already holds the plane chunk should call that directly.
    """
    return component_chunk_size(component_variant_chunk(group), length)


def component_variant_chunk(group: Any) -> int | None:
    """The variant-axis chunk of a group's component plane, or `None`.

    This is the writer/validator-side read that turns a group into the explicit
    `component_chunk` a layout is derived from.  It is deliberately *not* called
    by `create_array` or `chunk_layout`: the converter has no builder group and
    must be handed the plane chunk directly.
    """
    if group is None:
        return None
    for sibling in ("eaf", "z", "imputed", "variant_index"):
        if sibling in group and group[sibling].ndim:
            return int(inner_chunk_of(group[sibling])[0])
    return None


# ── v3 sharded physical layout ───────────────────────────────────────────────
#
# Format 0.2.0 (issue #245) stores every array as Zarr v3 with the sharding
# codec: the *inner chunk* is the unit a query reads, and the *shard* is the
# unit stored as one file.  The inner chunk keeps coming from `chunk_layout`;
# this section is its one companion, the role -> shard policy.  #247 switches
# the builders to read this same table, so the converter (which has no builder
# group) and the builders cannot disagree about the shard a role gets.
#
# The shard is always a whole multiple of the inner chunk on every axis, which
# zarr enforces as well; the duplicate check here fails with the role in the
# message rather than a bare zarr error naming only the sizes.

#: The v3 codec chain, as `COMPRESSOR_RECORD` is for v2 -- the same Blosc
#: zstd / clevel 3 / bitshuffle, expressed as zarr's own `BloscCodec` because a
#: v3 array refuses a numcodecs codec (`'Blosc' object is not iterable`).  The
#: `format` key is the extension the ticket asks for: the same record shape the
#: manifest and `index.sqlite` publish, saying which Zarr format it describes.
SHARDED_COMPRESSOR_RECORD: dict[str, Any] = {
    **COMPRESSOR_RECORD,
    "library": "zarr.codecs.BloscCodec",
    "format": "zarr_v3_sharding",
}


def sharded_compressor() -> BloscCodec:
    """A v3 `BloscCodec` in the Store Release's one configuration.

    The v3 spelling of `compressor()`: `BloscCodec` is zarr's own codec, and it
    serialises into the `sharding_indexed` codec's inner chain.  The frame it
    writes is the same Blosc zstd / clevel 3 / bitshuffle a v2 build writes, so
    the decoded codes are unchanged.
    """
    return BloscCodec(
        cname=COMPRESSOR_RECORD["cname"],
        clevel=COMPRESSOR_RECORD["clevel"],
        shuffle=COMPRESSOR_RECORD["shuffle"],
    )


# ── whole-shard writes ───────────────────────────────────────────────────────
#
# A write that covers part of a shard turns into a read-modify-write of the
# whole shard: correct, but it loses the throughput the shard exists for.  The
# Dense band writer and the row-block writers therefore work in whole shards,
# and this guard is what makes a later writer that stops doing so fail loudly
# instead of quietly getting slower.  It is off on the production path: the
# check runs only when `require_whole_shard_writes()` has been entered, or when
# the environment variable below is set (a real-data build opts in to prove its
# writes are aligned).

#: Set to ``1`` to check every Store write for whole-shard coverage.  Unset in
#: production, where the guard costs an environment lookup at import and
#: nothing per write.
_ENFORCE_WHOLE_SHARD_WRITES_ENV = "OPEN_GWASDB_REQUIRE_WHOLE_SHARD_WRITES"


class PartialShardWriteError(RuntimeError):
    """A write covers part of a shard; the shard would become a read-modify-write."""


def _axis_covers_whole_shard(selection: Any, shard: int, dim: int) -> bool:
    """Whether one axis' selection starts and ends on a shard boundary.

    An integer index selects a single position and never covers a shard.  A
    slice counts as aligned when it starts at a multiple of the shard and reaches
    either the next shard boundary or the end of the array: the final shard of
    an array need not be full, and a write that reaches the array's end covers
    it whole.
    """
    if isinstance(selection, slice):
        start, stop, step = selection.indices(dim)
        if step != 1:
            return False
        return start % shard == 0 and (stop == dim or stop % shard == 0)
    return False


def _normalise_selection(shape: tuple[int, ...], selection: Any) -> tuple[Any, ...]:
    """`selection` as one entry per axis, with `Ellipsis` and padding expanded."""
    if selection is Ellipsis:
        return (slice(None),) * len(shape)
    if not isinstance(selection, tuple):
        selection = (selection,)
    axes: list[Any] = []
    remaining = list(shape)
    seen_ellipsis = False
    for entry in selection:
        if entry is Ellipsis:
            if seen_ellipsis:
                return ()
            seen_ellipsis = True
            fill = len(shape) - (len(selection) - 1)
            axes.extend([slice(None)] * max(fill, 0))
            remaining = remaining[fill:]
        else:
            axes.append(entry)
            if remaining:
                remaining.pop(0)
    axes.extend([slice(None)] * len(remaining))
    return tuple(axes)


def require_whole_shard_write(array: Any, selection: Any) -> None:
    """Fail loudly unless `selection` covers whole shards of a sharded array.

    Every sharded array is judged, whatever its rank or shard size: the Dense
    statistic planes, the imputed mask and the SE coefficient table; the 1-D
    Ragged association sequences (`z`, `se`, `variant_index`, `eaf`, `imputed`),
    whose shard is 50,000,000 elements; and a 1-D array whose one shard is the
    whole array, such as an exception table.  A write that covers part of a
    shard is a read-modify-write of the whole shard, so the writer pays to
    decode and re-encode everything the shard already held (#249); on OGS-00011's
    overflow sequences that is about twelve rewrites of every
    50,000,000-element shard, and on a single-shard table it is one rewrite per
    writing band.  `encoding/se.py` buffers its SE exception tables and writes
    each once for exactly this reason.

    Unsharded arrays are not judged -- the rule is about shards.
    """
    shards = getattr(array, "shards", None)
    if shards is None:
        return
    axes = _normalise_selection(tuple(int(size) for size in array.shape), selection)
    if not axes:
        return
    for axis, (index_selection, shard, dim) in enumerate(
        zip(axes, (int(size) for size in shards), array.shape, strict=False)
    ):
        if not _axis_covers_whole_shard(index_selection, shard, int(dim)):
            raise PartialShardWriteError(
                f"array {getattr(array, 'path', '?')!r}: a write selecting {selection!r} "
                f"does not cover whole shards (axis {axis}: {index_selection!r} against "
                f"shard {shard} over {dim} rows). A partial shard write is a "
                "read-modify-write of the shard; write whole shards (issue #247)"
            )


def write_shard_cells(array: Any, rows: Any, cols: Any, values: Any) -> None:
    """Write `values` into `array[rows, cols]` with one whole-shard write each.

    Patching individual cells of a 2-D Dense plane through `vindex`/`oindex` is a
    read-modify-write of every shard they touch, and it is invisible to the
    whole-shard guard (`require_whole_shard_write`).  The cells are grouped by
    the shard they fall in; each touched shard's band is read once, patched in
    memory and written back whole.  A 1-D or unsharded array falls back to the
    direct write, which is what the element-capped and whole-array policies
    expect.
    """
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    values = np.asarray(values)
    if rows.size == 0:
        return
    shards = getattr(array, "shards", None)
    if shards is None or len(array.shape) != 2:
        array[rows, cols] = values
        return
    shard_rows = max(int(shards[0]), 1)
    shard_cols = max(int(shards[1]), 1)
    n_col_shards = -(-int(array.shape[1]) // shard_cols)
    keys = (rows // shard_rows) * n_col_shards + (cols // shard_cols)
    for key in np.unique(keys):
        mask = keys == key
        shard_row_idx = rows[mask]
        shard_col_idx = cols[mask]
        r0 = int(shard_row_idx[0]) // shard_rows * shard_rows
        c0 = int(shard_col_idx[0]) // shard_cols * shard_cols
        r1 = min(r0 + shard_rows, int(array.shape[0]))
        c1 = min(c0 + shard_cols, int(array.shape[1]))
        band = np.asarray(array[r0:r1, c0:c1])
        band[shard_row_idx - r0, shard_col_idx - c0] = values[mask]
        array[r0:r1, c0:c1] = band


def array_key_of_store_key(key: str) -> str | None:
    """The array path a store key belongs to, or `None` for metadata/group keys.

    A Zarr v3 sharded array stores one file per shard at ``<array>/c/<coords>``
    (coordinates joined by ``/``), and its metadata at ``<array>/zarr.json``.  A
    key with no ``/c/`` is metadata, a group, or a v2-style chunk, and does not
    identify an array this recorder attributes bytes to.
    """
    marker = key.find("/c/")
    return None if marker < 0 else key[:marker]


#: A Zarr v2 chunk key is ``<array>/<c0>[.<c1>...]`` with purely numeric
#: coordinates; every other suffix (``.zarray``, ``.zattrs``, ``zarr.json``,
#: groups) is metadata and does not name a compressed output chunk.  v2 has no
#: ``/c/`` marker, so `array_key_of_store_key` alone cannot see these.
_V2_CHUNK_KEY = re.compile(r"^(?P<array>.+)/(?P<coordinates>\d+(?:\.\d+)*)$")


def v2_chunk_array_key(key: str) -> str | None:
    """The array path a Zarr v2 chunk key names, or `None` for anything else.

    A Zarr v3 sharded array stores one file per shard at ``<array>/c/<coords>``;
    those keys are also numeric-suffixed, so ``/c/`` is refused explicitly
    rather than parsed as a v2 chunk of a nested array (#270 review).
    """
    if "/c/" in key:
        return None
    match = _V2_CHUNK_KEY.match(key)
    return match.group("array") if match else None


class ShardWriteRecorder:
    """Every byte the Store hands to its storage layer, attributable to an array.

    Installed by `count_shard_writes`.  It records one entry per storage write so
    a build's write amplification -- bytes written divided by final bytes -- can
    be computed per array, which is what tells a whole-shard writer from one that
    rewrites every shard once per region (#249).  The bytes are the real
    (compressed, as the codec pipeline wrote them) bytes, not the selection's
    logical size.

    ``record_bytes=False`` records only the key: a build that writes gigabytes
    through hundreds of thousands of chunks pays to ``to_bytes`` each write, and
    a caller counting *how many times* a chunk was encoded (#270) does not need
    the bytes.  ``chunk_writes`` reads v3 shard keys, ``v2_chunk_writes`` the
    v2 chunk files, so the same recorder answers both Store formats.
    """

    def __init__(self, *, record_bytes: bool = True) -> None:
        self.writes: list[tuple[str, int]] = []
        self.record_bytes = record_bytes

    def record(self, key: str, value: Any) -> None:
        if not self.record_bytes:
            self.writes.append((key, 0))
            return
        to_bytes = getattr(value, "to_bytes", None)
        self.writes.append((key, len(to_bytes() if callable(to_bytes) else bytes(value))))

    def chunk_writes(self) -> dict[str, dict[str, int]]:
        """`array path -> {shard key: times written}`, metadata excluded."""
        counted: dict[str, dict[str, int]] = {}
        for key, _size in self.writes:
            path = array_key_of_store_key(key)
            if path is None:
                continue
            counted.setdefault(path, {})
            counted[path][key] = counted[path].get(key, 0) + 1
        return counted

    def v2_chunk_writes(self) -> dict[str, dict[str, int]]:
        """`array path -> {v2 chunk key: times written}`, metadata excluded."""
        counted: dict[str, dict[str, int]] = {}
        for key, _size in self.writes:
            path = v2_chunk_array_key(key)
            if path is None:
                continue
            counted.setdefault(path, {})
            counted[path][key] = counted[path].get(key, 0) + 1
        return counted

    def bytes_written_by_array(self) -> dict[str, int]:
        """Real bytes written per array path, over every chunk of every shard."""
        totals: dict[str, int] = {}
        for key, size in self.writes:
            path = array_key_of_store_key(key)
            if path is not None:
                totals[path] = totals.get(path, 0) + size
        return totals


#: The active `ShardWriteRecorder`, or `None` when nothing is counting.
_WRITE_RECORDER: ShardWriteRecorder | None = None


@contextmanager
def count_shard_writes(*, record_bytes: bool = True) -> Iterator[ShardWriteRecorder]:
    """Record every Store write's bytes and shard key while the block runs.

    A benchmark/test hook, the counting companion of `require_whole_shard_writes`:
    the guard answers "is this write whole-shard?" and the recorder answers "how
    many bytes did it cost?", which is what #249's write-amplification table
    needs.  It is off in production (a `None` check per storage write).

    ``record_bytes=False`` skips the per-write ``to_bytes`` and is what a caller
    counting *encodings* (#270) uses: the key count is the measurement, and
    materialising every compressed chunk only to measure its length would make
    the measurement change the run it measures.
    """
    global _WRITE_RECORDER
    if _WRITE_RECORDER is not None:
        raise RuntimeError("a shard-write recorder is already installed")
    recorder = ShardWriteRecorder(record_bytes=record_bytes)
    _WRITE_RECORDER = recorder
    try:
        yield recorder
    finally:
        _WRITE_RECORDER = None


def _count_store_write(key: str, value: Any) -> None:
    if _WRITE_RECORDER is not None:
        _WRITE_RECORDER.record(key, value)


def _block_selection_to_elements(array: Any, selection: Any) -> tuple[Any, ...]:
    """A `set_block_selection` selection, as the element selection it covers.

    `array.blocks` indexes the **outer block grid**: on a v3 sharded array that
    is the shard shape (`array.shards`, the metadata chunk grid), not the
    sharding codec's inner chunk (`array.chunks`).  Every block is therefore a
    whole shard, so any block selection -- a stepped one included -- covers whole
    shards; the returned selection spans the outer blocks it addresses.  (For an
    unsharded array the block grid is the chunk grid, but the whole-shard rule
    does not apply there.)
    """
    block_shape = array.shards if getattr(array, "shards", None) is not None else array.chunks
    shape = tuple(int(size) for size in array.shape)
    blocks = tuple(int(size) for size in block_shape)
    axes: list[Any] = []
    for entry, block, dim in zip(
        _normalise_selection(shape, selection), blocks, shape, strict=True
    ):
        n_blocks = max(1, -(-dim // block))
        if isinstance(entry, slice):
            start, stop, _step = entry.indices(n_blocks)
            # The selected blocks are all whole shards, so the covering
            # contiguous range is whole-shard aligned even when the block
            # selection is stepped (the gaps are not written, but they are not
            # partial shards either).
            axes.append(slice(start * block, min(stop * block, dim)))
        else:
            index = int(entry)
            if index < 0:
                index += n_blocks
            axes.append(slice(index * block, min((index + 1) * block, dim)))
    return tuple(axes)


_ORIGINAL_SETITEM = zarr.Array.__setitem__
_ORIGINAL_SET_BASIC = zarr.Array.set_basic_selection
_ORIGINAL_SET_ORTHOGONAL = zarr.Array.set_orthogonal_selection
_ORIGINAL_SET_MASK = zarr.Array.set_mask_selection
_ORIGINAL_SET_COORDINATE = zarr.Array.set_coordinate_selection
_ORIGINAL_SET_BLOCK = zarr.Array.set_block_selection
_ORIGINAL_ASYNC_SETITEM = zarr.AsyncArray.setitem
_SHARD_WRITE_GUARD_INSTALLED = False


def _guarded_setitem(self: Any, selection: Any, value: Any) -> None:
    require_whole_shard_write(self, selection)
    _ORIGINAL_SETITEM(self, selection, value)


def _guarded_set_basic_selection(
    self: Any, selection: Any, value: Any, *args: Any, **kwargs: Any
) -> None:
    require_whole_shard_write(self, selection)
    _ORIGINAL_SET_BASIC(self, selection, value, *args, **kwargs)


def _guarded_set_orthogonal_selection(
    self: Any, selection: Any, value: Any, *args: Any, **kwargs: Any
) -> None:
    # An orthogonal selection is a tuple of integer arrays: it names individual
    # cells, never a whole shard, so it is a partial write by construction.
    require_whole_shard_write(self, selection)
    _ORIGINAL_SET_ORTHOGONAL(self, selection, value, *args, **kwargs)


def _guarded_set_mask_selection(
    self: Any, mask: Any, value: Any, *args: Any, **kwargs: Any
) -> None:
    require_whole_shard_write(self, mask)
    _ORIGINAL_SET_MASK(self, mask, value, *args, **kwargs)


def _guarded_set_coordinate_selection(
    self: Any, selection: Any, value: Any, *args: Any, **kwargs: Any
) -> None:
    require_whole_shard_write(self, selection)
    _ORIGINAL_SET_COORDINATE(self, selection, value, *args, **kwargs)


def _guarded_set_block_selection(
    self: Any, selection: Any, value: Any, *args: Any, **kwargs: Any
) -> None:
    # `array.blocks` addresses whole outer blocks, i.e. whole shards on a sharded
    # array, so this is aligned by construction; the translation also makes a
    # changed block-grid definition fail the check rather than pass silently.
    require_whole_shard_write(self, _block_selection_to_elements(self, selection))
    _ORIGINAL_SET_BLOCK(self, selection, value, *args, **kwargs)


async def _guarded_async_setitem(
    self: Any, selection: Any, value: Any, *args: Any, **kwargs: Any
) -> None:
    require_whole_shard_write(self, selection)
    await _ORIGINAL_ASYNC_SETITEM(self, selection, value, *args, **kwargs)


def _install_shard_write_guard() -> None:
    global _SHARD_WRITE_GUARD_INSTALLED
    # Every public Zarr method that writes a *selection of cells*, which is what
    # a partial shard write is:
    #
    # * sync: `__setitem__`, the five `set_*_selection` methods, and
    #   `set_block_selection` (what `array.blocks[...] = ...` calls).  `oindex`
    #   and `vindex` delegate to `set_orthogonal_selection` /
    #   `set_coordinate_selection` / `set_mask_selection`.
    # * async: `AsyncArray.setitem` (the async `oindex`/`vindex` are read-only).
    #
    # `resize` changes the array's shape (a metadata and chunk-delete operation,
    # not a region write), and attribute writes are metadata, so neither is a
    # cell write and neither is judged here.
    type.__setattr__(zarr.Array, "__setitem__", _guarded_setitem)
    type.__setattr__(zarr.Array, "set_basic_selection", _guarded_set_basic_selection)
    type.__setattr__(
        zarr.Array, "set_orthogonal_selection", _guarded_set_orthogonal_selection
    )
    type.__setattr__(zarr.Array, "set_mask_selection", _guarded_set_mask_selection)
    type.__setattr__(
        zarr.Array, "set_coordinate_selection", _guarded_set_coordinate_selection
    )
    type.__setattr__(zarr.Array, "set_block_selection", _guarded_set_block_selection)
    type.__setattr__(zarr.AsyncArray, "setitem", _guarded_async_setitem)
    _SHARD_WRITE_GUARD_INSTALLED = True


def _uninstall_shard_write_guard() -> None:
    global _SHARD_WRITE_GUARD_INSTALLED
    type.__setattr__(zarr.Array, "__setitem__", _ORIGINAL_SETITEM)
    type.__setattr__(zarr.Array, "set_basic_selection", _ORIGINAL_SET_BASIC)
    type.__setattr__(
        zarr.Array, "set_orthogonal_selection", _ORIGINAL_SET_ORTHOGONAL
    )
    type.__setattr__(zarr.Array, "set_mask_selection", _ORIGINAL_SET_MASK)
    type.__setattr__(
        zarr.Array, "set_coordinate_selection", _ORIGINAL_SET_COORDINATE
    )
    type.__setattr__(zarr.Array, "set_block_selection", _ORIGINAL_SET_BLOCK)
    type.__setattr__(zarr.AsyncArray, "setitem", _ORIGINAL_ASYNC_SETITEM)
    _SHARD_WRITE_GUARD_INSTALLED = False


@contextmanager
def require_whole_shard_writes() -> Iterator[None]:
    """Check every Store array write for whole-shard coverage (test-time hook).

    Installed by a test that builds a store or exercises a writer; production
    never enters it unless ``OPEN_GWASDB_REQUIRE_WHOLE_SHARD_WRITES=1`` is set,
    which the real-data pilot does, so a genuine multi-shard build proves its
    writers are aligned rather than asserting it.

    Covered: every public Zarr method that writes a **selection of cells** --
    sync ``__setitem__``, ``set_basic_selection``, ``set_orthogonal_selection``,
    ``set_mask_selection``, ``set_coordinate_selection`` and
    ``set_block_selection`` (which ``array.blocks[...] = ...`` calls), and async
    ``AsyncArray.setitem``.  ``oindex``/``vindex`` delegate to the orthogonal /
    coordinate / mask setters, and the async ``oindex``/``vindex`` are read-only.
    \"Selection\" is the test: ``resize`` changes shape (a metadata and
    chunk-delete operation, not a region write) and attribute writes are
    metadata, so neither is a partial shard write and neither is judged here.
    """
    global _SHARD_WRITE_GUARD_INSTALLED
    if _SHARD_WRITE_GUARD_INSTALLED:
        yield
        return
    _install_shard_write_guard()
    try:
        yield
    finally:
        _uninstall_shard_write_guard()


if os.environ.get(_ENFORCE_WHOLE_SHARD_WRITES_ENV) == "1":
    _install_shard_write_guard()


#: The array a residual `se` plan's per-Analysis fit lives in, as its path
#: spells it. `ArrayRole.SE_COEFFICIENTS` is the role; this is the key, spelled
#: once, by the Store Reader's codec (`SE_COEFFICIENTS_KEY`), whose decoded
#: planes read it (#464); the role maps below read it here.
SE_COEFFICIENTS = ArrayRole.SE_COEFFICIENTS.value

#: The Dense roles, by array path, for a Dense Observed-Only or
#: Reference-Completed `data.zarr`.  The converter walks the source tree and
#: refuses any array this cannot name a role for; the two Dense-only completion
#: arrays (`imputed`, `on_panel`) are here so the mapping is complete for the
#: layout even though #245 refuses a Reference-Completed source (#248 adds it).
_DENSE_ROLES_BY_NAME: Mapping[str, ArrayRole] = MappingProxyType(
    {
        "z": ArrayRole.DENSE_STATISTIC_PLANE,
        "se": ArrayRole.DENSE_STATISTIC_PLANE,
        "eaf": ArrayRole.DENSE_STATISTIC_PLANE,
        "imputed": ArrayRole.DENSE_IMPUTED_MASK,
        "on_panel": ArrayRole.DENSE_ON_PANEL,
        "eaf_baseline": ArrayRole.PER_VARIANT,
        "eaf_reference": ArrayRole.PER_VARIANT,
        SE_COEFFICIENTS: ArrayRole.SE_COEFFICIENTS,
        "se_exception_index": ArrayRole.EXCEPTION_TABLE,
        "se_exception_value": ArrayRole.EXCEPTION_TABLE,
        "z_overflow_index": ArrayRole.EXCEPTION_TABLE,
        "z_overflow_value": ArrayRole.EXCEPTION_TABLE,
        "eaf_exception_index": ArrayRole.EXCEPTION_TABLE,
        "eaf_exception_value": ArrayRole.EXCEPTION_TABLE,
    }
)


#: The Ragged CSR group's arrays, by their leaf name under `ragged/` (#248).
#: `z`, `se`, `variant_index`, `eaf`, `imputed` and the optional per-association
#: sample size `n` (ADR 0067) are the CSR's parallel sequences; `offsets`
#: indexes them by Analysis; `eaf_baseline` and
#: `eaf_reference` are per-variant side arrays; the `*_exception_*` and
#: `z_overflow_*` tables are the exact-value tables.  The roles are the Ragged
#: ones so their shards follow #248's element caps, not #246's Dense shapes.
_RAGGED_ROLES_BY_NAME: Mapping[str, ArrayRole] = MappingProxyType(
    {
        "z": ArrayRole.ASSOCIATION_SEQUENCE,
        "se": ArrayRole.ASSOCIATION_SEQUENCE,
        "variant_index": ArrayRole.ASSOCIATION_SEQUENCE,
        "eaf": ArrayRole.ASSOCIATION_SEQUENCE,
        "imputed": ArrayRole.ASSOCIATION_SEQUENCE,
        "n": ArrayRole.ASSOCIATION_SEQUENCE,
        "offsets": ArrayRole.ASSOCIATION_OFFSETS,
        "eaf_baseline": ArrayRole.RAGGED_PER_VARIANT,
        "eaf_reference": ArrayRole.RAGGED_PER_VARIANT,
        SE_COEFFICIENTS: ArrayRole.SE_COEFFICIENTS,
        "se_exception_index": ArrayRole.RAGGED_EXCEPTION_TABLE,
        "se_exception_value": ArrayRole.RAGGED_EXCEPTION_TABLE,
        "z_overflow_index": ArrayRole.RAGGED_EXCEPTION_TABLE,
        "z_overflow_value": ArrayRole.RAGGED_EXCEPTION_TABLE,
        "eaf_exception_index": ArrayRole.RAGGED_EXCEPTION_TABLE,
        "eaf_exception_value": ArrayRole.RAGGED_EXCEPTION_TABLE,
    }
)


#: The arrays of the variant-centric index (`ragged/by_variant/...`, ADR 0060).
#: The index is a re-keying of the component's cell-keyed arrays, so each leaf
#: takes the same role its Analysis-sorted sibling does: `offsets` is the one
#: `RAGGED_PER_VARIANT` array (with its own inner-chunk hint, see
#: `inner_chunk_hint_for_path`), the cell sequences and the imputed mask are
#: `ASSOCIATION_SEQUENCE`, and the re-keyed exception tables are
#: `RAGGED_EXCEPTION_TABLE`.  `imputed`, `eaf`, `n` and the `se_exception_*` pair
#: are present only when the Analysis-sorted component carries them; that coupling
#: is checked by validation, not here, so a one-sided member is a release error
#: rather than a mapper guess.
_BY_VARIANT_ROLES_BY_LEAF: Mapping[str, ArrayRole] = MappingProxyType(
    {
        "offsets": ArrayRole.RAGGED_PER_VARIANT,
        "analysis_index": ArrayRole.ASSOCIATION_SEQUENCE,
        "z": ArrayRole.ASSOCIATION_SEQUENCE,
        "se": ArrayRole.ASSOCIATION_SEQUENCE,
        "eaf": ArrayRole.ASSOCIATION_SEQUENCE,
        "imputed": ArrayRole.ASSOCIATION_SEQUENCE,
        "n": ArrayRole.ASSOCIATION_SEQUENCE,
        "z_overflow_index": ArrayRole.RAGGED_EXCEPTION_TABLE,
        "z_overflow_value": ArrayRole.RAGGED_EXCEPTION_TABLE,
        "eaf_exception_index": ArrayRole.RAGGED_EXCEPTION_TABLE,
        "eaf_exception_value": ArrayRole.RAGGED_EXCEPTION_TABLE,
        "se_exception_index": ArrayRole.RAGGED_EXCEPTION_TABLE,
        "se_exception_value": ArrayRole.RAGGED_EXCEPTION_TABLE,
    }
)

#: The one path whose `RAGGED_PER_VARIANT` layout carries an explicit hint
#: (ADR 0060): `by_variant/offsets` is read a variant at a time, so 1,000
#: elements an inner chunk.  It is a *path* fact, not a role fact, because the
#: sibling `ragged/eaf_baseline` keeps the plane-following default.
BY_VARIANT_OFFSETS_PATH = "ragged/by_variant/offsets"


def inner_chunk_hint_for_path(path: str) -> int | None:
    """The explicit inner-chunk hint an array's **path** requires, or `None`.

    One array in the format needs a layout its role's default would get wrong:
    `ragged/by_variant/offsets`, whose `RAGGED_PER_VARIANT` policy would
    otherwise follow the sibling plane and chunk 200,000 at a time (ADR 0060).
    The hint is a scalar element count because `RAGGED_PER_VARIANT` is 1-D:
    `_per_variant` takes an int, unlike the Dense grid's `(rows, cols)` tuple.
    Every creator that derives a layout from a path -- the converter's
    `_plan_one_array`, the legacy relayout and the augment command -- must ask
    here rather than restate the number, or a converted array and a rebuilt one
    would silently disagree about their inner chunk (and a query reading one
    variant would decompress 200,000 offsets, not 1,000).
    """
    return BY_VARIANT_OFFSETS_CHUNK if path.strip("/") == BY_VARIANT_OFFSETS_PATH else None


def role_for_array_path(path: str) -> ArrayRole | None:
    """The `ArrayRole` a `data.zarr` array's path maps to, or `None`.

    The one path -> role mapping a 0.2.0 conversion is allowed to use.  A
    converter must never guess a layout: an array this returns `None` for fails
    the conversion loudly rather than being copied with some default shard.
    Top-hit tiers are `top_hits/<tier>/...`: the per-Analysis `analysis_offsets`
    is its own role, every other column is `TOP_HIT_INDEX`.  Rho is `rho/...`.
    The Ragged CSR group is `ragged/...` (#248); its sequences, per-variant
    side arrays and exception tables have their own roles so their shards are
    decided independently of the Dense shapes #246 owns.
    """
    name = path.strip("/")
    if not name:
        return None
    head, _, rest = name.partition("/")
    if rest:
        return _grouped_role(head, rest)
    return _DENSE_ROLES_BY_NAME.get(name)


#: The top-hit tier's arrays, by leaf name (#248).  `analysis_offsets` is its
#: own role; every column the tier stores with the same flat layout is
#: `TOP_HIT_INDEX`.  `eaf` (ADR 0040) and `imputed` (a Reference-Completed
#: release) are optional members, present only on some tiers, so a leaf outside
#: this map is an unknown format member and is refused rather than given the
#: generic role.
_TOP_HIT_ROLES_BY_LEAF: Mapping[str, ArrayRole] = MappingProxyType(
    {
        "analysis_offsets": ArrayRole.TOP_HIT_ANALYSIS_OFFSETS,
        "variant_index": ArrayRole.TOP_HIT_INDEX,
        "analysis_index": ArrayRole.TOP_HIT_INDEX,
        "abs_z": ArrayRole.TOP_HIT_INDEX,
        "z": ArrayRole.TOP_HIT_INDEX,
        "se": ArrayRole.TOP_HIT_INDEX,
        "p_value": ArrayRole.TOP_HIT_INDEX,
        "eaf": ArrayRole.TOP_HIT_INDEX,
        "imputed": ArrayRole.TOP_HIT_INDEX,
    }
)

#: The Rho Matrix group's arrays, by leaf name (#248): `rho`, `n_null` and
#: `variant_index`.  An unknown leaf is refused, not given `RHO_ARRAY`.
_RHO_ROLES_BY_LEAF: Mapping[str, ArrayRole] = MappingProxyType(
    {
        "rho": ArrayRole.RHO_ARRAY,
        "n_null": ArrayRole.RHO_ARRAY,
        "variant_index": ArrayRole.RHO_ARRAY,
    }
)


def _grouped_role(head: str, rest: str) -> ArrayRole | None:
    """The role of an array under a group: `top_hits`, `rho` or `ragged`.

    Each group has an explicit allowed-leaf map (#248): a leaf the format does
    not define returns `None`, so a conversion refuses it rather than copying an
    unknown member under a guessed role.  `top_hits/<tier>` is exactly one
    segment, so a deeper path is refused here as it is by
    `is_recorded_group_path` for the group itself.  `ragged/by_variant/...` is
    ADR 0060's variant-centric index.
    """
    if head == "top_hits":
        return _top_hit_tier_role(rest)
    if head == "rho":
        return _RHO_ROLES_BY_LEAF.get(rest)
    if head == "ragged":
        return _ragged_group_role(rest)
    return None


def _top_hit_tier_role(rest: str) -> ArrayRole | None:
    """The role of `top_hits/<tier>/<leaf>`, or `None` for a malformed path."""
    _tier, separator, leaf = rest.partition("/")
    if not separator or not leaf or "/" in leaf:
        return None
    return _TOP_HIT_ROLES_BY_LEAF.get(leaf)


def _ragged_group_role(rest: str) -> ArrayRole | None:
    """The role of an array under `ragged/`, including `by_variant/` (ADR 0060)."""
    prefix = "by_variant/"
    if rest.startswith(prefix):
        leaf = rest[len(prefix) :]
        if not leaf or "/" in leaf:
            return None
        return _BY_VARIANT_ROLES_BY_LEAF.get(leaf)
    if rest == "by_variant":
        return None
    return _RAGGED_ROLES_BY_NAME.get(rest)


#: The Dense `data.zarr` group paths a conversion may carry over.  A group is a
#: container, not an array, so `role_for_array_path` cannot judge it; an *empty*
#: unknown group would otherwise be recreated unnoticed.  `top_hits/<tier>` is
#: exactly one segment below `top_hits`, so a deeper unknown group is refused.
#: `ragged` is the Ragged CSR group (#248).
_RECORDED_GROUP_NAMES = frozenset({"top_hits", "ragged", "rho"})


def is_recorded_group_path(path: str) -> bool:
    """Whether a `data.zarr` group path is one the format defines.

    Used by the converter (#245) to refuse an unknown group rather than
    recreating it: the brief's rule is that an unmapped array **or group** fails
    the conversion.  `top_hits`, `rho` and the Ragged CSR `ragged` (#248) are
    the groups; a tier is exactly `top_hits/<name>`.
    """
    name = path.strip("/")
    if name in _RECORDED_GROUP_NAMES:
        return True
    head, _, rest = name.partition("/")
    if head == "ragged" and rest == "by_variant":
        # The variant-centric index group (ADR 0060).  It holds arrays only, so
        # exactly one extra segment is recorded and `ragged/by_variant/<x>/<y>`
        # is refused like any other deeper group.
        return True
    return head == "top_hits" and bool(rest) and "/" not in rest


# ── creation ─────────────────────────────────────────────────────────────────

#: Marker meaning "no explicit fill value": Zarr then uses the dtype default,
#: which is a different `.zarray` from an explicit `fill_value=None`.
_NO_FILL = object()

#: Marker meaning "the seam compressor": `compressor=None` means *uncompressed*,
#: which several exception tables deliberately are.
_SEAM_COMPRESSOR = object()


def _resolve_shape(name: str, data: Any, shape: tuple[int, ...] | None) -> tuple[int, ...]:
    """The array's shape, from `data`, `shape`, or both when they agree."""
    if data is not None:
        resolved = tuple(int(size) for size in np.shape(data))
        if shape is not None and tuple(int(size) for size in shape) != resolved:
            raise ValueError(
                f"array {name!r}: shape {shape!r} disagrees with data shape {resolved!r}"
            )
        return resolved
    if shape is None:
        raise ValueError(f"array {name!r}: one of data= or shape= is required")
    return tuple(int(size) for size in shape)


def _creation_kwargs(
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    *,
    data: Any,
    dtype: Any,
    fill_value: Any,
    compressor: Any,
    filters: Any,
    order: str,
    shards: tuple[int, ...] | None,
) -> dict[str, Any]:
    """The `create_array` keyword arguments one role's array is made with.

    ``compressors`` is zarr 3's spelling; on a v2 group it takes a single
    numcodecs codec and writes the same ``.zarray`` zarr 2.18 did.  ``order``
    is a v2-only memory-layout hint, so it is passed only when there is no
    shard (zarr 3 warns it has no effect for a v3 array).  The dtype, fill and
    filter arguments are in `_cast_kwargs`.
    """
    kwargs: dict[str, Any] = {
        "chunks": chunks,
        "compressors": _new_compressor() if compressor is _SEAM_COMPRESSOR else compressor,
        "shape": shape,
    }
    if shards is None:
        kwargs["order"] = order
    else:
        kwargs["shards"] = tuple(int(size) for size in shards)
    kwargs.update(_cast_kwargs(data=data, dtype=dtype, fill_value=fill_value, filters=filters))
    return kwargs


def _cast_kwargs(
    *, data: Any, dtype: Any, fill_value: Any, filters: Any
) -> dict[str, Any]:
    """The dtype / fill / filter arguments a create call carries.

    ``dtype`` is inferred from `data` when omitted, because zarr 3's
    ``create_array`` refuses ``data`` and ``dtype`` together; zarr 2's
    whole-array write was ``zarr.array(data, dtype=...)``, i.e. create-then-
    assign, and `create_array` assigns `data` after creation so a dtype the
    data does not carry still casts.  Omitting `fill_value` means the dtype
    default, which is a different array from an explicit ``fill_value=None``.
    """
    kwargs: dict[str, Any] = {}
    resolved_dtype = dtype
    if resolved_dtype is None and data is not None:
        resolved_dtype = np.asanyarray(data).dtype
    if resolved_dtype is not None:
        kwargs["dtype"] = resolved_dtype
    if fill_value is not _NO_FILL:
        kwargs["fill_value"] = fill_value
    if filters is not None:
        kwargs["filters"] = filters
    return kwargs


def _group_zarr_format(group: Any) -> int:
    """The Zarr on-disk format of the group an array is created under.

    A v3 group needs a sharded array with a v3 codec; a v2 group needs the
    numcodecs spelling and no shard.  Defaulting to 2 when the metadata cannot
    be read keeps a synthetic/test group behaving as it did before #247 rather
    than silently writing v3 metadata.
    """
    metadata = getattr(group, "metadata", None)
    return int(getattr(metadata, "zarr_format", 2))


def _v3_compressor(codec: Any) -> Any:
    """A caller's codec as the v3 spelling a sharded array needs, or `None`.

    The Store format defines **one** compressor configuration, published in the
    manifest, the `index.sqlite` `dense` blob and the root attrs.  A v3 array
    refuses a numcodecs codec (``'Blosc' object is not iterable``), so a writer
    that names the seam's configuration in its numcodecs spelling -- as every
    Dense builder did under 0.1.0 -- gets the v3 `BloscCodec` for it.  A codec
    that is *not* the seam's one configuration is refused rather than silently
    converted, because the published record would then describe bytes that are
    not stored.  ``None`` means uncompressed and stays `None`.
    """
    if codec is None or isinstance(codec, BloscCodec):
        return codec
    if isinstance(codec, Blosc):
        config = codec.get_config()
        canonical = (
            str(COMPRESSOR_RECORD["cname"]),
            int(COMPRESSOR_RECORD["clevel"]),
            int(_SHUFFLE_CODES[str(COMPRESSOR_RECORD["shuffle"])]),
        )
        observed = (
            str(config.get("cname")),
            int(config.get("clevel", -1)),
            int(config.get("shuffle", -1)),
        )
        if observed != canonical:
            raise ValueError(
                f"a v3 Store array holds the format's one compressor {canonical!r}, but "
                f"{observed!r} was requested; the manifest publishes the canonical "
                "record, so storing another would be a false claim"
            )
        return sharded_compressor()
    raise ValueError(
        f"a v3 Store array holds the seam's BloscCodec or None, not {codec!r} (issue #247)"
    )


def create_array(
    group: Any,
    name: str,
    role: ArrayRole,
    *,
    data: np.ndarray | None = None,
    shape: tuple[int, ...] | None = None,
    dtype: Any = None,
    fill_value: Any = _NO_FILL,
    compressor: Any = _SEAM_COMPRESSOR,
    hint: Any = None,
    component_chunk: int | None = None,
    inner_chunk: tuple[int, ...] | None = None,
    filters: Any = None,
    order: str = "C",
    shards: tuple[int, ...] | None = None,
    overwrite: bool = False,
) -> Any:
    """Create one Store array under the seam's compressor and `role` layout.

    Exactly one of `data` (a whole-array write) and `shape` (an array to be
    filled later) describes the array.  `fill_value` is the plane's own missing
    marker (spec §15): omit it for the dtype default, pass `fill_value=None`
    only when the metadata must literally say `null`.  `compressor=None` stores
    uncompressed.  `hint` overrides the role's default layout; `component_chunk`
    is the variant-axis chunk of the plane a `PER_VARIANT` array serves, passed
    in rather than sniffed from `group`; `inner_chunk` is an already-resolved
    inner chunk (the converter's path, #245) and wins over `hint`.

    `_creation_layout` decides the shard and the codec from the group's Zarr
    format (see its docstring).  `overwrite` deletes an existing array of the
    same name first; a site that expects a fresh name leaves it False, so
    writing twice fails loudly.
    """
    shape = _resolve_shape(name, data, shape)
    chunks = _resolve_inner_chunk(name, role, shape, hint, component_chunk, inner_chunk)
    resolved_shards, resolved_compressor = _creation_layout(
        group, name, role, shape, chunks, component_chunk, shards, compressor
    )
    if overwrite and name in group:
        del group[name]
    return _create_and_fill(
        group,
        name,
        _creation_kwargs(
            shape,
            chunks,
            data=data,
            dtype=dtype,
            fill_value=fill_value,
            compressor=resolved_compressor,
            filters=filters,
            order=order,
            shards=resolved_shards,
        ),
        data,
    )


def _creation_layout(
    group: Any,
    name: str,
    role: ArrayRole,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    component_chunk: int | None,
    shards: tuple[int, ...] | None,
    compressor: Any,
) -> tuple[tuple[int, ...] | None, Any]:
    """The `(shards, codec)` an array is created with, by the group's format.

    **On a v3 group the shard comes from the seam's role policy.**  `shards`
    overrides it (the converter's path, #245); left unset, every array a builder
    writes is sharded, because a 0.2.0 release's every array carries the
    `sharding_indexed` codec (ADR 0057).  The shard is checked against the inner
    chunk here, before zarr sees it.  A `compressor` naming the seam's numcodecs
    configuration is translated to its v3 spelling; another codec is refused.
    On a v2 group `shards` must be unset and the compressor passes through
    unchanged, so a 0.2.0 writer and a v2 fixture cannot be confused.
    """
    if _group_zarr_format(group) == 3:
        resolved_shards = (
            shard_layout(role, shape, inner_chunk=chunks, component_chunk=component_chunk)
            if shards is None
            else tuple(int(size) for size in shards)
        )
        require_shard_multiple(role, chunks, resolved_shards)
        resolved_compressor = (
            sharded_compressor() if compressor is _SEAM_COMPRESSOR else _v3_compressor(compressor)
        )
        return resolved_shards, resolved_compressor
    if shards is not None:
        raise ValueError(
            f"array {name!r}: a Zarr v2 array cannot be sharded (issue #247); "
            "sharding is format 0.2.0's"
        )
    return None, _new_compressor() if compressor is _SEAM_COMPRESSOR else compressor


def _resolve_inner_chunk(
    name: str,
    role: ArrayRole,
    shape: tuple[int, ...],
    hint: Any,
    component_chunk: int | None,
    inner_chunk: tuple[int, ...] | None,
) -> tuple[int, ...]:
    """The inner chunk an array is created with: the caller's, or the policy's.

    `inner_chunk` is the converter's path (#245): it has already computed and
    validated the layout through `chunk_layout`, and passing it explicitly is
    what lets a v3 write reproduce exactly the inner chunk the manifest
    records.  Otherwise the role policy decides, as it always has.
    """
    if inner_chunk is None:
        return chunk_layout(role, shape, hint=hint, component_chunk=component_chunk)
    resolved = tuple(int(size) for size in inner_chunk)
    if len(resolved) != len(shape):
        raise ValueError(
            f"array {name!r}: inner chunk {resolved!r} does not match shape {shape!r}"
        )
    return resolved


def _create_and_fill(group: Any, name: str, kwargs: dict[str, Any], data: Any) -> Any:
    """Create the array, then assign `data` into it when it is a whole-array write.

    zarr 2's ``create_dataset(data=...)`` was create-then-assign; the assign is
    what casts `data` to the caller's ``dtype`` when the two differ, and zarr 3
    refuses ``data`` and ``dtype`` together, so the cast has to happen here.
    """
    array = group.create_array(name, **kwargs)
    if data is not None:
        array[...] = data
    return array


def create_group(group: Any, name: str, *, replace: bool = True) -> Any:
    """Create one Store group, replacing any existing one by default."""
    if replace and name in group:
        del group[name]
    return group.create_group(name)


def require_group(group: Any, name: str) -> Any:
    """Return the named group, creating it only when it is absent."""
    return group.require_group(name)


class ConsolidatedMetadataError(RuntimeError):
    """A write under consolidated metadata the package cannot keep up to date."""


def _is_zarr_group_directory(directory: Path) -> bool:
    return (directory / ".zgroup").is_file() or (directory / "zarr.json").is_file()


def consolidated_metadata_records(directory: Path) -> list[Path]:
    """Every consolidated-metadata record that describes the group at `directory`.

    A record describes a group when it sits in the group's own directory or in an
    enclosing directory that is still part of the same Zarr hierarchy: a
    consolidated root lists every array beneath it. Zarr v2 keeps the record in
    ``.zmetadata``; Zarr v3 keeps it under ``consolidated_metadata`` in the group's
    ``zarr.json``.
    """
    records: list[Path] = []
    current = directory
    while True:
        v2 = current / ".zmetadata"
        if v2.is_file():
            records.append(v2)
        v3 = current / "zarr.json"
        if v3.is_file() and json.loads(v3.read_text()).get("consolidated_metadata"):
            records.append(v3)
        parent = current.parent
        if parent == current or not _is_zarr_group_directory(parent):
            return records
        current = parent


def refuse_under_consolidated_metadata(
    directory: Path, action: str, *, wiped: bool = False
) -> None:
    """Fail before a write that would leave consolidated metadata stale.

    zarr-python 3 opens a group from its consolidated metadata whenever a record
    exists (zarr 2.18 did not), and nothing the package writes updates one:
    creating, deleting or moving an array under a record leaves it describing
    arrays that are gone or changed, and the next open reads that instead (#244
    review). The package never consolidates, so a record came from elsewhere;
    refusing loudly is the only answer that cannot return stale arrays.
    `wiped` is for ``mode="w"``, which deletes the group's own record with it.
    """
    records = consolidated_metadata_records(directory)
    if wiped:
        records = [record for record in records if record.parent != directory]
    if records:
        listed = ", ".join(str(record) for record in records)
        raise ConsolidatedMetadataError(
            f"{action} {directory}: consolidated metadata in {listed} describes this group. "
            "zarr 3 reads that record in place of the live metadata, and this write would not "
            "update it. Remove the record, write, and consolidate again if it is wanted."
        )


#: The files a consolidated record copies: changing or deleting one stales it.
#: Chunk files are not recorded, so a chunk write needs no check.
_RECORDED_METADATA = frozenset({".zarray", ".zgroup", ".zattrs", "zarr.json"})


class _GuardedLocalStore(LocalStore):
    """The local store every Store group is opened on.

    It refuses a metadata write or a delete that consolidated metadata describes.
    The check at `open_group` cannot see a record that appears after a handle was
    opened. Nor can it see every route a write takes: `create_array`,
    `create_group` and `require_group` go through the seam, but attribute writes
    and ``del group[name]`` go through zarr's own API. Every one of them reaches
    the store, so the store is where the check is repeated (#244 review round 2).
    Deleting a directory takes any record inside it along, so only records
    enclosing it count, as for ``mode="w"``.
    """

    def _check_write(self, key: str) -> None:
        if key.rsplit("/", 1)[-1] in _RECORDED_METADATA:
            refuse_under_consolidated_metadata((self.root / key).parent, f"writing {key!r} in")

    def _check_delete(self, key: str) -> None:
        """Refuse any delete beneath a record, chunk files included.

        A delete must be refused before the first destructive step, and zarr does
        not always change metadata first: a shrinking ``resize`` deletes the
        chunks beyond the new shape before it writes the new shape, so the
        metadata write would be refused only after chunks were gone (#244 review
        round 3). With ``write_empty_chunks`` on, an ordinary write never deletes
        a chunk, so this costs nothing on the build path.
        """
        target = self.root / key
        if target.is_dir():
            refuse_under_consolidated_metadata(target, f"deleting {key!r} in", wiped=True)
        else:
            refuse_under_consolidated_metadata(target.parent, f"deleting {key!r} in")

    async def set(self, key: str, value: Buffer) -> None:
        self._check_write(key)
        _count_store_write(key, value)
        await super().set(key, value)

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        self._check_write(key)
        _count_store_write(key, value)
        await super().set_if_not_exists(key, value)

    def set_sync(self, key: str, value: Buffer) -> None:
        self._check_write(key)
        _count_store_write(key, value)
        super().set_sync(key, value)

    async def delete(self, key: str) -> None:
        self._check_delete(key)
        await super().delete(key)

    def delete_sync(self, key: str) -> None:
        self._check_delete(key)
        super().delete_sync(key)

    async def delete_dir(self, prefix: str) -> None:
        self._check_delete(prefix)
        await super().delete_dir(prefix)

    async def clear(self) -> None:
        self._check_delete("")
        await super().clear()

    async def move(self, dest_root: Path | str) -> None:
        self._check_delete("")
        await super().move(dest_root)


def _open_local_store(path: str | Path, mode: ZarrMode) -> _GuardedLocalStore:
    """The store zarr would open for a local `path` in `mode`, guarded.

    This is the same `LocalStore.open` call zarr's own path handling makes, so the
    mode means exactly what it did: `r` and `r+` require the directory, the others
    create it.
    """
    opening = _GuardedLocalStore.open(root=Path(path), mode=mode, read_only=mode == "r")
    store: _GuardedLocalStore = zarr_sync(opening)
    return store


#: Zarr entries whose bytes live in a directory of their own, which is what a
#: filesystem rename moves as a unit.
_LOCAL_STORE_ATTR = "root"


def _local_group_directory(group: Any) -> Path | None:
    """The on-disk directory a LocalStore-backed group lives in, or `None`.

    zarr 3's ``LocalStore`` exposes its root as a ``Path``; the group's own
    location inside it is ``group.path``.  Any other store returns `None` and
    the caller refuses rather than guessing.
    """
    root = getattr(group.store, _LOCAL_STORE_ATTR, None)
    if not isinstance(root, (str, Path)):
        return None
    location = group.path
    return Path(root) / location if location else Path(root)


def move_in_group(group: Any, source: str, dest: str) -> None:
    """Rename the entry `source` to `dest` inside `group`.

    zarr-python 3.4's ``Group.move`` raises ``NotImplementedError``, but the
    package relies on a real rename in two places: the SE float16 fallback
    swaps its staged plane into place, and `repair` swaps a rechunked array in
    (and restores the original if the swap fails).  zarr 2 renamed the entry in
    the store; every Store Release is a local directory, so this is the same
    ``os.replace`` of the array directory: byte-preserving, which a
    copy-through-zarr would not be for Blosc (its default thread count makes
    recompression non-reproducible, see #243).

    Each move is atomic; a swap built from two moves is not.  A process that
    dies between them leaves neither name in place, and no exception handler
    runs to roll back.  The SE fallback swaps only inside a staged release,
    which a failed build discards.  `repair` swaps inside a published release,
    so its next run recovers whatever state a death left (#244 review).

    A store that is not a local directory fails loudly: silently degrading to a
    copy would change the bytes and could leave a half-swapped store behind. So
    does a group that consolidated metadata describes, which a move would leave
    stale (`refuse_under_consolidated_metadata`).
    """
    local = _local_group_directory(group)
    if local is None:
        raise NotImplementedError(
            f"move_in_group({source!r} -> {dest!r}) needs a local directory store; "
            f"{type(group.store).__name__} cannot rename an entry. Open the release "
            "from a path rather than an in-memory store."
        )
    refuse_under_consolidated_metadata(local, f"moving {source!r} to {dest!r} in")
    os.replace(local / source, local / dest)


#: The modes zarr-python 3's `zarr.open_group` accepts, by name.  A lookup table
#: rather than a cast, so the seam passes zarr a `Literal` it has checked.
ZarrMode = Literal["r", "r+", "a", "w", "w-"]
_ZARR_MODES: Mapping[str, ZarrMode] = MappingProxyType(
    {"r": "r", "r+": "r+", "a": "a", "w": "w", "w-": "w-"}
)

#: `zarr.open_group` modes that create the group (or wipe it) when it is not
#: there.  `r` and `r+` require it to exist and are not creation.  zarr 2's
#: ``x`` is gone: zarr 3 rejects it with a bare ``AssertionError``.
CREATION_MODES = frozenset({"w", "a", "w-"})


def _zarr_mode(mode: str) -> ZarrMode:
    """`mode` as the `Literal` zarr 3 takes, or a `ValueError` naming the valid ones."""
    try:
        return _ZARR_MODES[mode]
    except KeyError:
        allowed = sorted(_ZARR_MODES)
        raise ValueError(f"zarr 3 opens a group in one of {allowed}, not {mode!r}") from None

#: The Zarr on-disk format every *created* array and group is written in.
#: Format 0.2.0 is Zarr v3 with sharding (ADR 0057, ADR 0058), so as of #247
#: every builder creates v3: passing the format explicitly on every creation-mode
#: open is the whole point, because zarr-python 3's ``open_group(..., mode="w")``
#: defaults to ``zarr_format=None``, which *creates a Zarr v3 group* -- correct
#: now, but a silent Store format change if the constant ever drifts from
#: ``CURRENT_FORMAT_VERSION``.  Read-mode opens pass ``zarr_format=None`` so a
#: 0.1.0 v2 store (which is still readable) opens; zarr 3 auto-detects the format
#: from the existing metadata.
STORE_ZARR_FORMAT: Final = 3

#: The Zarr on-disk formats zarr-python 3 can be asked to create, as the
#: `Literal` it takes.  A lookup rather than a cast, so an unknown value is a
#: `ValueError` here instead of something zarr refuses obscurely.
_ZARR_FORMATS: Mapping[int, Literal[2, 3]] = MappingProxyType({2: 2, 3: 3})

#: The Zarr format a 0.2.0 Store Release is written in (issue #245).  Kept as a
#: name for the format 0.2.0 releases carry so a reader of the code can see the
#: pair; it *is* `STORE_ZARR_FORMAT` now that every builder writes 0.2.0.
SHARDED_STORE_ZARR_FORMAT: Final = STORE_ZARR_FORMAT


def _checked_zarr_format(zarr_format: int | None) -> Literal[2, 3]:
    """`zarr_format` as the `Literal` zarr takes, or a `ValueError`."""
    if zarr_format is None or zarr_format not in _ZARR_FORMATS:
        raise ValueError(
            f"a Store Release is written in Zarr format 2 or 3, not {zarr_format!r}"
        )
    return _ZARR_FORMATS[zarr_format]


def _open_group_format(mode: str, zarr_format: int | None = None) -> Literal[2, 3] | None:
    """The ``zarr_format`` an open in `mode` must use.

    A creating mode must declare a format -- v2 by default, so the builders are
    unchanged, or the caller's `zarr_format` for a v3 release (#245).  A read
    mode must not declare anything, so the existing metadata decides (which is
    how zarr 3 reads a v2 store and how #245's v3 store reads).
    """
    if mode not in CREATION_MODES:
        return None
    return STORE_ZARR_FORMAT if zarr_format is None else _checked_zarr_format(zarr_format)


def open_group(
    path: str | Path, mode: str = "r", *, zarr_format: int | None = None
) -> Any:
    """Open a Zarr group, read-only by default.

    With the read-only open, `opengwasdb.store_reader.arrays.open_group`, which
    mode ``r`` hands to, the one place `zarr.open_group` is called, so the Store
    format is declared once.  `mode` may be a creating mode here; `open_group_for_write` is the
    named entry point for those, and this general form exists for `r`/`r+` and
    for the release-envelope ``arrays(mode=...)`` methods that forward their
    mode.  A creating mode is pinned to `STORE_ZARR_FORMAT` unless `zarr_format`
    overrides it (the converter's v3 output, #245); a read mode leaves the
    format to the stored metadata.  Any mode but ``r`` refuses a group that
    consolidated metadata describes (`refuse_under_consolidated_metadata`).
    """
    zarr_mode = _zarr_mode(mode)
    if zarr_mode == "r":
        return reader_arrays.open_group(path)
    refuse_under_consolidated_metadata(
        Path(path), f"opening in mode {mode!r}", wiped=zarr_mode == "w"
    )
    store = _open_local_store(path, zarr_mode)
    return zarr.open_group(
        store, mode=zarr_mode, zarr_format=_open_group_format(mode, zarr_format)
    )


def consolidate_group_metadata(directory: str | Path, *, zarr_format: int) -> None:
    """Regenerate a group's consolidated-metadata record from its live metadata.

    The one place this package calls zarr's consolidation, so the array-creation
    seam remains the only module that talks to zarr directly
    (`tests/test_array_creation_seam.py`).  The augment path takes a store's
    record out of the way to write beneath it and must put a regenerated one
    back; that refresh goes through here rather than a second zarr call site.
    """
    zarr.consolidate_metadata(str(directory), zarr_format=_checked_zarr_format(zarr_format))


def open_group_for_write(
    path: str | Path, mode: str, *, zarr_format: int | None = None
) -> Any:
    """Open a Zarr group for writing, creating it when it is absent.

    `mode` must be one of ``w``/``a``/``w-`` and is **required**: ``a``
    and ``w`` differ on a resumed build (``w`` wipes the staged group, ``a``
    keeps it), so a silent default here would change what a resume finds.  A
    read mode is a bug (the caller meant `open_group`) and fails loudly.

    The group is created in `STORE_ZARR_FORMAT` (Zarr v2 until #247) unless
    `zarr_format` overrides it -- the converter's v3 output passes 3 (#245).
    Without the explicit format zarr-python 3 would create a Zarr v3 group and
    silently change the Store format; that is why every write-mode open routes
    here.
    """
    if mode not in CREATION_MODES:
        raise ValueError(
            f"open_group_for_write needs one of {sorted(CREATION_MODES)}, got {mode!r}; "
            "use open_group for a read mode"
        )
    refuse_under_consolidated_metadata(Path(path), f"opening in mode {mode!r}", wiped=mode == "w")
    zarr_mode = _zarr_mode(mode)
    store = _open_local_store(path, zarr_mode)
    return zarr.open_group(
        store, mode=zarr_mode, zarr_format=_open_group_format(mode, zarr_format)
    )
