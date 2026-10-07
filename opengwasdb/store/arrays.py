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
* **zarr's process-wide runtime configuration** (the section after the
  imports): chunks that are all fill value are still written, Blosc decodes
  with its internal threads, and every array goes through zarr's fused codec
  pipeline with one worker.  zarr-python 3 holds these in runtime config
  rather than array metadata, so they are set once, when this module is
  imported (#244).

The role -> layout policy is one table in this module (`_LAYOUTS`) and nothing
else chooses chunks.  A role that does not fit an existing entry gets a new
one rather than a call site that passes `chunks=` by hand.
"""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal

import numcodecs
import numpy as np
import zarr
from numcodecs import Blosc
from zarr.codecs import BloscCodec
from zarr.core.buffer import Buffer
from zarr.core.sync import sync as zarr_sync
from zarr.storage import LocalStore

# The module whose import switches Blosc's threads off.  ``import zarr`` loads
# it today; importing it by name makes sure it has run before the setting below
# turns them back on, should zarr ever load it lazily.  (`importlib` rather than
# an import statement so mypy does not follow into zarr's sources.)
importlib.import_module("zarr.codecs.blosc")

# ── zarr runtime configuration ───────────────────────────────────────────────
#
# zarr-python 3 keeps three behaviours in process-wide runtime config rather
# than in array metadata.  They are set here, once, in the module every Store
# array is created and opened through, so every array the package touches gets
# them and none can be opened without them.  ADR 0056 records why each is set,
# what was rejected and what it costs.

#: zarr 2 wrote every chunk, including one that is entirely the fill value: its
#: ``write_empty_chunks`` default was True.  zarr-python 3 defaults it to False,
#: which would silently drop those chunk files from a built store and change
#: the file set a release holds.  The setting is *runtime* config, not stored
#: metadata, so pinning it per array only covers the object `create_array`
#: returns -- every ``group[name]`` reopens with the default.  It is therefore
#: set once, here, the module that owns the Store's physical layout; every
#: array this package creates or reopens then writes empty chunks as zarr 2
#: did.  #247 must revisit this when builders move to Zarr v3 shards.
zarr.config.set({"array.write_empty_chunks": True})

#: Blosc's internal threads, back on (#244).  ``import zarr`` runs
#: ``numcodecs.blosc.use_threads = False`` for the whole process, and zarr 3
#: decodes on worker threads where numcodecs' adaptive default would say no
#: anyway, so every chunk decoded single-threaded: ~4.5 ms against ~0.6 ms with
#: Blosc's 8 threads, for a ``[1000, 1000]`` int16 chunk of OGS-00009
#: (``benchmarks/zarr3_blosc_decode.py``).  zarr 2.18 decoded on the main
#: thread with 8 Blosc threads; this restores that.
#:
#: On its own, under zarr's default pipeline, it is not a speed-up: decodes
#: issued concurrently from zarr's pool queue on numcodecs' lock (below), so
#: one Analysis genome-wide went from 76 s to 88 s and random lookups slowed by
#: 21-41%, while phewas went from 50 ms to 34 ms (medians of three fresh
#: processes).  It pays off with the one-worker fused pipeline below, which
#: decodes one chunk at a time with all of Blosc's threads.
#:
#: It is safe, from numcodecs 0.17 (the pinned floor) and its c-blosc 1.21.7:
#:
#: * threads: a threaded call uses Blosc's one global context, and numcodecs
#:   serialises every such call -- compress and decompress -- under a module
#:   ``threading.Lock``, with the GIL released inside it.  Concurrent decodes
#:   queue rather than race.  (0.16 did not take the lock on decompress.)
#: * forks: a forked process never uses the global context, because numcodecs
#:   compares the pid with the importing process's and runs single-threaded
#:   context functions in a child whatever ``use_threads`` says; it also
#:   re-creates its lock in the child, and c-blosc discards the inherited global
#:   context in its own ``pthread_atfork`` child handler.
#:
#: It is global because the flag is: numcodecs has no per-call or per-thread
#: form, so a "query-only" switch could only mean "on from the first query
#: onwards", a mode that depends on what a process did earlier.  The cost
#: falls on builds, and it is the one zarr 2.18 already had: a chunk of at
#: least two Blosc blocks (256 KiB uncompressed at zstd clevel 3, so every
#: ``[1000, 1000]`` Dense chunk) compressed in the parent writes its blocks in
#: completion order, so its bytes differ run to run while its decoded values
#: and its compressed size do not.  Compare built stores byte for byte under
#: ``BLOSC_NTHREADS=1``, as #243 did.  The SE plan reads compressed sizes only,
#: and forked workers stay single-threaded.
if not hasattr(numcodecs.blosc, "use_threads"):
    raise ImportError(
        "numcodecs.blosc no longer has `use_threads`; the seam cannot restore threaded "
        "Blosc decoding, and setting the old name would silently do nothing (#244)"
    )
numcodecs.blosc.use_threads = True

#: Every array reads and writes through zarr's ``FusedCodecPipeline`` (opt-in
#: from zarr 3.3), with one worker (#244).  The default ``BatchedCodecPipeline``
#: schedules each chunk's fetch and decode as separate event-loop tasks; the
#: fused one fetches, decodes and scatters a whole selection in one hop to a
#: worker thread.  On OGS-00009 with Blosc threads on it took one Analysis
#: genome-wide from 88 s to 24 s (zarr 2.18: 26 s), random lookups from 186 ms
#: and 1,051 ms to 103 ms and 575 ms, and phewas from 34 ms to 23 ms (medians
#: of three fresh processes each).
#:
#: ``max_workers = 1`` is the measured choice, and the fork-safe one:
#:
#: * With Blosc threads on, chunk decodes queue on numcodecs' lock anyway, so a
#:   pool of workers adds contention, not decode throughput: with its default
#:   pool (224 workers here) the same read took 41-46 s, and random lookups
#:   were 23-42% slower; only the one-window read (4,241,966 associations)
#:   gained, by ~7%.
#: * With more than one worker the pipeline keeps a module-level
#:   ``ThreadPoolExecutor``, and zarr 3.4's after-fork reset clears its event
#:   loop and executor but not that pool (zarr-developers/zarr-python#4478).
#:   A forked build worker inherits the pool without its threads.  A read there
#:   of more than one chunk, but of no more chunks than the idle permits the
#:   parent's pool left, queues work that nothing runs and never returns --
#:   reproduced with the package's own ``ordered_map`` and on OGS-00009's
#:   top-hit gather.  A single-chunk read never uses the pool, so one-chunk
#:   fixtures cannot show it.  With one worker the pool is never created; do
#:   not raise ``max_workers`` (ADR 0056).
#:
#: Writes take the same path: a band write encodes its chunks one at a time
#: (multi-threaded inside Blosc), as zarr 2.18 did.  A forked worker decodes
#: single-threaded and now also one chunk at a time.
_FUSED_PIPELINE = "zarr.core.codec_pipeline.FusedCodecPipeline"
if not hasattr(importlib.import_module("zarr.core.codec_pipeline"), "FusedCodecPipeline"):
    raise ImportError(
        f"{_FUSED_PIPELINE} is gone; the seam's read path and its fork guard were "
        "written for it (#244)"
    )
zarr.config.set({"codec_pipeline.path": _FUSED_PIPELINE, "codec_pipeline.max_workers": 1})

__all__ = [
    "ASSOCIATION_OFFSETS_CHUNK",
    "ASSOCIATION_SEQUENCE_CHUNK",
    "COMPRESSOR_RECORD",
    "DENSE_CHUNK_SHAPE",
    "DENSE_SHARD_SHAPE",
    "EXCEPTION_TABLE_CHUNK",
    "PER_VARIANT_CHUNK",
    "RAGGED_SEQUENCE_SHARD_ELEMENTS",
    "RAGGED_SIDE_SHARD_ELEMENTS",
    "RHO_CHUNK_ROWS",
    "SE_COEFFICIENTS_ROWS",
    "PartialShardWriteError",
    "SHARDED_COMPRESSOR_RECORD",
    "SHARDED_STORE_ZARR_FORMAT",
    "STORE_ZARR_FORMAT",
    "TOP_HIT_CHUNK_SIZE",
    "ArrayRole",
    "array_length",
    "chunk_layout",
    "component_chunk_size",
    "component_variant_chunk",
    "compressor",
    "compressor_of",
    "create_array",
    "create_group",
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
    "shard_layout",
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


def array_length(array: Any) -> int:
    """The leading dimension of an array -- what zarr 2's ``len(array)`` meant.

    zarr-python 3 removed ``Array.__len__``, so ``len(array)`` now raises
    ``TypeError``; the leading axis is the length every call site wanted.  A
    2-D top-hit tier reads the same way as a 1-D CSR plane.
    """
    return int(array.shape[0])


def inner_chunk_of(array: Any) -> tuple[int, ...]:
    """The *inner* chunk of a Store array, sharded or not.

    zarr-python 3's ``Array.chunks`` is the inner chunk for a sharded array and
    ``Array.shards`` is the outer shard -- the reverse of what a reader might
    assume from the v2 spelling.  Every rule about the unit a query reads (the
    per-variant chunk rule, issue #135; the recorded-layout rule, #245) must
    use this, never ``shards``, or it would judge the file unit instead of the
    read unit.  A store with one chunk per shard has ``chunks == shards``.
    """
    return tuple(int(size) for size in array.chunks)


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


#: `create_array`'s `compressor=` parameter would otherwise shadow the function
#: inside that function's body.
_new_compressor = compressor


# ── chunk-layout policy ──────────────────────────────────────────────────────

#: Length-indexed flat arrays (Ragged CSR association sequences): one chunk of
#: this many cells, never clipped -- a short array keeps the same declared chunk
#: as a long one so the chunk size does not encode the component's size.
ASSOCIATION_SEQUENCE_CHUNK = 200_000

#: The per-Analysis offset array that indexes an association sequence.
ASSOCIATION_OFFSETS_CHUNK = 10_000

#: Ceiling for a per-variant side array (EAF baseline/reference).  The policy
#: follows the chunk of a sibling variant-axis array when the group has one, so
#: a side array is read in the same tiles as the plane it serves.
PER_VARIANT_CHUNK = 200_000

#: Row count of the two SE decode coefficients per Analysis.
SE_COEFFICIENTS_ROWS = 1024

#: Row count of a Rho Matrix array.
RHO_CHUNK_ROWS = 1_000_000

#: The default inner chunk of a Dense grid: 1,000 variant rows and 64 Analyses.
#: This is `DEFAULT_CHUNK_SHAPE` in the Dense constants module, and the one
#: definition: the Dense module imports *this* name, so the converter and the
#: Dense builders cannot disagree about the default grid layout.  The Analysis
#: axis narrowed from 1,000 to 64 in format 0.2.0, decided by #246 and recorded
#: in ADR 0058; the 1,000-row variant axis is unchanged.
DENSE_CHUNK_SHAPE = (1000, 64)

#: One flat array of a top-hit threshold tier, when a caller supplies no
#: override.  `layouts/dense/top_hits.TOP_HIT_CHUNK_SIZE` is this value.
TOP_HIT_CHUNK_SIZE = 16_384

#: One half of an exact-value exception/overflow table, when a caller supplies
#: no override.  `encoding/codec.EXACT_TABLE_CHUNK` is this value.
EXCEPTION_TABLE_CHUNK = 200_000


class ArrayRole(StrEnum):
    """What an array *is*, which is what fixes its chunk layout.

    The role is about the array's part in the store, not its dtype: `z` and
    `se` are both `DENSE_STATISTIC_PLANE` and are stored as different dtypes.
    """

    #: A 2-D variant x Analysis Dense statistic grid (`z`, `se`, `eaf`) or the
    #: float32 scratch plane a builder stages before the encoding is decided.
    DENSE_STATISTIC_PLANE = "dense_statistic_plane"
    #: The 2-D `uint8` imputed mask that accompanies a completed Dense grid.
    DENSE_IMPUTED_MASK = "dense_imputed_mask"
    #: The 1-D per-variant `uint8` `on_panel` mask of a completed Dense grid.
    DENSE_ON_PANEL = "dense_on_panel"
    #: A 1-D Ragged CSR association sequence: `z`, `se`, `eaf`, `variant_index`,
    #: `imputed`, or an empty plane to be filled region by region.
    ASSOCIATION_SEQUENCE = "association_sequence"
    #: The per-Analysis offset array of a Ragged CSR component.
    ASSOCIATION_OFFSETS = "association_offsets"
    #: A 1-D per-variant side array: `eaf_baseline`, `eaf_reference`.
    PER_VARIANT = "per_variant"
    #: A Ragged group's 1-D per-variant side array (`ragged/eaf_baseline`,
    #: `ragged/eaf_reference`).  The same plane as `PER_VARIANT` but a distinct
    #: role because #248 sizes a Ragged side array's *shard* differently from a
    #: Dense one, and #246 owns the Dense decision.
    RAGGED_PER_VARIANT = "ragged_per_variant"
    #: A Ragged group's exact-value exception/overflow table
    #: (`ragged/z_overflow_index`, `ragged/eaf_exception_index`, ...).  Distinct
    #: from the Dense `EXCEPTION_TABLE` for the same reason as
    #: `RAGGED_PER_VARIANT`: #248 bounds its shard so a Ragged overflow table is
    #: not one multi-hundred-MB file, without changing #246's Dense shapes.
    RAGGED_EXCEPTION_TABLE = "ragged_exception_table"
    #: One flat array of a top-hit threshold tier (`variant_index`, `abs_z`,
    #: `z`, `se`, `p_value`, `imputed`, `eaf`).
    TOP_HIT_INDEX = "top_hit_index"
    #: The per-Analysis `analysis_offsets` array of a top-hit threshold tier.
    TOP_HIT_ANALYSIS_OFFSETS = "top_hit_analysis_offsets"
    #: One half of a plane's exact-value exception/overflow table.
    EXCEPTION_TABLE = "exception_table"
    #: The two SE decode coefficients per Analysis.
    SE_COEFFICIENTS = "se_coefficients"
    #: One array of the Rho Matrix group (`rho`, `n_null`, `variant_index`).
    RHO_ARRAY = "rho_array"


@dataclass(frozen=True)
class _LayoutContext:
    """Everything a layout policy is allowed to depend on.

    `component_chunk` is the variant-axis chunk of the component plane a
    per-variant side array serves.  It is a caller-supplied fact, not something
    the policy sniffs from a group: the converter (#245) has the plane's chunk
    and must be able to derive the same side-array layout the builder wrote
    (spec §6).
    """

    shape: tuple[int, ...]
    hint: Any
    component_chunk: int | None


def _dense_grid(ctx: _LayoutContext) -> tuple[int, ...]:
    """A 2-D Dense grid, clipped to its own dimensions (ADR 0021)."""
    hint = ctx.hint if ctx.hint is not None else DENSE_CHUNK_SHAPE
    return (min(int(hint[0]), ctx.shape[0]), min(int(hint[1]), ctx.shape[1]))


def _dense_on_panel(ctx: _LayoutContext) -> tuple[int, ...]:
    """The per-variant Dense mask, on the grid's row chunk."""
    hint = ctx.hint if ctx.hint is not None else DENSE_CHUNK_SHAPE
    return (min(int(hint[0]), ctx.shape[0]),)


def _association_sequence(ctx: _LayoutContext) -> tuple[int, ...]:
    """A flat CSR sequence: the declared chunk, or an explicit override.

    The declared chunk is fixed, not clipped to the component's length -- a
    short component keeps the same chunk as a long one.  A caller that passes
    an explicit `chunks=(...)` gets exactly that, as it did before the seam.
    """
    if ctx.hint is not None:
        return tuple(int(size) for size in ctx.hint)
    return (ASSOCIATION_SEQUENCE_CHUNK,)


def _association_offsets(ctx: _LayoutContext) -> tuple[int, ...]:
    """The CSR per-Analysis offset array, or an explicit override."""
    if ctx.hint is not None:
        return tuple(int(size) for size in ctx.hint)
    return (ASSOCIATION_OFFSETS_CHUNK,)


def _per_variant(ctx: _LayoutContext) -> tuple[int, ...]:
    """A per-variant side array, following the plane it serves.

    The component plane's variant-axis chunk (spec §6: the side array must be
    no coarser than the plane) comes in as `component_chunk`.  An explicit
    `hint` -- the `chunk` override the writers accept -- wins over it, clipped
    to the array length as before.
    """
    length = ctx.shape[0]
    if ctx.hint is not None:
        return (max(1, min(int(ctx.hint), max(length, 1))),)
    return (component_chunk_size(ctx.component_chunk, length),)


def _length_clipped(ctx: _LayoutContext, default: int) -> tuple[int, ...]:
    """A flat array whose chunk is an override (or `default`) clipped to length."""
    hint = default if ctx.hint is None else int(ctx.hint)
    return (max(1, min(ctx.shape[0], hint)),)


def _top_hit_index(ctx: _LayoutContext) -> tuple[int, ...]:
    """One top-hit tier's flat column, whole-array default clipped to its hits."""
    return _length_clipped(ctx, TOP_HIT_CHUNK_SIZE)


def _exception_table(ctx: _LayoutContext) -> tuple[int, ...]:
    """One half of a plane's exception table."""
    return _length_clipped(ctx, EXCEPTION_TABLE_CHUNK)


def _top_hit_analysis_offsets(ctx: _LayoutContext) -> tuple[int, ...]:
    """One top-hit tier's per-Analysis offsets, whole in one chunk."""
    return (max(1, ctx.shape[0]),)


def _se_coefficients(ctx: _LayoutContext) -> tuple[int, ...]:
    """The (n_analyses, 2) coefficient table."""
    return (max(1, min(ctx.shape[0], SE_COEFFICIENTS_ROWS)), ctx.shape[1])


def _rho_array(ctx: _LayoutContext) -> tuple[int, ...]:
    """One Rho Matrix array."""
    return (max(1, min(ctx.shape[0], RHO_CHUNK_ROWS)),)


#: The role -> physical-layout policy.  One entry per role; later tickets that
#: change the layout (Zarr v3 sharding, #247) change this table, and the
#: converter (#245) reads the same table so builders and converter agree.  Each
#: policy is a function of `role + shape + component_chunk` (the last used only
#: by `PER_VARIANT`, and `None` for every other role and for a component with no
#: plane); a caller's `hint` is an override the policy honours.
_LAYOUTS: Mapping[ArrayRole, Callable[[_LayoutContext], tuple[int, ...]]] = MappingProxyType(
    {
        ArrayRole.DENSE_STATISTIC_PLANE: _dense_grid,
        ArrayRole.DENSE_IMPUTED_MASK: _dense_grid,
        ArrayRole.DENSE_ON_PANEL: _dense_on_panel,
        ArrayRole.ASSOCIATION_SEQUENCE: _association_sequence,
        ArrayRole.ASSOCIATION_OFFSETS: _association_offsets,
        ArrayRole.PER_VARIANT: _per_variant,
        ArrayRole.RAGGED_PER_VARIANT: _per_variant,
        ArrayRole.RAGGED_EXCEPTION_TABLE: _exception_table,
        ArrayRole.TOP_HIT_INDEX: _top_hit_index,
        ArrayRole.TOP_HIT_ANALYSIS_OFFSETS: _top_hit_analysis_offsets,
        ArrayRole.EXCEPTION_TABLE: _exception_table,
        ArrayRole.SE_COEFFICIENTS: _se_coefficients,
        ArrayRole.RHO_ARRAY: _rho_array,
    }
)


def component_chunk_size(component_chunk: int | None, length: int) -> int:
    """The per-variant chunk for a component plane's variant-axis chunk.

    The explicit-component-chunk helper the `PER_VARIANT` policy and the
    converter (#245) use.  A side array must be no coarser than the plane it
    serves (spec §6), so a component chunk smaller than `PER_VARIANT_CHUNK`
    bounds the result; a component with no plane (`component_chunk=None`, e.g. a
    tiny synthetic group) falls back to `PER_VARIANT_CHUNK`.  The result is
    clipped to the array's own length.
    """
    cap = (
        PER_VARIANT_CHUNK
        if component_chunk is None
        else min(int(component_chunk), PER_VARIANT_CHUNK)
    )
    return min(cap, max(length, 1))


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


def chunk_layout(
    role: ArrayRole,
    shape: tuple[int, ...],
    *,
    hint: Any = None,
    component_chunk: int | None = None,
) -> tuple[int, ...]:
    """The chunks `role` requires for an array of `shape`.

    With no `hint` this is the role's **default** physical layout, derivable
    from `role + shape`, plus -- for `PER_VARIANT` only -- the component
    plane's `component_chunk`.  Every other role ignores `component_chunk`: the
    one role whose layout depends on a sibling is `PER_VARIANT` (spec §6), and
    making that dependency an argument is what lets the store converter (#245)
    reproduce the layout without the builder's group.  `hint` is the caller's
    explicit override (the Dense `chunk_shape`, the top-hit `chunk_size`, a
    `chunks=(...)` passed to a CSR writer); each policy decides whether it clips
    the override or takes it whole.
    """
    if role not in _LAYOUTS:
        raise ValueError(f"no chunk layout is registered for role {role!r}")
    return _LAYOUTS[role](
        _LayoutContext(shape=shape, hint=hint, component_chunk=component_chunk)
    )


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

#: The default shard for a Dense statistic grid, in (variant rows, Analyses)
#: units.  Decided by #246 and recorded in ADR 0058; the converter still takes
#: it as a parameter and this is the value its default and the builders both
#: use.  The Analysis axis must be bounded: the Dense VCF builder writes
#: `[all variants x band]` column bands, so a shard spanning every Analysis
#: would never be written whole (#240, ADR 0057).  `100000 x 1024` int16 is
#: 205 MB uncompressed, the largest unit a conversion worker holds.
DENSE_SHARD_SHAPE = (100_000, 1_024)

#: One shard of a per-variant or flat index array, in *elements*: rounded down
#: to a whole number of inner chunks, then clipped to the array.  These arrays
#: are 1-D, so a shard of about a million elements is about 4 MB per dtype.
SHARD_ELEMENT_CAP = 1_000_000

#: One shard of a Ragged association sequence (`ragged/z`, `se`,
#: `variant_index`, `eaf`, `imputed`), in *elements* (#248).  50,000,000 is 250
#: inner chunks of the sequence policy (200,000), so on OGS-00011's overflow
#: sequences (3,085,080,783 entries) it is 62 shard files per array instead of
#: 3,086 at the old one-million cap.  The widest sequence dtype is 4 bytes
#: (`variant_index`, a residual `eaf`), so one shard is at most about 200 MB
#: uncompressed -- the same per-worker ceiling the Dense plane shard
#: (`100000 x 1024` int16, 205 MB) sets.  Every shard is a whole multiple of
#: the inner chunk, so a reader still decodes only the chunks it selects.
RAGGED_SEQUENCE_SHARD_ELEMENTS = 50_000_000

#: One shard of a Ragged 1-D side array, in *elements* (#248): a
#: `RAGGED_PER_VARIANT` frequency or a `RAGGED_EXCEPTION_TABLE`.  These are
#: read whole or at a single position rather than streamed by Analysis, so the
#: shard exists only to bound the file and the converter worker's block: ten
#: million elements is 80 MB for an `int64` exception index and 40 MB for a
#: `float32` value, and 19 files for OGS-00011's 180,396,687-entry Ragged
#: `eaf_exception_index` -- tens of MB each rather than one 1.4 GB file the
#: whole-array policy would have produced.
RAGGED_SIDE_SHARD_ELEMENTS = 10_000_000

#: A top-hit tier's flat column keeps today's 16,384 inner chunk and shards 64
#: of them (about a million elements), the seed of a bounded point read.
TOP_HIT_SHARD_CHUNKS = 64

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


@dataclass(frozen=True)
class _ShardContext:
    """Everything a shard policy may depend on: role, shape, inner chunk, params.

    `top_hit_shard_chunks` is the converter's `--top-hit-shard-chunks` (#246):
    how many top-hit inner chunks one shard holds.  It applies to
    `TOP_HIT_INDEX` alone.  `None` means the default, `TOP_HIT_SHARD_CHUNKS`;
    `1` makes the shard one inner chunk, the "effectively unsharded" variant
    #246 measures the top-hit query against.  The array is still a v3 sharded
    array either way -- it is never written without the sharding codec.
    """

    shape: tuple[int, ...]
    inner_chunk: tuple[int, ...]
    dense_shard: tuple[int, int] | None
    top_hit_shard_chunks: int | None = None


def _covering_shard(inner: int, dim: int) -> int:
    """The smallest whole number of inner chunks that covers `dim`."""
    if inner <= 0:
        raise ValueError(f"inner chunk {inner} is not a positive length")
    return max(-(-max(dim, 1) // inner), 1) * inner


def _clip_shard_to_multiple(requested: int, inner: int, dim: int) -> int:
    """`requested` rounded to a whole number of inner chunks that covers `dim`.

    A shard must be a whole multiple of the inner chunk (zarr refuses one that
    is not), and clipping it to the array means the largest whole number of
    inner chunks no larger than the request *or* than the array needs -- so a
    small array gets one shard of one inner chunk rather than a shard wider
    than the array.
    """
    covering = _covering_shard(inner, dim)
    chunks = min(max(requested // inner, 1), covering // inner)
    return chunks * inner


def _divisors(value: int) -> list[int]:
    """Every divisor of `value`, ascending."""
    return [d for d in range(1, value + 1) if value % d == 0]


def _allowed_inner_chunks(wanted: int) -> str:
    """The inner chunks that tile a decided shard axis, for an error message."""
    divisors = _divisors(wanted)
    if len(divisors) <= 12:
        return ", ".join(str(d) for d in divisors)
    return f"any divisor of {wanted}"


def _require_dense_shard_multiple(
    request: tuple[int, int], inner_chunk: tuple[int, ...], shape: tuple[int, ...]
) -> None:
    """Refuse a Dense inner chunk that does not tile the decided shard.

    A build does not choose its shard: the Dense planes are stored as the decided
    `DENSE_SHARD_SHAPE`, clipped only when the array itself is smaller than the
    shard (ADR 0058).  An inner chunk that does not tile the shard axis would
    silently produce a **different** shard — e.g. `[100000, 1024]` rounded down
    to `[100000, 1000]` for a 1,000-Analysis inner chunk — which is the layout
    divergence #249 exists to catch but a reader cannot see.  Such an inner chunk
    is refused here, naming the values that tile the axis.

    An axis whose array is shorter than the requested shard is the one allowed
    exception: the shard is the array's own extent (one whole-array shard), so
    the inner chunk's divisibility is moot and the shard clips to cover the
    array.  This applies to the *default* shard and to an explicit one alike.
    """
    for axis, (wanted, inner, dim) in enumerate(zip(request, inner_chunk, shape, strict=True)):
        wanted = int(wanted)
        if dim < wanted:
            continue
        if wanted % inner:
            axis_name = "variant" if axis == 0 else "Analysis"
            raise ValueError(
                f"the Dense {axis_name}-axis inner chunk {inner} does not tile the "
                f"decided shard axis {wanted} (shard {tuple(int(size) for size in request)!r}); "
                "a build writes the format's shard, not one it chose. Allowed "
                f"{axis_name}-axis inner chunks: {_allowed_inner_chunks(wanted)}"
            )


def _shard_dense_grid(ctx: _ShardContext) -> tuple[int, ...]:
    """A Dense plane's `(V_s, A_s)` shard, from the conversion parameters."""
    request = DENSE_SHARD_SHAPE if ctx.dense_shard is None else ctx.dense_shard
    if len(request) != len(ctx.inner_chunk):
        raise ValueError(
            f"dense shard {tuple(request)!r} does not match the {len(ctx.inner_chunk)}-D "
            f"inner chunk {ctx.inner_chunk!r}"
        )
    _require_dense_shard_multiple(
        (int(request[0]), int(request[1])), ctx.inner_chunk, ctx.shape
    )
    return tuple(
        _clip_shard_to_multiple(int(wanted), inner, dim)
        for wanted, inner, dim in zip(request, ctx.inner_chunk, ctx.shape, strict=True)
    )


def _shard_element_cap(ctx: _ShardContext) -> tuple[int, ...]:
    """One shard of a 1-D array is about `SHARD_ELEMENT_CAP` elements."""
    inner = ctx.inner_chunk[0]
    return (_clip_shard_to_multiple(SHARD_ELEMENT_CAP, inner, ctx.shape[0]),)


def _shard_ragged_sequence(ctx: _ShardContext) -> tuple[int, ...]:
    """One shard of a Ragged association sequence (#248).

    A fixed element count, a whole number of inner chunks: the sequence is
    Analysis-sorted but its cells are not Analysis-aligned, so a shard bounded
    by cells (rather than by one Analysis's run) is the shape a future
    per-variant index beside it can be added to without re-sharding.  See
    `RAGGED_SEQUENCE_SHARD_ELEMENTS`.
    """
    inner = ctx.inner_chunk[0]
    return (_clip_shard_to_multiple(RAGGED_SEQUENCE_SHARD_ELEMENTS, inner, ctx.shape[0]),)


def _shard_ragged_side(ctx: _ShardContext) -> tuple[int, ...]:
    """One shard of a Ragged 1-D side array (#248), bounded by elements."""
    inner = ctx.inner_chunk[0]
    return (_clip_shard_to_multiple(RAGGED_SIDE_SHARD_ELEMENTS, inner, ctx.shape[0]),)


def _shard_top_hit_index(ctx: _ShardContext) -> tuple[int, ...]:
    """A top-hit tier's flat column: its `top_hit_shard_chunks` inner chunks.

    The default is the seam's `TOP_HIT_SHARD_CHUNKS`.  The converter passes an
    override for #246's "sharded against effectively unsharded" measurement;
    the override is validated here rather than left to zarr, so a bad value
    fails with the role in the message.
    """
    count = (
        TOP_HIT_SHARD_CHUNKS
        if ctx.top_hit_shard_chunks is None
        else int(ctx.top_hit_shard_chunks)
    )
    if count < 1:
        raise ValueError(
            f"top-hit shard must hold at least one inner chunk, got {count}"
        )
    return tuple(
        _clip_shard_to_multiple(count * inner, inner, dim)
        for inner, dim in zip(ctx.inner_chunk, ctx.shape, strict=True)
    )


def _shard_whole_array(ctx: _ShardContext) -> tuple[int, ...]:
    """One shard holds the whole array, so a small side array is one file.

    Used for the exact-value exception/overflow tables, the SE coefficient
    table, the top-hit per-Analysis offsets and the CSR offset array: each is
    read whole or at a single Analysis, so a shard split only adds files.
    """
    return tuple(
        _covering_shard(inner, dim)
        for inner, dim in zip(ctx.inner_chunk, ctx.shape, strict=True)
    )


#: The role -> shard policy, the companion of `_LAYOUTS`.  #247 reads this same
#: table; the converter must not keep a private copy (issue #245).
_SHARD_LAYOUTS: Mapping[ArrayRole, Callable[[_ShardContext], tuple[int, ...]]] = (
    MappingProxyType(
        {
            ArrayRole.DENSE_STATISTIC_PLANE: _shard_dense_grid,
            ArrayRole.DENSE_IMPUTED_MASK: _shard_dense_grid,
            ArrayRole.DENSE_ON_PANEL: _shard_element_cap,
            ArrayRole.ASSOCIATION_SEQUENCE: _shard_ragged_sequence,
            ArrayRole.ASSOCIATION_OFFSETS: _shard_whole_array,
            ArrayRole.PER_VARIANT: _shard_element_cap,
            ArrayRole.RAGGED_PER_VARIANT: _shard_ragged_side,
            ArrayRole.RAGGED_EXCEPTION_TABLE: _shard_ragged_side,
            ArrayRole.TOP_HIT_INDEX: _shard_top_hit_index,
            ArrayRole.TOP_HIT_ANALYSIS_OFFSETS: _shard_whole_array,
            ArrayRole.EXCEPTION_TABLE: _shard_whole_array,
            ArrayRole.SE_COEFFICIENTS: _shard_whole_array,
            ArrayRole.RHO_ARRAY: _shard_element_cap,
        }
    )
)


def _require_shard_multiple(
    role: ArrayRole, inner: tuple[int, ...], shard: tuple[int, ...]
) -> None:
    """Fail loudly unless `shard` is a whole multiple of `inner` on every axis."""
    if len(shard) != len(inner):
        raise ValueError(
            f"role {role!r}: shard {shard!r} has {len(shard)} dimensions but the inner "
            f"chunk {inner!r} has {len(inner)}"
        )
    for axis, (outer, inner_axis) in enumerate(zip(shard, inner, strict=True)):
        if inner_axis <= 0 or outer <= 0 or outer % inner_axis:
            raise ValueError(
                f"role {role!r}: shard {shard!r} is not a whole multiple of the inner "
                f"chunk {inner!r} (axis {axis}: {outer} is not a multiple of {inner_axis})"
            )


def shard_layout(
    role: ArrayRole,
    shape: tuple[int, ...],
    *,
    inner_chunk: tuple[int, ...] | None = None,
    dense_shard: tuple[int, int] | None = None,
    component_chunk: int | None = None,
    top_hit_shard_chunks: int | None = None,
) -> tuple[int, ...]:
    """The shard shape `role` requires for an array of `shape`.

    `inner_chunk` is the array's inner chunk, normally from `chunk_layout`; when
    omitted it is derived from `role + shape` (with `component_chunk` for
    `PER_VARIANT`).  `dense_shard` is the `(V_s, A_s)` conversion parameter and
    applies only to the Dense grid roles; every other role's shard is fixed by
    its policy.  `top_hit_shard_chunks` is the converter's override for
    `TOP_HIT_INDEX` (#246); it is ignored by every other role.

    The result is always a whole multiple of `inner_chunk` -- the presence of a
    shard that is not is a corrupt layout, so this refuses rather than passing
    it to zarr to fail with a message that names only two sizes.
    """
    if role not in _SHARD_LAYOUTS:
        raise ValueError(f"no shard layout is registered for role {role!r}")
    resolved = (
        chunk_layout(role, shape, component_chunk=component_chunk)
        if inner_chunk is None
        else tuple(int(size) for size in inner_chunk)
    )
    if len(resolved) != len(shape):
        raise ValueError(
            f"role {role!r}: inner chunk {resolved!r} does not match shape {tuple(shape)!r}"
        )
    shard = _SHARD_LAYOUTS[role](
        _ShardContext(
            shape=tuple(int(size) for size in shape),
            inner_chunk=resolved,
            dense_shard=(
                None if dense_shard is None else (int(dense_shard[0]), int(dense_shard[1]))
            ),
            top_hit_shard_chunks=top_hit_shard_chunks,
        )
    )
    _require_shard_multiple(role, resolved, shard)
    return shard


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
    """Fail loudly unless `selection` covers whole shards of a Dense grid.

    Applied to the two-dimensional sharded arrays -- the Dense statistic planes,
    the imputed mask and the SE coefficient table -- where a partial write is a
    read-modify-write of a shard that can be hundreds of megabytes.  The 1-D
    arrays whose shard policy is "one shard holds the whole array" (the
    exception tables) and the element-capped Ragged/Overflow arrays are written
    incrementally by design and are not judged here.
    """
    shards = getattr(array, "shards", None)
    if shards is None or len(array.shape) != 2:
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


def _block_selection_to_elements(array: Any, selection: Any) -> tuple[Any, ...] | None:
    """A `set_block_selection` selection, as the element selection it covers.

    `array.blocks` indexes the **chunk** grid, so a block index maps to
    `[i * chunk : (i + 1) * chunk]`.  Translating it lets the whole-shard rule
    apply to `array.blocks[...]` exactly as it does to an element selection.
    Returns `None` for a block selection with a step, which is not a contiguous
    range and is refused by the caller rather than guessed at.
    """
    shape = tuple(int(size) for size in array.shape)
    chunks = tuple(int(size) for size in array.chunks)
    axes: list[Any] = []
    for entry, chunk, dim in zip(
        _normalise_selection(shape, selection), chunks, shape, strict=True
    ):
        n_blocks = max(1, -(-dim // chunk))
        if isinstance(entry, slice):
            start, stop, step = entry.indices(n_blocks)
            if step != 1:
                return None
            axes.append(slice(start * chunk, min(stop * chunk, dim)))
        else:
            index = int(entry)
            if index < 0:
                index += n_blocks
            axes.append(slice(index * chunk, min((index + 1) * chunk, dim)))
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
    elements = _block_selection_to_elements(self, selection)
    if elements is None:
        raise PartialShardWriteError(
            f"array {getattr(self, 'path', '?')!r}: a `blocks` write selecting "
            f"{selection!r} steps through the block grid and cannot cover whole "
            "shards; write whole shards (issue #247)"
        )
    require_whole_shard_write(self, elements)
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
        "se_coefficients": ArrayRole.SE_COEFFICIENTS,
        "se_exception_index": ArrayRole.EXCEPTION_TABLE,
        "se_exception_value": ArrayRole.EXCEPTION_TABLE,
        "z_overflow_index": ArrayRole.EXCEPTION_TABLE,
        "z_overflow_value": ArrayRole.EXCEPTION_TABLE,
        "eaf_exception_index": ArrayRole.EXCEPTION_TABLE,
        "eaf_exception_value": ArrayRole.EXCEPTION_TABLE,
    }
)


#: The Ragged CSR group's arrays, by their leaf name under `ragged/` (#248).
#: `z`, `se`, `variant_index`, `eaf` and `imputed` are the CSR's parallel
#: sequences; `offsets` indexes them by Analysis; `eaf_baseline` and
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
        "offsets": ArrayRole.ASSOCIATION_OFFSETS,
        "eaf_baseline": ArrayRole.RAGGED_PER_VARIANT,
        "eaf_reference": ArrayRole.RAGGED_PER_VARIANT,
        "se_coefficients": ArrayRole.SE_COEFFICIENTS,
        "se_exception_index": ArrayRole.RAGGED_EXCEPTION_TABLE,
        "se_exception_value": ArrayRole.RAGGED_EXCEPTION_TABLE,
        "z_overflow_index": ArrayRole.RAGGED_EXCEPTION_TABLE,
        "z_overflow_value": ArrayRole.RAGGED_EXCEPTION_TABLE,
        "eaf_exception_index": ArrayRole.RAGGED_EXCEPTION_TABLE,
        "eaf_exception_value": ArrayRole.RAGGED_EXCEPTION_TABLE,
    }
)


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
    `is_recorded_group_path` for the group itself.
    """
    if head == "top_hits":
        tier, separator, leaf = rest.partition("/")
        if not tier or not separator or not leaf or "/" in leaf:
            return None
        return _TOP_HIT_ROLES_BY_LEAF.get(leaf)
    if head == "rho":
        return _RHO_ROLES_BY_LEAF.get(rest)
    if head == "ragged":
        return _RAGGED_ROLES_BY_NAME.get(rest)
    return None


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
        _require_shard_multiple(role, chunks, resolved_shards)
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
        await super().set(key, value)

    async def set_if_not_exists(self, key: str, value: Buffer) -> None:
        self._check_write(key)
        await super().set_if_not_exists(key, value)

    def set_sync(self, key: str, value: Buffer) -> None:
        self._check_write(key)
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

    The one place `zarr.open_group` is called, so the Store format is declared
    once.  `mode` may be a creating mode here; `open_group_for_write` is the
    named entry point for those, and this general form exists for `r`/`r+` and
    for the release-envelope ``arrays(mode=...)`` methods that forward their
    mode.  A creating mode is pinned to `STORE_ZARR_FORMAT` unless `zarr_format`
    overrides it (the converter's v3 output, #245); a read mode leaves the
    format to the stored metadata.  Any mode but ``r`` refuses a group that
    consolidated metadata describes (`refuse_under_consolidated_metadata`).
    """
    zarr_mode = _zarr_mode(mode)
    if zarr_mode != "r":
        refuse_under_consolidated_metadata(
            Path(path), f"opening in mode {mode!r}", wiped=zarr_mode == "w"
        )
    store = _open_local_store(path, zarr_mode)
    return zarr.open_group(
        store, mode=zarr_mode, zarr_format=_open_group_format(mode, zarr_format)
    )


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
