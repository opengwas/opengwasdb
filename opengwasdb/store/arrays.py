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
from collections.abc import Callable, Mapping
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
    "RHO_CHUNK_ROWS",
    "SE_COEFFICIENTS_ROWS",
    "SHARDED_COMPRESSOR_RECORD",
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
    "move_in_group",
    "open_group",
    "open_group_for_write",
    "per_variant_chunk_size",
    "require_group",
    "role_for_array_path",
    "shard_layout",
    "sharded_compressor",
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

#: The maximum chunk hint a Dense grid uses when a caller has none.  This is
#: `DEFAULT_CHUNK_SHAPE` in the Dense constants module, and the one definition:
#: the Dense module imports *this* name, so the converter and the Dense builders
#: cannot disagree about the default grid layout.
DENSE_CHUNK_SHAPE = (1000, 1000)

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
#: units.  Both are *parameters* of the conversion (#246 benchmarks and decides
#: them); this is the proposed default.  The Analysis axis must be bounded: the
#: Dense VCF builder writes `[all variants x band]` column bands, so a shard
#: spanning every Analysis would never be written whole (#240, ADR 0057).
#: `100000 x 1024` int16 is 205 MB uncompressed, the largest unit a conversion
#: worker holds.
DENSE_SHARD_SHAPE = (100_000, 1_024)

#: One shard of a per-variant or flat index array, in *elements*: rounded down
#: to a whole number of inner chunks, then clipped to the array.  These arrays
#: are 1-D, so a shard of about a million elements is about 4 MB per dtype.
SHARD_ELEMENT_CAP = 1_000_000

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
    """Everything a shard policy may depend on: role, shape, inner chunk, params."""

    shape: tuple[int, ...]
    inner_chunk: tuple[int, ...]
    dense_shard: tuple[int, int] | None


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


def _shard_dense_grid(ctx: _ShardContext) -> tuple[int, ...]:
    """A Dense plane's `(V_s, A_s)` shard, from the conversion parameters."""
    request = DENSE_SHARD_SHAPE if ctx.dense_shard is None else ctx.dense_shard
    if len(request) != len(ctx.inner_chunk):
        raise ValueError(
            f"dense shard {tuple(request)!r} does not match the {len(ctx.inner_chunk)}-D "
            f"inner chunk {ctx.inner_chunk!r}"
        )
    for wanted, inner in zip(request, ctx.inner_chunk, strict=True):
        if int(wanted) % inner:
            raise ValueError(
                f"dense shard {tuple(int(size) for size in request)!r} is not a whole "
                f"multiple of the inner chunk {ctx.inner_chunk!r} ({wanted} is not a "
                f"multiple of {inner})"
            )
    return tuple(
        _clip_shard_to_multiple(int(wanted), inner, dim)
        for wanted, inner, dim in zip(request, ctx.inner_chunk, ctx.shape, strict=True)
    )


def _shard_element_cap(ctx: _ShardContext) -> tuple[int, ...]:
    """One shard of a 1-D array is about `SHARD_ELEMENT_CAP` elements."""
    inner = ctx.inner_chunk[0]
    return (_clip_shard_to_multiple(SHARD_ELEMENT_CAP, inner, ctx.shape[0]),)


def _shard_inner_chunks(count: int) -> Callable[[_ShardContext], tuple[int, ...]]:
    """A shard of `count` inner chunks, axis by axis (used by 1-D tiers)."""

    def policy(ctx: _ShardContext) -> tuple[int, ...]:
        return tuple(
            _clip_shard_to_multiple(count * inner, inner, dim)
            for inner, dim in zip(ctx.inner_chunk, ctx.shape, strict=True)
        )

    return policy


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
            ArrayRole.ASSOCIATION_SEQUENCE: _shard_element_cap,
            ArrayRole.ASSOCIATION_OFFSETS: _shard_whole_array,
            ArrayRole.PER_VARIANT: _shard_element_cap,
            ArrayRole.TOP_HIT_INDEX: _shard_inner_chunks(TOP_HIT_SHARD_CHUNKS),
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
) -> tuple[int, ...]:
    """The shard shape `role` requires for an array of `shape`.

    `inner_chunk` is the array's inner chunk, normally from `chunk_layout`; when
    omitted it is derived from `role + shape` (with `component_chunk` for
    `PER_VARIANT`).  `dense_shard` is the `(V_s, A_s)` conversion parameter and
    applies only to the Dense grid roles; every other role's shard is fixed by
    its policy.

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
        )
    )
    _require_shard_multiple(role, resolved, shard)
    return shard


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


def role_for_array_path(path: str) -> ArrayRole | None:
    """The `ArrayRole` a Dense `data.zarr` array's path maps to, or `None`.

    The one path -> role mapping a 0.2.0 conversion is allowed to use.  A
    converter must never guess a layout: an array this returns `None` for fails
    the conversion loudly rather than being copied with some default shard.
    Top-hit tiers are `top_hits/<tier>/...`: the per-Analysis `analysis_offsets`
    is its own role, every other column is `TOP_HIT_INDEX`.  Rho is `rho/...`.
    """
    name = path.strip("/")
    if not name:
        return None
    head, _, rest = name.partition("/")
    if head == "top_hits" and rest:
        return (
            ArrayRole.TOP_HIT_ANALYSIS_OFFSETS
            if rest.endswith("analysis_offsets")
            else ArrayRole.TOP_HIT_INDEX
        )
    if head == "rho" and rest:
        return ArrayRole.RHO_ARRAY
    return _DENSE_ROLES_BY_NAME.get(name)


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
    inner chunk (the converter's path, #245) and wins over `hint`.  `shards`
    writes a Zarr v3 sharded array: the outer shard shape, with the inner chunk
    inside the `sharding_indexed` codec and a v3 codec (e.g.
    `sharded_compressor()`) as `compressor`.  The shard is checked against the
    inner chunk here, before zarr sees it.  `overwrite` deletes an existing
    array of the same name first; a site that expects a fresh name leaves it
    False, so writing twice fails loudly.
    """
    shape = _resolve_shape(name, data, shape)
    chunks = _resolve_inner_chunk(name, role, shape, hint, component_chunk, inner_chunk)
    resolved_shards = None if shards is None else tuple(int(size) for size in shards)
    if resolved_shards is not None:
        _require_shard_multiple(role, chunks, resolved_shards)
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
            compressor=compressor,
            filters=filters,
            order=order,
            shards=resolved_shards,
        ),
        data,
    )


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

#: The Zarr on-disk format every *created* array and group is written in until
#: #247 moves the builders to Zarr v3 (ADR 0041).  Passing it explicitly on
#: every creation-mode open is the whole point: zarr-python 3's
#: ``open_group(..., mode="w")`` defaults to ``zarr_format=None``, which
#: *creates a Zarr v3 group* -- a silent Store format change.  Read-mode opens
#: pass ``zarr_format=None`` so a converted v3 store (#245) still opens; zarr 3
#: auto-detects the format from the existing metadata.
STORE_ZARR_FORMAT: Final = 2

#: The Zarr on-disk formats zarr-python 3 can be asked to create, as the
#: `Literal` it takes.  A lookup rather than a cast, so an unknown value is a
#: `ValueError` here instead of something zarr refuses obscurely.
_ZARR_FORMATS: Mapping[int, Literal[2, 3]] = MappingProxyType({2: 2, 3: 3})

#: The Zarr format a 0.2.0 Store Release is written in (issue #245).  The
#: converters and #247's shard writers pass it; builders leave the default.
SHARDED_STORE_ZARR_FORMAT: Final = 3


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
