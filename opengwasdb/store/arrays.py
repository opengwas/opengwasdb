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
    "EXCEPTION_TABLE_CHUNK",
    "PER_VARIANT_CHUNK",
    "RHO_CHUNK_ROWS",
    "SE_COEFFICIENTS_ROWS",
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
    "move_in_group",
    "open_group",
    "open_group_for_write",
    "per_variant_chunk_size",
    "require_group",
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
            return int(group[sibling].chunks[0])
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
    role: ArrayRole,
    shape: tuple[int, ...],
    *,
    data: Any,
    dtype: Any,
    fill_value: Any,
    compressor: Any,
    hint: Any,
    component_chunk: int | None,
    filters: Any,
    order: str,
) -> dict[str, Any]:
    """The `create_array` keyword arguments one role's array is made with.

    ``compressors`` (plural, a v3 spelling) is what zarr-python 3's
    ``create_array`` takes; on a Zarr v2-format group it accepts a single
    numcodecs codec and writes the same ``.zarray`` ``create_dataset`` wrote
    in zarr 2.18.  ``create_dataset`` itself no longer exists in zarr 3.4, so
    this is the only compatible call.

    Two behaviours that zarr 2.18 had implicitly are now explicit:

    * ``write_empty_chunks``: zarr 2's ``create`` defaulted it to True, so a
      chunk that is entirely the fill value was still written as a file.  zarr
      3 defaults it to False, which would silently drop those chunk files and
      change a built store's file set.  The module sets the process-wide
      default back to True (see the comment at the import); the flag is *not*
      stored in array metadata, so a per-array value would be lost the moment
      a caller reopened the array.
    * ``dtype`` is inferred from `data` here, because zarr 3's ``create_array``
      refuses ``data`` and ``dtype`` together.  zarr 2's whole-array write went
      through ``zarr.array(data, dtype=...)``, i.e. create-then-assign; the
      caller's `data` is assigned after creation in `create_array` so a dtype
      the data does not already carry still casts, exactly as it did.
    """
    kwargs: dict[str, Any] = {
        "chunks": chunk_layout(role, shape, hint=hint, component_chunk=component_chunk),
        "compressors": _new_compressor() if compressor is _SEAM_COMPRESSOR else compressor,
        "order": order,
        "shape": shape,
    }
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
    filters: Any = None,
    order: str = "C",
    overwrite: bool = False,
) -> Any:
    """Create one Store array under the seam's compressor and `role` layout.

    Exactly one of `data` (a whole-array write) and `shape` (an array to be
    filled later, in bands or regions) must describe the array; `data`'s shape
    supplies `shape` when only `data` is given.  `dtype` is passed straight
    through, or inferred from `data` when omitted.

    `fill_value` is the plane's own missing marker (spec §15).  Omit it for the
    dtype default; pass `fill_value=None` only when the metadata must literally
    say `null`.  `compressor=None` stores uncompressed; omit it for the seam's
    compressor.

    `hint` overrides the role's default layout (the Dense `chunk_shape`, the
    top-hit `chunk_size`, a `chunks=(...)` passed to a CSR writer).
    `component_chunk` is the variant-axis chunk of the plane a `PER_VARIANT`
    array serves; the writer passes the chunk it read from the component, so
    the layout never depends on sniffing `group` and the converter can supply
    the same fact directly.

    `overwrite` deletes an existing array of the same name first, which the
    call sites that previously did `if name in group: del group[name]` relied
    on.  A site that expects a fresh name leaves it False, so writing twice
    still fails loudly.
    """
    shape = _resolve_shape(name, data, shape)
    if overwrite and name in group:
        del group[name]
    return _create_and_fill(
        group,
        name,
        _creation_kwargs(
            role,
            shape,
            data=data,
            dtype=dtype,
            fill_value=fill_value,
            compressor=compressor,
            hint=hint,
            component_chunk=component_chunk,
            filters=filters,
            order=order,
        ),
        data,
    )


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


def _open_group_format(mode: str) -> Literal[2] | None:
    """The ``zarr_format`` an open in `mode` must use.

    A creating mode must declare v2; a read mode must not declare anything, so
    the existing metadata decides (which is how zarr 3 reads a v2 store and how
    #245's v3 store will read).
    """
    return STORE_ZARR_FORMAT if mode in CREATION_MODES else None


def open_group(path: str | Path, mode: str = "r") -> Any:
    """Open a Zarr group, read-only by default.

    The one place `zarr.open_group` is called, so the Store format is declared
    once.  `mode` may be a creating mode here; `open_group_for_write` is the
    named entry point for those, and this general form exists for `r`/`r+` and
    for the release-envelope ``arrays(mode=...)`` methods that forward their
    mode.  A creating mode is pinned to `STORE_ZARR_FORMAT`; a read mode leaves
    the format to the stored metadata.  Any mode but ``r`` refuses a group that
    consolidated metadata describes (`refuse_under_consolidated_metadata`).
    """
    zarr_mode = _zarr_mode(mode)
    if zarr_mode != "r":
        refuse_under_consolidated_metadata(
            Path(path), f"opening in mode {mode!r}", wiped=zarr_mode == "w"
        )
    return zarr.open_group(str(path), mode=zarr_mode, zarr_format=_open_group_format(mode))


def open_group_for_write(path: str | Path, mode: str) -> Any:
    """Open a Zarr group for writing, creating it when it is absent.

    `mode` must be one of ``w``/``a``/``w-`` and is **required**: ``a``
    and ``w`` differ on a resumed build (``w`` wipes the staged group, ``a``
    keeps it), so a silent default here would change what a resume finds.  A
    read mode is a bug (the caller meant `open_group`) and fails loudly.

    The group is created in `STORE_ZARR_FORMAT` (Zarr v2 until #247).  Without
    the explicit format zarr-python 3 would create a Zarr v3 group and silently
    change the Store format; that is why every write-mode open routes here.
    """
    if mode not in CREATION_MODES:
        raise ValueError(
            f"open_group_for_write needs one of {sorted(CREATION_MODES)}, got {mode!r}; "
            "use open_group for a read mode"
        )
    refuse_under_consolidated_metadata(Path(path), f"opening in mode {mode!r}", wiped=mode == "w")
    return zarr.open_group(str(path), mode=_zarr_mode(mode), zarr_format=STORE_ZARR_FORMAT)
