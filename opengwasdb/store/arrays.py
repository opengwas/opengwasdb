"""The one seam through which every Store Release array and group is created.

Every Zarr array this package writes is created here, via `create_array`, and
every Zarr group via `create_group`/`require_group`.  That is not tidiness for
its own sake: array creation is where a Store Release's physical layout is
decided, and it has to be decided in exactly one place so the pieces that will
change it -- the switch to Zarr v3 sharding (#247) and the converter that has
to reproduce the same layout (#245) -- cannot disagree about how an array is
laid out.

The seam owns four things:

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

The role -> layout policy is one table in this module (`_LAYOUTS`) and nothing
else chooses chunks.  A role that does not fit an existing entry gets a new
one rather than a call site that passes `chunks=` by hand.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import zarr
from numcodecs import Blosc

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
    "chunk_layout",
    "component_chunk_size",
    "component_variant_chunk",
    "compressor",
    "create_array",
    "create_group",
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
    """The `create_dataset` keyword arguments one role's array is made with."""
    kwargs: dict[str, Any] = {
        "chunks": chunk_layout(role, shape, hint=hint, component_chunk=component_chunk),
        "compressor": _new_compressor() if compressor is _SEAM_COMPRESSOR else compressor,
        "order": order,
    }
    if dtype is not None:
        kwargs["dtype"] = dtype
    if data is not None:
        kwargs["data"] = data
    else:
        kwargs["shape"] = shape
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
    return group.create_dataset(
        name,
        **_creation_kwargs(
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
    )


def create_group(group: Any, name: str, *, replace: bool = True) -> Any:
    """Create one Store group, replacing any existing one by default."""
    if replace and name in group:
        del group[name]
    return group.create_group(name)


def require_group(group: Any, name: str) -> Any:
    """Return the named group, creating it only when it is absent."""
    return group.require_group(name)


#: `zarr.open_group` modes that create the group (or wipe it) when it is not
#: there.  `r` and `r+` require it to exist and are not creation.
CREATION_MODES = frozenset({"w", "a", "w-", "x"})


def open_group(path: str | Path, mode: str = "r") -> Any:
    """Open a Zarr group, read-only by default.

    The one place `zarr.open_group` is called, so the seam can later pass
    ``zarr_format`` (zarr-python 3, #244) without hunting down every opener.
    `mode` may be a creating mode here; `open_group_for_write` is the named
    entry point for those, and this general form exists for `r`/`r+` and for
    the release-envelope ``arrays(mode=...)`` methods that forward their mode.
    """
    return zarr.open_group(str(path), mode=mode)


def open_group_for_write(path: str | Path, mode: str) -> Any:
    """Open a Zarr group for writing, creating it when it is absent.

    `mode` must be one of ``w``/``a``/``w-``/``x`` and is **required**: ``a``
    and ``w`` differ on a resumed build (``w`` wipes the staged group, ``a``
    keeps it), so a silent default here would change what a resume finds.  A
    read mode is a bug (the caller meant `open_group`) and fails loudly.  No
    ``zarr_format`` argument yet: zarr 2.18 has none, and #244 adds the Zarr v3
    argument here once, for every writer.
    """
    if mode not in CREATION_MODES:
        raise ValueError(
            f"open_group_for_write needs one of {sorted(CREATION_MODES)}, got {mode!r}; "
            "use open_group for a read mode"
        )
    return zarr.open_group(str(path), mode=mode)
