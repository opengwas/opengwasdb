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
  names its array's `ArrayRole`; the role, plus (for the layouts that take one)
  a build-wide hint such as the dense `chunk_shape`, fixes the chunks.  No call
  site computes a chunk tuple itself.
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
from types import MappingProxyType
from typing import Any

import numpy as np
from numcodecs import Blosc

__all__ = [
    "ASSOCIATION_OFFSETS_CHUNK",
    "ASSOCIATION_SEQUENCE_CHUNK",
    "COMPRESSOR_RECORD",
    "PER_VARIANT_CHUNK",
    "RHO_CHUNK_ROWS",
    "SE_COEFFICIENTS_ROWS",
    "ArrayRole",
    "chunk_layout",
    "compressor",
    "create_array",
    "create_group",
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

#: The chunk hint a Dense grid uses when a caller has none: `DEFAULT_CHUNK_SHAPE`
#: from the Dense constants module.  Duplicated as a plain default here so the
#: seam imports nothing from the encoding/layout packages.
DEFAULT_DENSE_CHUNK_HINT = (1000, 1000)


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
    """Everything a layout policy is allowed to depend on."""

    shape: tuple[int, ...]
    hint: Any
    group: Any


def _dense_grid(ctx: _LayoutContext) -> tuple[int, ...]:
    """A 2-D Dense grid, clipped to its own dimensions (ADR 0021)."""
    hint = ctx.hint if ctx.hint is not None else DEFAULT_DENSE_CHUNK_HINT
    return (min(int(hint[0]), ctx.shape[0]), min(int(hint[1]), ctx.shape[1]))


def _dense_on_panel(ctx: _LayoutContext) -> tuple[int, ...]:
    """The per-variant Dense mask, on the grid's row chunk."""
    hint = ctx.hint if ctx.hint is not None else DEFAULT_DENSE_CHUNK_HINT
    return (min(int(hint[0]), ctx.shape[0]),)


def _association_sequence(ctx: _LayoutContext) -> tuple[int, ...]:
    """A flat CSR sequence: one fixed chunk, never clipped."""
    return (ASSOCIATION_SEQUENCE_CHUNK,)


def _association_offsets(ctx: _LayoutContext) -> tuple[int, ...]:
    """The CSR per-Analysis offset array."""
    return (ASSOCIATION_OFFSETS_CHUNK,)


def _per_variant(ctx: _LayoutContext) -> tuple[int, ...]:
    """A per-variant side array, following the plane it serves.

    An explicit hint (the `chunk` override the writers accept) wins over the
    sibling-derived size, clipped to the array length as before.
    """
    length = ctx.shape[0]
    if ctx.hint is not None:
        return (max(1, min(int(ctx.hint), max(length, 1))),)
    return (per_variant_chunk_size(ctx.group, length),)


def _length_clipped(ctx: _LayoutContext) -> tuple[int, ...]:
    """A flat array whose chunk is a hint clipped to its own length."""
    return (max(1, min(ctx.shape[0], int(ctx.hint))),)


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
#: converter (#245) reads the same table so builders and converter agree.
_LAYOUTS: Mapping[ArrayRole, Callable[[_LayoutContext], tuple[int, ...]]] = MappingProxyType(
    {
        ArrayRole.DENSE_STATISTIC_PLANE: _dense_grid,
        ArrayRole.DENSE_IMPUTED_MASK: _dense_grid,
        ArrayRole.DENSE_ON_PANEL: _dense_on_panel,
        ArrayRole.ASSOCIATION_SEQUENCE: _association_sequence,
        ArrayRole.ASSOCIATION_OFFSETS: _association_offsets,
        ArrayRole.PER_VARIANT: _per_variant,
        ArrayRole.TOP_HIT_INDEX: _length_clipped,
        ArrayRole.TOP_HIT_ANALYSIS_OFFSETS: _top_hit_analysis_offsets,
        ArrayRole.EXCEPTION_TABLE: _length_clipped,
        ArrayRole.SE_COEFFICIENTS: _se_coefficients,
        ArrayRole.RHO_ARRAY: _rho_array,
    }
)


def per_variant_chunk_size(group: Any, length: int) -> int:
    """Return the component-local chunk size for a per-variant side array.

    The side array must be read in the same tiles as the plane it belongs to,
    so it follows the first variant-axis sibling the group has; a group with no
    suitable sibling (a tiny synthetic one, or `None`) falls back to
    `PER_VARIANT_CHUNK`.
    """
    if group is not None:
        for sibling in ("eaf", "z", "imputed", "variant_index"):
            if sibling in group and group[sibling].ndim:
                return min(int(group[sibling].chunks[0]), PER_VARIANT_CHUNK, max(length, 1))
    return min(PER_VARIANT_CHUNK, max(length, 1))


def chunk_layout(
    role: ArrayRole,
    shape: tuple[int, ...],
    *,
    hint: Any = None,
    group: Any = None,
) -> tuple[int, ...]:
    """The chunks `role` requires for an array of `shape`.

    `create_array` is the only writer, but this is public because the store
    converter (#245) has to reproduce the same layout for arrays it did not
    create, and it must read the mapping from the same table.
    """
    if role not in _LAYOUTS:
        raise ValueError(f"no chunk layout is registered for role {role!r}")
    return _LAYOUTS[role](_LayoutContext(shape=shape, hint=hint, group=group))


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
    filters: Any,
    order: str,
    group: Any,
) -> dict[str, Any]:
    """The `create_dataset` keyword arguments one role's array is made with."""
    kwargs: dict[str, Any] = {
        "chunks": chunk_layout(role, shape, hint=hint, group=group),
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
    compressor.  `hint` is the build-wide chunk hint a length/clipped policy
    consumes (the Dense `chunk_shape`, the top-hit `chunk_size`, the
    exception-table chunk); policies that need none ignore it.

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
            filters=filters,
            order=order,
            group=group,
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
