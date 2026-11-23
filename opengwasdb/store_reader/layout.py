"""The physical layout of a Store Release's arrays: each array role's inner chunk and shard.

A layout is part of the format (ADR 0057, ADR 0058): which inner chunk and which
shard an array of a given role and shape has is decided by one table, so the
writer that creates an array, the converter that reproduces one and the reader
and validator that refuse one off the format cannot disagree. The table lives
here, at the bottom of the import graph, because the Store Reader checks it on
open -- an Indexed Variant Subset whose plane chunks span Analyses is refused
before it is read (#263, #464). Creating an array on it is
`opengwasdb.store.arrays`'s, the array-creation seam, which imports it.

These are the format's per-role chunk and shard facts, and all of them: readers
use them to check an array's layout, writers to create one. The writer-side
arithmetic -- a build's hint, the Dense shard override, `require_shard_multiple`
-- is here too, because splitting the table would give one fact two owners
(owner's ruling on #464).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from opengwasdb.store_reader.encoding.codec import SE_COEFFICIENTS as SE_COEFFICIENTS_KEY

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
#: override.  `top_hits.TOP_HIT_CHUNK_SIZE` is this value.
TOP_HIT_CHUNK_SIZE = 16_384

#: One half of an exact-value exception/overflow table, when a caller supplies
#: no override.  `encoding/codec.EXACT_TABLE_CHUNK` is this value.
EXCEPTION_TABLE_CHUNK = 200_000

#: An Indexed Variant Subset's variant-axis inner chunk.  The subset planes are
#: Analysis-major (`n_analyses x n_subset_variants`), so the *Analysis* axis is
#: the row of one variant, and an inner chunk of one row by this many variants
#: makes a single-Analysis read a contiguous run of chunks rather than a
#: genome-wide gather (ADR 0061; prototype `CACHE_ROWS`).
INDEXED_SUBSET_CHUNK = 65_536

#: An Indexed Variant Subset plane's variant-axis shard: 16 inner chunks.  The
#: shard is bounded along the variant axis only, so one shard never spans two
#: Analyses -- reading one Analysis never decompresses another's cells.
INDEXED_SUBSET_SHARD = 16 * INDEXED_SUBSET_CHUNK


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
    SE_COEFFICIENTS = SE_COEFFICIENTS_KEY
    #: One array of the Rho Matrix group (`rho`, `n_null`, `variant_index`).
    RHO_ARRAY = "rho_array"
    #: One statistic plane of an Indexed Variant Subset (`z`, `se`, `eaf`).
    #: Analysis-major (`n_analyses x n_subset_variants`) and chunked for narrow
    #: single-Analysis reads; a distinct role because neither axis's layout is
    #: the Dense grid's (ADR 0061, #264).
    INDEXED_SUBSET_PLANE = "indexed_subset_plane"
    #: An Indexed Variant Subset's sorted Store Variant Indices.
    INDEXED_SUBSET_VARIANT_INDEX = "indexed_subset_variant_index"


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


def _indexed_subset_plane(ctx: _LayoutContext) -> tuple[int, ...]:
    """An Indexed Variant Subset statistic plane, one Analysis per chunk row.

    Analysis-major, so the row is one Analysis and the column is a subset
    variant.  The default inner chunk is `(1, INDEXED_SUBSET_CHUNK)`, clipped
    to the array, which is what makes a single-Analysis query a sequential
    read; a caller's `hint` overrides it (the writer passes the same shape).
    """
    hint = ctx.hint if ctx.hint is not None else (1, INDEXED_SUBSET_CHUNK)
    return (
        max(1, min(int(hint[0]), ctx.shape[0])),
        max(1, min(int(hint[1]), ctx.shape[1])),
    )


def _indexed_subset_variant_index(ctx: _LayoutContext) -> tuple[int, ...]:
    """An Indexed Variant Subset's sorted Variant Index array."""
    return _length_clipped(ctx, INDEXED_SUBSET_CHUNK)

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
        ArrayRole.INDEXED_SUBSET_PLANE: _indexed_subset_plane,
        ArrayRole.INDEXED_SUBSET_VARIANT_INDEX: _indexed_subset_variant_index,
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


def _shard_indexed_subset_plane(ctx: _ShardContext) -> tuple[int, ...]:
    """An Indexed Variant Subset plane's shard: one Analysis row, bounded.

    The Analysis axis shard is one row, so a single-Analysis read never touches
    another Analysis's cells; the variant axis is `INDEXED_SUBSET_SHARD`
    rounded to a whole number of inner chunks and clipped to the array.
    """
    return (
        _clip_shard_to_multiple(1, ctx.inner_chunk[0], ctx.shape[0]),
        _clip_shard_to_multiple(INDEXED_SUBSET_SHARD, ctx.inner_chunk[1], ctx.shape[1]),
    )


def _shard_indexed_subset_variant_index(ctx: _ShardContext) -> tuple[int, ...]:
    """An Indexed Variant Subset's Variant Index array, sharded by element cap."""
    return _shard_element_cap(ctx)


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
            ArrayRole.INDEXED_SUBSET_PLANE: _shard_indexed_subset_plane,
            ArrayRole.INDEXED_SUBSET_VARIANT_INDEX: _shard_indexed_subset_variant_index,
        }
    )
)


def require_shard_multiple(
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
    require_shard_multiple(role, resolved, shard)
    return shard
