"""Writing a store's statistic planes: the encoded planes, their side tables and
per-variant arrays.

The decoded views over the planes -- `DenseZPlane`, `DenseSePlane`,
`DenseEafPlane`, `RaggedEafPlane`, `RaggedSePlane` -- are the Store Reader's
(`opengwasdb.store_reader.encoding.planes`, #464).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from opengwasdb.encoding.codec import (
    EafExceptionBuilder,
    SeExceptionBuilder,
    StoreCodec,
    ZOverflowBuilder,
    require_parallel,
    write_exact_table,
)
from opengwasdb.store.arrays import (
    SE_COEFFICIENTS,
    ArrayRole,
    component_variant_chunk,
    compressor,
    create_array,
    write_shard_cells,
)
from opengwasdb.store_reader.encoding.codec import (
    EAF_BASELINE,
    EAF_REFERENCE,
    EafExceptionTable,
    SeExceptionTable,
    ZOverflowTable,
    positions_pairs,
)
from opengwasdb.store_reader.encoding.plan import EafBaselineError, StoreEncoding

if TYPE_CHECKING:
    from opengwasdb.encoding.association_codec import Encoded

    # ---- writing ---------------------------------------------------------

# ── Writing an `eaf` plane ──────────────────────────────────────────────────
#
# The plane, its per-variant baseline and its exception table are one artifact
# in three arrays: a component that has any of them must have all of them.
# These helpers write them together for the two shapes that occur, so no
# builder can produce two of the three.


def _write_per_variant_array(
    group: Any,
    name: str,
    values: np.ndarray,
    *,
    compressor: Any = None,
    chunk: int | None = None,
    role: ArrayRole = ArrayRole.PER_VARIANT,
) -> None:
    """Write (or replace) one `float32` per variant of a component's axis.

    The component plane's variant-axis chunk is read here and passed in as
    `component_chunk`, so the layout policy itself never inspects `group`.  The
    `role` is explicit rather than sniffed from `group.path`: a Ragged CSR
    component opened directly at `data.zarr/ragged` reports `path == ""`, so a
    path check cannot tell it from a Dense root (#247 review round 1).
    """
    create_array(
        group,
        name,
        role,
        data=np.asarray(values, dtype=np.float32),
        dtype="float32",
        compressor=compressor,
        hint=chunk,
        component_chunk=component_variant_chunk(group),
        overwrite=True,
    )


def write_eaf_baseline(
    group: Any,
    baseline: np.ndarray,
    *,
    compressor: Any = None,
    chunk: int | None = None,
    role: ArrayRole = ArrayRole.PER_VARIANT,
) -> None:
    """Write the per-variant `eaf_baseline` the residual coding decodes against."""
    _write_per_variant_array(
        group, EAF_BASELINE, baseline, compressor=compressor, chunk=chunk, role=role
    )


def write_eaf_reference(
    group: Any,
    reference: np.ndarray,
    *,
    compressor: Any = None,
    chunk: int | None = None,
    role: ArrayRole = ArrayRole.PER_VARIANT,
) -> None:
    """Write the per-variant reference-panel frequency (ADR 0037 §4).

    One `float32` per variant, and ~0 bytes per cell: an imputed cell's EAF is
    the panel's, identical for every Analysis imputed at that variant, so it is
    a per-variant constant rather than per-cell data.
    """
    _write_per_variant_array(
        group, EAF_REFERENCE, reference, compressor=compressor, chunk=chunk, role=role
    )


def write_se_coefficients(group: Any, coefficients: np.ndarray, *, compressor: Any = None) -> None:
    """Write (or replace) the two `float32` decode parameters per Analysis."""
    create_array(
        group,
        SE_COEFFICIENTS,
        ArrayRole.SE_COEFFICIENTS,
        data=np.asarray(coefficients, dtype=np.float32),
        dtype="float32",
        compressor=compressor,
        overwrite=True,
    )


def _side_table_role(plane_role: ArrayRole) -> ArrayRole:
    """The exception/overflow table role for a plane's role (#248).

    A Ragged CSR component's tables have their own shard policy, so a table
    written beside a CSR plane takes the Ragged role.  The role is derived from
    the **plane** role the writer was already given, never from `group.path`:
    a component opened directly at `data.zarr/ragged` reports `path == ""`, so a
    path check cannot tell it from a Dense root (#247 review round 1).
    """
    return (
        ArrayRole.RAGGED_EXCEPTION_TABLE
        if plane_role is ArrayRole.ASSOCIATION_SEQUENCE
        else ArrayRole.EXCEPTION_TABLE
    )


def _write_se_arrays(
    group: Any,
    codec: StoreCodec,
    codes: np.ndarray,
    coefficients: np.ndarray | None,
    exceptions: SeExceptionTable | None,
    *,
    role: ArrayRole,
    compressor: Any,
    chunks: tuple[int, ...] | None,
) -> None:
    """Replace the plane and, for a residual plan, both of its side arrays.

    The plane, its coefficients and its exception table are one artifact in
    three arrays, exactly as the `eaf` writers above treat theirs: writing them
    from one place is what stops a builder producing two of the three.
    """
    create_array(
        group,
        "se",
        role,
        data=codes,
        dtype=codec.encoding.se.dtype,
        compressor=compressor,
        hint=chunks,
        overwrite=True,
    )
    if not codec.encoding.se.is_residual:
        return
    assert coefficients is not None and exceptions is not None
    write_se_coefficients(group, coefficients, compressor=compressor)
    write_exact_table(exceptions, group, compressor=compressor, role=_side_table_role(role))


# ── Writing an encoded batch ────────────────────────────────────────────────


def _write_eaf_arrays(
    group: Any,
    plan: StoreEncoding,
    encoded: Encoded,
    *,
    role: ArrayRole,
    compressor: Any,
    chunks: tuple[int, ...] | None,
    eaf_baseline: np.ndarray | None,
) -> None:
    """Replace the `eaf` plane and, for a residual plan, its baseline and exception table.

    The plane, its baseline and its exception table are one artifact in three
    arrays, exactly as `_write_se_arrays` treats the SE plan's: writing them
    from one place is what stops a builder producing two of the three. A batch
    with no frequency plane writes none of them.
    """
    if encoded.eaf is None:
        return
    create_array(
        group,
        "eaf",
        role,
        data=encoded.eaf,
        dtype=plan.eaf.dtype,
        compressor=compressor,
        hint=chunks,
        overwrite=True,
    )
    if not plan.eaf.is_residual:
        return
    if eaf_baseline is None:
        raise EafBaselineError("a residual eaf plane is written with its eaf_baseline")
    write_eaf_baseline(group, eaf_baseline, compressor=compressor, role=_per_variant_role(role))
    write_exact_table(encoded.tables.eaf_exceptions, group, role=_side_table_role(role))


def write_encoded_planes(
    group: Any,
    plan: StoreEncoding,
    encoded: Encoded,
    *,
    role: ArrayRole,
    compressor: Any,
    chunks: tuple[int, ...] | None = None,
    eaf_baseline: np.ndarray | None = None,
    se_coefficients: np.ndarray | None = None,
) -> None:
    """Write a whole component's statistic planes from one encoded batch.

    Each plane with the arrays it cannot be decoded without: z with its
    overflow table, `eaf` with its baseline and exception table, `se` with its
    coefficients and exception table. `role` is the planes' own (a Dense grid's
    or an association sequence's) and decides the side arrays' roles. The `z`
    plane takes `chunks` as a hint and the others take the chunks it resolved
    to, so the three planes are chunked alike.

    The batch's arrays are judged parallel before the first plane is created
    (issue #392): one that is not describes cells the component does not have,
    and a plane written from it holds one cell's statistic under another's
    variant.
    """
    require_parallel(_batch_arrays(encoded), where="write_encoded_planes")
    z_array = create_array(
        group, "z", role, data=encoded.z, dtype=plan.z.dtype, compressor=compressor, hint=chunks
    )
    resolved = None if chunks is None else tuple(int(size) for size in z_array.chunks)
    if plan.z.is_fixed_point:
        # A float plane has no overflow table, and must not acquire an empty one.
        write_exact_table(encoded.tables.z_overflow, group, role=_side_table_role(role))
    _write_eaf_arrays(
        group,
        plan,
        encoded,
        role=role,
        compressor=compressor,
        chunks=resolved,
        eaf_baseline=eaf_baseline,
    )
    _write_se_arrays(
        group,
        StoreCodec(plan),
        encoded.se,
        se_coefficients,
        encoded.tables.se_exceptions,
        role=role,
        compressor=compressor,
        chunks=resolved,
    )


def write_dense_cells(
    group: Any, plan: StoreEncoding, rows: np.ndarray, cols: np.ndarray, encoded: Encoded
) -> None:
    """Overwrite the Dense cells `(rows[i], cols[i])` of a built grid with an encoded batch.

    The one partial in-place write against a built grid -- Hybrid completion
    folding a crossed-over association into the Dense Component (issue #99).
    Each side table is rewritten with its plane, not after it: a cell can
    enter or leave a table, and a table left describing the value a cell used
    to have is the stale-by-one-call-site defect the codec exists to prevent.

    `rows`, `cols` and the batch address the same cells, and a call whose
    arrays disagree is refused before the grid is touched (issue #392).
    """
    rows, cols = np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
    require_parallel(
        (("rows", rows), ("cols", cols), *_batch_arrays(encoded)), where="write_dense_cells"
    )
    if len(rows) == 0:
        return
    positions = positions_pairs(rows, cols, int(group["z"].shape[1]))
    tables = encoded.tables
    # One whole-shard write per touched shard, not a per-cell `vindex` write:
    # the latter is a read-modify-write of the shard and is invisible to the
    # whole-shard guard (#247 review round 1).
    write_shard_cells(group["z"], rows, cols, encoded.z)
    if plan.z.is_fixed_point:
        # A float plane has no overflow table, and must not acquire an empty one.
        z_table = ZOverflowTable.read(group)
        write_exact_table(_merged(z_table, positions, tables.z_overflow, ZOverflowBuilder()), group)
    if encoded.imputed is not None:
        write_shard_cells(group["imputed"], rows, cols, encoded.imputed)
    if encoded.eaf is not None and "eaf" in group:
        write_shard_cells(group["eaf"], rows, cols, encoded.eaf)
        if plan.eaf.is_residual:
            eaf_table = EafExceptionTable.read(group)
            write_exact_table(
                _merged(eaf_table, positions, tables.eaf_exceptions, EafExceptionBuilder()), group
            )
    write_shard_cells(group["se"], rows, cols, encoded.se)
    if plan.se.is_residual:
        se_table = SeExceptionTable.read(group)
        merged = _merged(se_table, positions, tables.se_exceptions, SeExceptionBuilder())
        write_exact_table(merged, group, compressor=compressor())


def _batch_arrays(encoded: Encoded) -> tuple[tuple[str, Any], ...]:
    """An encoded batch's own arrays: every one of them describes the same cells.

    The tables are not among them -- they hold an entry per exceptional cell,
    not per cell -- so `require_parallel` judges only the batch.
    """
    return (
        ("z", encoded.z),
        ("se", encoded.se),
        ("eaf", encoded.eaf),
        ("imputed", encoded.imputed),
    )


_Table = ZOverflowTable | EafExceptionTable | SeExceptionTable
_Builder = ZOverflowBuilder | EafExceptionBuilder | SeExceptionBuilder


def _merged(previous: _Table, positions: np.ndarray, added: _Table, builder: _Builder) -> _Table:
    """`previous` with the entries at `positions` replaced by `added`'s.

    `previous` is the whole table read from the group: a codec's table is a
    windowed view whose in-memory arrays are empty, and merging with it would
    drop every existing entry (#99 review round 2, finding 1).
    """
    untouched = ~np.isin(previous.index, positions)
    builder.add(previous.index[untouched], previous.value[untouched])
    builder.add(added.index, added.value)
    return builder.table()


def _per_variant_role(plane_role: ArrayRole) -> ArrayRole:
    """The per-variant array role for a plane's role, as `_side_table_role` is for tables."""
    return (
        ArrayRole.RAGGED_PER_VARIANT
        if plane_role is ArrayRole.ASSOCIATION_SEQUENCE
        else ArrayRole.PER_VARIANT
    )
