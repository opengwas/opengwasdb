"""Mask a legacy Dense release's cells with an SE above 65,504 (#492, ADR 0068).

What `storable` leaves out of a build, written into a release that stored it: the
cell's z, SE and frequency at their plan's missing codes, its imputed flag 0, and its
entries in the exact-value side tables gone, in the grid and in each Indexed Variant
Subset; and its row in every Top-Hit tier gone. Only `legacy_conversion` calls these,
on a staged copy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from opengwasdb.encoding.codec import write_exact_table
from opengwasdb.store import arrays as store_arrays
from opengwasdb.store_reader.encoding.codec import (
    EafExceptionTable,
    SeExceptionTable,
    SparseExactTable,
    ZOverflowTable,
)
from opengwasdb.store_reader.encoding.plan import (
    EAF_ABSENT,
    SE_MISSING,
    Z_MISSING,
    StoreEncoding,
)
from opengwasdb.store_reader.indexed import INDEXED_SUBSETS_GROUP
from opengwasdb.store_reader.layout import ArrayRole
from opengwasdb.store_reader.top_hits import TOP_HIT_GROUP

#: The planes a masked cell is rewritten in. A sample-size plane or a Rho Matrix is
#: refused: what a build holds there for a left-out association is not this module's
#: to decide, and no release `--convert-legacy` is for carries one.
_REFUSED_PLANES = ("n", "rho")

_TABLES: tuple[type[SparseExactTable], ...] = (ZOverflowTable, SeExceptionTable, EafExceptionTable)


def missing_codes(encoding: StoreEncoding) -> dict[str, Any]:
    """Each plane's missing code under `encoding`: what an encoder writes for NaN."""
    return {
        "z": Z_MISSING if encoding.z.is_fixed_point else np.nan,
        "se": SE_MISSING if encoding.se.is_residual else np.nan,
        "eaf": EAF_ABSENT if encoding.eaf.is_residual else np.nan,
        "imputed": 0,
    }


def mask_cells(group: Any, positions: np.ndarray, encoding: StoreEncoding, n_analyses: int) -> None:
    """Mask the cells at flat `positions` of the grid in `group`."""
    if not positions.size:
        return
    _refuse_other_planes(group)
    rows, cols = positions // n_analyses, positions % n_analyses
    for name, code in missing_codes(encoding).items():
        if name in group:
            plane = group[name]
            values = np.full(rows.size, code, dtype=plane.dtype)
            store_arrays.write_shard_cells(plane, rows, cols, values)
    for table in _TABLES:
        _drop_entries(group, table, positions)


def _refuse_other_planes(group: Any) -> None:
    held = [name for name in _REFUSED_PLANES if name in group]
    if held:
        raise ValueError(
            f"{group.path or 'data.zarr'} carries {', '.join(held)}: a cell masked for an SE "
            "above 65,504 is not rewritten there, so the release is rebuilt from source"
        )


def _drop_entries(group: Any, table_type: type[SparseExactTable], positions: np.ndarray) -> None:
    """Remove the table's entries at `positions`, rewriting it only when it holds one."""
    if table_type.index_name not in group:
        return
    table = table_type.read(group)
    gone = np.isin(table.index, positions)
    if gone.any():
        kept = table_type(index=table.index[~gone], value=table.value[~gone])
        write_exact_table(kept, group, compressor=store_arrays.compressor())


def mask_subsets(
    root: Any, positions: np.ndarray, encoding: StoreEncoding, n_analyses: int
) -> None:
    """Mask the cells in every Indexed Variant Subset that holds their row."""
    if INDEXED_SUBSETS_GROUP not in root:
        return
    rows, cols = positions // n_analyses, positions % n_analyses
    subsets = root[INDEXED_SUBSETS_GROUP]
    for name in sorted(subsets.group_keys()):
        subset = subsets[name]
        held = np.asarray(subset["variant_index"][:], dtype=np.int64)
        slot = np.searchsorted(held, rows)
        inside = slot < held.size
        inside[inside] = held[slot[inside]] == rows[inside]
        mask_cells(subset, slot[inside] * n_analyses + cols[inside], encoding, n_analyses)


def mask_tiers(root: Any, positions: np.ndarray, n_analyses: int) -> None:
    """Drop the masked cells' rows from every Top-Hit tier, and renumber its offsets."""
    if TOP_HIT_GROUP not in root:
        return
    index = root[TOP_HIT_GROUP]
    for key in sorted(index.group_keys()):
        tier = index[key]
        cells = np.asarray(tier["variant_index"][:], np.int64) * n_analyses + np.asarray(
            tier["analysis_index"][:], np.int64
        )
        keep = ~np.isin(cells, positions)
        if not keep.all():
            _rewrite_tier(tier, keep, n_analyses)


def _rewrite_tier(tier: Any, keep: np.ndarray, n_analyses: int) -> None:
    analyses = np.asarray(tier["analysis_index"][:], dtype=np.int64)[keep]
    offsets = np.zeros(n_analyses + 1, dtype="uint64")
    np.cumsum(np.bincount(analyses, minlength=n_analyses), dtype=np.uint64, out=offsets[1:])
    for name in sorted(tier.array_keys()):
        if name == "analysis_offsets":
            data, role = offsets, ArrayRole.TOP_HIT_ANALYSIS_OFFSETS
        else:
            data, role = np.asarray(tier[name][:])[keep], ArrayRole.TOP_HIT_INDEX
        store_arrays.create_array(
            tier, name, role, data=data, dtype=data.dtype,
            compressor=store_arrays.compressor(), overwrite=True,
        )


__all__ = ["mask_cells", "mask_subsets", "mask_tiers", "missing_codes"]
