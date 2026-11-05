"""Plant a state in a built Dense grid, one plane at a time.

A Store's writers write associations whole, through `AssociationCodec` and
`write_dense_cells`, and refuse a cell Association Admission would not admit.
A test that has to plant such a cell -- a Z without its SE, an infinite SE, a
value moved on one plane only -- writes that plane here, through the plane's
own codec, with its side table rewritten beside it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from opengwasdb.encoding import codec as side
from opengwasdb.encoding.codec import EAF_BASELINE, SE_COEFFICIENTS, StoreCodec, positions_pairs
from opengwasdb.encoding.plan import StoreEncoding
from opengwasdb.encoding.planes import DenseEafPlane, DenseSePlane, DenseZPlane
from opengwasdb.store.arrays import compressor, write_shard_cells


def patch(
    group: Any,
    encoding: StoreEncoding,
    plane: type,
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
) -> None:
    """Overwrite the cells `(rows[i], cols[i])` of one plane (`DenseZPlane`,
    `DenseEafPlane` or `DenseSePlane`) with `values[i]`, and nothing else.

    EAF is coded against the baseline already recorded, and a residual SE
    against the EAF the cells decode to now.
    """
    rows, cols = np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
    if not len(rows):
        return
    positions = positions_pairs(rows, cols, int(group["z"].shape[1]))
    codec = StoreCodec(encoding)
    if plane is DenseZPlane:
        z_entries = side.ZOverflowBuilder()
        codes = codec.encode_z(values, positions=positions, overflow=z_entries)
        write_shard_cells(group["z"], rows, cols, codes)
        if encoding.z.is_fixed_point:
            _replace(
                group,
                side.ZOverflowTable.read(group),
                z_entries,
                positions,
                side.ZOverflowBuilder(),
            )
    elif plane is DenseEafPlane:
        if "eaf" not in group:
            return
        baseline = None if EAF_BASELINE not in group else group[EAF_BASELINE].oindex[rows]
        eaf_entries = side.EafExceptionBuilder()
        codes = codec.encode_eaf(
            values, baseline=baseline, positions=positions, exceptions=eaf_entries
        )
        write_shard_cells(group["eaf"], rows, cols, codes)
        if encoding.eaf.is_residual:
            previous = side.EafExceptionTable.read(group)
            _replace(group, previous, eaf_entries, positions, side.EafExceptionBuilder())
    elif plane is DenseSePlane:
        _patch_se(group, encoding, rows, cols, positions, values)
    else:
        raise TypeError(f"no Dense plane {plane!r}")


def _patch_se(
    group: Any,
    encoding: StoreEncoding,
    rows: np.ndarray,
    cols: np.ndarray,
    positions: np.ndarray,
    values: np.ndarray,
) -> None:
    se_entries = side.SeExceptionBuilder()
    if encoding.se.is_residual:
        codes = StoreCodec(encoding).encode_se(
            values,
            eaf=DenseEafPlane.open(group, encoding).points(rows, cols),
            analysis_index=cols,
            coefficients=np.asarray(group[SE_COEFFICIENTS][:], dtype=np.float32),
            positions=positions,
            exceptions=se_entries,
        )
    else:
        codes = StoreCodec(encoding).encode_se(
            values, eaf=values, analysis_index=cols, coefficients=np.zeros((0, 2))
        )
    write_shard_cells(group["se"], rows, cols, codes)
    if encoding.se.is_residual:
        previous = side.SeExceptionTable.read(group)
        _replace(
            group,
            previous,
            se_entries,
            positions,
            side.SeExceptionBuilder(),
            compressor=compressor(),
        )


def _replace(
    group: Any,
    previous: Any,
    entries: Any,
    positions: np.ndarray,
    builder: Any,
    **write: Any,
) -> None:
    """Write `previous` with the entries at `positions` swapped for `entries`'s."""
    added = entries.table()
    kept = ~np.isin(previous.index, positions)
    builder.add(previous.index[kept], previous.value[kept])
    builder.add(added.index, added.value)
    builder.table().write(group, **write)
