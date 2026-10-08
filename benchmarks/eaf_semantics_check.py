"""Which EAF does residual SE decode against on a Reference-Completed Dense release? (#253)

#253 shares one EAF read between SE decoding and the result's `eaf` column. That
is only correct if both consume the same decoded frequency. This read-only check
on OGS-00010 (Dense, Reference-Completed, `se` and `eaf` int8_residual, `eaf`
with `reference: true`) compares, for cells of one Analysis:

  se_eaf      what `DenseSePlane` decodes SE against (its own `DenseEafPlane`)
  result_eaf  what the query facade reports in the result's `eaf` column
  raw_eaf     the stored plane decoded without the panel substitution

and tries decoding SE against `raw_eaf`, separately for imputed and observed
cells. On OGS-00010 `se_eaf` equals `result_eaf` for both, `raw_eaf` is NaN on
every imputed cell, and decoding SE against it raises.

    pixi run -e dev python benchmarks/eaf_semantics_check.py \\
        --store /data/opengwasdb/stores/OGS-00010/store.opengwasdb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _cells(q: Any, rows: np.ndarray, col: int, n_cols: int) -> dict[str, object]:
    from opengwasdb.encoding.codec import StoreCodec
    from opengwasdb.encoding.planes import positions_pairs

    cols = np.full(len(rows), col, dtype=np.int64)
    se_eaf = q._se._eaf.points(rows, cols)
    result_eaf = q._eaf.points(rows, cols)
    raw_codec = StoreCodec(
        q._encoding.with_eaf_reference(False), eaf_exceptions=q._eaf._codec.eaf_exceptions
    )
    raw_eaf = raw_codec.decode_eaf(
        q._eaf._array.vindex[rows, cols],
        baseline=q._eaf._gather(q._eaf._baseline, rows),
        positions=positions_pairs(rows, cols, n_cols),
    )
    se_codes = np.asarray(q._se._array.vindex[rows, cols])
    rec: dict[str, object] = {
        "cells": int(len(rows)),
        "se_eaf_equals_result_eaf": bool(np.array_equal(se_eaf, result_eaf, equal_nan=True)),
        "raw_eaf_nan_fraction": float(np.isnan(raw_eaf).mean()) if len(rows) else None,
        "raw_eaf_equals_result_eaf": bool(np.array_equal(raw_eaf, result_eaf, equal_nan=True)),
        "cells_with_se": int((se_codes != -128).sum()),
    }
    try:
        positions = positions_pairs(rows, cols, n_cols)
        se_from_raw = q._se._decode(se_codes, raw_eaf, cols, positions)
        se_true = q._se.points(rows, cols)
    except ValueError as exc:  # residual SE refuses a NaN frequency: the answer recorded
        rec["se_decode_with_raw_eaf"] = f"{type(exc).__name__}: {exc}"
        return rec
    rec["se_decode_with_raw_eaf"] = "ok"
    rec["se_from_raw_equals_se"] = bool(np.array_equal(se_from_raw, se_true, equal_nan=True))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--store", type=Path, required=True, help="OGS-00010's store.opengwasdb")
    ap.add_argument("--column", type=int, default=0, help="the Analysis column checked")
    ap.add_argument("--cells", type=int, default=20_000, help="cells per kind")
    args = ap.parse_args()
    from opengwasdb.query import query_store

    out: dict[str, object] = {"store": str(args.store)}
    with query_store(args.store) as q:
        out["encoding"] = q._encoding.to_manifest()
        _, n_cols = q._root["z"].shape
        col = args.column
        imputed_col = np.asarray(q._imputed[:, col], dtype=bool)
        se_raw_col = np.asarray(q._root["se"][:, col])
        rows_imp = np.where(imputed_col)[0][: args.cells]
        rows_obs = np.where(~imputed_col & (se_raw_col != -128))[0][: args.cells]
        out["column"] = col
        out["n_imputed_in_column"] = int(imputed_col.sum())
        for label, rows in (("imputed", rows_imp), ("observed", rows_obs)):
            out[label] = _cells(q, rows, col, n_cols)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
