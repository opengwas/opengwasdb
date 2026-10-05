"""What each #242 harness shape reads on OGS-00009, re-expressed for candidate inner chunks.

Runs each harness shape once through the query facade, with this checkout's code
and reader configuration, and intercepts every zarr selection. For each array
read it records the array, its dtype size, the stored chunks it touches and,
for every candidate inner chunk `[R, C]`, the chunks it would touch:

* a 2-D plane `(n_variants, n_analyses)` takes the candidate chunk;
* a per-variant 1-D array (`eaf_baseline`, `eaf_reference`, `on_panel`)
  follows the plane's variant chunk R (the spec: no coarser than the plane's,
  never above 200,000);
* every other array (top-hit index, exception tables, coefficients) keeps its
  stored chunks.

`benchmarks/shape_screen.py` costs these reads per candidate. It also shows
that top hits makes six one-chunk reads (ADR 0056).

    pixi run -e dev python benchmarks/shape_harness_geometry.py \\
        --store /data/opengwasdb/stores/OGS-00009/store.opengwasdb \\
        --output /tmp/harness_geometry.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks import _zarr3_levers as levers

CANDIDATES = [(1000, 1000), (1000, 128), (1000, 64), (2000, 128), (4000, 256), (250, 512)]
CANDIDATES += [(2000, 256), (4000, 128), (8000, 64), (1000, 256), (500, 256), (2000, 64)]

Index = np.ndarray | range


def normalise(sel: Any, shape: tuple[int, ...]) -> list[Index]:
    """Per-dimension index sets: ranges for slices and ints, arrays for fancy indexes."""
    if not isinstance(sel, tuple):
        sel = (sel,)
    if any(s is Ellipsis for s in sel):
        i = next(j for j, s in enumerate(sel) if s is Ellipsis)
        sel = sel[:i] + (slice(None),) * (len(shape) - len(sel) + 1) + sel[i + 1 :]
    sel = sel + (slice(None),) * (len(shape) - len(sel))
    out: list[Index] = []
    for s, n in zip(sel, shape, strict=True):
        if isinstance(s, slice):
            out.append(range(*s.indices(n)))
        elif isinstance(s, int | np.integer):
            out.append(range(int(s) % n, int(s) % n + 1))
        else:
            out.append(np.asarray(s, dtype=np.int64))
    return out


def n_index_chunks(idx: Index, c: int) -> int:
    if isinstance(idx, range):
        if len(idx) == 0:
            return 0
        if idx.step == 1:
            return (idx[-1] // c) - (idx[0] // c) + 1
        idx = np.arange(idx.start, idx.stop, idx.step)
    return int(np.unique(idx // c).size)


def n_chunks(kind: str, dims: list[Index], chunk: tuple[int, ...]) -> int:
    if kind == "coordinate":
        if len(dims) == 1:
            return n_index_chunks(dims[0], chunk[0])
        keys = (np.asarray(dims[0]) // chunk[0]) * 1_000_000 + (np.asarray(dims[1]) // chunk[1])
        return int(np.unique(keys).size)
    total = 1
    for idx, c in zip(dims, chunk, strict=False):
        total *= n_index_chunks(idx, c)
    return total


class Recorder:
    """Wraps zarr's selection methods and logs every read made while a shape runs."""

    def __init__(self, n_variants: int) -> None:
        self.n_variants = n_variants
        self.shape: str | None = None
        self.log: list[dict[str, Any]] = []

    def record(self, array: Any, selection: Any, kind: str) -> None:
        shape = tuple(array.shape)
        if kind == "coordinate":
            parts = selection if isinstance(selection, tuple) else (selection,)
            dims: list[Index] = [np.asarray(s, dtype=np.int64).ravel() for s in parts]
        elif kind == "mask":
            dims = [np.asarray(x, dtype=np.int64) for x in np.nonzero(np.asarray(selection))]
        else:
            dims = normalise(selection, shape)
        kind = "coordinate" if kind in ("coordinate", "mask") else "box"
        stored = tuple(array.chunks)
        if len(shape) == 2 and shape[0] == self.n_variants:
            role = "plane"
        elif len(shape) == 1 and shape[0] == self.n_variants:
            role = "per_variant"
        else:
            role = "other"
        itemsize = int(np.dtype(array.dtype).itemsize)
        rec: dict[str, Any] = {
            "shape_name": self.shape,
            "array": array.path,
            "role": role,
            "itemsize": itemsize,
            "stored_chunks": list(stored),
            "kind": kind,
            "stored_n": n_chunks(kind, dims, stored),
        }
        cand = {}
        for r, c in CANDIDATES:
            ch = (
                (r, c)
                if role == "plane"
                else (min(r, 200_000),)
                if role == "per_variant"
                else stored
            )
            cand[f"{r}x{c}"] = {
                "n": n_chunks(kind, dims, ch),
                "chunk_bytes": int(np.prod(ch)) * itemsize,
            }
        rec["candidates"] = cand
        self.log.append(rec)

    def install(self) -> None:
        from zarr.core.array import Array

        methods = (
            ("get_basic_selection", "basic"),
            ("get_orthogonal_selection", "orthogonal"),
            ("get_coordinate_selection", "coordinate"),
            ("get_mask_selection", "mask"),
        )
        for name, kind in methods:
            original = getattr(Array, name)

            def wrapper(
                array: Any, selection: Any, *a: Any, _o: Any = original, _k: str = kind, **k: Any
            ) -> Any:
                if self.shape is not None:
                    self.record(array, selection, _k)
                return _o(array, selection, *a, **k)

            setattr(Array, name, wrapper)

    def run(self, name: str, fn: Callable[[], dict[str, Any]]) -> None:
        fn()  # warm: lazily opened planes open outside the recorded call
        self.shape = name
        t0 = time.perf_counter()
        result = fn()
        self.shape = None
        self.log.append(
            {
                "shape_name": name,
                "summary": True,
                "rows": len(result["z"]),
                "s": round(time.perf_counter() - t0, 3),
            }
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--selection", type=Path, default=levers.ZARR2_BASELINE)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--skip-bulk", action="store_true", help="omit one whole Analysis")
    args = ap.parse_args()
    import zarr

    from opengwasdb.query import query_store

    selection = levers.load_selection(args.selection)
    with query_store(args.store) as q:
        recorder = Recorder(int(q._root["z"].shape[0]))
        recorder.install()
        shapes = levers.lever_shapes(q, selection, with_bulk=not args.skip_bulk)
        for name, fn in shapes.items():
            recorder.run(name, fn)
    payload = {
        "n_variants": recorder.n_variants,
        "candidates": CANDIDATES,
        "zarr": zarr.__version__,
        "log": recorder.log,
    }
    args.output.write_text(json.dumps(payload))
    by_shape: dict[str, list[dict[str, Any]]] = {}
    for rec in recorder.log:
        if not rec.get("summary"):
            by_shape.setdefault(str(rec["shape_name"]), []).append(rec)
    for name, recs in by_shape.items():
        arrays = sorted({str(r["array"]) for r in recs})
        stored = sum(int(r["stored_n"]) for r in recs)
        print(name, "reads", len(recs), "stored chunks", stored, arrays)


if __name__ == "__main__":
    main()
