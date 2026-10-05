"""What the read levers do to writes, on real OGS-00009 tiles (#244, ADR 0056).

Two questions decided whether the levers could be set process-wide rather than
for queries only:

1. Do compressed sizes move? The SE encoding plan is chosen from
   `len(Blosc.encode(...))` (`encoding/se.py:_packed`), measured in the parent
   when serial and in forked workers (always single-threaded) when parallel. If
   threaded and single-threaded Blosc gave different sizes, the plan could
   differ between `n_workers=1` and `n_workers>1`. This prints every
   `[1000, 1000]` tile's compressed size.
2. What does a band write cost? Builders write from the parent only. This times
   a write of each plane's band through zarr.

The label is `<pipeline>[+bt]`, pipeline in default | fused | fused_mw1;
`numcodecs.blosc.use_threads` is set to whether `+bt` is given. Run each label
in a fresh process, and compare the `tile_sizes` across labels.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from benchmarks import _zarr3_levers as levers

LABELS = ("default", "default+bt", "fused", "fused+bt", "fused_mw1", "fused_mw1+bt")


def tile_sizes(planes: dict[str, np.ndarray]) -> dict[str, list[int]]:
    """Every `[1000, 1000]` tile's compressed size, as the SE measurement sizes them."""
    import numcodecs

    codec = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=2)
    sizes: dict[str, list[int]] = {}
    for name, data in planes.items():
        sizes[name] = [
            len(codec.encode(np.ascontiguousarray(data[r : r + 1000, c : c + 1000])))
            for r in range(0, data.shape[0], 1000)
            for c in range(0, data.shape[1], 1000)
        ]
    return sizes


def band_writes(planes: dict[str, np.ndarray], scratch: Path, out: dict[str, object]) -> str:
    """Time one zarr write per plane; returns the pipeline the arrays used."""
    import numcodecs
    import zarr

    timings = {}
    pipeline = ""
    for name, data in planes.items():
        path = scratch / name
        arr = zarr.create_array(
            str(path),
            shape=data.shape,
            chunks=(1000, 1000),
            dtype=data.dtype,
            zarr_format=2,
            compressors=numcodecs.Blosc(cname="zstd", clevel=3, shuffle=2),
            overwrite=True,
        )
        t0 = time.perf_counter()
        arr[:] = data
        timings[name] = round((time.perf_counter() - t0) * 1000, 1)
        back = np.asarray(zarr.open_array(str(path), mode="r")[:])
        if not np.array_equal(back, data, equal_nan=data.dtype.kind == "f"):
            raise SystemExit(f"{name}: the written band reads back differently")
        files = sorted(p for p in path.iterdir() if not p.name.startswith("."))
        out[f"{name}_files"] = len(files)
        out[f"{name}_bytes"] = sum(p.stat().st_size for p in files)
        pipeline = type(arr._async_array.codec_pipeline).__name__
    out["write_ms"] = timings
    return pipeline


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("label", choices=LABELS)
    ap.add_argument("--store", type=Path, required=True, help="OGS-00009's store.opengwasdb")
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument(
        "--rows",
        type=int,
        nargs=2,
        default=(6_500_000, 6_520_000),
        help="the row band [start, stop); the default is 20 row chunks x 3 column chunks",
    )
    args = ap.parse_args()
    import numcodecs.blosc
    import zarr

    pipeline, _, bt = args.label.partition("+")
    zarr.config.set(levers.PIPELINES[pipeline])
    numcodecs.blosc.use_threads = bt == "bt"
    zarr.config.set({"array.write_empty_chunks": True})
    src = zarr.open_group(str(args.store / "data.zarr"), mode="r", zarr_format=2)
    start, stop = args.rows
    planes = {name: np.asarray(src[name][start:stop, :]) for name in ("z", "se", "eaf")}
    out: dict[str, object] = {"label": args.label, "use_threads": numcodecs.blosc.use_threads}
    out["tile_sizes"] = tile_sizes(planes)
    out["pipeline"] = band_writes(planes, args.scratch / f"enc-{args.label}", out)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
