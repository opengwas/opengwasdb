"""Does a forked worker finish its read under each zarr 3 read lever? (#244, ADR 0056)

Each case runs in a fresh process; run one case per invocation. The parent
first does the lever's work, so any process-global state is live -- Blosc's
global-context thread pool, the `FusedCodecPipeline`'s module-level `_pool` --
then forks a `ProcessPoolExecutor` (fork start, as every build pool here does)
whose workers read multi-chunk selections and Blosc-encode them. A worker that
cannot finish in `--timeout` seconds is reported as HANG, the workers are
killed, and the process exits 3.

  baseline        zarr 3's defaults
  bt              + numcodecs.blosc.use_threads = True
  fused           FusedCodecPipeline with its default thread pool
  fused+bt        both
  fused+bt+reset  both, plus an at-fork hook clearing zarr's private `_pool`
  fused_mw1+bt    the shipped configuration: one worker, Blosc threads on

`fused` and `fused+bt` hang on zarr 3.4.0; the other cases finish
(zarr-developers/zarr-python#4478).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

import numpy as np

from benchmarks import _zarr3_levers as levers

CASES = ("baseline", "bt", "fused", "fused+bt", "fused+bt+reset", "fused_mw1+bt")


def _setup(case: str) -> None:
    import numcodecs.blosc
    import zarr

    parts = set(case.split("+"))
    for part in parts & set(levers.PIPELINES):
        zarr.config.set(levers.PIPELINES[part])
    if "bt" in parts:
        numcodecs.blosc.use_threads = True
    if "reset" in parts:
        import zarr.core.codec_pipeline as cp

        def _reset() -> None:
            cp._pool = None
            cp._pool_size = 0

        os.register_at_fork(after_in_child=_reset)


def _child_read(path: str) -> dict[str, object]:
    import numcodecs
    import numcodecs.blosc
    import zarr
    import zarr.core.codec_pipeline as cp

    t0 = time.perf_counter()
    arr = zarr.open_array(path, mode="r")
    block = np.asarray(arr[:, 1500:2500])  # chunk columns 1 and 2, all 4 row chunks
    codec = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=2)
    encoded = codec.encode(np.ascontiguousarray(block))
    return {
        "pid": os.getpid(),
        "sum": int(block.astype(np.int64).sum()),
        "encoded": len(encoded),
        "child_use_threads_effective": numcodecs.blosc._get_use_threads(),
        "pool_inherited": cp._pool is not None,
        "s": round(time.perf_counter() - t0, 3),
    }


def _parent_array(case: str, scratch: Path) -> tuple[str, np.ndarray, dict[str, object]]:
    """Write and read back 16 multi-block chunks in the parent, so the lever is live."""
    import numcodecs
    import numcodecs.blosc
    import zarr
    import zarr.core.codec_pipeline as cp

    rng = np.random.default_rng(244)
    data = rng.integers(-3000, 3000, size=(4000, 4000), dtype=np.int16)
    path = str(scratch / f"probe-{case}.zarr")
    arr = zarr.create_array(
        path,
        shape=data.shape,
        chunks=(1000, 1000),
        dtype="int16",
        zarr_format=2,
        compressors=numcodecs.Blosc(cname="zstd", clevel=3, shuffle=2),
        overwrite=True,
    )
    arr[:] = data  # 16 chunks of 2 MB, 16 Blosc blocks each
    back = np.asarray(zarr.open_array(path, mode="r")[:])
    if not np.array_equal(back, data):
        raise SystemExit("the parent read back different values")
    parent = {
        "use_threads": numcodecs.blosc.use_threads,
        "parent_use_threads_effective": numcodecs.blosc._get_use_threads(),
        "parent_pool_created": cp._pool is not None,
        "pipeline": type(zarr.open_array(path, mode="r")._async_array.codec_pipeline).__name__,
    }
    return path, data, parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("case", choices=CASES)
    ap.add_argument("--scratch", type=Path, required=True, help="where the probe array goes")
    ap.add_argument("--timeout", type=int, default=60, help="seconds before a worker is a HANG")
    args = ap.parse_args()
    _setup(args.case)
    args.scratch.mkdir(parents=True, exist_ok=True)
    path, data, parent = _parent_array(args.case, args.scratch)
    expect = int(data[:, 1500:2500].astype(np.int64).sum())
    result: dict[str, object] = {"case": args.case, "parent": parent}
    ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=4, mp_context=ctx) as pool:
        futures = [pool.submit(_child_read, path) for _ in range(8)]
        try:
            children = [f.result(timeout=args.timeout) for f in futures]
        except FutureTimeout:
            result["outcome"] = f"HANG (no result within {args.timeout}s)"
            print(json.dumps(result), flush=True)
            for proc in multiprocessing.active_children():
                proc.kill()
            os._exit(3)
    if not all(c["sum"] == expect for c in children):
        raise SystemExit("a forked worker read the wrong values")
    result["outcome"] = "ok"
    result["children"] = children[:2]
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
