"""Standalone reproducer: zarr's fused-pipeline thread pool hangs a forked reader.

Reported upstream as zarr-developers/zarr-python#4478; ADR 0056 records the
constraint it places on this package. It imports nothing from opengwasdb.

The parent reads a 16-chunk array through `FusedCodecPipeline`, which creates
the module-level `zarr.core.codec_pipeline._pool`. zarr 3.4's after-fork reset
does not clear that pool, so a forked child inherits it without its threads. A
child read of more than one chunk, but of no more chunks than the idle permits
the parent's pool left, queues work nothing runs. The two-chunk child read here
is in that window.

    python benchmarks/zarr3_pool_fork_repro.py                 # hangs: exits 1 after 30 s
    python benchmarks/zarr3_pool_fork_repro.py --max-workers 1  # finishes: exits 0

With `codec_pipeline.max_workers = 1` the pool is never created.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, TimeoutError

import numpy as np
import zarr
import zarr.core.codec_pipeline as cp

PATH = os.path.join(tempfile.mkdtemp(), "a.zarr")


def child(_: int) -> int:
    return int(zarr.open_array(PATH, mode="r")[:10, :20].sum())  # 2 chunks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--max-workers", type=int, default=None, help="codec_pipeline.max_workers")
    ap.add_argument("--timeout", type=int, default=30)
    args = ap.parse_args()
    cfg: dict[str, object] = {"codec_pipeline.path": "zarr.core.codec_pipeline.FusedCodecPipeline"}
    if args.max_workers is not None:
        cfg["codec_pipeline.max_workers"] = args.max_workers
    zarr.config.set(cfg)
    a = zarr.create_array(PATH, shape=(40, 40), chunks=(10, 10), dtype="i2")
    a[:] = np.arange(1600, dtype="i2").reshape(40, 40)  # 16 chunks: the parent uses cp._pool
    a[:]
    permits = cp._pool._idle_semaphore._value if cp._pool else None
    print("zarr", zarr.__version__, "| parent pool idle permits:", permits)
    pool = ProcessPoolExecutor(2, mp_context=mp.get_context("fork"))
    try:
        print("children:", list(pool.map(child, range(2), timeout=args.timeout)), flush=True)
    except TimeoutError:
        print(f"HANG: forked workers did not return in {args.timeout} s", flush=True)
        for proc in mp.active_children():
            proc.kill()
        os._exit(1)
    pool.shutdown()


if __name__ == "__main__":
    main()
