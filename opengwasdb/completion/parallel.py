"""Process-pool worker setup shared by dense and ragged block-parallel completion."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

# Kept alive for the worker's lifetime so the BLAS thread cap persists (a bare
# threadpool_limits() call would reset on garbage collection).
_worker_thread_limiter: Any = None


def init_block_worker() -> None:
    """Cap each pool worker's BLAS (OpenBLAS/MKL) to one thread. numpy's linear
    algebra and sklearn otherwise spawn one thread per core *inside every worker*,
    so n_workers processes each with ~n_core threads massively oversubscribe the
    CPU (~1000 threads on 256 cores). One BLAS thread per worker gives clean
    process-level parallelism — scale with n_workers, not threads."""
    global _worker_thread_limiter
    _worker_thread_limiter = threadpool_limits(limits=1)


def run_block_tasks(
    pending: list[Any],
    n_workers: int,
    run_block: Any,
    *,
    noun: str = "blocks",
    every: int = 200,
) -> None:
    """Run completion block tasks over a process pool, or serially when
    ``n_workers <= 1``, propagating worker errors and printing progress every
    ``every`` tasks. Each task writes its own checkpoint; the parent keeps
    nothing per block (issue 044). Shared by the dense and ragged completion
    drivers, which differ only in what ``run_block`` does."""
    if n_workers <= 1:
        for i, task in enumerate(pending):
            run_block(task)
            if (i + 1) % every == 0:
                print(f"  {i + 1:,} / {len(pending):,} {noun}")
    else:
        with ProcessPoolExecutor(max_workers=n_workers, initializer=init_block_worker) as pool:
            futures = [pool.submit(run_block, task) for task in pending]
            for i, fut in enumerate(as_completed(futures)):
                fut.result()  # propagate worker errors; result is on disk
                if (i + 1) % every == 0:
                    print(f"  {i + 1:,} / {len(pending):,} {noun}")
