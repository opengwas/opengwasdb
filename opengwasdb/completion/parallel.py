"""The process pools Dense and Ragged completion share: the LD-block pool, and
the pool that writes the completed planes a unit at a time.

The LD blocks run on a plain process pool whose workers write their own
checkpoints (ADR 0023). The write after them runs its units -- a Dense row band,
a Ragged Analysis -- on an ordered fork pool (`ordered_map`): each worker reads
and computes its own unit, the parent takes the results in unit order, so what
the parent assembles from them (side tables, counts, a CSR sequence) is the
serial pass's whichever worker finished first. A unit holds its cells in
memory, so the write runs no more workers than a quarter of the host's
available memory holds beside the parent (`encoding.memory.pool_workers`).
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, TypeVar

from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from opengwasdb.build.ordered_pool import ordered_map
from opengwasdb.encoding.memory import RESULTS_HELD_PER_WORKER
from opengwasdb.encoding.timing import log_progress

P = TypeVar("P")
U = TypeVar("U")
R = TypeVar("R")

log = logging.getLogger(__name__)

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
    ``n_workers <= 1``, propagating worker errors and logging progress, with an
    ETA, every ``every`` tasks and on the last. Each task writes its own
    checkpoint; the parent keeps nothing per block (issue 044). Shared by the
    dense and ragged completion drivers, which differ only in what
    ``run_block`` does.

    The ETA extrapolates the mean time per block so far; blocks differ in size,
    so it is a guide, not a promise."""
    total = len(pending)
    started = time.monotonic()
    label = f"LD {noun}"
    if n_workers <= 1:
        for i, task in enumerate(pending):
            run_block(task)
            log_progress(log, label, i + 1, total, started, every=every)
    else:
        with ProcessPoolExecutor(max_workers=n_workers, initializer=init_block_worker) as pool:
            futures = [pool.submit(run_block, task) for task in pending]
            for i, fut in enumerate(as_completed(futures)):
                fut.result()  # propagate worker errors; result is on disk
                log_progress(log, label, i + 1, total, started, every=every)


def ordered_write(
    fn: Callable[[P, U], R],
    plan: P,
    units: Iterable[U],
    workers: int,
) -> Iterator[R]:
    """``fn(plan, unit)`` for each unit, across ``workers`` forked processes,
    yielded in unit order.

    ``plan`` is everything the units read, opened by the parent before the pool
    forks and inherited, never pickled (`ordered_map`'s ``shared``). A worker's
    error is raised here at its unit's position, and the units not yet started
    are cancelled. The parent holds at most `RESULTS_HELD_PER_WORKER` results a
    worker, the share `pool_workers` budgets.
    """
    return ordered_map(
        fn, units, workers, max_in_flight=RESULTS_HELD_PER_WORKER * workers, shared=plan
    )
