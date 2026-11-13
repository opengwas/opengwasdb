"""The LD-block process pool Dense and Ragged completion share.

The LD blocks run on a plain process pool whose workers write their own
checkpoints (ADR 0023). The write after them runs its units -- a Dense row band,
a Ragged Analysis, a checkpoint to resolve -- on the ordered fork pool
(`ordered_map` with ``shared=``): each worker reads and computes its own unit,
the parent takes the results in unit order, so what the parent assembles from
them (side tables, counts, a CSR sequence) is the serial pass's whichever
worker finished first. A unit holds its cells in memory, so the write runs no
more workers than a quarter of the host's available memory holds beside the
parent (`encoding.memory.pool_workers`).
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import as_completed
from typing import Any

from opengwasdb.build.ordered_pool import single_threaded_fork_pool
from opengwasdb.encoding.timing import log_progress

log = logging.getLogger(__name__)


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
        # BLAS and OpenMP on one thread a worker, set before the workers fork
        # (`single_threaded_fork_pool` says why not in each worker): one block
        # a worker gives process-level parallelism that scales with n_workers.
        with single_threaded_fork_pool(n_workers) as pool:
            futures = [pool.submit(run_block, task) for task in pending]
            for i, fut in enumerate(as_completed(futures)):
                fut.result()  # propagate worker errors; result is on disk
                log_progress(log, label, i + 1, total, started, every=every)
