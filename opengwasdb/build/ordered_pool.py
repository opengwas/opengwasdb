"""Ordered, bounded parallel map for per-column build phases (#217).

The post-Pass-2 consolidation walks independent columns (or zarr row chunks)
whose results must be combined in input order: CSR offsets align with
``analysis_index``, overflow tables and top-hit harvests are ordered. A plain
``as_completed`` loop would reorder them silently, and ``pool.map`` over a list
submits every item at once and holds every result. ``ordered_map`` does
neither: it keeps at most ``max_in_flight`` items submitted, and yields results
strictly in input order.

Workers are forked, so large read-only state the parent set up before the call
(numpy arrays, not Python dicts, which fork copies page by page as reference
counts change) is shared without pickling. ``fn`` and each item must still be
picklable to cross the pool boundary; results are pickled back.

With ``n_workers <= 1`` no pool is created and ``fn`` runs in the calling
process, which is the serial path every caller keeps.

``fork_each`` is the per-Analysis form Pass 2 uses: results in completion
order, and one ``shared`` value -- a routing table, a spill directory --
handed to every call. Forked workers inherit it, so it is never pickled.
"""

from __future__ import annotations

import multiprocessing
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from typing import Any, TypeVar

T = TypeVar("T")
R = TypeVar("R")
S = TypeVar("S")

# The value `fork_each` hands its workers. Set in the parent before the pool
# forks and cleared when it closes, so a worker reads what the parent held.
_shared: Any = None
_sharing = False


def fork_pool(n_workers: int) -> ProcessPoolExecutor:
    """A ``ProcessPoolExecutor`` pinned to the fork start method.

    Its workers inherit the parent's read-only state, numpy lookup arrays among
    them, without re-pickling it per task. Only correct on platforms with fork
    (Linux).
    """
    fork_ctx = multiprocessing.get_context("fork")
    return ProcessPoolExecutor(max_workers=n_workers, mp_context=fork_ctx)


def ordered_map(
    fn: Callable[[T], R],
    items: Iterable[T],
    n_workers: int,
    max_in_flight: int | None = None,
) -> Iterator[R]:
    """Yield ``fn(item)`` for each item, in input order.

    At most ``max_in_flight`` items (default ``2 * n_workers``) are submitted
    but not yet yielded, which bounds the results the parent holds. An
    exception raised by ``fn`` propagates from the iteration at that item's
    position; remaining submitted work is cancelled.
    """
    if n_workers <= 1:
        for item in items:
            yield fn(item)
        return
    limit = max_in_flight if max_in_flight is not None else 2 * n_workers
    if limit < 1:
        raise ValueError(f"max_in_flight must be >= 1, got {limit}")
    pending: deque[Future[R]] = deque()
    with fork_pool(n_workers) as pool:
        try:
            for item in items:
                pending.append(pool.submit(fn, item))
                if len(pending) >= limit:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()
        finally:
            for future in pending:
                future.cancel()


def fork_each(
    fn: Callable[[S, T], R], shared: S, items: Iterable[T], n_workers: int
) -> Iterator[R]:
    """Yield ``fn(shared, item)`` for each item, in completion order.

    Every item is submitted at once, so this suits a modest number of large
    tasks -- one per Analysis -- whose results are small. ``shared`` reaches
    each call without pickling: a forked worker inherits it, numpy arrays and
    all. With ``n_workers <= 1`` the calls run here, in input order. Not
    re-entrant: a second ``fork_each`` while one has a pool open is refused.
    """
    if n_workers <= 1:
        for item in items:
            yield fn(shared, item)
        return
    global _shared, _sharing
    if _sharing:
        raise RuntimeError("fork_each is already sharing a value with a pool")
    _shared, _sharing = shared, True
    try:
        with fork_pool(n_workers) as pool:
            futures = [pool.submit(_call_shared, fn, item) for item in items]
            for future in as_completed(futures):
                yield future.result()
    finally:
        _shared, _sharing = None, False


def _call_shared(fn: Callable[[Any, T], R], item: T) -> R:
    """Run ``fn`` in a worker against the value its parent shared before forking."""
    return fn(_shared, item)
