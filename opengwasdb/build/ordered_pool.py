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
"""

from __future__ import annotations

import multiprocessing
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


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
    fork_ctx = multiprocessing.get_context("fork")
    pending: deque[Future[R]] = deque()
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=fork_ctx) as pool:
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
