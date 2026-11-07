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
picklable to cross the pool boundary; results are pickled back. A call that
also needs one value of its own -- a reference panel, a routing table -- passes
it as ``shared``, and the forked workers inherit it rather than receiving a
copy per item.

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
from contextlib import contextmanager
from functools import partial
from typing import Any, TypeVar, overload

T = TypeVar("T")
R = TypeVar("R")
S = TypeVar("S")

# The value a shared call hands its workers. Set in the parent before the pool
# forks and cleared when it closes, so a worker reads what the parent held.
_shared: Any = None
_sharing = False

#: Marks a call that carries no shared value, so that one which is deliberately
#: ``None`` is still shared rather than silently dropped.
_NO_SHARED: Any = object()


def fork_pool(n_workers: int) -> ProcessPoolExecutor:
    """A ``ProcessPoolExecutor`` pinned to the fork start method.

    Its workers inherit the parent's read-only state, numpy lookup arrays among
    them, without re-pickling it per task. Only correct on platforms with fork
    (Linux).
    """
    fork_ctx = multiprocessing.get_context("fork")
    return ProcessPoolExecutor(max_workers=n_workers, mp_context=fork_ctx)


@contextmanager
def _sharing_value(value: Any) -> Iterator[None]:
    """Hold ``value`` for the workers about to fork; refuse a nested share.

    A worker that opened its own pool would hand that pool's children a value
    whose owner is still open, so the second share is refused rather than
    guessed at. The value goes when the pool closes, whether or not the work
    succeeded.
    """
    global _shared, _sharing
    if _sharing:
        raise RuntimeError("a shared value is already with an open pool")
    _shared, _sharing = value, True
    try:
        yield
    finally:
        _shared, _sharing = None, False


@overload
def ordered_map(
    fn: Callable[[T], R],
    items: Iterable[T],
    n_workers: int,
    max_in_flight: int | None = None,
) -> Iterator[R]: ...


@overload
def ordered_map(
    fn: Callable[[S, T], R],
    items: Iterable[T],
    n_workers: int,
    max_in_flight: int | None = None,
    *,
    shared: S,
) -> Iterator[R]: ...


def ordered_map(
    fn: Callable[..., R],
    items: Iterable[T],
    n_workers: int,
    max_in_flight: int | None = None,
    *,
    shared: Any = _NO_SHARED,
) -> Iterator[R]:
    """Yield ``fn(item)`` for each item, in input order.

    With ``shared``, each call is ``fn(shared, item)`` instead, and the value
    reaches the forked workers by inheritance exactly as ``fork_each``'s does
    (issue #395): a reference panel or frequency table is not pickled once per
    item. Not re-entrant in that form -- a second shared call while one has a
    pool open is refused, as it is for ``fork_each``.

    At most ``max_in_flight`` items (default ``2 * n_workers``) are submitted
    but not yet yielded, which bounds the results the parent holds. An
    exception raised by ``fn`` propagates from the iteration at that item's
    position; remaining submitted work is cancelled.
    """
    if n_workers <= 1:
        if shared is _NO_SHARED:
            for item in items:
                yield fn(item)
        else:
            for item in items:
                yield fn(shared, item)
        return
    limit = max_in_flight if max_in_flight is not None else 2 * n_workers
    if limit < 1:
        raise ValueError(f"max_in_flight must be >= 1, got {limit}")
    if shared is _NO_SHARED:
        with fork_pool(n_workers) as pool:
            yield from _in_input_order(partial(pool.submit, fn), items, limit)
        return
    with _sharing_value(shared), fork_pool(n_workers) as pool:

        def _submit(item: T) -> Future[R]:
            return pool.submit(_call_shared, fn, item)

        yield from _in_input_order(_submit, items, limit)


def _in_input_order(
    submit: Callable[[T], Future[R]], items: Iterable[T], limit: int
) -> Iterator[R]:
    """Submit ``items`` through ``submit``, yielding results in submission order.

    At most ``limit`` results are held at once, and an exception propagates at
    its own item's position; whatever is still submitted is then cancelled.
    """
    pending: deque[Future[R]] = deque()
    try:
        for item in items:
            pending.append(submit(item))
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
    re-entrant: a second shared call while one has a pool open is refused.
    """
    if n_workers <= 1:
        for item in items:
            yield fn(shared, item)
        return
    with _sharing_value(shared), fork_pool(n_workers) as pool:
        futures = [pool.submit(_call_shared, fn, item) for item in items]
        for future in as_completed(futures):
            yield future.result()


def _call_shared(fn: Callable[[Any, T], R], item: T) -> R:
    """Run ``fn`` in a worker against the value its parent shared before forking."""
    return fn(_shared, item)
