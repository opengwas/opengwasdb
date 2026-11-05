"""`ordered_map` yields in input order and bounds work in flight (#217)."""

from __future__ import annotations

import multiprocessing
import os
from typing import Any

import pytest

from opengwasdb.build.ordered_pool import ordered_map

_ITEMS = 16
# Item 0 waits for every other item to finish before returning, and each item
# logs its completion position here. A shared counter rather than a sleep makes
# that ordering a condition, not a race with the clock; a map that yields in
# completion order therefore fails the order assertion below.
_fork = multiprocessing.get_context("fork")
_completed = _fork.Value("i", 0)
_completion_order = _fork.Array("i", _ITEMS)


def _reverse_completion(i: int) -> tuple[int, int]:
    if i == 0:
        while True:
            with _completed.get_lock():
                if _completed.value >= _ITEMS - 1:
                    break
            os.sched_yield()
    with _completed.get_lock():
        position = _completed.value
        _completed.value += 1
    _completion_order[position] = i
    return i, os.getpid()


def _identity(i: int) -> tuple[int, int]:
    return i, os.getpid()


def _reset_completion() -> None:
    with _completed.get_lock():
        _completed.value = 0


def _fail_on_three(i: int) -> int:
    if i == 3:
        raise RuntimeError("column 3 failed")
    return i


def test_parallel_results_come_back_in_input_order() -> None:
    _reset_completion()
    # max_in_flight covers the whole input so item 0 can block on later items
    # that would otherwise not be submitted until it yielded.
    results = list(
        ordered_map(_reverse_completion, range(_ITEMS), n_workers=4, max_in_flight=_ITEMS)
    )
    assert [i for i, _ in results] == list(range(_ITEMS))
    # Fixture is meaningful only if work really left the parent process...
    assert {pid for _, pid in results} - {os.getpid()}
    # ... and if completion order really differed from input order: item 0,
    # which input order demands is returned first, completed last.
    assert list(_completion_order)[-1] == 0


def test_serial_path_runs_in_process() -> None:
    results = list(ordered_map(_identity, range(4), n_workers=1))
    assert [i for i, _ in results] == [0, 1, 2, 3]
    assert {pid for _, pid in results} == {os.getpid()}


def test_items_are_pulled_lazily_with_bounded_in_flight() -> None:
    pulled: list[int] = []

    def source():
        for i in range(20):
            pulled.append(i)
            yield i

    it = ordered_map(abs, source(), n_workers=2, max_in_flight=3)
    first = next(it)
    assert first == 0
    assert len(pulled) <= 3
    assert list(it) == list(range(1, 20))


@pytest.mark.parametrize("n_workers", [1, 3])
def test_worker_error_propagates(n_workers: int) -> None:
    with pytest.raises(RuntimeError, match="column 3 failed"):
        list(ordered_map(_fail_on_three, range(10), n_workers=n_workers))


# ── fork_each: one shared value, results in completion order ─────────────────


def _read_shared(shared: dict[str, Any], item: int) -> tuple[int, int, int]:
    """The item, the shared value's tag, and the worker's process id."""
    return item, shared["tag"], os.getpid()


def test_fork_each_hands_every_worker_the_shared_value_unpickled() -> None:
    """The shared value reaches forked workers by inheritance: a lock cannot be
    pickled, so a value passed through the pool's pipe would fail to arrive."""
    import threading

    from opengwasdb.build.ordered_pool import fork_each

    shared = {"tag": 7, "lock": threading.Lock()}
    results = sorted(fork_each(_read_shared, shared, range(8), n_workers=3))

    assert [item for item, _tag, _pid in results] == list(range(8))
    assert {tag for _item, tag, _pid in results} == {7}
    assert os.getpid() not in {pid for _item, _tag, pid in results}


def test_fork_each_runs_in_order_in_process_when_serial() -> None:
    from opengwasdb.build.ordered_pool import fork_each

    results = list(fork_each(_read_shared, {"tag": 1}, [3, 1, 2], n_workers=1))

    assert results == [(3, 1, os.getpid()), (1, 1, os.getpid()), (2, 1, os.getpid())]


def _nested(shared: dict[str, Any], item: int) -> int:
    from opengwasdb.build.ordered_pool import fork_each

    return sum(fork_each(_read_shared_item, shared, [item], n_workers=2))


def _read_shared_item(shared: dict[str, Any], item: int) -> int:
    return item


def test_fork_each_refuses_a_second_share_while_one_is_open_and_clears_after() -> None:
    """A worker that started its own pool would hand its children a value its
    parent's pool still shares; that is refused, and the value goes when the
    pool closes, whether or not the work succeeded."""
    from opengwasdb.build import ordered_pool

    with pytest.raises(RuntimeError, match="already sharing"):
        list(ordered_pool.fork_each(_nested, {"tag": 0}, [1], n_workers=2))
    assert ordered_pool._shared is None
    assert not ordered_pool._sharing
