"""`ordered_map` yields in input order and bounds work in flight (#217)."""

from __future__ import annotations

import multiprocessing
import os

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
