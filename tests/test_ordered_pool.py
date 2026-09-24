"""`ordered_map` yields in input order and bounds work in flight (#217)."""

from __future__ import annotations

import multiprocessing
import os
from typing import Any

import pytest

from opengwasdb.build.ordered_pool import ordered_map


def _probe(i: int) -> tuple[int, int]:
    return i, os.getpid()


def _coordinated_probe(item: tuple[int, Any, Any]) -> tuple[int, int]:
    """Item 0 blocks until item 1 has recorded itself, so completion order is
    provably not input order -- by waiting on the condition, not the clock.
    """
    i, later_done, completion_order = item
    if i == 0:
        later_done.wait()
    completion_order.append(i)
    if i == 1:
        later_done.set()
    return i, os.getpid()


def _fail_on_three(i: int) -> int:
    if i == 3:
        raise RuntimeError("column 3 failed")
    return i


def test_parallel_results_come_back_in_input_order() -> None:
    manager = multiprocessing.Manager()
    later_done = manager.Event()
    completion_order = manager.list()
    results = list(
        ordered_map(
            _coordinated_probe,
            [(i, later_done, completion_order) for i in range(16)],
            n_workers=4,
        )
    )
    assert [i for i, _ in results] == list(range(16))
    # Fixture is meaningful only if work really left the parent process.
    assert {pid for _, pid in results} - {os.getpid()}
    # ... and only if completion really left input order while yielding did not.
    assert completion_order[0] != 0


def test_serial_path_runs_in_process() -> None:
    results = list(ordered_map(_probe, range(4), n_workers=1))
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
