"""`ordered_map` yields in input order and bounds work in flight (#217)."""

from __future__ import annotations

import multiprocessing
import os
from typing import Any

import pytest

from opengwasdb.build.ordered_pool import ordered_map

#: Per-item events a probe sets, so an early item can wait for a later one
#: without a clock: item 0 does not finish until item 1 has run, which makes
#: completion order the reverse of input order for the first pair.
_EVENTS: dict[int, Any] = {}


def _reordered_probe(i: int) -> tuple[int, int]:
    _EVENTS[i].set()
    if i == 0 and not _EVENTS[1].wait(timeout=30):
        raise RuntimeError("a later item never ran")
    return i, os.getpid()


def _identity_probe(i: int) -> tuple[int, int]:
    return i, os.getpid()


def _fail_on_three(i: int) -> int:
    if i == 3:
        raise RuntimeError("column 3 failed")
    return i


def _fork_events(n: int) -> None:
    global _EVENTS
    ctx = multiprocessing.get_context("fork")
    _EVENTS = {i: ctx.Event() for i in range(n)}


def test_parallel_results_come_back_in_input_order() -> None:
    _fork_events(16)
    results = list(ordered_map(_reordered_probe, range(16), n_workers=4))
    assert [i for i, _ in results] == list(range(16))
    # Fixture is meaningful only if work really left the parent process.
    assert {pid for _, pid in results} - {os.getpid()}


def test_serial_path_runs_in_process() -> None:
    results = list(ordered_map(_identity_probe, range(4), n_workers=1))
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
