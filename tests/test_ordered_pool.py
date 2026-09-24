"""`ordered_map` yields in input order and bounds work in flight (#217)."""

from __future__ import annotations

import multiprocessing
import os

import pytest

from opengwasdb.build.ordered_pool import ordered_map

#: The first four in-flight items wait on this until the next four set it, so
#: completion order is the reverse of input order. A synchronisation primitive
#: rather than a fixed sleep: the ordering is a fact about the work, and the
#: test-hygiene gate forbids waiting on the clock.
_RELEASE = multiprocessing.get_context("fork").Event()


def _reverse_completion(i: int) -> tuple[int, int]:
    # Early items finish last, so a map that yielded in completion order fails
    # the ordering assertion below.
    if i < 4:
        _RELEASE.wait(timeout=30)
    elif i < 8:
        _RELEASE.set()
    return i, os.getpid()


def _tag(i: int) -> tuple[int, int]:
    return i, os.getpid()


def _fail_on_three(i: int) -> int:
    if i == 3:
        raise RuntimeError("column 3 failed")
    return i


def test_parallel_results_come_back_in_input_order() -> None:
    results = list(ordered_map(_reverse_completion, range(16), n_workers=8))
    assert [i for i, _ in results] == list(range(16))
    # Fixture is meaningful only if work really left the parent process.
    assert {pid for _, pid in results} - {os.getpid()}


def test_serial_path_runs_in_process() -> None:
    results = list(ordered_map(_tag, range(4), n_workers=1))
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
