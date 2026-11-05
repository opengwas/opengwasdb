"""Record the order forked workers finish in (#395).

``fork_each`` yields in completion order, and a pipeline whose output is
input-ordered -- the Analysis Catalogue, the phenotype-SD TSV -- has to put
results back by index. A test that only compares one run's output with
another's can pass whether or not it does: a small fixture's workers finish in
the order they were submitted, so completion order and input order coincide.
So a test makes one worker finish last on purpose, records what each worker
claimed as it finished, and asserts both that the order really differed from
the input and that the output did not follow it.

The counter and array are made with the fork start method, so a worker inherits
them exactly as it inherits the pipeline's own pool state.
"""

from __future__ import annotations

import multiprocessing
import os
import time


class CompletionOrder:
    """Which index finished when, across one fork pool's workers."""

    #: How long a deliberately-slow worker waits for the others before giving
    #: up, so a pool that never ran them at the same time fails rather than
    #: hanging the suite.
    WAIT_SECONDS = 30.0

    def __init__(self, n_items: int) -> None:
        context = multiprocessing.get_context("fork")
        self._claimed = context.Value("i", 0)
        self._order = context.Array("i", n_items)
        self._n_items = n_items

    def reset(self) -> None:
        """Forget a previous run, before the same pool is measured again."""
        with self._claimed.get_lock():
            self._claimed.value = 0
            for position in range(self._n_items):
                self._order[position] = 0

    def wait_for_the_rest(self) -> None:
        """Block until every other worker has finished, so this one finishes last."""
        deadline = time.monotonic() + self.WAIT_SECONDS
        while self._claimed.value < self._n_items - 1:
            if time.monotonic() > deadline:
                raise AssertionError(
                    f"{self._claimed.value} of {self._n_items} workers finished within "
                    f"{self.WAIT_SECONDS}s; the pool did not run them in parallel"
                )
            os.sched_yield()

    def claim(self, index: int) -> None:
        """Record that the worker for ``index`` has finished."""
        with self._claimed.get_lock():
            self._order[self._claimed.value] = index
            self._claimed.value += 1

    def order(self) -> list[int]:
        """The finishing indices, first to last."""
        return list(self._order)
