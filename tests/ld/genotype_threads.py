"""A fresh process with a two-thread native default, for the public request lifetime test."""

from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from threadpoolctl import threadpool_info

import opengwasdb.ld
from opengwasdb.ld.genotypes import reader


def blas_threads() -> list[int]:
    return [pool["num_threads"] for pool in threadpool_info() if pool["user_api"] == "blas"]


def main(path: str) -> None:
    store = opengwasdb.ld.open(path)
    rows = store._reader.rows_in("22", 19000000, 21000000)[:10]
    names = store._reader.alids_of(rows).tolist()
    assert len(names) == 10
    assert blas_threads() == [2]
    original = reader.r
    fast_entered = threading.Event()
    slow_entered = threading.Event()
    fast_finished = threading.Event()
    seen: set[str] = set()

    def checked(rows, span):
        name = threading.current_thread().name
        if name not in seen:
            seen.add(name)
            if name.endswith("_0"):
                fast_entered.set()
                assert slow_entered.wait(2)
            else:
                slow_entered.set()
                assert fast_finished.wait(2)
                assert blas_threads() == [1]
        return original(rows, span)

    def fast():
        try:
            return store.matrix(names)
        finally:
            fast_finished.set()

    reader.r = checked
    try:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="request") as pool:
            first = pool.submit(fast)
            assert fast_entered.wait(2)
            second = pool.submit(store.matrix, names)
            np.testing.assert_array_equal(first.result().r, second.result().r)
        assert blas_threads() == [2], "restore the caller's default after the last request"
    finally:
        reader.r = original
        store._reader.close()


if __name__ == "__main__":
    main(sys.argv[1])
