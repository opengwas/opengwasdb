"""Wait for a quiet-machine window before timing a benchmark column (#250).

Both the store-comparison harness and the extras runner gate each column on the
1-minute load average; the wait lives here so the two do not carry a copy.
"""

from __future__ import annotations

import os
import time


def wait_for_quiet(
    max_load: float, timeout_s: float = 3600.0, poll_s: float = 15.0
) -> tuple[float, bool]:
    """Block until the 1-minute load is below `max_load`; return (seconds, timed_out).

    A `max_load` of 0 disables the wait. A permanently busy node stops the wait
    at `timeout_s` and reports that it gave up, so a contended column is
    recorded rather than silently timed.
    """
    if max_load <= 0:
        return 0.0, False
    started = time.perf_counter()
    while os.getloadavg()[0] >= max_load:
        if time.perf_counter() - started > timeout_s:
            return time.perf_counter() - started, True
        time.sleep(poll_s)
    return time.perf_counter() - started, False
