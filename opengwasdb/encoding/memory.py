"""How much of the host's memory a pass may hold, and how many workers fit.

Several passes are bounded by memory rather than by a worker count: a decoded
band of a Dense grid (`banding.py`, which bounds one band's cells), and every
pool that forks workers over units each holding a share of a release's cells --
the Dense build's residual `eaf` encode (#320), the Top-Hit Index scan (#313),
validation's band pass (#315) and Reference Completion's write pools (#321).
Each of those pools is sized by `pool_workers`, one rule: a quarter of the
host's `MemAvailable`, less what the parent holds, divided by what one worker
holds -- the fork's floor (`FORK_FLOOR_BYTES`), its unit, and the results the
parent keeps for it (`RESULTS_HELD_PER_WORKER`). A phase that sizes a pool by
memory must not re-derive "a quarter of available" on its own, and the figure
it must not substitute is the one that decides whether a shared host survives
the run.

The reading is deliberately strict. `/proc/meminfo` is a kernel format, not a
free-form file: a field that is not exactly a whole number of kilobytes is
`None`, so a caller runs one worker and says why, rather than dividing a number
nothing meant.

A worker that lives for a whole pass also gives back what it has freed between
units (`release_freed_heap`), so that what it holds is one unit's working set
rather than the high-water of every unit it has worked (#451).
"""

from __future__ import annotations

import ctypes
import functools
import logging
from collections.abc import Callable
from pathlib import Path

from opengwasdb.build.ordered_pool import RESULTS_HELD_PER_WORKER

log = logging.getLogger(__name__)

#: Where the kernel publishes the host's memory, in kilobytes.
MEMINFO = Path("/proc/meminfo")
#: Where it publishes this process's own, in kilobytes.
SELF_STATUS = Path("/proc/self/status")

#: The share of the host's available memory one phase may plan to use.
MEMORY_BUDGET_DIVISOR = 4


def available_memory_bytes(path: Path = MEMINFO) -> int | None:
    """The host's `MemAvailable` in bytes, or `None` when it cannot be read.

    `MemAvailable` is the kernel's estimate of what a new job can take without
    swapping, which is why it is the field to plan against and `MemFree` is not.
    """
    try:
        content = path.read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError):
        return None
    for line in content.splitlines():
        key, _, rest = line.partition(":")
        if key.strip() != "MemAvailable":
            continue
        fields = rest.split()
        if len(fields) != 2 or fields[1] != "kB" or not fields[0].isdigit():
            return None
        return int(fields[0]) * 1024
    return None


def memory_budget_bytes(available_bytes: int | None) -> int | None:
    """A phase's budget: a quarter of what the host has available, or unknown."""
    return None if available_bytes is None else available_bytes // MEMORY_BUDGET_DIVISOR


def workers_within_memory(
    requested: int, per_worker_bytes: int, budget_bytes: int | None
) -> int:
    """How many workers fit in `budget_bytes`, one a worker's working set.

    Never more than `requested`, and never zero: a budget too small for even one
    worker's working set still runs one, because refusing to do the work at all
    is not a memory policy. An unknown budget is also one worker -- a pool sized
    from a guess is the thing the budget exists to prevent.
    """
    if budget_bytes is None:
        return 1
    return max(1, min(requested, budget_bytes // max(per_worker_bytes, 1)))


#: What a forked worker holds before its first unit: its interpreter, the
#: pages of the parent's it writes to, a read's compressed chunks and a codec's
#: intermediates. A validation worker held 0.26-0.27 GiB of its own between
#: bands on OGS-00011's Dense Component and on OGS-00009 (#315), and the eaf
#: encode's band fit put a worker's fixed part at 135 MiB (#320), so this is
#: about twice the larger. `pool_workers` adds it to every worker's unit.
FORK_FLOOR_BYTES = 512 * 1024**2

#: The share of a unit's result an exception table may take, in cells per
#: byte: one byte for every 40 cells of the unit. An entry is 12 bytes, pickled
#: and unpickled, so this is room for one cell in a thousand to be an
#: exception; OGS-00009's completed release holds 6e-5 of its cells as SE
#: exceptions and its build 3.5e-7 as EAF ones (#320, #321).
EXCEPTION_CELLS_PER_RESULT_BYTE = 40


def worker_bytes(unit_bytes: int) -> int:
    """What one forked worker holds while it works a unit of ``unit_bytes``."""
    return FORK_FLOOR_BYTES + unit_bytes


def resident_bytes(path: Path = SELF_STATUS) -> int:
    """This process's resident set (`VmRSS`) in bytes.

    What a pool's parent already holds when the pool forks. Linux always
    publishes it, so a status file that does not give it as a whole number of
    kilobytes is refused rather than counted as nothing held: a parent counted
    at zero sizes its pool for memory the parent is already using (#321).
    """
    try:
        content = path.read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            f"cannot read this process's resident set from {path}: {exc}"
        ) from exc
    for line in content.splitlines():
        key, _, rest = line.partition(":")
        if key.strip() != "VmRSS":
            continue
        fields = rest.split()
        if len(fields) == 2 and fields[1] == "kB" and fields[0].isdigit():
            return int(fields[0]) * 1024
        raise RuntimeError(
            f"cannot read this process's resident set from {path}: VmRSS is {rest.strip()!r}, "
            "not a whole number of kB"
        )
    raise RuntimeError(f"cannot read this process's resident set from {path}: it has no VmRSS")


def pool_workers(
    requested: int,
    unit_bytes: int,
    *,
    what: str,
    unit: str,
    units: int | None,
    result_bytes: int,
    parent_bytes: int | None = None,
) -> int:
    """How many of ``requested`` workers a pool runs, and the lines saying why.

    Never more than its ``units``, when the pool can count them: a worker with
    no unit to work is still forked. A worker holds the fork's floor and
    ``unit_bytes`` while it works one ``unit`` (`worker_bytes`), and the parent holds up to
    `RESULTS_HELD_PER_WORKER` of its results, ``result_bytes`` each, until it
    takes them in order (`ordered_map`'s window). The workers, those results and
    the parent together stay within a phase's budget, a quarter of `MemAvailable`
    read when the pool starts. The parent's own share is ``parent_bytes``, its
    expected peak, or what it holds now (`resident_bytes`). That resident set
    is counted although `MemAvailable` already leaves it out: the double count
    is a margin kept on purpose.

    An unreadable host gets one worker rather than a guess, and so does a budget
    the parent and one worker overrun -- silently at one requested worker, where
    the reading changes nothing, and with a warning otherwise. The count is
    logged with the figures it came from.
    """
    requested = requested if units is None else min(requested, max(units, 1))
    if requested <= 1:
        return 1
    parent = resident_bytes() if parent_bytes is None else parent_bytes
    per_worker = worker_bytes(unit_bytes) + RESULTS_HELD_PER_WORKER * result_bytes
    budget = memory_budget_bytes(available_memory_bytes())
    if budget is None:
        log.warning(
            "%s: cannot read %s, so no memory budget can be set under %d worker(s); "
            "running on one worker",
            what, MEMINFO, requested,
        )
        workers = 1
    else:
        room = budget - parent
        if room < per_worker:
            log.warning(
                "%s: a %s needs %d MiB a worker beside the parent's %d MiB, and the budget "
                "is %d MiB; running on one worker, which will overrun it",
                what, unit, per_worker >> 20, parent >> 20, budget >> 20,
            )
        workers = workers_within_memory(requested, per_worker, max(room, 0))
    log.info(
        "%s: %d of %d worker(s); a %s is %d MiB a worker and %d MiB of results held by "
        "the parent, beside the parent's %d MiB, against a budget of %s",
        what, workers, requested, unit, worker_bytes(unit_bytes) >> 20,
        (RESULTS_HELD_PER_WORKER * result_bytes) >> 20, parent >> 20,
        "unknown" if budget is None else f"{budget >> 20} MiB",
    )
    return workers


@functools.cache
def _malloc_trim() -> Callable[[int], int] | None:
    """glibc's `malloc_trim`, or `None`, said once, where the C library has none."""
    try:
        trim: Callable[[int], int] = ctypes.CDLL(None).malloc_trim
    except AttributeError:
        log.warning(
            "this C library has no malloc_trim, so a long-lived worker keeps the memory "
            "it has freed (#451)"
        )
        return None
    return trim


def release_freed_heap() -> None:
    """Hand the pages the C heap holds free back to the host.

    A long-lived worker otherwise keeps the high-water of every unit it has
    worked: Python frees a source's buffers, but glibc keeps their pages for
    reuse, and a freed page under a live allocation is never returned (#451
    has the measurements). Only glibc offers `malloc_trim`; on any other C
    library nothing is released, and the first call says so.
    """
    trim = _malloc_trim()
    if trim is not None:
        trim(0)
