"""How much of the host's memory a pass may hold, and how many workers fit.

Several passes are bounded by memory rather than by a worker count: a decoded
band of a Dense grid (`banding.py`, which bounds one band's cells), the
residual `eaf` encode's pool (`build_vcf.py`, which bounds how many bands run
at once), and Reference Completion's write pools (`pool_workers`, #321). The
pools need the host's own figure, so reading it and dividing it belong here
rather than in the layout that happened to need them first: a phase that sizes
a pool by memory must not re-derive "a quarter of available" on its own, and
the figure it must not substitute is the one that decides whether a shared
host survives the run.

The reading is deliberately strict. `/proc/meminfo` is a kernel format, not a
free-form file: a field that is not exactly a whole number of kilobytes is
`None`, so a caller runs one worker and says why, rather than dividing a number
nothing meant.
"""

from __future__ import annotations

import logging
from pathlib import Path

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


def resident_bytes(path: Path = SELF_STATUS) -> int:
    """This process's resident set (`VmRSS`) in bytes; 0 when it cannot be read.

    What a pool's parent already holds when the pool forks. An unreadable
    figure counts as nothing held rather than refusing the pool: the host's
    own reading, which `pool_workers` does refuse to guess, still bounds it.
    """
    try:
        content = path.read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError):
        return 0
    for line in content.splitlines():
        key, _, rest = line.partition(":")
        if key.strip() != "VmRSS":
            continue
        fields = rest.split()
        if len(fields) == 2 and fields[1] == "kB" and fields[0].isdigit():
            return int(fields[0]) * 1024
    return 0


def pool_workers(
    requested: int,
    unit_bytes: int,
    *,
    what: str,
    unit: str,
    parent_bytes: int | None = None,
) -> int:
    """How many of ``requested`` workers a pool runs, and the lines saying why.

    A worker holds ``unit_bytes`` while it works one ``unit``. The workers and
    the parent together stay within a phase's budget, a quarter of
    `MemAvailable` read when the pool starts: the parent's share is
    ``parent_bytes``, its expected peak, or what it holds now (`resident_bytes`).
    An unreadable host gets one worker rather than a guess, and so does a budget
    the parent and one worker overrun -- silently at one requested worker, where
    the reading changes nothing, and with a warning otherwise. The count is
    logged with the figures it came from.
    """
    if requested <= 1:
        return 1
    parent = resident_bytes() if parent_bytes is None else parent_bytes
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
        if room < unit_bytes:
            log.warning(
                "%s: a %s needs %d MiB a worker beside the parent's %d MiB, and the budget "
                "is %d MiB; running on one worker, which will overrun it",
                what, unit, unit_bytes >> 20, parent >> 20, budget >> 20,
            )
        workers = workers_within_memory(requested, unit_bytes, max(room, 0))
    log.info(
        "%s: %d of %d worker(s); a %s is %d MiB a worker beside the parent's %d MiB, "
        "against a budget of %s",
        what, workers, requested, unit, unit_bytes >> 20, parent >> 20,
        "unknown" if budget is None else f"{budget >> 20} MiB",
    )
    return workers
