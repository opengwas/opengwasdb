"""How much of the host's memory a pass may hold, and how many workers fit.

Two passes are bounded by memory rather than by a worker count: a decoded band
of a Dense grid (`banding.py`, which bounds one band's cells) and the residual
`eaf` encode's pool (`build_vcf.py`, which bounds how many bands run at once).
The second needs the host's own figure, so reading it and dividing it belong
here rather than in the layout that happened to need them first: a second phase
that sizes a pool by memory must not re-derive "a quarter of available" on its
own, and the figure it must not substitute is the one that decides whether a
shared host survives the run.

The reading is deliberately strict. `/proc/meminfo` is a kernel format, not a
free-form file: a field that is not exactly a whole number of kilobytes is
`None`, so a caller runs one worker and says why, rather than dividing a number
nothing meant.
"""

from __future__ import annotations

from pathlib import Path

#: Where the kernel publishes the host's memory, in kilobytes.
MEMINFO = Path("/proc/meminfo")

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
