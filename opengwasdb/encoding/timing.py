"""Wall-clock accounting and progress logging for a build's long phases."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager


def format_duration(seconds: float) -> str:
    """Compact `1h28m` / `3m05s` / `7s` rendering of a wall-clock interval."""
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


@contextmanager
def log_phase(logger: logging.Logger, label: str) -> Iterator[None]:
    """Log `label`'s start and its elapsed time on the way out, even on failure."""
    logger.info("%s: start", label)
    started = time.monotonic()
    try:
        yield
    finally:
        logger.info("%s: done in %s", label, format_duration(time.monotonic() - started))


def log_progress(
    logger: logging.Logger,
    label: str,
    completed: int,
    total: int,
    started: float,
    *,
    every: int,
    extra: str = "",
) -> None:
    """One progress line per `every` items, and always on the last.

    Idle-cheap for the caller: it returns before reading the clock unless this
    item is one to report, so a phase may call it after every row chunk.
    """
    if completed % every and completed != total:
        return
    elapsed = time.monotonic() - started
    eta = elapsed / completed * (total - completed) if completed else 0.0
    suffix = f" ({extra})" if extra else ""
    logger.info(
        "%s: %d/%d%s — elapsed %s, ETA %s",
        label, completed, total, suffix,
        format_duration(elapsed), format_duration(eta),
    )

class PhaseTimer:
    """Seconds accumulated per named phase, over a run that visits each many times.

    Format 3.0 added four distinct passes over a Dense `se` plane -- the
    per-Analysis fit, the candidate measurement, the rewrite, and the top-hit
    index rebuild -- and the only number anyone has is their sum: 3,863 s to
    migrate the FinnGen R13 pilot, which extrapolates to 62.5 hours for
    `ukb-b`. A total cannot say which pass to make cheaper (issue #144).

    Phases **partition** the work; they do not nest. A phase entered once per
    row chunk reports the sum of its visits, so `total()` is the accounted work
    and each phase's share of it is a real fraction. Wrapping one phase inside
    another would make both shares meaningless, which is why the callers time
    leaves only.
    """

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = self.seconds.get(name, 0.0) + (time.perf_counter() - started)

    def total(self) -> float:
        return sum(self.seconds.values())

    def report(self) -> list[tuple[str, float, float]]:
        """`(phase, seconds, share of the accounted total)`, most expensive first."""
        total = self.total()
        return [
            (name, seconds, seconds / total if total else 0.0)
            for name, seconds in sorted(self.seconds.items(), key=lambda item: -item[1])
        ]

    def format_report(self) -> str:
        lines = [
            f"  {name:<24} {seconds:9.1f}s  {share:5.1%}" for name, seconds, share in self.report()
        ]
        lines.append(f"  {'total accounted':<24} {self.total():9.1f}s")
        return "\n".join(lines)
