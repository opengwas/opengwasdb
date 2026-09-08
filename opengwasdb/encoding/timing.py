"""Wall-clock accounting for the passes an encoding decision makes over a plane."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


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
