"""Shared helpers for tests that kill an in-place Store-array swap mid-rename.

`opengwasdb.store.swap` owns the two-rename swap both `repair` and `reencode`
use.  The repair and re-encode interruption suites each need the same way to
kill a run between the two renames, so the helper lives here rather than in one
suite's private copy.
"""

from __future__ import annotations

from typing import Any

import pytest

from opengwasdb.store import swap


class ProcessDeath(BaseException):
    """Stands in for a process dying: `except Exception` rollbacks do not see it."""


def die_on_second_move(monkeypatch: pytest.MonkeyPatch) -> None:
    """Kill the run before the second rename, i.e. between the two renames."""
    calls: list[tuple[str, str]] = []
    real = swap.move_in_group

    def move(group: Any, source: str, dest: str) -> None:
        calls.append((source, dest))
        if len(calls) == 2:
            raise ProcessDeath(f"killed before moving {source} -> {dest}")
        real(group, source, dest)

    monkeypatch.setattr(swap, "move_in_group", move)
