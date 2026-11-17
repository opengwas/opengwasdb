"""Size Reference Completion's write pools for a test's tiny units (#321).

Each pool is capped by a quarter of the host's available memory, one unit's
working set a worker, beside the parent. The production estimates are sized
for OGS-00009's bands (a Dense band alone budgets 4 GiB before its cells), so a
test that needs its units to run side by side on a small host -- a CI runner
with 16 GB -- pins each unit's fixed part to what a fixture's unit can hold.
The host's own reading, and the parent's, still decide.
"""

from __future__ import annotations

import pytest

import opengwasdb.encoding.memory as memory
import opengwasdb.layouts.dense.complete as dense_complete

#: A fixture's unit, whatever the pool: far above what its few cells hold.
FIXTURE_UNIT_BYTES = 16 * 1024**2


def pin_fork_floor(patch: pytest.MonkeyPatch) -> None:
    """A forked worker's floor, which every pool adds to its unit, at a fixture's size."""
    assert hasattr(memory, "FORK_FLOOR_BYTES"), "memory has no FORK_FLOOR_BYTES to pin"
    patch.setattr(memory, "FORK_FLOOR_BYTES", FIXTURE_UNIT_BYTES)


def pin_pool_units(patch: pytest.MonkeyPatch) -> None:
    """Every completion write pool's fixed part, at a fixture's size: the fork's
    floor they share, and the Dense band's step on top of it."""
    pin_fork_floor(patch)
    assert hasattr(dense_complete, "_BAND_FIXED_BYTES"), "no _BAND_FIXED_BYTES to pin"
    patch.setattr(dense_complete, "_BAND_FIXED_BYTES", FIXTURE_UNIT_BYTES)
