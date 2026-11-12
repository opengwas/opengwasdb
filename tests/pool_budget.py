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

import opengwasdb.completion.fill_shards as fill_shards
import opengwasdb.layouts.dense.complete as dense_complete
import opengwasdb.layouts.ragged.complete as ragged_complete
import opengwasdb.layouts.ragged.zarr_csr as zarr_csr

#: A fixture's unit, whatever the pool: far above what its few cells hold.
FIXTURE_UNIT_BYTES = 16 * 1024**2


def pin_pool_units(patch: pytest.MonkeyPatch) -> None:
    """Every completion write pool's per-unit fixed part, at a fixture's size."""
    for module, name in (
        (dense_complete, "_BAND_FIXED_BYTES"),
        (fill_shards, "_RESOLVE_FIXED_BYTES"),
        (ragged_complete, "_ASSEMBLY_FIXED_BYTES"),
        (zarr_csr, "_REGION_FIXED_BYTES"),
    ):
        assert hasattr(module, name), f"{module.__name__} has no {name} to pin"
        patch.setattr(module, name, FIXTURE_UNIT_BYTES)
