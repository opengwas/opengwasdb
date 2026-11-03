"""The read counter must tell one read of a sharded plane from two.

Validation holds every plane to one read per component; the counter that checks
it has to see a second read of a 0.2.0 plane, whose inner chunks share one shard
file and so one store key.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from tests.store_reads import element_reads, times_read


def _sharded_plane(path: Path) -> zarr.Array:
    group = zarr.open_group(path, mode="w", zarr_format=3)
    plane = group.create_array("z", shape=(40, 4), chunks=(10, 4), shards=(20, 4), dtype="i2")
    plane[:] = np.arange(160, dtype=np.int16).reshape(40, 4)
    return zarr.open_group(path, mode="r")["z"]


def test_one_read_of_a_sharded_plane_counts_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plane = _sharded_plane(tmp_path)
    with element_reads(monkeypatch) as counts:
        for start in range(0, 40, 10):
            np.asarray(plane[start : start + 10])
    assert times_read(counts, plane) == 1.0


def test_a_second_read_through_another_entry_point_is_seen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plane = _sharded_plane(tmp_path)
    with element_reads(monkeypatch) as counts:
        np.asarray(plane[:])
        np.asarray(plane.oindex[np.arange(0, 40, 2), :])
        np.asarray(plane.vindex[np.arange(20), np.zeros(20, dtype=int)])
    assert times_read(counts, plane) == 1.0 + 0.5 + 20 / 160
