"""The segment-order evidence runner must fail loudly (#252 review round 3).

`benchmarks/check_store_segment_order.py` publishes
`docs/benchmark-output/opengwasdb_252_segment_order.json`. Round 3 of #252's
review found that it could report an offset-implied `entries_read` after the
rule read nothing on a length mismatch, and report success for an empty or
wrong root. These tests pin the fail-loud guards.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks import check_store_segment_order as runner
from opengwasdb.store.arrays import ArrayRole, create_array, open_group_for_write


def _write_group(tmp_path, offsets: np.ndarray, variant_index: np.ndarray):
    """A bare Ragged group at `OGS-9999/store.opengwasdb` with the two arrays."""
    store = tmp_path / "OGS-9999" / "store.opengwasdb"
    root = open_group_for_write(store / "data.zarr" / "ragged", "w")
    create_array(
        root, "offsets", ArrayRole.ASSOCIATION_OFFSETS,
        data=offsets, dtype="int64", overwrite=True,
    )
    create_array(
        root, "variant_index", ArrayRole.ASSOCIATION_SEQUENCE,
        data=variant_index, dtype="int32", overwrite=True,
    )
    return store


def _malformed_store(tmp_path):
    """A Ragged group whose `offsets` imply three entries but hold none."""
    return _write_group(
        tmp_path, np.array([0, 3], dtype=np.int64), np.zeros(0, dtype=np.int32)
    )


def test_a_length_offset_mismatch_fails_before_the_rule(tmp_path) -> None:
    record = runner._check_one(_malformed_store(tmp_path))
    assert not record["ok"], "a length/offset mismatch must fail the record"
    assert record["entries_expected"] == 3
    assert record["entries_checked"] == 0, "the rule read nothing, so nothing was checked"
    assert any("offsets imply 3" in error for error in record["errors"])


def test_a_valid_segment_is_counted(tmp_path) -> None:
    store = _write_group(
        tmp_path, np.array([0, 3], dtype=np.int64), np.array([1, 2, 3], dtype=np.int32)
    )
    record = runner._check_one(store)
    assert record["ok"], record["errors"]
    assert record["entries_expected"] == 3
    assert record["entries_checked"] == 3


def test_an_empty_root_is_refused(tmp_path) -> None:
    with pytest.raises(SystemExit):
        runner._discover(tmp_path)


def test_an_ogs_directory_without_a_store_is_refused(tmp_path) -> None:
    (tmp_path / "OGS-9999").mkdir()
    with pytest.raises(SystemExit):
        runner._discover(tmp_path)
