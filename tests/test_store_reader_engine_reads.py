"""The named reads the Store Reader gained for the engine's readers (#473).

The validator's source-fidelity join reads each row's `source_alid`, the EAF
audit and the variant-table rule stream the table, and completion reads the side
arrays its source was coded against. Each is a Store Reader read now; these pin
what each returns, including what it says when the release cannot answer.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import zarr

from opengwasdb.store_reader.release import release_arrays
from opengwasdb.store_reader.stored import (
    eaf_baseline,
    se_coefficients,
    sequence_arrays,
    stored_attributes,
)
from opengwasdb.store_reader.top_hits import analysis_hit_counts
from opengwasdb.store_reader.variant_axis import VariantAxis


class _Table:
    """A `LegacyVariantTable` over rows given as text, counting the rows read."""

    def __init__(self, rows: list[str]) -> None:
        self._rows = rows
        self.read = 0

    def rows(self) -> Iterator[str]:
        for row in self._rows:
            self.read += 1
            yield row

    def fetch(self, chromosome: str, start: int, end: int) -> list[str]:
        raise AssertionError("a whole-table read fetches no range")

    def line_at(self, offset: int) -> str:
        raise AssertionError("a whole-table read seeks no row")

    def alias_index(self, alias: str) -> int | None:
        raise AssertionError("a whole-table read looks up no alias")

    def close(self) -> None:
        pass


def _axis(tmp_path: Path, rows: list[str]) -> tuple[VariantAxis, _Table]:
    np.save(tmp_path / "variant_offsets.npy", np.arange(len(rows), dtype=np.int64))
    table = _Table(rows)
    return VariantAxis(tmp_path, table), table


def _row(index: int, source_alid: str | None) -> str:
    alid = f"1:{1000 + index}:A:G"
    fields = ["1", str(1000 + index), str(index), "A", "G", alid, "."]
    if source_alid is not None:
        fields.append(source_alid)
    return "\t".join(fields) + "\n"


def test_source_alids_resolve_to_their_rows_and_the_scan_stops_once_all_are_found(
    tmp_path: Path,
):
    rows = [_row(i, f"1:{500 + i}:A:G") for i in range(6)]
    axis, table = _axis(tmp_path, rows)
    found = axis.indices_by_source_alid({"1:501:A:G", "1:503:A:G", "1:999:A:G"})
    assert found == {"1:501:A:G": 1, "1:503:A:G": 3}
    assert table.read == 6  # an ALID the table lacks is looked for to the end

    axis, table = _axis(tmp_path, rows)
    assert axis.indices_by_source_alid({"1:501:A:G", "1:503:A:G"}) == {
        "1:501:A:G": 1,
        "1:503:A:G": 3,
    }
    assert table.read == 4


def test_a_blank_source_alid_matches_nothing(tmp_path: Path):
    axis, _ = _axis(tmp_path, [_row(0, "."), _row(1, "1:7:C:T")])
    assert axis.indices_by_source_alid({"1:7:C:T"}) == {"1:7:C:T": 1}
    # `.` is how the table spells a row with no source ALID, not an ALID to match.
    assert axis.indices_by_source_alid({"."}) == {}
    assert axis.indices_by_source_alid({".", "1:7:C:T"}) == {"1:7:C:T": 1}
    assert axis.indices_by_source_alid({"1:8:C:T"}) == {}


def test_a_table_without_the_source_alid_column_says_so_rather_than_matching_nothing(
    tmp_path: Path,
):
    axis, _ = _axis(tmp_path, [_row(0, None), _row(1, None)])
    assert axis.indices_by_source_alid({"1:1000:A:G"}) is None


def test_records_stream_what_all_materialises(tmp_path: Path):
    rows = [_row(i, None if i % 2 else f"1:{i}:A:G") for i in range(5)]
    axis, table = _axis(tmp_path, rows)
    stream = axis.records()
    assert table.read == 0  # nothing read until the stream is
    first = next(stream)
    assert (first.variant_index, table.read) == (0, 1)
    assert [first, *stream] == _axis(tmp_path, rows)[0].all()


@pytest.fixture
def release(tmp_path: Path) -> Path:
    root = zarr.open_group(str(tmp_path / "data.zarr"), mode="w", zarr_format=3)
    root.attrs["chunk_shape"] = [4, 2]
    ragged = root.create_group("ragged")
    ragged.create_array("eaf_baseline", shape=(3,), dtype="float32")[:] = [0.1, 0.2, 0.3]
    ragged.create_array("se_coefficients", shape=(2, 2), dtype="float32")[:] = [[1, 2], [3, 4]]
    tier = root.create_group("top_hits").create_group("p_5e_08")
    tier.create_array("analysis_offsets", shape=(4,), dtype="int64")[:] = [0, 2, 2, 5]
    old = root["top_hits"].create_group("p_5e_06")
    old.create_array("analysis_index", shape=(4,), dtype="int32")[:] = [0, 2, 2, 2]
    return tmp_path


def test_the_side_arrays_read_as_stored_and_are_none_where_absent(release: Path):
    group = sequence_arrays(release)
    baseline = eaf_baseline(group)
    assert baseline is not None and baseline.dtype == np.float32
    np.testing.assert_array_equal(baseline, np.array([0.1, 0.2, 0.3], dtype=np.float32))
    coefficients = se_coefficients(group)
    assert coefficients is not None and coefficients.shape == (2, 2)
    root = release_arrays(release)
    assert eaf_baseline(root) is None
    assert se_coefficients(root) is None
    assert stored_attributes(release / "data.zarr") == {"chunk_shape": [4, 2]}


def test_hit_counts_come_from_the_offsets_or_from_counting_an_older_index(release: Path):
    root = release_arrays(release)
    assert analysis_hit_counts(root, 5e-8, 3) == [2, 0, 3]
    assert analysis_hit_counts(root, 5e-6, 3) == [1, 0, 3]
    with pytest.raises(KeyError):
        analysis_hit_counts(root, 5e-4, 3)
