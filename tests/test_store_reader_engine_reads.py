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

from opengwasdb.store_reader.alid import RSID_DTYPE
from opengwasdb.store_reader.alid_sidecar import write_alid_sidecar
from opengwasdb.store_reader.format import CURRENT_FORMAT_VERSION
from opengwasdb.store_reader.position_sidecar import write_position_sidecar
from opengwasdb.store_reader.release import release_arrays
from opengwasdb.store_reader.source_alid_sidecar import write_source_alid_sidecar
from opengwasdb.store_reader.stored import (
    eaf_baseline,
    se_coefficients,
    sequence_arrays,
    stored_attributes,
)
from opengwasdb.store_reader.top_hits import analysis_hit_counts
from opengwasdb.store_reader.variant_axis import VariantAxis, parse_variant_line


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


#: The two ways a release holds `source_alid`: the 0.2 series' table column, and 0.3.0's
#: sidecar (#470), from which the table is never read.
VERSIONS = ["0.2.3", CURRENT_FORMAT_VERSION]


def _axis(
    tmp_path: Path, rows: list[str], version: str = CURRENT_FORMAT_VERSION
) -> tuple[VariantAxis, _Table]:
    """An axis of `version` over `rows`, and an rsID index naming none (`_row`): on 0.3.0
    its sidecars, `source_alid`'s included; on 0.2.3 the fixed-width ALID pair and
    offsets the table is read by."""
    from format_0_2_3 import write_fixed_width_alid_index

    records = [parse_variant_line(row) for row in rows]
    alids = [record.alid for record in records]
    if version == "0.2.3":
        write_fixed_width_alid_index(tmp_path, alids)
        np.save(tmp_path / "variant_offsets.npy", np.arange(len(rows), dtype=np.uint64))
    else:
        write_alid_sidecar(tmp_path, alids)
        write_position_sidecar(
            tmp_path,
            [record.chromosome for record in records],
            [record.position for record in records],
            chromosome_key=None,
        )
        write_source_alid_sidecar(tmp_path, [record.source_alid for record in records], alids)
    np.save(tmp_path / "variant_rsid_bytes.npy", np.empty(0, dtype=RSID_DTYPE))
    np.save(tmp_path / "variant_rsid_rows.npy", np.empty(0, dtype="int32"))
    table = _Table(rows)
    return VariantAxis(tmp_path, table, format_version=version), table


def _row(index: int, source_alid: str | None) -> str:
    alid = f"1:{1000 + index}:A:G"
    fields = ["1", str(1000 + index), str(index), "A", "G", alid, "."]
    if source_alid is not None:
        fields.append(source_alid)
    return "\t".join(fields) + "\n"


def test_source_alids_resolve_to_their_rows_and_the_table_scan_stops_once_all_are_found(
    tmp_path: Path,
):
    rows = [_row(i, f"1:{500 + i}:A:G") for i in range(6)]
    axis, table = _axis(tmp_path, rows, "0.2.3")
    found = axis.indices_by_source_alid({"1:501:A:G", "1:503:A:G", "1:999:A:G"})
    assert found == {"1:501:A:G": 1, "1:503:A:G": 3}
    assert table.read == 6  # an ALID the table lacks is looked for to the end

    axis, table = _axis(tmp_path, rows, "0.2.3")
    assert axis.indices_by_source_alid({"1:501:A:G", "1:503:A:G"}) == {
        "1:501:A:G": 1,
        "1:503:A:G": 3,
    }
    assert table.read == 4


def test_on_0_3_0_source_alids_resolve_from_the_sidecar_and_the_table_is_not_read(
    tmp_path: Path,
):
    rows = [_row(i, f"1:{500 + i}:A:G") for i in range(6)]
    axis, table = _axis(tmp_path, rows)
    found = axis.indices_by_source_alid({"1:501:A:G", "1:503:A:G", "1:999:A:G"})
    assert found == {"1:501:A:G": 1, "1:503:A:G": 3}
    assert table.read == 0


@pytest.mark.parametrize("version", VERSIONS)
def test_a_blank_source_alid_matches_nothing(tmp_path: Path, version: str):
    axis, table = _axis(tmp_path, [_row(0, "."), _row(1, "1:7:C:T")], version)
    assert axis.indices_by_source_alid({"1:7:C:T"}) == {"1:7:C:T": 1}
    # `.` is how the table spells a row with no source ALID, not an ALID to match.
    assert axis.indices_by_source_alid({"."}) == {}
    assert axis.indices_by_source_alid({".", "1:7:C:T"}) == {"1:7:C:T": 1}
    assert axis.indices_by_source_alid({"1:8:C:T"}) == {}
    assert (table.read > 0) == (version == "0.2.3")


def test_a_table_without_the_source_alid_column_says_so_rather_than_matching_nothing(
    tmp_path: Path,
):
    axis, _ = _axis(tmp_path, [_row(0, None), _row(1, None)], "0.2.3")
    assert axis.indices_by_source_alid({"1:1000:A:G"}) is None


def test_on_0_3_0_a_sidecar_naming_none_says_so_rather_than_matching_nothing(tmp_path: Path):
    """A 0.3.0 release rewritten from a table without the column names no row's
    `source_alid` (#470): like that table, it says so (None), so the caller lifts the
    ALIDs itself, and reads no table to say it."""
    axis, table = _axis(tmp_path, [_row(0, None), _row(1, None)])
    assert axis.indices_by_source_alid({"1:1000:A:G"}) is None
    assert table.read == 0


@pytest.mark.parametrize("version", VERSIONS)
def test_records_stream_what_all_materialises(tmp_path: Path, version: str):
    rows = [_row(i, None if i % 2 else f"1:{i}:A:G") for i in range(5)]
    axis, table = _axis(tmp_path, rows, version)
    stream = axis.records()
    assert table.read == 0  # nothing read until the stream is
    first = next(stream)
    assert first.variant_index == 0
    assert table.read == (1 if version == "0.2.3" else 0)  # 0.3.0 reads the sidecars
    streamed = [first, *stream]
    assert streamed == _axis(tmp_path, rows, version)[0].all()
    assert streamed == [parse_variant_line(row) for row in rows]  # the table's rows, either way


def test_written_table_records_read_the_table_on_0_3_0(tmp_path: Path):
    """The validator's read: the table as it is written, which it checks the sidecars
    against, on 0.3.0 as on 0.2."""
    rows = [_row(i, None if i % 2 else f"1:{i}:A:G") for i in range(5)]
    axis, table = _axis(tmp_path, rows)
    assert list(axis.written_table_records()) == [parse_variant_line(row) for row in rows]
    assert table.read == 5


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
