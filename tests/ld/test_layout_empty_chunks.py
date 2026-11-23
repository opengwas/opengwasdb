"""A chunk holding only the fill value is not written, whatever the process has set (#455).

Importing `opengwasdb.ld` imports `opengwasdb`, which sets zarr's process-wide
`array.write_empty_chunks` to True for the GWAS Store (opengwasdb/store/arrays.py). ogld built
every LD Store and Genotype Store under zarr's default, False: such a chunk is left out and reads
as the fill value. Every array the LD engine creates is created in `opengwasdb.ld.store.layout`,
which keeps ogld's behaviour."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from zarr import Group, config, open_group

from opengwasdb.ld.format import REncoding
from opengwasdb.ld.store import layout

ENCODING = REncoding(511)
LENGTH = 40


def chunk_files(store: Path, name: str) -> list[str]:
    """The chunk (or shard) files written for the array `name` of the group at `store`."""
    return sorted(str(p.relative_to(store)) for p in (store / name).rglob("c/*") if p.is_file())


@pytest.fixture
def group(tmp_path: Path) -> Group:
    """An empty group, made under the GWAS engine's process-wide setting."""
    assert config.get("array.write_empty_chunks") is True, "the setting under test is gone"
    return open_group(str(tmp_path), mode="w", zarr_format=3)


CREATORS: list[tuple[str, Any, int]] = [
    ("numbers", lambda g: layout.new_numbers(g, "numbers", LENGTH, np.int32, -1, (8, 16)), -1),
    ("r", lambda g: layout.new_r_array(g, "r", LENGTH, (8, 16), ENCODING), ENCODING.reserved_code),
    (layout.FRAMES, lambda g: layout.new_frames_array(g, LENGTH), layout.FRAMES_FILL),
    (layout.FRAMES_INDEX, lambda g: layout.new_frames_index(g, LENGTH), layout.FRAMES_INDEX_FILL),
]


@pytest.mark.parametrize(("name", "make", "fill"), CREATORS, ids=[c[0] for c in CREATORS])
def test_a_chunk_of_only_the_fill_value_is_not_written(
    group: Group, tmp_path: Path, name: str, make: Any, fill: int
) -> None:
    array = make(group)
    array[:] = np.full(LENGTH, fill, dtype=array.dtype)
    assert chunk_files(tmp_path, name) == []
    np.testing.assert_array_equal(array[:], np.full(LENGTH, fill, dtype=array.dtype))
    array[3] = 7  # one value that is not the fill, so the scan above could have seen a chunk
    assert chunk_files(tmp_path, name) != []


def test_layout_is_the_one_place_an_array_is_created() -> None:
    """A `create_array` anywhere else would take the process-wide setting back."""
    package = Path(layout.__file__).resolve().parents[1]
    sources = sorted(package.rglob("*.py"))
    assert len(sources) > 40, f"only {len(sources)} sources under {package}; the glob is wrong"
    calls = [
        str(path.relative_to(package))
        for path in sources
        for line in path.read_text().splitlines()
        if ".create_array(" in line
    ]
    assert calls == ["store/layout.py"], calls  # the one call, inside `layout._create_array`


def test_arrays_written_whole_leave_out_chunks_of_fill(group: Group, tmp_path: Path) -> None:
    data = np.full(LENGTH, -1, dtype=np.int32)
    data[:8] = 5  # the first inner chunk holds data; the rest is fill
    layout.put_numbers(group, "numbers", data, -1, (8, 16))
    layout.put_strings(group, "strings", np.array([""] * LENGTH))
    assert chunk_files(tmp_path, "numbers") == ["numbers/c/0"]
    assert chunk_files(tmp_path, "strings") == []
