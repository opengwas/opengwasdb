"""The variable-length ALID sidecar's codec (format 0.3.0, #467, ADR 0105).

Row -> ALID is an offset into a byte array; ALID -> row is a binary search over a
fixed-width prefix key, settled by an exact compare on the full bytes. Every ALID is
indexed, however long, which the fixed-width S32 index of the 0.2 series was not
(#127: 115,043 ALIDs left out on OGS-00010).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from opengwasdb.store_reader import alid_sidecar
from opengwasdb.store_reader.alid_sidecar import (
    KEY_WIDTH,
    AlidSidecar,
    DuplicateAlidError,
    encode_alids,
    write_alid_sidecar,
)

#: A very long indel: a 600-base deletion, an ALID of 607 bytes.
VERY_LONG = "7:117559590:" + "ACGT" * 150 + ":A"

#: Two ALIDs over 32 bytes that agree over their first 32 and over the key width:
#: the pair the S32 index collapsed onto one key (#127).
LONG_A = "1:827587:GGGGCCGGGGCCGGGGCCGGGGCCGGGGCC:G"
LONG_B = "1:827587:GGGGCCGGGGCCGGGGCCGGGGCCGGGGCCGGGGCC:G"

ALIDS = [
    "1:100:A:G",
    LONG_B,
    "1:827587:G:GGGGCC",
    VERY_LONG,
    "10:100000000:C:T",
    LONG_A,
    "X:5:A:C",
    "1:100:A:C",
]


def _sidecar(tmp_path: Path, alids: list[str]) -> AlidSidecar:
    write_alid_sidecar(tmp_path, alids)
    opened = AlidSidecar.open(tmp_path)
    assert opened is not None
    return opened


def test_the_fixture_holds_what_the_s32_index_could_not() -> None:
    long = [alid for alid in ALIDS if len(alid.encode()) > 32]
    assert len(long) == 3
    assert LONG_A[:32] == LONG_B[:32] and LONG_A[:KEY_WIDTH] == LONG_B[:KEY_WIDTH]
    assert len(VERY_LONG) > 600


def test_round_trip_names_every_row(tmp_path: Path) -> None:
    sidecar = _sidecar(tmp_path, ALIDS)
    assert sidecar.n_rows == len(ALIDS)
    assert [sidecar.alid(row) for row in range(len(ALIDS))] == ALIDS
    assert sidecar.alids(np.arange(len(ALIDS))).tolist() == ALIDS
    assert sidecar.alids(np.array([3, 0, 3])).tolist() == [VERY_LONG, ALIDS[0], VERY_LONG]


def test_every_alid_resolves_to_its_row_however_long(tmp_path: Path) -> None:
    sidecar = _sidecar(tmp_path, ALIDS)
    for row, alid in enumerate(ALIDS):
        assert sidecar.row_of(alid) == row, alid
    assert sidecar.rows_of(ALIDS).tolist() == list(range(len(ALIDS)))


def test_a_batch_search_marks_misses_and_keeps_query_order(tmp_path: Path) -> None:
    sidecar = _sidecar(tmp_path, ALIDS)
    queries = [
        LONG_A,
        "1:100:A:T",  # absent
        LONG_A[:-1] + "T",  # absent, sharing LONG_A's key
        LONG_A[:KEY_WIDTH],  # absent: a key's bytes, not an ALID the store holds
        VERY_LONG,
        "1:100:A:G",
        VERY_LONG + "A",  # absent
    ]
    assert sidecar.rows_of(queries).tolist() == [5, -1, -1, -1, 3, 0, -1]
    assert sidecar.rows_of([]).tolist() == []


def test_the_order_is_by_full_bytes(tmp_path: Path) -> None:
    encoded = encode_alids(ALIDS)
    full = [ALIDS[row].encode() for row in encoded.order]
    assert full == sorted(full)
    assert encoded.keys.dtype == np.dtype(f"S{KEY_WIDTH}")
    assert np.all(encoded.keys[:-1] <= encoded.keys[1:])


def test_duplicates_are_refused() -> None:
    with pytest.raises(DuplicateAlidError, match="1:100:A:G"):
        encode_alids(["1:100:A:G", "2:1:A:C", "1:100:A:G"])
    with pytest.raises(DuplicateAlidError, match="GGGGCC"):
        encode_alids([LONG_A, "2:1:A:C", LONG_A])


def test_an_empty_axis_is_four_empty_arrays(tmp_path: Path) -> None:
    sidecar = _sidecar(tmp_path, [])
    assert sidecar.n_rows == 0
    assert sidecar.rows_of(["1:1:A:C"]).tolist() == [-1]


def test_the_files_are_plain_memory_mappable_npy(tmp_path: Path) -> None:
    write_alid_sidecar(tmp_path, ALIDS)
    names = sorted(path.name for path in tmp_path.iterdir())
    assert names == sorted(alid_sidecar.ALID_SIDECAR_FILENAMES)
    for name in names:
        array = np.load(tmp_path / name, mmap_mode="r")
        assert isinstance(array, np.memmap), name


def test_absent_files_open_as_none(tmp_path: Path) -> None:
    assert AlidSidecar.open(tmp_path) is None


def test_the_codec_imports_nothing_but_numpy_and_the_stdlib() -> None:
    code = (
        "import sys, opengwasdb.store_reader.alid_sidecar as m; "
        "bad = sorted(n for n in sys.modules if n.split('.')[0] in "
        "{'pysam', 'sqlite3', 'zarr', 'pandas', 'scipy'}); print(bad)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
