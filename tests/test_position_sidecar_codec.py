"""The position sidecar's codec: round trip, offsets, ranges and refusals (#468).

The fixture's rows are in genomic order and hold what a range must get right: three
chromosomes, a position shared by three rows, a chromosome of one row, and the first
and last rows of the axis. Each range is checked against a scan of the same rows, the
answer tabix gave on 0.2.x: every row on the chromosome with ``start <= position <= end``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from opengwasdb.store_reader.position_sidecar import (
    CHROMOSOME_OFFSETS_FILENAME,
    CHROMOSOMES_FILENAME,
    POSITION_FILENAME,
    POSITION_MAX,
    POSITION_SIDECAR_FILENAMES,
    MalformedPositionSidecar,
    PositionSidecar,
    UnorderedPositionsError,
    encode_positions,
    sidecar_problems,
    write_position_sidecar,
)
from opengwasdb.variants.normalise import chromosome_sort_key

#: ``(chromosome, position)`` per row, in row order.
ROWS = [
    ("1", 100),
    ("1", 100),
    ("1", 100),
    ("1", 250),
    ("1", 900),
    ("2", 5),
    ("X", 7),
    ("X", 7),
    ("X", 4_000_000_000),
]
CHROMOSOMES = [chromosome for chromosome, _ in ROWS]
POSITIONS = [position for _, position in ROWS]


def _sidecar(tmp_path: Path, rows: list[tuple[str, int]] = ROWS) -> PositionSidecar:
    write_position_sidecar(
        tmp_path,
        [c for c, _ in rows],
        [p for _, p in rows],
        chromosome_key=chromosome_sort_key,
    )
    sidecar = PositionSidecar.open(tmp_path)
    assert sidecar is not None
    return sidecar


def _scanned(chromosome: str, start: int, end: int) -> list[int]:
    return [i for i, (c, p) in enumerate(ROWS) if c == chromosome and start <= p <= end]


def test_the_fixture_holds_every_case() -> None:
    assert len(set(CHROMOSOMES)) == 3
    assert POSITIONS.count(100) == 3, "a position shared by several rows"
    assert CHROMOSOMES.count("2") == 1, "a chromosome of one row"
    assert max(POSITIONS) > 2**31, "a position only uint32 holds"


def test_round_trip(tmp_path: Path) -> None:
    sidecar = _sidecar(tmp_path)
    assert sidecar.n_rows == len(ROWS)
    assert np.asarray(sidecar.positions).tolist() == POSITIONS
    assert sidecar.chromosomes_of(np.arange(len(ROWS))) == CHROMOSOMES
    assert sidecar.chromosomes_of(np.array([len(ROWS)])) == [None]
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(POSITION_SIDECAR_FILENAMES)
    assert list(sidecar_problems(sidecar, chromosome_key=chromosome_sort_key)) == []


def test_the_offsets_name_each_chromosomes_rows(tmp_path: Path) -> None:
    encoded = encode_positions(CHROMOSOMES, POSITIONS)
    assert encoded.chromosomes.tolist() == [b"1", b"2", b"X"]
    assert encoded.offsets.tolist() == [0, 5, 6, 9]
    assert encoded.positions.dtype == np.dtype("<u4")
    assert encoded.offsets.dtype == np.dtype("<u8")


@pytest.mark.parametrize(
    ("chromosome", "start", "end"),
    [
        ("1", 100, 100),  # starts and ends on a shared position
        ("1", 1, 100),  # the first row, from before it
        ("1", 100, 250),  # from a shared position to the next
        ("1", 101, 249),  # between two rows: empty
        ("1", 900, 10**12),  # the chromosome's last row, end past what uint32 holds
        ("2", 5, 5),  # a chromosome of one row
        ("2", 1, 4),  # before it: empty
        ("X", 7, 7),  # a shared position at a chromosome's start
        ("X", 4_000_000_000, POSITION_MAX),  # the axis's last row
        ("X", -5, 0),  # nothing is at position 0 or below
        ("Y", 1, 10**9),  # a chromosome the release lacks
        ("1", 250, 249),  # end before start: empty, as tabix's fetch(249, 249)
    ],
)
def test_a_range_is_the_rows_a_scan_finds(
    tmp_path: Path, chromosome: str, start: int, end: int
) -> None:
    sidecar = _sidecar(tmp_path)
    assert sidecar.rows_in(chromosome, start, end).tolist() == _scanned(chromosome, start, end)


def test_a_range_stays_inside_its_chromosome(tmp_path: Path) -> None:
    """Position 5 is on chromosome 2 only; chromosome 1's slice ends before its row."""
    sidecar = _sidecar(tmp_path)
    assert sidecar.rows_in("1", 1, 10**9).tolist() == [0, 1, 2, 3, 4]
    assert sidecar.rows_in("2", 1, 10**9).tolist() == [5]
    assert sidecar.rows_in("X", 1, 10**9).tolist() == [6, 7]


def test_an_empty_release_has_no_rows_and_no_chromosomes(tmp_path: Path) -> None:
    sidecar = _sidecar(tmp_path, [])
    assert sidecar.n_rows == 0
    assert sidecar.rows_in("1", 1, 10**9).tolist() == []
    assert np.asarray(sidecar.offsets).tolist() == [0]
    assert list(sidecar_problems(sidecar, chromosome_key=chromosome_sort_key)) == []


@pytest.mark.parametrize(
    ("rows", "says"),
    [
        ([("1", 10), ("2", 5), ("1", 20)], "resumes at row 2"),
        ([("1", 10), ("1", 9)], "row 1 is at position 9, before row 0"),
        ([("2", 10), ("1", 5)], "out of the canonical chromosome order"),
        ([("X", 10), ("10", 5)], "out of the canonical chromosome order"),
    ],
)
def test_rows_out_of_genomic_order_are_refused(rows: list[tuple[str, int]], says: str) -> None:
    with pytest.raises(UnorderedPositionsError, match=says):
        encode_positions(
            [c for c, _ in rows], [p for _, p in rows], chromosome_key=chromosome_sort_key
        )


@pytest.mark.parametrize("position", [0, -1, POSITION_MAX + 1])
def test_a_position_outside_uint32_is_refused(position: int) -> None:
    with pytest.raises(ValueError, match="outside 1.."):
        encode_positions(["1", "1"], [5, position])


def test_an_empty_chromosome_label_is_refused() -> None:
    with pytest.raises(UnorderedPositionsError, match="empty chromosome"):
        encode_positions(["1", ""], [5, 6])


def test_a_half_written_sidecar_is_refused(tmp_path: Path) -> None:
    _sidecar(tmp_path)
    (tmp_path / CHROMOSOMES_FILENAME).unlink()
    with pytest.raises(FileNotFoundError, match="position sidecar is incomplete"):
        PositionSidecar.open(tmp_path)
    for name in (POSITION_FILENAME, CHROMOSOME_OFFSETS_FILENAME):
        (tmp_path / name).unlink()
    assert PositionSidecar.open(tmp_path) is None


def _edit(store: Path, name: str, array: np.ndarray) -> None:
    np.save(store / name, array)


@pytest.mark.parametrize(
    ("name", "array", "says"),
    [
        (CHROMOSOME_OFFSETS_FILENAME, np.array([0, 2, 5, 6, 9], dtype="<u8"),
         "has 5 offsets for 3 chromosome"),
        (CHROMOSOME_OFFSETS_FILENAME, np.array([0, 5, 9], dtype="<u8"),
         "has 3 offsets for 3 chromosome"),
        (CHROMOSOMES_FILENAME, np.array([b"1", b"2", b"1"]), "names chromosome '1' twice"),
        (CHROMOSOMES_FILENAME, np.array([b"1", b"\xff", b"X"]), "chromosome 1 is not UTF-8"),
    ],
)
def test_a_sidecar_whose_chromosomes_cannot_slice_the_rows_is_refused_at_open(
    tmp_path: Path, name: str, array: np.ndarray, says: str
) -> None:
    """Read as it stands, each would answer some chromosome's ranges from another's rows,
    or from none: an answer that looks like no association."""
    _sidecar(tmp_path)
    _edit(tmp_path, name, array)
    with pytest.raises(MalformedPositionSidecar, match=says):
        PositionSidecar.open(tmp_path)


def test_misplaced_names_each_row_the_sidecar_places_elsewhere(tmp_path: Path) -> None:
    sidecar = _sidecar(tmp_path)
    assert sidecar.misplaced(0, CHROMOSOMES, POSITIONS) == []
    moved, renamed = list(POSITIONS), list(CHROMOSOMES)
    moved[3], renamed[5] = 251, "3"
    assert sidecar.misplaced(0, renamed, moved) == [3, 5]
    assert sidecar.misplaced(5, renamed[5:], POSITIONS[5:]) == [5], "rows count from `start`"
    # A row past the sidecar's end is not compared: `n_rows` says the sidecar is short.
    assert sidecar.misplaced(8, ["X", "Y"], [4_000_000_000, 1]) == []
    assert sidecar.placements([3, 5, len(ROWS)]) == [("1", 250), ("2", 5), (None, None)]


def test_the_codec_imports_numpy_and_the_standard_library_only() -> None:
    code = (
        "import sys; import opengwasdb.store_reader.position_sidecar; "
        "print(sorted(m for m in ('pysam', 'sqlite3', 'zarr', 'pandas') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
