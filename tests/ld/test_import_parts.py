"""Hail 0.2.64's worked literal, independent of the production decoder (#93)."""

from pathlib import Path

import numpy as np
import pytest

from ogld.build.importing.parts import read_part

FIXTURE = Path(__file__).parent / "data" / "panukb"


def test_literal_blockmatrix_part() -> None:
    matrix = read_part(FIXTURE / "part-0")
    assert matrix.shape == (3, 3)
    np.testing.assert_array_equal(matrix, [[4, 1, -3], [1, 1, 0.75], [-3, 0.75, 9]])


@pytest.mark.parametrize("payload", [b"", b"\x01", b"\xff" * 16])
def test_corrupt_part_is_refused_by_name(tmp_path: Path, payload: bytes) -> None:
    part = tmp_path / "broken-part"
    part.write_bytes(payload)
    with pytest.raises(ValueError, match="broken-part"):
        read_part(part)


def test_truncated_part_is_refused_by_name(tmp_path: Path) -> None:
    part = tmp_path / "short-part"
    part.write_bytes((FIXTURE / "part-0").read_bytes()[:-1])
    with pytest.raises(ValueError, match="short-part"):
        read_part(part)
