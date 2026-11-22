"""Hail 0.2.64's worked literal, independent of the production decoder (#93)."""

import sys
from pathlib import Path

import numpy as np
import pytest

from ogld.build.importing.parts import read_part
from ogld.format import InputError

FIXTURE = Path(__file__).parent / "data" / "panukb"


def test_literal_blockmatrix_part() -> None:
    matrix = read_part(FIXTURE / "part-0")
    assert matrix.shape == (3, 3)
    np.testing.assert_array_equal(matrix, [[4, 1, -3], [1, 1, 0.75], [-3, 0.75, 9]])


def test_second_chromosome_has_an_independent_covariance_literal() -> None:
    matrix = read_part(FIXTURE / "part-1")
    assert not np.array_equal(matrix, read_part(FIXTURE / "part-0"))
    np.testing.assert_array_equal(matrix, [[16, 2, 2], [2, 4, -1], [2, -1, 1]])


@pytest.mark.parametrize("order", ["column", "row"])
def test_nonsymmetric_part_pins_matrix_storage_order(order: str) -> None:
    matrix = read_part(FIXTURE / f"part-{order}-major")
    assert matrix.shape == (2, 3)
    np.testing.assert_array_equal(matrix, [[1, 2, 3], [4, 5, 6]])


@pytest.mark.parametrize("payload", [b"", b"\x01", b"\xff" * 16])
def test_corrupt_part_is_refused_by_name(tmp_path: Path, payload: bytes) -> None:
    part = tmp_path / "broken-part"
    part.write_bytes(payload)
    with pytest.raises(ValueError, match="broken-part"):
        read_part(part)


def test_missing_import_dependency_is_a_named_extra_refusal(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "lz4.block", None)
    with pytest.raises(InputError, match=r"lz4.block.*ogld\[import\]"):
        read_part(FIXTURE / "part-0")


def test_truncated_part_is_refused_by_name(tmp_path: Path) -> None:
    part = tmp_path / "short-part"
    part.write_bytes((FIXTURE / "part-0").read_bytes()[:-1])
    with pytest.raises(ValueError, match="short-part"):
        read_part(part)
