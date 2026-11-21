"""Genotype Stores answer through the same public queries as the independent Test Store (#91)."""

from pathlib import Path

import build_fixtures
import numpy as np
import testdata
from contract.support import TEST_STORE

import ogld


def test_matrix_equals_the_independent_test_store(genotype_store: Path) -> None:
    rows = np.flatnonzero(testdata.variants().passes_filters)
    assert len(rows) > 100
    names = [*build_fixtures.alids(rows), "absent", build_fixtures.alids(rows[:1])[0]]
    expected = ogld.open(TEST_STORE).matrix(names)
    actual = ogld.open(genotype_store).matrix(names)
    assert actual.absent == expected.absent == ("absent",)
    np.testing.assert_array_equal(actual.r, expected.r)
    np.testing.assert_array_equal(actual.kinds(), expected.kinds())
