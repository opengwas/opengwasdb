"""Genotype Stores answer through the same public queries as the independent Test Store
(opengwas/ogld#91)."""

from pathlib import Path

import build_fixtures
import numpy as np
import testdata
from contract.support import TEST_STORE

import opengwasdb.ld


def test_matrix_equals_the_independent_test_store(genotype_store: Path) -> None:
    rows = np.flatnonzero(testdata.variants().passes_filters)
    assert len(rows) > 100
    names = [*build_fixtures.alids(rows), "absent", build_fixtures.alids(rows[:1])[0]]
    expected = opengwasdb.ld.open(TEST_STORE).matrix(names)
    actual = opengwasdb.ld.open(genotype_store).matrix(names)
    assert actual.absent == expected.absent == ("absent",)
    np.testing.assert_array_equal(actual.r, expected.r)
    np.testing.assert_array_equal(actual.kinds(), expected.kinds())


def test_provenance_claims_partner_rows_for_both_store_kinds(genotype_store: Path) -> None:
    """`claims_partner_rows` is the capability rule: both kinds answer `clump` from t_partner."""
    from ogquery.ld.format import claims_partner_rows

    ld = opengwasdb.ld.open(TEST_STORE).provenance
    sidecar = opengwasdb.ld.open(genotype_store).provenance
    assert claims_partner_rows(ld), "the LD Store records its Partner Threshold"
    assert claims_partner_rows(sidecar), "the sidecar answers clump and proxies from t_partner"
    without = {key: value for key, value in dict(ld).items() if key != "t_partner"}
    assert not claims_partner_rows(without), "a store that records no threshold claims none"
