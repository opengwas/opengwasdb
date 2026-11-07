"""The walking skeleton's contract: the thinnest real path, and the Test Store it writes.

pgen → r at k = 0 → the Near Band → write → read → one pair answered. The skeleton also
writes the Test Store from the Test Panel's `reference_r(10)` with the writer, into
`tests/data/test-panel/store/`, and gives `scripts/make_test_store.py` a
`write_test_store(k, out)` that ogld#6 uses for a k = 0 store. From then on ogld#4–#6 build
against real data in the format.

These tests go through the store seam, `StoreReader`, not the queries, which are ogld#4's
and ogld#6's. Paths the skeleton does not build raise NotImplementedError naming their
ticket, so the other modules' contract tests keep failing for the right reason.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from script_modules import load_script
from support import HALF_STEP, SLACK, T_FAR, TEST_STORE, alids, contract, reference

import ogld
from ogld.store import StoreReader

SKELETON = contract("the walking skeleton")


def near_pair_r(reader: StoreReader, k: int) -> tuple[float, float]:
    """A Near Band pair's r as the store reads it back, and as the reference has it."""
    ref = reference(k)
    pair = ref.where((ref.distance > 0) & (ref.distance <= 3_000_000), "a Near Band pair")
    rows = reader.rows_of(alids([pair.row_i[0], pair.row_j[0]]))
    assert (rows >= 0).all(), "both variants must be in the Test Store"
    return float(reader.pair_r(rows)[0, 1]), float(pair.r[0])


@SKELETON
def test_the_test_store_opens_through_ogld_open() -> None:
    store = ogld.open(TEST_STORE)
    assert (store.t_far, store.near_bp, store.far_bp) == (T_FAR, 3_000_000, 10_000_000)


@SKELETON
def test_one_near_band_pair_reads_back_its_reference_r() -> None:
    got, expected = near_pair_r(StoreReader.open(TEST_STORE), 10)
    assert abs(got - expected) <= HALF_STEP + SLACK


@SKELETON
def test_an_unstored_pair_reads_as_nan() -> None:
    reader = StoreReader.open(TEST_STORE)
    ref = reference()
    far = ref.where(ref.distance > 3_000_000, "a pair beyond the Near Band")
    rows = reader.rows_of(alids([far.row_i[0], far.row_j[0]]))
    r = reader.pair_r(rows)[0, 1]
    stored = abs(int(np.rint(far.r[0] * 32767))) >= 5426
    assert np.isnan(r) != stored


@SKELETON
def test_the_test_store_can_be_written_again_at_k0(tmp_path: Path) -> None:
    StoreReader.open(TEST_STORE)
    out = load_script("make_test_store").write_test_store(0, tmp_path / "k0")
    reader = StoreReader.open(out)
    assert reader.provenance["adjustment"]["k"] == 0
    got, expected = near_pair_r(reader, 0)
    assert abs(got - expected) <= HALF_STEP + SLACK
