"""The walking skeleton's contract: the thinnest real path, and the Test Store it writes.

pgen → r at k = 0 → the Near Band → write → read → one pair answered. The skeleton also
writes the Test Store from the Test Panel's `reference_r(10)` with the writer, into
`tests/data/test-panel/store/`, and gives `scripts/make_test_store.py` a
`write_test_store(k, out)` that ogld#6 uses for a k = 0 store. From then on ogld#4–#6 build
against real data in the format. Paths it does not build yet raise NotImplementedError
naming their ticket, so the other modules' contract tests keep failing for the right
reason.
"""

from __future__ import annotations

from pathlib import Path

from script_modules import load_script
from support import HALF_STEP, SLACK, T_FAR, TEST_STORE, alids, contract, reference

import ogld

SKELETON = contract("the walking skeleton")


@SKELETON
def test_the_test_store_opens() -> None:
    store = ogld.open(TEST_STORE)
    assert (store.t_far, store.near_bp, store.far_bp) == (T_FAR, 3_000_000, 10_000_000)


@SKELETON
def test_one_near_band_pair_reads_back_its_reference_r() -> None:
    store = ogld.open(TEST_STORE)
    ref = reference()
    pair = ref.where((ref.distance > 0) & (ref.distance <= 3_000_000), "a Near Band pair")
    m = store.matrix(alids([pair.row_i[0], pair.row_j[0]]))
    assert m.absent == ()
    assert abs(float(m.r[0, 1]) - pair.r[0]) <= HALF_STEP + SLACK


@SKELETON
def test_the_test_store_can_be_written_again_at_k0(tmp_path: Path) -> None:
    ogld.open(TEST_STORE)
    out = load_script("make_test_store").write_test_store(0, tmp_path / "k0")
    store = ogld.open(out)
    assert store.provenance["adjustment"]["k"] == 0
    ref = reference(0)
    pair = ref.where((ref.distance > 0) & (ref.distance <= 3_000_000), "a Near Band pair at k = 0")
    m = store.matrix(alids([pair.row_i[0], pair.row_j[0]]))
    assert abs(float(m.r[0, 1]) - pair.r[0]) <= HALF_STEP + SLACK
