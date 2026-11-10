"""The whole path, once: the Test Panel's pgen → `ogld.build` at k = 0 → `ogld.open` → one pair.

Nothing in it comes from `reference_r` except the expected value: the code the store answers with
was computed from the pgen's genotypes by the build (ogld#12's thin path, walked by ogld#5's
builder since its own k = 0 Near Band replaced the walking skeleton's).
"""

from __future__ import annotations

import build_fixtures
import numpy as np
import pytest
import testdata

import ogld
from ogld.format import NEAR_BP

HALF_STEP = 1 / 65534


def test_a_pair_from_the_pgen_is_answered_by_the_store_built_from_it(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    _, out = build_fixtures.test_panel_build(0, tmp_path_factory)
    store = ogld.open(out)
    ref = testdata.reference_r(0)
    v = testdata.variants()
    on22 = (v.chrom[ref.row_i] == "22") & (v.pos[ref.row_j] - v.pos[ref.row_i] <= NEAR_BP)
    sign = build_fixtures.a1_signs()
    # A chr22 Near Band pair whose two variants' A1s differ in kind: a lost sign would show.
    mixed = on22 & (sign[ref.row_i] != sign[ref.row_j]) & (np.abs(ref.r) > 0.05)
    assert mixed.any(), "the Test Panel must hold such a pair for this to mean anything"
    pick = int(np.flatnonzero(mixed)[0])
    i, j = int(ref.row_i[pick]), int(ref.row_j[pick])
    got = store.matrix(build_fixtures.alids([i, j])).r[0, 1]
    assert abs(got - ref.r[pick] * sign[i] * sign[j]) <= HALF_STEP + 1e-9
    assert store.near_bp == NEAR_BP
