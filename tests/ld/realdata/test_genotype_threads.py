"""The genotype code equality at eight BLAS threads, on the Test Panel (ogld#90).

`opengwasdb.ld.genotypes.r` must give one set of Stored codes at any BLAS thread count, and #90's
criterion
names one and eight (four is the widest the default tier ever held). Any count above one inside
each of many parallel workers oversubscribes a small CI runner's CPUs and spins until the test
times out, so the tier that runs on the IEU compute nodes holds them instead
(`pixi run -e dev test-realdata`, which `tests/ld/realdata/run.sh` names); the default tier holds the
one-thread check, its block sizes and single rows.

Nothing here reads the panels: the fixture is the committed Test Panel, and the assertion is the
default tier's, at the encoding the shipped HGDP+1kGP panels record (× 511).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from genotype_codes import assert_codes_identical

#: The tests directory, which holds `genotype_codes`: a realdata module is imported by path, so
#: its own directory is on `sys.path` and the tier's parent is not.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.mark.parametrize("k", [0, 10])
def test_the_genotype_codes_are_identical_at_one_four_and_eight_blas_threads(k: int) -> None:
    assert_codes_identical((1, 4, 8), k, 511)
