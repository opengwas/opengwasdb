"""The genotype code equality at eight BLAS threads, on the Test Panel (ogld#90).

`ogld.genotypes.r` must give one set of Stored codes at any BLAS thread count, and eight is the
widest count #90's criterion names. On a CI runner, eight OpenBLAS threads inside each of many
parallel workers oversubscribe the CPUs and spin, so the test times out there; the tier that runs
on the IEU compute nodes holds it instead (`pixi run -e dev test-realdata`, which
`tests/realdata/run.sh` names), and the default tier keeps the one- and four-thread check.

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
def test_the_genotype_codes_are_identical_at_one_and_eight_blas_threads(k: int) -> None:
    assert_codes_identical((1, 8), k, 511)
