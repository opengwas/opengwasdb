"""What the contract tests share: the Test Store, the marker, and the Test Panel read as ALIDs.

The contract tests are epic #1's tickets written as tests before the tickets start. Each is
marked `contract("ogld#<n>")`: a strict xfail that expects NotImplementedError, so it fails
until its ticket lands and errors the moment it passes while still marked. A ticket's job
is to make its own module pass, unmarking each test it makes pass, without changing a
public signature (tests/test_surface.py).

Every test calls the ogld function it is about before it reads the Test Panel or the Test
Store, so an unbuilt function surfaces as the expected failure, not as a missing file.
The Test Panel is lane 2d's `tests/testdata.py`; the Test Store is written from its
`reference_r(10)` by the walking skeleton into `tests/data/test-panel/store/`.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import ModuleType

import numpy as np
import numpy.typing as npt
import pytest

from ogld.format import far_code_floor, quantise, t_far
from ogld.identity import orient

TESTS = Path(__file__).resolve().parents[1]
TEST_STORE = TESTS / "data" / "test-panel" / "store"

#: The Test Panel: HGDP+1kGP EUR's 771 samples; the Test Store is built at k = 10.
N_SAMPLES = 771
K = 10
T_FAR = t_far(N_SAMPLES, K)
FLOOR = far_code_floor(T_FAR)
HALF_STEP = 1 / 65534
#: Two independent float64 computations of one r may differ by a few ulps before rounding.
SLACK = 1e-9


def contract(ticket: str) -> pytest.MarkDecorator:
    """The mark every contract test carries: fails until `ticket` builds what it tests."""
    return pytest.mark.xfail(strict=True, raises=NotImplementedError, reason=ticket)


def panel_data() -> ModuleType:
    """Lane 2d's loader, imported on first use so a test meets its ogld call first."""
    return importlib.import_module("testdata")


@cache
def oriented() -> tuple[list[str], npt.NDArray[np.float64]]:
    """Each Test Panel row's ALID, and +1 where A1 is its ALT allele, −1 where it is REF.

    The Test Panel's reference r is the correlation of ALT dosages, so a stored r, signed to
    A1, is the reference r times both rows' signs (spec §3).
    """
    v = panel_data().variants()
    found = [
        orient(str(c), int(p), str(ref), str(alt))
        for c, p, ref, alt in zip(v.chrom, v.pos, v.ref, v.alt, strict=True)
    ]
    return [o.alid for o in found], np.array([1.0 if o.a1_is_alt else -1.0 for o in found])


def alids(rows: npt.ArrayLike) -> list[str]:
    names, _ = oriented()
    return [names[int(row)] for row in np.atleast_1d(rows)]


def passing_rows() -> npt.NDArray[np.int64]:
    """The Test Panel rows that pass the build's filters: the Test Store's Variant Set."""
    return np.flatnonzero(panel_data().variants().passes_filters).astype(np.int64)


@dataclass(frozen=True)
class Pairs:
    """The reference pairs at one k, with distances and r signed to each row's A1."""

    row_i: npt.NDArray[np.int64]
    row_j: npt.NDArray[np.int64]
    distance: npt.NDArray[np.int64]
    r: npt.NDArray[np.float64]

    @property
    def code(self) -> npt.NDArray[np.int16]:
        return quantise(self.r)

    def where(self, mask: npt.NDArray[np.bool_], what: str) -> Pairs:
        """The pairs `mask` keeps. An empty fixture proves nothing, so there must be one."""
        assert mask.any(), f"the Test Panel must hold {what} for this test to mean anything"
        return Pairs(self.row_i[mask], self.row_j[mask], self.distance[mask], self.r[mask])


@cache
def reference(k: int = K) -> Pairs:
    ref = panel_data().reference_r(k)
    _, sign = oriented()
    pos = panel_data().variants().pos.astype(np.int64)
    i, j = ref.row_i.astype(np.int64), ref.row_j.astype(np.int64)
    return Pairs(i, j, pos[j] - pos[i], ref.r * sign[i] * sign[j])


def lookup(k: int = K) -> dict[tuple[int, int], float]:
    """Reference r by (row_i, row_j), either order."""
    ref = reference(k)
    out = {}
    for i, j, r in zip(ref.row_i.tolist(), ref.row_j.tolist(), ref.r.tolist(), strict=True):
        out[(i, j)] = out[(j, i)] = r
    return out
