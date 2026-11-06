"""The profile and LD-block sweep's guards (prototypes/locus_leads, #409).

The rules are judged on #403's shared set, so the sweep refuses to go on when
the rows it ranks differ from #403's by a single row.
"""

from __future__ import annotations

import numpy as np
import pytest
from prototype_modules import load_prototype

profile_sweep = load_prototype("locus_leads", "profile_sweep")

INDICES = np.array([12, 40, 7, 1_000_003, 99])


def _identity(indices: np.ndarray, analysis: str = "GCST1") -> dict[str, object]:
    identity: dict[str, object] = profile_sweep.shared_set_identity("OGS-1", analysis, indices)
    return identity


def test_the_same_rows_in_another_order_are_the_same_shared_set() -> None:
    reference = _identity(INDICES)
    assert reference["rows"] == 5

    profile_sweep.check_same_shared_set(_identity(INDICES[::-1]), reference)


@pytest.mark.parametrize("change", ["swap one", "drop one", "add one", "other Analysis"])
def test_a_shared_set_one_row_from_the_reference_is_refused(change: str) -> None:
    reference = _identity(INDICES)
    mine = {
        "swap one": lambda: _identity(np.r_[INDICES[:-1], 100]),
        "drop one": lambda: _identity(INDICES[:-1]),
        "add one": lambda: _identity(np.r_[INDICES, 100]),
        "other Analysis": lambda: _identity(INDICES, "GCST2"),
    }[change]()

    with pytest.raises(profile_sweep.SharedSetMismatch, match="differs from"):
        profile_sweep.check_same_shared_set(mine, reference)
