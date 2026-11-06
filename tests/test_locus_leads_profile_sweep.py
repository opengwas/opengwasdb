"""The profile and LD-block sweep's guards (prototypes/locus_leads, #409).

The rules are judged on #403's shared set, so the sweep refuses to go on when
the rows it ranks differ from #403's by a single row.
"""

from __future__ import annotations

import numpy as np
import pytest
from prototype_modules import load_prototype

profile_sweep = load_prototype("locus_leads", "profile_sweep")
profile_valleys = load_prototype("locus_leads", "profile_valleys")

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


def _valley(bin_index: int, variants: int, observed: int) -> dict[str, object]:
    return {"store": "OGS-1", "analysis_id": "GCST1", "bin_bp": 10_000, "chromosome": "1",
            "bin": bin_index, "variants": variants, "observed": observed, "imputed": 0,
            "max_observed_abs_z": None}


def test_judgement_is_withheld_when_most_sampled_valleys_are_gaps_in_coverage() -> None:
    # Every sampled valley's cells are missing: the profile cannot tell a valley from a gap.
    summary = profile_valleys.summarise_valley_cells([_valley(b, 5, 0) for b in range(4)])
    assert summary["pooled"]["share_gap"] == 1.0 and summary["pooled"]["stop_rule_trips"]
    judged: list[tuple[object, ...]] = []

    def judge(*given: object) -> dict[str, object]:
        judged.append(given)
        return {}

    out = profile_sweep.conclude([], {}, summary, judge=judge)

    assert judged == []
    assert out["judgement"] is None
    assert "10000" in out["judgement_withheld"] and "gaps" in out["judgement_withheld"]


def test_the_rules_are_judged_when_the_valleys_are_low_abs_z() -> None:
    summary = profile_valleys.summarise_valley_cells(
        [_valley(0, 5, 0), _valley(1, 5, 2), _valley(2, 5, 1)])
    assert not summary["pooled"]["stop_rule_trips"]

    out = profile_sweep.conclude(["a"], {"c": 1}, summary, judge=lambda *a: {"judged": a[:2]})

    assert out == {"judgement": {"judged": (["a"], {"c": 1})}}
