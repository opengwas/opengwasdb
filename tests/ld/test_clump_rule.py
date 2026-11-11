"""`scripts/realdata/clump_rule.py`: the rule that explains a disagreement with plink2 (ogld#6).

The realdata tier and the acceptance run call it on real stores; these tests give it small
matrices whose answer is known, and the ones that must fail are watched failing: a pair outside
the encoding's band is refused, not explained.
"""

from __future__ import annotations

import numpy as np
import pytest
from script_modules import load_script

rule = load_script("realdata/clump_rule")

E511 = 1 / (2 * 511)


def three_variants(stored_r: float, true_r2: float) -> dict[str, object]:
    """Variants 0, 1, 2 in order of strength; only the pair (0, 1) is judged differently."""
    r_hat = np.array([[1, stored_r, 0.0], [stored_r, 1, 0.9], [0.0, 0.9, 1]])
    truth = np.array([[1, true_r2, 0.0], [true_r2, 1, 0.81], [0.0, 0.81, 1]])
    return {
        "order": [0, 1, 2],
        "position": np.array([100, 200, 300]),
        "r_hat": r_hat,
        "true_r2": truth,
        "window_bp": 1000,
        "max_error": E511,
    }


def test_replay_clumps_strongest_first_and_each_variant_joins_the_strongest_lead() -> None:
    join = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=bool)
    assert rule.replay([1, 0, 2, 3], join) == [1, 3], "1 leads 0 and 2; 3 is alone"
    assert rule.replay([0, 1, 2, 3], join) == [0, 2, 3], "0 takes 1, so 1 cannot take 2"


def test_a_pair_on_the_threshold_that_rounds_the_other_way_explains_the_difference() -> None:
    # r̂ = 0.100009 (code 51 of 511 is 0.0998; here a code at ×32767, 3277, is 0.1000092), so the
    # store joins 0 and 1 at r2 = 0.01 and plink2's r² = 0.0099992 does not.
    data = {**three_variants(0.1000092, 0.0099992), "max_error": 1 / (2 * 32767)}
    got = rule.explain(r2=0.01, **data)
    assert got.leads_store == [0, 2], "the store joins 1 to 0, so 2, joined only to 1, leads"
    assert got.leads_true == [0, 1], "plink2 leaves 1 a lead, and it takes 2"
    assert [(p.i, p.j) for p in got.flipped] == [(0, 1)]


def test_a_difference_outside_the_band_is_refused_not_explained() -> None:
    # r̂² = 0.0121 against r2 = 0.01 is 0.0021 past it: far beyond 2|r|·e + e² = 0.0004 at ×511.
    with pytest.raises(rule.OutsideBand, match=r"0\.0121"):
        rule.explain(r2=0.01, **three_variants(0.11, 0.0099))


def test_a_threshold_moved_by_0_002_leaves_the_flipped_pair_outside_the_band() -> None:
    # The same pair that is explained at 0.01 (r̂² 0.0100018, true r² 0.0099992). Judged at 0.012
    # against a true r² of 0.0125 the two sides disagree 0.002 from the threshold, outside the band.
    data = three_variants(0.1000092, 0.0125)
    with pytest.raises(rule.OutsideBand):
        rule.explain(r2=0.012, **{**data, "max_error": 1 / (2 * 32767)})
    rule.explain(r2=0.01, **{**three_variants(0.1000092, 0.0099992), "max_error": 1 / (2 * 32767)})


def test_an_unstored_pair_inside_the_window_is_refused_not_read_as_no_correlation() -> None:
    data = three_variants(0.1000092, 0.0099992)
    data["r_hat"] = np.array([[1, np.nan, 0.0], [np.nan, 1, 0.9], [0.0, 0.9, 1]])
    with pytest.raises(rule.OutsideBand, match="not stored"):
        rule.explain(r2=0.01, **data)


def test_pairs_beyond_the_window_join_nothing_whatever_their_r() -> None:
    data = three_variants(0.5, 0.25)
    data["window_bp"] = 100  # the pair (0, 1) is 100 bp apart: not < 100
    got = rule.explain(r2=0.01, **data)
    assert got.flipped == []
    assert got.leads_store == got.leads_true == [0, 1, 2]
