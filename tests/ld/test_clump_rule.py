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


def test_the_band_is_two_r_half_codes_plus_the_square_of_one() -> None:
    # r̂² = 0.0121 against r2 = 0.01 is 0.0021 past it: far beyond 2|r|·e + e² = 0.0004 at × 511.
    assert not rule.within_band(0.11, 0.01, E511)
    assert rule.within_band(51 / 511, 0.01, E511), "code 51 is 0.0000394 from the threshold"
    assert rule.within_band(-51 / 511, 0.01, E511), "the sign is no part of it"
    assert not rule.within_band(0.11, 0.01, E511 / 64), "the band narrows with the step"


def test_a_difference_outside_the_band_is_refused_not_explained() -> None:
    with pytest.raises(rule.OutsideBand):
        rule.explain(r2=0.01, **three_variants(0.11, 0.0099))


def test_a_threshold_moved_by_0_002_leaves_the_flipped_pair_outside_the_band() -> None:
    # At × 511 the store holds r̂ = 52/511, r̂² = 0.010355. A true r² of 0.0102 (r = 0.10100, within
    # half a code of r̂) against r2 = 0.0103 is a pair the encoding explains: the store joins and
    # plink2 does not, 0.000055 from the threshold.
    stored = 52 / 511
    on_the_threshold = three_variants(stored, 0.0102)
    got = rule.explain(r2=0.0103, **on_the_threshold)
    assert [(p.i, p.j) for p in got.flipped] == [(0, 1)]
    # Moved by 0.002 on both sides, the threshold is 0.0123 and plink2's side is a true r² of
    # 0.0125: r = 0.1118 is 0.0100 from r̂, ten half-codes away, so no encoding gives it.
    moved = three_variants(stored, 0.0125)
    with pytest.raises(rule.OutsideBand):
        rule.explain(r2=0.0123, **moved)


def test_plink2_s_r_must_lie_within_half_a_code_of_the_stored_r_for_every_pair_in_the_window() -> (
    None
):
    # r̂ = 51/511 (r̂² = 0.0099606) is on the threshold's side that plink2's r² = 0.0121 is not: the
    # pair is flipped at r2 = 0.01 and within the band by r̂ alone, but plink2's r = 0.11 is 0.0102
    # from r̂, ten half-codes: the stored r is not the rounding of anything plink2 measured.
    with pytest.raises(rule.OutsideBand, match=r"plink2"):
        rule.explain(r2=0.01, **three_variants(51 / 511, 0.0121))
    # And a pair that is NOT flipped is checked too: both sides agree it joins, but plink2's r²
    # is 0.5 against a stored 0.81.
    data = three_variants(51 / 511, 0.0099)
    data["true_r2"] = np.array([[1, 0.0099, 0.0], [0.0099, 1, 0.5], [0.0, 0.5, 1]])
    with pytest.raises(rule.OutsideBand, match=r"plink2"):
        rule.explain(r2=0.01, **data)


def test_a_pair_plink2_did_not_give_is_refused_not_read_as_no_join() -> None:
    data = three_variants(0.1000092, 0.0099992)
    truth = np.array(data["true_r2"], dtype=float)
    truth[0, 1] = truth[1, 0] = np.nan
    data["true_r2"] = truth
    with pytest.raises(rule.OutsideBand, match="plink2 gave no r"):
        rule.explain(r2=0.01, **{**data, "max_error": 1 / (2 * 32767)})


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
