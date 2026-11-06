"""The sweep's grid, baseline, pooled rates and criteria (#403), on hand-worked numbers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from prototype_modules import load_prototype

genetic_map = load_prototype("locus_leads", "genetic_map")
rule = load_prototype("locus_leads", "rule")
sweep = load_prototype("locus_leads", "sweep")

# A hotspot of 2 cM at 1,499,000-1,501,000 on an otherwise cold chromosome 1.
HOTSPOT = genetic_map.GeneticMap(
    {"1": (np.array([1, 1_499_000, 1_501_000, 10_000_000]), np.array([0.0, 0.15, 2.15, 3.0]))})
POSITION = np.array([1_450_000, 1_550_000, 5_450_000, 5_550_000])
CHROMOSOME = np.array(["1"] * 4)
ABS_Z = np.array([9.0, 8.0, 9.0, 8.0])
INDEX = np.array([10, 11, 12, 13])


def _leads(d_cm: float, w_bp: int = 1_000_000) -> list[int]:
    cm = HOTSPOT.cm_at(CHROMOSOME, POSITION)
    return rule.locus_leads(CHROMOSOME, POSITION, cm, ABS_Z, INDEX, d_cm=d_cm,
                            w_bp=w_bp).tolist()


def test_the_grid_is_five_d_by_four_w_and_a_bp_only_baseline_at_each_w() -> None:
    configs = sweep.grid_configs()
    assert len(configs) == 5 * 4 + 4
    assert [c for c in configs if c.d_cm == float("inf")] == [
        sweep.Config(float("inf"), w) for w in (250_000, 1_000_000, 2_500_000, 10_000_000)]
    assert sweep.Config(float("inf"), 1_000_000).label() == {"d_cm": "bp_only", "w_bp": 1_000_000}
    assert sweep.Config(0.5, 1_000_000).label() == {"d_cm": 0.5, "w_bp": 1_000_000}


def test_bp_only_merges_across_the_hotspot_that_the_cm_rule_splits() -> None:
    cm = HOTSPOT.cm_at(CHROMOSOME, POSITION)
    assert cm[1] - cm[0] > 2.0, "fixture must straddle the hotspot"
    assert _leads(0.5) == [0, 1, 2, 2]
    assert _leads(float("inf")) == [0, 0, 2, 2]


def test_d_infinity_is_a_plain_window_on_any_positions() -> None:
    rng = np.random.default_rng(3)
    position = np.sort(rng.integers(1, 10_000_000, 60))
    abs_z = rng.permutation(60) + 1.0
    cm = HOTSPOT.cm_at(np.array(["1"] * 60), position)
    got = rule.locus_leads(np.array(["1"] * 60), position, cm, abs_z, np.arange(60),
                           d_cm=float("inf"), w_bp=1_000_000)
    expected, leads = [], []
    for row in np.argsort(-abs_z):
        near = [lead for lead in leads if abs(position[lead] - position[row]) <= 1_000_000]
        lead = near[0] if near else int(row)
        leads += [] if near else [int(row)]
        expected.append((int(row), lead))
    assert got.tolist() == [lead for _, lead in sorted(expected)]


# Two Analyses at one config: A has 10 leads, 20 index variants, 1 split, 4 lost;
# B has 30 leads, 40 index variants, 5 splits, 6 lost.
A = {"locus_leads": 10, "index_variants": 20, "spurious_splits": 1, "lost_signals": 4}
B = {"locus_leads": 30, "index_variants": 40, "spurious_splits": 5, "lost_signals": 6}


def test_pooled_rates_are_sums_over_sums_and_the_median_is_per_analysis() -> None:
    pooled = sweep.pool([A, B])
    assert pooled["spurious_split_rate"] == pytest.approx(6 / 40)  # not (1/10 + 5/30) / 2
    assert pooled["lost_signal_rate"] == pytest.approx(10 / 60)
    assert pooled["disagreement_rate"] == pytest.approx(16 / 100)
    assert pooled["median_spurious_split_rate"] == pytest.approx((1 / 10 + 5 / 30) / 2)
    assert pooled["median_lost_signal_rate"] == pytest.approx((4 / 20 + 6 / 40) / 2)


def _each(splits: int, lost: int, n: int = 10) -> dict[str, int]:
    return {"locus_leads": n, "index_variants": n, "spurious_splits": splits, "lost_signals": lost}


def test_clearly_better_needs_a_ten_percent_cut_and_two_thirds_of_the_analyses() -> None:
    bp = [_each(1, 7)] * 3
    assert sweep.clearly_better([_each(2, 3)] * 3, bp)  # 5 against 8 per 20, lower in 3 of 3
    # 5% lower pooled, lower in every Analysis: not clearly better.
    near_bp = [_each(10, 30, 100)] * 3
    assert not sweep.clearly_better([_each(10, 28, 100)] * 3, near_bp)
    # 25% lower pooled, but lower in only 1 of 3 Analyses.
    assert not sweep.clearly_better([_each(0, 0), _each(4, 5), _each(4, 5)], [_each(4, 4)] * 3)


def test_a_config_qualifies_only_when_both_rates_are_within_ten_percent_and_it_beats_bp() -> None:
    good = {"locus_leads": 100, "index_variants": 100, "spurious_splits": 10, "lost_signals": 10}
    bad = {"locus_leads": 100, "index_variants": 100, "spurious_splits": 11, "lost_signals": 10}
    lost_bad = {"locus_leads": 100, "index_variants": 100, "spurious_splits": 10,
                "lost_signals": 11}
    assert sweep.within_tolerance(sweep.pool([good]))
    assert not sweep.within_tolerance(sweep.pool([bad]))
    assert not sweep.within_tolerance(sweep.pool([lost_bad]))


def _summary(n: int, near: int, straddling: int) -> dict[str, Any]:
    band = {"n": n, "r2_ge_0.01": near, "r2_ge_0.001": near}
    return {"n": n, "r2_undefined": 0, "r2_histogram": {"[0,0.001)": 0, "[0.001,0.01)": n - near},
            "by_distance_band": {"<=1Mb": band, "1-2.5Mb": {"n": 0, "r2_ge_0.01": 0,
                                                            "r2_ge_0.001": 0}},
            "straddling_a_spike": straddling, "straddling": {"TSI": straddling}}


def test_pairs_pool_by_summing_counts_across_analyses() -> None:
    pooled = sweep.pool_pairs([_summary(4, 3, 1), _summary(6, 1, 0)])
    assert pooled["n"] == 10 and pooled["straddling_a_spike"] == 1
    assert pooled["by_distance_band"]["<=1Mb"] == {"n": 10, "r2_ge_0.01": 4, "r2_ge_0.001": 4}
    assert pooled["r2_histogram"]["[0.001,0.01)"] == 6


def test_chance_ld_pools_pairs_and_reports_the_share_at_each_threshold() -> None:
    pooled = sweep.pool_chance([
        {"pairs": 100, "pairs_r2_ge_0.01": 3, "pairs_r2_ge_0.001": 20, "pairs_r2_ge_0.05": 1,
         "pairs_r2_ge_0.1": 0, "r2_undefined": 0},
        {"pairs": 300, "pairs_r2_ge_0.01": 5, "pairs_r2_ge_0.001": 40, "pairs_r2_ge_0.05": 2,
         "pairs_r2_ge_0.1": 1, "r2_undefined": 0}])
    assert pooled["pairs"] == 400
    assert pooled["share_r2_ge_0.01"] == pytest.approx(8 / 400)
    assert pooled["share_r2_ge_0.001"] == pytest.approx(60 / 400)
    assert pooled["share_r2_ge_0.1"] == pytest.approx(1 / 400)
