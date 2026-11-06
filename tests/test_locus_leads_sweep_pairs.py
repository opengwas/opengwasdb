"""Pairs whose r^2 is undefined are counted beside the rest, never as zero (#403)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from prototype_modules import load_prototype

genetic_map = load_prototype("locus_leads", "genetic_map")
pairs = load_prototype("locus_leads", "pairs")
sweep = load_prototype("locus_leads", "sweep")


def _chance(pairs_total: int, undefined: int, reaching: int) -> dict[str, int]:
    return {"pairs": pairs_total, "r2_undefined": undefined, "pairs_r2_ge_0.001": reaching,
            "pairs_r2_ge_0.01": reaching, "pairs_r2_ge_0.05": 0, "pairs_r2_ge_0.1": 0}


def test_chance_shares_are_over_pairs_with_a_defined_r2() -> None:
    # One perfectly correlated pair and two undefined ones: the share is 1 of 1, not 1 of 3.
    pooled = sweep.pool_chance([_chance(3, 2, 1)])
    assert pooled["pairs_evaluable"] == 1
    assert pooled["share_r2_ge_0.01"] == pytest.approx(1.0)
    assert (pooled["pairs"], pooled["r2_undefined"]) == (3, 2)


def test_chance_shares_pool_defined_pairs_across_analyses() -> None:
    pooled = sweep.pool_chance([_chance(10, 0, 1), _chance(10, 8, 1)])
    assert pooled["share_r2_ge_0.01"] == pytest.approx(2 / 12)


def test_no_pair_with_a_defined_r2_is_refused_not_reported_as_a_share() -> None:
    with pytest.raises(ValueError, match="no pair with a defined r"):
        sweep.pool_chance([_chance(5, 5, 0)])


def test_a_split_whose_r2_is_undefined_fails_the_threshold_check() -> None:
    members = [pairs.Pair("1", (100, 200), 0.1, 0.5), pairs.Pair("1", (100, 300), 0.2, math.nan)]
    with pytest.raises(ValueError, match="undefined r"):
        pairs.check_pairs_follow_threshold(members, 0.01)


def test_a_split_below_the_threshold_still_fails() -> None:
    with pytest.raises(ValueError, match="below"):
        pairs.check_pairs_follow_threshold([pairs.Pair("1", (100, 200), 0.1, 0.001)], 0.01)


def test_each_distance_band_counts_its_undefined_pairs() -> None:
    members = [pairs.Pair("1", (100, 200), 0.1, 0.5), pairs.Pair("1", (100, 300), 0.2, math.nan)]
    band = pairs.summarise(members, ())["by_distance_band"]["<=1Mb"]
    assert (band["n"], band["r2_undefined"], band["r2_ge_0.01"]) == (2, 1, 1)


def _map(positions: list[int], cm: list[float]) -> object:
    return genetic_map.GeneticMap({"7": (np.array(positions), np.array(cm))})


def test_a_spikes_size_is_the_jump_from_its_point_to_the_next_in_its_population_map() -> None:
    population = _map([1000, 2000, 2444, 5000], [0.0, 1.0, 93.7, 94.0])
    mean = _map([1000, 2000, 2444, 5000], [0.0, 0.5, 19.1, 19.4])
    record = pairs.spike_record(pairs.Spike("7", 2000, "TSI"), population, mean)
    assert record["interval_bp"] == 444 and record["interval_end"] == 2444
    assert record["population_cm"] == pytest.approx(92.7)
    assert record["mean_map_cm"] == pytest.approx(18.6)
    assert (record["chromosome"], record["position"], record["population"]) == ("7", 2000, "TSI")


def test_a_spike_that_is_not_a_point_of_its_population_map_is_refused() -> None:
    population = _map([1000, 2000, 5000], [0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="not a point"):
        pairs.spike_record(pairs.Spike("7", 2001, "TSI"), population, population)
    with pytest.raises(ValueError, match="last point"):
        pairs.spike_record(pairs.Spike("7", 5000, "TSI"), population, population)
