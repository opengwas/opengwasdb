"""Chance LD at a panel's own sample size, from the pooled shared sets (#405).

On synthetic inputs: the pairs drawn are cross-chromosome and seeded, the share
is of pairs whose r^2 is defined, and the variant pool is capped by the seed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from prototype_modules import load_prototype

chance = load_prototype("locus_leads", "panel_chance")


def test_the_pairs_drawn_are_cross_chromosome_and_seeded() -> None:
    labels = np.array(["1", "1", "2", "2", "2", "3"], dtype=object)

    first, second = chance.sample_pairs(labels, 50, seed=405)

    assert first.size == second.size == 50
    assert np.all(labels[first] != labels[second]), "a within-chromosome pair is not chance LD"
    again = chance.sample_pairs(labels, 50, seed=405)
    np.testing.assert_array_equal(again[0], first)
    np.testing.assert_array_equal(again[1], second)
    assert not np.array_equal(chance.sample_pairs(labels, 50, seed=406)[0], first)


def test_a_pool_on_one_chromosome_cannot_give_a_cross_chromosome_pair() -> None:
    with pytest.raises(ValueError, match="cross-chromosome"):
        chance.sample_pairs(np.array(["7", "7"], dtype=object), 5, seed=405)


def test_the_share_is_of_pairs_with_a_defined_r2_and_the_rest_are_counted() -> None:
    shares = chance.summarise(np.array([0.005, 0.02, np.nan, 0.5]))

    assert shares["pairs_sampled"] == 4
    assert shares["r2_undefined"] == 1
    assert shares["pairs_evaluable"] == 3
    assert shares["pairs_r2_ge_0.01"] == 2
    assert shares["share_r2_ge_0.01"] == pytest.approx(2 / 3)  # not 2/4
    assert shares["share_r2_ge_0.001"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="no pair"):
        chance.summarise(np.array([np.nan, np.nan]))


def test_at_most_the_cap_is_used_and_the_seed_chooses_which_variants() -> None:
    pooled = chance.Pooled(np.array([str(1 + i % 3) for i in range(100)], dtype=object),
                           np.array([f"1:{i}:A:G" for i in range(100)], dtype=object), 7)

    used = chance.sample_variants(pooled, 10, seed=405)

    assert len(used) == 10 and used.analyses == 7
    assert set(used.panel_id) <= set(pooled.panel_id)
    assert list(used.panel_id) == list(chance.sample_variants(pooled, 10, seed=405).panel_id)
    assert len(chance.sample_variants(pooled, 200, seed=405)) == 100


def test_one_file_carries_every_ancestry_and_its_seed_is_fixed() -> None:
    first = chance.merge_chance(None, "AFR", {"share_r2_ge_0.01": 0.1})
    both = chance.merge_chance(first, "AMR", {"share_r2_ge_0.01": 0.2})

    assert list(both["ancestries"]) == ["AFR", "AMR"]
    assert both["seed"] == chance.SEED == 405
    replaced = chance.merge_chance(both, "AFR", {"share_r2_ge_0.01": 0.3})
    assert replaced["ancestries"]["AFR"] == {"share_r2_ge_0.01": 0.3}
    assert json.dumps(both, sort_keys=True) == json.dumps(
        chance.merge_chance(chance.merge_chance(None, "AMR", {"share_r2_ge_0.01": 0.2}),
                            "AFR", {"share_r2_ge_0.01": 0.1}), sort_keys=True)
    with pytest.raises(ValueError, match="seed 405"):
        chance.merge_chance(both, "EAS", {}, seed=403)
