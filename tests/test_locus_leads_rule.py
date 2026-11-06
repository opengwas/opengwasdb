"""The Locus Lead rule (prototypes/locus_leads, #402).

A row is a Locus Lead unless an already-accepted lead, taken in descending
abs_z with ties broken by ascending Store Variant Index, lies within W bp and
D cM of it.
"""

from __future__ import annotations

import numpy as np
from prototype_modules import load_prototype

genetic_map = load_prototype("locus_leads", "genetic_map")
rule = load_prototype("locus_leads", "rule")

D_CM = 0.5
W_BP = 1_000_000

# Chromosome 1 is cold (about 0.1 cM/Mb) except for a hotspot that adds 2 cM
# between 1,499,000 and 1,501,000.
HOTSPOT_MAP = genetic_map.GeneticMap(
    {"1": (np.array([1, 1_499_000, 1_501_000, 10_000_000]), np.array([0.0, 0.15, 2.15, 3.0]))}
)


def _leads(chromosome: list[str], position: list[int], cm: np.ndarray,
           abs_z: list[float], variant_index: list[int]) -> np.ndarray:
    result: np.ndarray = rule.locus_leads(
        np.array(chromosome), np.array(position), cm, np.array(abs_z),
        np.array(variant_index), d_cm=D_CM, w_bp=W_BP,
    )
    return result


def test_hits_across_a_hotspot_keep_two_leads_and_the_same_spacing_in_a_cold_region_one() -> None:
    chromosome = ["1"] * 4
    # Two pairs, each 100 kb apart: one straddles the hotspot, one is cold.
    position = [1_450_000, 1_550_000, 5_450_000, 5_550_000]
    cm = HOTSPOT_MAP.cm_at(np.array(chromosome), np.array(position))
    hotspot_gap, cold_gap = cm[1] - cm[0], cm[3] - cm[2]
    assert len(position) == 4 and position[1] - position[0] == position[3] - position[2]
    assert hotspot_gap > 2.0 > D_CM, "fixture must straddle the hotspot"
    assert cold_gap < 0.02, "fixture's second pair must sit in the cold region"

    lead_of = _leads(chromosome, position, cm, [9.0, 8.0, 9.0, 8.0], [10, 11, 12, 13])

    # Rows 0 and 1 lead their own loci; row 3 merges into row 2.
    np.testing.assert_array_equal(lead_of, [0, 1, 2, 2])


def test_w_caps_merging_where_cm_barely_moves_and_chromosomes_never_merge() -> None:
    chromosome = ["1", "1", "2"]
    # Rows 0 and 1 are 2 Mb apart in the cold stretch, about 0.2 cM.
    position = [3_000_000, 5_000_000, 3_000_000]
    cm = np.array([1.0, 1.2, 1.0])

    lead_of = _leads(chromosome, position, cm, [9.0, 8.0, 7.0], [1, 2, 3])

    np.testing.assert_array_equal(lead_of, [0, 1, 2])


def test_equal_abs_z_is_broken_by_ascending_store_variant_index() -> None:
    # Rows 0 and 1 tie on abs_z; row 1 has the smaller Store Variant Index.
    # Row 2 has the smallest index of all but a lower abs_z, so it cannot lead.
    chromosome = ["1"] * 3
    position = [2_000_000, 2_000_100, 2_000_200]
    cm = np.array([0.5, 0.5, 0.5])

    lead_of = _leads(chromosome, position, cm, [7.0, 7.0, 6.0], [30, 12, 5])

    np.testing.assert_array_equal(lead_of, [1, 1, 1])
