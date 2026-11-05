"""cM interpolation from a genetic map (prototypes/locus_leads, #402)."""

from __future__ import annotations

import numpy as np
import pytest
from prototype_modules import load_prototype

genetic_map = load_prototype("locus_leads", "genetic_map")


def _three_point_map() -> object:
    return genetic_map.GeneticMap(
        {"1": (np.array([100, 200, 400]), np.array([1.0, 2.0, 2.5]))}
    )


def test_cm_is_linear_between_points_flat_beyond_the_ends_and_nan_off_the_map() -> None:
    chromosome = np.array(["1", "1", "1", "1", "1", "X", "2"])
    position = np.array([150, 300, 200, 50, 1_000, 150, 150])

    cm = _three_point_map().cm_at(chromosome, position)

    # 150 is halfway from (100, 1.0) to (200, 2.0); 300 halfway from
    # (200, 2.0) to (400, 2.5); 50 and 1,000 lie beyond the ends.
    np.testing.assert_array_equal(cm[:5], [1.5, 2.25, 2.0, 1.0, 2.5])
    assert np.isnan(cm[5:]).all(), "a chromosome the map lacks gets NaN, never a value"


def test_a_map_whose_cm_decreases_with_position_is_refused() -> None:
    decreasing = {"7": (np.array([100, 200, 300]), np.array([1.0, 1.5, 1.4]))}

    with pytest.raises(genetic_map.GeneticMapError, match="chromosome 7"):
        genetic_map.GeneticMap(decreasing)
