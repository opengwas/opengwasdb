"""Building the comparison genetic maps (prototypes/locus_leads, #404).

#403 swept D and W against the equal-weight mean of the five pyrho EUR
population maps. #404 compares it with HapMap II lifted to GRCh38 as Eagle ships
it and with the median of the same five maps, so both have to arrive in the one
layout the sweep reads: `chromosome position cM`, autosomes only, cM
non-decreasing within each chromosome.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from prototype_modules import load_prototype

genetic_map = load_prototype("locus_leads", "genetic_map")
maps = load_prototype("locus_leads", "maps")

# Eagle's reference map: four columns, the rate column unused, chrX spelled 23.
EAGLE_TEXT = """\
chr position COMBINED_rate(cM/Mb) Genetic_Map(cM)
1 55550 0.0000000000 0.000000000000000
1 632942 0.0000000000 0.000000000000000
1 785910 2.6858076690 0.410292036939447
22 16050000 1.0000000000 3.250000000000000
23 100000 2.0000000000 4.500000000000000
"""


def test_eagles_four_column_map_normalises_to_position_and_cm_per_autosome() -> None:
    parsed = maps.parse_points(EAGLE_TEXT, maps.EAGLE_HAPMAP2)

    assert list(parsed.points) == ["1", "22"], "chrX is not an autosome and is kept out"
    np.testing.assert_array_equal(parsed.points["1"][0], [55550, 632942, 785910])
    np.testing.assert_array_equal(parsed.points["1"][1], [0.0, 0.0, 0.410292036939447])
    np.testing.assert_array_equal(parsed.points["22"][0], [16_050_000])
    np.testing.assert_array_equal(parsed.points["22"][1], [3.25])
    assert parsed.excluded == {"23": 1}, "the chrX points are counted, not silently dropped"


def test_a_map_in_another_layout_is_refused_rather_than_read_by_column_number() -> None:
    # Beagle's three-column spelling of the same map, which lacks the rate column.
    other = "chr position Genetic_Map(cM)\n1 55550 0.0\n"

    with pytest.raises(maps.MapBuildError, match="COMBINED_rate"):
        maps.parse_points(other, maps.EAGLE_HAPMAP2)


def test_a_map_label_that_is_no_known_chromosome_is_refused() -> None:
    # `chr1` instead of `1`: read leniently, a differently spelled file is an
    # empty map, or one with a chromosome silently missing.
    prefixed = EAGLE_TEXT.replace("\n1 ", "\nchr1 ")

    with pytest.raises(maps.MapBuildError, match="'chr1'"):
        maps.parse_points(prefixed, maps.EAGLE_HAPMAP2)


def test_every_autosome_must_be_present_or_the_map_is_refused() -> None:
    parsed = maps.parse_points(EAGLE_TEXT, maps.EAGLE_HAPMAP2)

    with pytest.raises(maps.MapBuildError, match="missing"):
        maps.autosome_map(parsed, name="eagle")


def _two_inversions() -> tuple[np.ndarray, np.ndarray]:
    return (np.array([100, 200, 300, 400, 500, 600]),
            np.array([0.0, 1.0, 0.9, 1.2, 1.1, 1.5]))


def test_the_points_that_fall_below_the_running_maximum_are_the_ones_dropped() -> None:
    position, cm = _two_inversions()

    kept, dropped = maps.drop_decreasing(position, cm)

    np.testing.assert_array_equal(kept[0], [100, 200, 400, 600])
    np.testing.assert_array_equal(kept[1], [0.0, 1.0, 1.2, 1.5])
    np.testing.assert_array_equal(dropped, [300, 500])
    assert maps.drop_decreasing(position, np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))[1].size == 0


def test_a_chromosome_that_would_lose_more_than_a_tenth_of_a_percent_is_refused() -> None:
    # 1,000 points on a clean map, with two inversions in it: 2 of 1,000 is 0.2%,
    # past the 0.1% a lift may cost.
    position = np.arange(1, 1_001, dtype="int64") * 1_000
    cm = np.arange(1_000, dtype="float64") * 0.1
    broken = cm.copy()
    broken[500], broken[900] = cm[499] - 0.05, cm[899] - 0.05

    with pytest.raises(maps.MapBuildError, match="chromosome 1: 2 of 1,000"):
        maps.build_genetic_map({"1": (position, broken)})
    # One point is exactly 0.1% and is allowed: the limit is `more than`.
    one = cm.copy()
    one[500] = cm[499] - 0.05
    built, dropped = maps.build_genetic_map({"1": (position, one)})
    assert dropped.counts() == {"1": 1} and built.points("1")[0].size == 999


def _loaded(one: object) -> object:
    return genetic_map.LoadedMap(one, {"sha256": "0", "assembly": "GRCh38"})


def _population(positions: list[int], cms: list[float]) -> object:
    return genetic_map.GeneticMap({"1": (np.array(positions), np.array(cms))})


def _one_spike_at_the_middle_knot() -> dict[str, object]:
    """Four maps running to 40, 40, 20 and 20 cM, and one that jumps 40 cM midway."""
    return {
        "A": _population([100, 200], [0.0, 40.0]),
        "B": _population([100, 200], [0.0, 40.0]),
        "C": _population([100, 200], [0.0, 20.0]),
        "D": _population([100, 200], [0.0, 20.0]),
        "E": _population([100, 150, 200], [0.0, 40.0, 50.0]),
    }


def test_the_median_map_is_the_median_rate_integrated_not_the_median_cM() -> None:
    populations = _one_spike_at_the_middle_knot()

    median = maps.median_map(populations)

    position, cm = median.points("1")
    np.testing.assert_array_equal(position, [100, 150, 200])
    # Each interval is 50 bp. In the second the five rates are 0.4, 0.4, 0.2, 0.2
    # and 0.2 cM/bp, so the median rate is 0.2 and the interval gains 10 cM on
    # the 20 the first gained.
    np.testing.assert_allclose(cm, [0.0, 20.0, 30.0])
    # The mutation: the median of cumulative cM ends at 40, because at the last
    # knot the middle of {40, 40, 20, 20, 50} is A's or B's value — the middle
    # winner changes at E's jump, and the step is a whole population's distance.
    interpolated = np.array([np.interp(position, *map_points)
                             for map_points in
                             (populations[name].points("1") for name in populations)])
    np.testing.assert_allclose(np.median(interpolated, axis=0), [0.0, 20.0, 40.0])


def test_the_median_map_starts_at_the_median_cM_of_the_first_knot() -> None:
    populations = {"A": _population([100, 200], [1.0, 2.0]),
                   "B": _population([100, 200], [2.0, 3.0]),
                   "C": _population([100, 200], [3.0, 4.0])}

    position, cm = maps.median_map(populations).points("1")

    np.testing.assert_array_equal(position, [100, 200])
    np.testing.assert_allclose(cm, [2.0, 3.0])


def test_one_populations_spike_moves_the_mean_and_not_the_median_rate() -> None:
    flat = [([100, 200, 300], [0.0, 1.0, 2.0])] * 4
    spiked = ([100, 150, 200, 300], [0.0, 21.0, 22.0, 23.0])
    populations = {name: _population(*points) for name, points in
                   zip(["CEU", "GBR", "FIN", "TSI", "IBS"], [*flat, spiked], strict=True)}

    median = maps.median_map(populations).points("1")
    mean = load_prototype("locus_leads", "eur_map").average_maps(populations).points("1")

    # The spike is one map of five: at 150 the other four interpolate to 0.5, so
    # the mean carries a fifth of it and the median rate carries none of it.
    np.testing.assert_array_equal(median[0], [100, 150, 200, 300])
    np.testing.assert_allclose(median[1], [0.0, 0.5, 1.0, 2.0])
    assert mean[1][1] == pytest.approx(0.5 + (21.0 - 0.5) / 5)


def test_the_median_map_refuses_population_maps_covering_different_chromosomes() -> None:
    both = {"CEU": _population([100, 300], [0.0, 2.0]),
            "GBR": genetic_map.GeneticMap({"1": (np.array([100, 300]), np.array([0.0, 2.0])),
                                           "2": (np.array([100, 300]), np.array([0.0, 2.0]))})}

    with pytest.raises(genetic_map.GeneticMapError, match="different chromosomes"):
        maps.median_map(both)


def test_a_source_map_round_trips_through_its_sidecar_and_source_checksum(tmp_path: Path) -> None:
    directory = tmp_path / "eagle-hapmap2-hg38"
    original = directory / "original" / "genetic_map_hg38_withX.txt.gz"
    original.parent.mkdir(parents=True)
    original.write_text(EAGLE_TEXT)
    parsed = maps.parse_points(EAGLE_TEXT, maps.EAGLE_HAPMAP2)
    # The snippet holds two autosomes; the other twenty are filled so the map
    # is complete, as a real download's is.
    points = parsed.points | {chromosome: (np.array([1_000, 2_000]), np.array([0.0, 1.0]))
                              for chromosome in maps.AUTOSOMES
                              if chromosome not in parsed.points}
    map_path = directory / "HapMap2.tsv.gz"

    identity = maps.write_source_map(points, map_path, provenance={"source": {
        "path": "original/genetic_map_hg38_withX.txt.gz", "url": "https://example.invalid",
        "retrieved": "2026-11-06", "licence": "public",
        "sha256": hashlib.sha256(EAGLE_TEXT.encode()).hexdigest()}}, excluded=parsed.excluded)

    assert identity["points_per_chromosome"]["1"] == 3
    assert identity["excluded_chromosomes"] == {"23": 1}
    loaded = genetic_map.load_map(map_path)
    # The file holds nine decimals, as the pyrho map's does.
    np.testing.assert_allclose(loaded.map.points("1")[1], [0.0, 0.0, 0.410292037])
    original.write_text(EAGLE_TEXT.replace("0.410292036939447", "0.410292036939448"))
    with pytest.raises(genetic_map.SourceChecksumError, match="sha256"):
        genetic_map.load_map(map_path)


def test_a_spike_interval_is_measured_from_the_map_and_a_position_off_it_refuses() -> None:
    one = genetic_map.GeneticMap({"7": (np.array([100, 200, 300]), np.array([0.0, 1.0, 3.0]))})

    assert maps.interval_jump(one, "7", 200) == {
        "chromosome": "7", "position": 200, "interval_start": 200, "interval_end": 300,
        "interval_bp": 100, "cm_gain": 2.0, "at_a_map_point": True}
    # A map with its own points has no interval starting at 250; the one that
    # contains it is the honest measure there.
    assert maps.interval_jump(one, "7", 250) == {
        "chromosome": "7", "position": 250, "interval_start": 200, "interval_end": 300,
        "interval_bp": 100, "cm_gain": 2.0, "at_a_map_point": False}
    assert maps.map_measurements({"one": _loaded(one)}, spikes=())["one"] == {
        "chromosomes": 1, "points": 3, "total_cM": 3.0, "spike_intervals": []}
    with pytest.raises(maps.MapBuildError, match="no interval after it"):
        maps.interval_jump(one, "7", 300)
    with pytest.raises(maps.MapBuildError, match="no interval after it"):
        maps.interval_jump(one, "7", 50)


def test_a_map_whose_identity_names_no_source_is_refused(tmp_path: Path) -> None:
    directory = tmp_path / "eagle-hapmap2-hg38"
    original = directory / "original" / "genetic_map_hg38_withX.txt.gz"
    original.parent.mkdir(parents=True)
    original.write_text(EAGLE_TEXT)
    map_path = directory / "HapMap2.tsv.gz"
    points = {chromosome: (np.array([1_000, 2_000]), np.array([0.0, 1.0]))
              for chromosome in maps.AUTOSOMES}
    import json  # only this test writes a sidecar by hand

    identity = maps.write_source_map(points, map_path, provenance={}, excluded={})
    assert "source" not in identity and "archive" not in identity

    with pytest.raises(genetic_map.GeneticMapError, match="exactly one of"):
        genetic_map.load_map(map_path)
    # Two sources is as unreadable as none: which one is the map's provenance?
    identity["source"] = {"path": "original/genetic_map_hg38_withX.txt.gz",
                          "sha256": hashlib.sha256(EAGLE_TEXT.encode()).hexdigest()}
    identity["archive"] = identity["source"]
    genetic_map.sidecar_path(map_path).write_text(json.dumps(identity))
    with pytest.raises(genetic_map.GeneticMapError, match="exactly one of"):
        genetic_map.load_map(map_path)
    del identity["archive"]
    genetic_map.sidecar_path(map_path).write_text(json.dumps(identity))
    np.testing.assert_allclose(genetic_map.load_map(map_path).map.points("1")[1], [0.0, 1.0])
