"""cM interpolation from a genetic map (prototypes/locus_leads, #402)."""

from __future__ import annotations

from pathlib import Path

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


def _write_map_beside_archive(tmp_path: Path, archive_bytes: bytes) -> Path:
    archive = tmp_path / "original" / "hg38_maps.tar.gz"
    archive.parent.mkdir()
    archive.write_bytes(archive_bytes)
    map_path = tmp_path / "EUR.tsv.gz"
    genetic_map.write_map(_three_point_map(), map_path, archive=archive, populations={"CEU": 1.0})
    return map_path


def test_a_map_whose_source_archive_differs_from_the_record_fails_the_run(tmp_path: Path) -> None:
    map_path = _write_map_beside_archive(tmp_path, b"abc")

    # md5("abc") is 900150983cd24fb0d6963f7d28e17f72, not the Zenodo record's.
    with pytest.raises(genetic_map.ArchiveChecksumError, match="900150983cd24fb0d6963f7d28e17f72"):
        genetic_map.load_map(map_path)
    loaded = genetic_map.load_map(map_path, archive_md5="900150983cd24fb0d6963f7d28e17f72")
    assert loaded.map.cm_at(np.array(["1"]), np.array([150]))[0] == 1.5
    assert genetic_map.ARCHIVE_MD5 == "45ebb10c4acf939ea03a3932c483f638"


def test_the_eur_map_averages_population_maps_at_the_union_of_their_positions() -> None:
    eur_map = load_prototype("locus_leads", "eur_map")
    # Three population maps on chromosome 1 with different knots: the second
    # adds one at 200 and runs on to 400, the third starts late (150) and ends
    # early (250). Each is flat beyond its own ends.
    maps = {
        "P1": genetic_map.GeneticMap({"1": (np.array([100, 300]), np.array([0.0, 2.0]))}),
        "P2": genetic_map.GeneticMap({"1": (np.array([100, 200, 400]), np.array([0.0, 0.5, 1.5]))}),
        "P3": genetic_map.GeneticMap({"1": (np.array([150, 250]), np.array([0.0, 1.0]))}),
    }

    position, cm = eur_map.average_maps(maps).points("1")

    # Worked by hand, at each union position (P1, P2, P3 -> mean):
    #   100: 0,    0,    0   (P3 flat before 150)  -> 0
    #   150: 0.5,  0.25, 0                         -> 0.75/3 = 1/4
    #   200: 1.0,  0.5,  0.5                       -> 2/3
    #   250: 1.5,  0.75, 1.0                       -> 3.25/3 = 13/12
    #   300: 2.0,  1.0,  1.0 (P3 flat after 250)   -> 4/3
    #   400: 2.0 (P1 flat after 300), 1.5, 1.0     -> 3/2
    np.testing.assert_array_equal(position, [100, 150, 200, 250, 300, 400])
    np.testing.assert_allclose(cm, [0.0, 1 / 4, 2 / 3, 13 / 12, 4 / 3, 3 / 2], rtol=0, atol=1e-12)
