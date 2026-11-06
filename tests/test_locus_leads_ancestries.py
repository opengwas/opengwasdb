"""The pyrho average generalised from EUR to any ancestry's populations (#405).

The EUR map is #403's committed file, so the generalised code is held to the
EUR path's bytes on the same archive, and the averaging is given a
three-population set worked by hand.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from prototype_modules import load_prototype

ancestries = load_prototype("locus_leads", "ancestries")
eur_map = load_prototype("locus_leads", "eur_map")
genetic_map = load_prototype("locus_leads", "genetic_map")

AUTOSOMES = tuple(str(number) for number in range(1, 23))
# One chromosome's points per population, and the same trivial map elsewhere:
# cM rising one per 1,000 bp, so any chromosome the archive holds is well formed.
ELSEWHERE = [(1_000, 0.0), (2_000, 1.0)]
MEMBER = ("hg38/{p}/{p}_recombination_map_hapmap_format_hg38_chr_{c}.txt")


def _hapmap_text(points: list[tuple[int, float]], chromosome: str) -> bytes:
    rows = [f"chr{chromosome}\t{position}\t1.0\t{cm!r}" for position, cm in points]
    return ("Chromosome\tPosition(bp)\tRate(cM/Mb)\tMap(cM)\n" + "\n".join(rows) + "\n").encode()


def write_archive(path: Path, populations: dict[str, list[tuple[int, float]]],
                  omit: dict[str, tuple[str, ...]] | None = None) -> Path:
    """A stand-in for `hg38_maps.tar.gz`: each population's chr1 to chr22 HapMap maps.

    `omit` names the chromosomes a population is left without, to stand in for a
    population the archive holds only in part.
    """
    missing = omit or {}
    with tarfile.open(path, "w:gz") as archive:
        for population, chr1 in populations.items():
            for chromosome in AUTOSOMES:
                if chromosome in missing.get(population, ()):
                    continue
                text = _hapmap_text(chr1 if chromosome == "1" else ELSEWHERE, chromosome)
                info = tarfile.TarInfo(MEMBER.format(p=population, c=chromosome))
                info.size = len(text)
                archive.addfile(info, io.BytesIO(text))
    return path


def _md5(path: Path) -> str:
    return genetic_map.file_digest(path, "md5")


def _written_cm(map_path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(map_path, sep="\t", dtype={"chromosome": str})
    chromosome_1 = frame[frame["chromosome"] == "1"]
    return chromosome_1["position"].to_numpy(), chromosome_1["cM"].to_numpy()


# Three populations whose points sit at different places on chromosome 1. A is
# flat before 1,000 and after 3,000; B runs on to 4,000; C starts at 500 and
# stops at 2,500.
THREE = {
    "AAA": [(1_000, 0.0), (3_000, 2.0)],
    "BBB": [(1_000, 0.0), (2_000, 1.0), (4_000, 3.0)],
    "CCC": [(500, 0.5), (2_500, 1.5)],
}


def test_the_average_of_three_populations_matches_cm_worked_by_hand(tmp_path: Path) -> None:
    archive = write_archive(tmp_path / "hg38_maps.tar.gz", THREE)
    map_path = tmp_path / "THREE.tsv.gz"

    identity = eur_map.build_population_map(archive, map_path, ("AAA", "BBB", "CCC"),
                                            expected_md5=_md5(archive))

    position, cm = _written_cm(map_path)
    np.testing.assert_array_equal(position, [500, 1_000, 2_000, 2_500, 3_000, 4_000])
    # At each union position, (AAA, BBB, CCC) and their mean:
    #   500: 0, 0, 0.5 -> 1/6        1,000: 0, 0, 0.75 -> 0.25
    #   2,000: 1, 1, 1.25 -> 13/12   2,500: 1.5, 1.5, 1.5 -> 1.5
    #   3,000: 2, 2, 1.5 -> 11/6     4,000: 2, 3, 1.5 -> 13/6
    np.testing.assert_allclose(cm, [1 / 6, 0.25, 13 / 12, 1.5, 11 / 6, 13 / 6], atol=1e-9)
    assert identity["populations"] == {"AAA": pytest.approx(1 / 3), "BBB": pytest.approx(1 / 3),
                                       "CCC": pytest.approx(1 / 3)}
    assert json.loads((tmp_path / "THREE.json").read_text())["populations"] == identity[
        "populations"]


def test_the_generalised_average_rebuilds_the_eur_map_byte_for_byte(tmp_path: Path) -> None:
    archive = write_archive(tmp_path / "hg38_maps.tar.gz",
                            {population: [(100, 0.0), (900, 1.5)]
                             for population in eur_map.EUR_POPULATIONS})
    asked, generalised = tmp_path / "EUR.tsv.gz", tmp_path / "GENERAL.tsv.gz"

    eur_map.build_eur_map(archive, asked, expected_md5=_md5(archive))
    eur_map.build_population_map(archive, generalised, eur_map.EUR_POPULATIONS,
                                 expected_md5=_md5(archive))

    assert generalised.read_bytes() == asked.read_bytes()
    identities = [json.loads((tmp_path / f"{name}.json").read_text())
                  for name in ("EUR", "GENERAL")]
    assert identities[1] | {"map": "EUR.tsv.gz"} == identities[0]


def test_a_population_the_archive_lacks_is_refused_by_name(tmp_path: Path) -> None:
    archive = write_archive(tmp_path / "hg38_maps.tar.gz",
                            {"AAA": ELSEWHERE, "BBB": ELSEWHERE, "CCC": ELSEWHERE})

    with pytest.raises(genetic_map.GeneticMapError, match="no map for population DDD"):
        eur_map.build_population_map(archive, tmp_path / "X.tsv.gz", ("AAA", "DDD"),
                                     expected_md5=_md5(archive))


def test_a_population_missing_a_chromosome_is_refused_by_name(tmp_path: Path) -> None:
    archive = write_archive(tmp_path / "hg38_maps.tar.gz",
                            {"AAA": ELSEWHERE, "BBB": ELSEWHERE}, omit={"BBB": ("7",)})

    with pytest.raises(genetic_map.GeneticMapError, match="no map for BBB chr7"):
        eur_map.build_population_map(archive, tmp_path / "X.tsv.gz", ("AAA", "BBB"),
                                     expected_md5=_md5(archive))


def test_each_ancestry_averages_the_populations_the_archive_holds_for_it() -> None:
    assert ancestries.populations("AFR") == ("YRI", "LWK", "GWD", "MSL", "ESN", "ACB", "ASW")
    assert ancestries.populations("EAS") == ("CHB", "JPT", "CHS", "CDX", "KHV")
    assert ancestries.populations("AMR") == ("MXL", "PUR", "CLM", "PEL")
    assert ancestries.populations("EUR") == eur_map.EUR_POPULATIONS

    with pytest.raises(ancestries.UnknownAncestry, match="SAS.*AFR, AMR, EAS, EUR"):
        ancestries.populations("SAS")
