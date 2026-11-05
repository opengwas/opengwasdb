"""The EUR genetic map, averaged from the published pyrho population maps.

The pyrho GRCh38 maps (Spence & Song 2019) cover the 26 1000 Genomes
populations. EUR is the equal-weight average of its five: CEU, GBR, FIN, TSI
and IBS (#62's ancestry-specific revision). Their points sit at different
positions, so each is interpolated at the union of all five's positions, by the
same rule a Store position is, and the cM values are averaged there.
"""

from __future__ import annotations

import re
import tarfile
from collections.abc import Iterator, Mapping
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .genetic_map import GeneticMap, GeneticMapError, verify_archive, write_map

EUR_POPULATIONS = ("CEU", "GBR", "FIN", "TSI", "IBS")
AUTOSOMES = tuple(str(number) for number in range(1, 23))
AVERAGING = (
    "equal-weight mean of the population maps' cM, each interpolated (linear between "
    "its points, flat beyond its ends) at the union of their positions"
)

_MEMBER = re.compile(
    r"hg38/(?P<population>[A-Z]+)/(?P=population)"
    r"_recombination_map_hapmap_format_hg38_chr_(?P<chromosome>\w+)\.txt$"
)


def build_eur_map(archive: Path, map_path: Path) -> dict[str, Any]:
    """Check the archive, average its EUR population maps, and write the result."""
    verify_archive(archive)
    maps = read_population_maps(archive, EUR_POPULATIONS)
    weights = {population: 1 / len(maps) for population in maps}
    return write_map(average_maps(maps), map_path, archive=archive, populations=weights,
                     averaging=AVERAGING)


def read_population_maps(archive: Path, populations: tuple[str, ...]) -> dict[str, GeneticMap]:
    """Each population's autosome maps, read in one pass through the archive."""
    points: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {p: {} for p in populations}
    for population, chromosome, map_points in _population_files(archive, populations):
        points[population][chromosome] = map_points
    missing = [f"{p} chr{c}" for p in populations for c in AUTOSOMES if c not in points[p]]
    if missing:
        raise GeneticMapError(f"{archive}: no map for {', '.join(missing)}")
    return {population: GeneticMap(points[population]) for population in populations}


def _population_files(
    archive: Path, populations: tuple[str, ...]
) -> Iterator[tuple[str, str, tuple[np.ndarray, np.ndarray]]]:
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            match = _MEMBER.search(member.name)
            if match is None or match["population"] not in populations:
                continue
            handle = stream.extractfile(member)
            if handle is None:
                raise GeneticMapError(f"{archive}: {member.name} is not a file")
            chromosome = match["chromosome"]
            yield match["population"], chromosome, _hapmap_points(
                handle.read(), chromosome, member.name
            )


def average_maps(maps: Mapping[str, GeneticMap]) -> GeneticMap:
    """The equal-weight mean of `maps`, on the union of their positions."""
    averaged = {}
    for chromosome in AUTOSOMES:
        each = [genetic_map.points(chromosome) for genetic_map in maps.values()]
        union = np.unique(np.concatenate([position for position, _ in each]))
        cm = np.mean([np.interp(union, position, cm) for position, cm in each], axis=0)
        averaged[chromosome] = (union, cm)
    return GeneticMap(averaged)


def _hapmap_points(text: bytes, chromosome: str, name: str) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(BytesIO(text), sep="\t")
    labels = set(frame["Chromosome"].astype(str))
    if labels != {f"chr{chromosome}"}:
        raise GeneticMapError(f"{name}: chromosome column holds {sorted(labels)}")
    return frame["Position(bp)"].to_numpy(), frame["Map(cM)"].to_numpy()
