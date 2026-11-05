"""A genetic map, and cM interpolated from it at Store positions.

cM is linear between map points and held flat beyond a chromosome's first and
last points. A chromosome the map does not cover gets NaN, never a substituted
value: a lead on chrX must not look like one on a cold stretch of an autosome
(#62's proposal, "Where cM comes from").
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# The pyrho GRCh38 maps, Zenodo record 11437540 (CC-BY 4.0), `hg38_maps.tar.gz`.
ARCHIVE_MD5 = "45ebb10c4acf939ea03a3932c483f638"
ARCHIVE_RECORD = "https://zenodo.org/records/11437540"
INTERPOLATION = "linear between map points; flat beyond a chromosome's ends; NaN off the map"


class GeneticMapError(ValueError):
    """The map cannot be trusted as a map: cM falls, or positions repeat."""


class ArchiveChecksumError(RuntimeError):
    """The map's source archive is not the one the record names."""


class GeneticMap:
    """Per chromosome, strictly increasing positions and non-decreasing cM."""

    def __init__(self, points: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> None:
        self._points: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for chromosome, (position, cm) in points.items():
            position = np.asarray(position, dtype="int64")
            cm = np.asarray(cm, dtype="float64")
            _check_chromosome(chromosome, position, cm)
            self._points[chromosome] = (position, cm)

    @property
    def chromosomes(self) -> list[str]:
        return list(self._points)

    def points(self, chromosome: str) -> tuple[np.ndarray, np.ndarray]:
        return self._points[chromosome]

    def cm_at(self, chromosome: np.ndarray, position: np.ndarray) -> np.ndarray:
        """cM at each (chromosome, position); NaN where the map lacks the chromosome."""
        chromosome = np.asarray(chromosome)
        position = np.asarray(position, dtype="int64")
        cm = np.full(position.shape, np.nan)
        for label, (map_position, map_cm) in self._points.items():
            on = chromosome == label
            # np.interp holds the end values flat beyond the first and last points.
            cm[on] = np.interp(position[on], map_position, map_cm)
        return cm


def _check_chromosome(chromosome: str, position: np.ndarray, cm: np.ndarray) -> None:
    if position.size == 0 or position.shape != cm.shape:
        raise GeneticMapError(f"chromosome {chromosome}: no points, or positions and cM differ")
    if not np.all(np.isfinite(cm)):
        raise GeneticMapError(f"chromosome {chromosome}: cM is not finite at every point")
    if np.any(np.diff(position) <= 0):
        raise GeneticMapError(f"chromosome {chromosome}: positions do not strictly increase")
    falls = np.flatnonzero(np.diff(cm) < 0)
    if falls.size:
        at = int(position[falls[0] + 1])
        raise GeneticMapError(
            f"chromosome {chromosome}: cM decreases with position at {at:,} "
            f"({falls.size:,} such steps); a misordered or wrongly lifted map"
        )


@dataclass(frozen=True)
class LoadedMap:
    map: GeneticMap
    identity: dict[str, Any]


def file_digest(path: Path, algorithm: str) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, algorithm).hexdigest()


def verify_archive(archive: Path, expected_md5: str = ARCHIVE_MD5) -> str:
    """The archive's md5, after refusing an archive whose md5 is not `expected_md5`."""
    md5 = file_digest(archive, "md5")
    if md5 != expected_md5:
        raise ArchiveChecksumError(
            f"{archive}: md5 {md5} differs from the record's {expected_md5} ({ARCHIVE_RECORD})"
        )
    return md5


def sidecar_path(map_path: Path) -> Path:
    if not map_path.name.endswith(".tsv.gz"):
        raise GeneticMapError(f"{map_path}: a map file is named <name>.tsv.gz")
    return map_path.with_name(map_path.name.removesuffix(".tsv.gz") + ".json")


def write_map(genetic_map: GeneticMap, map_path: Path, *, archive: Path,
              populations: Mapping[str, float], averaging: str = "") -> dict[str, Any]:
    """Write `<name>.tsv.gz` (chromosome, position, cM) and its `<name>.json` identity.

    The gzip header carries no time or name, so the same map gives the same
    bytes and the same sha256 on every build.
    """
    frame = pd.concat(
        [pd.DataFrame({"chromosome": label, "position": position, "cM": cm})
         for label in genetic_map.chromosomes
         for position, cm in [genetic_map.points(label)]],
        ignore_index=True,
    )
    with map_path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0,
                                                    filename="") as handle:
        handle.write(frame.to_csv(sep="\t", index=False, float_format="%.9f").encode())
    identity = {
        "map": map_path.name,
        "sha256": file_digest(map_path, "sha256"),
        "assembly": "GRCh38",
        "archive": {"path": str(archive.relative_to(map_path.parent)),
                    "md5": file_digest(archive, "md5"), "record": ARCHIVE_RECORD},
        "populations": dict(populations),
        "averaging": averaging,
        "interpolation": INTERPOLATION,
        "points_per_chromosome": {label: int(genetic_map.points(label)[0].size)
                                  for label in genetic_map.chromosomes},
    }
    sidecar_path(map_path).write_text(json.dumps(identity, indent=2) + "\n")
    return identity


def load_map(map_path: Path, *, archive_md5: str = ARCHIVE_MD5) -> LoadedMap:
    """The map at `map_path`, once its source archive and its own bytes check out."""
    identity: dict[str, Any] = json.loads(sidecar_path(map_path).read_text())
    verify_archive(map_path.parent / identity["archive"]["path"], archive_md5)
    sha256 = file_digest(map_path, "sha256")
    if sha256 != identity["sha256"]:
        raise GeneticMapError(
            f"{map_path}: sha256 {sha256} differs from {identity['sha256']} in its identity file"
        )
    frame = pd.read_csv(io.BytesIO(gzip.decompress(map_path.read_bytes())), sep="\t",
                        dtype={"chromosome": str, "position": "int64", "cM": "float64"})
    points = {str(label): (group["position"].to_numpy(), group["cM"].to_numpy())
              for label, group in frame.groupby("chromosome", sort=False)}
    return LoadedMap(GeneticMap(points), identity | {"path": str(map_path)})
