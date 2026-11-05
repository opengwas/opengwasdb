"""A genetic map, and cM interpolated from it at Store positions.

cM is linear between map points and held flat beyond a chromosome's first and
last points. A chromosome the map does not cover gets NaN, never a substituted
value: a lead on chrX must not look like one on a cold stretch of an autosome
(#62's proposal, "Where cM comes from").
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


class GeneticMapError(ValueError):
    """The map cannot be trusted as a map: cM falls, or positions repeat."""


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
