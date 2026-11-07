"""`LDStore`: the object a caller queries (spec §8, §9).

The four queries are methods of one opened store, because each needs the same state:
the reader, t_far and the distances the store measured. That keeps the interface to
`ogld.open` and four calls, while the reader, the codes and the refusal rules stay
behind it. Each method delegates to a module of its own, so the tickets that implement
them (ogld#4: matrices; ogld#6: proxies and clumping) edit different files.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy.typing as npt

from ogld.format import Provenance
from ogld.query import clumping, matrices
from ogld.query import proxies as proxies_query
from ogld.query.results import Clumps, LDMatrix, Proxies
from ogld.store import StoreReader


class LDStore:
    """An opened LD Store. Get one from `ogld.open`."""

    def __init__(self, reader: StoreReader) -> None:
        self._reader = reader

    @property
    def provenance(self) -> Provenance:
        """The store's provenance, as recorded (spec §7)."""
        return self._reader.provenance

    @property
    def t_far(self) -> float:
        """The Far Threshold this store recorded, the one every answer uses (spec §3)."""
        return self._reader.provenance["t_far"]

    @property
    def near_bp(self) -> int:
        """The distance up to which the store holds every pair (spec §7)."""
        return self._reader.provenance["near_bp"]

    @property
    def far_bp(self) -> int:
        """The distance up to which the store measured pairs (spec §7)."""
        return self._reader.provenance["far_bp"]

    def matrix(self, alids: Sequence[str], *, fill: float | None = None) -> LDMatrix:
        """The signed-r matrix of `alids`, in request order; NaN where unstored unless `fill`."""
        return matrices.matrix(self._reader, alids, fill=fill)

    def region(
        self, chrom: str, centre: int, radius: int, *, fill: float | None = None
    ) -> LDMatrix:
        """The signed-r matrix within `radius` bp of `centre`; refused if radius > far_bp / 2."""
        return matrices.region(self._reader, chrom, centre, radius, fill=fill)

    def proxies(self, alids: Sequence[str], *, r2: float = 0.8, kb: float = 10_000) -> Proxies:
        """Each ALID's proxies at r̂² ≥ r2 within kb × 1,000 bp, with signed r."""
        return proxies_query.proxies(self._reader, alids, r2=r2, kb=kb)

    def clump(self, alids: Sequence[str], score: npt.ArrayLike, *, r2: float, kb: float) -> Clumps:
        """Clump `alids` by `score` (|z| or −log10 p) at r̂² ≥ r2 within kb × 1,000 bp."""
        return clumping.clump(self._reader, alids, score, r2=r2, kb=kb)


def open(location: str | os.PathLike[str]) -> LDStore:
    """Open the LD Store at a local path or an HTTP(S) URL (spec §1; ogld#4)."""
    return LDStore(StoreReader.open(location))
