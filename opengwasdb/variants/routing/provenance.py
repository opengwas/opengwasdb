"""The build-time rule for `source_alid` provenance: no coordinate no source supplied (#261).

A Store Variant Table row's ``source_alid`` names the coordinate its source spelt it
by. A plain Variant Reference once gave every ALID it listed its own ALID as
``source_alid``, as if a source had supplied it, and nothing downstream could tell. A
Store cannot tell either, after the fact: a row whose every association Admission
dropped was observed all the same, and holds no association, as an unobserved row
does. So the rule is the build's, and it is kept apart from what it checks: the
routing's provenance comes from its record of what was observed
(`VariantRouting.source_alids`), and the rule checks the map a builder writes against
the coordinates its sources supplied, recorded from the sites read
(`SuppliedCoordinates`), never from that record.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from opengwasdb.variants.routing.keys import SourceKey, source_alid

__all__ = ["SuppliedCoordinates", "require_observed_provenance"]


def _digest(alid: str) -> int:
    return int.from_bytes(hashlib.blake2b(alid.encode(), digest_size=8).digest(), "little")


def _digests_in_order(alids: Iterable[str]) -> np.ndarray:
    """Each ALID as a 64-bit BLAKE2b digest, in order: 8 bytes an ALID, so a
    genome-scale union is held in megabytes, not gigabytes of strings."""
    return np.fromiter((_digest(alid) for alid in alids), dtype=np.uint64)


def _digests(alids: Iterable[str]) -> np.ndarray:
    """`_digests_in_order`, sorted and unique."""
    return np.unique(_digests_in_order(alids))


@dataclass(frozen=True)
class SuppliedCoordinates:
    """The source-build coordinates a build's sources supplied, as the ALIDs
    ``source_alid`` spells them (`keys.source_alid`).

    Recorded from the sites read: Pass 1's union for a lifted union or a plain
    reference, an artifact's recorded source keys (which its extraction read), and the
    Hybrid fold's resolved keys (which Pass 2 read). Held as digests, so a coordinate
    not supplied is found absent with a chance of a false match near 2^-64 a lookup.
    """

    digests: np.ndarray

    @classmethod
    def of_sites(cls, sites: Iterable[SourceKey]) -> SuppliedCoordinates:
        """The coordinates of ``sites``, each spelt as its ``source_alid``."""
        return cls(_digests(source_alid(site) for site in sites))

    @classmethod
    def of_alids(cls, alids: Iterable[str]) -> SuppliedCoordinates:
        """Coordinates already spelt as ``source_alid`` values."""
        return cls(_digests(alids))

    def __or__(self, other: SuppliedCoordinates) -> SuppliedCoordinates:
        return SuppliedCoordinates(np.union1d(self.digests, other.digests))

    def __len__(self) -> int:
        return len(self.digests)

    def holds(self, alids: list[str]) -> np.ndarray:
        """Per ALID, whether a source supplied it."""
        wanted = _digests_in_order(alids)
        at = np.searchsorted(self.digests, wanted)
        found = np.zeros(len(wanted), dtype=bool)
        inside = at < len(self.digests)
        found[inside] = self.digests[at[inside]] == wanted[inside]
        return found


def require_observed_provenance(
    origin: Mapping[str, str | None], supplied: SuppliedCoordinates, *, where: str
) -> int:
    """Refuse a ``source_alid`` map that names a coordinate no source supplied; return
    how many ALIDs carry one (#261).

    ``origin`` is the map a builder writes, ALID to ``source_alid`` (``None`` is ``.``).
    ``supplied`` is recorded from the sites the build read, apart from the routing's
    record of what was observed, so a regression in that record is refused here
    rather than passed. ``where`` names the caller in the refusal."""
    named = [(alid, source) for alid, source in origin.items() if source is not None]
    held = supplied.holds([source for _alid, source in named])
    fabricated = sorted(pair for pair, ok in zip(named, held.tolist(), strict=True) if not ok)
    if fabricated:
        alid, source = fabricated[0]
        raise ValueError(
            f"{where}: {len(fabricated)} ALID(s) would record a source_alid that no source "
            f"supplied (first: {alid} <- {source}); a row no source observed has no source "
            "coordinate, and its source_alid is '.'"
        )
    return len(named)
