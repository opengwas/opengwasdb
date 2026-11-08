"""Lifting a Ragged SSF build's non-GRCh38 sources onto GRCh38 (#386).

A Ragged SSF Store is one build, GRCh38, and every manifest row declares its
source's build. A source on hg19 is read oriented to its own canonical A1, its
sites are lifted together through `place_sites` (the placement Variant Routing
and the Hybrid fold share), and each row is re-oriented by the same rule if a
reverse-strand chain block made the other allele A1.

The builder does not call `route_variants`. That reads every source twice, and
its first read sees the rows Association Admission would still drop, so the
lift's failure rate and its both-builds drop would judge sites no Store row has.
This build admits each Analysis's rows once, as it reads them, and so places
the sites of the rows it admitted.

What the lift drops, and how each is counted (Admission `Reason`s):

- ``lift_unmapped``: the chain places the site nowhere.
- ``lift_ambiguous``: a Routing Key that a GRCh38 source and a lifted source both
  declare. It is two loci that share a string, as `place_union` treats it, so
  every row of the key is dropped from both builds' Analyses.

Two source sites that lift onto one variant are not dropped here: they are one
Analysis's duplicate rows, which the build's duplicate rule (`collapse_duplicates`)
resolves after the lift, as it does in a Dense or Hybrid build.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from opengwasdb.readers.orientation import swap
from opengwasdb.readers.rows import ReaderBatch
from opengwasdb.variants.routing.keys import SourceKey, encode_routing_keys, routing_key
from opengwasdb.variants.routing.lift import (
    cross_assembly_collisions,
    log_cross_assembly_drop,
    place_sites,
)

log = logging.getLogger(__name__)

#: The one build a Ragged SSF Store holds, as `normalise_build` spells it.
STORE_BUILD = "hg38"


@dataclass(frozen=True)
class LiftOptions:
    """How a build lifts: the chain, and the most sites the chain may fail to place."""

    chain_file: Path | None = None
    failure_threshold: float = 0.01


def require_liftable(declared: Sequence[tuple[str, str]], chain_file: Path | None) -> None:
    """Refuse, before any source is read, a lift the build cannot perform.

    ``declared`` is each Analysis's ``(analysis_id, normalised build)``. A source
    on another build needs a chain that exists: with none, pyliftover would
    fetch UCSC's, and a build must name the chain it lifted by.
    """
    for analysis_id, build in declared:
        if build == STORE_BUILD:
            continue
        if chain_file is None:
            raise ValueError(
                f"analysis {analysis_id!r} declares source_genome_build {build}; "
                f"build-ragged-ssf lifts it to GRCh38 and needs --chain-file"
            )
        if not Path(chain_file).is_file():
            raise FileNotFoundError(
                f"analysis {analysis_id!r} declares source_genome_build {build}: "
                f"liftover chain file not found: {chain_file}"
            )


@dataclass(frozen=True)
class SitePlacement:
    """Where a build's lifted sites land, and what the lift could not place.

    ``alid`` and ``reoriented`` are `place_sites`'. ``ambiguous`` holds the
    Routing Keys both a GRCh38 and a lifted source declare. ``n_unmapped`` is
    the number of lifted sites the chain placed nowhere, and ``off_primary`` the
    sites it placed on an alt, random or unplaced contig (#376).
    """

    alid: dict[SourceKey, str]
    reoriented: set[SourceKey]
    ambiguous: frozenset[str]
    n_unmapped: int
    off_primary: frozenset[SourceKey] = frozenset()

    def provenance(self, chain_file: Path | None) -> dict[str, object]:
        """The `provenance.lift` block, as `build-ragged-besd` records its lift.

        A lift here is never checked against a reference, so `reference` is
        ``null`` and its counts are too: an unchecked lift does not read as a
        checked one. ``n_lifted`` counts the sites the chain placed, the
        ambiguous ones among them.
        """
        return {
            "source_build": "hg19",
            "chain_file": str(chain_file) if chain_file is not None else None,
            "n_lifted": len(self.alid),
            "n_unmapped": self.n_unmapped,
            "n_off_primary": len(self.off_primary),
            "n_ambiguous": len(self.ambiguous),
            "reference": None,
            "n_alleles_verified": None,
            "n_reference_other_variant": None,
            "n_reference_absent": None,
        }


def place_held_sources(
    held: Sequence[tuple[bool, Sequence[ReaderBatch]]], options: LiftOptions
) -> SitePlacement:
    """Place every lifted source's sites, and find the keys both builds declare.

    ``held`` is each Analysis's ``(lifted, admitted batches)`` in Manifest Row
    order. Raises `LiftoverFailureError` when the chain fails more of the lifted
    sites than ``options.failure_threshold`` allows.
    """
    sites = {
        site for lifted, batches in held if lifted for batch in batches for site in _sites(batch)
    }
    placement = place_sites(
        {"hg19": sites}, chain_file=options.chain_file,
        failure_threshold=options.failure_threshold,
    )
    ambiguous = _both_builds_declare(
        [batches for lifted, batches in held if not lifted],
        placement.alid.keys() | placement.off_primary,
    )
    log_cross_assembly_drop(len(ambiguous))
    return SitePlacement(
        placement.alid,
        placement.reoriented,
        frozenset(ambiguous),
        len(sites) - len(placement.alid) - len(placement.off_primary),
        frozenset(placement.off_primary),
    )


def _sites(batch: ReaderBatch) -> list[SourceKey]:
    """The source key of each row of ``batch``: its coordinate and alleles as the
    source spells them."""
    keys: list[SourceKey] = []
    for alid, ref, alt in zip(batch.alid.tolist(), batch.ref.tolist(), batch.alt.tolist(),
                              strict=True):
        chromosome, position, _a1, _a2 = alid.split(":")
        keys.append((chromosome, int(position), ref, alt))
    return keys


def _both_builds_declare(
    grch38: Sequence[Sequence[ReaderBatch]], lifted: AbstractSet[SourceKey]
) -> set[str]:
    """The Routing Keys a GRCh38 source declares that a placed lifted site spells too.

    `cross_assembly_collisions` is the rule; only the GRCh38 rows whose key a
    lifted site spells are formed into sites for it, so a build holds the lifted
    group's keys and never the GRCh38 group's.
    """
    if not lifted:
        return set()
    wanted = np.array(sorted({routing_key(site) for site in lifted}), dtype="S")
    shared: set[SourceKey] = set()
    for batches in grch38:
        for batch in batches:
            shared.update(_sites_spelling(batch, wanted))
    return cross_assembly_collisions(shared, lifted)


def _sites_spelling(batch: ReaderBatch, wanted: np.ndarray) -> list[SourceKey]:
    """The sites of ``batch`` whose Routing Key is one of ``wanted``."""
    sites = _sites(batch)
    keys = encode_routing_keys(
        [s[0] for s in sites], [s[1] for s in sites], [s[2] for s in sites], [s[3] for s in sites]
    )
    return [sites[i] for i in np.flatnonzero(np.isin(keys, wanted)).tolist()]


@dataclass(frozen=True)
class PlacedBatch:
    """One batch after the lift: the rows it keeps, and the rows it dropped.

    ``batch`` holds the kept rows, their ALIDs on GRCh38 and their statistics
    on its A1. ``source_alid`` is each kept row's ALID as its own source spelt
    it, which the rsid harvest compares by. ``unmapped``, ``ambiguous`` and
    ``off_primary`` name the dropped rows by the source key Admission logs.
    """

    batch: ReaderBatch
    source_alid: np.ndarray
    unmapped: list[str]
    ambiguous: list[str]
    off_primary: list[str] = field(default_factory=list)


def place_batch(batch: ReaderBatch, lifted: bool, placement: SitePlacement) -> PlacedBatch:
    """``batch`` on GRCh38: a lifted source's rows moved and re-oriented, every
    row of an ambiguous Routing Key dropped, and a GRCh38 source's rows otherwise
    untouched."""
    if not lifted and not placement.ambiguous:
        return PlacedBatch(batch, batch.alid, [], [], [])
    sites = _sites(batch)
    ambiguous = [
        i for i, site in enumerate(sites) if routing_key(site) in placement.ambiguous
    ]
    moves, unmapped, off_primary = (
        _moves(sites, set(ambiguous), placement) if lifted else ({}, [], [])
    )
    alid = batch.alid.copy()
    swapped = np.zeros(len(sites), dtype=bool)
    for i, (moved, reoriented) in moves.items():
        alid[i], swapped[i] = moved, reoriented
    keep = np.ones(len(sites), dtype=bool)
    keep[ambiguous + unmapped + off_primary] = False
    z, eaf = swap(swapped, batch.z, batch.eaf)
    placed = replace(
        batch,
        alid=alid,
        flipped=batch.flipped ^ swapped,
        z=z,
        eaf=eaf,
        ref_first=None if batch.ref_first is None else batch.ref_first ^ swapped,
    ).take(keep)
    return PlacedBatch(
        placed, batch.alid[keep], batch.keys_at(np.array(unmapped, dtype=np.int64)),
        batch.keys_at(np.array(ambiguous, dtype=np.int64)),
        batch.keys_at(np.array(off_primary, dtype=np.int64)),
    )


def _moves(
    sites: list[SourceKey], skip: set[int], placement: SitePlacement
) -> tuple[dict[int, tuple[str, bool]], list[int], list[int]]:
    """Each lifted row's ALID on GRCh38 and whether the lift re-oriented it, the
    rows the chain placed nowhere, and those it placed off the primary assembly.
    A row in ``skip`` is dropped already."""
    moves: dict[int, tuple[str, bool]] = {}
    unmapped: list[int] = []
    off_primary: list[int] = []
    for i, site in enumerate(sites):
        if i in skip:
            continue
        moved = placement.alid.get(site)
        if moved is None:
            (off_primary if site in placement.off_primary else unmapped).append(i)
        else:
            moves[i] = (moved, site in placement.reoriented)
    return moves, unmapped, off_primary
