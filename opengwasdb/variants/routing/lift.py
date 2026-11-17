"""Placing Pass 1's union on GRCh38: the hg38 passthrough and the hg19 lift (issue #85).

Each source declares its own assembly. A site already on hg38 maps straight to
its ALID; running it through the hg19->hg38 chain a second time would shift it
to the wrong position, and pyliftover cannot tell from the coordinate alone.
Sites declared hg19 go through one shared lift, whose failure threshold applies
to that group alone. A reverse-strand block re-orients a site's statistics as
well as renaming it, so the lift returns the keys it re-oriented (issue #284).

A raw tuple declared under both assemblies is two physical loci that share one
string, not one variant reported twice; it is dropped from both groups rather
than guessed.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Collection, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from opengwasdb.build.liftover import (
    LiftedRow,
    lift_hg19_to_hg38,
    lifted_rows,
    statistics_swap,
)
from opengwasdb.encoding.timing import format_duration
from opengwasdb.variants.normalise import canonical_alid, is_primary_contig
from opengwasdb.variants.routing.keys import SourceKey, routing_key

log = logging.getLogger(__name__)

#: A source coordinate -> the hg38 ALID it routes to (Pass 1's product).
SourceLookup = dict[SourceKey, str]
#: The source keys a reverse-strand lift re-oriented, whose statistics follow
#: the projected alleles (issue #284).
ReorientedKeys = set[SourceKey]


class LiftSplit(NamedTuple):
    """A lift's rows by where they landed: ``primary`` on a primary-assembly contig,
    and the source keys of ``off_primary``, which landed on an alt, random or
    unplaced contig (#376)."""

    primary: dict[SourceKey, LiftedRow]
    off_primary: set[SourceKey]


def split_off_primary(rows: Mapping[SourceKey, LiftedRow]) -> LiftSplit:
    """Split a lift's rows by `is_primary_contig`: the one place the rule is applied
    to a lift's result, which every path that lifts a source calls."""
    primary = {key: row for key, row in rows.items() if is_primary_contig(row.chromosome)}
    return LiftSplit(primary, rows.keys() - primary.keys())


def lifted_off_primary(sites: Collection[SourceKey], *, chain_file: str | Path | None) -> set[str]:
    """The routing keys of the hg19 ``sites`` the chain places off the primary assembly.

    For a build routed by a Variant Reference, whose unmatched hg19 rows are the
    only ones not lifted already: no failure threshold applies, since a site the
    chain places nowhere is `off_axis` as before."""
    rows, _attempts, _failures = lifted_rows(sites, chain_file=chain_file)
    return {routing_key(site) for site in split_off_primary(rows).off_primary}


class UnionPlacement(NamedTuple):
    """`place_union`'s result: the union's site lookup, the keys the lift
    re-oriented (#284), the routing keys both assemblies declared (#255), and the
    routing keys of the sites the lift placed off the primary assembly (#376)."""

    lookup: SourceLookup
    reoriented: ReorientedKeys
    dropped: set[str]
    off_primary: set[str]


@dataclass(frozen=True)
class SitePlacement:
    """Where source sites land on GRCh38.

    ``alid`` maps each placed site to its hg38 ALID; a site whose lift failed
    is absent. ``reoriented`` holds the sites a reverse-strand chain block
    re-oriented: their stored alleles call the other allele A1, so a z signed
    by the source must be negated and an EAF complemented (issue #284).
    ``off_primary`` holds the sites the lift placed on an alt, random or
    unplaced contig: they are in neither ``alid`` nor ``reoriented``, and a
    build drops and counts their rows (#376).
    """

    alid: SourceLookup
    reoriented: ReorientedKeys
    off_primary: set[SourceKey] = field(default_factory=set)


def place_sites(
    sites_by_assembly: Mapping[str, Collection[SourceKey]],
    *,
    chain_file: str | Path | None = None,
    failure_threshold: float = 0.01,
) -> SitePlacement:
    """Place each site, grouped by its declared assembly, on GRCh38.

    An hg38 site is its own ALID (`canonical_alid`). hg19 sites go through one
    lift, whose failure threshold applies to that group alone, so hg38 sites can
    neither dilute nor inflate it; more failures than it allows raise
    `LiftoverFailureError`. Any other assembly is refused with ``ValueError``,
    and so is a tuple given under both: it is two loci, each with its own
    place, and a result keyed by the tuple could keep only one. Pass 1's union
    and the Hybrid fold's off-reference keys are both placed here, so the lift
    and its orientation have one rule.
    """
    unknown = set(sites_by_assembly) - {"hg19", "hg38"}
    if unknown:
        raise ValueError(f"unhandled source_assembly values: {sorted(unknown)}")
    hg38 = sites_by_assembly.get("hg38", ())
    hg19 = sites_by_assembly.get("hg19", ())
    if hg38 and hg19:
        _refuse_both_assemblies(hg38, hg19)
    placement = SitePlacement({}, set())
    if hg38:
        _place_hg38(hg38, placement)
    if hg19:
        _lift_hg19(hg19, placement, chain_file=chain_file, failure_threshold=failure_threshold)
    return placement


def _place_hg38(sites: Collection[SourceKey], placement: SitePlacement) -> None:
    """Name each hg38 site by its own coordinate (`canonical_alid`)."""
    log.info("%d variants already GRCh38 -- no liftover needed", len(sites))
    for chrom, pos, ref, alt in sites:
        placement.alid[(chrom, pos, ref, alt)] = canonical_alid(chrom, pos, ref, alt)


def _lift_hg19(
    sites: Collection[SourceKey],
    placement: SitePlacement,
    *,
    chain_file: str | Path | None,
    failure_threshold: float,
) -> None:
    """Lift the hg19 sites, recording the ones a reverse-strand block re-oriented.

    The richer `lift_hg19_to_hg38` rather than `build_liftover_lookup`: a
    reverse-strand block re-orients a variant's statistics as well as renaming
    it, and an ALID string cannot say which -- a palindromic pair's ALID is
    identical under either projection.
    """
    log.info("Running liftover hg19 → hg38 (%d variants)", len(sites))
    lifted = lift_hg19_to_hg38(sites, chain_file=chain_file, failure_threshold=failure_threshold)
    split = split_off_primary(lifted)
    placement.off_primary.update(split.off_primary)
    for key, row in split.primary.items():
        placement.alid[key] = row.alid
        if statistics_swap(key, row):
            placement.reoriented.add(key)
    log.info("Liftover complete: %d variants mapped", len(lifted))
    if placement.off_primary:
        log.warning(
            "%d lifted variant(s) landed on an alt, random or unplaced contig, "
            "which the Store does not hold -- their rows are dropped (first: %s)",
            len(placement.off_primary), _first_key(placement.off_primary),
        )


def _first_key(sites: Collection[SourceKey]) -> str:
    return ":".join(map(str, min(sites)))


def _refuse_both_assemblies(
    hg38: Collection[SourceKey], hg19: Collection[SourceKey]
) -> None:
    """Refuse the tuples declared under both assemblies, naming the first."""
    both = set(hg38) & set(hg19)
    if both:
        raise ValueError(
            f"{len(both)} site(s) given under both hg38 and hg19 (first: {min(both)!r}); "
            "each is two loci with two places, so place each assembly's sites apart "
            "and decide which locus owns the row before placing them"
        )


def _lift_union_by_assembly(
    tuples_by_assembly: dict[str, set[SourceKey]],
    *,
    chain_file: str | Path | None,
    liftover_failure_threshold: float,
) -> tuple[SitePlacement, SitePlacement]:
    """Place each assembly's group; return the hg38 passthrough and the lifted
    placement (its re-oriented keys, issue #284, and the sites it placed off the
    primary assembly, #376). Kept apart so the both-assembly drop can compare them."""
    passthrough = place_sites({"hg38": tuples_by_assembly.pop("hg38", set())})
    lifted = place_sites(
        {"hg19": tuples_by_assembly.pop("hg19", set())},
        chain_file=chain_file,
        failure_threshold=liftover_failure_threshold,
    )
    if tuples_by_assembly:
        raise ValueError(f"unhandled source_assembly values: {sorted(tuples_by_assembly)}")
    return passthrough, lifted


def cross_assembly_collisions(
    hg38: AbstractSet[SourceKey], hg19: AbstractSet[SourceKey]
) -> set[str]:
    """The routing keys both groups declare, which a placement must not route.

    ``hg19`` is the hg19 sites the lift placed. A routing key spells a site's
    identity -- its alleles upper-cased -- so a key in both groups is one
    identity declared under both assemblies: two physical loci, not one variant
    reported twice, whether the two spell it alike or one writes `g/a` and the
    other `G/A`. Binding both to one stored row would misattribute one
    Analysis's association to the other's variant, so every site any of these
    keys spells is dropped from both groups (issues #85, #255, #360).

    The union's placement and extraction's windows both call this, so a
    Variant Reference drops exactly what the union does.
    """
    if not hg38 or not hg19:
        return set()
    return {routing_key(site) for site in hg38} & {routing_key(site) for site in hg19}


def log_cross_assembly_drop(n_tuples: int) -> None:
    """Say how many identities both assemblies declared, which were dropped."""
    if n_tuples:
        log.warning(
            "%d raw variant tuple(s) declared both hg38 and hg19 in this manifest "
            "(same chrom/pos/ref/alt string, two different builds -> two different "
            "physical loci) -- dropped from both, with every other spelling of them, "
            "rather than guessed which one owns the stored row",
            n_tuples,
        )


def _drop_cross_assembly_collisions(
    passthrough_lookup: SourceLookup,
    lifted_lookup: SourceLookup,
    swapped: ReorientedKeys,
    off_primary: set[SourceKey],
) -> set[str]:
    """Apply `cross_assembly_collisions` to the union's two placed groups, and
    return the dropped routing keys, which a harvest also drops from its
    candidates.

    A site lifted off the primary assembly is a lifted locus all the same: a
    GRCh38 source spelling its key would otherwise take the row that is not its.
    """
    dropped = cross_assembly_collisions(
        passthrough_lookup.keys(), lifted_lookup.keys() | off_primary
    )
    log_cross_assembly_drop(len(dropped))
    if dropped:
        for lookup in (passthrough_lookup, lifted_lookup):
            for site in [site for site in lookup if routing_key(site) in dropped]:
                del lookup[site]
                swapped.discard(site)
        off_primary.difference_update(
            [site for site in off_primary if routing_key(site) in dropped]
        )
    return dropped


def place_union(
    tuples_by_assembly: dict[str, set[SourceKey]],
    *,
    chain_file: str | Path | None,
    failure_threshold: float,
) -> UnionPlacement:
    """Place the union on hg38 (`UnionPlacement`): the both-assembly tuples it
    dropped are routing keys (issue #255 rounds 4-5), and so are the sites the
    lift placed off the primary assembly, whose rows a build drops and counts (#376).
    """
    finalise_start = time.monotonic()
    passthrough, lifted = _lift_union_by_assembly(
        tuples_by_assembly,
        chain_file=chain_file,
        liftover_failure_threshold=failure_threshold,
    )
    passthrough_lookup, lifted_lookup = passthrough.alid, lifted.alid
    swapped, off_primary = lifted.reoriented, lifted.off_primary
    dropped_keys = _drop_cross_assembly_collisions(
        passthrough_lookup, lifted_lookup, swapped, off_primary
    )
    source_lookup = {**passthrough_lookup, **lifted_lookup}
    log.info(
        "Pass 1 liftover/finalisation: %d source variants → %d hg38 ALIDs in %s",
        len(source_lookup),
        len(set(source_lookup.values())),
        format_duration(time.monotonic() - finalise_start),
    )
    return UnionPlacement(
        source_lookup, swapped, dropped_keys, {routing_key(site) for site in off_primary}
    )
