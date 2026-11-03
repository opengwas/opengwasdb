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
from dataclasses import dataclass
from pathlib import Path

from opengwasdb.build.liftover import lift_hg19_to_hg38, statistics_swap
from opengwasdb.encoding.timing import format_duration
from opengwasdb.variants.routing.keys import RoutingKey, SourceKey

log = logging.getLogger(__name__)

#: A source coordinate -> the hg38 ALID it routes to (Pass 1's product).
SourceLookup = dict[SourceKey, str]
#: The source keys a reverse-strand lift re-oriented, whose statistics follow
#: the projected alleles (issue #284).
ReorientedKeys = set[SourceKey]


@dataclass(frozen=True)
class SitePlacement:
    """Where source sites land on GRCh38.

    ``alid`` maps each placed site to its hg38 ALID; a site whose lift failed
    is absent. ``reoriented`` holds the sites a reverse-strand chain block
    re-oriented: their stored alleles call the other allele A1, so a z signed
    by the source must be negated and an EAF complemented (issue #284).
    """

    alid: SourceLookup
    reoriented: ReorientedKeys


def place_sites(
    sites_by_assembly: Mapping[str, Collection[SourceKey]],
    *,
    chain_file: str | Path | None = None,
    failure_threshold: float = 0.01,
) -> SitePlacement:
    """Place each site, grouped by its declared assembly, on GRCh38.

    An hg38 site is its own ALID, its alleles sorted. hg19 sites go through one
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
    """Name each hg38 site by its own coordinate, its alleles sorted."""
    log.info("%d variants already GRCh38 -- no liftover needed", len(sites))
    for chrom, pos, ref, alt in sites:
        a1, a2 = sorted((ref, alt))
        placement.alid[(chrom, pos, ref, alt)] = f"{chrom}:{pos}:{a1}:{a2}"


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
    for key, row in lifted.items():
        placement.alid[key] = row.alid
        if statistics_swap(key, row):
            placement.reoriented.add(key)
    log.info("Liftover complete: %d variants mapped", len(lifted))


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
) -> tuple[SourceLookup, SourceLookup, ReorientedKeys]:
    """Place each assembly's group; return the hg38 passthrough, the lifted, and
    the keys the lift re-oriented (issue #284). Kept apart so the
    both-assembly drop can compare them."""
    passthrough = place_sites({"hg38": tuples_by_assembly.pop("hg38", set())})
    lifted = place_sites(
        {"hg19": tuples_by_assembly.pop("hg19", set())},
        chain_file=chain_file,
        failure_threshold=liftover_failure_threshold,
    )
    if tuples_by_assembly:
        raise ValueError(f"unhandled source_assembly values: {sorted(tuples_by_assembly)}")
    return passthrough.alid, lifted.alid, lifted.reoriented


def _drop_cross_assembly_collisions(
    passthrough_lookup: SourceLookup,
    lifted_lookup: SourceLookup,
    swapped: ReorientedKeys,
    *,
    key: RoutingKey,
) -> set[str]:
    """Drop the raw tuples declared under both assemblies, from both lookups.

    A shared raw tuple is two physical loci, not one variant reported twice, and
    binding both to one stored row would misattribute one Analysis's
    association to the other's variant. Returns the dropped tuples as ``key``
    spells them, which a harvest also drops from its candidates (issue #255).

    A key that folds can also make two *different* raw tuples declared under
    different assemblies collide; the two-pass build refuses that rather than
    guess, so this refuses too. An exact key cannot collide that way.
    """
    ambiguous = passthrough_lookup.keys() & lifted_lookup.keys()
    if ambiguous:
        log.warning(
            "%d raw variant tuple(s) declared both hg38 and hg19 in this manifest "
            "(same chrom/pos/ref/alt string, two different builds -> two different "
            "physical loci) -- dropped from both rather than guessed which one owns "
            "the stored row",
            len(ambiguous),
        )
        for site in ambiguous:
            del passthrough_lookup[site]
            del lifted_lookup[site]
            swapped.discard(site)
    dropped_keys = {key.of(site) for site in ambiguous}
    if not key.folds:
        return dropped_keys
    cross = (
        {key.of(site) for site in passthrough_lookup} & {key.of(site) for site in lifted_lookup}
    ) - dropped_keys
    if cross:
        raise ValueError(
            f"{len(cross)} source routing key(s) are declared under both hg38 and hg19 "
            f"by different coordinates (e.g. {sorted(cross)[0]!r}); the two-pass "
            "build refuses to guess which locus owns the stored row, so this build "
            "refuses too"
        )
    return dropped_keys


def place_union(
    tuples_by_assembly: dict[str, set[SourceKey]],
    *,
    chain_file: str | Path | None,
    failure_threshold: float,
    key: RoutingKey,
) -> tuple[SourceLookup, ReorientedKeys, set[str]]:
    """Place the union on hg38: its lookup, the keys the lift re-oriented, and
    the both-assembly tuples it dropped, as ``key`` spells them.

    ``key`` is the spelling the drop is computed on: the raw tuple for the
    union a build routes by, or the build's own key for a harvest whose
    candidates are keyed that way (issue #255 rounds 4-5).
    """
    finalise_start = time.monotonic()
    passthrough_lookup, lifted_lookup, swapped = _lift_union_by_assembly(
        tuples_by_assembly,
        chain_file=chain_file,
        liftover_failure_threshold=failure_threshold,
    )
    dropped_keys = _drop_cross_assembly_collisions(
        passthrough_lookup, lifted_lookup, swapped, key=key
    )
    source_lookup = {**passthrough_lookup, **lifted_lookup}
    log.info(
        "Pass 1 liftover/finalisation: %d source variants → %d hg38 ALIDs in %s",
        len(source_lookup),
        len(set(source_lookup.values())),
        format_duration(time.monotonic() - finalise_start),
    )
    return source_lookup, swapped, dropped_keys
