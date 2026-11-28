"""`route_variants`: which Variant Index place every source coordinate reaches.

The authority decides. Under `LiftedUnion` the sources themselves are the
axis: Pass 1 reads each once, lifts the hg19 group to GRCh38 and names each
ALID by its first-named rsid. Under `FromReference` a Variant Reference is
the axis and its source keys the routing; Pass 1 runs to harvest names: the
reference's own names win, and the harvest fills every variant it leaves
unnamed (issue #258). A reference that names no source keys (a plain ALID
list, a Store Variant Table) can route only a GRCh38 row, by identity: when an
hg19 source is in the build, Pass 1's lifted union routes every site and the
reference stays the axis (issue #260); otherwise Pass 1's read of the sources
also says which of its ALIDs a source observed, the only ones that carry a
``source_alid`` (issue #261). Either harvest compares the names a site's
spellings carry, by one rule (issue #379).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path

from opengwasdb.build.liftover import lifted_positions
from opengwasdb.readers.registry import SourceSpec, open_source
from opengwasdb.store_reader.identity import (
    is_primary_contig,
)
from opengwasdb.variants.axis import sorted_alids
from opengwasdb.variants.reference import VariantReference
from opengwasdb.variants.routing.keys import (
    SourceKey,
    contig_label,
    folded_site,
    note_origin,
    raw_site_key,
    routing_key,
    source_alid,
)
from opengwasdb.variants.routing.lift import (
    NO_LIFTED_AWAY,
    LiftCollisions,
    LiftDrops,
    LiftedAway,
    LiftedAwayReason,
    UnionPlacement,
    collided_sites,
    find_lift_collisions,
    place_union,
)
from opengwasdb.variants.routing.names import (
    HARVEST_LABEL,
    NO_REFERENCE_NAMES,
    RoutedNames,
    SiteNames,
    harvested_names,
    union_names,
)
from opengwasdb.variants.routing.provenance import SuppliedCoordinates
from opengwasdb.variants.routing.table import RouteTable
from opengwasdb.variants.routing.union import (
    Pass1Stats,
    ReaderFactory,
    UnionOptions,
    consume_shards,
    materialize_site_union,
)
from opengwasdb.variants.rsid import RsidConflictPolicy

log = logging.getLogger(__name__)

#: What a refusal or a warning calls a build routed by its sources' own
#: union, unless the build says otherwise.
UNION_LABEL = "Two-pass build"


@dataclass(frozen=True)
class LiftedUnion:
    """The sources are the axis: every site read once, hg19 sites lifted.

    ``failure_threshold`` is the fraction of the hg19 group allowed to fail
    the lift before the build refuses (`LiftoverFailureError`).
    """

    chain_file: str | Path | None = None
    failure_threshold: float = 0.01


@dataclass(frozen=True)
class FromReference:
    """A Variant Reference is the axis, and its source keys route.

    A key the reference records as re-oriented has its statistics re-signed
    exactly as Pass 1's lift re-signs them, so a single-pass build stores what
    the two-pass one does (issue #354). Pass 1 runs to harvest the sources'
    rsids, which name every variant the reference leaves unnamed (issue #258);
    its lift then uses ``chain_file`` and ``failure_threshold``, though it
    routes nothing. A reference with no source keys is read beside its sources
    either way: with an hg19 source, Pass 1's lift, with the same chain and
    threshold, routes every site (issue #260); with none, Pass 1 says which
    ALIDs a source observed (issue #261).
    """

    reference: VariantReference
    chain_file: str | Path | None = None
    failure_threshold: float = 0.01


class VariantRouting:
    """Where each source coordinate goes, and what each routed ALID is called.

    ``names`` and ``stats`` are values. The site map behind `alids`,
    `source_alids` and `table` is held only until `table` consumes it, so a
    genome-scale map is not resident through Pass 2: read `alids` and
    `source_alids` first, and call `table` once, before forking.

    ``lift_drops`` is each source site dropped because another source locus lifted
    onto its ALID (#325), by both identities, and the routing keys by which Pass 2
    counts their rows as `lift_collision`. ``lifted_away`` is the routing key of each
    site a lift placed where no Store row can be (Lifted Away), with the reason Pass 2
    counts its rows under (`split_lifted`, #376, #410).

    ``placed_every_site`` says Pass 1 placed every source site, as a two-pass build
    does: a row the site map does not hold is one the routing dropped (a failed or
    off-primary lift, a collision, a tuple both assemblies declare) and is counted
    as such, never placed later by a Hybrid fold (issue #260).

    ``observed`` is the sites some source supplied, each with the ALID it routes to:
    what `source_alids` reads (#261). It is the site map itself, except under a plain
    reference (``plain_reference``), whose site map is the identity of every ALID it
    lists, observed or not; there Pass 1's sites, routed as Pass 2 routes them, are
    passed, and a plain reference that passes none is refused. ``supplied`` is the
    coordinates the sources supplied, recorded from the sites read and never from
    ``observed``: what a builder checks the provenance it writes against
    (`require_observed_provenance`).
    """

    def __init__(
        self,
        site_alid: Mapping[SourceKey, str],
        reoriented: Collection[SourceKey],
        *,
        axis: Sequence[str] | None,
        names: RoutedNames,
        stats: Pass1Stats,
        supplied: SuppliedCoordinates,
        lifted_away: LiftedAway = NO_LIFTED_AWAY,
        lift_drops: LiftDrops | None = None,
        placed_every_site: bool = False,
        observed: Mapping[SourceKey, str] | None = None,
        plain_reference: bool = False,
    ) -> None:
        if plain_reference and observed is None:
            raise ValueError(
                "a Variant Reference with no source keys routes every ALID it lists by "
                "identity, so its routing must say which ones a source observed; it says "
                "none, and every listed ALID would record a source_alid no source supplied"
            )
        self._site_alid: Mapping[SourceKey, str] | None = site_alid
        self._observed = observed
        self.supplied = supplied
        self._reoriented = reoriented
        self._axis = axis
        self.names = names
        self.stats = stats
        self.lifted_away = lifted_away
        self.lift_drops = lift_drops if lift_drops is not None else LiftDrops()
        self.placed_every_site = placed_every_site

    def _sites(self) -> Mapping[SourceKey, str]:
        if self._site_alid is None:
            raise ValueError("the site map was consumed by table(); read it before")
        return self._site_alid

    @cached_property
    def alids(self) -> list[str]:
        """Every ALID the authority routes to, unique and in genomic order."""
        return sorted_alids(self._axis if self._axis is not None else self.reached_alids)

    @cached_property
    def reached_alids(self) -> frozenset[str]:
        """Every ALID a site of the site map reaches: under a plain reference, every
        ALID it lists, observed or not (`source_alids` holds the observed ones)."""
        return frozenset(self._sites().values())

    def _observed_sites(self) -> Mapping[SourceKey, str]:
        return self._sites() if self._observed is None else self._observed

    @cached_property
    def observed_alids(self) -> frozenset[str]:
        """The ALIDs a site some source supplied routes to: the only ones `source_alids`
        names a coordinate for (#261)."""
        return frozenset(self._observed_sites().values())

    def source_alids(self) -> dict[str, str | None]:
        """Each routed ALID's source-build ALID: the own coordinate of the sites a source
        supplied that reach it.

        The Store Variant Table's ``source_alid`` provenance. Several sites can
        reach one ALID (a liftover collapse, two assemblies); when they name
        different origins the provenance is ambiguous and is ``None``, never a
        guess (issue #85). A routed ALID no source observed has no entry, and is
        written ``.``: a plain reference routes it, but no source supplied a
        coordinate for it (#261). Computed afresh on each call.
        """
        by_alid: dict[str, str | None] = {}
        for site, alid in self._observed_sites().items():
            note_origin(by_alid, alid, source_alid(site))
        return by_alid

    def table(
        self, axes: Sequence[Sequence[str]], *, keep_unmatched: bool = False
    ) -> RouteTable:
        """The fork-safe Pass 2 lookup onto ``axes``; the first axis holding an ALID places it.

        ``keep_unmatched`` keeps the associations no axis holds, by routing
        key, for a caller that places them later; otherwise they are only
        counted. Consumes the site map, so it runs once.
        """
        table = RouteTable.build(
            self._sites(), self._reoriented, axes, keep_unmatched=keep_unmatched
        )
        self._site_alid = None
        self._observed = None
        self._reoriented = ()
        return table


def route_variants(
    sources: Sequence[SourceSpec],
    authority: LiftedUnion | FromReference,
    *,
    on_rsid_conflict: RsidConflictPolicy = RsidConflictPolicy.REFUSE,
    n_workers: int = 1,
    union: UnionOptions | None = None,
    readers: ReaderFactory = open_source,
    label: str | None = None,
) -> VariantRouting:
    """Route every source coordinate to the Variant Index, and name what it reaches.

    Every coordinate is matched by its routing key (spec §4). ``n_workers`` and
    ``union`` shape Pass 1's parallel union and never change its result.
    ``readers`` opens a source; a test passes a fake. ``label`` is what a
    harvest's refusals and warnings call the build: by default a two-pass
    build under `LiftedUnion` and a single-pass one under `FromReference`.

    Raises `LiftoverFailureError` when more of the hg19 group fails the lift
    than the threshold allows, and ``ValueError`` when one routing key spells
    two assemblies' coordinates, when a disputed name meets
    ``on_rsid_conflict=refuse``, or when a rekeyed name disagrees with the
    independent walk of the raw names.
    """
    union_pass = _UnionPass(
        sources, n_workers, union if union is not None else UnionOptions(), Pass1Stats(), readers
    )
    if isinstance(authority, FromReference) and not _needs_lift_or_refuses(
        authority.reference, sources
    ):
        return _route_by_reference(
            union_pass, authority, on_rsid_conflict, label or HARVEST_LABEL
        )
    default = HARVEST_LABEL if isinstance(authority, FromReference) else UNION_LABEL
    return _route_by_union(union_pass, authority, on_rsid_conflict, label or default)


def _needs_lift_or_refuses(reference: VariantReference, sources: Sequence[SourceSpec]) -> bool:
    """Whether ``reference`` cannot route ``sources`` itself, so Pass 1's lift must;
    raises ``ValueError`` for a source neither can place.

    A reference with source keys (an extraction artifact) routes every source it was
    extracted from. One without them routes a row only by its own coordinate, which
    is an hg38 ALID only for a GRCh38 source: an hg19 source's rows are lifted, by
    the lift the two-pass build uses (issue #260). A source declared under any other
    assembly cannot be placed either way, and is refused."""
    if reference.explicit_source_keys:
        return False
    unknown = sorted({s.assembly for s in sources} - {"hg19", "hg38"})
    if unknown:
        raise ValueError(
            f"the Variant Reference names no source keys, so it routes only a GRCh38 "
            f"source by its own coordinate, and an hg19 source by the lift; the manifest "
            f"declares {', '.join(map(repr, unknown))}, which neither places"
        )
    return any(s.assembly == "hg19" for s in sources)


def _route_by_union(
    union_pass: _UnionPass,
    authority: LiftedUnion | FromReference,
    policy: RsidConflictPolicy,
    label: str,
) -> VariantRouting:
    """Every site lifted and placed, each named by its first rsid. A collision's
    sources name nothing: they reach no ALID (#325).

    The sources' own union is the axis, or, under a `FromReference` whose reference
    names no source keys, the reference is (issue #260): it decides which ALIDs
    exist, and a site placed on none of them is off its axis, as an off-panel
    variant is to a two-pass build. A reference that names rsids names those
    variants, and the sources name the rest (issue #258)."""
    placed, rsid_by_site, supplied = _place_union_pass(union_pass, authority)
    reference = authority.reference if isinstance(authority, FromReference) else None
    names = union_names(
        rsid_by_site.without(placed.dropped, collided_sites(placed.collisions)),
        placed.lookup,
        policy=policy,
        label=label,
        named=NO_REFERENCE_NAMES if reference is None else reference.rsid_by_alid,
    )
    return VariantRouting(
        placed.lookup, placed.reoriented,
        axis=None if reference is None else reference.alids,
        names=names, stats=union_pass.stats, supplied=supplied,
        lifted_away=placed.away,
        lift_drops=LiftDrops.of(placed.collisions, rsid_by_site.lifted),
        placed_every_site=True,
    )


def _place_union_pass(
    union_pass: _UnionPass, authority: LiftedUnion | FromReference
) -> tuple[UnionPlacement, SiteNames, SuppliedCoordinates]:
    """Pass 1's union placed on GRCh38: the placement, each site's first-named rsid, and
    the coordinates the union read, recorded before anything is placed (#261)."""
    tuples_by_assembly, rsid_by_site = union_pass.sites()
    supplied = SuppliedCoordinates.of_sites(
        site for sites in tuples_by_assembly.values() for site in sites
    )
    placed = place_union(
        tuples_by_assembly,
        chain_file=authority.chain_file,
        failure_threshold=authority.failure_threshold,
    )
    return placed, rsid_by_site, supplied


def _unrouted_collisions(
    collisions: LiftCollisions, reference: VariantReference
) -> LiftCollisions:
    """``collisions`` less the sources ``reference`` routes itself, which Pass 2 places.

    Compared by Routing Key, case-folded as Pass 2 matches a row, so a source the
    reference stores under another spelling is not recorded as dropped."""
    if not collisions:
        return {}
    routed = {routing_key(site) for site in reference.source_lookup}
    kept = {
        alid: tuple(s for s in sites if routing_key(s) not in routed)
        for alid, sites in collisions.items()
    }
    return {alid: sites for alid, sites in kept.items() if sites}


@dataclass(frozen=True)
class _UnionPass:
    """Pass 1 over one build's sources, read when a routing needs their sites."""

    sources: Sequence[SourceSpec]
    n_workers: int
    options: UnionOptions
    stats: Pass1Stats
    readers: ReaderFactory

    def sites(self) -> tuple[dict[str, set[SourceKey]], SiteNames]:
        """The site union by assembly, and each site's first-named rsid in
        ``(rank, site)`` order."""
        result: tuple[dict[str, set[SourceKey]], SiteNames] = consume_shards(
            self.sources,
            materialize_site_union,
            n_workers=self.n_workers,
            options=self.options,
            stats=self.stats,
            readers=self.readers,
        )
        return result


def _without_off_primary_lifts(
    reference: VariantReference,
) -> tuple[VariantReference, LiftedAway]:
    """``reference`` less the source sites a lift placed on an alt, random or
    unplaced contig, and those sites' routing keys, each `LiftedAwayReason.OFF_PRIMARY` (#376).

    An extraction since #376 leaves such a site out, but one made before it routes
    an hg19 row onto ``11_KI270927v1_alt:…``. A site whose own contig is not the
    contig of the ALID it reaches was lifted; where that ALID is off the primary
    assembly the site is dropped, as a lift drops it, and so is an ALID only such
    sites reach. A site on an alt contig of its own assembly is a chromosome."""
    stale = _lifted_off_primary_sites(reference)
    if not stale:
        return reference, NO_LIFTED_AWAY
    trimmed = _without_sites(reference, stale)
    log.warning(
        "%d source key(s) of the Variant Reference were lifted onto an alt, random or "
        "unplaced contig by the extraction -- their rows are dropped, and so are the %d "
        "ALID(s) only they reach",
        len(stale), len(reference.alids) - len(trimmed.alids),
    )
    return trimmed, {routing_key(site): LiftedAwayReason.OFF_PRIMARY for site in stale}


def _without_sites(reference: VariantReference, sites: set[SourceKey]) -> VariantReference:
    """``reference`` without ``sites``, and without any off-primary ALID only they reach."""
    lookup = {site: alid for site, alid in reference.source_lookup.items() if site not in sites}
    reached = set(lookup.values())
    alids = [
        a for a in reference.alids if is_primary_contig(a.split(":", 1)[0]) or a in reached
    ]
    kept = set(alids)
    return replace(
        reference,
        alids=alids,
        source_lookup=lookup,
        rsid_by_alid={a: r for a, r in reference.rsid_by_alid.items() if a in kept},
        reoriented=reference.reoriented - sites,
    )


def _lifted_off_primary_sites(reference: VariantReference) -> set[SourceKey]:
    """The reference's source sites a lift moved onto the off-primary ALID they reach."""
    off_primary = {a for a in reference.alids if not is_primary_contig(a.split(":", 1)[0])}
    if not off_primary:
        return set()
    return {
        site
        for site, alid in reference.source_lookup.items()
        if alid in off_primary and _moved_by_a_lift(site, alid)
    }


def _without_frozen_collisions(
    reference: VariantReference, sources: Sequence[SourceSpec], chain_file: str | Path | None
) -> tuple[VariantReference, LiftCollisions]:
    """``reference`` less the source keys of every ALID it routes two lifted loci to,
    and those keys by ALID (#325, #260).

    An artifact extracted before #325 keeps such an ALID with both loci's keys, and
    routed as given it would store one locus's statistics under the other's rsid. The
    rule is the lift's (`find_lift_collisions`), held to the lifted keys
    (`_lifted_keys`). The ALID stays on the axis, which the reference decides; no
    source key reaches it. A build with no hg19 source lifted nothing, so its keys are
    not scanned."""
    assemblies = {source.assembly for source in sources}
    if not reference.explicit_source_keys or "hg19" not in assemblies:
        return reference, {}
    collisions = find_lift_collisions(
        _lifted_keys(reference.source_lookup, every_source_hg19=assemblies == {"hg19"},
                     chain_file=chain_file)
    )
    if not collisions:
        return reference, {}
    sites = collided_sites(collisions)
    first, first_sites = next(iter(collisions.items()))
    log.warning(
        "the Variant Reference routes %d ALID(s) from two or more lifted source loci "
        "(first: %s, from %s) -- all %d of their source keys are dropped, as a lift "
        "collision is (#325)",
        len(collisions), first, ", ".join(map(raw_site_key, first_sites)), len(sites),
    )
    return _without_sites(reference, sites), collisions


def _lifted_keys(
    lookup: Mapping[SourceKey, str], *, every_source_hg19: bool, chain_file: str | Path | None
) -> dict[SourceKey, str]:
    """The keys of an artifact's ``lookup`` that an hg19 source's lift placed, by ALID.

    With every source hg19, that is every key. In a mixed manifest, a key a lift moved
    (`_moved_by_a_lift`) was lifted. A key at its ALID's own locus may be a GRCh38
    key, or an hg19 key the chain leaves where it is: the artifact does not say which
    assembly spelled it. It counts as lifted where the chain places its coordinate,
    read as hg19, on its own ALID's locus. Only a key beside a moved key at one ALID can
    collide, so only those are lifted. No failure threshold applies, since a key the
    chain places elsewhere, or nowhere, is a GRCh38 key."""
    if every_source_hg19:
        return dict(lookup)
    moved = {site: alid for site, alid in lookup.items() if _moved_by_a_lift(site, alid)}
    reached = set(moved.values())
    beside = {
        site: alid for site, alid in lookup.items() if alid in reached and site not in moved
    }
    return moved | _left_in_place(beside, chain_file=chain_file)


def _left_in_place(
    lookup: Mapping[SourceKey, str], *, chain_file: str | Path | None
) -> dict[SourceKey, str]:
    """The keys of ``lookup`` whose coordinate, read as hg19, the chain leaves on the
    locus of the ALID they reach."""
    sites = list(lookup)
    landed = lifted_positions([(str(s[0]), int(s[1])) for s in sites], chain_file=chain_file)
    return {
        site: lookup[site]
        for site, place in zip(sites, landed, strict=True)
        if place is not None and not _moved_by_a_lift(place, lookup[site])
    }


def _moved_by_a_lift(site: tuple[str, int] | SourceKey, alid: str) -> bool:
    """Whether ``site`` reaches ``alid`` from another ``(contig, position)``: the one
    rule for whether a reference's source key was lifted, since a GRCh38 key names its
    ALID's own locus."""
    contig, position, _rest = alid.split(":", 2)
    return int(site[1]) != int(position) or contig_label(site[0]) != contig


def _merged_collisions(*groups: LiftCollisions) -> LiftCollisions:
    """The collisions of every group, each ALID's sites once, in sorted order."""
    sites_by_alid: dict[str, set[SourceKey]] = {}
    for group in groups:
        for alid, sites in group.items():
            sites_by_alid.setdefault(alid, set()).update(sites)
    return {alid: tuple(sorted(sites)) for alid, sites in sorted(sites_by_alid.items())}


def _route_by_reference(
    union_pass: _UnionPass,
    authority: FromReference,
    policy: RsidConflictPolicy,
    label: str,
) -> VariantRouting:
    """A reference's routing, named by the reference and Pass 1's harvest.

    The harvest is Pass 1's candidates, by routing key, minus the tuples both
    assemblies declare (issue #255 rounds 4-5). It names every variant the
    reference leaves unnamed, and the reference's own names win (issue #258),
    so a reference that names some rsids is never used as the whole answer. The
    reference routes, and the keys it records as re-oriented are the ones
    re-oriented (issue #354).

    An artifact's source keys are coordinates its sources supplied, so they are its
    provenance. A plain reference has none: its site map is the identity of every
    ALID it lists, so the sites Pass 1 read, routed as Pass 2 routes them, are what
    is observed (#261).
    """
    reference, off_primary = _without_off_primary_lifts(authority.reference)
    reference, frozen = _without_frozen_collisions(
        reference, union_pass.sources, authority.chain_file
    )
    placed, rsid_by_site, read = _place_union_pass(union_pass, authority)
    observed = _observed_by(reference, lambda: placed.lookup)
    # A tuple declared under both assemblies is two loci; naming the lifted
    # one from the other's row would attach an rsid to a variant a different
    # association owns.
    candidates = rsid_by_site.without(placed.dropped)
    # What the harvest's own lift collided on, less the sources the reference routes
    # itself: those rows are placed, so nothing is dropped of them. Each is named by
    # the rsid its own source gave it, whether or not the reference names rsids.
    lift_drops = LiftDrops.of(
        _merged_collisions(_unrouted_collisions(placed.collisions, reference), frozen),
        rsid_by_site.lifted,
    )
    names = harvested_names(
        candidates, reference.source_lookup, placed.lookup, policy=policy, label=label,
        named=reference.rsid_by_alid,
    )
    del rsid_by_site
    return VariantRouting(
        reference.source_lookup,
        reference.reoriented,
        axis=reference.alids,
        names=names,
        stats=union_pass.stats,
        supplied=_supplied_by(reference, read),
        lifted_away=off_primary,
        lift_drops=lift_drops,
        observed=observed,
        plain_reference=not reference.explicit_source_keys,
    )


def _supplied_by(reference: VariantReference, read: SuppliedCoordinates) -> SuppliedCoordinates:
    """The coordinates a reference's build may record as ``source_alid``: an artifact's
    recorded source keys, which its extraction read, or the sites Pass 1 ``read``."""
    if reference.explicit_source_keys:
        return SuppliedCoordinates.of_sites(reference.source_lookup)
    return read


def _observed_by(
    reference: VariantReference, sites: Callable[[], Iterable[SourceKey]]
) -> dict[SourceKey, str] | None:
    """What a plain reference observed: of the ``sites`` a source supplied, each one
    it routes, with the ALID it reaches, matched by Routing Key as Pass 2 matches a row
    (`folded_site`) (#261). None for an artifact, whose source keys are observations
    already; ``sites`` is read only for a plain reference."""
    if reference.explicit_source_keys:
        return None
    routed: dict[SourceKey, str] = {}
    for site in sites():
        alid = reference.source_lookup.get(folded_site(site))
        if alid is not None:
            routed[site] = alid
    return routed

