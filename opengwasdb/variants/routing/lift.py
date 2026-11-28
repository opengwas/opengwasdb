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

Two distinct source loci a lift puts on one ALID are a collision (issue #325):
every lifted source reaching that ALID is dropped from the placement, none is
merged with another, and each is returned so the build can record it. A locus is
a source ``(chromosome, position)``: one locus's alleles reported either way round
are one variant, which the canonical ALID already merges. The rule holds the
lifted group to itself: a GRCh38 source's variant at that ALID is one variant seen
from another build, as it was before (issue #85).
"""

from __future__ import annotations

import logging
import time
from collections.abc import (
    Callable,
    Collection,
    Iterable,
    Mapping,
    MutableMapping,
    MutableSet,
    Sequence,
)
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

from opengwasdb.build.liftover import (
    LiftedRow,
    lift_hg19_to_hg38,
    lifted_rows,
    statistics_swap,
)
from opengwasdb.encoding.timing import format_duration
from opengwasdb.store_reader.identity import is_primary_contig
from opengwasdb.variants import assembly_gaps
from opengwasdb.variants.assembly_gaps import grch38
from opengwasdb.variants.normalise import LiftedOffPrimary, canonical_alid
from opengwasdb.variants.routing.keys import (
    SourceKey,
    contig_label,
    raw_site_key,
    routing_key,
    split_raw_site_key,
)

log = logging.getLogger(__name__)

#: A source coordinate -> the hg38 ALID it routes to (Pass 1's product).
SourceLookup = dict[SourceKey, str]
#: The source keys a reverse-strand lift re-oriented, whose statistics follow
#: the projected alleles (issue #284).
ReorientedKeys = set[SourceKey]

class LiftedAwayReason(StrEnum):
    """Why a lift placed a site where no Store row can be (CONTEXT: Lifted Away): the
    Admission reason its rows are dropped and counted under. Each value is the
    Admission reason's own string, so `admission.Reason` reads it unchanged."""

    #: On an alt, random or unplaced contig (#376).
    OFF_PRIMARY = LiftedOffPrimary.reason
    #: On a primary contig, in an assembly gap (#410).
    IN_ASSEMBLY_GAP = assembly_gaps.IN_ASSEMBLY_GAP
    #: On a primary contig, past its last base (#410).
    PAST_CONTIG_END = assembly_gaps.PAST_CONTIG_END


#: The sites a lift placed where no Store row can be, by the key a build finds their
#: rows by (a Routing Key, or the Hybrid fold's raw key), each with its reason.
LiftedAway = Mapping[str, LiftedAwayReason]
#: The one empty `LiftedAway`: the default of every signature that takes one.
NO_LIFTED_AWAY: LiftedAway = MappingProxyType({})


def away_counts(away: Mapping[SourceKey, LiftedAwayReason]) -> dict[str, int]:
    """A lift's ``provenance.lift`` counts of the sites it placed where no Store row can
    be: ``n_off_primary`` (#376) and ``n_outside_sequence`` (#410), which with the sites
    it placed make up its ``n_lifted``."""
    outside = sum(reason is not LiftedAwayReason.OFF_PRIMARY for reason in away.values())
    return {"n_off_primary": len(away) - outside, "n_outside_sequence": outside}


class LiftSplit(NamedTuple):
    """A lift's rows by where they landed: ``primary`` where a Store can hold them, and
    ``away``, the source keys of the rest (Lifted Away), each with its reason: an alt,
    random or unplaced contig (#376), or a primary contig where GRCh38 has no sequence
    (#410)."""

    primary: dict[SourceKey, LiftedRow]
    away: dict[SourceKey, LiftedAwayReason]


def split_lifted(rows: Mapping[SourceKey, LiftedRow]) -> LiftSplit:
    """Split a lift's rows by where they landed (`LiftSplit`): the one place a lift's
    result is judged, which every path that lifts a source calls.

    The judgement is of the lifted coordinate, so it is made here and not at
    Admission, whose rows still carry the source's: the hg19->hg38 chain aligns
    2,372 bases of 9 blocks into GRCh38 gaps, and a row lifted there names no variant."""
    away = {key: LiftedAwayReason.OFF_PRIMARY for key, row in rows.items()
            if not is_primary_contig(row.chromosome)}
    placed = [(key, row) for key, row in rows.items() if key not in away]
    outside = _outside_sequence(placed)
    away.update(outside)
    _warn_outside_sequence(outside, rows)
    return LiftSplit({key: row for key, row in placed if key not in outside}, away)


def _outside_sequence(
    placed: Sequence[tuple[SourceKey, LiftedRow]],
) -> dict[SourceKey, LiftedAwayReason]:
    """The ``placed`` sites a lift put where GRCh38 has no sequence, each with its reason
    (`AssemblySequence.reasons_at`)."""
    reasons = grch38().reasons_at(
        (contig_label(row.chromosome) for _key, row in placed),
        (row.position for _key, row in placed),
    )
    return {
        key: LiftedAwayReason(reason)
        for (key, _row), reason in zip(placed, reasons, strict=True) if reason
    }


def _warn_outside_sequence(
    outside: Mapping[SourceKey, LiftedAwayReason], rows: Mapping[SourceKey, LiftedRow]
) -> None:
    """Say how many lifted sites landed where GRCh38 has no sequence, naming the first."""
    if outside:
        first = min(outside)
        row = rows[first]
        log.warning(
            "%d lifted variant(s) landed in a GRCh38 assembly gap or past their contig's "
            "end, where no variant is -- their rows are dropped (first: %s -> %s:%d)",
            len(outside), raw_site_key(first), row.chromosome, row.position,
        )


def lifted_away(
    sites: Collection[SourceKey], *, chain_file: str | Path | None
) -> dict[str, LiftedAwayReason]:
    """The routing keys of the hg19 ``sites`` the chain places where a Store holds no
    row (`split_lifted`), each with its Admission reason.

    For a build routed by a Variant Reference, whose unmatched hg19 rows are the
    only ones not lifted already: no failure threshold applies, since a site the
    chain places nowhere is `off_axis` as before."""
    rows, _attempts, _failures = lifted_rows(sites, chain_file=chain_file)
    return {routing_key(site): reason for site, reason in split_lifted(rows).away.items()}


class UnionPlacement(NamedTuple):
    """`place_union`'s result: the union's site lookup, the keys the lift
    re-oriented (#284), the routing keys both assemblies declared (#255), the
    routing keys of the sites the lift placed where no Store row can be, each with
    its Admission reason (#376, #410), and the lift collisions it dropped (#325)."""

    lookup: SourceLookup
    reoriented: ReorientedKeys
    dropped: set[str]
    away: dict[str, LiftedAwayReason]
    collisions: LiftCollisions


@dataclass(frozen=True)
class SitePlacement:
    """Where source sites land on GRCh38.

    ``alid`` maps each placed site to its hg38 ALID; a site whose lift failed
    is absent. ``reoriented`` holds the sites a reverse-strand chain block
    re-oriented: their stored alleles call the other allele A1, so a z signed
    by the source must be negated and an EAF complemented (issue #284).
    ``away`` holds the sites the lift placed where no Store row can be, each with
    its Admission reason (`split_lifted`): they are in neither ``alid`` nor
    ``reoriented``, and a build drops and counts their rows (#376, #410).
    ``collisions`` holds the sites dropped because another source locus reached
    the same ALID, by that ALID:
    they too are in neither ``alid`` nor ``reoriented`` (#325).
    """

    alid: SourceLookup
    reoriented: ReorientedKeys
    away: dict[SourceKey, LiftedAwayReason] = field(default_factory=dict)
    collisions: LiftCollisions = field(default_factory=dict)

    def drops(
        self,
        rsid_of: Callable[[SourceKey], str] = lambda site: "",
        *,
        spell: Callable[[SourceKey], str] = routing_key,
    ) -> LiftDrops:
        """The sites a lift collision dropped, as `LiftDrops` (#325)."""
        return LiftDrops.of(self.collisions, rsid_of, spell=spell)


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
    place, and a result keyed by the tuple could keep only one. Sites of two
    loci that reach one ALID are all dropped (`_drop_collisions`). Pass 1's union
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
    split = split_lifted(lifted)
    placement.away.update(split.away)
    alid = {key: row.alid for key, row in split.primary.items()}
    reoriented = {key for key, row in split.primary.items() if statistics_swap(key, row)}
    # The collision rule holds the lifted group to itself: a GRCh38 source's variant
    # is no lifted locus, and a lifted site onto it is one variant seen from two builds.
    placement.collisions.update(_drop_collisions(alid, reoriented))
    placement.alid.update(alid)
    placement.reoriented.update(reoriented)
    log.info("Liftover complete: %d variants mapped", len(lifted))
    off_primary = [
        key for key, reason in placement.away.items() if reason is LiftedAwayReason.OFF_PRIMARY
    ]
    if off_primary:
        log.warning(
            "%d lifted variant(s) landed on an alt, random or unplaced contig, "
            "which the Store does not hold -- their rows are dropped (first: %s)",
            len(off_primary), _first_key(off_primary),
        )


#: A lift collision: the ALID two or more source loci reached, and every site of
#: each of them, which are all dropped (issue #325).
LiftCollisions = dict[str, tuple[SourceKey, ...]]


def find_lift_collisions(lookup: Mapping[SourceKey, str]) -> LiftCollisions:
    """The ALIDs that sites of two or more source loci reach, with those sites.

    ``lookup`` is a lifted group's placement, which the caller holds to lifted sites
    alone: a GRCh38 source's variant at an ALID is that variant seen from another
    build, not a second locus (issue #85).

    A locus is a source ``(contig, position)``. The sites ``A:G`` and ``G:A`` at one
    position are one locus, which the canonical ALID merges; sites at two positions
    that reach one ALID are two variants, whose statistics must never be stored as
    one. One pass: the first site an ALID is reached by is kept by reference, and
    each later one is collected, so a collision's sites are known without a second
    scan of the map.
    """
    contigs: dict[str, str] = {}
    first: dict[str, SourceKey] = {}
    later: dict[str, list[SourceKey]] = {}
    for site, alid in lookup.items():
        if first.setdefault(alid, site) is not site:
            later.setdefault(alid, []).append(site)

    def locus(site: SourceKey) -> tuple[str, int]:
        contig = contigs.get(site[0])
        if contig is None:
            contig = contigs[site[0]] = contig_label(site[0])
        return contig, int(site[1])

    return {
        alid: tuple(sorted((first[alid], *sites)))
        for alid, sites in sorted(later.items())
        if any(locus(site) != locus(first[alid]) for site in sites)
    }


def require_one_locus_per_alid(lookup: Mapping[SourceKey, str], *, where: str) -> None:
    """Refuse a site map that puts two source loci on one ALID (issue #325).

    The guard a caller that places sites itself calls: a map that reached a write
    with a colliding ALID would store one source's statistics under another's
    identity and rsid, and nothing downstream could tell. ``where`` names the
    caller in the refusal. `place_sites` drops collisions by the same scan, so the
    map it returns passes by construction."""
    collisions = find_lift_collisions(lookup)
    if collisions:
        alid, sites = next(iter(collisions.items()))
        raise ValueError(
            f"{where}: {len(collisions)} ALID(s) are reached by two or more source loci "
            f"(first: {alid} by {', '.join(map(raw_site_key, sites))}); "
            "a lift collision must be dropped, never routed"
        )


def _drop_collisions(
    lookup: MutableMapping[SourceKey, str], reoriented: MutableSet[SourceKey]
) -> LiftCollisions:
    """Drop from ``lookup`` every site of every ALID that two or more source loci
    reach, and from ``reoriented`` too; return what was dropped, by ALID.

    The one rule for a lift collision, which every path that places sites reaches
    through `place_sites`: no source locus keeps the ALID, since which one owns it
    is not knowable from the chain, and keeping either would attach its statistics
    to the other's rsid. A precedent is `_drop_cross_assembly_collisions`'
    both-assembly drop."""
    collisions = find_lift_collisions(lookup)
    dropped = collided_sites(collisions)
    for site in dropped:
        del lookup[site]
        reoriented.discard(site)
    if collisions:
        first, sites = next(iter(collisions.items()))
        log.warning(
            "%d ALID(s) are each reached by two or more source loci (first: %s, by %s) "
            "-- all %d source site(s) are dropped rather than one given the ALID",
            len(collisions),
            first,
            ", ".join(map(raw_site_key, sites)),
            len(dropped),
        )
    return collisions


def collided_sites(collisions: Mapping[str, tuple[SourceKey, ...]]) -> set[SourceKey]:
    """Every site in ``collisions``: the one owner of that set."""
    return {site for sites in collisions.values() for site in sites}


class DroppedSource(NamedTuple):
    """One source site a lift collision dropped, by both identities (issue #325):
    the ALID it was lifted to, the site as ``chrom:pos:ref:alt``, and the rsid its
    source named it, ``""`` when none did."""

    alid: str
    source_key: str
    rsid: str


@dataclass(frozen=True)
class LiftDrops:
    """What a lift collision dropped: each source site with both identities
    (``sources``, the Admission record's ``lift_collisions.tsv``), and ``keys``, the
    spelling of each by which a build recognises its rows (the routing key Pass 2's
    unmatched rows carry, or the raw key the Hybrid fold's do), so their rows are
    counted as `lift_collision`."""

    sources: tuple[DroppedSource, ...] = ()
    keys: frozenset[str] = frozenset()
    #: Each source's key as ``keys`` spells it, parallel to ``sources``.
    spelled: tuple[str, ...] = ()

    @classmethod
    def of(
        cls,
        collisions: LiftCollisions,
        rsid_of: Callable[[SourceKey], str] = lambda site: "",
        *,
        spell: Callable[[SourceKey], str] = routing_key,
    ) -> LiftDrops:
        """``collisions`` as drops, each named by the rsid ``rsid_of`` gives its site."""
        entries = [(alid, site) for alid, sites in collisions.items() for site in sites]
        spelled = tuple(spell(site) for _alid, site in entries)
        return cls(
            tuple(DroppedSource(alid, raw_site_key(site), rsid_of(site)) for alid, site in entries),
            frozenset(spelled),
            spelled,
        )

    def found_by(self, collisions: LiftCollisions) -> LiftDrops:
        """``collisions``, which a Hybrid fold found among its spills' keys, as these
        drops -- its harvest's -- name the same sites: as the source wrote the site, which
        a spill does not keep (it upper-cases, #360), with the rsid its hg19 sources named
        it. ``keys`` spell each site as the fold's rows are found by, `raw_site_key`.

        The harvest's lift read every hg19 site the fold spills, so it met each of these
        at the same ALID; one it did not is refused, never named by a guess. Of several
        spellings of one Routing Key, a named one comes first, then the fold's own."""
        harvested: dict[tuple[str, str], list[DroppedSource]] = {}
        for source in self.sources:
            key = (source.alid, routing_key(split_raw_site_key(source.source_key)))
            harvested.setdefault(key, []).append(source)
        sources: list[DroppedSource] = []
        spelled: list[str] = []
        for alid, sites in collisions.items():
            for site in sites:
                spill = raw_site_key(site)
                found = harvested.get((alid, routing_key(site)))
                if not found:
                    raise ValueError(
                        f"the fold dropped {spill} as a lift collision at {alid}, which the "
                        "harvest's own lift did not; the two disagree on what collided"
                    )
                sources.append(
                    min(found, key=lambda s: (not s.rsid, s.source_key != spill, s.source_key))
                )
                spelled.append(spill)
        return LiftDrops(tuple(sources), frozenset(spelled), tuple(spelled))

    @classmethod
    def read_back(cls, sources: Iterable[DroppedSource], spelled: Iterable[str]) -> LiftDrops:
        """Drops recorded as ``sources``, with the key each was found by (``spelled``)."""
        held, keys = tuple(sources), tuple(spelled)
        return cls(held, frozenset(keys), keys)

    def among(self, seen: Collection[str]) -> tuple[DroppedSource, ...]:
        """The sources whose rows a build really dropped, as ``seen`` names them
        (``chrom:pos:ref:alt``, the way the dropped row's source spells it).

        The record names decisions, not candidates: a candidate no row of the build met
        is not listed. Every row a gate dropped as a collision was read from a source the
        harvest read too, so its spelling is a candidate's; one that is not means the two
        disagree on what a source spells, and the build refuses rather than name a guess."""
        by_raw = {source.source_key: source for source in self.sources}
        unknown = sorted(set(seen) - by_raw.keys())
        if unknown:
            raise ValueError(
                f"{len(unknown)} row(s) were dropped as lift collisions that no harvested site "
                f"spells (first: {unknown[0]}); the harvest and the rows disagree on a spelling"
            )
        return tuple(sorted(by_raw[raw] for raw in seen))


def _first_key(sites: Collection[SourceKey]) -> str:
    return raw_site_key(min(sites))


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
    placement (its re-oriented keys, issue #284, and the sites it placed where no
    Store row can be, #376, #410). Kept apart so the both-assembly drop can compare them."""
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
    away: dict[SourceKey, LiftedAwayReason],
    collided: AbstractSet[SourceKey] = frozenset(),
) -> set[str]:
    """Apply `cross_assembly_collisions` to the union's two placed groups, and
    return the dropped routing keys, which a harvest also drops from its
    candidates.

    A site lifted where no Store row can be (``away``, #376, #410) is a lifted locus
    all the same: a GRCh38 source spelling its key would otherwise take the row that
    is not its, and so is a site dropped as a lift collision (``collided``, #325): its
    rows reach no ALID, and must not route by its raw key onto a GRCh38 row that spells it.
    """
    dropped = cross_assembly_collisions(
        passthrough_lookup.keys(), lifted_lookup.keys() | away.keys() | collided
    )
    log_cross_assembly_drop(len(dropped))
    if dropped:
        for lookup in (passthrough_lookup, lifted_lookup, away):
            for site in _spelling(lookup, dropped):
                del lookup[site]
                swapped.discard(site)
    return dropped


def _spelling(sites: Iterable[SourceKey], keys: AbstractSet[str]) -> list[SourceKey]:
    """The ``sites`` whose routing key is one of ``keys``."""
    return [site for site in sites if routing_key(site) in keys]


def place_union(
    tuples_by_assembly: dict[str, set[SourceKey]],
    *,
    chain_file: str | Path | None,
    failure_threshold: float,
) -> UnionPlacement:
    """Place the union on hg38 (`UnionPlacement`): the both-assembly tuples it
    dropped are routing keys (issue #255 rounds 4-5), and so are the sites the
    lift placed where no Store row can be, whose rows a build drops and counts by
    the reason each carries (#376, #410).
    """
    finalise_start = time.monotonic()
    passthrough, lifted = _lift_union_by_assembly(
        tuples_by_assembly,
        chain_file=chain_file,
        liftover_failure_threshold=failure_threshold,
    )
    passthrough_lookup, lifted_lookup = passthrough.alid, lifted.alid
    swapped, away = lifted.reoriented, lifted.away
    dropped_keys = _drop_cross_assembly_collisions(
        passthrough_lookup, lifted_lookup, swapped, away,
        collided_sites(lifted.collisions),
    )
    source_lookup = {**passthrough_lookup, **lifted_lookup}
    log.info(
        "Pass 1 liftover/finalisation: %d source variants → %d hg38 ALIDs in %s",
        len(source_lookup),
        len(set(source_lookup.values())),
        format_duration(time.monotonic() - finalise_start),
    )
    return UnionPlacement(
        source_lookup, swapped, dropped_keys,
        {routing_key(site): reason for site, reason in away.items()}, lifted.collisions,
    )
