"""`route_variants`: which Variant Index place every source coordinate reaches.

The authority decides. Under `LiftedUnion` the sources themselves are the
axis: Pass 1 reads each once, lifts the hg19 group to GRCh38 and names each
ALID by its first-named rsid. Under `FromReference` a Variant Reference is
the axis and its source keys the routing; Pass 1 runs only to harvest names
the reference lacks, and only when it names none. Either harvest compares the
names a site's spellings carry, by one rule (issue #379).
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from opengwasdb.readers.registry import SourceSpec, open_source
from opengwasdb.variants.axis import sorted_alids
from opengwasdb.variants.reference import (
    VariantReference,
    warn_reference_left_rows_blank,
)
from opengwasdb.variants.routing.keys import SourceKey, routing_key, source_alid
from opengwasdb.variants.routing.lift import place_union
from opengwasdb.variants.routing.names import (
    HARVEST_LABEL,
    RoutedNames,
    harvested_names,
    union_names,
)
from opengwasdb.variants.routing.table import RouteTable
from opengwasdb.variants.routing.union import (
    Pass1Stats,
    ReaderFactory,
    UnionOptions,
    consume_shards,
    materialize_site_union,
)
from opengwasdb.variants.rsid import RsidConflictPolicy, RsidConflicts

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
    the two-pass one does (issue #354). Pass 1 runs only when the reference
    names no rsids, to harvest them; its lift then uses ``chain_file`` and
    ``failure_threshold``, though it routes nothing.
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
    """

    def __init__(
        self,
        site_alid: Mapping[SourceKey, str],
        reoriented: Collection[SourceKey],
        *,
        axis: Sequence[str] | None,
        names: RoutedNames,
        stats: Pass1Stats,
        off_primary: frozenset[str] = frozenset(),
    ) -> None:
        self._site_alid: Mapping[SourceKey, str] | None = site_alid
        self._reoriented = reoriented
        self._axis = axis
        self.names = names
        self.stats = stats
        self.off_primary = off_primary

    def _sites(self) -> Mapping[SourceKey, str]:
        if self._site_alid is None:
            raise ValueError("the site map was consumed by table(); read it before")
        return self._site_alid

    @cached_property
    def alids(self) -> list[str]:
        """Every ALID the authority routes to, unique and in genomic order."""
        if self._axis is not None:
            return sorted_alids(self._axis)
        return sorted_alids(self._sites().values())

    def source_alids(self) -> dict[str, str | None]:
        """Each routed ALID's source-build ALID: its sites' own coordinate.

        The Store Variant Table's ``source_alid`` provenance. Several sites can
        reach one ALID (a liftover collapse, two assemblies); when they name
        different origins the provenance is ambiguous and is ``None``, never a
        guess (issue #85). Computed afresh on each call.
        """
        by_alid: dict[str, str | None] = {}
        for site, alid in self._sites().items():
            source = source_alid(site)
            if alid not in by_alid:
                by_alid[alid] = source
            elif by_alid[alid] != source:
                by_alid[alid] = None
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
    if isinstance(authority, FromReference):
        return _route_by_reference(
            union_pass, authority, on_rsid_conflict, label or HARVEST_LABEL
        )
    tuples_by_assembly, rsid_by_site = union_pass.sites()
    site_alid, reoriented, dropped, off_primary = place_union(
        tuples_by_assembly,
        chain_file=authority.chain_file,
        failure_threshold=authority.failure_threshold,
    )
    names = union_names(
        _undropped(rsid_by_site, dropped),
        site_alid,
        policy=on_rsid_conflict,
        label=label or UNION_LABEL,
    )
    return VariantRouting(
        site_alid, reoriented, axis=None, names=names, stats=union_pass.stats,
        off_primary=frozenset(off_primary),
    )


def _undropped(
    rsid_by_site: dict[SourceKey, str], dropped: Collection[str]
) -> dict[SourceKey, str]:
    """The named sites but those whose key both assemblies declare.

    Such a key is two loci and is placed nowhere (issue #255 rounds 4-5), so
    its sites name nothing; comparing their names would count, or refuse, a
    dispute no stored variant has.
    """
    if not dropped:
        return rsid_by_site
    return {site: rsid for site, rsid in rsid_by_site.items() if routing_key(site) not in dropped}


@dataclass(frozen=True)
class _UnionPass:
    """Pass 1 over one build's sources, read when a routing needs their sites."""

    sources: Sequence[SourceSpec]
    n_workers: int
    options: UnionOptions
    stats: Pass1Stats
    readers: ReaderFactory

    def sites(self) -> tuple[dict[str, set[SourceKey]], dict[SourceKey, str]]:
        """The site union by assembly, and each site's first-named rsid in
        ``(rank, site)`` order."""
        result: tuple[dict[str, set[SourceKey]], dict[SourceKey, str]] = consume_shards(
            self.sources,
            materialize_site_union,
            n_workers=self.n_workers,
            options=self.options,
            stats=self.stats,
            readers=self.readers,
        )
        return result


def _route_by_reference(
    union_pass: _UnionPass,
    authority: FromReference,
    policy: RsidConflictPolicy,
    label: str,
) -> VariantRouting:
    """A reference's routing, named by the reference or by Pass 1's harvest.

    A reference that names rsids is used as given, and nothing is read. One
    that names none is harvested: Pass 1's candidates, by routing key, minus
    the tuples both assemblies declare (issue #255 rounds 4-5). Either way the
    reference routes, and the keys it records as re-oriented are the ones
    re-oriented (issue #354).
    """
    reference = authority.reference
    if reference.rsid_by_alid:
        warn_reference_left_rows_blank(reference)
        names = RoutedNames(
            dict(reference.rsid_by_alid), RsidConflicts(policy, None), reference_named=True
        )
    else:
        tuples_by_assembly, rsid_by_site = union_pass.sites()
        pass1_lookup, _reoriented, dropped, _off_primary = place_union(
            tuples_by_assembly,
            chain_file=authority.chain_file,
            failure_threshold=authority.failure_threshold,
        )
        # A tuple declared under both assemblies is two loci; naming the lifted
        # one from the other's row would attach an rsid to a variant a different
        # association owns.
        candidates = _undropped(rsid_by_site, dropped)
        del rsid_by_site
        names = harvested_names(
            candidates, reference.source_lookup, pass1_lookup, policy=policy, label=label
        )
    return VariantRouting(
        reference.source_lookup,
        reference.reoriented,
        axis=reference.alids,
        names=names,
        stats=union_pass.stats,
    )
