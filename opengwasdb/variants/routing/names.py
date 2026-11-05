"""The rs identifier each routed variant carries (issues #109, #192, #255, #306, #379).

Pass 1 reads each source's names with its sites, first-named in ``(rank,
site)`` order. Every harvest keys them by Routing Key, rekeys them once
through the routing the build actually uses, and compares the result with an
independent walk of the raw names (the oracles in
`opengwasdb.variants.reference`): a union the build routes by
(`union_names`), or a Variant Reference that names none of its variants
(`harvested_names`). A key whose sources name it differently is refused or
left unnamed, as ``--on-rsid-conflict`` says, and counted, whichever routes.

A Hybrid build routes the keys the reference does not know only in its fold,
after Pass 2, so its names stay pending until `RoutedNames.finish` is given
that routing.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from opengwasdb.variants.reference import (
    expected_rsids_by_routed_key,
    require_rsids_match_expected,
    require_written_rsids_match,
)
from opengwasdb.variants.routing.keys import SourceKey, routing_key
from opengwasdb.variants.rsid import RsidConflictPolicy, RsidConflicts, withhold_disputed

log = logging.getLogger(__name__)

#: Where the keys the authority did not route ended up: a routing key -> its
#: ALID, or None when nothing routed it.
FoldRoute = Callable[[str], str | None]

#: What a refusal or a warning calls the build whose names are harvested,
#: unless the build says otherwise.
HARVEST_LABEL = "Single-pass build"


#: A candidate or routing key: the raw source tuple, or the string form the
#: build's routing table keys it by (`routing_key`).
_SiteKeyT = TypeVar("_SiteKeyT")


def _rekey_rsids_to_alids(
    source_lookup: Mapping[_SiteKeyT, str],
    rsid_by_site: Mapping[_SiteKeyT, str],
) -> dict[str, str]:
    """The rsid each ALID carries: the first non-empty in ``(rank, site)`` order.

    ``rsid_by_site`` is produced in that order (issue #192), so iterating it
    directly pins the winner for an ALID several source keys resolve to.
    Iterating ``source_lookup`` would fall back to the union set's per-process
    string hash order and let the answer move between runs (issue #192).
    """
    by_alid: dict[str, str] = {}
    for site, rsid in rsid_by_site.items():
        alid = source_lookup.get(site)
        if alid and rsid:
            by_alid.setdefault(alid, rsid)
    return by_alid


def _candidates_by_routing_key(
    rsid_by_site: Mapping[SourceKey, str],
    *,
    label: str,
    on_conflict: RsidConflictPolicy = RsidConflictPolicy.REFUSE,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """The harvest's candidates by routing key, order kept, and the keys whose
    sources name them differently.

    A routing key is not injective -- it upper-cases alleles -- and two raw
    sites that collapse onto one key are one stored variant, so the
    first non-empty rsid in `(rank, site)` order wins. Two *different* non-empty
    names for one key cannot be resolved honestly, so this never picks one:
    under ``refuse`` the first such key fails the build (issue #255 round 4);
    under ``drop`` every such key's candidate is "", and the second value maps
    each to its distinct names in `(rank, site)` order, for the build to report
    and for `withhold_disputed` to unname the variants they reach (issue
    #306).
    """
    candidates: dict[str, str] = {}
    first_site: dict[str, SourceKey] = {}
    conflicts: dict[str, list[str]] = {}
    for site, rsid in rsid_by_site.items():
        folded = routing_key(site)
        existing = candidates.get(folded)
        if existing is None:
            candidates[folded] = rsid
            first_site[folded] = site
        elif not existing:
            candidates[folded] = rsid
        elif rsid and rsid != existing:
            if on_conflict is RsidConflictPolicy.REFUSE:
                raise _rsid_conflict_error(label, folded, first_site[folded], existing, site, rsid)
            _note_conflict(conflicts, folded, existing, rsid)
    # Blanked only now: a key blanked mid-walk would take the next name it met.
    for folded in conflicts:
        candidates[folded] = ""
    return candidates, conflicts


def _note_conflict(
    conflicts: dict[str, list[str]], folded: str, existing: str, rsid: str
) -> None:
    """Add ``rsid`` to the distinct names ``folded``'s sources gave it."""
    names = conflicts.setdefault(folded, [existing])
    if rsid not in names:
        names.append(rsid)


def _rsid_conflict_error(
    label: str, folded: str, first: SourceKey, existing: str, site: SourceKey, rsid: str
) -> ValueError:
    """The refusal for two names on one routing key (issue #255 round 4)."""
    return ValueError(
        f"{label}: two source variants case-fold to one routing key "
        f"{folded!r} with different rsids: {first!r} names "
        f"{existing!r} but {site!r} names {rsid!r}; both associations route "
        "to one stored variant, so refusing to choose a name -- a wrong "
        "answer indistinguishable from a right one"
    )


def _routing_lookup(source_lookup: Mapping[SourceKey, str]) -> dict[str, str]:
    """A site -> ALID routing, by routing key."""
    return {routing_key(site): alid for site, alid in source_lookup.items()}


def _log_routing_disagreements(
    candidates: Mapping[str, str],
    routing: Mapping[str, str],
    pass1_lookup: Mapping[SourceKey, str] | None,
    *,
    label: str,
) -> None:
    """Log named sites the build's routing places differently from Pass 1's.

    The build's routing is the authority; Pass 1's lift is only a second
    opinion, so a disagreement is reported rather than resolved.
    """
    if pass1_lookup is None:
        return
    # Only a site the reference actually routes can disagree with Pass 1: a site
    # it does not route is off-reference and is dropped (Dense) or resolved by
    # the fold (Hybrid), which is the documented behaviour, not a disagreement.
    disagreed = sum(
        1
        for site, pass1_alid in pass1_lookup.items()
        if (key := routing_key(site)) in candidates
        and (routed := routing.get(key)) is not None
        and routed != pass1_alid
    )
    if disagreed:
        log.warning(
            "%s: %d source variant(s) that named an rsid route to a different ALID "
            "under the reference than under Pass 1's own liftover; the reference "
            "routing wins, so the name stays on the variant its association uses",
            label,
            disagreed,
        )


def _rekey_and_verify(
    rsid_by_site: Mapping[SourceKey, str],
    routing: Mapping[str, str],
    *,
    label: str,
    on_conflict: RsidConflictPolicy,
) -> tuple[dict[str, str], set[str], dict[str, list[str]]]:
    """Rekey harvested candidates ONCE through ONE routing, and verify exactly.

    ``routing`` must be keyed by routing key and be the *final* routing -- a
    reference's source keys combined with, for a Hybrid build, the fold's
    off-reference resolution. One rekey in the candidates' `(rank, site)` order
    and one global oracle comparison, so no partition can win a name the global
    first-candidate rule would not give it (issue #255).

    The oracle walks the **raw** ``rsid_by_site`` through `routing_key` itself rather
    than reading the keyed candidates, so the key preparation is checked: a
    collapse that keeps the wrong name disagrees with the raw walk and fails
    (issue #255 round 4). Under ``drop`` it finds the disputed keys itself too,
    so a variant one reaches that kept a name fails as well (issue #306).
    Returns the resolved map, the variants left unnamed under ``drop``, and
    each disputed key's names.
    """
    candidates, conflicts = _candidates_by_routing_key(
        rsid_by_site, label=label, on_conflict=on_conflict
    )
    resolved = _rekey_rsids_to_alids(routing, candidates)
    del candidates
    withheld = withhold_disputed(resolved, routing, conflicts)
    expected = expected_rsids_by_routed_key(
        rsid_by_site, routing, routing_key, drop_disputed=on_conflict is RsidConflictPolicy.DROP
    )
    require_rsids_match_expected(resolved, expected)
    return resolved, withheld, conflicts


def _log_dropped_rsids(label: str, conflicts: RsidConflicts) -> None:
    """Say how many variants were left unnamed because their sources disagree."""
    if not conflicts.n_keys:
        return
    first = conflicts.sample[0]
    log.warning(
        "%s: %d routing key(s) are named differently by their sources, and the %d "
        "stored variant(s) they reach have no rsid (--on-rsid-conflict drop), e.g. %s "
        "named %s",
        label,
        conflicts.n_keys,
        conflicts.n_variants,
        first.key,
        " and ".join(first.names),
    )


@dataclass(frozen=True)
class _Harvest:
    """What a pending harvest still needs: the raw names, in ``(rank, site)``
    order, and the authority's routing of the keys they name."""

    rsid_by_site: Mapping[SourceKey, str]
    candidate_routing: Mapping[str, str]
    label: str


@dataclass(frozen=True)
class RoutedNames:
    """The rs identifier each routed ALID carries, and what the sources disputed.

    ``rsid_by_alid`` names the variants the authority routes. It is final
    unless `pending`: a harvest over a Variant Reference also names source
    keys the reference does not route, and only the caller knows where those
    end up. `finish` takes that routing, rekeys every name once over the
    combined routing and checks the result exactly; until then the map holds
    only the reference-routed names, and `require_written` refuses it.

    ``conflicts`` counts the routing keys whose sources name them differently
    and, once final, the variants they reach. ``reference_named`` says the
    reference supplied its own names and no source was read for them.
    """

    rsid_by_alid: dict[str, str]
    conflicts: RsidConflicts
    reference_named: bool = False
    _harvest: _Harvest | None = None

    @property
    def pending(self) -> bool:
        """Whether `finish` must still be given the routing of the unrouted keys."""
        return self._harvest is not None

    def finish(self, route: FoldRoute | None = None) -> RoutedNames:
        """The final names, once every key the authority did not route is placed.

        ``route`` gives a key's ALID or None; omitted, nothing routes those
        keys -- their associations are dropped, and so are their names. Raises
        ``ValueError`` on names already final, so a second routing can never
        be ignored in silence.
        """
        harvest = self._harvest
        if harvest is None:
            raise ValueError("these names are already final; finish() runs once")
        combined = harvest.candidate_routing
        if route is not None:
            combined = {**combined, **_unrouted_keys(harvest, route)}
        label = harvest.label
        merged, withheld, _disputed = _rekey_and_verify(
            harvest.rsid_by_site,
            combined,
            label=label,
            on_conflict=self.conflicts.policy,
        )
        if merged == self.rsid_by_alid:
            # Unchanged: keep the map a caller may already have written, order and all.
            merged = self.rsid_by_alid
        else:
            log.info(
                "Single-pass build: %d named candidate(s) rekeyed against the combined "
                "reference/fold routing",
                len(combined),
            )
        # Only the combined routing says which stored variant each disputed key reaches.
        conflicts = self.conflicts.reaching(len(withheld))
        _log_dropped_rsids(label, conflicts)
        return RoutedNames(merged, conflicts, reference_named=self.reference_named)

    def require_written(self, store: str | Path, alids: Sequence[str]) -> None:
        """Refuse a written Store Variant Table that disagrees with these names.

        Read back row for row (`require_written_rsids_match`), so a partial
        loss or a table written from a stale map fails on the bytes a query
        reads (issue #255). Pending names are refused outright.
        """
        if self.pending:
            raise ValueError(
                "the routed names are pending the routing of off-reference keys; "
                "finish() them before checking a written table"
            )
        require_written_rsids_match(store, alids, self.rsid_by_alid)


def _unrouted_keys(harvest: _Harvest, route: FoldRoute) -> dict[str, str]:
    """``route``'s ALID for every named key the authority did not route.

    Keyed exactly as the candidates are, so the combined routing speaks the
    key form the association routing uses. A named key ``route`` does not
    place has no entry, and its name stays unspent (issue #255).
    """
    routing: dict[str, str] = {}
    for site in harvest.rsid_by_site:
        key = routing_key(site)
        if key in harvest.candidate_routing:
            continue
        alid = route(key)
        if alid is not None:
            routing[key] = alid
    return routing


def harvested_names(
    rsid_by_site: Mapping[SourceKey, str],
    reference_lookup: Mapping[SourceKey, str],
    pass1_lookup: Mapping[SourceKey, str],
    *,
    policy: RsidConflictPolicy,
    label: str = HARVEST_LABEL,
) -> RoutedNames:
    """Pending names for a reference that names none, from Pass 1's harvest.

    The harvest is keyed as the build routes, so two raw sites one routing key
    spells -- one site in two cases -- are one stored variant; a key named two
    ways is refused or left unnamed, as ``policy`` says (issue #306). The
    reference-routed subset is rekeyed now, for a table written before the
    unrouted keys are placed; the raw names travel on, not a keyed copy, so
    `RoutedNames.finish`'s oracle can walk them through `routing_key` itself.
    """
    candidates, conflicts = _candidates_by_routing_key(
        rsid_by_site, label=label, on_conflict=policy
    )
    reference_routing = _routing_lookup(reference_lookup)
    candidate_routing = {
        routed: alid for routed, alid in reference_routing.items() if routed in candidates
    }
    _log_routing_disagreements(candidates, reference_routing, pass1_lookup, label=label)
    del reference_routing
    on_reference = _rekey_rsids_to_alids(candidate_routing, candidates)
    withhold_disputed(on_reference, candidate_routing, conflicts)
    return RoutedNames(
        on_reference,
        RsidConflicts.found(policy, conflicts),
        _harvest=_Harvest(rsid_by_site, candidate_routing, label),
    )


def refuse_disputed_names(rsid_by_site: Mapping[SourceKey, str], *, label: str) -> None:
    """Refuse the first routing key ``rsid_by_site``'s sites name two ways.

    The test every harvest applies (`_candidates_by_routing_key`), for a
    writer that has no policy to offer and keeps its own first-named rule:
    with no key disputed, that rule names every variant as the union does.
    """
    _candidates_by_routing_key(rsid_by_site, label=label, on_conflict=RsidConflictPolicy.REFUSE)


def union_names(
    rsid_by_site: Mapping[SourceKey, str],
    site_alid: Mapping[SourceKey, str],
    *,
    policy: RsidConflictPolicy,
    label: str,
) -> RoutedNames:
    """Final names for a union the build routes by: every site is placed already.

    The same harvest as `harvested_names`: candidates keyed by Routing Key,
    so a site's spellings are compared, rekeyed once through the union's own
    placement and checked against the raw walk. A key
    named two ways is refused or left unnamed, as ``policy`` says, so which
    source is listed first never decides a name its sources dispute (issue
    #379). Only named sites are keyed: they are every key a name can reach.
    """
    routing = {
        routing_key(site): alid for site in rsid_by_site if (alid := site_alid.get(site))
    }
    resolved, withheld, disputed = _rekey_and_verify(
        rsid_by_site, routing, label=label, on_conflict=policy
    )
    conflicts = RsidConflicts.found(policy, disputed).reaching(len(withheld))
    _log_dropped_rsids(label, conflicts)
    return RoutedNames(resolved, conflicts)
