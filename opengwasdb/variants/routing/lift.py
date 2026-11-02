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


def _lift_union_by_assembly(
    tuples_by_assembly: dict[str, set[SourceKey]],
    *,
    chain_file: str | Path | None,
    liftover_failure_threshold: float,
) -> tuple[SourceLookup, SourceLookup, ReorientedKeys]:
    """Lift each assembly's group; return the hg38 passthrough, the lifted, and
    the keys the lift re-oriented (issue #284)."""
    passthrough_lookup: SourceLookup = {}
    passthrough = tuples_by_assembly.pop("hg38", set())
    if passthrough:
        log.info("%d variants already GRCh38 -- no liftover needed", len(passthrough))
        for chrom, pos, ref, alt in passthrough:
            a1, a2 = sorted((ref, alt))
            passthrough_lookup[(chrom, pos, ref, alt)] = f"{chrom}:{pos}:{a1}:{a2}"
    lifted_lookup, swapped = _lift_hg19_union(
        tuples_by_assembly.pop("hg19", set()),
        chain_file=chain_file,
        liftover_failure_threshold=liftover_failure_threshold,
    )
    assert not tuples_by_assembly, f"unhandled source_assembly values: {sorted(tuples_by_assembly)}"
    return passthrough_lookup, lifted_lookup, swapped


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


def _lift_hg19_union(
    hg19_tuples: set[SourceKey],
    *,
    chain_file: str | Path | None,
    liftover_failure_threshold: float,
) -> tuple[SourceLookup, ReorientedKeys]:
    """Lift Pass 1's hg19 group; return its ALIDs and the keys it re-oriented.

    The richer `lift_hg19_to_hg38` rather than `build_liftover_lookup`, because
    a reverse-strand block re-orients each variant's statistics as well as
    renaming it (issue #284), and an ALID string cannot say which: a palindromic
    pair's ALID is identical under either projection.
    """
    if not hg19_tuples:
        return {}, set()
    log.info("Running liftover hg19 → hg38 (%d variants)", len(hg19_tuples))
    lifted = lift_hg19_to_hg38(
        hg19_tuples, chain_file=chain_file, failure_threshold=liftover_failure_threshold
    )
    lookup = {key: row.alid for key, row in lifted.items()}
    log.info("Liftover complete: %d variants mapped", len(lookup))
    return lookup, {key for key, row in lifted.items() if statistics_swap(key, row)}


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
