"""The Indexed Variant Subsets a release publishes, judged over the association scan.

`subsets.indexed` holds every published Indexed Variant Subset to the rules the
build's publication gate runs on it, there over a deterministic sample and here over
every cell (#270). Comparing every cell used to read the primary planes again, a
subset band at a time, after the association scan had read them once already: on
OGS-00009 that second read was 11,030 s of a 16,974 s validation. The rule now rides
the scan (issue #314): the pass hands it each band it decodes, and the subset's
share of the release's EAF exception table and per-variant arrays, so the rule reads
only the index. A subset the pass could not carry is compared as before.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from functools import partial

import numpy as np

from opengwasdb.encoding import EAF_BASELINE, EAF_REFERENCE
from opengwasdb.layouts.dense.indexed_subsets import (
    INDEXED_SUBSETS_GROUP,
    SubsetRide,
    subset_rides,
    validate_indexed_subsets,
)
from opengwasdb.validation.rule import Check, Kind, Ride, Rule, Target
from opengwasdb.validation.rules.associations import grid_counts
from opengwasdb.validation.rules.release import REQUIRED
from opengwasdb.validation.scans import ASSOCIATIONS, Block, eaf_table, per_variant

#: The rule's name.
INDEXED = "subsets.indexed"
#: Where a pass leaves each subset's share of it for the rule, judged after the pass.
_RIDES = f"{INDEXED}/rides"


def _board(target: Target) -> Callable[[Block], None] | None:
    """Ready a ride for every published subset, and say what each band is handed to."""
    group = target.group
    if INDEXED_SUBSETS_GROUP not in group:
        return None
    _n_variants, n_analyses = grid_counts(target)
    rides = subset_rides(group, target.encoding, n_analyses)
    if not rides:
        return None
    table = eaf_table(target)
    for ride in rides.values():
        _share_per_variant(target, ride)
        if table is not None:
            table.watch(ride.watch)
            ride.saw_table = partial(table.saw_all, ride.watch)
    target.remember(_RIDES, lambda: rides)
    return partial(_see, rides)


def _share_per_variant(target: Target, ride: SubsetRide) -> None:
    """The subset's share of the per-variant arrays the scan holds for its decode."""
    group, eaf = target.group, target.encoding.eaf
    if eaf.is_residual and EAF_BASELINE in group:
        ride.baseline = per_variant(target, EAF_BASELINE)[ride.subset]
    if eaf.reference and EAF_REFERENCE in group:
        ride.reference = per_variant(target, EAF_REFERENCE)[ride.subset]


def _see(rides: Mapping[str, SubsetRide], block: Block) -> None:
    """Hand one band to every subset's ride; one the band breaks reads for itself."""
    r0 = block.offset // block.n_analyses
    for ride in rides.values():
        if not ride.broken:
            try:
                ride.see(r0, r0 + block.rows, block.n_analyses, partial(_plane, block))
            except Exception:  # that subset is compared against a read of its own
                ride.broken = True


def _plane(block: Block, name: str) -> np.ndarray:
    """One decoded plane of the band, as the comparison compares it; raises where the
    band holds none to compare."""
    if name == "z":
        return block.z
    if name == "se":
        return block.se
    held = block.eaf if name == "eaf" else block.mask
    if held is None:
        raise ValueError(f"the band holds no {name} to compare")
    return held


def _indexed_subsets(target: Target) -> Iterator[str]:
    """Every published Indexed Variant Subset is self-consistent and value-equal.

    The rules are the ones the build's publication gate runs on a subset before
    publishing it, there over a deterministic sample and here over every cell
    (#270).
    """
    errors: list[str] = []
    store = target.store
    n_variants, n_analyses = grid_counts(target)
    rides: Mapping[str, SubsetRide] | None = target.forget(_RIDES)
    # The optional indexed-subset namespace (ADR 0061, #264).  A release without it
    # stays valid and unchanged; a release with it must hold only published,
    # self-consistent, value-equal subsets.
    validate_indexed_subsets(
        store.path, target.group, store.manifest, n_variants, n_analyses, errors, rides=rides
    )
    yield from errors


RULES: tuple[Rule, ...] = (
    Check(
        name=INDEXED,
        kinds=Kind.GRID,
        after=(REQUIRED,),
        run=_indexed_subsets,
        ride=Ride(ASSOCIATIONS, _board),
    ),
)
