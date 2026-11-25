"""Rules for the optional Rho Matrix, `data.zarr/rho` (PRD "Validation", issue 049, ADR 0074).

A release without it stays valid: Rho is opt-in, add-in-place metadata (ADR 0025),
not association data. These rules are structural and consistency rules only;
none re-runs the estimator. A Hybrid release's Rho Matrix is its Dense Component's
and records exactly that component's Overflow-only Analyses, which `rho.overflow_only`
judges against the association scan (#496).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import numpy as np

from opengwasdb.store_reader.arrays import array_length
from opengwasdb.store_reader.associations.record import State, states
from opengwasdb.store_reader.rho import OVERFLOW_ONLY_ATTR, packed_rho, recorded_overflow_only
from opengwasdb.validation.rule import Check, Fold, Kind, Rule, Target
from opengwasdb.validation.rules.associations import VALUES_AFTER, grid_counts
from opengwasdb.validation.rules.release import REQUIRED
from opengwasdb.validation.scans import ASSOCIATIONS, Block

#: The provenance every Rho Matrix records.
REQUIRED_ATTRS = (
    "z_thresh",
    "min_nulls",
    "grid_n",
    "window_bp",
    "n_variants_used",
    "observed_only",
    "method",
    "n_analyses",
)
#: The rule every other rule here runs after: the group holds what it must.
MEMBERS = "rho.members"
#: The rule the matrix is read after: it is sized for the release's Analyses.
ANALYSES = "rho.analyses"
#: The rule the matrix's values are read after: each array is one pair per entry.
LENGTHS = "rho.lengths"


def has_rho(target: Target) -> bool:
    return "rho" in target.group


def members(target: Target) -> Iterator[str]:
    """The group records its provenance, and carries its three arrays."""
    group = target.group["rho"]
    missing_attrs = [a for a in REQUIRED_ATTRS if a not in group.attrs]
    if missing_attrs:
        yield f"rho group is missing provenance attrs: {', '.join(missing_attrs)}"
        return
    for name in ("rho", "n_null", "variant_index"):
        if name not in group:
            yield f"rho group is missing data.zarr/rho/{name}"


def analyses(target: Target) -> Iterator[str]:
    """The matrix was estimated over the Analyses `analyses.tsv` holds."""
    recorded = int(target.group["rho"].attrs["n_analyses"])
    n_analyses = grid_counts(target)[1]
    if recorded != n_analyses:
        yield (
            f"rho group n_analyses attr {recorded} does not match "
            f"analyses.tsv's {n_analyses}"
        )


def _pairs(target: Target) -> int:
    n_analyses = grid_counts(target)[1]
    return n_analyses * (n_analyses - 1) // 2


def lengths(target: Target) -> Iterator[str]:
    """`rho` and `n_null` hold one entry per pair of Analyses."""
    group = target.group["rho"]
    expected = _pairs(target)
    for name in ("rho", "n_null"):
        held = array_length(group[name])
        if held != expected:
            yield f"{name} array has {held} entries but expected {expected}"


def values(target: Target) -> Iterator[str]:
    """A pair has a finite rho exactly when it has the null support, and rho is in [-1, 1]."""
    group = target.group["rho"]
    rho, n_null = packed_rho(group)
    min_nulls = int(group.attrs["min_nulls"])
    finite = np.isfinite(rho)
    if not np.array_equal(finite, n_null >= min_nulls):
        yield f"rho finiteness is inconsistent with n_null support (min_nulls={min_nulls})"
    held = rho[finite]
    if np.any(held < -1.0) or np.any(held > 1.0):
        yield "rho contains values outside [-1, 1]"


def variants(target: Target) -> Iterator[str]:
    """`variant_index` holds the variants the estimate says it used."""
    group = target.group["rho"]
    used = int(group.attrs["n_variants_used"])
    held = array_length(group["variant_index"])
    if held != used:
        yield (
            f"rho variant_index has {held} entries but "
            f"n_variants_used attr says {used}"
        )


def _observed_per_analysis(target: Target, block: Block) -> np.ndarray:
    """How many observed associations each Analysis has in the block: present, and not
    flagged imputed (spec §15 `states`). Nothing on a standalone Dense release, whose
    judgement reads only the group's attributes: its cells are not counted."""
    if not target.is_component:
        return np.zeros((), dtype=np.int64)
    observed = states(block.z_missing, np.isnan(block.se), block.imputed) == State.OBSERVED
    return block.per_analysis(observed)


def _judge_overflow_only(target: Target, observed: np.ndarray) -> Iterator[str]:
    """A Hybrid's Dense Component records exactly its Overflow-only Analyses -- those with
    no observed association in it, as this pass counts them -- and a standalone Dense
    release, which has none, records none."""
    group = target.group["rho"]
    if not target.is_component:
        if OVERFLOW_ONLY_ATTR in group.attrs:
            yield (
                f"rho group records `{OVERFLOW_ONLY_ATTR}`, which only a Hybrid release's "
                "Dense Component has"
            )
        return
    n_analyses = grid_counts(target)[1]
    try:
        recorded = recorded_overflow_only(group, n_analyses, "rho group")
    except ValueError as exc:
        yield str(exc)
        return
    counts = np.zeros(n_analyses, dtype=np.int64) + observed
    actual = {int(i) for i in np.flatnonzero(counts == 0)}
    if recorded != actual:
        yield (
            f"rho group `{OVERFLOW_ONLY_ATTR}` records {sorted(recorded)}, but the Analyses "
            f"with no observed association in the Dense Component are {sorted(actual)}"
        )


def _rule(
    name: str, run: Callable[[Target], Iterable[str]], after: tuple[str, ...]
) -> Rule:
    return Check(name=name, kinds=Kind.GRID, when=has_rho, after=after, run=run)


RULES: tuple[Rule, ...] = (
    _rule(MEMBERS, members, (REQUIRED,)),
    _rule(ANALYSES, analyses, (MEMBERS, "analyses.readable")),
    _rule(LENGTHS, lengths, (ANALYSES,)),
    _rule("rho.values", values, (LENGTHS,)),
    _rule("rho.variants", variants, (MEMBERS,)),
    Fold(
        name="rho.overflow_only",
        kinds=Kind.GRID,
        # After the grid's own value prerequisites only: a fold cannot wait on a check
        # registered after the scan's first fold, and this one reads no Rho array --
        # only the record, sized by the grid.
        after=VALUES_AFTER,
        when=has_rho,
        scan=ASSOCIATIONS,
        zero=np.zeros((), dtype=np.int64),
        measure=_observed_per_analysis,
        judge=_judge_overflow_only,
    ),
)
