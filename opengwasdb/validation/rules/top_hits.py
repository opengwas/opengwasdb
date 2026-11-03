"""Rules for a component's Top-Hit Index, one set for every layout (ADR 0040, ADR 0074).

A tier lists every association whose |z| clears its threshold's cutoff, with the
decoded `z`, `se`, `eaf` and imputed flag a query reads. Its arrays, its structure
and its values are checked the same way on a Dense grid and an association
sequence, each in the words it has always been reported in. Its values are
checked against the component's own planes by folds over the association scan:
every tier cell is one the component holds, agrees with the decoded plane it was
harvested from, and the tier holds every cell that clears the cutoff. Those folds
share the scan's one read with every other value rule, so no Top-Hit check reads
a plane a second time, and none samples.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from opengwasdb import top_hits
from opengwasdb.top_hits import z_critical
from opengwasdb.validation.rule import Check, Fold, Kind, Rule, Target
from opengwasdb.validation.rules.associations import VALUES_AFTER
from opengwasdb.validation.rules.release import REQUIRED
from opengwasdb.validation.scans import ASSOCIATIONS, Block, CellIndex

#: The targets whose Top-Hit Index the folds check against the planes.
CHECKED_AGAINST_PLANES = Kind.GRID | Kind.CSR


def tier_root(target: Target) -> Any:
    """The group holding a component's `top_hits`: a grid's own, a sequence's release's."""
    if target.kind is Kind.CSR:
        return target.remember("data_root", lambda: target.store.arrays(mode="r"))
    return target.group


def has_index(target: Target) -> bool:
    return bool(top_hits.has_index(tier_root(target)))


def _tier_arrays(target: Target, key: str) -> dict[str, Any]:
    group = top_hits.index_group(tier_root(target))[key]
    return target.remember(f"top_hits/{key}", lambda: top_hits.load_tier_arrays(group))


def _keys(target: Target) -> list[str]:
    return list(top_hits.index_group(tier_root(target)))


def arrays(target: Target) -> Iterator[str]:
    """Every tier carries the arrays a tier is made of."""
    index = top_hits.index_group(tier_root(target))
    for key in _keys(target):
        missing = top_hits.missing_arrays(index[key])
        if missing:
            yield f"top-hit index {key} is missing {', '.join(missing)}"


def _n_analyses(target: Target) -> int:
    if target.kind is Kind.CSR:
        return int(target.group["offsets"].shape[0]) - 1
    return int(target.group["z"].shape[1])


def _expects_imputed(target: Target) -> bool:
    return "imputed" in target.group


def structure(target: Target) -> Iterator[str]:
    """Every complete tier has parallel arrays, sound offsets and ordered slices."""
    index = top_hits.index_group(tier_root(target))
    for key in _keys(target):
        if top_hits.missing_arrays(index[key]):
            continue
        problem = top_hits.structural_problem(
            _tier_arrays(target, key), index[key], _n_analyses(target), _expects_imputed(target)
        )
        if problem is not None:
            yield f"top-hit index {key} {problem}"


@dataclass(frozen=True)
class Tier:
    """One structurally sound tier, ready to be held against the planes."""

    key: str
    z_crit: float
    cells: CellIndex
    z: np.ndarray
    se: np.ndarray
    imputed: np.ndarray | None
    eaf: np.ndarray | None


def sound_tiers(target: Target) -> list[Tier]:
    """The tiers the value rules check: those with every array and a sound structure.

    A tier that fails either is reported by `top_hits.arrays` or `top_hits.structure`
    and its cells are not compared, since a comparison against arrays that do not
    line up would only repeat that finding as a value one.
    """
    return target.remember("top_hits/sound", lambda: list(_sound(target)))


def _sound(target: Target) -> Iterator[Tier]:
    index = top_hits.index_group(tier_root(target))
    for key in _keys(target):
        group = index[key]
        if top_hits.missing_arrays(group):
            continue
        loaded = _tier_arrays(target, key)
        if top_hits.structural_problem(
            loaded, group, _n_analyses(target), _expects_imputed(target)
        ):
            continue
        yield Tier(
            key=key,
            z_crit=z_critical(top_hits.tier_threshold(key, group)),
            cells=CellIndex(loaded["rows"], loaded["cols"]),
            z=loaded["z_values"],
            se=loaded["se_values"],
            imputed=loaded["imputed_values"],
            eaf=loaded["eaf_values"],
        )


def _per_tier(
    target: Target, block: Block, disagree: Callable[[Tier, np.ndarray, np.ndarray], int]
) -> Counter[str]:
    """Each tier's cells in this block that `disagree` counts as wrong."""
    counts: Counter[str] = Counter()
    for tier in sound_tiers(target):
        found, flat = block.locate(tier.cells)
        if len(flat):
            counts[tier.key] += disagree(tier, found, flat)
    return counts


def _z_disagrees(block: Block) -> Callable[[Tier, np.ndarray, np.ndarray], int]:
    def count(tier: Tier, found: np.ndarray, flat: np.ndarray) -> int:
        stored = block.z.ravel()[flat]
        wrong = ~np.isfinite(stored)
        wrong |= ~np.isclose(stored, tier.z[found], rtol=1e-3, atol=1e-3)
        return int(np.count_nonzero(wrong))

    return count


def _below_cutoff(block: Block) -> Callable[[Tier, np.ndarray, np.ndarray], int]:
    def count(tier: Tier, _found: np.ndarray, flat: np.ndarray) -> int:
        return int(np.count_nonzero(np.abs(block.z.ravel()[flat]) < tier.z_crit))

    return count


def _held(_block: Block) -> Callable[[Tier, np.ndarray, np.ndarray], int]:
    def count(_tier: Tier, _found: np.ndarray, flat: np.ndarray) -> int:
        return len(flat)

    return count


def _se_disagrees(block: Block) -> Callable[[Tier, np.ndarray, np.ndarray], int]:
    # rtol=1e-6, atol=0: the index is *written* from the decoded plane, so the two
    # are bit-identical in a sound store and anything looser hides the defect this
    # checks for. A stale index left over from a format migration differs by the
    # coding's half step -- 0.197% at range 0.5 -- which against a typical SE of
    # 0.05 is 1e-4, and an atol of 1e-3 waves it straight through. Measured on the
    # FinnGen R13 pilot, where exactly that happened.
    def count(tier: Tier, found: np.ndarray, flat: np.ndarray) -> int:
        stored = block.se.ravel()[flat]
        same = np.isclose(stored, tier.se[found], rtol=1e-6, atol=0.0, equal_nan=True)
        return int(np.count_nonzero(~same))

    return count


def _imputed_disagrees(block: Block) -> Callable[[Tier, np.ndarray, np.ndarray], int]:
    def count(tier: Tier, found: np.ndarray, flat: np.ndarray) -> int:
        if tier.imputed is None or block.imputed is None:
            return 0
        stored = block.imputed.ravel()[flat].astype(np.uint8)
        return int(np.count_nonzero(stored != tier.imputed[found]))

    return count


def _eaf_disagrees(block: Block) -> Callable[[Tier, np.ndarray, np.ndarray], int]:
    # The index's `eaf` is written from the decoded plane, panel frequency
    # included on imputed cells, so it is held to numpy's default tolerance.
    def count(tier: Tier, found: np.ndarray, flat: np.ndarray) -> int:
        if tier.eaf is None:
            return 0
        if block.eaf is None:
            stored = np.full(len(flat), np.nan, dtype=np.float32)
        else:
            stored = block.eaf.ravel()[flat]
        return int(np.count_nonzero(~np.isclose(stored, tier.eaf[found], equal_nan=True)))

    return count


def _measure_z(target: Target, block: Block) -> Counter[str]:
    """A grid's tier cell is wrong whose Z disagrees or falls below the cutoff; a
    sequence's says which, so the two are counted apart there."""
    wrong = _per_tier(target, block, _z_disagrees(block))
    below = _per_tier(target, block, _below_cutoff(block))
    if target.kind is Kind.GRID:
        return wrong + below
    return wrong + Counter({f"{key}{_BELOW}": n for key, n in below.items()})


def _measure_held(target: Target, block: Block) -> Counter[str]:
    return _per_tier(target, block, _held(block))


def _measure_se(target: Target, block: Block) -> Counter[str]:
    return _per_tier(target, block, _se_disagrees(block))


def _measure_imputed(target: Target, block: Block) -> Counter[str]:
    return _per_tier(target, block, _imputed_disagrees(block))


def _measure_eaf(target: Target, block: Block) -> Counter[str]:
    return _per_tier(target, block, _eaf_disagrees(block))


def _measure_passing(target: Target, block: Block) -> Counter[str]:
    """Each tier's cutoff cleared by how many of this block's cells."""
    absolute = np.abs(block.z)
    counts: Counter[str] = Counter()
    for tier in sound_tiers(target):
        counts[tier.key] += int(np.count_nonzero(absolute >= tier.z_crit))
    return counts


#: Where a sequence's Z count keeps the cells below the cutoff, apart from the rest.
_BELOW = "/below"


def _judged(grid: str, sequence: str) -> Callable[[Target, Counter[str]], Iterator[str]]:
    """A judge naming each tier with a wrong cell, in the words each layout has used."""

    def judge(target: Target, wrong: Counter[str]) -> Iterator[str]:
        wording = grid if target.kind is Kind.GRID else sequence
        for tier in sound_tiers(target):
            if wrong[tier.key]:
                yield f"top-hit index {tier.key} {wording}"

    return judge


def _judge_z(target: Target, wrong: Counter[str]) -> Iterator[str]:
    yield from _judged(
        "contains z value inconsistent with z array", "z value inconsistent with CSR"
    )(target, wrong)
    if target.kind is Kind.CSR:
        for tier in sound_tiers(target):
            if wrong[f"{tier.key}{_BELOW}"]:
                yield f"top-hit index {tier.key} contains association above threshold"


def _judge_held(target: Target, held: Counter[str]) -> Iterator[str]:
    """Every tier cell is an association the sequence holds.

    A grid holds every cell its axes span, which the tier's structure already
    checks; a sequence holds only the cells it lists.
    """
    for tier in sound_tiers(target):
        if held[tier.key] != len(tier.cells.rows):
            yield f"top-hit index {tier.key} references missing association"


def abs_z(target: Target) -> Iterator[str]:
    """Every tier's `abs_z` is its `z`'s magnitude: a query filters on it unread."""
    for tier in sound_tiers(target):
        stored = _tier_arrays(target, tier.key)["abs_z"]
        if not np.allclose(stored, np.abs(tier.z), rtol=1e-3, atol=1e-3):
            yield f"top-hit index {tier.key} abs_z inconsistent with z"


def _judge_complete(target: Target, passing: Counter[str]) -> Iterator[str]:
    """A tier holds every cell whose |z| clears its cutoff, and no other."""
    for tier in sound_tiers(target):
        if len(tier.cells.rows) != passing[tier.key]:
            yield f"top-hit index {tier.key} does not match stored z values"


def _against_planes(
    name: str,
    measure: Callable[[Target, Block], Counter[str]],
    judge: Callable[[Target, Counter[str]], Iterator[str]],
    kinds: Kind = CHECKED_AGAINST_PLANES,
) -> Fold[Block, Counter[str]]:
    return Fold(
        name=name,
        kinds=kinds,
        when=has_index,
        after=VALUES_AFTER,
        scan=ASSOCIATIONS,
        zero=Counter(),
        measure=measure,
        judge=judge,
    )


#: The rules about a tier's own arrays.
STRUCTURE_RULES: tuple[Rule, ...] = (
    Check(
        name="top_hits.arrays",
        kinds=Kind.GRID | Kind.CSR,
        when=has_index,
        after=(REQUIRED,),
        run=arrays,
    ),
    Check(
        name="top_hits.structure",
        kinds=Kind.GRID | Kind.CSR,
        when=has_index,
        after=(REQUIRED,),
        run=structure,
    ),
    Check(name="top_hits.abs_z", kinds=Kind.CSR, when=has_index, after=(REQUIRED,), run=abs_z),
)

#: The rules holding a tier against the planes, in the order they always reported.
VALUE_RULES: tuple[Rule, ...] = (
    _against_planes("top_hits.held", _measure_held, _judge_held, kinds=Kind.CSR),
    _against_planes("top_hits.z", _measure_z, _judge_z),
    _against_planes(
        "top_hits.se",
        _measure_se,
        _judged(
            "contains se value inconsistent with decoded se plane",
            "se value inconsistent with decoded CSR se",
        ),
    ),
    _against_planes(
        "top_hits.imputed",
        _measure_imputed,
        _judged(
            "contains imputed value inconsistent with imputed array",
            "imputed value inconsistent with CSR",
        ),
    ),
    _against_planes(
        "top_hits.eaf",
        _measure_eaf,
        _judged(
            "contains eaf value inconsistent with eaf plane", "eaf value inconsistent with CSR"
        ),
    ),
    _against_planes("top_hits.complete", _measure_passing, _judge_complete),
)
