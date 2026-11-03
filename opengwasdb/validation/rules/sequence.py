"""Rules about how an association sequence's cells are laid out, over its one read (ADR 0074).

Each Analysis's segment of `variant_index` is strictly increasing, so the
variant-side binary search answers from it and no Analysis holds two
associations at one variant (spec §11, §20). An Overflow's cells address the
shared variant table and none of them is a Dense Component row (spec §16). Each
rule folds over the association scan's windows, which already hold every cell's
Analysis and variant.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from opengwasdb.store.arrays import array_length
from opengwasdb.validation.rule import Fold, Kind, Rule, Target
from opengwasdb.validation.rules.csr import ARRAYS, LENGTHS, is_overflow
from opengwasdb.validation.rules.encoding import PLAN
from opengwasdb.validation.rules.hybrid import DENSE_TO_SHARED, dense_to_shared, shared_rows
from opengwasdb.validation.scans import ASSOCIATIONS, Block, analyses_of

#: Cells the standalone ordering check reads at once (4 MB of `int32`).
_ORDER_WINDOW = 1_000_000


@dataclass(frozen=True)
class SegmentBreak:
    """The first cell of an Analysis's segment that is not above the cell before it."""

    analysis: int
    offset: int
    previous: int
    held: int

    def message(self, label: str) -> str:
        """A repeat and a decrease are different defects, and are named differently.

        A repeat is two associations at one variant of one Analysis (a post-lift
        collision, or a duplicate source row), which no reader can resolve; a
        decrease is a segment out of genomic order, on which the variant-side
        binary search answers with a plausible, wrong row.
        """
        if self.held == self.previous:
            return (
                f"{label}/variant_index repeats variant {self.held} within Analysis "
                f"{self.analysis}'s segment: offsets {self.offset - 1} and {self.offset} "
                "both hold it, so one Analysis carries two associations at one variant"
            )
        return (
            f"{label}/variant_index is not non-decreasing within Analysis "
            f"{self.analysis}'s segment: offset {self.offset} holds {self.held} after "
            f"{self.previous}"
        )


@dataclass(frozen=True)
class FirstBreak:
    """The earliest break found so far; adding keeps the earlier one."""

    found: SegmentBreak | None = None

    def __add__(self, other: FirstBreak) -> FirstBreak:
        return self if self.found is not None else other


def first_break(
    analysis: np.ndarray,
    variant: np.ndarray,
    offset: int,
    previous: tuple[int, int] | None,
) -> SegmentBreak | None:
    """The first break among consecutive cells starting at flat `offset`.

    `previous` is the `(analysis, variant)` of the cell before them, so a break
    across two windows is found where it is. The comparison is `<=`, so a repeat
    is found at the same cell a decrease would be.
    """
    if len(variant) == 0:
        return None
    if previous is not None and previous[0] == int(analysis[0]) and int(variant[0]) <= previous[1]:
        return SegmentBreak(int(analysis[0]), offset, previous[1], int(variant[0]))
    breaks = np.flatnonzero((analysis[1:] == analysis[:-1]) & (variant[1:] <= variant[:-1]))
    if not len(breaks):
        return None
    i = int(breaks[0]) + 1
    return SegmentBreak(int(analysis[i]), offset + i, int(variant[i - 1]), int(variant[i]))


def segment_order_errors(
    group: Any, offsets: np.ndarray, n_assoc: int, errors: list[str], label: str
) -> int:
    """The ordering rule alone, over a group's `variant_index`; the entries it read.

    For an evidence runner that checks this one rule across Stores without the
    rest of a validation (#252). Read `_ORDER_WINDOW` cells at a time, so the
    sequence is never resident; it reads nothing when the length disagrees with
    `offsets`, which the length rule reports.
    """
    if array_length(group["variant_index"]) != n_assoc:
        return 0
    previous: tuple[int, int] | None = None
    read = 0
    for lo in range(0, n_assoc, _ORDER_WINDOW):
        hi = min(lo + _ORDER_WINDOW, n_assoc)
        variant = np.asarray(group["variant_index"][lo:hi], dtype=np.int32)
        analysis = analyses_of(offsets, lo, hi)
        read += len(variant)
        found = first_break(analysis, variant, lo, previous)
        if found is not None:
            errors.append(found.message(label))
            return read
        previous = (int(analysis[-1]), int(variant[-1]))
    return read


def _segment_order(_target: Target, block: Block) -> FirstBreak:
    assert block.analysis is not None and block.variant is not None
    return FirstBreak(first_break(block.analysis, block.variant, block.offset, block.previous))


def _judge_segment_order(_target: Target, total: FirstBreak) -> Iterator[str]:
    """`variant_index` strictly increases within every Analysis's segment.

    Checked on every release, not only the ones this build wrote: the variant-side
    binary search (`RaggedCSRReader.segment_positions`) is correct only on an
    ordered segment.
    """
    if total.found is not None:
        yield total.found.message("data.zarr/ragged")


def _outside_shared(target: Target, block: Block) -> int:
    assert block.variant is not None
    variant = block.variant
    return int(np.count_nonzero((variant < 0) | (variant >= shared_rows(target))))


def _judge_outside_shared(target: Target, n: int) -> Iterator[str]:
    """An Overflow's `variant_index` addresses rows of the shared variant table."""
    if n:
        yield (
            "Ragged Overflow variant_index falls outside the shared variant table "
            f"[0, {shared_rows(target)})"
        )


def _on_panel(target: Target) -> np.ndarray:
    """Which shared rows are Dense Component rows, held once for the run."""

    def build() -> np.ndarray:
        rows = np.zeros(shared_rows(target), dtype=bool)
        rows[dense_to_shared(target)] = True
        return rows

    return target.remember("on_panel_shared", build)


def _on_panel_cells(target: Target, block: Block) -> int:
    assert block.variant is not None
    on_panel = _on_panel(target)
    variant = block.variant
    inside = variant[(variant >= 0) & (variant < len(on_panel))]
    return int(np.count_nonzero(on_panel[inside]))


def _judge_partition(_target: Target, n: int) -> Iterator[str]:
    """No Overflow association is at a Dense Component (on-panel) variant (spec §16)."""
    if n:
        yield (
            "disjoint-partition violated: an overflow association references an "
            "on-panel (Dense Component) variant"
        )


_LAID_OUT = (ARRAYS, LENGTHS, *PLAN)

RULES: tuple[Rule, ...] = (
    Fold(
        name="csr.segment_order",
        kinds=Kind.CSR,
        after=_LAID_OUT,
        scan=ASSOCIATIONS,
        zero=FirstBreak(),
        measure=_segment_order,
        judge=_judge_segment_order,
    ),
    Fold(
        name="csr.shared_rows",
        kinds=Kind.CSR,
        when=is_overflow,
        after=_LAID_OUT,
        scan=ASSOCIATIONS,
        zero=0,
        measure=_outside_shared,
        judge=_judge_outside_shared,
    ),
    Fold(
        name="hybrid.partition",
        kinds=Kind.CSR,
        when=is_overflow,
        after=(*_LAID_OUT, DENSE_TO_SHARED),
        scan=ASSOCIATIONS,
        zero=0,
        measure=_on_panel_cells,
        judge=_judge_partition,
    ),
)
