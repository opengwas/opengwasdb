"""Rules that hold each exact-value side table to the cells its plane marks (ADR 0074).

A plane that says "look in the table" for a cell the table does not describe has
lost that association outright, and a table entry for a cell that is not marked
is never read. For `z` the lost cell is the *largest* |z| in the store, for
`eaf` the frequency furthest from its baseline, and for a residual `se` an
exception the coding could not express. Each rule folds the marked cells'
positions out of the association scan and matches them against the table once
the scan is done.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from opengwasdb.encoding import (
    EAF_EXCEPTION,
    SE_EXCEPTION,
    Z_OVERFLOW,
    EafExceptionTable,
    SeExceptionTable,
    ZOverflowTable,
)
from opengwasdb.validation.rule import Fold, Rule, Target
from opengwasdb.validation.rules.associations import VALUE_KINDS, VALUES_AFTER
from opengwasdb.validation.rules.encoding import plane_label
from opengwasdb.validation.scans import ASSOCIATIONS, Block, judges_frequencies


@dataclass(frozen=True)
class Marked:
    """The flat positions of the cells a plane marks, block by block, in block order."""

    chunks: tuple[np.ndarray, ...] = ()
    undecodable: bool = False

    def __add__(self, other: Marked) -> Marked:
        return Marked(self.chunks + other.chunks, self.undecodable or other.undecodable)

    @property
    def positions(self) -> np.ndarray:
        if not self.chunks:
            return np.empty(0, dtype=np.int64)
        joined: np.ndarray = np.concatenate(self.chunks)
        return joined


def positions_match(
    label: str,
    observed: np.ndarray,
    table: ZOverflowTable | EafExceptionTable | SeExceptionTable,
    errors: list[str],
    *,
    what: str = "out-of-range z",
    table_name: str = "overflow",
) -> None:
    """Every side-table cell has an exact value, and nothing else does.

    A plane that says "look in the table" for a cell the table does not
    describe has lost that association outright. For `z` it would be the
    *largest* |z| in the store -- exactly the value nothing may be quietly
    wrong about -- and for `eaf` it is the frequency furthest from its
    baseline, which is the rare variant a user is filtering on.
    """
    if np.array_equal(np.sort(observed), table.index):
        return
    orphan = np.setdiff1d(observed, table.index)
    stray = np.setdiff1d(table.index, observed)
    if len(orphan):
        errors.append(
            f"{label}: {len(orphan)} {what} cell(s) have no entry in the {table_name} "
            f"table (first at flat position {int(orphan[0])})"
        )
    if len(stray):
        errors.append(
            f"{label}: {table_name} table describes {len(stray)} cell(s) that are not marked "
            f"{what} (first at flat position {int(stray[0])})"
        )


def _z_overflow(target: Target, block: Block) -> Marked:
    if not target.encoding.z.is_fixed_point:
        return Marked()
    return Marked((block.positions(block.z_codes == Z_OVERFLOW),))


def _judge_z_overflow(target: Target, marked: Marked) -> Iterator[str]:
    """The z overflow table describes exactly the `Z_OVERFLOW` cells (issue #114)."""
    if not target.encoding.z.is_fixed_point:
        return
    errors: list[str] = []
    label = plane_label(target, "z")
    positions_match(label, marked.positions, ZOverflowTable.read(target.group), errors)
    yield from errors


def _eaf_exceptions(target: Target, block: Block) -> Marked:
    if block.eaf_codes is None or not target.encoding.eaf.is_residual:
        return Marked(undecodable=block.eaf_undecodable)
    found = block.positions(block.eaf_codes == EAF_EXCEPTION)
    return Marked((found,), block.eaf_undecodable)


def _judge_eaf_exceptions(target: Target, marked: Marked) -> Iterator[str]:
    """Every eaf exception cell has an exact table entry, and no entry is stray.

    An undecodable band is only reported once the positions have been reconciled:
    a table that disagrees with the plane is the *reason* the plane cannot be
    decoded, and saying "cannot decode" before naming that reason would hide it.
    """
    if not target.encoding.eaf.is_residual or "eaf" not in target.group:
        return
    errors: list[str] = []
    label = plane_label(target, "eaf")
    positions_match(
        label,
        marked.positions,
        EafExceptionTable.read(target.group),
        errors,
        what="eaf exception",
        table_name="eaf exception",
    )
    if marked.undecodable and not errors:
        errors.append(f"{label} cannot be decoded under the declared plan")
    yield from errors


def _se_exceptions(target: Target, block: Block) -> Marked:
    if not target.encoding.se.is_residual:
        return Marked()
    return Marked((block.positions(block.se_codes == SE_EXCEPTION),))


def _judge_se_exceptions(target: Target, marked: Marked) -> Iterator[str]:
    """Every `-127` in an `se` plane has a table entry, and no entry is stray."""
    if not target.encoding.se.is_residual:
        return
    errors: list[str] = []
    positions_match(
        plane_label(target, "se"),
        marked.positions,
        SeExceptionTable.read(target.group),
        errors,
        what="se exception",
        table_name="se exception",
    )
    yield from errors


RULES: tuple[Rule, ...] = (
    Fold(
        name="side_tables.z_overflow",
        kinds=VALUE_KINDS,
        after=VALUES_AFTER,
        scan=ASSOCIATIONS,
        zero=Marked(),
        measure=_z_overflow,
        judge=_judge_z_overflow,
    ),
    Fold(
        name="side_tables.eaf_exceptions",
        kinds=VALUE_KINDS,
        when=judges_frequencies,
        after=VALUES_AFTER,
        scan=ASSOCIATIONS,
        zero=Marked(),
        measure=_eaf_exceptions,
        judge=_judge_eaf_exceptions,
    ),
    Fold(
        name="side_tables.se_exceptions",
        kinds=VALUE_KINDS,
        after=VALUES_AFTER,
        scan=ASSOCIATIONS,
        zero=Marked(),
        measure=_se_exceptions,
        judge=_judge_se_exceptions,
    ),
)
