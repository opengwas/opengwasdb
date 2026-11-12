"""Rules that hold each exact-value side table to the cells its plane marks (ADR 0074).

A plane that says "look in the table" for a cell the table does not describe has
lost that association outright, and a table entry for a cell that is not marked
is never read. For `z` the lost cell is the *largest* |z| in the store, for
`eaf` the frequency furthest from its baseline, and for a residual `se` an
exception the coding could not express.

`z` and `se` fold the marked positions out of the association scan and match them
against the table once it is done; both tables are small. The `eaf` table is not:
it is one Zarr v3 shard of up to 1,136,991,041 entries, so that rule reconciles
each block against the span of the table it must occupy as the scan reaches it,
holding one page rather than every marked position and the whole table beside
them (issue #387). The span is the one the block's own decode took from the
pass's single read of the table, so the rule reads none of it again (issue #314).
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
    StoreCodec,
    ZOverflowTable,
)
from opengwasdb.store.arrays import array_length
from opengwasdb.store.open import parse_format_version
from opengwasdb.validation.rule import (
    Check,
    Finding,
    Fold,
    Rule,
    Severity,
    Target,
)
from opengwasdb.validation.rules.associations import EAF_EXCEPTIONS, VALUE_KINDS, VALUES_AFTER
from opengwasdb.validation.rules.encoding import plane_label
from opengwasdb.validation.scans import ASSOCIATIONS, Block


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


@dataclass(frozen=True)
class Reconciled:
    """How a plane's marks and its side table disagreed, block by block, so far.

    Counts and the first position of each kind, not the positions themselves: a
    block's marks and the span it must occupy are compared as the scan reaches
    them, so nothing accumulates but this.
    """

    orphan: int = 0
    orphan_first: int | None = None
    stray: int = 0
    stray_first: int | None = None
    undecodable: bool = False
    #: Entries of the table the scan has taken, in order.
    returned: int = 0

    def __add__(self, other: Reconciled) -> Reconciled:
        return Reconciled(
            orphan=self.orphan + other.orphan,
            orphan_first=_first(self.orphan_first, other.orphan_first),
            stray=self.stray + other.stray,
            stray_first=_first(self.stray_first, other.stray_first),
            undecodable=self.undecodable or other.undecodable,
            returned=max(self.returned, other.returned),
        )


def _first(one: int | None, other: int | None) -> int | None:
    """The first of two flat positions in block order, either of which may be absent."""
    if one is None:
        return other
    if other is None:
        return one
    return min(one, other)


def _eaf_exceptions(target: Target, block: Block) -> Reconciled:
    """One block's marked cells, against the table span they must occupy.

    The comparison is made here rather than in the judge because here the block's
    cells and the table's span are both in hand: the judge would have to hold every
    marked position until the scan ended and then read the table whole.
    """
    if block.eaf_codes is None or not target.encoding.eaf.is_residual:
        return Reconciled(undecodable=block.eaf_undecodable)
    if block.eaf_span is None:
        # A residual plane with no table to take a span of: the plan rule says so,
        # and this rule meets the absent array as it always has.
        target.group[EafExceptionTable.index_name]
        target.group[EafExceptionTable.value_name]
        raise KeyError(EafExceptionTable.index_name)
    marked = block.positions(block.eaf_codes == EAF_EXCEPTION)
    span = block.eaf_span[0]
    orphan = np.setdiff1d(marked, span)
    stray = np.setdiff1d(span, marked)
    return Reconciled(
        orphan=len(orphan),
        orphan_first=int(orphan[0]) if len(orphan) else None,
        stray=len(stray),
        stray_first=int(stray[0]) if len(stray) else None,
        undecodable=block.eaf_undecodable,
        returned=block.eaf_taken,
    )


def _left_behind(target: Target, total: Reconciled) -> tuple[int, int | None]:
    """The table entries after the last block's span, and the first of them.

    A table that describes a cell past the last one the scan read leaves them
    here: the scan never reaches them, and the rule is that no entry is stray.
    """
    # The table's own length, not the scan's count of it: a component with no cell
    # to read takes nothing, and every entry it holds is then past the last span.
    left = array_length(target.group[EafExceptionTable.index_name]) - total.returned
    if left <= 0:
        return 0, None
    first = np.asarray(
        target.group[EafExceptionTable.index_name][total.returned : total.returned + 1],
        dtype=np.int64,
    )
    return left, int(first[0]) if len(first) else None


def _judge_eaf_exceptions(target: Target, total: Reconciled) -> Iterator[str]:
    """Every eaf exception cell has an exact table entry, and no entry is stray.

    An undecodable band is only reported once the positions have been reconciled:
    a table that disagrees with the plane is the *reason* the plane cannot be
    decoded, and saying "cannot decode" before naming that reason would hide it.
    The rules reading frequencies are skipped behind this one, so it reports every
    plane that cannot be decoded, with an exception table or without.
    """
    label = plane_label(target, "eaf")
    if not target.encoding.eaf.is_residual or "eaf" not in target.group:
        if total.undecodable:
            yield f"{label} cannot be decoded under the declared plan"
        return
    errors: list[str] = []
    left, left_first = _left_behind(target, total)
    errors.extend(
        _reconciled_errors(
            label,
            Reconciled(
                orphan=total.orphan,
                orphan_first=total.orphan_first,
                stray=total.stray + left,
                stray_first=_first(total.stray_first, left_first),
                undecodable=total.undecodable,
            ),
        )
    )
    if total.undecodable and not errors:
        errors.append(f"{label} cannot be decoded under the declared plan")
    yield from errors


def _reconciled_errors(label: str, reconciled: Reconciled) -> Iterator[str]:
    """The words a plane's disagreement with its table has always been reported in."""
    if reconciled.orphan:
        yield (
            f"{label}: {reconciled.orphan} eaf exception cell(s) have no entry in the "
            f"eaf exception table (first at flat position {reconciled.orphan_first})"
        )
    if reconciled.stray:
        yield (
            f"{label}: eaf exception table describes {reconciled.stray} cell(s) that are "
            f"not marked eaf exception (first at flat position {reconciled.stray_first})"
        )


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


#: The share of a residual plane's SE exception table above which the plane was
#: rounded before its plan was chosen (issue #355).
FLOAT16_EXACT_SHARE = 0.01
#: The fewest exceptions that share is judged over. Below it, one exact entry is
#: as likely to be a source reporting SEs at `float16`'s own precision as a
#: rounded plane, so the rule says nothing rather than guessing.
FLOAT16_EXACT_MINIMUM = 1_000
#: The first `format_version` whose builders hold an SE exception at the source's
#: `float32` (issue #355): this package's, which write 0.2.2 (spec §21.3). A release
#: written before it, package 0.9.0's 0.2.1 among them, may hold a rounded table, so
#: the rule reports that one and does not fail it: #367 owns re-validating the
#: published Stores.
SE_EXCEPTIONS_EXACT_SINCE = (0, 2, 2)


def _se_exceptions_judged(target: Target) -> bool:
    """A residual plan has a table, and only a table of `FLOAT16_EXACT_MINIMUM` is judged.

    Read from the array's shape, not its values: the minimum is fixed, so a
    Store either owes the whole judgment or owes none of it.
    """
    if not target.encoding.se.is_residual:
        return False
    name = SeExceptionTable.value_name
    if name not in target.group:
        return False
    return array_length(target.group[name]) >= FLOAT16_EXACT_MINIMUM


def _rounded_se_exceptions(target: Target) -> Iterator[str]:
    """How much of the SE exception table is exactly what a `float16` plane stores.

    Spec §6a has a residual plan hold an exception at the source's own `float32`.
    A build that rounded every SE to `float16` before the plan was chosen wrote
    the rounding instead, and one rounded cell cannot be told from a right one
    afterwards -- but a Store can: a source SE is a `float32` rounding of the
    decimal the source reported, so its low `23 - 10 = 13` mantissa bits are
    effectively uniform and a value inside `float16`'s range is `float16`-exact
    with probability `2**-13` (0.0122%). OGS-00003's table measures 0.0084% of
    its exceptions `float16`-exact when the builder held the source's value, and
    100% when it rounded (issue #355).

    So the rule fails a table whose share exceeds `FLOAT16_EXACT_SHARE`: 1% is 82
    times chance, and over `FLOAT16_EXACT_MINIMUM` exceptions the expected number
    of exact values is 0.12, which makes the ten a 1% share needs a 1-in-5.5e15
    run rather than a real table (Poisson tail, probability 1.8e-16). A source
    that itself stores `float16`-exact SEs would fail it; nothing in the corpus
    does.
    """
    values = np.asarray(target.group[SeExceptionTable.value_name][:], dtype=np.float32)
    exact = np.isfinite(values) & (
        StoreCodec(target.encoding).float16_se(values).astype(np.float32) == values
    )
    n_exact = int(np.count_nonzero(exact))
    share = n_exact / len(values)
    if share <= FLOAT16_EXACT_SHARE:
        return
    owed = (
        "rebuild the release"
        if parse_format_version(target.store.manifest.format_version) >= SE_EXCEPTIONS_EXACT_SINCE
        else "the release predates the fix, which #367 owns"
    )
    yield (
        f"{plane_label(target, 'se')}: {n_exact} of {len(values)} se exceptions "
        f"({share:.2%}) are exactly what a float16 plane stores, against "
        f"{2**-13:.2%} by chance: the plane was rounded to float16 before its encoding "
        f"plan was chosen (issue #355); {owed}"
    )


@dataclass(frozen=True, kw_only=True)
class _SeverityByFormat(Check):
    """A check whose severity says what the release's *format* owes, not what a reader saw.

    An error fails the release, so only one written by a builder that had to hold
    the value exactly can be failed: a release from before `exact_since` is
    reported as a warning, which says what it is without failing a Store nobody
    can fix in place (issue #355, #367). Every other rule's severity is its own.
    """

    exact_since: tuple[int, int, int]

    def findings(self, target: Target) -> Iterator[Finding]:
        version = parse_format_version(target.store.manifest.format_version)
        severity = self.severity if version >= self.exact_since else Severity.WARNING
        for message in self.run(target):
            yield self.finding(target, message, severity)


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
        name=EAF_EXCEPTIONS,
        kinds=VALUE_KINDS,
        after=VALUES_AFTER,
        scan=ASSOCIATIONS,
        zero=Reconciled(),
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
    _SeverityByFormat(
        name="side_tables.se_exceptions_rounded",
        kinds=VALUE_KINDS,
        after=("side_tables.se_exceptions",),
        when=_se_exceptions_judged,
        exact_since=SE_EXCEPTIONS_EXACT_SINCE,
        run=_rounded_se_exceptions,
    ),
)
