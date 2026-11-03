"""Rules that judge the associations a component stores, over one streamed read (ADR 0074).

Each value rule is a fold over `ASSOCIATIONS`: it measures every block the scan
reads and is judged on the total, so the planes are read once however many rules
there are. A Dense grid and an association sequence are judged by the same rules,
each in the words it has always been reported in. The presence rules count
through `Tally`, the one presence rule every reader shares (spec §15, ADR 0070).
`grid.planes` and `grid.eaf_side_lengths`, and a sequence's `csr.lengths`, come
first: no value means anything on a plane that does not span the axes it is read
against.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import TypeVar

import numpy as np

from opengwasdb.associations import State, Tally
from opengwasdb.encoding import EAF_BASELINE, EAF_REFERENCE
from opengwasdb.model.analyses import read_analyses
from opengwasdb.store.arrays import array_length
from opengwasdb.validation.rule import Check, Fold, Kind, Rule, Severity, Target
from opengwasdb.validation.rules.csr import ARRAYS, LENGTHS, OBSERVED_ONLY, prefix
from opengwasdb.validation.rules.encoding import PLAN, plane_label
from opengwasdb.validation.rules.release import REQUIRED
from opengwasdb.validation.scans import ASSOCIATIONS, Block

#: The rule every value rule runs after for a grid's planes to span its axes.
GRID_PLANES = "grid.planes"
#: The rule a read that fails part-way is reported by.
DECODABLE = "associations.decodable"
#: The targets the value rules judge.
VALUE_KINDS = Kind.GRID | Kind.CSR
#: What every value rule runs after: the plan its planes decode under, planes that
#: span the axes, on a completed grid a mask the same shape, and on a sequence
#: parallel arrays one entry per association -- none of them imputed in an Overflow.
VALUES_AFTER = (
    *PLAN,
    GRID_PLANES,
    "grid.eaf_side_lengths",
    "completion.mask",
    ARRAYS,
    LENGTHS,
    OBSERVED_ONLY,
)

#: What a value rule measures in a block and adds up.
M = TypeVar("M")

#: Analyses one infinite-se warning names. The count says how much of the release is
#: affected; five ids say where to start reading it.
INFINITE_SE_ANALYSES_NAMED = 5


def grid_counts(target: Target) -> tuple[int, int]:
    """The grid's expected shape: the variant axis's rows by `analyses.tsv`'s."""
    n_variants = target.remember("n_variants", lambda: _n_variants(target))
    n_analyses = target.remember(
        "n_analyses", lambda: len(read_analyses(target.store.analyses_path).rows)
    )
    return n_variants, n_analyses


def _n_variants(target: Target) -> int:
    release = target.parent if target.parent is not None else target
    return release.variant_axis.n_variants


def planes(target: Target) -> Iterator[str]:
    """`z` and `se` are present, and `z`, `se` and `eaf` span `(n_variants, n_analyses)`.

    `eaf` is optional (ADR 0036) -- absent on any build whose sources report no
    frequency -- but when present it is a third parallel plane. Both missing
    planes are named, which saves a repair-and-revalidate round trip.
    """
    group = target.group
    missing = [name for name in ("z", "se") if name not in group]
    if missing:
        yield from (f"missing data.zarr/{name}" for name in missing)
        return
    expected = grid_counts(target)
    for name in ("z", "se", "eaf"):
        if name in group and tuple(group[name].shape) != expected:
            yield f"{name} shape {tuple(group[name].shape)} does not match {expected}"


def eaf_side_lengths(target: Target) -> Iterator[str]:
    """The per-variant eaf side arrays are the length of the variant axis.

    Checked whether or not there is an `eaf` plane: a completed release whose
    Analyses reported no frequency carries `eaf_reference` and nothing else, and a
    mis-sized one there would hand every imputed cell the frequency of some other
    variant (issue #113).
    """
    group = target.group
    n_variants = grid_counts(target)[0]
    for name in (EAF_BASELINE, EAF_REFERENCE):
        if name in group and array_length(group[name]) != n_variants:
            yield (
                f"{name} has {array_length(group[name])} entries but the variant axis "
                f"has {n_variants}"
            )


def _decoded(_target: Target, block: Block) -> int:
    """The cells whose SE decodes: decoded here first, so a failure is this rule's.

    Every later rule reading SE shares the decode; one that cannot be made is
    reported here, once, and those rules are skipped.
    """
    return int(block.se.size)


def _judge_decoded(_target: Target, _decoded_cells: int) -> Iterator[str]:
    yield from ()


def _infinite_se(target: Target, block: Block) -> np.ndarray:
    # Only a floating-point plane can hold an infinity; a residual `int8` plane's
    # decoded values are finite by construction, so there is nothing to count.
    if target.encoding.se.is_residual:
        return np.zeros(block.n_analyses, dtype=np.int64)
    return block.per_analysis(np.isinf(block.se))


def infinite_se_message(
    label: str, per_analysis: np.ndarray | int, analyses_path: Path | None
) -> str | None:
    """The warning for one plane's infinite `se` cells, naming the count and first Analyses.

    An SE above a `float16` plane's range is narrowed to `+inf` beside a finite Z
    (issue #338). Spec §15 makes that cell present, so the release is valid -- but
    `beta = z * se` reads as infinity, and nothing else in a Store says so (#337).
    """
    counts = np.asarray(per_analysis)
    total = int(counts.sum())
    if not total:
        return None
    message = (
        f"{total} cell(s) of {label} hold an infinite se: a source standard error above "
        "the plane's range is stored as +inf, so beta = z * se reads as infinity although "
        "the source value was finite (issue #338)"
    )
    named = _first_holding(counts, analyses_path)
    if named:
        message += f". The first Analyses holding one: {', '.join(map(repr, named))}"
    return message


def _first_holding(counts: np.ndarray, analyses_path: Path | None) -> list[str]:
    """The `analysis_id` of up to `INFINITE_SE_ANALYSES_NAMED` Analyses holding one."""
    holding = np.flatnonzero(counts)[:INFINITE_SE_ANALYSES_NAMED].tolist()
    if not holding or analyses_path is None:
        return []
    ids: dict[int, str] = {}
    for row in read_analyses(analyses_path).rows:
        try:
            ids[int(row.get("analysis_index", ""))] = str(row.get("analysis_id", ""))
        except ValueError:
            continue  # a malformed row is reported by the analyses.analysis_index rule
    return [ids[i] for i in holding if i in ids]


def _infinite_se_label(target: Target) -> str:
    # A Hybrid release's Dense Component is a Dense release at `<store>/dense` (spec
    # §16), so a warning naming plain `data.zarr` would send its reader to the wrong one.
    if target.kind is Kind.CSR:
        return "data.zarr/ragged/se"
    return "dense/data.zarr/se" if target.is_component else "data.zarr/se"


def _judge_infinite_se(target: Target, per_analysis: np.ndarray) -> Iterator[str]:
    label = _infinite_se_label(target)
    message = infinite_se_message(label, per_analysis, target.store.analyses_path)
    if message is not None:
        yield message


def _negative_se(_target: Target, block: Block) -> int:
    return int(np.count_nonzero(np.isfinite(block.se) & (block.se < 0)))


def _judge_negative_se(target: Target, n: int) -> Iterator[str]:
    if n:
        yield f"{prefix(target)}se contains negative finite values"


def tally(_target: Target, block: Block) -> Tally:
    """A block's cells under the presence rule, counted once and shared by its rules."""
    return block.tally


def _judge_paired(target: Target, total: Tally) -> Iterator[str]:
    """Z and SE are missing in exactly the same cells (spec §15).

    A sequence has always named which of the two is missing alone; a grid, that
    they disagree. An infinite SE is present, so a missing Z beside one fails.
    """
    if target.kind is Kind.GRID:
        if total.count(State.PARTIAL):
            yield "z and se missingness is inconsistent"
        return
    if total.z_only_missing:
        yield f"{prefix(target)}missing z rows have a present se value (inconsistent missingness)"
    if total.se_only_missing:
        yield f"{prefix(target)}z and se missingness is inconsistent"


def _judge_canonical(target: Target, total: Tally) -> Iterator[str]:
    """Every missing marker is the canonical one (spec §15, ADR 0070)."""
    if total.count(State.NON_CANONICAL):
        yield (
            f"{prefix(target)}z or se holds a non-canonical missing marker "
            "(a NaN other than np.nan)"
        )


def _eaf_out_of_range(_target: Target, block: Block) -> int:
    if block.eaf is None:
        return 0
    eaf = block.eaf
    return int(np.count_nonzero(np.isfinite(eaf) & ((eaf < 0.0) | (eaf > 1.0))))


def _judge_eaf_range(target: Target, n: int) -> Iterator[str]:
    """Decoded frequencies are frequencies: none may fall outside [0, 1].

    Decoded, not raw: an `int8` residual plane's bytes are codes, and checking those
    for "in [0, 1]" would pass every store while saying nothing about the
    frequencies a query returns.
    """
    if n:
        yield f"{plane_label(target, 'eaf')} contains finite values outside [0, 1]"


CHECKS: tuple[Rule, ...] = (
    Check(name=GRID_PLANES, kinds=Kind.GRID, after=(REQUIRED,), run=planes),
    Check(name="grid.eaf_side_lengths", kinds=Kind.GRID, after=(REQUIRED,), run=eaf_side_lengths),
)


def value_fold(
    name: str,
    zero: M,
    measure: Callable[[Target, Block], M],
    judge: Callable[[Target, M], Iterable[str]],
    *,
    severity: Severity = Severity.ERROR,
    when: Callable[[Target], bool] | None = None,
    kinds: Kind = VALUE_KINDS,
) -> Fold[Block, M]:
    """A value rule over `ASSOCIATIONS`, after `VALUES_AFTER`, on every value target."""
    return Fold(
        name=name,
        kinds=kinds,
        after=VALUES_AFTER,
        when=when if when is not None else _always,
        severity=severity,
        scan=ASSOCIATIONS,
        zero=zero,
        measure=measure,
        judge=judge,
    )


def _always(_target: Target) -> bool:
    return True


#: The read every value rule shares, and the presence rules.
PRESENCE_RULES: tuple[Rule, ...] = (
    value_fold(DECODABLE, 0, _decoded, _judge_decoded),
    value_fold(
        "associations.se_infinite",
        np.zeros((), dtype=np.int64),
        _infinite_se,
        _judge_infinite_se,
        severity=Severity.WARNING,
    ),
    value_fold("associations.se_negative", 0, _negative_se, _judge_negative_se),
    value_fold("associations.markers_paired", Tally(), tally, _judge_paired),
    value_fold("associations.markers_canonical", Tally(), tally, _judge_canonical),
)

#: The frequency rule, reported after the completion rules as it always has been.
EAF_RULES: tuple[Rule, ...] = (
    value_fold("associations.eaf_range", 0, _eaf_out_of_range, _judge_eaf_range),
)
