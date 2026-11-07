"""Rules about the optional per-association sample-size plane (ADR 0067, ADR 0074).

Every rule here is one way the `n` plane could read as a plausible sample size
while being wrong: a length out of step with the sequence, so every later cell is
another association's N; a code that is neither a count nor the missing code; a
plane that holds no N at all, which reads exactly like a source without one; and
an Analysis whose `sample_size_scope` claims one N describes all its associations
while the plane holds a different one per association. A plane the plan does not
declare, or one it declares and the component lacks, is `encoding.n`'s.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

import numpy as np

from opengwasdb.encoding import N_MAX, N_MISSING
from opengwasdb.model.analyses import read_analyses
from opengwasdb.model.enums import SampleSizeScope
from opengwasdb.store.arrays import array_length
from opengwasdb.validation.rule import Check, Fold, Kind, Rule, Target
from opengwasdb.validation.rules.associations import VALUES_AFTER
from opengwasdb.validation.rules.csr import ARRAYS, is_ragged
from opengwasdb.validation.scans import ASSOCIATIONS, Block, sequence_offsets

#: The rule the plane's values are read after: one code per association.
LENGTH = "sample_size.length"


def holds_plane(target: Target) -> bool:
    """Whether a Ragged release declares and carries an `n` plane."""
    return is_ragged(target) and target.encoding.n is not None and "n" in target.group


def length(target: Target) -> Iterator[str]:
    """The plane holds one code per association the offsets bound."""
    n_assoc = int(sequence_offsets(target)[-1])
    held = array_length(target.group["n"])
    if held != n_assoc:
        yield f"data.zarr/ragged/n has {held} entries but offsets imply {n_assoc}"


@dataclass(frozen=True)
class Held:
    """Each Analysis's cells holding an N, and the cells holding nonsense."""

    per_analysis: np.ndarray = field(default_factory=lambda: np.zeros((), dtype=np.int64))
    n_invalid: int = 0
    first_invalid: int | None = None

    def __add__(self, other: Held) -> Held:
        first = self.first_invalid if self.first_invalid is not None else other.first_invalid
        return Held(self.per_analysis + other.per_analysis, self.n_invalid + other.n_invalid, first)


def _held(_target: Target, block: Block) -> Held:
    codes = block.n_codes
    if codes is None:
        return Held()
    invalid = (codes < 1) | ((codes > N_MAX) & (codes != N_MISSING))
    n_invalid = int(np.count_nonzero(invalid))
    first = block.offset + int(np.argmax(invalid)) if n_invalid else None
    return Held(block.per_analysis(~invalid & (codes != N_MISSING)), n_invalid, first)


def _judge_codes(target: Target, total: Held) -> Iterator[str]:
    """Every code is a sample size or the missing code, and some cell holds a size."""
    if total.n_invalid:
        yield (
            f"data.zarr/ragged/n holds {total.n_invalid} cell(s) that are neither a sample "
            f"size from 1 to {N_MAX} nor the missing code {N_MISSING} (first at flat "
            f"position {total.first_invalid})"
        )
    if int(sequence_offsets(target)[-1]) and not total.per_analysis.any():
        yield (
            "data.zarr/ragged/n holds no sample size at all; a plane of missing cells "
            "reads exactly like a release without one, and no build writes it (ADR 0067)"
        )


def _judge_scope(target: Target, total: Held) -> Iterator[str]:
    """An Analysis holding per-association N declares `variant_level` scope.

    The converse is not required: `variant_level` with no stored N says the N
    varies by variant and the release does not hold it, which is true of every
    Analysis whose pairs the table omitted.
    """
    held = np.zeros(len(sequence_offsets(target)) - 1, dtype=np.int64) + total.per_analysis
    for row in read_analyses(target.store.analyses_path).rows:
        index = int(row["analysis_index"])
        if index >= len(held) or held[index] == 0:
            continue
        scope = row.get("sample_size_scope", "")
        if scope != SampleSizeScope.VARIANT_LEVEL.value:
            yield (
                f"analysis {row.get('analysis_id', '')!r} holds a per-association N for "
                f"{int(held[index])} association(s) but declares sample_size_scope="
                f"{scope!r}; per-association sample sizes are "
                f"{SampleSizeScope.VARIANT_LEVEL.value} (ADR 0067)"
            )


#: The length rule, which the folds below read after.
CHECKS: tuple[Rule, ...] = (
    Check(name=LENGTH, kinds=Kind.CSR, when=holds_plane, after=(ARRAYS,), run=length),
)


def _value_rule(name: str, judge: Callable[[Target, Held], Iterable[str]]) -> Fold[Block, Held]:
    return Fold(
        name=name,
        kinds=Kind.CSR,
        when=holds_plane,
        after=(*VALUES_AFTER, LENGTH),
        scan=ASSOCIATIONS,
        zero=Held(),
        measure=_held,
        judge=judge,
    )


#: The value rules, which share the plane's one read.
RULES: tuple[Rule, ...] = (
    _value_rule("sample_size.codes", _judge_codes),
    _value_rule("sample_size.scope", _judge_scope),
)
