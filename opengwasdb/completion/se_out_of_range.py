"""What a Reference Completion left out for an SE above `float16`'s range (issue #338, ADR 0068).

A completed release's `n_missing_se_out_of_range` is what the completion itself
left out, plus its source's count where the source records one (0.2.3 or later;
before it the count is unknown, never 0), in two parts recorded apart in
`provenance.completion.se_out_of_range`:

* `observed` -- source associations the completion did not carry, because their
  SE is above 65,504. Only a release written before format 0.2.3 holds one: its
  writers stored such an SE (as `+inf` in a `float16` plane, or as an exact
  residual exception). A crossover a Hybrid completion does not fold is one.
* `imputed` -- fills whose SE is above 65,504, left missing.

The parts are what make the count checkable exactly: the Observed-View Check
samples cells, so it can find only some of the `observed` part, never the whole
of it, and none of the `imputed` one (ADR 0068, defect 2 of the 2 Nov review).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from opengwasdb.model.analyses import N_MISSING_SE_OUT_OF_RANGE, Analysis, with_se_out_of_range
from opengwasdb.store.open import records_se_out_of_range

#: The key of `provenance.completion` that records the parts.
PROVENANCE_KEY = "se_out_of_range"
_PARTS = ("observed", "imputed")


def carried_counts(
    analyses: Sequence[Analysis], source_version: str, *, source: str
) -> np.ndarray | None:
    """Per Analysis, the count the source records, which its completion carries
    forward; `None` for a source before 0.2.3, whose count is unknown.

    A source at format 0.2.3 or later records one for every Analysis, and a blank
    there is refused, never read as 0. A source before 0.2.3 records none, and no
    0 stands in for it: its completion counts instead the source associations
    above 65,504 it meets (`observed`), which is exact because a completion reads
    every observed cell of its source (`counted`).
    """
    if not records_se_out_of_range(source_version):
        return None
    out = np.zeros(len(analyses), dtype=np.int64)
    for i, analysis in enumerate(analyses):
        value = analysis.n_missing_se_out_of_range
        if not value:
            raise ValueError(
                f"{source}: analysis {analysis.analysis_id!r} records no "
                f"{N_MISSING_SE_OUT_OF_RANGE}, which a release at format_version "
                f"{source_version} records for every Analysis (spec §7a, issue #338)"
            )
        out[i] = int(value)
    return out


def counted(carried: np.ndarray | None, met: np.ndarray) -> np.ndarray:
    """The count a completed release records: what its completion `met` and left
    out, plus its source's count where that is known (`carried_counts`)."""
    return met if carried is None else carried + met


@dataclass(frozen=True)
class LeftOut:
    """Per Analysis, what one completion left out, in its two parts."""

    observed: np.ndarray
    imputed: np.ndarray

    @classmethod
    def none(cls, n_analyses: int) -> LeftOut:
        return cls(np.zeros(n_analyses, np.int64), np.zeros(n_analyses, np.int64))

    @classmethod
    def of(cls, observed: Any, imputed: Any) -> LeftOut:
        """The parts, each a non-negative count per Analysis, judged on its own: a
        negative part would cancel the other in the sum, an undercount that looks
        exact (spec §7a)."""
        out = cls(np.asarray(observed, dtype=np.int64), np.asarray(imputed, dtype=np.int64))
        if out.observed.shape != out.imputed.shape:
            raise ValueError(f"not a count per Analysis: {out.provenance()}")
        negative = [part for part in _PARTS if (getattr(out, part) < 0).any()]
        if negative:
            raise ValueError(
                f"{' and '.join(negative)} must each be a non-negative count per "
                f"Analysis: {out.provenance()}"
            )
        return out

    @classmethod
    def from_merge(cls, by_index: Mapping[int, int], n_analyses: int) -> np.ndarray:
        """The merge's per-Analysis-index counts of fills it left missing, as an array."""
        out = np.zeros(n_analyses, dtype=np.int64)
        for index, count in by_index.items():
            out[index] += count
        return out

    @property
    def total(self) -> np.ndarray:
        total: np.ndarray = self.observed + self.imputed
        return total

    def __add__(self, other: LeftOut) -> LeftOut:
        return LeftOut.of(self.observed + other.observed, self.imputed + other.imputed)

    def provenance(self) -> dict[str, list[int]]:
        """The parts as `provenance.completion.se_out_of_range` records them, one
        entry per Analysis in `analysis_index` order."""
        return {"observed": self.observed.tolist(), "imputed": self.imputed.tolist()}

    @classmethod
    def from_provenance(cls, completion: Mapping[str, Any], n_analyses: int) -> LeftOut:
        """The parts a completed release records; raises `ValueError` when they are
        absent or are not a count for each of its `n_analyses` Analyses."""
        block = completion.get(PROVENANCE_KEY)
        if not isinstance(block, Mapping) or set(block) != set(_PARTS):
            raise ValueError(
                f"provenance.completion.{PROVENANCE_KEY} is {block!r}, not the "
                f"{' and '.join(_PARTS)} counts a completion records (issue #338)"
            )
        parts = [block[part] for part in _PARTS]
        if not all(_one_count_each(part, n_analyses) for part in parts):
            raise ValueError(
                f"provenance.completion.{PROVENANCE_KEY} does not hold one integer per "
                f"Analysis for each of its {n_analyses} Analyses"
            )
        return cls.of(*parts)

    def recorded(
        self, analyses: Sequence[Analysis], carried: np.ndarray | None
    ) -> list[Analysis]:
        """`analyses` with the count a completed release records: both parts, plus
        `carried` where the source's count is known (`counted`)."""
        return with_se_out_of_range(list(analyses), counted(carried, self.total))


def _one_count_each(part: Any, n_analyses: int) -> bool:
    """A list of `n_analyses` integers (a bool is not one)."""
    return (
        isinstance(part, list)
        and len(part) == n_analyses
        and all(isinstance(v, int) and not isinstance(v, bool) for v in part)
    )
