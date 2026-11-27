"""The stored-z audit: an Analysis whose stored z is implausible as a genome-wide scan (#510).

ADR 0069's check compares each stored z with its source's own p-value, and since
#510 every build runs it. A published Store keeps no source p-value, so a Store
built before then -- or from a bundle whose Resolution Records predate the check,
as OGS-00032's and OGS-00033's did -- cannot be checked that way after the fact.
What it does keep is every z. In a genome-wide scan most variants are not at 5e-8.
An effect on the wrong scale puts nearly every row there:

- an odds ratio in `beta` beside a log-scale SE gives |z| of about 1/SE (the ten
  Omae Analyses of PMID 40222985: 81% to 87% of rows at 5e-8);
- a mis-scaled `beta` gives |z| in the millions (GCST90278111: every row);
- GCST002047, ADR 0069's own example: every row.

So the rule counts, per Analysis, the observed associations and those that clear
5e-8, and flags an Analysis with more than 30% at 5e-8. A Hybrid's Analysis is
judged on its counts summed over the Dense Component and the Overflow. The
measurement is `/data/opengwasdb/work/orchestrate/opengwasdb-367/scratch/510/signals_table.md`
(each source's first 500,000 rows: the twelve wrong Analyses at 81% or more, 206
sound ones at 8.2% or less) and, over every published Store,
`scratch/510/survey.jsonl`: the largest share of any sound Analysis is 9.7%
(GCST90245992, height, OGS-00011), and only GCST002047 is flagged.

A share of rows above |z| 3 (a median |z| above 3) was measured and dropped (#510
ruling): it flags genome-wide sources published only below a p threshold, whose
effects are sound -- GCST90444395 (every row at p <= 0.007, 55% above |z| 3) and
GCST90271757 (p <= 0.00067, every row above |z| 3) -- while their share at 5e-8 is
0.7% and 2.2%.

It is a fold over the association scan, so it costs no extra read: each block adds
two counts per Analysis, and nothing is kept per association. Imputed cells are
not counted: a Reference Completion's values are the panel's, not the source's.

Only a genome-wide release is judged: one whose manifest declares
`association_coverage` `gwas`. A cis release holds the window around each gene, and
a signals-only one holds what its source chose to keep, so in neither are most rows
null: a strong eGene legitimately has most of its window at 5e-8 (OGS-00027 would
flag 560 genes, OGS-00001 two).

The severity follows the release's format, as `SeverityByFormat`'s does: an error on
a release this package's builders write, a warning on an earlier one, which nothing
rewrites (#367).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from opengwasdb.store_reader.associations.record import State, states
from opengwasdb.store_reader.enums import AssociationCoverage, PrimaryStorageLayout
from opengwasdb.store_reader.stats import z_critical
from opengwasdb.validation.rule import Finding, Fold, Kind, Rule, Target
from opengwasdb.validation.rules.associations import VALUES_AFTER
from opengwasdb.validation.rules.release import severity_owed
from opengwasdb.validation.scans import ASSOCIATIONS, Block

__all__ = [
    "MAX_SHARE_GENOME_WIDE",
    "MIN_OBSERVED",
    "RULE",
    "RULES",
    "ZPlausibility",
    "block_counts",
    "implausible",
]

RULE = "associations.z_plausible"

#: The largest share of an Analysis's observed associations that may clear 5e-8.
MAX_SHARE_GENOME_WIDE = 0.3
#: Fewest observed associations in a component an Analysis is judged from. Below
#: it a share is a handful of rows, and one locus could decide it.
MIN_OBSERVED = 1_000

_Z_GENOME_WIDE = z_critical(5e-8)
#: Rows of the counts a block adds: observed, and at 5e-8.
_OBSERVED, _GENOME_WIDE = range(2)
#: Where a Hybrid's first-judged component leaves its counts for the second.
_WAITING = "z_plausible/first_component"


@dataclass(frozen=True)
class ZPlausibility:
    """One Analysis's counts in one component, and the share at 5e-8 they make."""

    analysis_index: int
    observed: int
    genome_wide: int

    @property
    def share_genome_wide(self) -> float:
        return self.genome_wide / self.observed


def implausible(counts: np.ndarray) -> list[ZPlausibility]:
    """The Analyses whose counts (`block_counts`' `2 x n_analyses` total) exceed the bound."""
    flagged = []
    for index in range(counts.shape[1]):
        observed, genome_wide = (int(v) for v in counts[:, index])
        if observed < MIN_OBSERVED:
            continue
        measured = ZPlausibility(index, observed, genome_wide)
        if measured.share_genome_wide > MAX_SHARE_GENOME_WIDE:
            flagged.append(measured)
    return flagged


def _measure(_target: Target, block: Block) -> np.ndarray:
    return block_counts(block)


def block_counts(block: Block) -> np.ndarray:
    """Per Analysis: observed associations, and those at 5e-8."""
    observed = states(block.z_missing, np.isnan(block.se), block.imputed) == State.OBSERVED
    with np.errstate(invalid="ignore"):
        abs_z = np.abs(block.z)
    counts = np.zeros((2, block.n_analyses), dtype=np.int64)
    counts[_OBSERVED] = block.per_analysis(observed)
    counts[_GENOME_WIDE] = block.per_analysis(observed & (abs_z >= _Z_GENOME_WIDE))
    return counts


def _judge(target: Target, counts: np.ndarray) -> Iterator[str]:
    """A standalone component's Analyses on its own counts; a Hybrid's on both
    components' counts summed, judged when the second component's pass ends."""
    summed = _hybrid_total(target, np.asarray(counts).reshape(2, -1))
    if summed is None:
        return
    label, total = summed
    flagged = implausible(total)
    if not flagged:
        return
    ids = _analysis_ids(target)
    named = "; ".join(
        f"{ids.get(m.analysis_index, f'analysis_index {m.analysis_index}')!r}: "
        f"{m.share_genome_wide:.1%} of {m.observed:,} observed associations at 5e-8"
        for m in flagged
    )
    yield (
        f"{len(flagged)} Analysis/Analyses in {label} have an implausible stored z "
        f"for a genome-wide scan (more than {MAX_SHARE_GENOME_WIDE:.0%} of observed "
        f"associations at 5e-8): {named}. Their effects are probably on another scale (an "
        "odds ratio in `beta`, or a mis-scaled effect); exclude them and rebuild (#510, "
        "ADR 0069)"
    )


def _hybrid_total(target: Target, counts: np.ndarray) -> tuple[str, np.ndarray] | None:
    """The counts to judge, and what they describe; None while a Hybrid waits.

    A Hybrid's Analysis is one Analysis over two components, so it is judged on its
    counts summed over both: judged per component, the 1,000-observation floor
    left unjudged a part too small to decide alone, and an Analysis whose Dense
    part is all at 5e-8 but under the floor passed on a sound-looking Overflow
    (#510). The first component's counts wait on the release; the second's pass
    sums and judges. If the other component's pass never ends -- its fold skipped
    behind a failed prerequisite -- that skip is the record, as for any fold.
    """
    root = target.outer_release()
    if root.layout is not PrimaryStorageLayout.HYBRID:
        return target.label, counts
    waiting = root.forget(_WAITING)
    if waiting is None:
        root.remember(_WAITING, lambda: (target.label, counts))
        return None
    other_label, other = waiting
    return f"{other_label} and {target.label}", _summed(other, counts)


def _summed(one: np.ndarray, two: np.ndarray) -> np.ndarray:
    """Two components' counts, per Analysis; a component holding no block is zero."""
    if one.shape == two.shape:
        total: np.ndarray = one + two
        return total
    for empty, full in ((one, two), (two, one)):
        if empty.shape[1] == 1 and not empty.any():
            return full
    raise ValueError(
        f"the Hybrid's components count {one.shape[1]} and {two.shape[1]} Analyses; "
        "they hold the same Analyses"
    )


def _analysis_ids(target: Target) -> dict[int, str]:
    """Each Analysis's id by index, from the outer release's `analyses.tsv`, which
    every layout's release carries (a Hybrid's root, its Analyses' copy)."""
    ids: dict[int, str] = {}
    release = target.outer_release()
    if not release.store.analyses_path.exists():
        return ids  # the analyses.tsv rules report a release without one
    for row in release.analyses.rows:
        try:
            ids[int(row.get("analysis_index", ""))] = str(row.get("analysis_id", ""))
        except ValueError:
            continue  # a malformed row is the analyses.analysis_index rule's
    return ids


def _genome_wide(target: Target) -> bool:
    return target.store.manifest.association_coverage is AssociationCoverage.GWAS


@dataclass(frozen=True, kw_only=True)
class _FoldByFormat(Fold[Block, np.ndarray]):
    """A fold whose severity is the one the release's format owes (`severity_owed`)."""

    def judged(self, target: Target, total: np.ndarray) -> Iterator[Finding]:
        owed = severity_owed(target, self.severity)
        for message in self.judge(target, total):
            yield self.finding(target, message, None if owed is self.severity else owed)


RULES: tuple[Rule, ...] = (
    _FoldByFormat(
        name=RULE,
        kinds=Kind.GRID | Kind.CSR,
        after=VALUES_AFTER,
        when=_genome_wide,
        scan=ASSOCIATIONS,
        zero=np.zeros((2, 1), dtype=np.int64),
        measure=_measure,
        judge=_judge,
    ),
)
