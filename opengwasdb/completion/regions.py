"""The regions an Analysis is completed against, and the LD blocks holding them (ADR 0014,
ADR 0039).

One definition for `complete-ragged`, which completes inside these regions, and for the
Observed-View Check (issue #296), which rebuilds them to test that every row completion
added lies inside one. An Analysis's regions are its cis region, where it has a Trait
position, and its significant trans regions (issue #430):

* the **cis region** is the region table's `cis` row where it has one, and otherwise the
  Trait position +/- `cis_window_bp`;
* each **significant trans region** is one of the table's `significant_trans` rows;
* `suggestive_trans` and `qc_panel` rows never make a region, however many of them fall
  in one block.

The table is the one the Store's build recorded: a Release Bundle's
`sidecars/sparse_regions.tsv`, which `build-ragged-ssf --regions` checks, writes into
the Store as `regions.tsv` (`model.regions`), beside `analyses.tsv`, and declares in its
manifest, and which completion carries into the completed Store unchanged. Only a Store
whose Association Coverage is `cis` holds no region but its cis windows, so only one may
be completed without it; any other would have its significant trans regions completed
nowhere.

A block is completed for an Analysis when its extent overlaps one of the Analysis's
regions (`region_blocks`), and the fit uses every observation the Analysis holds at the
block's panel variants. Only the panel variants inside a region's inclusive
`[start, end]` (`inside`) become imputed or missing rows; the rest of the block is left
as the source holds it.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from opengwasdb.completion.ld_panel import LDBlock, list_all_blocks, list_chromosomes
from opengwasdb.model.analyses import Analysis
from opengwasdb.model.enums import AssociationCoverage
from opengwasdb.model.regions import (
    REGION_KINDS,
    REGIONS_FILE,
    Region,
    RegionTable,
    RegionTableError,
    check_region_table,
)
from opengwasdb.variants.normalise import chromosome_sort_key, normalise_chromosome

__all__ = [
    "RegionBlocks",
    "analysis_regions",
    "has_trait_position",
    "inside",
    "region_blocks",
    "region_summary",
]


def has_trait_position(a: Analysis) -> bool:
    """Whether the Analysis names a Trait genomic position to centre a cis window on."""
    return bool(a.trait_chr) and bool(a.trait_bp)


def region_summary(table: RegionTable, analyses: Sequence[Analysis]) -> dict[str, object]:
    """What `opengwasdb info` says of a Store's regions: its rows by kind, and how many of
    its Analyses have a completion region (a `cis` or `significant_trans` row, or a Trait
    position to centre a cis window on) and how many have none."""
    counts = {kind: 0 for kind in REGION_KINDS}
    for row in table.rows:
        counts[row["region_kind"]] += 1
    with_region = sum(1 for a in analyses if _regions_of(a, table, cis_window_bp=0))
    return {
        "rows_by_kind": counts,
        "analyses_with_region": with_region,
        "analyses_without_region": len(analyses) - with_region,
    }


def _cis_window(a: Analysis, cis_window_bp: int) -> Region:
    """The Trait position +/- `cis_window_bp`, for an Analysis the table gives no cis row."""
    centre = int(a.trait_bp)
    return Region(
        normalise_chromosome(a.trait_chr),
        max(1, centre - cis_window_bp),
        centre + cis_window_bp,
        "cis",
        f"{a.analysis_id}__cis_window",
    )


def analysis_regions(
    analyses: Sequence[Analysis],
    wanted: Iterable[int],
    table: RegionTable | None,
    cis_window_bp: int,
    coverage: AssociationCoverage,
) -> dict[int, tuple[Region, ...]]:
    """Each wanted Analysis's regions, by its index in `analyses` (the Store's Analyses, in
    `analysis_index` order). An Analysis with none is absent: it is carried Observed-Only.

    Refuses a Store whose Association Coverage (`coverage`) is not `cis` with no table, a
    table row naming an Analysis the Store lacks, and a wanted Analysis with no Trait
    position the table says nothing of, rather than complete any by another rule.
    """
    wanted = list(wanted)
    _refuse_missing_table(table, coverage)
    _refuse_foreign_rows(analyses, table)
    _refuse_unplaced(analyses, wanted, table)
    regions: dict[int, tuple[Region, ...]] = {}
    for index in wanted:
        held = _regions_of(analyses[index], table, cis_window_bp)
        if held:
            regions[index] = held
    return regions


def _refuse_missing_table(table: RegionTable | None, coverage: AssociationCoverage) -> None:
    """Only a `cis` Store's regions are its cis windows alone. Any other records significant
    regions only its build's table holds, which completing without it would leave out."""
    if table is None and coverage is not AssociationCoverage.CIS:
        raise RegionTableError(
            f"the Store's Association Coverage is {coverage.value} and it holds no "
            f"{REGIONS_FILE}, the region table its build records: rebuild it with "
            "build-ragged-ssf --regions. Only a cis Store is completed without one, on each "
            "Trait position +/- the cis window"
        )


def _refuse_foreign_rows(analyses: Sequence[Analysis], table: RegionTable | None) -> None:
    """A table naming an Analysis the Store lacks is another Store's table, or a stale one."""
    if table is not None:
        check_region_table(table, (a.analysis_id for a in analyses))


def _refuse_unplaced(
    analyses: Sequence[Analysis], wanted: Sequence[int], table: RegionTable | None
) -> None:
    """An Analysis with no Trait position has its regions from the table alone, so one the
    table says nothing of is refused, never completed by another rule (issue #430)."""
    known = frozenset() if table is None else table.analysis_ids
    unplaced = [
        analyses[i].analysis_id
        for i in wanted
        if not has_trait_position(analyses[i]) and analyses[i].analysis_id not in known
    ]
    if not unplaced:
        return
    where = (
        f"and the Store holds no {REGIONS_FILE}, so there is no region to complete them "
        "in: rebuild it with build-ragged-ssf --regions"
        if table is None
        else f"and no row in region table {table.path}"
    )
    raise RegionTableError(
        f"{len(unplaced):,} Analyses have no Trait position {where}; first: {unplaced[:5]}"
    )


def _regions_of(
    a: Analysis, table: RegionTable | None, cis_window_bp: int
) -> tuple[Region, ...]:
    recorded = () if table is None else table.regions.get(a.analysis_id, ())
    if has_trait_position(a) and not any(r.kind == "cis" for r in recorded):
        return (_cis_window(a, cis_window_bp), *recorded)
    return recorded


def inside(regions: Iterable[Region], chromosome: str, positions: np.ndarray) -> np.ndarray:
    """Which of `positions` on `chromosome` lie inside one of `regions`, bounds included.

    The one test of whether a variant is inside an Analysis's regions: completion's fills
    and rows, and the Observed-View Check's, are all decided by it.
    """
    chromosome = normalise_chromosome(chromosome)
    positions = np.asarray(positions, dtype=np.int64)
    found = np.zeros(len(positions), dtype=bool)
    for region in regions:
        if region.chromosome == chromosome:
            found |= (positions >= region.start) & (positions <= region.end)
    return found


def _overlaps(block: LDBlock, region: Region) -> bool:
    return not (block.end_bp < region.start or block.start_bp > region.end)


def _blocks_overlapping(blocks: Sequence[LDBlock], held: Sequence[Region]) -> list[LDBlock]:
    """The blocks, in panel order, whose extent overlaps one of `held`."""
    return [b for b in blocks if any(_overlaps(b, r) for r in held)]


def _overlapping_none(held: Sequence[Region], blocks: Sequence[LDBlock]) -> list[Region]:
    """The regions no block's extent overlaps: nothing is completed in them."""
    return [r for r in held if not any(_overlaps(b, r) for b in blocks)]


def _by_chromosome(
    regions: Mapping[int, Sequence[Region]],
) -> list[tuple[str, dict[int, list[Region]]]]:
    """Each chromosome some region is on, in Store order, with each Analysis's regions there."""
    by_chromosome: dict[str, dict[int, list[Region]]] = defaultdict(lambda: defaultdict(list))
    for index, held in regions.items():
        for region in held:
            by_chromosome[region.chromosome][index].append(region)
    return [(c, by_chromosome[c]) for c in sorted(by_chromosome, key=chromosome_sort_key)]


@dataclass(frozen=True)
class RegionBlocks:
    """Per Analysis, the LD blocks its regions overlap, and the regions no block overlaps."""

    blocks: dict[int, list[LDBlock]]
    uncovered: tuple[Region, ...]


def region_blocks(
    regions: Mapping[int, Sequence[Region]], ld_dir: str | Path, ancestry: str
) -> RegionBlocks:
    """Per Analysis, the LD blocks whose extent overlaps one of its regions, in panel order.

    Each chromosome's blocks are read once for every Analysis, rather than once per
    region, and only the blocks some region overlaps are kept. A region no block
    overlaps, in a gap of the panel or on a chromosome it lacks, is completed nowhere:
    legitimately, so it is returned (`uncovered`) for the caller to report, not refused.
    """
    directory = {normalise_chromosome(c): c for c in list_chromosomes(ld_dir, ancestry)}
    found: dict[int, list[LDBlock]] = defaultdict(list)
    uncovered: list[Region] = []
    for chromosome, by_analysis in _by_chromosome(regions):
        blocks = (
            list_all_blocks(ld_dir, ancestry, directory[chromosome])
            if chromosome in directory
            else []
        )
        for index, held in by_analysis.items():
            uncovered += _overlapping_none(held, blocks)
            found[index] += _blocks_overlapping(blocks, held)
    return RegionBlocks(dict(found), tuple(uncovered))
