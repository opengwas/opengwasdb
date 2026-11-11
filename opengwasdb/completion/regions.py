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

The table is the one the Store's build recorded, a bundle's `sidecars/sparse_regions.tsv`
(`read_region_table`): a Store records which rows it holds, not why it holds them. Only a
Store whose Association Coverage is `cis` holds no region but its cis windows, so only one
may be completed without a table; any other would have its significant trans regions
completed nowhere.

A block is completed for an Analysis when its extent overlaps one of the Analysis's
regions (`region_blocks`), and the fit uses every observation the Analysis holds at the
block's panel variants. Only the panel variants inside a region's inclusive
`[start, end]` (`inside`) become imputed or missing rows; the rest of the block is left
as the source holds it.
"""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from opengwasdb.completion.ld_panel import LDBlock, list_all_blocks, list_chromosomes
from opengwasdb.model.analyses import Analysis
from opengwasdb.model.enums import AssociationCoverage
from opengwasdb.variants.normalise import chromosome_sort_key, normalise_chromosome

__all__ = [
    "COMPLETED_KINDS",
    "REGION_KINDS",
    "Region",
    "RegionTable",
    "RegionTableError",
    "analysis_regions",
    "has_trait_position",
    "RegionBlocks",
    "inside",
    "read_region_table",
    "region_blocks",
]


@dataclass(frozen=True)
class Region:
    """One region an Analysis is completed inside: a chromosome and an inclusive span,
    named by its table's `region_id` (or, for a cis window, by its Analysis)."""

    chromosome: str
    start: int
    end: int
    kind: str
    region_id: str


def has_trait_position(a: Analysis) -> bool:
    """Whether the Analysis names a Trait genomic position to centre a cis window on."""
    return bool(a.trait_chr) and bool(a.trait_bp)


#: Every kind a region table may give a row, and the two that make a region. A
#: suggestive lead is one row, kept to say a locus exists; a QC-panel row is kept for the
#: build's ancestry and frequency checks. Neither is evidence completion fits from.
REGION_KINDS = ("cis", "significant_trans", "suggestive_trans", "qc_panel")
COMPLETED_KINDS = frozenset({"cis", "significant_trans"})
#: The columns `read_region_table` reads; a table may carry more, which it ignores.
TABLE_COLUMNS = (
    "analysis_id", "region_id", "region_kind", "chromosome", "start", "end", "source_genome_build",
)


class RegionTableError(ValueError):
    """A region table completion cannot complete against, or an Analysis it cannot place."""


@dataclass(frozen=True)
class RegionTable:
    """A build's per-Analysis region table, read and checked whole (`read_region_table`).

    `regions` holds each Analysis's `cis` and `significant_trans` rows only, and
    `analysis_ids` every Analysis the table gives a row of any kind.
    """

    path: Path
    sha256: str
    regions: Mapping[str, tuple[Region, ...]]
    analysis_ids: frozenset[str]


def read_region_table(path: str | Path, assembly: str) -> RegionTable:
    """Read a region table (a bundle's `sidecars/sparse_regions.tsv`), refusing it whole.

    Every row must name a kind in `REGION_KINDS`, coordinates in `assembly` (the Store's,
    GRCh38) and an inclusive span `1 <= start <= end` on a chromosome; a row that does
    not is refused by line, never skipped, since a skipped region is a region completed
    nowhere.
    """
    path = Path(path)
    data = path.read_bytes()
    lines = data.decode("utf-8").splitlines()
    rows = list(csv.DictReader(lines, delimiter="\t"))
    header = next(csv.reader(lines[:1], delimiter="\t"), [])
    missing = [c for c in TABLE_COLUMNS if c not in header]
    if missing:
        raise RegionTableError(f"region table {path} has no {', '.join(missing)} column")
    regions: dict[str, list[Region]] = defaultdict(list)
    for line, row in enumerate(rows, start=2):
        region = _table_region(row, assembly, f"region table {path} line {line}")
        if region.kind in COMPLETED_KINDS:
            regions[row["analysis_id"]].append(region)
    return RegionTable(
        path=path,
        sha256=hashlib.sha256(data).hexdigest(),
        regions={a: tuple(r) for a, r in regions.items()},
        analysis_ids=frozenset(row["analysis_id"] for row in rows),
    )


def _table_region(row: Mapping[str, str], assembly: str, where: str) -> Region:
    kind = row["region_kind"]
    if kind not in REGION_KINDS:
        raise RegionTableError(
            f"{where}: region_kind {kind!r} is none of {', '.join(REGION_KINDS)}"
        )
    if row["source_genome_build"] != assembly:
        raise RegionTableError(
            f"{where}: source_genome_build {row['source_genome_build']!r}, not the "
            f"Store's {assembly}"
        )
    try:
        region = Region(
            normalise_chromosome(row["chromosome"]), int(row["start"]), int(row["end"]), kind,
            row["region_id"],
        )
    except ValueError as exc:
        raise RegionTableError(f"{where}: {exc}") from exc
    if not 1 <= region.start <= region.end:
        raise RegionTableError(f"{where}: start {region.start} and end {region.end} are no span")
    return region


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
            f"the Store's Association Coverage is {coverage.value}, so its regions are in the "
            "region table its build recorded: give complete-ragged that table (--regions). "
            "Only a cis Store is completed without one, on each Trait position +/- the cis "
            "window"
        )


def _refuse_foreign_rows(analyses: Sequence[Analysis], table: RegionTable | None) -> None:
    """A table naming an Analysis the Store lacks is another Store's table, or a stale one."""
    if table is None:
        return
    foreign = sorted(table.analysis_ids - {a.analysis_id for a in analyses})
    if foreign:
        raise RegionTableError(
            f"region table {table.path} names {len(foreign):,} Analyses the Store does "
            f"not hold; first: {foreign[:5]}"
        )


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
        "and no region table, so there is no region to complete them in: give "
        "complete-ragged the region table the Store's build recorded (--regions)"
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
