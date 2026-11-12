"""A Ragged Store Release's region table, `regions.tsv` (spec §1, §17, issue #430).

The table a Release Bundle's prep filter writes as `sidecars/sparse_regions.tsv`: one row
per region a build kept rows for, of four kinds, of which only `cis` and
`significant_trans` are regions completion fills (`completion.regions`). A build reads and
checks it whole (`read_region_table`, `check_region_table`) and writes it into the Store
beside `analyses.tsv`, through the same writer (`store_regions`), and declares it in
`manifest.json` (`model.manifest.RegionsEntry`); completion and `validate` find it through
that entry (`read_store_regions`). The rule for which rows complete which Analysis lives
in `completion.regions`; this module is only the table.
"""

from __future__ import annotations

import csv
import hashlib
import io
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from opengwasdb.model.analyses import AnalysesTable, write_analyses
from opengwasdb.model.manifest import RegionsEntry, StoreManifest
from opengwasdb.variants.normalise import normalise_chromosome

__all__ = [
    "COMPLETED_KINDS",
    "REGIONS_FILE",
    "REGION_KINDS",
    "Region",
    "RegionTable",
    "RegionTableError",
    "check_region_table",
    "read_region_table",
    "read_store_regions",
    "store_regions",
]

#: The region table's file in a Ragged Store Release, beside `analyses.tsv` (spec §1, §17).
REGIONS_FILE = "regions.tsv"


@dataclass(frozen=True)
class Region:
    """One region an Analysis is completed inside: a chromosome and an inclusive span,
    named by its table's `region_id` (or, for a cis window, by its Analysis)."""

    chromosome: str
    start: int
    end: int
    kind: str
    region_id: str


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
    `analysis_ids` every Analysis the table gives a row of any kind. `fieldnames` and
    `rows` are the table as read, every column of every row, which is what a build
    writes into the Store (`store_regions`).
    """

    path: Path
    sha256: str
    regions: Mapping[str, tuple[Region, ...]]
    analysis_ids: frozenset[str]
    fieldnames: tuple[str, ...]
    rows: tuple[dict[str, str], ...]


def read_region_table(path: str | Path, assembly: str) -> RegionTable:
    """Read a region table (a bundle's `sidecars/sparse_regions.tsv`), refusing it whole.

    Every row must name a kind in `REGION_KINDS`, coordinates in `assembly` (the Store's,
    GRCh38) and an inclusive span `1 <= start <= end` on a chromosome; a row that does
    not is refused by line, never skipped, since a skipped region is a region completed
    nowhere.
    """
    path = Path(path)
    data = path.read_bytes()
    # One intact stream, so a quoted value spanning lines is read whole, as written.
    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""), delimiter="\t")
    header = _table_header(reader, path)
    rows: list[dict[str, str]] = []
    regions: dict[str, list[Region]] = defaultdict(list)
    for row in reader:
        rows.append(row)
        region = _table_region(row, assembly, f"region table {path} line {reader.line_num}")
        if region.kind in COMPLETED_KINDS:
            regions[row["analysis_id"]].append(region)
    return RegionTable(
        path=path,
        sha256=hashlib.sha256(data).hexdigest(),
        regions={a: tuple(r) for a, r in regions.items()},
        analysis_ids=frozenset(row["analysis_id"] for row in rows),
        fieldnames=tuple(header),
        rows=tuple(rows),
    )


def _table_header(reader: csv.DictReader[str], path: Path) -> list[str]:
    """The table's columns, refusing a table without one `read_region_table` reads."""
    header = list(reader.fieldnames or [])
    missing = [c for c in TABLE_COLUMNS if c not in header]
    if missing:
        raise RegionTableError(f"region table {path} has no {', '.join(missing)} column")
    return header


def read_store_regions(store: str | Path) -> RegionTable | None:
    """The region table a Store Release's manifest declares, read in the Store's assembly,
    or None where it declares none (spec §17, §21.2).

    Found through the manifest, never by the file alone: a `regions.tsv` the manifest does
    not declare, a declared one that is missing, and one whose bytes are not those the
    manifest's SHA-256 names are each a Store that disagrees with itself, and refused.
    """
    store = Path(store)
    manifest = StoreManifest.load(store)
    declared = manifest.regions
    if declared is None:
        if (store / REGIONS_FILE).exists():
            raise RegionTableError(
                f"{store / REGIONS_FILE} is present but manifest.json declares no regions "
                "entry; a Store's region table is found through its manifest (spec §17)"
            )
        return None
    path = store / declared.path
    if declared.path != REGIONS_FILE or not path.is_file():
        raise RegionTableError(
            f"manifest.json declares its region table at {declared.path!r}, but the Store "
            f"holds no such file; a Store's region table is {REGIONS_FILE} (spec §17)"
        )
    table = read_region_table(path, manifest.reference_assembly)
    if table.sha256 != declared.sha256:
        raise RegionTableError(
            f"{path} has sha256 {table.sha256}, not the sha256 manifest.json declares "
            f"({declared.sha256})"
        )
    return table


def store_regions(store: str | Path, table: RegionTable) -> RegionsEntry:
    """Write `table` into a Store as `regions.tsv`, every row it holds, through the writer
    `analyses.tsv` is written by, and return the manifest entry that declares it."""
    path = write_analyses(
        Path(store) / REGIONS_FILE, AnalysesTable(fieldnames=table.fieldnames, rows=table.rows)
    )
    return RegionsEntry(path=REGIONS_FILE, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def check_region_table(table: RegionTable, analysis_ids: Iterable[str]) -> None:
    """Refuse a table naming an Analysis the Store does not hold: another Store's table, or a
    stale one. The one check a build, `validate` and completion all make of it."""
    foreign = sorted(table.analysis_ids - set(analysis_ids))
    if foreign:
        raise RegionTableError(
            f"region table {table.path} names {len(foreign):,} Analyses the Store does "
            f"not hold; first: {foreign[:5]}"
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
