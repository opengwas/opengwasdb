"""The one writer of Ragged completion region tables in the tests (issue #430).

A region table is what a Release Bundle records as `sidecars/sparse_regions.tsv`, what
`build-ragged-ssf --regions` reads and writes into the Store as `regions.tsv`, and what
completion and the Observed-View Check read from there. The tests write theirs here,
with the sidecar's columns, so every fixture's table has the shape opengwasdb-stores
passes.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from opengwasdb.model.analyses import read_analysis_records

#: The sidecar's columns, in its order.
COLUMNS = (
    "analysis_id", "region_id", "region_kind", "chromosome", "start", "end",
    "source_genome_build", "target_id", "target_label", "lead_variant_id",
    "pvalue_threshold", "n_variants_retained", "region_policy_id",
)

#: One region: `(analysis_id, region_kind, chromosome, start, end)`.
Row = tuple[str, str, str, int, int]


def write_region_table(
    path: Path,
    rows: Iterable[Row],
    *,
    assembly: str = "GRCh38",
    without: str | None = None,
    target_label: str = "",
) -> Path:
    """Write `rows` as a region table at `path`, every row in `assembly`. Each region is
    named `{analysis_id}__{region_kind}__{n}`; the columns completion does not read are
    filler, `target_label` among them. `without` leaves one column out, for a test of a
    table that lacks it."""
    columns = [c for c in COLUMNS if c != without]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)
        for n, (analysis_id, kind, chromosome, start, end) in enumerate(rows):
            record = {
                "analysis_id": analysis_id, "region_id": f"{analysis_id}__{kind}__{n:03d}",
                "region_kind": kind, "chromosome": chromosome, "start": start, "end": end,
                "source_genome_build": assembly, "target_id": "", "target_label": target_label,
                "lead_variant_id": '""', "pvalue_threshold": "5e-08",
                "n_variants_retained": 1, "region_policy_id": "test-policy",
            }
            writer.writerow([record[c] for c in columns])
    return path


def every_analysis(store: Path, kind: str, chromosome: str, start: int, end: int) -> list[Row]:
    """One `kind` region, `chromosome:start-end`, for every Analysis of `store`."""
    return [
        (a.analysis_id, kind, chromosome, start, end)
        for a in read_analysis_records(store / "analyses.tsv")
    ]


#: The table of a `cis_and_signals` or `signals_only` fixture whose build recorded no region
#: but its Analyses' cis windows: completed on each Trait position +/- the window, as a `cis`
#: Store is with no table (issue #430).
NO_SIGNIFICANT_REGIONS: tuple[Row, ...] = ()


def with_regions(store: Path, rows: Iterable[Row]) -> Path:
    """`store`, given `rows` as its `regions.tsv`, its manifest's `regions` entry, and the
    `format_version` a build given the table stamps: the file written by the writer a build
    uses (`model.regions.store_regions`), for a fixture whose builder takes no table, the
    BESD one, or a test that changes the table a Store holds."""
    from opengwasdb.model.regions import read_region_table, store_regions
    from opengwasdb.store.open import format_version_for, indels_reference_first

    table = write_region_table(store.parent / f"{store.name}.regions-input.tsv", rows)
    entry = store_regions(store, read_region_table(table, "GRCh38"))
    edit_manifest(
        store,
        lambda manifest: {
            **manifest,
            "regions": entry.to_manifest(),
            # The Store's own indel spelling stays what its stamp says (ADR 0079).
            "format_version": format_version_for(
                carries_regions=True,
                reference_first_indels=indels_reference_first(manifest["format_version"]),
            ),
        },
    )
    return store


def edit_manifest(store: Path, edit: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
    """Rewrite `store`'s `manifest.json` as `edit` returns it, as a build writes one."""
    path = store / "manifest.json"
    edited = edit(json.loads(path.read_text(encoding="utf-8")))
    path.write_text(json.dumps(edited, indent=2, sort_keys=True) + "\n", encoding="utf-8")
