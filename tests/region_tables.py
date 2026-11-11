"""The one writer of Ragged completion region tables in the tests (issue #430).

A region table is what a Release Bundle records as `sidecars/sparse_regions.tsv`, and
what `complete-ragged --regions` and the Observed-View Check's `--regions` read. The
tests write theirs here, with the sidecar's columns, so every fixture's table has the
shape opengwasdb-stores passes.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

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
    path: Path, rows: Iterable[Row], *, assembly: str = "GRCh38", without: str | None = None
) -> Path:
    """Write `rows` as a region table at `path`, every row in `assembly`. Each region is
    named `{analysis_id}__{region_kind}__{n}`; the columns completion does not read are
    filler. `without` leaves one column out, for a test of a table that lacks it."""
    columns = [c for c in COLUMNS if c != without]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)
        for n, (analysis_id, kind, chromosome, start, end) in enumerate(rows):
            record = {
                "analysis_id": analysis_id, "region_id": f"{analysis_id}__{kind}__{n:03d}",
                "region_kind": kind, "chromosome": chromosome, "start": start, "end": end,
                "source_genome_build": assembly, "target_id": "", "target_label": "",
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
