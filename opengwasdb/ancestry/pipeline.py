"""Annotate a raw source manifest into the Analysis Catalogue (issue 063).

Reads a raw source manifest (all candidate Analyses), extracts allele frequencies
at the reference sites in parallel (targeted ``bcftools -R``, not a full scan),
runs the NNLS mixture + gates per Analysis, and writes the versioned Analysis
Catalogue. Non-EUR and Unassigned Analyses are annotated and retained (parked),
never dropped. Results are order-preserving and independent of worker count, so a
re-run with the same inputs and versions reproduces the Catalogue byte-for-byte.
"""

from __future__ import annotations

import csv
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from opengwasdb.ancestry.catalogue import CatalogueRow, write_catalogue
from opengwasdb.ancestry.mixture import AncestryAssignment, Gates, assign_from_source
from opengwasdb.ancestry.reference import AncestryReference
from opengwasdb.readers.gwas_vcf import GWAS_VCF_CAPABILITY, write_regions_file

log = logging.getLogger(__name__)

#: Canonical ``analyses.tsv`` names -> pre-ADR-0034 source-manifest aliases
#: (issue #170). The canonical name wins when a manifest carries both.
_SOURCE_MANIFEST_COLUMN_ALIASES: dict[str, str] = {
    "analysis_id": "trait_id",
    "source_file": "file_path",
    "analysis_label": "trait_name",
    "sample_size": "n",
}


def _source_manifest_column(fieldnames: set[str], canonical: str) -> str | None:
    """Resolve `canonical`'s column name, or None when neither spelling is
    present: the canonical ``analyses.tsv`` name when the manifest carries it,
    otherwise its legacy source-manifest name (issue #170)."""
    if canonical in fieldnames:
        return canonical
    legacy = _SOURCE_MANIFEST_COLUMN_ALIASES[canonical]
    return legacy if legacy in fieldnames else None


def _required_source_manifest_column(
    fieldnames: set[str], canonical: str, path: str | Path
) -> str:
    """`_source_manifest_column`, but raise when neither spelling is present."""
    column = _source_manifest_column(fieldnames, canonical)
    if column is None:
        legacy = _SOURCE_MANIFEST_COLUMN_ALIASES[canonical]
        raise ValueError(
            f"source manifest {path} is missing required column: "
            f"{canonical!r} (legacy name {legacy!r})"
        )
    return column


def _source_row(
    row: dict[str, str],
    analysis_id_col: str,
    source_file_col: str,
    label_col: str | None,
    sample_size_col: str | None,
) -> SourceRow:
    """Build one `SourceRow`, applying the alias defaults (issue #170)."""
    trait_id = row[analysis_id_col]
    return SourceRow(
        trait_id=trait_id,
        file_path=row[source_file_col],
        trait_name=(row.get(label_col) or trait_id) if label_col else trait_id,
        n=int((row.get(sample_size_col) or 0) if sample_size_col else 0),
        reported_population=row.get("reported_population", "").strip(),
        source_reader_capability=(
            row.get("source_reader_capability") or GWAS_VCF_CAPABILITY
        ),
    )


@dataclass(frozen=True)
class SourceRow:
    """One row of the raw source manifest (build columns + reported population)."""

    trait_id: str
    file_path: str
    trait_name: str
    n: int
    reported_population: str
    source_reader_capability: str = GWAS_VCF_CAPABILITY


def read_source_manifest(path: str | Path) -> list[SourceRow]:
    """Read the raw source manifest TSV.

    Requires ``analysis_id`` and ``source_file`` (issue #170; the pre-ADR-0034
    ``trait_id``/``file_path`` names remain readable as aliases);
    ``analysis_label`` (legacy ``trait_name``) defaults to the analysis id,
    ``sample_size`` (legacy ``n``) to 0, ``reported_population`` to empty, and
    ``source_reader_capability`` to GWAS-VCF when absent -- the only format
    ancestry assignment could read before issue #115, so a manifest written
    without the column keeps its meaning. The canonical ``analyses.tsv`` name
    wins when a manifest carries both spellings.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fieldnames = set(reader.fieldnames or ())
        raw_rows = list(reader)
    analysis_id_col = _required_source_manifest_column(fieldnames, "analysis_id", path)
    source_file_col = _required_source_manifest_column(fieldnames, "source_file", path)
    label_col = _source_manifest_column(fieldnames, "analysis_label")
    sample_size_col = _source_manifest_column(fieldnames, "sample_size")
    rows = [
        _source_row(row, analysis_id_col, source_file_col, label_col, sample_size_col)
        for row in raw_rows
    ]
    if not rows:
        raise ValueError(f"empty source manifest: {path}")
    return rows


# Fork-inherited worker state, set in the parent immediately before the pool is
# created. The reference (~1 GB) is inherited by fork rather than pickled per task.
_WORKER_REFERENCE: AncestryReference | None = None
_WORKER_GATES: Gates | None = None
_WORKER_REGIONS: Path | None = None


def _fork_pool(n_workers: int) -> ProcessPoolExecutor:
    fork_ctx = multiprocessing.get_context("fork")
    return ProcessPoolExecutor(max_workers=n_workers, mp_context=fork_ctx)


def _annotate_one(task: tuple[str, str]) -> AncestryAssignment:
    """Worker: extract AF at reference sites and assign ancestry for one source."""
    assert _WORKER_REFERENCE is not None and _WORKER_GATES is not None
    file_path, capability = task
    return assign_from_source(
        file_path,
        _WORKER_REFERENCE,
        _WORKER_GATES,
        capability=capability,
        regions_file=_WORKER_REGIONS,
    )


def annotate_catalogue(
    source_rows: list[SourceRow],
    reference: AncestryReference,
    gates: Gates,
    out_path: str | Path,
    *,
    catalogue_version: str,
    ancestry_reference_version: str,
    n_workers: int = 1,
    regions_file: str | Path | None = None,
) -> list[CatalogueRow]:
    """Annotate every Analysis and write the Catalogue TSV; return the rows.

    A ``bcftools -R`` regions file is built once from the reference sites (unless
    supplied) so each study is read targeted, not full-scanned. Assignments are
    gathered back into input order, so the Catalogue does not depend on worker
    count or completion order.
    """
    global _WORKER_REFERENCE, _WORKER_GATES, _WORKER_REGIONS

    out_path = Path(out_path)
    if regions_file is None:
        regions_file = write_regions_file(
            reference.index.keys(), out_path.with_suffix(".regions.txt")
        )
    regions_file = Path(regions_file)

    _WORKER_REFERENCE = reference
    _WORKER_GATES = gates
    _WORKER_REGIONS = regions_file

    t0 = time.monotonic()
    tasks = [(row.file_path, row.source_reader_capability) for row in source_rows]
    if n_workers <= 1:
        assignments = [_annotate_one(t) for t in tasks]
    else:
        with _fork_pool(n_workers) as pool:
            # executor.map preserves input order regardless of completion order.
            assignments = list(pool.map(_annotate_one, tasks))

    rows = [
        CatalogueRow(
            trait_id=src.trait_id,
            file_path=src.file_path,
            trait_name=src.trait_name,
            n=src.n,
            reported_population=src.reported_population,
            assignment=a,
            source_reader_capability=src.source_reader_capability,
        )
        for src, a in zip(source_rows, assignments, strict=True)
    ]

    write_catalogue(
        out_path,
        rows,
        reference.superpops,
        catalogue_version=catalogue_version,
        ancestry_reference_version=ancestry_reference_version,
        gates=gates,
    )

    n_assigned = sum(1 for r in rows if r.assignment.assigned_ancestry is not None)
    log.info(
        "Catalogue: %d analyses annotated (%d assigned, %d parked) in %.1fs → %s",
        len(rows),
        n_assigned,
        len(rows) - n_assigned,
        time.monotonic() - t0,
        out_path,
    )
    return rows
