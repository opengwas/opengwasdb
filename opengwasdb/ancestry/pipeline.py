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
import time
from dataclasses import dataclass
from pathlib import Path

from opengwasdb.ancestry.catalogue import CatalogueRow, write_catalogue
from opengwasdb.ancestry.mixture import AncestryAssignment, Gates, assign_from_source
from opengwasdb.ancestry.reference import AncestryReference
from opengwasdb.build.ordered_pool import fork_each
from opengwasdb.model.manifest_columns import (
    manifest_n,
    manifest_trait_name,
    resolve_manifest_columns,
)
from opengwasdb.readers.assembly import assembly_plan, declared_source_build
from opengwasdb.readers.gwas_vcf import (
    GWAS_VCF_CAPABILITY,
    load_liftover,
    write_regions_file,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceRow:
    """One row of the raw source manifest (build columns + reported population)."""

    trait_id: str
    file_path: str
    trait_name: str
    n: int
    reported_population: str
    source_genome_build: str
    source_reader_capability: str = GWAS_VCF_CAPABILITY


def read_source_manifest(
    path: str | Path, *, default_source_genome_build: str | None = None
) -> list[SourceRow]:
    """Read the raw source manifest TSV.

    Requires ``analysis_id`` and ``source_file`` (issue #170; the pre-ADR-0034
    ``trait_id``/``file_path`` names remain readable as aliases);
    ``analysis_label`` (legacy ``trait_name``) defaults to the analysis id,
    ``sample_size`` (legacy ``n``) to 0, ``reported_population`` to empty, and
    ``source_reader_capability`` to GWAS-VCF when absent -- the only format
    ancestry assignment could read before issue #115, so a manifest written
    without the column keeps its meaning. The canonical ``analyses.tsv`` name
    wins when a manifest carries both spellings.

    ``source_genome_build`` is required unless the caller supplies an explicit
    default. ``source_assembly`` remains a readable alias, but two populated
    spellings must agree after normalisation. No filename, coordinate, or VCF
    header is used to guess a missing build (issue #277).
    """
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fieldnames = set(reader.fieldnames or ())
        cols = resolve_manifest_columns(fieldnames, path)
        raw_rows = list(reader)
    rows = [
        SourceRow(
            trait_id=r[cols.analysis_id],
            file_path=r[cols.source_file],
            trait_name=manifest_trait_name(
                r, cols.analysis_label, r[cols.analysis_id], fallback_on_blank=True
            ),
            n=manifest_n(r, cols.sample_size),
            reported_population=r.get("reported_population", "").strip(),
            source_genome_build=declared_source_build(
                r,
                default=default_source_genome_build,
                context=f"{path}: Analysis {r[cols.analysis_id]!r}",
            ),
            source_reader_capability=(
                r.get("source_reader_capability") or GWAS_VCF_CAPABILITY
            ),
        )
        for r in raw_rows
    ]
    if not rows:
        raise ValueError(f"empty source manifest: {path}")
    return rows


@dataclass(frozen=True)
class _AnnotationContext:
    """What every worker reads: the Ancestry Reference Panel and the run's inputs.

    ``fork_each`` hands this to each forked worker by inheritance, so the panel
    (~1 GB) crosses no pickle boundary once per Analysis (issue #395).
    """

    reference: AncestryReference
    gates: Gates
    regions_file: Path
    liftovers: dict[str, object]


def _annotate_one(
    context: _AnnotationContext, task: tuple[int, str, str, str]
) -> tuple[int, AncestryAssignment]:
    """Worker: extract AF at reference sites and assign ancestry for one source.

    The task's index comes back beside its assignment because ``fork_each``
    yields in completion order and the Catalogue is written in input order.
    """
    index, file_path, capability, source_build = task
    return index, assign_from_source(
        file_path,
        context.reference,
        context.gates,
        capability=capability,
        regions_file=context.regions_file,
        liftover=context.liftovers.get(source_build),
    )


def _configured_liftovers(
    source_rows: list[SourceRow], reference_build: str, chain_file: str | Path | None
) -> dict[str, object]:
    """Validate every format/build pair, then load each required chain once."""
    plans = {
        (row.source_reader_capability, row.source_genome_build): assembly_plan(
            capability=row.source_reader_capability,
            source_build=row.source_genome_build,
            reference_build=reference_build,
            chain_file=chain_file,
        )
        for row in source_rows
    }
    return {
        plan.source_build: load_liftover(
            from_build=plan.source_build,
            to_build=plan.reference_build,
            chain_file=str(plan.chain_file),
        )
        for plan in plans.values()
        if plan.requires_liftover
    }


def _catalogue_rows(
    source_rows: list[SourceRow], assignments: list[AncestryAssignment]
) -> list[CatalogueRow]:
    return [
        CatalogueRow(
            trait_id=source.trait_id,
            file_path=source.file_path,
            trait_name=source.trait_name,
            n=source.n,
            reported_population=source.reported_population,
            assignment=assignment,
            source_reader_capability=source.source_reader_capability,
        )
        for source, assignment in zip(source_rows, assignments, strict=True)
    ]


def annotate_catalogue(
    source_rows: list[SourceRow],
    reference: AncestryReference,
    gates: Gates,
    out_path: str | Path,
    *,
    catalogue_version: str,
    ancestry_reference_version: str,
    reference_build: str,
    chain_file: str | Path | None = None,
    n_workers: int = 1,
    regions_file: str | Path | None = None,
) -> list[CatalogueRow]:
    """Annotate every Analysis and write the Catalogue TSV; return the rows.

    A ``bcftools -R`` regions file is built once from the reference sites (unless
    supplied) so each same-build study is read targeted, not full-scanned. A
    supported cross-build GWAS-VCF uses the explicitly configured chain and a
    full source-coordinate scan; every other mismatch is refused before any
    source is read (issue #277). ``fork_each`` hands the Ancestry Reference
    Panel to forked workers by inheritance and yields in completion order, so
    the assignments are put back in input order before the Catalogue is
    written: it does not depend on worker count or completion order.
    """
    out_path = Path(out_path)
    if regions_file is None:
        regions_file = write_regions_file(
            reference.index.keys(), out_path.with_suffix(".regions.txt")
        )
    context = _AnnotationContext(
        reference,
        gates,
        Path(regions_file),
        _configured_liftovers(source_rows, reference_build, chain_file),
    )

    t0 = time.monotonic()
    tasks = [
        (index, row.file_path, row.source_reader_capability, row.source_genome_build)
        for index, row in enumerate(source_rows)
    ]
    # `fork_each` yields in completion order; restore input order by the index
    # each task carries, before anything is written.
    assignments = [
        assignment
        for _index, assignment in sorted(fork_each(_annotate_one, context, tasks, n_workers))
    ]

    rows = _catalogue_rows(source_rows, assignments)

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
