"""Analysis Catalogue reader/writer (ADR 0027).

The Catalogue is a versioned TSV with one row per Analysis: the four build-manifest
columns (``trait_id``, ``file_path``, ``trait_name``, ``n``) followed by the
ancestry annotations. Because it is a **superset** of the build manifest and the
builder reads only its four columns via ``DictReader``, a build is a pure row
filter over the Catalogue with no format translation.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from opengwasdb.ancestry.mixture import AncestryAssignment, Gates
from opengwasdb.model.enums import AncestryAssignmentMethod
from opengwasdb.readers.gwas_vcf import GWAS_VCF_CAPABILITY

# The build-manifest columns, first and in this order (superset invariant).
BUILD_COLUMNS = ["trait_id", "file_path", "trait_name", "n"]

# Annotation columns written after the build columns and before the per-super-pop
# composition columns and the version stamps.
_ANNOTATION_COLUMNS = [
    "assigned_ancestry",
    "ancestry_assignment_method",
    "reported_population",
    "af_overlap",
    "nnls_residual",
    "dominant_superpop",
    "dominant_proportion",
    "runner_up_margin",
    "gate_reason",
    # Issue #115. Annotation rather than build columns, so the superset
    # invariant above is untouched -- but `source_reader_capability` is read by
    # the Dense/Hybrid builders' manifest reader when present, so carrying it
    # here is what lets a Catalogue of non-GWAS-VCF sources drive a build.
    "eaf_orientation",
    "eaf_orientation_r",
    "source_reader_capability",
]
_VERSION_COLUMNS = ["catalogue_version", "ancestry_reference_version"]
# Gate provenance: the admission thresholds that produced this Catalogue's labels.
# Recorded so a calibrated τ/δ pick (issue 064) is explicit and reproducible.
_GATE_COLUMNS = [
    "gate_tau",
    "gate_delta",
    "gate_n_min",
    "gate_residual_max",
    "gate_orientation_flip_r",
]

UNASSIGNED = "Unassigned"
_UNASSIGNED = UNASSIGNED


@dataclass(frozen=True)
class CatalogueRow:
    """One annotated Analysis: build columns + ancestry assignment."""

    trait_id: str
    file_path: str
    trait_name: str
    n: int
    reported_population: str
    assignment: AncestryAssignment
    source_reader_capability: str = GWAS_VCF_CAPABILITY


def _prop_column(superpop: str) -> str:
    return f"ancestry_prop_{superpop}"


def catalogue_fieldnames(superpops: list[str]) -> list[str]:
    """Full Catalogue header for the given super-population set."""
    return [
        *BUILD_COLUMNS,
        *_ANNOTATION_COLUMNS,
        *[_prop_column(sp) for sp in superpops],
        *_VERSION_COLUMNS,
        *_GATE_COLUMNS,
    ]


def _row_to_dict(
    row: CatalogueRow,
    superpops: list[str],
    gates: Gates,
    *,
    catalogue_version: str,
    ancestry_reference_version: str,
) -> dict[str, str]:
    a = row.assignment
    out: dict[str, str] = {
        "trait_id": row.trait_id,
        "file_path": row.file_path,
        "trait_name": row.trait_name,
        "n": str(row.n),
        "assigned_ancestry": a.assigned_ancestry or UNASSIGNED,
        "ancestry_assignment_method": (
            AncestryAssignmentMethod.AF_ASSIGNED.value
            if a.assigned_ancestry
            else AncestryAssignmentMethod.UNASSIGNED.value
        ),
        "reported_population": row.reported_population,
        "af_overlap": str(a.af_overlap),
        "nnls_residual": _fmt(a.residual),
        "dominant_superpop": a.dominant_superpop or "",
        "dominant_proportion": _fmt(a.dominant_proportion),
        "runner_up_margin": _fmt(a.runner_up_margin),
        "gate_reason": a.gate_reason,
        "eaf_orientation": a.eaf_orientation,
        "eaf_orientation_r": _fmt(a.eaf_orientation_r),
        "source_reader_capability": row.source_reader_capability,
        "catalogue_version": catalogue_version,
        "ancestry_reference_version": ancestry_reference_version,
        "gate_tau": _fmt(gates.tau),
        "gate_delta": _fmt(gates.delta),
        "gate_n_min": str(gates.n_min),
        "gate_residual_max": _fmt(gates.residual_max),
        "gate_orientation_flip_r": _fmt(gates.orientation_flip_r),
    }
    for sp in superpops:
        out[_prop_column(sp)] = _fmt(a.superpop_composition.get(sp, 0.0))
    return out


def write_catalogue(
    path: str | Path,
    rows: list[CatalogueRow],
    superpops: list[str],
    *,
    catalogue_version: str,
    ancestry_reference_version: str,
    gates: Gates | None = None,
) -> Path:
    """Write the Catalogue TSV (superset of the build manifest).

    ``gates`` (the admission thresholds that produced the labels) are recorded as
    provenance columns; a default ``Gates()`` is used when not supplied.
    """
    path = Path(path)
    gates = gates or Gates()
    fieldnames = catalogue_fieldnames(superpops)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                _row_to_dict(
                    row,
                    superpops,
                    gates,
                    catalogue_version=catalogue_version,
                    ancestry_reference_version=ancestry_reference_version,
                )
            )
    return path


def read_catalogue(path: str | Path) -> list[dict[str, str]]:
    """Read a Catalogue TSV back as a list of raw string dict rows."""
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _fmt(value: float) -> str:
    if value != value:  # NaN
        return "nan"
    return f"{value:.6g}"
