"""Dense fixtures from the rows tests already define.

A test describes its associations as one TSV whose rows each name their Analysis
and repeat its metadata. `build-dense` reads a build manifest and one
associations-only TSV per Analysis, so this module splits such a TSV:
`write_dense_manifest` writes the manifest and the per-Analysis files beside it,
and `build_dense_fixture` builds a Store from them.

The manifest lists Analyses in `analysis_id` order, the order a Store's columns
took when `build-dense` read the combined TSV.
"""

from __future__ import annotations

import csv
import gzip
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from opengwasdb.admission.rules import AnalysisRules
from opengwasdb.build.observed import build_dense_observed_from_manifest
from opengwasdb.layouts.dense import (
    AnalysisSource,
    DenseBuildResult,
    build_dense_observed_store,
)
from opengwasdb.layouts.dense.build import NormalisedAssociation
from opengwasdb.model.analyses import Analysis
from opengwasdb.readers.interface import ImputationScoreStatus, ReferenceAllele, StreamTally
from opengwasdb.readers.rows import ReaderBatch

#: Each Analysis-level column the combined TSV may carry, by the manifest column
#: it becomes. Aliases are listed in the order the combined reader tried them.
_DECLARED = {
    "analysis_label": ("analysis_label", "phenotype_label", "trait_label", "trait"),
    "trait_ontology_id": ("trait_ontology_id",),
    "trait_ontology_label": ("trait_ontology_label", "trait_ontology_name"),
    "license": ("license",),
    "publication_doi": ("publication_doi",),
    "publication_pmid": ("publication_pmid",),
    "consortium": ("consortium",),
    "first_author": ("first_author",),
    "stored_effect_scale": ("stored_effect_scale", "effect_scale"),
    "maf_threshold": ("maf_threshold",),
    "info_score_threshold": ("info_score_threshold",),
}
_ANALYSIS_ID = ("analysis_id", "study_id", "trait_id")
_MISSING = {"", ".", "NA", "NaN", "nan"}
_ANALYSIS_COLUMNS = {name for names in _DECLARED.values() for name in names} | set(_ANALYSIS_ID)


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    gz = path.suffix.lower() == ".gz"
    logical = path.with_suffix("").suffix.lower() if gz else path.suffix.lower()
    with (gzip.open(path, "rt", encoding="utf-8", newline="") if gz
          else path.open(encoding="utf-8", newline="")) as handle:
        reader = csv.DictReader(handle, delimiter="," if logical == ".csv" else "\t")
        return list(reader.fieldnames or []), list(reader)


def _declared(row: dict[str, str], names: tuple[str, ...]) -> str:
    lookup = {key.lower(): value for key, value in row.items() if key is not None}
    for name in names:
        value = (lookup.get(name) or "").strip()
        if value not in _MISSING:
            return value
    return ""


def write_dense_manifest(*sources: Path) -> Path:
    """The manifest of the Analyses ``sources`` hold, beside the first of them.

    Each Analysis's declarations are those of its first row, as the combined
    reader took them; its file holds only its own rows, with the columns that
    declared something left out.
    """
    first = sources[0]
    directory = first.parent / f"{first.name}.dense"
    directory.mkdir(exist_ok=True)
    by_analysis: dict[str, list[dict[str, str]]] = {}
    header: list[str] = []
    for source in sources:
        fieldnames, rows = _read(source)
        header = [name for name in fieldnames if name.lower() not in _ANALYSIS_COLUMNS]
        for row in rows:
            by_analysis.setdefault(_declared(row, _ANALYSIS_ID), []).append(row)
    manifest = directory / "manifest.tsv"
    columns = ["analysis_id", "source_file", "original_sd_method", *_DECLARED]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, columns, delimiter="\t")
        writer.writeheader()
        for analysis_id in sorted(by_analysis):
            rows = by_analysis[analysis_id]
            path = directory / f"{analysis_id}.tsv"
            with path.open("w", encoding="utf-8", newline="") as out:
                associations = csv.DictWriter(
                    out, header, delimiter="\t", extrasaction="ignore", lineterminator="\n"
                )
                associations.writeheader()
                associations.writerows(rows)
            declared = {key: _declared(rows[0], names) for key, names in _DECLARED.items()}
            writer.writerow(
                {
                    "analysis_id": analysis_id,
                    "source_file": str(path),
                    "original_sd_method": "declared_standardised",
                    **declared,
                    "stored_effect_scale": declared["stored_effect_scale"] or "sd",
                    "maf_threshold": declared["maf_threshold"] or "NaN",
                    "info_score_threshold": declared["info_score_threshold"] or "NaN",
                }
            )
    return manifest


def build_dense_fixture(
    sources: Sequence[Path], output_path: Path, **options: Any
) -> DenseBuildResult:
    """A Dense Store of the Analyses ``sources`` hold, built as `build-dense` builds one."""
    return build_dense_observed_from_manifest(
        write_dense_manifest(*sources), output_path, **options
    )


def build_dense_from_records(
    records: Sequence[NormalisedAssociation], output_path: Path, **options: Any
) -> DenseBuildResult:
    """A Dense Store of associations a test holds in memory, one column per Analysis.

    The columns are in `analysis_id` order. Each Analysis declares nothing beyond
    its id and the SD effect scale, and each record is already on its canonical A1.
    """
    by_analysis: dict[str, list[NormalisedAssociation]] = {}
    for record in records:
        by_analysis.setdefault(record.analysis_id, []).append(record)
    sources = [
        AnalysisSource(
            analysis=Analysis(analysis_id=analysis_id, stored_effect_scale="sd"),
            rules=AnalysisRules(analysis_id),
            se_divisor=1.0,
            batches=[_batch(rows)],
            tally=StreamTally(),
        )
        for analysis_id, rows in sorted(by_analysis.items())
    ]
    return build_dense_observed_store(sources, output_path, **options)


def _batch(rows: Sequence[NormalisedAssociation]) -> ReaderBatch:
    n = len(rows)
    return ReaderBatch(
        alid=np.array([r.variant.alid for r in rows], dtype=object),
        source_row=np.arange(n, dtype=np.int64),
        flipped=np.zeros(n, dtype=bool),
        z=np.array([r.z for r in rows], dtype=np.float64),
        se=np.array([r.se for r in rows], dtype=np.float64),
        eaf=np.array([np.nan if r.eaf is None else r.eaf for r in rows], dtype=np.float64),
        imputation_score=np.full(n, np.nan),
        imputation_status=np.full(n, ImputationScoreStatus.UNDECLARED, dtype=object),
        rsid=np.array([r.rsid or "" for r in rows], dtype=object),
        reference=ReferenceAllele.UNDECLARED,
    )

