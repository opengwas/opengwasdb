"""Estimate phenotype SD over a canonical ``analyses.tsv`` (issue #176).

`opengwasdb.build.phenotype_sd.estimate_phenotype_sd` is pure computation over
arrays (ADR 0029): it owns the formula and the method tiers, and nothing about
files. This module is the one seam that joins it to real sources, so a Store
Family does not have to re-open each source file and re-align alleles itself:
it reads a canonical `analyses.tsv` (ADR 0034), resolves a `SourceReader` per
row through `opengwasdb.readers.registry` (issue #115), streams each source
once for the se/af/beta evidence its chosen method needs, and writes the
estimates as a TSV whose column names are the shared-core `analyses.tsv`
spellings -- so a caller merges a column rather than translating a table.

The boundary is ADR 0029's computation-versus-acceptance split. This command
*computes* the number: it does not choose the method tier, the tolerance, or
whether a disagreement blocks a release. Those stay with the caller who wrote
`original_sd_method` into the manifest; the manifest column is read, never
inferred from whichever arrays happen to be present.
"""

from __future__ import annotations

import csv
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np

from opengwasdb.build.ordered_pool import ordered_map
from opengwasdb.build.phenotype_sd import ESTIMATION_METHODS, estimate_phenotype_sd
from opengwasdb.model.enums import OriginalSdMethod, StoredEffectScale
from opengwasdb.model.manifest_columns import require_columns, resolve_manifest_columns
from opengwasdb.readers.gwas_vcf import GWAS_VCF_CAPABILITY
from opengwasdb.readers.interface import ReaderAssociation, SourceReader
from opengwasdb.readers.registry import known_capabilities, resolve_reader
from opengwasdb.variants.normalise import orient_to_canonical

log = logging.getLogger(__name__)

__all__ = [
    "AfSource",
    "PhenotypeSdRow",
    "SdManifestRow",
    "estimate_manifest_phenotype_sd",
    "load_af_reference",
    "read_sd_manifest",
    "write_sd_estimates",
]

#: The columns the output TSV carries -- the shared-core `analyses.tsv`
#: spellings a caller merges back in (ADR 0034), plus the free-text `notes`
#: the estimator already produces.
SD_OUTPUT_COLUMNS: tuple[str, ...] = (
    "analysis_id",
    "original_sd",
    "original_sd_method",
    "original_sd_dispersion",
    "notes",
)


class AfSource(StrEnum):
    """Where the allele frequency in the ADR-0029 estimator comes from."""

    source = "source"
    reference = "reference"


@dataclass(frozen=True)
class SdManifestRow:
    """One Analysis's inputs, resolved from a manifest but not yet read."""

    analysis_id: str
    source_file: str
    source_reader_capability: str
    sample_size: float | None
    method: OriginalSdMethod


@dataclass(frozen=True)
class PhenotypeSdRow:
    """One Analysis's estimate, in the output TSV's own column terms.

    Values are strings because this is the row that gets written and merged:
    an unavailable estimate writes a blank `original_sd` and
    ``original_sd_method=unavailable``, never a fabricated ``0``.
    """

    analysis_id: str
    original_sd: str
    original_sd_method: str
    original_sd_dispersion: str
    notes: str

    def as_dict(self) -> dict[str, str]:
        return {
            "analysis_id": self.analysis_id,
            "original_sd": self.original_sd,
            "original_sd_method": self.original_sd_method,
            "original_sd_dispersion": self.original_sd_dispersion,
            "notes": self.notes,
        }


def read_sd_manifest(path: str | Path) -> list[SdManifestRow]:
    """Read a canonical ``analyses.tsv`` into per-Analysis estimation inputs.

    Column names resolve through the shared manifest alias resolver (issue
    #170/#172), so a legacy ``n``/``file_path`` manifest reads as well as a
    canonical one. ``source_reader_capability`` defaults to GWAS-VCF -- the
    format every manifest meant before the column existed -- and an unknown
    capability fails naming the ones this build knows. A malformed
    ``sample_size`` fails naming the Analysis and the value rather than being
    silently read as absent; a genuinely missing/blank one is `None`, which
    the estimator reports as ``unavailable``. A method tier this command does
    not compute is a caller error and fails loudly, naming both. A repeated
    ``analysis_id`` is refused rather than silently over-written and merged
    twice: `analysis_id` keys the output, so a duplicate is an ambiguous row.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        raw_rows = list(reader)
    require_columns(fieldnames, path, "original_sd_method")
    cols = resolve_manifest_columns(fieldnames, path)
    known = known_capabilities()

    rows: list[SdManifestRow] = []
    seen_ids: set[str] = set()
    for raw in raw_rows:
        analysis_id = raw[cols.analysis_id]
        if analysis_id in seen_ids:
            raise ValueError(
                f"analyses manifest {path} contains duplicate analysis_id: "
                f"{analysis_id!r}"
            )
        seen_ids.add(analysis_id)
        capability = (raw.get("source_reader_capability") or GWAS_VCF_CAPABILITY).strip()
        if capability not in known:
            raise ValueError(
                f"analyses manifest {path}: analysis {analysis_id!r} has unknown "
                f"source_reader_capability {capability!r}; known: {', '.join(known)}"
            )
        method_raw = (raw.get("original_sd_method") or "").strip()
        try:
            method = OriginalSdMethod(method_raw)
        except ValueError as exc:
            allowed = ", ".join(m.value for m in OriginalSdMethod)
            raise ValueError(
                f"analyses manifest {path}: analysis {analysis_id!r} has invalid "
                f"original_sd_method {method_raw!r}; expected one of {allowed}"
            ) from exc
        if method not in ESTIMATION_METHODS:
            computable = ", ".join(sorted(m.value for m in ESTIMATION_METHODS))
            raise ValueError(
                f"analyses manifest {path}: analysis {analysis_id!r} requests "
                f"original_sd_method={method_raw!r}, which this command does not "
                f"estimate (it computes {computable}); the tier is the caller's decision"
            )
        rows.append(
            SdManifestRow(
                analysis_id=analysis_id,
                source_file=raw[cols.source_file],
                source_reader_capability=capability,
                sample_size=_parse_sample_size(raw.get(cols.sample_size), analysis_id, path),
                method=method,
            )
        )
    if not rows:
        raise ValueError(f"analyses manifest {path} contains no rows")
    return rows


def _parse_sample_size(
    value: str | None, analysis_id: str, path: str | Path
) -> float | None:
    """A blank/missing sample size is `None`; a malformed one fails loudly.

    `None` is what `estimate_phenotype_sd` reads as ``unavailable``; a
    non-numeric string is a manifest defect, not a missing value, and is
    named rather than folded into the same outcome (matching the BESD overlay,
    issue #173).
    """
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"analyses manifest {path}: analysis {analysis_id!r} has invalid "
            f"sample_size {value!r}"
        ) from exc


def load_af_reference(
    path: str | Path, *, ancestry: str | None = None
) -> dict[str, float]:
    """Every A1-oriented ``{alid: eaf}`` a reference frequency source holds.

    Shares `opengwasdb.build.eaf_orientation.iter_eaf_reference` with the
    build-time EAF orientation check, so "how a reference frequency is read
    and oriented" has one implementation (issue #176).
    """
    from opengwasdb.build.eaf_orientation import iter_eaf_reference

    return dict(iter_eaf_reference(path, ancestry=ancestry))


@dataclass(frozen=True)
class _EstimationContext:
    """What every worker reads: where the ADR-0029 estimate's frequency comes from.

    ``ordered_map`` hands this to each forked worker by inheritance, so a
    reference frequency table is not pickled once per Analysis (issue #395).
    """

    af_source: AfSource
    af_reference: Mapping[str, float] | None


def _canonical_alid(association: ReaderAssociation) -> str:
    """The association's canonical ALID, through the shared orientation code."""
    return orient_to_canonical(
        association.chromosome,
        association.position,
        association.ref,
        association.alt,
    ).variant.alid


def _gather_evidence(
    reader: SourceReader,
    method: OriginalSdMethod,
    reference: Mapping[str, float] | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """``(se, af, beta)`` the chosen tier needs, from one source read.

    The source's own frequencies (``ReaderAssociation.eaf``, already oriented
    to the stored effect allele, ADR 0036) serve `estimated_from_source_maf`;
    the reference table serves `estimated_from_reference_maf`, keyed by the
    association's canonical ALID; `estimated_from_beta_distribution` needs
    only `beta = z * se`. Rows without the evidence a tier needs are left out
    of the arrays -- the estimator reports ``unavailable`` when too few
    survive, rather than a fabricated number.
    """
    if method is OriginalSdMethod.ESTIMATED_FROM_BETA_DISTRIBUTION:
        betas = [
            association.z * association.se
            for association in reader.stream_associations()
            if math.isfinite(association.z) and math.isfinite(association.se)
        ]
        return None, None, np.asarray(betas, dtype=np.float64)

    se_values: list[float] = []
    af_values: list[float] = []
    for association in reader.stream_associations():
        # A row with no effect is not an association, whatever its SE.
        if not (math.isfinite(association.se) and math.isfinite(association.z)):
            continue
        if method is OriginalSdMethod.ESTIMATED_FROM_SOURCE_MAF:
            frequency = association.eaf
        else:
            assert reference is not None, "reference AF required for this tier"
            frequency = reference.get(_canonical_alid(association))
        if frequency is None or not math.isfinite(frequency):
            continue
        se_values.append(association.se)
        af_values.append(frequency)
    return (
        np.asarray(se_values, dtype=np.float64),
        np.asarray(af_values, dtype=np.float64),
        None,
    )


def _check_af_source(row: SdManifestRow, af_source: AfSource) -> None:
    """Refuse to label reference-derived evidence with a source-MAF tier.

    `--af-source` chooses where the frequency comes from and the manifest's
    `original_sd_method` names the tier the caller wants applied; if the two
    disagree the estimate would be computed from one frequency source and
    labelled as another -- a plausible number recorded under the wrong method.
    """
    expected = {
        OriginalSdMethod.ESTIMATED_FROM_SOURCE_MAF: AfSource.source,
        OriginalSdMethod.ESTIMATED_FROM_REFERENCE_MAF: AfSource.reference,
    }.get(row.method)
    if expected is not None and expected is not af_source:
        raise ValueError(
            f"analysis {row.analysis_id!r} requests "
            f"original_sd_method={row.method.value!r}, which needs --af-source "
            f"{expected.value}, but --af-source {af_source.value} was given"
        )


def _estimate_one(
    context: _EstimationContext, row: SdManifestRow
) -> PhenotypeSdRow:
    """Worker: read one Analysis's source and estimate its phenotype SD."""
    _check_af_source(row, context.af_source)
    try:
        reader = resolve_reader(
            row.source_reader_capability, row.source_file, StoredEffectScale.SD
        )
        se, af, beta = _gather_evidence(reader, row.method, context.af_reference)
    except OSError as exc:
        raise ValueError(
            f"analysis {row.analysis_id!r} source {row.source_file!r} could not be read: {exc}"
        ) from exc
    try:
        estimate = estimate_phenotype_sd(
            row.method, row.sample_size, se=se, af=af, beta=beta
        )
    except ValueError as exc:
        raise ValueError(f"analysis {row.analysis_id!r}: {exc}") from exc
    return PhenotypeSdRow(
        analysis_id=row.analysis_id,
        original_sd=_format_float(estimate.sd),
        original_sd_method=estimate.method.value,
        original_sd_dispersion=_format_float(estimate.dispersion),
        notes=estimate.notes or "",
    )


def _format_float(value: float) -> str:
    """A blank cell for a non-finite value -- absence is blank, not ``nan``."""
    if not math.isfinite(value):
        return ""
    return f"{value:.10g}"


def estimate_manifest_phenotype_sd(
    rows: list[SdManifestRow],
    *,
    af_source: AfSource = AfSource.source,
    af_reference: Mapping[str, float] | None = None,
    n_workers: int = 1,
) -> list[PhenotypeSdRow]:
    """Estimate every row's phenotype SD, preserving input order.

    ``af_reference`` is required for :attr:`AfSource.reference` and must be the
    mapping `load_af_reference` returns. ``ordered_map`` hands the frequency
    source to forked workers by inheritance and yields in input order, so the
    output does not depend on worker count or completion order (spec §20.2,
    matching `opengwasdb.ancestry.pipeline.annotate_catalogue`).
    """
    if af_source is AfSource.reference and af_reference is None:
        raise ValueError(
            "--af-source reference requires a reference AF table; none was supplied"
        )
    context = _EstimationContext(af_source, af_reference)
    return list(ordered_map(_estimate_one, rows, n_workers, shared=context))


def write_sd_estimates(path: str | Path, rows: list[PhenotypeSdRow]) -> None:
    """Write the estimates TSV, keyed by `analysis_id` in input order."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(SD_OUTPUT_COLUMNS), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())
