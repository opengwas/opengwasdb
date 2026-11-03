"""Tiny tabular source reader and normalised association records."""

from __future__ import annotations

import csv
import gzip
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.variants import CanonicalVariant, VariantNormalisationError, orient_to_canonical


class SourceRowError(ValueError):
    """Raised when a source row cannot be converted to an association record."""


@dataclass(frozen=True)
class NormalisedAssociation:
    """One source association oriented to canonical Store ALID convention."""

    analysis_id: str
    variant: CanonicalVariant
    z: float
    se: float
    rsid: str | None = None
    analysis_label: str | None = None
    trait_ontology_id: str | None = None
    trait_ontology_label: str | None = None
    license: str | None = None
    publication_doi: str | None = None
    publication_pmid: str | None = None
    consortium: str | None = None
    first_author: str | None = None
    stored_effect_scale: StoredEffectScale = StoredEffectScale.SD
    eaf: float | None = None


def read_normalised_associations(path: str | Path) -> list[NormalisedAssociation]:
    """Read a tiny TSV/CSV source file into normalised association records.

    Required logical fields are analysis id, chromosome, position, effect allele,
    other allele, SE, and either Z or beta. Column aliases cover the fixture
    style used in tests and common GWAS-SSF-ish names.
    """

    source_path = Path(path)
    delimiter = "," if _logical_suffix(source_path) == ".csv" else "\t"
    opener = gzip.open if source_path.suffix.lower() == ".gz" else Path.open
    with opener(source_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise SourceRowError(f"{source_path} has no header")
        return [_normalise_row(row, row_number=i + 2) for i, row in enumerate(reader)]


def stream_normalised_associations(paths: Iterable[str | Path]) -> list[NormalisedAssociation]:
    """Read one or more source files preserving row order within each file."""

    records: list[NormalisedAssociation] = []
    for path in paths:
        records.extend(read_normalised_associations(path))
    return records


def _normalise_row(row: dict[str, str], row_number: int) -> NormalisedAssociation:
    lookup = {key.lower(): value for key, value in row.items() if key is not None}

    def field(*names: str, required: bool = True) -> str | None:
        for name in names:
            value = lookup.get(name.lower())
            if value is not None and str(value).strip() not in {"", ".", "NA", "NaN", "nan"}:
                return str(value).strip()
        if required:
            raise SourceRowError(f"row {row_number}: missing required field {names[0]!r}")
        return None

    analysis_id = field("analysis_id", "study_id", "trait_id")
    assert analysis_id is not None

    try:
        chromosome = field("chromosome", "chrom", "chr")
        position = field("position", "pos", "base_pair_location")
        effect_allele = field("effect_allele", "ea", "a1")
        other_allele = field("other_allele", "oa", "a2", "non_effect_allele")
        assert chromosome is not None
        assert position is not None
        assert effect_allele is not None
        assert other_allele is not None
        orientation = orient_to_canonical(
            chromosome=chromosome,
            position=position,
            source_effect_allele=effect_allele,
            source_other_allele=other_allele,
        )
    except VariantNormalisationError as exc:
        raise SourceRowError(f"row {row_number}: {exc}") from exc

    # A missing, non-positive or non-finite SE is the source's: Association
    # Admission drops and logs the row, and its z is NaN when it depends on it.
    se_text = field("se", "standard_error", required=False)
    se = math.nan if se_text is None else _parse_float(se_text, row_number, "se")
    usable_se = math.isfinite(se) and se > 0

    z_text = field("z", "zscore", "z_score", required=False)
    if z_text is None:
        beta_text = field("beta", "effect", "effect_size")
        assert beta_text is not None
        beta = _parse_float(beta_text, row_number, "beta")
        z = beta / se if usable_se else math.nan
    else:
        z = _parse_float(z_text, row_number, "z")
    if usable_se and not math.isfinite(z):
        raise SourceRowError(f"row {row_number}: z must be finite")

    eaf_text = field("eaf", "effect_allele_frequency", required=False)
    eaf = _parse_float(eaf_text, row_number, "eaf") if eaf_text is not None else None
    if eaf is not None and orientation.flipped:
        eaf = 1.0 - eaf

    if orientation.flipped:
        z = -z

    scale_text = field("stored_effect_scale", "effect_scale", required=False)
    try:
        scale = StoredEffectScale(scale_text) if scale_text else StoredEffectScale.SD
    except ValueError as exc:
        message = f"row {row_number}: invalid stored_effect_scale {scale_text!r}"
        raise SourceRowError(message) from exc

    return NormalisedAssociation(
        analysis_id=analysis_id,
        variant=orientation.variant,
        z=z,
        se=se,
        rsid=field("rsid", "variant_id", required=False),
        analysis_label=field(
            "analysis_label", "phenotype_label", "trait_label", "trait", required=False
        ),
        trait_ontology_id=field("trait_ontology_id", required=False),
        trait_ontology_label=field("trait_ontology_label", "trait_ontology_name", required=False),
        license=field("license", required=False),
        publication_doi=field("publication_doi", required=False),
        publication_pmid=field("publication_pmid", required=False),
        consortium=field("consortium", required=False),
        first_author=field("first_author", required=False),
        stored_effect_scale=scale,
        eaf=eaf,
    )


def _parse_float(value: str | None, row_number: int, field_name: str) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise SourceRowError(f"row {row_number}: invalid {field_name} {value!r}") from exc


def _logical_suffix(path: Path) -> str:
    suffixes = path.suffixes
    if suffixes and suffixes[-1].lower() == ".gz" and len(suffixes) > 1:
        return suffixes[-2].lower()
    return path.suffix.lower()
