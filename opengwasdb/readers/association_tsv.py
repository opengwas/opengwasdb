"""The association TSV Source Reader: one Analysis's associations in a small
delimited file, the format `build-dense` reads.

A file holds associations only. Which Analysis it is, and what is declared of
it (label, Trait Ontology, licence, publication, MAF and INFO policy, effect
scale), is the build manifest's: one row per Analysis, one file per row, as for
every other Source Format. A header that names such a column is refused, since a
reader that went on would either drop the declaration or read several Analyses
as one.

The columns, by name (case-insensitive; the first alias present wins):

- chromosome: `chromosome`, `chrom`, `chr`;
- position: `position`, `pos`, `base_pair_location`;
- effect allele: `effect_allele`, `ea`, `a1`;
- other allele: `other_allele`, `oa`, `a2`, `non_effect_allele`;
- standard error: `se`, `standard_error`;
- effect: `z`, `zscore`, `z_score`, else `beta`, `effect`, `effect_size`;
- optional: `eaf` or `effect_allele_frequency`; `rsid` or `variant_id`.

A `.csv` (or `.csv.gz`) file is comma-delimited, any other tab-delimited. The
format names no reference allele: its effect and other alleles need not be the
genome's REF and ALT (`interface.ReferenceAllele.UNDECLARED`). A cell of `.`, `NA`, `NaN`,
`nan` or nothing is missing.

A row it cannot read refuses the whole read (`SourceRowError`): a missing
required field, or a number that is not one. A row whose identity no ALID can
spell (a missing chromosome, an allele outside A, C, G and T, or an indel,
since the format declares no reference allele to spell first, ADR 0079) is
not an unreadable row but a malformed one (#365): with a ``tally`` it is left
out and named there, for Association Admission to log. Without one an indel
is left out, as every reader's projections leave it, and any other malformed
row is refused.
"""

from __future__ import annotations

import csv
import gzip
import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import IO

from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.readers import interface
from opengwasdb.readers.rows import MetricsChunk, ReaderBatch, SourceRow, stream_batches
from opengwasdb.readers.tabular import (
    extract_at_sites,
    metrics_chunks_from_rows,
    parse_positive_float,
    stream_associations,
    stream_variants,
)
from opengwasdb.stats import parse_af
from opengwasdb.variants import VariantNormalisationError, orient_to_canonical
from opengwasdb.variants.normalise import (
    MalformedIdentity,
    Orientation,
    UndeclaredReference,
    identity_fault,
)

__all__ = ["ASSOCIATION_TSV_CAPABILITY", "AssociationTsvReader", "SourceRowError"]

ASSOCIATION_TSV_CAPABILITY = "opengwasdb.association-tsv"

#: The format names no reference allele.
_REFERENCE = interface.ReferenceAllele.UNDECLARED

#: The sites an extraction asks for.
_Sites = Iterable[str]

_MISSING = frozenset({"", ".", "NA", "NaN", "nan"})

_CHROMOSOME = ("chromosome", "chrom", "chr")
_POSITION = ("position", "pos", "base_pair_location")
_EFFECT_ALLELE = ("effect_allele", "ea", "a1")
_OTHER_ALLELE = ("other_allele", "oa", "a2", "non_effect_allele")

#: Columns that declare something of an Analysis. They belong to the manifest.
_ANALYSIS_COLUMNS = frozenset(
    {
        "analysis_id", "study_id", "trait_id",
        "analysis_label", "phenotype_label", "trait_label", "trait",
        "trait_ontology_id", "trait_ontology_label", "trait_ontology_name",
        "license", "publication_doi", "publication_pmid", "consortium", "first_author",
        "stored_effect_scale", "effect_scale",
        "maf_threshold", "info_score_threshold",
    }
)


class SourceRowError(ValueError):
    """Raised when a source row cannot be read as an association."""


def _logical_suffix(path: Path) -> str:
    suffixes = path.suffixes
    if suffixes and suffixes[-1].lower() == ".gz" and len(suffixes) > 1:
        return suffixes[-2].lower()
    return path.suffix.lower()


def _open(path: Path) -> IO[str]:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open(encoding="utf-8", newline="")


def _refuse_analysis_columns(path: Path, fieldnames: Iterable[str | None]) -> None:
    declared = sorted({name.lower() for name in fieldnames if name} & _ANALYSIS_COLUMNS)
    if declared:
        raise SourceRowError(
            f"{path} declares {', '.join(declared)}: an association TSV holds associations "
            "only, and the Analysis's own declarations belong in the build manifest, one "
            "file per Analysis"
        )


def _iter_rows(path: str | Path, tally: interface.StreamTally | None) -> Iterator[SourceRow]:
    """Every readable row, in file order; see the module docstring for the refusals."""
    source = Path(path)
    delimiter = "," if _logical_suffix(source) == ".csv" else "\t"
    with _open(source) as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise SourceRowError(f"{source} has no header")
        _refuse_analysis_columns(source, reader.fieldnames)
        for i, row in enumerate(reader):
            parsed = _read_row(_Cells(row, i + 2), tally)
            if parsed is not None:
                yield parsed


class _Cells:
    """One row's cells by lower-cased column name, with the row's number for refusals."""

    def __init__(self, row: Mapping[str | None, str | None], number: int) -> None:
        self.number = number
        self._cells = {key.lower(): value for key, value in row.items() if key is not None}

    def get(self, *names: str, required: bool = True) -> str | None:
        """The first of ``names`` the row holds a value for, else ``None`` or a refusal."""
        for name in names:
            value = self._cells.get(name)
            if value is not None and str(value).strip() not in _MISSING:
                return str(value).strip()
        if required:
            raise SourceRowError(f"row {self.number}: missing required field {names[0]!r}")
        return None

    def required(self, *names: str) -> str:
        value = self.get(*names)
        assert value is not None
        return value

    def number_of(self, text: str, field_name: str) -> float:
        try:
            return float(text)
        except ValueError as exc:
            raise SourceRowError(f"row {self.number}: invalid {field_name} {text!r}") from exc

    def raw(self, names: tuple[str, ...]) -> str:
        """The cell as the source spelt it, missing-value tokens and all."""
        cells = (self._cells.get(name) for name in names)
        return next((str(cell).strip() for cell in cells if cell is not None), "")


def _read_row(cells: _Cells, tally: interface.StreamTally | None) -> SourceRow | None:
    fault = _identity_fault(cells)
    if fault is not None:
        key, malformed = fault
        if tally is not None:
            tally.faults.record(key, malformed)
            return None
        # An indel is a well-formed row the format cannot place (ADR 0079): a
        # projection with no tally leaves it out, as every reader's does.
        if isinstance(malformed, UndeclaredReference):
            return None
        raise SourceRowError(f"row {cells.number}: {malformed}") from malformed

    orientation, effect_allele, other_allele = _orientation(cells)
    se, z, beta = _effect(cells)
    eaf = None
    eaf_text = cells.get("eaf", "effect_allele_frequency", required=False)
    if eaf_text is not None:
        reported = cells.number_of(eaf_text, "eaf")
        eaf = parse_af(reported, None if tally is None else tally.frequencies)
    return SourceRow(
        chromosome=orientation.variant.chromosome,
        position=orientation.variant.position,
        ref=other_allele,
        alt=effect_allele,
        alid=orientation.variant.alid,
        flipped=orientation.flipped,
        af_alt=eaf,
        beta=beta,
        se=se,
        rsid=cells.get("rsid", "variant_id", required=False) or "",
        z=z,
    )


def _orientation(cells: _Cells) -> tuple[Orientation, str, str]:
    """The row's variant on canonical A1, and its alleles as the source spelt them."""
    chromosome = cells.required(*_CHROMOSOME)
    position = cells.required(*_POSITION)
    effect_allele = cells.required(*_EFFECT_ALLELE)
    other_allele = cells.required(*_OTHER_ALLELE)
    try:
        orientation = orient_to_canonical(
            chromosome=chromosome,
            position=position,
            source_effect_allele=effect_allele,
            source_other_allele=other_allele,
            other_is_reference=False,
        )
    except VariantNormalisationError as exc:
        raise SourceRowError(f"row {cells.number}: {exc}") from exc
    return orientation, effect_allele, other_allele


def _effect(cells: _Cells) -> tuple[float, float, float | None]:
    """The row's ``(se, z, beta)`` on its own effect allele.

    A missing, non-positive or non-finite SE is the source's: Association
    Admission drops and logs the row, and its z is NaN when it depends on it. A
    non-finite z is the source's too.
    """
    se_text = cells.get("se", "standard_error", required=False)
    se = math.nan if se_text is None else cells.number_of(se_text, "se")
    z_text = cells.get("z", "zscore", "z_score", required=False)
    if z_text is not None:
        return se, cells.number_of(z_text, "z"), None
    beta = cells.number_of(cells.required("beta", "effect", "effect_size"), "beta")
    return se, beta / se if math.isfinite(se) and se > 0 else math.nan, beta


def _identity_fault(cells: _Cells) -> tuple[str, MalformedIdentity] | None:
    """The row's malformed identity and the key its log names it by, or ``None``.

    Judged on the cells as the source spelt them, before any other field is
    required of the row.
    """
    chromosome, position = cells.raw(_CHROMOSOME), cells.raw(_POSITION)
    effect, other = cells.raw(_EFFECT_ALLELE), cells.raw(_OTHER_ALLELE)
    fault = identity_fault(chromosome, effect, other, other_is_reference=False)
    if fault is None:
        return None
    return f"{chromosome}:{position}:{other}:{effect}", fault


def _usable_se(row: SourceRow) -> SourceRow:
    """``row`` with an unusable SE absent, as every metrics projection reads it."""
    return replace(row, se=parse_positive_float(str(row.se)))


@dataclass(frozen=True)
class AssociationTsvReader:
    """Reader for one Analysis's association TSV."""

    path: str | Path
    stored_effect_scale: StoredEffectScale

    def _rows(self, tally: interface.StreamTally | None = None) -> Iterator[SourceRow]:
        return _iter_rows(self.path, tally)

    def stream_associations(
        self, tally: interface.StreamTally | None = None
    ) -> Iterator[interface.ReaderAssociation]:
        return stream_associations(self._rows(tally), self.stored_effect_scale, _REFERENCE)

    def stream_batches(self, tally: interface.StreamTally | None = None) -> Iterator[ReaderBatch]:
        return stream_batches(self._rows(tally), _REFERENCE)

    def stream_variants(self) -> Iterator[interface.SourceVariant]:
        return stream_variants(self._rows())

    def stream_metric_chunks(self) -> Iterator[MetricsChunk]:
        return metrics_chunks_from_rows(_usable_se(row) for row in self._rows())

    def extract_at_sites(
        self,
        alids: _Sites,
        *,
        region: str | None = None,
        regions_file: interface.RegionsFile = None,
    ) -> dict[str, interface.SiteMetrics]:
        return extract_at_sites(
            (_usable_se(row) for row in self._rows()), alids, region=region
        )
