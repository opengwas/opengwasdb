"""The Source Reader interface (issue #19; extended by issues #20 and #21).

`opengwasdb-stores` declares one `source_reader_capability` string per Source
Collection (ADR-0009, e.g. `"opengwasdb.gwas-vcf"`). This module defines the
interface that string resolves to (see `opengwasdb.readers.registry`): the
things the pipeline needs from source data -- streaming every variant
(issue #20, a builder's union-variant pass), streaming associations for the
build, and extracting allele frequency and standard error at a requested set
of sites for annotation (ancestry assignment, phenotype-SD estimation --
issue #21).

The interface is structural (`typing.Protocol`), not an ABC -- this package
has no abstract-base-class precedent elsewhere, and Protocol lets
`GwasVcfReader` and `FakeReader` satisfy it without a shared base class,
consistent with the codebase's existing preference for dataclasses and duck
typing over inheritance.

The dense and hybrid builders (issue #20) and ancestry assignment / phenotype-SD
estimation (issue #21) now resolve a reader through this interface instead of
importing a source module directly.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np

from opengwasdb.model.enums import StoredEffectScale


class ImputationScoreStatus(StrEnum):
    """A score's validity, independently of whether an association has an effect.

    `USABLE` and `OUT_OF_RANGE` both carry a finite number (stores #176): the
    filter compares either against a threshold, so `OUT_OF_RANGE` is a recorded
    fact about where the number sits, not a reason to drop the row. `UNDECLARED`,
    `MISSING`, `MALFORMED` and `NONFINITE` carry no number and are retained.
    """

    UNDECLARED = "undeclared"
    USABLE = "usable"
    MISSING = "missing"
    MALFORMED = "malformed"
    NONFINITE = "nonfinite"
    OUT_OF_RANGE = "out_of_range"


class ImputationScoreKind(StrEnum):
    IMPUTATION_INFO = "imputation_info"
    IMPUTATION_R2 = "imputation_r2"


@dataclass(frozen=True)
class ImputationScoreDeclaration:
    """Explicit provider-backed mapping; a header name alone is not provenance.

    The caller must supply independent evidence that `column_name` measures
    imputation INFO or R² for this Analysis. Neither EAF nor a familiar-looking
    header is evidence. Preserve this declaration in release metadata (#175).
    """

    column_name: str
    kind: ImputationScoreKind
    provenance: str

    def __post_init__(self) -> None:
        if not self.column_name or self.column_name != self.column_name.strip():
            raise ValueError("imputation score column name must be exact and nonempty")
        if not isinstance(self.kind, ImputationScoreKind):
            raise ValueError("imputation score kind must be imputation_info or imputation_r2")
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("imputation score requires independent provider provenance")


@dataclass(frozen=True)
class ImputationScore:
    value: float | None = None
    status: ImputationScoreStatus = ImputationScoreStatus.UNDECLARED


def parse_imputation_score(value: str | None, *, declared: bool) -> ImputationScore:
    """Parse a declared score, without substituting a default (stores #176).

    Any finite number is a usable score, whatever its value: a score above 1 is
    usable and passes any threshold <= 1, and a negative score is usable and
    falls below any positive threshold. `OUT_OF_RANGE` therefore records the
    fact that a finite score lies outside [0, 1] -- nothing more; it is not a
    disposition that drops the row.
    """
    if not declared:
        return ImputationScore()
    if value is None or value.strip() in ("", ".", "NA", "NaN", "nan", "None"):
        return ImputationScore(status=ImputationScoreStatus.MISSING)
    try:
        number = float(value)
    except ValueError:
        return ImputationScore(status=ImputationScoreStatus.MALFORMED)
    if not np.isfinite(number):
        return ImputationScore(status=ImputationScoreStatus.NONFINITE)
    if 0.0 <= number <= 1.0:
        return ImputationScore(number, ImputationScoreStatus.USABLE)
    return ImputationScore(number, ImputationScoreStatus.OUT_OF_RANGE)


@dataclass(frozen=True)
class ReaderAssociation:
    """One source association's position, effect, and precision.

    `ref`/`alt` are the source's own allele labelling, not reordered to
    canonical A1/A2 -- `z` is already sign-corrected to the A1 = min(ref, alt)
    convention every reader in this package follows. `eaf` follows `z`: it is
    the frequency of the *stored* effect allele, so a reader that negated `z`
    also stores `1 - af` (ADR 0036). None where the source reports no usable
    frequency -- never fabricated, and never a substitute 0.5. There is no
    `analysis_id`: a source file may cover one Analysis (GWAS-VCF) or many (a
    multi-analysis tabular file), so identity assignment stays the caller's
    responsibility. `stored_effect_scale` is likewise never derived from the
    source file itself (issue #17): it is Analytical Metadata a reader
    receives from its caller (ultimately the build manifest, validated
    against issue #16's schema) and attaches to every association it yields.
    """

    chromosome: str
    position: int
    ref: str
    alt: str
    z: float
    se: float
    stored_effect_scale: StoredEffectScale
    eaf: float | None = None
    # Present only with an independent provider-backed declaration (#175).
    imputation_score: ImputationScore = ImputationScore()

    def __post_init__(self) -> None:
        if self.se < 0:
            raise ValueError(f"se must be non-negative, got {self.se!r}")
        if self.eaf is not None and not 0.0 <= self.eaf <= 1.0:
            raise ValueError(f"eaf must be in [0, 1], got {self.eaf!r}")


@dataclass(frozen=True)
class SourceVariant:
    """One variant a source names, with the source's own identifier for it.

    `chromosome`/`position`/`ref`/`alt` are the source's own labelling, same
    as `ReaderAssociation`. `rsid` is whatever identifier the source records
    for the row -- blank when it records none, never fabricated.

    A dataclass rather than the bare 4-tuple this used to be (issue #109):
    Dense and Hybrid builds discarded rsids entirely because the variant
    stream had nowhere to carry them, so every rsid lookup against those
    stores returned an empty result indistinguishable from "no association."
    Dedup by `site`, not by the whole record -- two rows for one variant may
    disagree on the identifier without being two different variants.
    """

    chromosome: str
    position: int
    ref: str
    alt: str
    rsid: str = ""

    @property
    def site(self) -> tuple[str, int, str, str]:
        """The variant's identity, independent of what the source calls it."""
        return (self.chromosome, self.position, self.ref, self.alt)


@dataclass(frozen=True)
class SiteMetrics:
    """A1-oriented allele frequency and standard error at one canonical site."""

    af: float
    se: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.af <= 1.0:
            raise ValueError(f"af must be in [0, 1], got {self.af!r}")
        if self.se < 0:
            raise ValueError(f"se must be non-negative, got {self.se!r}")


class SourceReader(Protocol):
    """One Source Format's reader: association streaming plus per-site AF/SE
    lookup for annotation."""

    def stream_associations(self) -> Iterator[ReaderAssociation]:
        """Yield every association in the source, oriented per ReaderAssociation."""
        ...

    def stream_variants(self) -> Iterator[SourceVariant]:
        """Yield a `SourceVariant` for every biallelic variant in the source,
        independent of whether it carries a usable association.

        A superset of what :meth:`stream_associations` yields positions for --
        a record dropped there for an invalid/missing effect size or SE may
        still belong on a builder's union-variant axis (issue #20), so callers
        needing full variant coverage use this rather than filtering
        :meth:`stream_associations` themselves.
        """
        ...

    def extract_at_sites(self, alids: Iterable[str]) -> dict[str, SiteMetrics]:
        """Return `{canonical_alid: SiteMetrics}` for the requested sites found
        in the source. A requested alid absent from the result was not found
        or was dropped (palindromic, unparseable AF/SE, ...) -- callers must
        not assume every requested alid comes back.
        """
        ...


def site_metrics_arrays(sites: dict[str, SiteMetrics]) -> tuple[np.ndarray, np.ndarray]:
    """``(se, af)`` arrays from an :meth:`SourceReader.extract_at_sites` result.

    The shape :func:`opengwasdb.build.phenotype_sd.estimate_phenotype_sd` needs
    (issue #21) -- the one adapter between the reader interface and the
    estimator, so ancestry assignment (`{alid: af}`) and SD estimation
    (`se`/`af` arrays) share the same underlying extraction call rather than
    each reshaping it independently.
    """
    se = np.fromiter((m.se for m in sites.values()), dtype=np.float64, count=len(sites))
    af = np.fromiter((m.af for m in sites.values()), dtype=np.float64, count=len(sites))
    return se, af


def af_only(sites: dict[str, SiteMetrics]) -> dict[str, float]:
    """``{alid: af}`` from an :meth:`SourceReader.extract_at_sites` result --
    the shape :func:`opengwasdb.ancestry.mixture.assign_ancestry` needs.
    Ancestry assignment has no use for `se`, but the extraction it drives is
    the same combined AF+SE call SD estimation uses (issue #21)."""
    return {alid: metrics.af for alid, metrics in sites.items()}
