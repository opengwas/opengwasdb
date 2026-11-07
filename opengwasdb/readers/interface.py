"""The Source Reader interface (issues #19, #20, #21, #56).

`opengwasdb-stores` declares one `source_reader_capability` string per Source
Collection (ADR-0009, e.g. `"opengwasdb.gwas-vcf"`). This module defines the
interface that string resolves to (see `opengwasdb.readers.registry`): every
projection of a source the pipeline reads, each in the source's declared
assembly unless the reader was opened aligned to another:

- `stream_variants`, every variant, for a builder's union-variant pass;
- `stream_associations`, one association per row, for Variant Routing's Pass 2;
- `stream_batches`, the same associations as columns, for a Ragged build;
- `stream_metric_chunks`, identity and statistics in blocks, for the resolver;
- `extract_at_sites`, A1 frequency and SE at requested sites, for ancestry
  assignment and phenotype-SD estimation.

Every reader presents them from its rows through `opengwasdb.readers.rows`, so
each is oriented to A1 by one rule. The interface is structural
(`typing.Protocol`), not an ABC: every reader and `FakeReader` satisfy it
without a shared base class.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.stats import FrequencyTally
from opengwasdb.variants.normalise import IdentityFaults

if TYPE_CHECKING:
    from opengwasdb.readers.rows import MetricsChunk, ReaderBatch

#: A ``CHROM POS`` file of the sites an extraction asks for, or none.
RegionsFile = str | Path | None


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
    if value is None or value.strip() in ("", ".", "NA", "#NA", "NaN", "nan", "None"):
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


class ReferenceAllele(StrEnum):
    """Which of a row's two alleles its source declares to be the genome's reference.

    Declared by the format, never inferred: GWAS-VCF's `REF` and FinnGen's
    `ref` are the reference allele (`REF`, the row's own `ref`), while
    GWAS-SSF, BESD and an effect/other Column Map name an effect allele and an
    other one, which need not be the reference (`UNDECLARED`). `ALT` is a row
    whose `alt` its source declares the reference; no reader declares one, so
    it is unreachable, and #316 removes it with the reversed-indel rule.

    An indel's ALID is formed from the declaration when the row is read (ADR
    0079), and one whose source declares none is dropped there, so every
    indel a reader yields has its reference in `ref`. Variant Routing still
    carries the declaration as each row's reference code, which #357's
    reversed-indel rule reads.
    """

    UNDECLARED = "undeclared"
    REF = "ref"
    ALT = "alt"

    @property
    def declared(self) -> bool:
        """Whether the format names either allele the genome's reference."""
        return self is not ReferenceAllele.UNDECLARED


@dataclass(frozen=True)
class ReaderAssociation:
    """One source association's position, effect, and precision.

    `ref`/`alt` are the source's own allele labelling, case included, not
    reordered to canonical A1/A2 -- `z` is already sign-corrected to the ALID's
    A1 every reader in this package forms (`alid_alleles`: the two sorted, or an
    indel's reference allele, which is then `ref`), the two compared upper-cased
    (`opengwasdb.variants.normalise.identity_allele`, spec §4: a lower-case base
    is the same base). `z` is NaN for a source row that
    reports no effect: such a row is not an association, and Association
    Admission drops and logs it (`effect_missing`). `eaf` follows `z`: it is
    the frequency of the *stored* effect allele, so a reader that negated `z`
    also stores `1 - af` (ADR 0036). None where the source reports no usable
    frequency -- never fabricated, and never a substitute 0.5. There is no
    `analysis_id`: a source file may cover one Analysis (GWAS-VCF) or many (a
    multi-analysis tabular file), so identity assignment stays the caller's
    responsibility. `stored_effect_scale` is likewise never derived from the
    source file itself (issue #17): it is Analytical Metadata a reader
    receives from its caller (ultimately the build manifest, validated
    against issue #16's schema) and attaches to every association it yields.
    `reference` says which of `ref` and `alt` the source declares to be the
    reference allele, if it declares one. It is keyword-only and has no
    default: every reader states its format's declaration.
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
    reference: ReferenceAllele = field(kw_only=True)

    def __post_init__(self) -> None:
        # An unusable `se` (NaN, not positive, not finite) is the source's, as
        # read: Association Admission drops and logs the row.
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


@dataclass
class StreamTally:
    """What one Source Reader stream did to the rows it read, for Association
    Admission to record (issue #357): the rows it dropped because no ALID can
    spell their identity (#365), the reported frequencies it made absent (D7),
    and the indels a GWAS-SSF harmoniser code placed (#381)."""

    faults: IdentityFaults = field(default_factory=IdentityFaults)
    frequencies: FrequencyTally = field(default_factory=FrequencyTally)
    indel_rows_kept_by_hm_code: int = 0


class SourceReader(Protocol):
    """One Source Format's reader: every projection of one source."""

    def stream_associations(self, tally: StreamTally | None = None) -> Iterator[ReaderAssociation]:
        """Yield every association in the source, oriented per ReaderAssociation.

        A row whose identity is malformed (a missing chromosome, an allele
        outside A, C, G and T) is not yielded. ``tally``, given, names each
        such row, and counts the reported frequencies the stream made absent
        because they lie outside (0, 1) (issue #357: D7, #365).
        """
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

    def stream_batches(self, tally: StreamTally | None = None) -> Iterator[ReaderBatch]:
        """`stream_associations`' rows as columns, in blocks, in the same order.

        The same z, SE and frequency on A1, with each row's ALID, its alleles
        as the source spells them, its place in the source and its rsid;
        ``tally`` as `stream_associations` fills it.
        """
        ...

    def stream_metric_chunks(self) -> Iterator[MetricsChunk]:
        """Every row naming a variant, with its statistics, a block at a time.

        The resolver's projection: a row is kept with whatever statistics it
        has, so ancestry assignment and phenotype-SD estimation each apply
        their own eligibility rule.
        """
        ...

    def extract_at_sites(
        self, alids: Iterable[str], *, region: str | None = None, regions_file: RegionsFile = None
    ) -> dict[str, SiteMetrics]:
        """Return `{canonical_alid: SiteMetrics}` for the requested sites found
        in the source. A requested alid absent from the result was not found
        or was dropped (palindromic, unparseable AF/SE, ...) -- callers must
        not assume every requested alid comes back.

        ``region`` restricts the extraction to one chromosome of the source, as
        the source names it. ``regions_file`` is a ``CHROM POS`` file of the
        requested sites' positions, built once by a caller extracting from many
        sources: a reader that can seek by position reads only those, and any
        other reads the whole source, with the same answer.
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
