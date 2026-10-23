"""Controlled vocabularies for the store contract."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType


class PrimaryStorageLayout(StrEnum):
    DENSE = "dense"
    RAGGED = "ragged"
    HYBRID = "hybrid"


class AssociationCoverage(StrEnum):
    """Which of a source Analysis's associations a Store Release retains.

    A property of the Store Release, not of its layout (ADR 0009): two Ragged
    releases of the same Source Collection can hold different amounts of it, and
    a manifest that says `cis_and_signals` for a Store holding only significant
    signals is a wrong answer that looks like a right one. So the value is a
    build input, never a layout default (#298); which values a layout can
    *represent* is `COVERAGES_BY_LAYOUT` (ADR 0064).
    """

    #: Every association the source reports, genome-wide.
    GWAS = "gwas"
    #: A cis window around each Analysis's target, and nothing else.
    CIS = "cis"
    #: A cis window plus significant signals elsewhere.
    CIS_AND_SIGNALS = "cis_and_signals"
    #: Significant signals and their flanks, with no cis window.
    SIGNALS_ONLY = "signals_only"

    @classmethod
    def _missing_(cls, value: object) -> AssociationCoverage | None:
        # `full` is what every release built before #298 declares, and the
        # rename must not orphan them (issue #298). Reading the old spelling as
        # its new name is the whole compatibility promise; nothing writes
        # `full` again, and the value is never propagated back out.
        if value == "full":
            return cls.GWAS
        return None


#: The Association Coverages a Store Release of each Primary Storage Layout can
#: carry (issue #298).
#:
#: A Dense or Hybrid release holds one cell per (reference-panel variant,
#: Analysis). An association a narrower coverage excludes is therefore not
#: absent from the matrix -- it is a cell, and a cell left missing is what
#: Reference Completion fills in, and what a query reports as an imputed or
#: unobserved association. Only `gwas` makes every missing cell mean "the
#: source did not report this", which is why a Dense or Hybrid build refuses
#: the other three. A Ragged sequence is a list of the associations kept, so
#: absence states itself there and all four are representable.
COVERAGES_BY_LAYOUT: Mapping[PrimaryStorageLayout, frozenset[AssociationCoverage]] = (
    MappingProxyType(
        {
            PrimaryStorageLayout.DENSE: frozenset({AssociationCoverage.GWAS}),
            PrimaryStorageLayout.HYBRID: frozenset({AssociationCoverage.GWAS}),
            PrimaryStorageLayout.RAGGED: frozenset(AssociationCoverage),
        }
    )
)


def coverage_for_layout(
    coverage: AssociationCoverage | str, layout: PrimaryStorageLayout
) -> AssociationCoverage:
    """The Association Coverage ``layout`` will write, or a refusal naming why.

    One function rather than a check in each builder and a second in validation:
    the set of values a layout can carry is `COVERAGES_BY_LAYOUT`, and every
    caller that has to honour it -- every builder, and `validate` -- reads it
    from there (issue #298).
    """
    value = AssociationCoverage(coverage)
    accepted = COVERAGES_BY_LAYOUT[layout]
    if value not in accepted:
        raise ValueError(
            f"a {layout.value} Store Release cannot carry "
            f"association_coverage={value.value}: it accepts "
            f"{', '.join(sorted(c.value for c in accepted))}. A {layout.value} "
            "release has a cell for every (variant, Analysis), so an "
            "association outside the coverage would be indistinguishable from "
            "one the source did not report (issue #298)"
        )
    return value


class CompletionState(StrEnum):
    OBSERVED_ONLY = "observed_only"
    REFERENCE_COMPLETED = "reference_completed"


class StoredEffectScale(StrEnum):
    SD = "sd"
    LOG_OR = "log_or"
    LOG_HAZARD = "log_hazard"


class SampleSizeKind(StrEnum):
    TOTAL = "total"
    CASE_CONTROL = "case_control"
    EFFECTIVE = "effective"
    VARIANT_LEVEL = "variant_level"


class SampleSizeScope(StrEnum):
    ANALYSIS_LEVEL = "analysis_level"
    VARIANT_LEVEL = "variant_level"


# Analytical Metadata vocabularies (store-format spec §7a). Literal values match
# the registry's already-shipped analyses.tsv vocabulary (opengwasdb-stores
# docs/release-metadata-schema.md), adopted as canonical here rather than
# reworked registry-side, since opengwasdb had not yet implemented analyses.tsv
# I/O when the two vocabularies diverged.
class AncestryAssignmentMethod(StrEnum):
    AF_ASSIGNED = "af_assigned"
    SOURCE_FALLBACK = "source_fallback"
    SOURCE_TRUSTED_NO_AF = "source_trusted_no_af"
    UNASSIGNED = "unassigned"


# Priority order documented in ADR-0029.
class OriginalSdMethod(StrEnum):
    DECLARED_STANDARDISED = "declared_standardised"
    SOURCE_PROVIDED = "source_provided"
    ESTIMATED_FROM_SOURCE_MAF = "estimated_from_source_maf"
    ESTIMATED_FROM_REFERENCE_MAF = "estimated_from_reference_maf"
    ESTIMATED_FROM_BETA_DISTRIBUTION = "estimated_from_beta_distribution"
    BINARY_TRAIT = "binary_trait"
    UNAVAILABLE = "unavailable"


class EafScope(StrEnum):
    ABSENT = "absent"
    VARIANT = "variant"
    ASSOCIATION = "association"


# EAF orientation verification (issue #115, ADR 0037 §6). A source that reports
# `effect_allele_frequency` against the *other* allele produces a store whose
# frequency column is systematically wrong with nothing in the row to show it,
# so the outcome of the build-time check is persisted per Analysis rather than
# left implicit in the build having succeeded.
class EafOrientationOutcome(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    UNVERIFIED = "unverified"


class EafOrientationMethod(StrEnum):
    """What the check correlated each Analysis's EAF against."""

    REFERENCE_PANEL = "reference_panel"
    CONSENSUS = "consensus"
    NONE = "none"


class InfoScope(StrEnum):
    ABSENT = "absent"
    VARIANT = "variant"
    ASSOCIATION = "association"


class AssociationStatus(StrEnum):
    MISSING = "missing"
    OBSERVED = "observed"
    IMPUTED = "imputed"

