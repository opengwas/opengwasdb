"""The controlled vocabularies a reader needs to open a Store Release.

A manifest's `primary_layout`, `association_coverage` and `completion_state`
(spec §2). The rest of the store contract's vocabularies, and which coverages a
layout can carry, are `opengwasdb.model.enums`'.
"""

from __future__ import annotations

from enum import StrEnum


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


class CompletionState(StrEnum):
    OBSERVED_ONLY = "observed_only"
    REFERENCE_COMPLETED = "reference_completed"
