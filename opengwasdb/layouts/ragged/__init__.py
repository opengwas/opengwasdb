"""Ragged layout implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from opengwasdb.model.enums import AssociationCoverage

#: Every Ragged builder lifts to hg38, so every Ragged release states it.
RAGGED_REFERENCE_ASSEMBLY = "GRCh38"


@dataclass(frozen=True)
class RaggedBuildResult:
    """Paths, dimensions and the coverage a built Ragged release declares.

    Shared by both Ragged builders: a caller switching between SSF and BESD
    inputs should not have to learn a second result type, and two copies of the
    field list is how the two drift apart.
    """

    output_path: Path
    n_variants: int
    n_analyses: int
    n_associations: int
    association_coverage: AssociationCoverage


__all__ = ["RAGGED_REFERENCE_ASSEMBLY", "RaggedBuildResult"]
