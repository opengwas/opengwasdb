"""Fail-closed assembly configuration for source readers (issues #277, #278).

Coordinates have meaning only relative to an assembly.  This module resolves the
manifest's authoritative declaration and decides whether a reader can align it
to the explicitly declared reference build.  It never guesses from a filename,
coordinate, or source header.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from opengwasdb.build.liftover import normalise_build
from opengwasdb.readers.gwas_vcf import GWAS_VCF_CAPABILITY


class UnsupportedAssemblyConfiguration(ValueError):
    """A declared source/reference build pair has no configured reader seam."""


@dataclass(frozen=True)
class AssemblyPlan:
    """Normalised source/reference builds and the configured chain, if required."""

    source_build: str
    reference_build: str
    chain_file: Path | None = None

    @property
    def requires_liftover(self) -> bool:
        return self.source_build != self.reference_build


def _optional_build(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return normalise_build(value)


def _declared_aliases(row: Mapping[str, str], context: str) -> str | None:
    genome_build = _optional_build(row.get("source_genome_build"))
    assembly_alias = _optional_build(row.get("source_assembly"))
    if genome_build is not None and assembly_alias is not None and genome_build != assembly_alias:
        raise ValueError(
            f"{context}: conflicting source_genome_build={row['source_genome_build']!r} and "
            f"source_assembly={row['source_assembly']!r}"
        )
    return genome_build or assembly_alias


def declared_source_build(
    row: Mapping[str, str], *, default: str | None, context: str
) -> str:
    """Resolve one row's authoritative build without guessing (issue #277)."""
    resolved = _declared_aliases(row, context)
    if resolved is not None:
        return resolved
    fallback = _optional_build(default)
    if fallback is not None:
        return fallback
    raise ValueError(
        f"{context}: source_genome_build is required (source_assembly is a supported alias); "
        "supply an explicit CLI default when the manifest omits both"
    )


def assembly_plan(
    *,
    capability: str,
    source_build: str,
    reference_build: str,
    chain_file: str | Path | None,
) -> AssemblyPlan:
    """Validate one reader's assembly alignment before its source is scanned."""
    source = normalise_build(source_build)
    reference = normalise_build(reference_build)
    if source == reference:
        return AssemblyPlan(source, reference)
    if capability != GWAS_VCF_CAPABILITY:
        raise UnsupportedAssemblyConfiguration(
            f"unsupported cross-build configuration for {capability!r}: "
            f"{source}->{reference}; this reader has no liftover seam"
        )
    if (source, reference) != ("hg19", "hg38"):
        raise UnsupportedAssemblyConfiguration(
            f"unsupported cross-build configuration for {capability!r}: "
            f"{source}->{reference}; only hg19->hg38 is implemented"
        )
    if chain_file is None:
        raise UnsupportedAssemblyConfiguration(
            f"cross-build configuration {source}->{reference} requires --chain-file"
        )
    chain = Path(chain_file)
    if not chain.is_file():
        raise FileNotFoundError(f"liftover chain file not found: {chain}")
    return AssemblyPlan(source, reference, chain)
