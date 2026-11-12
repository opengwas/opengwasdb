"""An Analysis's Trait position, lifted with its variants (issue #433).

A Store's Trait positions are in the Store's `reference_assembly`, as its
variants are. A builder that lifts a source's variants to GRCh38 lifts the
Trait positions the source or manifest declared in the same build, through the
same chain and the same engine (`opengwasdb.build.liftover.lifted_rows`), and
this module is the one place that does it.

A position the chain does not carry over -- placed nowhere, or onto another
chromosome -- gives its Analysis no Trait position: never the GRCh37 value
under a GRCh38 Store, which would centre a cis window off the gene without any
error. Such Analyses are counted and named, and a share past the variant
lift's failure threshold refuses the build.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from opengwasdb.admission.record import UNLIFTED_TRAIT_POSITIONS
from opengwasdb.build.liftover import (
    LiftoverFailureError,
    enforce_liftover_threshold,
    lifted_positions,
)
from opengwasdb.variants.normalise import VariantNormalisationError, normalise_chromosome

__all__ = ["LiftedTraitPositions", "lift_trait_positions", "write_unlifted"]

#: How many unlifted Analyses the printed summary names; the file names them all.
_NAMED_IN_SUMMARY = 10


@dataclass(frozen=True)
class LiftedTraitPositions:
    """What lifting a build's Trait positions produced."""

    #: Analysis id -> its ``(bare chromosome, GRCh38 bp)``, for each that lifted.
    positions: dict[str, tuple[str, int]]
    attempted: int
    #: The Analyses whose position did not lift, in the order they were given.
    unlifted: tuple[str, ...]

    def report(self) -> None:
        """Say on stdout how many positions lifted and which Analyses lost theirs."""
        print(
            f"Trait positions lifted: {len(self.positions)}/{self.attempted}; "
            f"{len(self.unlifted)} Analyses have no Trait position because theirs did not lift"
            + (
                f": {', '.join(self.unlifted[:_NAMED_IN_SUMMARY])}"
                + (", ..." if len(self.unlifted) > _NAMED_IN_SUMMARY else "")
                + f" (all named in {UNLIFTED_TRAIT_POSITIONS} in the Admission record)"
                if self.unlifted
                else ""
            )
        )


def write_unlifted(lifted: LiftedTraitPositions | None, directory: Path) -> None:
    """Name every unlifted Analysis in ``directory`` (the Admission record's).

    ``None`` -- a build that lifted nothing -- writes nothing: the record's own
    reset already removed any list an earlier build left there.
    """
    if lifted is None or not lifted.unlifted:
        return
    directory.mkdir(parents=True, exist_ok=True)
    (directory / UNLIFTED_TRAIT_POSITIONS).write_text(
        "analysis_id\n" + "".join(f"{a}\n" for a in lifted.unlifted)
    )


def lift_trait_positions(
    declared: Sequence[tuple[str, str, int]],
    *,
    chain_file: str | Path | None,
    failure_threshold: float,
) -> LiftedTraitPositions:
    """Lift each ``(analysis_id, chromosome, bp)`` from GRCh37 to GRCh38.

    Raises `LiftoverFailureError` when the unlifted share exceeds
    ``failure_threshold`` -- the variant lift's own threshold, applied by its
    own function (`enforce_liftover_threshold`).
    """
    placed = lifted_positions([(chrom, bp) for _id, chrom, bp in declared], chain_file=chain_file)
    attempts = len(declared)
    positions: dict[str, tuple[str, int]] = {}
    unlifted: list[str] = []
    for (analysis_id, chrom, _bp), lifted in zip(declared, placed, strict=True):
        if lifted is not None and _same_chromosome(chrom, lifted[0]):
            positions[analysis_id] = (normalise_chromosome(lifted[0]), lifted[1])
        else:
            unlifted.append(analysis_id)
    result = LiftedTraitPositions(positions, attempts, tuple(unlifted))
    try:
        enforce_liftover_threshold(
            attempts,
            len(unlifted),
            from_build="hg19",
            to_build="hg38",
            failure_threshold=failure_threshold,
        )
    except LiftoverFailureError as exc:
        raise LiftoverFailureError(
            f"Trait-position {exc} -- {len(unlifted)} of {attempts} Analyses' Trait "
            f"positions did not lift (first: {list(unlifted[:5])})"
        ) from exc
    return result


def _same_chromosome(source: str, lifted: str) -> bool:
    """Whether the chain kept the position on its chromosome; a chain that moves
    it elsewhere has not lifted the Trait's position, whatever it returned."""
    try:
        return normalise_chromosome(source) == normalise_chromosome(lifted)
    except VariantNormalisationError:
        return False
